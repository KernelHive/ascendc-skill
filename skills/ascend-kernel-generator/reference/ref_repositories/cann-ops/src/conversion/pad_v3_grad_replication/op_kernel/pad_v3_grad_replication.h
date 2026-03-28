/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file pad_v3_grad_replication.h
 */
#ifndef PAD_V3_GRAD_REPLICATION_H
#define PAD_V3_GRAD_REPLICATION_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
using namespace AscendC;

template <typename dtype, bool isB16>
class PadV3GradReplication {
public:
__aicore__ inline PadV3GradReplication(PadV3GradReplicationTilingData &tilingData)
    : tilingData(tilingData) {}

__aicore__ inline void Init(GM_ADDR x, GM_ADDR z, GM_ADDR workspace)
{
    xGm.SetGlobalBuffer((__gm__ dtype *)x, tilingData.inputSize);
    zGm.SetGlobalBuffer((__gm__ dtype *)z, tilingData.outputSize);
    ws.SetGlobalBuffer((__gm__ float *)GetUserWorkspace(workspace), tilingData.workspaceSize);
    InitGlobalMemory(ws, tilingData.workspaceSize, (float)0);
    SyncAll();

    dtypeByteSize = sizeof(dtype);
    eleNumPer32Block = 32U / dtypeByteSize;  // 32Block能存储多少个元素
    isLastCore = (GetBlockIdx() == GetBlockNum() - 1);
    if (isLastCore) {
        cubeNumCurrentCore = tilingData.cubeNumLastCore;
    } else {
        cubeNumCurrentCore = tilingData.cubeNumEachCore;
    }
}

__aicore__ inline void Process()
{
    // 初始化同步事件
    InitSync();
    // 申请累加buffer
    ApplyComputeBuf();
    // 上下面
    ProcessTopAndBottom();
    // 左右面
    ProcessLeftAndRight();
    // corner处理
    ProcessCorner();
    // 释放所有buffer
    ReleaseBuf();
    // 申请搬运buffer
    ApplyMoveBuf();
    // 搬出
    MoveToOut();
    // 关闭同步
    CloseSync();
}

private:
// —— —— —— —— —— —— —— —— —— —— —— 基础接口 —— —— —— —— —— —— —— —— —— —— —— —— ——

__aicore__ inline void InitSync()
{
    evtIDMTE3ToMTE2 = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    evtIDVToMTE2 = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    evtIDMTE2ToV = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    evtIDVToMTE3 = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    evtIDMTE2ToMTE3 = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
}

__aicore__ inline void ApplyComputeBuf()
{
    pipe.Init();
    pipe.InitBuffer(x1Buf, tilingData.addTensorByteSize);
    pipe.InitBuffer(x2Buf, tilingData.addTensorByteSize);
    pipe.InitBuffer(yBuf, tilingData.addTensorByteSize);
    x1Local = x1Buf.Get<float>();
    yLocal = yBuf.Get<float>();
    x2Local = x2Buf.Get<float>();
    if constexpr (isB16) {
        x1LocalB16Big = x1Buf.GetWithOffset<dtype>(tilingData.addTensorSize, tilingData.addTensorByteSize / PAIR);
        yLocalB16Big = yBuf.GetWithOffset<dtype>(tilingData.addTensorSize, tilingData.addTensorByteSize / PAIR);
    }
}

__aicore__ inline void ApplyMoveBuf()
{
    pipe.Init();
    pipe.InitBuffer(mBuf, tilingData.moveTensorByteSize);
    mLocal = mBuf.Get<float>();
    if constexpr (isB16) {
        mLocalB16Big = mBuf.GetWithOffset<dtype>(tilingData.moveTensorSize, tilingData.moveTensorByteSize / PAIR);
        mLocalB16Little = mBuf.Get<dtype>(tilingData.moveTensorSize);
    }
}

__aicore__ inline void ReleaseBuf()
{
    PipeBarrier<PIPE_ALL>();
    pipe.Destroy();
}

__aicore__ inline void CloseSync()
{
    PipeBarrier<PIPE_ALL>();
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(evtIDMTE3ToMTE2);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(evtIDVToMTE2);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(evtIDMTE2ToV);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(evtIDVToMTE3);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(evtIDMTE2ToMTE3);
}

__aicore__ inline uint32_t GetDataLength32BAligned(uint32_t dataLength)
{
    return ((dataLength - 1) / eleNumPer32Block + 1) * eleNumPer32Block;
}

__aicore__ inline bool IsBlockNumOddPerDataLength(uint32_t dataLength)
{
    return ((dataLength - 1) / ELE_NUM_B32_PER_BLOCK + 1) % PAIR != 0;
}

// —— —— —— —— —— —— —— —— —— —— —— Top/Bottom处理接口 —— —— —— —— —— —— —— —— —— —— —— —— ——

__aicore__ inline void CopyInTop(uint64_t startPos, uint32_t dataLength, uint32_t fullDataLength, uint16_t repeatTimes, bool isFirst)
{
    DataCopyExtParams copyParams{repeatTimes, dataLength * dtypeByteSize,
                                    (tilingData.layerInputSize - tilingData.topSize) * dtypeByteSize,
                                    0, 0};
    DataCopyPadExtParams<dtype> padParams{false, 0, 0, 0};
    if constexpr (isB16) {
        if (isFirst) {
            DataCopyPad(x1LocalB16Big, xGm[startPos], copyParams, padParams);
        }
        DataCopyPad(yLocalB16Big, xGm[startPos + tilingData.topSize], copyParams, padParams);

        SetFlag<HardEvent::MTE2_V>(evtIDMTE2ToV);  // 计算等搬入
        WaitFlag<HardEvent::MTE2_V>(evtIDMTE2ToV);
        if (isFirst) {
            Cast(x1Local, x1LocalB16Big, RoundMode::CAST_NONE, repeatTimes * fullDataLength);
        }
        Cast(yLocal, yLocalB16Big, RoundMode::CAST_NONE, repeatTimes * fullDataLength);
        PipeBarrier<PIPE_V>();
    } else {
        if (isFirst) {
            DataCopyPad(x1Local, xGm[startPos], copyParams, padParams);
        }
        DataCopyPad(yLocal, xGm[startPos + tilingData.topSize], copyParams, padParams);
    }
}

__aicore__ inline void ComputeTop(bool &atX1, uint32_t fullDataLength, uint16_t repeatTimes)
{
    LocalTensor<float> xLocal = atX1 ? x1Local : x2Local;
    LocalTensor<float> zLocal = atX1 ? x2Local : x1Local;
    Add(zLocal, xLocal, yLocal, (int32_t)(fullDataLength * repeatTimes));
    atX1 = !atX1;
}

__aicore__ inline void CopyOutTop(bool atX1, uint64_t startPos, uint32_t dataLength, uint32_t fullDataLength, uint16_t repeatTimes)
{
    LocalTensor<float> zLocal = atX1 ? x1Local : x2Local;
    DataCopyExtParams copyParams{repeatTimes, dataLength * 4U,  // 4U为float的字节大小
                                    (fullDataLength - dataLength >= 8) ? 1U : 0U,  // 如果每块脏数据达到8个，为b16（最大15），则这里UB自动对齐少算一个32Block
                                    0, 0};
    DataCopyPad(ws[startPos], zLocal, copyParams);
}

__aicore__ inline void ProcessTopYield(uint64_t yieldStart, uint32_t dataLength, uint64_t wsYieldStart,
                                        uint32_t computeTimes, uint16_t repeatTimes)
{
    bool atX1 = true;  // 加数是否在X1
    SetFlag<HardEvent::MTE3_MTE2>(evtIDMTE3ToMTE2);  // 搬入等搬出
    WaitFlag<HardEvent::MTE3_MTE2>(evtIDMTE3ToMTE2);
    uint32_t fullDataLength = GetDataLength32BAligned(dataLength);  // 每面的计算长度（32B对齐）

    for (auto k = 0; k < computeTimes; k++) {  // 待处理的一轮（两行相加）
        uint64_t startPos = yieldStart + tilingData.topSize * k;  // 搬运起始序号

        SetFlag<HardEvent::V_MTE2>(evtIDVToMTE2);  // 搬入等计算
        WaitFlag<HardEvent::V_MTE2>(evtIDVToMTE2);
        // 搬入一轮计算的数据到UB，搬入X1（首次）、Y（k的下一行）
        CopyInTop(startPos, dataLength, fullDataLength, repeatTimes, k == 0);
        
        SetFlag<HardEvent::MTE2_V>(evtIDMTE2ToV);  // 计算等搬入
        WaitFlag<HardEvent::MTE2_V>(evtIDMTE2ToV);
        // X1、X2交替成为VECIN和VECCALC
        ComputeTop(atX1, fullDataLength, repeatTimes);
    }

    SetFlag<HardEvent::V_MTE3>(evtIDVToMTE3);  // 搬出等计算
    WaitFlag<HardEvent::V_MTE3>(evtIDVToMTE3);
    // 最后一个存储结果的X搬到WS
    CopyOutTop(atX1, wsYieldStart, dataLength, fullDataLength, repeatTimes);
}

__aicore__ inline void ProcessTopCaseLarge(EdgeTiling &topTiling, bool bottomMode=false)
{
    uint64_t taskStart = GetBlockIdx() * tilingData.cubeNumEachCore * tilingData.cubeInputSize + \
                            (bottomMode ? tilingData.topToBottomSize: 0);  // 本核任务起始序号
    uint64_t wsTaskStart = GetBlockIdx() * tilingData.cubeNumEachCore * tilingData.totalTopInputSizeEachCube + \
                            (bottomMode ? tilingData.topResultSize : 0);  // 任务ws起始序号
    for (auto l = 0; l < tilingData.inputShape[DIM_1] * cubeNumCurrentCore; l++) {  // 待处理面
        uint64_t layerStart = taskStart + tilingData.layerInputSize * l;  // 面起始序号
        uint64_t wsLayerStart = wsTaskStart + tilingData.topSize * l;  // 面WS起始序号
    
        for (auto j = 0; j < topTiling.tileCount; j++) {  // 待处理的轮区（满）
            uint64_t yieldStart = layerStart + tilingData.addTensorSize * j;  // 轮区起始序号
            uint32_t dataLength = tilingData.addTensorSize;  // 轮区实际搬运长度
            uint64_t wsYieldStart = wsLayerStart + tilingData.addTensorSize * j;  // 轮区WS起始序号
            ProcessTopYield(yieldStart, dataLength, wsYieldStart,
                            (bottomMode ? tilingData.paddings[DIM_3] : tilingData.paddings[DIM_2]), 1);
        }
        if (topTiling.additionalCount) {  // 待处理的轮区（不满）
            uint64_t yieldStart = layerStart + tilingData.addTensorSize * topTiling.tileCount;  // 轮区起始序号
            uint32_t dataLength = topTiling.additionalCount;  // 轮区实际搬运长度
            uint64_t wsYieldStart = wsLayerStart + tilingData.addTensorSize * topTiling.tileCount;  // 轮区WS起始序号
            ProcessTopYield(yieldStart, dataLength, wsYieldStart,
                            (bottomMode ? tilingData.paddings[DIM_3] : tilingData.paddings[DIM_2]), 1);
        }
    }
}

__aicore__ inline void ProcessTopCaseSmall(EdgeTiling &topTiling, bool bottomMode=false)
{
    uint64_t taskStart = GetBlockIdx() * tilingData.cubeNumEachCore * tilingData.cubeInputSize + \
                            (bottomMode ? tilingData.topToBottomSize : 0);  // 本核任务起始序号
    uint64_t wsTaskStart = GetBlockIdx() * tilingData.cubeNumEachCore * tilingData.totalTopInputSizeEachCube + \
                            (bottomMode ? tilingData.topResultSize : 0);  // 任务ws起始序号
    uint32_t dataLength = tilingData.topSize;  // 每面的搬运长度

    for (auto i = 0; i < topTiling.tileCount; i++) {  // 待处理的轮区（满）
        uint64_t yieldStart = taskStart + topTiling.edgeCount * tilingData.layerInputSize * i;  // 轮区起始序号
        uint64_t wsYieldStart = wsTaskStart + topTiling.edgeCount * tilingData.topSize * i;  // 轮区WS起始序号
        uint16_t repeatTimes = topTiling.edgeCount;  // 要搬几个面
        ProcessTopYield(yieldStart, dataLength, wsYieldStart,
                        (bottomMode ? tilingData.paddings[DIM_3] : tilingData.paddings[DIM_2]), repeatTimes);
    }
    if (topTiling.additionalCount) {  // 待处理的轮区（不满）
        uint64_t yieldStart = taskStart + topTiling.edgeCount * tilingData.layerInputSize * topTiling.tileCount;  // 轮区起始序号
        uint64_t wsYieldStart = wsTaskStart + topTiling.edgeCount * tilingData.topSize * topTiling.tileCount;  // 轮区WS起始序号
        uint16_t repeatTimes = topTiling.additionalCount;  // 要搬几个面
        ProcessTopYield(yieldStart, dataLength, wsYieldStart,
                        (bottomMode ? tilingData.paddings[DIM_3] : tilingData.paddings[DIM_2]), repeatTimes);
    }
}

// —— —— —— —— —— —— —— —— —— —— —— 处理上下边 —— —— —— —— —— —— —— —— —— —— —— —— ——

__aicore__ inline void ProcessTopAndBottom()
{
    auto topTiling = (isLastCore) ? tilingData.topTilingLastCore : tilingData.topTiling;
    if (topTiling.edgeCount) {  // 一轮UB能处理一个及以上完整面
        if (tilingData.paddings[DIM_2]) ProcessTopCaseSmall(topTiling);
        if (tilingData.paddings[DIM_3]) ProcessTopCaseSmall(topTiling, true);
    } else {  // 一轮UB（轮区）不足一面边长的情况
        if (tilingData.paddings[DIM_2]) ProcessTopCaseLarge(topTiling);
        if (tilingData.paddings[DIM_3]) ProcessTopCaseLarge(topTiling, true);
    }
}

// —— —— —— —— —— —— —— —— —— —— —— Left/Right处理接口 —— —— —— —— —— —— —— —— —— —— —— —— ——

__aicore__ inline void CopyInLeft(uint64_t startPos, uint16_t dataLength, uint32_t fullDataLength, uint16_t repeatTimes, bool isFirst)
// dataLength为实际的列长，repeatTimes还是面数，但无法多面调用一次接口，必须一面面搬运
{
    uint32_t eleNumPerRepeat = dataLength * eleNumPer32Block;
    for (auto i = 0; i < repeatTimes; i++) {
        DataCopyExtParams copyParams{dataLength, dtypeByteSize,
                                        (tilingData.topSize - 1) * dtypeByteSize,
                                        0, 0};
        DataCopyPadExtParams<dtype> padParams{false, 0, 0, 0};
        if constexpr (isB16) {
            if (isFirst) {
                DataCopyPad(x1LocalB16Big[i * eleNumPerRepeat], xGm[startPos + i * tilingData.layerInputSize], copyParams, padParams);
            }
            DataCopyPad(yLocalB16Big[i * eleNumPerRepeat], xGm[startPos + 1 + i * tilingData.layerInputSize], copyParams, padParams);
        } else {
            if (isFirst) {
                DataCopyPad(x1Local[i * eleNumPerRepeat], xGm[startPos + i * tilingData.layerInputSize], copyParams, padParams);
            }
            DataCopyPad(yLocal[i * eleNumPerRepeat], xGm[startPos + 1 + i * tilingData.layerInputSize], copyParams, padParams);
        }
    }
    if constexpr (isB16) {
        SetFlag<HardEvent::MTE2_V>(evtIDMTE2ToV);  // 计算等搬入
        WaitFlag<HardEvent::MTE2_V>(evtIDMTE2ToV);
        if (isFirst) {
            Cast(x1Local, x1LocalB16Big, RoundMode::CAST_NONE, eleNumPerRepeat * repeatTimes);
        }
        Cast(yLocal, yLocalB16Big, RoundMode::CAST_NONE, eleNumPerRepeat * repeatTimes);
        pipe_barrier(PIPE_V);
    }
}

/* fullDataLength须为dataLength向32位对齐取整，对左右来说就是乘以eleNumPer32Block */
__aicore__ inline void ComputeLeft(bool &atX1, uint32_t fullDataLength, uint16_t repeatTimes)
{
    LocalTensor<float> xLocal = atX1 ? x1Local : x2Local;
    LocalTensor<float> zLocal = atX1 ? x2Local : x1Local;
    Add(zLocal, xLocal, yLocal, (int32_t)(fullDataLength * repeatTimes));
    atX1 = !atX1;
}

__aicore__ inline void CopyOutLeft(bool atX1, uint64_t startPos, uint16_t dataLength, uint16_t repeatTimes)
{
    LocalTensor<float> zLocal = atX1 ? x1Local : x2Local;
    DataCopyExtParams copyParams{(uint16_t)(repeatTimes * dataLength), 4U, (isB16) ? 1U : 0U, 0, 0};  // b16场景每块恒有15个脏数据，UB自然对齐必少一个32Block
    DataCopyPad(ws[startPos], zLocal, copyParams);
}

__aicore__ inline void ProcessLeftYield(uint64_t yieldStart, uint16_t dataLength, uint64_t wsYieldStart,
                                        uint32_t computeTimes, uint16_t repeatTimes)
{
    bool atX1 = true;  // 加数是否在X1
    SetFlag<HardEvent::MTE3_MTE2>(evtIDMTE3ToMTE2);  // 搬入等搬出
    WaitFlag<HardEvent::MTE3_MTE2>(evtIDMTE3ToMTE2);
    uint32_t fullDataLength = dataLength * eleNumPer32Block;  // 每面的计算长度（32B对齐，一个数占32Block）

    for (auto k = 0; k < computeTimes; k++) {  // 待处理的一轮（两列相加）
        uint64_t startPos = yieldStart + k;  // 搬运起始序号

        SetFlag<HardEvent::V_MTE2>(evtIDVToMTE2);  // 搬入等计算
        WaitFlag<HardEvent::V_MTE2>(evtIDVToMTE2);
        // 搬入一轮计算的数据到UB，搬入X1（首次）、Y（k的下一行）
        CopyInLeft(startPos, dataLength, fullDataLength, repeatTimes, k == 0);
        
        SetFlag<HardEvent::MTE2_V>(evtIDMTE2ToV);  // 计算等搬入
        WaitFlag<HardEvent::MTE2_V>(evtIDMTE2ToV);
        // X1、X2交替成为VECIN和VECCALC
        ComputeLeft(atX1, fullDataLength, repeatTimes);
    }

    SetFlag<HardEvent::V_MTE3>(evtIDVToMTE3);  // 搬出等计算
    WaitFlag<HardEvent::V_MTE3>(evtIDVToMTE3);
    // 最后一个存储结果的X搬到WS
    CopyOutLeft(atX1, wsYieldStart, dataLength, repeatTimes);
}

__aicore__ inline void ProcessLeftCaseLarge(EdgeTiling &leftTiling, bool rightMode=false)
{
    uint64_t taskStart = GetBlockIdx() * tilingData.cubeNumEachCore * tilingData.cubeInputSize + \
                            (tilingData.paddings[DIM_2] ? (tilingData.paddings[DIM_2] + 1) : 0) * tilingData.topSize + \
                            (rightMode ? tilingData.leftToRightSize : 0);  // 本核任务起始序号
    uint64_t wsTaskStart = tilingData.topResultSize * 2 + \
                            GetBlockIdx() * tilingData.cubeNumEachCore * tilingData.totalLeftInputSizeEachCube + \
                            (rightMode ? tilingData.leftResultSize : 0);  // 任务ws起始序号
    for (auto l = 0; l < tilingData.inputShape[DIM_1] * cubeNumCurrentCore; l++) {  // 待处理面
        uint64_t layerStart = taskStart + tilingData.layerInputSize * l;  // 面起始序号
        uint64_t wsLayerStart = wsTaskStart + tilingData.leftSize * l;  // 面WS起始序号

        for (auto j = 0; j < leftTiling.tileCount; j++) {  // 待处理的轮区（满）
            uint64_t yieldStart = layerStart + tilingData.addTensorBlockNum * j * tilingData.topSize;  // 轮区起始序号
            uint16_t dataLength = tilingData.addTensorBlockNum;  // 轮区实际搬运长度
            uint64_t wsYieldStart = wsLayerStart + tilingData.addTensorBlockNum * j;  // 轮区WS起始序号
            ProcessLeftYield(yieldStart, dataLength, wsYieldStart,
                                (rightMode ? tilingData.paddings[DIM_1] : tilingData.paddings[DIM_0]), 1);
        }
        if (leftTiling.additionalCount) {  // 待处理的轮区（不满）
            uint64_t yieldStart = layerStart + tilingData.addTensorBlockNum * leftTiling.tileCount * tilingData.topSize;  // 轮区起始序号
            uint16_t dataLength = leftTiling.additionalCount;  // 轮区实际搬运长度
            uint64_t wsYieldStart = wsLayerStart + tilingData.addTensorBlockNum * leftTiling.tileCount;  // 轮区WS起始序号
            ProcessLeftYield(yieldStart, dataLength, wsYieldStart,
                                (rightMode ? tilingData.paddings[DIM_1] : tilingData.paddings[DIM_0]), 1);
        }
    }
}

__aicore__ inline void ProcessLeftCaseSmall(EdgeTiling &leftTiling, bool rightMode=false)
{
    uint64_t taskStart = GetBlockIdx() * tilingData.cubeNumEachCore * tilingData.cubeInputSize + \
                            (tilingData.paddings[DIM_2] ? (tilingData.paddings[DIM_2] + 1) : 0) * tilingData.topSize + \
                            (rightMode ? tilingData.leftToRightSize : 0);  // 本核任务起始序号
    uint64_t wsTaskStart = tilingData.topResultSize * 2 + \
                            GetBlockIdx() * tilingData.cubeNumEachCore * tilingData.totalLeftInputSizeEachCube + \
                            (rightMode ? tilingData.leftResultSize : 0);  // 任务ws起始序号
    uint32_t dataLength = tilingData.leftSize;  // 每面的搬运长度

    for (auto i = 0; i < leftTiling.tileCount; i++) {  // 待处理的轮区（满）
        uint64_t yieldStart = taskStart + leftTiling.edgeCount * tilingData.layerInputSize * i;  // 轮区起始序号
        uint64_t wsYieldStart = wsTaskStart + leftTiling.edgeCount * tilingData.leftSize * i;  // 轮区WS起始序号
        uint16_t repeatTimes = leftTiling.edgeCount;  // 要搬几个面
        ProcessLeftYield(yieldStart, dataLength, wsYieldStart, (rightMode ? tilingData.paddings[DIM_1] : tilingData.paddings[DIM_0]),
                            repeatTimes);
    }
    if (leftTiling.additionalCount) {  // 待处理的轮区（不满）
        uint64_t yieldStart = taskStart + leftTiling.edgeCount * tilingData.layerInputSize * leftTiling.tileCount;  // 轮区起始序号
        uint64_t wsYieldStart = wsTaskStart + leftTiling.edgeCount * tilingData.leftSize * leftTiling.tileCount;  // 轮区WS起始序号
        uint16_t repeatTimes = leftTiling.additionalCount;  // 要搬几个面
        ProcessLeftYield(yieldStart, dataLength, wsYieldStart, (rightMode ? tilingData.paddings[DIM_1] : tilingData.paddings[DIM_0]),
                            repeatTimes);
    }
}

// —— —— —— —— —— —— —— —— —— —— —— 处理左右边（无上下padding的部分） —— —— —— —— —— —— —— —— —— —— —— —— ——

__aicore__ inline void ProcessLeftAndRight()
{
    if (!tilingData.leftSize) return;
    auto leftTiling = (isLastCore) ? tilingData.leftTilingLastCore : tilingData.leftTiling;;
    if (leftTiling.edgeCount) {  // 一轮UB能处理一个及以上完整面
        if (tilingData.paddings[DIM_0]) ProcessLeftCaseSmall(leftTiling);
        if (tilingData.paddings[DIM_1]) ProcessLeftCaseSmall(leftTiling, true);
    } else {  // 一轮UB（轮区）不足一面边长的情况
        if (tilingData.paddings[DIM_0]) ProcessLeftCaseLarge(leftTiling);
        if (tilingData.paddings[DIM_1]) ProcessLeftCaseLarge(leftTiling, true);
    }
}

// —— —— —— —— —— —— —— —— —— —— —— corner处理接口 —— —— —— —— —— —— —— —— —— —— —— —— ——

__aicore__ inline void CopyInCorner(uint64_t startPos, uint16_t dataLength, bool isFirst)
{
    DataCopyExtParams copyParams{dataLength, 4U,
                                    (tilingData.topSize - 1) * 4U, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    if (isFirst) {
        DataCopyPad(x1Local, ws[startPos], copyParams, padParams);
    }
    DataCopyPad(yLocal, ws[startPos + 1], copyParams, padParams);
}

__aicore__ inline void ComputeCorner(bool &atX1, uint32_t fullDataLength)
{
    LocalTensor<float> xLocal = atX1 ? x1Local : x2Local;
    LocalTensor<float> zLocal = atX1 ? x2Local : x1Local;
    Add(zLocal, xLocal, yLocal, (int32_t)fullDataLength);
    atX1 = !atX1;
}

__aicore__ inline void CopyOutCorner(bool atX1, uint64_t startPos, uint16_t dataLength)
{
    LocalTensor<float> zLocal = atX1 ? x1Local : x2Local;
    DataCopyExtParams copyParams{dataLength, 4U, 0,
                                    (tilingData.topSize - 1) * 4U, 0};
    DataCopyPad(ws[startPos], zLocal, copyParams);
}

__aicore__ inline void ProcessCornerYield(uint64_t yieldStart, uint16_t dataLength, uint64_t wsYieldStart,
                                            uint32_t computeTimes, uint16_t repeatTimes=0)
{
    bool atX1 = true;  // 加数是否在X1
    SetFlag<HardEvent::MTE3_MTE2>(evtIDMTE3ToMTE2);  // 搬入等搬出
    WaitFlag<HardEvent::MTE3_MTE2>(evtIDMTE3ToMTE2);
    uint32_t fullDataLength = dataLength * eleNumPer32Block;  // 每面的计算长度（32B对齐，一个数占32Block）

    for (auto k = 0; k < computeTimes; k++) {  // 待处理的一轮（两列相加）
        uint64_t startPos = yieldStart + k;  // 搬运起始序号

        SetFlag<HardEvent::V_MTE2>(evtIDVToMTE2);  // 搬入等计算
        WaitFlag<HardEvent::V_MTE2>(evtIDVToMTE2);
        // 搬入一轮计算的数据到UB，搬入X1（首次）、Y（k的下一行）
        CopyInCorner(startPos, dataLength, k == 0);
        
        SetFlag<HardEvent::MTE2_V>(evtIDMTE2ToV);  // 计算等搬入
        WaitFlag<HardEvent::MTE2_V>(evtIDMTE2ToV);
        // X1、X2交替成为VECIN和VECCALC
        ComputeCorner(atX1, fullDataLength);
    }

    SetFlag<HardEvent::V_MTE3>(evtIDVToMTE3);  // 搬出等计算
    WaitFlag<HardEvent::V_MTE3>(evtIDVToMTE3);
    // 最后一个存储结果的X搬到WS
    CopyOutCorner(atX1, wsYieldStart, dataLength);
}

/* 这里是把整个当前核心的任务（在WS上）视为一个大面，然后按照LargeShape的模式left tiling
    edgeCount恒为0，每次搬满能搬tilingData.addTensorBlockNum行，只需关注需要搬满几轮、剩余几行
    由于每面连续，每次可以视为dataLength为一轮行数的1次repeat
    */
__aicore__ inline void ProcessTopCorner(EdgeTiling &cornerTiling, bool bottomMode=false)
{
    // 左边需要累加
    if (tilingData.paddings[DIM_0]) {
        uint64_t taskInputStart = GetBlockIdx() * tilingData.cubeNumEachCore * tilingData.totalTopInputSizeEachCube + \
                                    (bottomMode ? tilingData.topResultSize : 0);  // 任务ws输入起始序号
        uint64_t taskOutputStart = GetBlockIdx() * tilingData.cubeNumEachCore * tilingData.totalTopInputSizeEachCube + \
                                    (bottomMode ? tilingData.topResultSize : 0) + \
                                    tilingData.paddings[DIM_0];  // 任务ws输出起始序号

        for (auto i = 0; i < cornerTiling.tileCount; i++) {  // 待处理的轮区（满）
            uint64_t yieldStart = taskInputStart + tilingData.addTensorBlockNum * tilingData.topSize * i;  // 轮区输入起始序号
            uint64_t wsYieldStart = taskOutputStart + tilingData.addTensorBlockNum * tilingData.topSize * i;  // 轮区输出起始序号
            uint16_t dataLength = tilingData.addTensorBlockNum;  // 要搬几个面
            ProcessCornerYield(yieldStart, dataLength, wsYieldStart, tilingData.paddings[DIM_0], 1);
        }
        if (cornerTiling.additionalCount) {  // 待处理的轮区（不满）
            uint64_t yieldStart = taskInputStart + \
                                    tilingData.addTensorBlockNum * tilingData.topSize * cornerTiling.tileCount;  // 轮区输入起始序号
            uint64_t wsYieldStart = taskOutputStart + \
                                    tilingData.addTensorBlockNum * tilingData.topSize * cornerTiling.tileCount;  // 轮区输出起始序号
            uint16_t dataLength = cornerTiling.additionalCount;  // 要搬几个面
            ProcessCornerYield(yieldStart, dataLength, wsYieldStart, tilingData.paddings[DIM_0], 1);
        }
    }

    // 右边需要累加
    if (tilingData.paddings[DIM_1]) {
        uint64_t taskInputStart = GetBlockIdx() * tilingData.cubeNumEachCore * tilingData.totalTopInputSizeEachCube + \
                                    (bottomMode ? tilingData.topResultSize : 0) + \
                                    tilingData.leftToRightSize;  // 任务ws输入起始序号
        uint64_t taskOutputStart = GetBlockIdx() * tilingData.cubeNumEachCore * tilingData.totalTopInputSizeEachCube + \
                                    (bottomMode ? tilingData.topResultSize : 0) + \
                                    tilingData.leftToRightSize;  // 任务ws输出起始序号

        for (auto i = 0; i < cornerTiling.tileCount; i++) {  // 待处理的轮区（满）
            uint64_t yieldStart = taskInputStart + tilingData.addTensorBlockNum * tilingData.topSize * i;  // 轮区输入起始序号
            uint64_t wsYieldStart = taskOutputStart + tilingData.addTensorBlockNum * tilingData.topSize * i;  // 轮区输出起始序号
            uint16_t dataLength = tilingData.addTensorBlockNum;  // 要搬几个面
            ProcessCornerYield(yieldStart, dataLength, wsYieldStart, tilingData.paddings[DIM_1], 1);
        }
        if (cornerTiling.additionalCount) {  // 待处理的轮区（不满）
            uint64_t yieldStart = taskInputStart + \
                                    tilingData.addTensorBlockNum * tilingData.topSize * cornerTiling.tileCount;  // 轮区输入起始序号
            uint64_t wsYieldStart = taskOutputStart + \
                                    tilingData.addTensorBlockNum * tilingData.topSize * cornerTiling.tileCount;  // 轮区输出起始序号
            uint16_t dataLength = cornerTiling.additionalCount;  // 要搬几个面
            ProcessCornerYield(yieldStart, dataLength, wsYieldStart, tilingData.paddings[DIM_1], 1);
        }
    }
}

// —— —— —— —— —— —— —— —— —— —— —— 处理WS上corner的部分 —— —— —— —— —— —— —— —— —— —— —— —— ——

__aicore__ inline void ProcessCorner()
{
    auto cornerTiling = (isLastCore) ? tilingData.cornerTilingLastCore : tilingData.cornerTiling;;
    // 一轮UB一定能处理一个及以上完整面
    if (tilingData.paddings[DIM_2]) ProcessTopCorner(cornerTiling);
    if (tilingData.paddings[DIM_3]) ProcessTopCorner(cornerTiling, true);
}

// —— —— —— —— —— —— —— —— —— —— —— 搬出处理接口 —— —— —— —— —— —— —— —— —— —— —— —— ——

__aicore__ inline void CopyInInner(uint64_t startPos, uint32_t dataLength, uint16_t repeatTimes)
{
    DataCopyExtParams copyParams{repeatTimes, dataLength * dtypeByteSize,
                                    (uint32_t)(((tilingData.paddings[DIM_0] ? (tilingData.paddings[DIM_0] + 1) : 0) + \
                                    (tilingData.paddings[DIM_1] ? (tilingData.paddings[DIM_1] + 1) : 0)) * dtypeByteSize), 0, 0};
    DataCopyPadExtParams<dtype> padParams{false, 0, 0, 0};
    if constexpr (isB16) {
        DataCopyPad(mLocalB16Big, xGm[startPos], copyParams, padParams);
    } else {
        DataCopyPad(mLocal, xGm[startPos], copyParams, padParams);
    }
}

__aicore__ inline void CopyInInnerPadding(uint64_t startPos, uint32_t dataLength, uint32_t fullDataLength, uint16_t repeatTimes)
{
    CopyInInner(startPos, dataLength, repeatTimes);
    if constexpr (isB16) {
        SetFlag<HardEvent::MTE2_V>(evtIDMTE2ToV);  // 计算等搬入
        WaitFlag<HardEvent::MTE2_V>(evtIDMTE2ToV);
        Cast(mLocal, mLocalB16Big, RoundMode::CAST_NONE, fullDataLength * repeatTimes);
    }
}

__aicore__ inline void CopyOutInner(uint64_t startPos, uint32_t dataLength, uint16_t repeatTimes)
{
    DataCopyExtParams copyParams{repeatTimes, dataLength * dtypeByteSize, 0,
                                    ((tilingData.paddings[DIM_0] ? 1 : 0) + \
                                    (tilingData.paddings[DIM_1] ? 1 : 0)) * dtypeByteSize, 0};
    if constexpr (isB16) {
        DataCopyPad(zGm[startPos], mLocalB16Big, copyParams);
    } else {
        DataCopyPad(zGm[startPos], mLocal, copyParams);
    }
}

// atomic add到WS
__aicore__ inline void CopyOutInnerPadding(uint64_t startPos, uint32_t dataLength, uint32_t fullDataLength, uint16_t repeatTimes)
{
    DataCopyExtParams copyParams{repeatTimes, dataLength * 4U,  // 已转为float
                                    (fullDataLength - dataLength >= 8) ? 1U : 0U,  // 如果每块脏数据达到8个，为b16（最大15），则这里UB自动对齐少算一个32Block
                                    ((tilingData.paddings[DIM_0] ? 1 : 0) + \
                                    (tilingData.paddings[DIM_1] ? 1 : 0)) * 4U, 0};
    PipeBarrier<PIPE_ALL>();
    SetAtomicAdd<float>();
    DataCopyPad(ws[startPos], mLocal, copyParams);
    SetAtomicNone();
}

__aicore__ inline void ProcessInnerYield(uint64_t yieldStart, uint32_t dataLength, uint64_t outYieldStart,
                                            uint16_t repeatTimes, bool isPadding)
{
    uint32_t fullDataLength = GetDataLength32BAligned(dataLength);
    SetFlag<HardEvent::MTE3_MTE2>(evtIDMTE3ToMTE2);  // 搬入等搬出
    WaitFlag<HardEvent::MTE3_MTE2>(evtIDMTE3ToMTE2);
    // 搬入一轮
    if (isPadding) {
        CopyInInnerPadding(yieldStart, dataLength, fullDataLength, repeatTimes);
    } else {
        CopyInInner(yieldStart, dataLength, repeatTimes);
    }

    // 搬出一轮
    if (isPadding) {
        SetFlag<HardEvent::V_MTE3>(evtIDVToMTE3);  // 搬出等计算
        WaitFlag<HardEvent::V_MTE3>(evtIDVToMTE3);
        CopyOutInnerPadding(outYieldStart, dataLength, fullDataLength, repeatTimes);
    } else {
        SetFlag<HardEvent::MTE2_MTE3>(evtIDMTE2ToMTE3);  // 搬出等搬入
        WaitFlag<HardEvent::MTE2_MTE3>(evtIDMTE2ToMTE3);
        CopyOutInner(outYieldStart, dataLength, repeatTimes);
    }
}

__aicore__ inline void GetStartPosInner(uint64_t &taskInputStart, uint64_t &taskOutputStart, uint64_t &taskOutputStartPadding)
{
    taskInputStart = GetBlockIdx() * tilingData.cubeNumEachCore * tilingData.cubeInputSize + \
                    (tilingData.paddings[DIM_2] ? ((tilingData.paddings[DIM_2] + 1) * tilingData.topSize) : 0) + \
                    (tilingData.paddings[DIM_0] ? (tilingData.paddings[DIM_0] + 1) : 0);  // 本核GM输入起始序号
    taskOutputStart = GetBlockIdx() * tilingData.cubeNumEachCore * tilingData.cubeOutputSize + \
                    (tilingData.paddings[DIM_2] ? tilingData.outputShape[DIM_3] : 0) + \
                    (tilingData.paddings[DIM_0] ? 1 : 0);  // 本核GM输出起始序号
    taskOutputStartPadding = GetBlockIdx() * tilingData.cubeNumEachCore * tilingData.layerOutputSize + \
                            (tilingData.paddings[DIM_2] ? tilingData.outputShape[DIM_3] : 0) + \
                            (tilingData.paddings[DIM_0] ? 1 : 0) + \
                            (tilingData.topResultSize + tilingData.leftResultSize) * PAIR;  // 本核WS输出起始序号(padding)
}

__aicore__ inline void GetPaddingOutputPos(const uint64_t &cubeOutputStart, const uint64_t &cubeOutputStartPadding, const uint32_t &l,
                                            uint64_t &layerOutputStart, bool &isPadding)
{
    if (tilingData.paddings[DIM_4] && (l <= tilingData.paddings[DIM_4])) {
        layerOutputStart = cubeOutputStartPadding;
        isPadding = true;
    } else if (tilingData.paddings[DIM_5] && (l >= tilingData.inputShape[DIM_1] - tilingData.paddings[DIM_5] - 1)) {
        layerOutputStart = cubeOutputStartPadding + tilingData.layerOutputSize * tilingData.inputShape[DIM_0];
        isPadding = true;
    } else {
        layerOutputStart = cubeOutputStart + tilingData.layerOutputSize * (l - tilingData.paddings[DIM_4]);
        isPadding = false;
    }
}

__aicore__ inline void MoveInnerCaseLarge(EdgeTiling &innerTiling)
{
    uint64_t taskInputStart;
    uint64_t taskOutputStart;
    uint64_t taskOutputStartPadding;
    GetStartPosInner(taskInputStart, taskOutputStart, taskOutputStartPadding);

    for (auto c = 0; c < cubeNumCurrentCore; c++) {  // 对于每个要处理的cube
        uint64_t cubeInputStart = taskInputStart + tilingData.cubeInputSize * c;
        uint64_t cubeOutputStart = taskOutputStart + tilingData.cubeOutputSize * c;
        uint64_t cubeOutputStartPadding = taskOutputStartPadding + tilingData.layerOutputSize * c;
        for (uint32_t l = 0; l < tilingData.inputShape[DIM_1]; l++) {  // 对于每个要处理的layer
            uint64_t layerInputStart = cubeInputStart + tilingData.layerInputSize * l;
            uint64_t layerOutputStart;
            bool isPadding;
            GetPaddingOutputPos(cubeOutputStart, cubeOutputStartPadding, l, layerOutputStart, isPadding);
            
            for (auto k = 0; k < tilingData.leftSize; k++) {  // 对于每一行
                uint64_t lineInputStart = layerInputStart + tilingData.topSize * k;
                uint64_t lineOutputStart = layerOutputStart + tilingData.outputShape[DIM_3] * k;
                for (auto i = 0; i < innerTiling.tileCount; i++) {  // 待处理的轮区（满）
                    uint64_t yieldInputStart = lineInputStart + tilingData.moveTensorSize * i;  // 轮区起始序号
                    uint64_t yieldOutputStart = lineOutputStart + tilingData.moveTensorSize * i;  // 轮区输出起始序号
                    uint32_t dataLength = tilingData.moveTensorSize;  // 每行的搬运长度
                    ProcessInnerYield(yieldInputStart, dataLength, yieldOutputStart, 1, isPadding);
                }
                if (innerTiling.additionalCount) {  // 待处理的轮区（不满）
                    uint64_t yieldInputStart = lineInputStart + tilingData.moveTensorSize * innerTiling.tileCount;  // 轮区起始序号
                    uint64_t yieldOutputStart = lineOutputStart + tilingData.moveTensorSize * innerTiling.tileCount;  // 轮区输出起始序号
                    uint32_t dataLength = innerTiling.additionalCount;  // 每行的搬运长度
                    ProcessInnerYield(yieldInputStart, dataLength, yieldOutputStart, 1, isPadding);
                }
            }
        }
    }
}

__aicore__ inline void MoveInnerCaseSmall(EdgeTiling &innerTiling)
{
    uint64_t taskInputStart;
    uint64_t taskOutputStart;
    uint64_t taskOutputStartPadding;
    GetStartPosInner(taskInputStart, taskOutputStart, taskOutputStartPadding);
    uint32_t dataLength = tilingData.innerRowLength;  // 每行的搬运长度

    for (auto c = 0; c < cubeNumCurrentCore; c++) {  // 对于每个要处理的cube
        uint64_t cubeInputStart = taskInputStart + tilingData.cubeInputSize * c;
        uint64_t cubeOutputStart = taskOutputStart + tilingData.cubeOutputSize * c;
        uint64_t cubeOutputStartPadding = taskOutputStartPadding + tilingData.layerOutputSize * c;
        for (uint32_t l = 0; l < tilingData.inputShape[DIM_1]; l++) {  // 对于每个要处理的layer
            uint64_t layerInputStart = cubeInputStart + tilingData.layerInputSize * l;
            uint64_t layerOutputStart;
            bool isPadding;
            GetPaddingOutputPos(cubeOutputStart, cubeOutputStartPadding, l, layerOutputStart, isPadding);

            for (auto i = 0; i < innerTiling.tileCount; i++) {  // 待处理的轮区（满）
                uint64_t yieldInputStart = layerInputStart + innerTiling.edgeCount * tilingData.topSize * i;  // 轮区起始序号
                uint64_t yieldOutputStart = layerOutputStart + innerTiling.edgeCount * tilingData.outputShape[DIM_3] * i;  // 轮区输出起始序号
                uint16_t repeatTimes = innerTiling.edgeCount;  // 要搬几行
                ProcessInnerYield(yieldInputStart, dataLength, yieldOutputStart, repeatTimes, isPadding);
            }
            if (innerTiling.additionalCount) {  // 待处理的轮区（不满）
                uint64_t yieldInputStart = layerInputStart + \
                                            innerTiling.edgeCount * tilingData.topSize * innerTiling.tileCount;  // 轮区起始序号
                uint64_t yieldOutputStart = layerOutputStart + \
                                            innerTiling.edgeCount * tilingData.outputShape[DIM_3] * innerTiling.tileCount;  // 轮区输出起始序号
                uint16_t repeatTimes = innerTiling.additionalCount;  // 要搬几行
                ProcessInnerYield(yieldInputStart, dataLength, yieldOutputStart, repeatTimes, isPadding);
            }
        }
    }
}

__aicore__ inline void CopyInMoveTop(uint64_t startPos, uint32_t dataLength)
{
    DataCopyExtParams copyParams{1, dataLength * 4U, 0, 0, 0};  // WS上数据为float
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(mLocal, ws[startPos], copyParams, padParams);
}

__aicore__ inline void CopyOutMoveTop(uint64_t startPos, uint32_t dataLength)
{
    DataCopyExtParams copyParams{1, dataLength * dtypeByteSize, 0, 0, 0};
    if constexpr (isB16) {
        SetFlag<HardEvent::MTE2_V>(evtIDMTE2ToV);  // 计算等搬入
        WaitFlag<HardEvent::MTE2_V>(evtIDMTE2ToV);
        Cast(mLocalB16Little, mLocal, RoundMode::CAST_RINT, dataLength);

        SetFlag<HardEvent::V_MTE3>(evtIDVToMTE3);  // 搬出等计算
        WaitFlag<HardEvent::V_MTE3>(evtIDVToMTE3);
        DataCopyPad(zGm[startPos], mLocalB16Little, copyParams);
    } else {
        DataCopyPad(zGm[startPos], mLocal, copyParams);
    }
}

__aicore__ inline void CopyOutMoveTopPadding(uint64_t startPos, uint32_t dataLength)
{
    DataCopyExtParams copyParams{1, dataLength * 4U, 0, 0, 0};  // WS上数据为float
    PipeBarrier<PIPE_ALL>();
    SetAtomicAdd<float>();
    DataCopyPad(ws[startPos], mLocal, copyParams);
    SetAtomicNone();
}

__aicore__ inline void ProcessMoveTopYield(uint64_t yieldStart, uint32_t dataLength, uint64_t outYieldStart,
                                            bool isPadding)
{
    SetFlag<HardEvent::MTE3_MTE2>(evtIDMTE3ToMTE2);  // 搬入等搬出
    WaitFlag<HardEvent::MTE3_MTE2>(evtIDMTE3ToMTE2);
    // 搬入一轮
    CopyInMoveTop(yieldStart, dataLength);

    SetFlag<HardEvent::MTE2_MTE3>(evtIDMTE2ToMTE3);  // 搬出等搬入
    WaitFlag<HardEvent::MTE2_MTE3>(evtIDMTE2ToMTE3);
    // 搬出一轮
    if (isPadding) {
        CopyOutMoveTopPadding(outYieldStart, dataLength);
    } else {
        CopyOutMoveTop(outYieldStart, dataLength);
    }
}

__aicore__ inline void MoveTop(bool bottomMode=false)
{
    uint64_t taskInputStart = GetBlockIdx() * tilingData.cubeNumEachCore * tilingData.totalTopInputSizeEachCube + \
                                tilingData.paddings[DIM_0] + \
                                (bottomMode ? tilingData.topResultSize : 0);  // 本核ws输入起始序号
    uint64_t taskOutputStart = GetBlockIdx() * tilingData.cubeNumEachCore * tilingData.cubeOutputSize + \
                                (bottomMode ? (tilingData.layerOutputSize - tilingData.outputShape[DIM_3]) : 0);  // 本核GM输出起始序号
    uint64_t taskOutputStartPadding = GetBlockIdx() * tilingData.cubeNumEachCore * tilingData.layerOutputSize + \
                                        (bottomMode ? (tilingData.layerOutputSize - tilingData.outputShape[DIM_3]) : 0) + \
                                        (tilingData.topResultSize + tilingData.leftResultSize) * 2;  // 本核WS输出起始序号(padding)

    for (auto c = 0; c < cubeNumCurrentCore; c++) {  // 对于每个要处理的cube
        uint64_t cubeInputStart = taskInputStart + (tilingData.inputShape[DIM_1] * tilingData.topSize) * c;
        uint64_t cubeOutputStart = taskOutputStart + tilingData.cubeOutputSize * c;
        uint64_t cubeOutputStartPadding = taskOutputStartPadding + tilingData.layerOutputSize * c;
        for (uint32_t l = 0; l < tilingData.inputShape[DIM_1]; l++) {  // 对于每个要处理的layer
            uint64_t layerInputStart = cubeInputStart + tilingData.topSize * l;
            uint64_t layerOutputStart;
            bool isPadding;
            GetPaddingOutputPos(cubeOutputStart, cubeOutputStartPadding, l, layerOutputStart, isPadding);
            
            uint32_t repeatTimes = tilingData.outputShape[DIM_3] / tilingData.moveTensorSize;  // 本edge要搬几次
            uint32_t lastCount = tilingData.outputShape[DIM_3] % tilingData.moveTensorSize;
            for (auto i = 0; i < repeatTimes ; i++) {  // 待处理的轮区
                uint32_t dataLength = tilingData.moveTensorSize;
                uint64_t yieldInputStart = layerInputStart + tilingData.moveTensorSize * i;  // 轮区起始序号
                uint64_t yieldOutputStart = layerOutputStart + tilingData.moveTensorSize * i;  // 轮区输出起始序号
                ProcessMoveTopYield(yieldInputStart, dataLength, yieldOutputStart, isPadding);
            }
            if (lastCount) {
                uint32_t dataLength = lastCount;
                uint64_t yieldInputStart = layerInputStart + tilingData.moveTensorSize * repeatTimes;  // 轮区起始序号
                uint64_t yieldOutputStart = layerOutputStart + tilingData.moveTensorSize * repeatTimes;  // 轮区输出起始序号
                ProcessMoveTopYield(yieldInputStart, dataLength, yieldOutputStart, isPadding);
            }
        }
    }
}

__aicore__ inline void CopyInMoveLeft(uint64_t startPos, uint16_t dataLength, bool isPadding)
{
    DataCopyExtParams copyParams{dataLength, 4U, 0, (isB16 && !isPadding) ? 1U : 0U, 0};  // WS上数据为float；如果为b16直搬，留出一个32Block以供cast
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(mLocal, ws[startPos], copyParams, padParams);
}

__aicore__ inline void CopyOutMoveLeft(uint64_t startPos, uint16_t dataLength)
{
    DataCopyExtParams copyParams{dataLength, dtypeByteSize, 0,
                                    (tilingData.outputShape[DIM_3] - 1) * dtypeByteSize, 0};
    if constexpr (isB16) {
        SetFlag<HardEvent::MTE2_V>(evtIDMTE2ToV);  // 计算等搬入
        WaitFlag<HardEvent::MTE2_V>(evtIDMTE2ToV);
        Cast(mLocalB16Little, mLocal, RoundMode::CAST_RINT, dataLength * 16U);  // 2块32Block包含16个float

        SetFlag<HardEvent::V_MTE3>(evtIDVToMTE3);  // 搬出等计算
        WaitFlag<HardEvent::V_MTE3>(evtIDVToMTE3);
        DataCopyPad(zGm[startPos], mLocalB16Little, copyParams);
    } else {
        DataCopyPad(zGm[startPos], mLocal, copyParams);
    }
}

__aicore__ inline void CopyOutMoveLeftPadding(uint64_t startPos, uint16_t dataLength)
{
    DataCopyExtParams copyParams{dataLength, 4U, 0,
                                    (tilingData.outputShape[DIM_3] - 1) * 4U, 0};
    PipeBarrier<PIPE_ALL>();
    SetAtomicAdd<float>();
    DataCopyPad(ws[startPos], mLocal, copyParams);
    SetAtomicNone();
}

__aicore__ inline void ProcessMoveLeftYield(uint64_t yieldStart, uint16_t dataLength, uint64_t outYieldStart,
                                            bool isPadding)
{
    SetFlag<HardEvent::MTE3_MTE2>(evtIDMTE3ToMTE2);  // 搬入等搬出
    WaitFlag<HardEvent::MTE3_MTE2>(evtIDMTE3ToMTE2);
    // 搬入一轮
    CopyInMoveLeft(yieldStart, dataLength, isPadding);

    SetFlag<HardEvent::MTE2_MTE3>(evtIDMTE2ToMTE3);  // 搬出等搬入
    WaitFlag<HardEvent::MTE2_MTE3>(evtIDMTE2ToMTE3);
    // 搬出一轮
    if (isPadding) {
        CopyOutMoveLeftPadding(outYieldStart, dataLength);
    } else {
        CopyOutMoveLeft(outYieldStart, dataLength);
    }
}

__aicore__ inline void MoveLeft(bool rightMode=false)
{
    uint64_t taskInputStart = tilingData.topResultSize * 2 + (rightMode ? tilingData.leftResultSize : 0) + \
                                GetBlockIdx() * tilingData.cubeNumEachCore * tilingData.totalLeftInputSizeEachCube;  // 本核ws输入起始序号
    uint64_t taskOutputStart = GetBlockIdx() * tilingData.cubeNumEachCore * tilingData.cubeOutputSize + \
                                (tilingData.paddings[DIM_2] ? tilingData.outputShape[DIM_3] : 0) + \
                                (rightMode ? (tilingData.outputShape[DIM_3] - 1) : 0);  // 本核GM输出起始序号
    uint64_t taskOutputStartPadding = GetBlockIdx() * tilingData.cubeNumEachCore * tilingData.layerOutputSize + \
                                        (tilingData.paddings[DIM_2] ? tilingData.outputShape[DIM_3] : 0) + \
                                        (rightMode ? (tilingData.outputShape[DIM_3] - 1) : 0) + \
                                        (tilingData.topResultSize + tilingData.leftResultSize) * 2;  // 本核WS输出起始序号(padding)

    for (auto c = 0; c < cubeNumCurrentCore; c++) {  // 对于每个要处理的cube
        uint64_t cubeInputStart = taskInputStart + (tilingData.inputShape[DIM_1] * tilingData.leftSize) * c;
        uint64_t cubeOutputStart = taskOutputStart + tilingData.cubeOutputSize * c;
        uint64_t cubeOutputStartPadding = taskOutputStartPadding + tilingData.layerOutputSize * c;
        for (uint32_t l = 0; l < tilingData.inputShape[DIM_1]; l++) {  // 对于每个要处理的layer
            uint64_t layerInputStart = cubeInputStart + tilingData.leftSize * l;
            uint64_t layerOutputStart;
            bool isPadding;
            GetPaddingOutputPos(cubeOutputStart, cubeOutputStartPadding, l, layerOutputStart, isPadding);
            
            uint32_t maxMoveNum = (tilingData.moveTensorBlockNum > PARAM_LIMIT_4095) ? PARAM_LIMIT_4095 : tilingData.moveTensorBlockNum;
            uint32_t repeatTimes = tilingData.leftSize / maxMoveNum;  // 本edge要搬几次
            uint32_t lastCount = tilingData.leftSize % maxMoveNum;
            for (auto i = 0; i < repeatTimes ; i++) {  // 待处理的轮区
                uint32_t dataLength = maxMoveNum;
                uint64_t yieldInputStart = layerInputStart + maxMoveNum * i;  // 轮区起始序号
                uint64_t yieldOutputStart = layerOutputStart + \
                                            maxMoveNum * tilingData.outputShape[DIM_3] * i;  // 轮区输出起始序号
                ProcessMoveLeftYield(yieldInputStart, dataLength, yieldOutputStart, isPadding);
            }
            if (lastCount) {
                uint32_t dataLength = lastCount;
                uint64_t yieldInputStart = layerInputStart + maxMoveNum * repeatTimes;  // 轮区起始序号
                uint64_t yieldOutputStart = layerOutputStart + \
                                            maxMoveNum * tilingData.outputShape[DIM_3] * repeatTimes;  // 轮区输出起始序号
                ProcessMoveLeftYield(yieldInputStart, dataLength, yieldOutputStart, isPadding);
            }
        }
    }
}

__aicore__ inline void CopyInPaddingLayer(uint64_t startPos, uint32_t dataLength, uint32_t fullDataLength, uint16_t repeatTimes)
{
    DataCopyExtParams copyParams{repeatTimes, dataLength * 4U,  // 从WS搬入中间结果
                                    0,
                                    (isB16 && IsBlockNumOddPerDataLength(dataLength)) ? 1U : 0U,  // 保持一个repeat搬入后占32Block数量的偶数倍（仅b16场景）
                                    0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    if constexpr (isB16) {
        DataCopyPad(mLocal, ws[startPos], copyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(evtIDMTE2ToV);  // 计算等搬入
        WaitFlag<HardEvent::MTE2_V>(evtIDMTE2ToV);
        Cast(mLocalB16Little, mLocal, RoundMode::CAST_RINT, repeatTimes * fullDataLength);
        SetFlag<HardEvent::V_MTE3>(evtIDVToMTE3);  // 搬出等计算
        WaitFlag<HardEvent::V_MTE3>(evtIDVToMTE3);
    } else {
        DataCopyPad(mLocal, ws[startPos], copyParams, padParams);
    }
}

__aicore__ inline void CopyOutPaddingLayer(uint64_t startPos, uint32_t dataLength, uint16_t repeatTimes)
{
    DataCopyExtParams copyParams{repeatTimes, dataLength * dtypeByteSize,
                                    0, (tilingData.cubeOutputSize - tilingData.layerOutputSize) * dtypeByteSize, 0};
    if constexpr (isB16) {
        DataCopyPad(zGm[startPos], mLocalB16Little, copyParams);
    } else {
        DataCopyPad(zGm[startPos], mLocal, copyParams);
    }
}

__aicore__ inline void ProcessPaddingLayerYield(uint64_t yieldStart, uint32_t dataLength, uint64_t outYieldStart,
                                                uint16_t repeatTimes)
{
    SetFlag<HardEvent::MTE3_MTE2>(evtIDMTE3ToMTE2);  // 搬入等搬出
    WaitFlag<HardEvent::MTE3_MTE2>(evtIDMTE3ToMTE2);
    uint32_t fullDataLength = GetDataLength32BAligned(dataLength);  // 每面的计算长度（32B对齐）

    // 搬入一轮
    CopyInPaddingLayer(yieldStart, dataLength, fullDataLength, repeatTimes);

    SetFlag<HardEvent::MTE2_MTE3>(evtIDMTE2ToMTE3);  // 搬出等搬入
    WaitFlag<HardEvent::MTE2_MTE3>(evtIDMTE2ToMTE3);
    // 搬出一轮
    CopyOutPaddingLayer(outYieldStart, dataLength, repeatTimes);
}

__aicore__ inline void MovePaddingLayerCaseSmall(EdgeTiling &paddingLayerTiling, bool backMode=false)
{
    uint64_t taskInputStart = GetBlockIdx() * tilingData.cubeNumEachCore * tilingData.layerOutputSize + \
                                ((backMode) ? tilingData.layerOutputSize * tilingData.inputShape[DIM_0] : 0) + \
                                (tilingData.topResultSize + tilingData.leftResultSize) * 2;  // 本核WS输入起始序号
    uint64_t taskOutputStart = GetBlockIdx() * tilingData.cubeNumEachCore * tilingData.cubeOutputSize + \
                                ((backMode) ? tilingData.layerOutputSize * (tilingData.outputShape[DIM_1] - 1) : 0);  // 本核GM输出起始序号
    uint32_t dataLength = tilingData.layerOutputSize;  // 每行的搬运长度

    for (auto i = 0; i < paddingLayerTiling.tileCount; i++) {  // 待处理的轮区（满）
        uint64_t yieldInputStart = taskInputStart + paddingLayerTiling.edgeCount * tilingData.layerOutputSize * i;  // 轮区起始序号
        uint64_t yieldOutputStart = taskOutputStart + paddingLayerTiling.edgeCount * tilingData.cubeOutputSize * i;  // 轮区输出起始序号
        uint16_t repeatTimes = paddingLayerTiling.edgeCount;  // 要搬几个面
        ProcessPaddingLayerYield(yieldInputStart, dataLength, yieldOutputStart, repeatTimes);
    }
    if (paddingLayerTiling.additionalCount) {  // 待处理的轮区（不满）
        uint64_t yieldInputStart = taskInputStart + paddingLayerTiling.edgeCount * tilingData.layerOutputSize * paddingLayerTiling.tileCount;  // 轮区起始序号
        uint64_t yieldOutputStart = taskOutputStart + paddingLayerTiling.edgeCount * tilingData.cubeOutputSize * paddingLayerTiling.tileCount;  // 轮区输出起始序号
        uint16_t repeatTimes = paddingLayerTiling.additionalCount;  // 要搬几个面
        ProcessPaddingLayerYield(yieldInputStart, dataLength, yieldOutputStart, repeatTimes);
    }
}

__aicore__ inline void MovePaddingLayerCaseLarge(EdgeTiling &paddingLayerTiling, bool backMode=false)
{
    uint64_t taskInputStart = GetBlockIdx() * tilingData.cubeNumEachCore * tilingData.layerOutputSize + \
                                ((backMode) ? tilingData.layerOutputSize * tilingData.inputShape[DIM_0] : 0) + \
                                (tilingData.topResultSize + tilingData.leftResultSize) * 2;  // 本核WS输入起始序号
    uint64_t taskOutputStart = GetBlockIdx() * tilingData.cubeNumEachCore * tilingData.cubeOutputSize + \
                                ((backMode) ? tilingData.layerOutputSize * (tilingData.outputShape[DIM_1] - 1) : 0);  // 本核输出起始序号

    for (auto l = 0; l < cubeNumCurrentCore; l++) {  // 待处理padding面
        uint64_t layerStart = taskInputStart + tilingData.layerOutputSize * l;  // 面WS起始序号
        uint64_t outLayerStart = taskOutputStart + tilingData.cubeOutputSize * l;  // 面GM起始序号

        for (auto i = 0; i < paddingLayerTiling.tileCount; i++) {  // 待处理的轮区（满）
            uint32_t dataLength = tilingData.moveTensorSize;  // 每轮的搬运长度
            uint64_t yieldInputStart = layerStart + tilingData.moveTensorSize * i;  // 轮区起始序号
            uint64_t yieldOutputStart = outLayerStart + tilingData.moveTensorSize * i;  // 轮区输出起始序号
            uint16_t repeatTimes = 1;
            ProcessPaddingLayerYield(yieldInputStart, dataLength, yieldOutputStart, repeatTimes);
        }
        if (paddingLayerTiling.additionalCount) {  // 待处理的轮区（不满）
            uint32_t dataLength = paddingLayerTiling.additionalCount;  // 每轮的搬运长度
            uint64_t yieldInputStart = layerStart + tilingData.moveTensorSize * paddingLayerTiling.tileCount;  // 轮区起始序号
            uint64_t yieldOutputStart = outLayerStart + tilingData.moveTensorSize * paddingLayerTiling.tileCount;  // 轮区输出起始序号
            uint16_t repeatTimes = 1;
            ProcessPaddingLayerYield(yieldInputStart, dataLength, yieldOutputStart, repeatTimes);
        }
    }
}

// —— —— —— —— —— —— —— —— —— —— —— 处理WS（edge）和GM（inner）的搬出 —— —— —— —— —— —— —— —— —— —— —— —— ——

__aicore__ inline void MoveToOut()
{
    if (tilingData.leftSize != 0 && tilingData.innerRowLength != 0) {
        EdgeTiling innerTiling = tilingData.innerTiling;
        if (innerTiling.edgeCount) {
            MoveInnerCaseSmall(innerTiling);
        } else {
            MoveInnerCaseLarge(innerTiling);
        }
    }

    if (tilingData.paddings[DIM_2]) MoveTop();
    if (tilingData.paddings[DIM_3]) MoveTop(true);
    if (tilingData.leftSize) {
        if (tilingData.paddings[DIM_0]) MoveLeft();
        if (tilingData.paddings[DIM_1]) MoveLeft(true);
    }

    EdgeTiling paddingLayerTiling = (isLastCore) ? tilingData.paddingLayerTilingLastCore : tilingData.paddingLayerTiling;
    if (paddingLayerTiling.edgeCount) {
        // 一次处理多面
        if (tilingData.paddings[DIM_4]) MovePaddingLayerCaseSmall(paddingLayerTiling);
        if (tilingData.paddings[DIM_5]) MovePaddingLayerCaseSmall(paddingLayerTiling, true);
    } else {
        // 一次处理不到一面
        if (tilingData.paddings[DIM_4]) MovePaddingLayerCaseLarge(paddingLayerTiling);
        if (tilingData.paddings[DIM_5]) MovePaddingLayerCaseLarge(paddingLayerTiling, true);
    }
}

private:
TPipe pipe;
GlobalTensor<dtype> xGm;
GlobalTensor<dtype> zGm;
GlobalTensor<float> ws;
TBuf<TPosition::VECCALC> x1Buf;
TBuf<TPosition::VECCALC> x2Buf;
TBuf<TPosition::VECCALC> yBuf;
TBuf<TPosition::VECCALC> mBuf;
LocalTensor<float> x1Local;
LocalTensor<float> x2Local;
LocalTensor<float> yLocal;
LocalTensor<float> mLocal;
LocalTensor<dtype> x1LocalB16Big;
LocalTensor<dtype> yLocalB16Big;
LocalTensor<dtype> mLocalB16Little;
LocalTensor<dtype> mLocalB16Big;
TEventID evtIDMTE3ToMTE2;  // 搬入等搬出
TEventID evtIDVToMTE2;  // 搬入等计算
TEventID evtIDMTE2ToV;  // 计算等搬入
TEventID evtIDVToMTE3;  // 搬出等计算
TEventID evtIDMTE2ToMTE3;  // 搬出等搬入
uint32_t dtypeByteSize;
uint32_t eleNumPer32Block;
uint32_t cubeNumCurrentCore;
bool isLastCore;
PadV3GradReplicationTilingData &tilingData;
private:
static constexpr uint32_t PAIR = 2;
static constexpr uint32_t DIM_0 = 0;
static constexpr uint32_t DIM_1 = 1;
static constexpr uint32_t DIM_2 = 2;
static constexpr uint32_t DIM_3 = 3;
static constexpr uint32_t DIM_4 = 4;
static constexpr uint32_t DIM_5 = 5;
static constexpr uint32_t ELE_NUM_B32_PER_BLOCK = 8;
static constexpr uint32_t PARAM_LIMIT_4095 = 4095;
};

#endif  // PAD_V3_GRAD_REPLICATION_H