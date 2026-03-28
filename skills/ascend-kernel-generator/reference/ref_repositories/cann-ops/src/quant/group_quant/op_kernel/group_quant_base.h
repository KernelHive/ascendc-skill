/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file group_quant_base.h
 * \brief
 */
#ifndef ASCENDC_GROUP_QUANT
#define ASCENDC_GROUP_QUANT

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"

namespace GroupQuant {

using namespace AscendC;

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class GroupQuantBase {
public:
    __aicore__ inline GroupQuantBase() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR groupIndex, GM_ADDR offset, GM_ADDR y,
                                const GroupQuantTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const GroupQuantTilingData* tilingData);
    __aicore__ inline void Compute();
    __aicore__ inline void PerLoopCompute(int64_t hIdx);
    __aicore__ inline void CopyInOffset();
    __aicore__ inline void CopyInScale(int64_t scaleAddr);
    __aicore__ inline void CopyInX(int64_t xAddr, int64_t perSSize);
    __aicore__ inline void VecCompute(uint32_t dataCount, int64_t scaleExpandLoop, LocalTensor<int8_t> outputLocal);
    __aicore__ inline void CopyOutY(int64_t xAddr, int64_t perSSize, LocalTensor<int8_t> outputLocal);
    __aicore__ inline void LoadGrpIdxFirstBatch(LocalTensor<T3> grpIdxUb, LocalTensor<T3> xRowCntUb,
                                                int64_t grpIdxGmOffset, int64_t loadElems);
    __aicore__ inline void LoadGrpIdxMoreBatch(LocalTensor<T3> grpIdxUb, LocalTensor<T3> xRowCntUb,
                                               int64_t grpIdxGmOffset, int64_t loadElems);
    __aicore__ inline void CheckGrpIdxOneBatch(LocalTensor<T3> grpIdxUb, int64_t loadElems,
                                               LocalTensor<T3> xRowCntUb, LocalTensor<T3> grpIdxInt32TmpUb,
                                               LocalTensor<float> grpIdxFp32Ub, LocalTensor<float> grpIdxDiffUb,
                                               LocalTensor<float> grpIdxWorkLocalUb);
    __aicore__ inline void CheckGrpIdxLastValue(LocalTensor<T3> grpIdxUb);
    __aicore__ inline void CheckGrpIdxAllValue();
    __aicore__ inline void CheckGrpIdxAll(int64_t grpIdxGmOffset, int64_t loadElems, LocalTensor<T3> grpIdxUb,
                                          LocalTensor<T3> xRowCntUb, LocalTensor<T3> grpIdxInt32TmpUb,
                                          LocalTensor<float> grpIdxFp32Ub, LocalTensor<float> grpIdxDiffUb,
                                          LocalTensor<float> grpIdxWorkLocalUb);
    __aicore__ inline void CalcGrpCnt(LocalTensor<T3> grpIdxUb, int64_t loadElems, LocalTensor<T3> xRowCntUb);
    __aicore__ inline void WalkGrpIdx(LocalTensor<T3> grpIdxUb, LocalTensor<T3> xRowCntUb);
    __aicore__ inline void LookupExpertStateful(int64_t xStartRowCurCore, int64_t xRowNumCurCore);

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> tmpBuf1_;   // 每块tensor的大小 预期留32k
    TBuf<QuePosition::VECCALC> tmpBuf2_;
    TBuf<QuePosition::VECCALC> tmpBuf3_;
    TBuf<QuePosition::VECCALC> scaleBuf_;  // 装scale的大小 预期留32k
    TBuf<QuePosition::VECCALC> mulBuf_;    // 装x乘scale结果 预期留32k
    TBuf<QuePosition::VECCALC> offsetBuf_; // 预留256B
    TBuf<QuePosition::VECCALC> outBuf_;    // 输出的大小 预期8k

    TBuf<QuePosition::VECCALC> grpIdxBuf;
    TBuf<QuePosition::VECCALC> xRowCntBuf;
    TBuf<QuePosition::VECCALC> grpIdxInt32TmpBuf;
    TBuf<QuePosition::VECCALC> grpIdxFp32Buf;
    TBuf<QuePosition::VECCALC> grpIdxDiffBuf;
    TBuf<QuePosition::VECCALC> grpIdxWorkLocalBuf;

    // Inputs
    GlobalTensor<T1> gmX_;
    GlobalTensor<T2> gmScale_;
    GlobalTensor<T3> gmGrpIdx_;
    GlobalTensor<T4> gmOffset_;

    // Output
    GlobalTensor<int8_t> gmY_;

    // Constants
    constexpr static int64_t UB_TENSOE_BUF = 32 * 1024;
    constexpr static int64_t OFFEST_TENSOE_BUF = 256;
    constexpr static int64_t PEER_H = 8 * 1024;
    constexpr static int64_t BLOCK_SIZE = 32;
    constexpr static int64_t ALIGN_SIZE = 64;
    constexpr static int64_t INT4_NUMS_IN_INT8_SPACE = 2;
    constexpr static int64_t GROUP_INDEX_UB_SIZE = 1024;  // lookup expert index and x row counts

    int64_t blockIdx = 0;
    int64_t currExpertIdx = -1;  // response to current row of input_scale and current index of input_group_index
    int64_t xRowCntOfCurrExpert = 0;
    int64_t grpIdxUbCursor = 0;

    int64_t dimS_ = 0;
    int64_t dimE_ = 0;
    int64_t dimH_ = 0;
    int64_t hasOffset_ = 0;
    int64_t needCoreNum_ = 0;
    int64_t preCoreNum_ = 0;
    int64_t xRowNum_ = 0;
    int64_t xRowIndex_ = 0;
    int64_t xRowNumPreCore_ = 0;
    int64_t xRowNumPostCore_ = 0;
    int64_t calH = 0;
    int64_t calHBlock = 0;
    int64_t maxSSize = 1;
    float offset_ = 0;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5>
__aicore__ inline void GroupQuantBase<T1, T2, T3, T4, T5>::ParseTilingData(const GroupQuantTilingData* tilingData) {
    dimS_ = tilingData->dimS;
    dimE_ = tilingData->dimE;
    dimH_ = tilingData->dimH;
    hasOffset_ = tilingData->hasOffset;
    needCoreNum_ = tilingData->needCoreNum;
    preCoreNum_ = tilingData->preCoreNum;
    xRowNumPreCore_ = tilingData->xRowNumPreCore;
    xRowNumPostCore_ = tilingData->xRowNumPostCore;

    xRowNum_ = xRowNumPreCore_;
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
__aicore__ inline void GroupQuantBase<T1, T2, T3, T4, T5>::Init(GM_ADDR x,
                                                                GM_ADDR scale,
                                                                GM_ADDR groupIndex,
                                                                GM_ADDR offset,
                                                                GM_ADDR y,
                                                                const GroupQuantTilingData* tilingData) {
    blockIdx = GetBlockIdx();
    ParseTilingData(tilingData);  // initialize tiling data

    gmX_.SetGlobalBuffer((__gm__ T1 *)x);
    gmScale_.SetGlobalBuffer((__gm__ T2 *)scale);
    gmGrpIdx_.SetGlobalBuffer((__gm__ T3 *)groupIndex);
    gmOffset_.SetGlobalBuffer((__gm__ T4 *)offset);
    gmY_.SetGlobalBuffer((__gm__ int8_t *)y);

    pipe.InitBuffer(tmpBuf1_, UB_TENSOE_BUF);            // 32KB
    pipe.InitBuffer(tmpBuf2_, UB_TENSOE_BUF);
    pipe.InitBuffer(tmpBuf3_, UB_TENSOE_BUF);
    pipe.InitBuffer(scaleBuf_, UB_TENSOE_BUF);           // 32KB
    pipe.InitBuffer(mulBuf_, UB_TENSOE_BUF);             // 32KB
    pipe.InitBuffer(outBuf_, PEER_H);                    //  8KB
    if (hasOffset_) {
        pipe.InitBuffer(offsetBuf_, OFFEST_TENSOE_BUF);  // 256B
    }

    pipe.InitBuffer(grpIdxBuf, GROUP_INDEX_UB_SIZE);           // 1KB
    pipe.InitBuffer(xRowCntBuf, GROUP_INDEX_UB_SIZE);          // 1KB
    pipe.InitBuffer(grpIdxInt32TmpBuf, GROUP_INDEX_UB_SIZE);   // 1KB
    pipe.InitBuffer(grpIdxFp32Buf, GROUP_INDEX_UB_SIZE);       // 1KB
    pipe.InitBuffer(grpIdxDiffBuf, GROUP_INDEX_UB_SIZE);       // 1KB
    pipe.InitBuffer(grpIdxWorkLocalBuf, GROUP_INDEX_UB_SIZE);  // 1KB
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
__aicore__ inline void GroupQuantBase<T1, T2, T3, T4, T5>::LoadGrpIdxFirstBatch(LocalTensor<T3> grpIdxUb,
                                                                                LocalTensor<T3> xRowCntUb,
                                                                                int64_t grpIdxGmOffset,
                                                                                int64_t loadElems) {
    event_t eventIdSToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    SetFlag<HardEvent::S_MTE2>(eventIdSToMte2);
    WaitFlag<HardEvent::S_MTE2>(eventIdSToMte2);

    DataCopyExtParams copyParams0;
    copyParams0.blockCount = 1;
    copyParams0.blockLen = loadElems * sizeof(T3);
    copyParams0.srcStride = 0;
    copyParams0.dstStride = 0;
    DataCopyPadExtParams<T3> padParams0{false, 0, 0, 0};
    DataCopyPad(grpIdxUb, gmGrpIdx_[grpIdxGmOffset], copyParams0, padParams0);

    if (loadElems == 1) {
        DataCopyExtParams copyParams1;
        copyParams1.blockCount = 1;
        copyParams1.blockLen = sizeof(T3);
        copyParams1.srcStride = 0;
        copyParams1.dstStride = 0;
        DataCopyPadExtParams<T3> padParams1{true, 1, 0, 0};
        DataCopyPad(xRowCntUb, gmGrpIdx_[grpIdxGmOffset], copyParams1, padParams1);
    } else {
        DataCopyExtParams copyParams1;
        copyParams1.blockCount = 1;
        copyParams1.blockLen = (loadElems - 1) * sizeof(T3);
        copyParams1.srcStride = 0;
        copyParams1.dstStride = 0;
        DataCopyPadExtParams<T3> padParams1{true, 1, 0, 0};
        DataCopyPad(xRowCntUb, gmGrpIdx_[grpIdxGmOffset], copyParams1, padParams1);
    }
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
__aicore__ inline void GroupQuantBase<T1, T2, T3, T4, T5>::LoadGrpIdxMoreBatch(LocalTensor<T3> grpIdxUb,
                                                                               LocalTensor<T3> xRowCntUb,
                                                                               int64_t grpIdxGmOffset,
                                                                               int64_t loadElems) {
    event_t eventIdSToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    SetFlag<HardEvent::S_MTE2>(eventIdSToMte2);
    WaitFlag<HardEvent::S_MTE2>(eventIdSToMte2);

    DataCopyExtParams copyParams0;
    copyParams0.blockCount = 1;
    copyParams0.blockLen = loadElems * sizeof(T3);
    copyParams0.srcStride = 0;
    copyParams0.dstStride = 0;
    DataCopyPadExtParams<T3> padParams0{false, 0, 0, 0};
    DataCopyPad(grpIdxUb, gmGrpIdx_[grpIdxGmOffset], copyParams0, padParams0);

    DataCopyExtParams copyParams1;
    copyParams1.blockCount = 1;
    copyParams1.blockLen = loadElems * sizeof(T3);
    copyParams1.srcStride = 0;
    copyParams1.dstStride = 0;
    DataCopyPadExtParams<T3> padParams1{false, 0, 0, 0};
    DataCopyPad(xRowCntUb, gmGrpIdx_[grpIdxGmOffset - 1], copyParams1, padParams1);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
__aicore__ inline void GroupQuantBase<T1, T2, T3, T4, T5>::CheckGrpIdxOneBatch(LocalTensor<T3> grpIdxUb,
                                                                               int64_t loadElems,
                                                                               LocalTensor<T3> xRowCntUb,
                                                                               LocalTensor<T3> grpIdxInt32TmpUb,
                                                                               LocalTensor<float> grpIdxFp32Ub,
                                                                               LocalTensor<float> grpIdxDiffUb,
                                                                               LocalTensor<float> grpIdxWorkLocalUb) {
    // group_index values are in the range [0, S], and the sequence is non-decreasing
    if constexpr (IsSameType<T3, int32_t>::value) {  // group_index data type is int32_t
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        // compare with 0
        PipeBarrier<PIPE_V>();
        Cast(grpIdxFp32Ub, grpIdxUb, RoundMode::CAST_NONE, loadElems);
        PipeBarrier<PIPE_V>();
        ReduceMin(grpIdxDiffUb, grpIdxFp32Ub, grpIdxWorkLocalUb, loadElems, false);

        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);

        if (grpIdxDiffUb.GetValue(0) < 0.0f) {
            Trap(); // "input group_index value should be greater than or equal 0"
        }

        // compare with S
        // values are all greater than 0, so the result of subtraction does not flip
        PipeBarrier<PIPE_V>();
        Adds(grpIdxInt32TmpUb, grpIdxUb, static_cast<T3>(-1 * dimS_), loadElems);
        PipeBarrier<PIPE_V>();
        Cast(grpIdxFp32Ub, grpIdxInt32TmpUb, RoundMode::CAST_NONE, loadElems);
        PipeBarrier<PIPE_V>();
        ReduceMax(grpIdxDiffUb, grpIdxFp32Ub, grpIdxWorkLocalUb, loadElems, false);

        event_t eventVS1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS1);
        WaitFlag<HardEvent::V_S>(eventVS1);

        if (grpIdxDiffUb.GetValue(0) > 0.0f) {
            Trap(); // "input group_index value should be less than or equal x shape[0]"
        }

        // compare with previous value
        PipeBarrier<PIPE_V>();
        Sub(grpIdxInt32TmpUb, grpIdxUb, xRowCntUb, loadElems);
        PipeBarrier<PIPE_V>();
        Cast(grpIdxFp32Ub, grpIdxInt32TmpUb, RoundMode::CAST_NONE, loadElems);
        PipeBarrier<PIPE_V>();
        ReduceMin(grpIdxDiffUb, grpIdxFp32Ub, grpIdxWorkLocalUb, loadElems, false);
        PipeBarrier<PIPE_V>();

        event_t eventVS2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS2);
        WaitFlag<HardEvent::V_S>(eventVS2);

        if (grpIdxDiffUb.GetValue(0) < 0.0f) {
            Trap(); // "input group_index should be an non-decreasing sequence"
        }
    } else {                                          // group_index data type is int64_t
        event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);

        for (int64_t i = 0; i < loadElems; i++) {
            // "input group_index value should be greater than or equal 0"
            // "input group_index value should be less than or equal x shape[0]"
            // "input group_index should be an non-decreasing sequence"
            int64_t valueOfGroupIndex = grpIdxUb.GetValue(i);
            if ((valueOfGroupIndex < 0) || (valueOfGroupIndex > dimS_) || (valueOfGroupIndex - xRowCntUb.GetValue(i) < 0)) {
                Trap();
            }
        }
    }
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
__aicore__ inline void GroupQuantBase<T1, T2, T3, T4, T5>::CheckGrpIdxLastValue(LocalTensor<T3> grpIdxUb) {
    PipeBarrier<PIPE_ALL>();

    DataCopyExtParams copyParams0;
    copyParams0.blockCount = 1;
    copyParams0.blockLen = sizeof(T3);
    copyParams0.srcStride = 0;
    copyParams0.dstStride = 0;
    DataCopyPadExtParams<T3> padParams0{false, 0, 0, 0};
    DataCopyPad(grpIdxUb, gmGrpIdx_[dimE_ - 1], copyParams0, padParams0);

    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);

    int64_t lastValueOfGroupIndex = static_cast<int64_t>(grpIdxUb.GetValue(0));
    int64_t firstDimOfXShape = dimS_;
    if (lastValueOfGroupIndex != firstDimOfXShape) {
        Trap(); // "last data of group_index should be same as input x shape[0]"
    }
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
__aicore__ inline void GroupQuantBase<T1, T2, T3, T4, T5>::CheckGrpIdxAllValue() {
    LocalTensor<T3> grpIdxUb = grpIdxBuf.Get<T3>();
    if (dimE_ > 1) {
        LocalTensor<T3> xRowCntUb = xRowCntBuf.Get<T3>();
        LocalTensor<T3> grpIdxInt32TmpUb = grpIdxInt32TmpBuf.Get<T3>();
        LocalTensor<float> grpIdxFp32Ub = grpIdxFp32Buf.Get<float>();
        LocalTensor<float> grpIdxDiffUb = grpIdxDiffBuf.Get<float>();
        LocalTensor<float> grpIdxWorkLocalUb = grpIdxWorkLocalBuf.Get<float>();

        int64_t grpIdxBaseNum = GROUP_INDEX_UB_SIZE / sizeof(T3);

        int64_t grpIdxGmOffset = 0;
        int64_t loadElems = grpIdxBaseNum;
        if (loadElems > dimE_) {
            loadElems = dimE_;
        }
        LoadGrpIdxFirstBatch(grpIdxUb, xRowCntUb, grpIdxGmOffset, loadElems);
        CheckGrpIdxOneBatch(grpIdxUb, loadElems, xRowCntUb,
                            grpIdxInt32TmpUb, grpIdxFp32Ub, grpIdxDiffUb, grpIdxWorkLocalUb);

        if (dimE_ > grpIdxBaseNum) {
            int64_t loopTimes = dimE_ / grpIdxBaseNum;
            for (int64_t i = 1; i < loopTimes; ++i) {
                grpIdxGmOffset = i * grpIdxBaseNum;
                loadElems = grpIdxBaseNum;
                CheckGrpIdxAll(grpIdxGmOffset, loadElems, grpIdxUb, xRowCntUb, 
                               grpIdxInt32TmpUb, grpIdxFp32Ub, grpIdxDiffUb, grpIdxWorkLocalUb);
            }

            int64_t tailE = dimE_ % grpIdxBaseNum;
            if (tailE > 0) {
                grpIdxGmOffset = dimE_ / grpIdxBaseNum * grpIdxBaseNum;
                loadElems = tailE;
                CheckGrpIdxAll(grpIdxGmOffset, loadElems, grpIdxUb, xRowCntUb, 
                               grpIdxInt32TmpUb, grpIdxFp32Ub, grpIdxDiffUb, grpIdxWorkLocalUb);
            }
        }
    }
    CheckGrpIdxLastValue(grpIdxUb);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
__aicore__ inline void GroupQuantBase<T1, T2, T3, T4, T5>::CheckGrpIdxAll(int64_t grpIdxGmOffset,
                                                                          int64_t loadElems,
                                                                          LocalTensor<T3> grpIdxUb,
                                                                          LocalTensor<T3> xRowCntUb,
                                                                          LocalTensor<T3> grpIdxInt32TmpUb,
                                                                          LocalTensor<float> grpIdxFp32Ub,
                                                                          LocalTensor<float> grpIdxDiffUb,
                                                                          LocalTensor<float> grpIdxWorkLocalUb) {
    if constexpr (IsSameType<T3, int32_t>::value) {   // group_index data type is int32_t
        event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    } else {
        event_t eventIdSToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
        SetFlag<HardEvent::S_MTE2>(eventIdSToMte2);
        WaitFlag<HardEvent::S_MTE2>(eventIdSToMte2);
    }
    LoadGrpIdxMoreBatch(grpIdxUb, xRowCntUb, grpIdxGmOffset, loadElems);
    CheckGrpIdxOneBatch(grpIdxUb, loadElems, xRowCntUb,
                        grpIdxInt32TmpUb, grpIdxFp32Ub, grpIdxDiffUb, grpIdxWorkLocalUb);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
__aicore__ inline void GroupQuantBase<T1, T2, T3, T4, T5>::CalcGrpCnt(LocalTensor<T3> grpIdxUb,
                                                                      int64_t loadElems,
                                                                      LocalTensor<T3> xRowCntUb) {
    if constexpr (IsSameType<T3, int32_t>::value) {  // group_index data type is int32_t
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        Sub(xRowCntUb, grpIdxUb, xRowCntUb, loadElems);

        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
    } else {                                          // group_index data type is int64_t
        event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);

        for (int64_t i = 0; i < loadElems; i++) {
            xRowCntUb.SetValue(i, grpIdxUb.GetValue(i) - xRowCntUb.GetValue(i));
        }
    }
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
__aicore__ inline void GroupQuantBase<T1, T2, T3, T4, T5>::WalkGrpIdx(LocalTensor<T3> grpIdxUb,
                                                                      LocalTensor<T3> xRowCntUb) {
    if (currExpertIdx < 0) {
        // load the first part of group_index data
        int64_t grpIdxGmOffset = 0;
        int64_t loadElems = GROUP_INDEX_UB_SIZE / sizeof(T3);
        if (loadElems > dimE_) {
            loadElems = dimE_;
        }
        LoadGrpIdxFirstBatch(grpIdxUb, xRowCntUb, grpIdxGmOffset, loadElems);
        CalcGrpCnt(grpIdxUb, loadElems, xRowCntUb);

        currExpertIdx = 0;
        grpIdxUbCursor = 0;
    } else if (grpIdxUbCursor + 1 == GROUP_INDEX_UB_SIZE / sizeof(T3)) {
        // load more data of group_index
        int64_t grpIdxGmOffset = currExpertIdx + 1;
        int64_t loadElems = GROUP_INDEX_UB_SIZE / sizeof(T3);
        if (grpIdxGmOffset + loadElems > dimE_) {
            loadElems = dimE_ - grpIdxGmOffset;
        }
        LoadGrpIdxMoreBatch(grpIdxUb, xRowCntUb, grpIdxGmOffset, loadElems);
        CalcGrpCnt(grpIdxUb, loadElems, xRowCntUb);

        currExpertIdx = currExpertIdx + 1;
        grpIdxUbCursor = 0;
    } else {
        currExpertIdx = currExpertIdx + 1;
        grpIdxUbCursor = grpIdxUbCursor + 1;
    }
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
__aicore__ inline void GroupQuantBase<T1, T2, T3, T4, T5>::LookupExpertStateful(int64_t xStartRowCurrCore,
                                                                                int64_t xRowNumCurrCore) {
    // This function is stateful.
    // When called for the first time, currExpertIdx shoulde be initialzed to -1.
    // When called again, the value of currExpertIdx updated last time is used.

    // 处理专家策略，将专家策略中的E的地址以及对应多少行进行重排
    LocalTensor<T3> grpIdxUb = grpIdxBuf.Get<T3>();
    LocalTensor<T3> xRowCntUb = xRowCntBuf.Get<T3>();

    while (currExpertIdx < dimE_) {
        WalkGrpIdx(grpIdxUb, xRowCntUb);

        int64_t grpCumSum = static_cast<int64_t>(grpIdxUb.GetValue(grpIdxUbCursor));
        int64_t grpCnt = static_cast<int64_t>(xRowCntUb.GetValue(grpIdxUbCursor));
        if (grpCumSum > xStartRowCurrCore && grpCnt > 0) {
            int64_t preXptEndRow = grpCumSum - grpCnt;
            if (preXptEndRow < xStartRowCurrCore) {
                if (xStartRowCurrCore + xRowNumCurrCore < grpCumSum) {
                    // ----|----------|-----------------|------------------|---------------|---------->(xRow)
                    //     0    preExpertEnd     xStartRowCurrCore   xEndRowCurrCore   grpCumSum
                    xRowCntOfCurrExpert = xRowNumCurrCore;
                } else {
                    // ----|----------|-----------------|------------------|---------------|---------->(xRow)
                    //     0    preExpertEnd     xStartRowCurrCore     grpCumSum     xEndRowCurrCore
                    xRowCntOfCurrExpert = grpCumSum - xStartRowCurrCore;
                }
            } else {
                if (xStartRowCurrCore + xRowNumCurrCore < grpCumSum) {
                    // ----|----------|-----------------|------------------|---------------|---------->(xRow)
                    //     0   xStartRowCurrCore  preExpertEnd      xEndRowCurrCore    grpCumSum
                    xRowCntOfCurrExpert = xStartRowCurrCore + xRowNumCurrCore - preXptEndRow;
                } else {
                    // ----|----------|-----------------|------------------|---------------|---------->(xRow)
                    //     0   xStartRowCurrCore  preExpertEnd         grpCumSum     xEndRowCurrCore
                    xRowCntOfCurrExpert = grpCnt;
                }
            }
            break;
        }
    }
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
__aicore__ inline void GroupQuantBase<T1, T2, T3, T4, T5>::CopyInOffset()
{
    if (hasOffset_) {
        LocalTensor<T4> offsetLocal = offsetBuf_.Get<T4>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = sizeof(T4);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<T4> padParams = {false, 0, 0, 0};
        DataCopyPad(offsetLocal, gmOffset_, copyParams, padParams);

        event_t eventIdMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);

        if constexpr (IsSameType<DTYPE_SCALE, bfloat16_t>::value) {
            offset_ = ToFloat(offsetLocal.GetValue(0));
        } else {
            offset_ = static_cast<float>(*((__ubuf__ T4 *)offsetLocal.GetPhyAddr()));
        }
    }
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
__aicore__ inline void GroupQuantBase<T1, T2, T3, T4, T5>::CopyInX(int64_t xAddr, int64_t perSSize)
{
    LocalTensor<T1> tmpLocal = tmpBuf1_.Get<T1>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = (uint16_t)perSSize;
    copyParams.blockLen = calH * sizeof(T1);
    copyParams.srcStride = (dimH_ - calH) * sizeof(T1);
    copyParams.dstStride = (calHBlock - calH) * sizeof(T1) / BLOCK_SIZE;
    DataCopyPadExtParams<T1> padParams = {false, 0, 0, 0};
    DataCopyPad(tmpLocal, gmX_[xAddr], copyParams, padParams);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
__aicore__ inline void GroupQuantBase<T1, T2, T3, T4, T5>::CopyInScale(int64_t scaleAddr)
{
    LocalTensor<T2> scaleLocal = scaleBuf_.Get<T2>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = calH * sizeof(T2);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T2> padParams = {false, 0, 0, 0};
    DataCopyPad(scaleLocal, gmScale_[scaleAddr], copyParams, padParams);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
__aicore__ inline void GroupQuantBase<T1, T2, T3, T4, T5>::VecCompute(uint32_t dataCount, int64_t scaleExpandLoop,
                                                                      LocalTensor<int8_t> outputLocal)
{
    // 计算过程是 (float) x * scale + offset
    event_t eventMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    LocalTensor<T2> oriScaleLocal = scaleBuf_.Get<T2>();
    LocalTensor<T1> tmpLocal1 = tmpBuf1_.Get<T1>();
    LocalTensor<float> tmpLocal2 = tmpBuf2_.Get<float>();
    LocalTensor<float> tmpLocal3 = tmpBuf3_.Get<float>();

    LocalTensor<float> xLocal;
    LocalTensor<float> scaleLocal;
    LocalTensor<float> mulLocal = mulBuf_.Get<float>();;

    // 将x cast成fp32 放在xLocal
    if constexpr (IsSameType<T1, float>::value) {
        xLocal = tmpLocal1;
    } else {
        Cast(tmpLocal2, tmpLocal1, RoundMode::CAST_NONE, dataCount);
        xLocal = tmpLocal2;
    }

    // 将scale cast成fp32 放在scaleLocal
    if constexpr (IsSameType<T2, float>::value) {
        scaleLocal = oriScaleLocal;
    } else {
        PipeBarrier<PIPE_V>();
        Cast(tmpLocal3, oriScaleLocal, RoundMode::CAST_NONE, calHBlock);
        scaleLocal = tmpLocal3;
    }

    // 将scale进行broadcast
    if (scaleExpandLoop > 1) {
        for (int64_t eIdx = 0; eIdx < scaleExpandLoop; eIdx++) {
            int64_t scaleAddr = eIdx * calHBlock;
            PipeBarrier<PIPE_V>();
            Muls(scaleLocal[scaleAddr], scaleLocal, 1.0f, calHBlock);
        }
    }

    // scale乘x
    PipeBarrier<PIPE_V>();
    Mul(mulLocal, xLocal, scaleLocal, dataCount);
    PipeBarrier<PIPE_V>();

    event_t eventVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventVToMte2);
    WaitFlag<HardEvent::V_MTE2>(eventVToMte2);

    if (hasOffset_) {
        Adds(mulLocal, mulLocal, (float)offset_, dataCount);
        PipeBarrier<PIPE_V>();
    }

    // float->int32
    Cast(mulLocal.ReinterpretCast<int32_t>(), mulLocal, RoundMode::CAST_RINT, dataCount);
    PipeBarrier<PIPE_V>();

    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();
    
    // int32->fp16
    Cast(mulLocal.ReinterpretCast<half>(), mulLocal.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, dataCount);
    PipeBarrier<PIPE_V>();

    // fp16->in8/int4
    if (ORIG_DTYPE_Y == DT_INT4) {
        Cast(outputLocal.ReinterpretCast<int4b_t>(),
        mulLocal.ReinterpretCast<half>(),
        RoundMode::CAST_RINT, dataCount);
        return;
    }
    Cast(outputLocal, mulLocal.ReinterpretCast<half>(), RoundMode::CAST_RINT, dataCount);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
__aicore__ inline void GroupQuantBase<T1, T2, T3, T4, T5>::CopyOutY(int64_t xAddr, int64_t perSSize,
                                                                    LocalTensor<int8_t> outputLocal)
{
    event_t eventVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventVToMte3);

    int64_t yLenReal = calH;
    if (ORIG_DTYPE_Y == DT_INT4) {
        yLenReal = yLenReal / INT4_NUMS_IN_INT8_SPACE;
    }

    DataCopyExtParams copyParams;
    copyParams.blockCount = (uint16_t)perSSize;
    copyParams.blockLen = yLenReal *  sizeof(int8_t);

    if (ORIG_DTYPE_Y == DT_INT4) {
        copyParams.srcStride = (calHBlock / INT4_NUMS_IN_INT8_SPACE - yLenReal) * sizeof(int8_t) / BLOCK_SIZE;
        copyParams.dstStride = (dimH_ / INT4_NUMS_IN_INT8_SPACE - yLenReal) * sizeof(int8_t);
        DataCopyPad(gmY_[xAddr / INT4_NUMS_IN_INT8_SPACE], outputLocal, copyParams);
    } else {
        copyParams.srcStride = (calHBlock - yLenReal) * sizeof(int8_t) / BLOCK_SIZE;
        copyParams.dstStride = (dimH_ - yLenReal) * sizeof(int8_t);
        DataCopyPad(gmY_[xAddr], outputLocal, copyParams);
    }

    event_t eventMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventMte3ToV);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
__aicore__ inline void GroupQuantBase<T1, T2, T3, T4, T5>::PerLoopCompute(int64_t hIdx)
{
    // 状态复位
    currExpertIdx = -1;
    int64_t cumSum = 0;
    int64_t sSizePerCore = 0; // 每个核x[S,H]中S维度在PerLoopCompute过程中计算到的行数

    while (cumSum < xRowNum_) {
        LookupExpertStateful(xRowIndex_, xRowNum_);
        cumSum += xRowCntOfCurrExpert;

        int64_t eSize = currExpertIdx;  // 这个专家策略在scale[E,H]中的行数 $$
        int64_t hNumPerE = xRowCntOfCurrExpert;  // 这个专家策略对应的行数 $$
        int64_t loopS = Ceil(hNumPerE, maxSSize); // 这个专家策略需要几次循环做完
        int64_t perSSize = maxSSize; // 一次处理的S维度的行数
        int64_t lastSSize = hNumPerE - (loopS - 1) * maxSSize;
        int64_t scaleAddr = eSize * dimH_ + hIdx * PEER_H; // scale的起始地址偏移

        event_t eventIdSToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
        SetFlag<HardEvent::S_MTE2>(eventIdSToMte2);
        WaitFlag<HardEvent::S_MTE2>(eventIdSToMte2);

        CopyInScale(scaleAddr);
        for (int64_t loopSIdx = 0; loopSIdx < loopS; loopSIdx ++) {
            if (loopSIdx == loopS - 1) {
                perSSize = lastSSize;
            }
            // x的起始地址偏移
            int64_t xAddr = (xRowIndex_ + sSizePerCore) * dimH_ + hIdx * PEER_H;
            // 一次搬运x的数据量 最大8k个数
            uint32_t dataCount = perSSize * calHBlock;
            CopyInX(xAddr, perSSize);

            LocalTensor<int8_t> outputLocal = outBuf_.Get<int8_t>();
            VecCompute(dataCount, perSSize, outputLocal);

            CopyOutY(xAddr, perSSize, outputLocal);
            sSizePerCore += perSSize;
        }
    }
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
__aicore__ inline void GroupQuantBase<T1, T2, T3, T4, T5>::Compute()
{
    // x=[S,H]维度，对H进行切分 一次循环最大处理8192个数
    // calH 是处理的H的大小
    int64_t loopH = Ceil(dimH_, PEER_H);
    calH = PEER_H;
    int64_t lastH = dimH_ - (loopH - 1) * PEER_H;

    // 搬入offset
    CopyInOffset();

    // 切H的循环
    for (int64_t loopIdx = 0; loopIdx < loopH; loopIdx ++) {
        if (loopIdx == loopH - 1) {
            calH = lastH;
        }
        // 64对齐之后的大小
        calHBlock = Ceil(calH, ALIGN_SIZE) * ALIGN_SIZE;
        if (calHBlock != 0) {
            // maxSSize是S维度可以处理的最大行数
            maxSSize = UB_TENSOE_BUF / sizeof(float) / calHBlock;
        }
        PerLoopCompute(loopIdx);
    }
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
__aicore__ inline void GroupQuantBase<T1, T2, T3, T4, T5>::Process() {
    if (blockIdx >= needCoreNum_) {
        return;
    }

    // kernel support empty tensor
    if (dimS_ == 0 || dimH_ == 0) {
        return;
    }
    int64_t firstDimOfScaleShape = dimE_;
    if (firstDimOfScaleShape <= 0) {
      Trap(); // "op GroupQuant no support for scale shape[0] is 0"
    }


    // Need to check all group_index value on each core to defense exception when group_index being used.
    // If group_index is int32, calculate by vector instructions for high performance.
    // If group_index is int64, calculate by scalar calculation.
    CheckGrpIdxAllValue();
    PipeBarrier<PIPE_ALL>();

    // preCoreNum_个核处理xRowNumPreCore_行，preCoreNum_之后的核处理xRowNumPostCore_行
    if (blockIdx >= preCoreNum_) {
        xRowNum_ = xRowNumPostCore_;
    }
    // xRowIndex_是每个核中 S维度起始的行号
    xRowIndex_ = (blockIdx < preCoreNum_) ? blockIdx * xRowNumPreCore_ :
        preCoreNum_ * xRowNumPreCore_ + (blockIdx - preCoreNum_) * xRowNumPostCore_;

    Compute();
}

}  // namespace GroupQuant
#endif  // ASCENDC_GROUP_QUANT
