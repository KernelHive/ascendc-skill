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
 * \file layer_norm_grad_v3_single_read.h
 * \brief
 */

#ifndef LAYER_NORM_GRAD_V3
#define LAYER_NORM_GRAD_V3

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace LayerNormGradV3 {
using namespace AscendC;

constexpr int32_t DB_BUFFER_NUM = 2;
constexpr int32_t BLOCK_BYTES = 32;
constexpr int32_t B32_BLOCK_SIZE = 8;
constexpr int32_t B32_REPEAT_SIZE = 64;
constexpr int32_t CONSTANT_TWO = 2;
constexpr int32_t CONSTANT_EIGHT = 8;
constexpr int32_t MAX_REPEAT_STRIDE = 255;
constexpr int32_t TMP_BUFFER_SIZE_0 = 16;
constexpr int32_t TMP_BUFFER_SIZE_1 = 128;
constexpr int32_t COUNT_NUM = 64;

template <typename T, typename U, bool isDeterministic>
class LayerNormGradV3SingleRead {
public:
    __aicore__ inline LayerNormGradV3SingleRead(){};
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean, GM_ADDR gamma, GM_ADDR pdX,
        GM_ADDR pdGamma, GM_ADDR pdBeta, GM_ADDR workspace, const LayerNormGradV3TilingDataSingleRead *tilingData)
    {
        // init gm inputs
        int64_t curRowsNum =
            (GetBlockIdx() != tilingData->blockNum - 1) ? tilingData->blockFormer : tilingData->blockTail;
        int64_t formerBlockLength = tilingData->blockFormer * tilingData->col;
        int64_t curBlockLength = curRowsNum * tilingData->col;
        dyInTensorGM.SetGlobalBuffer((__gm__ T *)dy + formerBlockLength * GetBlockIdx(), curBlockLength);
        xInTensorGM.SetGlobalBuffer((__gm__ T *)x + formerBlockLength * GetBlockIdx(), curBlockLength);
        rstdInTensorGM.SetGlobalBuffer((__gm__ float *)rstd + tilingData->blockFormer * GetBlockIdx(), curRowsNum);
        meanInTensorGM.SetGlobalBuffer((__gm__ float *)mean + tilingData->blockFormer * GetBlockIdx(), curRowsNum);
        gammaInTensorGM.SetGlobalBuffer((__gm__ U *)gamma, tilingData->col);
        // init gm outputs
        pdXOutTensorGM.SetGlobalBuffer((__gm__ T *)pdX + formerBlockLength * GetBlockIdx(), curBlockLength);
        pdGammaOutTensorGM.SetGlobalBuffer((__gm__ float *)pdGamma, tilingData->col);
        pdBetaOutTensorGM.SetGlobalBuffer((__gm__ float *)pdBeta, tilingData->col);
        // init workspace
        int64_t wsLenPerBlock = tilingData->colAlignV * CONSTANT_TWO;
        workspaceGM.SetGlobalBuffer((__gm__ float *)workspace + wsLenPerBlock * GetBlockIdx(), wsLenPerBlock);
        // clear workspace and pdGamma/pdBeta outputs
        InitOutput<float>(workspaceGM, wsLenPerBlock, 0.0);
        if constexpr (!isDeterministic) {
            if (GetBlockIdx() == 0) {
                InitOutput<float>(pdGammaOutTensorGM, tilingData->col, 0.0);
                InitOutput<float>(pdBetaOutTensorGM, tilingData->col, 0.0);
            }
            CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_AIV_ONLY_ALL);
        }
        PipeBarrier<PIPE_ALL>();
        // init queues
        int64_t bufferSize = tilingData->bufferElemNums * sizeof(float);
        pipe.InitBuffer(queue0, 1, bufferSize);
        pipe.InitBuffer(queue1, 1, bufferSize);
        pipe.InitBuffer(queue2, 1, bufferSize);
        pipe.InitBuffer(queue3, 1, bufferSize);
        pipe.InitBuffer(queue4, DB_BUFFER_NUM, TMP_BUFFER_SIZE_0 * sizeof(float));
        pipe.InitBuffer(queue5, DB_BUFFER_NUM, TMP_BUFFER_SIZE_0 * sizeof(float));
        pipe.InitBuffer(tmpTensor, (TMP_BUFFER_SIZE_1 * CONSTANT_TWO + TMP_BUFFER_SIZE_0) * sizeof(float));
        // init attrs
        coff = static_cast<float>(1.0) / static_cast<float>(tilingData->col);
    }

    __aicore__ inline void Process(const LayerNormGradV3TilingDataSingleRead *tilingData)
    {
        int64_t ubLoopCount = (GetBlockIdx() == tilingData->blockNum - 1) ? tilingData->ubLoopOfTailBlock
                                                                          : tilingData->ubLoopOfFormerBlock;
        int64_t tailRowsNum = (GetBlockIdx() == tilingData->blockNum - 1) ? tilingData->ubTailOfTailBlock
                                                                          : tilingData->ubTailOfFormerBlock;
        // preload
        int64_t preloadRowsNum = (0 == ubLoopCount - 1) ? tailRowsNum : tilingData->ubFormer;
        CopyInPhase1(0, preloadRowsNum, tilingData);
        CopyInPhase3(0, preloadRowsNum, tilingData);
        for (int64_t i = 0; i < ubLoopCount - 1; i++) {
            int64_t nextRowsNum = (i == ubLoopCount - 1 - 1) ? tailRowsNum : tilingData->ubFormer;
            CopyInPhase0(i, tilingData->ubFormer, tilingData);
            ComputePhase0(i, tilingData->ubFormer, tilingData);
            CopyOutPhase0(i, tilingData->ubFormer, tilingData);
            CopyInPhase1(i + 1, nextRowsNum, tilingData);
            CopyInPhase2(i, tilingData->ubFormer, tilingData);
            ComputePhase1(i, tilingData->ubFormer, tilingData);
            CopyInPhase3(i + 1, nextRowsNum, tilingData);
            ComputePhase2(i, tilingData->ubFormer, tilingData);
            CopyInPhase4(i, tilingData->ubFormer, tilingData);
            ComputePhase3(i, tilingData->ubFormer, tilingData);
            CopyOutPhase1(i, tilingData->ubFormer, tilingData);
            ComputePhase4(i, tilingData->ubFormer, tilingData);
            CopyOutPhase2(i, tilingData->ubFormer, tilingData);
        }
        CopyInPhase0(ubLoopCount - 1, tailRowsNum, tilingData);
        ComputePhase0(ubLoopCount - 1, tailRowsNum, tilingData);
        CopyOutPhase0(ubLoopCount - 1, tailRowsNum, tilingData);
        CopyInPhase2(ubLoopCount - 1, tailRowsNum, tilingData);
        ComputePhase1(ubLoopCount - 1, tailRowsNum, tilingData);
        ComputePhase2(ubLoopCount - 1, tailRowsNum, tilingData);
        CopyInPhase4(ubLoopCount - 1, tailRowsNum, tilingData);
        ComputePhase3(ubLoopCount - 1, tailRowsNum, tilingData);
        CopyOutPhase1(ubLoopCount - 1, tailRowsNum, tilingData);
        ComputePhase4(ubLoopCount - 1, tailRowsNum, tilingData);
        CopyOutPhase2(ubLoopCount - 1, tailRowsNum, tilingData);
        FinalProcess(tilingData);
    }

private:
    __aicore__ inline void CopyInPhase0(
        const int64_t outerIdx, const int64_t curRowsNum, const LayerNormGradV3TilingDataSingleRead *tilingData)
    {
        // copy_in dy to buffer1
        buffer1 = queue1.AllocTensor<float>();
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyParams intriParams;
        if (likely(tilingData->colAlignV == tilingData->col)) {
            intriParams.blockCount = 1;
            intriParams.blockLen = curRowsNum * tilingData->col * sizeof(T);
        } else {
            intriParams.blockCount = curRowsNum;
            intriParams.blockLen = tilingData->col * sizeof(T);
            padParams.isPad = true;
            padParams.rightPadding = tilingData->colAlignM - tilingData->col;
        }
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(buffer1.ReinterpretCast<T>(),
                dyInTensorGM[tilingData->ubFormer * tilingData->col * outerIdx],
                intriParams,
                padParams);
        } else {
            DataCopyPad(buffer1.ReinterpretCast<T>()[tilingData->bufferElemNums],
                dyInTensorGM[tilingData->ubFormer * tilingData->col * outerIdx],
                intriParams,
                padParams);
        }
        queue1.EnQue(buffer1);
    }

    __aicore__ inline void CopyInPhase1(
        const int64_t outerIdx, const int64_t curRowsNum, const LayerNormGradV3TilingDataSingleRead *tilingData)
    {
        // copy_in mean to buffer4
        buffer4 = queue4.AllocTensor<float>();
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = curRowsNum * sizeof(float);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        DataCopyPad(buffer4, meanInTensorGM[tilingData->ubFormer * outerIdx], intriParams, padParams);
        queue4.EnQue(buffer4);
    }

    __aicore__ inline void CopyInPhase2(
        const int64_t outerIdx, const int64_t curRowsNum, const LayerNormGradV3TilingDataSingleRead *tilingData)
    {
        // copy_in x to buffer0
        buffer0 = queue0.AllocTensor<float>();
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyParams intriParams;
        if (likely(tilingData->colAlignV == tilingData->col)) {
            intriParams.blockCount = 1;
            intriParams.blockLen = curRowsNum * tilingData->col * sizeof(T);
        } else {
            intriParams.blockCount = curRowsNum;
            intriParams.blockLen = tilingData->col * sizeof(T);
        }
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(buffer0.ReinterpretCast<T>(),
                xInTensorGM[tilingData->ubFormer * tilingData->col * outerIdx],
                intriParams,
                padParams);
        } else {
            DataCopyPad(buffer0.ReinterpretCast<T>()[tilingData->bufferElemNums],
                xInTensorGM[tilingData->ubFormer * tilingData->col * outerIdx],
                intriParams,
                padParams);
        }
        queue0.EnQue(buffer0);
    }

    __aicore__ inline void CopyInPhase3(
        const int64_t outerIdx, const int64_t curRowsNum, const LayerNormGradV3TilingDataSingleRead *tilingData)
    {
        // copy_in rstd to buffer5
        buffer5 = queue5.AllocTensor<float>();
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = curRowsNum * sizeof(float);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        DataCopyPad(buffer5, rstdInTensorGM[tilingData->ubFormer * outerIdx], intriParams, padParams);
        queue5.EnQue(buffer5);
    }

    __aicore__ inline void CopyInPhase4(
        const int64_t outerIdx, const int64_t curRowsNum, const LayerNormGradV3TilingDataSingleRead *tilingData)
    {
        // copy_in gamma to buffer1
        buffer1 = queue1.AllocTensor<float>();
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = tilingData->col * sizeof(U);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        if constexpr (IsSameType<U, float>::value) {
            DataCopyPad(buffer1.ReinterpretCast<U>(), gammaInTensorGM, intriParams, padParams);
        } else {
            DataCopyPad(buffer1.ReinterpretCast<U>()[tilingData->colAlignM], gammaInTensorGM, intriParams, padParams);
        }
        queue1.EnQue(buffer1);
    }

    __aicore__ inline void ComputePhase0(
        const int64_t outerIdx, const int64_t curRowsNum, const LayerNormGradV3TilingDataSingleRead *tilingData)
    {
        // copy dy to buffer3
        buffer1 = queue1.DeQue<float>();
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = curRowsNum * tilingData->colAlignM * sizeof(T) / BLOCK_BYTES;
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        buffer3 = queue3.AllocTensor<float>();
        if constexpr (IsSameType<T, float>::value) {
            DataCopy(buffer3, buffer1, intriParams);
            PipeBarrier<PIPE_V>();
        } else {
            DataCopy(buffer3.ReinterpretCast<T>()[tilingData->bufferElemNums],
                buffer1.ReinterpretCast<T>()[tilingData->bufferElemNums],
                intriParams);
            PipeBarrier<PIPE_V>();
            CastToFloat(buffer3, curRowsNum, tilingData);
        }
        queue1.FreeTensor(buffer1);
        queue3.EnQue(buffer3);
    }

    __aicore__ inline void ComputePhase1(
        const int64_t outerIdx, const int64_t curRowsNum, const LayerNormGradV3TilingDataSingleRead *tilingData)
    {
        // block_broadcast_2 mean to buffer6
        buffer6 = tmpTensor.Get<float>();
        buffer4 = queue4.DeQue<float>();
        BlockBroadcast<float>(buffer6, buffer4, curRowsNum);
        queue4.FreeTensor(buffer4);
        buffer0 = queue0.DeQue<float>();
        // sub_0 = x - block_broadcast_2 tp buffer0
        if constexpr (!IsSameType<T, float>::value) {
            CastToFloat(buffer0, curRowsNum, tilingData);
        }
        BinElemWithInlinedLastBrcFP32(buffer0, buffer0, buffer6, curRowsNum, tilingData, Sub);
    }

    __aicore__ inline void ComputePhase2(
        const int64_t outerIdx, const int64_t curRowsNum, const LayerNormGradV3TilingDataSingleRead *tilingData)
    {
        // block_broadcast_1 rstd to buffer7(buffer6[TMP_BUFFER_SIZE_1])
        buffer5 = queue5.DeQue<float>();
        BlockBroadcast<float>(buffer6[TMP_BUFFER_SIZE_1], buffer5, curRowsNum);
        queue5.FreeTensor(buffer5);
    }

    __aicore__ inline void ComputePhase3(
        const int64_t outerIdx, const int64_t curRowsNum, const LayerNormGradV3TilingDataSingleRead *tilingData)
    {
        // mul_1 = sub_0 * block_broadcast_1 to buffer0
        BinElemWithInlinedLastBrcFP32(buffer0, buffer0, buffer6[TMP_BUFFER_SIZE_1], curRowsNum, tilingData, Mul);
        // mul_2 = dy * mul_1 to buffer2, buffer3 is freed after copy_out while data is still in tensor
        buffer2 = queue2.AllocTensor<float>();
        buffer3 = queue3.AllocTensor<float>();
        Mul(buffer2, buffer3, buffer0, curRowsNum * tilingData->colAlignV);
        PipeBarrier<PIPE_V>();
        queue2.EnQue(buffer2);
    }

    __aicore__ inline void ComputePhase4(
        const int64_t outerIdx, const int64_t curRowsNum, const LayerNormGradV3TilingDataSingleRead *tilingData)
    {
        // mul_3 = dy * gamma to buffer3
        buffer1 = queue1.DeQue<float>();
        if constexpr (!IsSameType<U, float>::value) {
            Cast(buffer1,
                buffer1.ReinterpretCast<U>()[tilingData->colAlignM],
                RoundMode::CAST_NONE,
                tilingData->colAlignV);
            PipeBarrier<PIPE_V>();
        }
        MulWithInlinedNLastBrcFP32(buffer3, buffer3, buffer1, curRowsNum, tilingData);
        // mul_4 = mul_3 * mul_1 to buffer1
        Mul(buffer1, buffer3, buffer0, curRowsNum * tilingData->colAlignV);
        PipeBarrier<PIPE_V>();
        // reduce_3 = reduce mul_4 to buffer8(buffer6[TMP_BUFFER_SIZE_1 * 2])
        ReduceMeanModeOne(buffer6[TMP_BUFFER_SIZE_1 * CONSTANT_TWO], buffer1, curRowsNum, tilingData);
        queue1.FreeTensor(buffer1);
        // block_broadcast_4 reduce_3 to buffer6
        BlockBroadcast<float>(buffer6, buffer6[TMP_BUFFER_SIZE_1 * CONSTANT_TWO], curRowsNum);
        // mul_7 = mul_1 * block_broadcast_4 to buffer2, buffer2 is freed after copy_out
        buffer2 = queue2.AllocTensor<float>();
        BinElemWithInlinedLastBrcFP32(buffer2, buffer0, buffer6, curRowsNum, tilingData, Mul);
        queue0.FreeTensor(buffer0);
        // reduce_2 = reduce mul_3 to buffer8(buffer6[TMP_BUFFER_SIZE_1 * 2])
        ReduceMeanModeOne(buffer6[TMP_BUFFER_SIZE_1 * CONSTANT_TWO], buffer3, curRowsNum, tilingData);
        // block_broadcast_3 reduce_2 to buffer6
        BlockBroadcast<float>(buffer6, buffer6[TMP_BUFFER_SIZE_1 * CONSTANT_TWO], curRowsNum);
        // sub_8 = mul_3 - mul_7 to buffer3
        Sub(buffer3, buffer3, buffer2, curRowsNum * tilingData->colAlignV);
        queue2.FreeTensor(buffer2);
        PipeBarrier<PIPE_V>();
        // sub_9 = sub_8 - block_broadcast_3 to buffer3
        BinElemWithInlinedLastBrcFP32(buffer3, buffer3, buffer6, curRowsNum, tilingData, Sub);
        // mul_10 = sub_9 - add_8 to block_broadcast_1
        BinElemWithInlinedLastBrcFP32(buffer3, buffer3, buffer6[TMP_BUFFER_SIZE_1], curRowsNum, tilingData, Mul);
        // cast mul_10
        if constexpr (!IsSameType<T, float>::value) {
            CastToB16(buffer3, curRowsNum, tilingData);
        }
        queue3.EnQue(buffer3);
    }

    __aicore__ inline void CopyOutPhase0(
        const int64_t outerIdx, const int64_t curRowsNum, const LayerNormGradV3TilingDataSingleRead *tilingData)
    {
        // copy_out copy_dy from buffer3 and do atomic
        buffer3 = queue3.DeQue<float>();
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = tilingData->colAlignV * sizeof(float);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        SetAtomicAdd<float>();
        for (int64_t i = 0; i < curRowsNum; i++) {
            // allocate one ws space while two workspaces are needed, so the second worspace has an offset
            if constexpr (isDeterministic) {
                PipeBarrier<PIPE_MTE3>();
            }
            DataCopyPad(workspaceGM[tilingData->colAlignV], buffer3[tilingData->colAlignV * i], intriParams);
        }
        queue3.FreeTensor(buffer3);
        SetAtomicNone();
    }

    __aicore__ inline void CopyOutPhase1(
        const int64_t outerIdx, const int64_t curRowsNum, const LayerNormGradV3TilingDataSingleRead *tilingData)
    {
        // copy_out mul_2 from buffer2 and do atomic
        buffer2 = queue2.DeQue<float>();
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = tilingData->colAlignV * sizeof(float);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        SetAtomicAdd<float>();
        for (int64_t i = 0; i < curRowsNum; i++) {
            if constexpr (isDeterministic) {
                PipeBarrier<PIPE_MTE3>();
            }
            DataCopyPad(workspaceGM, buffer2[tilingData->colAlignV * i], intriParams);
        }
        queue2.FreeTensor(buffer2);
        SetAtomicNone();
    }

    __aicore__ inline void CopyOutPhase2(
        const int64_t outerIdx, const int64_t curRowsNum, const LayerNormGradV3TilingDataSingleRead *tilingData)
    {
        // copy_out mul_10 to pdX
        buffer3 = queue3.DeQue<float>();
        DataCopyParams intriParams;
        if (likely(tilingData->colAlignV == tilingData->col)) {
            intriParams.blockCount = 1;
            intriParams.blockLen = curRowsNum * tilingData->col * sizeof(T);
        } else {
            intriParams.blockCount = curRowsNum;
            intriParams.blockLen = tilingData->col * sizeof(T);
        }
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        DataCopyPad(pdXOutTensorGM[tilingData->ubFormer * tilingData->col * outerIdx],
            buffer3.ReinterpretCast<T>(),
            intriParams);
        queue3.FreeTensor(buffer3);
    }

    __aicore__ inline void FinalProcess(const LayerNormGradV3TilingDataSingleRead *tilingData)
    {
        // copy_in tmp data from workspace and do atomic
        PipeBarrier<PIPE_ALL>();
        if constexpr (isDeterministic) {
            FinalProcessDeterministic(tilingData);
        } else {
            buffer0 = queue0.AllocTensor<float>();
            DataCopyPadParams padParams{false, 0, 0, 0};
            DataCopyParams intriParams;
            intriParams.blockCount = 1;
            intriParams.blockLen = tilingData->col * sizeof(float);
            intriParams.srcStride = 0;
            intriParams.dstStride = 0;
            DataCopyPad(buffer0, workspaceGM, intriParams, padParams);
            event_t event0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
            SetFlag<HardEvent::MTE2_MTE3>(event0);
            WaitFlag<HardEvent::MTE2_MTE3>(event0);
            SetAtomicAdd<float>();
            CrossCoreWaitFlag(SYNC_AIV_ONLY_ALL);
            DataCopyPad(pdGammaOutTensorGM, buffer0, intriParams);
            queue0.FreeTensor(buffer0);
            SetAtomicNone();
            buffer1 = queue1.AllocTensor<float>();
            DataCopyPad(buffer1, workspaceGM[tilingData->colAlignV], intriParams, padParams);
            event_t event1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
            SetFlag<HardEvent::MTE2_MTE3>(event1);
            WaitFlag<HardEvent::MTE2_MTE3>(event1);
            SetAtomicAdd<float>();
            DataCopyPad(pdBetaOutTensorGM, buffer1, intriParams);
            queue1.FreeTensor(buffer1);
            SetAtomicNone();
        }
    }

    __aicore__ inline void FinalProcessDeterministic(const LayerNormGradV3TilingDataSingleRead *tilingData)
    {
        SyncAll();
        if (GetBlockIdx() == 0) {
            buffer2 = queue2.AllocTensor<float>();
            buffer3 = queue3.AllocTensor<float>();
            Duplicate(buffer2, static_cast<float>(0.0), tilingData->colAlignV);
            Duplicate(buffer3, static_cast<float>(0.0), tilingData->colAlignV);
            PipeBarrier<PIPE_V>();
            DataCopyPadParams padParams{false, 0, 0, 0};
            DataCopyParams intriParams;
            intriParams.blockCount = 1;
            intriParams.blockLen = tilingData->colAlignV * sizeof(float);
            intriParams.srcStride = 0;
            intriParams.dstStride = 0;
            for (int64_t i = 0; i < tilingData->blockNum; i++) {
                buffer0 = queue0.AllocTensor<float>();
                DataCopyPad(buffer0, workspaceGM[tilingData->colAlignV * CONSTANT_TWO * i], intriParams, padParams);
                queue0.EnQue(buffer0);
                buffer1 = queue1.AllocTensor<float>();
                DataCopyPad(buffer1,
                    workspaceGM[tilingData->colAlignV * CONSTANT_TWO * i + tilingData->colAlignV],
                    intriParams,
                    padParams);
                queue1.EnQue(buffer1);
                PipeBarrier<PIPE_MTE2>();
                buffer0 = queue0.DeQue<float>();
                Add(buffer2, buffer2, buffer0, tilingData->colAlignV);
                queue0.FreeTensor(buffer0);
                buffer1 = queue1.DeQue<float>();
                Add(buffer3, buffer3, buffer1, tilingData->colAlignV);
                queue1.FreeTensor(buffer1);
                PipeBarrier<PIPE_V>();
            }
            queue2.EnQue(buffer2);
            buffer2 = queue2.DeQue<float>();
            intriParams.blockLen = tilingData->col * sizeof(float);
            DataCopyPad(pdGammaOutTensorGM, buffer2, intriParams);
            queue2.FreeTensor(buffer2);
            queue3.EnQue(buffer3);
            buffer3 = queue3.DeQue<float>();
            DataCopyPad(pdBetaOutTensorGM, buffer3, intriParams);
            queue3.FreeTensor(buffer3);
        }
    }

    __aicore__ inline void CastToFloat(const LocalTensor<float> &buffer, const int64_t curRowsNum,
        const LayerNormGradV3TilingDataSingleRead *tilingData)
    {
        if (tilingData->colAlignM == tilingData->colAlignV || tilingData->colAlignV == tilingData->col) {
            Cast(buffer,
                buffer.ReinterpretCast<T>()[tilingData->bufferElemNums],
                RoundMode::CAST_NONE,
                curRowsNum * tilingData->colAlignV);
        } else {
            for (int64_t i = 0; i < curRowsNum; i++) {
                Cast(buffer[i * tilingData->colAlignV],
                    buffer.ReinterpretCast<T>()[tilingData->bufferElemNums + i * tilingData->colAlignM],
                    RoundMode::CAST_NONE,
                    tilingData->colAlignV);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CastToB16(const LocalTensor<float> &buffer, const int64_t curRowsNum,
        const LayerNormGradV3TilingDataSingleRead *tilingData)
    {
        RoundMode b16RoundMode = IsSameType<T, bfloat16_t>::value ? RoundMode::CAST_ROUND : RoundMode::CAST_NONE;
        if (tilingData->colAlignM == tilingData->colAlignV || tilingData->colAlignV == tilingData->col) {
            Cast(buffer.ReinterpretCast<T>(), buffer, b16RoundMode, curRowsNum * tilingData->colAlignV);
        } else {
            for (int64_t i = 0; i < curRowsNum; i++) {
                Cast(buffer.ReinterpretCast<T>()[i * tilingData->colAlignM],
                    buffer[i * tilingData->colAlignV],
                    b16RoundMode,
                    tilingData->colAlignV);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    template <typename dType>
    __aicore__ inline void BlockBroadcast(
        const LocalTensor<dType> &dst, const LocalTensor<dType> &src, const int64_t curRowsNum)
    {
        Brcb(dst, src, (curRowsNum + CONSTANT_EIGHT - 1) / CONSTANT_EIGHT, {1, CONSTANT_EIGHT});
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BinElemWithInlinedLastBrcFP32(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
        const LocalTensor<float> &src1, const int64_t curRowsNum, const LayerNormGradV3TilingDataSingleRead *tilingData,
        void (*func)(const LocalTensor<float> &, const LocalTensor<float> &, const LocalTensor<float> &, uint64_t,
            uint8_t, const BinaryRepeatParams &))
    {
        // src1 need to do inline broadcast
        BinaryRepeatParams intriParams;
        intriParams.dstBlkStride = 1;
        intriParams.src0BlkStride = 1;
        intriParams.src1BlkStride = 0;
        intriParams.dstRepStride = CONSTANT_EIGHT;
        intriParams.src0RepStride = CONSTANT_EIGHT;
        intriParams.src1RepStride = 0;
        int64_t formerLoops = tilingData->colAlignV / B32_REPEAT_SIZE;
        int64_t tail = tilingData->colAlignV - formerLoops * B32_REPEAT_SIZE;
        for (int64_t i = 0; i < curRowsNum; i++) {
            int64_t formerOffset = i * tilingData->colAlignV;
            func(dst[formerOffset],
                src0[formerOffset],
                src1[i * B32_BLOCK_SIZE],
                B32_REPEAT_SIZE,
                formerLoops,
                intriParams);
            if (tail != 0) {
                int64_t tailOffset = formerOffset + formerLoops * B32_REPEAT_SIZE;
                func(dst[tailOffset], src0[tailOffset], src1[i * B32_BLOCK_SIZE], tail, 1, intriParams);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MulWithInlinedNLastBrcFP32(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
        const LocalTensor<float> &src1, const int64_t curRowsNum, const LayerNormGradV3TilingDataSingleRead *tilingData)
    {
        // src1 need to do inline broadcast
        for (int64_t i = 0; i < curRowsNum; i++) {
            int64_t formerOffset = i * tilingData->colAlignV;
            Mul(dst[formerOffset], src0[formerOffset], src1, tilingData->colAlignV);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ReduceMeanModeOne(const LocalTensor<float> &dst, const LocalTensor<float> &src,
        const int64_t curRowsNum, const LayerNormGradV3TilingDataSingleRead *tilingData)
    {
        AscendCUtils::SetMaskCount<float>();
        SetVectorMask<float>(0, tilingData->colAlignV);
        for (int64_t i = 0; i < curRowsNum; i++) {
            ReduceSum(src[i * tilingData->colAlignV], src[i * tilingData->colAlignV],
                src[i * tilingData->colAlignV], COUNT_NUM);
            PipeBarrier<PIPE_V>();
            event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            (*(__ubuf__ int32_t *)((__ubuf__ float *)dst[i].GetPhyAddr())) = GetAccVal();
        }
        AscendCUtils::SetMaskNorm<float>();
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        Muls(dst, dst, coff, curRowsNum);
        PipeBarrier<PIPE_V>();
    }

private:
    TPipe pipe;
    constexpr static uint16_t SYNC_AIV_ONLY_ALL = 14;

    TQue<QuePosition::VECIN, 1> queue0;
    TQue<QuePosition::VECIN, 1> queue1;
    TQue<QuePosition::VECOUT, 1> queue2;
    TQue<QuePosition::VECOUT, 1> queue3;
    TQue<QuePosition::VECIN, DB_BUFFER_NUM> queue4;
    TQue<QuePosition::VECIN, DB_BUFFER_NUM> queue5;
    TBuf<> tmpTensor;

    LocalTensor<float> buffer0;
    LocalTensor<float> buffer1;
    LocalTensor<float> buffer2;
    LocalTensor<float> buffer3;
    LocalTensor<float> buffer4;
    LocalTensor<float> buffer5;
    LocalTensor<float> buffer6;

    GlobalTensor<T> dyInTensorGM;
    GlobalTensor<T> xInTensorGM;
    GlobalTensor<float> rstdInTensorGM;
    GlobalTensor<float> meanInTensorGM;
    GlobalTensor<U> gammaInTensorGM;
    GlobalTensor<T> pdXOutTensorGM;
    GlobalTensor<float> pdGammaOutTensorGM;
    GlobalTensor<float> pdBetaOutTensorGM;
    GlobalTensor<float> workspaceGM;

    float coff;
};
}  // namespace LayerNormGradV3
#endif
