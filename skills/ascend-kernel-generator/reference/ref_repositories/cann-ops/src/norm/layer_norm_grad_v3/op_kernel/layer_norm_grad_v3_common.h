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
 * \file layer_norm_grad_v3_common.h
 * \brief
 */

#ifndef LAYER_NORM_GRAD_V3_COMMON
#define LAYER_NORM_GRAD_V3_COMMON

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace LayerNormGradV3 {
using namespace AscendC;

constexpr int32_t COMMON_B32_BLOCK_SIZE = 8;
constexpr int32_t COMMON_B16_BLOCK_SIZE = 16;
constexpr int32_t COMMON_B32_REPEAT_SIZE = 64;
constexpr int32_t COMMON_CONSTANT_TWO = 2;
constexpr int32_t COMMON_CONSTANT_EIGHT = 8;
constexpr int32_t COMMON_CONSTANT_SIXTEEN = 16;
constexpr int32_t COMMON_MAX_REPEAT = 255;
constexpr int32_t COMMON_VC_MAX_REPEAT = 248;

template <typename T, typename U, bool isDeterministic>
class LayerNormGradV3Common {
public:
    __aicore__ inline LayerNormGradV3Common(){};
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean, GM_ADDR gamma, GM_ADDR pdX,
        GM_ADDR pdGamma, GM_ADDR pdBeta, GM_ADDR workspace, const LayerNormGradV3TilingDataCommon *tilingData)
    {
        // init gm inputs
        int64_t curRowsNum =
            (GetBlockIdx() != tilingData->blockNum - 1) ? tilingData->blockFormer : tilingData->blockTail;
        int64_t formerBlockLength = tilingData->blockFormer * tilingData->col;
        int64_t curBlockLength = curRowsNum * tilingData->col;
        dyInTensorGM_.SetGlobalBuffer((__gm__ T *)dy + formerBlockLength * GetBlockIdx(), curBlockLength);
        xInTensorGM_.SetGlobalBuffer((__gm__ T *)x + formerBlockLength * GetBlockIdx(), curBlockLength);
        rstdInTensorGM_.SetGlobalBuffer((__gm__ float *)rstd + tilingData->blockFormer * GetBlockIdx(), curRowsNum);
        meanInTensorGM_.SetGlobalBuffer((__gm__ float *)mean + tilingData->blockFormer * GetBlockIdx(), curRowsNum);
        gammaInTensorGM_.SetGlobalBuffer((__gm__ U *)gamma, tilingData->col);
        // init gm outputs
        pdXOutTensorGM_.SetGlobalBuffer((__gm__ T *)pdX + formerBlockLength * GetBlockIdx(), curBlockLength);
        pdGammaOutTensorGM_.SetGlobalBuffer((__gm__ float *)pdGamma, tilingData->col);
        pdBetaOutTensorGM_.SetGlobalBuffer((__gm__ float *)pdBeta, tilingData->col);
        // init workspace
        int64_t wsLenPerBlock = tilingData->colAlignV * COMMON_CONSTANT_TWO;
        workspaceGM_.SetGlobalBuffer((__gm__ float *)workspace + wsLenPerBlock * GetBlockIdx(), wsLenPerBlock);
        // clear pdGamma/pdBeta outputs
        if constexpr (!isDeterministic) {
            if (GetBlockIdx() == 0) {
                InitOutput<float>(pdGammaOutTensorGM_, tilingData->col, 0.0);
                InitOutput<float>(pdBetaOutTensorGM_, tilingData->col, 0.0);
            }
            CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_AIV_ONLY_ALL);
        }
        PipeBarrier<PIPE_ALL>();
        // init queues
        pipe_.InitBuffer(queue0_, 1, tilingData->wholeBufferBytes);
        pipe_.InitBuffer(queue1_, 1, tilingData->wholeBufferBytes);
        pipe_.InitBuffer(queue2_, 1, tilingData->lastRBufferBytes);
        pipe_.InitBuffer(queue3_, 1, tilingData->lastRBufferBytes);
        pipe_.InitBuffer(queue4_, 1, tilingData->nlastRBufferBytes);
        pipe_.InitBuffer(queue5_, 1, tilingData->wholeBufferBytes);
        pipe_.InitBuffer(queue6_, 1, tilingData->nlastRBufferBytes);
        pipe_.InitBuffer(queue7_, 1, tilingData->nlastRBufferBytes);
        pipe_.InitBuffer(tmpTensor0_, tilingData->wholeBufferBytes);
        pipe_.InitBuffer(tmpTensor1_, tilingData->lastBrcbBufferBytes);
        pipe_.InitBuffer(tmpTensor2_, tilingData->lastRBufferBytes);
        // init attrs
        coff_ = static_cast<float>(1.0) / static_cast<float>(tilingData->col);
    }

    __aicore__ inline void Process(const LayerNormGradV3TilingDataCommon *tilingData)
    {
        int64_t ubLoopCount = (GetBlockIdx() == tilingData->blockNum - 1) ? tilingData->ubLoopOfTailBlock
                                                                          : tilingData->ubLoopOfFormerBlock;
        int64_t tailRowsNum = (GetBlockIdx() == tilingData->blockNum - 1) ? tilingData->ubTailOfTailBlock
                                                                          : tilingData->ubTailOfFormerBlock;
        // pre process
        CopyInPhase0(tilingData);
        ComputePhase0(tilingData);
        for (int64_t i = 0; i < ubLoopCount - 1; i++) {
            CopyInPhase1(i, tilingData->ubFormer, tilingData);
            ComputePhase1(i, tilingData->ubFormer, tilingData);
            CopyInPhase2(i, tilingData->ubFormer, tilingData);
            ComputePhase2(i, tilingData->ubFormer, tilingData);
            CopyInPhase3(i, tilingData->ubFormer, tilingData);
            ComputePhase3(i, tilingData->ubFormer, tilingData);
            CopyInPhase4(i, tilingData->ubFormer, tilingData);
            ComputePhase4(i, tilingData->ubFormer, tilingData);
            CopyOutPhase0(i, tilingData->ubFormer, tilingData);
        }
        CopyInPhase1(ubLoopCount - 1, tailRowsNum, tilingData);
        ComputePhase1(ubLoopCount - 1, tailRowsNum, tilingData);
        CopyInPhase2(ubLoopCount - 1, tailRowsNum, tilingData);
        ComputePhase2(ubLoopCount - 1, tailRowsNum, tilingData);
        CopyInPhase3(ubLoopCount - 1, tailRowsNum, tilingData);
        ComputePhase3(ubLoopCount - 1, tailRowsNum, tilingData);
        CopyInPhase4(ubLoopCount - 1, tailRowsNum, tilingData);
        ComputePhase4(ubLoopCount - 1, tailRowsNum, tilingData);
        CopyOutPhase0(ubLoopCount - 1, tailRowsNum, tilingData);
        // post process
        if constexpr (isDeterministic) {
            CopyOutPhase1Deterministic(tilingData);
        } else {
            CrossCoreWaitFlag(SYNC_AIV_ONLY_ALL);
            CopyOutPhase1(tilingData);
        }
        queue4_.FreeTensor(buffer4_);
    }

private:
    __aicore__ inline void CopyInPhase0(const LayerNormGradV3TilingDataCommon *tilingData)
    {
        // copy_in gamma to buffer4_
        buffer4_ = queue4_.AllocTensor<float>();
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = tilingData->col * sizeof(U);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        if constexpr (IsSameType<U, float>::value) {
            DataCopyPad(buffer4_.ReinterpretCast<U>(), gammaInTensorGM_, intriParams, padParams);
        } else {
            DataCopyPad(buffer4_.ReinterpretCast<U>()[tilingData->colAlignM], gammaInTensorGM_, intriParams, padParams);
        }
        queue4_.EnQue(buffer4_);
    }

    __aicore__ inline void ComputePhase0(const LayerNormGradV3TilingDataCommon *tilingData)
    {
        // cast gamma to buffer4_
        buffer4_ = queue4_.DeQue<float>();
        if constexpr (!IsSameType<U, float>::value) {
            Cast(buffer4_,
                buffer4_.ReinterpretCast<U>()[tilingData->colAlignM],
                RoundMode::CAST_NONE,
                tilingData->colAlignV);
            PipeBarrier<PIPE_V>();
        }
        // set reduce init value
        buffer6_ = queue6_.AllocTensor<float>();
        Duplicate(buffer6_, static_cast<float>(0.0), tilingData->colAlignV);
        buffer7_ = queue7_.AllocTensor<float>();
        Duplicate(buffer7_, static_cast<float>(0.0), tilingData->colAlignV);
        PipeBarrier<PIPE_V>();
        buffer8_ = tmpTensor0_.Get<float>();
        buffer9_ = tmpTensor1_.Get<float>();
        buffer10_ = tmpTensor2_.Get<float>();
    }

    __aicore__ inline void CopyInPhase1(
        const int64_t outerIdx, const int64_t curRowsNum, const LayerNormGradV3TilingDataCommon *tilingData)
    {
        // copy_in dy to buffer1_
        buffer1_ = queue1_.AllocTensor<float>();
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
            DataCopyPad(buffer1_.ReinterpretCast<T>(),
                dyInTensorGM_[tilingData->ubFormer * tilingData->col * outerIdx],
                intriParams,
                padParams);
        } else {
            DataCopyPad(buffer1_.ReinterpretCast<T>()[tilingData->wholeBufferElemNums],
                dyInTensorGM_[tilingData->ubFormer * tilingData->col * outerIdx],
                intriParams,
                padParams);
        }
        queue1_.EnQue(buffer1_);
    }

    __aicore__ inline void ComputePhase1(
        const int64_t outerIdx, const int64_t curRowsNum, const LayerNormGradV3TilingDataCommon *tilingData)
    {
        // reduce0 = reduce dy to tensor6
        buffer1_ = queue1_.DeQue<float>();
        if constexpr (!IsSameType<T, float>::value) {
            CastToFloat(buffer1_, buffer8_, curRowsNum, tilingData);
        }
        PipeBarrier<PIPE_V>();
        NlastReduceSum(buffer6_, buffer1_, curRowsNum, tilingData);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyInPhase2(
        const int64_t outerIdx, const int64_t curRowsNum, const LayerNormGradV3TilingDataCommon *tilingData)
    {
        // copy_in mean to buffer2_
        buffer2_ = queue2_.AllocTensor<float>();
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = curRowsNum * sizeof(float);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        DataCopyPad(buffer2_, meanInTensorGM_[tilingData->ubFormer * outerIdx], intriParams, padParams);
        queue2_.EnQue(buffer2_);
    }

    __aicore__ inline void ComputePhase2(
        const int64_t outerIdx, const int64_t curRowsNum, const LayerNormGradV3TilingDataCommon *tilingData)
    {
        // block_broadcast_2 mean to buffer9_
        buffer2_ = queue2_.DeQue<float>();
        BlockBroadcast<float>(buffer9_, buffer2_, curRowsNum);
        queue2_.FreeTensor(buffer2_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyInPhase3(
        const int64_t outerIdx, const int64_t curRowsNum, const LayerNormGradV3TilingDataCommon *tilingData)
    {
        // copy_in x to buffer0_
        buffer0_ = queue0_.AllocTensor<float>();
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
            DataCopyPad(buffer0_.ReinterpretCast<T>(),
                xInTensorGM_[tilingData->ubFormer * tilingData->col * outerIdx],
                intriParams,
                padParams);
        } else {
            DataCopyPad(buffer0_.ReinterpretCast<T>()[tilingData->wholeBufferElemNums],
                xInTensorGM_[tilingData->ubFormer * tilingData->col * outerIdx],
                intriParams,
                padParams);
        }
        queue0_.EnQue(buffer0_);
    }

    __aicore__ inline void ComputePhase3(
        const int64_t outerIdx, const int64_t curRowsNum, const LayerNormGradV3TilingDataCommon *tilingData)
    {
        buffer0_ = queue0_.DeQue<float>();
        // sub_0 = x - block_broadcast_2 to buffer0_
        if constexpr (!IsSameType<T, float>::value) {
            CastToFloat(buffer0_, buffer8_, curRowsNum, tilingData);
        }
        PipeBarrier<PIPE_V>();
        BinElemWithInlinedLastBrcFP32(buffer0_, buffer0_, buffer9_, curRowsNum, tilingData, Sub);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyInPhase4(
        const int64_t outerIdx, const int64_t curRowsNum, const LayerNormGradV3TilingDataCommon *tilingData)
    {
        // copy_in rstd to buffer3_
        buffer3_ = queue3_.AllocTensor<float>();
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = curRowsNum * sizeof(float);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        DataCopyPad(buffer3_, rstdInTensorGM_[tilingData->ubFormer * outerIdx], intriParams, padParams);
        queue3_.EnQue(buffer3_);
    }

    __aicore__ inline void ComputePhase4(
        const int64_t outerIdx, const int64_t curRowsNum, const LayerNormGradV3TilingDataCommon *tilingData)
    {
        // block_broadcast_1 rstd to buffer9_
        buffer3_ = queue3_.DeQue<float>();
        BlockBroadcast<float>(buffer9_, buffer3_, curRowsNum);
        PipeBarrier<PIPE_V>();
        queue3_.FreeTensor(buffer3_);
        // mul_1 = sub_0 * block_broadcast_1 to buffer5_
        buffer5_ = queue5_.AllocTensor<float>();
        BinElemWithInlinedLastBrcFP32(buffer5_, buffer0_, buffer9_, curRowsNum, tilingData, Mul);
        PipeBarrier<PIPE_V>();
        // mul_2 = dy * mul_1 to buffer0_
        Mul(buffer0_, buffer5_, buffer1_, curRowsNum * tilingData->colAlignV);
        PipeBarrier<PIPE_V>();
        // reduce_1 = reduce mul_2 to buffer7_
        NlastReduceSum(buffer7_, buffer0_, curRowsNum, tilingData);
        PipeBarrier<PIPE_V>();
        // mul_3 = dy * gamma to buffer8_
        MulWithInlinedNLastBrcFP32(buffer8_, buffer1_, buffer4_, curRowsNum, tilingData);
        PipeBarrier<PIPE_V>();
        // mul_4 = mul_3 * mul_1 to buffer1_
        Mul(buffer1_, buffer8_, buffer5_, curRowsNum * tilingData->colAlignV);
        PipeBarrier<PIPE_V>();
        // reduce_3 = reduce mul_4 to buffer10_ and use buffer0_ as tmpBuffer
        LastReduceSum(buffer10_, buffer1_, buffer0_, curRowsNum, tilingData);
        queue1_.FreeTensor(buffer1_);
        PipeBarrier<PIPE_V>();
        // mul_6 = reduce_3 * coff to buffer10_
        Muls(buffer10_, buffer10_, coff_, curRowsNum);
        PipeBarrier<PIPE_V>();
        // block_broadcast_4 mul_6 to buffer0_
        BlockBroadcast<float>(buffer0_, buffer10_, curRowsNum);
        PipeBarrier<PIPE_V>();
        // mul_7 = mul_1 * block_broadcast_4 to buffer5_
        BinElemWithInlinedLastBrcFP32(buffer5_, buffer5_, buffer0_, curRowsNum, tilingData, Mul);
        PipeBarrier<PIPE_V>();
        // sub_8 = mul_3 - mul_7 to buffer5_
        Sub(buffer5_, buffer8_, buffer5_, curRowsNum * tilingData->colAlignV);
        PipeBarrier<PIPE_V>();
        // reduce_2 = reduce mul_3 to buffer10_ and use buffer0_ as tmpBuffer
        LastReduceSum(buffer10_, buffer8_, buffer0_, curRowsNum, tilingData);
        queue0_.FreeTensor(buffer0_);
        PipeBarrier<PIPE_V>();
        // mul_5 = reduce_2 * coff to buffer10_
        Muls(buffer10_, buffer10_, coff_, curRowsNum);
        PipeBarrier<PIPE_V>();
        // block_broadcast_3 mul_5 to buffer8_
        BlockBroadcast<float>(buffer8_, buffer10_, curRowsNum);
        PipeBarrier<PIPE_V>();
        // sub_9 = sub_8 - block_broadcast_3 to buffer5_
        BinElemWithInlinedLastBrcFP32(buffer5_, buffer5_, buffer8_, curRowsNum, tilingData, Sub);
        PipeBarrier<PIPE_V>();
        // mul_10 = sub_9 * block_broadcast_1 to buffer5_
        BinElemWithInlinedLastBrcFP32(buffer5_, buffer5_, buffer9_, curRowsNum, tilingData, Mul);
        // cast mul_10
        if constexpr (!IsSameType<T, float>::value) {
            PipeBarrier<PIPE_V>();
            CastToB16(buffer5_, buffer8_, curRowsNum, tilingData);
        }
        queue5_.EnQue(buffer5_);
    }

    __aicore__ inline void CopyOutPhase0(
        const int64_t outerIdx, const int64_t curRowsNum, const LayerNormGradV3TilingDataCommon *tilingData)
    {
        // copy_out mul_10 from buffer5_ to pdX
        buffer5_ = queue5_.DeQue<float>();
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
        DataCopyPad(pdXOutTensorGM_[tilingData->ubFormer * tilingData->col * outerIdx],
            buffer5_.ReinterpretCast<T>(),
            intriParams);
        queue5_.FreeTensor(buffer5_);
    }

    __aicore__ inline void CopyOutPhase1(const LayerNormGradV3TilingDataCommon *tilingData)
    {
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = tilingData->col * sizeof(float);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        SetAtomicAdd<float>();
        // copy_out reduce_0 from buffer6_ to pdBeta and do atomic
        queue6_.EnQue(buffer6_);
        buffer6_ = queue6_.DeQue<float>();
        DataCopyPad(pdBetaOutTensorGM_, buffer6_, intriParams);
        queue6_.FreeTensor(buffer6_);
        // copy_out reduce_1 from buffer7_ to pdGamma and do atomic
        queue7_.EnQue(buffer7_);
        buffer7_ = queue7_.DeQue<float>();
        DataCopyPad(pdGammaOutTensorGM_, buffer7_, intriParams);
        queue7_.FreeTensor(buffer7_);
        SetAtomicNone();
    }

    __aicore__ inline void CopyOutPhase1Deterministic(const LayerNormGradV3TilingDataCommon *tilingData)
    {
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = tilingData->colAlignV * sizeof(float);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        queue6_.EnQue(buffer6_);
        buffer6_ = queue6_.DeQue<float>();
        DataCopyPad(workspaceGM_[tilingData->colAlignV], buffer6_, intriParams);
        queue6_.FreeTensor(buffer6_);
        queue7_.EnQue(buffer7_);
        buffer7_ = queue7_.DeQue<float>();
        DataCopyPad(workspaceGM_, buffer7_, intriParams);
        queue7_.FreeTensor(buffer7_);
        PipeBarrier<PIPE_ALL>();
        SyncAll();
        if (GetBlockIdx() == 0) {
            buffer6_ = queue6_.AllocTensor<float>();
            buffer7_ = queue7_.AllocTensor<float>();
            Duplicate(buffer6_, static_cast<float>(0.0), tilingData->colAlignV);
            Duplicate(buffer7_, static_cast<float>(0.0), tilingData->colAlignV);
            PipeBarrier<PIPE_V>();
            DataCopyPadParams padParams{false, 0, 0, 0};
            for (int64_t i = 0; i < tilingData->blockNum; i++) {
                buffer0_ = queue0_.AllocTensor<float>();
                DataCopyPad(buffer0_, workspaceGM_[tilingData->colAlignV * CONSTANT_TWO * i], intriParams, padParams);
                queue0_.EnQue(buffer0_);
                buffer1_ = queue1_.AllocTensor<float>();
                DataCopyPad(buffer1_,
                    workspaceGM_[tilingData->colAlignV * CONSTANT_TWO * i + tilingData->colAlignV],
                    intriParams,
                    padParams);
                queue1_.EnQue(buffer1_);
                buffer0_ = queue0_.DeQue<float>();
                Add(buffer7_, buffer7_, buffer0_, tilingData->colAlignV);
                queue0_.FreeTensor(buffer0_);
                buffer1_ = queue1_.DeQue<float>();
                Add(buffer6_, buffer6_, buffer1_, tilingData->colAlignV);
                queue1_.FreeTensor(buffer1_);
                PipeBarrier<PIPE_V>();
            }
            queue6_.EnQue(buffer6_);
            buffer6_ = queue6_.DeQue<float>();
            intriParams.blockLen = tilingData->col * sizeof(float);
            DataCopyPad(pdBetaOutTensorGM_, buffer6_, intriParams);
            queue6_.FreeTensor(buffer6_);
            queue7_.EnQue(buffer7_);
            buffer7_ = queue7_.DeQue<float>();
            DataCopyPad(pdGammaOutTensorGM_, buffer7_, intriParams);
            queue7_.FreeTensor(buffer7_);
        }
    }

    __aicore__ inline void CastToFloat(const LocalTensor<float> &buffer, const LocalTensor<float> &tmpBuffer,
        const int64_t curRowsNum, const LayerNormGradV3TilingDataCommon *tilingData)
    {
        if (tilingData->colAlignM == tilingData->colAlignV || tilingData->colAlignV == tilingData->col) {
            Cast(buffer,
                buffer.ReinterpretCast<T>()[tilingData->wholeBufferElemNums],
                RoundMode::CAST_NONE,
                curRowsNum * tilingData->colAlignV);
        } else {
            DataCopyParams copyIntriParams;
            copyIntriParams.blockCount = 1;
            copyIntriParams.blockLen = curRowsNum * tilingData->colAlignM / COMMON_CONSTANT_SIXTEEN;
            copyIntriParams.srcStride = 0;
            copyIntriParams.dstStride = 0;
            DataCopy(tmpBuffer.ReinterpretCast<T>(),
                buffer.ReinterpretCast<T>()[tilingData->wholeBufferElemNums],
                copyIntriParams);
            PipeBarrier<PIPE_V>();
            int64_t formerColLoops = tilingData->colAlignV / COMMON_B32_REPEAT_SIZE;
            int64_t remainderCol = tilingData->colAlignV - formerColLoops * COMMON_B32_REPEAT_SIZE;
            int64_t repeatLoops = curRowsNum / COMMON_MAX_REPEAT;
            int64_t remainderRepeat = curRowsNum - repeatLoops * COMMON_MAX_REPEAT;
            UnaryRepeatParams intriParams;
            intriParams.dstBlkStride = 1;
            intriParams.srcBlkStride = 1;
            intriParams.dstRepStride = tilingData->colAlignV / COMMON_CONSTANT_EIGHT;
            intriParams.srcRepStride = tilingData->colAlignM / COMMON_CONSTANT_SIXTEEN;
            for (int64_t i = 0; i < repeatLoops; i++) {
                int64_t srcRepeatOffset = i * COMMON_MAX_REPEAT * tilingData->colAlignM;
                int64_t dstRepeatOffset = i * COMMON_MAX_REPEAT * tilingData->colAlignV;
                for (int64_t j = 0; j < formerColLoops; j++) {
                    int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
                    Cast(buffer[dstRepeatOffset + colOffset],
                        tmpBuffer.ReinterpretCast<T>()[srcRepeatOffset + colOffset],
                        RoundMode::CAST_NONE,
                        COMMON_B32_REPEAT_SIZE,
                        COMMON_MAX_REPEAT,
                        intriParams);
                }
                if (likely(remainderCol != 0)) {
                    int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
                    Cast(buffer[dstRepeatOffset + colOffset],
                        tmpBuffer.ReinterpretCast<T>()[srcRepeatOffset + colOffset],
                        RoundMode::CAST_NONE,
                        remainderCol,
                        COMMON_MAX_REPEAT,
                        intriParams);
                }
            }
            if (likely(remainderRepeat != 0)) {
                int64_t srcRepeatOffset = repeatLoops * COMMON_MAX_REPEAT * tilingData->colAlignM;
                int64_t dstRepeatOffset = repeatLoops * COMMON_MAX_REPEAT * tilingData->colAlignV;
                for (int64_t j = 0; j < formerColLoops; j++) {
                    int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
                    Cast(buffer[dstRepeatOffset + colOffset],
                        tmpBuffer.ReinterpretCast<T>()[srcRepeatOffset + colOffset],
                        RoundMode::CAST_NONE,
                        COMMON_B32_REPEAT_SIZE,
                        remainderRepeat,
                        intriParams);
                }
                if (likely(remainderCol != 0)) {
                    int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
                    Cast(buffer[dstRepeatOffset + colOffset],
                        tmpBuffer.ReinterpretCast<T>()[srcRepeatOffset + colOffset],
                        RoundMode::CAST_NONE,
                        remainderCol,
                        remainderRepeat,
                        intriParams);
                }
            }
        }
    }

    __aicore__ inline void CastToB16(const LocalTensor<float> &buffer, const LocalTensor<float> &tmpBuffer,
        const int64_t curRowsNum, const LayerNormGradV3TilingDataCommon *tilingData)
    {
        RoundMode b16RoundMode = IsSameType<T, bfloat16_t>::value ? RoundMode::CAST_ROUND : RoundMode::CAST_NONE;
        if (tilingData->colAlignM == tilingData->colAlignV || tilingData->colAlignV == tilingData->col) {
            Cast(buffer.ReinterpretCast<T>(), buffer, b16RoundMode, curRowsNum * tilingData->colAlignV);
        } else {
            DataCopyParams copyIntriParams;
            copyIntriParams.blockCount = 1;
            copyIntriParams.blockLen = curRowsNum * tilingData->colAlignV / COMMON_CONSTANT_EIGHT;
            copyIntriParams.srcStride = 0;
            copyIntriParams.dstStride = 0;
            DataCopy(tmpBuffer, buffer, copyIntriParams);
            PipeBarrier<PIPE_V>();
            int64_t formerColLoops = tilingData->colAlignV / COMMON_B32_REPEAT_SIZE;
            int64_t remainderCol = tilingData->colAlignV - formerColLoops * COMMON_B32_REPEAT_SIZE;
            int64_t repeatLoops = curRowsNum / COMMON_MAX_REPEAT;
            int64_t remainderRepeat = curRowsNum - repeatLoops * COMMON_MAX_REPEAT;
            UnaryRepeatParams intriParams;
            intriParams.dstBlkStride = 1;
            intriParams.srcBlkStride = 1;
            intriParams.dstRepStride = tilingData->colAlignM / COMMON_CONSTANT_SIXTEEN;
            intriParams.srcRepStride = tilingData->colAlignV / COMMON_CONSTANT_EIGHT;
            for (int64_t i = 0; i < repeatLoops; i++) {
                int64_t srcRepeatOffset = i * COMMON_MAX_REPEAT * tilingData->colAlignV;
                int64_t dstRepeatOffset = i * COMMON_MAX_REPEAT * tilingData->colAlignM;
                for (int64_t j = 0; j < formerColLoops; j++) {
                    int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
                    Cast(buffer.ReinterpretCast<T>()[dstRepeatOffset + colOffset],
                        tmpBuffer[srcRepeatOffset + colOffset],
                        b16RoundMode,
                        COMMON_B32_REPEAT_SIZE,
                        COMMON_MAX_REPEAT,
                        intriParams);
                }
                if (likely(remainderCol != 0)) {
                    int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
                    Cast(buffer.ReinterpretCast<T>()[dstRepeatOffset + colOffset],
                        tmpBuffer[srcRepeatOffset + colOffset],
                        b16RoundMode,
                        remainderCol,
                        COMMON_MAX_REPEAT,
                        intriParams);
                }
            }
            if (likely(remainderRepeat != 0)) {
                int64_t srcRepeatOffset = repeatLoops * COMMON_MAX_REPEAT * tilingData->colAlignV;
                int64_t dstRepeatOffset = repeatLoops * COMMON_MAX_REPEAT * tilingData->colAlignM;
                for (int64_t j = 0; j < formerColLoops; j++) {
                    int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
                    Cast(buffer.ReinterpretCast<T>()[dstRepeatOffset + colOffset],
                        tmpBuffer[srcRepeatOffset + colOffset],
                        b16RoundMode,
                        COMMON_B32_REPEAT_SIZE,
                        remainderRepeat,
                        intriParams);
                }
                if (likely(remainderCol != 0)) {
                    int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
                    Cast(buffer.ReinterpretCast<T>()[dstRepeatOffset + colOffset],
                        tmpBuffer[srcRepeatOffset + colOffset],
                        b16RoundMode,
                        remainderCol,
                        remainderRepeat,
                        intriParams);
                }
            }
        }
    }

    template <typename dType>
    __aicore__ inline void BlockBroadcast(
        const LocalTensor<dType> &dst, const LocalTensor<dType> &src, const int64_t curRowsNum)
    {
        // repeat must less than 255
        Brcb(dst, src, (curRowsNum + CONSTANT_EIGHT - 1) / CONSTANT_EIGHT, {1, CONSTANT_EIGHT});
    }

    __aicore__ inline void BinElemWithInlinedLastBrcFP32(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
        const LocalTensor<float> &src1, const int64_t curRowsNum, const LayerNormGradV3TilingDataCommon *tilingData,
        void (*func)(const LocalTensor<float> &, const LocalTensor<float> &, const LocalTensor<float> &, uint64_t,
            uint8_t, const BinaryRepeatParams &))
    {
        // src1 need to do inline broadcast
        int64_t formerColLoops = tilingData->colAlignV / COMMON_B32_REPEAT_SIZE;
        int64_t remainderCol = tilingData->colAlignV - formerColLoops * COMMON_B32_REPEAT_SIZE;
        int64_t repeatLoops = curRowsNum / COMMON_MAX_REPEAT;
        int64_t remainderRepeat = curRowsNum - repeatLoops * COMMON_MAX_REPEAT;
        BinaryRepeatParams intriParams;
        intriParams.dstBlkStride = 1;
        intriParams.src0BlkStride = 1;
        intriParams.src1BlkStride = 0;
        intriParams.dstRepStride = tilingData->colAlignV / COMMON_CONSTANT_EIGHT;
        intriParams.src0RepStride = tilingData->colAlignV / COMMON_CONSTANT_EIGHT;
        intriParams.src1RepStride = 1;
        for (int64_t i = 0; i < repeatLoops; i++) {
            int64_t repeatOffset = i * COMMON_MAX_REPEAT * tilingData->colAlignV;
            for (int64_t j = 0; j < formerColLoops; j++) {
                int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
                func(dst[repeatOffset + colOffset],
                    src0[repeatOffset + colOffset],
                    src1[i * COMMON_MAX_REPEAT * COMMON_B32_BLOCK_SIZE],
                    COMMON_B32_REPEAT_SIZE,
                    COMMON_MAX_REPEAT,
                    intriParams);
            }
            if (likely(remainderCol != 0)) {
                int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
                func(dst[repeatOffset + colOffset],
                    src0[repeatOffset + colOffset],
                    src1[i * COMMON_MAX_REPEAT * COMMON_B32_BLOCK_SIZE],
                    remainderCol,
                    COMMON_MAX_REPEAT,
                    intriParams);
            }
        }
        if (likely(remainderRepeat != 0)) {
            int64_t repeatOffset = repeatLoops * COMMON_MAX_REPEAT * tilingData->colAlignV;
            for (int64_t j = 0; j < formerColLoops; j++) {
                int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
                func(dst[repeatOffset + colOffset],
                    src0[repeatOffset + colOffset],
                    src1[repeatLoops * COMMON_MAX_REPEAT * COMMON_B32_BLOCK_SIZE],
                    COMMON_B32_REPEAT_SIZE,
                    remainderRepeat,
                    intriParams);
            }
            if (likely(remainderCol != 0)) {
                int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
                func(dst[repeatOffset + colOffset],
                    src0[repeatOffset + colOffset],
                    src1[repeatLoops * COMMON_MAX_REPEAT * COMMON_B32_BLOCK_SIZE],
                    remainderCol,
                    remainderRepeat,
                    intriParams);
            }
        }
    }

    __aicore__ inline void MulWithInlinedNLastBrcFP32(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
        const LocalTensor<float> &src1, const int64_t curRowsNum, const LayerNormGradV3TilingDataCommon *tilingData)
    {
        // src1 need to do inline broadcast
        int64_t formerColLoops = tilingData->colAlignV / COMMON_B32_REPEAT_SIZE;
        int64_t remainderCol = tilingData->colAlignV - formerColLoops * COMMON_B32_REPEAT_SIZE;
        int64_t repeatLoops = curRowsNum / COMMON_MAX_REPEAT;
        int64_t remainderRepeat = curRowsNum - repeatLoops * COMMON_MAX_REPEAT;
        BinaryRepeatParams intriParams;
        intriParams.dstBlkStride = 1;
        intriParams.src0BlkStride = 1;
        intriParams.src1BlkStride = 1;
        intriParams.dstRepStride = tilingData->colAlignV / COMMON_CONSTANT_EIGHT;
        intriParams.src0RepStride = tilingData->colAlignV / COMMON_CONSTANT_EIGHT;
        intriParams.src1RepStride = 0;
        for (int64_t i = 0; i < repeatLoops; i++) {
            int64_t repeatOffset = i * COMMON_MAX_REPEAT * tilingData->colAlignV;
            for (int64_t j = 0; j < formerColLoops; j++) {
                int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
                Mul(dst[repeatOffset + colOffset],
                    src0[repeatOffset + colOffset],
                    src1[colOffset],
                    COMMON_B32_REPEAT_SIZE,
                    COMMON_MAX_REPEAT,
                    intriParams);
            }
            if (likely(remainderCol != 0)) {
                int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
                Mul(dst[repeatOffset + colOffset],
                    src0[repeatOffset + colOffset],
                    src1[colOffset],
                    remainderCol,
                    COMMON_MAX_REPEAT,
                    intriParams);
            }
        }
        if (likely(remainderRepeat != 0)) {
            int64_t repeatOffset = repeatLoops * COMMON_MAX_REPEAT * tilingData->colAlignV;
            for (int64_t j = 0; j < formerColLoops; j++) {
                int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
                Mul(dst[repeatOffset + colOffset],
                    src0[repeatOffset + colOffset],
                    src1[colOffset],
                    COMMON_B32_REPEAT_SIZE,
                    remainderRepeat,
                    intriParams);
            }
            if (likely(remainderCol != 0)) {
                int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
                Mul(dst[repeatOffset + colOffset],
                    src0[repeatOffset + colOffset],
                    src1[colOffset],
                    remainderCol,
                    remainderRepeat,
                    intriParams);
            }
        }
    }

    __aicore__ inline void NlastReduceSum(const LocalTensor<float> &dst, const LocalTensor<float> &src,
        const int64_t curRowsNum, const LayerNormGradV3TilingDataCommon *tilingData)
    {
        int64_t formerColLoops = tilingData->colAlignV / COMMON_B32_REPEAT_SIZE;
        int64_t remainderCol = tilingData->colAlignV - formerColLoops * COMMON_B32_REPEAT_SIZE;
        int64_t repeatLoops = curRowsNum / COMMON_MAX_REPEAT;
        int64_t remainderRepeat = curRowsNum - repeatLoops * COMMON_MAX_REPEAT;
        BinaryRepeatParams intriParams;
        intriParams.dstBlkStride = 1;
        intriParams.src0BlkStride = 1;
        intriParams.src1BlkStride = 1;
        intriParams.dstRepStride = 0;
        intriParams.src0RepStride = tilingData->colAlignV / COMMON_CONSTANT_EIGHT;
        intriParams.src1RepStride = 0;
        for (int64_t i = 0; i < repeatLoops; i++) {
            int64_t repeatOffset = i * COMMON_MAX_REPEAT * tilingData->colAlignV;
            for (int64_t j = 0; j < formerColLoops; j++) {
                int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
                Add(dst[colOffset],
                    src[repeatOffset + colOffset],
                    dst[colOffset],
                    COMMON_B32_REPEAT_SIZE,
                    COMMON_MAX_REPEAT,
                    intriParams);
            }
            if (likely(remainderCol != 0)) {
                int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
                Add(dst[colOffset],
                    src[repeatOffset + colOffset],
                    dst[colOffset],
                    remainderCol,
                    COMMON_MAX_REPEAT,
                    intriParams);
            }
            PipeBarrier<PIPE_V>();
        }
        if (likely(remainderRepeat != 0)) {
            int64_t repeatOffset = repeatLoops * COMMON_MAX_REPEAT * tilingData->colAlignV;
            for (int64_t j = 0; j < formerColLoops; j++) {
                int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
                Add(dst[colOffset],
                    src[repeatOffset + colOffset],
                    dst[colOffset],
                    COMMON_B32_REPEAT_SIZE,
                    remainderRepeat,
                    intriParams);
            }
            if (likely(remainderCol != 0)) {
                int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
                Add(dst[colOffset],
                    src[repeatOffset + colOffset],
                    dst[colOffset],
                    remainderCol,
                    remainderRepeat,
                    intriParams);
            }
        }
    }

    __aicore__ inline void LastReduceSum(const LocalTensor<float> &dst, const LocalTensor<float> &src,
        const LocalTensor<float> &tmp, const int64_t curRowsNum, const LayerNormGradV3TilingDataCommon *tilingData)
    {
        if (tilingData->colAlignV <= COMMON_B32_REPEAT_SIZE) {
            // curRowsNum may larger than 255
            int64_t repeatLoops = curRowsNum / COMMON_VC_MAX_REPEAT;
            int64_t remainderRepeat = curRowsNum - repeatLoops * COMMON_VC_MAX_REPEAT;
            for (int64_t i = 0; i < repeatLoops; i++) {
                WholeReduceSum(dst[i * COMMON_VC_MAX_REPEAT],
                    src[i * COMMON_VC_MAX_REPEAT * tilingData->colAlignV],
                    tilingData->colAlignV,
                    COMMON_VC_MAX_REPEAT,
                    1,
                    1,
                    tilingData->colAlignV / COMMON_CONSTANT_EIGHT);
            }
            if (likely(remainderRepeat != 0)) {
                WholeReduceSum(dst[repeatLoops * COMMON_VC_MAX_REPEAT],
                    src[repeatLoops * COMMON_VC_MAX_REPEAT * tilingData->colAlignV],
                    tilingData->colAlignV,
                    remainderRepeat,
                    1,
                    1,
                    tilingData->colAlignV / COMMON_CONSTANT_EIGHT);
            }
        } else {
            // curRowsNum must less than 255
            DataCopyParams copyIntriParams;
            copyIntriParams.blockCount = curRowsNum;
            copyIntriParams.blockLen = COMMON_CONSTANT_EIGHT;
            copyIntriParams.srcStride = tilingData->colAlignV / COMMON_CONSTANT_EIGHT - COMMON_CONSTANT_EIGHT;
            copyIntriParams.dstStride = 0;
            // copy first REPEAT_B32_SIZE elements of each row
            DataCopy(tmp, src, copyIntriParams);
            PipeBarrier<PIPE_V>();
            int64_t formerColLoops = tilingData->colAlignV / COMMON_B32_REPEAT_SIZE;
            int64_t remainderCol = tilingData->colAlignV - formerColLoops * COMMON_B32_REPEAT_SIZE;
            BinaryRepeatParams intriParams;
            intriParams.dstBlkStride = 1;
            intriParams.src0BlkStride = 1;
            intriParams.src1BlkStride = 1;
            intriParams.dstRepStride = COMMON_CONSTANT_EIGHT;
            intriParams.src0RepStride = COMMON_CONSTANT_EIGHT;
            intriParams.src1RepStride = tilingData->colAlignV / COMMON_CONSTANT_EIGHT;
            for (int64_t i = 1; i < formerColLoops; i++) {
                Add(tmp, tmp, src[i * COMMON_B32_REPEAT_SIZE], COMMON_B32_REPEAT_SIZE, curRowsNum, intriParams);
                PipeBarrier<PIPE_V>();
            }
            if (likely(remainderCol != 0)) {
                Add(tmp, tmp, src[formerColLoops * COMMON_B32_REPEAT_SIZE], remainderCol, curRowsNum, intriParams);
            }
            PipeBarrier<PIPE_V>();
            WholeReduceSum(dst, tmp, COMMON_B32_REPEAT_SIZE, curRowsNum, 1, 1, COMMON_CONSTANT_EIGHT);
        }
    }

private:
    TPipe pipe_;
    constexpr static uint16_t SYNC_AIV_ONLY_ALL = 14;

    TQue<QuePosition::VECIN, 1> queue0_;
    TQue<QuePosition::VECIN, 1> queue1_;
    TQue<QuePosition::VECIN, 1> queue2_;
    TQue<QuePosition::VECIN, 1> queue3_;
    TQue<QuePosition::VECIN, 1> queue4_;
    TQue<QuePosition::VECOUT, 1> queue5_;
    TQue<QuePosition::VECOUT, 1> queue6_;
    TQue<QuePosition::VECOUT, 1> queue7_;
    TBuf<> tmpTensor0_;
    TBuf<> tmpTensor1_;
    TBuf<> tmpTensor2_;

    LocalTensor<float> buffer0_;
    LocalTensor<float> buffer1_;
    LocalTensor<float> buffer2_;
    LocalTensor<float> buffer3_;
    LocalTensor<float> buffer4_;
    LocalTensor<float> buffer5_;
    LocalTensor<float> buffer6_;
    LocalTensor<float> buffer7_;
    LocalTensor<float> buffer8_;
    LocalTensor<float> buffer9_;
    LocalTensor<float> buffer10_;

    GlobalTensor<T> dyInTensorGM_;
    GlobalTensor<T> xInTensorGM_;
    GlobalTensor<float> rstdInTensorGM_;
    GlobalTensor<float> meanInTensorGM_;
    GlobalTensor<U> gammaInTensorGM_;
    GlobalTensor<T> pdXOutTensorGM_;
    GlobalTensor<float> pdGammaOutTensorGM_;
    GlobalTensor<float> pdBetaOutTensorGM_;
    GlobalTensor<float> workspaceGM_;

    float coff_;
};
}  // namespace LayerNormGradV3
#endif
