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
 * \file layer_norm_grad_v3_workspace.h
 * \brief
 */

#ifndef LAYER_NORM_GRAD_V3_WORKSPACE
#define LAYER_NORM_GRAD_V3_WORKSPACE

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace LayerNormGradV3 {
using namespace AscendC;

template <typename T, typename U, bool isDeterministic>
class LayerNormGradV3Workspace {
public:
    __aicore__ inline LayerNormGradV3Workspace(){};
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean, GM_ADDR gamma, GM_ADDR pdX,
        GM_ADDR pdGamma, GM_ADDR pdBeta, GM_ADDR workspace, const LayerNormGradV3TilingDataWorkspace *tilingData)
    {
        // init gm inputs
        int64_t formerBlockLength = tilingData->blockFormer * tilingData->col;
        int64_t curRowsNum =
            (GetBlockIdx() != tilingData->blockNum - 1) ? tilingData->blockFormer : tilingData->blockTail;
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
        colLen = tilingData->col;
        colAlign = tilingData->colAlignV;
        int64_t wsLenPerBlock = colAlign * WORKSPACE_NUM;
        workspaceGM.SetGlobalBuffer((__gm__ float *)workspace + wsLenPerBlock * GetBlockIdx(), wsLenPerBlock);
        dGammaWorkspaceGM.SetGlobalBuffer(
            (__gm__ float *)workspace + wsLenPerBlock * GetBlockIdx() + colAlign, colAlign);
        mul1WorkspaceGM.SetGlobalBuffer(
            (__gm__ float *)workspace + wsLenPerBlock * GetBlockIdx() + colAlign * 2, colAlign);
        mul3WorkspaceGM.SetGlobalBuffer(
            (__gm__ float *)workspace + wsLenPerBlock * GetBlockIdx() + colAlign * 3, colAlign);

        // clear workspace and pdGamma/pdBeta outputs
        InitOutput<float>(workspaceGM, colAlign + colAlign, 0.0);
        if constexpr (!isDeterministic) {
            if (GetBlockIdx() == 0) {
                InitOutput<float>(pdGammaOutTensorGM, tilingData->col, 0.0);
                InitOutput<float>(pdBetaOutTensorGM, tilingData->col, 0.0);
            }
            CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_AIV_ONLY_ALL);
        }
        PipeBarrier<PIPE_ALL>();
        // init queues
        int64_t bufferSize = tilingData->ubFormer * sizeof(float);
        pipe.InitBuffer(queIn0, 1, bufferSize);
        pipe.InitBuffer(queIn1, 1, bufferSize);
        pipe.InitBuffer(queIn2, 1, bufferSize);
        pipe.InitBuffer(queOut3, 1, bufferSize);
        pipe.InitBuffer(queOut4, 1, bufferSize);
        pipe.InitBuffer(queOut5, 1, bufferSize);
    }

    __aicore__ inline void Process(const LayerNormGradV3TilingDataWorkspace *tilingData)
    {
        int64_t rowCount =
            (GetBlockIdx() == tilingData->blockNum - 1) ? tilingData->blockTail : tilingData->blockFormer;
        for (int64_t rowIndex = 0; rowIndex < rowCount; rowIndex++) {
            float reduce2 = 0.0f;
            float reduce3 = 0.0f;
            float meanIn = meanInTensorGM.GetValue(rowIndex);
            float rstdIn = rstdInTensorGM.GetValue(rowIndex);

            // step 1. calc reduce2 reduce3 mul_1 mul3
            for (int64_t ubIndex = 0; ubIndex < tilingData->ubLoop - 1; ubIndex++) {
                CopyInPhase0(rowIndex, ubIndex, tilingData->ubFormer, tilingData->ubFormer);
                ComputePhase0(ubIndex, tilingData->ubFormer, tilingData->ubFormer, meanIn, rstdIn);
                reduce2 += ReducePhase0(buffer5, tilingData->ubFormer);
                reduce3 += ReducePhase0(buffer4, tilingData->ubFormer);
            }
            CopyInPhase0(rowIndex, tilingData->ubLoop - 1, tilingData->ubFormer, tilingData->ubTail);
            ComputePhase0(tilingData->ubLoop - 1, tilingData->ubFormer, tilingData->ubTail, meanIn, rstdIn);
            reduce2 += ReducePhase0(buffer5, tilingData->ubTail);
            reduce3 += ReducePhase0(buffer4, tilingData->ubTail);
            reduce2 = reduce2 / tilingData->col * (-1.0f);
            reduce3 = reduce3 / tilingData->col;

            // step 2. calc dx
            PipeBarrier<PIPE_ALL>();
            for (int64_t ubIndex = 0; ubIndex < tilingData->ubLoop - 1; ubIndex++) {
                CopyInPhase1(ubIndex, tilingData->ubFormer, tilingData->ubFormer);
                ComputePhase1(rowIndex, ubIndex, tilingData->ubFormer, tilingData->ubFormer, reduce2, reduce3, rstdIn);
            }
            CopyInPhase1(tilingData->ubLoop - 1, tilingData->ubFormer, tilingData->ubTail);
            ComputePhase1(
                rowIndex, tilingData->ubLoop - 1, tilingData->ubFormer, tilingData->ubTail, reduce2, reduce3, rstdIn);
        }

        // step3. calc pgamma and pbeta form workspace
        if constexpr (isDeterministic) {
            SyncAll();
        } else {
            PipeBarrier<PIPE_ALL>();
            CrossCoreWaitFlag(SYNC_AIV_ONLY_ALL);
        }
        for (int64_t ubIndex = 0; ubIndex < tilingData->ubLoop - 1; ubIndex++) {
            FinalProcess(ubIndex, tilingData->ubFormer, tilingData->ubFormer, tilingData->blockNum);
        }
        FinalProcess(tilingData->ubLoop - 1, tilingData->ubFormer, tilingData->ubTail, tilingData->blockNum);
    }

    __aicore__ inline void CopyInPhase0(int64_t rowIndex, int64_t ubIndex, int64_t ubFormer, int64_t calcNum)
    {
        buffer0 = queIn0.AllocTensor<float>();
        buffer1 = queIn1.AllocTensor<float>();
        buffer2 = queIn2.AllocTensor<float>();

        DataCopyParams intriParamsT = {1, static_cast<uint16_t>(calcNum * sizeof(T)), 0, 0};
        DataCopyParams intriParamsU = {1, static_cast<uint16_t>(calcNum * sizeof(U)), 0, 0};
        DataCopyPadParams padParams = {false, 0, 0, 0};
        int64_t colOffset = ubIndex * ubFormer;
        int64_t rowOffset = rowIndex * colLen + colOffset;
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(buffer0, dyInTensorGM[rowOffset], intriParamsT, padParams);
            DataCopyPad(buffer1, xInTensorGM[rowOffset], intriParamsT, padParams);
        } else {
            DataCopyPad(buffer0.ReinterpretCast<T>()[ubFormer], dyInTensorGM[rowOffset], intriParamsT, padParams);
            DataCopyPad(buffer1.ReinterpretCast<T>()[ubFormer], xInTensorGM[rowOffset], intriParamsT, padParams);
        }
        queIn0.EnQue(buffer0);
        queIn1.EnQue(buffer1);

        if constexpr (IsSameType<U, float>::value) {
            DataCopyPad(buffer2, gammaInTensorGM[colOffset], intriParamsU, padParams);
        } else {
            DataCopyPad(buffer2.ReinterpretCast<U>()[ubFormer], gammaInTensorGM[colOffset], intriParamsU, padParams);
        }
        queIn2.EnQue(buffer2);
    }

    __aicore__ inline void ComputePhase0(int64_t ubIndex, int64_t ubFormer, int64_t calcNum, float meanIn, float rstdIn)
    {
        buffer3 = queOut3.AllocTensor<float>();
        buffer4 = queOut4.AllocTensor<float>();
        buffer5 = queOut5.AllocTensor<float>();

        DataCopyExtParams intriParams = {1, static_cast<uint32_t>(calcNum * sizeof(float)), 0, 0, 0};

        buffer0 = queIn0.DeQue<float>();
        if constexpr (!IsSameType<T, float>::value) {
            Cast(buffer0, buffer0.ReinterpretCast<T>()[ubFormer], RoundMode::CAST_NONE, calcNum);
            PipeBarrier<PIPE_V>();
        }
        Adds(buffer3, buffer0, 0.0f, calcNum);

        buffer1 = queIn1.DeQue<float>();
        if constexpr (!IsSameType<T, float>::value) {
            Cast(buffer1, buffer1.ReinterpretCast<T>()[ubFormer], RoundMode::CAST_NONE, calcNum);
            PipeBarrier<PIPE_V>();
        }
        Adds(buffer1, buffer1, -meanIn, calcNum);

        queOut3.EnQue(buffer3);
        buffer3 = queOut3.DeQue<float>();

        SetAtomicAdd<float>();
        DataCopyPad(workspaceGM[ubIndex * ubFormer], buffer3, intriParams);  // workspace dbeta
        SetAtomicNone();
        queOut3.FreeTensor(buffer3);

        PipeBarrier<PIPE_V>();
        Muls(buffer4, buffer1, rstdIn, calcNum);

        buffer3 = queOut3.AllocTensor<float>();
        PipeBarrier<PIPE_V>();
        Mul(buffer3, buffer0, buffer4, calcNum);

        queOut4.EnQue(buffer4);
        buffer4 = queOut4.DeQue<float>();
        DataCopyPad(mul1WorkspaceGM[ubIndex * ubFormer], buffer4, intriParams);  // workspace mul_1
        queOut4.FreeTensor(buffer4);

        buffer2 = queIn2.DeQue<float>();
        if constexpr (!IsSameType<U, float>::value) {
            Cast(buffer2, buffer2.ReinterpretCast<U>()[ubFormer], RoundMode::CAST_NONE, calcNum);
            PipeBarrier<PIPE_V>();
        }
        Mul(buffer5, buffer2, buffer0, calcNum);
        PipeBarrier<PIPE_V>();

        queOut3.EnQue(buffer3);
        buffer3 = queOut3.DeQue<float>();
        SetAtomicAdd<float>();
        DataCopyPad(dGammaWorkspaceGM[ubIndex * ubFormer], buffer3, intriParams);  // workspace dgamma
        SetAtomicNone();

        queOut5.EnQue(buffer5);
        buffer5 = queOut5.DeQue<float>();
        DataCopyPad(mul3WorkspaceGM[ubIndex * ubFormer], buffer5, intriParams);  // workspace mul_3

        buffer4 = queOut4.AllocTensor<float>();
        Mul(buffer4, buffer4, buffer5, calcNum);

        FreeTensor();
    }

    __aicore__ inline void FreeTensor()
    {
        queIn0.FreeTensor(buffer0);
        queIn1.FreeTensor(buffer1);
        queIn2.FreeTensor(buffer2);
        queOut3.FreeTensor(buffer3);
        queOut4.FreeTensor(buffer4);
        queOut5.FreeTensor(buffer5);
    }

    __aicore__ inline float ReducePhase0(const LocalTensor<float> &src, int64_t reduceNum)
    {
        AscendCUtils::SetMaskCount<float>();
        SetVectorMask<float>(0, reduceNum);
        ReduceSum(src, src, src, COUNT_NUM);
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        uint64_t acc_val = GetAccVal();
        AscendCUtils::SetMaskNorm<float>();

        float value = *reinterpret_cast<float *>(&acc_val);
        return value;
    }

    __aicore__ inline void CopyInPhase1(int64_t ubIndex, int64_t ubFormer, int64_t calcNum)
    {
        buffer0 = queIn0.AllocTensor<float>();
        buffer1 = queIn1.AllocTensor<float>();
        DataCopyExtParams intriParams = {1, static_cast<uint32_t>(calcNum * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams padParams = {false, 0, 0, 0.0f};
        DataCopyPad(buffer0, mul1WorkspaceGM[ubIndex * ubFormer], intriParams, padParams);
        queIn0.EnQue(buffer0);
        DataCopyPad(buffer1, mul3WorkspaceGM[ubIndex * ubFormer], intriParams, padParams);
        queIn1.EnQue(buffer1);
    }

    __aicore__ inline void ComputePhase1(int64_t rowIndex, int64_t ubIndex, int64_t ubFormer, int64_t calcNum,
        float reduce2, float reduce3, float rstdIn)
    {
        buffer4 = queOut4.AllocTensor<float>();
        buffer0 = queIn0.DeQue<float>();
        buffer1 = queIn1.DeQue<float>();
        Muls(buffer0, buffer0, reduce3, calcNum);
        PipeBarrier<PIPE_V>();
        Sub(buffer1, buffer1, buffer0, calcNum);
        PipeBarrier<PIPE_V>();
        Adds(buffer1, buffer1, reduce2, calcNum);
        PipeBarrier<PIPE_V>();

        if constexpr (IsSameType<T, float>::value) {
            Muls(buffer4, buffer1, rstdIn, calcNum);
        } else {
            Muls(buffer5, buffer1, rstdIn, calcNum);
            PipeBarrier<PIPE_V>();
            if constexpr (IsSameType<T, half>::value) {
                Cast(buffer4.ReinterpretCast<T>(), buffer5, RoundMode::CAST_NONE, calcNum);
            } else {
                Cast(buffer4.ReinterpretCast<T>(), buffer5, RoundMode::CAST_ROUND, calcNum);
            }
        }
        queOut4.EnQue(buffer4);
        buffer4 = queOut4.DeQue<float>();
        DataCopyExtParams intriParams = {1, static_cast<uint32_t>(calcNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(pdXOutTensorGM[rowIndex * colLen + ubIndex * ubFormer], buffer4.ReinterpretCast<T>(), intriParams);
        queIn0.FreeTensor(buffer0);
        queIn1.FreeTensor(buffer1);
        queOut4.FreeTensor(buffer4);
    }

    __aicore__ inline void FinalProcess(int64_t ubIndex, int64_t ubFormer, int64_t calcNum, int64_t blockNum)
    {
        if constexpr (isDeterministic) {
            FinalProcessDeterministic(ubIndex, ubFormer, calcNum, blockNum);
        } else {
            buffer0 = queIn0.AllocTensor<float>();
            buffer1 = queIn1.AllocTensor<float>();
            buffer3 = queOut3.AllocTensor<float>();
            buffer4 = queOut4.AllocTensor<float>();
            DataCopyExtParams intriParams = {1, static_cast<uint32_t>(calcNum * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams padParams = {false, 0, 0, 0.0f};
            int64_t offset = ubIndex * ubFormer;

            DataCopyPad(buffer0, dGammaWorkspaceGM[offset], intriParams, padParams);
            queIn0.EnQue(buffer0);
            buffer0 = queIn0.DeQue<float>();
            Adds(buffer3, buffer0, 0.0f, calcNum);
            queOut3.EnQue(buffer3);
            buffer3 = queOut3.DeQue<float>();
            SetAtomicAdd<float>();
            DataCopyPad(pdGammaOutTensorGM[offset], buffer3, intriParams);
            SetAtomicNone();

            DataCopyPad(buffer1, workspaceGM[offset], intriParams, padParams);
            queIn1.EnQue(buffer1);
            buffer1 = queIn1.DeQue<float>();
            Adds(buffer4, buffer1, 0.0f, calcNum);
            queOut4.EnQue(buffer4);
            buffer4 = queOut4.DeQue<float>();
            SetAtomicAdd<float>();
            DataCopyPad(pdBetaOutTensorGM[offset], buffer4, intriParams);
            SetAtomicNone();
            queIn0.FreeTensor(buffer0);
            queIn1.FreeTensor(buffer1);
            queOut3.FreeTensor(buffer3);
            queOut4.FreeTensor(buffer4);
        }
    }

    __aicore__ inline void FinalProcessDeterministic(
        int64_t ubIndex, int64_t ubFormer, int64_t calcNum, int64_t blockNum)
    {
        if (GetBlockIdx() == 0) {
            buffer3 = queOut3.AllocTensor<float>();
            buffer4 = queOut4.AllocTensor<float>();
            DataCopyExtParams intriParams = {1, static_cast<uint32_t>(calcNum * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams padParams = {false, 0, 0, 0.0f};
            int64_t offset = ubIndex * ubFormer;

            Duplicate(buffer3, static_cast<float>(0.0), ubFormer);
            Duplicate(buffer4, static_cast<float>(0.0), ubFormer);
            for (int64_t i = 0; i < blockNum; i++) {
                buffer0 = queIn0.AllocTensor<float>();
                DataCopyPad(
                    buffer0, workspaceGM[i * colAlign * WORKSPACE_NUM + colAlign + offset], intriParams, padParams);
                queIn0.EnQue(buffer0);
                buffer0 = queIn0.DeQue<float>();
                Add(buffer3, buffer3, buffer0, calcNum);
                queIn0.FreeTensor(buffer0);

                buffer1 = queIn1.AllocTensor<float>();
                DataCopyPad(buffer1, workspaceGM[i * colAlign * WORKSPACE_NUM + offset], intriParams, padParams);
                queIn1.EnQue(buffer1);
                buffer1 = queIn1.DeQue<float>();
                Add(buffer4, buffer4, buffer1, calcNum);
                queIn1.FreeTensor(buffer1);
            }
            queOut3.EnQue(buffer3);
            buffer3 = queOut3.DeQue<float>();
            DataCopyPad(pdGammaOutTensorGM[offset], buffer3, intriParams);

            queOut4.EnQue(buffer4);
            buffer4 = queOut4.DeQue<float>();
            DataCopyPad(pdBetaOutTensorGM[offset], buffer4, intriParams);
            queOut3.FreeTensor(buffer3);
            queOut4.FreeTensor(buffer4);
        }
    }

private:
    constexpr static int64_t WORKSPACE_NUM = 4;
    constexpr static uint16_t SYNC_AIV_ONLY_ALL = 14;
    constexpr static uint32_t COUNT_NUM = 64;

    TPipe pipe;

    TQue<QuePosition::VECIN, 1> queIn0;
    TQue<QuePosition::VECIN, 1> queIn1;
    TQue<QuePosition::VECIN, 1> queIn2;
    TQue<QuePosition::VECOUT, 1> queOut3;
    TQue<QuePosition::VECOUT, 1> queOut4;
    TQue<QuePosition::VECOUT, 1> queOut5;

    LocalTensor<float> buffer0;
    LocalTensor<float> buffer1;
    LocalTensor<float> buffer2;
    LocalTensor<float> buffer3;
    LocalTensor<float> buffer4;
    LocalTensor<float> buffer5;

    GlobalTensor<T> dyInTensorGM;
    GlobalTensor<T> xInTensorGM;
    GlobalTensor<float> rstdInTensorGM;
    GlobalTensor<float> meanInTensorGM;
    GlobalTensor<U> gammaInTensorGM;
    GlobalTensor<T> pdXOutTensorGM;
    GlobalTensor<float> pdGammaOutTensorGM;
    GlobalTensor<float> pdBetaOutTensorGM;
    GlobalTensor<float> workspaceGM;
    GlobalTensor<float> dGammaWorkspaceGM;
    GlobalTensor<float> mul1WorkspaceGM;
    GlobalTensor<float> mul3WorkspaceGM;

    int64_t colLen;
    int64_t colAlign;
};
}  // namespace LayerNormGradV3
#endif