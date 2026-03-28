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
 * \file add_layer_norm_grad_cut_n.h
 * \brief
 */
#ifndef OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_LAYER_NORM_GRAD_ADD_LAYER_NORM_GRAD_CUT_N
#define OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_LAYER_NORM_GRAD_ADD_LAYER_NORM_GRAD_CUT_N
#include "add_layer_norm_grad_utils.h"

using namespace AscendC;

template <typename T, int TILING_KEY>
class KernelAddLayerNormGrad {
#define HAS_ADDITIONAL_INPUT ((TILING_KEY % 10) == 1)
public:
    __aicore__ inline KernelAddLayerNormGrad()
    {}

    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x_1, GM_ADDR x_2, GM_ADDR rstd, GM_ADDR mean, GM_ADDR gamma,
        GM_ADDR dsum, GM_ADDR d_x, GM_ADDR d_gamma, GM_ADDR d_beta, AddLayerNormGradTilingData tiling,
        GM_ADDR workspace)
    {
        InitVar(tiling);
        InitInputGmBuffer(dy, x_1, x_2, rstd, mean, gamma, dsum);
        InitInputQueue();
        InitOutputQueue();
        InitTmpBuffer(workspace);
        InitOuputGmBuffer(d_x, d_gamma, d_beta);
#if __CCE_AICORE__ == 220
        SyncAll();
#else
        uint32_t each_core_handle_num = BLOCK_AlIGN / sizeof(int32_t);
        GlobalTensor<int32_t> syncGlobal_;
        syncGlobal_.SetGlobalBuffer((__gm__ int32_t *)workspace, numCore * FLOAT_BLOCK_ELEM);

        LocalTensor<int32_t> tmp_init_buf = dBetaQue.AllocTensor<int32_t>();
        Duplicate(tmp_init_buf, 0, each_core_handle_num);
        DataCopy(syncGlobal_[each_core_handle_num * GetBlockIdx()], tmp_init_buf, each_core_handle_num);

        LocalTensor<int32_t> workLocal = dGammaQue.AllocTensor<int32_t>();
        SyncAll(syncGlobal_, workLocal);
        dGammaQue.FreeTensor(workLocal);
        dBetaQue.FreeTensor(tmp_init_buf);
#endif
    }
    __aicore__ inline void InitVar(AddLayerNormGradTilingData tiling)
    {
        numCore = tiling.numCore;
        numLastDim = tiling.numLastDim;
        numFirstDim = tiling.numFirstDim;
        nInOnecoreLength = tiling.nInOneCoreLength;
        nInOnecoreLengthTail = tiling.nInOneCoreLengthTail;
        nDInOnecoreLength = tiling.ndInOneCoreLength;
        nAvailInUb = tiling.nAvailInUb;
        dInnerLength = tiling.dInnerLength;
        dInnerLengthTail = tiling.dInnerLengthTail;
        dOuterLength = tiling.dOuterLength;
        dyPadRight = tiling.dyPadRight;
        rstdPadRight = tiling.rstdPadRight;
        roundUpNumLastDim = tiling.roundUpNumLastDim;
        roundUpNumLastDimDtype = tiling.roundUpNumLastDimDtype;
        roundUp1Dtype = tiling.roundUp1Dtype;
        roundUpNumLastDimFloat = tiling.roundUpNumLastDimFloat;

        if (GetBlockIdx() != numCore - 1) {
            nInOneCore = tiling.nInOneCoreNorm;
            gmOneCoreElemXY = tiling.gmOneCoreElemXYNorm;
            nAvailInUb = tiling.nAvailInUbNorm;
            nMiddleCount = tiling.nMiddleCountNorm;
            nInUbTotalTail = tiling.nInUbTotalNormTail;
            nDRoundUpDtype = tiling.ndRoundUpDtypeNorm;
            n1RoundUpFloat = tiling.n1RoundUpFloatNorm;
        } else {
            nInOneCore = tiling.nInOneCoreTail;
            gmOneCoreElemXY = tiling.gmOneCoreElemXYTail;
            nAvailInUb = tiling.nAvailInUbTail;
            nMiddleCount = tiling.nMiddleCountTail;
            nInUbTotalTail = tiling.nInUbTotalTailTail;
            nDRoundUpDtype = tiling.ndRoundUpDtypeTail;
            n1RoundUpFloat = tiling.n1RoundUpFloatTail;
        }
        dyPadRight = tiling.dyPadRight;
        rstdPadRight = tiling.rstdPadRight;

        blockNumber = BLOCK_AlIGN / sizeof(float);

#if __CCE_AICORE__ == 220
        if constexpr (IsSame<T, half>::value || IsSame<T, bfloat16_t>::value) {
#else
        if constexpr (IsSame<T, half>::value) {
#endif
            blockNumberTdtype = BLOCK_AlIGN / sizeof(half);
        } else {
            blockNumberTdtype = BLOCK_AlIGN / sizeof(float);
        }
        eachCoreHandleNum = BLOCK_AlIGN / sizeof(int32_t);

        offsetGmXY = GetBlockIdx() * nDInOnecoreLength;
        offsetGmMeanVar = GetBlockIdx() * nInOnecoreLength;
        offsetGmGamma = 0;
    }

    __aicore__ inline void InitInputGmBuffer(
        GM_ADDR dy, GM_ADDR x_1, GM_ADDR x_2, GM_ADDR rstd, GM_ADDR mean, GM_ADDR gamma, GM_ADDR dsum)
    {
        dyGm.SetGlobalBuffer((__gm__ T *)dy + GetBlockIdx() * nDInOnecoreLength, gmOneCoreElemXY);
        x1Gm.SetGlobalBuffer((__gm__ T *)x_1 + GetBlockIdx() * nDInOnecoreLength, gmOneCoreElemXY);
        x2Gm.SetGlobalBuffer((__gm__ T *)x_2 + GetBlockIdx() * nDInOnecoreLength, gmOneCoreElemXY);
        rstdGm.SetGlobalBuffer((__gm__ float *)rstd + GetBlockIdx() * nInOnecoreLength, nInOneCore);
        meanGm.SetGlobalBuffer((__gm__ float *)mean + GetBlockIdx() * nInOnecoreLength, nInOneCore);
        gammaGm.SetGlobalBuffer((__gm__ T *)gamma, numLastDim);
        if constexpr (HAS_ADDITIONAL_INPUT) {
            dSumGm.SetGlobalBuffer((__gm__ T *)dsum + GetBlockIdx() * nDInOnecoreLength, gmOneCoreElemXY);
        }
    }

    __aicore__ inline void InitOuputGmBuffer(GM_ADDR d_x, GM_ADDR d_gamma, GM_ADDR d_beta)
    {
        dXGm.SetGlobalBuffer((__gm__ T *)d_x + GetBlockIdx() * nDInOnecoreLength, gmOneCoreElemXY);
        dGammaGm.SetGlobalBuffer((__gm__ float *)d_gamma, numLastDim);
        dBetaGm.SetGlobalBuffer((__gm__ float *)d_beta, numLastDim);
        LocalTensor<float> temp_local_tensor = dGammaQue.AllocTensor<float>();
        InitGmData(dGammaGm, dBetaGm, numLastDim, temp_local_tensor, roundUpNumLastDimFloat);
        dGammaQue.FreeTensor(temp_local_tensor);
    }

    __aicore__ inline void InitInputQueue()
    {
        pipe.InitBuffer(dyQue, BUFFER_NUM, nDRoundUpDtype);
        pipe.InitBuffer(x1Que, BUFFER_NUM, nDRoundUpDtype);
        pipe.InitBuffer(x2Que, BUFFER_NUM, nDRoundUpDtype);
        pipe.InitBuffer(rstdQue, BUFFER_NUM, n1RoundUpFloat);
        pipe.InitBuffer(meanQue, BUFFER_NUM, n1RoundUpFloat);
        pipe.InitBuffer(GammaQue, BUFFER_NUM, roundUpNumLastDimDtype);
        if constexpr (HAS_ADDITIONAL_INPUT) {
            pipe.InitBuffer(dSumQue, BUFFER_NUM, nDRoundUpDtype);
        }
    }

    __aicore__ inline void InitOutputQueue()
    {
        pipe.InitBuffer(dXQue, BUFFER_NUM, nDRoundUpDtype);
        pipe.InitBuffer(dGammaQue, BUFFER_NUM, roundUpNumLastDimFloat);
        pipe.InitBuffer(dBetaQue, BUFFER_NUM, roundUpNumLastDimFloat);
    }

    __aicore__ inline void InitTmpBuffer(GM_ADDR workspace)
    {
        if constexpr (IsSame<T, float>::value) {
        } else {
            pipe.InitBuffer(dyFp32Buf, roundUpNumLastDimFloat);
            pipe.InitBuffer(x1Fp32Buf, roundUpNumLastDimFloat);
            pipe.InitBuffer(x2Fp32Buf, roundUpNumLastDimFloat);
            pipe.InitBuffer(gammaFp32Buf, roundUpNumLastDimFloat);
            pipe.InitBuffer(dXFp32Buf, roundUpNumLastDimFloat);
            if constexpr (HAS_ADDITIONAL_INPUT) {
                pipe.InitBuffer(dSumFp32Buf, roundUpNumLastDimFloat);
            }
        }
    }

    __aicore__ inline void CutNProcess()
    {
        CopyInGamma(numLastDim, dyPadRight);
        LocalTensor<T> inputGamma = GammaQue.DeQue<T>();
        float reduceAxisSize = (float)1.0 / numLastDim;

        for (int32_t NOuterUbIndex = 0; NOuterUbIndex < nMiddleCount; ++NOuterUbIndex) {
            uint32_t nInOnceUb = (NOuterUbIndex != nMiddleCount - 1) ? nAvailInUb : nInUbTotalTail;
            uint32_t offsetUbXY = NOuterUbIndex * nAvailInUb * numLastDim;
            uint32_t offsetUbMeanVar = NOuterUbIndex * nAvailInUb;
            uint32_t DRstdInUb = 1;

            CopyIn(offsetUbXY, offsetUbMeanVar, numLastDim, DRstdInUb, nInOnceUb, dyPadRight, rstdPadRight);
            PrecisionCompute(nInOnceUb, inputGamma, reduceAxisSize);
            CopyOut(offsetUbXY, numLastDim, nInOnceUb);
        }
        GammaQue.FreeTensor(inputGamma);
    }

private:
    __aicore__ inline void CopyInGamma(const int32_t d_y_in_ub, const int32_t dyPadRight)
    {
        LocalTensor<T> gammaLocal = GammaQue.AllocTensor<T>();
#if __CCE_AICORE__ == 220
        DataCopyParams gamma_data_copy_params = {1, (uint16_t)(d_y_in_ub * sizeof(T)), 0, 0};
        DataCopyPadParams dy_pad_params{true, 0, (uint8_t)dyPadRight, 0};
        DataCopyPad(gammaLocal, gammaGm[0], gamma_data_copy_params, dy_pad_params);
#else
        DataCopy(gammaLocal, gammaGm[0], ROUND_UP(d_y_in_ub, blockNumberTdtype));
#endif
        GammaQue.EnQue(gammaLocal);
    }

    __aicore__ inline void CopyIn(const int32_t offsetUbXY, const int32_t offsetUbMeanVar, const int32_t d_y_in_ub,
        const int32_t DRstdInUb, const int32_t nInOnceUb, const int32_t dyPadRight, const int32_t rstdPadRight)
    {
        // AllocTensor
        LocalTensor<T> dyLocal = dyQue.AllocTensor<T>();
        LocalTensor<T> x1Local = x1Que.AllocTensor<T>();
        LocalTensor<T> x2Local = x2Que.AllocTensor<T>();
        LocalTensor<float> rstdLocal = rstdQue.AllocTensor<float>();
        LocalTensor<float> meanLocal = meanQue.AllocTensor<float>();
        LocalTensor<T> dSumLocal;
        if constexpr (HAS_ADDITIONAL_INPUT) {
            dSumLocal = dSumQue.AllocTensor<T>();
        }
#if __CCE_AICORE__ == 220
        DataCopyParams dy_data_copy_params{(uint16_t)nInOnceUb, (uint16_t)(d_y_in_ub * sizeof(T)), 0, 0};
        DataCopyPadParams dy_pad_params{true, 0, (uint8_t)dyPadRight, 0};
        DataCopyParams rstd_data_copy_params{(uint16_t)nInOnceUb, (uint16_t)(DRstdInUb * sizeof(float)), 0, 0};
        DataCopyPadParams rstd_pad_params{true, 0, (uint8_t)rstdPadRight, 0};

        DataCopyPad(dyLocal, dyGm[offsetUbXY], dy_data_copy_params, dy_pad_params);
        DataCopyPad(x1Local, x1Gm[offsetUbXY], dy_data_copy_params, dy_pad_params);
        DataCopyPad(x2Local, x2Gm[offsetUbXY], dy_data_copy_params, dy_pad_params);

        DataCopyPad(rstdLocal, rstdGm[offsetUbMeanVar], rstd_data_copy_params, rstd_pad_params);
        DataCopyPad(meanLocal, meanGm[offsetUbMeanVar], rstd_data_copy_params, rstd_pad_params);

        if constexpr (HAS_ADDITIONAL_INPUT) {
            DataCopyPad(dSumLocal, dSumGm[offsetUbXY], dy_data_copy_params, dy_pad_params);
        }
#else
        for (uint32_t idx = 0; idx < nInOnceUb; idx++) {
            DataCopy(dyLocal[idx * ROUND_UP(d_y_in_ub, blockNumberTdtype)],
                dyGm[offsetUbXY + idx * d_y_in_ub],
                ROUND_UP(d_y_in_ub, blockNumberTdtype));
            DataCopy(x1Local[idx * ROUND_UP(d_y_in_ub, blockNumberTdtype)],
                x1Gm[offsetUbXY + idx * d_y_in_ub],
                ROUND_UP(d_y_in_ub, blockNumberTdtype));
            DataCopy(x2Local[idx * ROUND_UP(d_y_in_ub, blockNumberTdtype)],
                x2Gm[offsetUbXY + idx * d_y_in_ub],
                ROUND_UP(d_y_in_ub, blockNumberTdtype));

            DataCopy(rstdLocal[idx * ROUND_UP(DRstdInUb, blockNumber)],
                rstdGm[offsetUbMeanVar + idx * DRstdInUb],
                ROUND_UP(DRstdInUb, blockNumber));
            DataCopy(meanLocal[idx * ROUND_UP(DRstdInUb, blockNumber)],
                meanGm[offsetUbMeanVar + idx * DRstdInUb],
                ROUND_UP(DRstdInUb, blockNumber));

            if (HAS_ADDITIONAL_INPUT) {
                DataCopy(dSumLocal[idx * ROUND_UP(d_y_in_ub, blockNumberTdtype)],
                    dSumGm[offsetUbXY + idx * d_y_in_ub],
                    ROUND_UP(d_y_in_ub, blockNumberTdtype));
            }
        }
#endif

        PipeBarrier<PIPE_ALL>();
        dyQue.EnQue(dyLocal);
        x1Que.EnQue(x1Local);
        x2Que.EnQue(x2Local);
        rstdQue.EnQue(rstdLocal);
        meanQue.EnQue(meanLocal);
        if constexpr (HAS_ADDITIONAL_INPUT) {
            dSumQue.EnQue(dSumLocal);
        }
    }

    __aicore__ inline void PrecisionCompute(
        const uint32_t nInOnceUb, const LocalTensor<T> inputGamma, const float reduceAxisSize)
    {
        LocalTensor<T> inputDy = dyQue.DeQue<T>();
        LocalTensor<T> inputX1 = x1Que.DeQue<T>();
        LocalTensor<T> inputX2 = x2Que.DeQue<T>();
        LocalTensor<float> inputRstd = rstdQue.DeQue<float>();
        LocalTensor<float> inputMean = meanQue.DeQue<float>();
        LocalTensor<T> dSumLocal;
        if constexpr (HAS_ADDITIONAL_INPUT) {
            dSumLocal = dSumQue.DeQue<T>();
        }

        LocalTensor<T> outputDx = dXQue.AllocTensor<T>();
        LocalTensor<float> outputDgamma = dGammaQue.AllocTensor<float>();
        LocalTensor<float> outputDbeta = dBetaQue.AllocTensor<float>();
#if __CCE_AICORE__ == 220
        Duplicate<float>(outputDgamma, 0.0, numLastDim);
        Duplicate<float>(outputDbeta, 0.0, numLastDim);
#else
        Duplicate<float>(outputDgamma, 0.0, ROUND_UP(numLastDim, blockNumber));
        Duplicate<float>(outputDbeta, 0.0, ROUND_UP(numLastDim, blockNumber));
#endif

        for (int32_t nInnerIndex = 0; nInnerIndex < nInOnceUb; ++nInnerIndex) {
            uint32_t offsetDXY = nInnerIndex * roundUpNumLastDim;
            uint32_t offsetDMeanVar = nInnerIndex * roundUp1Dtype;
#if __CCE_AICORE__ == 220
            if constexpr (IsSame<T, half>::value || IsSame<T, bfloat16_t>::value) {
#else
            if constexpr (IsSame<T, half>::value) {
#endif
                LocalTensor<float> dyFp32Local = dyFp32Buf.Get<float>();
                LocalTensor<float> x1Fp32Local = x1Fp32Buf.Get<float>();
                LocalTensor<float> x2Fp32Local = x2Fp32Buf.Get<float>();
                LocalTensor<float> gammaFp32Local = gammaFp32Buf.Get<float>();
                LocalTensor<float> dXLocal = dXFp32Buf.Get<float>();
                LocalTensor<float> dSumFp32Local;
                if constexpr (HAS_ADDITIONAL_INPUT) {
                    dSumFp32Local = dSumFp32Buf.Get<float>();
                    Cast(dSumFp32Local, dSumLocal[offsetDXY], RoundMode::CAST_NONE, numLastDim);
                }
                Cast(dyFp32Local, inputDy[offsetDXY], RoundMode::CAST_NONE, numLastDim);
                Cast(x1Fp32Local, inputX1[offsetDXY], RoundMode::CAST_NONE, numLastDim);
                Cast(x2Fp32Local, inputX2[offsetDXY], RoundMode::CAST_NONE, numLastDim);
                Cast(gammaFp32Local, inputGamma, RoundMode::CAST_NONE, numLastDim);
                PipeBarrier<PIPE_V>();

                MainCompute(dyFp32Local,
                    x1Fp32Local,
                    x2Fp32Local,
                    inputRstd[offsetDMeanVar],
                    inputMean[offsetDMeanVar],
                    gammaFp32Local,
                    dSumFp32Local,
                    dXLocal,
                    outputDgamma,
                    outputDbeta,
                    numLastDim,
                    numLastDim,
                    reduceAxisSize);

                if constexpr (IsSame<T, half>::value) {
                    Cast(outputDx[offsetDXY], dXLocal, RoundMode::CAST_NONE, numLastDim);
                } else {
                    Cast(outputDx[offsetDXY], dXLocal, RoundMode::CAST_RINT, numLastDim);
                }
            } else {
                MainCompute(inputDy[offsetDXY],
                    inputX1[offsetDXY],
                    inputX2[offsetDXY],
                    inputRstd[offsetDMeanVar],
                    inputMean[offsetDMeanVar],
                    inputGamma,
                    dSumLocal[offsetDXY],
                    outputDx[offsetDXY],
                    outputDgamma,
                    outputDbeta,
                    numLastDim,
                    numLastDim,
                    reduceAxisSize);
            }
        }
        dyQue.FreeTensor(inputDy);
        x1Que.FreeTensor(inputX1);
        x2Que.FreeTensor(inputX2);
        rstdQue.FreeTensor(inputRstd);
        meanQue.FreeTensor(inputMean);
        if constexpr (HAS_ADDITIONAL_INPUT) {
            dSumQue.FreeTensor(dSumLocal);
        }
        dXQue.EnQue(outputDx);
        dGammaQue.EnQue(outputDgamma);
        dBetaQue.EnQue(outputDbeta);
    }

    __aicore__ inline void MainCompute(const LocalTensor<float> &inputDy, const LocalTensor<float> &inputX1,
        const LocalTensor<float> &inputX2, const LocalTensor<float> &inputRstd, const LocalTensor<float> &inputMean,
        const LocalTensor<float> &inputGamma, const LocalTensor<float> &input_dx, const LocalTensor<float> &outputDx,
        const LocalTensor<float> &outputDgamma, const LocalTensor<float> &outputDbeta, const uint32_t elem_cout_d_x_y,
        const uint32_t numLastDim, const float reduceAxisSize)
    {
        Add(inputX1, inputX1, inputX2, elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();
        // 1. x1Tensor = inputDy * inputGamma
        Mul(inputX2, inputDy, inputGamma, elem_cout_d_x_y);

        // 2. x2Tensor = inputX - inputMean
        event_t event_mte2_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(event_mte2_s);
        WaitFlag<HardEvent::MTE2_S>(event_mte2_s);
        float inputMeanNum = inputMean.GetValue(0);
        float inputRstdNum = inputRstd.GetValue(0);
        float tmpLocalNum = inputRstdNum * inputRstdNum * inputRstdNum;
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);
        Adds(outputDx, inputX1, inputMeanNum * (-1.0f), elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();

        // 3. pd_var = np.sum((-0.5) * x1Tensor * x2Tensor * np.power(inputRstd, 3))
        // 3.1. duplicate
        Muls(inputX1, outputDx, tmpLocalNum, elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();

        Mul(inputX1, inputX2, inputX1, elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();

        // 3.2. pd_var = np.sum(res1)
        auto aveLocalTemp = ReduceSumCustom(inputX1, numLastDim);
        inputMeanNum = aveLocalTemp * -reduceAxisSize;
        event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);

        // 4. pd_mean = np.sum((-1.0) * x1Tensor * inputRstd)
        // output: gamma = x2Tensor * rstd * inputDy
        Muls(inputX1, outputDx, inputRstdNum, elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();

        Mul(inputX1, inputX1, inputDy, elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();
        Add(outputDgamma, outputDgamma, inputX1, elem_cout_d_x_y);
        Add(outputDbeta, outputDbeta, inputDy, elem_cout_d_x_y);

        // 4.1. res1 = (-1.0) * x1Tensor * rstd
        WaitFlag<HardEvent::S_V>(event_s_v);
        Muls(inputX1, outputDx, inputMeanNum, elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();
        Muls(outputDx, inputX2, inputRstdNum, elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();
        Muls(inputDy, outputDx, -1.0f, elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();

        // 4.2. pd_mean = np.sum(res1)
        Add(inputX2, inputX1, outputDx, elem_cout_d_x_y);
        aveLocalTemp = ReduceSumCustom(inputDy, elem_cout_d_x_y);
        inputMeanNum = aveLocalTemp * reduceAxisSize;
        event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);

        // 5. d_x = x1Tensor * inputRstd +
        //           pd_var * (2.0 / reduceAxisSize) * x2Tensor +
        //           pd_mean * (1.0 / reduceAxisSize)
        // 5.1. res0 = x1Tensor * np.power((inputVariace + EPSLON), (-0.5)), already store in resForGamma
        // 5.2. res1 = pd_var*(2.0 / reduceAxisSize)*(x2Tensor)
        // 5.3. res2 = pd_mean*(1.0 / reduceAxisSize)
        // 5.4. d_x = res0 + res1 + res2
        WaitFlag<HardEvent::S_V>(event_s_v);
        Adds(outputDx, inputX2, inputMeanNum, elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();
        if constexpr (HAS_ADDITIONAL_INPUT) {
            Add(outputDx, outputDx, input_dx, elem_cout_d_x_y);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CopyOut(const int32_t offsetUbXY, const int32_t d_y_in_ub, const int32_t nInOnceUb)
    {
        LocalTensor<T> outputDx = dXQue.DeQue<T>();
        LocalTensor<float> outputDgamma = dGammaQue.DeQue<float>();
        LocalTensor<float> outputDbeta = dBetaQue.DeQue<float>();
        PipeBarrier<PIPE_ALL>();
        DataCopyCustom<T>(dXGm, outputDx, d_y_in_ub, offsetUbXY, false, (uint16_t)nInOnceUb);
        PipeBarrier<PIPE_ALL>();

        SetAtomicAdd<float>();
        DataCopyAutomicAdd(dBetaGm, outputDbeta, numLastDim, 0, (uint16_t)1);
        PipeBarrier<PIPE_ALL>();
        SetAtomicNone();

        SetAtomicAdd<float>();
        DataCopyAutomicAdd(dGammaGm, outputDgamma, numLastDim, 0, (uint16_t)1);
        PipeBarrier<PIPE_ALL>();
        SetAtomicNone();

        dXQue.FreeTensor(outputDx);
        dGammaQue.FreeTensor(outputDgamma);
        dBetaQue.FreeTensor(outputDbeta);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> dyQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> x1Que;
    TQue<QuePosition::VECIN, BUFFER_NUM> x2Que;
    TQue<QuePosition::VECIN, BUFFER_NUM> rstdQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> meanQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> GammaQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> dSumQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dXQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dGammaQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dBetaQue;

    TBuf<TPosition::VECCALC> dyFp32Buf;
    TBuf<TPosition::VECCALC> x1Fp32Buf;
    TBuf<TPosition::VECCALC> x2Fp32Buf;
    TBuf<TPosition::VECCALC> gammaFp32Buf;
    TBuf<TPosition::VECCALC> dXFp32Buf;
    TBuf<TPosition::VECCALC> dSumFp32Buf;
    GlobalTensor<float> rstdGm;
    GlobalTensor<float> meanGm;
    GlobalTensor<float> dGammaGm;
    GlobalTensor<float> dBetaGm;
    GlobalTensor<T> dyGm;
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<T> dXGm;
    GlobalTensor<T> dSumGm;

    uint32_t numCore;
    uint32_t numFirstDim;
    int32_t numLastDim;
    uint32_t nInOnecoreLength;
    uint32_t nInOnecoreLengthTail;
    uint32_t nAvailInUb;
    uint32_t dInnerLength;
    uint32_t dInnerLengthTail;
    uint32_t dOuterLength;
    uint32_t offsetGmXY;
    uint32_t offsetGmMeanVar;
    uint32_t offsetGmGamma;
    uint32_t dyPadRight;
    uint32_t rstdPadRight;
    uint32_t roundUpNumLastDim;
    uint32_t roundUpNumLastDimDtype;
    uint32_t roundUp1Dtype;
    uint32_t roundUpNumLastDimFloat;

    uint32_t nInOneCore;
    uint32_t nMiddleCount;
    uint32_t nInUbTotalTail;
    uint32_t nDRoundUpDtype;
    uint32_t n1RoundUpFloat;
    uint32_t nDInOnecoreLength;
    uint32_t gmOneCoreElemXY;
    uint32_t blockNumber;
    uint32_t blockNumberTdtype;
    uint32_t eachCoreHandleNum;
};

#endif  // OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_LAYER_NORM_GRAD_ADD_LAYER_NORM_GRAD_CUT_N