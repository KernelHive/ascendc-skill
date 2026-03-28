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
 * \file add_layer_norm_grad_cut_d.h
 * \brief
 */
#ifndef OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_LAYER_NORM_GRAD_ADD_LAYER_NORM_GRAD_CUT_D
#define OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_LAYER_NORM_GRAD_ADD_LAYER_NORM_GRAD_CUT_D
#include "add_layer_norm_grad_utils.h"

using namespace AscendC;

template <typename T, int TILING_KEY>
class KernelAddLayerNormGradLarge {
#define HAS_ADDITIONAL_INPUT ((TILING_KEY % 10) == 1)
public:
    __aicore__ inline KernelAddLayerNormGradLarge()
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
        nAvailInUb = tiling.nAvailInUb;
        dInnerLength = tiling.dInnerLength;
        dInnerLengthTail = tiling.dInnerLengthTail;
        dOuterLength = tiling.dOuterLength;

        nInOneCore = (GetBlockIdx() != numCore - 1) ? nInOnecoreLength : nInOnecoreLengthTail;
        gmOneCoreElemXY = nInOneCore * numLastDim;

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

        offsetGmXY = GetBlockIdx() * nInOnecoreLength * numLastDim;
        offsetGmMeanVar = GetBlockIdx() * nInOnecoreLength;
        offsetGmGamma = 0;
    }

    __aicore__ inline void InitInputGmBuffer(
        GM_ADDR dy, GM_ADDR x_1, GM_ADDR x_2, GM_ADDR rstd, GM_ADDR mean, GM_ADDR gamma, GM_ADDR dsum)
    {
        dyGm.SetGlobalBuffer((__gm__ T *)dy + GetBlockIdx() * nInOnecoreLength * numLastDim, gmOneCoreElemXY);
        x1Gm.SetGlobalBuffer((__gm__ T *)x_1 + GetBlockIdx() * nInOnecoreLength * numLastDim, gmOneCoreElemXY);
        x2Gm.SetGlobalBuffer((__gm__ T *)x_2 + GetBlockIdx() * nInOnecoreLength * numLastDim, gmOneCoreElemXY);
        rstdGm.SetGlobalBuffer((__gm__ float *)rstd + GetBlockIdx() * nInOnecoreLength, nInOneCore);
        meanGm.SetGlobalBuffer((__gm__ float *)mean + GetBlockIdx() * nInOnecoreLength, nInOneCore);
        gammaGm.SetGlobalBuffer((__gm__ T *)gamma, numLastDim);
        if constexpr (HAS_ADDITIONAL_INPUT) {
            dSumGm.SetGlobalBuffer((__gm__ T *)dsum + GetBlockIdx() * nInOnecoreLength * numLastDim, gmOneCoreElemXY);
        }
    }

    __aicore__ inline void InitOuputGmBuffer(GM_ADDR d_x, GM_ADDR d_gamma, GM_ADDR d_beta)
    {
        dXGm.SetGlobalBuffer((__gm__ T *)d_x + GetBlockIdx() * nInOnecoreLength * numLastDim, gmOneCoreElemXY);
        dGammaGm.SetGlobalBuffer((__gm__ float *)d_gamma, numLastDim);
        dBetaGm.SetGlobalBuffer((__gm__ float *)d_beta, numLastDim);
        LocalTensor<float> temp_local_tensor = dGammaQue.AllocTensor<float>();
        InitGmData(dGammaGm, dBetaGm, numLastDim, temp_local_tensor, ROUND_UP(dInnerLength, blockNumber));
        dGammaQue.FreeTensor(temp_local_tensor);
    }

    __aicore__ inline void InitInputQueue()
    {
        pipe.InitBuffer(dyQue, BUFFER_NUM, ROUND_UP(dInnerLength, blockNumberTdtype) * sizeof(T));
        pipe.InitBuffer(x1Que, BUFFER_NUM, ROUND_UP(dInnerLength, blockNumberTdtype) * sizeof(T));
        pipe.InitBuffer(x2Que, BUFFER_NUM, ROUND_UP(dInnerLength, blockNumberTdtype) * sizeof(T));
        pipe.InitBuffer(rstdQue, BUFFER_NUM, ROUND_UP(1, blockNumber) * sizeof(float));
        pipe.InitBuffer(meanQue, BUFFER_NUM, ROUND_UP(1, blockNumber) * sizeof(float));
        pipe.InitBuffer(gammaQue, BUFFER_NUM, ROUND_UP(dInnerLength, blockNumberTdtype) * sizeof(T));
        if constexpr (HAS_ADDITIONAL_INPUT) {
            pipe.InitBuffer(dSumQue, BUFFER_NUM, ROUND_UP(dInnerLength, blockNumberTdtype) * sizeof(T));
        }
    }

    __aicore__ inline void InitOutputQueue()
    {
        pipe.InitBuffer(dXQue, BUFFER_NUM, ROUND_UP(dInnerLength, blockNumberTdtype) * sizeof(T));
        pipe.InitBuffer(dGammaQue, BUFFER_NUM, ROUND_UP(dInnerLength, blockNumber) * sizeof(float));
        pipe.InitBuffer(dBetaQue, BUFFER_NUM, ROUND_UP(dInnerLength, blockNumber) * sizeof(float));
    }

    __aicore__ inline void InitTmpBuffer(GM_ADDR workspace)
    {
        pipe.InitBuffer(tmpMeanPdBuf, ROUND_UP(1, blockNumber) * sizeof(float));
        pipe.InitBuffer(tmpVarPdBuf, ROUND_UP(1, blockNumber) * sizeof(float));
        if constexpr (IsSame<T, float>::value) {
        } else {
            pipe.InitBuffer(dyFp32Buf, ROUND_UP(dInnerLength, blockNumber) * sizeof(float));
            pipe.InitBuffer(x1Fp32Buf, ROUND_UP(dInnerLength, blockNumber) * sizeof(float));
            pipe.InitBuffer(x2Fp32Buf, ROUND_UP(dInnerLength, blockNumber) * sizeof(float));
            pipe.InitBuffer(dgammaFp32Buf, ROUND_UP(dInnerLength, blockNumber) * sizeof(float));
            pipe.InitBuffer(dXFp32Buf, ROUND_UP(dInnerLength, blockNumber) * sizeof(float));
            if constexpr (HAS_ADDITIONAL_INPUT) {
                pipe.InitBuffer(dSumFp32Buf, ROUND_UP(dInnerLength, blockNumber) * sizeof(float));
            }
        }
    }

    __aicore__ inline void CutDProcess()
    {
        LocalTensor<float> tmpMeanPdLocal = tmpMeanPdBuf.Get<float>();
        LocalTensor<float> tmpVarPdLocal = tmpVarPdBuf.Get<float>();
        LocalTensor<float> dyFp32Local;
        LocalTensor<float> x1Fp32Local;
        LocalTensor<float> x2Fp32Local;
        LocalTensor<float> gammaFp32Local;
        LocalTensor<float> dXLocal;
        LocalTensor<float> dSumFp32Local;
#if __CCE_AICORE__ == 220
        if constexpr (IsSame<T, half>::value || IsSame<T, bfloat16_t>::value) {
#else
        if constexpr (IsSame<T, half>::value) {
#endif
            dyFp32Local = dyFp32Buf.Get<float>();
            x1Fp32Local = x1Fp32Buf.Get<float>();
            x2Fp32Local = x2Fp32Buf.Get<float>();
            gammaFp32Local = dgammaFp32Buf.Get<float>();
            dXLocal = dXFp32Buf.Get<float>();
            if constexpr (HAS_ADDITIONAL_INPUT) {
                dSumFp32Local = dSumFp32Buf.Get<float>();
            }
        }
        for (int32_t nInnerIndex = 0; nInnerIndex < nInOneCore; ++nInnerIndex) {
            Duplicate(tmpMeanPdLocal, 0.0f, blockNumber);
            Duplicate(tmpVarPdLocal, 0.0f, blockNumber);
            PipeBarrier<PIPE_V>();
            for (int32_t DOuterUbIndex = 0; DOuterUbIndex < dOuterLength; ++DOuterUbIndex) {
                uint32_t DInOnceUb = (DOuterUbIndex != dOuterLength - 1) ? dInnerLength : dInnerLengthTail;
                uint32_t offsetUbXY = DOuterUbIndex * dInnerLength + nInnerIndex * numLastDim;
                uint32_t offsetUbMeanVar = nInnerIndex;
                uint32_t offsetUbGamma = DOuterUbIndex * dInnerLength;
                uint32_t elemCoutXYUb = DInOnceUb;
                uint32_t DRstdInUb = 1;

                CopyIn(offsetUbXY, offsetUbMeanVar, elemCoutXYUb, DRstdInUb, nAvailInUb, offsetUbGamma);
                event_t event_mte2_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(event_mte2_v);
                WaitFlag<HardEvent::MTE2_V>(event_mte2_v);
                LocalTensor<T> inputDy = dyQue.DeQue<T>();
                LocalTensor<T> inputX1 = x1Que.DeQue<T>();
                LocalTensor<T> inputX2 = x2Que.DeQue<T>();
                LocalTensor<float> inputRstd = rstdQue.DeQue<float>();
                LocalTensor<float> inputMean = meanQue.DeQue<float>();
                LocalTensor<T> inputGamma = gammaQue.DeQue<T>();

                LocalTensor<T> outputDx = dXQue.AllocTensor<T>();
                LocalTensor<float> outputDgamma = dGammaQue.AllocTensor<float>();
                LocalTensor<float> outputDbeta = dBetaQue.AllocTensor<float>();
#if __CCE_AICORE__ == 220
                if constexpr (IsSame<T, half>::value || IsSame<T, bfloat16_t>::value) {
#else
                if constexpr (IsSame<T, half>::value) {
#endif
                    Cast(dyFp32Local, inputDy, RoundMode::CAST_NONE, elemCoutXYUb);
                    Cast(x1Fp32Local, inputX1, RoundMode::CAST_NONE, elemCoutXYUb);
                    Cast(x2Fp32Local, inputX2, RoundMode::CAST_NONE, elemCoutXYUb);
                    Cast(gammaFp32Local, inputGamma, RoundMode::CAST_NONE, elemCoutXYUb);
                    MainComputeFirstPart(dyFp32Local,
                        x1Fp32Local,
                        x2Fp32Local,
                        inputRstd,
                        inputMean,
                        gammaFp32Local,
                        outputDgamma,
                        outputDbeta,
                        dXLocal,
                        tmpVarPdLocal,
                        tmpMeanPdLocal,
                        elemCoutXYUb);
                } else {
                    MainComputeFirstPart(inputDy,
                        inputX1,
                        inputX2,
                        inputRstd,
                        inputMean,
                        inputGamma,
                        outputDgamma,
                        outputDbeta,
                        outputDx,
                        tmpVarPdLocal,
                        tmpMeanPdLocal,
                        elemCoutXYUb);
                }
                dyQue.FreeTensor(inputDy);
                x1Que.FreeTensor(inputX1);
                x2Que.FreeTensor(inputX2);
                rstdQue.FreeTensor(inputRstd);
                meanQue.FreeTensor(inputMean);
                gammaQue.FreeTensor(inputGamma);
                dXQue.FreeTensor(outputDx);
                dGammaQue.EnQue(outputDgamma);
                dBetaQue.EnQue(outputDbeta);
                event_t event_v_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(event_v_mte3);
                WaitFlag<HardEvent::V_MTE3>(event_v_mte3);
                CopyOutBetaGamma(offsetUbGamma, elemCoutXYUb);
                PipeBarrier<PIPE_ALL>();
            }

            for (int32_t DOuterUbIndex = 0; DOuterUbIndex < dOuterLength; ++DOuterUbIndex) {
                uint32_t DInOnceUb = (DOuterUbIndex != dOuterLength - 1) ? dInnerLength : dInnerLengthTail;
                uint32_t offsetUbXY = DOuterUbIndex * dInnerLength + nInnerIndex * numLastDim;
                uint32_t offsetUbMeanVar = nInnerIndex;
                uint32_t offsetUbGamma = DOuterUbIndex * dInnerLength;
                uint32_t elemCoutXYUb = DInOnceUb;
                uint32_t DRstdInUb = 1;

                CopyIn(offsetUbXY, offsetUbMeanVar, elemCoutXYUb, DRstdInUb, nAvailInUb, offsetUbGamma, true);
                event_t event_mte2_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(event_mte2_v);
                WaitFlag<HardEvent::MTE2_V>(event_mte2_v);
                LocalTensor<T> inputDy = dyQue.DeQue<T>();
                LocalTensor<T> inputX1 = x1Que.DeQue<T>();
                LocalTensor<T> inputX2 = x2Que.DeQue<T>();
                LocalTensor<float> inputRstd = rstdQue.DeQue<float>();
                LocalTensor<float> inputMean = meanQue.DeQue<float>();
                LocalTensor<T> inputGamma = gammaQue.DeQue<T>();
                LocalTensor<T> inputDx;
                LocalTensor<T> outputDx = dXQue.AllocTensor<T>();
                if constexpr (HAS_ADDITIONAL_INPUT) {
                    inputDx = dSumQue.DeQue<T>();
                }
#if __CCE_AICORE__ == 220
                if constexpr (IsSame<T, half>::value || IsSame<T, bfloat16_t>::value) {
#else
                if constexpr (IsSame<T, half>::value) {
#endif
                    Cast(dyFp32Local, inputDy, RoundMode::CAST_NONE, elemCoutXYUb);
                    Cast(x1Fp32Local, inputX1, RoundMode::CAST_NONE, elemCoutXYUb);
                    Cast(x2Fp32Local, inputX2, RoundMode::CAST_NONE, elemCoutXYUb);
                    Cast(gammaFp32Local, inputGamma, RoundMode::CAST_NONE, elemCoutXYUb);
                    if constexpr (HAS_ADDITIONAL_INPUT) {
                        Cast(dSumFp32Local, inputDx, RoundMode::CAST_NONE, elemCoutXYUb);
                    }
                    PipeBarrier<PIPE_V>();
                    MainComputeSecondPart(dyFp32Local,
                        x1Fp32Local,
                        x2Fp32Local,
                        inputRstd,
                        inputMean,
                        gammaFp32Local,
                        dSumFp32Local,
                        dXLocal,
                        tmpVarPdLocal,
                        tmpMeanPdLocal,
                        elemCoutXYUb);
                    if constexpr (IsSame<T, half>::value) {
                        Cast(outputDx, dXLocal, RoundMode::CAST_NONE, elemCoutXYUb);
                        PipeBarrier<PIPE_V>();
                    } else {
                        Cast(outputDx, dXLocal, RoundMode::CAST_RINT, elemCoutXYUb);
                        PipeBarrier<PIPE_V>();
                    }
                } else {
                    MainComputeSecondPart(inputDy,
                        inputX1,
                        inputX2,
                        inputRstd,
                        inputMean,
                        inputGamma,
                        inputDx,
                        outputDx,
                        tmpVarPdLocal,
                        tmpMeanPdLocal,
                        elemCoutXYUb);
                }

                dyQue.FreeTensor(inputDy);
                x1Que.FreeTensor(inputX1);
                x2Que.FreeTensor(inputX2);
                rstdQue.FreeTensor(inputRstd);
                meanQue.FreeTensor(inputMean);
                gammaQue.FreeTensor(inputGamma);
                if constexpr (HAS_ADDITIONAL_INPUT) {
                    dSumQue.FreeTensor(inputDx);
                }
                dXQue.EnQue(outputDx);
                event_t event_v_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(event_v_mte3);
                WaitFlag<HardEvent::V_MTE3>(event_v_mte3);
                CopyOutX(offsetUbXY, elemCoutXYUb, nAvailInUb);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(const int32_t offsetUbXY, const int32_t offsetUbMeanVar, const int32_t d_y_in_ub,
        const int32_t DRstdInUb, const int32_t n_in_once_ub, const int32_t offsetUbGamma, const bool has_dsum = false)
    {
        // AllocTensor
        LocalTensor<T> dyLocal = dyQue.AllocTensor<T>();
        LocalTensor<T> x1Local = x1Que.AllocTensor<T>();
        LocalTensor<T> x2Local = x2Que.AllocTensor<T>();
        LocalTensor<float> rstdLocal = rstdQue.AllocTensor<float>();
        LocalTensor<float> meanLocal = meanQue.AllocTensor<float>();
        LocalTensor<T> gammaLocal = gammaQue.AllocTensor<T>();
        LocalTensor<T> dSumLocal;
#if __CCE_AICORE__ == 220
        DataCopyParams dy_data_copy_params{(uint16_t)n_in_once_ub, (uint16_t)(d_y_in_ub * sizeof(T)), 0, 0};
        uint8_t dyPadRight = ROUND_UP(d_y_in_ub, blockNumberTdtype) - d_y_in_ub;
        DataCopyPadParams dy_pad_params{true, 0, dyPadRight, 0};
        DataCopyParams rstd_data_copy_params{(uint16_t)n_in_once_ub, (uint16_t)(DRstdInUb * sizeof(float)), 0, 0};
        uint8_t rstdPadRight = ROUND_UP(DRstdInUb, blockNumber) - DRstdInUb;
        DataCopyPadParams rstd_pad_params{true, 0, rstdPadRight, 0};

        DataCopyPad(dyLocal, dyGm[offsetUbXY], dy_data_copy_params, dy_pad_params);
        DataCopyPad(x1Local, x1Gm[offsetUbXY], dy_data_copy_params, dy_pad_params);
        DataCopyPad(x2Local, x2Gm[offsetUbXY], dy_data_copy_params, dy_pad_params);

        DataCopyPad(rstdLocal, rstdGm[offsetUbMeanVar], rstd_data_copy_params, rstd_pad_params);
        DataCopyPad(meanLocal, meanGm[offsetUbMeanVar], rstd_data_copy_params, rstd_pad_params);
        DataCopyParams gamma_data_copy_params = {1, (uint16_t)(d_y_in_ub * sizeof(T)), 0, 0};
        DataCopyPad(gammaLocal, gammaGm[offsetUbGamma], gamma_data_copy_params, dy_pad_params);
        if (HAS_ADDITIONAL_INPUT && has_dsum) {
            dSumLocal = dSumQue.AllocTensor<T>();
            DataCopyPad(dSumLocal, dSumGm[offsetUbXY], dy_data_copy_params, dy_pad_params);
        }
#else
        for (uint32_t idx = 0; idx < n_in_once_ub; idx++) {
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

            if (HAS_ADDITIONAL_INPUT && has_dsum) {
                dSumLocal = dSumQue.AllocTensor<T>();
                DataCopy(dSumLocal[idx * ROUND_UP(d_y_in_ub, blockNumberTdtype)],
                    dSumGm[offsetUbXY + idx * d_y_in_ub],
                    ROUND_UP(d_y_in_ub, blockNumberTdtype));
            }
        }
        DataCopy(gammaLocal, gammaGm[offsetUbGamma], ROUND_UP(d_y_in_ub, blockNumberTdtype));

#endif

        PipeBarrier<PIPE_ALL>();
        dyQue.EnQue(dyLocal);
        x1Que.EnQue(x1Local);
        x2Que.EnQue(x2Local);
        rstdQue.EnQue(rstdLocal);
        meanQue.EnQue(meanLocal);
        gammaQue.EnQue(gammaLocal);
        if (HAS_ADDITIONAL_INPUT && has_dsum) {
            dSumQue.EnQue(dSumLocal);
        }
    }

    __aicore__ inline void MainComputeFirstPart(const LocalTensor<float> &inputDy, const LocalTensor<float> &inputX1,
        const LocalTensor<float> &inputX2, const LocalTensor<float> &inputRstd, const LocalTensor<float> &inputMean,
        const LocalTensor<float> &inputGamma, const LocalTensor<float> &outputDgamma,
        const LocalTensor<float> &outputDbeta, const LocalTensor<float> &outputDx,
        const LocalTensor<float> &tmpVarPdLocal, const LocalTensor<float> &tmpMeanPdLocal,
        const uint32_t elem_cout_d_x_y)
    {
        Adds(outputDbeta, inputDy, 0.0f, elem_cout_d_x_y);
        // 1. x1Tensor = inputDy * inputGamma
        Add(inputX1, inputX1, inputX2, elem_cout_d_x_y);
        event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s);
        Mul(inputGamma, inputDy, inputGamma, elem_cout_d_x_y);
        // 2. x2Tensor = inputX - inputMean
        WaitFlag<HardEvent::V_S>(event_v_s);
        float inputMeanNum = inputMean.GetValue(0);
        float inputRstdNum = inputRstd.GetValue(0);
        Duplicate<float>(outputDx, inputMeanNum, elem_cout_d_x_y);
        Duplicate<float>(inputX2, inputRstdNum, elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();
        float tmpLocalNum = inputRstdNum * inputRstdNum * inputRstdNum;
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);
        Sub(outputDx, inputX1, outputDx, elem_cout_d_x_y);

        // 3. pd_var = np.sum((-0.5) * x1Tensor * x2Tensor * np.power(inputRstd, 3))
        // 3.1. duplicate
        Duplicate<float>(inputX1, tmpLocalNum, elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();

        // 3.2. res1 = (-0.5) * x1Tensor * (x2Tensor) * res
        Mul(inputX1, outputDx, inputX1, elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();
        Mul(inputX1, inputGamma, inputX1, elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();

        // 3.3. pd_var = np.sum(res1)
        auto aveLocalTemp = ReduceSumCustom(inputX1, elem_cout_d_x_y);

        // 3.4. pd_mean = np.sum((-1.0) * x1Tensor * inputRstd)
        // gamma = x2Tensor * rstd * inputDy
        Mul(inputX1, outputDx, inputX2, elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();
        Mul(outputDgamma, inputX1, inputDy, elem_cout_d_x_y);

        // 5.1. res1 = (-1.0) * x1Tensor * rstd
        Mul(inputX1, inputGamma, inputX2, elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();
        Muls(inputX2, inputX1, -1.0f, elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();

        // 5.2. pd_mean = np.sum(res1)
        auto aveLocalTemp2 = ReduceSumCustom(inputX2, elem_cout_d_x_y);

        Adds(tmpVarPdLocal, tmpVarPdLocal, aveLocalTemp, blockNumber);
        Adds(tmpMeanPdLocal, tmpMeanPdLocal, aveLocalTemp2, blockNumber);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MainComputeSecondPart(const LocalTensor<float> &inputDy, const LocalTensor<float> &inputX1,
        const LocalTensor<float> &inputX2, const LocalTensor<float> &inputRstd, const LocalTensor<float> &inputMean,
        const LocalTensor<float> &inputGamma, const LocalTensor<float> &inputDx, const LocalTensor<float> &outputDx,
        const LocalTensor<float> &tmpVarPdLocal, const LocalTensor<float> &tmpMeanPdLocal,
        const uint32_t elem_cout_d_x_y)
    {
        // 1. x1Tensor = inputDy * inputGamma
        Add(inputX1, inputX1, inputX2, elem_cout_d_x_y);
        Mul(inputGamma, inputDy, inputGamma, elem_cout_d_x_y);
        // 2. x2Tensor = inputX - inputMean
        PipeBarrier<PIPE_V>();
        Adds(inputX1, inputX1, -1.0f * inputMean.GetValue(0), elem_cout_d_x_y);
        Muls(inputDy, inputGamma, inputRstd.GetValue(0), elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();

        // 5. d_x = x1Tensor * inputRstd +
        //           pd_var * (2.0 / reduceAxisSize) * x2Tensor +
        //           pd_mean * (1.0 / reduceAxisSize)
        float reduceAxisSize = 1.0f;
        if (numLastDim != 0) {
            reduceAxisSize = 1.0f / numLastDim;
        }
        float inputRstdNum = tmpVarPdLocal.GetValue(0) * -reduceAxisSize;
        Muls(inputX2, inputX1, inputRstdNum, elem_cout_d_x_y);
        // 5.3. res2 = pd_mean*(1.0 / reduceAxisSize)
        float inputMeanNum = tmpMeanPdLocal.GetValue(0) * reduceAxisSize;
        Duplicate<float>(inputGamma, inputMeanNum, elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);

        // 5.4. d_x = res0 + res1 + res2
        Add(inputX2, inputGamma, inputX2, elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();
        Add(outputDx, inputX2, inputDy, elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();
        if constexpr (HAS_ADDITIONAL_INPUT) {
            Add(outputDx, outputDx, inputDx, elem_cout_d_x_y);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CopyOutBetaGamma(const int32_t offsetUbGamma, const int32_t elem_cout_d_x_y)
    {
        LocalTensor<float> d_gammaLocal = dGammaQue.DeQue<float>();
        LocalTensor<float> d_beta_local = dBetaQue.DeQue<float>();

        SetAtomicAdd<float>();
        DataCopyAutomicAdd(dGammaGm, d_gammaLocal, elem_cout_d_x_y, offsetUbGamma, (uint16_t)1);
        PipeBarrier<PIPE_ALL>();
        SetAtomicNone();

        SetAtomicAdd<float>();
        DataCopyAutomicAdd(dBetaGm, d_beta_local, elem_cout_d_x_y, offsetUbGamma, (uint16_t)1);
        PipeBarrier<PIPE_ALL>();
        SetAtomicNone();

        dGammaQue.FreeTensor(d_gammaLocal);
        dBetaQue.FreeTensor(d_beta_local);
    }

    __aicore__ inline void CopyOutX(const int32_t offsetUbXY, const int32_t d_y_in_ub, const int32_t n_in_once_ub)
    {
        LocalTensor<T> dXLocal = dXQue.DeQue<T>();

        DataCopyCustom<T>(dXGm, dXLocal, d_y_in_ub, offsetUbXY, false, (uint16_t)n_in_once_ub);
        PipeBarrier<PIPE_ALL>();

        dXQue.FreeTensor(dXLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> dyQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> x1Que;
    TQue<QuePosition::VECIN, BUFFER_NUM> x2Que;
    TQue<QuePosition::VECIN, BUFFER_NUM> rstdQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> meanQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> gammaQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> dSumQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dXQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dGammaQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dBetaQue;

    TBuf<TPosition::VECCALC> tmpMeanPdBuf;
    TBuf<TPosition::VECCALC> tmpVarPdBuf;
    TBuf<TPosition::VECCALC> dyFp32Buf;
    TBuf<TPosition::VECCALC> x1Fp32Buf;
    TBuf<TPosition::VECCALC> x2Fp32Buf;
    TBuf<TPosition::VECCALC> dgammaFp32Buf;
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

    uint32_t nInOneCore;
    uint32_t gmOneCoreElemXY;
    uint32_t blockNumber;
    uint32_t blockNumberTdtype;
    uint32_t eachCoreHandleNum;
};

#endif  // OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_LAYER_NORM_GRAD_ADD_LAYER_NORM_GRAD_CUT_D