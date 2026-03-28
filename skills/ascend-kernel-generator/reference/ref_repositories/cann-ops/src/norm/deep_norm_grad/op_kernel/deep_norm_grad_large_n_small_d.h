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
 * \file deep_norm_grad_large_n_small_d.h
 * \brief
 */

#ifndef OPS_BUILT_IN_TBE_IMPL_ASCENDC_DEEP_NORM_GRAD_DEEP_NORM_GRAD_LARGE_N_SMALL_D_H_
#define OPS_BUILT_IN_TBE_IMPL_ASCENDC_DEEP_NORM_GRAD_DEEP_NORM_GRAD_LARGE_N_SMALL_D_H_
#include "deep_norm_grad_common.h"

template <typename T>
class KernelDeepNormGradLargeNSmallD : public KernelDeepNormGradBase<T> {
public:
    __aicore__ inline KernelDeepNormGradLargeNSmallD()
    {}

    __aicore__ inline void InitGM(GM_ADDR dataDy, GM_ADDR dataX, GM_ADDR dataGx, GM_ADDR dataRstd, GM_ADDR dataMean,
        GM_ADDR dataGamma, GM_ADDR outputDx, GM_ADDR outputDgx, GM_ADDR outputDgamma, GM_ADDR outputDbeta)
    {
        dyGm.SetGlobalBuffer((__gm__ T *)dataDy + GetBlockIdx() * nDealPerCore * dDimNum, nDeal * dDimNum);
        xGm.SetGlobalBuffer((__gm__ T *)dataX + GetBlockIdx() * nDealPerCore * dDimNum, nDeal * dDimNum);
        gxGm.SetGlobalBuffer((__gm__ T *)dataGx + GetBlockIdx() * nDealPerCore * dDimNum, nDeal * dDimNum);
        meanGm.SetGlobalBuffer((__gm__ float *)dataMean + GetBlockIdx() * nDealPerCore, nDeal);
        rstdGm.SetGlobalBuffer((__gm__ float *)dataRstd + GetBlockIdx() * nDealPerCore, nDeal);
        gammaGm.SetGlobalBuffer((__gm__ T *)dataGamma, dDimNum);

        outputDxGm.SetGlobalBuffer((__gm__ T *)outputDx + GetBlockIdx() * nDealPerCore * dDimNum, nDeal * dDimNum);
        outputDgxGm.SetGlobalBuffer((__gm__ T *)outputDgx + GetBlockIdx() * nDealPerCore * dDimNum, nDeal * dDimNum);
        outputDbetaGm.SetGlobalBuffer((__gm__ float *)outputDbeta, dDimNum);
        outputDgammaGm.SetGlobalBuffer((__gm__ float *)outputDgamma, dDimNum);
    }

    __aicore__ inline void InitQueue()
    {
        uint32_t sizeND = mergeNCountUpdatePer * elemWithDInUB * sizeof(T);
        uint32_t sizeNDFp32 = mergeNCountUpdatePer * elemWithDInUB * sizeof(float);
        uint32_t sizeD = elemWithDInUB * sizeof(T);
        uint32_t sizeDFp32 = elemWithDInUBFp32 * sizeof(float);
        uint32_t brcbNFp32 = brcbLineAlignedPer * elemWithoutDInUBFp32 * sizeof(float);
        uint32_t brcbNDFp32 = brcbLineAlignedPer * elemWithDInUB * sizeof(float);

        pipe.InitBuffer(dyQue, BUFFER_NUM, sizeND);
        pipe.InitBuffer(xQue, BUFFER_NUM, sizeND);
        pipe.InitBuffer(gxQue, BUFFER_NUM, sizeND);
        pipe.InitBuffer(meanQue, BUFFER_NUM, brcbNFp32);
        pipe.InitBuffer(rstdQue, BUFFER_NUM, brcbNFp32);
        pipe.InitBuffer(gammaQue, BUFFER_NUM, sizeD);
        pipe.InitBuffer(outputPdXQue, BUFFER_NUM, sizeND);
        pipe.InitBuffer(outputPdGxQue, BUFFER_NUM, sizeND);
        pipe.InitBuffer(outputPdBetaQue, BUFFER_NUM, sizeDFp32);
        pipe.InitBuffer(outputPdGammaQue, BUFFER_NUM, sizeDFp32);

        // tmp buffer
        pipe.InitBuffer(tmpNDBuf, sizeNDFp32);
        pipe.InitBuffer(brcbNDBuf1, brcbNDFp32);
        pipe.InitBuffer(brcbNDBuf2, brcbNDFp32);
#if __CCE_AICORE__ == 220
        if constexpr (IsSame<T, half>::value || IsSame<T, bfloat16_t>::value) {
#else
        if constexpr (IsSame<T, half>::value) {
#endif
            pipe.InitBuffer(dyFp32Buf, sizeNDFp32);
            pipe.InitBuffer(xFp32Buf, sizeNDFp32);
            pipe.InitBuffer(gxFp32Buf, sizeNDFp32);
            pipe.InitBuffer(gammaFp32Buf, sizeDFp32);
            pipe.InitBuffer(outputPdXFp32Buf, sizeNDFp32);
            pipe.InitBuffer(outputPdGxFp32Buf, sizeNDFp32);
        }
    }

    __aicore__ inline void Init(GM_ADDR dataDy, GM_ADDR dataX, GM_ADDR dataGx, GM_ADDR dataRstd, GM_ADDR dataMean,
        GM_ADDR dataGamma, GM_ADDR outputDx, GM_ADDR outputDgx, GM_ADDR outputDgamma, GM_ADDR outputDbeta,
        DeepNormGradTilingData tiling, GM_ADDR usrWorkspace)
    {
        useCoreNum = tiling.useCoreNum;
        nDimNum = tiling.nDimNum;
        dDimNum = tiling.dDimNum;
        nDealPerCore = tiling.nDealPerCore;
        nDealLastCore = tiling.nDealLastCore;
        alphaVal = *reinterpret_cast<float *>(&tiling.alpha);
        fixedOutputFlag = tiling.fixedOutputFlag;

        mergeNCount = tiling.mergeNCount;  // >1: no cut;

        // init GM
        nDeal = (GetBlockIdx() != useCoreNum - 1) ? nDealPerCore : nDealLastCore;
        InitGM(dataDy, dataX, dataGx, dataRstd, dataMean, dataGamma, outputDx, outputDgx, outputDgamma, outputDbeta);

        // merge N
        mergeNCountUpdatePer = (mergeNCount > nDeal) ? nDeal : mergeNCount;
        mergeNTime = this->CeilDiv(nDeal, mergeNCountUpdatePer);
        mergeNCountUpdateTail = nDeal - (mergeNCountUpdatePer * (mergeNTime - 1));
        brcbLineAlignedPer = this->BlockAlign(mergeNCountUpdatePer, BRCB_ONCE_ELEM);
        brcbLineAlignedTail = this->BlockAlign(mergeNCountUpdateTail, BRCB_ONCE_ELEM);

        // init queue
        blockElem = BLOCK_ALIGN_SIZE / sizeof(T);
        blockElemFp32 = BLOCK_ALIGN_SIZE / sizeof(float);

        elemWithDInUB = this->BlockAlign(dDimNum, blockElem);
        elemWithoutDInUB = this->BlockAlign(1, blockElem);
        elemWithDInUBFp32 = this->BlockAlign(dDimNum, blockElemFp32);
        elemWithoutDInUBFp32 = this->BlockAlign(1, blockElemFp32);

        InitQueue();

        // use atomicadd, need init beta&gamma
        LocalTensor<float> temp_local_tensor = outputPdGammaQue.AllocTensor<float>();
        this->InitGmData(outputDgammaGm, outputDbetaGm, dDimNum, temp_local_tensor, elemWithoutDInUBFp32);
        outputPdGammaQue.FreeTensor(temp_local_tensor);
        // avoid muti cal in UB
        oneDivD = (float)-1.0 / dDimNum;
#if __CCE_AICORE__ == 220
        SyncAll();
#else
        uint32_t each_core_handle_num = BLOCK_ALIGN_SIZE / sizeof(int32_t);
        GlobalTensor<int32_t> syncGlobal_;
        syncGlobal_.SetGlobalBuffer((__gm__ int32_t *)usrWorkspace, useCoreNum * blockElemFp32);

        LocalTensor<int32_t> tmp_init_buf = outputPdGammaQue.AllocTensor<int32_t>();
        Duplicate(tmp_init_buf, 0, each_core_handle_num);
        DataCopy(syncGlobal_[each_core_handle_num * GetBlockIdx()], tmp_init_buf, each_core_handle_num);

        LocalTensor<int32_t> workLocal = outputPdBetaQue.AllocTensor<int32_t>();
        SyncAll(syncGlobal_, workLocal);
        outputPdGammaQue.FreeTensor(tmp_init_buf);
        outputPdBetaQue.FreeTensor(workLocal);
#endif
    }

    __aicore__ inline void Process()
    {
        CopyInPre(dDimNum);
        LocalTensor<T> gamma = gammaQue.DeQue<T>();
        LocalTensor<float> dbeta = outputPdBetaQue.AllocTensor<float>();
        LocalTensor<float> dgamma = outputPdGammaQue.AllocTensor<float>();

        // init atomic Tensor
        Duplicate(dbeta, 0.0f, elemWithDInUBFp32);
        Duplicate(dgamma, 0.0f, elemWithDInUBFp32);
#if __CCE_AICORE__ == 220
        if constexpr (IsSame<T, half>::value || IsSame<T, bfloat16_t>::value) {
#else
        if constexpr (IsSame<T, half>::value) {
#endif
            LocalTensor<float> gammaFp32 = gammaFp32Buf.Get<float>();
            Cast(gammaFp32, gamma, RoundMode::CAST_NONE, dDimNum);
            ProcessMergeNFp16(gammaFp32, dbeta, dgamma);
        } else {
            ProcessMergeNFp32(gamma, dbeta, dgamma);
        }

        gammaQue.FreeTensor(gamma);
        outputPdBetaQue.EnQue(dbeta);
        outputPdGammaQue.EnQue(dgamma);

        if (fixedOutputFlag == 0) {
            CopyOutAfter(dDimNum);
        } else {
            CopyOutAfterInOrder(dDimNum);
        }
    }

    __aicore__ inline void ProcessMergeNFp16(
        const LocalTensor<float> &gamma, const LocalTensor<float> &dbeta, const LocalTensor<float> &dgamma)
    {
        for (uint32_t iMerge = 0; iMerge < mergeNTime; ++iMerge) {
            uint32_t mergeNCountUpdate = (iMerge != mergeNTime - 1) ? mergeNCountUpdatePer : mergeNCountUpdateTail;
            uint32_t brcbLineAligned = (iMerge != mergeNTime - 1) ? brcbLineAlignedPer : brcbLineAlignedTail;

            CopyInMergeN(iMerge, mergeNCountUpdate, dDimNum);
            ComputeMergeNFp16(iMerge, mergeNCountUpdate, brcbLineAligned, dDimNum, gamma, dbeta, dgamma);
            CopyOutMergeN(iMerge, mergeNCountUpdate, dDimNum);
        }
    }

    __aicore__ inline void ProcessMergeNFp32(
        const LocalTensor<float> &gamma, const LocalTensor<float> &dbeta, const LocalTensor<float> &dgamma)
    {
        for (uint32_t iMerge = 0; iMerge < mergeNTime; ++iMerge) {
            uint32_t mergeNCountUpdate = (iMerge != mergeNTime - 1) ? mergeNCountUpdatePer : mergeNCountUpdateTail;
            uint32_t brcbLineAligned = (iMerge != mergeNTime - 1) ? brcbLineAlignedPer : brcbLineAlignedTail;

            CopyInMergeN(iMerge, mergeNCountUpdate, dDimNum);
            ComputeMergeNFp32(iMerge, mergeNCountUpdate, brcbLineAligned, dDimNum, gamma, dbeta, dgamma);
            CopyOutMergeN(iMerge, mergeNCountUpdate, dDimNum);
        }
    }

private:
    __aicore__ inline void CopyInPre(uint32_t processElem)
    {
        LocalTensor<T> gammaLocal = gammaQue.AllocTensor<T>();
        uint32_t offsetD = 0;
#if __CCE_AICORE__ == 220
        // gamma
        DataCopyParams dataCopyParamsD{(uint16_t)1, (uint16_t)(processElem * sizeof(T)), 0, 0};
        uint8_t rightPadElemD = this->BlockAlign(processElem, blockElem) - processElem;
        DataCopyPadParams rightPadParamsD{true, 0, rightPadElemD, 0};

        DataCopyPad(gammaLocal, gammaGm[offsetD], dataCopyParamsD, rightPadParamsD);
#else
        DataCopy(gammaLocal, gammaGm[offsetD], this->BlockAlign(processElem, blockElem));
#endif
        gammaQue.EnQue(gammaLocal);
    }

    __aicore__ inline void CopyOutAfter(uint32_t processElem)
    {
        LocalTensor<float> dbeta = outputPdBetaQue.DeQue<float>();
        LocalTensor<float> dgamma = outputPdGammaQue.DeQue<float>();

        SetAtomicAdd<float>();
        DataCopyAutomicAdd(outputDgammaGm, dgamma, processElem, 0, (uint16_t)1);
        DataCopyAutomicAdd(outputDbetaGm, dbeta, processElem, 0, (uint16_t)1);

        SetAtomicNone();

        outputPdBetaQue.FreeTensor(dbeta);
        outputPdGammaQue.FreeTensor(dgamma);
    }

    __aicore__ inline void CopyOutAfterInOrder(uint32_t processElem)
    {
        uint32_t alreadyFixOutputSyncVal = GetBlockIdx();
        for (int32_t count = 0; count < INT_MAX; count++) {
            if (g_FixedOutputSync[0] == alreadyFixOutputSyncVal) {
                break;
            }
        }

        LocalTensor<float> dbeta = outputPdBetaQue.DeQue<float>();
        LocalTensor<float> dgamma = outputPdGammaQue.DeQue<float>();

        SetAtomicAdd<float>();
        DataCopyAutomicAdd(outputDgammaGm, dgamma, processElem, 0, (uint16_t)1);
        DataCopyAutomicAdd(outputDbetaGm, dbeta, processElem, 0, (uint16_t)1);

        SetAtomicNone();

        event_t eventMTE3S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventMTE3S);
        WaitFlag<HardEvent::MTE3_S>(eventMTE3S);

        outputPdBetaQue.FreeTensor(dbeta);
        outputPdGammaQue.FreeTensor(dgamma);
        g_FixedOutputSync[0]++;
    }

    __aicore__ inline void CopyInMergeN(uint32_t processID, uint32_t processNCount, uint32_t processElem)
    {
        LocalTensor<T> dyLocal = dyQue.AllocTensor<T>();
        LocalTensor<T> xLocal = xQue.AllocTensor<T>();
        LocalTensor<T> gxLocal = gxQue.AllocTensor<T>();
        LocalTensor<float> meanLocal = meanQue.AllocTensor<float>();
        LocalTensor<float> rstdLocal = rstdQue.AllocTensor<float>();

        uint32_t offsetND = processID * mergeNCountUpdatePer * dDimNum;
        uint32_t offsetN = processID * mergeNCountUpdatePer;
#if __CCE_AICORE__ == 220
        // dy&x&gx
        DataCopyParams dataCopyParamsND{(uint16_t)processNCount, (uint16_t)(processElem * sizeof(T)), 0, 0};
        uint8_t rightPadElemND = this->BlockAlign(processElem, blockElem) - processElem;
        DataCopyPadParams rightPadParamsND{true, 0, rightPadElemND, 0};

        DataCopyPad(dyLocal, dyGm[offsetND], dataCopyParamsND, rightPadParamsND);
        DataCopyPad(xLocal, xGm[offsetND], dataCopyParamsND, rightPadParamsND);
        DataCopyPad(gxLocal, gxGm[offsetND], dataCopyParamsND, rightPadParamsND);

        // mean&rstd
        DataCopyParams data_copy_params_N{(uint16_t)processNCount, (uint16_t)(1 * sizeof(float)), 0, 0};
        uint8_t right_pad_elem_N = this->BlockAlign(1, blockElemFp32) - 1;
        DataCopyPadParams right_pad_params_N{true, 0, right_pad_elem_N, 0};

        DataCopyPad(meanLocal, meanGm[offsetN], data_copy_params_N, right_pad_params_N);
        DataCopyPad(rstdLocal, rstdGm[offsetN], data_copy_params_N, right_pad_params_N);

#else
        for (uint32_t idx = 0; idx < processNCount; idx++) {
            DataCopy(dyLocal[idx * this->BlockAlign(processElem, blockElem)],
                dyGm[offsetND + idx * processElem],
                this->BlockAlign(processElem, blockElem));
            DataCopy(xLocal[idx * this->BlockAlign(processElem, blockElem)],
                xGm[offsetND + idx * processElem],
                this->BlockAlign(processElem, blockElem));
            DataCopy(gxLocal[idx * this->BlockAlign(processElem, blockElem)],
                gxGm[offsetND + idx * processElem],
                this->BlockAlign(processElem, blockElem));
        }
        // mean&rstd
        DataCopy(meanLocal, meanGm[offsetN], this->BlockAlign(processNCount, BRCB_ONCE_ELEM));
        DataCopy(rstdLocal, rstdGm[offsetN], this->BlockAlign(processNCount, BRCB_ONCE_ELEM));
#endif
        dyQue.EnQue(dyLocal);
        xQue.EnQue(xLocal);
        gxQue.EnQue(gxLocal);
        meanQue.EnQue(meanLocal);
        rstdQue.EnQue(rstdLocal);
    }

    __aicore__ inline void CopyOutMergeN(uint32_t processID, uint32_t processNCount, uint32_t processElem)
    {
        LocalTensor<T> outputPdXLocal = outputPdXQue.DeQue<T>();
        LocalTensor<T> outputPdGxLocal = outputPdGxQue.DeQue<T>();

        uint32_t offsetND = processID * mergeNCountUpdatePer * dDimNum;

        DataCopyCustom<T>(outputDxGm, outputPdXLocal, processElem, offsetND, false, (uint16_t)processNCount);
        DataCopyCustom<T>(outputDgxGm, outputPdGxLocal, processElem, offsetND, false, (uint16_t)processNCount);
        outputPdXQue.FreeTensor(outputPdXLocal);
        outputPdGxQue.FreeTensor(outputPdGxLocal);
    }

    __aicore__ inline void ComputeMergeNFp32(uint32_t processID, uint32_t processNCount, uint32_t brcbLineAligned,
        uint32_t processElem, const LocalTensor<float> &gamma, const LocalTensor<float> &dbeta,
        const LocalTensor<float> &dgamma)
    {
        LocalTensor<T> inputDy = dyQue.DeQue<T>();
        LocalTensor<T> inputX = xQue.DeQue<T>();
        LocalTensor<T> inputGx = gxQue.DeQue<T>();
        LocalTensor<float> inputRstd = rstdQue.DeQue<float>();
        LocalTensor<float> inputMean = meanQue.DeQue<float>();
        LocalTensor<T> outputDx = outputPdXQue.AllocTensor<T>();
        LocalTensor<T> outputDgx = outputPdGxQue.AllocTensor<T>();

        MainCompute(inputDy,
            inputX,
            inputGx,
            inputRstd,
            inputMean,
            gamma,
            outputDx,
            outputDgx,
            dbeta,
            dgamma,
            processNCount,
            brcbLineAligned,
            processElem);

        dyQue.FreeTensor(inputDy);
        xQue.FreeTensor(inputX);
        gxQue.FreeTensor(inputGx);
        rstdQue.FreeTensor(inputRstd);
        meanQue.FreeTensor(inputMean);
        outputPdXQue.EnQue(outputDx);
        outputPdGxQue.EnQue(outputDgx);
    }

    __aicore__ inline void ComputeMergeNFp16(uint32_t processID, uint32_t processNCount, uint32_t brcbLineAligned,
        uint32_t processElem, const LocalTensor<float> &gamma, const LocalTensor<float> &dbeta,
        const LocalTensor<float> &dgamma)
    {
        LocalTensor<T> inputDy = dyQue.DeQue<T>();
        LocalTensor<T> inputX = xQue.DeQue<T>();
        LocalTensor<T> inputGx = gxQue.DeQue<T>();
        LocalTensor<float> inputRstd = rstdQue.DeQue<float>();
        LocalTensor<float> inputMean = meanQue.DeQue<float>();
        LocalTensor<T> outputDx = outputPdXQue.AllocTensor<T>();
        LocalTensor<T> outputDgx = outputPdGxQue.AllocTensor<T>();
        LocalTensor<float> tmpNDBufLocal = tmpNDBuf.Get<float>();
        LocalTensor<float> brcbNDBufLocal1 = brcbNDBuf1.Get<float>();
        LocalTensor<float> brcbNDBufLocal2 = brcbNDBuf2.Get<float>();

        LocalTensor<float> dyFp32Local = dyFp32Buf.Get<float>();
        LocalTensor<float> xFp32Local = xFp32Buf.Get<float>();
        LocalTensor<float> gxFp32Local = gxFp32Buf.Get<float>();
        LocalTensor<float> outputPdXFp32Local = outputPdXFp32Buf.Get<float>();
        LocalTensor<float> outputPdGxFp32Local = outputPdGxFp32Buf.Get<float>();

        uint32_t processElemND = processNCount * elemWithDInUB;

        Cast(dyFp32Local, inputDy, RoundMode::CAST_NONE, processElemND);
        Cast(xFp32Local, inputX, RoundMode::CAST_NONE, processElemND);
        Cast(gxFp32Local, inputGx, RoundMode::CAST_NONE, processElemND);
        PipeBarrier<PIPE_V>();

        MainCompute(dyFp32Local,
            xFp32Local,
            gxFp32Local,
            inputRstd,
            inputMean,
            gamma,
            outputPdXFp32Local,
            outputPdGxFp32Local,
            dbeta,
            dgamma,
            processNCount,
            brcbLineAligned,
            processElem);

        if constexpr (IsSame<T, half>::value) {
            Cast(outputDx, outputPdXFp32Local, RoundMode::CAST_NONE, processElemND);
            Cast(outputDgx, outputPdGxFp32Local, RoundMode::CAST_NONE, processElemND);
        } else {
            Cast(outputDx, outputPdXFp32Local, RoundMode::CAST_RINT, processElemND);
            Cast(outputDgx, outputPdGxFp32Local, RoundMode::CAST_RINT, processElemND);
        }

        dyQue.FreeTensor(inputDy);
        xQue.FreeTensor(inputX);
        gxQue.FreeTensor(inputGx);
        rstdQue.FreeTensor(inputRstd);
        meanQue.FreeTensor(inputMean);
        outputPdXQue.EnQue(outputDx);
        outputPdGxQue.EnQue(outputDgx);
    }

    __aicore__ inline void MainCompute(const LocalTensor<float> &inputDy, const LocalTensor<float> &inputX,
        const LocalTensor<float> &inputGx, const LocalTensor<float> &inputRstd, const LocalTensor<float> &inputMean,
        const LocalTensor<float> &inputGamma, const LocalTensor<float> &outputDx, const LocalTensor<float> &outputDgx,
        const LocalTensor<float> &outputDbeta, const LocalTensor<float> &outputDgamma, uint32_t processNCount,
        uint32_t brcbLineAligned, uint32_t processElem)
    {
        LocalTensor<float> tmpNDBufLocal = tmpNDBuf.Get<float>();
        LocalTensor<float> brcbNDBufLocal1 = brcbNDBuf1.Get<float>();
        LocalTensor<float> brcbNDBufLocal2 = brcbNDBuf2.Get<float>();

        uint32_t processElemND = processNCount * elemWithDInUB;
        uint32_t brcbRepTimes = brcbLineAligned / BRCB_ONCE_ELEM;
        uint8_t brcbBlockStride = elemWithDInUB / FLOAT_BLOCK_ELEM;
        uint16_t brcbRepStride = brcbBlockStride * BRCB_ONCE_ELEM;

        // 1.1. x_sum = alpha * x + gx
        Axpy(inputGx, inputX, alphaVal, processElemND);
        // 1.2. tmpTensor1 = dy * gamma
        this->Level0MulFp32Short(outputDgx, inputDy, inputGamma, elemWithDInUB, processNCount, processElem);
        // 1.3. brcb mean
        for (uint32_t elemIndex = 0; elemIndex < elemWithDInUB; elemIndex += FLOAT_BLOCK_ELEM) {
            Brcb(brcbNDBufLocal1[elemIndex], inputMean, brcbRepTimes, {brcbBlockStride, brcbRepStride});
        }
        // 1.4. brcb rstd
        for (uint32_t elemIndex = 0; elemIndex < elemWithDInUB; elemIndex += FLOAT_BLOCK_ELEM) {
            Brcb(brcbNDBufLocal2[elemIndex], inputRstd, brcbRepTimes, {brcbBlockStride, brcbRepStride});
        }
        PipeBarrier<PIPE_V>();

        // 2.1. tmpTensor2 = x_sum - mean
        Sub(inputGx, inputGx, brcbNDBufLocal1, processElemND);
        // 2.2. d_var process: rstd * rstd
        Mul(inputX, inputRstd, inputRstd, processNCount);
        // 2.3. d_mean/d_gx process: tmpTensor1 * rstd
        Mul(tmpNDBufLocal, outputDgx, brcbNDBufLocal2, processElemND);
        PipeBarrier<PIPE_V>();

        // 3.1. d_gamma process: tmpTensor2 * rstd
        Mul(outputDx, inputGx, brcbNDBufLocal2, processElemND);
        // 3.2. d_var process: rstd^3
        Mul(inputRstd, inputRstd, inputX, processNCount);
        PipeBarrier<PIPE_V>();

        // 4.1. brcb rstd^3
        for (uint32_t elemIndex = 0; elemIndex < elemWithDInUB; elemIndex += FLOAT_BLOCK_ELEM) {
            Brcb(brcbNDBufLocal1[elemIndex], inputRstd, brcbRepTimes, {brcbBlockStride, brcbRepStride});
        }
        PipeBarrier<PIPE_V>();

        // 5.1. d_var process: tmpTensor2 * rstd^3
        Mul(inputX, inputGx, brcbNDBufLocal1, processElemND);
        PipeBarrier<PIPE_V>();

        // 6.1. d_beta end: add(dy)
        this->Level0AddFp32Short(outputDbeta, inputDy, elemWithDInUB, processNCount, processElem);
        // 6.2. d_gamma process: tmpTensor2 * rstd * dy
        Mul(outputDx, outputDx, inputDy, processElemND);
        // 6.3. d_var process: tmpTensor2 * rstd^3 * tmpTensor1
        Mul(inputX, inputX, outputDgx, processElemND);
        PipeBarrier<PIPE_V>();

        // 7.1. d_var end = reducesum(tmpTensor2 * rstd^3 * tmpTensor1)
        Duplicate<float>(inputDy, 0, processNCount * FLOAT_BLOCK_ELEM);
        PipeBarrier<PIPE_V>();
        this->ReduceSumFp32Short(inputMean, inputX, inputDy, elemWithDInUB, processNCount, processElem);
        PipeBarrier<PIPE_V>();

        // 8.1. d_gx process: -1/D * d_var
        Muls(inputMean, inputMean, oneDivD, processNCount);
        Duplicate<float>(inputDy, 0, processNCount * FLOAT_BLOCK_ELEM);
        PipeBarrier<PIPE_V>();

        // 9.1. d_mean end: reducesum(tmpTensor1 * rstd)
        this->ReduceSumFp32Short(inputRstd, tmpNDBufLocal, inputDy, elemWithDInUB, processNCount, processElem);
        PipeBarrier<PIPE_V>();

        // 10.1. d_gx process: -1/D * d_mean
        Muls(inputRstd, inputRstd, oneDivD, processNCount);
        // 10.2. vbrcb
        for (uint32_t elemIndex = 0; elemIndex < elemWithDInUB; elemIndex += FLOAT_BLOCK_ELEM) {
            Brcb(brcbNDBufLocal1[elemIndex], inputMean, brcbRepTimes, {brcbBlockStride, brcbRepStride});
        }
        PipeBarrier<PIPE_V>();

        // 11.1. d_gx process: -1/D * d_var * tmpTensor2
        Mul(outputDgx, inputGx, brcbNDBufLocal1, processElemND);
        // 11.2. vbrcb
        for (uint32_t elemIndex = 0; elemIndex < elemWithDInUB; elemIndex += FLOAT_BLOCK_ELEM) {
            Brcb(brcbNDBufLocal2[elemIndex], inputRstd, brcbRepTimes, {brcbBlockStride, brcbRepStride});
        }
        PipeBarrier<PIPE_V>();

        // 12.1. d_gx end: (-1/D * d_var * tmpTensor1) + (-1/D * d_mean)
        Add(outputDgx, outputDgx, brcbNDBufLocal2, processElemND);
        PipeBarrier<PIPE_V>();

        // 13.1. d_gamma end: add(tmpTensor2 * rstd * dy)
        this->Level0AddFp32Short(outputDgamma, outputDx, elemWithDInUB, processNCount, processElem);

        // 13.2. d_gx process: (-1/D * d_var * tmpTensor1) + (-1/D * d_mean) + (tmpTensor1 * rstd)
        Add(outputDgx, outputDgx, tmpNDBufLocal, processElemND);
        PipeBarrier<PIPE_V>();

        // 14.1. d_x end: alpha * dgx
        Muls(outputDx, outputDgx, alphaVal, processElemND);
        PipeBarrier<PIPE_V>();
    }

public:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> dyQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> gxQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> rstdQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> meanQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> gammaQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputPdXQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputPdGxQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputPdBetaQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputPdGammaQue;

    // cast buf for fp16&bf16
    TBuf<TPosition::VECCALC> dyFp32Buf;
    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> gxFp32Buf;
    TBuf<TPosition::VECCALC> gammaFp32Buf;
    TBuf<TPosition::VECCALC> outputPdXFp32Buf;
    TBuf<TPosition::VECCALC> outputPdGxFp32Buf;

    TBuf<TPosition::VECCALC> tmpNDBuf;
    TBuf<TPosition::VECCALC> brcbNDBuf1;
    TBuf<TPosition::VECCALC> brcbNDBuf2;

    // input
    GlobalTensor<T> dyGm;
    GlobalTensor<T> xGm;
    GlobalTensor<T> gxGm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<float> meanGm;

    // output
    GlobalTensor<T> outputDxGm;
    GlobalTensor<T> outputDgxGm;
    GlobalTensor<float> outputDgammaGm;
    GlobalTensor<float> outputDbetaGm;

    uint32_t useCoreNum;
    uint32_t nDimNum;
    uint32_t dDimNum;
    uint32_t nDealPerCore;
    uint32_t nDealLastCore;

    uint32_t nDeal;
    uint32_t blockElem;
    uint32_t blockElemFp32;

    // merge N params
    uint32_t mergeNCount;
    uint32_t mergeNCountUpdatePer;
    uint32_t mergeNCountUpdateTail;
    uint32_t mergeNTime;
    uint32_t brcbLineAlignedPer;
    uint32_t brcbLineAlignedTail;

    uint32_t elemWithDInUB;
    uint32_t elemWithoutDInUB;
    uint32_t elemWithDInUBFp32;
    uint32_t elemWithoutDInUBFp32;

    float oneDivD;
    float alphaVal;
    uint32_t fixedOutputFlag;
};

#endif
