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
 * \file deep_norm_grad_cut_d.h
 * \brief
 */
#ifndef OPS_BUILT_IN_TBE_IMPL_ASCENDC_DEEP_NORM_GRAD_DEEP_NORM_GRAD_CUT_D_H_
#define OPS_BUILT_IN_TBE_IMPL_ASCENDC_DEEP_NORM_GRAD_DEEP_NORM_GRAD_CUT_D_H_
#include "deep_norm_grad_common.h"

template <typename T>
class KernelDeepNormGradCutD : public KernelDeepNormGradBase<T> {
public:
    __aicore__ inline KernelDeepNormGradCutD()
    {}
    __aicore__ inline void InitGM(GM_ADDR dataDy, GM_ADDR dataX, GM_ADDR dataGx, GM_ADDR dataRstd, GM_ADDR dataMean,
        GM_ADDR dataGamma, GM_ADDR outputPdX, GM_ADDR outputPdGx, GM_ADDR outputPdGamma, GM_ADDR outputPdBeta)
    {
        dyGm.SetGlobalBuffer((__gm__ T *)dataDy + GetBlockIdx() * nDealPerCore * dDimNum, nDeal * dDimNum);
        xGm.SetGlobalBuffer((__gm__ T *)dataX + GetBlockIdx() * nDealPerCore * dDimNum, nDeal * dDimNum);
        gxGm.SetGlobalBuffer((__gm__ T *)dataGx + GetBlockIdx() * nDealPerCore * dDimNum, nDeal * dDimNum);
        meanGm.SetGlobalBuffer((__gm__ float *)dataMean + GetBlockIdx() * nDealPerCore, nDeal);
        rstdGm.SetGlobalBuffer((__gm__ float *)dataRstd + GetBlockIdx() * nDealPerCore, nDeal);
        gammaGm.SetGlobalBuffer((__gm__ T *)dataGamma, dDimNum);

        outputPdXGm.SetGlobalBuffer((__gm__ T *)outputPdX + GetBlockIdx() * nDealPerCore * dDimNum, nDeal * dDimNum);
        outputPdGxGm.SetGlobalBuffer((__gm__ T *)outputPdGx + GetBlockIdx() * nDealPerCore * dDimNum, nDeal * dDimNum);
        outputPdBetaGm.SetGlobalBuffer((__gm__ float *)outputPdBeta, dDimNum);
        outputPdGammaGm.SetGlobalBuffer((__gm__ float *)outputPdGamma, dDimNum);
        // use atomicadd, need init beta&gamma
    }

    __aicore__ inline void InitQueue()
    {
        pipe.InitBuffer(dyQue, BUFFER_NUM, elemWithDInUB * sizeof(T));
        pipe.InitBuffer(xQue, BUFFER_NUM, elemWithDInUB * sizeof(T));
        pipe.InitBuffer(gxQue, BUFFER_NUM, elemWithDInUB * sizeof(T));
        pipe.InitBuffer(meanQue, BUFFER_NUM, elemWithoutDInUBFp32 * sizeof(float));
        pipe.InitBuffer(rstdQue, BUFFER_NUM, elemWithoutDInUBFp32 * sizeof(float));
        pipe.InitBuffer(gammaQue, BUFFER_NUM, elemWithDInUB * sizeof(T));
        pipe.InitBuffer(outputPdXQue, BUFFER_NUM, elemWithDInUB * sizeof(T));
        pipe.InitBuffer(outputPdGxQue, BUFFER_NUM, elemWithDInUB * sizeof(T));
        pipe.InitBuffer(outputPdGammaQue, BUFFER_NUM, elemWithDInUBFp32 * sizeof(float));
        pipe.InitBuffer(outputPdBetaQue, BUFFER_NUM, elemWithDInUBFp32 * sizeof(float));

        pipe.InitBuffer(tmpMeanPdBuf, elemWithoutDInUBFp32 * sizeof(float));
        pipe.InitBuffer(tmpVarPdBuf, elemWithoutDInUBFp32 * sizeof(float));
#if __CCE_AICORE__ == 220
        if constexpr (IsSame<T, half>::value || IsSame<T, bfloat16_t>::value) {
#else
        if constexpr (IsSame<T, half>::value) {
#endif
            pipe.InitBuffer(dyFp32Buf, elemWithDInUBFp32 * sizeof(float));
            pipe.InitBuffer(xFp32Buf, elemWithDInUBFp32 * sizeof(float));
            pipe.InitBuffer(gxFp32Buf, elemWithDInUBFp32 * sizeof(float));
            pipe.InitBuffer(gammaFp32Buf, elemWithDInUBFp32 * sizeof(float));
            pipe.InitBuffer(outputPdXFp32Buf, elemWithDInUBFp32 * sizeof(float));
            pipe.InitBuffer(outputPdGxFp32Buf, elemWithDInUBFp32 * sizeof(float));
        }
    }

    __aicore__ inline void Init(GM_ADDR dataDy, GM_ADDR dataX, GM_ADDR dataGx, GM_ADDR dataRstd, GM_ADDR dataMean,
        GM_ADDR dataGamma, GM_ADDR outputPdX, GM_ADDR outputPdGx, GM_ADDR outputPdGamma, GM_ADDR outputPdBeta,
        DeepNormGradTilingData tiling, GM_ADDR usrWorkspace)
    {
        useCoreNum = tiling.useCoreNum;
        nDimNum = tiling.nDimNum;
        dDimNum = tiling.dDimNum;
        nDealPerCore = tiling.nDealPerCore;
        nDealLastCore = tiling.nDealLastCore;
        alphaVal = *reinterpret_cast<float *>(&tiling.alpha);

        cutDTime = tiling.cutDTime;  // 1: no cut; >1: cut
        cutDPerTime = tiling.cutDPerTime;
        cutDLastTime = tiling.cutDLastTime;
        fixedOutputFlag = tiling.fixedOutputFlag;

        // init GM
        nDeal = (GetBlockIdx() != useCoreNum - 1) ? nDealPerCore : nDealLastCore;
        InitGM(
            dataDy, dataX, dataGx, dataRstd, dataMean, dataGamma, outputPdX, outputPdGx, outputPdGamma, outputPdBeta);

        // cut D
        uint32_t dDimNumAlloc = dDimNum;
        if (cutDTime > 1) {
            dDimNumAlloc = cutDPerTime;
        }

        // init queue
        blockElem = BLOCK_ALIGN_SIZE / sizeof(T);
        blockElemFp32 = BLOCK_ALIGN_SIZE / sizeof(float);

        elemWithDInUB = this->BlockAlign(dDimNumAlloc, blockElem);
        elemWithoutDInUB = this->BlockAlign(1, blockElem);
        elemWithDInUBFp32 = this->BlockAlign(dDimNumAlloc, blockElemFp32);
        elemWithoutDInUBFp32 = this->BlockAlign(1, blockElemFp32);

        InitQueue();

        // use atomicadd, need init beta&gamma
        LocalTensor<float> temp_local_tensor = outputPdGammaQue.AllocTensor<float>();
        this->InitGmData(outputPdGammaGm, outputPdBetaGm, dDimNum, temp_local_tensor, elemWithoutDInUBFp32);
        outputPdGammaQue.FreeTensor(temp_local_tensor);
        // avoid muti cal in UB
        oneDivD = (float)-1.0 / dDimNum;
#if __CCE_AICORE__ == 220
        SyncAll();
#else
        uint32_t each_core_handle_num = BLOCK_ALIGN_SIZE / sizeof(int32_t);
        GlobalTensor<int32_t> syncGlobal_;
        syncGlobal_.SetGlobalBuffer((__gm__ int32_t *)usrWorkspace, useCoreNum * blockElemFp32);

        LocalTensor<int32_t> tmp_init_buf = outputPdBetaQue.AllocTensor<int32_t>();
        Duplicate(tmp_init_buf, 0, each_core_handle_num);
        DataCopy(syncGlobal_[each_core_handle_num * GetBlockIdx()], tmp_init_buf, each_core_handle_num);

        LocalTensor<int32_t> workLocal = outputPdGammaQue.AllocTensor<int32_t>();
        SyncAll(syncGlobal_, workLocal);
        outputPdGammaQue.FreeTensor(workLocal);
        outputPdBetaQue.FreeTensor(tmp_init_buf);
#endif
    }

    __aicore__ inline void ProcessFirstPart(const LocalTensor<float> &dyFp32Local, const LocalTensor<float> &xFp32Local,
        const LocalTensor<float> &gxFp32Local, const LocalTensor<float> &gammaFp32Local,
        const LocalTensor<float> &tmpVarPdLocal, const LocalTensor<float> &tmpMeanPdLocal, uint32_t dDimNumUB,
        const uint32_t processID)
    {
        for (uint32_t cutDIndex = 0; cutDIndex < cutDTime; cutDIndex++) {
            dDimNumUB = dDimNum;
            if (cutDIndex != cutDTime - 1) {
                dDimNumUB = cutDPerTime;
            } else {
                dDimNumUB = cutDLastTime;
            }

            CopyInCutD(processID, cutDIndex, dDimNumUB);
            CopyInGamma(cutDIndex, dDimNumUB);
            PrecisionComputeCutDFirstPart(
                dyFp32Local, xFp32Local, gxFp32Local, gammaFp32Local, tmpVarPdLocal, tmpMeanPdLocal, dDimNumUB);
        }
    }

    __aicore__ inline void ProcessSecondPart(const LocalTensor<float> &dyFp32Local,
        const LocalTensor<float> &xFp32Local, const LocalTensor<float> &gxFp32Local,
        const LocalTensor<float> &gammaFp32Local, const LocalTensor<float> &outputPdXLocal,
        const LocalTensor<float> &outputPdGxLocal, const LocalTensor<float> &tmpVarPdLocal,
        const LocalTensor<float> &tmpMeanPdLocal, uint32_t dDimNumUB, const uint32_t processID)
    {
        for (uint32_t cutDIndex = 0; cutDIndex < cutDTime; ++cutDIndex) {
            dDimNumUB = cutDPerTime;
            if (cutDIndex == cutDTime - 1) {
                dDimNumUB = cutDLastTime;
            }

            CopyInCutD(processID, cutDIndex, dDimNumUB);
            CopyInGamma(cutDIndex, dDimNumUB);
            PrecisionComputeCutDSecondPart(dyFp32Local,
                xFp32Local,
                gxFp32Local,
                gammaFp32Local,
                outputPdXLocal,
                outputPdGxLocal,
                tmpVarPdLocal,
                tmpMeanPdLocal,
                dDimNumUB);
            CopyOutX(processID, cutDIndex, dDimNumUB);
        }
    }

    __aicore__ inline void ProcessThirdPart(const LocalTensor<float> &dyFp32Local, const LocalTensor<float> &xFp32Local,
        const LocalTensor<float> &gxFp32Local)
    {
        for (uint32_t cutDIndex = 0; cutDIndex < cutDTime; cutDIndex++) {
            uint32_t dDimNumUB = dDimNum;
            if (cutDIndex != cutDTime - 1) {
                dDimNumUB = cutDPerTime;
            } else {
                dDimNumUB = cutDLastTime;
            }

            LocalTensor<float> dgammaFp32 = outputPdGammaQue.AllocTensor<float>();
            LocalTensor<float> dbetaFp32 = outputPdBetaQue.AllocTensor<float>();
            Duplicate(dgammaFp32, 0.0f, dDimNumUB);
            Duplicate(dbetaFp32, 0.0f, dDimNumUB);

            for (uint32_t processID = 0; processID < nDeal; ++processID) {
                CopyInCutD(processID, cutDIndex, dDimNumUB);
                CopyInMeanRstd(processID);
                PrecisionComputeCutDThirdPart(dyFp32Local, xFp32Local, gxFp32Local, dgammaFp32, dbetaFp32, dDimNumUB);
            }
            outputPdGammaQue.EnQue(dgammaFp32);
            outputPdBetaQue.EnQue(dbetaFp32);
            if (fixedOutputFlag == 0) {
                CopyOutDbetaDgamma(cutDIndex, dDimNumUB);
            } else {
                CopyOutDbetaDgammaInOrder(cutDIndex, dDimNumUB);
            }
        }
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> tmpMeanPdLocal = tmpMeanPdBuf.Get<float>();
        LocalTensor<float> tmpVarPdLocal = tmpVarPdBuf.Get<float>();

        LocalTensor<float> dyFp32Local;
        LocalTensor<float> xFp32Local;
        LocalTensor<float> gxFp32Local;
        LocalTensor<float> gammaFp32Local;
        LocalTensor<float> outputPdXLocal;
        LocalTensor<float> outputPdGxLocal;
#if __CCE_AICORE__ == 220
        if constexpr (IsSame<T, half>::value || IsSame<T, bfloat16_t>::value) {
#else
        if constexpr (IsSame<T, half>::value) {
#endif
            dyFp32Local = dyFp32Buf.Get<float>();
            xFp32Local = xFp32Buf.Get<float>();
            gxFp32Local = gxFp32Buf.Get<float>();
            gammaFp32Local = gammaFp32Buf.Get<float>();
            outputPdXLocal = outputPdXFp32Buf.Get<float>();
            outputPdGxLocal = outputPdGxFp32Buf.Get<float>();
        }

        for (uint32_t iDeal = 0; iDeal < nDeal; ++iDeal) {
            // init reducesum buf
            Duplicate(tmpMeanPdLocal, 0.0f, elemWithoutDInUBFp32);
            Duplicate(tmpVarPdLocal, 0.0f, elemWithoutDInUBFp32);

            // mean&rstd
            CopyInMeanRstd(iDeal);

            ProcessFirstPart(
                dyFp32Local, xFp32Local, gxFp32Local, gammaFp32Local, tmpVarPdLocal, tmpMeanPdLocal, dDimNum, iDeal);

            ProcessSecondPart(dyFp32Local,
                xFp32Local,
                gxFp32Local,
                gammaFp32Local,
                outputPdXLocal,
                outputPdGxLocal,
                tmpVarPdLocal,
                tmpMeanPdLocal,
                dDimNum,
                iDeal);

            CopyOutMeanRstd();
        }

        ProcessThirdPart(dyFp32Local, xFp32Local, gxFp32Local);
    }

private:
    __aicore__ inline void CopyInCutD(const uint32_t processID, const uint32_t cutDIndex, const uint32_t processElem)
    {
        LocalTensor<T> dyLocal = dyQue.AllocTensor<T>();
        LocalTensor<T> xLocal = xQue.AllocTensor<T>();
        LocalTensor<T> gxLocal = gxQue.AllocTensor<T>();

        uint32_t offsetND = processID * dDimNum + cutDIndex * cutDPerTime;
#if __CCE_AICORE__ == 220
        // dy&x&gx
        DataCopyParams dataCopyParamsND{(uint16_t)1, (uint16_t)(processElem * sizeof(T)), 0, 0};
        uint8_t rightPadElemND = this->BlockAlign(processElem, blockElem) - processElem;
        DataCopyPadParams rightPadParamsND{true, 0, rightPadElemND, 0};

        DataCopyPad(dyLocal, dyGm[offsetND], dataCopyParamsND, rightPadParamsND);
        DataCopyPad(xLocal, xGm[offsetND], dataCopyParamsND, rightPadParamsND);
        DataCopyPad(gxLocal, gxGm[offsetND], dataCopyParamsND, rightPadParamsND);
#else
        DataCopy(dyLocal, dyGm[offsetND], this->BlockAlign(processElem, blockElem));
        DataCopy(xLocal, xGm[offsetND], this->BlockAlign(processElem, blockElem));
        DataCopy(gxLocal, gxGm[offsetND], this->BlockAlign(processElem, blockElem));
#endif
        dyQue.EnQue(dyLocal);
        xQue.EnQue(xLocal);
        gxQue.EnQue(gxLocal);
    }

    __aicore__ inline void CopyInMeanRstd(const uint32_t processID)
    {
        LocalTensor<float> meanLocal = meanQue.AllocTensor<float>();
        LocalTensor<float> rstdLocal = rstdQue.AllocTensor<float>();
        uint32_t offsetN = processID;
#if __CCE_AICORE__ == 220
        // mean&rstd
        DataCopyParams dataCopyParamsN{(uint16_t)1, (uint16_t)(1 * sizeof(float)), 0, 0};
        uint8_t rightPadElemN = this->BlockAlign(1, blockElemFp32) - 1;
        DataCopyPadParams rightPadParamsN{true, 0, rightPadElemN, 0};

        DataCopyPad(meanLocal, meanGm[offsetN], dataCopyParamsN, rightPadParamsN);
        DataCopyPad(rstdLocal, rstdGm[offsetN], dataCopyParamsN, rightPadParamsN);
#else
        DataCopy(meanLocal, meanGm[offsetN], this->BlockAlign(1, blockElemFp32));
        DataCopy(rstdLocal, rstdGm[offsetN], this->BlockAlign(1, blockElemFp32));
#endif
        meanQue.EnQue(meanLocal);
        rstdQue.EnQue(rstdLocal);
    }

    __aicore__ inline void CopyOutMeanRstd()
    {
        LocalTensor<float> inputRstd = rstdQue.DeQue<float>();
        LocalTensor<float> inputMean = meanQue.DeQue<float>();
        meanQue.FreeTensor(inputRstd);
        rstdQue.FreeTensor(inputMean);
    }

    __aicore__ inline void CopyInGamma(const uint32_t cutDIndex, const uint32_t processElem)
    {
        LocalTensor<T> gammaLocal = gammaQue.AllocTensor<T>();
        uint32_t offsetD = cutDIndex * cutDPerTime;
        // gamma
#if __CCE_AICORE__ == 220
        DataCopyParams dataCopyParamsD{(uint16_t)1, (uint16_t)(processElem * sizeof(T)), 0, 0};
        uint8_t rightPadElemD = this->BlockAlign(processElem, blockElem) - processElem;
        DataCopyPadParams rightPadParamsD{true, 0, rightPadElemD, 0};

        DataCopyPad(gammaLocal, gammaGm[offsetD], dataCopyParamsD, rightPadParamsD);
#else
        DataCopy(gammaLocal, gammaGm[offsetD], this->BlockAlign(processElem, blockElem));
#endif
        gammaQue.EnQue(gammaLocal);
    }

    __aicore__ inline void CopyOutDbetaDgamma(const uint32_t cutDIndex, const uint32_t processElem)
    {
        uint32_t offsetD = cutDIndex * cutDPerTime;
        LocalTensor<float> outputPdGammaLocal = outputPdGammaQue.DeQue<float>();
        LocalTensor<float> outputPdBetaLocal = outputPdBetaQue.DeQue<float>();

        SetAtomicAdd<float>();
        DataCopyAutomicAdd(outputPdGammaGm, outputPdGammaLocal, processElem, offsetD, (uint16_t)1);
        DataCopyAutomicAdd(outputPdBetaGm, outputPdBetaLocal, processElem, offsetD, (uint16_t)1);
        SetAtomicNone();

        outputPdGammaQue.FreeTensor(outputPdGammaLocal);
        outputPdBetaQue.FreeTensor(outputPdBetaLocal);
    }

    __aicore__ inline void CopyOutDbetaDgammaInOrder(const uint32_t cutDIndex, const uint32_t processElem)
    {
        uint32_t alreadyFixOutputSyncVal = GetBlockIdx() + cutDIndex * useCoreNum;
        for (int32_t count = 0; count < INT_MAX; count++) {
            if (g_FixedOutputSync[0] == alreadyFixOutputSyncVal) {
                break;
            }
        }

        uint32_t offsetD = cutDIndex * cutDPerTime;
        LocalTensor<float> outputPdGammaLocal = outputPdGammaQue.DeQue<float>();
        LocalTensor<float> outputPdBetaLocal = outputPdBetaQue.DeQue<float>();

        SetAtomicAdd<float>();
        DataCopyAutomicAdd(outputPdGammaGm, outputPdGammaLocal, processElem, offsetD, (uint16_t)1);
        DataCopyAutomicAdd(outputPdBetaGm, outputPdBetaLocal, processElem, offsetD, (uint16_t)1);
        SetAtomicNone();

        event_t eventMTE3S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventMTE3S);
        WaitFlag<HardEvent::MTE3_S>(eventMTE3S);

        outputPdGammaQue.FreeTensor(outputPdGammaLocal);
        outputPdBetaQue.FreeTensor(outputPdBetaLocal);
        g_FixedOutputSync[0]++;
    }

    __aicore__ inline void CopyOutX(const uint32_t processID, const uint32_t cutDIndex, const uint32_t processElem)
    {
        LocalTensor<T> outputPdXLocal = outputPdXQue.DeQue<T>();
        LocalTensor<T> outputPdGxLocal = outputPdGxQue.DeQue<T>();

        uint32_t offsetND = processID * dDimNum + cutDIndex * cutDPerTime;

        DataCopyCustom<T>(outputPdXGm, outputPdXLocal, processElem, offsetND, false, (uint16_t)1);
        DataCopyCustom<T>(outputPdGxGm, outputPdGxLocal, processElem, offsetND, false, (uint16_t)1);

        outputPdXQue.FreeTensor(outputPdXLocal);
        outputPdGxQue.FreeTensor(outputPdGxLocal);
    }

    __aicore__ inline void PrecisionComputeCutDFirstPart(const LocalTensor<float> &dyFp32Local,
        const LocalTensor<float> &xFp32Local, const LocalTensor<float> &gxFp32Local,
        const LocalTensor<float> &gammaFp32Local, const LocalTensor<float> &tmpVarPdLocal,
        const LocalTensor<float> &tmpMeanPdLocal, const uint32_t processElem)
    {
        LocalTensor<T> inputDy = dyQue.DeQue<T>();
        LocalTensor<T> inputX = xQue.DeQue<T>();
        LocalTensor<T> inputGx = gxQue.DeQue<T>();
        LocalTensor<float> inputRstd = rstdQue.DeQue<float>();
        LocalTensor<float> inputMean = meanQue.DeQue<float>();
        LocalTensor<T> inputGamma = gammaQue.DeQue<T>();
#if __CCE_AICORE__ == 220
        if constexpr (IsSame<T, half>::value || IsSame<T, bfloat16_t>::value) {
#else
        if constexpr (IsSame<T, half>::value) {
#endif
            Cast(dyFp32Local, inputDy, RoundMode::CAST_NONE, processElem);
            Cast(xFp32Local, inputX, RoundMode::CAST_NONE, processElem);
            Cast(gxFp32Local, inputGx, RoundMode::CAST_NONE, processElem);
            Cast(gammaFp32Local, inputGamma, RoundMode::CAST_NONE, processElem);
            PipeBarrier<PIPE_V>();
            dyQue.FreeTensor(inputDy);
            xQue.FreeTensor(inputX);
            gxQue.FreeTensor(inputGx);
            gammaQue.FreeTensor(inputGamma);

            MainComputeFirstPart(dyFp32Local,
                xFp32Local,
                gxFp32Local,
                inputRstd,
                inputMean,
                gammaFp32Local,
                tmpVarPdLocal,
                tmpMeanPdLocal,
                processElem);
        } else {
            MainComputeFirstPart(
                inputDy, inputX, inputGx, inputRstd, inputMean, inputGamma, tmpVarPdLocal, tmpMeanPdLocal, processElem);

            dyQue.FreeTensor(inputDy);
            xQue.FreeTensor(inputX);
            gxQue.FreeTensor(inputGx);
            gammaQue.FreeTensor(inputGamma);
        }

        meanQue.EnQue(inputMean);
        rstdQue.EnQue(inputRstd);
    }

    __aicore__ inline void MainComputeFirstPart(const LocalTensor<float> &inputDy, const LocalTensor<float> &inputX,
        const LocalTensor<float> &inputGx, const LocalTensor<float> &inputRstd, const LocalTensor<float> &inputMean,
        const LocalTensor<float> &inputGamma, const LocalTensor<float> &tmpVarPdLocal,
        const LocalTensor<float> &tmpMeanPdLocal, const uint32_t processElem)
    {
        // 1.1. x_sum = alpha * x1 + x2
        Axpy(inputGx, inputX, alphaVal, processElem);
        // 1.2. tmpTensor1 = dy * gamma
        Mul(inputGamma, inputDy, inputGamma, processElem);
        PipeBarrier<PIPE_V>();

        event_t event_mte2_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));

        // 2.1. scalar
        SetFlag<HardEvent::MTE2_S>(event_mte2_s);
        WaitFlag<HardEvent::MTE2_S>(event_mte2_s);
        float inputMeanNum = inputMean.GetValue(0);
        float inputRstdNum = inputRstd.GetValue(0);
        float rstdSqrtTmpNum = inputRstdNum * inputRstdNum * inputRstdNum;
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);

        // 3.1. tmpTensor2 = x_sum - mean
        Adds(inputGx, inputGx, inputMeanNum * (-1.0f), processElem);
        // 3.2. dvar part process: tmpTensor1 * rstd^3
        Muls(inputX, inputGamma, rstdSqrtTmpNum, processElem);
        PipeBarrier<PIPE_V>();

        // 4.1. dvar part process: tmpTensor1 * rstd^3 * tmpTensor2
        Mul(inputX, inputX, inputGx, processElem);
        // 4.2. dmean part process: tmpTensor1 * rstd
        Muls(inputGamma, inputGamma, inputRstdNum, processElem);  // can't use in d_gx
        PipeBarrier<PIPE_V>();

        // 5.1. dvar part end: reducesum(tmpTensor1 * rstd^3 * tmpTensor2)
        // 5.2. dmean part end: reducesum(tmpTensor1 * rstd)
        auto reduceTmpNum = this->ReduceSumCustom(inputX, processElem);
        auto reduceTmpNum2 = this->ReduceSumCustom(inputGamma, processElem);
        reduceTmpNum = reduceTmpNum * oneDivD;
        reduceTmpNum2 = reduceTmpNum2 * oneDivD;
        PipeBarrier<PIPE_V>();

        // 6.1. add to tmp for second cal part
        Adds(tmpVarPdLocal, tmpVarPdLocal, reduceTmpNum, blockElemFp32);
        Adds(tmpMeanPdLocal, tmpMeanPdLocal, reduceTmpNum2, blockElemFp32);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PrecisionComputeCutDSecondPart(const LocalTensor<float> &dyFp32Local,
        const LocalTensor<float> &xFp32Local, const LocalTensor<float> &gxFp32Local,
        const LocalTensor<float> &gammaFp32Local, const LocalTensor<float> &outputPdXLocal,
        const LocalTensor<float> &outputPdGxLocal, const LocalTensor<float> &tmpVarPdLocal,
        const LocalTensor<float> &tmpMeanPdLocal, const uint32_t processElem)
    {
        LocalTensor<T> inputDy = dyQue.DeQue<T>();
        LocalTensor<T> inputX = xQue.DeQue<T>();
        LocalTensor<T> inputGx = gxQue.DeQue<T>();
        LocalTensor<float> inputRstd = rstdQue.DeQue<float>();
        LocalTensor<float> inputMean = meanQue.DeQue<float>();
        LocalTensor<T> inputGamma = gammaQue.DeQue<T>();

        LocalTensor<T> outputPdX = outputPdXQue.AllocTensor<T>();
        LocalTensor<T> outputPdGx = outputPdGxQue.AllocTensor<T>();

        if constexpr (IsSame<T, float>::value) {
            MainComputeSecondPart(inputDy,
                inputX,
                inputGx,
                inputRstd,
                inputMean,
                inputGamma,
                outputPdX,
                outputPdGx,
                tmpVarPdLocal,
                tmpMeanPdLocal,
                processElem);

            dyQue.FreeTensor(inputDy);
            xQue.FreeTensor(inputX);
            gxQue.FreeTensor(inputGx);
            gammaQue.FreeTensor(inputGamma);
        } else {
            Cast(dyFp32Local, inputDy, RoundMode::CAST_NONE, processElem);
            Cast(xFp32Local, inputX, RoundMode::CAST_NONE, processElem);
            Cast(gxFp32Local, inputGx, RoundMode::CAST_NONE, processElem);
            Cast(gammaFp32Local, inputGamma, RoundMode::CAST_NONE, processElem);
            PipeBarrier<PIPE_V>();
            dyQue.FreeTensor(inputDy);
            xQue.FreeTensor(inputX);
            gxQue.FreeTensor(inputGx);
            gammaQue.FreeTensor(inputGamma);

            MainComputeSecondPart(dyFp32Local,
                xFp32Local,
                gxFp32Local,
                inputRstd,
                inputMean,
                gammaFp32Local,
                outputPdXLocal,
                outputPdGxLocal,
                tmpVarPdLocal,
                tmpMeanPdLocal,
                processElem);

            if constexpr (IsSame<T, half>::value) {
                Cast(outputPdX, outputPdXLocal, RoundMode::CAST_NONE, processElem);
                Cast(outputPdGx, outputPdGxLocal, RoundMode::CAST_NONE, processElem);
            } else {
                Cast(outputPdX, outputPdXLocal, RoundMode::CAST_RINT, processElem);
                Cast(outputPdGx, outputPdGxLocal, RoundMode::CAST_RINT, processElem);
            }
            PipeBarrier<PIPE_V>();
        }

        meanQue.EnQue(inputMean);
        rstdQue.EnQue(inputRstd);
        outputPdXQue.EnQue(outputPdX);
        outputPdGxQue.EnQue(outputPdGx);
    }

    __aicore__ inline void MainComputeSecondPart(const LocalTensor<float> &inputDy, const LocalTensor<float> &inputX,
        const LocalTensor<float> &inputGx, const LocalTensor<float> &inputRstd, const LocalTensor<float> &inputMean,
        const LocalTensor<float> &inputGamma, const LocalTensor<float> &outputPdX, const LocalTensor<float> &outputPdGx,
        const LocalTensor<float> &tmpVarPdLocal, const LocalTensor<float> &tmpMeanPdLocal, const uint32_t processElem)
    {
        // 1.1. x_sum = alpha * x1 + x2
        Axpy(inputGx, inputX, alphaVal, processElem);
        // 1.2. tmpTensor1 = dy * gamma
        Mul(inputGamma, inputDy, inputGamma, processElem);
        PipeBarrier<PIPE_V>();

        event_t event_mte2_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));

        // 2.1. scalar
        SetFlag<HardEvent::MTE2_S>(event_mte2_s);
        WaitFlag<HardEvent::MTE2_S>(event_mte2_s);
        float inputMeanNum = inputMean.GetValue(0);
        float inputRstdNum = inputRstd.GetValue(0);
        float tmpVarPdNum = tmpVarPdLocal.GetValue(0);
        float tmpMeanPdNum = tmpMeanPdLocal.GetValue(0);
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);

        // 2.1. tmpTensor2 = x - mean
        Adds(inputGx, inputGx, inputMeanNum * (-1.0f), processElem);
        // 2.2. dgx process: tmpTensor1 * rstd
        Muls(inputX, inputGamma, inputRstdNum, processElem);
        PipeBarrier<PIPE_V>();

        // 3.1. dgx process: (-1.0/D) * d_var * tmpTensor2
        Muls(outputPdGx, inputGx, tmpVarPdNum, processElem);
        // 3.2. dgx process:  (-1.0/D * d_mean) + (tmpTensor1 * rstd)
        Adds(inputX, inputX, tmpMeanPdNum, processElem);
        PipeBarrier<PIPE_V>();

        // 4.1. dgx end: (-1.0/D * d_var * tmpTensor1) + (-1.0/D * d_mean) + (tmpTensor1 * rstd)
        Add(outputPdGx, outputPdGx, inputX, processElem);
        PipeBarrier<PIPE_V>();

        // 5.1. dx end: dx = alpha * dgx
        Muls(outputPdX, outputPdGx, alphaVal, processElem);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PrecisionComputeCutDThirdPart(const LocalTensor<float> &dyFp32Local,
        const LocalTensor<float> &xFp32Local, const LocalTensor<float> &gxFp32Local,
        const LocalTensor<float> &dgammaFp32, const LocalTensor<float> &dbetaFp32, const uint32_t processElem)
    {
        LocalTensor<T> inputDy = dyQue.DeQue<T>();
        LocalTensor<T> inputX = xQue.DeQue<T>();
        LocalTensor<T> inputGx = gxQue.DeQue<T>();
        LocalTensor<float> inputRstd = rstdQue.DeQue<float>();
        LocalTensor<float> inputMean = meanQue.DeQue<float>();
#if __CCE_AICORE__ == 220
        if constexpr (IsSame<T, half>::value || IsSame<T, bfloat16_t>::value) {
#else
        if constexpr (IsSame<T, half>::value) {
#endif
            Cast(dyFp32Local, inputDy, RoundMode::CAST_NONE, processElem);
            Cast(xFp32Local, inputX, RoundMode::CAST_NONE, processElem);
            Cast(gxFp32Local, inputGx, RoundMode::CAST_NONE, processElem);
            PipeBarrier<PIPE_V>();
            dyQue.FreeTensor(inputDy);
            xQue.FreeTensor(inputX);
            gxQue.FreeTensor(inputGx);

            MainComputeThirdPart(
                dyFp32Local, xFp32Local, gxFp32Local, inputRstd, inputMean, dgammaFp32, dbetaFp32, processElem);
        } else {
            MainComputeThirdPart(inputDy, inputX, inputGx, inputRstd, inputMean, dgammaFp32, dbetaFp32, processElem);

            dyQue.FreeTensor(inputDy);
            xQue.FreeTensor(inputX);
            gxQue.FreeTensor(inputGx);
        }
    }

    __aicore__ inline void MainComputeThirdPart(const LocalTensor<float> &inputDy, const LocalTensor<float> &inputX,
        const LocalTensor<float> &inputGx, LocalTensor<float> &inputRstd, LocalTensor<float> &inputMean,
        const LocalTensor<float> &outputPdGamma, const LocalTensor<float> &outputPdBeta, const uint32_t processElem)
    {
        // 1.1. x_sum = alpha * x1 + x2
        Axpy(inputGx, inputX, alphaVal, processElem);

        event_t event_mte2_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));

        // 2.1. scalar
        SetFlag<HardEvent::MTE2_S>(event_mte2_s);
        WaitFlag<HardEvent::MTE2_S>(event_mte2_s);
        float inputMeanNum = inputMean.GetValue(0);
        float inputRstdNum = inputRstd.GetValue(0);
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);

        rstdQue.FreeTensor(inputRstd);
        meanQue.FreeTensor(inputMean);
        PipeBarrier<PIPE_V>();

        // 3.1. tmpTensor2 = x_sum - mean
        Adds(inputGx, inputGx, inputMeanNum * (-1.0f), processElem);
        // 3.2. dgamma process: rstd * dy
        Muls(inputX, inputDy, inputRstdNum, processElem);
        PipeBarrier<PIPE_V>();

        // 4.1. dgamma process: tmpTensor2 * rstd * dy
        Mul(inputGx, inputGx, inputX, processElem);
        PipeBarrier<PIPE_V>();

        // 5.1. dgamma end: atomicadd (tmpTensor2 * rstd * dy)
        Add(outputPdGamma, outputPdGamma, inputGx, processElem);
        // 5.2. dbeta end: atomicadd(dy)
        Add(outputPdBeta, outputPdBeta, inputDy, processElem);
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
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputPdGammaQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputPdBetaQue;

    TBuf<TPosition::VECCALC> tmpMeanPdBuf;
    TBuf<TPosition::VECCALC> tmpVarPdBuf;

    // cast buf for fp16&bf16
    TBuf<TPosition::VECCALC> dyFp32Buf;
    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> gxFp32Buf;
    TBuf<TPosition::VECCALC> gammaFp32Buf;
    TBuf<TPosition::VECCALC> outputPdXFp32Buf;
    TBuf<TPosition::VECCALC> outputPdGxFp32Buf;
    TBuf<TPosition::VECCALC> x_buf_fp32;

    // input
    GlobalTensor<T> dyGm;
    GlobalTensor<T> xGm;
    GlobalTensor<T> gxGm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<float> meanGm;

    // output
    GlobalTensor<T> outputPdXGm;
    GlobalTensor<T> outputPdGxGm;
    GlobalTensor<float> outputPdGammaGm;
    GlobalTensor<float> outputPdBetaGm;

    uint32_t useCoreNum;
    uint32_t nDimNum;
    uint32_t dDimNum;
    uint32_t nDealPerCore;
    uint32_t nDealLastCore;

    uint32_t nDeal;
    uint32_t blockElem;
    uint32_t blockElemFp32;

    // cut D params
    uint32_t cutDTime;
    uint32_t cutDPerTime;
    uint32_t cutDLastTime;

    uint32_t elemWithDInUB;
    uint32_t elemWithoutDInUB;
    uint32_t elemWithDInUBFp32;
    uint32_t elemWithoutDInUBFp32;

    float oneDivD;
    float alphaVal;
    uint32_t fixedOutputFlag;
};

#endif
