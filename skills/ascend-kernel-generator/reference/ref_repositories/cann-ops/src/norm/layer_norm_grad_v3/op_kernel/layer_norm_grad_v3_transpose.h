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
 * \file layer_norm_grad_v3_transpose.h
 * \brief
 */

#ifndef LAYER_NORM_GRAD_V3_TRANSPOSE_H
#define LAYER_NORM_GRAD_V3_TRANSPOSE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace LayerNormGradV3 {
using namespace AscendC;

template <typename Tdy, typename Tgamma, bool isDeterministic>
class LayerNormGradV3Transpose {
public:
    __aicore__ inline LayerNormGradV3Transpose()
    {}
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean, GM_ADDR gamma, GM_ADDR pdX,
        GM_ADDR pdGamma, GM_ADDR pdBeta, GM_ADDR workspace,
        const LayerNormGradV3TilingDataTranspose *__restrict tilingData)
    {
        // load tiling data
        col = tilingData->col;
        row = tilingData->row;
        coff = tilingData->coefficient;
        blockDim = tilingData->blockDim;
        blockFormer = tilingData->blockFormer;
        blockTail = tilingData->blockTail;
        ubFormer = tilingData->ubFormer;
        bFormer = tilingData->bFormer;
        dichotomizeAddDiffSize = tilingData->dichotomizeAddDiffSize;
        ubLoopOfFormerBlock = tilingData->ubLoopOfFormerBlock;
        ubLoopOfTailBlock = tilingData->ubLoopOfTailBlock;
        ubTailOfFormerBlock = tilingData->ubTailOfFormerBlock;
        ubTailOfTailBlock = tilingData->ubTailOfTailBlock;

        // set input global buffer
        dyGm.SetGlobalBuffer((__gm__ Tdy *)dy + blockIdx * blockFormer * col);
        xGm.SetGlobalBuffer((__gm__ Tdy *)x + blockIdx * blockFormer * col);
        rstdGm.SetGlobalBuffer((__gm__ float *)rstd + blockIdx * blockFormer);
        meanGm.SetGlobalBuffer((__gm__ float *)mean + blockIdx * blockFormer);
        gammaGm.SetGlobalBuffer((__gm__ Tgamma *)gamma);
        // set output global buffer
        pdXGm.SetGlobalBuffer((__gm__ Tdy *)pdX + blockIdx * blockFormer * col);
        pdGammaGm.SetGlobalBuffer((__gm__ float *)pdGamma);
        pdBetaGm.SetGlobalBuffer((__gm__ float *)pdBeta);
        pdGammaWsp.SetGlobalBuffer((__gm__ float *)workspace);
        pdBetaWsp.SetGlobalBuffer((__gm__ float *)workspace + tilingData->deterministicComputeWspSize / FLOAT_SIZE);

        // pipe init buffer
        uint64_t inQueueDySize = (bFormer * col + B16_BLOCK_ALIGN_NUM - 1) / B16_BLOCK_ALIGN_NUM * B16_BLOCK_ALIGN_NUM *
                                 TRANSPOSE_C0_SIZE * FLOAT_SIZE;
        uint64_t inQueueMeanSize = (bFormer + B16_BLOCK_ALIGN_NUM - 1) / B16_BLOCK_ALIGN_NUM * B16_BLOCK_ALIGN_NUM *
                                   TRANSPOSE_C0_SIZE * FLOAT_SIZE;
        pipe.InitBuffer(inQueueDy, 1, inQueueDySize);
        pipe.InitBuffer(inQueueX, 1, inQueueDySize);
        pipe.InitBuffer(inQueueMean, 1, inQueueMeanSize);
        pipe.InitBuffer(inQueueRstd, 1, inQueueMeanSize);
        pipe.InitBuffer(inQueueGamma, 1, TRANSPOSE_COL_LIMIT * FLOAT_SIZE);
        pipe.InitBuffer(outQueuePdX, 1, inQueueDySize);
        pipe.InitBuffer(outQueuePdGamma, 1, TRANSPOSE_COL_LIMIT * FLOAT_SIZE);
        pipe.InitBuffer(outQueuePdBeta, 1, TRANSPOSE_COL_LIMIT * FLOAT_SIZE);
        pipe.InitBuffer(tmpBuf0, inQueueDySize);
        pipe.InitBuffer(tmpBuf1, (col + BLOCK_NUM_PER_REP - 1) / BLOCK_NUM_PER_REP * BLOCK_NUM_PER_REP * BLOCK);
        pipe.InitBuffer(tmpBuf2, inQueueMeanSize);

        // clear pdGamma/pdBeta outputs
        if constexpr (!isDeterministic) {
            if (blockIdx == 0) {
                InitOutput<float>(pdGammaGm, col, 0.0);
                InitOutput<float>(pdBetaGm, col, 0.0);
            }
            CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_AIV_ONLY_ALL);
        }
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Process()
    {
        uint64_t ubLoopCount;
        uint64_t ubTailLoopBlockLength;
        if (blockIdx < (blockDim - 1)) {
            ubLoopCount = ubLoopOfFormerBlock;
            ubTailLoopBlockLength = ubTailOfFormerBlock;
        }
        if (blockIdx == (blockDim - 1)) {
            ubLoopCount = ubLoopOfTailBlock;
            ubTailLoopBlockLength = ubTailOfTailBlock;
        }
        rowLength = ubFormer;
        CalcGeneralParams();
        CopyInGamma();
        PipeBarrier<PIPE_V>();
        pdGammaOut = outQueuePdGamma.AllocTensor<float>();
        pdBetaOut = outQueuePdBeta.AllocTensor<float>();
        Duplicate<float>(pdGammaOut, float(0.0), TRANSPOSE_COL_LIMIT);
        Duplicate<float>(pdBetaOut, float(0.0), TRANSPOSE_COL_LIMIT);
        PipeBarrier<PIPE_V>();
        // do basic block
        for (uint64_t loopIdx = 0; loopIdx < ubLoopCount; loopIdx++) {
            if (loopIdx == (ubLoopCount - 1)) {
                rowLength = ubTailLoopBlockLength;
                CalcGeneralParams();
            }
            xGmOffset = loopIdx * ubFormer * col;
            meanGmOffset = loopIdx * ubFormer;
            ProcessBasicBlock();
        }
        // post process
        colB32Align = (col + B32_BLOCK_ALIGN_NUM - 1) / B32_BLOCK_ALIGN_NUM * B32_BLOCK_ALIGN_NUM;
        if constexpr (!isDeterministic) {
            SetAtomicAdd<float>();
            CrossCoreWaitFlag(SYNC_AIV_ONLY_ALL);
        }
        DataCopyExtParams copyOutParams{1, uint32_t(col * sizeof(float)), 0, 0, 0};
        outQueuePdGamma.EnQue(pdGammaOut);
        outQueuePdGamma.DeQue<float>();
        if constexpr (!isDeterministic) {
            DataCopyPad(pdGammaGm, pdGammaOut, copyOutParams);
        } else {
            DataCopyPad(pdGammaWsp[blockIdx * colB32Align], pdGammaOut, copyOutParams);
        }
        outQueuePdGamma.FreeTensor(pdGammaOut);
        outQueuePdBeta.EnQue(pdBetaOut);
        outQueuePdBeta.DeQue<float>();
        if constexpr (!isDeterministic) {
            DataCopyPad(pdBetaGm, pdBetaOut, copyOutParams);
        } else {
            DataCopyPad(pdBetaWsp[blockIdx * colB32Align], pdBetaOut, copyOutParams);
        }
        outQueuePdBeta.FreeTensor(pdBetaOut);
        SetAtomicNone();

        if constexpr (isDeterministic) {
            DeterministicComputeGammaBeta();
        }
    }

private:
    __aicore__ inline void DeterministicComputeGammaBeta()
    {
        SyncAll();
        if (blockIdx < 1) {
            DataCopyExtParams copyInParams{1, uint32_t(colB32Align * FLOAT_SIZE), 0, 0, 0};
            DataCopyExtParams copyOutParams{1, uint32_t(col * FLOAT_SIZE), 0, 0, 0};
            LocalTensor<float> inTensor;
            LocalTensor<float> outGammaTensor = outQueuePdGamma.AllocTensor<float>();
            LocalTensor<float> outBetaTensor = outQueuePdBeta.AllocTensor<float>();
            Duplicate(outGammaTensor, static_cast<float>(0.0), colB32Align);
            Duplicate(outBetaTensor, static_cast<float>(0.0), colB32Align);
            PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < blockDim; i++) {
                inTensor = inQueueGamma.AllocTensor<float>();
                DataCopyPad(inTensor, pdGammaWsp[i * colB32Align], copyInParams, {false, 0, 0, 0});
                inQueueGamma.EnQue(inTensor);
                inQueueGamma.DeQue<float>();
                Add(outGammaTensor, outGammaTensor, inTensor, colB32Align);
                inQueueGamma.FreeTensor(inTensor);
            }
            outQueuePdGamma.EnQue(outGammaTensor);
            outQueuePdGamma.DeQue<float>();
            DataCopyPad(pdGammaGm, outGammaTensor, copyOutParams);
            outQueuePdGamma.FreeTensor(outGammaTensor);
            PipeBarrier<PIPE_ALL>();
            for (uint32_t i = 0; i < blockDim; i++) {
                inTensor = inQueueGamma.AllocTensor<float>();
                DataCopyPad(inTensor, pdBetaWsp[i * colB32Align], copyInParams, {false, 0, 0, 0});
                inQueueGamma.EnQue(inTensor);
                inQueueGamma.DeQue<float>();
                Add(outBetaTensor, outBetaTensor, inTensor, colB32Align);
                inQueueGamma.FreeTensor(inTensor);
            }
            outQueuePdBeta.EnQue(outBetaTensor);
            outQueuePdBeta.DeQue<float>();
            DataCopyPad(pdBetaGm, outBetaTensor, copyOutParams);
            outQueuePdBeta.FreeTensor(outBetaTensor);
        }
    }

    __aicore__ inline void CalcGeneralParams()
    {
        bFormerFactor = (rowLength + TRANSPOSE_C0_SIZE - 1) / TRANSPOSE_C0_SIZE;
        rFormerAxisAlign = (bFormerFactor * col + DY_NUM_PER_BLOCK - 1) / DY_NUM_PER_BLOCK * DY_NUM_PER_BLOCK;
        bTailFactor = bFormerFactor - 1;
        rTailAxisAlign = (bTailFactor * col + DY_NUM_PER_BLOCK - 1) / DY_NUM_PER_BLOCK * DY_NUM_PER_BLOCK;
        formerLoops = rowLength - TRANSPOSE_C0_SIZE * (bFormerFactor - 1);
        calcXElements = col * bFormerFactor * TRANSPOSE_C0_SIZE;
        colLineElements = bFormerFactor * TRANSPOSE_C0_SIZE;
    }

    __aicore__ inline void CopyInGamma()
    {
        // 搬入inputGamma,cast,再将每个数BroadCast到一个Block
        DataCopyExtParams copyParams;
        DataCopyPadExtParams<Tgamma> padParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = col * sizeof(Tgamma);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.paddingValue = 0;
        uint64_t localOffset = (col + GAMMA_NUM_PER_BLOCK - 1) / GAMMA_NUM_PER_BLOCK * GAMMA_NUM_PER_BLOCK;
        padParams.rightPadding = localOffset - col;
        LocalTensor<Tgamma> gammaLocal = inQueueGamma.AllocTensor<Tgamma>();
        LocalTensor<float> gammaFp32;
        if constexpr (std::is_same<Tgamma, float>::value) {
            localOffset = 0;
        }
        DataCopyPad(gammaLocal[localOffset], gammaGm, copyParams, padParams);
        inQueueGamma.EnQue(gammaLocal);
        inQueueGamma.DeQue<Tgamma>();
        if constexpr (std::is_same<Tgamma, float>::value) {
            gammaFp32 = gammaLocal;
        } else {
            gammaFp32 = gammaLocal.template ReinterpretCast<float>();
            Cast(gammaFp32, gammaLocal[localOffset], RoundMode::CAST_NONE, col);
            PipeBarrier<PIPE_V>();
        }
        gammaBrc = tmpBuf1.Get<float>();
        Brcb(gammaBrc, gammaFp32, (col + BLOCK_NUM_PER_REP - 1) / BLOCK_NUM_PER_REP, {1, BLOCK_NUM_PER_REP});
        inQueueGamma.FreeTensor(gammaFp32);
    }

    template <typename T_COPY>
    __aicore__ inline void CopyInPad(
        LocalTensor<T_COPY> &dstTensor, GlobalTensor<T_COPY> inGm, uint64_t ubLoopGmOffset, uint32_t lastAxisSize)
    {
        /*
        支持x，dy，mean，rstd的补pad搬出，抽象为二维的数据；
        lastAxisSize: 搬入数据的尾轴大小
        */
        uint64_t ubOffset = 0;
        uint64_t gmOffset = 0;
        // 每行数据对齐后的size
        uint32_t formerLineAlignSize = 0;
        uint32_t tailLineAlignSize = 0;
        if constexpr (std::is_same<T_COPY, float>::value) {
            formerLineAlignSize =
                (bFormerFactor * lastAxisSize + B32_BLOCK_ALIGN_NUM - 1) / B32_BLOCK_ALIGN_NUM * B32_BLOCK_ALIGN_NUM;
            tailLineAlignSize =
                (bTailFactor * lastAxisSize + B32_BLOCK_ALIGN_NUM - 1) / B32_BLOCK_ALIGN_NUM * B32_BLOCK_ALIGN_NUM;
        } else {
            formerLineAlignSize =
                (bFormerFactor * lastAxisSize + B16_BLOCK_ALIGN_NUM - 1) / B16_BLOCK_ALIGN_NUM * B16_BLOCK_ALIGN_NUM;
            tailLineAlignSize =
                (bTailFactor * lastAxisSize + B16_BLOCK_ALIGN_NUM - 1) / B16_BLOCK_ALIGN_NUM * B16_BLOCK_ALIGN_NUM;
        }
        DataCopyPadExtParams<T_COPY> padParams;
        DataCopyExtParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = lastAxisSize * bFormerFactor * sizeof(T_COPY);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = formerLineAlignSize - lastAxisSize * bFormerFactor;
        padParams.paddingValue = 0;
        for (uint32_t i = 0; i < TRANSPOSE_C0_SIZE; i++) {
            if (i < formerLoops) {
                DataCopyPad(dstTensor[ubOffset], inGm[ubLoopGmOffset + gmOffset], intriParams, padParams);
                gmOffset += bFormerFactor * lastAxisSize;
                ubOffset += formerLineAlignSize;
            } else {
                if (bTailFactor > 0) {
                    intriParams.blockLen = lastAxisSize * bTailFactor * sizeof(T_COPY);
                    padParams.rightPadding = tailLineAlignSize - lastAxisSize * bTailFactor;
                    DataCopyPad(dstTensor[ubOffset], inGm[ubLoopGmOffset + gmOffset], intriParams, padParams);
                    gmOffset += bTailFactor * lastAxisSize;
                    ubOffset += formerLineAlignSize;
                }
                if ((formerLineAlignSize - tailLineAlignSize) > 0) {
                    Duplicate<T_COPY>(dstTensor[i * formerLineAlignSize + tailLineAlignSize],
                        0.0,
                        (formerLineAlignSize - tailLineAlignSize));
                }
            }
        }
    }

    template <typename T_TRANS>
    __aicore__ inline void DoTranspose(
        LocalTensor<T_TRANS> &dstTensor, LocalTensor<T_TRANS> &srcTensor, uint32_t lastAxisSize)
    {
        /*
        tiling限制repeat不大于255
        支持float16,float32
        */
        // 每行数据对齐后的size
        uint32_t lineAlignSize = 0;
        if constexpr (std::is_same<T_TRANS, float>::value) {
            lineAlignSize =
                (bFormerFactor * lastAxisSize + B32_BLOCK_ALIGN_NUM - 1) / B32_BLOCK_ALIGN_NUM * B32_BLOCK_ALIGN_NUM;
        } else {
            lineAlignSize =
                (bFormerFactor * lastAxisSize + B16_BLOCK_ALIGN_NUM - 1) / B16_BLOCK_ALIGN_NUM * B16_BLOCK_ALIGN_NUM;
        }
        __ubuf__ T_TRANS *srcAddr = (__ubuf__ T_TRANS *)srcTensor.GetPhyAddr();
        __ubuf__ T_TRANS *dstAddr = (__ubuf__ T_TRANS *)dstTensor.GetPhyAddr();
        __ubuf__ T_TRANS *srcLocalList[TRANSPOSE_C0_SIZE];
        __ubuf__ T_TRANS *dstLocalList[TRANSPOSE_C0_SIZE];
        for (uint32_t i = 0; i < TRANSPOSE_C0_SIZE; i++) {
            srcLocalList[i] = srcAddr + lineAlignSize * i;
            if constexpr (std::is_same<T_TRANS, float>::value) {
                dstLocalList[i] = dstAddr + B32_BLOCK_ALIGN_NUM * i;
            } else {
                dstLocalList[i] = dstAddr + B16_BLOCK_ALIGN_NUM * i;
            }
        }
        struct TransDataTo5HDParams transDataParams;
        if constexpr (std::is_same<T_TRANS, float>::value) {
            transDataParams.repeatTimes = lineAlignSize / B32_BLOCK_ALIGN_NUM;
        } else {
            transDataParams.repeatTimes = lineAlignSize / B16_BLOCK_ALIGN_NUM;
        }
        transDataParams.srcRepStride = 1;
        transDataParams.dstRepStride = TRANSPOSE_C0_SIZE;
        if (transDataParams.repeatTimes == 1) {
            transDataParams.srcRepStride = 0;
            transDataParams.dstRepStride = 0;
        }
        TransDataTo5HDImpl(dstLocalList, srcLocalList, transDataParams);
    }

    template <typename T_RESHAPE>
    __aicore__ inline void DoReshape(
        LocalTensor<T_RESHAPE> &dstTensor, LocalTensor<T_RESHAPE> &srcTensor, uint32_t lastAxisSize)
    {
        /*
        支持fp32，fp16
        */
        // 一个repeat处理（128 / IN_NUM_PER_BLOCK）行数据
        uint32_t repeatTimes = lastAxisSize / BLOCK_NUM_PER_REP;
        uint32_t remainRepeat = lastAxisSize % BLOCK_NUM_PER_REP;
        uint32_t mask = 0;
        uint32_t lineBlockNum = 0;
        if constexpr (std::is_same<T_RESHAPE, float>::value) {
            mask = B32_BLOCK_ALIGN_NUM * BLOCK_NUM_PER_REP;
            lineBlockNum = TRANSPOSE_C0_SIZE / B32_BLOCK_ALIGN_NUM;
        } else {
            mask = B16_BLOCK_ALIGN_NUM * BLOCK_NUM_PER_REP;
            lineBlockNum = TRANSPOSE_C0_SIZE / B16_BLOCK_ALIGN_NUM;
        }
        if ((bFormerFactor * BLOCK_NUM_PER_REP * lineBlockNum) < MAX_REP_NUM) {
            if (repeatTimes) {
                for (uint32_t i = 0; i < bFormerFactor; i++) {
                    Copy(dstTensor[i * TRANSPOSE_C0_SIZE],
                        srcTensor[i * TRANSPOSE_C0_SIZE * lastAxisSize],
                        mask,
                        repeatTimes,
                        {(uint16_t)(bFormerFactor * lineBlockNum),
                            (uint16_t)lineBlockNum,
                            (uint8_t)(BLOCK_NUM_PER_REP * bFormerFactor * lineBlockNum),
                            (uint8_t)(BLOCK_NUM_PER_REP * lineBlockNum)});
                    if constexpr (std::is_same<T_RESHAPE, float>::value) {
                        Copy(dstTensor[i * TRANSPOSE_C0_SIZE + B32_BLOCK_ALIGN_NUM],
                            srcTensor[i * TRANSPOSE_C0_SIZE * lastAxisSize + B32_BLOCK_ALIGN_NUM],
                            mask,
                            repeatTimes,
                            {(uint16_t)(bFormerFactor * lineBlockNum),
                                (uint16_t)lineBlockNum,
                                (uint8_t)(BLOCK_NUM_PER_REP * bFormerFactor * lineBlockNum),
                                (uint8_t)(BLOCK_NUM_PER_REP * lineBlockNum)});
                    }
                }
            }
            if (remainRepeat) {
                if constexpr (std::is_same<T_RESHAPE, float>::value) {
                    mask = remainRepeat * B32_BLOCK_ALIGN_NUM;
                } else {
                    mask = remainRepeat * B16_BLOCK_ALIGN_NUM;
                }
                for (uint32_t i = 0; i < bFormerFactor; i++) {
                    Copy(dstTensor[i * TRANSPOSE_C0_SIZE +
                                   repeatTimes * TRANSPOSE_C0_SIZE * BLOCK_NUM_PER_REP * bFormerFactor],
                        srcTensor[i * TRANSPOSE_C0_SIZE * lastAxisSize +
                                  repeatTimes * TRANSPOSE_C0_SIZE * BLOCK_NUM_PER_REP],
                        mask,
                        1,
                        {(uint16_t)(bFormerFactor * lineBlockNum), (uint16_t)lineBlockNum, 0, 0});
                    if constexpr (std::is_same<T_RESHAPE, float>::value) {
                        Copy(dstTensor[i * TRANSPOSE_C0_SIZE + B32_BLOCK_ALIGN_NUM +
                                       repeatTimes * TRANSPOSE_C0_SIZE * BLOCK_NUM_PER_REP * bFormerFactor],
                            srcTensor[i * TRANSPOSE_C0_SIZE * lastAxisSize + B32_BLOCK_ALIGN_NUM +
                                      repeatTimes * TRANSPOSE_C0_SIZE * BLOCK_NUM_PER_REP],
                            mask,
                            1,
                            {(uint16_t)(bFormerFactor * lineBlockNum), (uint16_t)lineBlockNum, 0, 0});
                    }
                }
            }
        } else {
            DataCopyParams copyParams;
            copyParams.blockCount = bFormerFactor;
            copyParams.blockLen = lineBlockNum;
            copyParams.srcStride = (lastAxisSize - 1) * copyParams.blockLen;
            copyParams.dstStride = 0;
            for (uint32_t i = 0; i < lastAxisSize; i++) {
                DataCopy(
                    dstTensor[i * bFormerFactor * TRANSPOSE_C0_SIZE], srcTensor[i * TRANSPOSE_C0_SIZE], copyParams);
            }
        }
    }

    __aicore__ inline void DoReduce(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor)
    {
        /*
        srcTensor为reduce之前的Tensor: col * colLineElements
        dstTensor为存放reduce结果的tensor: colLineElements
        */
        uint64_t nowColSize = col;
        if (nowColSize == 1) {
            Adds<float>(dstTensor, srcTensor, 0, colLineElements);
            PipeBarrier<PIPE_V>();
            return;
        }
        // col为非二次幂，先将二次幂差值行加到前面
        if (dichotomizeAddDiffSize != 0) {
            Add(srcTensor,
                srcTensor,
                srcTensor[(nowColSize - dichotomizeAddDiffSize) * colLineElements],
                dichotomizeAddDiffSize * colLineElements);
            PipeBarrier<PIPE_V>();
            nowColSize = nowColSize - dichotomizeAddDiffSize;
        }
        while (nowColSize > 1) {
            nowColSize = nowColSize / TWO_NUM;
            if (nowColSize == 1) {
                Add(dstTensor, srcTensor, srcTensor[colLineElements], colLineElements);
            } else {
                Add(srcTensor, srcTensor, srcTensor[nowColSize * colLineElements], nowColSize * colLineElements);
            }
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void DoLastAxisReduce(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor)
    {
        /*
        srcTensor为reduce之前的Tensor: col * colLineElements
        dstTensor为待累加当前reduce结果的tensor: col
        */
        LocalTensor<float> TempReduceResult = tmpBuf0.Get<float>();
        AscendCUtils::SetMaskCount<float>();
        SetVectorMask<float>(0, colLineElements);
        for (int64_t i = 0; i < col; i++) {
            ReduceSum(srcTensor[i * colLineElements], srcTensor[i * colLineElements],
                srcTensor[i * colLineElements], COUNT_NUM);
            PipeBarrier<PIPE_V>();
            event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            (*(__ubuf__ int32_t *)((__ubuf__ float *)TempReduceResult[i].GetPhyAddr())) = GetAccVal();
        }
        AscendCUtils::SetMaskNorm<float>();
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        Add(dstTensor, dstTensor, TempReduceResult, col);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DoInlineBrcSub(
        LocalTensor<float> &dstTensor, LocalTensor<float> &src0Tensor, LocalTensor<float> &src1Tensor)
    {
        /*
        src0，大小为col * colLineElements
        src1大小为 colLineElements
        做inline broadcast的sub计算
        */
        if (((colLineElements / ELEM_PER_REP_FP32) < col) && (colLineElements < (MAX_REP_NUM * BLOCK_NUM_PER_REP))) {
            for (uint32_t i = 0; i < (colLineElements / ELEM_PER_REP_FP32); i++) {
                Sub(dstTensor[i * ELEM_PER_REP_FP32],
                    src0Tensor[i * ELEM_PER_REP_FP32],
                    src1Tensor[i * ELEM_PER_REP_FP32],
                    ELEM_PER_REP_FP32,
                    col,
                    {1,
                        1,
                        1,
                        (uint8_t)(colLineElements / BLOCK_NUM_PER_REP),
                        (uint8_t)(colLineElements / BLOCK_NUM_PER_REP),
                        0});
            }
            if (colLineElements % ELEM_PER_REP_FP32 > 0) {
                Sub(dstTensor[colLineElements / ELEM_PER_REP_FP32 * ELEM_PER_REP_FP32],
                    src0Tensor[colLineElements / ELEM_PER_REP_FP32 * ELEM_PER_REP_FP32],
                    src1Tensor[colLineElements / ELEM_PER_REP_FP32 * ELEM_PER_REP_FP32],
                    colLineElements % ELEM_PER_REP_FP32,
                    col,
                    {1,
                        1,
                        1,
                        (uint8_t)(colLineElements / BLOCK_NUM_PER_REP),
                        (uint8_t)(colLineElements / BLOCK_NUM_PER_REP),
                        0});
            }
        } else {
            for (uint64_t i = 0; i < col; i++) {
                Sub(dstTensor[i * colLineElements], src0Tensor[i * colLineElements], src1Tensor, colLineElements);
            }
        }
    }

    __aicore__ inline void DoInlineBrcMul(
        LocalTensor<float> &dstTensor, LocalTensor<float> &src0Tensor, LocalTensor<float> &src1Tensor)
    {
        /*
        src0，大小为col * colLineElements
        src1大小为 colLineElements
        做inline broadcast的mul计算
        */
        if (((colLineElements / ELEM_PER_REP_FP32) < col) && (colLineElements < (MAX_REP_NUM * BLOCK_NUM_PER_REP))) {
            for (uint32_t i = 0; i < (colLineElements / ELEM_PER_REP_FP32); i++) {
                Mul(dstTensor[i * ELEM_PER_REP_FP32],
                    src0Tensor[i * ELEM_PER_REP_FP32],
                    src1Tensor[i * ELEM_PER_REP_FP32],
                    ELEM_PER_REP_FP32,
                    col,
                    {1,
                        1,
                        1,
                        (uint8_t)(colLineElements / BLOCK_NUM_PER_REP),
                        (uint8_t)(colLineElements / BLOCK_NUM_PER_REP),
                        0});
            }
            if (colLineElements % ELEM_PER_REP_FP32 > 0) {
                Mul(dstTensor[colLineElements / ELEM_PER_REP_FP32 * ELEM_PER_REP_FP32],
                    src0Tensor[colLineElements / ELEM_PER_REP_FP32 * ELEM_PER_REP_FP32],
                    src1Tensor[colLineElements / ELEM_PER_REP_FP32 * ELEM_PER_REP_FP32],
                    colLineElements % ELEM_PER_REP_FP32,
                    col,
                    {1,
                        1,
                        1,
                        (uint8_t)(colLineElements / BLOCK_NUM_PER_REP),
                        (uint8_t)(colLineElements / BLOCK_NUM_PER_REP),
                        0});
            }
        } else {
            for (uint64_t i = 0; i < col; i++) {
                Mul(dstTensor[i * colLineElements], src0Tensor[i * colLineElements], src1Tensor, colLineElements);
            }
        }
    }

    __aicore__ inline void DoMulGamma(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor)
    {
        uint32_t repeatTimes = rowLength / ELEM_PER_REP_FP32;
        uint32_t remainRepeat = rowLength % ELEM_PER_REP_FP32;
        uint32_t mask = ELEM_PER_REP_FP32;
        for (uint64_t i = 0; i < col; i++) {
            Mul(dstTensor[i * colLineElements],
                srcTensor[i * colLineElements],
                gammaBrc[i * B32_BLOCK_ALIGN_NUM],
                mask,
                repeatTimes,
                {1, 1, 0, 8, 8, 0});
        }
        if (remainRepeat) {
            mask = remainRepeat;
            for (uint64_t i = 0; i < col; i++) {
                Mul(dstTensor[i * colLineElements + ELEM_PER_REP_FP32 * repeatTimes],
                    srcTensor[i * colLineElements + ELEM_PER_REP_FP32 * repeatTimes],
                    gammaBrc[i * B32_BLOCK_ALIGN_NUM],
                    mask,
                    1,
                    {1, 1, 0, 0, 0, 0});
            }
        }
    }

    template <typename T_POST_RESHAPE>
    __aicore__ inline void DoPostReshape(LocalTensor<T_POST_RESHAPE> &dstTensor, LocalTensor<T_POST_RESHAPE> &srcTensor)
    {
        /*
        支持fp32，fp16
        */
        // 一个repeat处理（128 / IN_NUM_PER_BLOCK）行数据
        uint32_t repeatTimes = col / BLOCK_NUM_PER_REP;
        uint32_t remainRepeat = col % BLOCK_NUM_PER_REP;
        uint32_t mask = 0;
        uint32_t lineBlockNum = 0;
        if constexpr (std::is_same<T_POST_RESHAPE, float>::value) {
            mask = B32_BLOCK_ALIGN_NUM * BLOCK_NUM_PER_REP;
            lineBlockNum = TRANSPOSE_C0_SIZE / B32_BLOCK_ALIGN_NUM;
        } else {
            mask = B16_BLOCK_ALIGN_NUM * BLOCK_NUM_PER_REP;
            lineBlockNum = TRANSPOSE_C0_SIZE / B16_BLOCK_ALIGN_NUM;
        }
        if ((bFormerFactor * BLOCK_NUM_PER_REP * lineBlockNum) < MAX_REP_NUM) {
            if (repeatTimes) {
                for (uint32_t i = 0; i < bFormerFactor; i++) {
                    Copy(dstTensor[i * TRANSPOSE_C0_SIZE * col],
                        srcTensor[i * TRANSPOSE_C0_SIZE],
                        mask,
                        repeatTimes,
                        {(uint16_t)lineBlockNum,
                            (uint16_t)(bFormerFactor * lineBlockNum),
                            (uint8_t)(BLOCK_NUM_PER_REP * lineBlockNum),
                            (uint8_t)(BLOCK_NUM_PER_REP * bFormerFactor * lineBlockNum)});
                    if constexpr (std::is_same<T_POST_RESHAPE, float>::value) {
                        Copy(dstTensor[i * TRANSPOSE_C0_SIZE * col + B32_BLOCK_ALIGN_NUM],
                            srcTensor[i * TRANSPOSE_C0_SIZE + B32_BLOCK_ALIGN_NUM],
                            mask,
                            repeatTimes,
                            {(uint16_t)lineBlockNum,
                                (uint16_t)(bFormerFactor * lineBlockNum),
                                (uint8_t)(BLOCK_NUM_PER_REP * lineBlockNum),
                                (uint8_t)(BLOCK_NUM_PER_REP * bFormerFactor * lineBlockNum)});
                    }
                }
            }
            if (remainRepeat) {
                if constexpr (std::is_same<T_POST_RESHAPE, float>::value) {
                    mask = remainRepeat * B32_BLOCK_ALIGN_NUM;
                } else {
                    mask = remainRepeat * B16_BLOCK_ALIGN_NUM;
                }
                for (uint32_t i = 0; i < bFormerFactor; i++) {
                    Copy(dstTensor[i * TRANSPOSE_C0_SIZE * col + repeatTimes * TRANSPOSE_C0_SIZE * BLOCK_NUM_PER_REP],
                        srcTensor[i * TRANSPOSE_C0_SIZE +
                                  repeatTimes * TRANSPOSE_C0_SIZE * BLOCK_NUM_PER_REP * bFormerFactor],
                        mask,
                        1,
                        {(uint16_t)lineBlockNum, (uint16_t)(bFormerFactor * lineBlockNum), 0, 0});
                    if constexpr (std::is_same<T_POST_RESHAPE, float>::value) {
                        Copy(dstTensor[i * TRANSPOSE_C0_SIZE * col + B32_BLOCK_ALIGN_NUM +
                                       repeatTimes * TRANSPOSE_C0_SIZE * BLOCK_NUM_PER_REP],
                            srcTensor[i * TRANSPOSE_C0_SIZE + B32_BLOCK_ALIGN_NUM +
                                      repeatTimes * TRANSPOSE_C0_SIZE * BLOCK_NUM_PER_REP * bFormerFactor],
                            mask,
                            1,
                            {(uint16_t)lineBlockNum, (uint16_t)(bFormerFactor * lineBlockNum), 0, 0});
                    }
                }
            }
        } else {
            DataCopyParams copyParams;
            copyParams.blockCount = bFormerFactor;
            copyParams.blockLen = lineBlockNum;
            copyParams.srcStride = 0;
            copyParams.dstStride = (col - 1) * copyParams.blockLen;
            for (uint32_t i = 0; i < col; i++) {
                DataCopy(
                    dstTensor[i * TRANSPOSE_C0_SIZE], srcTensor[i * bFormerFactor * TRANSPOSE_C0_SIZE], copyParams);
            }
        }
    }

    template <typename T_POST_TRANS>
    __aicore__ inline void DoPostTranspose(LocalTensor<T_POST_TRANS> &dstTensor, LocalTensor<T_POST_TRANS> &srcTensor)
    {
        /*
        tiling限制repeat不大于255
        支持float16,float32
        反向的转置过程，会使行对齐为16的倍数
        */
        // 每行数据对齐后的size
        uint32_t lineAlignSize = 0;
        lineAlignSize = (bFormerFactor * col + TRANSPOSE_C0_SIZE - 1) / TRANSPOSE_C0_SIZE * TRANSPOSE_C0_SIZE;
        __ubuf__ T_POST_TRANS *srcAddr = (__ubuf__ T_POST_TRANS *)srcTensor.GetPhyAddr();
        __ubuf__ T_POST_TRANS *dstAddr = (__ubuf__ T_POST_TRANS *)dstTensor.GetPhyAddr();
        __ubuf__ T_POST_TRANS *srcLocalList[TRANSPOSE_C0_SIZE];
        __ubuf__ T_POST_TRANS *dstLocalList[TRANSPOSE_C0_SIZE];
        for (uint32_t i = 0; i < TRANSPOSE_C0_SIZE; i++) {
            srcLocalList[i] = srcAddr + TRANSPOSE_C0_SIZE * i;
            if constexpr (std::is_same<T_POST_TRANS, float>::value) {
                dstLocalList[i] = dstAddr + lineAlignSize * (i / TWO_NUM) + B32_BLOCK_ALIGN_NUM * (i % TWO_NUM);
            } else {
                dstLocalList[i] = dstAddr + lineAlignSize * i;
            }
        }
        struct TransDataTo5HDParams transDataParams;
        if constexpr (std::is_same<T_POST_TRANS, float>::value) {
            transDataParams.repeatTimes = lineAlignSize / B32_BLOCK_ALIGN_NUM / TWO_NUM;
            transDataParams.srcRepStride = TRANSPOSE_C0_SIZE * TWO_NUM;
            transDataParams.dstRepStride = TWO_NUM;
        } else {
            transDataParams.repeatTimes = lineAlignSize / B16_BLOCK_ALIGN_NUM;
            transDataParams.srcRepStride = TRANSPOSE_C0_SIZE;
            transDataParams.dstRepStride = 1;
        }
        if (transDataParams.repeatTimes == 1) {
            transDataParams.srcRepStride = 0;
            transDataParams.dstRepStride = 0;
        }
        TransDataTo5HDImpl(dstLocalList, srcLocalList, transDataParams);
        if constexpr (std::is_same<T_POST_TRANS, float>::value) {
            for (uint32_t i = 0; i < TRANSPOSE_C0_SIZE; i++) {
                srcLocalList[i] = srcAddr + TRANSPOSE_C0_SIZE * i + B32_BLOCK_ALIGN_NUM;
                dstLocalList[i] = dstAddr + B32_BLOCK_ALIGN_NUM * lineAlignSize + lineAlignSize * (i / TWO_NUM) +
                                  B32_BLOCK_ALIGN_NUM * (i % TWO_NUM);
            }
            TransDataTo5HDImpl(dstLocalList, srcLocalList, transDataParams);
        }
    }

    __aicore__ inline void CopyOutPdX(LocalTensor<Tdy> &srcTensor)
    {
        uint32_t curFactor = bFormerFactor;
        uint64_t ubOffset = 0;
        uint64_t gmOffset = 0;
        // 每行数据对齐后的size
        uint32_t lineAlignSize = 0;
        lineAlignSize = (bFormerFactor * col + TRANSPOSE_C0_SIZE - 1) / TRANSPOSE_C0_SIZE * TRANSPOSE_C0_SIZE;

        DataCopyExtParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = col * bFormerFactor * sizeof(Tdy);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;

        for (uint32_t i = 0; i < TRANSPOSE_C0_SIZE; i++) {
            if (i < formerLoops) {
                curFactor = bFormerFactor;
            } else {
                if (bTailFactor > 0) {
                    intriParams.blockLen = col * bTailFactor * sizeof(Tdy);
                    curFactor = bTailFactor;
                } else {
                    break;
                }
            }
            DataCopyPad(pdXGm[xGmOffset + gmOffset], srcTensor[ubOffset], intriParams);
            // gm偏移连续，每次偏curFactor * col
            gmOffset += curFactor * col;
            // ub偏移对齐，每次固定lineAlignSize
            ubOffset += lineAlignSize;
        }
    }

    __aicore__ inline void ProcessBasicBlock()
    {
        LocalTensor<Tdy> dyLocal = inQueueDy.AllocTensor<Tdy>();
        CopyInPad<Tdy>(dyLocal, dyGm, xGmOffset, col);
        inQueueDy.EnQue(dyLocal);
        inQueueDy.DeQue<Tdy>();

        LocalTensor<Tdy> transposeTemp = tmpBuf0.Get<Tdy>();
        if constexpr (std::is_same<Tdy, bfloat16_t>::value) {
            LocalTensor<half> transposeTempHalf = transposeTemp.template ReinterpretCast<half>();
            LocalTensor<half> dyLocalHalf = dyLocal.template ReinterpretCast<half>();
            DoTranspose<half>(transposeTempHalf, dyLocalHalf, col);
        } else {
            DoTranspose<Tdy>(transposeTemp, dyLocal, col);
        }

        PipeBarrier<PIPE_V>();

        if constexpr (std::is_same<Tdy, float>::value) {
            DoReshape<Tdy>(dyLocal, transposeTemp, col);
            PipeBarrier<PIPE_V>();
            dyLocalFp32 = dyLocal;
        } else {
            LocalTensor<Tdy> dyLocalSecond = dyLocal[calcXElements];
            if constexpr (std::is_same<Tdy, bfloat16_t>::value) {
                LocalTensor<half> dyLocalSecondRHalf = dyLocalSecond.template ReinterpretCast<half>();
                LocalTensor<half> transposeTempRHalf = transposeTemp.template ReinterpretCast<half>();
                DoReshape<half>(dyLocalSecondRHalf, transposeTempRHalf, col);
            } else {
                DoReshape<Tdy>(dyLocalSecond, transposeTemp, col);
            }
            PipeBarrier<PIPE_V>();
            dyLocalFp32 = dyLocal.template ReinterpretCast<float>();
            Cast(dyLocalFp32, dyLocalSecond, RoundMode::CAST_NONE, calcXElements);
            PipeBarrier<PIPE_V>();
        }

        DoLastAxisReduce(pdBetaOut, dyLocalFp32);

        LocalTensor<Tdy> xLocal = inQueueX.AllocTensor<Tdy>();
        CopyInPad<Tdy>(xLocal, xGm, xGmOffset, col);
        inQueueX.EnQue(xLocal);
        inQueueX.DeQue<Tdy>();

        LocalTensor<Tdy> transposeTemp1 = tmpBuf0.Get<Tdy>();
        if constexpr (std::is_same<Tdy, bfloat16_t>::value) {
            LocalTensor<half> transposeTemp1Half = transposeTemp1.template ReinterpretCast<half>();
            LocalTensor<half> xLocalHalf = xLocal.template ReinterpretCast<half>();
            DoTranspose<half>(transposeTemp1Half, xLocalHalf, col);
        } else {
            DoTranspose<Tdy>(transposeTemp1, xLocal, col);
        }

        PipeBarrier<PIPE_V>();

        if constexpr (std::is_same<Tdy, float>::value) {
            DoReshape<Tdy>(xLocal, transposeTemp1, col);
            PipeBarrier<PIPE_V>();
            xLocalFp32 = xLocal;
        } else {
            LocalTensor<Tdy> xLocalSecond = xLocal[calcXElements];
            if constexpr (std::is_same<Tdy, bfloat16_t>::value) {
                LocalTensor<half> xLocalSecondRHalf = xLocalSecond.template ReinterpretCast<half>();
                LocalTensor<half> transposeTemp1RHalf = transposeTemp1.template ReinterpretCast<half>();
                DoReshape<half>(xLocalSecondRHalf, transposeTemp1RHalf, col);
            } else {
                DoReshape<Tdy>(xLocalSecond, transposeTemp1, col);
            }
            PipeBarrier<PIPE_V>();
            xLocalFp32 = xLocal.template ReinterpretCast<float>();
            Cast(xLocalFp32, xLocalSecond, RoundMode::CAST_NONE, calcXElements);
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<float> rstdLocal = inQueueRstd.AllocTensor<float>();
        CopyInPad<float>(rstdLocal, rstdGm, meanGmOffset, 1);
        inQueueRstd.EnQue(rstdLocal);
        inQueueRstd.DeQue<float>();

        LocalTensor<float> transposeTemp2 = tmpBuf0.Get<float>();
        DoTranspose<float>(transposeTemp2, rstdLocal, 1);
        PipeBarrier<PIPE_V>();
        Adds<float>(rstdLocal, transposeTemp2, 0, rowLength);

        LocalTensor<float> meanLocal = inQueueMean.AllocTensor<float>();
        CopyInPad<float>(meanLocal, meanGm, meanGmOffset, 1);
        inQueueMean.EnQue(meanLocal);
        inQueueMean.DeQue<float>();

        LocalTensor<float> transposeTemp3 = tmpBuf0.Get<float>();
        DoTranspose<float>(transposeTemp3, meanLocal, 1);
        PipeBarrier<PIPE_V>();
        Adds<float>(meanLocal, transposeTemp3, 0, rowLength);
        PipeBarrier<PIPE_V>();

        DoInlineBrcSub(xLocalFp32, xLocalFp32, meanLocal);
        inQueueMean.FreeTensor(meanLocal);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> outPdxLocal = outQueuePdX.AllocTensor<float>();
        DoInlineBrcMul(outPdxLocal, xLocalFp32, rstdLocal);
        PipeBarrier<PIPE_V>();

        Mul(xLocalFp32, outPdxLocal, dyLocalFp32, calcXElements);
        PipeBarrier<PIPE_V>();

        DoLastAxisReduce(pdGammaOut, xLocalFp32);
        inQueueX.FreeTensor(xLocalFp32);

        LocalTensor<float> mulGammaTensor = tmpBuf0.Get<float>();
        DoMulGamma(mulGammaTensor, dyLocalFp32);
        PipeBarrier<PIPE_V>();

        Mul(dyLocalFp32, mulGammaTensor, outPdxLocal, calcXElements);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> reduce3Tensor = tmpBuf2.Get<float>();
        DoReduce(reduce3Tensor, dyLocalFp32);
        inQueueDy.FreeTensor(dyLocalFp32);
        PipeBarrier<PIPE_V>();

        Muls(reduce3Tensor, reduce3Tensor, coff, colLineElements);
        PipeBarrier<PIPE_V>();

        DoInlineBrcMul(outPdxLocal, outPdxLocal, reduce3Tensor);
        PipeBarrier<PIPE_V>();

        Sub(outPdxLocal, mulGammaTensor, outPdxLocal, calcXElements);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> reduce2Tensor = tmpBuf2.Get<float>();
        DoReduce(reduce2Tensor, mulGammaTensor);
        PipeBarrier<PIPE_V>();

        Muls(reduce2Tensor, reduce2Tensor, coff, colLineElements);
        PipeBarrier<PIPE_V>();

        DoInlineBrcSub(outPdxLocal, outPdxLocal, reduce2Tensor);
        PipeBarrier<PIPE_V>();

        DoInlineBrcMul(outPdxLocal, outPdxLocal, rstdLocal);
        inQueueRstd.FreeTensor(rstdLocal);
        PipeBarrier<PIPE_V>();

        LocalTensor<Tdy> outPdxTdy;
        if constexpr (!IsSameType<Tdy, float>::value) {
            RoundMode b16RoundMode = IsSameType<Tdy, bfloat16_t>::value ? RoundMode::CAST_ROUND : RoundMode::CAST_NONE;
            outPdxTdy = outPdxLocal.template ReinterpretCast<Tdy>();
            Cast(outPdxTdy, outPdxLocal, b16RoundMode, calcXElements);
            PipeBarrier<PIPE_V>();
        } else {
            outPdxTdy = outPdxLocal;
        }

        LocalTensor<Tdy> reshapeTemp = tmpBuf0.Get<Tdy>();
        if constexpr (std::is_same<Tdy, bfloat16_t>::value) {
            LocalTensor<half> reshapeTempRHalf = reshapeTemp.template ReinterpretCast<half>();
            LocalTensor<half> outPdxTdyRHalf = outPdxTdy.template ReinterpretCast<half>();
            DoPostReshape<half>(reshapeTempRHalf, outPdxTdyRHalf);
        } else {
            DoPostReshape<Tdy>(reshapeTemp, outPdxTdy);
        }
        PipeBarrier<PIPE_V>();

        if constexpr (std::is_same<Tdy, bfloat16_t>::value) {
            LocalTensor<half> outPdxTdyHalf = outPdxTdy.template ReinterpretCast<half>();
            LocalTensor<half> reshapeTempHalf = reshapeTemp.template ReinterpretCast<half>();
            DoPostTranspose<half>(outPdxTdyHalf, reshapeTempHalf);
        } else {
            DoPostTranspose<Tdy>(outPdxTdy, reshapeTemp);
        }
        PipeBarrier<PIPE_V>();

        outQueuePdX.EnQue(outPdxTdy);
        outQueuePdX.DeQue<Tdy>();
        CopyOutPdX(outPdxTdy);
        outQueuePdX.FreeTensor(outPdxTdy);
    }

private:
    constexpr static uint32_t BLOCK = 32;
    constexpr static uint32_t DY_NUM_PER_BLOCK = BLOCK / sizeof(Tdy);
    constexpr static uint32_t MEAN_NUM_PER_BLOCK = BLOCK / sizeof(float);
    constexpr static uint32_t GAMMA_NUM_PER_BLOCK = BLOCK / sizeof(Tgamma);
    constexpr static uint32_t TRANSPOSE_COL_LIMIT = 64;
    constexpr static uint32_t FLOAT_SIZE = 4;
    constexpr static uint32_t TRANSPOSE_C0_SIZE = 16;
    constexpr static uint32_t MAX_REP_NUM = 255;
    constexpr static uint32_t ELEM_PER_REP_FP32 = 64;
    constexpr static uint32_t BLOCK_NUM_PER_REP = 8;
    constexpr static uint32_t B32_BLOCK_ALIGN_NUM = 8;
    constexpr static uint32_t B16_BLOCK_ALIGN_NUM = 16;
    constexpr static uint32_t TWO_NUM = 2;
    constexpr static uint16_t SYNC_AIV_ONLY_ALL = 14;
    constexpr static uint32_t COUNT_NUM = 64;

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueDy;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueMean;
    TQue<QuePosition::VECIN, 1> inQueueRstd;
    TQue<QuePosition::VECIN, 1> inQueueGamma;
    TQue<QuePosition::VECOUT, 1> outQueuePdX;
    TQue<QuePosition::VECOUT, 1> outQueuePdGamma;
    TQue<QuePosition::VECOUT, 1> outQueuePdBeta;
    TBuf<TPosition::VECCALC> tmpBuf0;
    TBuf<TPosition::VECCALC> tmpBuf1;
    TBuf<TPosition::VECCALC> tmpBuf2;

    GlobalTensor<Tdy> dyGm;
    GlobalTensor<Tdy> xGm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<float> meanGm;
    GlobalTensor<Tgamma> gammaGm;
    GlobalTensor<Tdy> pdXGm;
    GlobalTensor<float> pdGammaGm;
    GlobalTensor<float> pdBetaGm;
    GlobalTensor<float> pdGammaWsp;
    GlobalTensor<float> pdBetaWsp;

    uint32_t blockIdx = GetBlockIdx();

    // calculate xGm and meanGm offset for ub loop
    uint64_t xGmOffset = 0;
    uint64_t meanGmOffset = 0;
    // in ub loop col size
    uint64_t rowLength = 0;

    // x搬入时整块的借轴因子
    uint32_t bFormerFactor = 0;
    // x搬入时一行整块的对齐长度
    uint32_t rFormerAxisAlign = 0;
    // x搬入时尾块的借轴因子
    uint32_t bTailFactor = 0;
    // x搬入时一行尾块的对齐长度
    uint32_t rTailAxisAlign = 0;
    // 整块搬入的循环次数
    uint32_t formerLoops = 0;
    // 重排后x基本块的元素个数
    uint64_t calcXElements;
    // 重排后x做col Reduce后的元素个数
    uint64_t colLineElements;
    // col以8向上对齐值
    uint32_t colB32Align;

    LocalTensor<float> gammaBrc;
    LocalTensor<float> dyLocalFp32;
    LocalTensor<float> xLocalFp32;
    LocalTensor<float> pdGammaOut;
    LocalTensor<float> pdBetaOut;
    // tilingData
    uint64_t col;
    uint64_t row;
    uint64_t blockDim;
    uint64_t blockFormer;
    uint64_t blockTail;
    uint64_t ubFormer;
    uint64_t bFormer;
    uint64_t dichotomizeAddDiffSize;
    uint64_t ubLoopOfFormerBlock;
    uint64_t ubLoopOfTailBlock;
    uint64_t ubTailOfFormerBlock;
    uint64_t ubTailOfTailBlock;
    float coff;
};

}  // namespace LayerNormGradV3
#endif  // LAYER_NORM_GRAD_V3_TRANSPOSE_H
