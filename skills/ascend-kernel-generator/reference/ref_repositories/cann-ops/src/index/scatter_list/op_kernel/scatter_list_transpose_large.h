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
 * \file scatter_list_transpose_large.h
 * \brief
 */
#ifndef SCATTER_LIST_TRANSPOSE_LARGE_H_
#define SCATTER_LIST_TRANSPOSE_LARGE_H_

#include "kernel_operator.h"
#include "scatter_list_base.h"

namespace ScatterList {
using namespace AscendC;

template <typename T1, typename T2>
class ScatterListTransposeLarge : public ScatterListBase<T1> {
public:
    __aicore__ inline ScatterListTransposeLarge(){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indice, GM_ADDR updates, GM_ADDR mask, GM_ADDR tempOut,
                                GM_ADDR workspace, const ScatterListTilingData* tilingData);
    __aicore__ inline void Process();

private:
    struct copyPreParams {
        int64_t inner_loop_idx = 0;
        int64_t onceNum = 0;
        int64_t onceNumAlign = 0;
        int64_t repeatTimes = 0;
        int64_t eachCoreBatchIdx = 0;
        int64_t dim0Idx = 0;
        int64_t allCoreBatchIdx = 0;
        int64_t copyNum = 0;
        int64_t dstGmOffset = 0;
    };
    __aicore__ inline void CopyIn(const uint64_t &CoreBatchNum);
    __aicore__ inline void CopyOut(const uint64_t &CoreBatchNum);
    __aicore__ inline void CopyOutPre(copyPreParams &preCopyParams, LocalTensor<T1> &updatesUb,
                                      LocalTensor<T2> &indiceUb, LocalTensor<T1> &dataUbSize,
                                      LocalTensor<T1> &transposeUbSize);
    __aicore__ inline void TransposeB2(LocalTensor<T1> &dataUbSize, LocalTensor<T1> &transposeUbSize,
                                       LocalTensor<T1> &updatesUb, copyPreParams &params);
    __aicore__ inline void TransposeB4(LocalTensor<T1> &dataUbSize, LocalTensor<T1> &transposeUbSize,
                                       LocalTensor<T1> &updatesUb, copyPreParams &params);
    __aicore__ inline void TransposeB1(LocalTensor<T1> &dataUbSize, LocalTensor<T1> &transposeUbSize,
                                       LocalTensor<T1> &updatesUb, copyPreParams &params);
    constexpr static int32_t bufferNum = 1;

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> indiceInQueue;
    TQue<QuePosition::VECIN, bufferNum> updatesInQueue;
    TQue<QuePosition::VECIN, bufferNum> maskInQueue;

    GlobalTensor<T1> varGm;
    GlobalTensor<T2> indiceGm;
    GlobalTensor<T1> updatesGm;
    GlobalTensor<uint8_t> maskGm;
    TBuf<QuePosition::VECCALC> dataUb;
    TBuf<QuePosition::VECCALC> transposeUb;

    uint64_t blockIdx = 0;
    GM_ADDR varPtr = nullptr;
    bool maskIsNull = false;
    ScatterListTilingData m_tilingData;
};

template <typename T1, typename T2>
__aicore__ inline void ScatterListTransposeLarge<T1, T2>::Init(GM_ADDR var, GM_ADDR indice, GM_ADDR updates,
                                                               GM_ADDR mask, GM_ADDR tempOut, GM_ADDR workspace,
                                                               const ScatterListTilingData* tilingData) {
    blockIdx = GetBlockIdx();
    this->ParseTilingData(tilingData, m_tilingData);
    varPtr = var;
    indiceGm.SetGlobalBuffer((__gm__ T2*)indice);
    updatesGm.SetGlobalBuffer((__gm__ T1*)updates);;
    if (mask == nullptr) {
        maskIsNull = true;
    } else {
        maskGm.SetGlobalBuffer((__gm__ uint8_t*)mask);
        pipe.InitBuffer(maskInQueue, bufferNum, m_tilingData.maskUbSize);
    }

    pipe.InitBuffer(indiceInQueue, bufferNum, m_tilingData.indiceUbSize);
    pipe.InitBuffer(updatesInQueue, bufferNum, m_tilingData.updatesUbSize);
    pipe.InitBuffer(dataUb, m_tilingData.dataUbSize);
    pipe.InitBuffer(transposeUb,m_tilingData.transposeUbSize);
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListTransposeLarge<T1, T2>::Process() {
    if (blockIdx >= m_tilingData.useCoreNum) {
        return;
    }
    if (blockIdx == m_tilingData.useCoreNum - 1) {
        CopyIn(m_tilingData.lastCoreBatchNum);
        CopyOut(m_tilingData.lastCoreBatchNum);
    } else {
        CopyIn(m_tilingData.preCoreBatchNum);
        CopyOut(m_tilingData.preCoreBatchNum);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListTransposeLarge<T1, T2>::CopyIn(const uint64_t &CoreBatchNum) {
    LocalTensor<T2> indiceUb = indiceInQueue.AllocTensor<T2>();
    LocalTensor<T1> updatesUb = updatesInQueue.AllocTensor<T1>();
    DataCopy(indiceUb, indiceGm, m_tilingData.indiceCount);
    if (!maskIsNull) {
        LocalTensor<uint8_t> maskUb = maskInQueue.AllocTensor<uint8_t>();
        DataCopy(maskUb, maskGm, m_tilingData.maskCount);
        maskInQueue.EnQue(maskUb);
    }

    indiceInQueue.EnQue(indiceUb);
    updatesInQueue.EnQue(updatesUb);
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListTransposeLarge<T1, T2>::TransposeB2(LocalTensor<T1> &dataUbSize,
                                                                      LocalTensor<T1> &transposeUbSize,
                                                                      LocalTensor<T1> &updatesUb,
                                                                      copyPreParams &transParams) {
    TransposeParams params;
    TransposeParams paramsBack;
    params.src = 16;
    params.dst = transParams.copyNum;
    params.src_rep = 16;
    params.dst_rep = 1;
    params.repeat_times = transParams.repeatTimes;
    paramsBack.src = transParams.copyNum;
    paramsBack.dst = 16;
    paramsBack.src_rep = 1;
    paramsBack.dst_rep = 16;
    paramsBack.repeat_times = transParams.repeatTimes;
    if (params.repeat_times == 1) {
        params.src_rep = 0;
        params.dst_rep = 0;
        paramsBack.src_rep = 0;
        paramsBack.dst_rep = 0;
    }
    this->TransposeB16(params, dataUbSize, transposeUbSize);
    DataCopy(transposeUbSize, updatesUb[0], transParams.onceNumAlign);
    this->TransposeB16(paramsBack, transposeUbSize, dataUbSize);
    DataCopyParams copyParams;
    copyParams.blockCount = transParams.onceNum;
    copyParams.blockLen = 1;
    copyParams.srcStride = 0;
    copyParams.dstStride = m_tilingData.varDim3Stride;
    this->VToMte3();
    DataCopy(varGm[transParams.dstGmOffset], dataUbSize, copyParams);
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListTransposeLarge<T1, T2>::TransposeB4(LocalTensor<T1> &dataUbSize,
                                                                      LocalTensor<T1> &transposeUbSize,
                                                                      LocalTensor<T1> &updatesUb,
                                                                      copyPreParams &transParams) {
    TransposeParams params;
    TransposeParams paramsBack;
    params.src = 8;
    params.dst = transParams.copyNum;
    params.repeat_times = transParams.repeatTimes;
    paramsBack.src = transParams.copyNum;
    paramsBack.dst = 8;
    paramsBack.repeat_times = transParams.repeatTimes;
    if (params.repeat_times == 1) {
        params.src_rep = 0;
        params.dst_rep = 0;
        paramsBack.src_rep = 0;
        paramsBack.dst_rep = 0;
    } else {
        params.src_rep = 16;
        params.dst_rep = 2;
        paramsBack.src_rep = 2;
        paramsBack.dst_rep = 16;
    }
    this->TransposeB32(params, dataUbSize, transposeUbSize);
    DataCopy(transposeUbSize, updatesUb[0], transParams.onceNumAlign);
    this->TransposeB32Back(paramsBack, transposeUbSize, dataUbSize);
    DataCopyParams copyParams;
    copyParams.blockCount = transParams.onceNum;
    copyParams.blockLen = 1;
    copyParams.srcStride = 0;
    copyParams.dstStride = m_tilingData.varDim3Stride;
    this->VToMte3();
    DataCopy(varGm[transParams.dstGmOffset], dataUbSize, copyParams);
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListTransposeLarge<T1, T2>::TransposeB1(LocalTensor<T1> &dataUbSize,
                                                                      LocalTensor<T1> &transposeUbSize,
                                                                      LocalTensor<T1> &updatesUb,
                                                                      copyPreParams &transParams) {
    TransposeParams params;
    TransposeParams paramsBack;
    params.src = 32;
    params.dst = transParams.copyNum;
    params.src_rep = 32;
    params.dst_rep = 1;
    params.repeat_times = transParams.repeatTimes;

    paramsBack.src = transParams.copyNum;
    paramsBack.dst = 32;
    paramsBack.src_rep = 1;
    paramsBack.dst_rep = 32;
    paramsBack.repeat_times = transParams.repeatTimes;
    if (params.repeat_times == 1) {
        params.src_rep = 0;
        params.dst_rep = 0;
        paramsBack.src_rep = 0;
        paramsBack.dst_rep = 0;
    }
    this->TransposeB8(params, dataUbSize, transposeUbSize);
    DataCopy(transposeUbSize, updatesUb[0], transParams.onceNumAlign);
    this->TransposeB8(paramsBack, transposeUbSize, dataUbSize);
    DataCopyParams copyParams;
    copyParams.blockCount = transParams.onceNum;
    copyParams.blockLen = 1;
    copyParams.srcStride = 0;
    copyParams.dstStride = m_tilingData.varDim3Stride;
    this->VToMte3();
    DataCopy(varGm[transParams.dstGmOffset], dataUbSize, copyParams);
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListTransposeLarge<T1, T2>::CopyOutPre(copyPreParams &preCopyParams,
                                                                     LocalTensor<T1> &updatesUb,
                                                                     LocalTensor<T2> &indiceUb,
                                                                     LocalTensor<T1> &dataUbSize,
                                                                     LocalTensor<T1> &transposeUbSize) {
    this->Mte3ToMte2();
    int64_t srcOffset = (blockIdx * m_tilingData.preCoreBatchNum + preCopyParams.eachCoreBatchIdx) *
                        m_tilingData.varDim2Count + preCopyParams.inner_loop_idx * m_tilingData.eachPreLoopEle;
    DataCopy(updatesUb, updatesGm[srcOffset], preCopyParams.onceNumAlign);
    varGm.SetGlobalBuffer(this->GetTensorAddr(varPtr, preCopyParams.dim0Idx));
    uint64_t dim1Idx = preCopyParams.allCoreBatchIdx % m_tilingData.dim1Count;
    uint64_t dim2OffsetIdx = indiceUb.GetValue(preCopyParams.dim0Idx);
    uint64_t dstGmOffset = dim1Idx * m_tilingData.dstBatchStride + dim2OffsetIdx +
                           preCopyParams.inner_loop_idx * m_tilingData.eachPreLoopEle * m_tilingData.varDim3Count;

    DataCopyParams copyParams;
    copyParams.blockCount = preCopyParams.onceNum;
    copyParams.blockLen = 1;
    copyParams.srcStride = m_tilingData.varDim3Stride;
    copyParams.dstStride = 0;
    DataCopy(dataUbSize, varGm[dstGmOffset], copyParams);
    this->Mte2ToV();
    preCopyParams.dstGmOffset = dstGmOffset;
    if (sizeof(T1) == 2) {
        TransposeB2(dataUbSize, transposeUbSize, updatesUb, preCopyParams);
    } else if (sizeof(T1) == 4) {
        TransposeB4(dataUbSize, transposeUbSize, updatesUb, preCopyParams);
    } else if (sizeof(T1) == 1) {
        TransposeB1(dataUbSize, transposeUbSize, updatesUb, preCopyParams);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListTransposeLarge<T1, T2>::CopyOut(const uint64_t &CoreBatchNum) {
    LocalTensor<T2> indiceUb = indiceInQueue.DeQue<T2>();
    LocalTensor<T1> updatesUb = updatesInQueue.DeQue<T1>();
    LocalTensor<T1> dataUbSize = dataUb.Get<T1>();
    LocalTensor<T1> transposeUbSize = transposeUb.Get<T1>();
    LocalTensor<uint8_t> maskUb;
    if (!maskIsNull) {
        maskUb = maskInQueue.DeQue<uint8_t>();
    }

    this->Mte2ToS();
    copyPreParams preCopyParams;

    for (uint64_t eachCoreBatchIdx = 0; eachCoreBatchIdx < CoreBatchNum; eachCoreBatchIdx++) {
        int64_t allCoreBatchIdx = blockIdx * m_tilingData.preCoreBatchNum + eachCoreBatchIdx;
        int64_t dim0Idx = allCoreBatchIdx / m_tilingData.dim1Count;
        if ((!maskIsNull) && (maskUb.GetValue(dim0Idx) == 0)) {
            continue;
        }
        for (int64_t inner_loop_idx = 0; inner_loop_idx < m_tilingData.eachLoopNum; inner_loop_idx++) {
            preCopyParams.inner_loop_idx = inner_loop_idx;
            preCopyParams.onceNum = m_tilingData.eachPreLoopEle;
            preCopyParams.onceNumAlign = m_tilingData.updateDim23Align;
            preCopyParams.repeatTimes = m_tilingData.transRepeatTimes;
            preCopyParams.eachCoreBatchIdx = eachCoreBatchIdx;
            preCopyParams.dim0Idx = dim0Idx;
            preCopyParams.allCoreBatchIdx = allCoreBatchIdx;
            preCopyParams.copyNum = m_tilingData.eachPreLoopEle;
            CopyOutPre(preCopyParams, updatesUb, indiceUb, dataUbSize, transposeUbSize);
        }
        preCopyParams.inner_loop_idx = m_tilingData.eachLoopNum;
        preCopyParams.onceNum = m_tilingData.eachLastLoopEle;
        preCopyParams.onceNumAlign = m_tilingData.srcBatchStrideAlign;
        preCopyParams.repeatTimes = m_tilingData.transRepeatTimesTail;
        preCopyParams.eachCoreBatchIdx = eachCoreBatchIdx;
        preCopyParams.dim0Idx = dim0Idx;
        preCopyParams.allCoreBatchIdx = allCoreBatchIdx;
        preCopyParams.copyNum = m_tilingData.eachLastSize;
        CopyOutPre(preCopyParams, updatesUb, indiceUb, dataUbSize, transposeUbSize);
    }

    indiceInQueue.FreeTensor(indiceUb);
    updatesInQueue.FreeTensor(updatesUb);
    if (!maskIsNull) {
        maskInQueue.FreeTensor(maskUb);
    }
}

}  // namespace ScatterList

#endif  // SCATTER_LIST_TRANSPOSE_LARGE_H_
