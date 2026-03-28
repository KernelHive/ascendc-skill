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
 * \file scatter_list_neg_small.h
 * \brief
 */
#ifndef SCATTER_LIST_NEG_SMALL_H_
#define SCATTER_LIST_NEG_SMALL_H_

#include "kernel_operator.h"
#include "scatter_list_base.h"

namespace ScatterList {
using namespace AscendC;

template <typename T1, typename T2>
class ScatterListNegSmall : public ScatterListBase<T1> {
public:
    __aicore__ inline ScatterListNegSmall(){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indice, GM_ADDR updates, GM_ADDR mask, GM_ADDR tempOut,
                                GM_ADDR workspace, const ScatterListTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(const uint64_t &CoreBatchNum);
    __aicore__ inline void CopyOut(const uint64_t &CoreBatchNum);
    __aicore__ inline void DataCopySmallPad(DataCopyExtParams &copyParams, LocalTensor<T1> &updatesUb,
                                            LocalTensor<T2> &indiceUb, const int64_t &dim0Idx, const int64_t &dim1Idx,
                                            const int64_t &srcGmOffset);
    __aicore__ inline void DataCopySmallNoPad(DataCopyParams &copyParams, LocalTensor<T1> &updatesUb,
                                              LocalTensor<T2> &indiceUb, const int64_t &dim0Idx, const int64_t &dim1Idx,
                                              const int64_t &srcGmOffset);
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

    uint64_t blockIdx = 0;
    GM_ADDR varPtr = nullptr;
    bool maskIsNull = false;
    ScatterListTilingData m_tilingData;
};

template <typename T1, typename T2>
__aicore__ inline void ScatterListNegSmall<T1, T2>::Init(GM_ADDR var, GM_ADDR indice, GM_ADDR updates,
                                                         GM_ADDR mask, GM_ADDR tempOut, GM_ADDR workspace,
                                                         const ScatterListTilingData* tilingData) {
    blockIdx = GetBlockIdx();
    this->ParseTilingData(tilingData, m_tilingData);
    varPtr = var;
    indiceGm.SetGlobalBuffer((__gm__ T2*)indice);
    updatesGm.SetGlobalBuffer((__gm__ T1*)updates);
    if (mask == nullptr) {
        maskIsNull = true;
    } else {
        maskGm.SetGlobalBuffer((__gm__ uint8_t*)mask);
        pipe.InitBuffer(maskInQueue, bufferNum, m_tilingData.maskUbSize);
    }

    pipe.InitBuffer(indiceInQueue, 1, m_tilingData.indiceUbSize);
    pipe.InitBuffer(updatesInQueue, bufferNum, m_tilingData.updatesUbSize);
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListNegSmall<T1, T2>::Process() {
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
__aicore__ inline void ScatterListNegSmall<T1, T2>::CopyIn(const uint64_t &CoreBatchNum) {
    LocalTensor<T2> indiceUb = indiceInQueue.AllocTensor<T2>();
    LocalTensor<T1> updatesUb = updatesInQueue.AllocTensor<T1>();
    DataCopy(indiceUb, indiceGm, m_tilingData.indiceCount);
    DataCopy(updatesUb, updatesGm[blockIdx * m_tilingData.preCoreUpdateDim23],
             CoreBatchNum * m_tilingData.srcBatchStride);
    if (!maskIsNull) {
        LocalTensor<uint8_t> maskUb = maskInQueue.AllocTensor<uint8_t>();
        DataCopy(maskUb, maskGm, m_tilingData.maskCount);
        maskInQueue.EnQue(maskUb);
    }

    indiceInQueue.EnQue(indiceUb);
    updatesInQueue.EnQue(updatesUb);
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListNegSmall<T1, T2>::DataCopySmallPad(DataCopyExtParams &copyParams,
                                                                     LocalTensor<T1> &updatesUb,
                                                                     LocalTensor<T2> &indiceUb,
                                                                     const int64_t &dim0Idx, const int64_t &dim1Idx,
                                                                     const int64_t &srcGmOffset) {
    if (m_tilingData.indiceDims == 1) {
        int64_t dim2OffsetIdx = indiceUb.GetValue(dim0Idx);
        int64_t dstGmOffset = dim1Idx * m_tilingData.dstBatchStride + dim2OffsetIdx;
        copyParams.blockCount = m_tilingData.varDim2Count;
        copyParams.blockLen = m_tilingData.dim3CountSize;
        copyParams.srcStride = 0;
        copyParams.dstStride = m_tilingData.varDim3Stride;
        DataCopyPad(varGm[dstGmOffset], updatesUb[srcGmOffset], copyParams);
    } else {
        int64_t dim2OffsetIdx = indiceUb.GetValue(dim0Idx * 2);
        int64_t dim2UpdateLen = indiceUb.GetValue(dim0Idx * 2 + 1);
        int64_t dstGmOffset = dim1Idx * m_tilingData.dstBatchStride + dim2OffsetIdx;
        int64_t block = (dim2UpdateLen + m_tilingData.updatesOneBlock - 1) / m_tilingData.updatesOneBlock;
        copyParams.blockCount = m_tilingData.varDim2Count;
        copyParams.blockLen = dim2UpdateLen * sizeof(T1);
        copyParams.srcStride = m_tilingData.dim3CountAlign - block;
        copyParams.dstStride = (m_tilingData.varDim3Count - dim2UpdateLen) * sizeof(T1);
        DataCopyPad(varGm[dstGmOffset], updatesUb[srcGmOffset], copyParams);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListNegSmall<T1, T2>::DataCopySmallNoPad(DataCopyParams &copyParams,
                                                                       LocalTensor<T1> &updatesUb,
                                                                       LocalTensor<T2> &indiceUb,
                                                                       const int64_t &dim0Idx, const int64_t &dim1Idx,
                                                                       const int64_t &srcGmOffset) {
    int64_t dim2OffsetIdx = indiceUb.GetValue(dim0Idx);
    int64_t dstGmOffset = dim1Idx * m_tilingData.dstBatchStride + dim2OffsetIdx;
    copyParams.blockCount = m_tilingData.varDim2Count;
    copyParams.blockLen = m_tilingData.dim3CountSize;
    copyParams.srcStride = 0;
    copyParams.dstStride = m_tilingData.varDim3Stride;
    DataCopy(varGm[dstGmOffset], updatesUb[srcGmOffset], copyParams);
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListNegSmall<T1, T2>::CopyOut(const uint64_t &CoreBatchNum) {
    LocalTensor<T2> indiceUb = indiceInQueue.DeQue<T2>();
    LocalTensor<T1> updatesUb = updatesInQueue.DeQue<T1>();
    LocalTensor<uint8_t> maskUb;
    if (!maskIsNull) {
        maskUb = maskInQueue.DeQue<uint8_t>();
    }

    this->Mte2ToS();

    for (uint64_t eachCoreBatchIdx = 0; eachCoreBatchIdx < CoreBatchNum; eachCoreBatchIdx++) {
        int64_t allCoreBatchIdx = blockIdx * m_tilingData.preCoreBatchNum + eachCoreBatchIdx;
        int64_t dim0Idx = allCoreBatchIdx / m_tilingData.dim1Count;
        if ((!maskIsNull) && (maskUb.GetValue(dim0Idx) == 0)) {
            continue;
        }
        varGm.SetGlobalBuffer(this->GetTensorAddr(varPtr, dim0Idx));
        uint64_t dim1Idx = allCoreBatchIdx % m_tilingData.dim1Count;
        uint64_t srcGmOffset = eachCoreBatchIdx * m_tilingData.srcBatchStride;
        DataCopyParams copyParams;
        DataCopyExtParams copyextParams;
        if constexpr(platform::IsDataCopyPadSupport()) {
            DataCopySmallPad(copyextParams, updatesUb, indiceUb, dim0Idx, dim1Idx, srcGmOffset);
        } else {
            DataCopySmallNoPad(copyParams, updatesUb, indiceUb, dim0Idx, dim1Idx, srcGmOffset);
        }
    }
    indiceInQueue.FreeTensor(indiceUb);
    updatesInQueue.FreeTensor(updatesUb);
    if (!maskIsNull) {
        maskInQueue.FreeTensor(maskUb);
    }
}
}  // namespace ScatterList

#endif  // SCATTER_LIST_NEG_SMALL_H_
