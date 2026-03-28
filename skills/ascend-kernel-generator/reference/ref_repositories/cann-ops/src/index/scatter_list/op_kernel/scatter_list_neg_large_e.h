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
 * \file scatter_list_neg_large_e.h
 * \brief
 */
#ifndef SCATTER_LIST_NEG_LARGE_E_H_
#define SCATTER_LIST_NEG_LARGE_E_H_

#include "scatter_list_base.h"

namespace ScatterList {
using namespace AscendC;

template <typename T1, typename T2>
class ScatterListNegLargeE : public ScatterListBase<T1> {
public:
    __aicore__ inline ScatterListNegLargeE(){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indice, GM_ADDR update, GM_ADDR mask,GM_ADDR tempOut,
                                GM_ADDR workspace, const ScatterListTilingData *tilingData);
    __aicore__ inline void Process();

private:
    struct copyOutLargeParams {
        int64_t inner_loop_idx = 0;
        int64_t copyNum = 0;
        int64_t eachCoreBatchIdx = 0;
        int64_t allCoreBatchIdx = 0;
        int64_t dim0Idx = 0;
    };
    __aicore__ inline void CopyIn(const int64_t &numPerCore);

    __aicore__ inline void CopyOut(const int64_t &numPerCore);
    __aicore__ inline void CopyLast(copyOutLargeParams &params, LocalTensor<T1> &updatesUb,
                                    LocalTensor<T2> &indiceUb);
    __aicore__ inline void CopyOnce(copyOutLargeParams &params, LocalTensor<T1> &updatesUb,
                                    LocalTensor<T2> &indiceUb);
    __aicore__ inline void CopyLargeEPad(copyOutLargeParams &params, LocalTensor<T1> &updatesUb,
                                         LocalTensor<T2> &indiceUb, const int64_t &dim1Idx);
    __aicore__ inline void ProcessPerCore(const int64_t &numPerCore);

    constexpr static int32_t bufferNum = 1;

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> indiceInQueue;
    TQue<QuePosition::VECIN, bufferNum> updatesInQueue;
    TQue<QuePosition::VECIN, bufferNum> maskInQueue;
    GlobalTensor<T2> indiceGM;
    GlobalTensor<T1> updatesGM;
    GlobalTensor<uint8_t> maskGM;
    GlobalTensor<T1> varGm;
    int64_t blockIdx = 0;
    bool maskIsNull = false;
    GM_ADDR varPtr = nullptr;
    ScatterListTilingData m_tilingData;
};

template <typename T1, typename T2>
__aicore__ inline void ScatterListNegLargeE<T1, T2>::Init(GM_ADDR var, GM_ADDR indice, GM_ADDR updates, GM_ADDR mask,
                                                          GM_ADDR tempOut, GM_ADDR workspace,
                                                          const ScatterListTilingData* tilingData) {
    blockIdx = GetBlockIdx();
    this->ParseTilingData(tilingData, m_tilingData);
    varPtr = var;
    indiceGM.SetGlobalBuffer((__gm__ T2*)indice);
    updatesGM.SetGlobalBuffer((__gm__ T1*)updates);
    if (mask == nullptr) {
        maskIsNull = true;
    } else {
        maskGM.SetGlobalBuffer((__gm__ uint8_t*)mask);
        pipe.InitBuffer(maskInQueue, bufferNum, m_tilingData.maskUbSize);
    }

    pipe.InitBuffer(indiceInQueue, bufferNum, m_tilingData.indiceUbSize);
    pipe.InitBuffer(updatesInQueue, bufferNum, m_tilingData.updatesUbSize);
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListNegLargeE<T1, T2>::Process() {
    if (blockIdx >= m_tilingData.useCoreNum) {
        return;
    }
    if (blockIdx == m_tilingData.useCoreNum - 1) {
        ProcessPerCore(m_tilingData.lastCoreBatchNum);
    } else {
        ProcessPerCore(m_tilingData.preCoreBatchNum);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListNegLargeE<T1, T2>::ProcessPerCore(const int64_t &numPerCore) {
    CopyIn(numPerCore);
    CopyOut(numPerCore);
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListNegLargeE<T1, T2>::CopyIn(const int64_t &numPerCore) {
    LocalTensor<T2> indiceUb = indiceInQueue.AllocTensor<T2>();
    DataCopy(indiceUb, indiceGM, m_tilingData.indiceCount);
    LocalTensor<T1> updatesUb = updatesInQueue.AllocTensor<T1>();
    if (!maskIsNull) {
        LocalTensor<uint8_t> maskUb = maskInQueue.AllocTensor<uint8_t>();
        DataCopy(maskUb, maskGM, m_tilingData.maskCount);
        maskInQueue.EnQue(maskUb);
    }

    indiceInQueue.EnQue(indiceUb);
    updatesInQueue.EnQue(updatesUb);
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListNegLargeE<T1, T2>::CopyOnce(copyOutLargeParams &params, LocalTensor<T1> &updatesUb,
                                                              LocalTensor<T2> &indiceUb) {
    this->Mte3ToMte2();
    int64_t srcOffset = blockIdx * m_tilingData.preCoreBatchNum * m_tilingData.dim3Count +
                        params.eachCoreBatchIdx * m_tilingData.dim3Count +
                        params.inner_loop_idx * m_tilingData.eachPreLoopEle;
    DataCopy(updatesUb, updatesGM[srcOffset], params.copyNum);

    varGm.SetGlobalBuffer(this->GetTensorAddr(varPtr, params.dim0Idx));
    int64_t dim1Idx = params.allCoreBatchIdx % m_tilingData.dim1Count;
    if constexpr(platform::IsDataCopyPadSupport()) {
        int64_t dim2OffsetIdx = indiceUb.GetValue(params.dim0Idx);
        int64_t dstGmOffset = dim1Idx * m_tilingData.varDim3Count + dim2OffsetIdx +
                              params.inner_loop_idx * m_tilingData.eachPreLoopEle;

        this->Mte2ToMte3();
        DataCopy(varGm[dstGmOffset], updatesUb, params.copyNum);
    } else {
        int64_t dim2OffsetIdx = indiceUb.GetValue(params.dim0Idx);
        int64_t dstGmOffset = dim1Idx * m_tilingData.varDim3Count + dim2OffsetIdx +
                              params.inner_loop_idx * m_tilingData.eachPreLoopEle;

        this->Mte2ToMte3();
        DataCopy(varGm[dstGmOffset], updatesUb, params.copyNum);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListNegLargeE<T1, T2>::CopyLargeEPad(copyOutLargeParams &params,
                                                                   LocalTensor<T1> &updatesUb,
                                                                   LocalTensor<T2> &indiceUb,
                                                                   const int64_t &dim1Idx) {
    int64_t src_offset = blockIdx * m_tilingData.preCoreBatchNum * m_tilingData.dim3Count +
                         params.eachCoreBatchIdx * m_tilingData.dim3Count +
                         params.inner_loop_idx * m_tilingData.eachPreLoopEle;
    DataCopyExtParams copyParams;
    DataCopy(updatesUb, updatesGM[src_offset], m_tilingData.eachLastLoopEleAlign);

    int64_t dim2OffsetIdx = indiceUb.GetValue(params.dim0Idx);
    int64_t dstGmOffset = dim1Idx * m_tilingData.varDim3Count + dim2OffsetIdx +
                          params.inner_loop_idx * m_tilingData.eachPreLoopEle;
    copyParams.blockCount = 1;
    copyParams.blockLen = m_tilingData.eachLastSize;
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    this->Mte2ToMte3();
    DataCopyPad(varGm[dstGmOffset], updatesUb, copyParams);
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListNegLargeE<T1, T2>::CopyLast(copyOutLargeParams &params,
                                                              LocalTensor<T1> &updatesUb, LocalTensor<T2> &indiceUb) {
    this->Mte3ToMte2();

    varGm.SetGlobalBuffer(this->GetTensorAddr(varPtr, params.dim0Idx));
    int64_t dim1Idx = params.allCoreBatchIdx % m_tilingData.dim1Count;
    if constexpr(platform::IsDataCopyPadSupport()) {
        CopyLargeEPad(params, updatesUb, indiceUb, dim1Idx);
    } else {
        int64_t src_offset = blockIdx * m_tilingData.preCoreBatchNum * m_tilingData.dim3Count +
                             params.eachCoreBatchIdx * m_tilingData.dim3Count +
                             params.inner_loop_idx * m_tilingData.eachPreLoopEle;
        DataCopy(updatesUb, updatesGM[src_offset], m_tilingData.eachLastLoopEle);
        int64_t dim2OffsetIdx = indiceUb.GetValue(params.dim0Idx);
        int64_t dstGmOffset = dim1Idx * m_tilingData.varDim3Count + dim2OffsetIdx +
                              params.inner_loop_idx * m_tilingData.eachPreLoopEle;
        this->Mte2ToMte3();
        DataCopy(varGm[dstGmOffset], updatesUb, m_tilingData.eachLastLoopEle);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListNegLargeE<T1, T2>::CopyOut(const int64_t &eachCoreBatchNum) {
    LocalTensor<T2> indiceUb = indiceInQueue.DeQue<T2>();
    LocalTensor<T1> updatesUb = updatesInQueue.DeQue<T1>();
    LocalTensor<uint8_t> maskUb;
    if (!maskIsNull) {
        maskUb = maskInQueue.DeQue<uint8_t>();
    }
    this->Mte2ToS();

    for (int64_t eachCoreBatchIdx = 0; eachCoreBatchIdx < eachCoreBatchNum; eachCoreBatchIdx++) {
        int64_t allCoreBatchIdx = blockIdx * m_tilingData.preCoreBatchNum + eachCoreBatchIdx;
        int64_t dim0Idx = allCoreBatchIdx / m_tilingData.dim1Count;
        if ((!maskIsNull) && (maskUb.GetValue(dim0Idx) == 0)) {
          continue;
        }
        copyOutLargeParams params;
        for (int64_t inner_loop_idx = 0; inner_loop_idx < m_tilingData.eachLoopNum; inner_loop_idx++) {
            params.inner_loop_idx = inner_loop_idx;
            params.copyNum = m_tilingData.eachPreLoopEle;
            params.eachCoreBatchIdx = eachCoreBatchIdx;
            params.allCoreBatchIdx = allCoreBatchIdx;
            params.dim0Idx = dim0Idx;
            CopyOnce(params, updatesUb, indiceUb);
        }
        params.inner_loop_idx = m_tilingData.eachLoopNum;
        params.copyNum = m_tilingData.eachLastLoopEle;
        params.eachCoreBatchIdx = eachCoreBatchIdx;
        params.allCoreBatchIdx = allCoreBatchIdx;
        params.dim0Idx = dim0Idx;
        CopyLast(params, updatesUb, indiceUb);
    }
    indiceInQueue.FreeTensor(indiceUb);
    updatesInQueue.FreeTensor(updatesUb);
    if (!maskIsNull) {
        maskInQueue.FreeTensor(maskUb);
    }
}

}  // namespace ScatterList

#endif  // SCATTER_LIST_NEG_LARGE_E_H_
