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
 * \file scatter_list_neg_dim2_large.h
 * \brief
 */
#ifndef SCATTER_LIST_NEG_DIM2_LARGE_E_H_
#define SCATTER_LIST_NEG_DIM2_LARGE_E_H_

#include "scatter_list_base.h"

namespace ScatterList {
using namespace AscendC;

template <typename T1, typename T2>
class ScatterListNegDim2Large : public ScatterListBase<T1> {
public:
    __aicore__ inline ScatterListNegDim2Large(){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indice, GM_ADDR update, GM_ADDR mask,GM_ADDR tempOut,
                                GM_ADDR workspace, const ScatterListTilingData *tilingData);
    __aicore__ inline void Process();

private:
    struct copyDim2LargeParams {
        int64_t inner_loop_idx = 0;
        int64_t copyNum = 0;
        int64_t eachCoreBatchIdx = 0;
        int64_t dim1Idx = 0;
        int64_t dim2OffsetIdx = 0;
        int64_t copyOutNum = 0;
        int64_t dim2UpdateLen = 0;
        int64_t dim2UpdateLenAlign = 0;
    };
    __aicore__ inline void CopyIn(const int64_t &numPerCore);

    __aicore__ inline void CopyOut(const int64_t &numPerCore);
    __aicore__ inline void CopyLast(copyDim2LargeParams &params, LocalTensor<T1> &updatesUb,
                                    LocalTensor<T2> &indiceUb);
    __aicore__ inline void CopyOnce(copyDim2LargeParams &params, LocalTensor<T1> &updatesUb,
                                    LocalTensor<T2> &indiceUb);
    __aicore__ inline void CopyOutDim2(copyDim2LargeParams &params, LocalTensor<T1> &updatesUb,
                                       LocalTensor<T2> &indiceUb);
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
__aicore__ inline void ScatterListNegDim2Large<T1, T2>::Init(GM_ADDR var, GM_ADDR indice, GM_ADDR updates, GM_ADDR mask,
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
__aicore__ inline void ScatterListNegDim2Large<T1, T2>::Process() {
    if (blockIdx >= m_tilingData.useCoreNum) {
        return;
    }
    if (blockIdx == m_tilingData.useCoreNum - 1) {
        ProcessPerCore(m_tilingData.lastCoreBatchNum);
    } else {
        ProcessPerCore( m_tilingData.preCoreBatchNum);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListNegDim2Large<T1, T2>::ProcessPerCore(const int64_t &numPerCore) {
    CopyIn(numPerCore);
    CopyOut(numPerCore);
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListNegDim2Large<T1, T2>::CopyIn(const int64_t &numPerCore) {
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
__aicore__ inline void ScatterListNegDim2Large<T1, T2>::CopyOnce(copyDim2LargeParams &params,
                                                                 LocalTensor<T1> &updatesUb,
                                                                 LocalTensor<T2> &indiceUb) {
    if constexpr(platform::IsDataCopyPadSupport()) {
        this->Mte3ToMte2();
        int64_t srcOffset = blockIdx * m_tilingData.preCoreBatchNum * m_tilingData.dim3Count +
                            params.eachCoreBatchIdx * m_tilingData.dim3Count +
                            params.inner_loop_idx * m_tilingData.eachPreLoopEle;
        DataCopy(updatesUb, updatesGM[srcOffset], params.copyNum);

        int64_t dstGmOffset = params.dim1Idx * m_tilingData.varDim3Count + params.dim2OffsetIdx +
                              params.inner_loop_idx * m_tilingData.eachPreLoopEle;
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = params.copyOutNum * sizeof(T1);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        this->Mte2ToMte3();
        DataCopyPad(varGm[dstGmOffset], updatesUb, copyParams);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListNegDim2Large<T1, T2>::CopyOutDim2(copyDim2LargeParams &params,
                                                                    LocalTensor<T1> &updatesUb,
                                                                     LocalTensor<T2> &indiceUb) {
    if (params.dim2UpdateLenAlign <= m_tilingData.eachPreLoopEle) {
        params.inner_loop_idx = 0;
        params.copyNum = params.dim2UpdateLenAlign;
        params.copyOutNum = params.dim2UpdateLen;
        CopyOnce(params, updatesUb, indiceUb);
    } else {
        int64_t eachBatchLoopNum = (params.dim2UpdateLen + m_tilingData.eachPreLoopEle - 1) /
                                   m_tilingData.eachPreLoopEle - 1;
        int64_t eachBatchLast = params.dim2UpdateLen - m_tilingData.eachPreLoopEle * eachBatchLoopNum;
        int64_t eachBatchLastAlign = ((eachBatchLast + m_tilingData.updatesOneBlock - 1) /
                                      m_tilingData.updatesOneBlock) * m_tilingData.updatesOneBlock;

        for (int64_t inner_loop_idx = 0; inner_loop_idx < eachBatchLoopNum; inner_loop_idx++) {
            params.inner_loop_idx = inner_loop_idx;
            params.copyNum = m_tilingData.eachPreLoopEle;
            params.copyOutNum = m_tilingData.eachPreLoopEle;
            CopyOnce(params, updatesUb, indiceUb);
        }
        params.inner_loop_idx = eachBatchLoopNum;
        params.copyNum = eachBatchLastAlign;
        params.copyOutNum = eachBatchLast;
        CopyOnce(params, updatesUb, indiceUb);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListNegDim2Large<T1, T2>::CopyLast(copyDim2LargeParams &params,
                                                                 LocalTensor<T1> &updatesUb,
                                                                 LocalTensor<T2> &indiceUb) {
    if constexpr(platform::IsDataCopyPadSupport()) {
        this->Mte3ToMte2();
        int64_t src_offset = blockIdx * m_tilingData.preCoreBatchNum * m_tilingData.dim3Count +
                             params.eachCoreBatchIdx * m_tilingData.dim3Count +
                             params.inner_loop_idx * m_tilingData.eachPreLoopEle;
        DataCopyExtParams copyParams;
        DataCopy(updatesUb, updatesGM[src_offset], params.copyNum);
        int64_t dstGmOffset = params.dim1Idx * m_tilingData.varDim3Count + params.dim2OffsetIdx +
                              params.inner_loop_idx * m_tilingData.eachPreLoopEle;
        copyParams.blockCount = 1;
        copyParams.blockLen = params.copyOutNum * sizeof(T1);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        this->Mte2ToMte3();
        DataCopyPad(varGm[dstGmOffset], updatesUb[0], copyParams);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListNegDim2Large<T1, T2>::CopyOut(const int64_t &eachCoreBatchNum) {
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
        int64_t dim2OffsetIdx = indiceUb.GetValue(dim0Idx * 2);
        int64_t dim2UpdateLen = indiceUb.GetValue(dim0Idx * 2 + 1);
        int64_t dim2UpdateLenAlign = ((dim2UpdateLen + m_tilingData.updatesOneBlock -1) /
                                      m_tilingData.updatesOneBlock) * m_tilingData.updatesOneBlock;
        int64_t dim1Idx = allCoreBatchIdx % m_tilingData.dim1Count;
        varGm.SetGlobalBuffer(this->GetTensorAddr(varPtr, dim0Idx));
        copyDim2LargeParams params;
        params.eachCoreBatchIdx = eachCoreBatchIdx;
        params.dim1Idx = dim1Idx;
        params.dim2OffsetIdx = dim2OffsetIdx;
        params.dim2UpdateLen = dim2UpdateLen;
        params.dim2UpdateLenAlign = dim2UpdateLenAlign;
        CopyOutDim2(params, updatesUb, indiceUb);
    }
    indiceInQueue.FreeTensor(indiceUb);
    updatesInQueue.FreeTensor(updatesUb);
    if (!maskIsNull) {
        maskInQueue.FreeTensor(maskUb);
    }
}

}  // namespace ScatterList

#endif  // SCATTER_LIST_NEG_DIM2_LARGE_E_H_
