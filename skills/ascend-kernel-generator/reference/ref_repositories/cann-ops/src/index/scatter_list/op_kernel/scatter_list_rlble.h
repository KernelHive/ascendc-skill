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
 * \file scatter_list_rlble.h
 * \brief
 */
// 此文件实现所有芯片、-2轴更新、大batch大element、updates的后两维乘积对齐block场景，对应模板230
// R: row, L: large, B: batch(updates的前两维除以核数得到), E: element(updates的后两维乘积)
#ifndef SCATTER_LIST_RLBLE_H_
#define SCATTER_LIST_RLBLE_H_

#include "kernel_operator.h"
#include "scatter_list_base.h"

namespace ScatterList {
using namespace AscendC;

template <typename T1, typename T2>
class ScatterListRLBLE : public ScatterListBase<T1> {
public:
    __aicore__ inline ScatterListRLBLE(){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indice, GM_ADDR updates, GM_ADDR mask, GM_ADDR varOut,
                                GM_ADDR workspace, const ScatterListTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessEqLen(const int64_t &eachCoreBatchNum);
    __aicore__ inline void ProcessNotEqLen(const int64_t &eachCoreBatchNum);
    __aicore__ inline void CopyIn(const int64_t &eachCoreBatchIdx, const int64_t &eachBatchInnerLoopIdx);
    __aicore__ inline void CopyOutEqLen(const int64_t &eachCoreBatchIdx, const int64_t &eachBatchInnerLoopIdx,
                                        const int64_t &eachBatchInnerLoopEle);
    __aicore__ inline void CopyOutNotEqLen(const int64_t &eachCoreBatchIdx, const int64_t &eachBatchInnerLoopIdx,
                                           const int64_t &eachBatchInnerLoopEle);

    constexpr static int32_t bufferNum = 1;

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> indiceInQueue;
    TQue<QuePosition::VECIN, bufferNum> updatesInQueue;
    TQue<QuePosition::VECIN, 1> maskInQueue;

    GlobalTensor<T1> varGm;
    GlobalTensor<T2> indiceGm;
    GlobalTensor<T1> updatesGm;
    GlobalTensor<uint8_t> maskGm;

    int64_t blockIdx = 0;
    GM_ADDR varPtr = nullptr;
    bool maskIsNull = false;
    int64_t preCoreBatchIdx = 0;
    int64_t curCoreBatchIdx = 0;
    int64_t dim0Idx = 0;
    int64_t dim1Idx = 0;
    int64_t dim2OffsetIdx = 0;
    int64_t dim2UpdateLen = 0;
    int64_t dstGmOffset = 0;
    int64_t srcDim2OffsetIdx = 0;
    int64_t needCopyCount = 0;
    int64_t haveCopyCount = 0;
    int64_t copyCount = 0;

    // tiling params
    ScatterListTilingData m_tilingData;
};

template <typename T1, typename T2>
__aicore__ inline void ScatterListRLBLE<T1, T2>::Init(GM_ADDR var, GM_ADDR indice, GM_ADDR updates, GM_ADDR mask,
                                                      GM_ADDR varOut, GM_ADDR workspace,
                                                      const ScatterListTilingData* tilingData) {
    blockIdx = GetBlockIdx();
    varPtr = var;
    indiceGm.SetGlobalBuffer((__gm__ T2*)indice);
    updatesGm.SetGlobalBuffer((__gm__ T1*)updates);

    this->ParseTilingData(tilingData, m_tilingData);
    preCoreBatchIdx = blockIdx * m_tilingData.preCoreBatchNum;
    pipe.InitBuffer(indiceInQueue, 1, m_tilingData.indiceUbSize);
    pipe.InitBuffer(updatesInQueue, bufferNum, m_tilingData.updatesUbSize);

    if (mask == nullptr) {
        maskIsNull = true;
    } else {
        maskGm.SetGlobalBuffer((__gm__ uint8_t*)mask);
        pipe.InitBuffer(maskInQueue, 1, m_tilingData.maskUbSize);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListRLBLE<T1, T2>::Process() {
    if (blockIdx >= m_tilingData.useCoreNum) {
        return;
    }
    if (blockIdx == m_tilingData.useCoreNum - 1) {
        if (m_tilingData.indiceDims == 1) {
            ProcessEqLen(m_tilingData.lastCoreBatchNum);
        } else {
            ProcessNotEqLen(m_tilingData.lastCoreBatchNum);
        }
    } else {
        if (m_tilingData.indiceDims == 1) {
            ProcessEqLen(m_tilingData.preCoreBatchNum);
        } else {
            ProcessNotEqLen(m_tilingData.preCoreBatchNum);
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListRLBLE<T1, T2>::ProcessEqLen(const int64_t &eachCoreBatchNum) {
    for (int64_t eachCoreBatchIdx = 0; eachCoreBatchIdx < eachCoreBatchNum; eachCoreBatchIdx++) {
        for (int64_t eachBatchInnerLoopIdx = 0; eachBatchInnerLoopIdx < m_tilingData.eachLoopNum - 1;
             eachBatchInnerLoopIdx++) {
            CopyIn(eachCoreBatchIdx, eachBatchInnerLoopIdx);
            CopyOutEqLen(eachCoreBatchIdx, eachBatchInnerLoopIdx, m_tilingData.eachPreLoopEle);
        }
        CopyIn(eachCoreBatchIdx, m_tilingData.eachLoopNum - 1);
        CopyOutEqLen(eachCoreBatchIdx, m_tilingData.eachLoopNum - 1, m_tilingData.eachLastLoopEle);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListRLBLE<T1, T2>::ProcessNotEqLen(const int64_t &eachCoreBatchNum) {
    for (int64_t eachCoreBatchIdx = 0; eachCoreBatchIdx < eachCoreBatchNum; eachCoreBatchIdx++) {
        for (int64_t eachBatchInnerLoopIdx = 0; eachBatchInnerLoopIdx < m_tilingData.eachLoopNum - 1;
             eachBatchInnerLoopIdx++) {
            CopyIn(eachCoreBatchIdx, eachBatchInnerLoopIdx);
            CopyOutNotEqLen(eachCoreBatchIdx, eachBatchInnerLoopIdx, m_tilingData.eachPreLoopEle);
        }
        CopyIn(eachCoreBatchIdx, m_tilingData.eachLoopNum - 1);
        CopyOutNotEqLen(eachCoreBatchIdx, m_tilingData.eachLoopNum - 1,
                        m_tilingData.eachLastLoopEle);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListRLBLE<T1, T2>::CopyIn(const int64_t &eachCoreBatchIdx,
                                                        const int64_t &eachBatchInnerLoopIdx) {
    LocalTensor<T1> updatesUb = updatesInQueue.AllocTensor<T1>();
    DataCopy(updatesUb,
             updatesGm[(preCoreBatchIdx + eachCoreBatchIdx) * m_tilingData.srcBatchStride +
                       eachBatchInnerLoopIdx * m_tilingData.eachPreLoopEle],
             m_tilingData.updatesCount);
    updatesInQueue.EnQue(updatesUb);

    LocalTensor<T2> indiceUb = indiceInQueue.AllocTensor<T2>();
    DataCopy(indiceUb, indiceGm, m_tilingData.indiceCount);
    indiceInQueue.EnQue(indiceUb);

    if (!maskIsNull) {
        LocalTensor<uint8_t> maskUb = maskInQueue.AllocTensor<uint8_t>();
        DataCopy(maskUb, maskGm, m_tilingData.maskCount);
        maskInQueue.EnQue(maskUb);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListRLBLE<T1, T2>::CopyOutEqLen(const int64_t &eachCoreBatchIdx,
                                                              const int64_t &eachBatchInnerLoopIdx,
                                                              const int64_t &eachBatchInnerLoopEle) {
    LocalTensor<T1> updatesUb = updatesInQueue.DeQue<T1>();
    LocalTensor<T2> indiceUb = indiceInQueue.DeQue<T2>();
    LocalTensor<uint8_t> maskUb;
    if (!maskIsNull) {
        maskUb = maskInQueue.DeQue<uint8_t>();
    }
    this->Mte2ToS();
    this->Mte2ToMte3();

    curCoreBatchIdx = preCoreBatchIdx + eachCoreBatchIdx;
    dim0Idx = curCoreBatchIdx / m_tilingData.dim1Count;
    if (maskIsNull || maskUb.GetValue(dim0Idx) == 1) {
        varGm.SetGlobalBuffer(this->GetTensorAddr(varPtr, dim0Idx));
        dim1Idx = curCoreBatchIdx % m_tilingData.dim1Count;
        dim2OffsetIdx = indiceUb.GetValue(dim0Idx);
        dstGmOffset = dim1Idx * m_tilingData.dstBatchStride + dim2OffsetIdx * m_tilingData.dim3Count +
                      eachBatchInnerLoopIdx * m_tilingData.eachPreLoopEle;
        DataCopy(varGm[dstGmOffset], updatesUb, eachBatchInnerLoopEle);
        this->Mte3ToMte2();
    }

    indiceInQueue.FreeTensor(indiceUb);
    updatesInQueue.FreeTensor(updatesUb);
    if (!maskIsNull) {
        maskInQueue.FreeTensor(maskUb);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListRLBLE<T1, T2>::CopyOutNotEqLen(const int64_t &eachCoreBatchIdx,
                                                                 const int64_t &eachBatchInnerLoopIdx,
                                                                 const int64_t &eachBatchInnerLoopEle) {
    LocalTensor<T1> updatesUb = updatesInQueue.DeQue<T1>();
    LocalTensor<T2> indiceUb = indiceInQueue.DeQue<T2>();
    LocalTensor<uint8_t> maskUb;
    if (!maskIsNull) {
        maskUb = maskInQueue.DeQue<uint8_t>();
    }
    this->Mte2ToS();
    this->Mte2ToMte3();

    curCoreBatchIdx = preCoreBatchIdx + eachCoreBatchIdx;
    dim0Idx = curCoreBatchIdx / m_tilingData.dim1Count;
    if (maskIsNull || maskUb.GetValue(dim0Idx) == 1) {
        varGm.SetGlobalBuffer(this->GetTensorAddr(varPtr, dim0Idx));
        dim1Idx = curCoreBatchIdx % m_tilingData.dim1Count;
        dim2OffsetIdx = indiceUb.GetValue(dim0Idx * 2);
        dim2UpdateLen = indiceUb.GetValue(dim0Idx * 2 + 1);
        needCopyCount = dim2UpdateLen * m_tilingData.dim3Count;
        haveCopyCount = eachBatchInnerLoopIdx * m_tilingData.eachPreLoopEle;
        if (haveCopyCount < needCopyCount) {
            dstGmOffset = dim1Idx * m_tilingData.dstBatchStride + dim2OffsetIdx * m_tilingData.dim3Count +
                          haveCopyCount;
            copyCount = needCopyCount - haveCopyCount;
            copyCount = copyCount > eachBatchInnerLoopEle ? eachBatchInnerLoopEle : copyCount;
            DataCopy(varGm[dstGmOffset], updatesUb, copyCount);
            this->Mte3ToMte2();
        }
    }

    indiceInQueue.FreeTensor(indiceUb);
    updatesInQueue.FreeTensor(updatesUb);
    if (!maskIsNull) {
        maskInQueue.FreeTensor(maskUb);
    }
}

}  // namespace ScatterList

#endif  // SCATTER_LIST_RLBLE_H_
