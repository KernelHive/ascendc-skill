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
 * \file scatter_list_base.h
 * \brief
 */
#ifndef SCATTER_LIST_BASE_H_
#define SCATTER_LIST_BASE_H_

#include "kernel_operator.h"

namespace platform {

#define MID_THREAD_NUM 1024

__aicore__ inline constexpr bool IsDataCopyPadSupport()
{
#if __CCE_AICORE__ == 220
    return true;
#else
    return false;
#endif
}
}

namespace ScatterList {
using namespace AscendC;
struct TransposeParams {
    int64_t src = 0;
    int64_t dst = 0;
    int64_t src_rep = 0;
    int64_t dst_rep = 0;
    int64_t repeat_times = 0;
};

template <typename T>
class ScatterListBase {
public:
    __aicore__ inline ScatterListBase(){};

protected:
    __aicore__ inline void ParseTilingData(const ScatterListTilingData *tilingData,
                                           ScatterListTilingData &m_tilingData);
    __aicore__ inline __gm__ T* GetTensorAddr(GM_ADDR tensorListPtr, const uint64_t &batchIdx);
    __aicore__ inline int64_t CeilDivMul(const int64_t &value, const int64_t &factor);
    __aicore__ inline void Mte2ToS();
    __aicore__ inline void Mte2ToMte3();
    __aicore__ inline void Mte3ToMte2();
    __aicore__ inline void VToMte3();
    __aicore__ inline void Mte2ToV();
    __aicore__ inline void TransposeB8(const TransposeParams &params, LocalTensor<T> &srcUbSize,
                                       LocalTensor<T> &dstUbSize);
    __aicore__ inline void TransposeB16(const TransposeParams &params, LocalTensor<T> &srcUbSize,
                                        LocalTensor<T> &dstUbSize);
    __aicore__ inline void TransposeB32(const TransposeParams &params, LocalTensor<T> &srcUbSize,
                                        LocalTensor<T> &dstUbSize);
    __aicore__ inline void TransposeB32Back(const TransposeParams &params, LocalTensor<T> &srcUbSize,
                                            LocalTensor<T> &dstUbSize);
};

template <typename T>
__aicore__ inline void ScatterListBase<T>::ParseTilingData(const ScatterListTilingData *tilingData,
                                                           ScatterListTilingData &m_tilingData) {
    m_tilingData.dim0Count = tilingData->dim0Count;
    m_tilingData.dim1Count = tilingData->dim1Count;
    m_tilingData.varDim2Count = tilingData->varDim2Count;
    m_tilingData.dim2Count = tilingData->dim2Count;
    m_tilingData.dim3Count = tilingData->dim3Count;
    m_tilingData.dim3CountAlign = tilingData->dim3CountAlign;
    m_tilingData.updatesOneBlock = tilingData->updatesOneBlock;
    m_tilingData.indiceDims = tilingData->indiceDims;
    m_tilingData.indiceCount = tilingData->indiceCount;
    m_tilingData.indiceUbSize = tilingData->indiceUbSize;
    m_tilingData.maskCount = tilingData->maskCount;
    m_tilingData.maskUbSize = tilingData->maskUbSize;
    m_tilingData.srcBatchStride = tilingData->srcBatchStride;
    m_tilingData.srcBatchStrideAlign = tilingData->srcBatchStrideAlign;
    m_tilingData.dstBatchStride = tilingData->dstBatchStride;
    m_tilingData.useCoreNum = tilingData->useCoreNum;
    m_tilingData.preCoreBatchNum = tilingData->preCoreBatchNum;
    m_tilingData.lastCoreBatchNum = tilingData->lastCoreBatchNum;
    m_tilingData.eachLoopNum = tilingData->eachLoopNum;
    m_tilingData.eachPreLoopEle = tilingData->eachPreLoopEle;
    m_tilingData.eachLastLoopEle = tilingData->eachLastLoopEle;
    m_tilingData.eachLastLoopEleAlign = tilingData->eachLastLoopEleAlign;
    m_tilingData.updatesCount = tilingData->updatesCount;
    m_tilingData.updatesUbSize = tilingData->updatesUbSize;
    m_tilingData.dataUbSize = tilingData->dataUbSize;
    m_tilingData.transposeUbSize = tilingData->transposeUbSize;
    m_tilingData.transRepeatTimes = tilingData->transRepeatTimes;
    m_tilingData.transRepeatTimesTail = tilingData->transRepeatTimesTail;
    m_tilingData.updateDim23Align = tilingData->updateDim23Align;
    m_tilingData.preCoreUpdateDim23 = tilingData->preCoreUpdateDim23;
    m_tilingData.varDim3Stride = tilingData->varDim3Stride;
    m_tilingData.varDim3Count = tilingData->varDim3Count;
    m_tilingData.dim3CountSize = tilingData->dim3CountSize;
    m_tilingData.eachLastSize = tilingData->eachLastSize;
    m_tilingData.tilingKey = tilingData->tilingKey;
}

template <typename T>
__aicore__ inline __gm__ T* ScatterListBase<T>::GetTensorAddr(GM_ADDR tensorListPtr, const uint64_t &batchIdx) {
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorListPtr);
    uint64_t tensorPtrOffset = *dataAddr;  // The offset of the data address from the first address.
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(tensorPtr + batchIdx));
}

template <typename T>
__aicore__ inline int64_t ScatterListBase<T>::CeilDivMul(const int64_t &value, const int64_t &factor) {
  if (factor == 0) {
    return value;
  }
  return (value + factor - 1) / factor * factor;
}

template <typename T>
__aicore__ inline void ScatterListBase<T>::Mte2ToS() {
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
}

template <typename T>
__aicore__ inline void ScatterListBase<T>::Mte2ToMte3() {
    event_t eventIdMte2ToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
    WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
}

template <typename T>
__aicore__ inline void ScatterListBase<T>::Mte3ToMte2() {
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
}

template <typename T>
__aicore__ inline void ScatterListBase<T>::VToMte3() {
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
}

template <typename T>
__aicore__ inline void ScatterListBase<T>::Mte2ToV() {
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
}

template <typename T>
__aicore__ inline void ScatterListBase<T>::TransposeB8(const TransposeParams &params,
                                                       LocalTensor<T> &srcUbSize,
                                                       LocalTensor<T> &dstUbSize) {
    __ubuf__ T* srcUb = (__ubuf__ T*)srcUbSize.GetPhyAddr();
    __ubuf__ T* dstUb = (__ubuf__ T*)dstUbSize.GetPhyAddr();
    __ubuf__ T* srcLocalList[16] = {srcUb, srcUb + params.src, srcUb + params.src * 2, srcUb + params.src * 3,
                                    srcUb + params.src * 4, srcUb + params.src * 5, srcUb + params.src * 6,
                                    srcUb + params.src * 7, srcUb + params.src * 8, srcUb + params.src * 9,
                                    srcUb + params.src * 10, srcUb + params.src * 11, srcUb + params.src * 12,
                                    srcUb + params.src * 13, srcUb + params.src * 14, srcUb + params.src * 15};
    __ubuf__ T* dstLocalList[16] = {dstUb, dstUb + params.dst, dstUb + params.dst * 2, dstUb + params.dst * 3,
                                    dstUb + params.dst * 4, dstUb + params.dst * 5, dstUb + params.dst * 6,
                                    dstUb + params.dst * 7, dstUb + params.dst * 8, dstUb + params.dst * 9,
                                    dstUb + params.dst * 10, dstUb + params.dst * 11, dstUb + params.dst * 12,
                                    dstUb + params.dst * 13, dstUb + params.dst * 14, dstUb + params.dst * 15};

    TransDataTo5HDParams transposeParams;
    transposeParams.dstHighHalf = 0;
    transposeParams.srcHighHalf = 0;
    transposeParams.repeatTimes = params.repeat_times;
    transposeParams.dstRepStride = params.dst_rep;
    transposeParams.srcRepStride = params.src_rep;
    TransDataTo5HDImpl(dstLocalList, srcLocalList, transposeParams);
    __ubuf__ T* dstLocalList1[16] = {dstUb + params.dst * 16, dstUb + params.dst * 17, dstUb + params.dst * 18,
                                     dstUb + params.dst * 19, dstUb + params.dst * 20, dstUb + params.dst * 21,
                                     dstUb + params.dst * 22, dstUb + params.dst * 23, dstUb + params.dst * 24,
                                     dstUb + params.dst * 25, dstUb + params.dst * 26, dstUb + params.dst * 27,
                                     dstUb + params.dst * 28, dstUb + params.dst * 29, dstUb + params.dst * 30,
                                     dstUb + params.dst * 31};

    transposeParams.dstHighHalf = 0;
    transposeParams.srcHighHalf = 1;
    TransDataTo5HDImpl(dstLocalList1, srcLocalList, transposeParams);

    __ubuf__ T* srcLocalList2[16] = {srcUb + params.src * 16, srcUb + params.src * 17, srcUb + params.src * 18,
                                     srcUb + params.src * 19, srcUb + params.src * 20, srcUb + params.src * 21,
                                     srcUb + params.src * 22, srcUb + params.src * 23, srcUb + params.src * 24,
                                     srcUb + params.src * 25, srcUb + params.src * 26, srcUb + params.src * 27,
                                     srcUb + params.src * 28, srcUb + params.src * 29, srcUb + params.src * 30,
                                     srcUb + params.src * 31};
    __ubuf__ T* dstLocalList2[16] = {dstUb, dstUb + params.dst, dstUb + params.dst * 2, dstUb + params.dst * 3,
                                     dstUb + params.dst * 4, dstUb + params.dst * 5, dstUb + params.dst * 6,
                                     dstUb + params.dst * 7, dstUb + params.dst * 8, dstUb + params.dst * 9,
                                     dstUb + params.dst * 10, dstUb + params.dst * 11, dstUb + params.dst * 12,
                                     dstUb + params.dst * 13, dstUb + params.dst * 14, dstUb + params.dst * 15};
    transposeParams.dstHighHalf = 1;
    transposeParams.srcHighHalf = 0;
    TransDataTo5HDImpl(dstLocalList2, srcLocalList2, transposeParams);
    __ubuf__ T* dstLocalList3[16] = {dstUb + params.dst * 16, dstUb + params.dst * 17, dstUb + params.dst * 18,
                                     dstUb + params.dst * 19, dstUb + params.dst * 20, dstUb + params.dst * 21,
                                     dstUb + params.dst * 22, dstUb + params.dst * 23, dstUb + params.dst * 24,
                                     dstUb + params.dst * 25, dstUb + params.dst * 26, dstUb + params.dst * 27,
                                     dstUb + params.dst * 28, dstUb + params.dst * 29, dstUb + params.dst * 30,
                                     dstUb + params.dst * 31};
    transposeParams.dstHighHalf = 1;
    transposeParams.srcHighHalf = 1;
    TransDataTo5HDImpl(dstLocalList3, srcLocalList2, transposeParams);
}

template <typename T>
__aicore__ inline void ScatterListBase<T>::TransposeB16(const TransposeParams &params, LocalTensor<T> &srcUbSize,
                                                        LocalTensor<T> &dstUbSize) {
    __ubuf__ T* srcUb = (__ubuf__ T*)srcUbSize.GetPhyAddr();
    __ubuf__ T* dstUb = (__ubuf__ T*)dstUbSize.GetPhyAddr();
    __ubuf__ T* srcLocalList[16] = {srcUb, srcUb + params.src, srcUb + params.src * 2, srcUb + params.src * 3,
                                    srcUb + params.src * 4, srcUb + params.src * 5, srcUb + params.src * 6,
                                    srcUb + params.src * 7, srcUb + params.src * 8, srcUb + params.src * 9,
                                    srcUb + params.src * 10, srcUb + params.src * 11, srcUb + params.src * 12,
                                    srcUb + params.src * 13, srcUb + params.src * 14, srcUb + params.src * 15};
    __ubuf__ T* dstLocalList[16] = {dstUb, dstUb + params.dst, dstUb + params.dst * 2, dstUb + params.dst * 3,
                                    dstUb + params.dst * 4, dstUb + params.dst * 5, dstUb + params.dst * 6,
                                    dstUb + params.dst * 7, dstUb + params.dst * 8, dstUb + params.dst * 9,
                                    dstUb + params.dst * 10, dstUb + params.dst * 11, dstUb + params.dst * 12,
                                    dstUb + params.dst * 13, dstUb + params.dst * 14, dstUb + params.dst * 15};
    TransDataTo5HDParams transposeParams;
    transposeParams.dstHighHalf = 0;
    transposeParams.srcHighHalf = 0;
    transposeParams.repeatTimes = params.repeat_times;
    transposeParams.dstRepStride = params.dst_rep;
    transposeParams.srcRepStride = params.src_rep;
    TransDataTo5HDImpl(dstLocalList, srcLocalList, transposeParams);
}

template <typename T>
__aicore__ inline void ScatterListBase<T>::TransposeB32(const TransposeParams &params, LocalTensor<T> &srcUbSize,
                                                        LocalTensor<T> &dstUbSize) {
    __ubuf__ T* srcUb = (__ubuf__ T*)srcUbSize.GetPhyAddr();
    __ubuf__ T* dstUb = (__ubuf__ T*)dstUbSize.GetPhyAddr();
    __ubuf__ T* srcLocalList[16] = {srcUb, srcUb + params.src, srcUb + params.src * 2, srcUb + params.src * 3,
                                    srcUb + params.src * 4, srcUb + params.src * 5, srcUb + params.src * 6,
                                    srcUb + params.src * 7, srcUb + params.src * 8, srcUb + params.src * 9,
                                    srcUb + params.src * 10, srcUb + params.src * 11, srcUb + params.src * 12,
                                    srcUb + params.src * 13, srcUb + params.src * 14, srcUb + params.src * 15};

    __ubuf__ T* dstLocalList[16] = {dstUb, dstUb + 8, dstUb + params.dst, dstUb + params.dst + 8,
                                    dstUb + params.dst * 2, dstUb + params.dst * 2 + 8, dstUb + params.dst * 3,
                                    dstUb + params.dst * 3 + 8, dstUb + params.dst * 4, dstUb + params.dst * 4 + 8,
                                    dstUb + params.dst * 5, dstUb + params.dst * 5 + 8, dstUb + params.dst * 6,
                                    dstUb + params.dst * 6 + 8, dstUb + params.dst * 7, dstUb + params.dst * 7 + 8};
    TransDataTo5HDParams transposeParams;
    transposeParams.dstHighHalf = 0;
    transposeParams.srcHighHalf = 0;
    transposeParams.repeatTimes = params.repeat_times;
    transposeParams.dstRepStride = params.dst_rep;
    transposeParams.srcRepStride = params.src_rep;

    TransDataTo5HDImpl(dstLocalList,srcLocalList, transposeParams);
}

template <typename T>
__aicore__ inline void ScatterListBase<T>::TransposeB32Back(const TransposeParams &params, LocalTensor<T> &srcUbSize,
                                                            LocalTensor<T> &dstUbSize) {
    __ubuf__ T* srcUb = (__ubuf__ T*)srcUbSize.GetPhyAddr();
    __ubuf__ T* dstUb = (__ubuf__ T*)dstUbSize.GetPhyAddr();
    __ubuf__ T* srcLocalList[16] = {srcUb, srcUb + params.src, srcUb + params.src * 2, srcUb + params.src * 3,
                                    srcUb + params.src * 4, srcUb + params.src * 5, srcUb + params.src * 6,
                                    srcUb + params.src * 7, srcUb + 8, srcUb + params.src + 8,
                                    srcUb + params.src * 2 + 8, srcUb + params.src * 3 + 8, srcUb + params.src * 4 + 8,
                                    srcUb + params.src * 5 + 8, srcUb + params.src * 6 + 8, srcUb + params.src * 7 + 8};
    __ubuf__ T* dstLocalList[16] = {dstUb, dstUb + 64, dstUb + params.dst, dstUb + params.dst + 64,
                                    dstUb + params.dst * 2, dstUb + params.dst * 2 + 64, dstUb + params.dst * 3,
                                    dstUb + params.dst * 3 + 64, dstUb + params.dst * 4, dstUb + params.dst * 4 + 64,
                                    dstUb + params.dst * 5, dstUb + params.dst * 5 + 64, dstUb + params.dst * 6,
                                    dstUb + params.dst * 6 + 64, dstUb + params.dst * 7, dstUb + params.dst * 7 + 64};
    TransDataTo5HDParams transposeParams;
    transposeParams.dstHighHalf = 0;
    transposeParams.srcHighHalf = 0;
    transposeParams.repeatTimes = params.repeat_times;
    transposeParams.dstRepStride = params.dst_rep;
    transposeParams.srcRepStride = params.src_rep;
    TransDataTo5HDImpl(dstLocalList, srcLocalList, transposeParams);
}
}  // namespace ScatterList

#endif  // SCATTER_LIST_BASE_H_
