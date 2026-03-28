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
 * \file lin_space_small_shape.h
 * \brief
 */
#ifndef LINSPACE_SMALL_SHAPE_H
#define LINSPACE_SMALL_SHAPE_H

#include "lin_space_base.h"

namespace LinSpace {
using namespace AscendC;

template <typename T>
class LinSpaceSmallShape : public LinSpaceBase<T> {
public:
    __aicore__ inline LinSpaceSmallShape(){};
    __aicore__ inline void Init(GM_ADDR start, GM_ADDR stop, GM_ADDR num, GM_ADDR output,
                                GM_ADDR workspace, const LinSpaceTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut();

    constexpr static int32_t bufferNum = 1;
    constexpr static int64_t HALF_NUM = 2;
    constexpr static int32_t outSize = 64 * 4;
    constexpr static int64_t blockSize = 32;
    constexpr static int64_t elementPerBlock = blockSize / sizeof(T);

private:
    TPipe pipe;
    TQue<QuePosition::VECOUT, bufferNum> outQueue;
    GlobalTensor<T> outputGm;

    int32_t blockIdx = 0;
    T blockOffset = 0;
    // tiling params
    LinSpaceTilingData m_tilingData;
};

template <typename T>
__aicore__ inline void LinSpaceSmallShape<T>::Init(GM_ADDR start, GM_ADDR stop, GM_ADDR num, GM_ADDR output,
                                                   GM_ADDR workspace, const LinSpaceTilingData *tilingData)
{
    blockIdx = GetBlockIdx();
    outputGm.SetGlobalBuffer((__gm__ T *)output);
    this->ParseTilingData(tilingData, m_tilingData);

    pipe.InitBuffer(outQueue, bufferNum, outSize);
}

template <typename T>
__aicore__ inline void LinSpaceSmallShape<T>::Process()
{
    if (m_tilingData.num == 0 || blockIdx >= m_tilingData.realCoreNum) {
        return;
    }

    Compute();
    CopyOut();
}

template <typename T>
__aicore__ inline void LinSpaceSmallShape<T>::Compute()
{
    LocalTensor<T> outLocal = outQueue.AllocTensor<T>();

    int64_t halfway = m_tilingData.num / HALF_NUM;

    if (m_tilingData.num == 1) {
        outLocal.SetValue(0, m_tilingData.start);
    } else {
        for (int64_t idx = 0; idx < m_tilingData.num; idx++) {
            if (idx < halfway) {
                outLocal.SetValue(idx, m_tilingData.start + m_tilingData.scalar * idx);
            } else {
                outLocal.SetValue(idx, m_tilingData.stop - m_tilingData.scalar * (m_tilingData.num - idx - 1));
            }
        }
    }

    outQueue.EnQue(outLocal);
}

template <typename T>
__aicore__ inline void LinSpaceSmallShape<T>::CopyOut()
{
    LocalTensor<T> outLocal = outQueue.DeQue<T>();
    if constexpr (PlatformSocInfo::IsDataCopyPadSupport()) {
        DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
        copyParams.blockLen = m_tilingData.tailNum * sizeof(T);
        DataCopyPad(outputGm, outLocal, copyParams);
    } else {
        int64_t tailAlignNum = this->CeilDiv(m_tilingData.tailNum, elementPerBlock) * elementPerBlock;
        DataCopy(outputGm, outLocal, tailAlignNum);
    }
    outQueue.FreeTensor(outLocal);
}
}  // namespace LinSpace

#endif  // LINSPACE_SMALL_SHAPE_H