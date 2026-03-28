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
 * \file lin_space_small_shape_and_cast.h
 * \brief
 */
#ifndef LINSPACE_SMALL_SHAPE_AND_CAST_H
#define LINSPACE_SMALL_SHAPE_AND_CAST_H

#include "lin_space_base.h"

namespace LinSpace {
using namespace AscendC;

template <typename T1, typename T2>
class LinSpaceSmallShapeAndCast : public LinSpaceBase<T2> {
public:
    __aicore__ inline LinSpaceSmallShapeAndCast(){};
    __aicore__ inline void Init(GM_ADDR start, GM_ADDR stop, GM_ADDR num, GM_ADDR output,
                                GM_ADDR workspace, const LinSpaceTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut();

    constexpr static int32_t bufferNum = 1;
    constexpr static int32_t processNum = 64;
    constexpr static int32_t HALF_NUM = 2;
    constexpr static int64_t blockSize = 32;
    constexpr static int64_t elementPerBlock = blockSize / sizeof(T1);

private:
    TPipe pipe;
    TQue<QuePosition::VECOUT, bufferNum> outQueue;
    TBuf<QuePosition::VECCALC> castBuf;
    GlobalTensor<T1> outputGm;

    int32_t blockIdx = 0;
    T2 blockOffset = 0;
    // tiling params
    LinSpaceTilingData m_tilingData;
};

template <typename T1, typename T2>
__aicore__ inline void LinSpaceSmallShapeAndCast<T1, T2>::Init(GM_ADDR start, GM_ADDR stop, GM_ADDR num,
                                                               GM_ADDR output, GM_ADDR workspace,
                                                               const LinSpaceTilingData *tilingData)
{
    blockIdx = GetBlockIdx();
    outputGm.SetGlobalBuffer((__gm__ T1 *)output);

    this->ParseTilingData(tilingData, m_tilingData);

    pipe.InitBuffer(outQueue, bufferNum, processNum * sizeof(T1));
    pipe.InitBuffer(castBuf, processNum * sizeof(T2));
}

template <typename T1, typename T2>
__aicore__ inline void LinSpaceSmallShapeAndCast<T1, T2>::Process()
{
    if (m_tilingData.num == 0 || blockIdx >= m_tilingData.realCoreNum) {
        return;
    }

    Compute();
    CopyOut();
}

template <typename T1, typename T2>
__aicore__ inline void LinSpaceSmallShapeAndCast<T1, T2>::Compute()
{
    LocalTensor<T1> outLocal = outQueue.AllocTensor<T1>();
    LocalTensor<T2> ubNeedCast = castBuf.Get<T2>();

    int64_t halfway = m_tilingData.num / HALF_NUM;

    if (m_tilingData.num == 1) {
        ubNeedCast.SetValue(0, m_tilingData.start);
    } else {
        for (int64_t idx = 0; idx < m_tilingData.num; idx++) {
            if (idx < halfway) {
                ubNeedCast.SetValue(idx, m_tilingData.start + m_tilingData.scalar * idx);
            } else {
                ubNeedCast.SetValue(idx, m_tilingData.stop - m_tilingData.scalar * (m_tilingData.num - idx - 1));
            }
        }
    }

    RoundMode retR = RoundMode::CAST_TRUNC;
    if (sizeof(T1) == sizeof(int16_t) && sizeof(T2) == sizeof(float)) {
        retR = RoundMode::CAST_ROUND;
    }

    Cast(outLocal, ubNeedCast, retR, m_tilingData.num);
    outQueue.EnQue(outLocal);
}

template <typename T1, typename T2>
__aicore__ inline void LinSpaceSmallShapeAndCast<T1, T2>::CopyOut()
{
    LocalTensor<T1> outLocal = outQueue.DeQue<T1>();
    if constexpr (PlatformSocInfo::IsDataCopyPadSupport()) {
        DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
        copyParams.blockLen = m_tilingData.tailNum * sizeof(T1);
        DataCopyPad(outputGm, outLocal, copyParams);
    } else {
        int64_t tailAlignNum = this->CeilDiv(m_tilingData.tailNum, elementPerBlock) * elementPerBlock;
        DataCopy(outputGm, outLocal, tailAlignNum);
    }
    outQueue.FreeTensor(outLocal);
}
}  // namespace LinSpace

#endif  // LINSPACE_SMALL_SHAPE_AND_CAST_H