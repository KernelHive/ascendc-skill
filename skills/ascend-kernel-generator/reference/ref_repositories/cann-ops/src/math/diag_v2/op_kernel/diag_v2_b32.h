/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file diag_v2_b32.h
 * \brief
 */
#ifndef DIAG_V2_B32_H
#define DIAG_V2_B32_H

#include "diag_v2_base.h"

namespace DiagV2 {
using namespace AscendC;

template <typename T>
class DiagV2B32 : public DiagV2Base<T> {
public:
    __aicore__ inline DiagV2B32(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const DiagV2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessPerBlock(const LocalTensor<int16_t> &ubAssist, int64_t loopNum, int64_t loopTail);
    __aicore__ inline void ProcessLastBlock(const LocalTensor<int16_t> &ubAssist, int64_t loopNum, int64_t loopTail);
    __aicore__ inline void CopyInB32(int64_t index, int64_t dataCount);
    __aicore__ inline void CopyInB32NotAlign(int64_t index, int64_t dataCount);
    __aicore__ inline void CopyInB32WithPad(int64_t index, int64_t dataCount);
    __aicore__ inline void Compute(const LocalTensor<int16_t> &ubAssist);
    __aicore__ inline void CopyOut(const int64_t dataCount);
    __aicore__ inline void CopyOutWithPad(const int64_t dataCount);

    constexpr static int32_t matrixSize = 64 * 64;
    constexpr static int32_t bufferNum = 2;
    constexpr static int32_t perBlockNum = 8;
    constexpr static int32_t mask = 128;

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> inQueueX;
    TQue<QuePosition::VECOUT, bufferNum> outQueueY;
    TBuf<QuePosition::VECCALC> matrixBuf;
    GlobalTensor<T> xGm, yGm;
    GlobalTensor<int32_t> gmAssist;

    int32_t blockIdx = 0;
    int64_t gmOutOffset = 0;

    // tiling params
    DiagV2TilingData m_tilingData;
};

template <typename T>
__aicore__ inline void DiagV2B32<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const DiagV2TilingData *tilingData)
{
    blockIdx = GetBlockIdx();
    xGm.SetGlobalBuffer((__gm__ T *)x);
    yGm.SetGlobalBuffer((__gm__ T *)y);
    gmAssist.SetGlobalBuffer((__gm__ int32_t *)assistGmB32, matrixSize);
    this->ParseTilingData(tilingData, m_tilingData);
    pipe.InitBuffer(inQueueX, bufferNum, matrixSize * sizeof(T));
    pipe.InitBuffer(outQueueY, bufferNum, matrixSize * sizeof(T));
    pipe.InitBuffer(matrixBuf, matrixSize * sizeof(T));
}

template <typename T>
__aicore__ inline void DiagV2B32<T>::Process()
{
    if (GetBlockIdx() >= m_tilingData.realCoreNum) {
        return;
    }

    // load matrix
  #if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    OOMCheckAddrRange(gmAssist.GetPhyAddr(), matrixSize * sizeof(int32_t));
  #endif
    LocalTensor<int32_t> ubAssistInt32 = matrixBuf.Get<int32_t>();
    DataCopy(ubAssistInt32, gmAssist, matrixSize);
    LocalTensor<int16_t> ubAssist;
    this->LocalTensor2NewTensor(ubAssist, ubAssistInt32);

    int64_t loopNum = 0;
    int64_t loopTail = 0;
    if (blockIdx == m_tilingData.realCoreNum - 1) {  // process last block
        loopNum = m_tilingData.tailNum / m_tilingData.matrixRowLength;
        loopTail = m_tilingData.tailNum % m_tilingData.matrixRowLength;
        ProcessLastBlock(ubAssist, loopNum, loopTail);
    } else {
        loopNum = this->CeilDiv(m_tilingData.numPerCore, m_tilingData.matrixRowLength);
        loopTail = m_tilingData.numPerCore % m_tilingData.matrixRowLength;
        ProcessPerBlock(ubAssist, loopNum, loopTail);
    }
}

template <typename T>
__aicore__ inline void DiagV2B32<T>::ProcessPerBlock(
    const LocalTensor<int16_t> &ubAssist, int64_t loopNum, int64_t loopTail)
{
    for (int64_t idx = 0; idx < loopNum; idx++) {
        gmOutOffset = blockIdx * m_tilingData.numPerCore + idx * m_tilingData.matrixRowLength;
        if ((idx == loopNum - 1) && (loopTail != 0)) {
            CopyInB32(idx, loopTail);
            Compute(ubAssist);
            CopyOut(loopTail);
        } else {
            CopyInB32(idx, m_tilingData.matrixRowLength);
            Compute(ubAssist);
            CopyOut(m_tilingData.matrixRowLength);
        }
    }
}

template <typename T>
__aicore__ inline void DiagV2B32<T>::ProcessLastBlock(
    const LocalTensor<int16_t> &ubAssist, int64_t loopNum, int64_t loopTail)
{
    for (int64_t idx = 0; idx < loopNum; idx++) {
        gmOutOffset = blockIdx * m_tilingData.numPerCore + idx * m_tilingData.matrixRowLength;
        CopyInB32(idx, m_tilingData.matrixRowLength);
        Compute(ubAssist);
        CopyOut(m_tilingData.matrixRowLength);
    }
    if (loopTail > 0) {
        gmOutOffset = blockIdx * m_tilingData.numPerCore + loopNum * m_tilingData.matrixRowLength;
        if constexpr (PlatformSocInfo::IsDataCopyPadSupport()) {
            CopyInB32WithPad(loopNum, loopTail);
            Compute(ubAssist);
            CopyOutWithPad(loopTail);
        } else {
            int64_t loopTailAlign = this->CeilDiv(loopTail, perBlockNum) * perBlockNum;
            if (m_tilingData.numOut < perBlockNum || loopTailAlign == loopTail) {
                CopyInB32(loopNum, loopTail);
            } else {
                gmOutOffset -= loopTailAlign - loopTail;
                CopyInB32NotAlign(loopNum, loopTail);
            }
            Compute(ubAssist);
            CopyOut(loopTailAlign);
        }
    }
}

template <typename T>
__aicore__ inline void DiagV2B32<T>::Compute(const LocalTensor<int16_t> &ubAssist)
{
    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    LocalTensor<int16_t> xLocalInt16;
    this->LocalTensor2NewTensor(xLocalInt16, xLocal);
    LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
    LocalTensor<int16_t> yLocalInt16;
    this->LocalTensor2NewTensor(yLocalInt16, yLocal);
    And(yLocalInt16, ubAssist, xLocalInt16, mask, m_tilingData.matrixRowLength, {1, 1, 1, 8, 8, 8});
    inQueueX.FreeTensor(xLocal);

    for (int64_t idx = 2; idx <= m_tilingData.matrixRowLength; idx = idx * 2) {
        AscendC::PipeBarrier<PIPE_V>();
        Or(yLocalInt16[0],
            yLocalInt16[m_tilingData.matrixRowLength * m_tilingData.matrixRowLength * 2 / idx],
            yLocalInt16[0],
            mask,
            m_tilingData.matrixRowLength / idx,
            {1, 1, 1, 8, 8, 8});
    }
    outQueueY.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void DiagV2B32<T>::CopyOut(const int64_t dataCount)
{
    LocalTensor<T> outLocal = outQueueY.DeQue<T>();
    DataCopy(yGm[gmOutOffset], outLocal, dataCount);
    outQueueY.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void DiagV2B32<T>::CopyOutWithPad(int64_t dataCount)
{
    LocalTensor<T> outLocal = outQueueY.DeQue<T>();
    DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
    copyParams.blockLen = dataCount * sizeof(T);
    DataCopyPad(yGm[gmOutOffset], outLocal, copyParams);
    outQueueY.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void DiagV2B32<T>::CopyInB32(int64_t index, int64_t dataCount)
{
    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    this->CopyIn(xLocal, xGm, m_tilingData, index, dataCount);
    inQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void DiagV2B32<T>::CopyInB32NotAlign(int64_t index, int64_t dataCount)
{
    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    this->CopyInNotAlign(xLocal, xGm, m_tilingData, index, dataCount);
    inQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void DiagV2B32<T>::CopyInB32WithPad(int64_t index, int64_t dataCount)
{
    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    this->CopyInWithPad(xLocal, xGm, m_tilingData, index, dataCount);
    inQueueX.EnQue(xLocal);
}

}  // namespace DiagV2
#endif  // DIAG_V2_B64_H
