/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
 * \file diag_v2_t4.h
 * \brief
 */
#ifndef DIAG_V2_T4_H
#define DIAG_V2_T4_H

#include "diag_v2_base.h"

namespace DiagV2 {
using namespace AscendC;

template <typename T>
class DiagV2T4 : public DiagV2Base<T> {
public:
    __aicore__ inline DiagV2T4(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const DiagV2TilingData *tilingData);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void ProcessPerBlock(const int64_t &num);
    __aicore__ inline void CopyIn(const int64_t &dataCount);
    __aicore__ inline void Compute(const int64_t &dataCount);
    __aicore__ inline void CopyOut(const int64_t &dataCount);
    __aicore__ inline void GetAndSetValue(LocalTensor<T> &xLocal, LocalTensor<T> &yLocal, const int64_t &dataCount);

    constexpr static int32_t bufferNum = 2;  // double buffer
    constexpr static int32_t perBlockNum = 32;
    constexpr static int32_t SIZE_1 = 1;
    constexpr static int32_t SIZE_2 = 2;
    constexpr static int32_t SIZE_4 = 4;
    constexpr static int32_t SIZE_8 = 8;

protected:
    TPipe pipe;
    GlobalTensor<T> xGM, yGM;
    TQue<QuePosition::VECIN, bufferNum> inQueueX;
    TQue<QuePosition::VECOUT, bufferNum> outQueueY;

    int32_t blockIdx = 0;
    int64_t blockOffset = 0;
    int64_t gmOutOffset = 0;

    // tiling params
    DiagV2TilingData m_tilingData;
};

template <typename T>
__aicore__ inline void DiagV2T4<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const DiagV2TilingData *tilingData)
{
    blockIdx = GetBlockIdx();
    xGM.SetGlobalBuffer((__gm__ T *)x);
    yGM.SetGlobalBuffer((__gm__ T *)y);
    this->ParseTilingData(tilingData, m_tilingData);

    pipe.InitBuffer(inQueueX,
        bufferNum,
        this->CeilDiv((m_tilingData.numPerCore * m_tilingData.xWidth) * sizeof(T), perBlockNum) * perBlockNum);
    pipe.InitBuffer(
        outQueueY, bufferNum, this->CeilDiv((m_tilingData.numPerCore) * sizeof(T), perBlockNum) * perBlockNum);
}

template <typename T>
__aicore__ inline void DiagV2T4<T>::Process()
{
    blockOffset = blockIdx * m_tilingData.numPerCore * (m_tilingData.xWidth + 1) + m_tilingData.gmOffset;
    if (blockIdx == m_tilingData.realCoreNum - 1) {
        ProcessPerBlock(m_tilingData.tailNum);
    } else {
        ProcessPerBlock(m_tilingData.numPerCore);
    }
}

template <typename T>
__aicore__ inline void DiagV2T4<T>::ProcessPerBlock(const int64_t &num)
{
    int64_t total_num = (m_tilingData.numPerCore - 1) * (m_tilingData.xWidth + 1) + 1;
    if (blockIdx == m_tilingData.realCoreNum - 1) {
        total_num = (m_tilingData.tailNum - 1) * (m_tilingData.xWidth + 1) + 1;
    }
    int64_t alignNum = perBlockNum / sizeof(T);
    int64_t realNum = this->CeilDiv(total_num, alignNum) * alignNum;
    gmOutOffset = blockIdx * m_tilingData.numPerCore;
    CopyIn(realNum);
    Compute(num);
    CopyOut(num);
}

template <typename T>
__aicore__ inline void DiagV2T4<T>::CopyIn(const int64_t &dataCount)
{
    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    DataCopy(xLocal, xGM[blockOffset], dataCount);
    inQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void DiagV2T4<T>::Compute(const int64_t &dataCount)
{
    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
    GetAndSetValue(xLocal, yLocal, dataCount);
    outQueueY.EnQue(yLocal);
    inQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void DiagV2T4<T>::GetAndSetValue(
    LocalTensor<T> &xLocal, LocalTensor<T> &yLocal, const int64_t &dataCount)
{
    switch (sizeof(T)) {
        case SIZE_1: {
            LocalTensor<int8_t> xLocalInt8;
            LocalTensor<int8_t> yLocalInt8;
            this->LocalTensor2NewTensor(xLocalInt8, xLocal);
            this->LocalTensor2NewTensor(yLocalInt8, yLocal);
            for (int64_t idx = 0; idx < dataCount; idx++) {
                yLocalInt8.SetValue(idx, xLocalInt8.GetValue(idx * (m_tilingData.xWidth + 1)));
            }
            break;
        }
        case SIZE_2: {
            LocalTensor<int16_t> xLocalInt16;
            LocalTensor<int16_t> yLocalInt16;
            this->LocalTensor2NewTensor(xLocalInt16, xLocal);
            this->LocalTensor2NewTensor(yLocalInt16, yLocal);
            for (int64_t idx = 0; idx < dataCount; idx++) {
                yLocalInt16.SetValue(idx, xLocalInt16.GetValue(idx * (m_tilingData.xWidth + 1)));
            }
            break;
        }
        case SIZE_4: {
            LocalTensor<int32_t> xLocalInt32;
            LocalTensor<int32_t> yLocalInt32;
            this->LocalTensor2NewTensor(xLocalInt32, xLocal);
            this->LocalTensor2NewTensor(yLocalInt32, yLocal);
            for (int64_t idx = 0; idx < dataCount; idx++) {
                yLocalInt32.SetValue(idx, xLocalInt32.GetValue(idx * (m_tilingData.xWidth + 1)));
            }
            break;
        }
        case SIZE_8: {
            LocalTensor<int64_t> xLocalInt64;
            LocalTensor<int64_t> yLocalInt64;
            this->LocalTensor2NewTensor(xLocalInt64, xLocal);
            this->LocalTensor2NewTensor(yLocalInt64, yLocal);
            for (int64_t idx = 0; idx < dataCount; idx++) {
                yLocalInt64.SetValue(idx, xLocalInt64.GetValue(idx * (m_tilingData.xWidth + 1)));
            }
            break;
        }
        default:
            break;
    }
}

template <typename T>
__aicore__ inline void DiagV2T4<T>::CopyOut(const int64_t &dataCount)
{
    LocalTensor<T> yLocal = outQueueY.DeQue<T>();
    int64_t alignNum = perBlockNum / sizeof(T);
    int64_t realNum = this->CeilDiv(dataCount, alignNum) * alignNum;
    DataCopy(yGM[gmOutOffset], yLocal, realNum);
    outQueueY.FreeTensor(yLocal);
}
}  // namespace DiagV2

#endif  // DIAG_V2_T4_H