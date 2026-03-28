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
 * \file diag_v2_base.h
 * \brief
 */
#ifndef DIAG_V2_BASE_H
#define DIAG_V2_BASE_H

#include "diag_v2_assist.h"
#include "platform.h"

namespace DiagV2 {
using namespace AscendC;

enum DiagV2TilingKey {
    ASSIST_SIZE_1 = 2101,
    ASSIST_SIZE_2 = 2102,
    ASSIST_SIZE_4 = 2103,
    ASSIST_SIZE_8 = 2104,
    ASSIST_SIZE_16 = 2105,

    SCALAR_SIZE_1 = 2401,
    SCALAR_SIZE_2 = 2402,
    SCALAR_SIZE_4 = 2403,
    SCALAR_SIZE_8 = 2404
};

template <typename T>
class DiagV2Base {
public:
    __aicore__ inline DiagV2Base(){};

protected:
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilDiv(T1 a, T2 b)
    {
        if (b == 0) {
            return 0;
        }
        return (a + b - 1) / b;
    };
    template <typename T1, typename T2>
    __aicore__ inline void LocalTensor2NewTensor(LocalTensor<T1> &tensor_new, const LocalTensor<T2> &tensor_old)
    {
        tensor_new = tensor_old.template ReinterpretCast<T1>();
    };
    __aicore__ inline void ParseTilingData(const DiagV2TilingData *tilingData, DiagV2TilingData &m_tilingData);
    __aicore__ inline void CopyIn(LocalTensor<T> &xLocal, const GlobalTensor<T> &xGm,
        const DiagV2TilingData &m_tilingData, const int64_t &index, const int64_t &dataCount);
    __aicore__ inline void CopyInNotAlign(LocalTensor<T> &xLocal, const GlobalTensor<T> &xGm,
        const DiagV2TilingData &m_tilingData, const int64_t &index, const int64_t &dataCount);
    __aicore__ inline void CopyInWithPad(LocalTensor<T> &xLocal, const GlobalTensor<T> &xGm,
        const DiagV2TilingData &m_tilingData, const int64_t &index, const int64_t &dataCount);

private:
    constexpr static int32_t blockSize = 32;
    constexpr static int32_t maxBlockStride = 65535;
    int32_t perBlockNum = blockSize / sizeof(T);
};

template <typename T>
__aicore__ inline void DiagV2Base<T>::ParseTilingData(const DiagV2TilingData *tilingData,
                                                      DiagV2TilingData &m_tilingData)
{
    m_tilingData.xWidth = tilingData->xWidth;
    m_tilingData.xHeight = tilingData->xHeight;
    m_tilingData.gmOffset = tilingData->gmOffset;
    m_tilingData.numOut = tilingData->numOut;
    m_tilingData.realCoreNum = tilingData->realCoreNum;
    m_tilingData.numPerCore = tilingData->numPerCore;
    m_tilingData.tailNum = tilingData->tailNum;
    m_tilingData.tilingKey = tilingData->tilingKey;
    m_tilingData.matrixRowLength = tilingData->matrixRowLength;
}

template <typename T>
__aicore__ inline void DiagV2Base<T>::CopyInWithPad(LocalTensor<T> &xLocal, const GlobalTensor<T> &xGm,
    const DiagV2TilingData &m_tilingData, const int64_t &index, const int64_t &dataCount)
{
    int64_t blockOffset =
        (GetBlockIdx() * m_tilingData.numPerCore + index * m_tilingData.matrixRowLength) * (m_tilingData.xWidth + 1) +
        m_tilingData.gmOffset;

    for (int64_t idx = 0; idx < dataCount; idx++) {
        DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
        copyParams.blockLen = dataCount * sizeof(T);
        DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
        DataCopyPad(xLocal[idx * m_tilingData.matrixRowLength], xGm[blockOffset + idx * m_tilingData.xWidth],
                    copyParams, padParams);
    }
}

template <typename T>
__aicore__ inline void DiagV2Base<T>::CopyInNotAlign(LocalTensor<T>& xLocal, const GlobalTensor<T>& xGm,
                                                     const DiagV2TilingData& m_tilingData, const int64_t& index,
                                                     const int64_t& dataCount) {
    int64_t loopTailAlign = this->CeilDiv(dataCount, perBlockNum) * perBlockNum;
    int64_t blockOffset = (GetBlockIdx() * m_tilingData.numPerCore + index * m_tilingData.matrixRowLength -
                           (loopTailAlign - dataCount)) * (m_tilingData.xWidth + 1) + m_tilingData.gmOffset;
    for (int64_t idx = 0; idx < loopTailAlign; idx++) {
        DataCopy(xLocal[idx * m_tilingData.matrixRowLength], xGm[blockOffset + idx * m_tilingData.xWidth],
                 loopTailAlign);
    }
}

template <typename T>
__aicore__ inline void DiagV2Base<T>::CopyIn(LocalTensor<T> &xLocal, const GlobalTensor<T> &xGm,
    const DiagV2TilingData &m_tilingData, const int64_t &index, const int64_t &dataCount)
{
    int64_t blockOffset =
        (GetBlockIdx() * m_tilingData.numPerCore + index * m_tilingData.matrixRowLength) * (m_tilingData.xWidth + 1) +
        m_tilingData.gmOffset;
    int64_t loopTailAlign = this->CeilDiv(dataCount, perBlockNum) * perBlockNum;
    if ((m_tilingData.xWidth - loopTailAlign) % perBlockNum == 0 &&
        (m_tilingData.xWidth - loopTailAlign) / perBlockNum <= maxBlockStride) {
        int64_t srcStride = (m_tilingData.xWidth - loopTailAlign) / perBlockNum;
        int64_t dstStride = (m_tilingData.matrixRowLength - loopTailAlign) / perBlockNum;
        DataCopyParams intriParams;
        intriParams.blockCount = dataCount;
        intriParams.blockLen = loopTailAlign / perBlockNum;
        intriParams.srcStride = srcStride;
        intriParams.dstStride = dstStride;
        DataCopy(xLocal, xGm[blockOffset], intriParams);
    } else {
        for (int64_t idx = 0; idx < dataCount; idx++) {
            DataCopy(xLocal[idx * m_tilingData.matrixRowLength], xGm[blockOffset + idx * m_tilingData.xWidth],
                     loopTailAlign);
        }
    }
}

}  // namespace DiagV2
 
#endif  // DIAG_V2_BASE_H
