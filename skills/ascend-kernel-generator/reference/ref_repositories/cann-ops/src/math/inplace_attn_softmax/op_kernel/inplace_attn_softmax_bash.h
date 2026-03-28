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

#ifndef INPLACE_ATTN_SOFTMAX_BASE_H
#define INPLACE_ATTN_SOFTMAX_BASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
namespace InplaceAttnSoftmaxOpt {
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t BLOCK_SIZE = 32;

// 单输入场景，一个tile需要的偏置参数
struct InplaceAttnSoftmaxOffsetParam {
    uint64_t tmpVecGmOffset;
};
template <typename inType, typename outType, bool isCast, bool isBigshape>
class InplaceAttnSoftmaxBase {
public:
    __aicore__ inline InplaceAttnSoftmaxBase()
    {}

    __aicore__ inline void ParseTilingData(const InplaceAttnSoftmaxTilingData *tilingData)
    {
        tilingData_.rowLen = tilingData->rowLen;                            // 多少行数据
        tilingData_.colLen = tilingData->colLen;                            // 列数，对输入x的一半
        tilingData_.rowLenPerHeadCore = tilingData->rowLenPerHeadCore;      // 每核处理的行数
        tilingData_.rowLenPerTailCore = tilingData->rowLenPerTailCore;      // 每核处理的行数
        tilingData_.basicRowLenHeadCore = tilingData->basicRowLenHeadCore;  // 头核每次计算的行数
        tilingData_.basicRowLenTailCore = tilingData->basicRowLenTailCore;  // 尾核每次计算的行数
        tilingData_.basicColLen = tilingData->basicColLen;                  // 每次计算的列数
        tilingData_.headCoreNum = tilingData->headCoreNum;                  // 使用的head核数
        tilingData_.realCoreNum = tilingData->realCoreNum;                  // 使用的核数
    }

    __aicore__ inline void InitParamsComm()
    {
        colLen = tilingData_.colLen;
        basicColLen = tilingData_.basicColLen;

        coreIdx = static_cast<uint32_t>(GetBlockIdx());
        headCoreNum = tilingData_.headCoreNum;
        if (coreIdx < headCoreNum) {
            rowLenPerCore = tilingData_.rowLenPerHeadCore;
            basicRowLen = tilingData_.basicRowLenHeadCore;
        } else if (coreIdx >= headCoreNum && coreIdx < tilingData_.realCoreNum) {
            rowLenPerCore = tilingData_.rowLenPerTailCore;
            basicRowLen = tilingData_.basicRowLenTailCore;
        } 
        if constexpr(isBigshape) {
            rowLoop = CeilDiv(rowLenPerCore, basicRowLen);
            colLoop = CeilDiv(colLen, basicColLen);
            lastcolLen = Ceilabs(colLen, basicColLen);
            rightPadding = basicColLen - lastcolLen;
        } else 
        {
            if (coreIdx < headCoreNum) {
                baseRow = coreIdx * rowLenPerCore;
            } else if (coreIdx >= headCoreNum && coreIdx < tilingData_.realCoreNum) {
                baseRow = headCoreNum * tilingData_.rowLenPerHeadCore + (coreIdx - headCoreNum) * rowLenPerCore;
            } 
            rowLoop = CeilDiv(rowLenPerCore, basicRowLen);

            uint32_t alignedNum = BLOCK_SIZE / sizeof(inType);
            sizeHalfLen = AlignUp(basicColLen, alignedNum);
            // 若basicColLen比32B还小 -> sizeHalfLen == 0 -> sizeHalfLen直接按32B字节算
            tileLength = basicRowLen * (sizeHalfLen == 0 ? (BLOCK_SIZE / sizeof(inType)) : sizeHalfLen);
            rightPadding = sizeHalfLen - basicColLen;
        }
    }

    template <typename T>
    __aicore__ inline T CeilDiv(T x, T y)
    {
        return y == 0 ? 0 : (x + y - 1) / y;
    }

    template <typename T>
    __aicore__ inline T AlignUp(T num, T div)
    {
        return (div == 0) ? 0 : (num + div - 1) / div * div;
    }

    template <typename T>
    __aicore__ inline T Ceilabs(T x, T y)
    {
        if(x > y){
            return x % y;
        }else {
            return y - x;
        }
    }

public:
    TPipe *pPipe = nullptr;
    /* tiling data */
    InplaceAttnSoftmaxTilingData tilingData_;
    SoftMaxTiling softmaxTilingData_;
    /* variable */
    uint32_t rowLen;
    uint32_t colLen;
    uint32_t ridx;
    uint32_t rowLenPerHeadCore;
    uint32_t rowLenPerTailCore;
    uint32_t basicRowLen;
    uint32_t rowLenPerCore;
    uint32_t basicRowLenHeadCore;
    uint32_t basicRowLenTailCore;
    uint32_t basicColLen;
    uint32_t headCoreNum;
    uint32_t realCoreNum;
    uint32_t outAlignLen;
    uint32_t sizeHalfLen;
    uint32_t outLen;
    uint8_t rightPadding;
    bool isPad = false;
    uint16_t blockUnit;
    uint32_t coreIdx;
    uint32_t rowLoop = 1;
    uint32_t colLoop = 1;
    uint32_t lastcolLen = 0;
    uint32_t baseRow = 0;  // 记录开始处理的行数
    uint16_t basicRowLenCal;
    uint64_t tileLength;
    InplaceAttnSoftmaxOffsetParam offsetParam;
};
}  
#endif  