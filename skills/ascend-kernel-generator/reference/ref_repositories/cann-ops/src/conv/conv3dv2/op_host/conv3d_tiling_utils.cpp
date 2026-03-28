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
 * \file conv3d_tiling_utils.cpp
 * \brief
 */

#include <cmath>
#include "conv3d_tiling_utils.h"

using namespace conv3d_tiling;
namespace optiling {
namespace conv3d_ops_tiling {
uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

void CalcCommFactor(const uint64_t num, const uint32_t numMax, std::vector<uint32_t> &reslist)
{
    uint32_t sqrtMax = static_cast<uint32_t>(sqrt(num));
    for (uint32_t i = 1; i <= sqrtMax; ++i) {
        if (num % i == 0) {
            if (i <= numMax) {
                reslist.emplace_back(i);
            }
            uint32_t right = num / i;
            if (right != i && right <= numMax) {
                reslist.emplace_back(right);
            }
        }
    }
    sort(reslist.begin(), reslist.end());
}

bool IsArrayEqual(std::vector<ConvDtype>& arr1, const std::vector<ConvDtype>& arr2, uint32_t size)
{
    if (arr1.size() < size || arr2.size() < size) {
        return false;
    }
    for (size_t i = 0; i < size; i++) {
        if (arr1[i] != arr2[i]) {
            return false;
        }
    }
    return true;
}

uint64_t AlignB(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return 0;
    }
    return ((a + b - 1) / b) * b;
}

uint64_t InferHiL1(uint64_t inputHoL1, uint64_t hi, uint64_t singlekH, uint32_t dilationH, uint32_t strideH)
{
    uint64_t khDilated = (singlekH - 1) * dilationH + 1;
    uint64_t tmpHiL1 = (inputHoL1 - 1) * strideH + khDilated;
    if (tmpHiL1 > hi) {
        tmpHiL1 = hi;
    }

    return tmpHiL1;
}

uint64_t InferWiL1(uint64_t inputWoL1, uint64_t wi, uint64_t singlekW, uint32_t dilationW, uint32_t strideW)
{
    uint64_t kwDilated = (singlekW - 1) * dilationW + 1;
    uint64_t tmpWiL1 = (inputWoL1 - 1) * strideW + kwDilated;
    if (tmpWiL1 > wi) {
        tmpWiL1 = wi;
    }

    return tmpWiL1;
}

}
}