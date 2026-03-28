/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file operator_desc.cpp
 */
#include "operator_desc.h"

#include "common.h"

using namespace std;

OperatorDesc::OperatorDesc(std::string opType) : opType(std::move(opType))
{
    opAttr = aclopCreateAttr();
}

OperatorDesc::~OperatorDesc()
{
    for (auto *desc : inputDesc) {
        aclDestroyTensorDesc(desc);
    }

    for (auto *desc : outputDesc) {
        aclDestroyTensorDesc(desc);
    }

    aclopDestroyAttr(opAttr);
}

OperatorDesc &OperatorDesc::AddInputTensorDesc(aclDataType dataType, int numDims, const int64_t *dims, aclFormat format)
{
    aclTensorDesc *desc = aclCreateTensorDesc(dataType, numDims, dims, format);
    if (desc == nullptr) {
        ERROR_LOG("create tensor failed");
        return *this;
    }

    inputDesc.emplace_back(desc);
    return *this;
}

OperatorDesc &OperatorDesc::AddOutputTensorDesc(aclDataType dataType, int numDims, const int64_t *dims,
                                                aclFormat format)
{
    aclTensorDesc *desc = aclCreateTensorDesc(dataType, numDims, dims, format);
    if (desc == nullptr) {
        ERROR_LOG("create tensor failed");
        return *this;
    }

    outputDesc.emplace_back(desc);
    return *this;
}
