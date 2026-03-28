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
 * @file operator_desc.h
 */
#ifndef OPERATOR_DESC_H
#define OPERATOR_DESC_H

#include <string>
#include <vector>

#include "acl/acl.h"

/**
 * Op description
 */
struct OperatorDesc {
    /**
     * Constructor
     * @param [in] opType: op type
     */
    explicit OperatorDesc(std::string opType);

    /**
     * Destructor
     */
    virtual ~OperatorDesc();

    /**
     * Add an input tensor description
     * @param [in] dataType: data type
     * @param [in] numDims: number of dims
     * @param [in] dims: dims
     * @param [in] format: format
     * @return OperatorDesc
     */
    OperatorDesc &AddInputTensorDesc(aclDataType dataType, int numDims, const int64_t *dims, aclFormat format);

    /**
     * Add an output tensor description
     * @param [in] dataType: data type
     * @param [in] numDims: number of dims
     * @param [in] dims: dims
     * @param [in] format: format
     * @return OperatorDesc
     */
    OperatorDesc &AddOutputTensorDesc(aclDataType dataType, int numDims, const int64_t *dims, aclFormat format);

    std::string opType;
    std::vector<aclTensorDesc *> inputDesc;
    std::vector<aclTensorDesc *> outputDesc;
    aclopAttr *opAttr;
};

#endif // OPERATOR_DESC_H
