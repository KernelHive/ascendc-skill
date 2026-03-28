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
 * \file cube_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_TILING_NEW_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_TILING_NEW_H_

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

// #include "external/graph/operator.h"
#include "op_tiling.h"

namespace optiling {
const size_t kConv2dDimNumLimit = 4;
const int32_t kConv2dNDim = 0;
const int32_t kConv2dHDim = 2;
const int32_t kConv2dWDim = 3;

/*
 * @brief: tiling function of cube category operators
 * @param [in] curShape: execution time shape info
 * @param [in] opInfo: compile time generated info of operator
 * @param [out] runInfo: result data
 * @return int: tiling id
 */
bool cube_tiling(const std::string &op_type, const std::vector<int64_t> &input_shape,
                 const std::vector<int64_t> &var_value);
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_TILING_H_