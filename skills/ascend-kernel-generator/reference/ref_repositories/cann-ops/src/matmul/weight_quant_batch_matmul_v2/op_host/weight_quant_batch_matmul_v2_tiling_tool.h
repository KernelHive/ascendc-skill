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
 * \file weight_quant_batch_matmul_v2_tiling_tool.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_TOOL_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_TOOL_H

#include "cache_tiling.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_templates_registry.h"
#include "tiling/tiling_type.h"

using matmul_tiling::MatrixTraverse;
using AscendC::BLOCK_CUBE;
using AscendC::ONE_BLK_SIZE;

namespace optiling {
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
}

namespace optiling {

constexpr uint64_t BASIC_BLOCK = 512UL;

template <typename T1, typename T2>
T2 CalcTailSize(T1 num1, T2 num2)
{
    if (num2 == 0) {
        return 0;
    }

    T1 mod = num1 % static_cast<T1>(num2);
    return mod != 0 ? static_cast<T2>(mod) : num2;
}

uint64_t GetBlockAlignSizeByDataType(ge::DataType dtype);

uint64_t GetShapeSizeWithDataType(uint64_t shapeSize, ge::DataType dtype);

bool CheckOptionalInputByShape(const gert::StorageShape *storageShape);

matmul_tiling::DataType GetMatmulTilingDtype(ge::DataType dtype);

ge::Format GetInputStorageFormat(const gert::TilingContext *context, size_t id);
}  // namespace optiling
#endif  // WEIGHT_QUANT_BATCH_MATMUL_V2_TOOL_H
