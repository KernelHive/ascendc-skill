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
 * \file weight_quant_batch_matmul_v2_tiling_tool.cpp
 * \brief
 */
#include "weight_quant_batch_matmul_v2_tiling_tool.h"

namespace optiling {

uint64_t GetBlockAlignSizeByDataType(ge::DataType dtype)
{
    if (dtype == ge::DT_INT4) {
        return ONE_BLK_SIZE + ONE_BLK_SIZE;
    } else {
        return ONE_BLK_SIZE / static_cast<uint32_t>(ge::GetSizeByDataType(dtype));
    }
}

uint64_t GetShapeSizeWithDataType(uint64_t shapeSize, ge::DataType dtype)
{
    if (dtype == ge::DT_INT4) {
        return (shapeSize + 1) >> 1;
    } else {
        return shapeSize * static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
    }
}

bool CheckOptionalInputByShape(const gert::StorageShape *storageShape)
{
    return storageShape != nullptr && storageShape->GetStorageShape().GetShapeSize() != 0;
}

const std::map<ge::DataType, matmul_tiling::DataType> DTYPE_MAP = {
    {ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16}, {ge::DT_FLOAT, matmul_tiling::DataType::DT_FLOAT},
    {ge::DT_INT8, matmul_tiling::DataType::DT_INT8},       {ge::DT_BF16, matmul_tiling::DataType::DT_BF16},
    {ge::DT_INT4, matmul_tiling::DataType::DT_INT4},
};

matmul_tiling::DataType GetMatmulTilingDtype(ge::DataType dtype)
{
    auto it = DTYPE_MAP.find(dtype);
    // impossible to get runtime error
    return it != DTYPE_MAP.end() ? it->second : matmul_tiling::DataType::DT_FLOAT16;
}

ge::Format GetInputStorageFormat(const gert::TilingContext *context, size_t id)
{
    auto desc = context->GetInputDesc(id);
    OP_TILING_CHECK(
        desc == nullptr,
        VECTOR_INNER_ERR_REPORT_TILIING("weight_quant_batch_matmul_v2_tiling", "get input[%zu] Desc is null!", id),
        return ge::FORMAT_NULL);
    return static_cast<ge::Format>(GetPrimaryFormat(desc->GetStorageFormat()));
}
}  // namespace optiling
