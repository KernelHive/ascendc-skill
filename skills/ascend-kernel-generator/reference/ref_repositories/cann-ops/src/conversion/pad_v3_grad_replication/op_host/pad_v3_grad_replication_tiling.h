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
 * @file pad_v3_grad_replication_tiling.h
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_PAD_V3_GRAD_REPLICATION_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_PAD_V3_GRAD_REPLICATION_H_
#include <cstdint>
#include "register/tilingdata_base.h"


namespace optiling {
constexpr uint32_t X_INPUT_INDEX = 0;
constexpr uint32_t PADDING_INPUT_INDEX = 1;
constexpr uint32_t Y_OUTPUT_INDEX = 0;
constexpr uint32_t VIRTUAL_INPUT_DIM = 4;
constexpr uint32_t PADDING_LENGTH = 6;
constexpr uint32_t PADDING_PAIR_NUM = 3;
constexpr uint32_t PADDING_API_LENGTH = 10;

constexpr uint32_t BYTE_32BLOCK = 32;
constexpr uint32_t BYTE_64BLOCK = 64;
constexpr uint32_t MAX_UINT16 = 65535;
constexpr uint16_t PARAM_LIMIT_4095 = 4095;
constexpr uint32_t ADD_TERSOR_NUM = 3;
constexpr uint32_t DIM_4D = 4;
constexpr uint32_t DIM_5D = 5;
constexpr uint32_t B32_BYTE_SIZE = 4;
constexpr uint32_t DIM_0 = 0;
constexpr uint32_t DIM_1 = 1;
constexpr uint32_t DIM_2 = 2;
constexpr uint32_t DIM_3 = 3;
constexpr uint32_t DIM_4 = 4;
constexpr uint32_t DIM_5 = 5;
constexpr uint32_t PAIR = 2;

constexpr uint32_t REPLICATION_FP32_KEY = 1;
constexpr uint32_t REPLICATION_FP16_KEY = 2;
constexpr uint32_t REPLICATION_BF16_KEY = 3;

BEGIN_TILING_DATA_DEF(EdgeTiling)
    TILING_DATA_FIELD_DEF(uint16_t, edgeCount);
    TILING_DATA_FIELD_DEF(uint64_t, tileCount);
    TILING_DATA_FIELD_DEF(uint16_t, additionalCount);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(EdgeTilingOp, EdgeTiling)

BEGIN_TILING_DATA_DEF(PadV3GradReplicationTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, addTensorBlockNum);  // 32B块数量或64B块数量（b16）
    TILING_DATA_FIELD_DEF(uint32_t, addTensorByteSize);
    TILING_DATA_FIELD_DEF(uint32_t, addTensorSize);
    TILING_DATA_FIELD_DEF(uint32_t, moveTensorBlockNum);
    TILING_DATA_FIELD_DEF(uint32_t, moveTensorByteSize);
    TILING_DATA_FIELD_DEF(uint32_t, moveTensorSize);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, VIRTUAL_INPUT_DIM, inputShape);
    TILING_DATA_FIELD_DEF(uint64_t, inputSize);
    TILING_DATA_FIELD_DEF(uint64_t, cubeInputSize);
    TILING_DATA_FIELD_DEF(uint32_t, layerInputSize);
    TILING_DATA_FIELD_DEF(uint32_t, cubeNumEachCore);
    TILING_DATA_FIELD_DEF(uint32_t, realUsedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, cubeNumLastCore);

    TILING_DATA_FIELD_DEF_ARR(uint32_t, VIRTUAL_INPUT_DIM, outputShape);
    TILING_DATA_FIELD_DEF(uint64_t, outputSize);
    TILING_DATA_FIELD_DEF(uint32_t, cubeOutputSize);
    TILING_DATA_FIELD_DEF(uint32_t, layerOutputSize);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, PADDING_LENGTH, paddings);
    
    TILING_DATA_FIELD_DEF(uint32_t, topSize);
    TILING_DATA_FIELD_DEF(uint64_t, totalTopInputSizeEachCube);
    TILING_DATA_FIELD_DEF(uint32_t, leftSize);
    TILING_DATA_FIELD_DEF(uint64_t, totalLeftInputSizeEachCube);
    TILING_DATA_FIELD_DEF(uint32_t, innerRowLength);
    TILING_DATA_FIELD_DEF(uint32_t, topToBottomSize);
    TILING_DATA_FIELD_DEF(uint64_t, topResultSize);
    TILING_DATA_FIELD_DEF(uint32_t, leftToRightSize);
    TILING_DATA_FIELD_DEF(uint64_t, leftResultSize);
    TILING_DATA_FIELD_DEF(uint64_t, workspaceSize);

    TILING_DATA_FIELD_DEF_STRUCT(EdgeTiling, topTiling);
    TILING_DATA_FIELD_DEF_STRUCT(EdgeTiling, leftTiling);
    TILING_DATA_FIELD_DEF_STRUCT(EdgeTiling, cornerTiling);
    TILING_DATA_FIELD_DEF_STRUCT(EdgeTiling, innerTiling);
    TILING_DATA_FIELD_DEF_STRUCT(EdgeTiling, paddingLayerTiling);
    TILING_DATA_FIELD_DEF_STRUCT(EdgeTiling, topTilingLastCore);
    TILING_DATA_FIELD_DEF_STRUCT(EdgeTiling, leftTilingLastCore);
    TILING_DATA_FIELD_DEF_STRUCT(EdgeTiling, cornerTilingLastCore);
    TILING_DATA_FIELD_DEF_STRUCT(EdgeTiling, paddingLayerTilingLastCore);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PadV3GradReplication, PadV3GradReplicationTilingData)

struct PadV3GradReplicationCompileInfo {
    uint32_t vectorCoreNum;
    uint32_t sysWorkspaceByteSize;
    uint32_t ubByteSize;
};
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_PAD_V3_GRAD_REPLICATION_H_