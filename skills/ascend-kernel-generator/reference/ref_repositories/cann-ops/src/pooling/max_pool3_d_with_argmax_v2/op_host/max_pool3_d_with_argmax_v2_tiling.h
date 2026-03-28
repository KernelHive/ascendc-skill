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
 * \file max_pool3_d_with_argmax_v2_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MAX_POOL3D_WITH_AGRMAX_V2_TILING_BASE_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MAX_POOL3D_WITH_AGRMAX_V2_TILING_BASE_H_

#define OPS_UTILS_LOG_SUB_MOD_NAME "OP_MAXPOOL3DWITHARGMAX"
#define OPS_UTILS_LOG_PACKAGE_TYPE "OP_MAXPOOL3DWITHARGMAX"
#include <array>

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_base.h"

using namespace std;

namespace ops {

#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
    if ((ptr) == nullptr) {                                                                        \
        std::printf("nullptr error!");                                                             \
        return ge::GRAPH_FAILED;                                                                     \
    }

const gert::Shape g_vec_1_shape = {1};
/**
 * Ensure that the returned shape is non-scalar.
 * When the dim num of shape is 0, this shape is considered to express a scalar.
 * This function returns the original shape when it receives a non-scalar shape, 
 * and returns the vector shape that returns a {1} when it receives a scalar shape
 * @param in_shape input shape
 * @return non-scalar shape
 */
inline const gert::Shape &EnsureNotScalar(const gert::Shape &in_shape) {
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}
}

namespace optiling {
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)

#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)

const int DHW_DIMS = 3;
const int DHW_PAD_DIMS = 6;
const int MAX_CORE_NUM = 50;
const uint32_t D_DIM = 0;
const uint32_t H_DIM = 1;
const uint32_t W_DIM = 2;
const uint32_t MAX_DIV = 2;
const uint32_t NCHW_CONV_ADDR_LIST_SIZE = 16;
const uint32_t MIN_TRANSPOSE_ROWS = 16;
const uint32_t INT64_FP32 = 2;
const uint32_t BINARY_SEARCH_COEFF = 2;
const uint32_t BLOCK_LEN_FP32 = 8;
const uint32_t BLOCK_LEN_FP16 = 16;

BEGIN_TILING_DATA_DEF(MaxPool3DWithArgmaxV2TilingData)
    TILING_DATA_FIELD_DEF_ARR(uint64_t, DHW_DIMS, inputShapes);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, DHW_DIMS, outShapes);
    TILING_DATA_FIELD_DEF(uint64_t, kD);
    TILING_DATA_FIELD_DEF(uint64_t, kW);
    TILING_DATA_FIELD_DEF(uint64_t, kH);
    TILING_DATA_FIELD_DEF(uint64_t, sD);
    TILING_DATA_FIELD_DEF(uint64_t, sW);
    TILING_DATA_FIELD_DEF(uint64_t, sH);
    TILING_DATA_FIELD_DEF(uint64_t, pD);
    TILING_DATA_FIELD_DEF(uint64_t, pW);
    TILING_DATA_FIELD_DEF(uint64_t, pH);
    TILING_DATA_FIELD_DEF(uint64_t, dD);
    TILING_DATA_FIELD_DEF(uint64_t, dW);
    TILING_DATA_FIELD_DEF(uint64_t, dH);
    TILING_DATA_FIELD_DEF(float, minFloat);
    TILING_DATA_FIELD_DEF(uint64_t, sizeUb1);
    TILING_DATA_FIELD_DEF(uint64_t, sizeUb2);
    TILING_DATA_FIELD_DEF(uint64_t, sizeValues);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(MaxPool3DWithArgmaxV2NoSplitTilingData)
    TILING_DATA_FIELD_DEF_ARR(uint64_t, DHW_DIMS, inputShapes);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, DHW_DIMS, outShapes);
    TILING_DATA_FIELD_DEF(uint64_t, kD);
    TILING_DATA_FIELD_DEF(uint64_t, kW);
    TILING_DATA_FIELD_DEF(uint64_t, kH);
    TILING_DATA_FIELD_DEF(uint64_t, sD);
    TILING_DATA_FIELD_DEF(uint64_t, sW);
    TILING_DATA_FIELD_DEF(uint64_t, sH);
    TILING_DATA_FIELD_DEF(uint64_t, pD);
    TILING_DATA_FIELD_DEF(uint64_t, pW);
    TILING_DATA_FIELD_DEF(uint64_t, pH);
    TILING_DATA_FIELD_DEF(uint64_t, dD);
    TILING_DATA_FIELD_DEF(uint64_t, dW);
    TILING_DATA_FIELD_DEF(uint64_t, dH);
    TILING_DATA_FIELD_DEF(uint64_t, batchesPerCore);
    TILING_DATA_FIELD_DEF(uint64_t, leftOverBatches);
    TILING_DATA_FIELD_DEF(uint64_t, partD);
    TILING_DATA_FIELD_DEF(uint64_t, partH);
    TILING_DATA_FIELD_DEF(uint64_t, partW);
    TILING_DATA_FIELD_DEF(float, minFloat);
    TILING_DATA_FIELD_DEF(uint64_t, ceilD);
    TILING_DATA_FIELD_DEF(uint64_t, ceilH);
    TILING_DATA_FIELD_DEF(uint64_t, ceilW);
    TILING_DATA_FIELD_DEF(uint64_t, sizeUb1);
    TILING_DATA_FIELD_DEF(uint64_t, sizeUb2);
    TILING_DATA_FIELD_DEF(uint64_t, sizeValues);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(MaxPool3DWithArgmaxV2SplitDTilingData)
    TILING_DATA_FIELD_DEF_ARR(uint64_t, DHW_DIMS, inputShapes);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, DHW_DIMS, outShapes);
    TILING_DATA_FIELD_DEF(uint64_t, kD);
    TILING_DATA_FIELD_DEF(uint64_t, kW);
    TILING_DATA_FIELD_DEF(uint64_t, kH);
    TILING_DATA_FIELD_DEF(uint64_t, sD);
    TILING_DATA_FIELD_DEF(uint64_t, sW);
    TILING_DATA_FIELD_DEF(uint64_t, sH);
    TILING_DATA_FIELD_DEF(uint64_t, pD);
    TILING_DATA_FIELD_DEF(uint64_t, pW);
    TILING_DATA_FIELD_DEF(uint64_t, pH);
    TILING_DATA_FIELD_DEF(uint64_t, dD);
    TILING_DATA_FIELD_DEF(uint64_t, dW);
    TILING_DATA_FIELD_DEF(uint64_t, dH);
    TILING_DATA_FIELD_DEF(uint64_t, batchesPerCore);
    TILING_DATA_FIELD_DEF(uint64_t, leftOverBatches);
    TILING_DATA_FIELD_DEF(uint64_t, partD);
    TILING_DATA_FIELD_DEF(uint64_t, partH);
    TILING_DATA_FIELD_DEF(uint64_t, partW);
    TILING_DATA_FIELD_DEF(float, minFloat);
    TILING_DATA_FIELD_DEF(uint64_t, ceilD);
    TILING_DATA_FIELD_DEF(uint64_t, ceilH);
    TILING_DATA_FIELD_DEF(uint64_t, ceilW);
    TILING_DATA_FIELD_DEF(uint64_t, sizeUb1);
    TILING_DATA_FIELD_DEF(uint64_t, sizeUb2);
    TILING_DATA_FIELD_DEF(uint64_t, sizeValues);
    TILING_DATA_FIELD_DEF(uint64_t, partsPerCore);
    TILING_DATA_FIELD_DEF(uint64_t, leftOverParts);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_CORE_NUM, splitPointDIn);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_CORE_NUM, splitPointDOut);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_CORE_NUM, sizeIn);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_CORE_NUM, batchStart);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(MaxPool3DWithArgmaxV2SplitHTilingData)
    TILING_DATA_FIELD_DEF_ARR(uint64_t, DHW_DIMS, inputShapes);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, DHW_DIMS, outShapes);
    TILING_DATA_FIELD_DEF(uint64_t, kD);
    TILING_DATA_FIELD_DEF(uint64_t, kW);
    TILING_DATA_FIELD_DEF(uint64_t, kH);
    TILING_DATA_FIELD_DEF(uint64_t, sD);
    TILING_DATA_FIELD_DEF(uint64_t, sW);
    TILING_DATA_FIELD_DEF(uint64_t, sH);
    TILING_DATA_FIELD_DEF(uint64_t, pD);
    TILING_DATA_FIELD_DEF(uint64_t, pW);
    TILING_DATA_FIELD_DEF(uint64_t, pH);
    TILING_DATA_FIELD_DEF(uint64_t, dD);
    TILING_DATA_FIELD_DEF(uint64_t, dW);
    TILING_DATA_FIELD_DEF(uint64_t, dH);
    TILING_DATA_FIELD_DEF(uint64_t, batchesPerCore);
    TILING_DATA_FIELD_DEF(uint64_t, leftOverBatches);
    TILING_DATA_FIELD_DEF(uint64_t, partD);
    TILING_DATA_FIELD_DEF(uint64_t, partH);
    TILING_DATA_FIELD_DEF(uint64_t, partW);
    TILING_DATA_FIELD_DEF(float, minFloat);
    TILING_DATA_FIELD_DEF(uint64_t, ceilD);
    TILING_DATA_FIELD_DEF(uint64_t, ceilH);
    TILING_DATA_FIELD_DEF(uint64_t, ceilW);
    TILING_DATA_FIELD_DEF(uint64_t, sizeUb1);
    TILING_DATA_FIELD_DEF(uint64_t, sizeUb2);
    TILING_DATA_FIELD_DEF(uint64_t, sizeValues);
    TILING_DATA_FIELD_DEF(uint64_t, partsPerCore);
    TILING_DATA_FIELD_DEF(uint64_t, leftOverParts);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_CORE_NUM, splitPointDIn);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_CORE_NUM, splitPointHIn);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_CORE_NUM, splitPointDOut);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_CORE_NUM, splitPointHOut);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_CORE_NUM, sizeIn);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_CORE_NUM, batchStart);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(MaxPool3DWithArgmaxV2SplitWTilingData)
    TILING_DATA_FIELD_DEF_ARR(uint64_t, DHW_DIMS, inputShapes);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, DHW_DIMS, outShapes);
    TILING_DATA_FIELD_DEF(uint64_t, kD);
    TILING_DATA_FIELD_DEF(uint64_t, kW);
    TILING_DATA_FIELD_DEF(uint64_t, kH);
    TILING_DATA_FIELD_DEF(uint64_t, sD);
    TILING_DATA_FIELD_DEF(uint64_t, sW);
    TILING_DATA_FIELD_DEF(uint64_t, sH);
    TILING_DATA_FIELD_DEF(uint64_t, pD);
    TILING_DATA_FIELD_DEF(uint64_t, pW);
    TILING_DATA_FIELD_DEF(uint64_t, pH);
    TILING_DATA_FIELD_DEF(uint64_t, dD);
    TILING_DATA_FIELD_DEF(uint64_t, dW);
    TILING_DATA_FIELD_DEF(uint64_t, dH);
    TILING_DATA_FIELD_DEF(uint64_t, batchesPerCore);
    TILING_DATA_FIELD_DEF(uint64_t, leftOverBatches);
    TILING_DATA_FIELD_DEF(uint64_t, partsPerCore);
    TILING_DATA_FIELD_DEF(uint64_t, leftOverParts);
    TILING_DATA_FIELD_DEF(uint64_t, partD);
    TILING_DATA_FIELD_DEF(uint64_t, partH);
    TILING_DATA_FIELD_DEF(uint64_t, partW);
    TILING_DATA_FIELD_DEF(float, minFloat);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_CORE_NUM, splitPointDIn);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_CORE_NUM, splitPointHIn);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_CORE_NUM, splitPointWIn);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_CORE_NUM, splitPointDOut);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_CORE_NUM, splitPointHOut);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_CORE_NUM, splitPointWOut);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_CORE_NUM, sizeIn);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_CORE_NUM, batchStart);
    TILING_DATA_FIELD_DEF(uint64_t, ceilD);
    TILING_DATA_FIELD_DEF(uint64_t, ceilH);
    TILING_DATA_FIELD_DEF(uint64_t, ceilW);
    TILING_DATA_FIELD_DEF(uint64_t, sizeUb1);
    TILING_DATA_FIELD_DEF(uint64_t, sizeUb2);
    TILING_DATA_FIELD_DEF(uint64_t, sizeValues);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(MaxPool3DWithArgmaxV2HugeKernelTilingData)
    TILING_DATA_FIELD_DEF_ARR(uint64_t, DHW_DIMS, inputShapes);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, DHW_DIMS, outShapes);
    TILING_DATA_FIELD_DEF(uint64_t, kD);
    TILING_DATA_FIELD_DEF(uint64_t, kW);
    TILING_DATA_FIELD_DEF(uint64_t, kH);
    TILING_DATA_FIELD_DEF(uint64_t, sD);
    TILING_DATA_FIELD_DEF(uint64_t, sW);
    TILING_DATA_FIELD_DEF(uint64_t, sH);
    TILING_DATA_FIELD_DEF(uint64_t, pD);
    TILING_DATA_FIELD_DEF(uint64_t, pW);
    TILING_DATA_FIELD_DEF(uint64_t, pH);
    TILING_DATA_FIELD_DEF(uint64_t, dD);
    TILING_DATA_FIELD_DEF(uint64_t, dW);
    TILING_DATA_FIELD_DEF(uint64_t, dH);
    TILING_DATA_FIELD_DEF(uint64_t, batchesPerCore);
    TILING_DATA_FIELD_DEF(uint64_t, leftOverBatches);
    TILING_DATA_FIELD_DEF(uint64_t, partD);
    TILING_DATA_FIELD_DEF(uint64_t, partH);
    TILING_DATA_FIELD_DEF(uint64_t, partW);
    TILING_DATA_FIELD_DEF(float, minFloat);
    TILING_DATA_FIELD_DEF(uint64_t, ceilD);
    TILING_DATA_FIELD_DEF(uint64_t, ceilH);
    TILING_DATA_FIELD_DEF(uint64_t, ceilW);
    TILING_DATA_FIELD_DEF(uint64_t, sizeUb1);
    TILING_DATA_FIELD_DEF(uint64_t, sizeUb2);
    TILING_DATA_FIELD_DEF(uint64_t, sizeValues);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(MaxPool3DWithArgmaxV2BigKernelTilingData)
    TILING_DATA_FIELD_DEF_ARR(uint32_t, DHW_DIMS, inputShapes);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, DHW_DIMS, outShapes);
    TILING_DATA_FIELD_DEF(uint32_t, kD);
    TILING_DATA_FIELD_DEF(uint32_t, kW);
    TILING_DATA_FIELD_DEF(uint32_t, kH);
    TILING_DATA_FIELD_DEF(uint32_t, sD);
    TILING_DATA_FIELD_DEF(uint32_t, sW);
    TILING_DATA_FIELD_DEF(uint32_t, sH);
    TILING_DATA_FIELD_DEF(uint32_t, pD);
    TILING_DATA_FIELD_DEF(uint32_t, pW);
    TILING_DATA_FIELD_DEF(uint32_t, pH);
    TILING_DATA_FIELD_DEF(uint32_t, dD);
    TILING_DATA_FIELD_DEF(uint32_t, dW);
    TILING_DATA_FIELD_DEF(uint32_t, dH);
    TILING_DATA_FIELD_DEF(uint64_t, blockFactor);
    TILING_DATA_FIELD_DEF(uint64_t, blockTail);
    TILING_DATA_FIELD_DEF(uint64_t, totalIdx);
    TILING_DATA_FIELD_DEF(uint64_t, coreNums);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(MaxPool3DWithArgmaxV2NoExpandIndicesTilingData)
TILING_DATA_FIELD_DEF(uint64_t, nc);
TILING_DATA_FIELD_DEF(uint64_t, dx);
TILING_DATA_FIELD_DEF(uint64_t, hx);
TILING_DATA_FIELD_DEF(uint64_t, wx);
TILING_DATA_FIELD_DEF(uint64_t, kd);
TILING_DATA_FIELD_DEF(uint64_t, kh);
TILING_DATA_FIELD_DEF(uint64_t, kw);
TILING_DATA_FIELD_DEF(uint64_t, sd);
TILING_DATA_FIELD_DEF(uint64_t, sh);
TILING_DATA_FIELD_DEF(uint64_t, sw);
TILING_DATA_FIELD_DEF(uint64_t, pf);
TILING_DATA_FIELD_DEF(uint64_t, pb);
TILING_DATA_FIELD_DEF(uint64_t, pl);
TILING_DATA_FIELD_DEF(uint64_t, pr);
TILING_DATA_FIELD_DEF(uint64_t, pt);
TILING_DATA_FIELD_DEF(uint64_t, pd);
TILING_DATA_FIELD_DEF(uint64_t, dy);
TILING_DATA_FIELD_DEF(uint64_t, hy);
TILING_DATA_FIELD_DEF(uint64_t, wy);
TILING_DATA_FIELD_DEF(uint64_t, ncFactor);
TILING_DATA_FIELD_DEF(uint64_t, ncTail);
TILING_DATA_FIELD_DEF(uint64_t, ncOuter);
TILING_DATA_FIELD_DEF(uint64_t, dyFactor);
TILING_DATA_FIELD_DEF(uint64_t, dyTail);
TILING_DATA_FIELD_DEF(uint64_t, dyOuter);
TILING_DATA_FIELD_DEF(uint64_t, hyFactor);
TILING_DATA_FIELD_DEF(uint64_t, hyTail);
TILING_DATA_FIELD_DEF(uint64_t, hyOuter);
TILING_DATA_FIELD_DEF(uint64_t, wyFactor);
TILING_DATA_FIELD_DEF(uint64_t, wyTail);
TILING_DATA_FIELD_DEF(uint64_t, wyOuter);
TILING_DATA_FIELD_DEF(uint64_t, blockFactor);
TILING_DATA_FIELD_DEF(uint64_t, blockTail);
TILING_DATA_FIELD_DEF(uint64_t, totalIdx);
TILING_DATA_FIELD_DEF(uint64_t, coreNums);
TILING_DATA_FIELD_DEF(uint64_t, inputBufferSize);
TILING_DATA_FIELD_DEF(uint64_t, outputMaxBufferSize);
TILING_DATA_FIELD_DEF(uint64_t, outputIndiceBufferSize);
TILING_DATA_FIELD_DEF(uint64_t, indiceTempBufferSize);
TILING_DATA_FIELD_DEF(uint64_t, maskBufferSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaxPool3DWithArgmaxV2, MaxPool3DWithArgmaxV2TilingData);

//1, splitD=0, splitH=0, splitW=0, splitKernel = 0, dtype=float=0
//1, splitD=0, splitH=0, splitW=0, splitKernel = 0, dtype=half=1
//1, splitD=0, splitH=0, splitW=0, splitKernel = 0, dtype=bfloat16=2
REGISTER_TILING_DATA_CLASS(MaxPool3DWithArgmaxV2_100000, MaxPool3DWithArgmaxV2NoSplitTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3DWithArgmaxV2_100001, MaxPool3DWithArgmaxV2NoSplitTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3DWithArgmaxV2_100002, MaxPool3DWithArgmaxV2NoSplitTilingData);

//1, splitD=1, splitH=0, splitW=0, splitKernel = 0, dtype=float=0
//1, splitD=1, splitH=0, splitW=0, splitKernel = 0, dtype=half=1
//1, splitD=1, splitH=0, splitW=0, splitKernel = 0, dtype=bfloat16=2
REGISTER_TILING_DATA_CLASS(MaxPool3DWithArgmaxV2_110000, MaxPool3DWithArgmaxV2SplitDTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3DWithArgmaxV2_110001, MaxPool3DWithArgmaxV2SplitDTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3DWithArgmaxV2_110002, MaxPool3DWithArgmaxV2SplitDTilingData);

//1, splitD=1, splitH=1, splitW=0, splitKernel = 0, dtype=float=0
//1, splitD=1, splitH=1, splitW=0, splitKernel = 0, dtype=half=1
//1, splitD=1, splitH=1, splitW=0, splitKernel = 0, dtype=bfloat16=2
REGISTER_TILING_DATA_CLASS(MaxPool3DWithArgmaxV2_111000, MaxPool3DWithArgmaxV2SplitHTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3DWithArgmaxV2_111001, MaxPool3DWithArgmaxV2SplitHTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3DWithArgmaxV2_111002, MaxPool3DWithArgmaxV2SplitHTilingData);

//1, splitD=1, splitH=1, splitW=1, splitKernel = 0, dtype=float=0
//1, splitD=1, splitH=1, splitW=1, splitKernel = 0, dtype=half=1
//1, splitD=1, splitH=1, splitW=1, splitKernel = 0, dtype=bfloat16=2
REGISTER_TILING_DATA_CLASS(MaxPool3DWithArgmaxV2_111100, MaxPool3DWithArgmaxV2SplitWTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3DWithArgmaxV2_111101, MaxPool3DWithArgmaxV2SplitWTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3DWithArgmaxV2_111102, MaxPool3DWithArgmaxV2SplitWTilingData);

//1, splitD=1, splitH=1, splitW=1, splitKernel = 1, dtype=float=0
//1, splitD=1, splitH=1, splitW=1, splitKernel = 1, dtype=half=1
//1, splitD=1, splitH=1, splitW=1, splitKernel = 1, dtype=bfloat16=2
REGISTER_TILING_DATA_CLASS(MaxPool3DWithArgmaxV2_111110, MaxPool3DWithArgmaxV2HugeKernelTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3DWithArgmaxV2_111111, MaxPool3DWithArgmaxV2HugeKernelTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3DWithArgmaxV2_111112, MaxPool3DWithArgmaxV2HugeKernelTilingData);

REGISTER_TILING_DATA_CLASS(MaxPool3DWithArgmaxV2_311110, MaxPool3DWithArgmaxV2BigKernelTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3DWithArgmaxV2_311111, MaxPool3DWithArgmaxV2BigKernelTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3DWithArgmaxV2_311112, MaxPool3DWithArgmaxV2BigKernelTilingData);

// 300001 - no padding, 300002 - padding
REGISTER_TILING_DATA_CLASS(MaxPool3DWithArgmaxV2_300001, MaxPool3DWithArgmaxV2NoExpandIndicesTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3DWithArgmaxV2_300002, MaxPool3DWithArgmaxV2NoExpandIndicesTilingData);

struct InputInfo {
    uint64_t batches;
    array<uint64_t, DHW_DIMS> inputShape;
    array<uint64_t, DHW_DIMS> outShape;
    array<uint64_t, DHW_DIMS> kernelSize;
    array<uint64_t, DHW_DIMS> stride;
    array<uint64_t, DHW_DIMS> pad;
    array<uint64_t, DHW_DIMS> dilation;
    bool ceilMode;
};

struct PadInputInfo {
    array<uint64_t, DHW_DIMS> padInputShape;
    array<uint64_t, DHW_DIMS> ceil;
};

struct UBBufferSize {
    uint64_t sizeUb1;
    uint64_t sizeUb2;
    uint64_t valSize;
};

struct SplitDataD {
    uint64_t batchesPerCore;
    uint64_t leftOverBatches;
    uint64_t partD;
    uint64_t partOutD;
    uint64_t partsPerCore;
    uint64_t leftOverParts;
    uint64_t splitPointDIn[MAX_CORE_NUM];
    uint64_t splitPointDOut[MAX_CORE_NUM];
    uint64_t sizeIn[MAX_CORE_NUM];
    uint64_t batchStart[MAX_CORE_NUM];
};

struct SplitDataH {
    uint64_t batchesPerCore;
    uint64_t leftOverBatches;
    uint64_t partD;
    uint64_t partOutD;
    uint64_t partH;
    uint64_t partOutH;
    uint64_t partsPerCore = 0;
    uint64_t leftOverParts = 0;
    uint64_t splitPointDIn[MAX_CORE_NUM] = {0};
    uint64_t splitPointHIn[MAX_CORE_NUM] = {0};
    uint64_t splitPointDOut[MAX_CORE_NUM] = {0};
    uint64_t splitPointHOut[MAX_CORE_NUM] = {0};
    uint64_t sizeIn[MAX_CORE_NUM] = {0};
    uint64_t batchStart[MAX_CORE_NUM] = {0};
};

struct SplitDataW {
    uint64_t batchesPerCore;
    uint64_t leftOverBatches;
    uint64_t partD;
    uint64_t partOutD;
    uint64_t partH;
    uint64_t partOutH;
    uint64_t partW;
    uint64_t partOutW;
    uint64_t partsPerCore = 0;
    uint64_t leftOverParts = 0;
    uint64_t splitPointDIn[MAX_CORE_NUM] = {0};
    uint64_t splitPointHIn[MAX_CORE_NUM] = {0};
    uint64_t splitPointWIn[MAX_CORE_NUM] = {0};
    uint64_t splitPointDOut[MAX_CORE_NUM] = {0};
    uint64_t splitPointHOut[MAX_CORE_NUM] = {0};
    uint64_t splitPointWOut[MAX_CORE_NUM] = {0};
    uint64_t sizeIn[MAX_CORE_NUM] = {0};
    uint64_t batchStart[MAX_CORE_NUM] = {0};
};

struct SplitDataHugeKernel {
    uint64_t batchesPerCore;
    uint64_t leftOverBatches;
    uint64_t partD;
    uint64_t partOutD;
    uint64_t partH;
    uint64_t partOutH;
    uint64_t partW;
    uint64_t partOutW;
};

struct SplitInfo {
  uint64_t ncFactor = 1;
  uint64_t ncTail = 1;
  uint64_t ncOuter = 1;
  // split out
  uint64_t dyFactor = 1;
  uint64_t dyTail = 1;
  uint64_t dyOuter = 1;
  uint64_t wyFactor = 1;
  uint64_t wyTail = 1;
  uint64_t wyOuter = 1;
  uint64_t hyFactor = 1;
  uint64_t hyTail = 1;
  uint64_t hyOuter = 1;
  // split kernel d
  uint64_t kdFactor = 1;
  uint64_t kdTail = 1;
  uint64_t kdOuter = 1;
  // split kernel h
  uint64_t khFactor = 1;
  uint64_t khTail = 1;
  uint64_t khOuter = 1;
  // split kernel w
  uint64_t kwFactor = 1;
  uint64_t kwTail = 1;
  uint64_t kwOuter = 1;

  uint64_t blockFactor = 1;
  uint64_t blockTail = 1;
  uint64_t totalIdx = 1;
  uint64_t coreNums = 1;
  array<uint64_t, DHW_DIMS> outShape;
  array<uint64_t, DHW_PAD_DIMS> pad;
};

struct BufferInfo {
  uint64_t inputBufferSize = 1;  // input buffer
  uint64_t outputMaxBufferSize = 1;  // output max
  uint64_t outputIndiceBufferSize = 1;  // output indice
  uint64_t indiceTempBufferSize = 1; // indice local/template/update buffer
  uint64_t maskBufferSize = 1;  // mask buffer

  // intermediate variables
  uint64_t inputCastBytes;
  uint64_t blockDataNum;
  uint64_t ncMaxFactor;
  uint64_t ncFactorAlign;
  uint64_t lbDyFactor;
  uint64_t lbHyFactor;
  uint64_t lbWyFactor;
  uint64_t ubDyFactor;
  uint64_t ubHyFactor;
  uint64_t ubWyFactor;
  uint64_t tmpDyFactor;
  uint64_t tmpHyFactor;
  uint64_t tmpWyFactor;
  uint64_t tmpWyFactorAlign;
  uint64_t outputMaxPoolSize;
  // different when dtype is float16
  uint64_t outputIndicePoolSize;
  uint64_t tmpDxFactor;
  uint64_t tmpHxFactor;
  uint64_t tmpWxFactorAlign;
  uint64_t inputPoolSize;
  uint64_t indiceTempPoolSize;
  uint64_t maskPoolSize;
  uint64_t tmpTotalSize;
};

struct MaxPool3DWithArgmaxV2CompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

class MaxPool3DWithArgmaxV2BaseTiling : public TilingBaseClass {
    public:
        explicit MaxPool3DWithArgmaxV2BaseTiling(gert::TilingContext* context) : TilingBaseClass(context) {
        }
  
      ~MaxPool3DWithArgmaxV2BaseTiling() override {
      }

    protected:
        bool IsCapable() override;
        ge::graphStatus GetPlatformInfo() override;
        ge::graphStatus GetShapeAttrsInfo() override;
        ge::graphStatus DoOpTiling() override;
        ge::graphStatus DoLibApiTiling() override;
        uint64_t GetTilingKey() const override;
        ge::graphStatus GetWorkspaceSize() override;
        ge::graphStatus PostTiling() override;

    public:
        InputInfo inputData;
        ge::DataType dtype = ge::DataType::DT_FLOAT;
        uint64_t coreNum = 1;
        uint64_t ubSize = 0;
};

class MaxPool3DWithArgmaxV2BaseSplitTiling : public MaxPool3DWithArgmaxV2BaseTiling {
    public:
        explicit MaxPool3DWithArgmaxV2BaseSplitTiling(gert::TilingContext* context) : MaxPool3DWithArgmaxV2BaseTiling(context) {
        }
  
        ~MaxPool3DWithArgmaxV2BaseSplitTiling() override {
        }
    protected:
        ge::graphStatus GetPlatformInfo() override;
        ge::graphStatus GetShapeAttrsInfo() override;
        ge::graphStatus GetWorkspaceSize() override;
        uint64_t CalcBufferSizes(const array<uint64_t, DHW_DIMS> part, const array<uint64_t, DHW_DIMS> partOut,
                                 const uint64_t partHwInp, UBBufferSize& bufSizes);
        ge::graphStatus FindSplitParts(array<uint64_t, DHW_DIMS> &inParts, array<uint64_t, DHW_DIMS> &outParts,
                                       UBBufferSize& bufSizes, uint64_t dim);
        uint64_t RoundUpBlock(const uint64_t& src, const uint64_t& blockLen);
        uint64_t RoundDownBlock(const uint64_t& src, const uint64_t& blockLen);

    private:
        ge::graphStatus PadCalc();
        uint64_t InputCalc(uint64_t dim);
        ge::graphStatus InputPadCalc(const array<int64_t, DHW_DIMS> inpDiff);
    
    public:
        PadInputInfo padInputData;
        UBBufferSize bufSizes;
        uint64_t blockLength;
        uint64_t blockLengthS;
        uint64_t ubSizeNew;
};

class MaxPool3DWithArgmaxV2NoSplitTiling : public MaxPool3DWithArgmaxV2BaseSplitTiling {
    public:
        explicit MaxPool3DWithArgmaxV2NoSplitTiling(gert::TilingContext* context) : MaxPool3DWithArgmaxV2BaseSplitTiling(context) {
        }
  
        ~MaxPool3DWithArgmaxV2NoSplitTiling() override {
        }

    private:
        void DoUBTiling();
        void SetTilingData();
        uint64_t GetTilingKey() const override;
        bool IsCapable() override;
        ge::graphStatus DoOpTiling() override;
        ge::graphStatus PostTiling() override;

        MaxPool3DWithArgmaxV2NoSplitTilingData tiling;
        uint64_t batchesPerCore;
        uint64_t leftOverBatches;
};

class MaxPool3DWithArgmaxV2SplitDTiling : public MaxPool3DWithArgmaxV2BaseSplitTiling {
    public:
        explicit MaxPool3DWithArgmaxV2SplitDTiling(gert::TilingContext* context) : MaxPool3DWithArgmaxV2BaseSplitTiling(context) {
        }

        ~MaxPool3DWithArgmaxV2SplitDTiling() override {
        }

    private:
        void DoUBTiling();
        void SetTilingData();
        ge::graphStatus SmallBatchesTiling(void);
        uint64_t GetTilingKey() const override;
        bool IsCapable() override;
        ge::graphStatus DoOpTiling() override;
        ge::graphStatus PostTiling() override;

        MaxPool3DWithArgmaxV2SplitDTilingData tiling;
        SplitDataD splitData;
};

class MaxPool3DWithArgmaxV2SplitHTiling : public MaxPool3DWithArgmaxV2BaseSplitTiling {
    public:
        explicit MaxPool3DWithArgmaxV2SplitHTiling(gert::TilingContext* context) : MaxPool3DWithArgmaxV2BaseSplitTiling(context) {
        }
  
        ~MaxPool3DWithArgmaxV2SplitHTiling() override {
        }

    private:
        void DoUBTiling();
        void SetTilingData();
        ge::graphStatus SmallBatchesTiling(void);
        uint64_t GetTilingKey() const override;
        bool IsCapable() override;
        ge::graphStatus DoOpTiling() override;
        ge::graphStatus PostTiling() override;

        MaxPool3DWithArgmaxV2SplitHTilingData tiling;
        SplitDataH splitData;
};

class MaxPool3DWithArgmaxV2SplitWTiling : public MaxPool3DWithArgmaxV2BaseSplitTiling {
    public:
        explicit MaxPool3DWithArgmaxV2SplitWTiling(gert::TilingContext* context) : MaxPool3DWithArgmaxV2BaseSplitTiling(context) {
        }
  
        ~MaxPool3DWithArgmaxV2SplitWTiling() override {
        }

    private:
        void DoUBTiling();
        void SetTilingData();
        ge::graphStatus SmallBatchesTiling(void);
        uint64_t GetTilingKey() const override;
        bool IsCapable() override;
        ge::graphStatus DoOpTiling() override;
        ge::graphStatus PostTiling() override;

        MaxPool3DWithArgmaxV2SplitWTilingData tiling;
        SplitDataW splitData;
};

class MaxPool3DWithArgmaxV2HugeKernelTiling : public MaxPool3DWithArgmaxV2BaseSplitTiling {
    public:
        explicit MaxPool3DWithArgmaxV2HugeKernelTiling(gert::TilingContext* context) : MaxPool3DWithArgmaxV2BaseSplitTiling(context) {
        }

        ~MaxPool3DWithArgmaxV2HugeKernelTiling() override {
        }

    private:
        void DoUBTiling();
        void SetTilingData();
        uint64_t GetTilingKey() const;
        bool IsCapable() override;
        ge::graphStatus DoOpTiling() override;
        ge::graphStatus PostTiling() override;
        ge::graphStatus GetWorkspaceSize() override;
        ge::graphStatus LargeKernelCalcParts(array<uint64_t, DHW_DIMS> &inParts, UBBufferSize& bufSizes, uint64_t dim);
        uint64_t CalcBufferSizes(const array<uint64_t, DHW_DIMS> part, UBBufferSize& bufSizes);

        MaxPool3DWithArgmaxV2HugeKernelTilingData tiling;
        SplitDataHugeKernel splitData;
};

class MaxPool3DWithArgmaxV2BigKernelTiling : public MaxPool3DWithArgmaxV2BaseSplitTiling {
    public:
        explicit MaxPool3DWithArgmaxV2BigKernelTiling(gert::TilingContext* context) : MaxPool3DWithArgmaxV2BaseSplitTiling(context) {
        }

        ~MaxPool3DWithArgmaxV2BigKernelTiling() override {
        }

    private:
        void DoUBTiling();
        void SetTilingData();
        uint64_t GetTilingKey() const;
        bool IsCapable() override;
        ge::graphStatus DoOpTiling() override;
        ge::graphStatus PostTiling() override;

        MaxPool3DWithArgmaxV2BigKernelTilingData tiling;
        uint32_t totalIdx;
        uint32_t blockFactor;
        uint32_t blockTail;
        uint32_t coreNums;
};

class MaxPool3DWithArgmaxV2NoExpandIndicesTiling : public MaxPool3DWithArgmaxV2BaseTiling {
    public:
        explicit MaxPool3DWithArgmaxV2NoExpandIndicesTiling(gert::TilingContext* context) : MaxPool3DWithArgmaxV2BaseTiling(context) {
        }

        ~MaxPool3DWithArgmaxV2NoExpandIndicesTiling() override {
        }

    private:
        void DoDAdjustment();
        void DoHAdjustment();
        void DoWAdjustment();
        void DoBoundaryAdjustment();
        void DoOutputPadAdjustment();
        void DoUBTiling();
        void SetTilingData();
        uint64_t GetTilingKey() const;
        void DoBlockTiling();
        void DoBufferCalculate();
        bool IsCapable() override;
        ge::graphStatus DoOpTiling() override;
        ge::graphStatus PostTiling() override;

        MaxPool3DWithArgmaxV2NoExpandIndicesTilingData tiling;
        uint64_t inputBytes;
        SplitInfo splitData;
        BufferInfo bufferData;
};

}  // namespace optiling

#endif