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
 * \file max_pool3d_grad_with_argmax_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_MAX_POOL3D_GRAD_WITH_ARGMAX_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_MAX_POOL3D_GRAD_WITH_ARGMAX_H

#define OPS_UTILS_LOG_SUB_MOD_NAME "OP_MAXPOOL3DGRADWIHTARGMAX"
#define OPS_UTILS_LOG_PACKAGE_TYPE "OP_MAXPOOL3DGRADWIHTARGMAX"

#include <string>
#include "tiling/tiling_base.h"
#include "tiling/tiling_type.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_templates_registry.h"

#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
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

#define OP_LOGI(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__) 
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)                     
#define unlikely(x) __builtin_expect((x), 0)
#define OP_LOGE_IF(condition, return_value, op_name, fmt, ...)                                                 \
  static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool"); \
  do {                                                                                                         \
    if (unlikely(condition)) {                                                                                 \
      OP_LOGE(op_name, fmt, ##__VA_ARGS__);                                                                    \
      return return_value;                                                                                     \
    }                                                                                                          \
  } while (0)

namespace vectorutil {
int64_t GetElementByType(const ge::DataType& dtype);
int64_t CeilDiv(const int64_t dividend, const int64_t divisor);
int64_t FloorAlign(const int64_t dividend, const int64_t divisor);
int64_t CeilAlign(const int64_t dividend, const int64_t divisor);
} // namespace vectorutil

const uint32_t DHW_DIMS = 3;
const uint32_t D_DIM = 0;
const uint32_t H_DIM = 1;
const uint32_t W_DIM = 2;
const uint32_t MAX_DIV = 2;
const uint32_t NCHW_CONV_ADDR_LIST_SIZE = 16;
const uint32_t MIN_TRANSPOSE_ROWS = 16;
const uint32_t INT64_FP32 = 2;
const uint32_t BINARY_SEARCH_COEFF = 2;

BEGIN_TILING_DATA_DEF(MaxPool3DGradWithArgmaxTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, ncDim);
    TILING_DATA_FIELD_DEF(uint64_t, diDim);
    TILING_DATA_FIELD_DEF(uint64_t, hiDim);
    TILING_DATA_FIELD_DEF(uint64_t, wiDim);
    TILING_DATA_FIELD_DEF(uint64_t, doDim);
    TILING_DATA_FIELD_DEF(uint64_t, hoDim);
    TILING_DATA_FIELD_DEF(uint64_t, woDim);
    TILING_DATA_FIELD_DEF(uint64_t, kd);
    TILING_DATA_FIELD_DEF(uint64_t, kh);
    TILING_DATA_FIELD_DEF(uint64_t, kw);
    TILING_DATA_FIELD_DEF(uint64_t, sd);
    TILING_DATA_FIELD_DEF(uint64_t, sh);
    TILING_DATA_FIELD_DEF(uint64_t, sw);
    TILING_DATA_FIELD_DEF(uint64_t, padDTop);
    TILING_DATA_FIELD_DEF(uint64_t, padHTop);
    TILING_DATA_FIELD_DEF(uint64_t, padWTop);
    TILING_DATA_FIELD_DEF(uint64_t, padDBottom);
    TILING_DATA_FIELD_DEF(uint64_t, padHBottom);
    TILING_DATA_FIELD_DEF(uint64_t, padWBottom);
    TILING_DATA_FIELD_DEF(uint64_t, baseNc);
    TILING_DATA_FIELD_DEF(uint64_t, baseDo);
    TILING_DATA_FIELD_DEF(uint64_t, baseHo);
    TILING_DATA_FIELD_DEF(uint64_t, baseWo);
    TILING_DATA_FIELD_DEF(uint64_t, ncTail);
    TILING_DATA_FIELD_DEF(uint64_t, doTail);
    TILING_DATA_FIELD_DEF(uint64_t, hoTail);
    TILING_DATA_FIELD_DEF(uint64_t, woTail);
    TILING_DATA_FIELD_DEF(uint64_t, ncCnt);
    TILING_DATA_FIELD_DEF(uint64_t, doCnt);
    TILING_DATA_FIELD_DEF(uint64_t, hoCnt);
    TILING_DATA_FIELD_DEF(uint64_t, woCnt);
    TILING_DATA_FIELD_DEF(uint64_t, totalCnt);
    TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint64_t, totalUBSize);

    // normal
    TILING_DATA_FIELD_DEF(uint64_t, singleCoreNc);
    TILING_DATA_FIELD_DEF(uint64_t, singleCoreDo);
    TILING_DATA_FIELD_DEF(uint64_t, singleCoreHo);
    TILING_DATA_FIELD_DEF(uint64_t, singleCoreWo);

    // scatter
    TILING_DATA_FIELD_DEF(uint64_t, ncRound);
    TILING_DATA_FIELD_DEF(uint64_t, ncRoundTail);
    TILING_DATA_FIELD_DEF(uint64_t, totalRound);
    TILING_DATA_FIELD_DEF(uint64_t, preCoreNum);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(MaxPool3DGradWithArgmaxNoSplitTilingData)
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
    TILING_DATA_FIELD_DEF(uint64_t, partOutD);
    TILING_DATA_FIELD_DEF(uint64_t, partOutH);
    TILING_DATA_FIELD_DEF(uint64_t, partOutW);
    TILING_DATA_FIELD_DEF(uint64_t, ceilD);
    TILING_DATA_FIELD_DEF(uint64_t, ceilH);
    TILING_DATA_FIELD_DEF(uint64_t, ceilW);
    TILING_DATA_FIELD_DEF(uint64_t, sizeUb1);
    TILING_DATA_FIELD_DEF(uint64_t, sizeUb2);
    TILING_DATA_FIELD_DEF(uint64_t, sizeValues);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(MaxPool3DGradWithArgmaxSplitDTilingData)
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
    TILING_DATA_FIELD_DEF(uint64_t, partOutD);
    TILING_DATA_FIELD_DEF(uint64_t, partOutH);
    TILING_DATA_FIELD_DEF(uint64_t, partOutW);
    TILING_DATA_FIELD_DEF(uint64_t, ceilD);
    TILING_DATA_FIELD_DEF(uint64_t, ceilH);
    TILING_DATA_FIELD_DEF(uint64_t, ceilW);
    TILING_DATA_FIELD_DEF(uint64_t, sizeUb1);
    TILING_DATA_FIELD_DEF(uint64_t, sizeUb2);
    TILING_DATA_FIELD_DEF(uint64_t, sizeValues);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(MaxPool3DGradWithArgmaxSplitHTilingData)
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
    TILING_DATA_FIELD_DEF(uint64_t, partOutD);
    TILING_DATA_FIELD_DEF(uint64_t, partOutH);
    TILING_DATA_FIELD_DEF(uint64_t, partOutW);
    TILING_DATA_FIELD_DEF(uint64_t, ceilD);
    TILING_DATA_FIELD_DEF(uint64_t, ceilH);
    TILING_DATA_FIELD_DEF(uint64_t, ceilW);
    TILING_DATA_FIELD_DEF(uint64_t, sizeUb1);
    TILING_DATA_FIELD_DEF(uint64_t, sizeUb2);
    TILING_DATA_FIELD_DEF(uint64_t, sizeValues);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(MaxPool3DGradWithArgmaxSplitWTilingData)
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
    TILING_DATA_FIELD_DEF(uint64_t, partOutD);
    TILING_DATA_FIELD_DEF(uint64_t, partOutH);
    TILING_DATA_FIELD_DEF(uint64_t, partOutW);
    TILING_DATA_FIELD_DEF(uint64_t, ceilD);
    TILING_DATA_FIELD_DEF(uint64_t, ceilH);
    TILING_DATA_FIELD_DEF(uint64_t, ceilW);
    TILING_DATA_FIELD_DEF(uint64_t, sizeUb1);
    TILING_DATA_FIELD_DEF(uint64_t, sizeUb2);
    TILING_DATA_FIELD_DEF(uint64_t, sizeValues);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaxPool3DGradWithArgmax, MaxPool3DGradWithArgmaxTilingData);

//1, splitD=0, splitH=0, splitW=0, splitKernel = 0, dtype=float=0
//no overlap = 1, splitD=0, splitH=0, splitW=0, splitKernel = 0, dtype=half=1
//overlap = 2, splitD=0, splitH=0, splitW=0, splitKernel = 0, dtype=half=1
//1, splitD=0, splitH=0, splitW=0, splitKernel = 0, dtype=bfloat16=2
REGISTER_TILING_DATA_CLASS(MaxPool3DGradWithArgmax_100000, MaxPool3DGradWithArgmaxNoSplitTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3DGradWithArgmax_100001, MaxPool3DGradWithArgmaxNoSplitTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3DGradWithArgmax_200001, MaxPool3DGradWithArgmaxNoSplitTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3DGradWithArgmax_100002, MaxPool3DGradWithArgmaxNoSplitTilingData);

//1, splitD=1, splitH=0, splitW=0, splitKernel = 0, dtype=float=0
//no overlap = 1, splitD=1, splitH=0, splitW=0, splitKernel = 0, dtype=half=1
//overlap = 2, splitD=1, splitH=0, splitW=0, splitKernel = 0, dtype=half=1
//1, splitD=1, splitH=0, splitW=0, splitKernel = 0, dtype=bfloat16=2
REGISTER_TILING_DATA_CLASS(MaxPool3DGradWithArgmax_110000, MaxPool3DGradWithArgmaxSplitDTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3DGradWithArgmax_110001, MaxPool3DGradWithArgmaxSplitDTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3DGradWithArgmax_210001, MaxPool3DGradWithArgmaxSplitDTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3DGradWithArgmax_110002, MaxPool3DGradWithArgmaxSplitDTilingData);

//1, splitD=1, splitH=1, splitW=0, splitKernel = 0, dtype=float=0
//no overlap = 1, splitD=1, splitH=1, splitW=0, splitKernel = 0, dtype=half=1
//overlap = 2, splitD=1, splitH=1, splitW=0, splitKernel = 0, dtype=half=1
//1, splitD=1, splitH=1, splitW=0, splitKernel = 0, dtype=bfloat16=2
REGISTER_TILING_DATA_CLASS(MaxPool3DGradWithArgmax_111000, MaxPool3DGradWithArgmaxSplitHTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3DGradWithArgmax_111001, MaxPool3DGradWithArgmaxSplitHTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3DGradWithArgmax_211001, MaxPool3DGradWithArgmaxSplitHTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3DGradWithArgmax_111002, MaxPool3DGradWithArgmaxSplitHTilingData);

//1, splitD=1, splitH=1, splitW=1, splitKernel = 0, dtype=float=0
//no overlap = 1, splitD=1, splitH=1, splitW=1, splitKernel = 0, dtype=half=1
//overlap = 2, splitD=1, splitH=1, splitW=1, splitKernel = 0, dtype=half=1
//1, splitD=1, splitH=1, splitW=1, splitKernel = 0, dtype=bfloat16=2
REGISTER_TILING_DATA_CLASS(MaxPool3DGradWithArgmax_111100, MaxPool3DGradWithArgmaxSplitWTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3DGradWithArgmax_111101, MaxPool3DGradWithArgmaxSplitWTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3DGradWithArgmax_211101, MaxPool3DGradWithArgmaxSplitWTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3DGradWithArgmax_111102, MaxPool3DGradWithArgmaxSplitWTilingData);

struct InputInfo {
    uint64_t batches;
    std::array<uint64_t, DHW_DIMS> inputShape;
    std::array<uint64_t, DHW_DIMS> outShape;
    std::array<uint64_t, DHW_DIMS> kernelSize;
    std::array<uint64_t, DHW_DIMS> stride;
    std::array<uint64_t, DHW_DIMS> pad;
    std::array<uint64_t, DHW_DIMS> dilation;
    bool ceilMode;
    bool isOverlap;
};

struct PadOutputInfo {
    std::array<uint64_t, DHW_DIMS> padOutputShape;
    std::array<uint64_t, DHW_DIMS> ceil;
};

struct UBBufferSize {
    uint64_t sizeUb1;
    uint64_t sizeUb2;
    uint64_t valSize;
};

struct SplitDataDGrad {
    uint64_t batchesPerCore;
    uint64_t leftOverBatches;
    uint64_t partD;
    uint64_t partOutD;
};

struct SplitDataHGrad {
    uint64_t batchesPerCore;
    uint64_t leftOverBatches;
    uint64_t partD;
    uint64_t partOutD;
    uint64_t partH;
    uint64_t partOutH;
};

struct SplitDataWGrad {
    uint64_t batchesPerCore;
    uint64_t leftOverBatches;
    uint64_t partD;
    uint64_t partOutD;
    uint64_t partH;
    uint64_t partOutH;
    uint64_t partW;
    uint64_t partOutW;
};

// Index const
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t GRAD_INDEX = 1;
constexpr uint32_t ARGMAX_INDEX = 2;
constexpr uint32_t Y_INDEX = 0;
constexpr size_t KSIZE_ATTR_INDEX = 0U;
constexpr size_t STRIDES_ATTR_INDEX = 1U;
constexpr size_t PADS_ATTR_INDEX = 2U;
constexpr size_t DILATION_ATTR_INDEX = 3U;
constexpr size_t CEIL_MODE_ATTR_INDEX = 4U;
// Params const
constexpr uint64_t NUM_TWO = 2;
constexpr size_t NC_DIM_NUM = 2;
constexpr size_t DHW_DIM_NUM = 3;
constexpr size_t NCDHW_DIM_NUM = 5;
constexpr uint32_t DTYPE_LEN_B8 = 1;
constexpr uint32_t DTYPE_LEN_B16 = 2;
constexpr uint32_t DTYPE_LEN_B32 = 4;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t MAX_BLOCK_COUNT = 4095;
constexpr uint64_t MAX_INT32 = 2147483647;
constexpr uint32_t NUM_PER_REP_B16 = 128;
constexpr uint32_t NUM_PER_REP_B32 = 64;
constexpr uint32_t SELECT_RESERVED_UB_SIZE = 8192;
// Tiling const
constexpr uint32_t TILING_OVERLAP = 100;
constexpr uint32_t TILING_UB_NO_CUT = 0;
constexpr uint32_t TILING_UB_CUT_NC = 10;
constexpr uint32_t TILING_UB_CUT_DO = 20;
constexpr uint32_t TILING_UB_CUT_HO = 30;
constexpr uint32_t TILING_UB_CUT_WO = 40;
constexpr uint32_t TILING_UB_CUT_KD = 50;
constexpr uint32_t TILING_UB_CUT_KH = 60;
constexpr uint32_t TILING_UB_CUT_KW = 70;
constexpr uint32_t TILING_TYPE_NORMAL = 0;
constexpr uint32_t TILING_TYPE_CUTK = 1;
constexpr uint32_t TILING_TYPE_SCATTER = 2;

struct Tiling4MaxPool3DGradWithArgmaxCompileInfo {
    platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
    uint64_t totalCoreNum = 0;
    uint64_t maxUbSize = 0;
};

struct MaxPoolGradWithArgmaxTilingParams {
    // Platform
    uint64_t maxUbSize{0};
    uint64_t totalCoreNum{0};
    // Input shape
    uint64_t xDtypeSize{0};
    uint64_t indexDtypeSize{0};
    uint64_t ncDim{0};
    uint64_t diDim{0};
    uint64_t hiDim{0};
    uint64_t wiDim{0};
    uint64_t doDim{0};
    uint64_t hoDim{0};
    uint64_t woDim{0};
    // Kernel size
    uint64_t kd{0};
    uint64_t kh{0};
    uint64_t kw{0};
    // Stride size
    uint64_t sd{0};
    uint64_t sh{0};
    uint64_t sw{0};
    uint64_t vl{0};
    // Pad Size
    uint64_t pDTop{0};
    uint64_t pHTop{0};
    uint64_t pWTop{0};
    uint64_t pDBottom{0};
    uint64_t pHBottom{0};
    uint64_t pWBottom{0};
    // Dilation size
    uint64_t dilationD{0};
    uint64_t dilationH{0};
    uint64_t dilationW{0};
    // Cal params
    uint64_t baseNc{0};
    uint64_t baseDo{0};
    uint64_t baseHo{0};
    uint64_t baseWo{0};
    uint64_t ncTail{0};
    uint64_t doTail{0};
    uint64_t hoTail{0};
    uint64_t woTail{0};
    uint64_t ncCnt{0};
    uint64_t doCnt{0};
    uint64_t hoCnt{0};
    uint64_t woCnt{0};
    uint64_t totalCnt{0};
    uint64_t usedCoreNum{0};

    // Normal params
    uint64_t singleCoreNc{0};
    uint64_t singleCoreDo{0};
    uint64_t singleCoreHo{0};
    uint64_t singleCoreWo{0};
    // Scatter params
    uint64_t ncRound{0};
    uint64_t ncRoundTail{0};
    uint64_t totalRound{0};
    uint64_t preCoreNum{0};

    // Workspace
    uint64_t workspaceSize{0};
    // Tiling key parmas
    uint64_t tilingType{0};
    uint32_t ubCutAxis{0};
    bool ceilMode{false};
    bool isOverLap{false};
};

class MaxPool3DGradWithArgmaxTilingBase : public TilingBaseClass {
public:
    explicit MaxPool3DGradWithArgmaxTilingBase(gert::TilingContext *context_) : TilingBaseClass(context_) {}
    ~MaxPool3DGradWithArgmaxTilingBase() override {}

    const std::string nodeName = "MaxPool3DGradWithArgmax";
    MaxPool3DGradWithArgmaxTilingData tilingData;
    MaxPoolGradWithArgmaxTilingParams maxPoolGradParams;

    bool CheckInputShape();
    ge::graphStatus CheckInputDtype();
    ge::graphStatus CheckAttrShape();
    ge::graphStatus CheckInputValid();
    ge::graphStatus SetInputParams();
    ge::graphStatus SetAttrParams();
    void SetCntTailTilingParams();
    void SetOtherInputParams();
    void SetBaseTilingData();
    void PrintTilingData();

protected:
    // Order: GetShapeAttrsInfo->GetPlatformInfo->
    //        IsCapable->DoOpTiling->DoLibApiTiling->
    //        GetWorkspaceSize->GetTilingKey
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
};

class MaxPool3DGradWithArgmaxNormalTiling : public MaxPool3DGradWithArgmaxTilingBase {
public:
    explicit MaxPool3DGradWithArgmaxNormalTiling(gert::TilingContext *context_)
        : MaxPool3DGradWithArgmaxTilingBase(context_)
    {}
    ~MaxPool3DGradWithArgmaxNormalTiling() override {}

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;

private:
    uint64_t CalUBTotalSize(uint64_t baseDo, uint64_t baseHo, uint64_t baseWo);
    uint64_t CalUBTotalSize(uint64_t baseDo, uint64_t baseHo, uint64_t baseWo, bool isCoreOverlap);
    uint64_t CalBestBaseSize(uint64_t baseXoStart, uint64_t baseXoEnd, const uint32_t ubCutAxis);
    bool SetNormalParamsNotCutUB(const uint32_t ubCutAxis);
    bool SetNormalParamsNotCutUB(const uint32_t ubCutAxis, bool isCoreOverlap);
    bool SetNormalParamsCutUB();
    bool SetNormalTilingParams();
    void SetOtherTilingParams();
    void SetNormalTilingData();
    void PrintNormalTilingData();
    void CalcBaseNc();
};

class MaxPool3DGradWithArgmaxCutKTiling : public MaxPool3DGradWithArgmaxTilingBase {
public:
    explicit MaxPool3DGradWithArgmaxCutKTiling(gert::TilingContext *context_)
        : MaxPool3DGradWithArgmaxTilingBase(context_)
    {}
    ~MaxPool3DGradWithArgmaxCutKTiling() override {}

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;

private:
    void CalXp(uint64_t &baseDo, uint64_t &baseHo, uint64_t &baseWo, uint64_t &baseDp, uint64_t &baseHp,
        uint64_t &baseWp, const uint32_t ubCutAxis);
    uint64_t CalUBTotalSize(uint64_t baseDo, uint64_t baseHo, uint64_t baseWo, const uint32_t ubCutAxis);
    uint64_t CalCloseBaseSize(uint64_t notCutDimXpMul, uint64_t notCutDimXoMul,
                              uint64_t wiDim, uint64_t kCutDim, uint64_t sCutDim,
                              const uint32_t ubCutAxis);
    bool SetCutKParamsNotCutUB(const uint32_t ubCutAxis);
    bool SetCutKParamsCutUB();
    bool SetCutKTilingParams();
    void SetOtherTilingParams();
};

class MaxPool3DGradWithArgmaxScatterTiling : public MaxPool3DGradWithArgmaxTilingBase {
public:
    explicit MaxPool3DGradWithArgmaxScatterTiling(gert::TilingContext *context_)
        : MaxPool3DGradWithArgmaxTilingBase(context_)
    {}
    ~MaxPool3DGradWithArgmaxScatterTiling() override {}

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    bool IsCapable() override;

private:
    bool SetScatterTilingParams();
    void SetOtherTilingParams();
    void SetScatterTilingData();
    void PrintScatterTilingData();
};

class MaxPool3DGradWithArgmaxBaseSplitTiling : public MaxPool3DGradWithArgmaxTilingBase {
	public:
		explicit MaxPool3DGradWithArgmaxBaseSplitTiling(gert::TilingContext* context) : MaxPool3DGradWithArgmaxTilingBase(context) {
		}

		~MaxPool3DGradWithArgmaxBaseSplitTiling() override {
		}
        ge::graphStatus ParamsDefinition();
	protected:
        ge::graphStatus GetPlatformInfo() override;
		ge::graphStatus GetShapeAttrsInfo() override;
		ge::graphStatus GetWorkspaceSize() override;
		uint64_t CalcBufferSizes(const std::array<uint64_t, DHW_DIMS> part, const std::array<uint64_t, DHW_DIMS> partOut,
		                         const uint64_t partHwInp, UBBufferSize& bufSizes, uint64_t dim);
		ge::graphStatus FindSplitParts(std::array<uint64_t, DHW_DIMS> &inParts, std::array<uint64_t, DHW_DIMS> &outParts,
									   UBBufferSize& bufSizes, uint64_t dim);
		uint64_t RoundUpBlock(const uint64_t& src, const uint64_t& blockLen);
		uint64_t RoundDownBlock(const uint64_t& src, const uint64_t& blockLen);

    private:
		ge::graphStatus PadCalc();
		uint64_t InputCalc(uint64_t dim);
		ge::graphStatus InputPadCalc(const std::array<int64_t, DHW_DIMS> inpDiff);

	public:
        InputInfo inputData;
		PadOutputInfo padOutputData;
		UBBufferSize bufSizes;
		uint64_t blockLength;
		uint64_t blockLengthS;
        uint64_t ubSizeNew;
        bool isOverlap;
        ge::DataType dtype = ge::DataType::DT_FLOAT;
};

class MaxPool3DGradWithArgmaxNoSplitTiling : public MaxPool3DGradWithArgmaxBaseSplitTiling {
	public:
		explicit MaxPool3DGradWithArgmaxNoSplitTiling(gert::TilingContext* context) : MaxPool3DGradWithArgmaxBaseSplitTiling(context) {
		}

		~MaxPool3DGradWithArgmaxNoSplitTiling() override {
		}

	private:
		void DoUBTiling();
		void SetTilingData();
		uint64_t GetTilingKey() const;
		bool IsCapable();
		ge::graphStatus DoOpTiling();
		ge::graphStatus PostTiling();

		MaxPool3DGradWithArgmaxNoSplitTilingData tiling;
		uint64_t batchesPerCore;
		uint64_t leftOverBatches;
};

class MaxPool3DGradWithArgmaxSplitDTiling : public MaxPool3DGradWithArgmaxBaseSplitTiling {
	public:
		explicit MaxPool3DGradWithArgmaxSplitDTiling(gert::TilingContext* context) : MaxPool3DGradWithArgmaxBaseSplitTiling(context) {
		}
  
		~MaxPool3DGradWithArgmaxSplitDTiling() override {
		}

	private:
		void DoUBTiling();
		void SetTilingData();
		uint64_t GetTilingKey() const;
		bool IsCapable();
		ge::graphStatus DoOpTiling();
		ge::graphStatus PostTiling();

		MaxPool3DGradWithArgmaxSplitDTilingData tiling;
		SplitDataDGrad splitData;
};

class MaxPool3DGradWithArgmaxSplitHTiling : public MaxPool3DGradWithArgmaxBaseSplitTiling {
	public:
		explicit MaxPool3DGradWithArgmaxSplitHTiling(gert::TilingContext* context) : MaxPool3DGradWithArgmaxBaseSplitTiling(context) {
		}

		~MaxPool3DGradWithArgmaxSplitHTiling() override {
		}

	private:
		void DoUBTiling();
		void SetTilingData();
		uint64_t GetTilingKey() const;
		bool IsCapable();
		ge::graphStatus DoOpTiling();
		ge::graphStatus PostTiling();

		MaxPool3DGradWithArgmaxSplitHTilingData tiling;
		SplitDataHGrad splitData;
};

class MaxPool3DGradWithArgmaxSplitWTiling : public MaxPool3DGradWithArgmaxBaseSplitTiling {
	public:
		explicit MaxPool3DGradWithArgmaxSplitWTiling(gert::TilingContext* context) : MaxPool3DGradWithArgmaxBaseSplitTiling(context) {
		}

		~MaxPool3DGradWithArgmaxSplitWTiling() override {
		}

	private:
		void DoUBTiling();
		void SetTilingData();
		uint64_t GetTilingKey() const;
		bool IsCapable();
		ge::graphStatus DoOpTiling();
		ge::graphStatus PostTiling();

		MaxPool3DGradWithArgmaxSplitWTilingData tiling;
		SplitDataWGrad splitData;
};

}  // namespace optiling

#endif