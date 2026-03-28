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
 * \file weight_quant_batch_matmul_v2_tiling.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_H

#include "weight_quant_batch_matmul_v2_tiling_tool.h"
#include "weight_quant_batch_matmul_v2_tiling_key.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_tiling.h"
#include "register/op_def_registry.h"
#include "op_util.h"

#define OP_LOGI(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGD(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGW(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGE(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)

#define unlikely(x) __builtin_expect((x), 0)
#define OP_LOGE_IF(condition, return_value, op_name, fmt, ...)                                                 \
  static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool"); \
  do {                                                                                                         \
    if (unlikely(condition)) {                                                                                 \
      OP_LOGE(op_name, fmt, ##__VA_ARGS__);                                                                    \
      return return_value;                                                                                     \
    }                                                                                                          \
  } while (0)

namespace optiling {
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define CUBE_CALL_ERR_REPORT(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)
}  // namespace optiling

#define OP_LOGI_IF_RETURN(condition, return_value, op_name, fmt, ...)                                          \
  static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool"); \
  do {                                                                                                         \
    if (unlikely(condition)) {                                                                                 \
      OP_LOGI(op_name, fmt, ##__VA_ARGS__);                                                                    \
      return return_value;                                                                                     \
    }                                                                                                          \
  } while (0)

#define OP_CHECK(cond, log_func, ...) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      return ge::GRAPH_FAILED;                                 \
    }                                         \
  } while (0)

namespace ge {
template <typename T>
std::string Shape2String(const T& shape) {
  std::ostringstream oss;
  oss << "[";
  if (shape.GetDimNum() > 0) {
    for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
      oss << shape.GetDim(i) << ", ";
    }
    oss << shape.GetDim(shape.GetDimNum() - 1);
  }
  oss << "]";
  return oss.str();
}
}

namespace optiling {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }
}  // namespace optiling

namespace optiling {

enum class QuantType {
    NONE = 0,
    PER_TENSOR = 1,
    PER_CHANNEL = 2,
    PER_GROUP = 3,
};

enum class KernelTemplateType {
    SERIAL = 0,
    GENERAL_PARALLEL = 1,
    SPLIT_K = 2,
    CUSTOM_ANTIQUANT = 3,
    MSD_MULTI_CORE = 6,
    MSD_GROUP = 7,
    WEIGHT_NZ = 8,
    MIX_SPLIT_K = 9,
    ANTI_REG = 10,
};

enum class WeightFormat {
    ND = 0,
    FRACTAL_NZ = 1,
};

enum class KernelTemplateTypeExtra {
    MSD_GENERAL = 1,
    HIGH_PRECISION = 2,
};

struct WeightQuantBatchMatmulInfo {
    bool transA = false;
    bool transB = false;
    bool hasBias = false;
    bool hasAntiQuantOffset = false;
    uint64_t groupSize = 0L;
    uint64_t mSize = 0L;
    uint64_t kSize = 0L;
    uint64_t nSize = 0L;
    ge::DataType aDtype = ge::DT_FLOAT16;
    ge::DataType bDtype = ge::DT_INT8;
    ge::DataType cDtype = ge::DT_FLOAT16;
    ge::DataType biasDtype = ge::DT_FLOAT16;
    ge::DataType antiQuantScaleDtype = ge::DT_FLOAT16;
    QuantType antiQuantType = QuantType::NONE;
    QuantType quantType = QuantType::PER_TENSOR;
    // 整改Base类时统一换成使用opName_
    const char *opName;
    ge::Format aFormat = ge::FORMAT_ND;
    ge::Format bFormat = ge::FORMAT_ND;
    uint64_t innerPrecise = 0;
};

struct WeightQuantBatchMatmulV2CompileInfo {
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0cSize;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint32_t workspaceNum;
    uint32_t aivNum;
    uint32_t aicNum;
    platform_ascendc::SocVersion socVersion;
};

class WeightQuantBatchMatmulV2Tiling : public TilingBaseClass {
public:
    using TilingBaseClass::Reset;

    explicit WeightQuantBatchMatmulV2Tiling(gert::TilingContext *context) : TilingBaseClass(context)
    {
    }

    ~WeightQuantBatchMatmulV2Tiling() override = default;

protected:
    bool IsCapable() override { return true; }
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    void SetCommonTilingKeyElement(TilingKeyConfigure &tilingKeyConfigure) const;

    void Reset(){};
    void InitCompileInfo();
    // 算子名称
    const char *opName_;

    // 伪量化输入信息
    std::unique_ptr<WeightQuantBatchMatmulInfo> matmulInfoPtr_;

    // 平台相关信息
    std::unique_ptr<WeightQuantBatchMatmulV2CompileInfo> compileInfoPtr_;
};

ge::graphStatus CheckPara(gert::TilingContext *context, platform_ascendc::SocVersion socVersion);

void GetDtype(WeightQuantBatchMatmulInfo &matmulInfo, const gert::TilingContext *context);

void GetAttrs(WeightQuantBatchMatmulInfo &matmulInfo, const gert::TilingContext *context);

void GetInputs(WeightQuantBatchMatmulInfo &matmulInfo, const gert::TilingContext *context);

bool CheckInputShape(WeightQuantBatchMatmulInfo *inputParams, const gert::StorageShape *xShape,
                     const gert::StorageShape *weightShape);

bool CheckDtype(gert::TilingContext *context, WeightQuantBatchMatmulInfo *inputParams,
                platform_ascendc::SocVersion socVersion);

bool CheckInputDtype(gert::TilingContext *context, WeightQuantBatchMatmulInfo *inputParams,
                     platform_ascendc::SocVersion socVersion);

bool CheckAntiQuantDtype(gert::TilingContext *context, WeightQuantBatchMatmulInfo *inputParams,
                         platform_ascendc::SocVersion socVersion);

bool CheckQuantDtype(gert::TilingContext *context, WeightQuantBatchMatmulInfo *inputParams);

bool CheckShapeDims(WeightQuantBatchMatmulInfo *inputParams);

bool CheckBiasShape(WeightQuantBatchMatmulInfo *inputParams, const gert::StorageShape *biasShape);

bool CheckQuantShape(WeightQuantBatchMatmulInfo *inputParams, const gert::StorageShape *quantScaleShape,
                     const gert::StorageShape *quantOffsetShape);

bool CheckShape(gert::TilingContext *context, WeightQuantBatchMatmulInfo *inputParams);

bool CheckAntiQuantShape(WeightQuantBatchMatmulInfo *inputParams, const gert::StorageShape *antiQuantScaleShape,
                         const gert::StorageShape *antiQuantOffsetShape);

bool CheckAttr(gert::TilingContext *context, WeightQuantBatchMatmulInfo *inputParams);
}  // namespace optiling
#endif  // WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_H

