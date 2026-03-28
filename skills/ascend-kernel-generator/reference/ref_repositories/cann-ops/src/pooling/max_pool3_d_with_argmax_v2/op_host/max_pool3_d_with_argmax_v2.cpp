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
 *  \file max_pool3_d_with_argmax_v2.cpp
 */
#include "max_pool3_d_with_argmax_v2_tiling.h"
#include "tiling/tiling_templates_registry.h"
#include "register/op_def_registry.h"

using namespace AscendC;

#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)

int32_t CheckLogLevel(int32_t moduleId, int32_t logLevel) {
    (void)moduleId;
    (void)logLevel;
    return 1;
}

namespace optiling {
#define OP_LOGE(opname, ...)
#define unlikely(x) __builtin_expect((x), 0)
#define OP_LOGE_IF(condition, return_value, op_name, fmt, ...)                                                 \
  static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool"); \
  do {                                                                                                         \
    if (unlikely(condition)) {                                                                                 \
      OP_LOGE(op_name, fmt, ##__VA_ARGS__);                                                                    \
      return return_value;                                                                                     \
    }                                                                                                          \
  } while (0)

static ge::graphStatus Tiling4MaxPool3DWithArgmaxV2(gert::TilingContext* context) {
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4MaxPool3DWithArgmaxV2(gert::TilingParseContext* context) {
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "platformInfoPtr is null");

    auto compileInfoPtr = context->GetCompiledInfo<MaxPool3DWithArgmaxV2CompileInfo>();
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "compileInfoPtr is null");

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNum();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    
    OP_TILING_CHECK(compileInfoPtr->coreNum <= 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "MaxPool3DWithArgmaxV2: coreNum = %zu, should be greater than 0",
                                                    compileInfoPtr->coreNum),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(compileInfoPtr->ubSize <= 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "MaxPool3DWithArgmaxV2: ubSize = %zu, should be greater than 0",
                                                    compileInfoPtr->ubSize),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MaxPool3DWithArgmaxV2)
    .Tiling(Tiling4MaxPool3DWithArgmaxV2)
    .TilingParse<MaxPool3DWithArgmaxV2CompileInfo>(TilingPrepare4MaxPool3DWithArgmaxV2);

}  // namespace optiling

namespace ops {
class MaxPool3DWithArgmaxV2 : public OpDef {
 public:
  explicit MaxPool3DWithArgmaxV2(const char* name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW})
        .UnknownShapeFormat({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW});

    this->Attr("ksize").AttrType(REQUIRED).ListInt();

    this->Attr("strides").AttrType(REQUIRED).ListInt();

    this->Attr("pads").AttrType(REQUIRED).ListInt();

    this->Attr("dilation").AttrType(OPTIONAL).ListInt({1, 1, 1});

    this->Attr("ceil_mode").AttrType(OPTIONAL).Bool(false);

    this->Attr("data_format").AttrType(OPTIONAL).String("NCDHW");

    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW})
        .UnknownShapeFormat({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW});

    this->Output("argmax")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
        .Format({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW})
        .UnknownShapeFormat({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW});

    OpAICoreConfig aicore_config;
    aicore_config.DynamicCompileStaticFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .ExtendCfgInfo("opFile.value", "max_pool3d_with_argmax_v2")
        .ExtendCfgInfo("opInterface.value", "max_pool3d_with_argmax_v2")
        .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");

    this->AICore().AddConfig("ascend910b", aicore_config);
  }
};

OP_ADD(MaxPool3DWithArgmaxV2);
}  // namespace ops
