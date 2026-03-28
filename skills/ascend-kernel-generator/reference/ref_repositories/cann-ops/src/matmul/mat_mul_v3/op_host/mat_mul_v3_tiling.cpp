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
 * @file mat_mul_v3.cpp
 */
#include "mat_mul_v3_tiling.h"
#include "mat_mul_v3_compile_info.h"
#include "mat_mul_v3_base_tiling.h"

#include "tiling/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "cube_tiling_runtime.h"
#include "cache_tiling.h"

#define OP_LOGI(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)

#define OP_LOGD(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)

namespace optiling {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }
}  // namespace optiling

namespace optiling {
#define CUBE_INNER_ERR_REPORT(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)
}  // namespace optiling

using namespace optiling::matmul_v3;

namespace {
static const size_t DEST_MAX = 100;
static const size_t MAX_LEN_SIMPLIFIED_KEY = 256;
static const int32_t INPUT0_INDEX = 0;
static const int32_t INPUT1_INDEX = 1;
static const int32_t BIAS_INDEX = 2;
}

namespace optiling {

REGISTER_TILING_TEMPLATE("MatMulV3", MatmulV3BaseTiling, 0);

static ge::graphStatus MatmulV3TilingFunc(gert::TilingContext* context)
{
    OP_TILING_CHECK(context == nullptr,
            CUBE_INNER_ERR_REPORT("MatMulV3", "context is null"),
            return ge::GRAPH_FAILED);
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForMatmulV3(gert::TilingParseContext *context) {
    OP_TILING_CHECK(context == nullptr,
                CUBE_INNER_ERR_REPORT("MatMulV3", "context is null"),
                return ge::GRAPH_FAILED);
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "platformInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<MatmulV3CompileInfo>();
    OP_TILING_CHECK(compileInfoPtr == nullptr,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "compileInfoPtr is null"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    platformInfo->GetPlatformRes("version", "SoC_version", compileInfoPtr->socVersionStr);
    std::string val;
    std::string dataMoveL12Bt;
    platformInfo->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_fix_pipe_l0c2out", val);
    platformInfo->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_data_move_l12bt", dataMoveL12Bt);
    compileInfoPtr->supportL0c2out = !val.empty();
    compileInfoPtr->supportL12BtBf16 = (dataMoveL12Bt.find("bf16") != std::string::npos);
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->btSize = compileInfoPtr->supportL0c2out ? 1024UL : 0UL; // 1024 is btSize
    compileInfoPtr->btSize = compileInfoPtr->supportL12BtBf16 ? 4096UL : compileInfoPtr->btSize; // 4096 is btSize
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);

    gert::GemmCompileInfo tbeCompileInfo;
    tbeCompileInfo.ParseRuntimePlatformInfo(context->GetNodeName(), *platformInfo);
    tbeCompileInfo.core_num = compileInfoPtr->aicNum;
    OP_TILING_CHECK(tbeCompileInfo.core_num <= 0L,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "aicNum value is [%d]", tbeCompileInfo.core_num),
                return ge::GRAPH_FAILED);
    optiling::PlatformInfo::GetInstance().SetInstance(tbeCompileInfo);
    OP_LOGI(context->GetNodeName(),
            "parse compile info success soc:%d, l1Size:%lu, l2Size:%lu, coreNum:%lu, supportL0c2out:%d, supportL12BtBf16:%d",
            static_cast<int>(compileInfoPtr->socVersion),
            compileInfoPtr->l1Size,
            compileInfoPtr->l2Size,
            compileInfoPtr->aicNum,
            compileInfoPtr->supportL0c2out,
            compileInfoPtr->supportL12BtBf16);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GenSimplifiedKeyForMatMulV3(gert::TilingContext *context, ge::char_t *simplifiedKey) {
    OP_LOGI(context->GetNodeName(), "Enter GenSimplifiedKeyForMatMulV3.");
    OP_TILING_CHECK(simplifiedKey == nullptr, CUBE_INNER_ERR_REPORT("MatMulV3", "simplifiedKey is null"),
                    return ge::GRAPH_FAILED);
    std::string simpKeyTemp = "";
    strcat_s(simplifiedKey, DEST_MAX, "diy,");
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(INPUT0_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(INPUT1_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetOutputDesc(0));
    if (context->GetInputDesc(BIAS_INDEX) != nullptr) {
        simpKeyTemp = std::to_string(context->GetInputDesc(INPUT0_INDEX)->GetStorageFormat()) + "/" +
                      std::to_string(context->GetInputDesc(INPUT1_INDEX)->GetStorageFormat()) + "/" +
                      std::to_string(ge::FORMAT_ND) + "/" + // bias的format均为FormatND，因此约束为仅通过FORMAT_ND参与匹配
                      std::to_string(context->GetOutputDesc(0)->GetStorageFormat()) + "/" +
                      std::to_string(context->GetInputDesc(INPUT0_INDEX)->GetDataType()) + "/" +
                      std::to_string(context->GetInputDesc(INPUT1_INDEX)->GetDataType()) + "/" +
                      std::to_string(context->GetInputDesc(BIAS_INDEX)->GetDataType()) + "/" +
                      std::to_string(context->GetOutputDesc(0)->GetDataType());
        strcat_s(simplifiedKey, DEST_MAX, simpKeyTemp.c_str());
    } else {
        // 二进制发布json有无bias时合并为同一个json发布，当无法获取bias信息时，当前约定使用input0的信息代替
        simpKeyTemp = std::to_string(context->GetInputDesc(INPUT0_INDEX)->GetStorageFormat()) + "/" +
                      std::to_string(context->GetInputDesc(INPUT1_INDEX)->GetStorageFormat()) + "/" +
                      std::to_string(ge::FORMAT_ND) + "/" +
                      std::to_string(context->GetOutputDesc(0)->GetStorageFormat()) + "/" +
                      std::to_string(context->GetInputDesc(INPUT0_INDEX)->GetDataType()) + "/" +
                      std::to_string(context->GetInputDesc(INPUT1_INDEX)->GetDataType()) + "/" +
                      std::to_string(context->GetInputDesc(INPUT0_INDEX)->GetDataType()) + "/" +
                      std::to_string(context->GetOutputDesc(0)->GetDataType());
        strcat_s(simplifiedKey, DEST_MAX, simpKeyTemp.c_str());
    }
    OP_TILING_CHECK(strlen(simplifiedKey) > MAX_LEN_SIMPLIFIED_KEY,
                           CUBE_INNER_ERR_REPORT("MatMulV3", "len of simplifiedKey exceeds max length."),
                           return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MatMulV3)
    .Tiling(MatmulV3TilingFunc)
    .TilingParse<MatmulV3CompileInfo>(TilingPrepareForMatmulV3)
    .GenSimplifiedKey(GenSimplifiedKeyForMatMulV3);
}
