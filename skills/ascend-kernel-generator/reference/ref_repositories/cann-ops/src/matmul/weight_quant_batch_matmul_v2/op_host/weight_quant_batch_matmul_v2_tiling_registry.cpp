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
 * \file weight_quant_batch_matmul_v2_tiling_registry.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "weight_quant_batch_matmul_v2_adaptive_split_tiling.h"
#include "weight_quant_batch_matmul_v2_tiling_custom.h"
#include "weight_quant_batch_matmul_v2_tiling_custom_nz_splitk.h"
#include "weight_quant_batch_matmul_v2_tiling_fixpipe.h"
#include "weight_quant_batch_matmul_v2_tiling_msd.h"
#include "weight_quant_batch_matmul_v2_tiling_msd_group.h"
#include "weight_quant_batch_matmul_v2_tiling_splitk.h"
#include "weight_quant_batch_matmul_v2_weight_nz_tiling.h"

namespace optiling {

constexpr int32_t SPLIT_K_PRIORITY = 0;
constexpr int32_t MSD_GROUP_PRIORITY = 1;
constexpr int32_t MSD_PRIORITY = 2;
constexpr int32_t CUSTOM_SPLITK_PRIORITY = 3;
constexpr int32_t CUSTOM_PRIORITY = 4;
constexpr int32_t FIXPIPE_PRIORITY = 5;
constexpr int32_t WEIGHT_NZ_PRIORITY = 6;
constexpr int32_t ANTI_REG_PRIORITY = 7;
constexpr int32_t ADAPTIVE_SPLIT_PRIORITY = 8;

REGISTER_TILING_TEMPLATE("WeightQuantBatchMatmulV2", WeightQuantBatchMatmulV2TilingSplitK, SPLIT_K_PRIORITY);
REGISTER_TILING_TEMPLATE("WeightQuantBatchMatmulV2", WeightQuantBatchMatmulV2TilingMsdGroup, MSD_GROUP_PRIORITY);
REGISTER_TILING_TEMPLATE("WeightQuantBatchMatmulV2", WeightQuantBatchMatmulV2Msd, MSD_PRIORITY);
REGISTER_TILING_TEMPLATE("WeightQuantBatchMatmulV2", WeightQuantBatchMatmulV2CustomNzSplitK, CUSTOM_SPLITK_PRIORITY);
REGISTER_TILING_TEMPLATE("WeightQuantBatchMatmulV2", WeightQuantBatchMatmulV2TilingCustom, CUSTOM_PRIORITY);
REGISTER_TILING_TEMPLATE("WeightQuantBatchMatmulV2", WeightQuantBatchMatmulV2TilingFixpipe, FIXPIPE_PRIORITY);
REGISTER_TILING_TEMPLATE("WeightQuantBatchMatmulV2", WeightQuantBatchMatmulV2WeightNz, WEIGHT_NZ_PRIORITY);
REGISTER_TILING_TEMPLATE("WeightQuantBatchMatmulV2", WeightQuantBatchMatmulV2TilingAS, ADAPTIVE_SPLIT_PRIORITY);

static ge::graphStatus WeightQuantBatchMatmulV2TilingFunc(gert::TilingContext *context)
{
    platform_ascendc::SocVersion socVersion;
    OP_LOGE_IF(context == nullptr, ge::GRAPH_FAILED, "WeightQuantBatchMatmulV2", "tilingContext is null");
    auto compileInfoPtr = reinterpret_cast<const WeightQuantBatchMatmulV2CompileInfo *>(context->GetCompileInfo());
    if (compileInfoPtr == nullptr) {
        auto platformInfoPtr = context->GetPlatformInfo();
        OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "platformInfoPtr is null");
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        socVersion = ascendcPlatform.GetSocVersion();
    } else {
        socVersion = compileInfoPtr->socVersion;
    }
    if (socVersion != platform_ascendc::SocVersion::ASCEND310P) {
        OP_LOGI(context->GetNodeName(), "Platform support Intrinsic_fix_pipe_l0c2out");
        OP_TILING_CHECK(
            CheckPara(context, socVersion) != ge::GRAPH_SUCCESS,
            VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "WeightQuantBatchMatMul para is illegal"),
            return ge::GRAPH_FAILED);
        if (socVersion != platform_ascendc::SocVersion::ASCEND910B) {
            OP_LOGI(context->GetNodeName(), "Platform support Intrinsic_data_move_l12bt bf16");
            vector<int32_t> registerList = {ANTI_REG_PRIORITY, ADAPTIVE_SPLIT_PRIORITY};
            return TilingRegistry::GetInstance().DoTilingImpl(context, registerList);
        } else {
            OP_LOGI(context->GetNodeName(), "Platform not support Intrinsic_data_move_l12bt bf16");
            return TilingRegistry::GetInstance().DoTilingImpl(context);
        }
    } else {
        OP_LOGI(context->GetNodeName(), "Platform not support Intrinsic_fix_pipe_l0c2out");
        vector<int32_t> registerList = {WEIGHT_NZ_PRIORITY};
        return TilingRegistry::GetInstance().DoTilingImpl(context, registerList);
    }
}

static ge::graphStatus TilingParseForWeightQuantBatchMatmulV2(gert::TilingParseContext *context)
{
    OP_LOGE_IF(context == nullptr, ge::GRAPH_FAILED, "WeightQuantBatchMatmulV2", "tilingParseContext is null");
    auto platformInfoPtr = context->GetPlatformInfo();
    // 在tilingParse获取不到GetPlatformInfo时，返回成功。会在之后的InitCompileInfo阶段设置compileInfo信息。
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_SUCCESS, context->GetNodeName(), "platformInfoPtr is null");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    auto compileInfoPtr = context->GetCompiledInfo<WeightQuantBatchMatmulV2CompileInfo>();
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "compileInfoPtr is null");

    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    OP_LOGE_IF(compileInfoPtr->aicNum == 0, ge::GRAPH_FAILED, context->GetNodeName(), "aicNum is 0");
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0bSize);
    compileInfoPtr->workspaceNum = ascendcPlatform.GetLibApiWorkSpaceSize();
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();

    gert::GemmCompileInfo compileInfo;
    compileInfo.ParseRuntimePlatformInfo(context->GetNodeName(), *platformInfoPtr);
    compileInfo.core_num = compileInfoPtr->aicNum;
    optiling::PlatformInfo::GetInstance().SetInstance(compileInfo);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(WeightQuantBatchMatmulV2)
    .Tiling(WeightQuantBatchMatmulV2TilingFunc)
    .TilingParse<WeightQuantBatchMatmulV2CompileInfo>(TilingParseForWeightQuantBatchMatmulV2);  // 向框架注册入口函数
} // namespace optiling

