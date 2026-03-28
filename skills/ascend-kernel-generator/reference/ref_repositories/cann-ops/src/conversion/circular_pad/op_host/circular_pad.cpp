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
 * @file circular_pad.cpp
 */

#define OPS_UTILS_LOG_PACKAGE_TYPE "context"
#define OPS_UTILS_LOG_SUB_MOD_NAME "context"
#include "register/op_def_registry.h"
#include "circular_pad_tiling.h"
#include "tiling/tiling_templates_registry.h"

namespace ops {
#define OP_TILING_CHECK(cond, log_func, expr)   \
  do {                                          \
    if (cond) {                                 \
      std::printf(log_func);                    \
      expr;                                     \
    }                                           \
  } while (0)

#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    const char* name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName(); \
    std::printf(name, "is nullptr!");                                                            \
    return ge::GRAPH_FAILED;                                                                     \
  }
}//namespace ops

template <typename T>
inline T* GetCompileInfoPtr(gert::TilingParseContext* context) {
  return context->GetCompiledInfo<T>();
}

#define OP_LOGD(opname, ...)

namespace optiling {
constexpr int32_t ALIGN = 32;
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t TYPE_MODE1 = 1;
constexpr int64_t TYPE_MODE2 = 2;
constexpr int64_t TYPE_MODE3 = 3;
constexpr int64_t TYPE_MODE4 = 4;
constexpr int64_t TYPE_MODE5 = 5;
// CircularPadTiling
ge::graphStatus CircularPadTiling::CheckLeftAndRight()
{
    std::stringstream ss;
    if (left > 0 && right > 0 && (left > inputW || right > inputW)) {
        ss << "left/right should not be greater than inputW," <<
                "when left/right is greater than 0.";
        return ge::GRAPH_FAILED;
    } else if (left > 0 && right < 0 && (left > inputW + right || -right >= inputW)) {
        ss << "left should not be greater than inputW + right," <<
                "and abs(right) should not be greater than inputW," <<
                "when left is greater than 0 and right is not greater than 0.";
        return ge::GRAPH_FAILED;
    } else if (left < 0 && right > 0 && (-left >= inputW || right > inputW + left)) {
        ss << "right should not be greater than inputW + left," <<
                "and abs(left) should not be greater than inputW," <<
                "when right is greater than 0 and left is not greater than 0.";
        return ge::GRAPH_FAILED;
    } else if (left < 0 && right < 0 && (-left - right >= inputW)) {
        ss << "abs(left + right) should not be greater than inputW.";
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CircularPadTiling::CheckTopAndBottom()
{
    std::stringstream ss;
    if (top > 0 && bottom > 0 && (top > inputH || bottom > inputH)) {
        ss << "top/bottom should not be greater than inputH," <<
                "when top/bottom is greater than 0.";
        return ge::GRAPH_FAILED;
    } else if (top > 0 && bottom < 0 && (top > inputH + bottom || -bottom >= inputH)) {
        ss << "top should not be greater than inputH + bottom," <<
                "and abs(bottom) should not be greater than inputH," <<
                "when top is greater than 0 and bottom is not greater than 0.";
        return ge::GRAPH_FAILED;
    } else if (top < 0 && bottom > 0 && (-top >= inputH || bottom > inputH + top)) {
        ss << "bottom should not be greater than inputH + top," <<
                "and abs(top) should not be greater than inputH," <<
                "when bottom is greater than 0 and top is not greater than 0.";
        return ge::GRAPH_FAILED;
    } else if (top < 0 && bottom < 0 && (-top - bottom >= inputH)) {
        ss << "abs(top + bottom) should not be greater than inputH.";
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CircularPadTiling::CheckFrontAndBack()
{
    std::stringstream ss;
    if (front > 0 && back > 0 && (front > inputL || back > inputL)) {
        ss << "front/back should not be greater than inputL," <<
                "when front/back is greater than 0.";
        return ge::GRAPH_FAILED;
    } else if (front > 0 && back < 0 && (front > inputL + back || -back >= inputL)) {
        ss << "front should not be greater than inputL + back," <<
                "and abs(back) should not be greater than inputL," <<
                "when front is greater than 0 and back is not greater than 0.";
        return ge::GRAPH_FAILED;
    } else if (front < 0 && back > 0 && (-front >= inputL || back > inputL + front)) {
        ss << "back should not be greater than inputL + front," <<
                "and abs(front) should not be greater than inputL," <<
                "when back is greater than 0 and front is not greater than 0.";
        return ge::GRAPH_FAILED;
    } else if (front < 0 && back < 0 && (-front - back >= inputL)) {
        ss << "abs(front + back) should not be greater than inputL.";
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CircularPadTiling::CheckInput()
{
    if ((inputW >= (int64_t)ubSize / BUFFER_NUM / tSize)) {
        return ge::GRAPH_FAILED;
    }
    if ((outputH != inputH + top + bottom ||
            outputW != inputW + left + right ||
            outputL != inputL + front + back)) {
        return ge::GRAPH_FAILED;
    }
    auto ret = CheckLeftAndRight();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckTopAndBottom();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckFrontAndBack();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CircularPadTiling::CheckDtype()
{
    // 读取dtype，确定数据类型
    auto xDataType = context_->GetInputDesc(0)->GetDataType();
    if (xDataType == ge::DataType::DT_INT8){
        dataType = TYPE_MODE1;
        tSize = sizeof(int8_t);
    } else if (xDataType == ge::DataType::DT_FLOAT16){
        dataType = TYPE_MODE2;
        tSize = sizeof(uint16_t);
    } else if (xDataType == ge::DataType::DT_BF16){
        dataType = TYPE_MODE3;
        tSize = sizeof(uint16_t);
    } else if (xDataType == ge::DataType::DT_FLOAT){
        dataType = TYPE_MODE4;
        tSize = sizeof(float);
    } else if (xDataType == ge::DataType::DT_INT32){
        dataType = TYPE_MODE5;
        tSize = sizeof(int32_t);
    } else {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CircularPadTiling::DoOpTiling()
{
    auto ret = CheckDtype();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    CalculateParams();
    inOutputH = inputH + nTop + nBottom;
    inOutputW = inputW + nLeft + nRight;
    inOutputL = inputL + nFront + nBack;
    outputWAlign = (outputW * tSize + ALIGN - DIM_1) / ALIGN * ALIGN / tSize;
    inOutputWAlign = (inOutputW * tSize + ALIGN - DIM_1) / ALIGN * ALIGN / tSize;
    ret = CheckInput();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    
    if (inOutputH * outputWAlign < (int64_t)ubSize / BUFFER_NUM / tSize) {
        shapeType = DIM_1;
        workspaceLen = inOutputH * (leftAlign + inOutputWAlign + rightAlign);
    } else {
        shapeType = DIM_2;
        leftAlign = left > 0 ? leftAlign : ALIGN / tSize;
        rightAlign = right > 0 ? rightAlign : ALIGN / tSize;
        leftAlign = leftAlign > inputW ? left : leftAlign;
        rightAlign = rightAlign > inputW ? right : rightAlign;
        workspaceLen = inOutputH * (leftAlign + rightAlign);
    }
    DivCore();
    SetTilingKey();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4CircularPad(gert::TilingContext* context) {
    CircularPadTiling circularPadTiling(context);
    auto ret = circularPadTiling.GetShapeAttrsInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = circularPadTiling.GetPlatformInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = circularPadTiling.DoOpTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = circularPadTiling.GetWorkspaceSize();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = circularPadTiling.PostTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    context->SetTilingKey(circularPadTiling.GetTilingKey());
    circularPadTiling.DumpTilingInfo();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4CircularPad(gert::TilingParseContext* context) {
    OP_LOGD(context->GetNodeName(), "TilingPrepare4CircularPad enter.");
    auto compileInfo = GetCompileInfoPtr<Tiling4CircularPadCommonCompileInfo>(context);
    OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_TILING_CHECK((compileInfo->coreNum <= 0), "Get core num failed",
    return ge::GRAPH_FAILED);
    
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    OP_TILING_CHECK((compileInfo->ubSize <= 0), ("Get ub size failed"),
    return ge::GRAPH_FAILED);
    
    compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_LOGD(context->GetNodeName(), "TilingPrepare4CircularPad exit.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CircularPad)
    .Tiling(Tiling4CircularPad)
    .TilingParse<Tiling4CircularPadCommonCompileInfo>(TilingPrepare4CircularPad)
    .TilingInputsDataDependency({1});
}  // namespace optiling