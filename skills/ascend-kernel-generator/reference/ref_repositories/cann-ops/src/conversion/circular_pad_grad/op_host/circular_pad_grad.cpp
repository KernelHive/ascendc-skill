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
 * @file circular_pad_grad.cpp
 */

 #define OPS_UTILS_LOG_PACKAGE_TYPE "context"
 #define OPS_UTILS_LOG_SUB_MOD_NAME "context"
 #include "register/op_def_registry.h"
 #include "circular_pad_grad_tiling.h"
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

#define OP_LOGD(opname, ...)
}//namespace ops

template <typename T>
inline T* GetCompileInfoPtr(gert::TilingParseContext* context) {
   return context->GetCompiledInfo<T>();
}

namespace optiling {
    constexpr int32_t ALIGN = 32;
    constexpr int32_t WTYPE_SIZE = 4;
    constexpr int64_t BUFFER_NUM = 2;
    constexpr int64_t DIM_1 = 1;
    constexpr int64_t DIM_2 = 2;
    constexpr int64_t DIM_3 = 3;
    constexpr int64_t TYPE_MODE1 = 1;
    constexpr int64_t TYPE_MODE2 = 2;
    constexpr int64_t TYPE_MODE3 = 3;

    // CircularPadGradTiling
    ge::graphStatus CircularPadGradTiling::CheckLeftAndRight()
    {
        std::stringstream ss;
        if (left > 0 && right > 0 && (left + right > inputW * DIM_2 / DIM_3)) {
            ss << "left + right should not be greater than inputW * 2 / 3," <<
                  "when left/right is greater than 0.";
            return ge::GRAPH_FAILED;
        } else if (left > 0 && right <= 0 && (left > inputW / DIM_2)) {
            ss << "left should not be greater than inputW / 2," <<
                  "when left is greater than 0 and right is not greater than 0.";
            return ge::GRAPH_FAILED;
        } else if (left <= 0 && right > 0 && (right > inputW / DIM_2)) {
            ss << "right should not be greater than inputW / 2," <<
                  "when right is greater than 0 and left is not greater than 0.";
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CircularPadGradTiling::CheckTopAndBottom()
    {
        std::stringstream ss;
        if (top > 0 && bottom > 0 && (top + bottom > inputH * DIM_2 / DIM_3)) {
            ss << "top + bottom should not be greater than inputH * 2 / 3," <<
                  "when top/bottom is greater than 0.";
            return ge::GRAPH_FAILED;
        } else if (top > 0 && bottom <= 0 && (top > inputH / DIM_2)) {
            ss << "top should not be greater than inputH / 2," <<
                  "when top is greater than 0 and bottom is not greater than 0.";
            return ge::GRAPH_FAILED;
        } else if (top <= 0 && bottom > 0 && (bottom > inputH / DIM_2)) {
            ss << "bottom should not be greater than inputH  / 2," <<
                  "when bottom is greater than 0 and top is not greater than 0.";
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CircularPadGradTiling::CheckFrontAndBack()
    {
        std::stringstream ss;
        if (front > 0 && back > 0 && (front + back > inputL * DIM_2 / DIM_3)) {
            ss << "front + back should not be greater than inputL / 2," <<
                  "when front/back is greater than 0.";
            return ge::GRAPH_FAILED;
        } else if (front > 0 && back <= 0 && (front > inputL / DIM_2)) {
            ss << "front should not be greater than inputL / 2," <<
                  "when front is greater than 0 and back is not greater than 0.";
            return ge::GRAPH_FAILED;
        } else if (front <= 0 && back > 0 && (back > inputL / DIM_2)) {
            ss << "back should not be greater than inputL / 2," <<
                  "when back is greater than 0 and front is not greater than 0.";
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CircularPadGradTiling::CheckInput()
    {
        if (inputW >= usedUBSize / tSize) {
            return ge::GRAPH_FAILED;
        }
        if ((outputH != inputH - top - bottom ||
             outputW != inputW - left - right ||
             outputL != inputL - front - back)) {
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

    ge::graphStatus CircularPadGradTiling::CheckDtype()
    {
        // 读取dtype，确定数据类型
        auto xDataType = context_->GetInputDesc(0)->GetDataType();
        if (xDataType == ge::DataType::DT_FLOAT){
            dataType = TYPE_MODE1;
            tSize = sizeof(float);
        } else if (xDataType == ge::DataType::DT_FLOAT16){
            dataType = TYPE_MODE2;
            tSize = sizeof(uint16_t);
        } else if (xDataType == ge::DataType::DT_BF16){
            dataType = TYPE_MODE3;
            tSize = sizeof(uint16_t);
        } else {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CircularPadGradTiling::DoOpTiling()
    {
        auto ret = CheckDtype();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        CalculateParams();
        inOutputH = inputH - pTop - pBottom;
        inOutputW = inputW - pLeft - pRight;
        inOutputL = inputL - pFront - pBack;
        if (dataType == TYPE_MODE1) {
            usedUBSize = ubSize / BUFFER_NUM;
        } else {
            usedUBSize  = ubSize / BUFFER_NUM / DIM_3;
        }
        ret = CheckInput();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        
        if (inputH * inputWAlign < usedUBSize / tSize) {
            shapeType = DIM_1;
            workspaceLen = inputH * (ALIGN / WTYPE_SIZE + inputWAlign + ALIGN / WTYPE_SIZE);
        } else {
            shapeType = DIM_2;
            workspaceLen = inOutputH * (ALIGN / WTYPE_SIZE + inputWAlign + ALIGN / WTYPE_SIZE);
        }

        DivCore();
        SetTilingKey();
        SetTilingData();
        return ge::GRAPH_SUCCESS;
    }


    static ge::graphStatus Tiling4CircularPadGrad(gert::TilingContext* context) {
        CircularPadGradTiling circularPadGradTiling(context);
        auto ret = circularPadGradTiling.GetShapeAttrsInfo();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        ret = circularPadGradTiling.GetPlatformInfo();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        ret = circularPadGradTiling.DoOpTiling();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        ret = circularPadGradTiling.GetWorkspaceSize();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        ret = circularPadGradTiling.PostTiling();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        context->SetTilingKey(circularPadGradTiling.GetTilingKey());
        circularPadGradTiling.DumpTilingInfo();
        return ge::GRAPH_SUCCESS;
    }
    
    static ge::graphStatus TilingPrepare4CircularPadGrad(gert::TilingParseContext* context) {
      OP_LOGD(context->GetNodeName(), "TilingPrepare4CircularPadGrad enter.");
      auto compileInfo = GetCompileInfoPtr<Tiling4CircularPadCommonCompileInfo>(context);
      OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
      auto platformInfo = context->GetPlatformInfo();
      OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
      auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
      compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
      OP_TILING_CHECK((compileInfo->coreNum <= 0),
                      "Get core num failed ", return ge::GRAPH_FAILED);
    
      ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
      OP_TILING_CHECK((compileInfo->ubSize <= 0),
                      "Get ub size failed ", return ge::GRAPH_FAILED);
    
      compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
      OP_LOGD(context->GetNodeName(), "TilingPrepare4CircularPadGrad exit.");
      return ge::GRAPH_SUCCESS;
    }
    
    IMPL_OP_OPTILING(CircularPadGrad)
        .Tiling(Tiling4CircularPadGrad)
        .TilingParse<Tiling4CircularPadCommonCompileInfo>(TilingPrepare4CircularPadGrad)
        .TilingInputsDataDependency({1});
    }  // namespace optiling