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
 * \file circular_pad_common.cc
 * \brief
 */
#define OPS_UTILS_LOG_PACKAGE_TYPE "context"
#define OPS_UTILS_LOG_SUB_MOD_NAME "context"

#include "circular_pad_common.h"
#include <sstream>
constexpr int32_t ALIGN = 32;
constexpr int32_t DATA_TYPE = 10;
constexpr int32_t TYPE_2D = 200;
constexpr int32_t TYPE_3D = 300;
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t DIM_4 = 4;
constexpr int64_t DIM_5 = 5;
constexpr int64_t DIM_6 = 6;

namespace ops{
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                  \
  if ((ptr) == nullptr) {                                                                          \
    const char* name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName();   \
    std::printf(name, "is nullptr!");                                                              \
    REPORT_INNER_ERR_MSG("EZ9999", "op[%s], %s is nullptr!", name, #ptr);                             \
    return ge::GRAPH_FAILED;                                                                       \
  }
    
#define OP_TILING_CHECK(cond, log_func, expr)  \
  do {                                         \
    if (cond) {                                \
    std::printf(log_func);                     \
    expr;                                      \
    }                                          \
  } while (0)
    
#define OPS_CHECK_NULL_WITH_CONTEXT_RET(context, ptr, ret)                                        \
  if ((ptr) == nullptr) {                                                                         \
    const char* name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName();  \
    std::printf(name, "is nullptr!");                                                             \
    REPORT_INNER_ERR_MSG("EZ9999", "op[%s], %s is nullptr!", name, #ptr);                            \
    return ret;                                                                                   \
  }
}
    
bool AddWorkspace(gert::TilingContext* context, const size_t workspace) {
    size_t* workspace_size = context->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, workspace_size, false);
    *workspace_size = workspace;
    return true;
}

namespace optiling {
// CircularPadCommonTiling
ge::graphStatus CircularPadCommonTiling::GetShapeAttrsInfo()
{
    // get paddings
    auto paddings = context_->GetInputTensor(DIM_1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, paddings);
    auto paddingsNum = paddings->GetShapeSize();
    OP_TILING_CHECK(paddingsNum < DIM_4, "paddings num should be greater than 4",
        return ge::GRAPH_FAILED);
    auto padding_valus = paddings->GetData<int64_t>();
    int32_t paddingDim = paddingsNum - DIM_4;
    top = padding_valus[paddingDim];
    bottom = padding_valus[paddingDim + DIM_1];
    left = padding_valus[paddingDim + DIM_2];
    right = padding_valus[paddingDim + DIM_3];
    if (paddingsNum >= DIM_6) {
        front = padding_valus[paddingDim - DIM_2];
        back = padding_valus[paddingDim - DIM_1];
    }

    // get x shape
    auto x_shape = context_->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x_shape);
    auto xDimNum = x_shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(xDimNum < DIM_2, "x dims should be greater than 2",
        return ge::GRAPH_FAILED);
    int32_t xDim = xDimNum - DIM_2;
    for (int32_t i = 0; i < xDim; i++) {
        auto shape = x_shape->GetStorageShape().GetDim(i);
        totalTasks *= (shape > 0 ? shape : DIM_1);
    }
    inputH = x_shape->GetStorageShape().GetDim(xDim);
    inputW = x_shape->GetStorageShape().GetDim(xDim + DIM_1);
    if (xDimNum >= DIM_3) {
        inputL = x_shape->GetStorageShape().GetDim(xDim - DIM_1);
    }

    // get y shape
    auto y_shape = context_->GetOutputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, y_shape);
    auto yDimNum = y_shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(yDimNum < DIM_2, "y dims should be greater than 2",
        return ge::GRAPH_FAILED);
    int32_t yDim = yDimNum - DIM_2;
    outputH = y_shape->GetStorageShape().GetDim(yDim);
    outputW = y_shape->GetStorageShape().GetDim(yDim + DIM_1);
    if (yDimNum >= DIM_3) {
        outputL = y_shape->GetStorageShape().GetDim(yDim - DIM_1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CircularPadCommonTiling::GetPlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const Tiling4CircularPadCommonCompileInfo*>(context_->GetCompileInfo());
        OPS_CHECK_NULL_WITH_CONTEXT(context_, compileInfoPtr);
        ubSize = compileInfoPtr->ubSize;
        coreNum = compileInfoPtr->coreNum;
        sysWorkspaceSize = compileInfoPtr->sysWorkspaceSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        coreNum = ascendcPlatform.GetCoreNum();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CircularPadCommonTiling::GetWorkspaceSize()
{
    AddWorkspace(context_, totalTasks * workspaceLen * sizeof(float) + sysWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CircularPadCommonTiling::PostTiling()
{
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void CircularPadCommonTiling::CalculateParams()
{
    if (front != 0 || back != 0) {
        dPad = true;
    }
    pLeft = left > 0 ? left : 0;
    pRight = right > 0 ? right : 0;
    pTop = top > 0 ? top : 0;
    pBottom = bottom > 0 ? bottom : 0;
    pFront = front > 0 ? front : 0;
    pBack = back > 0 ? back : 0;

    nLeft = left > 0 ? 0 : left;
    nRight = right > 0 ? 0 : right;
    nTop = top > 0 ? 0 : top;
    nBottom = bottom > 0 ? 0 : bottom;
    nFront = front > 0 ? 0 : front;
    nBack = back > 0 ? 0 : back;
    inputWAlign = (inputW * tSize + ALIGN - DIM_1) / ALIGN * ALIGN / tSize;
    
    leftAlign = (pLeft * tSize + ALIGN - DIM_1) / ALIGN * ALIGN / tSize;
    rightAlign = (pRight * tSize + ALIGN - DIM_1) / ALIGN * ALIGN / tSize;
}

void CircularPadCommonTiling::DivCore()
{
    if (dPad) {
        int64_t batchNum = totalTasks / inputL;
        perCoreTaskNum = (batchNum / coreNum) * inputL;
        tailTaskNum = (batchNum % coreNum) * inputL;
        useCoreNum = perCoreTaskNum > 0 ? coreNum : tailTaskNum / inputL;
    } else {
        perCoreTaskNum = totalTasks / coreNum;
        tailTaskNum = totalTasks % coreNum;
        useCoreNum = perCoreTaskNum > 0 ? coreNum : tailTaskNum;
    }
}

void CircularPadCommonTiling::SetTilingKey()
{
    if (dPad) {
        tilingKey_ = TYPE_3D + DATA_TYPE * dataType + shapeType;
    } else {
        tilingKey_ = TYPE_2D + DATA_TYPE * dataType + shapeType;
    }
}

uint64_t CircularPadCommonTiling::GetTilingKey()
{
    return tilingKey_;
}

void CircularPadCommonTiling::SetTilingData()
{
    tilingData_.set_inputH(inputH);
    tilingData_.set_inputW(inputW);
    tilingData_.set_inputL(inputL);
    tilingData_.set_outputH(outputH);
    tilingData_.set_outputW(outputW);
    tilingData_.set_outputL(outputL);
    tilingData_.set_left(left);
    tilingData_.set_right(right);
    tilingData_.set_top(top);
    tilingData_.set_bottom(bottom);
    tilingData_.set_front(front);
    tilingData_.set_back(back);
    tilingData_.set_perCoreTaskNum(perCoreTaskNum);
    tilingData_.set_tailTaskNum(tailTaskNum);
    tilingData_.set_workspaceLen(workspaceLen);
    context_->SetBlockDim(useCoreNum);
}

void CircularPadCommonTiling::DumpTilingInfo()
{
    int32_t enable = AlogCheckDebugLevel(static_cast<int32_t>(OP), DLOG_DEBUG);
    if (enable != 1) {
        return;
    }
    auto buf = (int64_t *)context_->GetRawTilingData()->GetData();
    auto bufLen = context_->GetRawTilingData()->GetDataSize();
    std::ostringstream oss;
    oss << "Start to dump tiling info. tilingkey:" << context_->GetTilingKey() << ", tiling data size:" << bufLen
        << ", content:";
    for (size_t i = 0; i < bufLen / sizeof(int64_t); i++) {
        oss << *(buf + i) << ",";
        if (oss.str().length() > 640) { // Split according to 640 to avoid truncation
            OPS_LOG_D(context_, "%s", oss.str().c_str());
            oss.str("");
        }
    }
    OPS_LOG_D(context_, "%s", oss.str().c_str());
}
}