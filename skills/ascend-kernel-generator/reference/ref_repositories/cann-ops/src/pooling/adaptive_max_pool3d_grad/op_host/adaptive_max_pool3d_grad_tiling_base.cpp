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
 * \file adaptive_max_pool3d_grad_tiling_base.cpp
 * \brief
 */
#include "adaptive_max_pool3d_grad_tiling.h"
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
#define OP_LOGI(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
namespace optiling {

constexpr uint64_t D_SHAPE_INDEX = 2;
constexpr uint64_t H_SHAPE_INDEX = 3;
constexpr uint64_t W_SHAPE_INDEX = 4;
constexpr uint64_t D_ATTR_INDEX = 0;
constexpr uint64_t H_ATTR_INDEX = 1;
constexpr uint64_t W_ATTR_INDEX = 2;

uint64_t AdaptiveMaxPool3DGradTilingBase::CeilDiv(const uint64_t dividend, const uint64_t divisor) {
    if (divisor == 0) {
      return 0;
    }
    return (dividend + divisor - 1) / divisor;
  }

uint64_t AdaptiveMaxPool3DGradTilingBase::Gcd(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }

    uint64_t c;
    if (a < b) {
        c = a;
        a = b;
        b = c;
    }

    while (a % b != 0) {
        c = a % b;
        a = b;
        b = c;
    }

    return b;
}
bool AdaptiveMaxPool3DGradTilingBase::CheckInputShape()
{
    const gert::StorageShape *xShape = context_->GetInputShape(X_INDEX);
    const gert::StorageShape *gradShape = context_->GetInputShape(GRAD_INDEX);
    const gert::StorageShape *argmaxShape = context_->GetInputShape(ARGMAX_INDEX);
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    size_t gradDimNum = gradShape->GetStorageShape().GetDimNum();
    size_t argmaxDimNum = argmaxShape->GetStorageShape().GetDimNum();

    // xDimNum should be 5(format:NCDHW)
    OP_TILING_CHECK((xDimNum != NCDHW_DIM_NUM) ||
                    (gradDimNum != NCDHW_DIM_NUM) ||
                    (argmaxDimNum != NCDHW_DIM_NUM),
                    OP_LOGE(context_->GetNodeName(),
                            "Input dim num should equal = %lu, actual is xDim: %lu, gradDim: %lu, argmaxDim: %lu.",
                            NCDHW_DIM_NUM, xDimNum, gradDimNum, argmaxDimNum),
                    return false);
    for (uint32_t i = 0; i < xDimNum; i++) {
        OP_TILING_CHECK(xShape->GetStorageShape().GetDim(i) == 0,
                        OP_LOGE(context_->GetNodeName(), "Input x shape can not be 0."),
                        return false);
    }

    // gradShape&argmaxShape's shape should be equal
    for (size_t i = 0; i < xDimNum; i++) {
        uint64_t gradDimValue = gradShape->GetStorageShape().GetDim(i);
        uint64_t argmaxDimValue = argmaxShape->GetStorageShape().GetDim(i);
        OP_TILING_CHECK(
            gradDimValue != argmaxDimValue,
            OP_LOGE(context_->GetNodeName(),
                    "Input dim check invalid, grad[%lu] is %lu, argmax[%lu] is %lu, not equal.",
                    i, gradDimValue, i, argmaxDimValue),
            return false);
    }

    // Input NCDim should be equal
    for (size_t i = 0; i < NC_DIM_NUM; i++) {
        uint64_t xDimValue = xShape->GetStorageShape().GetDim(i);
        uint64_t gradDimValue = gradShape->GetStorageShape().GetDim(i);
        OP_TILING_CHECK(
            (gradDimValue != xDimValue),
            OP_LOGE(context_->GetNodeName(),
                    "Input dim check invalid, grad[%lu] is %lu, x[%lu] is %lu, not equal.",
                    i, gradDimValue, i, xDimValue),
            return false);
    }

    return true;
}

ge::graphStatus AdaptiveMaxPool3DGradTilingBase::CheckInputDtype()
{
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(X_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(GRAD_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(ARGMAX_INDEX));
    auto xDataType = context_->GetInputDesc(X_INDEX)->GetDataType();
    auto gradDataType = context_->GetInputDesc(GRAD_INDEX)->GetDataType();
    auto argmaxDataType = context_->GetInputDesc(ARGMAX_INDEX)->GetDataType();

    OP_TILING_CHECK(xDataType != gradDataType,
                    OP_LOGE(context_->GetNodeName(), "Data type invalid, x data type not equal grad data type."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((xDataType != ge::DT_FLOAT) && (xDataType != ge::DT_FLOAT16) && (xDataType != ge::DT_BF16),
                    OP_LOGE(context_->GetNodeName(), "Data type invalid, x data type not fp32/fp16/bf16."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(argmaxDataType != ge::DT_INT32,
                    OP_LOGE(context_->GetNodeName(), "Data type invalid, argmax data type not equal int32."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveMaxPool3DGradTilingBase::CheckInputValid()
{
    // Check index range
    OP_TILING_CHECK(maxPoolGradParams.diDim * maxPoolGradParams.hiDim * maxPoolGradParams.wiDim > MAX_INT32,
                    OP_LOGE(context_->GetNodeName(),
                            "Shape too big, diDim * hiDim * wiDim should not bigger than max range of int32."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline uint64_t AdaptiveMaxPool3DGradTilingBase::CalKIndexStart(uint64_t &kIdx, uint64_t &innerDim, uint64_t &outerDim)
{
    return outerDim == 0 ? kIdx * innerDim : kIdx * innerDim / outerDim;
}

inline uint64_t AdaptiveMaxPool3DGradTilingBase::CalKIndexEnd(uint64_t &kIdx, uint64_t &innerDim, uint64_t &outerDim)
{
    return CeilDiv((kIdx + 1) * innerDim, outerDim);
}

inline uint64_t AdaptiveMaxPool3DGradTilingBase::CalKIndexLen(uint64_t &kIdx, uint64_t &innerDim, uint64_t &outerDim)
{
    return CalKIndexEnd(kIdx, innerDim, outerDim) - CalKIndexStart(kIdx, innerDim, outerDim);
}

inline uint64_t AdaptiveMaxPool3DGradTilingBase::GetMaxK(uint64_t &innerDim, uint64_t &outerDim)
{
    uint64_t maxKValue = 0;
    for (uint64_t kIdx = 0; kIdx < outerDim; kIdx++) {
        uint64_t kValue = CalKIndexLen(kIdx, innerDim, outerDim);
        if (kValue > maxKValue) {
            maxKValue = kValue;
        }
    }    
    return maxKValue;
}

ge::graphStatus AdaptiveMaxPool3DGradTilingBase::SetInputParams()
{
    const gert::Shape xShape = context_->GetInputShape(X_INDEX)->GetStorageShape();
    const gert::Shape gradShape = context_->GetInputShape(GRAD_INDEX)->GetStorageShape();
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    uint64_t n = xShape.GetDim(0);
    uint64_t c = xShape.GetDim(1);
    maxPoolGradParams.ncDim = n * c;
    maxPoolGradParams.diDim = xShape.GetDim(D_SHAPE_INDEX);
    maxPoolGradParams.hiDim = xShape.GetDim(H_SHAPE_INDEX);
    maxPoolGradParams.wiDim = xShape.GetDim(W_SHAPE_INDEX);
    maxPoolGradParams.doDim = gradShape.GetDim(D_SHAPE_INDEX);
    maxPoolGradParams.hoDim = gradShape.GetDim(H_SHAPE_INDEX);
    maxPoolGradParams.woDim = gradShape.GetDim(W_SHAPE_INDEX);
    maxPoolGradParams.kdMax = GetMaxK(maxPoolGradParams.diDim, maxPoolGradParams.doDim);
    maxPoolGradParams.khMax = GetMaxK(maxPoolGradParams.hiDim, maxPoolGradParams.hoDim);
    maxPoolGradParams.kwMax = GetMaxK(maxPoolGradParams.wiDim, maxPoolGradParams.woDim);
    maxPoolGradParams.dGcd = Gcd(maxPoolGradParams.doDim, maxPoolGradParams.diDim);
    bool isDOverLap = maxPoolGradParams.diDim % maxPoolGradParams.doDim != 0;
    bool isHOverLap = maxPoolGradParams.hiDim % maxPoolGradParams.hoDim != 0;
    bool isWOverLap = maxPoolGradParams.wiDim % maxPoolGradParams.woDim != 0;
    maxPoolGradParams.isOverLap = (isDOverLap || isHOverLap || isWOverLap);
    maxPoolGradParams.vl = NUM_PER_REP_B32;
    return ge::GRAPH_SUCCESS;
}

void AdaptiveMaxPool3DGradTilingBase::SetOtherInputParams()
{
    auto xDataType = context_->GetInputDesc(X_INDEX)->GetDataType();
    uint64_t xDtypeSize = GetSizeByDataType(xDataType);
    maxPoolGradParams.xDtypeSize = xDtypeSize;
    maxPoolGradParams.indexDtypeSize = GetSizeByDataType(ge::DT_INT32);
}

ge::graphStatus AdaptiveMaxPool3DGradTilingBase::GetShapeAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "Enter AdaptiveMaxPool3DGradTilingBase GetShapeAttrsInfo.");
    OP_TILING_CHECK(ge::GRAPH_SUCCESS != CheckInputDtype(),
                    OP_LOGE(context_->GetNodeName(), "The input dtype is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckInputShape(),
                    OP_LOGE(context_->GetNodeName(), "The input relationship is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(ge::GRAPH_SUCCESS != SetInputParams(),
                    OP_LOGE(context_->GetNodeName(), "Set input shape failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(ge::GRAPH_SUCCESS != CheckInputValid(),
                OP_LOGE(context_->GetNodeName(), "The input shape is invalid."), return ge::GRAPH_FAILED);
    SetOtherInputParams();
    return ge::GRAPH_SUCCESS;
}

bool AdaptiveMaxPool3DGradTilingBase::IsCapable()
{
    return false;
}

ge::graphStatus AdaptiveMaxPool3DGradTilingBase::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveMaxPool3DGradTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveMaxPool3DGradTilingBase::GetPlatformInfo() {
    auto platformInfo = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    maxPoolGradParams.totalCoreNum = ascendcPlatform.GetCoreNum();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, maxPoolGradParams.maxUbSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveMaxPool3DGradTilingBase::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveMaxPool3DGradTilingBase::PostTiling()
{
    context_->SetBlockDim(maxPoolGradParams.usedCoreNum);
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    size_t usrWorkspaceSize = maxPoolGradParams.workspaceSize;
    size_t sysWorkSpaceSize = 16 * 1024 * 1024;
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrWorkspaceSize + sysWorkSpaceSize;

    return ge::GRAPH_SUCCESS;
}

void AdaptiveMaxPool3DGradTilingBase::SetCntTailTilingParams()
{
    maxPoolGradParams.ncCnt = CeilDiv(maxPoolGradParams.ncDim, maxPoolGradParams.baseNc);
    maxPoolGradParams.doCnt = CeilDiv(maxPoolGradParams.doDim, maxPoolGradParams.baseDo);
    maxPoolGradParams.hoCnt = CeilDiv(maxPoolGradParams.hoDim, maxPoolGradParams.baseHo);
    maxPoolGradParams.woCnt = CeilDiv(maxPoolGradParams.woDim, maxPoolGradParams.baseWo);
    maxPoolGradParams.ncTail = maxPoolGradParams.ncDim - (maxPoolGradParams.ncCnt - 1) * maxPoolGradParams.baseNc;
    maxPoolGradParams.doTail = maxPoolGradParams.doDim - (maxPoolGradParams.doCnt - 1) * maxPoolGradParams.baseDo;
    maxPoolGradParams.hoTail = maxPoolGradParams.hoDim - (maxPoolGradParams.hoCnt - 1) * maxPoolGradParams.baseHo;
    maxPoolGradParams.woTail = maxPoolGradParams.woDim - (maxPoolGradParams.woCnt - 1) * maxPoolGradParams.baseWo;
    maxPoolGradParams.totalCnt =
        maxPoolGradParams.ncCnt * maxPoolGradParams.doCnt * maxPoolGradParams.hoCnt * maxPoolGradParams.woCnt;
}

void AdaptiveMaxPool3DGradTilingBase::SetBaseTilingData()
{
    tilingData.set_ncDim(maxPoolGradParams.ncDim);
    tilingData.set_diDim(maxPoolGradParams.diDim);
    tilingData.set_hiDim(maxPoolGradParams.hiDim);
    tilingData.set_wiDim(maxPoolGradParams.wiDim);
    tilingData.set_doDim(maxPoolGradParams.doDim);
    tilingData.set_hoDim(maxPoolGradParams.hoDim);
    tilingData.set_woDim(maxPoolGradParams.woDim);
    tilingData.set_kdMax(maxPoolGradParams.kdMax);
    tilingData.set_khMax(maxPoolGradParams.khMax);
    tilingData.set_kwMax(maxPoolGradParams.kwMax);
    tilingData.set_baseNc(maxPoolGradParams.baseNc);
    tilingData.set_baseDo(maxPoolGradParams.baseDo);
    tilingData.set_baseHo(maxPoolGradParams.baseHo);
    tilingData.set_baseWo(maxPoolGradParams.baseWo);
    tilingData.set_ncTail(maxPoolGradParams.ncTail);
    tilingData.set_doTail(maxPoolGradParams.doTail);
    tilingData.set_hoTail(maxPoolGradParams.hoTail);
    tilingData.set_woTail(maxPoolGradParams.woTail);
    tilingData.set_ncCnt(maxPoolGradParams.ncCnt);
    tilingData.set_doCnt(maxPoolGradParams.doCnt);
    tilingData.set_hoCnt(maxPoolGradParams.hoCnt);
    tilingData.set_woCnt(maxPoolGradParams.woCnt);
    tilingData.set_totalCnt(maxPoolGradParams.totalCnt);
    tilingData.set_usedCoreNum(maxPoolGradParams.usedCoreNum);
    tilingData.set_totalUBSize(maxPoolGradParams.maxUbSize);
}

void AdaptiveMaxPool3DGradTilingBase::PrintTilingData()
{
    OP_LOGI(context_->GetNodeName(),
        "TilingData nc: %lu, di: %lu, hi: %lu, wi: %lu do: %lu, ho: %lu, wo: %lu, "
        "kdMax: %lu, khMax: %lu, kwMax: %lu, "
        "baseNc: %lu, baseDo: %lu, baseHo: %lu, baseWo: %lu, ncTail: %lu, doTail: %lu, hoTail: %lu, woTail: %lu, "
        "ncCnt: %lu, doCnt: %lu, hoCnt: %lu, woCnt: %lu, totalCnt: %lu, usedCoreNum: %lu, totalUBSize: %lu.",
        tilingData.get_ncDim(), tilingData.get_diDim(), tilingData.get_hiDim(), tilingData.get_wiDim(),
        tilingData.get_doDim(), tilingData.get_hoDim(), tilingData.get_woDim(),
        tilingData.get_kdMax(), tilingData.get_khMax(), tilingData.get_kwMax(),
        tilingData.get_baseNc(), tilingData.get_baseDo(), tilingData.get_baseHo(), tilingData.get_baseWo(),
        tilingData.get_ncTail(), tilingData.get_doTail(), tilingData.get_hoTail(), tilingData.get_woTail(),
        tilingData.get_ncCnt(), tilingData.get_doCnt(), tilingData.get_hoCnt(), tilingData.get_woCnt(),
        tilingData.get_totalCnt(), tilingData.get_usedCoreNum(), tilingData.get_totalUBSize());
}

uint64_t AdaptiveMaxPool3DGradTilingBase::GetTilingKey() const
{
    uint64_t tilingKey = maxPoolGradParams.tilingType;
    if (maxPoolGradParams.isOverLap) {
        tilingKey += TILING_OVERLAP;
    }
    OP_LOGI(context_->GetNodeName(), "TilingKey is %lu.", tilingKey);
    return tilingKey;
}
} // namespace optiling
