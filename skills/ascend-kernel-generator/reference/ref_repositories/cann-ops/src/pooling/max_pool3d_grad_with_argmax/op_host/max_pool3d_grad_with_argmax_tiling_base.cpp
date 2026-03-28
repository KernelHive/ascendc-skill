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
 * \file max_pool3d_grad_with_argmax_tiling_base.cpp
 * \brief
 */

//#include "vector_tiling_util.h"
#include "max_pool3d_grad_with_argmax_tiling.h"

namespace optiling {

constexpr uint64_t D_SHAPE_INDEX = 2;
constexpr uint64_t H_SHAPE_INDEX = 3;
constexpr uint64_t W_SHAPE_INDEX = 4;
constexpr uint64_t D_ATTR_INDEX = 0;
constexpr uint64_t H_ATTR_INDEX = 1;
constexpr uint64_t W_ATTR_INDEX = 2;

bool MaxPool3DGradWithArgmaxTilingBase::CheckInputShape()
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

ge::graphStatus MaxPool3DGradWithArgmaxTilingBase::CheckInputDtype()
{
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(X_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(GRAD_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(ARGMAX_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(Y_INDEX));
    auto xDataType = context_->GetInputDesc(X_INDEX)->GetDataType();
    auto gradDataType = context_->GetInputDesc(GRAD_INDEX)->GetDataType();
    auto argmaxDataType = context_->GetInputDesc(ARGMAX_INDEX)->GetDataType();
    auto yOutDataType = context_->GetOutputDesc(Y_INDEX)->GetDataType();

    OP_TILING_CHECK(xDataType != gradDataType,
                    OP_LOGE(context_->GetNodeName(), "Data type invalid, x data type not equal grad data type."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(xDataType != yOutDataType,
                    OP_LOGE(context_->GetNodeName(), "Data type invalid, x data type not equal y data type."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((xDataType != ge::DT_FLOAT) && (xDataType != ge::DT_FLOAT16) && (xDataType != ge::DT_BF16),
                    OP_LOGE(context_->GetNodeName(), "Data type invalid, x data type not fp32/fp16/bf16."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(argmaxDataType != ge::DT_INT32,
                    OP_LOGE(context_->GetNodeName(), "Data type invalid, argmax data type not equal int32."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradWithArgmaxTilingBase::CheckAttrShape()
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    int32_t kSizeDimNum = attrs->GetListInt(KSIZE_ATTR_INDEX)->GetSize();
    int32_t stridesDimNum = attrs->GetListInt(STRIDES_ATTR_INDEX)->GetSize();
    int32_t padsDimNum = attrs->GetListInt(PADS_ATTR_INDEX)->GetSize();
    int32_t dilationsDimNum = attrs->GetListInt(DILATION_ATTR_INDEX)->GetSize();

    // Check attr dim num
    OP_TILING_CHECK((kSizeDimNum != DHW_DIM_NUM) && (kSizeDimNum != 1),
                    OP_LOGE(context_->GetNodeName(),
                            "Attr kSize dim num invalid, dim num should equal 3 or 1."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((stridesDimNum != DHW_DIM_NUM) && (stridesDimNum != 1) && (stridesDimNum != 0),
                    OP_LOGE(context_->GetNodeName(),
                            "Attr strides dim num invalid, dim num should equal 3 or 1 or 0."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((padsDimNum != DHW_DIM_NUM) && (padsDimNum != 1),
                    OP_LOGE(context_->GetNodeName(),
                            "Attr pads dim num invalid, dim num should equal 3 or 1."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((dilationsDimNum != DHW_DIM_NUM) && (dilationsDimNum != 1),
                    OP_LOGE(context_->GetNodeName(),
                            "Attr dilations dim num invalid, dim num should equal 3 or 1."),
                    return ge::GRAPH_FAILED);

    // Check attr value bigger than 0
    auto kSizeVector = attrs->GetListInt(KSIZE_ATTR_INDEX)->GetData();
    auto stridesVector = attrs->GetListInt(STRIDES_ATTR_INDEX)->GetData();
    auto padsVector = attrs->GetListInt(PADS_ATTR_INDEX)->GetData();
    auto dilationsVector = attrs->GetListInt(DILATION_ATTR_INDEX)->GetData();
    for (uint32_t i = 0; i < (uint32_t)kSizeDimNum; i++) {
        OP_TILING_CHECK((kSizeVector[i] <= 0), OP_LOGE(context_->GetNodeName(),
            "Attr value invalid, kSize[%u] is %ld, should bigger than 0.", i, kSizeVector[i]),
        return ge::GRAPH_FAILED);
    }
    for (uint32_t i = 0; i < (uint32_t)stridesDimNum; i++) {
        OP_TILING_CHECK((stridesVector[i] <= 0), OP_LOGE(context_->GetNodeName(),
            "Attr value invalid, strides[%u] is %ld, should bigger than 0.", i, stridesVector[i]),
        return ge::GRAPH_FAILED);
    }
    for (uint32_t i = 0; i < (uint32_t)padsDimNum; i++) {
        OP_TILING_CHECK((padsVector[i] < 0), OP_LOGE(context_->GetNodeName(),
            "Attr value invalid, pads[%u] is %ld, should bigger or equal 0.", i, padsVector[i]),
        return ge::GRAPH_FAILED);
    }
    for (uint32_t i = 0; i < (uint32_t)dilationsDimNum; i++) {
        OP_TILING_CHECK((dilationsVector[i] <= 0), OP_LOGE(context_->GetNodeName(),
            "Attr value invalid, dilations[%u] is %ld, should bigger than 0.", i, dilationsVector[i]),
        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradWithArgmaxTilingBase::SetInputParams()
{
    const gert::Shape xShape = context_->GetInputShape(X_INDEX)->GetStorageShape();
    const gert::Shape gradShape = context_->GetInputShape(GRAD_INDEX)->GetStorageShape();
    uint64_t n = xShape.GetDim(0);
    uint64_t c = xShape.GetDim(1);
    maxPoolGradParams.ncDim = n * c;
    maxPoolGradParams.diDim = xShape.GetDim(D_SHAPE_INDEX);
    maxPoolGradParams.hiDim = xShape.GetDim(H_SHAPE_INDEX);
    maxPoolGradParams.wiDim = xShape.GetDim(W_SHAPE_INDEX);
    maxPoolGradParams.doDim = gradShape.GetDim(D_SHAPE_INDEX);
    maxPoolGradParams.hoDim = gradShape.GetDim(H_SHAPE_INDEX);
    maxPoolGradParams.woDim = gradShape.GetDim(W_SHAPE_INDEX);
    maxPoolGradParams.vl = NUM_PER_REP_B32;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradWithArgmaxTilingBase::SetAttrParams()
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    int32_t kSizeDimNum = attrs->GetListInt(KSIZE_ATTR_INDEX)->GetSize();
    int32_t stridesDimNum = attrs->GetListInt(STRIDES_ATTR_INDEX)->GetSize();
    int32_t padsDimNum = attrs->GetListInt(PADS_ATTR_INDEX)->GetSize();
    int32_t dilationsDimNum = attrs->GetListInt(DILATION_ATTR_INDEX)->GetSize();
    auto kSizeVector = attrs->GetListInt(KSIZE_ATTR_INDEX)->GetData();
    auto stridesVector = attrs->GetListInt(STRIDES_ATTR_INDEX)->GetData();
    auto padsVector = attrs->GetListInt(PADS_ATTR_INDEX)->GetData();
    auto dilationsVector = attrs->GetListInt(DILATION_ATTR_INDEX)->GetData();
    bool ceilMode = *attrs->GetBool(CEIL_MODE_ATTR_INDEX);
    maxPoolGradParams.ceilMode = ceilMode;
    maxPoolGradParams.kd = kSizeVector[D_ATTR_INDEX];
    maxPoolGradParams.kh = (kSizeDimNum == 1) ? maxPoolGradParams.kd : kSizeVector[H_ATTR_INDEX];
    maxPoolGradParams.kw = (kSizeDimNum == 1) ? maxPoolGradParams.kd : kSizeVector[W_ATTR_INDEX];
    if (stridesDimNum == 0) {
        maxPoolGradParams.sd = maxPoolGradParams.kd;
        maxPoolGradParams.sh = maxPoolGradParams.kd;
        maxPoolGradParams.sw = maxPoolGradParams.kd;
    } else {
        maxPoolGradParams.sd = stridesVector[D_ATTR_INDEX];
        maxPoolGradParams.sh = (stridesDimNum == 1) ? maxPoolGradParams.sd : stridesVector[H_ATTR_INDEX];
        maxPoolGradParams.sw = (stridesDimNum == 1) ? maxPoolGradParams.sd : stridesVector[W_ATTR_INDEX];
    }
    maxPoolGradParams.pDTop = padsVector[D_ATTR_INDEX];
    maxPoolGradParams.pHTop = (padsDimNum == 1) ? maxPoolGradParams.pDTop : padsVector[H_ATTR_INDEX];
    maxPoolGradParams.pWTop = (padsDimNum == 1) ? maxPoolGradParams.pDTop : padsVector[W_ATTR_INDEX];
    maxPoolGradParams.dilationD = dilationsVector[D_ATTR_INDEX];
    maxPoolGradParams.dilationH = (dilationsDimNum == 1) ? maxPoolGradParams.dilationD : dilationsVector[H_ATTR_INDEX];
    maxPoolGradParams.dilationW = (dilationsDimNum == 1) ? maxPoolGradParams.dilationD : dilationsVector[W_ATTR_INDEX];
    bool isDOverLap = maxPoolGradParams.kd > maxPoolGradParams.sd ? true : false;
    bool isHOverLap = maxPoolGradParams.kh > maxPoolGradParams.sh ? true : false;
    bool isWOverLap = maxPoolGradParams.kw > maxPoolGradParams.sw ? true : false;
    maxPoolGradParams.isOverLap = (isDOverLap || isHOverLap || isWOverLap);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradWithArgmaxTilingBase::CheckInputValid()
{
    const uint64_t kd = maxPoolGradParams.kd;
    const uint64_t kh = maxPoolGradParams.kh;
    const uint64_t kw = maxPoolGradParams.kw;
    const uint64_t sd = maxPoolGradParams.sd;
    const uint64_t sh = maxPoolGradParams.sh;
    const uint64_t sw = maxPoolGradParams.sw;
    const uint64_t pDTop = maxPoolGradParams.pDTop;
    const uint64_t pHTop = maxPoolGradParams.pHTop;
    const uint64_t pWTop = maxPoolGradParams.pWTop;
    const uint64_t dilationD = maxPoolGradParams.dilationD;
    const uint64_t dilationH = maxPoolGradParams.dilationH;
    const uint64_t dilationW = maxPoolGradParams.dilationW;

    // check 1
    OP_TILING_CHECK((pDTop > (kd / 2)) || (pHTop > (kh / 2)) || (pWTop > (kw / 2)),
                    OP_LOGE(context_->GetNodeName(),
                            "Attr size invalid, padSize should smaller than kernelSize div 2"),
                    return ge::GRAPH_FAILED);
    // check 2
    OP_TILING_CHECK((pDTop > ((kd - 1) * dilationD + 1) / 2) ||
                    (pHTop > ((kh - 1) * dilationH + 1) / 2) ||
                    (pWTop > ((kw - 1) * dilationW + 1) / 2),
                    OP_LOGE(context_->GetNodeName(),
                            "Attr size invalid, padSize should smaller than ((kernelSize - 1) * dilation + 1) / 2."),
                    return ge::GRAPH_FAILED);
    // check 3
    // Check outerDim invaild
    int64_t doExpected, hoExpected, woExpected;
    if (maxPoolGradParams.ceilMode) {
        doExpected = vectorutil::CeilDiv((maxPoolGradParams.diDim + NUM_TWO * pDTop - dilationD * (kd - 1) - 1), sd) + 1;
        hoExpected = vectorutil::CeilDiv((maxPoolGradParams.hiDim + NUM_TWO * pHTop - dilationH * (kh - 1) - 1), sh) + 1;
        woExpected = vectorutil::CeilDiv((maxPoolGradParams.wiDim + NUM_TWO * pWTop - dilationW * (kw - 1) - 1), sw) + 1;
    } else {
        doExpected = (maxPoolGradParams.diDim + NUM_TWO * pDTop - dilationD * (kd - 1) - 1) / sd + 1;
        hoExpected = (maxPoolGradParams.hiDim + NUM_TWO * pHTop - dilationH * (kh - 1) - 1) / sh + 1;
        woExpected = (maxPoolGradParams.wiDim + NUM_TWO * pWTop - dilationW * (kw - 1) - 1) / sw + 1;
    }
    doExpected = ((doExpected - 1) * sd >= maxPoolGradParams.diDim + pDTop) ? doExpected - 1 : doExpected;
    hoExpected = ((hoExpected - 1) * sh >= maxPoolGradParams.hiDim + pHTop) ? hoExpected - 1 : hoExpected;
    woExpected = ((woExpected - 1) * sw >= maxPoolGradParams.wiDim + pWTop) ? woExpected - 1 : woExpected;
    OP_TILING_CHECK((doExpected <= 0) || ((uint64_t)doExpected != maxPoolGradParams.doDim) ||
                    (hoExpected <= 0) || ((uint64_t)hoExpected != maxPoolGradParams.hoDim) ||
                    (woExpected <= 0) || ((uint64_t)woExpected != maxPoolGradParams.woDim),
                    OP_LOGE(context_->GetNodeName(),
                            "OuterDim size invalid, doExpected: %ld, hoExpected: %ld, woExpected: %ld.",
                            doExpected, hoExpected, woExpected),
                    return ge::GRAPH_FAILED);
    // check 4
    // Check index range
    OP_TILING_CHECK(maxPoolGradParams.diDim * maxPoolGradParams.hiDim * maxPoolGradParams.wiDim > MAX_INT32,
                    OP_LOGE(context_->GetNodeName(),
                            "Shape too big, diDim * hiDim * wiDim should not bigger than max range of int32."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void MaxPool3DGradWithArgmaxTilingBase::SetOtherInputParams()
{
    auto xDataType = context_->GetInputDesc(X_INDEX)->GetDataType();
    uint64_t xDtypeSize = GetSizeByDataType(xDataType);
    maxPoolGradParams.xDtypeSize = xDtypeSize;
    maxPoolGradParams.indexDtypeSize = GetSizeByDataType(ge::DT_INT32);

    uint64_t dLenCalFromDo = (maxPoolGradParams.doDim - 1) * maxPoolGradParams.sd + maxPoolGradParams.kd;
    uint64_t dLenCalFromDi = maxPoolGradParams.diDim + maxPoolGradParams.pDTop;
    maxPoolGradParams.pDBottom = dLenCalFromDo < dLenCalFromDi ? 0 : dLenCalFromDo - dLenCalFromDi;
    uint64_t hLenCalFromHo = (maxPoolGradParams.hoDim - 1) * maxPoolGradParams.sh + maxPoolGradParams.kh;
    uint64_t hLenCalFromHi = maxPoolGradParams.hiDim + maxPoolGradParams.pHTop;
    maxPoolGradParams.pHBottom = hLenCalFromHo < hLenCalFromHi ? 0 : hLenCalFromHo - hLenCalFromHi;
    uint64_t wLenCalFromWo = (maxPoolGradParams.woDim - 1) * maxPoolGradParams.sw + maxPoolGradParams.kw;
    uint64_t wLenCalFromWi = maxPoolGradParams.wiDim + maxPoolGradParams.pWTop;
    maxPoolGradParams.pWBottom = wLenCalFromWo < wLenCalFromWi ? 0 : wLenCalFromWo - wLenCalFromWi;
}

ge::graphStatus MaxPool3DGradWithArgmaxTilingBase::GetShapeAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "Enter MaxPool3DGradWithArgmaxTilingBase GetShapeAttrsInfo.");
    OP_TILING_CHECK(ge::GRAPH_SUCCESS != CheckInputDtype(),
                    OP_LOGE(context_->GetNodeName(), "The input dtype is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckInputShape(),
                    OP_LOGE(context_->GetNodeName(), "The input relationship is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(ge::GRAPH_SUCCESS != CheckAttrShape(),
                    OP_LOGE(context_->GetNodeName(), "The attr shape is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(ge::GRAPH_SUCCESS != SetInputParams(),
                    OP_LOGE(context_->GetNodeName(), "Set input shape failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(ge::GRAPH_SUCCESS != SetAttrParams(),
                    OP_LOGE(context_->GetNodeName(), "Set attr shape failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(ge::GRAPH_SUCCESS != CheckInputValid(),
                    OP_LOGE(context_->GetNodeName(), "The input shape is invalid."), return ge::GRAPH_FAILED);
    SetOtherInputParams();
    return ge::GRAPH_SUCCESS;
}

bool MaxPool3DGradWithArgmaxTilingBase::IsCapable()
{
    return false;
}

ge::graphStatus MaxPool3DGradWithArgmaxTilingBase::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradWithArgmaxTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradWithArgmaxTilingBase::GetPlatformInfo() {
    auto compileInfo = reinterpret_cast<const Tiling4MaxPool3DGradWithArgmaxCompileInfo*>(context_->GetCompileInfo());
    maxPoolGradParams.totalCoreNum = compileInfo->totalCoreNum;
    maxPoolGradParams.maxUbSize = compileInfo->maxUbSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradWithArgmaxTilingBase::GetWorkspaceSize()
{
    context_->SetBlockDim(maxPoolGradParams.usedCoreNum);
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    size_t usrWorkspaceSize = maxPoolGradParams.workspaceSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    size_t sysWorkSpaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrWorkspaceSize + sysWorkSpaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradWithArgmaxTilingBase::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

void MaxPool3DGradWithArgmaxTilingBase::SetCntTailTilingParams()
{
    maxPoolGradParams.ncCnt = vectorutil::CeilDiv(maxPoolGradParams.ncDim, maxPoolGradParams.baseNc);
    maxPoolGradParams.doCnt = vectorutil::CeilDiv(maxPoolGradParams.doDim, maxPoolGradParams.baseDo);
    maxPoolGradParams.hoCnt = vectorutil::CeilDiv(maxPoolGradParams.hoDim, maxPoolGradParams.baseHo);
    maxPoolGradParams.woCnt = vectorutil::CeilDiv(maxPoolGradParams.woDim, maxPoolGradParams.baseWo);
    maxPoolGradParams.ncTail = maxPoolGradParams.ncDim - (maxPoolGradParams.ncCnt - 1) * maxPoolGradParams.baseNc;
    maxPoolGradParams.doTail = maxPoolGradParams.doDim - (maxPoolGradParams.doCnt - 1) * maxPoolGradParams.baseDo;
    maxPoolGradParams.hoTail = maxPoolGradParams.hoDim - (maxPoolGradParams.hoCnt - 1) * maxPoolGradParams.baseHo;
    maxPoolGradParams.woTail = maxPoolGradParams.woDim - (maxPoolGradParams.woCnt - 1) * maxPoolGradParams.baseWo;
    maxPoolGradParams.totalCnt =
        maxPoolGradParams.ncCnt * maxPoolGradParams.doCnt * maxPoolGradParams.hoCnt * maxPoolGradParams.woCnt;
}

void MaxPool3DGradWithArgmaxTilingBase::SetBaseTilingData()
{
    tilingData.set_ncDim(maxPoolGradParams.ncDim);
    tilingData.set_diDim(maxPoolGradParams.diDim);
    tilingData.set_hiDim(maxPoolGradParams.hiDim);
    tilingData.set_wiDim(maxPoolGradParams.wiDim);
    tilingData.set_doDim(maxPoolGradParams.doDim);
    tilingData.set_hoDim(maxPoolGradParams.hoDim);
    tilingData.set_woDim(maxPoolGradParams.woDim);
    tilingData.set_kd(maxPoolGradParams.kd);
    tilingData.set_kh(maxPoolGradParams.kh);
    tilingData.set_kw(maxPoolGradParams.kw);
    tilingData.set_sd(maxPoolGradParams.sd);
    tilingData.set_sh(maxPoolGradParams.sh);
    tilingData.set_sw(maxPoolGradParams.sw);
    tilingData.set_padDTop(maxPoolGradParams.pDTop);
    tilingData.set_padDBottom(maxPoolGradParams.pDBottom);
    tilingData.set_padHTop(maxPoolGradParams.pHTop);
    tilingData.set_padHBottom(maxPoolGradParams.pHBottom);
    tilingData.set_padWTop(maxPoolGradParams.pWTop);
    tilingData.set_padWBottom(maxPoolGradParams.pWBottom);
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

void MaxPool3DGradWithArgmaxTilingBase::PrintTilingData()
{
    OP_LOGI(context_->GetNodeName(),
        "TilingData nc: %lu, di: %lu, hi: %lu, wi: %lu do: %lu, ho: %lu, wo: %lu, "
        "kd: %lu, kh: %lu, kw: %lu sd: %lu, sh: %lu, sw: %lu, "
        "pDTop: %lu, pHTop: %lu, pWTop: %lu pDBottom: %lu, pHBottom: %lu, pWBottom: %lu, "
        "baseNc: %lu, baseDo: %lu, baseHo: %lu, baseWo: %lu ncTail: %lu, doTail: %lu, hoTail: %lu, woTail: %lu, "
        "ncCnt: %lu, doCnt: %lu, hoCnt: %lu, woCnt: %lu totalCnt: %lu, usedCoreNum: %lu, totalUBSize: %lu, "
        "ceilMode: %d, isOverLap: %d.",
        tilingData.get_ncDim(), tilingData.get_diDim(), tilingData.get_hiDim(), tilingData.get_wiDim(),
        tilingData.get_doDim(), tilingData.get_hoDim(), tilingData.get_woDim(), tilingData.get_kd(),
        tilingData.get_kh(), tilingData.get_kw(), tilingData.get_sd(), tilingData.get_sh(), tilingData.get_sw(),
        tilingData.get_padDTop(), tilingData.get_padHTop(), tilingData.get_padWTop(), tilingData.get_padDBottom(),
        tilingData.get_padHBottom(), tilingData.get_padWBottom(), tilingData.get_baseNc(), tilingData.get_baseDo(),
        tilingData.get_baseHo(), tilingData.get_baseWo(), tilingData.get_ncTail(), tilingData.get_doTail(),
        tilingData.get_hoTail(), tilingData.get_woTail(), tilingData.get_ncCnt(), tilingData.get_doCnt(),
        tilingData.get_hoCnt(), tilingData.get_woCnt(), tilingData.get_totalCnt(), tilingData.get_usedCoreNum(),
        tilingData.get_totalUBSize(), maxPoolGradParams.ceilMode, maxPoolGradParams.isOverLap);
}

uint64_t MaxPool3DGradWithArgmaxTilingBase::GetTilingKey() const
{
    uint64_t tilingKey = maxPoolGradParams.tilingType;
    if (maxPoolGradParams.isOverLap) {
        tilingKey += TILING_OVERLAP;
    }
    if (TILING_TYPE_CUTK == maxPoolGradParams.tilingType) {
        tilingKey += maxPoolGradParams.ubCutAxis;
    }
    OP_LOGI(context_->GetNodeName(), "TilingKey is %lu.", tilingKey);
    return tilingKey;
}
} // namespace optiling
