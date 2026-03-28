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
 * \file avg_pool_3d_grad.cpp
 * \brief
 */

#include <string>
#include <vector>
#include <math.h>
#include <algorithm>
#include "platform/platform_info.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_type.h"
#include "avg_pool3d_grad_tiling.h"

#include "graph/utils/type_utils.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"

namespace optiling {
// AvgPool3DGrad for vector
constexpr int64_t GRAD_SHAPE = 5;
constexpr int64_t ATTR_SIZE = 3;
constexpr int64_t SCALR_UB_SIZE = 20480;
constexpr int64_t BF16_BYTE_SIZE = 2;
constexpr int64_t BF16_UB_PART = 3;
constexpr int64_t BF16_UB_PART_NO_OVERLAP = 4;
constexpr int64_t FP32_UB_PART = 2;
constexpr int64_t D_DIM = 0;
constexpr int64_t H_DIM = 1;
constexpr int64_t W_DIM = 2;
constexpr int64_t W_DIM_OFFSET = 1;
constexpr int64_t H_DIM_OFFSET = 2;
constexpr int64_t D_DIM_OFFSET = 3;
constexpr int64_t COUNT_IDX = 4;
constexpr int64_t DIVISOR_IDX = 5;
constexpr int64_t FORMAT_IDX = 6;
constexpr int64_t ALIGN_BYTES = 32;
constexpr int64_t TILINGKEY_CAST = 1000;
constexpr int64_t TILINGKEY_NO_CAST = 2000;
constexpr int64_t TILINGKEY_ONLY_T_FP32 = 3000;
constexpr int64_t TILINGKEY_ONLY_T_BF16 = 4000;
constexpr int64_t NDHWC_D_DIM = 1;
constexpr int64_t NCDHW_D_DIM = 2;
constexpr size_t ORIG_INPUT_SHAPE_INDEX = 0;

struct DHWParam {
    int d = 0;
    int h = 0;
    int w = 0;
};

class AvgPool3dGradTiling {
public:
    explicit AvgPool3dGradTiling(gert::TilingContext* context) : tilingContext_(context) {}
    ge::graphStatus Init();
    ge::graphStatus SetKernelTiling();
private:
    inline ge::graphStatus InitDHW();
    inline void Tiling4NCParam(const uint64_t ubSizePlatform, const int64_t ncShape,
                               const int64_t dhwShape, const ge::DataType &dtype);
    inline void Tiling4HWParam(const uint64_t ubSizePlatform, const ge::DataType &dtype);
    inline void Tiling4CastCopyOut(const int64_t ubSizePlatform, const int64_t ncShape, const int64_t dhwShape);
    inline void SetTilingKey(const ge::DataType &dtype);
    inline int64_t AlignUp(int64_t num, int64_t rnd) const;
    inline int64_t AlignDown(int64_t num, int64_t rnd);

private:
    gert::TilingContext* tilingContext_ = nullptr;
    AvgPool3dGradTilingParam tilingData_;
    uint64_t coreNum_ = 0;
    uint64_t isDetermine_ = 0;
    uint64_t countIncludePad_ = 0;
    int64_t divisorOverride_ = 0;
    uint64_t isOverlap_ = 0;
    uint64_t isOnlyT_ = 0;
    uint64_t N_ = 0;
    uint64_t C_ = 0;
    std::string dataFormat_;
    DHWParam outDHW_;
    DHWParam inDHW_;
    DHWParam kDHW_;
    DHWParam dDHW_;
    DHWParam padDHW_;
private:
    // NC
    uint64_t normalCoreNCNum_ = 0;
    uint64_t lastCoreNCNum_ = 0;
    uint64_t ncAlign_ = 0;
    uint64_t ncTotal_ = 0;
    uint64_t ncNum_ = 0;
    uint64_t ncCount_ = 0;
    uint64_t ncTail_ = 0;
    uint64_t nLine_ = 0;
private:
    // HW
    uint64_t normalCoreHWNum_ = 0;
    uint64_t lastCoreHWNum_ = 0;
    uint64_t hwAlign_ = 0;
    uint64_t hwTotal_ = 0;
    uint64_t hwNum_ = 0;
    uint64_t hwCount_ = 0;
    uint64_t hwTail_ = 0;
    uint64_t hwLine_ = 0;
private:
    // copy cast
    uint64_t maxDataNumInUb_ = 0;
    uint64_t normalCoreNum_ = 0;
    uint64_t tailCoreNum_ = 0;
    uint64_t normalCoreDataNum_ = 0;
    uint64_t tailCoreDataNum_ = 0;
    uint64_t normalCoreFormerCopyTime_ = 0;
    uint64_t normalCoreTailCopyTime_ = 0;
    uint64_t normalCoreFormerDataNum_ = 0;
    uint64_t normalCoreTailDataNum_ = 0;
    uint64_t tailCoreFormerCopyTime_ = 0;
    uint64_t tailCoreTailCopyTime_ = 0;
    uint64_t tailCoreFormerDataNum_ = 0;
    uint64_t tailCoreTailDataNum_ = 0;
};

int64_t AvgPool3dGradTiling::AlignDown(int64_t num, int64_t rnd) {
    return ((rnd) == 0 ? 0 : ((num / rnd) * rnd));
}

int64_t AvgPool3dGradTiling::AlignUp(int64_t num, int64_t rnd) const {
    return ((rnd) == 0 ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}

ge::graphStatus AvgPool3dGradTiling::InitDHW()
{
    auto inputShape1 = tilingContext_->GetInputShape(1);
    auto gradShape = inputShape1->GetStorageShape();
    auto attrs = tilingContext_->GetAttrs();
    auto kSize = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(1);
    auto pads = attrs->GetAttrPointer<gert::ContinuousVector>(2);
    if (kSize->GetSize() != ATTR_SIZE || strides->GetSize() != ATTR_SIZE ||
        pads->GetSize() != ATTR_SIZE) {
        return ge::GRAPH_FAILED;
    }
    auto kSizeData = reinterpret_cast<const int64_t*>(kSize->GetData());
    auto stridesData = reinterpret_cast<const int64_t*>(strides->GetData());
    auto padsData = reinterpret_cast<const int64_t*>(pads->GetData());

    auto inputShape0 = tilingContext_->GetInputShape(0);
    auto shapeDim = inputShape0->GetStorageShape().GetDim(0);
    const gert::Tensor* shapeTensor = tilingContext_->GetInputTensor(ORIG_INPUT_SHAPE_INDEX);
    const int32_t* shapeValue = shapeTensor->GetData<int32_t>();
    if(shapeValue == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint64_t gradShapeDimD = (dataFormat_ == "NDHWC") ? NDHWC_D_DIM : NCDHW_D_DIM;
    inDHW_.d = shapeValue[shapeDim - D_DIM_OFFSET];
    inDHW_.h = shapeValue[shapeDim - H_DIM_OFFSET];
    inDHW_.w = shapeValue[shapeDim - W_DIM_OFFSET];
    outDHW_.d = gradShape.GetDim(gradShapeDimD);
    outDHW_.h = gradShape.GetDim(gradShapeDimD + H_DIM);
    outDHW_.w = gradShape.GetDim(gradShapeDimD + W_DIM);
    kDHW_.d = kSizeData[D_DIM];
    kDHW_.h = kSizeData[H_DIM];
    kDHW_.w = kSizeData[W_DIM];
    dDHW_.d = stridesData[D_DIM];
    dDHW_.h = stridesData[H_DIM];
    dDHW_.w = stridesData[W_DIM];
    padDHW_.d = padsData[D_DIM];
    padDHW_.h = padsData[H_DIM];
    padDHW_.w = padsData[W_DIM];
    isOverlap_ = static_cast<uint64_t>(dDHW_.d < kDHW_.d || dDHW_.h < kDHW_.h || dDHW_.w < kDHW_.w);
    isOnlyT_ = (kDHW_.h == 1) && (dDHW_.h == 1) && (kDHW_.w == 1) && (dDHW_.w == 1);
    return ge::GRAPH_SUCCESS;
}

void AvgPool3dGradTiling::Tiling4CastCopyOut(const int64_t ubSizePlatform, const int64_t ncShape,
                                             const int64_t dhwShape)
{
    auto gradShape = dhwShape * ncShape;
    auto ubSizeLeft = ubSizePlatform - SCALR_UB_SIZE;
    maxDataNumInUb_ = ubSizeLeft / BF16_BYTE_SIZE / BF16_UB_PART;
    normalCoreNum_ = gradShape % coreNum_;
    tailCoreNum_ = coreNum_ - normalCoreNum_;
    normalCoreDataNum_ = (gradShape + coreNum_ - 1) / coreNum_;
    tailCoreDataNum_ = gradShape / coreNum_;

    auto normalCoreCopyTime =   (normalCoreDataNum_ + maxDataNumInUb_ - 1) / maxDataNumInUb_;
    auto tailCoreCopyTime = (tailCoreDataNum_ + maxDataNumInUb_ - 1) / maxDataNumInUb_;
   
    normalCoreFormerCopyTime_ = normalCoreDataNum_ % normalCoreCopyTime;
    normalCoreTailCopyTime_ = normalCoreCopyTime - normalCoreFormerCopyTime_;
    normalCoreFormerDataNum_ = (normalCoreDataNum_ + normalCoreCopyTime - 1) / normalCoreCopyTime;
    normalCoreTailDataNum_ = normalCoreDataNum_ / normalCoreCopyTime;

    tailCoreFormerCopyTime_ = tailCoreDataNum_ % tailCoreCopyTime;
    tailCoreTailCopyTime_ = tailCoreCopyTime - tailCoreFormerCopyTime_;
    tailCoreFormerDataNum_ = (tailCoreDataNum_ + tailCoreCopyTime - 1) / tailCoreCopyTime;
    tailCoreTailDataNum_ = tailCoreDataNum_ / tailCoreCopyTime;
}

void AvgPool3dGradTiling::Tiling4NCParam(const uint64_t ubSizePlatform, const int64_t ncShape,
                                         const int64_t dhwShape, const ge::DataType &dtype)
{
    auto ubSizeLeft = ubSizePlatform - SCALR_UB_SIZE;
    auto ubPart = (dtype == ge::DT_FLOAT) ? FP32_UB_PART : ( (isOverlap_ != 0) ? BF16_UB_PART : BF16_UB_PART_NO_OVERLAP);
    auto byteSize = GetSizeByDataType(dtype);
    auto ubSize4NC = ubSizeLeft / ubPart;
    auto ubSize4NCAlign = AlignDown(ubSize4NC, ALIGN_BYTES);
    int64_t ubMaxNCNum = ubSize4NCAlign / byteSize;
    ncTotal_ = ncShape;
    if (ncShape > ubMaxNCNum) {
        ncAlign_ = ubMaxNCNum;
        ncCount_ = ubMaxNCNum;
        ncNum_ = (ncShape + ncCount_ - 1) / ncCount_;
        ncTail_ = ncShape - (ncNum_ - 1) * ncCount_;
        nLine_ = 1;
    } else {
        auto alignBlocks = ALIGN_BYTES / byteSize;
        ncAlign_ = AlignUp(ncShape, alignBlocks);
        if(ncAlign_ == 0) {
            return;
        }
        ncCount_ = ncShape;
        ncNum_ = 1;
        ncTail_ = ncCount_;
        nLine_ = ubMaxNCNum / ncAlign_;
    }
    auto dividShape = dhwShape * ncNum_;
    normalCoreNCNum_ = dividShape / coreNum_;
    lastCoreNCNum_ = dividShape - normalCoreNCNum_ * (coreNum_ - 1);
}

void AvgPool3dGradTiling::Tiling4HWParam(const uint64_t ubSizePlatform, const ge::DataType &dtype)
{
    auto ubSizeLeft = ubSizePlatform - SCALR_UB_SIZE;
    auto ubPart = (dtype == ge::DT_FLOAT) ? FP32_UB_PART : ( (isOverlap_ != 0) ? BF16_UB_PART : BF16_UB_PART_NO_OVERLAP);
    auto byteSize = GetSizeByDataType(dtype);
    auto ubSize4HW = ubSizeLeft / ubPart;
    auto ubSize4HWAlign = AlignDown(ubSize4HW, ALIGN_BYTES);
    int64_t ubMaxHWNum = ubSize4HWAlign / byteSize;
    auto hwShape = outDHW_.h * outDHW_.w;
    hwTotal_ = hwShape;
    if (hwShape > ubMaxHWNum) {
        hwAlign_ = ubMaxHWNum;
        hwCount_ = ubMaxHWNum;
        hwNum_ = (hwShape + hwCount_ - 1) / hwCount_;
        hwTail_ = hwShape - (hwNum_ - 1) * hwCount_;
        hwLine_ = 1;
    } else {
        auto alignBlocks = ALIGN_BYTES / byteSize;
        hwAlign_ = AlignUp(hwShape, alignBlocks);
        if(hwAlign_ == 0) {
            return;
        }
        hwCount_ = hwShape;
        hwNum_ = 1;
        hwTail_ = hwCount_;
        hwLine_ = ubMaxHWNum / hwAlign_;
    }
    auto dividShape = N_ * C_ * outDHW_.d * hwNum_;
    normalCoreHWNum_ = dividShape / coreNum_;
    lastCoreHWNum_ = dividShape - normalCoreHWNum_ * (coreNum_ - 1);
}

void AvgPool3dGradTiling::SetTilingKey(const ge::DataType &dtype)
{
    bool isFp32 = dtype == ge::DT_FLOAT;
    uint64_t tilingKey;
    if ( (isOnlyT_ != 0) && dataFormat_ == "NCDHW") {
        tilingKey = isFp32 ? TILINGKEY_ONLY_T_FP32 : TILINGKEY_ONLY_T_BF16;
    } else {
        tilingKey = isFp32 ? TILINGKEY_NO_CAST : TILINGKEY_CAST;
    }
    tilingContext_->SetTilingKey(tilingKey);
}

ge::graphStatus AvgPool3dGradTiling::Init()
{
    auto platformInfo = tilingContext_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t core_num = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatform;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
    uint64_t totalUbSize = 0;
    platformInfo->GetLocalMemSize(fe::LocalMemType::UB, totalUbSize);

    auto gradShape = tilingContext_->GetInputShape(1)->GetStorageShape();
    auto dtype = tilingContext_->GetInputDesc(1)->GetDataType();
    auto attrs = tilingContext_->GetAttrs();
    countIncludePad_ = static_cast<uint64_t>(*attrs->GetAttrPointer<bool>(COUNT_IDX));
    divisorOverride_ = static_cast<int64_t>(*attrs->GetAttrPointer<int>(DIVISOR_IDX));
    dataFormat_ = attrs->GetStr(FORMAT_IDX);
    if (dataFormat_ != "NDHWC" && dataFormat_ != "NCDHW") {
        return ge::GRAPH_FAILED;
    }
    if (dataFormat_ == "NDHWC" && gradShape.GetDim(0) != 1) {
        return ge::GRAPH_FAILED;
    }
    if (gradShape.GetDimNum() != GRAD_SHAPE) {
        return ge::GRAPH_FAILED;
    }

    int64_t dhwShape = 1;
    int64_t ncShape = 1;
    if (dataFormat_ == "NDHWC") {
        ncShape = gradShape.GetDim(GRAD_SHAPE - 1);
        for (int i = NDHWC_D_DIM; i < NDHWC_D_DIM + ATTR_SIZE; i++) {
            dhwShape *= gradShape.GetDim(i);
        }
    } else {
        N_ = gradShape.GetDim(0);
        C_ = gradShape.GetDim(1);
        ncShape = N_ * C_;
    }

    auto ret = InitDHW();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t castWorkspaceSize = 0;
    size_t *currentWorkSpace = tilingContext_->GetWorkspaceSizes(1);
    if (dtype == ge::DT_BF16 || dtype == ge::DT_FLOAT16) {
        castWorkspaceSize = inDHW_.d * inDHW_.h * inDHW_.w * ncShape * sizeof(float);
    }
    sysWorkspaceSize += castWorkspaceSize;
    currentWorkSpace[0] = sysWorkspaceSize;
    if (dataFormat_ == "NDHWC") {
        coreNum_ = std::min(core_num, static_cast<uint32_t>(dhwShape));
    } else {
        coreNum_ = std::min(core_num, static_cast<uint32_t>(ncShape * outDHW_.d));
    }
    if (tilingContext_->GetDeterministic() == 1) {
        coreNum_ = 1;
        isDetermine_ = 1;
    }
    if (coreNum_ == 0) {
        return ge::GRAPH_FAILED;
    }

    // tiling for HW or NC
    if ( (isOnlyT_ != 0) && dataFormat_ == "NCDHW") {
        Tiling4HWParam(ubSizePlatform, dtype);
    } else {
        Tiling4NCParam(ubSizePlatform, ncShape, dhwShape, dtype);
    }

    int64_t inDhwShape = inDHW_.d * inDHW_.h * inDHW_.w;
    if (dtype == ge::DT_BF16 || dtype == ge::DT_FLOAT16) {
        Tiling4CastCopyOut(ubSizePlatform, ncShape, inDhwShape);
    }
    // set tiling key
    SetTilingKey(dtype);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvgPool3dGradTiling::SetKernelTiling()
{
    tilingContext_->SetBlockDim(coreNum_);
    tilingData_.attrParam.set_N(N_);
    tilingData_.attrParam.set_C(C_);
    tilingData_.attrParam.set_inD(inDHW_.d);
    tilingData_.attrParam.set_inH(inDHW_.h);
    tilingData_.attrParam.set_inW(inDHW_.w);
    tilingData_.attrParam.set_outD(outDHW_.d);
    tilingData_.attrParam.set_outH(outDHW_.h);
    tilingData_.attrParam.set_outW(outDHW_.w);
    tilingData_.attrParam.set_kD(kDHW_.d);
    tilingData_.attrParam.set_kH(kDHW_.h);
    tilingData_.attrParam.set_kW(kDHW_.w);
    tilingData_.attrParam.set_dD(dDHW_.d);
    tilingData_.attrParam.set_dH(dDHW_.h);
    tilingData_.attrParam.set_dW(dDHW_.w);
    tilingData_.attrParam.set_padD(padDHW_.d);
    tilingData_.attrParam.set_padH(padDHW_.h);
    tilingData_.attrParam.set_padW(padDHW_.w);
    tilingData_.attrParam.set_countIncludePad(countIncludePad_);
    tilingData_.attrParam.set_divisorOverride(divisorOverride_);
    tilingData_.attrParam.set_isOverLap(isOverlap_);
    tilingData_.attrParam.set_isDetermine(isDetermine_);

    tilingData_.ncParam.set_normalCoreNCNum(normalCoreNCNum_);
    tilingData_.ncParam.set_lastCoreNCNum(lastCoreNCNum_);
    tilingData_.ncParam.set_ncAlign(ncAlign_);
    tilingData_.ncParam.set_ncTotal(ncTotal_);
    tilingData_.ncParam.set_ncCount(ncCount_);
    tilingData_.ncParam.set_ncNum(ncNum_);
    tilingData_.ncParam.set_nLine(nLine_);
    tilingData_.ncParam.set_ncTail(ncTail_);

    tilingData_.castCopyParam.set_maxDataNumInUb(maxDataNumInUb_);
    tilingData_.castCopyParam.set_normalCoreNum(normalCoreNum_);
    tilingData_.castCopyParam.set_tailCoreNum(tailCoreNum_);
    tilingData_.castCopyParam.set_normalCoreDataNum(normalCoreDataNum_);
    tilingData_.castCopyParam.set_tailCoreDataNum(tailCoreDataNum_);
    tilingData_.castCopyParam.set_normalCoreFormerCopyTime(normalCoreFormerCopyTime_);
    tilingData_.castCopyParam.set_normalCoreTailCopyTime(normalCoreTailCopyTime_);
    tilingData_.castCopyParam.set_normalCoreFormerDataNum(normalCoreFormerDataNum_);
    tilingData_.castCopyParam.set_normalCoreTailDataNum(normalCoreTailDataNum_);
    tilingData_.castCopyParam.set_tailCoreFormerCopyTime(tailCoreFormerCopyTime_);
    tilingData_.castCopyParam.set_tailCoreTailCopyTime(tailCoreTailCopyTime_);
    tilingData_.castCopyParam.set_tailCoreFormerDataNum(tailCoreFormerDataNum_);
    tilingData_.castCopyParam.set_tailCoreTailDataNum(tailCoreTailDataNum_);

    tilingData_.hwParam.set_normalCoreHWNum(normalCoreHWNum_);
    tilingData_.hwParam.set_lastCoreHWNum(lastCoreHWNum_);
    tilingData_.hwParam.set_hwAlign(hwAlign_);
    tilingData_.hwParam.set_hwTotal(hwTotal_);
    tilingData_.hwParam.set_hwCount(hwCount_);
    tilingData_.hwParam.set_hwNum(hwNum_);
    tilingData_.hwParam.set_nLine(hwLine_);
    tilingData_.hwParam.set_hwTail(hwTail_);

    tilingData_.SaveToBuffer(tilingContext_->GetRawTilingData()->GetData(),
                             tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingFuncForAvgPool3DGrad(gert::TilingContext* context)
{
    AvgPool3dGradTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.SetKernelTiling();
}

IMPL_OP_OPTILING(AvgPool3DGrad)
    .InputsDataDependency({ORIG_INPUT_SHAPE_INDEX})
    .Tiling(TilingFuncForAvgPool3DGrad);
} // namespace optiling

namespace ge {
using gert::InferShapeContext;
using ge::Format;
using ge::FORMAT_NCDHW;
using ge::FORMAT_NDHWC;
using ge::FORMAT_NCHW;
using ge::FORMAT_NHWC;
using ge::FORMAT_HWCN;
using ge::GRAPH_FAILED;
using ge::graphStatus;

constexpr size_t kConv3dDimSizeLimit = 5;

static graphStatus InferShapeForConvBackprop(InferShapeContext *context, size_t const_tensor_idx,
                                             const char *const_tensor_name, size_t dim_num) {
  const auto op_name = context->GetNodeName();
  auto y_shape = context->GetOutputShape(0);

  auto const_tensor = context->GetInputTensor(const_tensor_idx);
  size_t const_tensor_dim_num = static_cast<size_t>(const_tensor->GetOriginShape().GetShapeSize());
  y_shape->SetDimNum(dim_num);

  auto dtype = const_tensor->GetDataType();
  if (dtype == ge::DT_INT32) {
    auto tensor_data = const_tensor->GetData<int32_t>();
    for (size_t idx = 0; idx < const_tensor_dim_num; ++idx) {
      y_shape->SetDim(idx, tensor_data[idx]);
    }
  } else if (dtype == ge::DT_INT64) {
    auto tensor_data = const_tensor->GetData<int64_t>();
    for (size_t idx = 0; idx < const_tensor_dim_num; ++idx) {
      y_shape->SetDim(idx, tensor_data[idx]);
    }
  } else {
    return GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

static graphStatus InferShapeForAvgPool3DGrad(InferShapeContext *context) {
  return InferShapeForConvBackprop(context, 0, "orig_input_shape", kConv3dDimSizeLimit);
}

IMPL_OP_INFERSHAPE(AvgPool3DGrad)
    .InferShape(InferShapeForAvgPool3DGrad)
    .InputsDataDependency({0})
    .PrivateAttr("padding", "");
}  // namespace ge