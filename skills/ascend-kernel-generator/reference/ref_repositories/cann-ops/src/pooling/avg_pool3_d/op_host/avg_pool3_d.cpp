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
 * \file avg_pool3_d.cpp
 * \brief op_host of avg_pool3d
 */
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_info.h"
#include "graph/utils/type_utils.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"

#include "avg_pool3d_tiling.h"
#include "tiling/tiling_api.h"

namespace {
constexpr int32_t SHAPE_SIZE_6D = 6;
constexpr size_t X_INDEX = 0;
constexpr size_t Y_INDEX = 0;
constexpr size_t KSIZE_INDEX = 0;
constexpr size_t STRIDES_INDEX = 1;
constexpr size_t PADS_INDEX = 2;
constexpr size_t CEIL_MODE_INDEX = 3;
constexpr size_t COUNT_INCLUDE_PAD_INDEX = 4;
constexpr size_t DIVISOR_OVERRIDE_INDEX = 5;
constexpr size_t DATA_FORMAT_INDEX = 6;
constexpr size_t X_DIMS = 5;
constexpr size_t Y_DIMS = 5;
constexpr size_t DIM0 = 0;
constexpr size_t DIM1 = 1;
constexpr size_t DIM2 = 2;
constexpr size_t DIM3 = 3;
constexpr size_t DIM4 = 4;
constexpr size_t OUTPUT_SIZE_DIMS = 3;
constexpr size_t COMPATIABLE_PAD_DIM = 6;

constexpr int64_t WORK_SPACE_SIZE = 32;
constexpr uint32_t RESERVED_UB_SIZE = 10U * 1024U;
constexpr uint32_t INDEX_BUF_SIZE = 9U * 1024U;
constexpr uint32_t INDEX_BUF_NUM = 9;

constexpr int32_t FP32_DTYPE_KEY = 0;
constexpr int32_t FP16_DTYPE_KEY = 1;
constexpr int32_t BF16_DTYPE_KEY = 2;

constexpr int32_t MODE_SPLIT_C = 1;
constexpr int32_t MODE_SPLIT_W = 2;
constexpr int32_t MODE_MULTI_W = 3;
constexpr int32_t MODE_REDUCE_D = 4;

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t MAX_TILE_NUM = 4095;

struct TilingParams {
  uint64_t inN = 0;
  uint64_t inC = 0;
  uint64_t tileC = 0;
  uint64_t inD = 0;
  uint64_t inH = 0;
  uint64_t inW = 0;
  uint64_t outD = 0;
  uint64_t outH = 0;
  uint64_t outW = 0;
  uint64_t kD = 0;
  uint64_t kH = 0;
  uint64_t kW = 0;
  uint64_t dD = 0;
  uint64_t dH = 0;
  uint64_t dW = 0;
  uint64_t pD = 0;
  uint64_t pH = 0;
  uint64_t pW = 0;
  int64_t divisorOverride = 0;
  uint64_t countIncludePad = 0;
  uint64_t ceilMode = 0;
  uint64_t formerLength = 0;
  uint64_t formerNum = 0;
  uint64_t tailLength = 0;
  uint64_t tailNum = 0;
  uint64_t indexBufLen = 0;
  uint64_t windowWNum = 0;
  uint64_t tileInput = 0;
  uint64_t tileHW = 0;

  uint32_t ubSize = 0;
  uint32_t coreNum = 0;
  int32_t dataTypeKey = 0;
  std::string dataFormat = "NDHWC";
};

static inline uint64_t FindDivisorWindowNum(uint64_t len, uint64_t initWindowNum) {
  std::vector<uint64_t> divisors;
  uint64_t maxWindowNum = 0;

  for (uint64_t i = 1U; i <= static_cast<uint64_t>(std::sqrt(len)) + 1U; ++i) {
    if (len % i == 0U) {
      divisors.push_back(i);
      if (i != len / i) {
        divisors.push_back(len / i);
      }
    }
  }

  for (auto w: divisors) {
    if (w <= initWindowNum && maxWindowNum < w) {
      maxWindowNum = w;
    }
  }

  return maxWindowNum;
}

} // namespace

namespace optiling {
static void ComputeCoreTilingStrategy(TilingParams& params, int32_t& usedCoreNum) {
  uint64_t outputNum = 0;

  if (params.dataFormat == "NDHWC") {
    outputNum = params.inN * params.outD * params.outH * params.outW;
  } else {
    outputNum = params.inN * params.inC * params.outD;
  }

  if (outputNum < params.coreNum) {
    params.formerNum = outputNum;
    params.tailNum = 0U;
    params.formerLength = 1U;
    params.tailLength = 0U;
    usedCoreNum = outputNum;
  } else if (outputNum % params.coreNum == 0U) {
    params.formerNum = params.coreNum;
    params.tailNum = 0U;
    params.formerLength = outputNum / params.coreNum;
    params.tailLength = 0U;
    usedCoreNum = params.coreNum;
  } else {
    params.formerNum = outputNum % params.coreNum;
    params.tailNum = params.coreNum - params.formerNum;
    params.formerLength = outputNum / params.coreNum + 1U;
    params.tailLength = outputNum / params.coreNum;
    usedCoreNum = params.coreNum;
  }
}

static void ComputeUBTilingStrategy(TilingParams& params, int32_t& mode) {
  int32_t dataTypeSize = params.dataTypeKey == FP32_DTYPE_KEY ? 4 : 2;

  uint64_t alignNum = BLOCK_SIZE * 2U / dataTypeSize;
  uint64_t tileLen = params.ubSize / (dataTypeSize + sizeof(float) * 2U) / alignNum * alignNum;

  if (params.dataFormat == "NCDHW") {
    mode = MODE_REDUCE_D;
    uint64_t alignHW = (params.inH * params.inW + alignNum - 1U) / alignNum * alignNum;
    params.tileHW = alignHW > tileLen ? tileLen : alignHW;
    return;
  }

  uint64_t alignC = (params.inC + alignNum - 1) / alignNum * alignNum;
  
  uint64_t doubleC = 2U * alignC;
  if (doubleC > tileLen) {
    mode = MODE_SPLIT_C;
    params.tileC = alignC > tileLen ? tileLen : alignC;
    return;
  }

  uint64_t tileInput = (params.ubSize / alignC - dataTypeSize - sizeof(float)) / sizeof(float);
  if (tileInput < params.kW) {
    mode = MODE_SPLIT_W;
    params.tileInput = tileInput < MAX_TILE_NUM ? tileInput : MAX_TILE_NUM;
    return;
  }

  if (params.dW > params.kW) {
    mode = MODE_SPLIT_W;
    params.tileInput = params.kW;
    return;
  }

  uint64_t windowWNum = 
    (params.ubSize / alignC - params.kW * sizeof(float)) / ((params.dW + 1) * sizeof(float) + dataTypeSize);

  mode = MODE_MULTI_W;
  windowWNum = windowWNum * params.kW <= MAX_TILE_NUM ? windowWNum : MAX_TILE_NUM / params.kW;
  windowWNum = windowWNum < params.outW ? windowWNum : params.outW;
  params.windowWNum = FindDivisorWindowNum(params.indexBufLen, windowWNum);

  if (windowWNum == 0) {
    mode = MODE_SPLIT_C;
    params.tileC = alignC > tileLen ? tileLen : alignC;
  }
}

static void SetTiling(const TilingParams& params, AvgPool3DTilingData& tiling) {
  tiling.set_inN(params.inN);
  tiling.set_inC(params.inC);
  tiling.set_tileC(params.tileC);
  tiling.set_inD(params.inD);
  tiling.set_inH(params.inH);
  tiling.set_inW(params.inW);
  tiling.set_outD(params.outD);
  tiling.set_outH(params.outH);
  tiling.set_outW(params.outW);
  tiling.set_kD(params.kD);
  tiling.set_kH(params.kH);
  tiling.set_kW(params.kW);
  tiling.set_dD(params.dD);
  tiling.set_dH(params.dH);
  tiling.set_dW(params.dW);
  tiling.set_pD(params.pD);
  tiling.set_pH(params.pH);
  tiling.set_pW(params.pW);
  tiling.set_divisorOverride(params.divisorOverride);
  tiling.set_countIncludePad(params.countIncludePad);
  tiling.set_ceilMode(params.ceilMode);
  tiling.set_formerLength(params.formerLength);
  tiling.set_formerNum(params.formerNum);
  tiling.set_tailLength(params.tailLength);
  tiling.set_tailNum(params.tailNum);
  tiling.set_indexBufLen(params.indexBufLen);
  tiling.set_windowWNum(params.windowWNum);
  tiling.set_tileInput(params.tileInput);
  tiling.set_tileHW(params.tileHW);
}

static bool GetDataTypeKey(ge::DataType dataType, int32_t& dataTypeKey) {
  switch (dataType) {
    case ge::DT_FLOAT16:
        dataTypeKey = FP16_DTYPE_KEY;
        break;
    case ge::DT_BF16:
        dataTypeKey = BF16_DTYPE_KEY;
        break;
    case ge::DT_FLOAT:
        dataTypeKey = FP32_DTYPE_KEY;
        break;
    default:
        return false;
  }

  return true;
}

static ge::graphStatus KernelTiling(gert::TilingContext* context, TilingParams& params) {
  auto nodeName = context->GetNodeName();

  int32_t usedCoreNum = 0;
  ComputeCoreTilingStrategy(params, usedCoreNum);

  int32_t modeKey = MODE_SPLIT_C;
  ComputeUBTilingStrategy(params, modeKey);

  AvgPool3DTilingData tiling;
  SetTiling(params, tiling);

  uint32_t tilingKey = modeKey * 10U + params.dataTypeKey;
  context->SetTilingKey(tilingKey);
  context->SetBlockDim(usedCoreNum);

  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  size_t* currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = WORK_SPACE_SIZE;

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFuncForAvgPool3d(gert::TilingContext* context)
{
  auto nodeName = context->GetNodeName();

  const gert::Shape xShape = context->GetInputShape(X_INDEX)->GetStorageShape();

  auto dataType = context->GetInputDesc(X_INDEX)->GetDataType();
  int32_t dataTypeKey = FP32_DTYPE_KEY;
  GetDataTypeKey(dataType, dataTypeKey);

  const gert::Shape yShape = context->GetOutputShape(Y_INDEX)->GetStorageShape();
  
  auto attrPtr = context->GetAttrs();
  auto ksizePtr = attrPtr->GetAttrPointer<gert::ContinuousVector>(KSIZE_INDEX);
  auto ksize = reinterpret_cast<const int64_t*>(ksizePtr->GetData());

  auto stridesPtr = attrPtr->GetAttrPointer<gert::ContinuousVector>(STRIDES_INDEX);
  auto strides = reinterpret_cast<const int64_t*>(stridesPtr->GetData());

  auto padsPtr = attrPtr->GetAttrPointer<gert::ContinuousVector>(PADS_INDEX);
  auto pads = reinterpret_cast<const int64_t*>(padsPtr->GetData());
  
  const bool* ceilMode = attrPtr->GetAttrPointer<bool>(CEIL_MODE_INDEX);
  const bool* countIncludePad = attrPtr->GetAttrPointer<bool>(COUNT_INCLUDE_PAD_INDEX);
  const int64_t* divisorOverride = attrPtr->GetAttrPointer<int64_t>(DIVISOR_OVERRIDE_INDEX);

  const std::string dataFormat = static_cast<std::string>(attrPtr->GetStr(DATA_FORMAT_INDEX));

  TilingParams params;
  params.dataFormat = dataFormat;

  if (dataFormat == "NCDHW") {
    params.inN = xShape.GetDim(DIM0);
    params.inC = xShape.GetDim(DIM1);
    params.inD = xShape.GetDim(DIM2);
    params.inH = xShape.GetDim(DIM3);
    params.inW = xShape.GetDim(DIM4);

    params.outD = yShape.GetDim(DIM2);
    params.outH = yShape.GetDim(DIM3);
    params.outW = yShape.GetDim(DIM4);
  } else {
    params.inN = xShape.GetDim(DIM0);
    params.inC = xShape.GetDim(DIM4);
    params.inD = xShape.GetDim(DIM1);
    params.inH = xShape.GetDim(DIM2);
    params.inW = xShape.GetDim(DIM3);

    params.outD = yShape.GetDim(DIM1);
    params.outH = yShape.GetDim(DIM2);
    params.outW = yShape.GetDim(DIM3);
  }
  params.kD = ksize[DIM0];
  params.kH = ksizePtr->GetSize() == 1 ? ksize[DIM0] : ksize[DIM1];
  params.kW = ksizePtr->GetSize() == 1 ? ksize[DIM0] : ksize[DIM2];
  params.dD = strides[DIM0];
  params.dH = stridesPtr->GetSize() == 1 ? strides[DIM0] : strides[DIM1];
  params.dW = stridesPtr->GetSize() == 1 ? strides[DIM0] : strides[DIM2];

  if (padsPtr->GetSize() != COMPATIABLE_PAD_DIM) {
    params.pD = pads[DIM0];
    params.pH = padsPtr->GetSize() == 1 ? pads[DIM0] : pads[DIM1];
    params.pW = padsPtr->GetSize() == 1 ? pads[DIM0] : pads[DIM2];
  } else {
    params.pD = pads[DIM0];
    params.pH = pads[DIM2];
    params.pW = pads[DIM4];
  }
  
  params.countIncludePad = countIncludePad == nullptr ? 0 : static_cast<uint64_t>(*countIncludePad);
  params.divisorOverride = divisorOverride == nullptr ? 0 : static_cast<int64_t>(*divisorOverride);
  params.ceilMode = ceilMode == nullptr ? 0 : static_cast<uint64_t>(*ceilMode);
  
  params.indexBufLen = INDEX_BUF_SIZE / INDEX_BUF_NUM / sizeof(int64_t);
  
  auto platformInfo = context->GetPlatformInfo();
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
  params.coreNum = ascendcPlatform.GetCoreNumAiv();

  uint64_t ubSizePlatForm;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
  params.ubSize = static_cast<int64_t>(ubSizePlatForm) - RESERVED_UB_SIZE - INDEX_BUF_SIZE;
  params.dataTypeKey = dataTypeKey;

  ge::graphStatus tilingStatus = KernelTiling(context, params);

  return tilingStatus;
}
IMPL_OP_OPTILING(AvgPool3D)
    .Tiling(TilingFuncForAvgPool3d);

} // namespace optiling

namespace ge {
using gert::InferShapeContext;
using ge::Format;
using ge::FORMAT_NCHW;
using ge::FORMAT_NHWC;
using ge::FORMAT_NCDHW;
using ge::FORMAT_NDHWC;
using ge::FORMAT_DHWCN;

constexpr size_t kConv3dInputSizeLimit = 5;

// AvgPool3D
constexpr size_t kDHWSizeLimit = 3;
constexpr size_t kAvgPool3DKSizeIdx = 0;
constexpr size_t kAvgPool3DStridesIdx = 1;
constexpr size_t kAvgPool3DPadsIdx = 2;
constexpr size_t kAvgPool3DCeilModeIdx = 3;
constexpr size_t kAvgPool3DPaddingIdx = 7;

// NDHWC
constexpr size_t kDDimNDHWCIdx = 1;
constexpr size_t kHDimNDHWCIdx = 2;
constexpr size_t kWDimNDHWCIdx = 3;
// NCDHW
constexpr size_t kDDimNCDHWIdx = 2;
constexpr size_t kHDimNCDHWIdx = 3;
constexpr size_t kWDimNCDHWIdx = 4;

// DHWCN
constexpr size_t kDDimDHWCNIdx = 0;
constexpr size_t kHDimDHWCNIdx = 1;
constexpr size_t kWDimDHWCNIdx = 2;

struct Conv3DInputShapes {
  int64_t in = 0;
  int64_t id = 0;
  int64_t ih = 0;
  int64_t iw = 0;
  int64_t ic = 0;

  int64_t kn = 0;
  int64_t kd = 0;
  int64_t kh = 0;
  int64_t kw = 0;
  int64_t kc = 0;
};

struct Conv3DAttrs {
  bool ceil_mode = false;

  int64_t strd = 0;
  int64_t strh = 0;
  int64_t strw = 0;

  int64_t dild = 1;
  int64_t dilh = 1;
  int64_t dilw = 1;

  int64_t groups = 1;

  int64_t padf = 0;
  int64_t padb = 0;
  int64_t padu = 0;
  int64_t padd = 0;
  int64_t padl = 0;
  int64_t padr = 0;
};

bool GetConv3DXShape(InferShapeContext *context, size_t x_idx, Format x_format, bool avg_pool3d,
                     Conv3DInputShapes &shapes) {
  const auto x_shape = context->GetInputShape(x_idx);

  size_t idx = 0;
  if (x_format == FORMAT_NCDHW) {
    shapes.in = x_shape->GetDim(idx++);
    shapes.ic = x_shape->GetDim(idx++);
    shapes.id = x_shape->GetDim(idx++);
    shapes.ih = x_shape->GetDim(idx++);
    shapes.iw = x_shape->GetDim(idx++);
  } else if (x_format == FORMAT_NDHWC) {
    shapes.in = x_shape->GetDim(idx++);
    shapes.id = x_shape->GetDim(idx++);
    shapes.ih = x_shape->GetDim(idx++);
    shapes.iw = x_shape->GetDim(idx++);
    shapes.ic = x_shape->GetDim(idx++);
  } else if (avg_pool3d && x_format == FORMAT_DHWCN) {
    shapes.id = x_shape->GetDim(idx++);
    shapes.ih = x_shape->GetDim(idx++);
    shapes.iw = x_shape->GetDim(idx++);
    shapes.ic = x_shape->GetDim(idx++);
    shapes.in = x_shape->GetDim(idx++);
  } else {
    return false;
  }

  return true;
}

bool GetConv3DPads(const InferShapeContext *context, const Conv3DInputShapes &shapes, size_t pads_idx,
                   size_t padding_idx, Conv3DAttrs &attrs) {
  const auto runtime_attrs = context->GetAttrs();
  const auto pads_list = runtime_attrs->GetAttrPointer<gert::ContinuousVector>(pads_idx);
  const auto pads_list_data = reinterpret_cast<const int64_t *>(pads_list->GetData());

  size_t idx = 0;
  attrs.padf = pads_list_data[idx++];
  attrs.padb = pads_list_data[idx++];
  attrs.padu = pads_list_data[idx++];
  attrs.padd = pads_list_data[idx++];
  attrs.padl = pads_list_data[idx++];
  attrs.padr = pads_list_data[idx++];

  if (runtime_attrs->GetAttrNum() > padding_idx) {
    const auto padding = runtime_attrs->GetAttrPointer<char>(padding_idx);
    if (padding != nullptr && (strcmp(padding, "SAME") == 0)) {
      int64_t tails_d = shapes.id % attrs.strd;  // non zero, checked in shape range infer logic
      int64_t tails_h = shapes.ih % attrs.strh;  // non zero, checked in shape range infer logic
      int64_t tails_w = shapes.iw % attrs.strw;  // non zero, checked in shape range infer logic
      int64_t dilate_kernel_d = attrs.dild * (shapes.kd - 1) + 1;
      int64_t dilate_kernel_h = attrs.dilh * (shapes.kh - 1) + 1;
      int64_t dilate_kernel_w = attrs.dilw * (shapes.kw - 1) + 1;
      int64_t pad_d = tails_d > 0 ? dilate_kernel_d - tails_d : dilate_kernel_d - attrs.strd;
      int64_t pad_h = tails_h > 0 ? dilate_kernel_h - tails_h : dilate_kernel_h - attrs.strh;
      int64_t pad_w = tails_w > 0 ? dilate_kernel_w - tails_w : dilate_kernel_w - attrs.strw;
      pad_d = std::max(pad_d, 0L);
      pad_h = std::max(pad_h, 0L);
      pad_w = std::max(pad_w, 0L);
      attrs.padf = pad_d / 2;  // 2 means pad_up is half size of pad_d
      attrs.padb = pad_d - attrs.padf;
      attrs.padu = pad_h / 2;  // 2 means pad_up is half size of pad_h
      attrs.padd = pad_h - attrs.padu;
      attrs.padl = pad_w / 2;  // 2 means pad_up is half size of pad_w
      attrs.padr = pad_w - attrs.padl;
      return true;
    }
  }

  bool negative_pad =
      std::any_of(pads_list_data, pads_list_data + pads_list->GetSize(), [](int64_t val) -> bool { return val < 0; });
  return true;
}

static bool GetStridesForAvgPool3D(const InferShapeContext *context, ge::Format x_format, Conv3DAttrs &attrs) {
  const auto runtime_attrs = context->GetAttrs();

  const auto strides_list = runtime_attrs->GetAttrPointer<gert::ContinuousVector>(kAvgPool3DStridesIdx);
  const auto strides_list_data = reinterpret_cast<const int64_t *>(strides_list->GetData());

  if (strides_list->GetSize() == 1) {
    attrs.strd = strides_list_data[0];
    attrs.strh = strides_list_data[0];
    attrs.strw = strides_list_data[0];
  } else if (strides_list->GetSize() == kDHWSizeLimit) {
    size_t idx = 0;
    attrs.strd = strides_list_data[idx++];
    attrs.strh = strides_list_data[idx++];
    attrs.strw = strides_list_data[idx++];
  } else if (strides_list->GetSize() == kConv3dInputSizeLimit) {
    if (x_format == ge::FORMAT_NCDHW) {
      attrs.strd = strides_list_data[kDDimNCDHWIdx];
      attrs.strh = strides_list_data[kHDimNCDHWIdx];
      attrs.strw = strides_list_data[kWDimNCDHWIdx];
    } else if (x_format == ge::FORMAT_NDHWC) {
      attrs.strd = strides_list_data[kDDimNDHWCIdx];
      attrs.strh = strides_list_data[kHDimNDHWCIdx];
      attrs.strw = strides_list_data[kWDimNDHWCIdx];
    } else {
      // DHWCN
      attrs.strd = strides_list_data[kDDimDHWCNIdx];
      attrs.strh = strides_list_data[kHDimDHWCNIdx];
      attrs.strw = strides_list_data[kWDimDHWCNIdx];
    }
  }

  return true;
}

static bool GetKSize(const InferShapeContext *context, ge::Format x_format, Conv3DInputShapes &shapes) {
  const auto runtime_attrs = context->GetAttrs();

  const auto ksize_list = runtime_attrs->GetAttrPointer<gert::ContinuousVector>(kAvgPool3DKSizeIdx);
  const auto ksize_list_data = reinterpret_cast<const int64_t *>(ksize_list->GetData());

  if (ksize_list->GetSize() == 1) {
    shapes.kd = ksize_list_data[0];
    shapes.kh = ksize_list_data[0];
    shapes.kw = ksize_list_data[0];
  } else if (ksize_list->GetSize() == kDHWSizeLimit) {
    size_t idx = 0;
    shapes.kd = ksize_list_data[idx++];
    shapes.kh = ksize_list_data[idx++];
    shapes.kw = ksize_list_data[idx++];
  } else if (ksize_list->GetSize() == kConv3dInputSizeLimit) {
    if (x_format == ge::FORMAT_NCDHW) {
      shapes.kd = ksize_list_data[kDDimNCDHWIdx];
      shapes.kh = ksize_list_data[kHDimNCDHWIdx];
      shapes.kw = ksize_list_data[kWDimNCDHWIdx];
    } else if (x_format == ge::FORMAT_NDHWC) {
      shapes.kd = ksize_list_data[kDDimNDHWCIdx];
      shapes.kh = ksize_list_data[kHDimNDHWCIdx];
      shapes.kw = ksize_list_data[kWDimNDHWCIdx];
    } else {
      // DHWCN
      shapes.kd = ksize_list_data[kDDimDHWCNIdx];
      shapes.kh = ksize_list_data[kHDimDHWCNIdx];
      shapes.kw = ksize_list_data[kWDimDHWCNIdx];
    }
  }

  return true;
}
    
static bool CalcAvgPool3DOutputShape(const char *op_name, const Conv3DInputShapes &shapes, const Conv3DAttrs &attrs,
                                     ge::Format y_format, gert::Shape *y_shape) {
  int64_t outd = 0;
  int64_t outh = 0;
  int64_t outw = 0;
  if (attrs.ceil_mode) {
    outd = (shapes.id + attrs.padf + attrs.padb - shapes.kd + attrs.strd - 1) / attrs.strd + 1;
    outh = (shapes.ih + attrs.padu + attrs.padd - shapes.kh + attrs.strh - 1) / attrs.strh + 1;
    outw = (shapes.iw + attrs.padl + attrs.padr - shapes.kw + attrs.strw - 1) / attrs.strw + 1;
    if ((outd - 1) * attrs.strd >= shapes.id + attrs.padf) {
      outd--;
    }
    if ((outh - 1) * attrs.strh >= shapes.ih + attrs.padu) {
      outh--;
    }
    if ((outw - 1) * attrs.strw >= shapes.iw + attrs.padl) {
      outw--;
    }
  } else {
    outd = (shapes.id + attrs.padf + attrs.padb - shapes.kd) / attrs.strd + 1;
    outh = (shapes.ih + attrs.padu + attrs.padd - shapes.kh) / attrs.strh + 1;
    outw = (shapes.iw + attrs.padl + attrs.padr - shapes.kw) / attrs.strw + 1;
  }

  y_shape->SetDimNum(0);
  if (y_format == ge::FORMAT_NCDHW) {
    y_shape->AppendDim(shapes.in);
    y_shape->AppendDim(shapes.ic);
    y_shape->AppendDim(outd);
    y_shape->AppendDim(outh);
    y_shape->AppendDim(outw);
  } else if (y_format == ge::FORMAT_NDHWC) {
    y_shape->AppendDim(shapes.in);
    y_shape->AppendDim(outd);
    y_shape->AppendDim(outh);
    y_shape->AppendDim(outw);
    y_shape->AppendDim(shapes.ic);
  } else if (y_format == ge::FORMAT_DHWCN) {
    y_shape->AppendDim(outd);
    y_shape->AppendDim(outh);
    y_shape->AppendDim(outw);
    y_shape->AppendDim(shapes.ic);
    y_shape->AppendDim(shapes.in);
  } else {
    return false;
  }

  return true;
}

static ge::graphStatus InferShapeForAvgPool3D(InferShapeContext *context) {
  const auto op_name = context->GetNodeName() == nullptr ? "AvgPool3D" : context->GetNodeName();

  const auto runtime_attrs = context->GetAttrs();
  const auto ceil_mode = runtime_attrs->GetAttrPointer<bool>(kAvgPool3DCeilModeIdx);

  const auto x_desc = context->GetInputDesc(0);
  const auto x_format = x_desc->GetOriginFormat();

  const auto y_shape = context->GetOutputShape(0);

  const auto y_desc = context->GetOutputDesc(0);
  const auto y_format = y_desc->GetOriginFormat();

  Conv3DInputShapes shapes;
  Conv3DAttrs attrs;
  attrs.ceil_mode = *ceil_mode;
  if (GetConv3DXShape(context, 0UL, x_format, true, shapes) && GetKSize(context, x_format, shapes) &&
      GetStridesForAvgPool3D(context, x_format, attrs) &&
      GetConv3DPads(context, shapes, kAvgPool3DPadsIdx, kAvgPool3DPaddingIdx, attrs) &&
      CalcAvgPool3DOutputShape(op_name, shapes, attrs, y_format, y_shape)) {
    return ge::GRAPH_SUCCESS;
  }

  return ge::GRAPH_FAILED;
}
IMPL_OP_INFERSHAPE(AvgPool3D)
    .InferShape(InferShapeForAvgPool3D)
    .PrivateAttr("padding", "");
}  // namespace ge