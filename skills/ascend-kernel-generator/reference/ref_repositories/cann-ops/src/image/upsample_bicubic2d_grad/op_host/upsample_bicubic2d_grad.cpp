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
 * \file upsample_bicubic2d_grad.cpp
 * \brief
 */

#include <cmath>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "upsample_bicubic2d_grad_tiling.h"

#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
#define OP_LOGI(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)

namespace optiling {
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)
#define OPS_CHECK_NULL_WITH_CONTEXT_RET(context, ptr, ret)                                        \
  if ((ptr) == nullptr) {                                                                         \
    const char* name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName();  \
    std::printf(name, "is nullptr!");                                                             \
    REPORT_INNER_ERR_MSG("EZ9999", "op[%s], %s is nullptr!", name, #ptr);                            \
    return ret;                                                                                   \
  }

bool AddWorkspace(gert::TilingContext* context, const size_t workspace) {
    size_t* workspace_size = context->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, workspace_size, false);
    *workspace_size = workspace;
    return true;
}

}  // namespace optiling

namespace optiling {

inline bool FloatEqual(float a, float b) {
  float closeTo0 = float(1e-6);
  if (a > b){
    return a - b < closeTo0;
  } else {
    return b - a < closeTo0;
  }
}

bool UpsampleBicubic2dGradTiling::GetPlatformInfo(const gert::TilingContext * context) {
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  _Params.CoreNum = ascendcPlatform.GetCoreNumAiv();
  return true;
}

bool UpsampleBicubic2dGradTiling::GetCheckAttr(const gert::TilingContext * context) {
  auto attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, attrs, false);

  const bool *align_corners = attrs->GetAttrPointer<bool>(0);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, align_corners, false);
  const float *scales_h = attrs->GetAttrPointer<float>(1);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, scales_h, false);
  const float *scales_w = attrs->GetAttrPointer<float>(2);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, scales_w, false);

  _Params.alignCorners = *align_corners;
  _Params.scalesH = *scales_h;
  _Params.scalesW = *scales_w;
  return true;
}

bool UpsampleBicubic2dGradTiling::CheckInOutShapes(const gert::TilingContext * context) {
  // input
  auto input_tensor = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, input_tensor, false);
  auto input_shape = input_tensor->GetStorageShape();
  OP_TILING_CHECK(input_shape.GetDimNum() != 4,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                  "UpsampleBicubic2dGrad get input shape dim is %lu not 4[NCHW], please check.", input_shape.GetDimNum()),
                  return false);
  _Params.batch = input_shape.GetDim(0) * input_shape.GetDim(1);
  _Params.inputN = input_shape.GetDim(0);
  _Params.inputC = input_shape.GetDim(1);
  _Params.inputH = input_shape.GetDim(NUM_TWO);
  _Params.inputW = input_shape.GetDim(NUM_THREE);

  auto output_tensor = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, output_tensor, false);
  auto output_shape = output_tensor->GetStorageShape();
  OP_TILING_CHECK(output_shape.GetDimNum() != 4,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                  "UpsampleBicubic2dGrad get output shape dim is %lu not 4[NCHW], please check.", output_shape.GetDimNum()),
                  return false);
  OP_TILING_CHECK(output_shape.GetDim(0) != input_shape.GetDim(0),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                  "UpsampleBicubic2dGrad get output shape dim[0] %ld not match input shape dim[0] %ld, please check.",
                  output_shape.GetDim(0), input_shape.GetDim(0)),
                  return false);
  OP_TILING_CHECK(output_shape.GetDim(1) != input_shape.GetDim(1),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                  "UpsampleBicubic2dGrad get output shape dim[1] %ld not match input shape dim[1] %ld, please check.",
                  output_shape.GetDim(1), input_shape.GetDim(1)),
                  return false);
  _Params.outputH = output_shape.GetDim(NUM_TWO);
  _Params.outputW = output_shape.GetDim(NUM_THREE);

  return true;
}

void UpsampleBicubic2dGradTiling::InitPlatformInfo(const gert::TilingContext * context, matmul_tiling::PlatformInfo& platformInfo) const {
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

  platformInfo.socVersion = ascendcPlatform.GetSocVersion();
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, platformInfo.ubSize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, platformInfo.l1Size);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, platformInfo.l0ASize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, platformInfo.l0BSize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, platformInfo.l0CSize);
}

bool UpsampleBicubic2dGradTiling::GetMMTilingData(const gert::TilingContext * context) {
  auto dataType = static_cast<matmul_tiling::DataType>(_Params.dataType);
  matmul_tiling::PlatformInfo platformInfo;
  InitPlatformInfo(context, platformInfo);
  matmul_tiling::MatmulApiTiling matmul_h(platformInfo);
  auto ret = matmul_h.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
  OP_TILING_CHECK(ret == -1,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "matmul_h SetAType fail."),
                  return false);
  ret = matmul_h.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
  OP_TILING_CHECK(ret == -1,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "matmul_h SetBType fail."),
                  return false);
  ret = matmul_h.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
  OP_TILING_CHECK(ret == -1,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "matmul_h SetCType fail."),
                  return false);
  ret = matmul_h.SetOrgShape(_Params.baseNH, _Params.outputW, NUM_FRACTAL, _Params.inputH);
  OP_TILING_CHECK(ret == -1,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "matmul_h SetOrgShape fail."),
                  return false);
  ret = matmul_h.SetShape(_Params.baseNH, _Params.outputW, NUM_FRACTAL);
  OP_TILING_CHECK(ret == -1,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "matmul_h Set single shape fail."),
                  return false);
  ret = matmul_h.SetBufferSpace(-1, -1, -1);
  OP_TILING_CHECK(ret == -1,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "matmul_h SetBufferSpace fail."),
                  return false);
  ret = matmul_h.GetTiling(tilingData.MMParamH);
  OP_TILING_CHECK(ret == -1,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "matmul_h GetTiling fail."),
                  return false);
  matmul_tiling::MatmulApiTiling matmul_w(platformInfo);
  ret = matmul_w.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
  OP_TILING_CHECK(ret == -1,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "matmul_w SetAType fail."),
                  return false);
  ret = matmul_w.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
  OP_TILING_CHECK(ret == -1,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "matmul_w SetBType fail."),
                  return false);
  ret = matmul_w.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
  OP_TILING_CHECK(ret == -1,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "matmul_w SetCType fail."),
                  return false);
  int innerBatchW_ = _Params.innerBatchW;
  if(_Params.innerBatchW == 0) {
    innerBatchW_ = 1;
  }
  ret = matmul_w.SetOrgShape(innerBatchW_ * _Params.inputH, _Params.baseNW, _Params.inputW, NUM_FRACTAL);
  OP_TILING_CHECK(ret == -1,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "matmul_w SetOrgShape fail."),
                  return false);
  ret = matmul_w.SetShape(innerBatchW_ * _Params.inputH, _Params.baseNW, NUM_FRACTAL);
  OP_TILING_CHECK(ret == -1,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "matmul_w Set single shape fail."),
                  return false);
  ret = matmul_w.SetBufferSpace(-1, -1, -1);
  OP_TILING_CHECK(ret == -1,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "matmul_w SetBufferSpace fail."),
                  return false);
  ret = matmul_w.GetTiling(tilingData.MMParamW);
  OP_TILING_CHECK(ret == -1,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "matmul_w GetTiling fail."),
                  return false);
  return true;
}
uint32_t UpsampleBicubic2dGradTiling::GetNumPerBlock() {
  if(_Params.dataType == ge::DataType::DT_FLOAT){
    return NUM_PER_BLOCK_FLOAT32;
  }
  return NUM_PER_BLOCK_FLOAT16;
}

uint32_t UpsampleBicubic2dGradTiling::GetDtypeSize() {
  if(_Params.dataType == ge::DataType::DT_FLOAT){
    return sizeof(float);
  }
  return sizeof(short);
}

bool UpsampleBicubic2dGradTiling::GetClearTilingData() {
  _Params.clearBaseN = UB_CLEAR_SIZE / GetDtypeSize();
  uint64_t total_count = _Params.batch * _Params.inputH * _Params.outputW;
  uint64_t step_count = _Params.CoreNum * _Params.clearBaseN;

  _Params.clearInterLoop = total_count / step_count;
  uint64_t total_count_tail = total_count % step_count;
  uint64_t total_block_tail = (total_count_tail + GetNumPerBlock() - 1) / GetNumPerBlock();
  _Params.clearInterTailN = total_block_tail / _Params.CoreNum * GetNumPerBlock();
  _Params.clearInterTailCoreNum = total_block_tail % _Params.CoreNum;

  total_count = _Params.batch * _Params.outputH * _Params.outputW;

  _Params.clearOutLoop = total_count / step_count;
  total_count_tail = total_count % step_count;
  total_block_tail = (total_count_tail + GetNumPerBlock() - 1) / GetNumPerBlock();
  _Params.clearOutTailN = total_block_tail / _Params.CoreNum * GetNumPerBlock();
  _Params.clearOutTailCoreNum = total_block_tail % _Params.CoreNum;
  return true;
}

bool UpsampleBicubic2dGradTiling::GetTilingData(const gert::TilingContext * context) {
  uint64_t loop_H = (_Params.inputH + NUM_FRACTAL -  1) / NUM_FRACTAL;
  _Params.tailH = ((_Params.inputH - 1) % NUM_FRACTAL) + 1;
  _Params.innerCoreNumH = _Params.CoreNum / loop_H;
  _Params.innerCoreNumH = _Params.innerCoreNumH == 0 ? 1 : _Params.innerCoreNumH;
  _Params.CoreNumH = _Params.CoreNum / _Params.innerCoreNumH;
  _Params.loopH = loop_H / _Params.CoreNumH;
  _Params.loopTailCoreH = loop_H % _Params.CoreNumH;
  _Params.innerBatchH =  _Params.batch / _Params.innerCoreNumH;
  _Params.innerBatchTailCoreH =  _Params.batch % _Params.innerCoreNumH;

  uint64_t loop_W = (_Params.inputW + NUM_FRACTAL -  1) / NUM_FRACTAL;
  _Params.tailW = ((_Params.inputW - 1) % NUM_FRACTAL) + 1;
  _Params.innerCoreNumW = _Params.CoreNum / loop_W;
  _Params.innerCoreNumW = _Params.innerCoreNumW == 0 ? 1 : _Params.innerCoreNumW;
  _Params.CoreNumW = _Params.CoreNum / _Params.innerCoreNumW;
  _Params.loopW = loop_W / _Params.CoreNumW;
  _Params.loopTailCoreW = loop_W % _Params.CoreNumW;
  _Params.innerBatchW =  _Params.batch / _Params.innerCoreNumW;
  _Params.innerBatchTailCoreW =  _Params.batch % _Params.innerCoreNumW;

  _Params.baseNH = NUM_FRACTAL * static_cast<uint64_t>(ceil(_Params.scalesH + THRESHOLD));
  _Params.baseNW = NUM_FRACTAL * static_cast<uint64_t>(ceil(_Params.scalesW + THRESHOLD));
  GetClearTilingData();
  return GetMMTilingData(context);
}

bool UpsampleBicubic2dGradTiling::SetTilingData(gert::TilingContext* context) {
  tilingData.set_dataType(static_cast<uint32_t>(_Params.dataType));
  tilingData.set_CoreNum(_Params.CoreNum);
  tilingData.set_alignCorners(_Params.alignCorners);
  tilingData.set_scalesH(_Params.scalesH);
  tilingData.set_scalesW(_Params.scalesW);
  tilingData.set_baseNH(_Params.baseNH);
  tilingData.set_baseNW(_Params.baseNW);

  tilingData.set_batch(_Params.batch);
  tilingData.set_inputH(_Params.inputH);
  tilingData.set_inputW(_Params.inputW);
  tilingData.set_outputH(_Params.outputH);
  tilingData.set_outputW(_Params.outputW);

  tilingData.set_tailH(_Params.tailH);
  tilingData.set_CoreNumH(_Params.CoreNumH);
  tilingData.set_loopH(_Params.loopH);
  tilingData.set_loopTailCoreH(_Params.loopTailCoreH);
  tilingData.set_innerCoreNumH(_Params.innerCoreNumH);
  tilingData.set_innerBatchH(_Params.innerBatchH);
  tilingData.set_innerBatchTailCoreH(_Params.innerBatchTailCoreH);

  tilingData.set_tailW(_Params.tailW);
  tilingData.set_CoreNumW(_Params.CoreNumW);
  tilingData.set_loopW(_Params.loopW);
  tilingData.set_loopTailCoreW(_Params.loopTailCoreW);
  tilingData.set_innerCoreNumW(_Params.innerCoreNumW);
  tilingData.set_innerBatchW(_Params.innerBatchW);
  tilingData.set_innerBatchTailCoreW(_Params.innerBatchTailCoreW);

  tilingData.set_clearBaseN(_Params.clearBaseN);
  tilingData.set_clearInterLoop(_Params.clearInterLoop);
  tilingData.set_clearInterTailN(_Params.clearInterTailN);
  tilingData.set_clearInterTailCoreNum(_Params.clearInterTailCoreNum);
  tilingData.set_clearOutLoop(_Params.clearOutLoop);
  tilingData.set_clearOutTailN(_Params.clearOutTailN);
  tilingData.set_clearOutTailCoreNum(_Params.clearOutTailCoreNum);

  tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
  return true;
}

bool UpsampleBicubic2dGradTiling::SetLaunchInfo(gert::TilingContext* context) {
  context->SetBlockDim(_Params.CoreNum / NUM_TWO);
  context->SetTilingKey(static_cast<int64_t>(UpsampleBicubic2dGradTilingKey::BASE_MODE));

  int64_t workspaceSize = ((_Params.baseNH > _Params.baseNW) ? _Params.baseNH : _Params.baseNW)
                        * NUM_FRACTAL * _Params.CoreNum * GetDtypeSize()
                        + (_Params.batch * _Params.inputH * _Params.outputW + GetNumPerBlock() - 1) / GetNumPerBlock() * BLOCK_SIZE
                        + _Params.CoreNum * BLOCK_SIZE * NUM_TWO
                        + 16 * 1024 * 1024;
  AddWorkspace(context, workspaceSize);
  return true;
}

bool UpsampleBicubic2dGradTiling::GetTilingDataDC(const gert::TilingContext * context) {
  _Params.slideSize = NUM_FRACTAL;
  _Params.CoreNumW = 0;
  _Params.CoreNumH = 0;
  _Params.CoreNum = _Params.CoreNum / NUM_TWO;
  CalcScales();
  CalcNeedCoreNum();
  CalcSingleCoreK();
  CalcTCubeTiling(context);

  return true;
}

void UpsampleBicubic2dGradTiling::CalcScales() {
  _Params.needExpandW = _Params.inputW != _Params.outputW;
  _Params.needExpandH = _Params.inputH != _Params.outputH;
}

void UpsampleBicubic2dGradTiling::CalcNeedCoreNum() {
  if (_Params.needExpandW) {
    CalcNeedCoreNumW();
  }

  if (_Params.needExpandH || !_Params.needExpandW) {
    CalcNeedCoreNumH();
  }
  _Params.CoreNum = (_Params.CoreNumW > _Params.CoreNumH) ?_Params.CoreNumW:_Params.CoreNumH;
  _Params.CoreNum = _Params.CoreNum > 1 ?_Params.CoreNum:1;
}

void UpsampleBicubic2dGradTiling::CalcNeedCoreNumW() {
  // 当前须切分的元素总数
  uint64_t eleW = _Params.outputW;
  uint64_t eleH = _Params.inputH * _Params.batch;
  // 切分每块的元素
  uint64_t slideSize = _Params.slideSize;
  uint64_t perCoreSlideNum = eleW / (_Params.CoreNum * slideSize);

  uint64_t tailW = eleW - perCoreSlideNum * slideSize * _Params.CoreNum;
  uint64_t tailAvergingRow = eleH /_Params.CoreNum;
  uint64_t tailExtraRow = eleH % _Params.CoreNum;

  uint64_t needCoreNum = (tailAvergingRow == 0)? tailExtraRow:_Params.CoreNum;

  uint64_t perCoreSlideEle = perCoreSlideNum * slideSize;
  uint64_t curRowStartIdx = 0;
  for (uint64_t coreIdx = 0; coreIdx < _Params.CoreNum; coreIdx++) {
    _Params.slideStartListW[coreIdx] = coreIdx * perCoreSlideEle;
    // 计算当前方向切块的终点
    _Params.slideEndListW[coreIdx] = _Params.slideStartListW[coreIdx] + perCoreSlideEle;
    if(tailW > 0) {
      if (coreIdx < tailExtraRow) {
        _Params.tailSlideStartListW[coreIdx] = curRowStartIdx;
        curRowStartIdx += (tailAvergingRow + 1);
        _Params.tailSlideEndListW[coreIdx] = curRowStartIdx;
        // needCoreNum++;
      } else {
        _Params.tailSlideStartListW[coreIdx] = curRowStartIdx;
        curRowStartIdx += tailAvergingRow;
        _Params.tailSlideEndListW[coreIdx] = curRowStartIdx;
      }
    }
  }
  _Params.tailStartW = eleW - tailW;
  _Params.tailEndW = eleW;
  if(perCoreSlideNum > 0ULL || perCoreSlideEle > 0ULL){
    needCoreNum = _Params.CoreNum;
  }
  _Params.CoreNumW = needCoreNum;
}

void UpsampleBicubic2dGradTiling::CalcNeedCoreNumH() {
  // 当前须切分的元素总数
  uint64_t eleH = _Params.outputH;
  uint64_t eleW = _Params.outputW;
  // 切分每块的元素
  uint64_t slideSize = _Params.slideSize;
  // uint64_t slideNum = (eleH + slideSize - 1) / slideSize;
  uint64_t perCoreSlideNum = eleH / (_Params.CoreNum * slideSize);

  uint64_t tailH = eleH - perCoreSlideNum * slideSize * _Params.CoreNum;
  uint64_t tailAvergingRow = eleW /_Params.CoreNum;
  uint64_t tailExtraRow = eleW % _Params.CoreNum;

  uint64_t perCoreSlideEle = perCoreSlideNum * slideSize;
  uint64_t curRowStartIdx = 0;
  uint64_t needCoreNum = (tailAvergingRow == 0)? tailExtraRow:_Params.CoreNum;
  for (uint64_t coreIdx = 0; coreIdx < _Params.CoreNum; coreIdx++) {
    _Params.slideStartListH[coreIdx] = coreIdx * perCoreSlideEle;
    // 计算当前方向切块的终点
    _Params.slideEndListH[coreIdx] = _Params.slideStartListH[coreIdx] + perCoreSlideEle;
    if(tailH > 0) {
      if (coreIdx < tailExtraRow) {
        _Params.tailSlideStartListH[coreIdx] = curRowStartIdx;
        curRowStartIdx += (tailAvergingRow + 1);
        _Params.tailSlideEndListH[coreIdx] = curRowStartIdx;
      } else {
        _Params.tailSlideStartListH[coreIdx] = curRowStartIdx;
        curRowStartIdx += tailAvergingRow;
        _Params.tailSlideEndListH[coreIdx] = curRowStartIdx;
      }
    }
  }
  _Params.tailStartH = eleH - tailH;
  _Params.tailEndH = eleH;
  if(perCoreSlideNum > 0ULL || perCoreSlideEle > 0ULL){
    needCoreNum = _Params.CoreNum;
  }

  _Params.CoreNumH = needCoreNum;
}

void UpsampleBicubic2dGradTiling::CalcSingleCoreK() {
  // 计算singleCoreK,处理时增加余量
  if (!FloatEqual(_Params.scalesW, (float)0.0)) {
    _Params.singleCoreKW = int64_t((_Params.slideSize + NUM_FOUR) / _Params.scalesW) +1;
    _Params.singleCoreKW = _Params.singleCoreKW < _Params.inputW ? _Params.singleCoreKW : _Params.inputW;
  } else {
    _Params.singleCoreKW = _Params.inputW;
  }
  if (!FloatEqual(_Params.scalesH, (float)0.0)) {
    _Params.singleCoreKH = int64_t((_Params.slideSize + NUM_FOUR) / _Params.scalesH) +1;
    _Params.singleCoreKH = _Params.singleCoreKH < _Params.inputH ? _Params.singleCoreKH : _Params.inputH;
  } else {
    _Params.singleCoreKH = _Params.inputH;
  }
}

void UpsampleBicubic2dGradTiling::CalcTCubeTiling(const gert::TilingContext * context) {
  auto dataType = static_cast<matmul_tiling::DataType>(_Params.dataType);
  matmul_tiling::PlatformInfo platformInfo;
  InitPlatformInfo(context, platformInfo);
  _Params.radioMatrixSize = ((_Params.singleCoreKH > _Params.singleCoreKW) ? _Params.singleCoreKH : _Params.singleCoreKW)
                                      * _Params.slideSize;
  // 中间tensor
  _Params.intermediateMatrixSize = _Params.inputN * _Params.inputC * _Params.inputH *_Params.outputW;
  matmul_tiling::MatmulApiTiling matmul_h(platformInfo);
  // matmul_tiling::MatmulApiTiling matmul_h;
  matmul_h.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
  matmul_h.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
  matmul_h.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
  matmul_h.SetOrgShape(_Params.outputH, _Params.outputW, _Params.inputW);
  matmul_h.SetShape(_Params.slideSize, _Params.outputW, _Params.singleCoreKH);

  if (matmul_h.GetTiling(tilingData.MMParamH) == -1) {
    return ;
  }

  matmul_tiling::MatmulApiTiling matmul_w(platformInfo);
  matmul_w.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
  matmul_w.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
  matmul_w.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
  matmul_w.SetOrgShape(_Params.batch * _Params.inputH, _Params.outputW, _Params.inputW);
  matmul_w.SetShape(_Params.batch * _Params.inputH, _Params.slideSize, _Params.singleCoreKW);
  if (matmul_w.GetTiling(tilingData.MMParamW) == -1) {
    return ;
  }
}

bool UpsampleBicubic2dGradTiling::SetTilingDataDC(gert::TilingContext* context) {
  tilingData.set_dataType(static_cast<uint32_t>(_Params.dataType));
  tilingData.set_CoreNum(_Params.CoreNum);
  tilingData.set_CoreNumW(_Params.CoreNumW);
  tilingData.set_CoreNumH(_Params.CoreNumH);
  tilingData.set_alignCorners(_Params.alignCorners);
  tilingData.set_scalesH(_Params.scalesH);
  tilingData.set_scalesW(_Params.scalesW);
  tilingData.set_singleCoreKW(_Params.singleCoreKW);
  tilingData.set_singleCoreKH(_Params.singleCoreKH);
  tilingData.set_needExpandW(_Params.needExpandW);
  tilingData.set_needExpandH(_Params.needExpandH);

  tilingData.set_batch(_Params.batch);
  tilingData.set_inputN(_Params.inputN);
  tilingData.set_inputC(_Params.inputC);
  tilingData.set_inputH(_Params.inputH);
  tilingData.set_inputW(_Params.inputW);
  tilingData.set_outputH(_Params.outputH);
  tilingData.set_outputW(_Params.outputW);

  tilingData.set_tailStartW(_Params.tailStartW);
  tilingData.set_tailEndW(_Params.tailEndW);
  tilingData.set_tailStartH(_Params.tailStartH);
  tilingData.set_tailEndH(_Params.tailEndH);

  tilingData.set_slideStartListW(_Params.slideStartListW);
  tilingData.set_slideEndListW(_Params.slideEndListW);
  tilingData.set_tailSlideStartListW(_Params.tailSlideStartListW);
  tilingData.set_tailSlideEndListW(_Params.tailSlideEndListW);

  tilingData.set_slideStartListH(_Params.slideStartListH);
  tilingData.set_slideEndListH(_Params.slideEndListH);
  tilingData.set_tailSlideStartListH(_Params.tailSlideStartListH);
  tilingData.set_tailSlideEndListH(_Params.tailSlideEndListH);

  tilingData.set_slideSize(_Params.slideSize);
  tilingData.set_radioMatrixSize(_Params.radioMatrixSize);
  tilingData.set_intermediateMatrixSize(_Params.intermediateMatrixSize);

  tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
  return true;
}

bool UpsampleBicubic2dGradTiling::SetLaunchInfoDC(gert::TilingContext* context) {
  context->SetBlockDim(_Params.CoreNum);
  context->SetTilingKey(static_cast<int64_t>(UpsampleBicubic2dGradTilingKey::DETERMINISTIC_MODE));

  uint64_t workspaceSize = (_Params.intermediateMatrixSize + _Params.radioMatrixSize * _Params.CoreNum) * GetDtypeSize() + 16 * 1024 * 1024;
  AddWorkspace(context, workspaceSize);
  return true;
}

ge::graphStatus UpsampleBicubic2dGradTiling::runTiling(gert::TilingContext* context) {
  OP_TILING_CHECK(!GetPlatformInfo(context),
                VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get platforminfo fail."),
                return ge::GRAPH_FAILED);
  OP_TILING_CHECK(!GetCheckAttr(context),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "check attr fail."),
                  return ge::GRAPH_FAILED);
  OP_TILING_CHECK(!CheckInOutShapes(context),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "check shape fail."),
                  return ge::GRAPH_FAILED);
  auto tempGetInputDesc = context->GetInputDesc(0);
  OP_TILING_CHECK(tempGetInputDesc == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "inputDesc is nullptr."),
                  return ge::GRAPH_FAILED);
  _Params.dataType = tempGetInputDesc->GetDataType();
  OP_TILING_CHECK(_Params.dataType != ge::DataType::DT_FLOAT
                  && _Params.dataType != ge::DataType::DT_FLOAT16
                  && _Params.dataType != ge::DataType::DT_BF16,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "check dtype fail."),
                  return ge::GRAPH_FAILED);
  _Params.deterministic = context->GetDeterministic();

  if (_Params.deterministic) {
    OP_TILING_CHECK(!GetTilingDataDC(context),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get tiling data fail."),
                    return ge::GRAPH_FAILED);
    // tilingdata
    OP_TILING_CHECK(!SetTilingDataDC(context),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "set tiling data fail."),
                    return ge::GRAPH_FAILED);
    // launchinfo: tilingkey, workspace, blockdim
    OP_TILING_CHECK(!SetLaunchInfoDC(context),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "set launchinfo fail."),
                    return ge::GRAPH_FAILED);
  } else {
    OP_TILING_CHECK(!GetTilingData(context),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get tiling data fail."),
                    return ge::GRAPH_FAILED);
    // tilingdata
    OP_TILING_CHECK(!SetTilingData(context),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "set tiling data fail."),
                    return ge::GRAPH_FAILED);
    // launchinfo: tilingkey, workspace, blockdim
    OP_TILING_CHECK(!SetLaunchInfo(context),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "set launchinfo fail."),
                    return ge::GRAPH_FAILED);
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForUpsampleBicubic2dGrad(gert::TilingContext* context) {
  UpsampleBicubic2dGradTiling tiling_handle;
  return tiling_handle.runTiling(context);
}

IMPL_OP_OPTILING(UpsampleBicubic2dGrad).Tiling(TilingForUpsampleBicubic2dGrad);
}  // namespace optiling
