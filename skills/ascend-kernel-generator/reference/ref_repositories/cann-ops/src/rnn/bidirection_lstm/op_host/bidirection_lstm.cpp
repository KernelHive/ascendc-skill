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
 * @file bidirection_lstm.cpp
 */

#include <cmath>
#include <vector>
#include "bidirection_lstm_tiling.h"
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace optiling {

#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }

#define OPS_CHECK_NULL_WITH_CONTEXT_RET(context, ptr, ret)                                       \
  if ((ptr) == nullptr) {                                                                        \
    const char* name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName(); \
    std::printf("EZ9999 op[%s], %s is nullptr!", name, #ptr);                           \
    return ret;                                                                                  \
  }

#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)

bool AddWorkspaceLSTM(gert::TilingContext* context, const size_t workspace) {
  size_t* workspace_size = context->GetWorkspaceSizes(1);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, workspace_size, false);
  *workspace_size = workspace;
  return true;
}

bool BidirectionLSTMTiling::CheckTensorShape(gert::TilingContext* context, gert::Shape& shape, uint64_t ndim, std::vector<uint64_t> dims) {
  OP_TILING_CHECK(shape.GetDimNum() != ndim,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                  "BidirectionLSTM tensor shape ndim:%lu wrong expected %lu, please check.",
                  static_cast<uint64_t>(shape.GetDimNum()), ndim), return false);
  for(uint32_t i = 0; i < ndim; i++) {
    OP_TILING_CHECK(static_cast<uint64_t>(shape.GetDim(i)) != dims[i],
                VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                "BidirectionLSTM tensor shape dim[%d]:%lu wrong expected %lu, please check.",
                i, static_cast<uint64_t>(shape.GetDim(i)), dims[i]), return false);
  }
  return true;
}

bool BidirectionLSTMTiling::CheckInputShapes(gert::TilingContext* context) {
  auto x = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, x, false);
  auto shape = x->GetStorageShape();
  if(!_Params.packed) {
    OP_TILING_CHECK(!CheckTensorShape(context, shape, 3, {_Params.sequenceLength, _Params.batchSize, _Params.inputSize}),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                    "BidirectionLSTM get x shape wrong, please check."), return false);
  }

  auto init_h = context->GetInputShape(1);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, init_h, false);
  shape = init_h->GetStorageShape();
  OP_TILING_CHECK(!CheckTensorShape(context, shape, 3, {(_Params.bidirection ?  2 : 1) * _Params.numLayers, _Params.batchSize, _Params.hiddenSize}),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                  "BidirectionLSTM get init_h shape wrong, please check."), return false);

  auto init_c = context->GetInputShape(2);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, init_c, false);
  shape = init_c->GetStorageShape();
  OP_TILING_CHECK(!CheckTensorShape(context, shape, 3, {(_Params.bidirection ?  2 : 1) * _Params.numLayers, _Params.batchSize, _Params.hiddenSize}),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                  "BidirectionLSTM get init_h shape wrong, please check."), return false);

  auto w_ih = context->GetInputShape(3);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, w_ih, false);
  shape = w_ih->GetStorageShape();
  OP_TILING_CHECK(!CheckTensorShape(context, shape, 2, {NUM_OF_GATE * _Params.hiddenSize, _Params.inputSize}),
                VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                "BidirectionLSTM get w_ih shape wrong, please check."), return false);

  auto w_hh = context->GetInputShape(4);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, w_hh, false);
  shape = w_hh->GetStorageShape();
  OP_TILING_CHECK(!CheckTensorShape(context, shape, 2, {NUM_OF_GATE * _Params.hiddenSize, _Params.hiddenSize}),
                VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                "BidirectionLSTM get w_hh shape wrong, please check."), return false);

  return true;
}

bool BidirectionLSTMTiling::CheckOptionalInputShapes(gert::TilingContext* context) {
  if(_Params.isBias){
    auto b_ih = context->GetOptionalInputShape(5);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, b_ih, false);
    auto shape = b_ih->GetStorageShape();
    OP_TILING_CHECK(!CheckTensorShape(context, shape, 1, {NUM_OF_GATE * _Params.hiddenSize}),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                  "BidirectionLSTM get b_ih shape wrong, please check."), return false);

    auto b_hh = context->GetOptionalInputShape(6);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, b_hh, false);
    shape = b_hh->GetStorageShape();
    OP_TILING_CHECK(!CheckTensorShape(context, shape, 1, {NUM_OF_GATE * _Params.hiddenSize}),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                  "BidirectionLSTM get b_hh shape wrong, please check."), return false);
  }

  if(_Params.bidirection){
    auto w_ih_reverse = context->GetOptionalInputShape(7);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, w_ih_reverse, false);
    auto shape = w_ih_reverse->GetStorageShape();
    OP_TILING_CHECK(!CheckTensorShape(context, shape, 2, {NUM_OF_GATE * _Params.hiddenSize, _Params.inputSize}),
              VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
              "BidirectionLSTM get w_ih_reverse shape wrong, please check."), return false);

    auto w_hh_reverse = context->GetOptionalInputShape(8);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, w_hh_reverse, false);
    shape = w_hh_reverse->GetStorageShape();
    OP_TILING_CHECK(!CheckTensorShape(context, shape, 2, {NUM_OF_GATE * _Params.hiddenSize, _Params.hiddenSize}),
              VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
              "BidirectionLSTM get w_hh_reverse shape wrong, please check."), return false);

    if(_Params.isBias){
      auto b_ih_reverse = context->GetOptionalInputShape(9);
      OPS_CHECK_NULL_WITH_CONTEXT_RET(context, b_ih_reverse, false);
      shape = b_ih_reverse->GetStorageShape();
      OP_TILING_CHECK(!CheckTensorShape(context, shape, 1, {NUM_OF_GATE * _Params.hiddenSize}),
              VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
              "BidirectionLSTM get b_ih_reverse shape wrong, please check."), return false);

      auto b_hh_reverse = context->GetOptionalInputShape(10);
      OPS_CHECK_NULL_WITH_CONTEXT_RET(context, b_hh_reverse, false);
      shape = b_hh_reverse->GetStorageShape();
      OP_TILING_CHECK(!CheckTensorShape(context, shape, 1, {NUM_OF_GATE * _Params.hiddenSize}),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
        "BidirectionLSTM get b_hh_reverse shape wrong, please check."), return false);
    }
  }
  return true;
}

bool BidirectionLSTMTiling::CheckOutputShapes(gert::TilingContext* context) {
  auto output = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, output, false);
  auto shape = output->GetStorageShape();
  if(!_Params.packed) {
    OP_TILING_CHECK(!CheckTensorShape(context, shape, 3, {_Params.sequenceLength, _Params.batchSize, (_Params.bidirection ?  2 : 1) * _Params.hiddenSize}),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                  "BidirectionLSTM get output shape wrong, please check."), return false);
  } else {
    OP_TILING_CHECK(!CheckTensorShape(context, shape, 2, {_Params.totalBatchSizes, (_Params.bidirection ?  2 : 1) * _Params.hiddenSize}),
              VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
              "BidirectionLSTM get output shape wrong, please check."), return false);
  }

  auto output_h = context->GetOutputShape(1);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, output_h, false);
  shape = output_h->GetStorageShape();
  OP_TILING_CHECK(!CheckTensorShape(context, shape, 3, {(_Params.bidirection ?  2 : 1) * _Params.numLayers, _Params.batchSize, _Params.hiddenSize}),
              VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
              "BidirectionLSTM get output_h shape wrong, please check."), return false);

  auto output_c = context->GetOutputShape(2);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, output_c, false);
  shape = output_c->GetStorageShape();
  OP_TILING_CHECK(!CheckTensorShape(context, shape, 3, {(_Params.bidirection ?  2 : 1) * _Params.numLayers, _Params.batchSize, _Params.hiddenSize}),
              VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
              "BidirectionLSTM get output_c shape wrong, please check."), return false);

  return true;
}

bool BidirectionLSTMTiling::CheckInOutShapes(gert::TilingContext* context) {
  auto batch_size = context->GetOptionalInputShape(11);
  if(batch_size == nullptr) {
    _Params.isSeq = false;
    _Params.packed = false;
  } else {
    auto shape = batch_size->GetStorageShape();
    OP_TILING_CHECK(shape.GetDimNum() != 1,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                    "BidirectionLSTM get batch_size shape ndim is not 1, please check."), return false);
    OP_TILING_CHECK(shape.GetDim(0) > MAX_SEQ,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                    "BidirectionLSTM get batch_size length %lu exceed limit %u, please check.", static_cast<uint64_t>(shape.GetDim(0)), MAX_SEQ), return false);
    _Params.isSeq = true;
    _Params.sequenceLength = shape.GetDim(0);
  }

  auto x = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, x, false);
  auto shape = x->GetStorageShape();
  if(!_Params.packed){
    OP_TILING_CHECK(shape.GetDimNum() != NUMBER_THREE,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                    "BidirectionLSTM get x shape ndim is not 3, please check."), return false);
    _Params.inputSize = shape.GetDim(NUMBER_TWO);
  } else {
    OP_TILING_CHECK(shape.GetDimNum() != NUMBER_TWO,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                    "BidirectionLSTM get x shape ndim is not 2, please check."), return false);
    _Params.totalBatchSizes = shape.GetDim(0);
    _Params.inputSize = shape.GetDim(1);
  }
  if(!_Params.isSeq){
    _Params.sequenceLength = shape.GetDim(0);
  }

  auto init_h = context->GetInputShape(1);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, init_h, false);
  shape = init_h->GetStorageShape();
  OP_TILING_CHECK(shape.GetDimNum() != NUMBER_THREE,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                  "BidirectionLSTM get init_h shape ndim is not 3, please check."), return false);
  _Params.batchSize = shape.GetDim(1);
  _Params.hiddenSize = shape.GetDim(NUMBER_TWO);

  OP_TILING_CHECK(!CheckInputShapes(context),
                VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                "BidirectionLSTM CheckInputShapes failed, please check."), return false);
  OP_TILING_CHECK(!CheckOptionalInputShapes(context),
                VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                "BidirectionLSTM CheckOptionalInputShapes failed, please check."), return false);
  OP_TILING_CHECK(!CheckOutputShapes(context),
                VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                "BidirectionLSTM CheckOutputShapes failed, please check."), return false);
  return true;
}

bool BidirectionLSTMTiling::GetCheckAttr(gert::TilingContext* context) {
  auto attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, attrs, false);
  //  optional
  const int64_t *num_layers = attrs->GetAttrPointer<int64_t>(0);
  const bool *isbias = attrs->GetAttrPointer<bool>(1);
  const bool *batch_first = attrs->GetAttrPointer<bool>(2);
  const bool *bidirection = attrs->GetAttrPointer<bool>(3);
  const bool *packed = attrs->GetAttrPointer<bool>(4);

  OP_TILING_CHECK(*num_layers != 1,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                  "BidirectionLSTM attr num_layers only support 1, please check."), return false);
  OP_TILING_CHECK(*batch_first,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                  "BidirectionLSTM attr batch_first only support false, please check."), return false);

  _Params.numLayers = *num_layers;
  _Params.isBias = *isbias;
  _Params.batchFirst = *batch_first;
  _Params.bidirection = *bidirection;
  if(packed == nullptr) {
    _Params.packed = false;
  } else {
    _Params.packed = *packed;
  }
  return true;
}

bool BidirectionLSTMTiling::GetPlatformInfo(gert::TilingContext* context) {
  fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
  OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
  std::string val;
  platformInfo->GetPlatformRes("AICoreintrinsicDtypeMap","Intrinsic_fix_pipe_l0c2out",val);
  OP_TILING_CHECK(!val.empty(), VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                  "BidirectionLSTM support only ASCEND310P for now"), return false);
  _Params.CoreNum = NUM_OF_AICORE;
  platformInfo->GetLocalMemSize(fe::LocalMemType::UB, _Params.UBSize);
  platformInfo->GetLocalMemSize(fe::LocalMemType::L1, _Params.L1Size);
  platformInfo->GetLocalMemSize(fe::LocalMemType::L0_A, _Params.L0ASize);
  platformInfo->GetLocalMemSize(fe::LocalMemType::L0_B, _Params.L0BSize);
  platformInfo->GetLocalMemSize(fe::LocalMemType::L0_C, _Params.L0CSize);
  return true;
}

void BidirectionLSTMTiling::GetPaddataTilingData() {
  _Params.padBaseN = UB_PAD_SIZE / GetDtypeSize(ge::DataType::DT_FLOAT16);
  _Params.padInnerBaseN = PAD_BASEN;
  _Params.padInputBaseM = _Params.padBaseN / (_Params.inputSizeAligned < PAD_BASEN ? _Params.inputSizeAligned : PAD_BASEN);
  _Params.padInputBaseM = (_Params.padInputBaseM < PAD_BASEM) ? _Params.padInputBaseM : PAD_BASEM;
  _Params.padInputInnerLoop = (_Params.inputSizeAligned + PAD_BASEN - 1) / PAD_BASEN;
  _Params.padInputInnerTailN = _Params.inputSizeAligned % PAD_BASEN;
  _Params.padHiddenBaseM = _Params.padBaseN / (_Params.hiddenSizeAligned < PAD_BASEN ? _Params.hiddenSizeAligned : PAD_BASEN);
  _Params.padHiddenBaseM = (_Params.padHiddenBaseM < PAD_BASEM) ? _Params.padHiddenBaseM : PAD_BASEM;
  _Params.padHiddenInnerLoop = (_Params.hiddenSizeAligned + PAD_BASEN - 1) / PAD_BASEN;
  _Params.padHiddenInnerTailN = _Params.hiddenSizeAligned % PAD_BASEN;

  int64_t padInOutSingeCoreM = (_Params.batchSize * _Params.sequenceLength) / NUM_PER_BLOCK_FLOAT16;
  _Params.padInOutTailCoreNum = (_Params.batchSize * _Params.sequenceLength) % NUM_PER_BLOCK_FLOAT16;
  if(_Params.packed) {
    padInOutSingeCoreM = _Params.totalBatchSizes / NUM_PER_BLOCK_FLOAT16;
    _Params.padInOutTailCoreNum = _Params.totalBatchSizes % NUM_PER_BLOCK_FLOAT16;
  }
  int64_t padWeightSingeCoreM = (NUM_OF_GATE * _Params.hiddenSize) / NUM_PER_BLOCK_FLOAT16;
  _Params.padWeightTailCoreNum = (NUM_OF_GATE * _Params.hiddenSize) % NUM_PER_BLOCK_FLOAT16;
  int64_t padInitSingeCoreM = (_Params.batchSize) / NUM_PER_BLOCK_FLOAT16;
  _Params.padInitTailCoreNum = (_Params.batchSize) % NUM_PER_BLOCK_FLOAT16;

  _Params.padInputLoop = padInOutSingeCoreM / _Params.padInputBaseM;
  _Params.padInputTailM = padInOutSingeCoreM % _Params.padInputBaseM;
  _Params.padWeightIHLoop = padWeightSingeCoreM / _Params.padInputBaseM;
  _Params.padWeightIHTailM = padWeightSingeCoreM % _Params.padInputBaseM;
  _Params.padWeightHHLoop = padWeightSingeCoreM / _Params.padHiddenBaseM;
  _Params.padWeightHHTailM = padWeightSingeCoreM % _Params.padHiddenBaseM;
  _Params.padInitLoop = padInitSingeCoreM / _Params.padHiddenBaseM;
  _Params.padInitTailM = padInitSingeCoreM % _Params.padHiddenBaseM;
  _Params.padOutputLoop = padInOutSingeCoreM / _Params.padHiddenBaseM;
  _Params.padOutputTailM = padInOutSingeCoreM % _Params.padHiddenBaseM;

  int64_t padInputNum = _Params.inputSizeAligned - _Params.inputSize;
  _Params.padInputMask = 0;
  for(int i=0;i<NUM_PER_BLOCK_FLOAT16;i++){
    _Params.padInputMask = _Params.padInputMask << 1;
    if(i < padInputNum) _Params.padInputMask = (_Params.padInputMask | 1);
  }
  int64_t padHiddenNum = _Params.hiddenSizeAligned - _Params.hiddenSize;
  _Params.padHiddenMask = 0;
  for(int i=0;i<NUM_PER_BLOCK_FLOAT16;i++){
    _Params.padHiddenMask = _Params.padHiddenMask << 1;
    if(i < padHiddenNum) _Params.padHiddenMask = (_Params.padHiddenMask | 1);
  }
}

void BidirectionLSTMTiling::GetCleardataTilingData() {
  int64_t total_count = (_Params.bidirection ? NUMBER_TWO : 1) * _Params.sequenceLength * _Params.batchSize * _Params.hiddenSize;
  if(_Params.packed) {
    total_count = (_Params.bidirection ? NUMBER_TWO : 1) * _Params.totalBatchSizes * _Params.hiddenSize;
  }
  int64_t step_count = _Params.CoreNum * _Params.padBaseN;

  _Params.clearInOutLoop = total_count / step_count;
  int64_t total_count_tail = total_count % step_count;
  int64_t total_block_tail = (total_count_tail + NUM_PER_BLOCK_FLOAT16 - 1) / NUM_PER_BLOCK_FLOAT16;
  _Params.clearInOutTailN = total_block_tail / _Params.CoreNum * NUM_PER_BLOCK_FLOAT16;
  _Params.clearInOutTailCoreNum = total_block_tail % _Params.CoreNum;

  total_count = (_Params.bidirection ? NUMBER_TWO : 1) * _Params.batchSize * _Params.hiddenSize;

  _Params.clearInitLoop = total_count / step_count;
  total_count_tail = total_count % step_count;
  total_block_tail = (total_count_tail + NUM_PER_BLOCK_FLOAT16 - 1) / NUM_PER_BLOCK_FLOAT16;
  _Params.clearInitTailN = total_block_tail / _Params.CoreNum * NUM_PER_BLOCK_FLOAT16;
  _Params.clearInitTailCoreNum = total_block_tail % _Params.CoreNum;
}

void BidirectionLSTMTiling::GetTransdataTilingData() {
  _Params.transBaseN = UB_TRANS_SIZE / GetDtypeSize(ge::DataType::DT_FLOAT16);
  _Params.transInnerBaseN = _Params.transBaseN / NUM_PER_BLOCK_FLOAT16;
  _Params.transInputInnerLoop = (_Params.inputSizeAligned + _Params.transInnerBaseN - 1) / _Params.transInnerBaseN;
  _Params.transInputInnerTailN = _Params.inputSizeAligned % _Params.transInnerBaseN;
  _Params.transInputInnerTailN = _Params.transInputInnerTailN == 0 ? _Params.transInnerBaseN : _Params.transInputInnerTailN;
  _Params.transHiddenInnerLoop = (_Params.hiddenSizeAligned + _Params.transInnerBaseN - 1) / _Params.transInnerBaseN;
  _Params.transHiddenInnerTailN = _Params.hiddenSizeAligned % _Params.transInnerBaseN;
  _Params.transHiddenInnerTailN = _Params.transHiddenInnerTailN == 0 ? _Params.transInnerBaseN : _Params.transHiddenInnerTailN;

  int64_t transInOutM = _Params.sequenceLength * _Params.batchSizeAligned / NUM_PER_BLOCK_FLOAT16;
  _Params.transInOutLoopBatch = _Params.batchSizeAligned / NUM_PER_BLOCK_FLOAT16;
  _Params.transInOutLoop = transInOutM / _Params.CoreNum;
  _Params.transInOutTailCoreNum = transInOutM % _Params.CoreNum;
  int64_t transWeightM = NUM_OF_GATE * _Params.hiddenSizeAligned / NUM_PER_BLOCK_FLOAT16;
  _Params.transWeightLoopBatch = _Params.hiddenSizeAligned / NUM_PER_BLOCK_FLOAT16;
  _Params.transWeightLoop = transWeightM / _Params.CoreNum;
  _Params.transWeightTailCoreNum = transWeightM % _Params.CoreNum;
  int64_t transInitM = _Params.batchSizeAligned / NUM_PER_BLOCK_FLOAT16;
  _Params.transInitLoop = transInitM / _Params.CoreNum;
  _Params.transInitTailCoreNum = transInitM % _Params.CoreNum;
}

void BidirectionLSTMTiling::GetPadAndTransdataTilingData() {
  _Params.batchSizeAligned = (_Params.batchSize + NUM_PER_BLOCK_FLOAT16 - 1) / NUM_PER_BLOCK_FLOAT16 * NUM_PER_BLOCK_FLOAT16;
  _Params.inputSizeAligned = (_Params.inputSize + NUM_PER_BLOCK_FLOAT16 - 1) / NUM_PER_BLOCK_FLOAT16 * NUM_PER_BLOCK_FLOAT16;
  _Params.hiddenSizeAligned = (_Params.hiddenSize + NUM_PER_BLOCK_FLOAT16 - 1) / NUM_PER_BLOCK_FLOAT16 * NUM_PER_BLOCK_FLOAT16;

  // buffer size
  _Params.workspaceWihN = _Params.inputSizeAligned * _Params.hiddenSizeAligned;
  _Params.workspaceWhhN = _Params.hiddenSizeAligned * _Params.hiddenSizeAligned;
  if(_Params.isBias){
    _Params.workspaceBN = NUM_PER_BLOCK_FLOAT16 * _Params.hiddenSizeAligned;
  }
  _Params.workspaceGateN = _Params.batchSizeAligned * _Params.hiddenSizeAligned;
  _Params.workspaceInOutStep =  _Params.batchSizeAligned
                            * (_Params.inputSizeAligned > _Params.hiddenSizeAligned ? _Params.inputSizeAligned : _Params.hiddenSizeAligned);
  _Params.workspaceInOutN = (_Params.sequenceLength + 1) * _Params.workspaceInOutStep;
  _Params.workspaceTmpN = _Params.workspaceWihN > _Params.workspaceWhhN ? _Params.workspaceWihN : _Params.workspaceWhhN;
  _Params.workspaceTmpN *= NUM_OF_GATE;
  _Params.workspaceTmpN = _Params.workspaceTmpN > _Params.workspaceInOutN ? _Params.workspaceTmpN : _Params.workspaceInOutN;

  // pad
  GetPaddataTilingData();
  //clearbeforepad
  GetCleardataTilingData();
  // trans
  GetTransdataTilingData();
}

void BidirectionLSTMTiling::GetIntraCoreMMTilingData(uint32_t BaseMNFractalN, uint32_t &BaseMNum, uint32_t &BaseNNum) {
  uint32_t MNFractalN = _Params.IHSingleCoreM * _Params.IHSingleCoreN;
  if(MNFractalN <= BaseMNFractalN){
    BaseMNum = 1;
    BaseNNum = 1;
  } else {
    float MNPieces = (float)MNFractalN / (float)BaseMNFractalN;
    float MDivN = (float)_Params.IHSingleCoreM/(float)_Params.IHSingleCoreN;
    float tmp = std::sqrt(MNPieces / MDivN);
    uint32_t baseM = static_cast<uint32_t>(floor((float)_Params.IHSingleCoreM / (MDivN * tmp)));
    uint32_t baseN = static_cast<uint32_t>(floor((float)_Params.IHSingleCoreN / tmp));
    BaseMNum = (_Params.IHSingleCoreM + baseM - 1) / baseM;
    BaseNNum = (_Params.IHSingleCoreN + baseN - 1) / baseN;
    if(_Params.IHSingleCoreM > _Params.IHSingleCoreN) {
      uint32_t BaseMFractalN_ = (_Params.IHSingleCoreM + BaseMNum - 1) / BaseMNum;
      while(BaseNNum > 1 && BaseMFractalN_ * ((_Params.IHSingleCoreN + BaseNNum - 1 - 1) / (BaseNNum - 1)) <= BaseMNFractalN) {
        BaseNNum -= 1;
      }
      uint32_t BaseNFractalN_ = (_Params.IHSingleCoreN + BaseNNum - 1) / BaseNNum;
      while(BaseMNum > 1 && BaseNFractalN_ * ((_Params.IHSingleCoreM + BaseMNum - 1 - 1) / (BaseMNum - 1)) <= BaseMNFractalN) {
        BaseMNum -= 1;
      }
    } else {
      uint32_t BaseNFractalN_ = (_Params.IHSingleCoreN + BaseNNum - 1) / BaseNNum;
      while(BaseMNum > 1 && BaseNFractalN_ * ((_Params.IHSingleCoreM + BaseMNum - 1 - 1) / (BaseMNum - 1)) <= BaseMNFractalN) {
        BaseMNum -= 1;
      }
      uint32_t BaseMFractalN_ = (_Params.IHSingleCoreM + BaseMNum - 1) / BaseMNum;
      while(BaseNNum > 1 && BaseMFractalN_ * ((_Params.IHSingleCoreN + BaseNNum - 1 - 1) / (BaseNNum - 1)) <= BaseMNFractalN) {
        BaseNNum -= 1;
      }
    }
  }
}

bool BidirectionLSTMTiling::ChooseMMStrategy(uint32_t BaseMNumA, uint32_t BaseNNumA, uint32_t BaseMNumB, uint32_t BaseNNumB) {
  uint32_t A = _Params.IHSingleCoreM * _Params.IHSingleCoreN;
  uint32_t B = (_Params.IHSingleCoreM * (BaseNNumA - BaseNNumB) + _Params.IHSingleCoreN * (BaseMNumA - BaseMNumB))
               * (_Params.IHSingleCoreK + _Params.HHSingleCoreK);
  if(A_FACTOR * static_cast<float>(A) > B_FACTOR * static_cast<float>(B)) {
    return true;
  }
  return false;
}

void BidirectionLSTMTiling::GetMMTilingDataBaseMN() {
    uint32_t baseFractalSizeFP16 = _Params.baseMKN * _Params.baseMKN * GetDtypeSize(ge::DataType::DT_FLOAT16);
    uint32_t baseFractalSizeFP32 = _Params.baseMKN * _Params.baseMKN * GetDtypeSize(ge::DataType::DT_FLOAT);
    uint32_t BaseMNumA, BaseNNumA;
    uint32_t BaseMNFractalNA = _Params.L0CSize / (_Params.isBias ? NUMBER_THREE : NUMBER_TWO) / baseFractalSizeFP32;
    GetIntraCoreMMTilingData(BaseMNFractalNA, BaseMNumA, BaseNNumA);

    uint32_t BaseMNumB, BaseNNumB;
    uint32_t BaseMNFractalNB = _Params.L0CSize / baseFractalSizeFP32;
    GetIntraCoreMMTilingData(BaseMNFractalNB, BaseMNumB, BaseNNumB);
    if(ChooseMMStrategy(BaseMNumA, BaseNNumA, BaseMNumB, BaseNNumB)){
      _Params.activeMat = true;
      _Params.BaseMNFractalN = BaseMNFractalNA;
      _Params.IHBaseMNum = BaseMNumA;
      _Params.IHBaseNNum = BaseNNumA;
      _Params.HHBaseMNum = BaseMNumA;
      _Params.HHBaseNNum = BaseNNumA;
    } else {
      _Params.activeMat = false;
      _Params.BaseMNFractalN = BaseMNFractalNB;
      _Params.IHBaseMNum = BaseMNumB;
      _Params.IHBaseNNum = BaseNNumB;
      _Params.HHBaseMNum = BaseMNumB;
      _Params.HHBaseNNum = BaseNNumB;
    }
    _Params.BaseNKFractalN = _Params.L0ASize / baseFractalSizeFP16;
    _Params.BaseMKFractalN = _Params.L0BSize / baseFractalSizeFP16;
    uint32_t BaseMNumL0AB = (_Params.IHSingleCoreM + _Params.BaseMKFractalN - 1) / _Params.BaseMKFractalN;
    uint32_t BaseNNumL0AB = (_Params.IHSingleCoreN + _Params.BaseNKFractalN - 1) / _Params.BaseNKFractalN;
    if(_Params.IHBaseMNum < BaseMNumL0AB) {
      _Params.IHBaseMNum = BaseMNumL0AB;
      _Params.HHBaseMNum = BaseMNumL0AB;
    }
    if(_Params.IHBaseNNum < BaseNNumL0AB) {
      _Params.IHBaseNNum = BaseNNumL0AB;
      _Params.HHBaseNNum = BaseNNumL0AB;
    }

    _Params.matUBBaseN = _Params.BaseMNFractalN * baseFractalSizeFP32 / GetDtypeSize(ge::DataType::DT_FLOAT);
    _Params.l1BaseN = _Params.L1Size / NUMBER_FOUR / GetDtypeSize(ge::DataType::DT_FLOAT16);
}

void BidirectionLSTMTiling::GetMMTilingDataBaseK() {
    uint32_t MKFractalN = (_Params.IHSingleCoreM + _Params.IHBaseMNum - 1) / _Params.IHBaseMNum * _Params.IHSingleCoreK;
    uint32_t NKFractalN = (_Params.IHSingleCoreN + _Params.IHBaseNNum - 1) / _Params.IHBaseNNum * _Params.IHSingleCoreK;

    if (MKFractalN <= _Params.BaseMKFractalN && NKFractalN <= _Params.BaseNKFractalN) {
      _Params.IHBaseKNum = 1;
    } else {
      float MKPieces = (float)MKFractalN / (float)_Params.BaseMKFractalN;
      float NKPieces = (float)NKFractalN / (float)_Params.BaseNKFractalN;
      uint32_t baseK = static_cast<uint32_t>(floor((float)_Params.IHSingleCoreK / (MKPieces > NKPieces ? MKPieces : NKPieces)));
      _Params.IHBaseKNum = (_Params.IHSingleCoreK + baseK - 1) / baseK;
    }

    MKFractalN = (_Params.HHSingleCoreM + _Params.HHBaseMNum - 1) / _Params.HHBaseMNum * _Params.HHSingleCoreK;
    NKFractalN = (_Params.HHSingleCoreN + _Params.HHBaseNNum - 1) / _Params.HHBaseNNum * _Params.HHSingleCoreK;

    if (MKFractalN <= _Params.BaseMKFractalN && NKFractalN <= _Params.BaseNKFractalN) {
      _Params.HHBaseKNum = 1;
    } else {
      float MKPieces = (float)MKFractalN / (float)_Params.BaseMKFractalN;
      float NKPieces = (float)NKFractalN / (float)_Params.BaseNKFractalN;
      uint32_t baseK = static_cast<uint32_t>(floor((float)_Params.HHSingleCoreK / (MKPieces > NKPieces ? MKPieces : NKPieces)));
      _Params.HHBaseKNum = (_Params.HHSingleCoreK + baseK - 1) / baseK;
    }
}

void BidirectionLSTMTiling::GetMMTilingData() {
  _Params.baseMKN = NUM_PER_BLOCK_FLOAT16;
  // IH SingleCore
  _Params.IHM = _Params.batchSizeAligned / _Params.baseMKN;
  _Params.IHK = _Params.inputSizeAligned / _Params.baseMKN;
  _Params.IHN = _Params.hiddenSizeAligned / _Params.baseMKN;
  _Params.IHSingleCoreM = _Params.IHM;
  _Params.IHSingleCoreK = _Params.IHK;
  _Params.IHSingleCoreN = _Params.IHN;
  if(_Params.IHM >= _Params.IHN) {
    _Params.IHSingleCoreM = (_Params.IHSingleCoreM + NUM_MM_GROUP - 1) / NUM_MM_GROUP;
  } else {
    _Params.IHSingleCoreN = (_Params.IHSingleCoreN + NUM_MM_GROUP - 1) / NUM_MM_GROUP;
  }

  // HH SingleCore
  _Params.HHM = _Params.IHM;
  _Params.HHN = _Params.IHN;
  _Params.HHSingleCoreM = _Params.IHSingleCoreM;
  _Params.HHSingleCoreN = _Params.IHSingleCoreN;
  _Params.HHK = _Params.hiddenSizeAligned / _Params.baseMKN;
  _Params.HHSingleCoreK = _Params.HHK;

  // BaseM,N
  GetMMTilingDataBaseMN();
  // BaseK
  GetMMTilingDataBaseK();
}

uint64_t BidirectionLSTMTiling::GetDtypeSize(ge::DataType dtype) {
  switch(dtype) {
    case ge::DataType::DT_FLOAT16:
      return SIZE_OF_FLOAT16;
    case ge::DataType::DT_FLOAT:
      return SIZE_OF_FLOAT32;
    default :
      return SIZE_OF_FLOAT32;
  }
}

void BidirectionLSTMTiling::GetVecTilingData() {
  int64_t vecBaseRepeatMax = (UB_BUFF_SIZE / GetDtypeSize(ge::DataType::DT_FLOAT) + NUM_PER_REPEAT_FLOAT16 - 1) / NUM_PER_REPEAT_FLOAT16;
  int64_t total_count = _Params.batchSizeAligned * _Params.hiddenSizeAligned;
  int64_t total_repeat = (total_count + NUM_PER_REPEAT_FLOAT16 - 1) / NUM_PER_REPEAT_FLOAT16;
  int64_t step_repeat_max = _Params.CoreNum * vecBaseRepeatMax;

  int64_t vecLoop_ = (total_repeat + step_repeat_max - 1) / step_repeat_max;
  int64_t vecBaseRepeat = (((total_repeat + vecLoop_ - 1) / vecLoop_) + _Params.CoreNum - 1) / _Params.CoreNum;
  int64_t step_repeat_real = vecBaseRepeat * _Params.CoreNum;
  _Params.vecBaseN = vecBaseRepeat * NUM_PER_REPEAT_FLOAT16;
  _Params.vecLoop = total_repeat / step_repeat_real;
  int64_t total_repeat_tail = total_repeat % step_repeat_real;
  _Params.vecTailN = total_repeat_tail / _Params.CoreNum * NUM_PER_REPEAT_FLOAT16;
  _Params.vecTailCoreNum = total_repeat_tail % _Params.CoreNum;
}

uint64_t BidirectionLSTMTiling::GetTilingKey() {
  if(_Params.dataType == ge::DataType::DT_FLOAT16) {
    return static_cast<int64_t>(BidirectionLSTMTilingKey::FP16_BIDIR);
  }
  return static_cast<int64_t>(BidirectionLSTMTilingKey::UNDFINED);
}

bool BidirectionLSTMTiling::SetLaunchInfo(gert::TilingContext* context) {
  context->SetBlockDim(_Params.CoreNum);

  context->SetTilingKey(GetTilingKey());

  int64_t workspaceSize = _Params.workspaceGateN * (NUM_OF_GATE + 1) * GetDtypeSize(ge::DataType::DT_FLOAT)
                        + _Params.workspaceWihN * NUM_OF_GATE * GetDtypeSize(ge::DataType::DT_FLOAT16)
                        + _Params.workspaceWhhN * NUM_OF_GATE * GetDtypeSize(ge::DataType::DT_FLOAT16)
                        + (_Params.isBias ? _Params.workspaceBN * NUM_OF_GATE * GetDtypeSize(ge::DataType::DT_FLOAT) : 0)
                        + _Params.workspaceInOutN * GetDtypeSize(ge::DataType::DT_FLOAT16)
                        + _Params.workspaceTmpN * GetDtypeSize(ge::DataType::DT_FLOAT16)
                        + _Params.CoreNum * BLOCK_SIZE
                        + 2 * 1024 * 1024;
  AddWorkspaceLSTM(context, workspaceSize);
  return true;
}

void BidirectionLSTMTiling::_SetTilingDataA() {
  tilingData.set_CoreNum(_Params.CoreNum);
  tilingData.set_bidirection(_Params.bidirection);
  tilingData.set_isBias(_Params.isBias);
  tilingData.set_isSeq(_Params.isSeq);
  tilingData.set_packed(_Params.packed);
  tilingData.set_activeMat(_Params.activeMat);

  tilingData.set_sequenceLength(_Params.sequenceLength);
  tilingData.set_batchSize(_Params.batchSize);
  tilingData.set_inputSize(_Params.inputSize);
  tilingData.set_hiddenSize(_Params.hiddenSize);

  tilingData.set_batchSizeAligned(_Params.batchSizeAligned);
  tilingData.set_inputSizeAligned(_Params.inputSizeAligned);
  tilingData.set_hiddenSizeAligned(_Params.hiddenSizeAligned);
  tilingData.set_workspaceWihN(_Params.workspaceWihN);
  tilingData.set_workspaceWhhN(_Params.workspaceWhhN);
  tilingData.set_workspaceBN(_Params.workspaceBN);
  tilingData.set_workspaceGateN(_Params.workspaceGateN);
  tilingData.set_workspaceInOutN(_Params.workspaceInOutN);
  tilingData.set_workspaceInOutStep(_Params.workspaceInOutStep);
  tilingData.set_workspaceTmpN(_Params.workspaceTmpN);

  tilingData.set_padBaseN(_Params.padBaseN);
  tilingData.set_padInnerBaseN(_Params.padInnerBaseN);
  tilingData.set_padInputBaseM(_Params.padInputBaseM);
  tilingData.set_padInputInnerLoop(_Params.padInputInnerLoop);
  tilingData.set_padInputInnerTailN(_Params.padInputInnerTailN);
  tilingData.set_padHiddenBaseM(_Params.padHiddenBaseM);
  tilingData.set_padHiddenInnerLoop(_Params.padHiddenInnerLoop);
  tilingData.set_padHiddenInnerTailN(_Params.padHiddenInnerTailN);
  tilingData.set_padInputLoop(_Params.padInputLoop);
  tilingData.set_padInputTailM(_Params.padInputTailM);
  tilingData.set_padWeightIHLoop(_Params.padWeightIHLoop);
  tilingData.set_padWeightIHTailM(_Params.padWeightIHTailM);
  tilingData.set_padWeightHHLoop(_Params.padWeightHHLoop);
  tilingData.set_padWeightHHTailM(_Params.padWeightHHTailM);
  tilingData.set_padInitLoop(_Params.padInitLoop);
  tilingData.set_padInitTailM(_Params.padInitTailM);
  tilingData.set_padOutputLoop(_Params.padOutputLoop);
  tilingData.set_padOutputTailM(_Params.padOutputTailM);
  tilingData.set_padInOutTailCoreNum(_Params.padInOutTailCoreNum);
  tilingData.set_padWeightTailCoreNum(_Params.padWeightTailCoreNum);
  tilingData.set_padInitTailCoreNum(_Params.padInitTailCoreNum);
  tilingData.set_padInputMask(_Params.padInputMask);
  tilingData.set_padHiddenMask(_Params.padHiddenMask);
}

void BidirectionLSTMTiling::_SetTilingDataB() {
  tilingData.set_clearInOutLoop(_Params.clearInOutLoop);
  tilingData.set_clearInOutTailN(_Params.clearInOutTailN);
  tilingData.set_clearInOutTailCoreNum(_Params.clearInOutTailCoreNum);
  tilingData.set_clearInitLoop(_Params.clearInitLoop);
  tilingData.set_clearInitTailN(_Params.clearInitTailN);
  tilingData.set_clearInitTailCoreNum(_Params.clearInitTailCoreNum);

  tilingData.set_transBaseN(_Params.transBaseN);
  tilingData.set_transInOutLoopBatch(_Params.transInOutLoopBatch);
  tilingData.set_transInOutLoop(_Params.transInOutLoop);
  tilingData.set_transInOutTailCoreNum(_Params.transInOutTailCoreNum);
  tilingData.set_transWeightLoopBatch(_Params.transWeightLoopBatch);
  tilingData.set_transWeightLoop(_Params.transWeightLoop);
  tilingData.set_transWeightTailCoreNum(_Params.transWeightTailCoreNum);
  tilingData.set_transInitLoop(_Params.transInitLoop);
  tilingData.set_transInitTailCoreNum(_Params.transInitTailCoreNum);
  tilingData.set_transInnerBaseN(_Params.transInnerBaseN);
  tilingData.set_transInputInnerLoop(_Params.transInputInnerLoop);
  tilingData.set_transInputInnerTailN(_Params.transInputInnerTailN);
  tilingData.set_transHiddenInnerLoop(_Params.transHiddenInnerLoop);
  tilingData.set_transHiddenInnerTailN(_Params.transHiddenInnerTailN);

  tilingData.set_matUBBaseN(_Params.matUBBaseN);
  tilingData.set_l1BaseN(_Params.l1BaseN);
  tilingData.set_IHM(_Params.IHM);
  tilingData.set_IHK(_Params.IHK);
  tilingData.set_IHN(_Params.IHN);
  tilingData.set_IHSingleCoreM(_Params.IHSingleCoreM);
  tilingData.set_IHSingleCoreK(_Params.IHSingleCoreK);
  tilingData.set_IHSingleCoreN(_Params.IHSingleCoreN);
  tilingData.set_IHBaseMNum(_Params.IHBaseMNum);
  tilingData.set_IHBaseNNum(_Params.IHBaseNNum);
  tilingData.set_IHBaseKNum(_Params.IHBaseKNum);
  tilingData.set_HHM(_Params.HHM);
  tilingData.set_HHK(_Params.HHK);
  tilingData.set_HHN(_Params.HHN);
  tilingData.set_HHSingleCoreM(_Params.HHSingleCoreM);
  tilingData.set_HHSingleCoreK(_Params.HHSingleCoreK);
  tilingData.set_HHSingleCoreN(_Params.HHSingleCoreN);
  tilingData.set_HHBaseMNum(_Params.HHBaseMNum);
  tilingData.set_HHBaseNNum(_Params.HHBaseNNum);
  tilingData.set_HHBaseKNum(_Params.HHBaseKNum);

  tilingData.set_vecLoop(_Params.vecLoop);
  tilingData.set_vecBaseN(_Params.vecBaseN);
  tilingData.set_vecTailN(_Params.vecTailN);
  tilingData.set_vecTailCoreNum(_Params.vecTailCoreNum);

  tilingData.set_totalBatchSizes(_Params.totalBatchSizes);
}

bool BidirectionLSTMTiling::SetTilingData(gert::TilingContext* context) {
  _SetTilingDataA();
  _SetTilingDataB();
  tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

  return true;
}

ge::graphStatus BidirectionLSTMTiling::runTiling(gert::TilingContext* context) {
  // 310P AscendC platformINFO
  OP_TILING_CHECK(!GetPlatformInfo(context),
                VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get platforminfo fail."),
                return ge::GRAPH_FAILED);
  // attr
  OP_TILING_CHECK(!GetCheckAttr(context),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "check attr fail."),
                  return ge::GRAPH_FAILED);
  // shape
  OP_TILING_CHECK(!CheckInOutShapes(context),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "check shape fail."),
                  return ge::GRAPH_FAILED);

  // datatype;
  auto inputDesc = context->GetInputDesc(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
  _Params.dataType = inputDesc->GetDataType();
  OP_TILING_CHECK(_Params.dataType != ge::DataType::DT_FLOAT16,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "check dtype fail."),
                  return ge::GRAPH_FAILED);

  // calculate tilingdata
  GetPadAndTransdataTilingData();
  GetMMTilingData();
  GetVecTilingData();

  // tilingdata
  OP_TILING_CHECK(!SetTilingData(context),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "set tiling data fail."),
                  return ge::GRAPH_FAILED);

  // launchinfo: tilingkey, workspace, blockdim
  OP_TILING_CHECK(!SetLaunchInfo(context),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "set launchinfo fail."),
                  return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingBidirectionLSTM(gert::TilingContext* context) {
  BidirectionLSTMTiling tiling_handle;
  return tiling_handle.runTiling(context);
}
}  // namespace optiling

namespace ops {
class BidirectionLSTM : public OpDef {
   public:
    explicit BidirectionLSTM(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND,})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("init_h")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("init_c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w_ih")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w_hh")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});            
        this->Input("b_ih")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b_hh")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w_ih_reverse")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w_hh_reverse")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b_ih_reverse")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b_hh_reverse")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("output_h")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("output_c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("num_layers")
            .AttrType(OPTIONAL)
            .Int(1);
        this->Attr("isbias")
            .AttrType(OPTIONAL)
            .Bool(false);
        this->Attr("batch_first")
            .AttrType(OPTIONAL)
            .Bool(false);
        this->Attr("bidirection")
            .AttrType(OPTIONAL)
            .Bool(false);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);

        this->AICore()
            .SetTiling(optiling::TilingBidirectionLSTM)
            .AddConfig("ascend310p", aicore_config);
    }
};

OP_ADD(BidirectionLSTM);
}  // namespace ops