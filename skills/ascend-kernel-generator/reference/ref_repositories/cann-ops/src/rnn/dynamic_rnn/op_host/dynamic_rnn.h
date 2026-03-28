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
 * \file dynamic_rnn.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_RNN_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_RNN_H_
#include <cstdint>
#include <vector>
#include <string>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"

namespace optiling {

class CompileInfoBase;
using CompileInfoPtr = std::shared_ptr<CompileInfoBase>;

class CompileInfoBase {
public:
  CompileInfoBase() {}
  virtual ~CompileInfoBase() {}
};

constexpr int32_t DEFAULT_SHAPE_LIST_SIZE = 3;
constexpr int32_t DEFAULT_INDEX_TWO = 2;
constexpr int32_t DEFAULT_RETURN = -2;
constexpr int32_t DEFAULT_PARAS_INPUT_SIZE = 2;
constexpr int32_t DEFAULT_XSHAPE_SIZE = 3;
constexpr int32_t DEFAULT_WSHAPE_SIZE = 2;
constexpr int32_t DYNAMIC_DIM = -1;
constexpr int32_t CEIL_LENTH16 = 16;
constexpr int32_t BLOCK_DIM = 32;
constexpr int32_t WORKSPACE_SIZE = 4096;
constexpr int32_t L1_TO_L0A_MAX_SIZE = 64 * 1024;
constexpr int64_t MIN_BASE_BUFFER = 8 * 1024;
constexpr int64_t MIN_BASE_SHAPE = 2048;
constexpr int32_t UNKNOW_DIM = -2;
constexpr uint32_t SCHEDULE_MODE = 1; // batchmode
constexpr int32_t CONST_TWO = 2;

enum class GateOrder : int64_t {
  IJFO,
  IFJO
};

enum class RNNTilingKey : int64_t {
  MM_FP16_SPLIT = 10000001,
  MM_FP32_SPLIT,
  MM_HF32_SPLIT,
  MM_BF16_SPLIT
};

struct DynamicRnnTiling {
  // include 2 matmul tiling
  TCubeTiling matmulTiling;
  TCubeTiling inputMMParam;
  TCubeTiling hiddenMMParam;

  int64_t tilingKey;
  int64_t usedCoreNum;

  // rnn input tiling
  int64_t timeStep;
  int64_t batch;
  int64_t inputSize;
  int64_t hiddenSize;
  int64_t isBias;
  int64_t isInithc;
  int64_t isSeqLength;
  int64_t isHF32;

  // cache
  int64_t isCached;
  int64_t cacheLength;

  // rnn attr
  int64_t gateOrder;
  int64_t direction;
  int64_t isTraining;
  float cellClip;
  float forgetBias;

  // tmp
  uint64_t ubSize;
  uint64_t l1Size;
  int64_t maxUbSize;
  int64_t sysAicCoreNum;
  int64_t sysAivCoreNum;
  int32_t baseM;
  int32_t baseN;
  int32_t baseK;
  int32_t singleM;
  int32_t singleN;
  int32_t singleK;
  int32_t dataType;
  bool isUseMerged;
  bool isFullLoad;
};

BEGIN_TILING_DATA_DEF(DynamicRNNTilingData)

TILING_DATA_FIELD_DEF(int64_t, tilingKey);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);

// rnn input tiling
TILING_DATA_FIELD_DEF(int64_t, timeStep);
TILING_DATA_FIELD_DEF(int64_t, batch);
TILING_DATA_FIELD_DEF(int64_t, inputSize);
TILING_DATA_FIELD_DEF(int64_t, hiddenSize);
TILING_DATA_FIELD_DEF(int64_t, isBias);
TILING_DATA_FIELD_DEF(int64_t, isInithc);
TILING_DATA_FIELD_DEF(int64_t, isSeqLength);
TILING_DATA_FIELD_DEF(int64_t, isHF32);

// cache
TILING_DATA_FIELD_DEF(int64_t, isCached);
TILING_DATA_FIELD_DEF(int64_t, cacheLength);

// rnn attr
TILING_DATA_FIELD_DEF(int64_t, gateOrder);
TILING_DATA_FIELD_DEF(int64_t, direction);
TILING_DATA_FIELD_DEF(int64_t, isTraining);
TILING_DATA_FIELD_DEF(float, cellClip);
TILING_DATA_FIELD_DEF(float, forgetBias);

TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, inputMMParam);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, hiddenMMParam);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DynamicRNN, DynamicRNNTilingData)

class LstmTilingRNN {
 public:
    void GetAICoreIntrinsicDtype(fe::PlatFormInfos& platform_info, const std::string& intrinsic_name, bool& value);
    ge::graphStatus TilingWithAscendC(gert::TilingContext* context);
 protected:
    virtual bool CheckParamsShape(gert::TilingContext* context) {
        return true;
    };
    virtual bool CheckInitParamsShape(gert::TilingContext* context) {
        return true;
    };

    bool CheckParamsDtype(const gert::TilingContext* context);

    bool CheckAttr(gert::TilingContext* context);
    virtual bool CheckAttrOps(gert::TilingContext* context) {
        return true;
    };
    virtual bool CheckAttrTiling(gert::TilingContext* context) {
        return true;
    };

    virtual ge::graphStatus GetOpInfo(const gert::TilingContext* context, DynamicRnnTiling& rnnParams) {
        return ge::GRAPH_SUCCESS;
    };
    virtual ge::graphStatus GetAttr(const gert::TilingContext* context, DynamicRnnTiling& rnnParams) {
        return ge::GRAPH_SUCCESS;
    };

    ge::graphStatus GetMMTilingData(gert::TilingContext* context, DynamicRNNTilingData& tilingData,
                                                DynamicRnnTiling& rnnParams);
    ge::graphStatus GetMMTilingDataSplit(const gert::TilingContext* context, DynamicRNNTilingData& tilingData,
                                                     DynamicRnnTiling& rnnParams, matmul_tiling::DataType dataType);
    ge::graphStatus CalcTilingKey(DynamicRnnTiling& rnnParams);
    ge::graphStatus SetTilingData(gert::TilingContext* context, DynamicRNNTilingData& tilingData,
                                        DynamicRnnTiling& rnnParams);

 private:
  DynamicRNNTilingData tilingData;
  DynamicRnnTiling rnnParams;
};

struct DynamicRnnTilingData {
  int32_t sequenceLength;
  int32_t dynamicRnnBatch;
  int32_t chequeIndex;
};

struct DynamicRnnCompileInfo : public optiling::CompileInfoBase {
  std::vector<std::vector<int64_t>> tuneShapeList;
};

class DynamicRNNTiling : public LstmTilingRNN {
  protected:
    bool CheckParamsShape(gert::TilingContext* context) override;
    bool CheckInitParamsShape(gert::TilingContext* context) override;
    bool CheckAttrOps(gert::TilingContext* context) override;
    bool CheckAttrTiling(gert::TilingContext* context) override;

    ge::graphStatus GetOpInfo(const gert::TilingContext* context, DynamicRnnTiling& rnnParams) override;
    ge::graphStatus GetAttr(const gert::TilingContext* context, DynamicRnnTiling& rnnParams) override;
};

}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_RNN_H_