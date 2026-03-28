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
 * \file bidirection_lstm_tiling.h
 */
#ifndef BIDIRECTION_LSTM_TILING_H
#define BIDIRECTION_LSTM_TILING_H
#include <cstdint>
#include <vector>
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
namespace optiling {

constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t UB_FACTOR = 4 + 1 + 1;
constexpr int32_t UB_BUFF_SIZE = 42 * 1024;
constexpr int32_t UB_PAD_SIZE = 128 * 1024;
constexpr int32_t UB_TRANS_SIZE = 64 * 1024;
constexpr int32_t SIZE_OF_FLOAT16 = 2;
constexpr int32_t SIZE_OF_FLOAT32 = 4;
constexpr int32_t NUM_BITS_INT64 = 64;
constexpr int32_t NUM_PER_REPEAT_FLOAT16 = 128;
constexpr int32_t NUM_PER_BLOCK_FLOAT16 = 16;
constexpr int32_t NUM_PER_BLOCK_FLOAT32 = 8;
constexpr int32_t NUM_OF_GATE = 4;
constexpr int32_t NUM_OF_AICORE = 8;
constexpr int32_t NUM_MM_GROUP = 2;
constexpr int32_t NUMBER_TWO = 2;
constexpr int32_t NUMBER_THREE = 3;
constexpr int32_t NUMBER_FOUR = 4;
constexpr uint32_t PAD_BASEN = 4080;
constexpr uint32_t PAD_BASEM = 255;
constexpr uint32_t MAX_SEQ = 256;
constexpr float A_FACTOR = 1.1;
constexpr float B_FACTOR = 0.04;

enum class BidirectionLSTMTilingKey : uint64_t {
  FP16_BIDIR = 10000001,
  UNDFINED = 10000099
};

struct BidirectionLSTMCompileInfo {};
struct BidirectionLSTMParam {
  //platform
  uint64_t CoreNum;
  uint64_t UBSize;
  uint64_t L1Size;
  uint64_t L0ASize;
  uint64_t L0BSize;
  uint64_t L0CSize;

  //attr
  uint64_t inputSize;
  uint64_t hiddenSize;
  uint64_t numLayers;
  bool isBias;
  bool batchFirst;
  bool bidirection;
  bool activeMat;
  float dropout;    //reserved
  uint64_t projSize; //reserved

  // addition attr
  uint64_t sequenceLength;
  uint64_t batchSize;
  ge::DataType dataType;

  // tilingdata
  // workspace buff
  uint64_t batchSizeAligned;
  uint64_t inputSizeAligned;
  uint64_t hiddenSizeAligned;
  uint64_t workspaceWihN;
  uint64_t workspaceWhhN;
  uint64_t workspaceBN;
  uint64_t workspaceGateN;
  uint64_t workspaceInOutStep;
  uint64_t workspaceInOutN;
  uint64_t workspaceTmpN;
  // pad
  uint64_t padBaseN;
  uint64_t padInnerBaseN;
  uint64_t padInputBaseM;
  uint64_t padInputInnerLoop;
  uint64_t padInputInnerTailN;
  uint64_t padHiddenBaseM;
  uint64_t padHiddenInnerLoop;
  uint64_t padHiddenInnerTailN;
  uint64_t padInputLoop;
  uint64_t padInputTailM;
  uint64_t padWeightIHLoop;
  uint64_t padWeightIHTailM;
  uint64_t padWeightHHLoop;
  uint64_t padWeightHHTailM;
  uint64_t padInitLoop;
  uint64_t padInitTailM;
  uint64_t padOutputLoop;
  uint64_t padOutputTailM;
  uint64_t padInOutTailCoreNum;
  uint64_t padWeightTailCoreNum;
  uint64_t padInitTailCoreNum;
  uint64_t padInputMask;
  uint64_t padHiddenMask;

  uint64_t clearInOutLoop;
  uint64_t clearInOutTailN;
  uint64_t clearInOutTailCoreNum;
  uint64_t clearInitLoop;
  uint64_t clearInitTailN;
  uint64_t clearInitTailCoreNum;
  // trans
  uint64_t transBaseN;
  uint64_t transInOutLoopBatch;
  uint64_t transInOutLoop;
  uint64_t transInOutTailCoreNum;
  uint64_t transWeightLoopBatch;
  uint64_t transWeightLoop;
  uint64_t transWeightTailCoreNum;
  uint64_t transInitLoop;
  uint64_t transInitTailCoreNum;
  uint64_t transInnerBaseN;
  uint64_t transInputInnerLoop;
  uint64_t transInputInnerTailN;
  uint64_t transHiddenInnerLoop;
  uint64_t transHiddenInnerTailN;
  // mm
  uint64_t baseMKN;
  uint64_t BaseMNFractalN;
  uint64_t BaseNKFractalN;
  uint64_t BaseMKFractalN;
  uint64_t matUBBaseN;
  uint64_t l1BaseN;
  uint64_t IHM;
  uint64_t IHK;
  uint64_t IHN;
  uint64_t IHSingleCoreM;
  uint64_t IHSingleCoreK;
  uint64_t IHSingleCoreN;
  uint64_t IHBaseMNum;
  uint64_t IHBaseNNum;
  uint64_t IHBaseKNum;
  uint64_t HHM;
  uint64_t HHK;
  uint64_t HHN;
  uint64_t HHSingleCoreM;
  uint64_t HHSingleCoreK;
  uint64_t HHSingleCoreN;
  uint64_t HHBaseMNum;
  uint64_t HHBaseNNum;
  uint64_t HHBaseKNum;
  // vec
  uint64_t vecLoop;
  uint64_t vecBaseN;
  uint64_t vecTailCoreNum;
  uint64_t vecTailN;

  //seq
  bool isSeq;
  bool packed;
  uint64_t totalBatchSizes;
};

BEGIN_TILING_DATA_DEF(BidirectionLSTMTilingData)

TILING_DATA_FIELD_DEF(uint32_t, CoreNum);
TILING_DATA_FIELD_DEF(uint32_t, bidirection);
TILING_DATA_FIELD_DEF(uint32_t, isBias);
TILING_DATA_FIELD_DEF(uint32_t, isSeq);
TILING_DATA_FIELD_DEF(uint32_t, packed);
TILING_DATA_FIELD_DEF(uint32_t, activeMat);
//
TILING_DATA_FIELD_DEF(uint32_t, sequenceLength);
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, inputSize);
TILING_DATA_FIELD_DEF(uint32_t, hiddenSize);
//
TILING_DATA_FIELD_DEF(uint32_t, batchSizeAligned);
TILING_DATA_FIELD_DEF(uint32_t, inputSizeAligned);
TILING_DATA_FIELD_DEF(uint32_t, hiddenSizeAligned);
TILING_DATA_FIELD_DEF(uint32_t, workspaceWihN);
TILING_DATA_FIELD_DEF(uint32_t, workspaceWhhN);
TILING_DATA_FIELD_DEF(uint32_t, workspaceBN);
TILING_DATA_FIELD_DEF(uint32_t, workspaceGateN);
TILING_DATA_FIELD_DEF(uint32_t, workspaceInOutN);
TILING_DATA_FIELD_DEF(uint32_t, workspaceInOutStep);
TILING_DATA_FIELD_DEF(uint32_t, workspaceTmpN);
//
TILING_DATA_FIELD_DEF(uint32_t, padBaseN);
TILING_DATA_FIELD_DEF(uint32_t, padInnerBaseN);
TILING_DATA_FIELD_DEF(uint32_t, padInputBaseM);
TILING_DATA_FIELD_DEF(uint32_t, padInputInnerLoop);
TILING_DATA_FIELD_DEF(uint32_t, padInputInnerTailN);
TILING_DATA_FIELD_DEF(uint32_t, padHiddenBaseM);
TILING_DATA_FIELD_DEF(uint32_t, padHiddenInnerLoop);
TILING_DATA_FIELD_DEF(uint32_t, padHiddenInnerTailN);
TILING_DATA_FIELD_DEF(uint32_t, padInputLoop);
TILING_DATA_FIELD_DEF(uint32_t, padInputTailM);
TILING_DATA_FIELD_DEF(uint32_t, padWeightIHLoop);
TILING_DATA_FIELD_DEF(uint32_t, padWeightIHTailM);
TILING_DATA_FIELD_DEF(uint32_t, padWeightHHLoop);
TILING_DATA_FIELD_DEF(uint32_t, padWeightHHTailM);
TILING_DATA_FIELD_DEF(uint32_t, padInitLoop);
TILING_DATA_FIELD_DEF(uint32_t, padInitTailM);
TILING_DATA_FIELD_DEF(uint32_t, padOutputLoop);
TILING_DATA_FIELD_DEF(uint32_t, padOutputTailM);
TILING_DATA_FIELD_DEF(uint32_t, padInOutTailCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, padWeightTailCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, padInitTailCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, padInputMask);
TILING_DATA_FIELD_DEF(uint64_t, padHiddenMask);
//
TILING_DATA_FIELD_DEF(uint32_t, clearInOutLoop);
TILING_DATA_FIELD_DEF(uint32_t, clearInOutTailN);
TILING_DATA_FIELD_DEF(uint32_t, clearInOutTailCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, clearInitLoop);
TILING_DATA_FIELD_DEF(uint32_t, clearInitTailN);
TILING_DATA_FIELD_DEF(uint32_t, clearInitTailCoreNum);
//
TILING_DATA_FIELD_DEF(uint32_t, transBaseN);
TILING_DATA_FIELD_DEF(uint32_t, transInOutLoopBatch);
TILING_DATA_FIELD_DEF(uint32_t, transInOutLoop);
TILING_DATA_FIELD_DEF(uint32_t, transInOutTailCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, transWeightLoopBatch);
TILING_DATA_FIELD_DEF(uint32_t, transWeightLoop);
TILING_DATA_FIELD_DEF(uint32_t, transWeightTailCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, transInitLoop);
TILING_DATA_FIELD_DEF(uint32_t, transInitTailCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, transInnerBaseN);
TILING_DATA_FIELD_DEF(uint32_t, transInputInnerLoop);
TILING_DATA_FIELD_DEF(uint32_t, transInputInnerTailN);
TILING_DATA_FIELD_DEF(uint32_t, transHiddenInnerLoop);
TILING_DATA_FIELD_DEF(uint32_t, transHiddenInnerTailN);
//
TILING_DATA_FIELD_DEF(uint32_t, matUBBaseN);
TILING_DATA_FIELD_DEF(uint32_t, l1BaseN);
TILING_DATA_FIELD_DEF(uint32_t, IHM);
TILING_DATA_FIELD_DEF(uint32_t, IHK);
TILING_DATA_FIELD_DEF(uint32_t, IHN);
TILING_DATA_FIELD_DEF(uint32_t, IHSingleCoreM);
TILING_DATA_FIELD_DEF(uint32_t, IHSingleCoreK);
TILING_DATA_FIELD_DEF(uint32_t, IHSingleCoreN);
TILING_DATA_FIELD_DEF(uint32_t, IHBaseMNum);
TILING_DATA_FIELD_DEF(uint32_t, IHBaseNNum);
TILING_DATA_FIELD_DEF(uint32_t, IHBaseKNum);
TILING_DATA_FIELD_DEF(uint32_t, HHM);
TILING_DATA_FIELD_DEF(uint32_t, HHK);
TILING_DATA_FIELD_DEF(uint32_t, HHN);
TILING_DATA_FIELD_DEF(uint32_t, HHSingleCoreM);
TILING_DATA_FIELD_DEF(uint32_t, HHSingleCoreK);
TILING_DATA_FIELD_DEF(uint32_t, HHSingleCoreN);
TILING_DATA_FIELD_DEF(uint32_t, HHBaseMNum);
TILING_DATA_FIELD_DEF(uint32_t, HHBaseNNum);
TILING_DATA_FIELD_DEF(uint32_t, HHBaseKNum);
//
TILING_DATA_FIELD_DEF(uint32_t, vecLoop);
TILING_DATA_FIELD_DEF(uint32_t, vecBaseN);
TILING_DATA_FIELD_DEF(uint32_t, vecTailN);
TILING_DATA_FIELD_DEF(uint32_t, vecTailCoreNum);

TILING_DATA_FIELD_DEF(uint32_t, totalBatchSizes);
//

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BidirectionLSTM, BidirectionLSTMTilingData)
REGISTER_TILING_DATA_CLASS(BidirectionLSTMV2, BidirectionLSTMTilingData)

class BidirectionLSTMTiling {
public:
  ge::graphStatus runTiling(gert::TilingContext* context);
protected:
  uint64_t GetDtypeSize(ge::DataType dtype);

  uint64_t GetTilingKey();

  bool GetPlatformInfo(gert::TilingContext* context);

  bool CheckTensorShape(gert::TilingContext* context, gert::Shape& shape, uint64_t ndim, std::vector<uint64_t> dims);

  bool CheckInputShapes(gert::TilingContext* context);

  bool CheckOptionalInputShapes(gert::TilingContext* context);

  bool CheckOutputShapes(gert::TilingContext* context);

  bool CheckInOutShapes(gert::TilingContext* context);

  bool GetCheckAttr(gert::TilingContext* context);

  void GetIntraCoreMMTilingData(uint32_t BaseMNFractalN, uint32_t &BaseMNum, uint32_t &BaseNNum);

  bool ChooseMMStrategy(uint32_t BaseMNumA, uint32_t BaseNNumA, uint32_t BaseMNumB, uint32_t BaseNNumB);

  void GetMMTilingDataBaseMN();

  void GetMMTilingDataBaseK();

  void GetMMTilingData();

  void GetPaddataTilingData();

  void GetCleardataTilingData();

  void GetTransdataTilingData();

  void GetPadAndTransdataTilingData();

  void GetVecTilingData();

  bool SetLaunchInfo(gert::TilingContext* context);

  void _SetTilingDataA();

  void _SetTilingDataB();

  bool SetTilingData(gert::TilingContext* context);

private:
  BidirectionLSTMTilingData tilingData;
  BidirectionLSTMParam _Params;
};

}  // namespace optiling
#endif  // BIDIRECTION_LSTM_TILING_H
