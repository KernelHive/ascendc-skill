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
 * \file lstm_bidir_fp16.h
 * \brief
 */
#ifndef _ASCENDC_LSTM_BIDIR_FP16_H_
#define _ASCENDC_LSTM_BIDIR_FP16_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;

constexpr uint32_t UB_SIZE = 256 * 1024;
constexpr uint32_t L1_SIZE = 1024 * 1024;
constexpr uint32_t L0A_SIZE = 64 * 1024;
constexpr uint32_t L0B_SIZE = 64 * 1024;
constexpr uint32_t L0C_SIZE = 256 * 1024;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t NUM_PER_BLOCK_FLOAT32 = 8;
constexpr uint32_t NUM_PER_BLOCK_FLOAT16 = 16;
constexpr uint32_t NUM_PER_REPEAT_FLOAT32 = 64;
constexpr uint32_t NUM_PER_REPEAT_FLOAT16 = 128;
constexpr uint32_t NUM_PER_FRACTAL_FLOAT16 = 256;
constexpr uint32_t NUM_MM_GROUP = 2;
constexpr uint32_t NUM_OF_GATE = 4;
constexpr uint32_t NONESENSE = 0;
constexpr uint32_t NUMBER_SIXTEEN = 16;
constexpr uint32_t NUMBER_TWO = 2;
constexpr uint32_t NUMBER_THREE = 3;
constexpr uint32_t NUMBER_FOUR = 4;
constexpr uint32_t SPECIAL_INIT_VAL = 210715;
constexpr uint32_t MAX_SEQ = 256;

struct TransInnerParams{
  __aicore__ TransInnerParams(){}
  uint32_t baseN, innerBaseN;
  uint32_t innerLoop, innerTailN;
  uint64_t (*ZNLocalList)[NUM_PER_BLOCK_FLOAT16];
  uint64_t (*NDLocalList)[NUM_PER_BLOCK_FLOAT16];
  uint64_t (*NDLocalListTail)[NUM_PER_BLOCK_FLOAT16];
  DataCopyParams repeatParamsTransND, repeatParamsTransZN;
  DataCopyParams repeatParamsTransNDTail, repeatParamsTransZNTail;
  TransDataTo5HDParams transDataParams, transDataParamsTail;
};

struct PadInnerParams{
  __aicore__ PadInnerParams(){}
  uint32_t baseInN, baseOutN;
  uint32_t innerBaseN, innerBaseM, innerLoop, innerTailN;
  uint64_t mask[2];
  uint32_t srcStep, dstStep;
  uint32_t srcOffset, dstOffset;
  DataCopyParams repeatParamsIn, repeatParamsOut;
  DataCopyParams repeatParamsInTail, repeatParamsOutTail;
};

class LstmBidirFP16 {
 public:
  TPipe pipe;

  __aicore__ inline LstmBidirFP16(){}
  __aicore__ inline void Process();
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR init_h, GM_ADDR init_c,
                                                GM_ADDR w_ih, GM_ADDR w_hh, GM_ADDR b_ih, GM_ADDR b_hh, GM_ADDR w_ih_reverse, GM_ADDR w_hh_reverse,
                                                GM_ADDR b_ih_reverse, GM_ADDR b_hh_reverse, GM_ADDR batch_size, GM_ADDR y, GM_ADDR output_h,
                                                GM_ADDR output_c, GM_ADDR usrWorkspace, const BidirectionLSTMTilingData* __restrict tilingData);
 protected:
  __aicore__ inline void swapValue(uint16_t &a, uint16_t &b);
  __aicore__ inline void InitTransInnerParamSingle(TransInnerParams &innerParams);
  __aicore__ inline void InitPadInnerParamSingle(PadInnerParams &innerParams);
  __aicore__ inline void InitPadInnerParams();
  __aicore__ inline void InitTransInnerParams();
  __aicore__ inline void InitTransLocalLists();
  __aicore__ inline void InitSyncWs();
  __aicore__ inline void InitEventId();
  __aicore__ inline void UpdateGlobalTensors();
  __aicore__ inline void InitWorkspaceTensors(GM_ADDR usrWorkspace);
  __aicore__ inline void InitGlobalTensors(GM_ADDR x, GM_ADDR init_h, GM_ADDR init_c,
                                                GM_ADDR w_ih, GM_ADDR w_hh, GM_ADDR b_ih, GM_ADDR b_hh, GM_ADDR w_ih_reverse, GM_ADDR w_hh_reverse,
                                                GM_ADDR b_ih_reverse, GM_ADDR b_hh_reverse, GM_ADDR y, GM_ADDR output_h,
                                                GM_ADDR output_c);
  __aicore__ inline void InitLocalTensors();
  __aicore__ inline void LstmSyncAll();
  __aicore__ inline void PadAndND2ZN();
  __aicore__ inline void ZN2NDAndUnpad();
  __aicore__ inline void ClearOutput();
  __aicore__ inline void ClearBeforeUnpadCopy(const GlobalTensor<half> &dstGlobal, uint32_t offset, uint32_t loop,
                                                        uint32_t baseN, uint32_t tailN, uint32_t tailCoreNum);
  __aicore__ inline void PadCopyAndCastBias(const GlobalTensor<float> &dstGlobal, const GlobalTensor<half> &srcGlobal,
                                                      uint32_t originBaseN, uint64_t mask, TransInnerParams &innerParams);
  __aicore__ inline void PadCopy(const GlobalTensor<half> &dstGlobal, const GlobalTensor<half> &srcGlobal,
                                          uint32_t loop, uint32_t tailM, uint32_t tailCoreNum,
                                          bool needPad, bool unpad, bool spilt_line, PadInnerParams &innerParams);
  __aicore__ inline void BatchTrans(const GlobalTensor<half> &ZNGlobal, const GlobalTensor<half> &NDGlobal,
                                                  uint32_t dstStep, uint32_t srcStep, uint32_t batchTailM, uint32_t batch, uint32_t loop, uint32_t tailCoreNum, bool spilt_line,
                                                  TransInnerParams &innerParams, bool isND2ZN, bool packed=false, bool isSeq=false);
  __aicore__ inline void BatchTransCast(const GlobalTensor<float> &ZNGlobal, const GlobalTensor<half> &NDGlobal,
                                                  uint32_t dstStep, uint32_t srcStep, uint32_t batchTailM, uint32_t batch, uint32_t loop, uint32_t tailCoreNum,
                                                  TransInnerParams &innerParams, bool isND2ZN);
  __aicore__ inline void ProcessMM(uint32_t tIdx);
  __aicore__ inline void CalCrossStartEnd(uint32_t baseOffset, uint32_t len, uint32_t& cross_start, uint32_t& cross_end);
  __aicore__ inline void ProcessVector(uint32_t tIdx);
  __aicore__ inline void IterateVectorCopy(uint32_t tIdx, uint32_t offset, uint32_t len);
  __aicore__ inline void IterateVectorCross(uint32_t tIdx, uint32_t offset, uint32_t len, uint32_t cross_start, uint32_t cross_end, uint64_t mask[2]);
  __aicore__ inline void TanhFunc(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                              const LocalTensor<float>& buffer);
  __aicore__ inline void SigmoidFunc(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                              const LocalTensor<float>& buffer);
  // GlobalTensor
  GlobalTensor<half> xGm;
  GlobalTensor<half> yGm;
  GlobalTensor<half> outHGm;
  GlobalTensor<half> outCGm;
  GlobalTensor<half> initHGm;
  GlobalTensor<half> initCGm;
  GlobalTensor<half> wIHGm;
  GlobalTensor<half> bIHGm;
  GlobalTensor<half> wHHGm;
  GlobalTensor<half> bHHGm;
  GlobalTensor<half> wIHRevGm;
  GlobalTensor<half> bIHRevGm;
  GlobalTensor<half> wHHRevGm;
  GlobalTensor<half> bHHRevGm;

  GlobalTensor<float> workspaceIGm;
  GlobalTensor<float> workspaceFGm;
  GlobalTensor<float> workspaceJGm;
  GlobalTensor<float> workspaceOGm;
  GlobalTensor<float> workspaceCGm;
  GlobalTensor<float> workspaceMMGm;
  GlobalTensor<float> workspaceBiasGM;
  GlobalTensor<half> workspaceInOutGm;
  GlobalTensor<half> workspaceInGm;
  GlobalTensor<half> workspacePreOutGm;
  GlobalTensor<half> workspaceInitHGm;
  GlobalTensor<half> workspaceOutputHGm;
  GlobalTensor<half> workspaceWihGM;
  GlobalTensor<half> workspaceWihGMSelf;
  GlobalTensor<half> workspaceWhhGM;
  GlobalTensor<half> workspaceWhhGMSelf;
  GlobalTensor<half> workspaceTmpGM;
  GlobalTensor<int32_t> syncGmWs;
  GlobalTensor<int32_t> syncGmWsSelf;

  // LocalTensor
  TBuf<TPosition::VECCALC> UbBuf;
  TBuf<TPosition::TSCM> L1Buf;
  TBuf<TPosition::A2> L0ABuf;
  TBuf<TPosition::B2> L0BBuf;
  TBuf<TPosition::CO1> L0CBuf;
  // for vecCompute
  LocalTensor<float> ubCompute[5];
  LocalTensor<half> ubCompute_;
  // for pad
  LocalTensor<half> ubPad[2];
  // for trans
  LocalTensor<half> ubTransIn[2];
  LocalTensor<half> ubTransOut[2];
  LocalTensor<float> ubTransOut_;
  // for matmul
  LocalTensor<float> ubMatIn;
  LocalTensor<float> ubMatOut;
  LocalTensor<float> ubMatBuff;
  LocalTensor<half> L1MatA;
  LocalTensor<half> L1MatB;
  LocalTensor<half> L0MatA;
  LocalTensor<half> L0MatB;
  LocalTensor<float> L0MatC;
  // for sync
  LocalTensor<int32_t> syncUbWs;
  LocalTensor<int32_t> syncUbWsSelf;

  const BidirectionLSTMTilingData* __restrict tilingData;
  uint32_t block_id;
  uint32_t vecOffset;
  uint32_t vecTailOffset;
  uint32_t clearInOutOffset;
  uint32_t clearInitOffset;

  uint32_t realSingleCoreM;
  uint32_t realSingleCoreN;
  uint32_t matOffsetM;
  uint32_t matOffsetN;
  uint32_t baseM;
  uint32_t baseMMax;
  uint32_t baseTailM;
  uint32_t baseN;
  uint32_t baseTailN;
  uint32_t IHbaseK;
  uint32_t IHbaseTailK;
  uint32_t HHbaseK;
  uint32_t HHbaseTailK;
  uint32_t batchSizes[MAX_SEQ];
  uint32_t batchOffsets[MAX_SEQ];
  const uint8_t _padList[4] = {0, 0, 0, 0};

  uint64_t ZNLocalList[NUMBER_TWO][NUM_PER_BLOCK_FLOAT16];
  uint64_t NDLocalList[NUMBER_TWO][NUM_PER_BLOCK_FLOAT16];
  uint64_t NDLocalListInputTail[NUMBER_TWO][NUM_PER_BLOCK_FLOAT16];
  uint64_t NDLocalListHiddenTail[NUMBER_TWO][NUM_PER_BLOCK_FLOAT16];
  uint64_t NDLocalListBias[1][NUM_PER_BLOCK_FLOAT16];

  event_t eventIdVToMTE3[2];
  event_t eventIdVToMTE2[2];
  event_t eventIdMTE2ToV[2];
  event_t eventIdMTE3ToV[2];
  event_t eventIdMTE2ToMTE3[2];
  event_t eventIdMTE3ToMTE2[2];
  event_t eventIdMTE2ToS;
  event_t eventIdMTE3ToS;
  event_t eventIdSToMTE2;
  event_t eventIdSToMTE3;
  event_t eventIdMToV;
  event_t eventIdVToM;
  event_t eventIdMTE1ToM;
  event_t eventIdMToMTE1;
  event_t eventIdMTE1ToMTE2[2];
  event_t eventIdMTE2ToMTE1[2];

  UnaryRepeatParams unaryParams;
  UnaryRepeatParams unaryParamsH2F;
  UnaryRepeatParams unaryParamsF2H;
  BinaryRepeatParams binaryParams;
  DataCopyParams repeatParamsHalf;
  DataCopyParams repeatParamsFloat;
  DataCopyEnhancedParams enhancedParams;

  TransInnerParams inputND2ZNInnerParams;
  TransInnerParams hiddenND2ZNInnerParams;
  TransInnerParams hiddenZN2NDInnerParams;

  PadInnerParams inputPadInnerParams;
  PadInnerParams hiddenPadInnerParams;
  PadInnerParams hiddenUnPadInnerParams;
};
#endif