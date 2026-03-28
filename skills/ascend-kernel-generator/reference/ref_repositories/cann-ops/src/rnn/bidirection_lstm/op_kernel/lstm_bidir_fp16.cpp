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
 * \file lstm_bidir_fp16.cpp
 * \brief
 */

#include "lstm_bidir_fp16.h"

using namespace AscendC;

__aicore__ inline void LstmBidirFP16::Init(GM_ADDR x, GM_ADDR init_h, GM_ADDR init_c, GM_ADDR w_ih,
                                                GM_ADDR w_hh, GM_ADDR b_ih, GM_ADDR b_hh, GM_ADDR w_ih_reverse, GM_ADDR w_hh_reverse,
                                                GM_ADDR b_ih_reverse, GM_ADDR b_hh_reverse, GM_ADDR batch_size, GM_ADDR y, GM_ADDR output_h,
                                                GM_ADDR output_c, GM_ADDR usrWorkspace, const BidirectionLSTMTilingData* __restrict lstmTiling) {
  block_id = GetBlockIdx();
  tilingData = lstmTiling;

  InitEventId();

  InitGlobalTensors(x, init_h, init_c, w_ih, w_hh, b_ih, b_hh, w_ih_reverse, w_hh_reverse,
                    b_ih_reverse, b_hh_reverse, y, output_h, output_c);

  InitWorkspaceTensors(usrWorkspace);

  InitLocalTensors();

  InitSyncWs();

  InitTransInnerParams();

  InitPadInnerParams();

  // common scalar
  unaryParamsH2F.srcRepStride = HALF_DEFAULT_REPEAT_STRIDE;
  unaryParamsF2H.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;
  enhancedParams.blockMode = BlockMode::BLOCK_MODE_MATRIX;
  // matmul scalar
  matOffsetM = 0;
  matOffsetN = 0;
  realSingleCoreM = tilingData->IHSingleCoreM;
  realSingleCoreN = tilingData->IHSingleCoreN;
  if (block_id % NUM_MM_GROUP != 0) {
    if(tilingData->IHSingleCoreM != tilingData->IHM) {
      realSingleCoreM = tilingData->IHM - tilingData->IHSingleCoreM;
      matOffsetM = tilingData->IHSingleCoreM;
    } else if(tilingData->IHSingleCoreN != tilingData->IHN) {
      realSingleCoreN = tilingData->IHN - tilingData->IHSingleCoreN;
      matOffsetN = tilingData->IHSingleCoreN;
    } else {
      realSingleCoreM = 0;
      realSingleCoreN = 0;
    }
  }
  baseM = realSingleCoreM / tilingData->IHBaseMNum;
  baseTailM = realSingleCoreM % tilingData->IHBaseMNum;
  baseN = realSingleCoreN / tilingData->IHBaseNNum;
  baseTailN = realSingleCoreN % tilingData->IHBaseNNum;
  IHbaseK = tilingData->IHSingleCoreK / tilingData->IHBaseKNum;
  IHbaseTailK = tilingData->IHSingleCoreK % tilingData->IHBaseKNum;
  HHbaseK = tilingData->HHSingleCoreK / tilingData->HHBaseKNum;
  HHbaseTailK = tilingData->HHSingleCoreK % tilingData->HHBaseKNum;

  baseMMax = baseM + (0 < baseTailM ? 1 : 0);
  // clear scalar
  clearInOutOffset = (tilingData->clearInOutLoop * tilingData->padBaseN + tilingData->clearInOutTailN) * block_id
                   + (block_id < tilingData->clearInOutTailCoreNum ? block_id : tilingData->clearInOutTailCoreNum) * NUM_PER_BLOCK_FLOAT16;
  clearInitOffset = (tilingData->clearInitLoop * tilingData->padBaseN + tilingData->clearInitTailN) * block_id
                  + (block_id < tilingData->clearInitTailCoreNum ? block_id : tilingData->clearInitTailCoreNum) * NUM_PER_BLOCK_FLOAT16;
  // vecCompute scalar
  vecOffset = tilingData->vecBaseN * block_id;
  vecTailOffset = tilingData->vecBaseN * tilingData->CoreNum * tilingData->vecLoop + tilingData->vecTailN * block_id
                  + (block_id < tilingData->vecTailCoreNum ? block_id : tilingData->vecTailCoreNum) * NUM_PER_REPEAT_FLOAT16;

  // batch_sizes
  if(tilingData->isSeq) {
    const __gm__ uint32_t *batch_size_GM = (const __gm__ uint32_t *)batch_size;
    bool batchSize_limited = true;
    for(int i = 0; i < tilingData->sequenceLength; i++) {
      batchSizes[i] = batch_size_GM[i];
      batchSize_limited = batchSize_limited && (batchSizes[i]<=tilingData->batchSize);
    }
    assert(batchSize_limited, "batch_size[] elements can not exceed %u\n", tilingData->batchSize);
    if(tilingData->packed) {
      batchOffsets[0] = 0;
      for(int i = 1; i <= tilingData->sequenceLength; i++) {
        batchOffsets[i] = batchOffsets[i-1] + batchSizes[i-1];
      }
      assert(batchOffsets[tilingData->sequenceLength] == tilingData->totalBatchSizes, "sum of batch_size[] %u not equal to x dim[0]:%u\n",batchOffsets[tilingData->sequenceLength],tilingData->totalBatchSizes);
    }
  }
}

__aicore__ inline void LstmBidirFP16::InitTransLocalLists() {
  ZNLocalList[0][0] = reinterpret_cast<uint64_t>(ubTransOut[0].GetPhyAddr());
  ZNLocalList[1][0] = reinterpret_cast<uint64_t>(ubTransOut[1].GetPhyAddr());
  NDLocalList[0][0] = reinterpret_cast<uint64_t>(ubTransIn[0].GetPhyAddr());
  NDLocalList[1][0] = reinterpret_cast<uint64_t>(ubTransIn[1].GetPhyAddr());
  NDLocalListInputTail[0][0] = reinterpret_cast<uint64_t>(ubTransIn[0].GetPhyAddr());
  NDLocalListInputTail[1][0] = reinterpret_cast<uint64_t>(ubTransIn[1].GetPhyAddr());
  NDLocalListHiddenTail[0][0] = reinterpret_cast<uint64_t>(ubTransIn[0].GetPhyAddr());
  NDLocalListHiddenTail[1][0] = reinterpret_cast<uint64_t>(ubTransIn[1].GetPhyAddr());
  NDLocalListBias[0][0] = reinterpret_cast<uint64_t>(ubTransIn[0].GetPhyAddr());
  for(int i = 1;i < NUM_PER_BLOCK_FLOAT16;i++) {
    ZNLocalList[0][i]= ZNLocalList[0][i - 1] + BLOCK_SIZE;
    ZNLocalList[1][i]= ZNLocalList[1][i - 1] + BLOCK_SIZE;
    NDLocalList[0][i] = NDLocalList[0][i - 1] + tilingData->transInnerBaseN * sizeof(half);
    NDLocalList[1][i] = NDLocalList[1][i - 1] + tilingData->transInnerBaseN * sizeof(half);
    NDLocalListInputTail[0][i] = NDLocalListInputTail[0][i - 1] + tilingData->transInputInnerTailN * sizeof(half);
    NDLocalListInputTail[1][i] = NDLocalListInputTail[1][i - 1] + tilingData->transInputInnerTailN * sizeof(half);
    NDLocalListHiddenTail[0][i] = NDLocalListHiddenTail[0][i - 1] + tilingData->transHiddenInnerTailN * sizeof(half);
    NDLocalListHiddenTail[1][i] = NDLocalListHiddenTail[1][i - 1] + tilingData->transHiddenInnerTailN * sizeof(half);
    NDLocalListBias[0][i] = NDLocalListBias[0][0];
  }
}

__aicore__ inline void LstmBidirFP16::InitTransInnerParamSingle(TransInnerParams &innerParams) {
  innerParams.ZNLocalList = ZNLocalList;
  innerParams.NDLocalList = NDLocalList;
  innerParams.repeatParamsTransND = {NUM_PER_BLOCK_FLOAT16, static_cast<uint16_t>(innerParams.innerBaseN / NUM_PER_BLOCK_FLOAT16),
                                              static_cast<uint16_t>((innerParams.baseN - innerParams.innerBaseN) / NUM_PER_BLOCK_FLOAT16), DEFAULT_DATA_COPY_STRIDE};
  innerParams.repeatParamsTransZN = {1, static_cast<uint16_t>(innerParams.innerBaseN),
                                              DEFAULT_DATA_COPY_STRIDE, DEFAULT_DATA_COPY_STRIDE};
  innerParams.repeatParamsTransNDTail = {NUM_PER_BLOCK_FLOAT16, static_cast<uint16_t>(innerParams.innerTailN / NUM_PER_BLOCK_FLOAT16),
                                                  static_cast<uint16_t>((innerParams.baseN - innerParams.innerTailN) / NUM_PER_BLOCK_FLOAT16), DEFAULT_DATA_COPY_STRIDE};
  innerParams.repeatParamsTransZNTail = {1, static_cast<uint16_t>(innerParams.innerTailN),
                                                  DEFAULT_DATA_COPY_STRIDE, DEFAULT_DATA_COPY_STRIDE};
  innerParams.transDataParams = {false, false, static_cast<uint8_t>(innerParams.innerBaseN / NUM_PER_BLOCK_FLOAT16),
                                          NUM_PER_BLOCK_FLOAT16, DEFAULT_BLK_STRIDE};
  innerParams.transDataParamsTail = {false, false, static_cast<uint8_t>(innerParams.innerTailN / NUM_PER_BLOCK_FLOAT16),
                                              NUM_PER_BLOCK_FLOAT16, DEFAULT_BLK_STRIDE};
  if(innerParams.transDataParamsTail.repeatTimes == 1){
    innerParams.transDataParamsTail.dstRepStride = 0;
    innerParams.transDataParamsTail.srcRepStride = 0;
  }
}

__aicore__ inline void LstmBidirFP16::swapValue(uint16_t &a, uint16_t &b) {
  uint16_t tmp = a;
  a = b;
  b = tmp;
}

__aicore__ inline void LstmBidirFP16::InitTransInnerParams() {
  InitTransLocalLists();
  inputND2ZNInnerParams.baseN = tilingData->inputSizeAligned;
  inputND2ZNInnerParams.innerBaseN = tilingData->transInnerBaseN;
  inputND2ZNInnerParams.innerLoop = tilingData->transInputInnerLoop;
  inputND2ZNInnerParams.innerTailN = tilingData->transInputInnerTailN;
  inputND2ZNInnerParams.NDLocalListTail = NDLocalListInputTail;
  InitTransInnerParamSingle(inputND2ZNInnerParams);

  hiddenND2ZNInnerParams.baseN = tilingData->hiddenSizeAligned;
  hiddenND2ZNInnerParams.innerBaseN = tilingData->transInnerBaseN;
  hiddenND2ZNInnerParams.innerLoop = tilingData->transHiddenInnerLoop;
  hiddenND2ZNInnerParams.innerTailN = tilingData->transHiddenInnerTailN;
  hiddenND2ZNInnerParams.NDLocalListTail = NDLocalListHiddenTail;
  InitTransInnerParamSingle(hiddenND2ZNInnerParams);

  hiddenZN2NDInnerParams = hiddenND2ZNInnerParams;
  swapValue(hiddenZN2NDInnerParams.repeatParamsTransND.srcStride, hiddenZN2NDInnerParams.repeatParamsTransND.dstStride);
  swapValue(hiddenZN2NDInnerParams.repeatParamsTransNDTail.srcStride, hiddenZN2NDInnerParams.repeatParamsTransNDTail.dstStride);
  swapValue(hiddenZN2NDInnerParams.transDataParams.dstRepStride, hiddenZN2NDInnerParams.transDataParams.srcRepStride);
  swapValue(hiddenZN2NDInnerParams.transDataParamsTail.dstRepStride, hiddenZN2NDInnerParams.transDataParamsTail.srcRepStride);
}

__aicore__ inline void LstmBidirFP16::InitPadInnerParamSingle(PadInnerParams &innerParams) {
    innerParams.srcStep = NUM_PER_BLOCK_FLOAT16 * innerParams.baseInN * innerParams.innerBaseM;
    innerParams.dstStep = NUM_PER_BLOCK_FLOAT16 * innerParams.baseOutN * innerParams.innerBaseM;
    innerParams.srcOffset = (block_id * NUM_MM_GROUP) * innerParams.baseInN;
    innerParams.dstOffset = (block_id * NUM_MM_GROUP) * innerParams.baseOutN;
    innerParams.repeatParamsIn = {static_cast<uint16_t>(innerParams.innerBaseM), static_cast<uint16_t>(innerParams.innerBaseN / NUM_PER_BLOCK_FLOAT16),
                                        static_cast<uint16_t>(innerParams.baseInN - innerParams.innerBaseN / NUM_PER_BLOCK_FLOAT16), DEFAULT_DATA_COPY_STRIDE};
    innerParams.repeatParamsOut = {static_cast<uint16_t>(innerParams.innerBaseM), static_cast<uint16_t>(innerParams.innerBaseN / NUM_PER_BLOCK_FLOAT16),
                                        DEFAULT_DATA_COPY_STRIDE, static_cast<uint16_t>(innerParams.baseOutN - innerParams.innerBaseN / NUM_PER_BLOCK_FLOAT16)};
    innerParams.repeatParamsInTail = {static_cast<uint16_t>(innerParams.innerBaseM), static_cast<uint16_t>(innerParams.innerTailN / NUM_PER_BLOCK_FLOAT16),
                                        static_cast<uint16_t>(innerParams.baseInN - innerParams.innerTailN / NUM_PER_BLOCK_FLOAT16), DEFAULT_DATA_COPY_STRIDE};
    innerParams.repeatParamsOutTail = {static_cast<uint16_t>(innerParams.innerBaseM), static_cast<uint16_t>(innerParams.innerTailN / NUM_PER_BLOCK_FLOAT16),
                                        DEFAULT_DATA_COPY_STRIDE, static_cast<uint16_t>(innerParams.baseOutN - innerParams.innerTailN / NUM_PER_BLOCK_FLOAT16)};
}

__aicore__ inline void LstmBidirFP16::InitPadInnerParams() {
  if(tilingData->inputSizeAligned != tilingData->inputSize) {
    inputPadInnerParams.baseInN = tilingData->inputSize;
    inputPadInnerParams.baseOutN = tilingData->inputSizeAligned;
    inputPadInnerParams.innerBaseN = tilingData->padInnerBaseN;
    inputPadInnerParams.innerBaseM = tilingData->padInputBaseM;
    inputPadInnerParams.innerLoop = tilingData->padInputInnerLoop;
    inputPadInnerParams.innerTailN = tilingData->padInputInnerTailN;
    inputPadInnerParams.mask[0] = tilingData->padInputMask;
    inputPadInnerParams.mask[1] = 0;
    InitPadInnerParamSingle(inputPadInnerParams);
  }
  if(tilingData->hiddenSizeAligned != tilingData->hiddenSize) {
    hiddenPadInnerParams.baseInN = tilingData->hiddenSize;
    hiddenPadInnerParams.baseOutN = tilingData->hiddenSizeAligned;
    hiddenPadInnerParams.innerBaseN = tilingData->padInnerBaseN;
    hiddenPadInnerParams.innerBaseM = tilingData->padHiddenBaseM;
    hiddenPadInnerParams.innerLoop = tilingData->padHiddenInnerLoop;
    hiddenPadInnerParams.innerTailN = tilingData->padHiddenInnerTailN;
    hiddenPadInnerParams.mask[0] = tilingData->padHiddenMask;
    hiddenPadInnerParams.mask[1] = 0;
    InitPadInnerParamSingle(hiddenPadInnerParams);

    hiddenUnPadInnerParams.baseInN = tilingData->hiddenSizeAligned;
    hiddenUnPadInnerParams.baseOutN = tilingData->hiddenSize;
    hiddenUnPadInnerParams.innerBaseN = tilingData->padInnerBaseN;
    hiddenUnPadInnerParams.innerBaseM = tilingData->padHiddenBaseM;
    hiddenUnPadInnerParams.innerLoop = tilingData->padHiddenInnerLoop;
    hiddenUnPadInnerParams.innerTailN = tilingData->padHiddenInnerTailN;
    hiddenUnPadInnerParams.mask[0] = tilingData->padHiddenMask;
    hiddenUnPadInnerParams.mask[1] = 0;
    InitPadInnerParamSingle(hiddenUnPadInnerParams);
  }
}

__aicore__ inline void LstmBidirFP16::InitEventId(){
  eventIdVToMTE2[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
  eventIdVToMTE2[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
  eventIdVToMTE3[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
  eventIdVToMTE3[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
  eventIdMTE2ToV[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
  eventIdMTE2ToV[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
  eventIdMTE3ToV[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
  eventIdMTE3ToV[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
  eventIdMTE2ToMTE3[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>());
  eventIdMTE2ToMTE3[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>());
  eventIdMTE3ToMTE2[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
  eventIdMTE3ToMTE2[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
  eventIdMTE2ToS = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>());
  eventIdMTE3ToS = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_S>());
  eventIdSToMTE3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::S_MTE3>());
  eventIdSToMTE2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::S_MTE2>());
  eventIdMToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_V>());
  eventIdVToM = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_M>());
  eventIdMTE1ToM = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_M>());
  eventIdMToMTE1 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>());
  eventIdMTE1ToMTE2[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>());
  eventIdMTE1ToMTE2[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>());
  eventIdMTE2ToMTE1[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>());
  eventIdMTE2ToMTE1[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>());
}

__aicore__ inline void LstmBidirFP16::InitWorkspaceTensors(GM_ADDR usrWorkspace) {
  // workspace
  GM_ADDR usrWorkspace_ = usrWorkspace;

  workspaceMMGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(usrWorkspace + block_id / NUM_MM_GROUP * tilingData->workspaceGateN * sizeof(float)),
                                tilingData->workspaceGateN);

  workspaceIGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(usrWorkspace_),
                                tilingData->workspaceGateN);
  usrWorkspace_ += tilingData->workspaceGateN * sizeof(float);
  workspaceFGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(usrWorkspace_),
                                tilingData->workspaceGateN);
  usrWorkspace_ += tilingData->workspaceGateN * sizeof(float);
  workspaceJGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(usrWorkspace_),
                                tilingData->workspaceGateN);
  usrWorkspace_ += tilingData->workspaceGateN * sizeof(float);
  workspaceOGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(usrWorkspace_),
                                tilingData->workspaceGateN);
  usrWorkspace_ += tilingData->workspaceGateN * sizeof(float);
  workspaceCGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(usrWorkspace_),
                                tilingData->workspaceGateN);
  usrWorkspace_ += tilingData->workspaceGateN * sizeof(float);

  workspaceInOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(usrWorkspace_), tilingData->workspaceInOutN);
  workspaceInGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(usrWorkspace_ + tilingData->workspaceInOutStep * sizeof(half)), tilingData->sequenceLength * tilingData->workspaceInOutStep);
  workspacePreOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(usrWorkspace_), tilingData->sequenceLength * tilingData->workspaceInOutStep);
  workspaceInitHGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(usrWorkspace_), tilingData->workspaceInOutStep);
  workspaceOutputHGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(usrWorkspace_ + tilingData->sequenceLength * tilingData->workspaceInOutStep * sizeof(half)), tilingData->workspaceInOutStep);
  usrWorkspace_ += tilingData->workspaceInOutN * sizeof(half);
  workspaceWihGM.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(usrWorkspace_), NUM_OF_GATE * tilingData->workspaceWihN);
  workspaceWihGMSelf.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(usrWorkspace_ + block_id / NUM_MM_GROUP * tilingData->workspaceWihN * sizeof(half)), tilingData->workspaceWihN);
  usrWorkspace_ += NUM_OF_GATE * tilingData->workspaceWihN * sizeof(half);
  workspaceWhhGM.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(usrWorkspace_), NUM_OF_GATE * tilingData->workspaceWhhN);
  workspaceWhhGMSelf.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(usrWorkspace_ + block_id / NUM_MM_GROUP * tilingData->workspaceWhhN * sizeof(half)), tilingData->workspaceWhhN);
  usrWorkspace_ += NUM_OF_GATE * tilingData->workspaceWhhN * sizeof(half);
  if(tilingData->isBias != 0) {
    workspaceBiasGM.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(usrWorkspace_ + block_id / NUM_MM_GROUP * tilingData->workspaceBN * sizeof(float)), tilingData->workspaceBN);
    usrWorkspace_ += NUM_OF_GATE * tilingData->workspaceBN * sizeof(float);
  }
  workspaceTmpGM.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(usrWorkspace_), tilingData->workspaceTmpN);
  usrWorkspace_ += tilingData->workspaceTmpN * sizeof(half);

  syncGmWs.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(usrWorkspace_),
                              tilingData->CoreNum * NUM_PER_BLOCK_FLOAT32);
  syncGmWsSelf.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(usrWorkspace_ + block_id * NUM_PER_BLOCK_FLOAT32 * sizeof(float)),
                              NUM_PER_BLOCK_FLOAT32);
}

__aicore__ inline void LstmBidirFP16::InitGlobalTensors(GM_ADDR x, GM_ADDR init_h, GM_ADDR init_c,
                                                GM_ADDR w_ih, GM_ADDR w_hh, GM_ADDR b_ih, GM_ADDR b_hh, GM_ADDR w_ih_reverse, GM_ADDR w_hh_reverse,
                                                GM_ADDR b_ih_reverse, GM_ADDR b_hh_reverse, GM_ADDR y, GM_ADDR output_h,
                                                GM_ADDR output_c) {
  uint32_t BHSize = tilingData->batchSize * tilingData->hiddenSize;
  uint32_t BISize = tilingData->batchSize * tilingData->inputSize;
  uint32_t HHSize = tilingData->hiddenSize * tilingData->hiddenSize;
  uint32_t IHSize = tilingData->inputSize * tilingData->hiddenSize;
  uint32_t bidirec_coeff = tilingData->bidirection != 0 ? NUMBER_TWO : 1;

  xGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x), tilingData->sequenceLength * BISize);
  yGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(y), bidirec_coeff * tilingData->sequenceLength * BHSize);
  outHGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output_h), bidirec_coeff * BHSize);
  outCGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output_c), bidirec_coeff * BHSize);
  initHGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(init_h), bidirec_coeff * BHSize);
  initCGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(init_c), bidirec_coeff * BHSize);

  wIHGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(w_ih), NUM_OF_GATE * IHSize);
  wHHGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(w_hh), NUM_OF_GATE * HHSize);

  if(tilingData->isBias != 0) {
    bIHGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(b_ih + block_id / NUM_MM_GROUP * tilingData->hiddenSize * sizeof(half)), tilingData->hiddenSize);
    bHHGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(b_hh + block_id / NUM_MM_GROUP * tilingData->hiddenSize * sizeof(half)), tilingData->hiddenSize);
  }

  if(tilingData->bidirection != 0){
    wIHRevGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(w_ih_reverse), NUM_OF_GATE * IHSize);
    wHHRevGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(w_hh_reverse), NUM_OF_GATE * HHSize);
    if(tilingData->isBias != 0) {
      bIHRevGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(b_ih_reverse + block_id / NUM_MM_GROUP * tilingData->hiddenSize * sizeof(half)), tilingData->hiddenSize);
      bHHRevGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(b_hh_reverse + block_id / NUM_MM_GROUP * tilingData->hiddenSize * sizeof(half)), tilingData->hiddenSize);
    }
  }
}

__aicore__ inline void LstmBidirFP16::InitLocalTensors() {
  pipe.InitBuffer(UbBuf, UB_SIZE);
  LocalTensor<uint8_t> tmp = UbBuf.Get<uint8_t>();

  syncUbWs = tmp.ReinterpretCast<int32_t>();
  syncUbWsSelf = syncUbWs[block_id * NUM_PER_BLOCK_FLOAT32];

  ubPad[0] = tmp.ReinterpretCast<half>();
  ubPad[1] = ubPad[0][tilingData->padBaseN];

  ubTransIn[0] = tmp.ReinterpretCast<half>();
  ubTransOut[0] = ubTransIn[0][tilingData->transBaseN];
  ubTransOut_ = ubTransOut[0][tilingData->transBaseN].ReinterpretCast<float>();
  ubTransIn[1] = ubTransOut[0][tilingData->transBaseN];
  ubTransOut[1] = ubTransIn[1][tilingData->transBaseN];

  ubCompute[0] = tmp.ReinterpretCast<float>();
  ubCompute[1] = ubCompute[0][tilingData->vecBaseN];
  ubCompute[NUMBER_TWO] = ubCompute[1][tilingData->vecBaseN];
  ubCompute[NUMBER_THREE] = ubCompute[NUMBER_TWO][tilingData->vecBaseN];
  ubCompute[NUMBER_FOUR] = ubCompute[NUMBER_THREE][tilingData->vecBaseN];
  ubCompute_ = ubCompute[NUMBER_FOUR][tilingData->vecBaseN].ReinterpretCast<half>();

  ubMatOut = tmp.ReinterpretCast<float>();
  if(tilingData->activeMat != 0) {
    ubMatBuff = ubMatOut[tilingData->matUBBaseN];
    if(tilingData->isBias != 0) {
      ubMatIn = ubMatBuff[tilingData->matUBBaseN];
    }
  }

  pipe.InitBuffer(L1Buf, L1_SIZE);
  L1MatA = L1Buf.Get<half>();
  L1MatB = L1MatA[tilingData->l1BaseN];

  pipe.InitBuffer(L0ABuf, L0A_SIZE);
  L0MatA = L0ABuf.Get<half>();
  pipe.InitBuffer(L0BBuf, L0B_SIZE);
  L0MatB = L0BBuf.Get<half>();
  pipe.InitBuffer(L0CBuf, L0C_SIZE);
  L0MatC = L0CBuf.Get<float>();
}

__aicore__ inline void LstmBidirFP16::InitSyncWs() {
  Duplicate<int32_t>(syncUbWsSelf, SPECIAL_INIT_VAL, NUM_PER_BLOCK_FLOAT32);
  SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3[0]);
  WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3[0]);
  DataCopy<int32_t>(syncGmWsSelf, syncUbWsSelf, NUM_PER_BLOCK_FLOAT32);
  SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
  WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
}

__aicore__ inline void LstmBidirFP16::ProcessMM(uint32_t tIdx) {
  uint32_t realBaseM, realIHBaseK, realHHBaseK, realBaseN;
  uint32_t offsetM, offsetIHK, offsetHHK, offsetN = matOffsetN;
  DataCopyParams repeatParams;
  LoadData2dParams loadData2D;
  loadData2D.srcStride = 1;
  loadData2D.ifTranspose = true;
  LoadData3DParamsV2Pro loadData3DV2;
  MmadParams mmadParams;

  uint32_t IHBaseMNum_ = tilingData->IHBaseMNum;
  if(tilingData->isSeq) {
    uint32_t IHM_ = (batchSizes[tIdx] + NUM_PER_BLOCK_FLOAT16 - 1) / NUM_PER_BLOCK_FLOAT16;
    uint32_t IHSingleCoreM_ = (tilingData->IHM == tilingData->IHSingleCoreM) ? IHM_ : (IHM_ + NUM_MM_GROUP - 1) / NUM_MM_GROUP;
    matOffsetM = 0;
    realSingleCoreM = IHSingleCoreM_;
    if (block_id % NUM_MM_GROUP != 0) {
      if(IHSingleCoreM_ != IHM_) {
        realSingleCoreM = IHM_ - IHSingleCoreM_;
        matOffsetM = IHSingleCoreM_;
      } else if(tilingData->IHSingleCoreN == tilingData->IHN) {
        realSingleCoreM = 0;
      }
    }
    IHBaseMNum_ = (IHBaseMNum_ * IHM_ + tilingData->IHM - 1) / tilingData->IHM;
    baseM = realSingleCoreM / IHBaseMNum_;
    baseTailM = realSingleCoreM % IHBaseMNum_;
    baseMMax = baseM + (0 < baseTailM ? 1 : 0);
  }

  if(realSingleCoreM == 0 || realSingleCoreN == 0) {
    return;
  }
  if(tilingData->activeMat == 0) {
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
  } else {
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
    SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
  }
  SetFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[0]);
  SetFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[1]);
  for(int i = 0;i < tilingData->IHBaseNNum; i++) {
    offsetM = matOffsetM;
    realBaseN = baseN + (i < baseTailN ? 1 : 0);
    mmadParams.m = realBaseN * NUM_PER_BLOCK_FLOAT16;
    if(tilingData->activeMat != 0 && tilingData->isBias != 0 && i == 0) {
      repeatParams.blockCount = 1;
      repeatParams.blockLen = realBaseN * NUM_PER_FRACTAL_FLOAT16 / NUM_PER_BLOCK_FLOAT32;
      repeatParams.srcStride = (tilingData->IHN - realBaseN) * NUM_PER_FRACTAL_FLOAT16 / NUM_PER_BLOCK_FLOAT32;
      repeatParams.dstStride = DEFAULT_DATA_COPY_STRIDE;
      WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
      DataCopy<float>(ubMatIn, workspaceBiasGM[offsetN * NUM_PER_FRACTAL_FLOAT16], repeatParams);
      SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
      WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
      uint32_t offset_ = realBaseN * NUM_PER_FRACTAL_FLOAT16;
      uint32_t offset = offset_;
      repeatParamsFloat.blockLen = realBaseN * NUM_PER_FRACTAL_FLOAT16 / NUM_PER_BLOCK_FLOAT32;
      for(int x = 1;x < baseMMax; x++) {
        DataCopy<float>(ubMatIn[offset], ubMatIn, repeatParamsFloat);
        offset += offset_;
      }
      PipeBarrier<PIPE_V>();
      repeatParamsFloat.blockLen = realBaseN * baseMMax;
      DataCopy<float>(L0MatC, ubMatIn, repeatParamsFloat, enhancedParams);
      SetFlag<HardEvent::V_M>(eventIdVToM);
      WaitFlag<HardEvent::V_M>(eventIdVToM);
      if(1 == IHBaseMNum_) {
        SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
      }
    }
    for(int j = 0;j < IHBaseMNum_; j++) {
      realBaseM = baseM + (j < baseTailM ? 1 : 0);
      mmadParams.n = realBaseM * NUM_PER_BLOCK_FLOAT16;
      loadData3DV2.channelSize = realBaseM * NUM_PER_BLOCK_FLOAT16;
      repeatParamsFloat.blockLen = realBaseN * realBaseM;
      if(tilingData->isBias != 0) {
        if(tilingData->activeMat == 0) {
          WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
          repeatParamsFloat.blockLen = realBaseN * NUM_PER_FRACTAL_FLOAT16 / NUM_PER_BLOCK_FLOAT32;
          DataCopy<float>(ubMatOut, workspaceBiasGM[offsetN * NUM_PER_FRACTAL_FLOAT16], repeatParamsFloat);
          SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
          WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
          uint32_t offset_ = realBaseN * NUM_PER_FRACTAL_FLOAT16;
          uint32_t offset = offset_;
          for(int x = 1;x < realBaseM; x++) {
            DataCopy<float>(ubMatOut[offset], ubMatOut, repeatParamsFloat);
            offset += offset_;
          }
          PipeBarrier<PIPE_V>();
          repeatParamsFloat.blockLen = realBaseN * realBaseM;
          DataCopy<float>(L0MatC, ubMatOut, repeatParamsFloat, enhancedParams);
          SetFlag<HardEvent::V_M>(eventIdVToM);
          WaitFlag<HardEvent::V_M>(eventIdVToM);
        }
        mmadParams.cmatrixInitVal = false;
      } else {
        mmadParams.cmatrixInitVal = true;
      }

      offsetIHK = 0;
      for(int k = 0;k < tilingData->IHBaseKNum; k++){
        realIHBaseK = IHbaseK + (k < IHbaseTailK ? 1 : 0);
        mmadParams.k = realIHBaseK * NUM_PER_BLOCK_FLOAT16;

        WaitFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[0]);
        repeatParams.blockCount = realBaseN;
        repeatParams.blockLen = realIHBaseK * NUM_PER_BLOCK_FLOAT16;
        repeatParams.srcStride = (tilingData->IHK - realIHBaseK) * NUM_PER_BLOCK_FLOAT16;
        repeatParams.dstStride = DEFAULT_DATA_COPY_STRIDE;
        DataCopy<half>(L1MatA, workspaceWihGMSelf[(offsetN * tilingData->IHK + offsetIHK) * NUM_PER_FRACTAL_FLOAT16], repeatParams);
        SetFlag<HardEvent::MTE2_MTE1>(eventIdMTE2ToMTE1[0]);
        WaitFlag<HardEvent::MTE2_MTE1>(eventIdMTE2ToMTE1[0]);
        loadData2D.repeatTimes = realBaseN * realIHBaseK;
        LoadData<half>(L0MatA, L1MatA, loadData2D);
        SetFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[0]);

        WaitFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[1]);
        repeatParams.blockCount = realBaseM;
        DataCopy<half>(L1MatB, workspaceInGm[tIdx * tilingData->workspaceInOutStep + (offsetM * tilingData->IHK + offsetIHK) * NUM_PER_FRACTAL_FLOAT16], repeatParams);
        SetFlag<HardEvent::MTE2_MTE1>(eventIdMTE2ToMTE1[1]);
        WaitFlag<HardEvent::MTE2_MTE1>(eventIdMTE2ToMTE1[1]);
        loadData3DV2.extConfig =((uint64_t)(realIHBaseK * NUM_PER_BLOCK_FLOAT16) << NUMBER_SIXTEEN) | (uint64_t)(realBaseM * NUM_PER_BLOCK_FLOAT16);
        Load3DSetFMatrixCal(1, realIHBaseK * NUM_PER_BLOCK_FLOAT16, _padList);
        LoadData<half>(L0MatB, L1MatB, loadData3DV2);
        SetFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[1]);

        SetFlag<HardEvent::MTE1_M>(eventIdMTE1ToM);
        WaitFlag<HardEvent::MTE1_M>(eventIdMTE1ToM);
        Mmad<float, half, half>(L0MatC, L0MatA, L0MatB, mmadParams);
        SetFlag<HardEvent::M_MTE1>(eventIdMToMTE1);
        WaitFlag<HardEvent::M_MTE1>(eventIdMToMTE1);
        if(k==0) mmadParams.cmatrixInitVal = false;
        offsetIHK += realIHBaseK;
      }
      offsetHHK = 0;
      for(int k = 0;k < tilingData->HHBaseKNum; k++){
        realHHBaseK = HHbaseK + (k < HHbaseTailK ? 1 : 0);
        mmadParams.k = realHHBaseK * NUM_PER_BLOCK_FLOAT16;

        WaitFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[0]);
        repeatParams.blockCount = realBaseN;
        repeatParams.blockLen = realHHBaseK * NUM_PER_BLOCK_FLOAT16;
        repeatParams.srcStride = (tilingData->HHK - realHHBaseK) * NUM_PER_BLOCK_FLOAT16;
        repeatParams.dstStride = DEFAULT_DATA_COPY_STRIDE;
        DataCopy<half>(L1MatA, workspaceWhhGMSelf[(offsetN * tilingData->HHK + offsetHHK) * NUM_PER_FRACTAL_FLOAT16], repeatParams);
        SetFlag<HardEvent::MTE2_MTE1>(eventIdMTE2ToMTE1[0]);
        WaitFlag<HardEvent::MTE2_MTE1>(eventIdMTE2ToMTE1[0]);
        loadData2D.repeatTimes = realBaseN * realHHBaseK;
        LoadData<half>(L0MatA, L1MatA, loadData2D);
        SetFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[0]);

        WaitFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[1]);
        repeatParams.blockCount = realBaseM;
        DataCopy<half>(L1MatB, workspacePreOutGm[tIdx * tilingData->workspaceInOutStep + (offsetM * tilingData->HHK + offsetHHK) * NUM_PER_FRACTAL_FLOAT16], repeatParams);
        SetFlag<HardEvent::MTE2_MTE1>(eventIdMTE2ToMTE1[1]);
        WaitFlag<HardEvent::MTE2_MTE1>(eventIdMTE2ToMTE1[1]);
        loadData3DV2.extConfig =((uint64_t)(realHHBaseK * NUM_PER_BLOCK_FLOAT16) << NUMBER_SIXTEEN) | (uint64_t)(realBaseM * NUM_PER_BLOCK_FLOAT16);
        Load3DSetFMatrixCal(1, realHHBaseK * NUM_PER_BLOCK_FLOAT16, _padList);
        LoadData<half>(L0MatB, L1MatB, loadData3DV2);
        SetFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[1]);

        SetFlag<HardEvent::MTE1_M>(eventIdMTE1ToM);
        WaitFlag<HardEvent::MTE1_M>(eventIdMTE1ToM);

        Mmad<float, half, half>(L0MatC, L0MatA, L0MatB, mmadParams);
        SetFlag<HardEvent::M_MTE1>(eventIdMToMTE1);
        WaitFlag<HardEvent::M_MTE1>(eventIdMToMTE1);
        offsetHHK += realHHBaseK;
      }
      SetFlag<HardEvent::M_V>(eventIdMToV);
      WaitFlag<HardEvent::M_V>(eventIdMToV);
      if(tilingData->activeMat != 0) {
        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
      }
      DataCopy<float>(ubMatOut, L0MatC, repeatParamsFloat, enhancedParams);
      PipeBarrier<PIPE_V>();
      if(tilingData->isBias == 0) {
        SetFlag<HardEvent::V_M>(eventIdVToM);
        WaitFlag<HardEvent::V_M>(eventIdVToM);
      } else if(tilingData->activeMat != 0) {
        if((j + 1) < IHBaseMNum_){
          DataCopy<float>(L0MatC, ubMatIn, repeatParamsFloat, enhancedParams);
          SetFlag<HardEvent::V_M>(eventIdVToM);
          WaitFlag<HardEvent::V_M>(eventIdVToM);
          if((j + 1) == (IHBaseMNum_ - 1)) {
            SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
          }
        } else {
          if((i + 1) < tilingData->IHBaseNNum){
            uint32_t realBaseN_ = baseN + ((i + 1) < baseTailN ? 1 : 0);
            repeatParams.blockCount = 1;
            repeatParams.blockLen = realBaseN_ * NUM_PER_FRACTAL_FLOAT16 / NUM_PER_BLOCK_FLOAT32;
            repeatParams.srcStride = (tilingData->IHN - realBaseN_) * NUM_PER_FRACTAL_FLOAT16 / NUM_PER_BLOCK_FLOAT32;
            repeatParams.dstStride = DEFAULT_DATA_COPY_STRIDE;
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
            DataCopy<float>(ubMatIn, workspaceBiasGM[(offsetN + realBaseN)* NUM_PER_FRACTAL_FLOAT16], repeatParams);
            SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
            WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
            uint32_t offset_ = realBaseN_ * NUM_PER_FRACTAL_FLOAT16;
            uint32_t offset = offset_;
            repeatParamsFloat.blockLen = realBaseN_ * NUM_PER_FRACTAL_FLOAT16 / NUM_PER_BLOCK_FLOAT32;
            for(int x = 1;x < baseMMax; x++) {
                DataCopy<float>(ubMatIn[offset], ubMatIn, repeatParamsFloat);
                offset += offset_;
            }
            PipeBarrier<PIPE_V>();
            repeatParamsFloat.blockLen = realBaseN_ * baseMMax;
            DataCopy<float>(L0MatC, ubMatIn, repeatParamsFloat, enhancedParams);
            SetFlag<HardEvent::V_M>(eventIdVToM);
            WaitFlag<HardEvent::V_M>(eventIdVToM);
            if(1 == IHBaseMNum_) {
                SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
            }
          }
        }
      }
      if(tilingData->activeMat != 0) {
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, realBaseM * realBaseN * NUM_PER_FRACTAL_FLOAT16);
        if(block_id / NUM_MM_GROUP != NUMBER_TWO) {
          SigmoidFunc(ubMatOut, ubMatOut, ubMatBuff);
        } else {
          TanhFunc(ubMatOut, ubMatOut, ubMatBuff);
        }
        SetMaskNorm();
        ResetMask();
      }
      SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3[0]);
      WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3[0]);
      repeatParams.blockCount = realBaseM;
      repeatParams.blockLen = realBaseN * NUM_PER_FRACTAL_FLOAT16 / NUM_PER_BLOCK_FLOAT32;
      repeatParams.srcStride = DEFAULT_DATA_COPY_STRIDE;
      repeatParams.dstStride = (tilingData->IHN - realBaseN) * NUM_PER_FRACTAL_FLOAT16 / NUM_PER_BLOCK_FLOAT32;
      DataCopy<float>(workspaceMMGm[(offsetM * tilingData->IHN + offsetN) * NUM_PER_FRACTAL_FLOAT16], ubMatOut, repeatParams);
      if(tilingData->activeMat != 0) {
        SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
      } else if(tilingData->isBias != 0) {
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
      }
      offsetM += realBaseM;
    }
    offsetN += realBaseN;
  }
  if(tilingData->activeMat != 0) {
  SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
  WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
  WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
  }

  WaitFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[0]);
  WaitFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[1]);
  WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
}

__aicore__ inline void LstmBidirFP16::SigmoidFunc(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                                  const LocalTensor<float>& buffer) {
  Muls<float, false>(dst, src, static_cast<float>(-1.0), MASK_PLACEHOLDER, 1, unaryParams);
  PipeBarrier<PIPE_V>();
  Exp<float, false>(dst, dst, MASK_PLACEHOLDER, 1, unaryParams);
  PipeBarrier<PIPE_V>();
  Adds<float, false>(dst, dst, static_cast<float>(1.0), MASK_PLACEHOLDER, 1, unaryParams);
  Duplicate<float, false>(buffer, static_cast<float>(1.0), MASK_PLACEHOLDER, 1,
      DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
  PipeBarrier<PIPE_V>();
  Div<float, false>(dst, buffer, dst, MASK_PLACEHOLDER, 1, binaryParams);
}

__aicore__ inline void LstmBidirFP16::TanhFunc(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                              const LocalTensor<float>& buffer) {
  Mins<float, false>(buffer, src, FP32_MAX_V2, MASK_PLACEHOLDER, 1, unaryParams);
  PipeBarrier<PIPE_V>();
  Maxs<float, false>(buffer, buffer, FP32_MIN_V2, MASK_PLACEHOLDER, 1, unaryParams);
  PipeBarrier<PIPE_V>();
  Muls<float, false>(buffer, buffer, DOUBLE_X, MASK_PLACEHOLDER, 1, unaryParams);
  PipeBarrier<PIPE_V>();
  Exp<float, false>(buffer, buffer, MASK_PLACEHOLDER, 1, unaryParams);
  PipeBarrier<PIPE_V>();
  Adds<float, false>(dst, buffer, static_cast<float>(-1.0), MASK_PLACEHOLDER, 1, unaryParams);
  PipeBarrier<PIPE_V>();
  Adds<float, false>(buffer, buffer, static_cast<float>(1.0), MASK_PLACEHOLDER, 1, unaryParams);
  PipeBarrier<PIPE_V>();
  Div<float, false>(dst, dst, buffer, MASK_PLACEHOLDER, 1, binaryParams);
}

__aicore__ inline void LstmBidirFP16::IterateVectorCross(uint32_t tIdx, uint32_t offset, uint32_t len, uint32_t cross_start, uint32_t cross_end, uint64_t mask[2]) {
  SetMaskCount();
  SetVectorMask<float, MaskMode::COUNTER>(0, cross_end);
  repeatParamsFloat.blockLen = cross_end / NUM_PER_BLOCK_FLOAT32;
  if(tilingData->activeMat != 0) {
    DataCopy<float>(ubCompute[0], workspaceIGm[offset], repeatParamsFloat);
    DataCopy<float>(ubCompute[1], workspaceJGm[offset], repeatParamsFloat);
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
    DataCopy<float>(ubCompute[NUMBER_TWO], workspaceCGm[offset], repeatParamsFloat);
    DataCopy<float>(ubCompute[NUMBER_THREE], workspaceFGm[offset], repeatParamsFloat);
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[1]);

    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
    Mul<float, false>(ubCompute[0], ubCompute[0], ubCompute[1], MASK_PLACEHOLDER, 1, binaryParams);
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);

    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[1]);
    if(cross_start != 0){
      SetVectorMask<float, MaskMode::COUNTER>(0, cross_start);
      Mul<float, false>(ubCompute[NUMBER_TWO], ubCompute[NUMBER_TWO], ubCompute[NUMBER_THREE], MASK_PLACEHOLDER, 1, binaryParams);
    }
    if(cross_start != cross_end){
      SetMaskNorm();
      Mul<float>(ubCompute[NUMBER_TWO][cross_start], ubCompute[NUMBER_TWO][cross_start], ubCompute[NUMBER_THREE][cross_start], mask, static_cast<uint8_t>((cross_end - cross_start) / NUM_PER_REPEAT_FLOAT32), binaryParams);
      SetMaskCount();
    }

    PipeBarrier<PIPE_V>();
    if(cross_start != 0){
      SetVectorMask<float, MaskMode::COUNTER>(0, cross_start);
      Add<float, false>(ubCompute[NUMBER_TWO], ubCompute[NUMBER_TWO], ubCompute[0], MASK_PLACEHOLDER, 1, binaryParams);
    }
    if(cross_start != cross_end){
      SetMaskNorm();
      Add<float>(ubCompute[NUMBER_TWO][cross_start], ubCompute[NUMBER_TWO][cross_start], ubCompute[0][cross_start], mask, static_cast<uint8_t>((cross_end - cross_start) / NUM_PER_REPEAT_FLOAT32), binaryParams);
      SetMaskCount();
    }

    SetVectorMask<float, MaskMode::COUNTER>(0, cross_end);

    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3[0]);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3[0]);
    DataCopy<float>(workspaceCGm[offset], ubCompute[NUMBER_TWO], repeatParamsFloat);

    PipeBarrier<PIPE_V>();
    TanhFunc(ubCompute[NUMBER_THREE], ubCompute[NUMBER_TWO], ubCompute[NUMBER_FOUR]);

    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
    DataCopy<float>(ubCompute[1], workspaceOGm[offset], repeatParamsFloat);
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);

    PipeBarrier<PIPE_V>();
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
    Mul<float, false>(ubCompute[NUMBER_THREE], ubCompute[NUMBER_THREE], ubCompute[1], MASK_PLACEHOLDER, 1, binaryParams);
  } else {
    DataCopy<float>(ubCompute[0], workspaceIGm[offset], repeatParamsFloat);
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);

    DataCopy<float>(ubCompute[1], workspaceJGm[offset], repeatParamsFloat);
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[1]);

    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
    SigmoidFunc(ubCompute[NUMBER_TWO], ubCompute[0], ubCompute[NUMBER_FOUR]);
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);

    PipeBarrier<PIPE_V>();
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[1]);
    TanhFunc(ubCompute[NUMBER_THREE], ubCompute[1], ubCompute[NUMBER_FOUR]);
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[1]);

    PipeBarrier<PIPE_V>();
    Mul<float, false>(ubCompute[NUMBER_TWO], ubCompute[NUMBER_TWO], ubCompute[NUMBER_THREE], MASK_PLACEHOLDER, 1, binaryParams);

    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
    DataCopy<float>(ubCompute[0], workspaceFGm[offset], repeatParamsFloat);
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);

    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[1]);
    DataCopy<float>(ubCompute[1], workspaceCGm[offset], repeatParamsFloat);
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[1]);

    PipeBarrier<PIPE_V>();
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
    SigmoidFunc(ubCompute[NUMBER_THREE], ubCompute[0], ubCompute[NUMBER_FOUR]);
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);

    PipeBarrier<PIPE_V>();
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[1]);
    if(cross_start != 0){
      SetVectorMask<float, MaskMode::COUNTER>(0, cross_start);
      Mul<float, false>(ubCompute[1], ubCompute[1], ubCompute[NUMBER_THREE], MASK_PLACEHOLDER, 1, binaryParams);
    }
    if(cross_start != cross_end){
      SetMaskNorm();
      Mul<float>(ubCompute[1][cross_start], ubCompute[1][cross_start], ubCompute[NUMBER_THREE][cross_start], mask, static_cast<uint8_t>((cross_end - cross_start) / NUM_PER_REPEAT_FLOAT32), binaryParams);
      SetMaskCount();
    }

    PipeBarrier<PIPE_V>();
    if(cross_start != 0){
      SetVectorMask<float, MaskMode::COUNTER>(0, cross_start);
      Add<float, false>(ubCompute[1], ubCompute[1], ubCompute[NUMBER_TWO], MASK_PLACEHOLDER, 1, binaryParams);
    }
    if(cross_start != cross_end){
      SetMaskNorm();
      Add<float>(ubCompute[1][cross_start], ubCompute[1][cross_start], ubCompute[NUMBER_TWO][cross_start], mask, static_cast<uint8_t>((cross_end - cross_start) / NUM_PER_REPEAT_FLOAT32), binaryParams);
      SetMaskCount();
    }

    SetVectorMask<float, MaskMode::COUNTER>(0, cross_end);

    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3[0]);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3[0]);
    DataCopy<float>(workspaceCGm[offset], ubCompute[1], repeatParamsFloat);

    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
    DataCopy<float>(ubCompute[0], workspaceOGm[offset], repeatParamsFloat);
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);

    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
    SigmoidFunc(ubCompute[NUMBER_THREE], ubCompute[0], ubCompute[NUMBER_FOUR]);

    PipeBarrier<PIPE_V>();
    TanhFunc(ubCompute[NUMBER_TWO], ubCompute[1], ubCompute[NUMBER_FOUR]);

    PipeBarrier<PIPE_V>();
    Mul<float, false>(ubCompute[NUMBER_THREE], ubCompute[NUMBER_THREE], ubCompute[NUMBER_TWO], MASK_PLACEHOLDER, 1, binaryParams);
  }

  if(len > cross_start){
    repeatParamsHalf.blockLen = (len - cross_start) / NUM_PER_BLOCK_FLOAT16;
    DataCopy<half>(ubCompute_[cross_start], workspacePreOutGm[tIdx * tilingData->workspaceInOutStep + offset + cross_start], repeatParamsHalf);
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
  }

  PipeBarrier<PIPE_V>();
  if(cross_start != 0){
    SetVectorMask<float, MaskMode::COUNTER>(0, cross_start);
    Cast<half, float, false>(ubCompute_, ubCompute[NUMBER_THREE], RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParamsF2H);
  }
  if(cross_start != cross_end){
    SetMaskNorm();
    Cast<half, float>(ubCompute_[cross_start], ubCompute[NUMBER_THREE][cross_start], RoundMode::CAST_NONE, mask, static_cast<uint8_t>((cross_end - cross_start) / NUM_PER_REPEAT_FLOAT32), unaryParamsF2H);
  }
  SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3[0]);
  WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3[0]);
  repeatParamsHalf.blockLen = len / NUM_PER_BLOCK_FLOAT16;
  DataCopy<half>(workspaceInGm[tIdx * tilingData->workspaceInOutStep + offset], ubCompute_, repeatParamsHalf);

  SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
  WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);

  SetMaskNorm();
  ResetMask();
}

__aicore__ inline void LstmBidirFP16::IterateVectorCopy(uint32_t tIdx, uint32_t offset, uint32_t len) {
  repeatParamsHalf.blockLen = len / NUM_PER_BLOCK_FLOAT16;
  DataCopy<half>(ubCompute_, workspacePreOutGm[tIdx * tilingData->workspaceInOutStep + offset], repeatParamsHalf);
  SetFlag<HardEvent::MTE2_MTE3>(eventIdMTE2ToMTE3[0]);
  WaitFlag<HardEvent::MTE2_MTE3>(eventIdMTE2ToMTE3[0]);
  DataCopy<half>(workspaceInGm[tIdx * tilingData->workspaceInOutStep + offset], ubCompute_, repeatParamsHalf);
  SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
  WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
}

__aicore__ inline void LstmBidirFP16::CalCrossStartEnd(uint32_t baseOffset, uint32_t len, uint32_t& cross_start, uint32_t& cross_end) {
    uint32_t end = baseOffset + len;
    cross_start = cross_start > baseOffset ? cross_start : baseOffset;
    cross_start = end < cross_start ? end : cross_start;
    cross_start -= baseOffset;
    cross_end = cross_end > baseOffset ? cross_end : baseOffset;
    cross_end = end < cross_end ? end : cross_end;
    cross_end -= baseOffset;
}

__aicore__ inline void LstmBidirFP16::ProcessVector(uint32_t tIdx) {
  uint32_t vecOffset_ = vecOffset;
  if(!tilingData->isSeq) {
    uint64_t mask[2] = {0, 0};
    for(int i = 0; i < tilingData->vecLoop; i++) {
      IterateVectorCross(tIdx, vecOffset_, tilingData->vecBaseN, tilingData->vecBaseN, tilingData->vecBaseN, mask);
      vecOffset_ += tilingData->vecBaseN * tilingData->CoreNum;
    }

    int32_t tail = tilingData->vecTailN + (block_id < tilingData->vecTailCoreNum ? NUM_PER_REPEAT_FLOAT16 : 0);
    if(tail > 0){
      IterateVectorCross(tIdx, vecTailOffset, tail, tail, tail, mask);
    }
  } else {
    uint32_t tail_ = batchSizes[tIdx] % NUM_PER_BLOCK_FLOAT16;
    uint64_t mask[2] = {1, 0};
    mask[0] = (mask[0] << tail_) - 1;
    mask[0] = mask[0] | (mask[0] << NUM_PER_BLOCK_FLOAT16);
    mask[0] = mask[0] | (mask[0] << (NUM_PER_BLOCK_FLOAT16 * NUMBER_TWO));
    uint32_t cross_start = (batchSizes[tIdx] / NUM_PER_BLOCK_FLOAT16 * NUM_PER_BLOCK_FLOAT16) * tilingData->hiddenSizeAligned;
    uint32_t cross_end = ((batchSizes[tIdx] + NUM_PER_BLOCK_FLOAT16 - 1) / NUM_PER_BLOCK_FLOAT16 * NUM_PER_BLOCK_FLOAT16) * tilingData->hiddenSizeAligned;
    for(int i = 0; i < tilingData->vecLoop; i++) {
      uint32_t inner_cross_start = cross_start,inner_cross_end = cross_end;
      CalCrossStartEnd(vecOffset_, tilingData->vecBaseN, inner_cross_start, inner_cross_end);
      if(inner_cross_end == 0) {
        IterateVectorCopy(tIdx, vecOffset_, tilingData->vecBaseN);
      } else {
        IterateVectorCross(tIdx, vecOffset_, tilingData->vecBaseN, inner_cross_start, inner_cross_end, mask);
      }
      vecOffset_ += tilingData->vecBaseN * tilingData->CoreNum;
    }

    int32_t tail = tilingData->vecTailN + (block_id < tilingData->vecTailCoreNum ? NUM_PER_REPEAT_FLOAT16 : 0);
    if(tail > 0){
      uint32_t inner_cross_start = cross_start,inner_cross_end = cross_end;
      CalCrossStartEnd(vecTailOffset, tail, inner_cross_start, inner_cross_end);
      if(inner_cross_end == 0) {
        IterateVectorCopy(tIdx, vecTailOffset, tail);
      } else {
        IterateVectorCross(tIdx, vecTailOffset, tail, inner_cross_start, inner_cross_end, mask);
      }
    }
  }
}

__aicore__ inline void LstmBidirFP16::PadCopyAndCastBias(const GlobalTensor<float> &dstGlobal, const GlobalTensor<half> &srcGlobal,
                                                      uint32_t originBaseN, uint64_t mask_, TransInnerParams &innerParams) {
  innerParams.repeatParamsTransZN.blockLen *= NUM_PER_BLOCK_FLOAT16 / NUM_PER_BLOCK_FLOAT32;
  innerParams.repeatParamsTransZNTail.blockLen *= NUM_PER_BLOCK_FLOAT16 / NUM_PER_BLOCK_FLOAT32;
  innerParams.repeatParamsTransND.blockCount = 1;
  innerParams.repeatParamsTransNDTail.blockCount = 1;

  uint32_t realN;
  uint64_t (*srcLocalList)[NUM_PER_BLOCK_FLOAT16];
  uint64_t (*dstLocalList)[NUM_PER_BLOCK_FLOAT16];
  TransDataTo5HDParams *transDataParams;
  DataCopyParams *repeatParamsTransND, *repeatParamsTransZN;
  dstLocalList = innerParams.ZNLocalList;
  srcLocalList = NDLocalListBias;
  if(innerParams.innerLoop == 1){
    realN = innerParams.innerTailN;
    transDataParams = &innerParams.transDataParamsTail;
    repeatParamsTransND = &innerParams.repeatParamsTransNDTail;
    repeatParamsTransZN = &innerParams.repeatParamsTransZNTail;
  } else {
    realN = innerParams.innerBaseN;
    transDataParams = &innerParams.transDataParams;
    repeatParamsTransND = &innerParams.repeatParamsTransND;
    repeatParamsTransZN = &innerParams.repeatParamsTransZN;
  }
  SetMaskCount();
  uint32_t srcOffset = 0;
  uint32_t dstOffset = 0;
  for(int i = 0;i < innerParams.innerLoop;i++) {
    if(i != 0 && i == (innerParams.innerLoop - 1)) {
      realN = innerParams.innerTailN;
      transDataParams = &innerParams.transDataParamsTail;
      repeatParamsTransND = &innerParams.repeatParamsTransNDTail;
      repeatParamsTransZN = &innerParams.repeatParamsTransZNTail;
    }
    DataCopy<half>(ubTransIn[0], srcGlobal[srcOffset], *repeatParamsTransND);
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
    if(i == (innerParams.innerLoop - 1) && (originBaseN % NUM_PER_BLOCK_FLOAT16) != 0){
      SetMaskNorm();
      uint64_t mask[2] = {mask_, 0};
      Duplicate<half>(ubTransIn[0][realN - NUM_PER_BLOCK_FLOAT16], static_cast<half>(0), mask, 1,
                    DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
      PipeBarrier<PIPE_V>();
      SetMaskCount();
    }
    TransDataTo5HD<half>(dstLocalList[0], srcLocalList[0], *transDataParams);
    PipeBarrier<PIPE_V>();

    SetVectorMask<half, MaskMode::COUNTER>(0, NUM_PER_BLOCK_FLOAT16 * realN);
    Cast<float, half, false>(ubTransOut_, ubTransOut[0], RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParamsH2F);
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3[0]);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3[0]);
    DataCopy<float>(dstGlobal[dstOffset], ubTransOut_, *repeatParamsTransZN);
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
    srcOffset += realN;
    dstOffset += realN * NUM_PER_BLOCK_FLOAT16;
  }
  SetMaskNorm();
  ResetMask();
  innerParams.repeatParamsTransZN.blockLen /= NUM_PER_BLOCK_FLOAT16 / NUM_PER_BLOCK_FLOAT32;
  innerParams.repeatParamsTransZNTail.blockLen /= NUM_PER_BLOCK_FLOAT16 / NUM_PER_BLOCK_FLOAT32;
  innerParams.repeatParamsTransND.blockCount = NUM_PER_BLOCK_FLOAT16;
  innerParams.repeatParamsTransNDTail.blockCount = NUM_PER_BLOCK_FLOAT16;
}

__aicore__ inline void LstmBidirFP16::PadCopy(const GlobalTensor<half> &dstGlobal, const GlobalTensor<half> &srcGlobal,
                                          uint32_t loop, uint32_t tailM, uint32_t tailCoreNum,
                                          bool needPad, bool unpad, bool spilt_line, PadInnerParams &innerParams) {
  if(unpad) {
    SetAtomicAdd<half>();
    if(spilt_line) {
      innerParams.dstStep *= NUMBER_TWO;
      innerParams.dstOffset *= NUMBER_TWO;
      innerParams.repeatParamsOut.dstStride += innerParams.baseOutN;
      innerParams.repeatParamsOutTail.dstStride += innerParams.baseOutN;
    }
  }
  uint32_t srcOffset = innerParams.srcOffset;
  uint32_t dstOffset = innerParams.dstOffset;
  uint32_t realN = innerParams.innerTailN;
  DataCopyParams *repeatParamsIn = &innerParams.repeatParamsInTail;
  DataCopyParams *repeatParamsOut = &innerParams.repeatParamsOutTail;
  if(innerParams.innerLoop != 1){
    realN = innerParams.innerBaseN;
    repeatParamsIn = &innerParams.repeatParamsIn;
    repeatParamsOut = &innerParams.repeatParamsOut;
  }
  SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
  SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[1]);
  for(int32_t i = 0;i <= (NUMBER_TWO * loop + 1);i++) {
    uint32_t second = i % NUMBER_TWO;
    uint32_t realM = innerParams.innerBaseM;
    uint32_t srcOffset_ = srcOffset + innerParams.baseInN * second;
    uint32_t dstOffset_ = dstOffset + innerParams.baseOutN * second * ((unpad && spilt_line) ? NUMBER_TWO : 1);
    if((i / NUMBER_TWO) == loop){
      realM = tailM + ((block_id * NUM_MM_GROUP + second) < tailCoreNum ? 1 : 0);
      if(realM == 0) break;
      innerParams.repeatParamsIn.blockCount = realM;
      innerParams.repeatParamsOut.blockCount = realM;
      innerParams.repeatParamsInTail.blockCount = realM;
      innerParams.repeatParamsOutTail.blockCount = realM;
    }
    for(int j = 0;j < innerParams.innerLoop;j++) {
      uint32_t ping_pong = (i * innerParams.innerLoop + j) % NUMBER_TWO;
      if(j != 0 && j == (innerParams.innerLoop - 1)) {
        realN = innerParams.innerTailN;
        repeatParamsIn = &innerParams.repeatParamsInTail;
        repeatParamsOut = &innerParams.repeatParamsOutTail;
      }
      WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[ping_pong]);
      DataCopy<half>(ubPad[ping_pong], srcGlobal[srcOffset_], *repeatParamsIn);
      if(j == (innerParams.innerLoop - 1) && needPad) {
        SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[ping_pong]);
        WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[ping_pong]);
        Duplicate<half>(ubPad[ping_pong][realN - NUM_PER_BLOCK_FLOAT16], static_cast<half>(0), innerParams.mask, realM,
                    DEFAULT_BLK_STRIDE, realN / NUM_PER_BLOCK_FLOAT16);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3[ping_pong]);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3[ping_pong]);
      } else {
        SetFlag<HardEvent::MTE2_MTE3>(eventIdMTE2ToMTE3[ping_pong]);
        WaitFlag<HardEvent::MTE2_MTE3>(eventIdMTE2ToMTE3[ping_pong]);
      }
      DataCopy<half>(dstGlobal[dstOffset_], ubPad[ping_pong], *repeatParamsOut);
      SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[ping_pong]);

      srcOffset_ += realN;
      dstOffset_ += realN;

      if(j != 0 && j == (innerParams.innerLoop - 1)) {
        realN = innerParams.innerBaseN;
        repeatParamsIn = &innerParams.repeatParamsIn;
        repeatParamsOut = &innerParams.repeatParamsOut;
      }
    }
    srcOffset += second * innerParams.srcStep;
    dstOffset += second * innerParams.dstStep;
  }
  WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
  WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[1]);

  innerParams.repeatParamsIn.blockCount = innerParams.innerBaseM;
  innerParams.repeatParamsOut.blockCount = innerParams.innerBaseM;
  innerParams.repeatParamsInTail.blockCount = innerParams.innerBaseM;
  innerParams.repeatParamsOutTail.blockCount = innerParams.innerBaseM;
  if(unpad) {
    SetAtomicNone();
    if(spilt_line) {
      innerParams.dstStep /= NUMBER_TWO;
      innerParams.dstOffset /= NUMBER_TWO;
      innerParams.repeatParamsOut.dstStride -= innerParams.baseOutN;
      innerParams.repeatParamsOutTail.dstStride -= innerParams.baseOutN;
    }
  }
}

__aicore__ inline void LstmBidirFP16::BatchTrans(const GlobalTensor<half> &ZNGlobal, const GlobalTensor<half> &NDGlobal,
                                                  uint32_t dstStep, uint32_t srcStep, uint32_t batchTailM, uint32_t batch, uint32_t loop, uint32_t tailCoreNum, bool spilt_line,
                                                  TransInnerParams &innerParams, bool isND2ZN, bool packed, bool isSeq) {
  uint32_t offsetM = (loop * block_id + (block_id < tailCoreNum ? block_id : tailCoreNum));
  loop += static_cast<uint32_t>(block_id < tailCoreNum);
  if(loop == 0 || batch == 0) {
    return;
  }
  if(spilt_line) {
    innerParams.repeatParamsTransND.dstStride += innerParams.baseN / NUM_PER_BLOCK_FLOAT16;
    innerParams.repeatParamsTransNDTail.dstStride += innerParams.baseN / NUM_PER_BLOCK_FLOAT16;
  }
  uint32_t realN;
  uint64_t (*ZNLocalList)[NUM_PER_BLOCK_FLOAT16] = innerParams.ZNLocalList;
  uint64_t (*NDLocalList)[NUM_PER_BLOCK_FLOAT16];
  TransDataTo5HDParams *transDataParams;
  DataCopyParams *repeatParamsTransND, *repeatParamsTransZN;
  if(innerParams.innerLoop == 1){
    realN = innerParams.innerTailN;
    NDLocalList = innerParams.NDLocalListTail;
    transDataParams = &innerParams.transDataParamsTail;
    repeatParamsTransND = &innerParams.repeatParamsTransNDTail;
    repeatParamsTransZN = &innerParams.repeatParamsTransZNTail;
  } else {
    realN = innerParams.innerBaseN;
    NDLocalList = innerParams.NDLocalList;
    transDataParams = &innerParams.transDataParams;
    repeatParamsTransND = &innerParams.repeatParamsTransND;
    repeatParamsTransZN = &innerParams.repeatParamsTransZN;
  }
  SetMaskCount();
  SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
  SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[1]);
  SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
  SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[1]);
  for(int i = offsetM;i < (offsetM + loop);i++) {
    uint32_t srcOffset, dstOffset, tailM = 0, tailM_ = 0;
    uint32_t innerOffsetM = (i % batch) * NUM_PER_BLOCK_FLOAT16;
    if(!packed) {
      srcOffset = (i / batch) * srcStep + innerOffsetM * innerParams.baseN;
      dstOffset = (i / batch) * dstStep + innerOffsetM * innerParams.baseN;
      if(((i + 1) % batch) == 0) {
        tailM = batchTailM;
      }
      if(isSeq) {
        if(innerOffsetM >= batchSizes[i / batch]) {
          tailM_ = NUM_PER_BLOCK_FLOAT16;
        } else if((innerOffsetM + NUM_PER_BLOCK_FLOAT16) >= batchSizes[i / batch]) {
          tailM_ = batchSizes[i / batch] % NUM_PER_BLOCK_FLOAT16;
        }
      }
    } else {
      if(innerOffsetM >= batchSizes[i / batch]) {
        if(isND2ZN) {
          tailM = NUM_PER_BLOCK_FLOAT16;
          dstOffset = (i / batch) * dstStep + innerOffsetM * innerParams.baseN;
        } else {
          continue;
        }
      } else {
        uint32_t tmpOffset = (batchOffsets[i / batch] + innerOffsetM) * innerParams.baseN;
        if(isND2ZN) {
          srcOffset = tmpOffset;
          dstOffset = (i / batch) * dstStep + innerOffsetM * innerParams.baseN;
        } else {
          srcOffset = (i / batch) * srcStep + innerOffsetM * innerParams.baseN;
          dstOffset = tmpOffset;
        }
        if((innerOffsetM + NUM_PER_BLOCK_FLOAT16) >= batchSizes[i / batch]) {
          tailM = batchSizes[i / batch] % NUM_PER_BLOCK_FLOAT16;
        }
      }
    }
    dstOffset *= spilt_line ? NUMBER_TWO : 1;
    for(int j = 0;j < innerParams.innerLoop;j++) {
      uint32_t ping_pong = (i * innerParams.innerLoop + j) % NUMBER_TWO;
      if(j != 0 && j == (innerParams.innerLoop - 1)) {
        realN = innerParams.innerTailN;
        NDLocalList = innerParams.NDLocalListTail;
        transDataParams = &innerParams.transDataParamsTail;
        repeatParamsTransND = &innerParams.repeatParamsTransNDTail;
        repeatParamsTransZN = &innerParams.repeatParamsTransZNTail;
      }
      if(isND2ZN) {
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[ping_pong]);
        if(tailM == NUM_PER_BLOCK_FLOAT16) {
          PipeBarrier<PIPE_V>();
          SetVectorMask<half, MaskMode::COUNTER>(0, NUM_PER_BLOCK_FLOAT16 * realN);
          Duplicate<half, false>(ubTransIn[ping_pong], static_cast<half>(0), MASK_PLACEHOLDER, 1,
                      DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
          PipeBarrier<PIPE_V>();
        } else if(tailM != 0) {
          repeatParamsTransND->blockCount = tailM;
          DataCopy<half>(ubTransIn[ping_pong], NDGlobal[srcOffset], *repeatParamsTransND);
          repeatParamsTransND->blockCount = NUM_PER_BLOCK_FLOAT16;
          SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[ping_pong]);
          WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[ping_pong]);
          SetVectorMask<half, MaskMode::COUNTER>(0, (NUM_PER_BLOCK_FLOAT16 - tailM) * realN);
          Duplicate<half, false>(ubTransIn[ping_pong][tailM * realN], static_cast<half>(0), MASK_PLACEHOLDER, 1,
                      DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
          PipeBarrier<PIPE_V>();
        } else {
          DataCopy<half>(ubTransIn[ping_pong], NDGlobal[srcOffset], *repeatParamsTransND);
          SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[ping_pong]);
          WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[ping_pong]);
        }
        TransDataTo5HD<half>(ZNLocalList[ping_pong], NDLocalList[ping_pong], *transDataParams);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3[ping_pong]);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3[ping_pong]);

        DataCopy<half>(ZNGlobal[dstOffset], ubTransOut[ping_pong], *repeatParamsTransZN);
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[ping_pong]);

        srcOffset += realN;
        dstOffset += realN * NUM_PER_BLOCK_FLOAT16;
      } else {
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[ping_pong]);
        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[ping_pong]);
        if(tailM_ == NUM_PER_BLOCK_FLOAT16) {
          SetVectorMask<half, MaskMode::COUNTER>(0, NUM_PER_BLOCK_FLOAT16 * realN);
          Duplicate<half, false>(ubTransIn[ping_pong], static_cast<half>(0), MASK_PLACEHOLDER, 1,
                      DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
        } else {
          DataCopy<half>(ubTransOut[ping_pong], ZNGlobal[srcOffset], *repeatParamsTransZN);
          SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[ping_pong]);
          WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[ping_pong]);
          TransDataTo5HD<half>(NDLocalList[ping_pong], ZNLocalList[ping_pong], *transDataParams);
          if(tailM_ != 0) {
            PipeBarrier<PIPE_V>();
            SetVectorMask<half, MaskMode::COUNTER>(0, (NUM_PER_BLOCK_FLOAT16 - tailM_) * realN);
            Duplicate<half, false>(ubTransIn[ping_pong][tailM_ * realN], static_cast<half>(0), MASK_PLACEHOLDER, 1,
                      DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
          }
        }
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3[ping_pong]);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3[ping_pong]);
        if(tailM) {
          repeatParamsTransND->blockCount = tailM;
          DataCopy<half>(NDGlobal[dstOffset], ubTransIn[ping_pong], *repeatParamsTransND);
          repeatParamsTransND->blockCount = NUM_PER_BLOCK_FLOAT16;
        } else {
          DataCopy<half>(NDGlobal[dstOffset], ubTransIn[ping_pong], *repeatParamsTransND);
        }
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[ping_pong]);
        SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[ping_pong]);

        srcOffset += realN * NUM_PER_BLOCK_FLOAT16;
        dstOffset += realN;
      }
      if(j != 0 && j == (innerParams.innerLoop - 1)) {
        realN = innerParams.innerBaseN;
        NDLocalList = innerParams.NDLocalList;
        transDataParams = &innerParams.transDataParams;
        repeatParamsTransND = &innerParams.repeatParamsTransND;
        repeatParamsTransZN = &innerParams.repeatParamsTransZN;
      }
    }
  }
  WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
  WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[1]);
  WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
  WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[1]);
  SetMaskNorm();
  ResetMask();
  if(spilt_line) {
    innerParams.repeatParamsTransND.dstStride -= innerParams.baseN / NUM_PER_BLOCK_FLOAT16;
    innerParams.repeatParamsTransNDTail.dstStride -= innerParams.baseN / NUM_PER_BLOCK_FLOAT16;
  }
}

__aicore__ inline void LstmBidirFP16::BatchTransCast(const GlobalTensor<float> &ZNGlobal, const GlobalTensor<half> &NDGlobal,
                                                  uint32_t dstStep, uint32_t srcStep, uint32_t batchTailM, uint32_t batch, uint32_t loop, uint32_t tailCoreNum,
                                                  TransInnerParams &innerParams, bool isND2ZN) {
  uint32_t offsetM = (loop * block_id + (block_id < tailCoreNum ? block_id : tailCoreNum));
  loop += static_cast<uint32_t>(block_id < tailCoreNum);
  if(loop == 0 || batch == 0) {
      return;
  }
  innerParams.repeatParamsTransZN.blockLen *= NUM_PER_BLOCK_FLOAT16 / NUM_PER_BLOCK_FLOAT32;
  innerParams.repeatParamsTransZNTail.blockLen *= NUM_PER_BLOCK_FLOAT16 / NUM_PER_BLOCK_FLOAT32;
  uint32_t realN;
  uint64_t (*NDLocalList)[NUM_PER_BLOCK_FLOAT16];
  uint64_t (*ZNLocalList)[NUM_PER_BLOCK_FLOAT16] = innerParams.ZNLocalList;
  TransDataTo5HDParams *transDataParams;
  DataCopyParams *repeatParamsTransND, *repeatParamsTransZN;
  if(innerParams.innerLoop == 1){
    realN = innerParams.innerTailN;
    NDLocalList = innerParams.NDLocalListTail;
    transDataParams = &innerParams.transDataParamsTail;
    repeatParamsTransND = &innerParams.repeatParamsTransNDTail;
    repeatParamsTransZN = &innerParams.repeatParamsTransZNTail;
  } else {
    realN = innerParams.innerBaseN;
    NDLocalList = innerParams.NDLocalList;
    transDataParams = &innerParams.transDataParams;
    repeatParamsTransND = &innerParams.repeatParamsTransND;
    repeatParamsTransZN = &innerParams.repeatParamsTransZN;
  }
  SetMaskCount();
  for(int i = offsetM;i < (offsetM + loop);i++) {
    uint32_t srcOffset = (i / batch) * srcStep + (i % batch) * innerParams.baseN * NUM_PER_BLOCK_FLOAT16;
    uint32_t dstOffset = (i / batch) * dstStep + (i % batch) * innerParams.baseN * NUM_PER_BLOCK_FLOAT16;
    for(int j = 0;j < innerParams.innerLoop;j++) {
      if(j != 0 && j == (innerParams.innerLoop - 1)) {
        realN = innerParams.innerTailN;
        NDLocalList = innerParams.NDLocalListTail;
        transDataParams = &innerParams.transDataParamsTail;
        repeatParamsTransND = &innerParams.repeatParamsTransNDTail;
        repeatParamsTransZN = &innerParams.repeatParamsTransZNTail;
      }
      if(isND2ZN) {
        if(((i + 1) % batch) == 0 && batchTailM != 0){
          repeatParamsTransND->blockCount = batchTailM;
          DataCopy<half>(ubTransIn[0], NDGlobal[srcOffset], *repeatParamsTransND);
          repeatParamsTransND->blockCount = NUM_PER_BLOCK_FLOAT16;
          SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
          WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
          SetVectorMask<half, MaskMode::COUNTER>(0, (NUM_PER_BLOCK_FLOAT16 - batchTailM) * realN);
          Duplicate<half, false>(ubTransIn[0][batchTailM * realN], static_cast<half>(0), MASK_PLACEHOLDER, 1,
                      DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
          PipeBarrier<PIPE_V>();
        } else {
          DataCopy<half>(ubTransIn[0], NDGlobal[srcOffset], *repeatParamsTransND);
          SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
          WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
        }
        TransDataTo5HD<half>(ZNLocalList[0], NDLocalList[0], *transDataParams);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
        SetVectorMask<half, MaskMode::COUNTER>(0, realN * NUM_PER_BLOCK_FLOAT16);
        Cast<float, half, false>(ubTransOut_, ubTransOut[0], RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParamsH2F);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3[0]);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3[0]);

        DataCopy<float>(ZNGlobal[dstOffset], ubTransOut_, *repeatParamsTransZN);
        SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
        srcOffset += realN;
        dstOffset += realN * NUM_PER_BLOCK_FLOAT16;
      } else {
        DataCopy<float>(ubTransOut_, ZNGlobal[srcOffset], *repeatParamsTransZN);
        SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
        WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);

        SetVectorMask<half, MaskMode::COUNTER>(0, realN * NUM_PER_BLOCK_FLOAT16);
        Cast<half, float, false>(ubTransOut[0], ubTransOut_, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParamsF2H);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
        TransDataTo5HD<half>(NDLocalList[0], ZNLocalList[0], *transDataParams);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3[0]);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3[0]);

        if(((i + 1) % batch) == 0 && batchTailM != 0){
          repeatParamsTransND->blockCount = batchTailM;
          DataCopy<half>(NDGlobal[dstOffset], ubTransIn[0], *repeatParamsTransND);
          repeatParamsTransND->blockCount = NUM_PER_BLOCK_FLOAT16;
        } else {
          DataCopy<half>(NDGlobal[dstOffset], ubTransIn[0], *repeatParamsTransND);
        }
        SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);

        srcOffset += realN * NUM_PER_BLOCK_FLOAT16;
        dstOffset += realN;
      }
      if(j != 0 && j == (innerParams.innerLoop - 1)) {
        realN = innerParams.innerBaseN;
        NDLocalList = innerParams.NDLocalList;
        transDataParams = &innerParams.transDataParams;
        repeatParamsTransND = &innerParams.repeatParamsTransND;
        repeatParamsTransZN = &innerParams.repeatParamsTransZN;
      }
    }
  }
  SetMaskNorm();
  ResetMask();
  innerParams.repeatParamsTransZN.blockLen /= NUM_PER_BLOCK_FLOAT16 / NUM_PER_BLOCK_FLOAT32;
  innerParams.repeatParamsTransZNTail.blockLen /= NUM_PER_BLOCK_FLOAT16 / NUM_PER_BLOCK_FLOAT32;
}

__aicore__ inline void LstmBidirFP16::PadAndND2ZN() {
  if(tilingData->inputSizeAligned != tilingData->inputSize) {
    PadCopy(workspaceTmpGM, xGm,
            tilingData->padInputLoop, tilingData->padInputTailM, tilingData->padInOutTailCoreNum,
            false, false, false, inputPadInnerParams);
    LstmSyncAll();
    BatchTrans(workspaceInGm, workspaceTmpGM,
            tilingData->workspaceInOutStep, tilingData->batchSize * tilingData->inputSizeAligned, tilingData->batchSize % NUM_PER_BLOCK_FLOAT16,
            tilingData->transInOutLoopBatch, tilingData->transInOutLoop, tilingData->transInOutTailCoreNum, false,
            inputND2ZNInnerParams, true, tilingData->packed, tilingData->isSeq);
    LstmSyncAll();
    PadCopy(workspaceTmpGM, wIHGm,
            tilingData->padWeightIHLoop, tilingData->padWeightIHTailM, tilingData->padWeightTailCoreNum,
            true, false, false, inputPadInnerParams);
    LstmSyncAll();
    BatchTrans(workspaceWihGM, workspaceTmpGM,
            tilingData->workspaceWihN, tilingData->hiddenSize * tilingData->inputSizeAligned, tilingData->hiddenSize % NUM_PER_BLOCK_FLOAT16,
            tilingData->transWeightLoopBatch, tilingData->transWeightLoop, tilingData->transWeightTailCoreNum, false,
            inputND2ZNInnerParams, true);
  } else {
    BatchTrans(workspaceInGm, xGm,
            tilingData->workspaceInOutStep, tilingData->batchSize * tilingData->inputSizeAligned, tilingData->batchSize % NUM_PER_BLOCK_FLOAT16,
            tilingData->transInOutLoopBatch, tilingData->transInOutLoop, tilingData->transInOutTailCoreNum, false,
            inputND2ZNInnerParams, true, tilingData->packed, tilingData->isSeq);
    BatchTrans(workspaceWihGM, wIHGm,
            tilingData->workspaceWihN, tilingData->hiddenSize * tilingData->inputSizeAligned, tilingData->hiddenSize % NUM_PER_BLOCK_FLOAT16,
            tilingData->transWeightLoopBatch, tilingData->transWeightLoop, tilingData->transWeightTailCoreNum, false,
            inputND2ZNInnerParams, true);
  }
  LstmSyncAll();
  if(tilingData->hiddenSizeAligned != tilingData->hiddenSize) {
    PadCopy(workspaceTmpGM, wHHGm,
            tilingData->padWeightHHLoop, tilingData->padWeightHHTailM, tilingData->padWeightTailCoreNum,
            true, false, false, hiddenPadInnerParams);
    LstmSyncAll();
    BatchTrans(workspaceWhhGM, workspaceTmpGM,
            tilingData->workspaceWhhN, tilingData->hiddenSize * tilingData->hiddenSizeAligned, tilingData->hiddenSize % NUM_PER_BLOCK_FLOAT16,
            tilingData->transWeightLoopBatch, tilingData->transWeightLoop, tilingData->transWeightTailCoreNum, false,
            hiddenND2ZNInnerParams, true);
    LstmSyncAll();
    PadCopy(workspaceTmpGM, initHGm,
            tilingData->padInitLoop, tilingData->padInitTailM, tilingData->padInitTailCoreNum,
            false, false, false, hiddenPadInnerParams);
    LstmSyncAll();
    BatchTrans(workspaceInitHGm, workspaceTmpGM,
            NONESENSE, NONESENSE, tilingData->batchSize % NUM_PER_BLOCK_FLOAT16,
            tilingData->transInOutLoopBatch, tilingData->transInitLoop, tilingData->transInitTailCoreNum, false,
            hiddenND2ZNInnerParams, true);
    LstmSyncAll();
    PadCopy(workspaceTmpGM, initCGm,
            tilingData->padInitLoop, tilingData->padInitTailM, tilingData->padInitTailCoreNum,
            true, false, false, hiddenPadInnerParams);
    LstmSyncAll();
    BatchTransCast(workspaceCGm, workspaceTmpGM,
            NONESENSE, NONESENSE, tilingData->batchSize % NUM_PER_BLOCK_FLOAT16,
            tilingData->transInOutLoopBatch, tilingData->transInitLoop, tilingData->transInitTailCoreNum,
            hiddenND2ZNInnerParams, true);
  } else {
    BatchTrans(workspaceWhhGM, wHHGm,
            tilingData->workspaceWhhN, tilingData->hiddenSize * tilingData->hiddenSizeAligned, tilingData->hiddenSize % NUM_PER_BLOCK_FLOAT16,
            tilingData->transWeightLoopBatch, tilingData->transWeightLoop, tilingData->transWeightTailCoreNum, false,
            hiddenND2ZNInnerParams, true);
    BatchTrans(workspaceInitHGm, initHGm,
            NONESENSE, NONESENSE, tilingData->batchSize % NUM_PER_BLOCK_FLOAT16,
            tilingData->transInOutLoopBatch, tilingData->transInitLoop, tilingData->transInitTailCoreNum, false,
            hiddenND2ZNInnerParams, true);
    BatchTransCast(workspaceCGm, initCGm,
            NONESENSE, NONESENSE, tilingData->batchSize % NUM_PER_BLOCK_FLOAT16,
            tilingData->transInOutLoopBatch, tilingData->transInitLoop, tilingData->transInitTailCoreNum,
            hiddenND2ZNInnerParams, true);
  }
  if(tilingData->isBias != 0 && block_id % NUM_MM_GROUP == 0) {
    PadCopyAndCastBias(workspaceBiasGM, bIHGm, tilingData->hiddenSize, tilingData->padHiddenMask, hiddenND2ZNInnerParams);
    SetAtomicAdd<float>();
    PadCopyAndCastBias(workspaceBiasGM, bHHGm, tilingData->hiddenSize, tilingData->padHiddenMask, hiddenND2ZNInnerParams);
    SetAtomicNone();
  }
  LstmSyncAll();
}

__aicore__ inline void LstmBidirFP16::ClearBeforeUnpadCopy(const GlobalTensor<half> &dstGlobal, uint32_t offset, uint32_t loop,
                                                        uint32_t baseN, uint32_t tailN, uint32_t tailCoreNum) {
  uint32_t tail = tailN;
  if(block_id < tailCoreNum) {
    tail += NUM_PER_BLOCK_FLOAT16;
  }
  SetMaskCount();
  if(loop > 0) {
    SetVectorMask<half, MaskMode::COUNTER>(0, baseN);
  } else if(tail > 0) {
    SetVectorMask<half, MaskMode::COUNTER>(0, tail);
  } else {
    SetMaskNorm();
    ResetMask();
    return;
  }
  Duplicate<half, false>(ubPad[0], static_cast<half>(0), MASK_PLACEHOLDER, 1,
                  DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
  SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3[0]);
  WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3[0]);
  SetMaskNorm();
  ResetMask();

  repeatParamsHalf.blockLen = baseN / NUM_PER_BLOCK_FLOAT16;
  for(int i = 0; i < loop; i++) {
    DataCopy<half>(dstGlobal[offset], ubPad[0], repeatParamsHalf);
    offset += baseN;
  }
  if(tail > 0){
    repeatParamsHalf.blockLen = tail / NUM_PER_BLOCK_FLOAT16;
    DataCopy<half>(dstGlobal[offset], ubPad[0], repeatParamsHalf);
  }
}

__aicore__ inline void LstmBidirFP16::ClearOutput(){
  if(tilingData->hiddenSizeAligned != tilingData->hiddenSize) {
    ClearBeforeUnpadCopy(yGm, clearInOutOffset, tilingData->clearInOutLoop, tilingData->padBaseN, tilingData->clearInOutTailN, tilingData->clearInOutTailCoreNum);
    ClearBeforeUnpadCopy(outHGm, clearInitOffset, tilingData->clearInitLoop, tilingData->padBaseN, tilingData->clearInitTailN, tilingData->clearInitTailCoreNum);
    ClearBeforeUnpadCopy(outCGm, clearInitOffset, tilingData->clearInitLoop, tilingData->padBaseN, tilingData->clearInitTailN, tilingData->clearInitTailCoreNum);
    LstmSyncAll();
  }
}

__aicore__ inline void LstmBidirFP16::ZN2NDAndUnpad() {
  if(tilingData->hiddenSizeAligned != tilingData->hiddenSize) {
    BatchTrans(workspaceInGm, workspaceTmpGM,
            tilingData->batchSize * tilingData->hiddenSizeAligned, tilingData->workspaceInOutStep, tilingData->batchSize % NUM_PER_BLOCK_FLOAT16,
            tilingData->transInOutLoopBatch, tilingData->transInOutLoop, tilingData->transInOutTailCoreNum, false,
            hiddenZN2NDInnerParams, false, tilingData->packed, tilingData->isSeq);
    LstmSyncAll();
    PadCopy(yGm, workspaceTmpGM,
            tilingData->padOutputLoop, tilingData->padOutputTailM, tilingData->padInOutTailCoreNum,
            true, true, tilingData->bidirection != 0, hiddenUnPadInnerParams);
    LstmSyncAll();
    BatchTrans(workspaceOutputHGm, workspaceTmpGM,
            NONESENSE, NONESENSE, tilingData->batchSize % NUM_PER_BLOCK_FLOAT16,
            tilingData->transInOutLoopBatch, tilingData->transInitLoop, tilingData->transInitTailCoreNum, false,
            hiddenZN2NDInnerParams, false);
    LstmSyncAll();
    PadCopy(outHGm, workspaceTmpGM,
            tilingData->padInitLoop, tilingData->padInitTailM, tilingData->padInitTailCoreNum,
            true, true, false, hiddenUnPadInnerParams);
    LstmSyncAll();
    BatchTransCast(workspaceCGm, workspaceTmpGM,
            NONESENSE, NONESENSE, tilingData->batchSize % NUM_PER_BLOCK_FLOAT16,
            tilingData->transInOutLoopBatch, tilingData->transInitLoop, tilingData->transInitTailCoreNum,
            hiddenZN2NDInnerParams, false);
    LstmSyncAll();
    PadCopy(outCGm, workspaceTmpGM,
            tilingData->padInitLoop, tilingData->padInitTailM, tilingData->padInitTailCoreNum,
            true, true, false, hiddenUnPadInnerParams);
  } else {
    BatchTrans(workspaceInGm, yGm,
            tilingData->batchSize * tilingData->hiddenSizeAligned, tilingData->workspaceInOutStep, tilingData->batchSize % NUM_PER_BLOCK_FLOAT16,
            tilingData->transInOutLoopBatch, tilingData->transInOutLoop, tilingData->transInOutTailCoreNum, tilingData->bidirection != 0,
            hiddenZN2NDInnerParams, false, tilingData->packed, tilingData->isSeq);
    BatchTrans(workspaceOutputHGm, outHGm,
            NONESENSE, NONESENSE, tilingData->batchSize % NUM_PER_BLOCK_FLOAT16,
            tilingData->transInOutLoopBatch, tilingData->transInitLoop, tilingData->transInitTailCoreNum, false,
            hiddenZN2NDInnerParams, false);
    BatchTransCast(workspaceCGm, outCGm,
            NONESENSE, NONESENSE, tilingData->batchSize % NUM_PER_BLOCK_FLOAT16,
            tilingData->transInOutLoopBatch, tilingData->transInitLoop, tilingData->transInitTailCoreNum,
            hiddenZN2NDInnerParams, false);
  }
  LstmSyncAll();
}

__aicore__ inline void LstmBidirFP16::LstmSyncAll() {
    PipeBarrier<PIPE_ALL>();

    DataCopy<int32_t>(syncUbWsSelf, syncGmWsSelf, {1, 1, 0, 0});
    SetFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
    int32_t curValue = *(reinterpret_cast<__ubuf__ int32_t*>(syncUbWsSelf.GetPhyAddr())) + 1;
    *(reinterpret_cast<__ubuf__ int32_t*>(syncUbWsSelf.GetPhyAddr())) = curValue;
    SetFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
    DataCopy<int32_t>(syncGmWsSelf, syncUbWsSelf, {1, 1, 0, 0});
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
    while (true) {
        DataCopy<int32_t>(syncUbWs, syncGmWs, {1, (uint16_t)tilingData->CoreNum, 0, 0});
        SetFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
        bool arrival = true;
        for (int32_t i = 0; i < tilingData->CoreNum; i++) {
          int32_t tmp = *(reinterpret_cast<__ubuf__ int32_t*>(syncUbWs.GetPhyAddr()) + i * NUM_PER_BLOCK_FLOAT32);
          if(curValue != tmp && (curValue + 1) != tmp) {
            arrival = false;
            break;
          }
        }
        SetFlag<HardEvent::S_MTE2>(eventIdSToMTE2);
        WaitFlag<HardEvent::S_MTE2>(eventIdSToMTE2);
        if (arrival) {
            break;
        }
    }
}

__aicore__ inline void LstmBidirFP16::UpdateGlobalTensors() {
  wIHGm = wIHRevGm;
  wHHGm = wHHRevGm;
  bIHGm = bIHRevGm;
  bHHGm = bHHRevGm;
  yGm = yGm[tilingData->hiddenSize];
  outHGm = outHGm[tilingData->batchSize * tilingData->hiddenSize];
  outCGm = outCGm[tilingData->batchSize * tilingData->hiddenSize];
  initHGm = initHGm[tilingData->batchSize * tilingData->hiddenSize];
  initCGm = initCGm[tilingData->batchSize * tilingData->hiddenSize];
  workspaceInGm = workspaceInOutGm;
  workspacePreOutGm = workspaceInOutGm[tilingData->workspaceInOutStep];
  workspaceInitHGm = workspaceInOutGm[tilingData->sequenceLength * tilingData->workspaceInOutStep];
  workspaceOutputHGm = workspaceInOutGm;
}

__aicore__ inline void LstmBidirFP16::Process() {
  PadAndND2ZN();
  for(int tIdx = 0; tIdx < tilingData->sequenceLength; tIdx++) {
    ProcessMM(tIdx);
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
    LstmSyncAll();
    ProcessVector(tIdx);
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
    LstmSyncAll();
  }
  ClearOutput();
  ZN2NDAndUnpad();
  if(tilingData->bidirection != 0){
    UpdateGlobalTensors();
    PadAndND2ZN();
    for(int tIdx = tilingData->sequenceLength-1; tIdx >=0; tIdx--) {
      ProcessMM(tIdx);
      SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
      WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
      LstmSyncAll();
      ProcessVector(tIdx);
      SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
      WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
      LstmSyncAll();
    }
    ZN2NDAndUnpad();
  }

  PipeBarrier<PIPE_ALL>();
}