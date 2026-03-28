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
 * \file top_kv3.h
 * \brief
 */
#ifndef TOP_K_V3_H
#define TOP_K_V3_H

#include "kernel_operator.h"

constexpr int ONE_BIT = 1;
constexpr int THREE_BIT = 3;
constexpr int SEVEN_BIT = 7;
constexpr int FIFTEEN_BIT = 15;

#define FP16_NEG_INF ((half)-6.550400e+04f)

using namespace AscendC;

__aicore__ inline int CalcValidBit(uint32_t unsortedProposalNum) {
  if (unsortedProposalNum == 0) {
    return ONE_BIT; // 仅对前一条队列有效
  } else if (unsortedProposalNum == 1) {
    return THREE_BIT; // 仅对前两条队列有效
  } else if (unsortedProposalNum == 2) {
    return SEVEN_BIT; // 仅对前三条队列有效
  } else {
    return FIFTEEN_BIT; // 仅对前四条队列有效
  }
}

constexpr uint32_t FP16_BLK_SIZE = 16;
constexpr uint32_t PROPOSAL_NUM_PER_REP = 16;
constexpr uint32_t PROPOSAL_SIZE = 8;

template <typename T>
class KernelTopKV3 {
public:
  __aicore__ inline KernelTopKV3(TPipe *pipe) {
    Ppipe = pipe;
  }
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR k, GM_ADDR values, GM_ADDR indices, const TopKV3TilingData* tilingData)
  {
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    numRow = tilingData->numRow;
    numCol = tilingData->numCol;
    blockFactor = tilingData->blockFactor;
    rowFactor = tilingData->rowFactor;
    ubFactor = tilingData->ubFactor;
    kValue = tilingData->kValue;
    largest = tilingData->largest;

    if (GetBlockIdx() < GetBlockNum() - 1) {
      this->rowWork = blockFactor;
    } else if (GetBlockIdx() == GetBlockNum() - 1) {
      this->rowWork = numRow - (GetBlockNum() - 1) * blockFactor;
    } else {}
    // get start index for current core, core parallel
    xGm.SetGlobalBuffer((__gm__ T*)x + GetBlockIdx() * blockFactor * numCol);
    valuesGm.SetGlobalBuffer((__gm__ T*)values + GetBlockIdx() * blockFactor * kValue);
    indicesGm.SetGlobalBuffer((__gm__ int32_t*)indices + GetBlockIdx() * blockFactor * kValue);

    // pipe alloc memory to queue, the unit is Bytes
    Ppipe->InitBuffer(inQueueX, 1, ubFactor * sizeof(T));
    Ppipe->InitBuffer(outQueueValues, 1, rowFactor * kValue * sizeof(T));
    Ppipe->InitBuffer(outQueueIndices, 1, rowFactor * kValue * sizeof(int32_t));

    Ppipe->InitBuffer(baseIdxBuf, ubFactor * sizeof(int32_t));
    Ppipe->InitBuffer(idxBuf, ubFactor * sizeof(int32_t));
    Ppipe->InitBuffer(idxHigh16Buf, ubFactor * sizeof(T));
    Ppipe->InitBuffer(idxLow16Buf, ubFactor * sizeof(T));

    Ppipe->InitBuffer(proposalBuf, ubFactor * PROPOSAL_SIZE * sizeof(T));
    pingpongAddrBias = (kValue + 3 * PROPOSAL_NUM_PER_REP) * PROPOSAL_SIZE;
    Ppipe->InitBuffer(proposalTopkDoubleBuf, pingpongAddrBias * sizeof(T) * 2);
    Ppipe->InitBuffer(proposalOutBuf, rowFactor * kValue * PROPOSAL_SIZE * sizeof(T));

    iOuterRepTimes = DivCeil(rowWork, rowFactor);
    rowTail = rowWork - (iOuterRepTimes - 1) * rowFactor;
    jRepTimes = DivCeil(numCol, ubFactor);
    colTail = numCol - (jRepTimes - 1) * ubFactor;
  }

  __aicore__ inline void Process()
  {
    WriteBaseIdxBuf();
    for (uint32_t iOuter = 0; iOuter < iOuterRepTimes; iOuter++) {
      uint32_t calcRowNum = (iOuter == iOuterRepTimes - 1) ? rowTail : rowFactor;
      SubProcess(iOuter, calcRowNum);
    }
  }

  __aicore__ inline void SubProcess(uint32_t iOuter, uint32_t calcRowNum)
  {
    for (uint32_t iInner = 0; iInner < calcRowNum; iInner++) {
      pingpongIdx = 0;
      for (uint32_t j = 0; j < jRepTimes; j++) {
        uint32_t calcColNum = (j == jRepTimes - 1) ? colTail : ubFactor;
        WriteIdxBuf(j, calcColNum);
        EncodeIdxToProposal(calcColNum);
        CopyIn((iOuter * rowFactor + iInner) * numCol + j * ubFactor, calcColNum);
        Compute(iInner, j, calcColNum);
      }
    }
    DecodeValuesFromProposal(calcRowNum);
    CopyOutValues(iOuter * rowFactor * kValue, calcRowNum);

    DecodeIdxFromProposal(calcRowNum);
    CopyOutIndices(iOuter * rowFactor * kValue, calcRowNum);
  }

private:
  __aicore__ inline void WriteBaseIdxBuf()
  {
    LocalTensor<int32_t> baseIdxLocal = baseIdxBuf.Get<int32_t>();
    CreateVecIndex(baseIdxLocal, (int32_t)0, ubFactor);
    AscendC::PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void WriteIdxBuf(uint32_t j, uint32_t calcColNum)
  {
    LocalTensor<int32_t> baseIdxLocal = baseIdxBuf.Get<int32_t>();
    LocalTensor<int32_t> idxLocal = idxBuf.Get<int32_t>();
    Adds(idxLocal, baseIdxLocal, static_cast<int32_t>(j * ubFactor), calcColNum);
    AscendC::PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void EncodeIdxToProposal(uint32_t calcColNum)
  {
    LocalTensor<T> idxFp16Local = idxBuf.Get<T>(); // 以float16格式读取
    LocalTensor<T> idxLow16Local = idxLow16Buf.Get<T>();
    LocalTensor<T> idxHigh16Local = idxHigh16Buf.Get<T>();

    // normal mode
    uint32_t mask = 0; // normal模式下mask需要设置为0
    uint16_t gatherRepeat = static_cast<uint16_t>(DivCeil(calcColNum * 2, (uint32_t)128)); // 处理量翻倍
    uint64_t rsvdCntLow16 = 0;  //用于保存筛选后保留下来的元素个数
    uint8_t Low16Pattern = 1; // 每两个元素取第一个元素
    GatherMask(idxLow16Local, idxFp16Local, Low16Pattern, false, mask, {1, gatherRepeat, 8, 0}, rsvdCntLow16);
    uint64_t rsvdCntHigh16 = 0; // 用于保存筛选后保留下来的元素个数
    uint8_t High16Pattern = 2; // 每两个元素取第二个元素
    GatherMask(idxHigh16Local, idxFp16Local, High16Pattern, false, mask, {1, gatherRepeat, 8, 0}, rsvdCntHigh16);
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<T> proposalLocal = proposalBuf.Get<T>();
    uint32_t proposalRepeat = DivCeil(calcColNum, 16);
    ProposalConcat(proposalLocal, idxLow16Local, proposalRepeat, 0);
    AscendC::PipeBarrier<PIPE_V>();
    ProposalConcat(proposalLocal, idxHigh16Local, proposalRepeat, 1);
    AscendC::PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void DecodeValuesFromProposal(uint32_t calcRowNum)
  {
    LocalTensor<T> valuesLocal = outQueueValues.AllocTensor<T>();
    LocalTensor<T> proposalOutLocal = proposalOutBuf.Get<T>();
    ProposalExtract(valuesLocal, proposalOutLocal, DivCeil(calcRowNum * kValue, PROPOSAL_NUM_PER_REP), 4); // 从score位置取出
    AscendC::PipeBarrier<PIPE_V>();
    if (largest == 0) {
      Muls(valuesLocal, valuesLocal, static_cast<T>(-1.0), calcRowNum * kValue);
      AscendC::PipeBarrier<PIPE_V>();
    }
    outQueueValues.EnQue<T>(valuesLocal);
  }

  __aicore__ inline void DecodeIdxFromProposal(uint32_t calcRowNum)
  {
    LocalTensor<int32_t> indicesLocal = outQueueIndices.AllocTensor<int32_t>();
    LocalTensor<int32_t> proposalOutInt32Local = proposalOutBuf.Get<int32_t>(); // 看作是int32的tensor
    uint64_t rsvdCnt = 0;
    uint8_t int32Pattern = 3; // 每四个元素取第一个元素
    uint16_t gatherRepeat = static_cast<uint16_t>(DivCeil(calcRowNum * kValue * 4, 64));
    GatherMask(indicesLocal, proposalOutInt32Local, int32Pattern, false, 0, {1, gatherRepeat, 8, 0}, rsvdCnt);
    AscendC::PipeBarrier<PIPE_V>();
    outQueueIndices.EnQue<int32_t>(indicesLocal);
  }

  __aicore__ inline void CopyIn(uint32_t gmBias, uint32_t calcColNum)
  {
    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    DataCopy(xLocal, xGm[gmBias], DivCeil(calcColNum, FP16_BLK_SIZE) * FP16_BLK_SIZE); // 向上对齐
    inQueueX.EnQue(xLocal);
  }

  __aicore__ inline void MrgSortCustom(uint32_t iInner, uint32_t j, uint32_t proposalRepeat, LocalTensor<T>& proposalLocal)
  {
    LocalTensor<T> proposalTopkLocal = proposalTopkDoubleBuf.Get<T>();
    uint32_t headCount = (j == 0) ? 1 : 0; // 首次处理1个proposal，后续不处理
    uint32_t bodyRepeat = (proposalRepeat - headCount) / 3; // 处理bodyRepeat * 3个proposal
    uint32_t tailCount = (proposalRepeat - headCount) % 3;

    uint16_t proposalNum = 16;
    uint32_t elementNumPerRep = proposalNum * PROPOSAL_SIZE; // 16个proposals所含元素个数

    if (headCount == 1) {
      struct MrgSortSrcList<T> srcList(proposalLocal, proposalLocal, proposalLocal, proposalLocal);
      uint16_t elementLengths[4] = {static_cast<uint16_t>(kValue), proposalNum, proposalNum, proposalNum};
      struct MrgSort4Info srcInfo(elementLengths, true, 1, 1);
      MrgSort4(proposalTopkLocal[pingpongIdx * pingpongAddrBias], srcList, srcInfo);
      AscendC::PipeBarrier<PIPE_V>();
    }

    for (uint32_t n = 0; n < bodyRepeat; n++) {
      struct MrgSortSrcList<T> srcList(proposalTopkLocal[pingpongIdx * pingpongAddrBias],
                                       proposalLocal[elementNumPerRep * (headCount + 3 * n)],
                                       proposalLocal[elementNumPerRep * (headCount + 3 * n + 1)],
                                       proposalLocal[elementNumPerRep * (headCount + 3 * n + 2)]);
      uint16_t elementLengths[4] = {static_cast<uint16_t>(kValue), proposalNum, proposalNum, proposalNum};
      struct MrgSort4Info srcInfo(elementLengths, true, 15, 1); //当k小于16时，可将ifExhaustedSuspension置为true
      pingpongIdx = pingpongIdx ^ 1; // 1->0, 0->1
      MrgSort4(proposalTopkLocal[pingpongIdx * pingpongAddrBias], srcList, srcInfo);
      AscendC::PipeBarrier<PIPE_V>();
    }
    //tail
    struct MrgSortSrcList<T> srcList(proposalTopkLocal[pingpongIdx * pingpongAddrBias],
                                     proposalLocal[elementNumPerRep * (headCount + 3 * bodyRepeat)],
                                     proposalLocal[elementNumPerRep * (headCount + 3 * bodyRepeat + 1)],
                                     proposalLocal[elementNumPerRep * (headCount + 3 * bodyRepeat + 2)]);
    uint16_t elementLengths[4] = {static_cast<uint16_t>(kValue), proposalNum, proposalNum, proposalNum};
    struct MrgSort4Info srcInfo(elementLengths, true, CalcValidBit(tailCount), 1);

    if (j == jRepTimes - 1) {
      LocalTensor<T> proposalOutLocal = proposalOutBuf.Get<T>();
      MrgSort4(proposalOutLocal[iInner * kValue * PROPOSAL_SIZE], srcList, srcInfo);
    } else {
      pingpongIdx = pingpongIdx ^ 1;
      MrgSort4(proposalTopkLocal[pingpongIdx * pingpongAddrBias], srcList, srcInfo);
    }
    AscendC::PipeBarrier<PIPE_V>();
  }
  __aicore__ inline void Compute(uint32_t iInner, uint32_t j, uint32_t calcColNum)
  {
    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    if (largest == 0) {
      Muls(xLocal, xLocal, static_cast<T>(-1.0), calcColNum);
      AscendC::PipeBarrier<PIPE_V>();
    }
    uint32_t proposalRepeat = DivCeil(calcColNum, PROPOSAL_NUM_PER_REP);
    uint32_t calcColNumAlign = proposalRepeat * PROPOSAL_NUM_PER_REP;
    if (calcColNum != calcColNumAlign) {
      uint32_t padNum = calcColNumAlign - calcColNum;
      event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
      set_flag(PIPE_V, PIPE_S, event_v_s);
      wait_flag(PIPE_V, PIPE_S, event_v_s);
      for (uint32_t n = 0; n < padNum; n++) {
        xLocal.SetValue(calcColNumAlign - n - 1, FP16_NEG_INF);
      }
      event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
      set_flag(PIPE_S, PIPE_V, event_s_v);
      wait_flag(PIPE_S, PIPE_V, event_s_v);
    }

    LocalTensor<T> proposalLocal = proposalBuf.Get<T>();
    ProposalConcat(proposalLocal, xLocal, proposalRepeat, 4); // 合入score
    AscendC::PipeBarrier<PIPE_V>();
    inQueueX.FreeTensor(xLocal);

    RpSort16(proposalLocal, proposalLocal, proposalRepeat);
    AscendC::PipeBarrier<PIPE_V>();
    MrgSortCustom(iInner, j, proposalRepeat, proposalLocal);
  }

  __aicore__ inline void CopyOutValues(uint32_t gmBias, uint32_t calcRowNum)
  {
    LocalTensor<T> valuesLocal = outQueueValues.DeQue<T>();
    DataCopy(valuesGm[gmBias], valuesLocal, DivCeil(calcRowNum * kValue, FP16_BLK_SIZE) * FP16_BLK_SIZE);
    outQueueValues.FreeTensor(valuesLocal);
  }

  __aicore__ inline void CopyOutIndices(uint32_t gmBias, uint32_t calcRowNum)
  {
    LocalTensor<int32_t> indicesLocal = outQueueIndices.DeQue<int32_t>();
    DataCopy(indicesGm[gmBias], indicesLocal, DivCeil(calcRowNum * kValue, FP16_BLK_SIZE) * FP16_BLK_SIZE);
    outQueueIndices.FreeTensor(indicesLocal);
  }

private:
  TPipe *Ppipe = nullptr;
  // create queues for input, in this case depth is equal to buffer num
  TQue<QuePosition::VECIN, 1> inQueueX;
  // create queues for output, in this case depth is equal to buffer num
  TQue<QuePosition::VECOUT, 1> outQueueValues, outQueueIndices;
  TBuf<TPosition::VECCALC> baseIdxBuf, idxBuf, idxHigh16Buf, idxLow16Buf;
  TBuf<TPosition::VECCALC> proposalBuf, proposalTopkDoubleBuf, proposalOutBuf;

  GlobalTensor<T> xGm;
  GlobalTensor<T> valuesGm;
  GlobalTensor<int32_t> indicesGm;

  uint32_t numRow;
  uint32_t numCol;
  uint32_t blockFactor; // number of calculations rows on each core
  uint32_t rowFactor;
  uint32_t ubFactor;
  int32_t kValue;
  uint32_t largest;

  uint32_t rowWork = 0;
  uint32_t iOuterRepTimes;
  uint32_t rowTail;
  uint32_t jRepTimes;
  uint32_t colTail;

  uint32_t pingpongIdx;
  uint32_t pingpongAddrBias;
};
#endif // TOP_K_V3_H
