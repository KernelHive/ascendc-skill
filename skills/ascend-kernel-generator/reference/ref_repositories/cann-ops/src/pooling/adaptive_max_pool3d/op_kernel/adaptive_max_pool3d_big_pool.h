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
 * \file adaptive_max_pool3d_big_pool.h
 * \brief
 */
#ifndef ADAPTIVE_MAX_POOL3D_BIG_POOL_H_
#define ADAPTIVE_MAX_POOL3D_BIG_POOL_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr int64_t BLOCK_DATA = 32;
constexpr int64_t REPEAT_DATA = 256;

constexpr uint16_t FLOAT16_NEG_INF = 64512; // -inf 0xFC00
constexpr uint16_t FLOAT16_INF = 31744; // inf 0x7C00
constexpr uint16_t FLOAT16_NAN_END = 32768; // 0x8000

constexpr int32_t FLOAT32_NEG_INF = -2139095040; // -inf 0xFF800000
constexpr int32_t FLOAT32_INF = 2139095040; // inf 0x7F800000
constexpr int32_t FLOAT32_NEG_ZERO = -2147483648; // -0

template<typename T>
class InnerComputer {
public:
  __aicore__ inline void Compute(LocalTensor<T> &xLocal, LocalTensor<float> &castToFP32, TBuf<> &maxUB, TBuf<> &workLocalUB, uint32_t dataCount) {
    LocalTensor<T> maxOutLocal = maxUB.Get<T>();
    LocalTensor<T> workLocal = workLocalUB.Get<T>();
    ReduceMax<T>(maxOutLocal, xLocal, workLocal, dataCount, true);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void GetMask(LocalTensor<T> &xLocal, LocalTensor<float> &castToFP32, LocalTensor<uint16_t> &mask, uint32_t dataCount) {
    uint32_t dataCountAlign = (dataCount + REPEAT_DATA - 1) / REPEAT_DATA * REPEAT_DATA;  // Compare函数要求保证dataCountAlign个元素所占空间256字节对齐
    if(dataCountAlign > dataCount) {
      Duplicate(xLocal[dataCount], T(0), dataCountAlign - dataCount);
      PipeBarrier<PIPE_V>();
    }
    Compare(mask, xLocal, xLocal, CMPMODE::EQ, dataCountAlign);
    PipeBarrier<PIPE_V>();
    Not(mask, mask, dataCountAlign/sizeof(uint16_t));
    PipeBarrier<PIPE_V>();
  }
};

template <>
class InnerComputer<bfloat16_t> {
public:
  __aicore__ inline void Compute(LocalTensor<bfloat16_t> &xLocal, LocalTensor<float> &castToFP32, TBuf<> &maxUB, TBuf<> &workLocalUB, uint32_t dataCount) {
    LocalTensor<float> maxOutLocal = maxUB.Get<float>();
    LocalTensor<float> workLocal = workLocalUB.Get<float>();
    Cast(castToFP32, xLocal, RoundMode::CAST_NONE, dataCount);
    PipeBarrier<PIPE_V>();
    ReduceMax<float>(maxOutLocal, castToFP32, workLocal, dataCount, true);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void GetMask(LocalTensor<bfloat16_t> &xLocal, LocalTensor<float> &castToFP32, LocalTensor<uint16_t> &mask, uint32_t dataCount) {
    uint32_t dataCountAlign = (dataCount + REPEAT_DATA - 1) / REPEAT_DATA * REPEAT_DATA;  // Compare函数要求保证dataCountAlign个元素所占空间256字节对齐
    if(dataCountAlign > dataCount) {
      Duplicate(castToFP32[dataCount], float(0), dataCountAlign - dataCount);
      PipeBarrier<PIPE_V>();
    }
    Compare(mask, castToFP32, castToFP32, CMPMODE::EQ, dataCountAlign);
    PipeBarrier<PIPE_V>();
    Not(mask, mask, dataCountAlign/sizeof(uint16_t));
    PipeBarrier<PIPE_V>();
  }
};

template <typename T1, typename T2>
class AdaptiveMaxPool3dBigPool {
 public:
  __aicore__ inline AdaptiveMaxPool3dBigPool() {};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR indices, GM_ADDR workspace, TPipe* pipe_in,
                              const AdaptiveMaxPool3dBigPoolTilingData* __restrict tiling, int64_t dataType);
  __aicore__ inline void Process();

private:
  __aicore__ inline void Prepare(int64_t curIdx);
  __aicore__ inline void BaseCompute(int64_t curIdx);
  __aicore__ inline int64_t dhwCopyInput(int64_t offset, int64_t blockLen);
  __aicore__ inline int64_t hwCopyInput(int64_t offset, int64_t blockLen, int64_t blockCount);
  __aicore__ inline int64_t wCopyInput(int64_t offset, int64_t blockLen, int64_t blockCount, int64_t dLen);
  __aicore__ inline int32_t Compute(int64_t dataCount);
  __aicore__ inline int64_t RestoreIndex(int32_t index, int64_t dLen, int64_t hLen, int64_t wLen);
  __aicore__ inline void CopyMaxOut(int64_t curIdx);
  __aicore__ inline void CopyIndicesOut(int64_t maxIndex, int64_t curIdx);
  __aicore__ inline void NaNIndicesInit(LocalTensor<float> indicesLocal);
  __aicore__ inline void GetIndexWithLastNan(LocalTensor<uint16_t> maskNanLocal, int64_t dataCount, int32_t &index);
  __aicore__ inline int64_t dhwContinueProcess();
  __aicore__ inline int64_t hwContinueProcess();
  __aicore__ inline int64_t wContinueProcess();
  __aicore__ inline void UpdateMax(int64_t curMaxIndex, T2 &maxValue, int64_t &maxIndice);
  __aicore__ inline int32_t kernelRealIndex(int32_t index, int64_t blockLen);

  __aicore__ inline int64_t CeilValue(int64_t inputValue, int64_t upperValue) {
    if(upperValue == 0) {
      return inputValue;
    }
    return (inputValue + upperValue - 1) / upperValue * upperValue;
  }

  __aicore__ inline int64_t min(int64_t a, int64_t b) {
    return (a > b) ? b : a;
  }

  __aicore__ inline int64_t startIndex(int64_t idx, int64_t inLen, int64_t outLen) {
    if(outLen == 0) {
      return 0;
    }
    return idx * inLen / outLen;
  }

  __aicore__ inline int64_t endIndex(int64_t idx, int64_t inLen, int64_t outLen) {
    if(outLen == 0) {
      return 0;
    }
    return ((idx + 1) * inLen + outLen - 1) / outLen;
  }

  /*
   * 功能：判断输入值是否为NaN
   * 说明：
   *    NaN定义：符号位取值无关紧要，指数位全为1，小数部分不全为0，表示为NaN
   *    INF定义：符号为代表正负，指数位全为1，小数部分全为0，表示为INF
   *    以Float16为例，其表达式为 Sign*1 + exponent*5 + fraction*10，
   *                符号位为0，NaN的最小值为0111110000000001，unint16为31745
   *                符号位为0，NaN的最大值为011111111111111，unint16为32767
   *                符号位为1，其表示为NaN的最小值为1111110000000001，unint16为64513
   *                INF为 0111110000000000，unint16为31744
   *                -INF为1111110000000000，unint16为64512
   * 
   *    以Float32为例，其表达式为 Sign*1 + exponent*8 + fraction*23，
   *                符号位为0，NaN的最小值为01111111100000000000000000000001，int32为2139095041
   *                符号位为0，NaN的最大值为01111111111111111111111111111111，int32为2147483647
   *                符号位为1，NaN的最大值为11111111100000000000000000000001，int32为-2139095041
   *                INF为  01111111100000000000000000000000 int32为2139095040
   *                -INF为 11111111100000000000000000000000 int32为-2139095040
   */
  __aicore__ inline bool isnan(T2 value) {
    if(inputDataTypeKey == 1) {  // 输入数据为fp16
      uint16_t nan = *reinterpret_cast<uint16_t *>(&value);
      if((nan > FLOAT16_INF && nan < FLOAT16_NAN_END) || nan > FLOAT16_NEG_INF) {
        return true;
      }
    }else{  //输入数据为fp32或bf16
      int32_t nan = *reinterpret_cast<int32_t *>(&value);
      if((nan != FLOAT32_NEG_ZERO) && (nan > FLOAT32_INF || nan < FLOAT32_NEG_INF)) {
        return true;
      }
    }
    return false;
  }

  TPipe* pipe;
  // 输入队列
  TQue<QuePosition::VECIN, BUFFER_NUM> inputQue;
  // bf16转换 && nan indeices Select结果
  TBuf<> inputCastUB;
  // 最大值ub
  TBuf<> maxUB;
  // indices初始下标
  TBuf<> indicesInitUB;
  // ReduceMax所需临时workLocal空间
  TBuf<> reduceWorkLocalUB;
  // Compare结果mask
  TBuf<> maskNanUB;
  // nan场景最大值和下标
  TBuf<> nanMaxIndexUB;

  GlobalTensor<T1> xGm, maxGm;
  GlobalTensor<int32_t> indicesGm;

  const AdaptiveMaxPool3dBigPoolTilingData* tilingData;

  uint32_t cBlockIdx;
  // shape info
  int64_t nc = 1;
  int64_t d = 1;
  int64_t h = 1;
  int64_t w = 1;
  int64_t dout = 1;  // do是关键字，此处变量名改为dout
  int64_t ho = 1;
  int64_t wo = 1;

  // tiling info
  int64_t blockFactor = 1;
  int64_t blockTail = 1;
  int64_t coreNums = 1;
  int64_t totalIdx = 1;
  int64_t inputDataTypeKey = 0;

  int64_t inHW = 1;
  int64_t inDHW = 1;
  int64_t outHW = 1;
  int64_t outDHW = 1;
  int64_t curNc = 0;
  int64_t curOriginD = 0;
  int64_t curOriginH = 0;
  int64_t curOriginW = 0;
  int64_t curOriginIndex = 0;
  int64_t curkD = 1;
  int64_t curkH = 1;
  int64_t curkW = 1;
  int64_t curInOffset = 0;
  T1 minT1 = 0;
  T2 minT2 = 0;
  int32_t maxCount = 10 * 1024;

  constexpr static int64_t BYTE_T1 = sizeof(T1);
  constexpr static int64_t BLOCK_NUM_T1 = BLOCK_DATA / sizeof(T1);
};

template <typename T1, typename T2>
__aicore__ inline void AdaptiveMaxPool3dBigPool<T1, T2>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR indices, GM_ADDR workspace, TPipe* pipe_in,
    const AdaptiveMaxPool3dBigPoolTilingData* __restrict tiling, int64_t dataType) {
  pipe = pipe_in;
  tilingData = tiling;
  // base info
  cBlockIdx = GetBlockIdx();
  // shape info
  d = tilingData->Di;
  h = tilingData->Hi;
  w = tilingData->Wi;
  dout = tilingData->Do;
  ho = tilingData->Ho;
  wo = tilingData->Wo;

  // tiling info
  blockFactor = tilingData->blockFactor;
  blockTail = tilingData->blockTail;
  totalIdx = tilingData->totalIdx;
  inputDataTypeKey = dataType;

  inHW = h * w;
  inDHW = d * inHW;
  outHW = ho * wo;
  outDHW = dout * outHW;

  uint32_t MIN_FLOAT32 = 0xFF800000;// -inf
  uint16_t MIN_FLOAT16 = 0xFC00;// -inf
  uint16_t MIN_BFLOAT16 = 0xFF80;// -inf
  if(inputDataTypeKey == 0) {
    minT1 = *reinterpret_cast<T1 *>(&MIN_FLOAT32);
    minT2 = minT1;
  } else if (inputDataTypeKey == 1) {
    minT1 = *reinterpret_cast<T1 *>(&MIN_FLOAT16);
    minT2 = minT1;
  } else if (inputDataTypeKey == 2) {
    minT1 = *reinterpret_cast<T1 *>(&MIN_BFLOAT16);
    minT2 = *reinterpret_cast<T2 *>(&MIN_FLOAT32);
  }
  
  // GM
  xGm.SetGlobalBuffer((__gm__ T1*)x);
  maxGm.SetGlobalBuffer((__gm__ T1*)y);
  indicesGm.SetGlobalBuffer((__gm__ int32_t*)indices);

  pipe->InitBuffer(inputQue, BUFFER_NUM, maxCount * sizeof(T1));
  pipe->InitBuffer(inputCastUB, maxCount * sizeof(float));
  pipe->InitBuffer(maxUB, 256);
  pipe->InitBuffer(indicesInitUB, maxCount * sizeof(float));
  pipe->InitBuffer(reduceWorkLocalUB, maxCount / 16 * sizeof(float));
  pipe->InitBuffer(maskNanUB, maxCount / 4);
  pipe->InitBuffer(nanMaxIndexUB, 256);
}

template <typename T1, typename T2>
__aicore__ inline void AdaptiveMaxPool3dBigPool<T1, T2>::Process() {
  // init indices
  LocalTensor<float> indicesLocal = indicesInitUB.Get<float>();
  NaNIndicesInit(indicesLocal);

  int64_t beginIdx = 0;
  int64_t endIdx = 0;
  if(cBlockIdx < blockTail) {
    beginIdx = cBlockIdx * (blockFactor + 1);
    endIdx = beginIdx + blockFactor + 1;
  } else {
    beginIdx = cBlockIdx * blockFactor + blockTail;
    endIdx = beginIdx + blockFactor;
  }

  if(dout == 1 && ho == 1 && wo == 1) {
    curOriginD = 0;
    curOriginH = 0;
    curOriginW = 0;
    curOriginIndex = 0;
    curkD = d;
    curkH = h;
    curkW = w;
  }

  // current blockdim range
  for (int64_t idx = beginIdx; idx < endIdx; idx++) {
    Prepare(idx);
    BaseCompute(idx);
  }
}

/*
 * 功能：动态计算输出点curIdx所对应的kernel size大小    
 */
template <typename T1, typename T2>
__aicore__ inline void AdaptiveMaxPool3dBigPool<T1, T2>::Prepare(int64_t curIdx) {

  if(dout == 1 && ho == 1 && wo == 1) {
    curNc = curIdx;
    curInOffset = curNc * inDHW;
    return;
  }

  int64_t cur3D = curIdx % outDHW;  // 单个output中的输出点index
  int64_t cur2D = cur3D % outHW;  // 单个output中的d维中的index

  int64_t curNc = curIdx / outDHW;  // curIdx所属nc前的nc数量
  int64_t curDo = cur3D / outHW;  // curIdx所属d前的d数量
  int64_t curHo = cur2D / wo;  // curIdx所属h前的h数量
  int64_t curWo = cur2D % wo;  // curIdx的w维度位置

  // 计算输出点curIdx对应的kernel在原nc中的起始位置
  curOriginD = startIndex(curDo, d, dout);
  curkD = endIndex(curDo, d, dout) - curOriginD;

  curOriginH = startIndex(curHo, h, ho);
  curkH = endIndex(curHo, h, ho) - curOriginH;

  curOriginW = startIndex(curWo, w, wo);
  curkW = endIndex(curWo, w, wo) - curOriginW;

  // 计算输出点curIdx对应的kernel在原nc数据中的起始index
  curOriginIndex = curOriginD * inHW + curOriginH * w + curOriginW;
  // 计算输出点curIdx对应的kernel在原数据中的index，作为拷贝数据的起始偏移量
  curInOffset = curNc * inDHW + curOriginIndex;
}

template <typename T1, typename T2>
__aicore__ inline void AdaptiveMaxPool3dBigPool<T1, T2>::BaseCompute(int64_t curIdx) {
  int64_t realIndex = 0;
  if(curkW == w && curkH == h) {
    realIndex = dhwContinueProcess();
  } else if(curkW == w && curkH != h) {
    realIndex = hwContinueProcess();
  } else if(curkW != w) {
    realIndex = wContinueProcess();
  }

  // 3、将max拷出到gm
  CopyMaxOut(curIdx);
  // 4、将indices拷出到gm
  CopyIndicesOut(realIndex, curIdx);
}

template <typename T1, typename T2>
__aicore__ inline int64_t AdaptiveMaxPool3dBigPool<T1, T2>::dhwCopyInput(int64_t offset, int64_t blockLen) {

  LocalTensor<T1> xLocal = inputQue.AllocTensor<T1>();
  int64_t blockLenAlign = CeilValue(blockLen, BLOCK_NUM_T1);
  int64_t alignNum = blockLenAlign - blockLen;
  DataCopyPadExtParams<T1> padExtParams;
  padExtParams.isPad = alignNum != 0;
  padExtParams.leftPadding = 0;
  padExtParams.rightPadding = padExtParams.isPad ? alignNum : 0;
  padExtParams.paddingValue = minT1;

  DataCopyExtParams extParams;
  extParams.blockCount = 1;
  extParams.blockLen = blockLen * sizeof(T1);
  extParams.srcStride = 0;
  extParams.dstStride = 0;
  DataCopyPad(xLocal, xGm[offset], extParams, padExtParams);
  inputQue.EnQue(xLocal);
  return blockLenAlign;
}

template <typename T1, typename T2>
__aicore__ inline int64_t AdaptiveMaxPool3dBigPool<T1, T2>::hwCopyInput(int64_t offset, int64_t blockLen, int64_t blockCount) {

  LocalTensor<T1> xLocal = inputQue.AllocTensor<T1>();
  int64_t blockLenAlign = CeilValue(blockLen, BLOCK_NUM_T1);
  int64_t alignNum = blockLenAlign - blockLen;
  DataCopyPadExtParams<T1> padExtParams;
  padExtParams.isPad = alignNum != 0;
  padExtParams.leftPadding = 0;
  padExtParams.rightPadding = padExtParams.isPad ? alignNum : 0;
  padExtParams.paddingValue = minT1;

  if(UINT32_MAX >= (inHW - blockLen) * sizeof(T1)) {
    DataCopyExtParams extParams;
    extParams.blockCount = blockCount;
    extParams.blockLen = blockLen * sizeof(T1);
    extParams.srcStride = (inHW - blockLen) * sizeof(T1);
    extParams.dstStride = 0;
    DataCopyPad(xLocal, xGm[offset], extParams, padExtParams);
  } else {
    DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = blockLen * sizeof(T1);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    for(int i = 0; i < blockCount; i++) {
      DataCopyPad(xLocal[i * blockLenAlign], xGm[offset + i * inHW], extParams, padExtParams);
    }
  }
  inputQue.EnQue(xLocal);
  return blockLenAlign * blockCount;
}

template <typename T1, typename T2>
__aicore__ inline int64_t AdaptiveMaxPool3dBigPool<T1, T2>::wCopyInput(int64_t offset, int64_t blockLen, int64_t blockCount, int64_t dLen) {

  LocalTensor<T1> xLocal = inputQue.AllocTensor<T1>();
  int64_t blockLenAlign = CeilValue(blockLen, BLOCK_NUM_T1);
  int64_t alignNum = blockLenAlign - blockLen;
  DataCopyPadExtParams<T1> padExtParams;
  padExtParams.isPad = alignNum != 0;
  padExtParams.leftPadding = 0;
  padExtParams.rightPadding = padExtParams.isPad ? alignNum : 0;
  padExtParams.paddingValue = minT1;

  if(UINT32_MAX >= (w - blockLen) * sizeof(T1)) {
    DataCopyExtParams extParams;
    extParams.blockCount = blockCount;
    extParams.blockLen = blockLen * sizeof(T1);
    extParams.srcStride = (w - blockLen) * sizeof(T1);
    extParams.dstStride = 0;
    for(int i = 0; i < dLen; i++) {
      DataCopyPad(xLocal[i * blockLenAlign * blockCount], xGm[offset + i * inHW], extParams, padExtParams);
    }
  } else {
    DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = blockLen * sizeof(T1);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    for(int i = 0; i < dLen * blockCount; i++) {
      DataCopyPad(xLocal[i * blockLenAlign], xGm[offset + i * w], extParams, padExtParams);
    }
  }
  inputQue.EnQue(xLocal);
  return blockLenAlign * blockCount * dLen;
}

template <typename T1, typename T2>
__aicore__ inline int32_t AdaptiveMaxPool3dBigPool<T1, T2>::Compute(int64_t dataCount) {

  LocalTensor<T1> xLocal = inputQue.DeQue<T1>();
  LocalTensor<float> castToFP32 = inputCastUB.Get<float>();
  InnerComputer<T1> computer;
  computer.Compute(xLocal, castToFP32, maxUB, reduceWorkLocalUB, dataCount);

  event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
  SetFlag<HardEvent::V_S>(eventIDVToS);
  WaitFlag<HardEvent::V_S>(eventIDVToS);

  int32_t index = 0;
  LocalTensor<float> indicesLocal = indicesInitUB.Get<float>();
  LocalTensor<uint16_t> maskNanLocal = maskNanUB.Get<uint16_t>();
  // 输入为fp16时
  if(inputDataTypeKey == 1) {
    LocalTensor<half> maxOutLocal = maxUB.Get<half>();
    half maxIndex = maxOutLocal.GetValue(1);
    int16_t indexInt16 = *reinterpret_cast<int16_t *>(&maxIndex);
    index = static_cast<int32_t>(indexInt16);

    half maxValue = maxOutLocal.GetValue(0);
    if(isnan(maxValue)) {
      computer.GetMask(xLocal, castToFP32, maskNanLocal, dataCount);
      GetIndexWithLastNan(maskNanLocal, dataCount, index);
    }
  } else {
    LocalTensor<float> maxOutLocal = maxUB.Get<float>();
    LocalTensor<float> castToFP32 = inputCastUB.Get<float>();
    float maxIndex = maxOutLocal.GetValue(1);
    index = *reinterpret_cast<int32_t *>(&maxIndex);
    float maxValue = maxOutLocal.GetValue(0);
    if(isnan(maxValue)) {
      computer.GetMask(xLocal, castToFP32, maskNanLocal, dataCount);
      GetIndexWithLastNan(maskNanLocal, dataCount, index);
    }
  }
  inputQue.FreeTensor<T1>(xLocal);
  return index;
}

template <typename T1, typename T2>
__aicore__ inline void AdaptiveMaxPool3dBigPool<T1, T2>::GetIndexWithLastNan(LocalTensor<uint16_t> maskNanLocal, int64_t dataCount, int32_t &index) {

  LocalTensor<float> indicesLocal = indicesInitUB.Get<float>(); 
  LocalTensor<float> indicesMaxLocal = inputCastUB.Get<float>();
  Select(indicesMaxLocal, maskNanLocal, indicesLocal, float(-1), SELMODE::VSEL_TENSOR_SCALAR_MODE, dataCount);
  PipeBarrier<PIPE_V>();

  LocalTensor<float> workLocal = reduceWorkLocalUB.Get<float>();
  LocalTensor<float> nanMaxIndex = nanMaxIndexUB.Get<float>();
  ReduceMax<float>(nanMaxIndex, indicesMaxLocal, workLocal, dataCount, false);
  PipeBarrier<PIPE_V>();

  event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
  SetFlag<HardEvent::V_S>(eventIDVToS);
  WaitFlag<HardEvent::V_S>(eventIDVToS);
  index = ScalarCast<float, int32_t, RoundMode::CAST_ROUND>(nanMaxIndex.GetValue(0));  // indicesLocal初始化时为float类型，indicesLocal < 10240，类型转变不会有精度丢失问题
}

/*
 * 功能：计算所求max值在该nc中的真实Index
 */
template <typename T1, typename T2>
__aicore__ inline int64_t AdaptiveMaxPool3dBigPool<T1, T2>::RestoreIndex(int32_t index, int64_t dLen, int64_t hLen, int64_t wLen) {

  int64_t realIndex = 0;
  if(wLen == w && hLen == h) {
    realIndex = curOriginIndex + index;  // 数据连续，直接加偏移量
  } else if(wLen == w) {
    realIndex = curOriginIndex + index / (wLen * hLen) * inHW + index % (wLen * hLen);  // hw连续，根据index计算kernel内的d和h
  } else {
    realIndex = curOriginIndex + index / (hLen * wLen) * inHW + index % (hLen * wLen) / wLen * w + index % wLen;  // w连续，根据index计算kernel内d、h、w
  }
  return realIndex;
}

template <typename T1, typename T2>
__aicore__ inline void AdaptiveMaxPool3dBigPool<T1, T2>::CopyMaxOut(int64_t curIdx) {
  DataCopyExtParams extParams;
  extParams.blockCount = 1;
  extParams.blockLen = 1 * sizeof(T1);
  extParams.srcStride = 0;
  extParams.dstStride = 0;
  if (inputDataTypeKey == 2) {
    LocalTensor<float> maxfloat32 = maxUB.Get<float>();
    LocalTensor<bfloat16_t> maxbfloat16 = maxUB.Get<bfloat16_t>();
    PipeBarrier<PIPE_ALL>();
    Cast(maxbfloat16, maxfloat32, RoundMode::CAST_RINT, 8);
    PipeBarrier<PIPE_V>();
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
  }
  LocalTensor<T1> maxOutLocal = maxUB.Get<T1>();
  DataCopyPad(maxGm[curIdx], maxOutLocal[0], extParams);
}

template <typename T1, typename T2>
__aicore__ inline void AdaptiveMaxPool3dBigPool<T1, T2>::CopyIndicesOut(int64_t maxIndex, int64_t curIdx) {
  DataCopyExtParams extParams;
  extParams.blockCount = 1;
  extParams.blockLen = 1 * sizeof(int32_t);
  extParams.srcStride = 0;
  extParams.dstStride = 0;
  event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
  SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
  WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
  LocalTensor<int32_t> indexTensor = nanMaxIndexUB.Get<int32_t>();
  indexTensor[32].SetValue(0, maxIndex);
  event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
  SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
  WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
  DataCopyPad(indicesGm[curIdx], indexTensor[32], extParams);
}

template <typename T1, typename T2>
__aicore__ inline void AdaptiveMaxPool3dBigPool<T1, T2>::NaNIndicesInit(
    LocalTensor<float> indicesLocal) {
  for (int32_t idx = 0; idx < 8; idx++) {
    indicesLocal.SetValue(idx, float(idx));
  }
  event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
  SetFlag<HardEvent::S_V>(eventIDSToV);
  WaitFlag<HardEvent::S_V>(eventIDSToV);
  int64_t dhAlignkW = CeilValue(w / wo + 2, BLOCK_NUM_T1) * (h / ho + 2) * (d / dout + 2);
  int32_t InitIndicesNum = (dhAlignkW > maxCount) ? maxCount : dhAlignkW;
  if (InitIndicesNum > 8) {
    int32_t addNum = 8;
    for (int32_t idx = 8; idx < InitIndicesNum;) {
      int32_t addNumCount = (2*idx > InitIndicesNum) ? InitIndicesNum-idx : idx;
      Adds(indicesLocal[idx], indicesLocal, float(addNum), addNumCount);
      PipeBarrier<PIPE_V>();
      idx += addNumCount;
      addNum = 2 * addNum;
    }
  }
}

template <typename T1, typename T2>
__aicore__ inline int64_t AdaptiveMaxPool3dBigPool<T1, T2>::dhwContinueProcess() {
  int64_t realIndex = 0;
  int64_t curkDHW = curkD * curkH * curkW;
  int64_t AlignkDHW = CeilValue(curkDHW, BLOCK_NUM_T1);
  int64_t inputOffset = curInOffset;
  int64_t kernelOffset = 0;
  T2 maxValue = minT2;
  int64_t maxIndex = 0;
  if(AlignkDHW <= maxCount) {
    int32_t dataCount = dhwCopyInput(curInOffset, curkDHW);
    int32_t index = Compute(dataCount);
    realIndex = RestoreIndex(index, curkD, curkH, curkW);
  } else {
    int64_t dhwLoops = (curkDHW + maxCount - 1) / maxCount;
    int32_t dhwFactor = maxCount;
    int32_t dhwTail = curkDHW % maxCount;
    if(dhwTail == 0) {
      dhwTail = dhwFactor;
    }
    for(int32_t dhwLoop = 0; dhwLoop < dhwLoops; dhwLoop++) {
      int32_t curFactor = dhwLoop == dhwLoops - 1 ? dhwTail : dhwFactor;
      int32_t dataCount = dhwCopyInput(inputOffset, curFactor);
      int32_t index = Compute(dataCount);
      UpdateMax(kernelOffset + index, maxValue, maxIndex);
      inputOffset += curFactor;
      kernelOffset += curFactor;
    }
    realIndex = RestoreIndex(maxIndex, curkD, curkH, curkW);
    LocalTensor<T2> maxOutLocal = maxUB.Get<T2>();
    maxOutLocal.SetValue(0, maxValue);
    event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
  }
  return realIndex;
}

template <typename T1, typename T2>
__aicore__ inline int64_t AdaptiveMaxPool3dBigPool<T1, T2>::hwContinueProcess() {
  int64_t realIndex = 0;
  int64_t curkHW = curkH * curkW;
  int64_t AlignkHW = CeilValue(curkHW, BLOCK_NUM_T1);
  int64_t alignNum = 0;
  int64_t blockLenAlign = 0;
  int64_t inputOffset = curInOffset;
  int64_t kernelOffset = 0;
  T2 maxValue = minT2;
  int64_t maxIndex = 0;
  if(curkD * AlignkHW <= maxCount) {
    int32_t dataCount = hwCopyInput(curInOffset, curkHW, curkD);
    int32_t index = Compute(dataCount);
    index = kernelRealIndex(index, curkHW);
    realIndex = RestoreIndex(index, curkD, curkH, curkW);
  } else {
    if(AlignkHW > maxCount) {
      int32_t dLoops = curkD;
      int32_t dFactor = 1;
      int32_t dTail = 0;
      int32_t hwLoops = (curkHW + maxCount - 1) / maxCount;
      int32_t hwFactor = maxCount;
      int32_t hwTail = curkHW % maxCount;
      if(hwTail == 0) {
        hwTail = hwFactor;
      }

      for(int32_t dLoop = 0; dLoop < dLoops; dLoop++) {
        for(int32_t hwLoop = 0; hwLoop < hwLoops; hwLoop++) {
          int32_t curFactor = hwLoop == hwLoops - 1 ? hwTail : hwFactor;
          int32_t dataCount = hwCopyInput(inputOffset, curFactor, 1);
          int32_t index = Compute(dataCount);
          index = kernelRealIndex(index, curFactor);
          UpdateMax(kernelOffset + index, maxValue, maxIndex);
          inputOffset += curFactor;
          kernelOffset += curFactor;
        }
        inputOffset = curInOffset + (dLoop + 1) * inHW;
      }
    } else {
      int32_t dFactor = maxCount / AlignkHW;
      int32_t dLoops = (curkD + dFactor - 1) / dFactor;
      int32_t dTail = curkD % dFactor;
      if(dTail == 0) {
        dTail = dFactor;
      }
      int32_t hwLoops = 1;
      int32_t hwFactor = curkHW;
      int32_t hwTail = 0;
      for(int32_t dLoop = 0; dLoop < dLoops; dLoop++) {
        int32_t curdFactor = dLoop == dLoops - 1 ? dTail : dFactor;
        int32_t dataCount = hwCopyInput(inputOffset, hwFactor, curdFactor);
        int32_t index = Compute(dataCount);
        index = kernelRealIndex(index, hwFactor);
        UpdateMax(kernelOffset + index, maxValue, maxIndex);
        inputOffset = curInOffset + curdFactor * inHW * (dLoop + 1);
        kernelOffset += curdFactor * hwFactor;
      }
    }
    realIndex = RestoreIndex(maxIndex, curkD, curkH, curkW);
    LocalTensor<T2> maxOutLocal = maxUB.Get<T2>();
    maxOutLocal.SetValue(0, maxValue);
    event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
  }
  return realIndex;
}

template <typename T1, typename T2>
__aicore__ inline int64_t AdaptiveMaxPool3dBigPool<T1, T2>::wContinueProcess() {
  int64_t realIndex = 0;
  int64_t curkHW = curkH * curkW;
  int64_t AlignkW = CeilValue(curkW, BLOCK_NUM_T1);
  int64_t alignNum = 0;
  int64_t blockLenAlign = 0;
  int64_t inputOffset = curInOffset;
  int64_t kernelOffset = 0;
  T2 maxValue = minT2;
  int64_t maxIndex = 0;
  if (curkD * curkH * AlignkW <= maxCount) {
    int32_t dataCount = wCopyInput(curInOffset, curkW, curkH, curkD);
    int32_t index = Compute(dataCount);
    index = kernelRealIndex(index, curkW);
    realIndex = RestoreIndex(index, curkD, curkH, curkW);
  } else {
    if(AlignkW > maxCount) {
      int64_t dLoops = curkD;
      int64_t dFactor = 1;
      int64_t dTail = 0;
      int64_t hLoops = curkH;
      int64_t hFactor = 1;
      int64_t hTail = 0;
      int64_t wFactor = maxCount;
      int64_t wLoops = (curkW + wFactor - 1) / wFactor;
      int64_t wTail = curkW % wFactor;
      if(wTail == 0) {
        wTail = wFactor;
      }
      for(int32_t dLoop = 0; dLoop < dLoops; dLoop++) {
        for(int32_t hLoop = 0; hLoop < hLoops; hLoop++) {
          inputOffset = curInOffset + dLoop * inHW + hLoop * w;
          for(int32_t wLoop = 0; wLoop < wLoops; wLoop++) {
            int32_t curFactor = wLoop == wLoops - 1 ? wTail : wFactor;
            int32_t dataCount = dhwCopyInput(inputOffset, curFactor);
            int32_t index = Compute(dataCount);
            index = kernelRealIndex(index, curFactor);
            UpdateMax(kernelOffset + index, maxValue, maxIndex);
            inputOffset += curFactor;
            kernelOffset += curFactor;
          }
        }
      }
    } else if (AlignkW * curkH > maxCount) {
      int64_t dLoops = curkD;
      int64_t dFactor = 1;
      int64_t dTail = 0;
      int64_t hFactor = maxCount / AlignkW;
      int64_t hLoops = (curkH + hFactor - 1) / hFactor;
      int64_t hTail = curkH % hFactor;
      if(hTail == 0) {
        hTail = hFactor;
      }
      int64_t wLoops = 1;
      int64_t wFactor = curkW;
      int64_t wTail = 0;
      for(int32_t dLoop = 0; dLoop < dLoops; dLoop++) {
        inputOffset = curInOffset + dLoop * inHW;
        for(int32_t hLoop = 0; hLoop < hLoops; hLoop++) {
          int32_t curhFactor = hLoop == hLoops - 1 ? hTail : hFactor;
          int32_t curFactor = hLoop == hLoops - 1 ? hTail * wFactor : hFactor * wFactor;
          int32_t dataCount = wCopyInput(inputOffset, wFactor, curhFactor, 1);
          int32_t index = Compute(dataCount);
          index = kernelRealIndex(index, wFactor);
          UpdateMax(kernelOffset + index, maxValue, maxIndex);
          inputOffset += curhFactor * w;
          kernelOffset += curFactor;
        }
      }
    } else if(AlignkW * curkH * curkD > maxCount) {
      int64_t dFactor = maxCount / (AlignkW * curkH);
      int64_t dLoops = (curkD + dFactor - 1) / dFactor;
      int64_t dTail = curkD % dFactor;
      if(dTail == 0) {
        dTail = dFactor;
      }
      int64_t hLoops = 1;
      int64_t hFactor = curkH;
      int64_t hTail = 0;
      int64_t wLoops = 1;
      int64_t wFactor = curkW;
      int64_t wTail = 0;
      for(int32_t dLoop = 0; dLoop < dLoops; dLoop++) {
        int32_t curdFactor = dLoop == dLoops - 1 ? dTail : dFactor;
        int32_t curFactor = dLoop == dLoops - 1 ? dTail * hFactor * wFactor : dFactor * hFactor * wFactor;
        int32_t dataCount = wCopyInput(inputOffset, wFactor, hFactor, curdFactor);
        int32_t index = Compute(dataCount);
        index = kernelRealIndex(index, wFactor);
        UpdateMax(kernelOffset + index, maxValue, maxIndex);
        inputOffset = curInOffset + curdFactor * inHW * (dLoop + 1);
        kernelOffset += curFactor;
      }
    }
    realIndex = RestoreIndex(maxIndex, curkD, curkH, curkW);
    LocalTensor<T2> maxOutLocal = maxUB.Get<T2>();
    maxOutLocal.SetValue(0, maxValue);
    event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
  }
  return realIndex;
}

template <typename T1, typename T2>
__aicore__ inline void AdaptiveMaxPool3dBigPool<T1, T2>::UpdateMax(int64_t curMaxIndex, T2 &maxValue, int64_t &maxIndex) {
  LocalTensor<T2> maxOutLocal = maxUB.Get<T2>();
  T2 curMaxValue = maxOutLocal.GetValue(0);
  if(isnan(curMaxValue) || (!isnan(maxValue) && (static_cast<float>(curMaxValue) > static_cast<float>(maxValue)))) {
    maxIndex = curMaxIndex;
    maxValue = curMaxValue;
  }
}

/*
 * 功能：计算所求Max值在kernel中的真实index
 * 说明：
 *    在CopyInput操作中，因DataCopyPad要求拷入数据需要对齐，故对拷入数据做了pad处理。此函数反向计算pad数量
 *    计算所求Max值在kernel中的真实Index
 */
template <typename T1, typename T2>
__aicore__ inline int32_t AdaptiveMaxPool3dBigPool<T1, T2>::kernelRealIndex(int32_t index, int64_t blockLen) {
  int64_t blockLenAlign = CeilValue(blockLen, BLOCK_NUM_T1);
  int64_t alignNum = blockLenAlign - blockLen;
  if(alignNum != 0) {
    return index - alignNum * (index / blockLenAlign);
  } else {
    return index;
  }
}
#endif  // ADAPTIVE_MAX_POOL3D_BIG_POOL_H_
