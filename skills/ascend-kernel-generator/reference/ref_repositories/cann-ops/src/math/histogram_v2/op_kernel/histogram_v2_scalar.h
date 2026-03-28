/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file histogram_v2_scalar.h
 * \brief
 */
#ifndef HISTOGRAM_V2_SCALAR_H
#define HISTOGRAM_V2_SCALAR_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace HistogramV2NS {
using namespace AscendC;
constexpr int32_t BLOCK_NUM = 32;
constexpr int32_t DOUBLE_BUFFER = 2;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
constexpr int32_t STATIC_ARRAY_LENGTH = 2560;
#else
constexpr int32_t STATIC_ARRAY_LENGTH = 5120;
#endif
constexpr int32_t ALIGNED_NUM = 8;

template <typename MTE_T, typename CAST_T, typename COMPUTE_T>
class HistogramV2Scalar {
public:
  __aicore__ inline HistogramV2Scalar() {}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR min, GM_ADDR max, GM_ADDR y, GM_ADDR workspace,
                              const HistogramV2TilingData* tilingData, TPipe* tPipe) {
    this->bins = tilingData->bins;
    this->ubBinsLength = tilingData->ubBinsLength;
    this->coreNum = GetBlockNum();
    int64_t formerNum = tilingData->formerNum;
    int64_t formerLength = tilingData->formerLength;
    int64_t formerLengthAligned = tilingData->formerLengthAligned;
    int64_t tailLength = tilingData->tailLength;
    int64_t tailLengthAligned = tilingData->tailLengthAligned;
    int64_t typeMultiples = sizeof(CAST_T) / sizeof(MTE_T);

    if (GetBlockIdx() < formerNum) { // former core
        this->coreTotalDataLength = formerLength;
        this->tileNum = tilingData->formerTileNum;
        this->tileDataLength = tilingData->formerTileDataLength;
        this->tileLeftDataLength = tilingData->formerTileLeftDataLength;
        this->xGm.SetGlobalBuffer(reinterpret_cast<__gm__ MTE_T*>(x) + formerLength * typeMultiples * GetBlockIdx(),
                                       formerLengthAligned * typeMultiples);
    } else {
        this->coreTotalDataLength = tailLength;
        this->tileNum = tilingData->tailTileNum;
        this->tileDataLength = tilingData->tailTileDataLength;
        this->tileLeftDataLength = tilingData->tailTileLeftDataLength;
        this->xGm.SetGlobalBuffer(reinterpret_cast<__gm__ MTE_T*>(x) + typeMultiples * formerLength
                                       * formerNum + typeMultiples * tailLength * (GetBlockIdx() - formerNum),
                                        tailLengthAligned * typeMultiples);
    }
    this->minGm.SetGlobalBuffer(reinterpret_cast<__gm__ MTE_T*>(min), BLOCK_NUM);
    this->maxGm.SetGlobalBuffer(reinterpret_cast<__gm__ MTE_T*>(max), BLOCK_NUM);
    this->yGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(y), this->bins);
    this->pipe = tPipe;
    #if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
    auto userWorkSpace = GetUserWorkspace(workspace);
    this->syncWorkspaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(userWorkSpace), BLOCK_NUM * this->coreNum / sizeof(int32_t));
    this->yKernelGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(userWorkSpace + BLOCK_NUM * this->coreNum), this->coreNum * (this->bins + ALIGNED_NUM));
    InitGlobalMemory(this->syncWorkspaceGm, BLOCK_NUM * this->coreNum / sizeof(int32_t), 0);
    this->pipe->InitBuffer(this->syncWorkspaceQue, 1, BLOCK_NUM * this->coreNum);
    this->pipe->InitBuffer(this->y1Que, 1, (this->ubBinsLength + ALIGNED_NUM) * sizeof(int32_t));
    #endif
    this->pipe->InitBuffer(this->xQue, DOUBLE_BUFFER, this->tileDataLength * sizeof(CAST_T));
    this->pipe->InitBuffer(this->minQue, 1, BLOCK_NUM * sizeof(CAST_T));
    this->pipe->InitBuffer(this->maxQue, 1, BLOCK_NUM * sizeof(CAST_T));
    this->pipe->InitBuffer(this->yQue, 1, (this->ubBinsLength + ALIGNED_NUM) * sizeof(int32_t));
  }

  __aicore__ inline void Process() {
    ReadMinMaxValue();
    CheckMinMaxValueInt();
    if (this->bins <= STATIC_ARRAY_LENGTH) {
        int32_t outStaticArray[STATIC_ARRAY_LENGTH + ALIGNED_NUM];
        for (int32_t i = 0; i < this->bins + ALIGNED_NUM; i++) {
            outStaticArray[i] = 0;
        }
        for (int32_t i = 0; i < this->tileNum; i++) {
            CopyIn(i);
            ComputeStaticArray(this->tileDataLength, outStaticArray);
        }

        if (this->tileLeftDataLength > 0) {
            CopyIn(this->tileNum);
            ComputeStaticArray(this->tileLeftDataLength, outStaticArray);
        }
        CopyStaticArrayOut(outStaticArray);
    } else if (this->bins <= this->ubBinsLength) {
        auto yLocal = this->yQue.template AllocTensor<int32_t>();
        Duplicate<int32_t>(yLocal, 0, this->ubBinsLength + ALIGNED_NUM);
        SWaitV();
        for (int32_t i = 0; i < this->tileNum; i++) {
            CopyIn(i);
            ComputeLocalTensor(this->tileDataLength, yLocal);
        }

        if (this->tileLeftDataLength > 0) {
            CopyIn(this->tileNum);
            ComputeLocalTensor(this->tileLeftDataLength, yLocal);
        }
        this->yQue.EnQue(yLocal);
        CopyYLocalOut();
    } else {
        for (int32_t i = 0; i < this->tileNum; i++) {
            CopyIn(i);
            ComputeBigBins(i, this->tileDataLength);
        }
        if (this->tileLeftDataLength > 0) {
            CopyIn(this->tileNum);
            ComputeBigBins(tileNum, this->tileLeftDataLength);
        }
    }
    #if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
    auto ubSyncWorkspace = this->syncWorkspaceQue.template AllocTensor<int32_t>();
    SyncAll(this->syncWorkspaceGm, ubSyncWorkspace, this->coreNum);
    if (GetBlockIdx() == 0) {
        ReduceY();
    }
    this->syncWorkspaceQue.template FreeTensor<int32_t>(ubSyncWorkspace);
    #endif
  }

  __aicore__ inline void ReduceY() {
    uint64_t binsStep = this->bins / this->ubBinsLength;
    uint64_t binsLeft = this->bins - binsStep * this->ubBinsLength;
    for (uint64_t part = 0; part < binsStep; part++) {
        ReduceYPart(part, this->ubBinsLength);
    }
    if (binsLeft > 0) {
        ReduceYPart(binsStep, binsLeft);
    }
  }

  __aicore__ inline void ReduceYPart(int64_t part, int64_t length) {
    int64_t lengthAligned = (length + ALIGNED_NUM - 1) / ALIGNED_NUM * ALIGNED_NUM;
    auto y0Local = this->yQue.template AllocTensor<int32_t>();
    MTE2WaitMTE3();
    DataCopy(y0Local, this->yKernelGm[part * this->ubBinsLength], lengthAligned);
    SWaitMTE2();
    for (int64_t coreId = 1; coreId < this->coreNum; coreId++) {
        CopyYKernelIn(coreId, part, lengthAligned);
        auto y1Local = this->y1Que.template DeQue<int32_t>();
        Add(y0Local, y0Local, y1Local, lengthAligned);
        y1Que.template FreeTensor<int32_t>(y1Local);
    }
    MTE3WaitV();
    DataCopy(this->yGm[part * this->ubBinsLength], y0Local, lengthAligned);
    this->yQue.template FreeTensor<int32_t>(y0Local);
  }

  __aicore__ inline void CopyYKernelIn(int64_t coreId, int64_t part, uint64_t length) {
    auto y1Local = this->y1Que.template AllocTensor<int32_t>();
    DataCopy(y1Local, this->yKernelGm[(this->bins + ALIGNED_NUM) * coreId + this->ubBinsLength * part], length);
    this->y1Que.template EnQue(y1Local);
  }

  __aicore__ inline void ComputeBigBins(int32_t tile, int32_t computeLength) {
    auto xLocal = this->xQue.template DeQue<MTE_T>();
    auto dataLocal = xLocal.template ReinterpretCast<CAST_T>();
    COMPUTE_T minMaxLength = this->maxValue - this->minValue;

    int32_t binsStep = this->bins / this->ubBinsLength;
    int32_t binsLeft = this->bins - binsStep * this->ubBinsLength;
    for (int32_t i = 0; i <= binsStep; i++) {
        if (i == binsStep && binsLeft == 0) {
            break;
        }
        int32_t binsStart = this->ubBinsLength * i;
        int32_t binsEnd = i == binsStep ? binsStart + binsLeft : binsStart + this->ubBinsLength;
        auto yLocal = this->yQue.template AllocTensor<int32_t>();
        Duplicate<int32_t>(yLocal, 0, this->ubBinsLength);
        SWaitV();
        for (int32_t j = 0; j < computeLength; j++) {
            COMPUTE_T value = static_cast<COMPUTE_T>(dataLocal.GetValue(j));
            int32_t index = static_cast<int32_t>((value - this->minValue) * this->bins / (minMaxLength));
            if (value >= this->minValue && value <= this->maxValue && index >= binsStart && index <= binsEnd) {
                if (index == this->bins) {
                    index--;
                }
                index = index - i * this->ubBinsLength;
                int32_t count = yLocal.GetValue(index) + 1;
                yLocal.SetValue(index, count);
            }
        }
        this->yQue.EnQue(yLocal);
        auto mte3Nums = i == binsStep ? binsLeft : this->ubBinsLength;
        CopyOutPart(tile, i, mte3Nums);
    }
    this->xQue.template FreeTensor<MTE_T>(xLocal);
  }

  __aicore__ inline void ComputeStaticArray(int32_t computeLength, int32_t outStaticArray[]) {
    auto xLocal = this->xQue.template DeQue<MTE_T>();
    auto dataLocal = xLocal.template ReinterpretCast<CAST_T>();
    COMPUTE_T minMaxLength = this->maxValue - this->minValue;
    for (int32_t i = 0; i < computeLength; i++) {
        COMPUTE_T value = static_cast<COMPUTE_T>(dataLocal.GetValue(i));
        if (value >= this->minValue && value <= this->maxValue) {
            int32_t index = static_cast<int32_t>((value - this->minValue) * this->bins / (minMaxLength));
            outStaticArray[index] ++;
        }
    }
    this->xQue.template FreeTensor<MTE_T>(xLocal);
  }

  __aicore__ inline void ComputeLocalTensor(int32_t computeLength, LocalTensor<int32_t> yLocal) {
    auto xLocal = this->xQue.template DeQue<MTE_T>();
    auto dataLocal = xLocal.template ReinterpretCast<CAST_T>();
    COMPUTE_T minMaxLength = this->maxValue - this->minValue;
    for (int32_t i = 0; i < computeLength; i++) {
        COMPUTE_T value = static_cast<COMPUTE_T>(dataLocal.GetValue(i));
        if (value >= this->minValue && value <= this->maxValue) {
            int32_t index = static_cast<int32_t>((value - this->minValue) * this->bins / (minMaxLength));
            int32_t count = yLocal.GetValue(index) + 1;
            yLocal.SetValue(index, count);
        }
    }
    this->xQue.template FreeTensor<MTE_T>(xLocal);
  }

  __aicore__ inline void CopyIn(int32_t tileOffset) {
    auto xLocal = this->xQue.template AllocTensor<MTE_T>();
    int32_t start = tileOffset * this->tileDataLength * (sizeof(CAST_T) / sizeof(MTE_T));
    int32_t copy_bytes = this->tileDataLength * sizeof(CAST_T);
    if (tileOffset == this->tileNum) {
        copy_bytes = this->tileLeftDataLength * sizeof(CAST_T);
    }
    #if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
    if (copy_bytes % BLOCK_NUM != 0) {
        copy_bytes += BLOCK_NUM - copy_bytes % BLOCK_NUM;
    }
    DataCopy(xLocal, this->xGm[start], copy_bytes / sizeof(MTE_T));
    #else
    DataCopyParams copyParams{1, static_cast<uint16_t>(copy_bytes), 0, 0};
    DataCopyPadParams padParams{true, 0, 0, 0};
    DataCopyPad(xLocal, this->xGm[start], copyParams, padParams);
    #endif
    this->xQue.EnQue(xLocal);
  }

 __aicore__ inline void CopyStaticArrayOut(int32_t outStaticArray[]) {
    outStaticArray[this->bins - 1] = outStaticArray[this->bins - 1] + outStaticArray[this->bins];
    outStaticArray[this->bins] = 0;
    auto yLocal = this->yQue.template AllocTensor<int32_t>();
    for (int32_t i = 0; i < this->bins + ALIGNED_NUM; i++) {
        yLocal.SetValue(i, outStaticArray[i]);
    }
    #if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
    DataCopy(this->yKernelGm[(this->bins + ALIGNED_NUM) * GetBlockIdx()], yLocal, this->bins + ALIGNED_NUM);
    #else
    DataCopyParams copyParams{1, static_cast<uint16_t>(this->bins * sizeof(int32_t)), 0, 0};
    SetAtomicAdd<int32_t>();
    DataCopyPad(this->yGm, yLocal, copyParams);
    SetAtomicNone();
    #endif
    this->yQue.template FreeTensor<int32_t>(yLocal);
 }

 __aicore__ inline void CopyYLocalOut() {
    auto yLocal = this->yQue.template DeQue<int32_t>();
    int32_t count = yLocal.GetValue(this->bins - 1) + yLocal.GetValue(this->bins);
    yLocal.SetValue(this->bins - 1, count);
    #if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
    yLocal.SetValue(this->bins, 0);
    DataCopy(this->yKernelGm[(this->bins + ALIGNED_NUM) * GetBlockIdx()], yLocal, this->bins + ALIGNED_NUM);
    #else
    DataCopyParams copyParams{1, static_cast<uint16_t>(this->bins * sizeof(int32_t)), 0, 0};
    SetAtomicAdd<int32_t>();
    DataCopyPad(this->yGm, yLocal, copyParams);
    SetAtomicNone();
    #endif
    this->yQue.template FreeTensor<int32_t>(yLocal);
 }

 __aicore__ inline void CopyOutPart(int32_t tile, int32_t parts, int32_t length) {
    auto yLocal = this->yQue.template DeQue<int32_t>();
    int32_t gmStart = parts * this->ubBinsLength;
    #if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
    if (length % ALIGNED_NUM != 0) {
        length += ALIGNED_NUM - (length % ALIGNED_NUM);
    }
    if (tile != 0) {
        auto y1Local = this->y1Que.template AllocTensor<int32_t>();
        DataCopy(y1Local, this->yKernelGm[(this->bins + ALIGNED_NUM) * GetBlockIdx() + gmStart], length);
        VWaitMTE2();
        Add(yLocal, yLocal, y1Local, length);
        this->y1Que.template FreeTensor<int32_t>(y1Local);
        MTE3WaitV();
    }
    DataCopy(this->yKernelGm[(this->bins + ALIGNED_NUM) * GetBlockIdx() + gmStart], yLocal, length);
    #else
    DataCopyParams copyParams{1, static_cast<uint16_t>(length * sizeof(int32_t)), 0, 0};
    SetAtomicAdd<int32_t>();
    DataCopyPad(this->yGm[gmStart], yLocal, copyParams);
    SetAtomicNone();
    #endif
    this->yQue.template FreeTensor<int32_t>(yLocal);
 }

 __aicore__ inline void ReadMinMaxValue() {
    auto minLocalMte = this->minQue.template AllocTensor<MTE_T>();
    auto maxLocalMte = this->maxQue.template AllocTensor<MTE_T>();
    #if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
    DataCopy(minLocalMte, this->minGm, BLOCK_NUM);
    DataCopy(maxLocalMte, this->maxGm, BLOCK_NUM);
    #else
    DataCopyParams copyParams{1, static_cast<uint16_t>(sizeof(CAST_T)), 0, 0};
    DataCopyPadParams padParams{true, 0, 0, 0};
    DataCopyPad(minLocalMte, this->minGm, copyParams, padParams);
    DataCopyPad(maxLocalMte, this->maxGm, copyParams, padParams);
    #endif
    SWaitMTE2();
    auto minLocal = minLocalMte.template ReinterpretCast<CAST_T>();
    auto maxLocal = maxLocalMte.template ReinterpretCast<CAST_T>();

    this->minValue = static_cast<COMPUTE_T>(minLocal.GetValue(0));
    this->maxValue = static_cast<COMPUTE_T>(maxLocal.GetValue(0));
    this->minQue.template FreeTensor<MTE_T>(minLocalMte);
    this->maxQue.template FreeTensor<MTE_T>(maxLocalMte);
 }

__aicore__ inline void CheckMinMaxValueInt() {
    if (this->minValue == this->maxValue) {
        this->minValue = this->minValue - 1;
        this->maxValue = this->maxValue + 1;
    }
}

__aicore__ inline void SWaitMTE2() {
  event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
  SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
  WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
}

__aicore__ inline void SWaitV() {
  event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
  SetFlag<HardEvent::V_S>(eventIDVToS);
  WaitFlag<HardEvent::V_S>(eventIDVToS);
}

__aicore__ inline void MTE3WaitV() {
  event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
  SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
  WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
}

__aicore__ inline void MTE2WaitMTE3() {
  event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
  SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
  WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
}

__aicore__ inline void VWaitMTE2() {
  event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
  SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
  WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
}

protected:
  COMPUTE_T minValue;
  COMPUTE_T maxValue;
  int64_t bins;
  int64_t ubBinsLength;
  int64_t tileNum;
  int64_t coreTotalDataLength;
  int64_t tileDataLength;
  int64_t tileLeftDataLength;
  int64_t coreNum;

  GlobalTensor<MTE_T> xGm;
  GlobalTensor<MTE_T> minGm;
  GlobalTensor<MTE_T> maxGm;
  GlobalTensor<int32_t> yGm;
  GlobalTensor<int32_t> syncWorkspaceGm;
  GlobalTensor<int32_t> yKernelGm;

  TPipe* pipe;
  TQue<TPosition::VECIN, DOUBLE_BUFFER> xQue;
  TQue<TPosition::VECIN, 1> minQue;
  TQue<TPosition::VECIN, 1> maxQue;
  TQue<TPosition::VECIN, 1> syncWorkspaceQue;
  TQue<TPosition::VECIN, 1> y1Que;
  TQue<TPosition::VECOUT, 1> yQue;
};
} // HistogramV2Scalar
#endif  // HISTOGRAM_V2_SCALAR