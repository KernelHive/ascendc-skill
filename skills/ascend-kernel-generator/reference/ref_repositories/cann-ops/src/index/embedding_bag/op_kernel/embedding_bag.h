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
 * \file embedding_bag.h
 * \brief
 */

#ifndef EMBEDDING_BAG_H_DETERMINIST_H
#define EMBEDDING_BAG_H_DETERMINIST_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

constexpr uint32_t TENSOR_COUNT = 8;
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t UINT8_BITS = 8;
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t ELE_NUM_PER_REPEAT = 64;
constexpr uint32_t MODE_SUM = 0;
constexpr uint32_t MODE_MEAN = 1;
constexpr uint32_t MODE_MAX = 2;
constexpr uint32_t NUM_PER_BLOCK = 8;
constexpr uint32_t NUM_PER_BLOCK_16 = 16;
extern constexpr uint32_t TILINGKEY_BF16 = 3;
namespace AscendC {
template<typename T, typename DTYPE>
class EmbeddingBag {
public:
__aicore__ inline EmbeddingBag() = delete;
__aicore__ inline EmbeddingBag(GM_ADDR inputTensors[TENSOR_COUNT], const EmbeddingBagTilingData &tiling, TPipe &pipe)
{
    InitParams(inputTensors, tiling, pipe);
}

__aicore__ inline void Process()
{
    CopyIn();
    Compute();
    CopyOut(); 
}

private:

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 a, T2 b) {
    return (a + b - 1) / b;
};
template <typename T1, typename T2>
__aicore__ inline T1 CeilAlign(T1 a, T2 b) {
    return (a + b - 1) / b * b;
};
__aicore__ inline float intToFloatBits(int i) {
// Use a union to reinterpret the bits of an int as a float
    union {
        int i;
        float f;
    } u;
    u.i = i;
    return u.f;
}

__aicore__ inline void SyncM2toV() {
    event_t eventId = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);
};

__aicore__ inline void SyncM2toS() {
    event_t eventId = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventId);
    WaitFlag<HardEvent::MTE2_S>(eventId);
};

__aicore__ inline void SyncVtoM3() {
    event_t eventId = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventId);
    WaitFlag<HardEvent::V_MTE3>(eventId);
}

__aicore__ inline void SyncM3toS() {
    event_t eventId = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventId);
    WaitFlag<HardEvent::MTE3_S>(eventId);
};

__aicore__ inline void SyncVtoS() {
    event_t eventId = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventId);
    WaitFlag<HardEvent::V_S>(eventId);
};

__aicore__ inline void InitParams(GM_ADDR inputTensors[TENSOR_COUNT], const EmbeddingBagTilingData &tiling, TPipe &pipe)
{
    pipe_ = &pipe;
    auto blockIdx_ = GetBlockIdx();
    isLastBlock_ = blockIdx_ == GetBlockNum() - 1;
    if (isLastBlock_) {
        offsetNum_ = tiling.tailOffsetNum + 1;
    } else {
        offsetNum_ = tiling.formerOffsetNum + 1;
    }
    offsetNumCou_ = CeilAlign(offsetNum_, NUM_PER_BLOCK);
    formerOffsetNum_ = tiling.formerOffsetNum;
    computeRepTime_ = tiling.computeRepTime;
    numEmbeddings_ = tiling.numEmbeddings;
    indicesMaxMoveLength_ = tiling.indicesMaxMoveLength;
    paddingIdx_ = tiling.paddingIdx;
    numIndices_ = tiling.numIndices;
    mode_ = tiling.mode;
    hasPerSampleWeights_ = tiling.hasPerSampleWeights;

    allocatedSpaceSize_ = CeilAlign(computeRepTime_, ELE_NUM_PER_REPEAT);

    maskSize_ = CeilDiv(allocatedSpaceSize_, UINT8_BITS);
    offset_ = blockIdx_ * formerOffsetNum_;

    weightGm_.SetGlobalBuffer((__gm__ T*)inputTensors[0]);
    indicesGm_.SetGlobalBuffer((__gm__ DTYPE*)inputTensors[1]);
    offsetGm_.SetGlobalBuffer((__gm__ DTYPE*)inputTensors[2]);
    perSampleWeightsGm_.SetGlobalBuffer((__gm__ T*)inputTensors[3]);
    yGm_.SetGlobalBuffer((__gm__ T*)inputTensors[4]);
    offset2bagGm_.SetGlobalBuffer((__gm__ DTYPE*)inputTensors[5]);
    bagSizeGm_.SetGlobalBuffer((__gm__ DTYPE*)inputTensors[6]);
    maxIndicesGm_.SetGlobalBuffer((__gm__ DTYPE*)inputTensors[7]);

    if (mode_ == MODE_MAX) {    
        InitMaxBuffers(pipe);
    } else {
        InitOtherBuffers(pipe);
    }

#if __CCE_AICORE__ < 220
    auto moveOffset = formerOffsetNum_ * numEmbeddings_ * blockIdx_;
    auto totalNumber = (offsetNum_ - 1) * numEmbeddings_;
    auto moveNum = CeilDiv(totalNumber, allocatedSpaceSize_);
    Duplicate<T>(yDataLocal, 0, allocatedSpaceSize_);
    for(auto idx = 0; idx < moveNum; idx++) {
        auto length = allocatedSpaceSize_;
        auto zeroOffset = moveOffset +idx * allocatedSpaceSize_;
        if (idx == moveNum - 1) {
            length = CeilAlign(totalNumber - idx * allocatedSpaceSize_, NUM_PER_BLOCK);
        }
        DataCopy(yGm_[zeroOffset], yDataLocal, length);
    }
#endif
}

__aicore__ inline void InitMaxBuffers(TPipe &pipe)
{
    pipe.InitBuffer(weightQue_, BUFFER_NUM, allocatedSpaceSize_ * sizeof(float));
    pipe.InitBuffer(offsetQue_, BUFFER_NUM, offsetNumCou_ * sizeof(DTYPE));
    pipe.InitBuffer(indicesQue_, BUFFER_NUM, indicesMaxMoveLength_ * sizeof(DTYPE));

    pipe.InitBuffer(bagSizeQue_, BUFFER_NUM, offsetNumCou_ * sizeof(DTYPE));
    pipe.InitBuffer(yQue_, BUFFER_NUM, allocatedSpaceSize_ * sizeof(float));
    pipe.InitBuffer(offset2bagQue_, BUFFER_NUM, indicesMaxMoveLength_ * sizeof(DTYPE));
    pipe.InitBuffer(maxIndicesQue_, BUFFER_NUM, allocatedSpaceSize_ * sizeof(DTYPE));

    pipe.InitBuffer(maskBuf, maskSize_);  
    maskTensor = maskBuf.Get<uint8_t>(maskSize_);

    weightDataLocal = weightQue_.AllocTensor<float>();  
    indicesDataLocal = indicesQue_.AllocTensor<DTYPE>();

    yDataLocal = yQue_.AllocTensor<float>();
    bagSizeDataLocal = bagSizeQue_.AllocTensor<DTYPE>();
    offset2bagDataLocal = offset2bagQue_.AllocTensor<DTYPE>();
    maxIndicesDataLocal = maxIndicesQue_.AllocTensor<DTYPE>();
    maxIndicesDataLocalT = maxIndicesDataLocal.template ReinterpretCast<float>();
}

__aicore__ inline void InitOtherBuffers(TPipe &pipe)
{
    pipe.InitBuffer(weightQue_, BUFFER_NUM, allocatedSpaceSize_ * sizeof(float));
    pipe.InitBuffer(offsetQue_, BUFFER_NUM, offsetNumCou_ * sizeof(DTYPE));
    pipe.InitBuffer(indicesQue_, BUFFER_NUM, indicesMaxMoveLength_ * sizeof(DTYPE));

    if(mode_ == MODE_SUM && hasPerSampleWeights_) { 
        pipe.InitBuffer(perSamplerWEightQue_, BUFFER_NUM, indicesMaxMoveLength_ * sizeof(T));
        perSamplerWeightDataLocal = perSamplerWEightQue_.AllocTensor<T>();
    }

    pipe.InitBuffer(yQue_, BUFFER_NUM, allocatedSpaceSize_ * sizeof(float));
    pipe.InitBuffer(bagSizeQue_, BUFFER_NUM, offsetNumCou_ * sizeof(DTYPE));
    pipe.InitBuffer(offset2bagQue_, BUFFER_NUM, indicesMaxMoveLength_ * sizeof(DTYPE));

    weightDataLocal = weightQue_.AllocTensor<float>();

    indicesDataLocal = indicesQue_.AllocTensor<DTYPE>();

    yDataLocal = yQue_.AllocTensor<float>();
    offset2bagDataLocal = offset2bagQue_.AllocTensor<DTYPE>();
    bagSizeDataLocal = bagSizeQue_.AllocTensor<DTYPE>();
}

template <typename C>
__aicore__ inline void GMToUB(GlobalTensor<C>& gm_, LocalTensor<C>& tensor_, int64_t copyOffset_,
                            int32_t moveLength_, int32_t realLength_)
{
#if __CCE_AICORE__ < 220
    DataCopy(tensor_, gm_[copyOffset_], moveLength_);
#else
    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(realLength_ * sizeof(C)), 0, 0, 0};
    DataCopyPadExtParams<C> padParams = {true, 0, 0, 0};
    DataCopyPad(tensor_, gm_[copyOffset_], copyParams, padParams);
#endif  
}

template <typename C>
__aicore__ inline void UBToGM(GlobalTensor<C>& gm_, LocalTensor<C>& tensor_, int64_t copyOffset_,
                            int32_t realLength_)
{
#if __CCE_AICORE__ < 220
    auto numPerBlock = BLOCK_BYTES / sizeof(C);
    auto alignLength = realLength_ / numPerBlock * numPerBlock;
    if (alignLength != 0) {
         DataCopy(gm_[copyOffset_], tensor_, alignLength);
    }
    if(alignLength != realLength_) {
        uint64_t mask0 = (1ul << numPerBlock) - (1ul << (realLength_ - alignLength));
        uint64_t mask[2] = {mask0, 0};
        Duplicate<C>(tensor_[alignLength], 0, mask, 1, 1, 1);
        SetAtomicAdd<C>();
        DataCopy(gm_[copyOffset_ + alignLength], tensor_[alignLength], numPerBlock);
        SetAtomicNone();   
    }
#else
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(realLength_ * sizeof(C)), 0, 0, 0};
    DataCopyPad(gm_[copyOffset_], tensor_, copyParams);
#endif
} 

__aicore__ inline void CopyIn()
{
    LocalTensor<DTYPE> dataLocal = offsetQue_.AllocTensor<DTYPE>();
    int64_t realOffsetNum = offsetNum_;
    if (isLastBlock_) {
        realOffsetNum -= 1;
    }
    GMToUB(offsetGm_, dataLocal, offset_, offsetNumCou_, realOffsetNum);
    if (isLastBlock_) {
        dataLocal.SetValue(realOffsetNum, numIndices_);
    }

    if (GetBlockIdx() == 0) {
        dataLocal.SetValue(0, 0);
    }
    offsetQue_.EnQue(dataLocal);
}

__aicore__ inline void MoveAndCompute(DTYPE length, DTYPE startNumber, int i, bool flag){
    GMToUB(indicesGm_, indicesDataLocal, startNumber, indicesMaxMoveLength_, length);

    if (mode_ == MODE_SUM && hasPerSampleWeights_) {
        GMToUB(perSampleWeightsGm_, perSamplerWeightDataLocal, startNumber, indicesMaxMoveLength_, length); 
    }
    if (weightOffset_ == 0) { 
        Duplicate<DTYPE>(offset2bagDataLocal, static_cast<DTYPE>(offset_ + i), length);
        PipeBarrier<PIPE_ALL>();
        UBToGM(offset2bagGm_, offset2bagDataLocal, startNumber, length);
    }
    for(int j = 0; j < length; j++){
        if(indicesDataLocal.GetValue(j) != paddingIdx_){
            auto offset = indicesDataLocal.GetValue(j) * numEmbeddings_ + weightOffset_;
            if (mode_ == MODE_MAX) {
                if (isFirstMaxIndices_) {
                    GMToUB(weightGm_, yDataLocal, offset, allocatedSpaceSize_, realCountNum_);
                    SyncM2toS();
                    Duplicate<DTYPE>(maxIndicesDataLocal, indicesDataLocal.GetValue(j), realCountNum_);
                    isFirstMaxIndices_ = false;
                } else {
                    MaxWeight(offset, j);
                }
            } else {
                AddWeight(offset, j);
            }
            bagSize_++;
        }
    }
    if (flag) {
        TensorCopyOut();
    }
}

__aicore__ inline void TensorCopyOut()
{
    if (mode_ == MODE_MEAN && bagSize_ > 0) {
        float n = 1 / static_cast<float>(bagSize_);
        Muls(yDataLocal, yDataLocal, n, realCountNum_);
    }
    UBToGM(yGm_, yDataLocal, yOffset_, realCountNum_);
    if (mode_ == MODE_MAX) {
        UBToGM(maxIndicesGm_, maxIndicesDataLocal, yOffset_, realCountNum_);
    }
    SyncM3toS();
}

__aicore__ inline void Compute()
{
    offsetDataLocal = offsetQue_.DeQue<DTYPE>();
    auto count = CeilDiv(numEmbeddings_, allocatedSpaceSize_);
    for (int i = 0; i < offsetNum_ - 1; i++) {
        for(auto j = 0; j < count; j++){
            realCountNum_ = allocatedSpaceSize_;
            if (j == count - 1) {
                realCountNum_ = numEmbeddings_ - j * allocatedSpaceSize_;
            }
            yOffset_ = (offset_ + i) * numEmbeddings_ + j *allocatedSpaceSize_;
            weightOffset_ = j * allocatedSpaceSize_;
            bagSize_ = 0;
            isFirstMaxIndices_ = true;
            Duplicate<float>(yDataLocal, 0, realCountNum_);
            DTYPE startNumber = offsetDataLocal.GetValue(i);
            DTYPE endNumber = offsetDataLocal.GetValue(i + 1);
            if (mode_ == MODE_MAX) {
                Duplicate<DTYPE>(maxIndicesDataLocal, 0, realCountNum_);
            }

            while (endNumber - startNumber > indicesMaxMoveLength_){
                MoveAndCompute(indicesMaxMoveLength_, startNumber, i, false);
                startNumber += indicesMaxMoveLength_;      
            }
            auto length = endNumber - startNumber;
            MoveAndCompute(length, startNumber, i, true);
        }
        if (mode_ != MODE_SUM) {
            bagSizeDataLocal.SetValue(i, bagSize_);
        } else {
            bagSizeDataLocal.SetValue(i, 0);
        }
    }
    FreeUBTensor();
}

__aicore__ inline void FreeUBTensor()
{
    if (mode_ == MODE_MAX) {
        maxIndicesQue_.FreeTensor(maxIndicesDataLocal);
    }

    if (mode_ == MODE_SUM && hasPerSampleWeights_) {
        perSamplerWEightQue_.FreeTensor(perSamplerWeightDataLocal);
    }

    bagSizeQue_.EnQue(bagSizeDataLocal);
    offsetQue_.FreeTensor(offsetDataLocal);
    indicesQue_.FreeTensor(indicesDataLocal);
    weightQue_.FreeTensor(weightDataLocal);

    yQue_.FreeTensor(yDataLocal);
    offset2bagQue_.FreeTensor(offset2bagDataLocal);
}

__aicore__ inline void AddWeight(const int64_t offset, const int32_t idx)
{
    GMToUB(weightGm_, weightDataLocal, offset, allocatedSpaceSize_, realCountNum_);
    SyncM2toV();

    if (mode_ == MODE_SUM && hasPerSampleWeights_) {
        Muls(weightDataLocal, weightDataLocal, perSamplerWeightDataLocal.GetValue(idx), realCountNum_);
    }
    Add(yDataLocal, yDataLocal, weightDataLocal, realCountNum_);
    SyncVtoS();
}

__aicore__ inline void MaxWeight(const int64_t offset, const int32_t idx)
{
    GMToUB(weightGm_, weightDataLocal, offset, allocatedSpaceSize_, realCountNum_);
    SyncM2toV();
    auto num = CeilAlign(realCountNum_, ELE_NUM_PER_REPEAT);
    auto number = intToFloatBits(indicesDataLocal.GetValue(idx));
    Compare(maskTensor, yDataLocal, weightDataLocal, CMPMODE::GE, num);
    Select(yDataLocal, maskTensor, yDataLocal, weightDataLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, num);
    Select(maxIndicesDataLocalT, maskTensor, maxIndicesDataLocalT, number, SELMODE::VSEL_TENSOR_SCALAR_MODE, num);
    SyncVtoS();
}

__aicore__ inline void CopyOut()
{
    LocalTensor<DTYPE> dataLocal = bagSizeQue_.DeQue<DTYPE>();
    UBToGM(bagSizeGm_, dataLocal, offset_, (offsetNum_ - 1));
    if (mode_ != MODE_MAX) {
       UBToGM(maxIndicesGm_, dataLocal, offset_, (offsetNum_ - 1));
    }
    bagSizeQue_.FreeTensor(dataLocal);
}

private:
    TPipe* pipe_;
    GlobalTensor<T> weightGm_;
    GlobalTensor<DTYPE> indicesGm_;
    GlobalTensor<DTYPE> offsetGm_;
    GlobalTensor<T> perSampleWeightsGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<DTYPE> offset2bagGm_;
    GlobalTensor<DTYPE> bagSizeGm_;
    GlobalTensor<DTYPE> maxIndicesGm_;

    TQue<TPosition::VECIN, BUFFER_NUM> offsetQue_;
    TQue<TPosition::VECIN, BUFFER_NUM> indicesQue_;
    TQue<TPosition::VECIN, BUFFER_NUM> weightQue_;
    TQue<TPosition::VECIN, BUFFER_NUM> perSamplerWEightQue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> yQue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> offset2bagQue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> bagSizeQue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> maxIndicesQue_;

    LocalTensor<T> weightDataLocal;
    LocalTensor<DTYPE> offsetDataLocal;
    LocalTensor<DTYPE> indicesDataLocal;
    LocalTensor<T> perSamplerWeightDataLocal;
    LocalTensor<T> yDataLocal;
    LocalTensor<DTYPE> offset2bagDataLocal;
    LocalTensor<DTYPE> bagSizeDataLocal;
    LocalTensor<DTYPE> maxIndicesDataLocal;
    LocalTensor<float> maxIndicesDataLocalT;

    TBuf<TPosition::VECCALC> maskBuf;
    LocalTensor<uint8_t> maskTensor;

    int64_t offsetNum_ = 0;
    int64_t offsetNumCou_ = 0;
    int64_t numEmbeddings_ = 0;
    int64_t computeRepTime_ = 0;
    int64_t indicesMaxMoveLength_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t paddingIdx_ = 0;
    int64_t numIndices_ = 0;
    int64_t mode_ = 0;
    int64_t formerOffsetNum_ = 0;
    int64_t hasPerSampleWeights_ = 0;
    int64_t tilingKey_ = 0;
    int64_t offset_ = 0;
    int64_t realCountNum_ = 0;
    int64_t weightOffset_ = 0;
    int64_t yOffset_ = 0;
    int32_t allocatedSpaceSize_ = 0;
    bool isLastBlock_ = false;
    DTYPE bagSize_ = 0;
    uint32_t maskSize_ = 0;
    bool isFirstMaxIndices_ = false;
};
}

#endif // EMBEDDING_BAG_H_DETERMINIST_H