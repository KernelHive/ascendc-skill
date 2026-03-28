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
 * \file embedding_dense_grad_v2_small_dim.h
 * \brief
 */

#ifndef EMBEDDING_DENSE_GRAD_V2_H_SMALL_DIM_H
#define EMBEDDING_DENSE_GRAD_V2_H_SMALL_DIM_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

constexpr int64_t SORT_QUE_NUM = 3;

namespace AscendC {
template<typename T>
class EmbeddingDenseGradV2SmallDimKernel {
public:
__aicore__ inline EmbeddingDenseGradV2SmallDimKernel() = delete;
__aicore__ inline EmbeddingDenseGradV2SmallDimKernel(GM_ADDR grad, GM_ADDR indices, GM_ADDR backProps, GM_ADDR workSpace, 
                                                     const EmbeddingDenseGradV2TilingData &tiling, TPipe &pipe)
{
    // 1. 初始化tiling参数
    InitParams(tiling);
    // 2. 初始化que
    InitBuffers(pipe);
    // 3. 设置gm地址
    SetGmAddr(grad, indices, backProps, workSpace, tiling);
}
__aicore__ inline void Process()
{
    for (int i = 0; i < copyTime_ - 1; i++) {
        CopyIn(i, true);
        ComputeAndCopyOut(true);
    }
    CopyIn(copyTime_ - 1, false);
    ComputeAndCopyOut(false);
}

private:
__aicore__ inline void InitParams(const EmbeddingDenseGradV2TilingData &tiling)
{
    coreIdx_ = GetBlockIdx();
    ResetScaleCount();
    numWeights_ = tiling.params.numWeights;
    embeddingDim_ = tiling.params.embeddingDim;
    scaleGradByFreq_ = tiling.params.scaleGradByFreq;
    paddingIdx_ = tiling.params.paddingIdx;
    maxRowInUb_ = tiling.smallDimTiling.maxRowInUb;
    partNum_ = tiling.smallDimTiling.partNum;
    if (coreIdx_ < GetBlockNum() - 1) {
        rowNum_ = tiling.smallDimTiling.formerCopyRow;
        copyTime_ = tiling.smallDimTiling.formerCopyTime;
        lastRowNum_ = tiling.smallDimTiling.formerLastRow;
    } else {
        rowNum_ = tiling.smallDimTiling.tailCopyRow;
        copyTime_ = tiling.smallDimTiling.tailCopyTime;
        lastRowNum_ = tiling.smallDimTiling.tailLastRow;
    }
    copyRowNum_ = maxRowInUb_ > rowNum_ ? rowNum_ : maxRowInUb_;
}

__aicore__ inline void InitBuffers(TPipe &pipe)
{
    uint64_t gradAlignNum = BLOCK_SIZE / sizeof(T);
    embeddingDimAlign_ = (embeddingDim_ + gradAlignNum - 1) / gradAlignNum * gradAlignNum;
    gradAlignNum = embeddingDimAlign_ * copyRowNum_;
    idxAlignNum_ = (copyRowNum_ + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE; // 32个数align
    pipe.InitBuffer(indicesQue_, BUFFER_NUM, idxAlignNum_ * sizeof(int));
    pipe.InitBuffer(idxBuf_, idxAlignNum_ * sizeof(int));
    pipe.InitBuffer(tmp2Que_, BUFFER_NUM, idxAlignNum_ * sizeof(T));
    pipe.InitBuffer(tmpQue_, BUFFER_NUM, idxAlignNum_ * SORT_QUE_NUM * sizeof(T));
    pipe.InitBuffer(sortIndicesQue_, BUFFER_NUM, idxAlignNum_ * SORT_QUE_NUM * sizeof(int));
    pipe.InitBuffer(gradQue_, BUFFER_NUM, gradAlignNum * sizeof(T));
    pipe.InitBuffer(addResQue_, BUFFER_NUM, embeddingDim_ * sizeof(T));
}

__aicore__ inline void SetGmAddr(GM_ADDR grad, GM_ADDR indices, GM_ADDR backProps, GM_ADDR workSpace, 
                                 const EmbeddingDenseGradV2TilingData &tiling)
{
    uint64_t indicesAddrOffset = tiling.smallDimTiling.formerCopyRow * coreIdx_;
    uint64_t gradAddrOffset = tiling.smallDimTiling.formerCopyRow * embeddingDim_ * coreIdx_;
    gradGm_.SetGlobalBuffer((__gm__ T*)grad + gradAddrOffset);
    indicesGm_.SetGlobalBuffer((__gm__ int*)indices + indicesAddrOffset);
    outputGm_.SetGlobalBuffer((__gm__ T*)backProps);
    idxNumGm_.SetGlobalBuffer((__gm__ float*)workSpace);
}

__aicore__ inline void CopyIn(const uint32_t progress, bool formerFlag)
{
    LocalTensor<int> indicesLocal = indicesQue_.AllocTensor<int>();
    LocalTensor<T> gradLocal = gradQue_.AllocTensor<T>();
    uint64_t idxAddrOffset = copyRowNum_ * progress;
    uint64_t gradAddrOffset = copyRowNum_ * embeddingDim_ * progress;
    uint64_t copyRow = formerFlag ? copyRowNum_ : lastRowNum_;
    DataCopyExtParams idxCopyParams{1, static_cast<uint32_t>(copyRow * sizeof(int)), 0, 0, 0};
    DataCopyExtParams gradCopyParams{static_cast<uint16_t>(copyRow), static_cast<uint32_t>(embeddingDim_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams idxpadParams{true, 0, 0, 0};
    DataCopyPadExtParams gradPadParams{true, 0, static_cast<uint8_t>(embeddingDimAlign_ - embeddingDim_), (T)(0.0f)};
    DataCopyPad(indicesLocal, indicesGm_[idxAddrOffset], idxCopyParams, idxpadParams);
    DataCopyPad(gradLocal, gradGm_[gradAddrOffset], gradCopyParams, gradPadParams);
    indicesQue_.EnQue<int>(indicesLocal);
    gradQue_.EnQue<T>(gradLocal);
}

__aicore__ inline void SortIndices(bool formerFlag)
{
    LocalTensor<int> indicesLocal = indicesQue_.DeQue<int>();
    LocalTensor<int32_t> idxLocal = idxBuf_.Get<int32_t>();
    LocalTensor<T> tmpLocal = tmpQue_.AllocTensor<T>();
    LocalTensor<T> tmp2Local = tmp2Que_.AllocTensor<T>();
    LocalTensor<T> sortResLocal = sortIndicesQue_.AllocTensor<T>();

    uint32_t idxNum = formerFlag ? copyRowNum_ : lastRowNum_;
    uint32_t idxAlign32 = (idxNum + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    uint32_t sortRepeatTimes = idxAlign32 / BLOCK_SIZE;
    // 1. cast indices to fp32
    Duplicate<T>(tmp2Local, -1, idxAlign32);
    Cast(tmp2Local, indicesLocal, RoundMode::CAST_ROUND, idxNum);
    // 2. create posIdx
    CreateVecIndex<int32_t>(idxLocal, 0U, idxNum);
    LocalTensor<uint32_t> idxULocal = idxLocal.ReinterpretCast<uint32_t>();
    // 3. sort indices
    Sort<float, true>(sortResLocal, tmp2Local, idxULocal, tmpLocal, sortRepeatTimes);
    // 4. extract res
    Extract(tmp2Local, idxULocal, sortResLocal, sortRepeatTimes);
    // 5. cast sort res to int
    Cast(indicesLocal, tmp2Local, RoundMode::CAST_ROUND, idxNum);
    indicesQue_.EnQue<int>(indicesLocal);
    tmpQue_.FreeTensor<T>(tmpLocal);
    tmp2Que_.FreeTensor<T>(tmp2Local);
    sortIndicesQue_.FreeTensor<T>(sortResLocal);
}

__aicore__ inline void ResetAddQue()
{
    LocalTensor<T> addLocal = addResQue_.AllocTensor<T>();
    Duplicate<T>(addLocal, 0.0, embeddingDim_);
    addResQue_.EnQue<T>(addLocal);
}

__aicore__ inline void FreeAddQue()
{
    LocalTensor<T> addLocal = addResQue_.DeQue<T>();
    addResQue_.FreeTensor<T>(addLocal);
}

__aicore__ inline void ResetScaleCount()
{
    scaleCount_ = 0;
}

__aicore__ inline void AtomicAddInUb(LocalTensor<T> &gradLocal, const int addrOffset)
{
    ++scaleCount_;
    int offset  = addrOffset * embeddingDimAlign_;
    LocalTensor<T> addLocal = addResQue_.DeQue<T>();
    Add(addLocal, addLocal, gradLocal[offset], embeddingDim_);
    addResQue_.EnQue<T>(addLocal);
}

__aicore__ inline void ComputeAndCopyOut(bool formerFlag)
{
    // 1. sort
    SortIndices(formerFlag);
    // 2. process one row
    LocalTensor<int> indicesLocal = indicesQue_.DeQue<int>();
    LocalTensor<int> idxLocal = idxBuf_.Get<int>();
    LocalTensor<T> gradLocal = gradQue_.DeQue<T>();
    uint64_t processRowNum = formerFlag ? copyRowNum_ : lastRowNum_;
    ResetAddQue();
    ResetScaleCount();
    int rightPtr;
    for (int i = 0; i < processRowNum;) {
        int posIdx = idxLocal.GetValue(i);
        int currentId = indicesLocal.GetValue(i);
        if (currentId == paddingIdx_) {
            ++i;
            continue;
        }
        AtomicAddInUb(gradLocal, posIdx);
        for (rightPtr = i + 1; rightPtr < processRowNum; rightPtr++) {
            int nextId = indicesLocal.GetValue(rightPtr);
            if (currentId != nextId) {
                break;
            }
            int nextPos = idxLocal.GetValue(rightPtr);
            AtomicAddInUb(gradLocal, nextPos);
        }
        i = rightPtr;
        CopyOut(currentId);
    }
    FreeAddQue();
    indicesQue_.FreeTensor<int>(indicesLocal);
    gradQue_.FreeTensor<T>(gradLocal);
}

__aicore__ inline void CopyOut(int indice)
{
    LocalTensor<T> addLocal = addResQue_.DeQue<T>();
    uint32_t gmAddrOffset = indice * embeddingDim_;
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(embeddingDim_ * sizeof(T)), 0, 0, 0};
    SetAtomicAdd<T>();
    DataCopyPad(outputGm_[gmAddrOffset], addLocal, copyParams);
    SetAtomicNone();
    if (scaleGradByFreq_) {
        LocalTensor<float> tmpLocal = tmpQue_.AllocTensor<float>();
        tmpLocal.SetValue(0, static_cast<float>((int)scaleCount_));
        DataCopyExtParams scaleCopyParams{1, sizeof(uint32_t), 0, 0, 0};
        SetAtomicAdd<float>();
        DataCopyPad(idxNumGm_[indice], tmpLocal, scaleCopyParams);
        SetAtomicNone();
        tmpQue_.FreeTensor<float>(tmpLocal);
    }
    addResQue_.FreeTensor<T>(addLocal);
    ResetAddQue();
    ResetScaleCount();
}

private:
uint64_t coreIdx_;
uint64_t partNum_;
uint64_t rowNum_;
uint64_t copyTime_;
uint64_t lastRowNum_;
uint64_t maxRowInUb_;
uint64_t copyRowNum_;
int64_t paddingIdx_;

uint64_t embeddingDim_;
uint64_t numWeights_;
bool scaleGradByFreq_;
uint64_t idxAlignNum_;
uint32_t scaleCount_;
uint64_t embeddingDimAlign_;

int lastIndices_;
bool switchId_;

private:
GlobalTensor<T> gradGm_;
GlobalTensor<T> outputGm_;
GlobalTensor<int> indicesGm_;
GlobalTensor<float> idxNumGm_;

TQue<TPosition::VECIN, BUFFER_NUM> gradQue_;
TQue<TPosition::VECIN, BUFFER_NUM> indicesQue_;
TQue<TPosition::VECIN, BUFFER_NUM> sortIndicesQue_;
TQue<TPosition::VECIN, BUFFER_NUM> tmpQue_;
TQue<TPosition::VECIN, BUFFER_NUM> tmp2Que_;
TQue<TPosition::VECOUT, BUFFER_NUM> addResQue_;
TBuf<TPosition::VECCALC> idxBuf_;
};
}

#endif // EMBEDDING_DENSE_GRAD_V2_H_SMALL_DIM_H