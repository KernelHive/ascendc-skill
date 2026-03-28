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
 * \file embedding_dense_grad_v2_determinist.h
 * \brief
 */

#ifndef EMBEDDING_DENSE_GRAD_V2_H_DETERMINIST_H
#define EMBEDDING_DENSE_GRAD_V2_H_DETERMINIST_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "embedding_dense_grad_v2.h"

constexpr uint64_t ADDR_BACK_STEP = 1;
constexpr uint64_t FIRST_PROGRESS_FLAG = 0;

namespace AscendC {
template<typename T>
class EmbeddingDenseGradV2DeterministKernel {
public:
__aicore__ inline EmbeddingDenseGradV2DeterministKernel() = delete;
__aicore__ inline EmbeddingDenseGradV2DeterministKernel(GM_ADDR grad, GM_ADDR sortIndices, GM_ADDR posIdx, GM_ADDR backProps,
                                                        GM_ADDR workSpace, const EmbeddingDenseGradV2TilingData &tiling, TPipe &pipe)
{
    InitParams(tiling);
    InitBuffers(pipe);
    SetGmAddr(grad, sortIndices, posIdx, backProps, workSpace, tiling);
}

__aicore__ inline void Process()
{
    InitAddQue();
    for (uint64_t dimJ = 0; dimJ <= formerEmbeddingDimRepTime_; dimJ++) {
        UpdateParams(dimJ);
        if (nowEmbeddingDim_ == 0) continue;
        standIdice_ = INDICE_INIT_PARAM;
        if (blockIdx_ != 0) {
            InitStandIndice(0);
        }
        if (SubProcess(dimJ)) {
            ContinueCompute(dimJ);
        }
    }
    FreeAddQue();
}

private:
__aicore__ inline bool SubProcess(const uint64_t dimJ)
{
    bool isNeedContinueCopy = true;
    for (int64_t j = processRowNum_ - 1; j >= 0; j--) {
        if (!CopyIn(j, true, dimJ)) {
            // is last indices is same with last core‘s, break
            isNeedContinueCopy = j == processRowNum_ - 1 ? false : true;
            if (addParam_.lastIndices != INDICE_INIT_PARAM) {
                CopyOut(addParam_.switchId, addParam_.lastIndices, dimJ);
            }
            break;
        }
        ComputeAndCopyOut(j, 0, dimJ);
    }
    return isNeedContinueCopy;
}

__aicore__ inline void InitParams(const EmbeddingDenseGradV2TilingData &tiling)
{
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= tiling.determinTiling.formerRowNumRepTime) {
        processRowNum_ = tiling.determinTiling.tailRowNum;
    } else {
        processRowNum_ = tiling.determinTiling.formerRowNum;
    }
    gradRow_ = tiling.determinTiling.gradRow;
    numWeights_ = tiling.params.numWeights;
    embeddingDim_ = tiling.params.embeddingDim;
    paddingIdx_ = tiling.params.paddingIdx;
    scaleGradByFreq_ = tiling.params.scaleGradByFreq;
    standIdice_ = INDICE_INIT_PARAM;

    formerEmbeddingDimRepTime_ = tiling.params.formerDimRepTime;
    formerEmbeddingDim_ = tiling.params.formerEmbeddingDim;
    tailEmbeddingDim_ = tiling.params.tailEmbeddingDim;
    nowEmbeddingDim_ = formerEmbeddingDim_;

    addParam_.mask = tiling.determinTiling.computeMask;
    addParam_.formerRepeatTime = tiling.determinTiling.formerComputeRepTime;
    addParam_.formerComputeFormerNum = tiling.determinTiling.formerComputeFormerNum;
    addParam_.formerComputeTailNum = tiling.determinTiling.formerComputeTailNum;
    addParam_.lastIndices = INDICE_INIT_PARAM;
    addParam_.switchId = false;
    addParam_.tailRepeatTime = tiling.determinTiling.tailComputeRepTime;
    addParam_.tailComputeFormerNum = tiling.determinTiling.tailComputeFormerNum;
    addParam_.tailComputeTailNum = tiling.determinTiling.tailComputeTailNum;
    addCount_[0] = INIT_PARAM;
    addCount_[1] = INIT_PARAM;

    uint64_t formerRowNumLoops = blockIdx_ < tiling.determinTiling.formerRowNumRepTime ? blockIdx_ : tiling.determinTiling.formerRowNumRepTime;
    uint64_t tailRowNumLoops = blockIdx_ < tiling.determinTiling.formerRowNumRepTime ? 0 : blockIdx_ - tiling.determinTiling.formerRowNumRepTime;
    indicesAddrOffset_ = tiling.determinTiling.formerRowNum * formerRowNumLoops + tiling.determinTiling.tailRowNum * tailRowNumLoops;
}

__aicore__ inline void UpdateParams(const uint64_t dimJ)
{
    if (dimJ == formerEmbeddingDimRepTime_) {
        nowEmbeddingDim_ = tailEmbeddingDim_;
        addParam_.repeatTime = addParam_.tailRepeatTime;
        addParam_.computeFormerNum = addParam_.tailComputeFormerNum;
        addParam_.computeTailNum = addParam_.tailComputeTailNum;
    } else {
        nowEmbeddingDim_ = formerEmbeddingDim_;
        addParam_.repeatTime = addParam_.formerRepeatTime;
        addParam_.computeFormerNum = addParam_.formerComputeFormerNum;
        addParam_.computeTailNum = addParam_.formerComputeTailNum;
    }
}

__aicore__ inline void InitBuffers(TPipe &pipe)
{
    uint64_t idxAlignNum = BLOCK_SIZE / sizeof(int32_t);
    uint64_t gradAlignNum = BLOCK_SIZE / sizeof(T);
    gradAlignNum = ((formerEmbeddingDim_ + gradAlignNum - 1) / gradAlignNum) * gradAlignNum;
    pipe.InitBuffer(indiceQue_, BUFFER_NUM, idxAlignNum * sizeof(int));
    pipe.InitBuffer(posIdxQue_, BUFFER_NUM, idxAlignNum * sizeof(int));
    pipe.InitBuffer(tmpQue_, BUFFER_NUM, idxAlignNum * sizeof(float));
    pipe.InitBuffer(gradQue_, BUFFER_NUM, gradAlignNum * sizeof(T));
    pipe.InitBuffer(addResQue_[0], BUFFER_NUM, gradAlignNum * sizeof(T));
    pipe.InitBuffer(addResQue_[1], BUFFER_NUM, gradAlignNum * sizeof(T));
}

__aicore__ inline void SetGmAddr(GM_ADDR grad, GM_ADDR sortIndices, GM_ADDR posIdx, GM_ADDR backProps,
                                 GM_ADDR workSpace, const EmbeddingDenseGradV2TilingData &tiling)
{
    gradGm_.SetGlobalBuffer((__gm__ T*)grad);
    indiceGm_.SetGlobalBuffer((__gm__ int*)sortIndices + indicesAddrOffset_);
    indiceGlobalGm_.SetGlobalBuffer((__gm__ int*)sortIndices);
    posIdxGm_.SetGlobalBuffer((__gm__ int*)posIdx + indicesAddrOffset_);
    outputGm_.SetGlobalBuffer((__gm__ T*)backProps);
    idxNumGm_.SetGlobalBuffer((__gm__ float*)workSpace);
}

__aicore__ inline void InitStandIndice(const uint64_t offset)
{
    LocalTensor<int> indiceLocal = indiceQue_.AllocTensor<int>();
    uint64_t idAddrOffset = indicesAddrOffset_ + offset;
    DataCopyParams indiceCopyParams{1, static_cast<uint16_t>(sizeof(uint32_t)), 0, 0};
    DataCopyPadParams padParams{true, 0, 0, 0};
    DataCopyPad(indiceLocal, indiceGlobalGm_[idAddrOffset - ADDR_BACK_STEP], indiceCopyParams, padParams);
    standIdice_ = indiceLocal.GetValue(0);
    indiceQue_.FreeTensor<int>(indiceLocal);
}

__aicore__ inline bool CopyIn(const uint64_t progress, const bool flag, const uint64_t dimJ)
{
    // check is same indice with stand indice
    LocalTensor<int> indiceLocal = indiceQue_.AllocTensor<int>();
    uint64_t idAddrOffset = progress;
    DataCopyParams indiceCopyParams{1, static_cast<uint16_t>(sizeof(uint32_t)), 0, 0};
    DataCopyPadParams padParams{true, 0, 0, 0};
    DataCopyPad(indiceLocal, indiceGm_[idAddrOffset], indiceCopyParams, padParams);
    if ((indiceLocal.GetValue(0) == standIdice_) == flag){
        indiceQue_.FreeTensor<int>(indiceLocal);
        return false;
    }
    indiceQue_.EnQue<int>(indiceLocal);
    // copy in
    LocalTensor<T> gradLocal = gradQue_.AllocTensor<T>();
    LocalTensor<int> posIdxLocal = posIdxQue_.AllocTensor<int>();
    DataCopyParams gradCopyParams{1, static_cast<uint16_t>(nowEmbeddingDim_ * sizeof(T)), 0, 0};
    DataCopyPad(posIdxLocal, posIdxGm_[idAddrOffset], indiceCopyParams, padParams);
    uint64_t gradAddrOffset = posIdxLocal.GetValue(0) * embeddingDim_ + formerEmbeddingDim_ * dimJ;
    DataCopyPad(gradLocal, gradGm_[gradAddrOffset], gradCopyParams, padParams);
    gradQue_.EnQue<T>(gradLocal);
    posIdxQue_.FreeTensor<int>(posIdxLocal);
    return true;
}

__aicore__ inline void ComputeAndCopyOut(const uint64_t progress, const uint64_t loopLimit, const uint64_t dimJ)
{
    LocalTensor<int> indiceLocal = indiceQue_.DeQue<int>();
    LocalTensor<T> gradLocal = gradQue_.DeQue<T>();
    uint64_t currentId = indiceLocal.GetValue(0);
    if (currentId != paddingIdx_) {
        bool isCopyOut = CheckIsNeedSwitchAddQue(currentId);
        AtomicAddInUb(gradLocal);
        // reach to count num limit or last indices -> copy out
        if (addCount_[addParam_.switchId] == LIMIT_COUNT_NUM || progress == loopLimit) {
            CopyOut(addParam_.switchId, currentId, dimJ);
        }
        // switch add que -> copy out
        if (isCopyOut) {
            CopyOut(!addParam_.switchId, addParam_.lastIndices, dimJ);
        }
        addParam_.lastIndices = currentId;
    } else if (addParam_.lastIndices != INDICE_INIT_PARAM) {
        CopyOut(addParam_.switchId, addParam_.lastIndices, dimJ);
        addParam_.lastIndices = INDICE_INIT_PARAM;
    }
    gradQue_.FreeTensor<T>(gradLocal);
    indiceQue_.FreeTensor<int>(indiceLocal);
}

__aicore__ inline bool CheckIsNeedSwitchAddQue(const uint64_t currentId)
{
    if (addParam_.lastIndices != INDICE_INIT_PARAM && currentId != addParam_.lastIndices) {
        addParam_.switchId = !addParam_.switchId;
        ResetAddCount(addParam_.switchId);
        return true;
    }
    return false;
}

__aicore__ inline void AtomicAddInUb(LocalTensor<T> &gradLocal)
{
    LocalTensor<T> addLocal = addResQue_[static_cast<uint8_t>(addParam_.switchId)].DeQue<T>();
    if (addParam_.computeFormerNum > 0) {
        Add(addLocal, addLocal, gradLocal, addParam_.mask, addParam_.repeatTime, {1, 1, 1, 8, 8, 8});
    }
    if (addParam_.computeTailNum > 0) {
        Add(addLocal[addParam_.computeFormerNum], addLocal[addParam_.computeFormerNum],
            gradLocal[addParam_.computeFormerNum], addParam_.computeTailNum, 1, {1, 1, 1, 0, 0, 0});
    }
    addResQue_[static_cast<uint8_t>(addParam_.switchId)].EnQue<T>(addLocal);
    addCount_[addParam_.switchId]++;
}

__aicore__ inline void CopyOut(const uint64_t addrOffset, const uint64_t indice, const uint64_t dimJ)
{
    LocalTensor<T> addResLocal = addResQue_[addrOffset].DeQue<T>();
    uint64_t gmAddrOffset = indice * embeddingDim_ + formerEmbeddingDim_ * dimJ;
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(nowEmbeddingDim_ * sizeof(T)), 0, 0, 0};
    SetAtomicAdd<T>();
    DataCopyPad(outputGm_[gmAddrOffset], addResLocal, copyParams);
    SetAtomicNone();
    if (scaleGradByFreq_ && dimJ == 0) {
        LocalTensor<float> tmpLocal = tmpQue_.AllocTensor<float>();
        tmpLocal.SetValue(0, static_cast<float>((int)addCount_[addrOffset]));
        DataCopyExtParams scaleCopyParams{1, sizeof(uint32_t), 0, 0, 0};
        SetAtomicAdd<float>();
        DataCopyPad(idxNumGm_[indice], tmpLocal, scaleCopyParams);
        SetAtomicNone();
        tmpQue_.FreeTensor<float>(tmpLocal);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    ResetAddQue(addResLocal);
    ResetAddCount(addrOffset);
    addResQue_[addrOffset].EnQue<T>(addResLocal);
}

__aicore__ inline void ContinueCompute(const uint64_t dimJ)
{
    addParam_.lastIndices = INDICE_INIT_PARAM;
    InitStandIndice(processRowNum_);
    uint64_t preProcessRowNum = indicesAddrOffset_;
    for (uint64_t i = processRowNum_; i < gradRow_ - preProcessRowNum; i++) {
        if (!CopyIn(i, false, dimJ)) {
            if (addParam_.lastIndices != INDICE_INIT_PARAM) {
                CopyOut(addParam_.switchId, addParam_.lastIndices, dimJ);
            }
            break;
        }
        ComputeAndCopyOut(i, gradRow_ - preProcessRowNum - 1, dimJ);
    }
}

__aicore__ inline void InitAddQue()
{
    LocalTensor<T> addResLocal1 = addResQue_[0].AllocTensor<T>();
    LocalTensor<T> addResLocal2 = addResQue_[1].AllocTensor<T>();
    ResetAddQue(addResLocal1);
    ResetAddQue(addResLocal2);
    addResQue_[0].EnQue<T>(addResLocal1);
    addResQue_[1].EnQue<T>(addResLocal2);
}

__aicore__ inline void FreeAddQue()
{
    LocalTensor<T> addResLocal1 = addResQue_[0].DeQue<T>();
    LocalTensor<T> addResLocal2 = addResQue_[1].DeQue<T>();
    addResQue_[0].FreeTensor<T>(addResLocal1);
    addResQue_[1].FreeTensor<T>(addResLocal2);
}

__aicore__ inline void ResetAddQue(LocalTensor<T> &addRes)
{
    Duplicate<T>(addRes, 0.0, formerEmbeddingDim_);
}

__aicore__ inline void ResetAddCount(const uint64_t id)
{
    addCount_[id] = INIT_PARAM;
}

private:
    GlobalTensor<T> gradGm_;
    GlobalTensor<T> outputGm_;
    GlobalTensor<int> indiceGm_;
    GlobalTensor<int> indiceGlobalGm_;
    GlobalTensor<int> posIdxGm_;
    GlobalTensor<float> idxNumGm_;

    TQue<TPosition::VECIN, BUFFER_NUM> gradQue_;
    TQue<TPosition::VECIN, BUFFER_NUM> indiceQue_;
    TQue<TPosition::VECIN, BUFFER_NUM> posIdxQue_;
    TQue<TPosition::VECIN, BUFFER_NUM> tmpQue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> addResQue_[2];

    AddParam addParam_;
    int standIdice_;
    int lastAddIndice_;
    uint64_t blockIdx_;
    uint64_t addCount_[2];
    uint64_t numWeights_;
    uint64_t embeddingDim_;
    uint64_t paddingIdx_;
    uint64_t processRowNum_;
    uint64_t gradRow_;
    bool scaleGradByFreq_;

    // big shape
    uint64_t formerEmbeddingDimRepTime_;
    uint64_t formerEmbeddingDim_;
    uint64_t tailEmbeddingDim_;
    uint64_t nowEmbeddingDim_;

    uint64_t indicesAddrOffset_;
};
}

#endif // EMBEDDING_DENSE_GRAD_V2_H_DETERMINIST_H
