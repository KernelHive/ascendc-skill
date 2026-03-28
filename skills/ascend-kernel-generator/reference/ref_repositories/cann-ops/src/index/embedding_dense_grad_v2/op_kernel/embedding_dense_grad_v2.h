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
 * \file embedding_dense_grad_v2.h
 * \brief
 */

#ifndef EMBEDDING_DENSE_GRAD_V2_H
#define EMBEDDING_DENSE_GRAD_V2_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

constexpr uint64_t BUFFER_NUM = 1;
constexpr uint64_t DOUBLE_BUFFER = 1;
constexpr uint64_t INIT_PARAM = 0;
constexpr int64_t INDICE_INIT_PARAM = -1;
constexpr uint64_t LIMIT_COUNT_NUM = 10;
constexpr uint64_t COPY_ROW_NUM = 1;
constexpr uint64_t BLOCK_SIZE = 32;

struct AddParam {
    uint64_t mask;
    uint64_t repeatTime;
    uint64_t computeFormerNum;
    uint64_t computeTailNum;
    int64_t lastIndices;
    bool switchId;
    uint64_t tailRepeatTime;
    uint64_t tailComputeFormerNum;
    uint64_t tailComputeTailNum;
    uint64_t formerRepeatTime;
    uint64_t formerComputeFormerNum;
    uint64_t formerComputeTailNum;
};

namespace AscendC {
template<typename T>
class EmbeddingDenseGradV2Kernel {
public:
    __aicore__ inline EmbeddingDenseGradV2Kernel() = delete;
    __aicore__ inline EmbeddingDenseGradV2Kernel(GM_ADDR grad, GM_ADDR sortIndices, GM_ADDR posIdx, GM_ADDR backProps,
                                                 GM_ADDR workSpace, const EmbeddingDenseGradV2TilingData &tiling, TPipe &pipe)
    {
        InitParams(tiling);
        InitBuffers(pipe);
        SetGmAddr(grad, sortIndices, posIdx, backProps, workSpace, tiling);
    }
    __aicore__ inline void Process()
    {
        InitAddQue();
        for (uint64_t dimJ = 0; dimJ <= formerDimRepTime_; dimJ++) {
            UpdateParams(dimJ);
            if (curEmbeddingDim_ == 0) continue;
            for (uint64_t i = 0; i < rowNum_; i++) {
                CopyIn(i, dimJ);
                ComputeAndCopyOut(i, dimJ);
            }
        }
        FreeAddQue();
    }

private:
    __aicore__ inline void UpdateParams(const uint64_t dimJ)
    {
        if (dimJ == formerDimRepTime_) {
            curEmbeddingDim_ = tailEmbeddingDim_;
            addParam_.repeatTime = addParam_.tailRepeatTime;
            addParam_.computeFormerNum = addParam_.tailComputeFormerNum;
            addParam_.computeTailNum = addParam_.tailComputeTailNum;
        } else {
            curEmbeddingDim_ = formerEmbeddingDim_;
            addParam_.repeatTime = addParam_.formerRepeatTime;
            addParam_.computeFormerNum = addParam_.formerComputeFormerNum;
            addParam_.computeTailNum = addParam_.formerComputeTailNum;
        }
    }

    __aicore__ inline void InitParams(const EmbeddingDenseGradV2TilingData &tiling)
    {
        blockIdx_ = GetBlockIdx();
        if (blockIdx_ >= tiling.params.formerRowRepTime) {
            rowNum_ = tiling.params.tailRowNum;
        } else {
            rowNum_ = tiling.params.formerRowNum;
        }

        addParam_.mask = tiling.params.computeMask;
        addParam_.formerRepeatTime = tiling.params.formerComputeRepTime;
        addParam_.formerComputeFormerNum = tiling.params.formerComputeFormerNum;
        addParam_.formerComputeTailNum = tiling.params.formerComputeTailNum;
        addParam_.lastIndices = INDICE_INIT_PARAM;
        addParam_.switchId = false;
        addParam_.tailRepeatTime = tiling.params.tailComputeRepTime;
        addParam_.tailComputeFormerNum = tiling.params.tailComputeFormerNum;
        addParam_.tailComputeTailNum = tiling.params.tailComputeTailNum;

        addCount_[0] = INIT_PARAM;
        addCount_[1] = INIT_PARAM;

        numWeights_ = tiling.params.numWeights;
        embeddingDim_ = tiling.params.embeddingDim;
        paddingIdx_ = tiling.params.paddingIdx;
        scaleGradByFreq_ = tiling.params.scaleGradByFreq;

        formerDimRepTime_ = tiling.params.formerDimRepTime;
        formerEmbeddingDim_ = tiling.params.formerEmbeddingDim;
        tailEmbeddingDim_ = tiling.params.tailEmbeddingDim;
        curEmbeddingDim_ = formerEmbeddingDim_;
    }
    __aicore__ inline void InitBuffers(TPipe &pipe)
    {
        uint64_t idxAlignNum = BLOCK_SIZE / sizeof(int);
        uint64_t gradAlignNum = BLOCK_SIZE / sizeof(T);
        gradAlignNum = ((formerEmbeddingDim_ + gradAlignNum - 1) / gradAlignNum) * gradAlignNum;
        pipe.InitBuffer(indiceQue_, DOUBLE_BUFFER, idxAlignNum * COPY_ROW_NUM * sizeof(int));
        pipe.InitBuffer(posIdxQue_, DOUBLE_BUFFER, idxAlignNum * COPY_ROW_NUM * sizeof(int));
        pipe.InitBuffer(gradQue_, DOUBLE_BUFFER, gradAlignNum * sizeof(T));
        pipe.InitBuffer(tmpQue_, BUFFER_NUM, idxAlignNum * COPY_ROW_NUM * sizeof(int));
        pipe.InitBuffer(addResQue_[0], BUFFER_NUM, gradAlignNum * sizeof(T));
        pipe.InitBuffer(addResQue_[1], BUFFER_NUM, gradAlignNum * sizeof(T));
    }

    __aicore__ inline void SetGmAddr(GM_ADDR grad, GM_ADDR sortIndices, GM_ADDR posIdx, GM_ADDR backProps, GM_ADDR workSpace, const EmbeddingDenseGradV2TilingData &tiling)
    {
        uint64_t formerRowNumLoops = blockIdx_ < tiling.params.formerRowRepTime ? blockIdx_ : tiling.params.formerRowRepTime;
        uint64_t tailRowNumLoops = blockIdx_ < tiling.params.formerRowRepTime ? 0 : blockIdx_ - tiling.params.formerRowRepTime;
        uint64_t indiceAddrOffset = tiling.params.formerRowNum * formerRowNumLoops + tiling.params.tailRowNum * tailRowNumLoops;
        gradGm_.SetGlobalBuffer((__gm__ T*)grad);
        indiceGm_.SetGlobalBuffer((__gm__ uint32_t*)sortIndices + indiceAddrOffset);
        posIdxGm_.SetGlobalBuffer((__gm__ uint32_t*)posIdx + indiceAddrOffset);
        outputGm_.SetGlobalBuffer((__gm__ T*)backProps);
        idxNumGm_.SetGlobalBuffer((__gm__ float*)workSpace);
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

    __aicore__ inline void CopyIn(const uint64_t progress, const uint64_t dimJ)
    {
        LocalTensor<T> gradLocal = gradQue_.AllocTensor<T>();
        LocalTensor<uint32_t> indiceLocal = indiceQue_.AllocTensor<uint32_t>();
        LocalTensor<uint32_t> posIdxLocal = posIdxQue_.AllocTensor<uint32_t>();
        uint64_t gradAddrOffset = 0;
        uint64_t indicesOffset = progress;
        DataCopyParams gradCopyParams{1, static_cast<uint16_t>(curEmbeddingDim_ * sizeof(T)), 0, 0};
        DataCopyParams indiceCopyParams{1, static_cast<uint16_t>(sizeof(uint32_t)), 0, 0};
        DataCopyPadParams padParams{true, 0, 0, 0};
        DataCopyPad(posIdxLocal, posIdxGm_[indicesOffset], indiceCopyParams, padParams);
        DataCopyPad(indiceLocal, indiceGm_[indicesOffset], indiceCopyParams, padParams);
        
        gradAddrOffset = posIdxLocal.GetValue(0) * embeddingDim_ + formerEmbeddingDim_ * dimJ;
        DataCopyPad(gradLocal, gradGm_[gradAddrOffset], gradCopyParams, padParams);
        gradQue_.EnQue<T>(gradLocal);
        indiceQue_.EnQue<uint32_t>(indiceLocal);
        posIdxQue_.FreeTensor<uint32_t>(posIdxLocal);
    }

    __aicore__ inline void ComputeAndCopyOut(const uint64_t progress, const uint64_t dimJ)
    {
        LocalTensor<uint32_t> indiceLocal = indiceQue_.DeQue<uint32_t>();
        LocalTensor<T> gradLocal = gradQue_.DeQue<T>();
        uint64_t currentId = indiceLocal.GetValue(0);
        if (currentId != paddingIdx_) {
            // 1. decided change atomic add que
            bool isCopyOut = CheckIsNeedSwitchAddQue(currentId);
            // 2. atomic add in ub
            AtomicAddInUb(gradLocal);
            // 3. check is copyout
            if (addCount_[addParam_.switchId] == LIMIT_COUNT_NUM || progress == rowNum_ - 1) {
                CopyOut(addParam_.switchId, currentId, dimJ);
            } 
            if (isCopyOut) {
                CopyOut(!addParam_.switchId, addParam_.lastIndices, dimJ);
            }
            addParam_.lastIndices = currentId;
        } else if (addParam_.lastIndices != INDICE_INIT_PARAM) {
            CopyOut(addParam_.switchId, addParam_.lastIndices, dimJ);
            addParam_.lastIndices = INDICE_INIT_PARAM;
        }
        gradQue_.FreeTensor<T>(gradLocal);
        indiceQue_.FreeTensor<uint32_t>(indiceLocal);
    }

    __aicore__ inline void CopyOut(const uint64_t addAddrOffset, const uint64_t indice, const uint64_t dimJ)
    {
        // 1. copy out to releative grad row
        LocalTensor<T> addResLocal = addResQue_[addAddrOffset].DeQue<T>();
        uint64_t gmAddrOffset = indice * embeddingDim_ + formerEmbeddingDim_ * dimJ;
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curEmbeddingDim_ * sizeof(T)), 0, 0, 0};
        SetAtomicAdd<T>();
        DataCopyPad(outputGm_[gmAddrOffset], addResLocal, copyParams);
        SetAtomicNone();
        // 2. copy out indices counts
        if (scaleGradByFreq_ && dimJ == 0) {
            LocalTensor<float> tmpLocal = tmpQue_.AllocTensor<float>();
            tmpLocal.SetValue(0, static_cast<float>((int)addCount_[addAddrOffset]));
            DataCopyExtParams scaleCopyParams{1, sizeof(uint32_t), 0, 0, 0};
            SetAtomicAdd<float>();
            DataCopyPad(idxNumGm_[indice], tmpLocal, scaleCopyParams);
            SetAtomicNone();
            tmpQue_.FreeTensor<float>(tmpLocal);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        ResetAddQue(addResLocal);
        ResetAddCount(addAddrOffset);
        addResQue_[addAddrOffset].EnQue<T>(addResLocal);
    }

    __aicore__ inline void ResetAddCount(const uint64_t id)
    {
        addCount_[id] = INIT_PARAM;
    }

    __aicore__ inline bool CheckIsNeedSwitchAddQue(const uint64_t currentId)
    {
        // 1. indice is not equal to last indice
        if (addParam_.lastIndices != INDICE_INIT_PARAM && currentId != addParam_.lastIndices) {
            // reset
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

private:
    GlobalTensor<T> gradGm_;
    GlobalTensor<T> outputGm_;
    GlobalTensor<uint32_t> indiceGm_;
    GlobalTensor<uint32_t> posIdxGm_;
    GlobalTensor<float> idxNumGm_;

    TQue<TPosition::VECIN, DOUBLE_BUFFER> gradQue_;
    TQue<TPosition::VECIN, DOUBLE_BUFFER> indiceQue_;
    TQue<TPosition::VECIN, DOUBLE_BUFFER> posIdxQue_;
    TQue<TPosition::VECIN, BUFFER_NUM> tmpQue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> addResQue_[2];

    AddParam addParam_;
    uint64_t rowNum_;
    uint64_t addCount_[2];

    uint64_t blockIdx_;
    // attr
    uint64_t numWeights_;
    uint64_t embeddingDim_;
    uint64_t paddingIdx_;
    bool scaleGradByFreq_;

    // big shape
    uint64_t formerDimRepTime_;
    uint64_t formerEmbeddingDim_;
    uint64_t tailEmbeddingDim_;
    uint64_t curEmbeddingDim_;
};
}

#endif // EMBEDDING_DENSE_GRAD_V2_H