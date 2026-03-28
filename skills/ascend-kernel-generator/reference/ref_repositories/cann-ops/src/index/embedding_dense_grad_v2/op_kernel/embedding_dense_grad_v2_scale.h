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
 * \file embedding_dense_grad_v2_scale.h
 * \brief
 */

#ifndef EMBEDDING_DENSE_GRAD_V2_H_SCALE_H
#define EMBEDDING_DENSE_GRAD_V2_H_SCALE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

struct ComputeParam {
    uint64_t mask;
    uint64_t repeatTime;
    uint64_t computeFormerNum;
    uint64_t computeTailNum;
    uint64_t tailRepeatTime;
    uint64_t tailComputeFormerNum;
    uint64_t tailComputeTailNum;
    uint64_t formerRepeatTime;
    uint64_t formerComputeFormerNum;
    uint64_t formerComputeTailNum;
};


namespace AscendC {
template<typename T>
class EmbeddingDenseGradV2ScaleKernel {
public:
    __aicore__ inline EmbeddingDenseGradV2ScaleKernel() = delete;
    __aicore__ inline EmbeddingDenseGradV2ScaleKernel(GM_ADDR backProps, GM_ADDR workSpace, const EmbeddingDenseGradV2TilingData &tiling, TPipe &pipe)
    {
        InitParams(tiling);
        InitBuffers(pipe);
        SetGmAddr(backProps, workSpace, tiling);
    }

    __aicore__ inline void Process()
    {
        for (uint64_t dimJ = 0; dimJ <= formerDimRepTime_; dimJ++) {
            UpdateParams(dimJ);
            for (uint64_t i = 0; i < scaleRowNum_; i++) {
                CopyIn(i, dimJ);
                Compute();
                CopyOut(i, dimJ);
            }
        }
    }

private:
    __aicore__ inline void UpdateParams(const uint64_t dimJ)
    {
        if (dimJ == formerDimRepTime_) {
            curEmbeddingDim_ = tailEmbeddingDim_;
            computeParam_.repeatTime = computeParam_.tailRepeatTime;
            computeParam_.computeFormerNum = computeParam_.tailComputeFormerNum;
            computeParam_.computeTailNum = computeParam_.tailComputeTailNum;
        } else {
            curEmbeddingDim_ = formerEmbeddingDim_;
            computeParam_.repeatTime = computeParam_.formerRepeatTime;
            computeParam_.computeFormerNum = computeParam_.formerComputeFormerNum;
            computeParam_.computeTailNum = computeParam_.formerComputeTailNum;
        }
    }

    __aicore__ inline void CopyIn(const uint64_t progress, const uint64_t dimJ)
    {
        LocalTensor<T> gradLocal = gradInQue_.AllocTensor<T>();
        LocalTensor<float> scaleLocal = scaleQue_.AllocTensor<float>();
        uint64_t gradAddrOffset = progress * embeddingDim_ + formerEmbeddingDim_ * dimJ;
        uint64_t scaleAddrOffset = progress;
        DataCopyParams gradCopyParams{1, static_cast<uint16_t>(curEmbeddingDim_ * sizeof(T)), 0, 0};
        DataCopyParams scaleCopyParams{1, static_cast<uint16_t>(sizeof(uint32_t)), 0, 0};
        DataCopyPadParams padParams{true, 0, 0, 0};
        DataCopyPad(gradLocal, backPropGm_[gradAddrOffset], gradCopyParams, padParams);
        DataCopyPad(scaleLocal, indexCountGm_[scaleAddrOffset], scaleCopyParams, padParams);
        gradInQue_.EnQue<T>(gradLocal);
        scaleQue_.EnQue<float>(scaleLocal);
    }

    __aicore__ inline void Compute()
    {
        // 1. get same indices count 
        // 2. 1 / scale
        // 3. grad * (1 / scale)
        LocalTensor<T> gradInLocal = gradInQue_.DeQue<T>();
        LocalTensor<float> scaleLocal = scaleQue_.DeQue<float>();
        LocalTensor<T> gradOutLocal = gradOutQue_.AllocTensor<T>();
        float scale = scaleLocal.GetValue(0);
        T reciScale = scale == 0 ? 1.0F : 1.0F / static_cast<T>(scale);
        if (computeParam_.computeFormerNum > 0) {
            Muls(gradOutLocal, gradInLocal, reciScale, computeParam_.mask, computeParam_.repeatTime, {1, 1, 8, 8});
        }
        if (computeParam_.computeTailNum > 0) {
            Muls(gradOutLocal[computeParam_.computeFormerNum], gradInLocal[computeParam_.computeFormerNum], 
                    reciScale, computeParam_.computeTailNum, 1, {1, 1, 0, 0});
        }
        gradInQue_.FreeTensor<T>(gradInLocal);
        scaleQue_.FreeTensor<float>(scaleLocal);
        gradOutQue_.EnQue<T>(gradOutLocal);
    }

    __aicore__ inline void CopyOut(const uint64_t progress, const uint64_t dimJ)
    {
        LocalTensor<T> gradOutLocal = gradOutQue_.DeQue<T>();
        uint64_t gradAddrOffset = progress * embeddingDim_ + formerEmbeddingDim_ * dimJ;
        DataCopyParams copyParams{1, static_cast<uint16_t>(curEmbeddingDim_ * sizeof(T)), 0, 0};
        DataCopyPad(backPropGm_[gradAddrOffset], gradOutLocal, copyParams);
        gradOutQue_.FreeTensor<T>(gradOutLocal);
    }   

    __aicore__ inline void InitParams(const EmbeddingDenseGradV2TilingData &tiling)
    {
        blockIdx_ = GetBlockIdx();
        if (blockIdx_ >= tiling.scaleTiling.formerCoreRowRepTime) {
            scaleRowNum_ = tiling.scaleTiling.tailCoreRowNum;
        } else {
            scaleRowNum_ = tiling.scaleTiling.formerCoreRowNum;
        }
        computeParam_.mask = tiling.scaleTiling.mask;
        computeParam_.formerRepeatTime = tiling.scaleTiling.formerComputeRepTime;
        computeParam_.formerComputeFormerNum = tiling.scaleTiling.formerComputeFormerNum;
        computeParam_.formerComputeTailNum = tiling.scaleTiling.formerComputeTailNum;
        computeParam_.tailRepeatTime = tiling.scaleTiling.tailComputeRepTime;
        computeParam_.tailComputeFormerNum = tiling.scaleTiling.tailComputeFormerNum;
        computeParam_.tailComputeTailNum = tiling.scaleTiling.tailComputeTailNum;
        embeddingDim_ = tiling.params.embeddingDim;

        formerDimRepTime_ = tiling.params.formerDimRepTime;
        formerEmbeddingDim_ = tiling.params.formerEmbeddingDim;
        tailEmbeddingDim_ = tiling.params.tailEmbeddingDim;
        curEmbeddingDim_ = formerEmbeddingDim_;
    }

    __aicore__ inline void InitBuffers(TPipe &pipe)
    {
        uint64_t alignNum = BLOCK_SIZE / sizeof(T);
        uint64_t scaleAlign = BLOCK_SIZE / sizeof(float);
        uint64_t gradAlign = ((formerEmbeddingDim_ + alignNum - 1) / alignNum) * alignNum;
        pipe.InitBuffer(gradInQue_, DOUBLE_BUFFER, gradAlign * sizeof(T));
        pipe.InitBuffer(scaleQue_, DOUBLE_BUFFER, scaleAlign * sizeof(float));
        pipe.InitBuffer(gradOutQue_, DOUBLE_BUFFER, gradAlign * sizeof(T));
    }

    __aicore__ inline void SetGmAddr(GM_ADDR backProps, GM_ADDR indiceCountGm, const EmbeddingDenseGradV2TilingData &tiling)
    {
        uint64_t formerCoreRowNumLoops = blockIdx_ < tiling.scaleTiling.formerCoreRowRepTime ? blockIdx_ : tiling.scaleTiling.formerCoreRowRepTime;
        uint64_t tailCoreRowNumLoops = blockIdx_ < tiling.scaleTiling.formerCoreRowRepTime ? 0 : blockIdx_ - tiling.scaleTiling.formerCoreRowRepTime;
        uint64_t addrOffset = tiling.scaleTiling.formerCoreRowNum * formerCoreRowNumLoops + tiling.scaleTiling.tailCoreRowNum * tailCoreRowNumLoops;
        backPropGm_.SetGlobalBuffer((__gm__ T*)backProps + addrOffset * embeddingDim_);
        indexCountGm_.SetGlobalBuffer((__gm__ float*)indiceCountGm + addrOffset);
    }

private:
    GlobalTensor<T> backPropGm_;
    GlobalTensor<float> indexCountGm_;

    TQue<TPosition::VECIN, DOUBLE_BUFFER> gradInQue_;
    TQue<TPosition::VECIN, DOUBLE_BUFFER> scaleQue_;
    TQue<TPosition::VECOUT, DOUBLE_BUFFER> gradOutQue_;

private:
    ComputeParam computeParam_;

    uint64_t blockIdx_;
    uint64_t embeddingDim_;
    uint64_t scaleRowNum_;

    // big shape
    uint64_t formerDimRepTime_;
    uint64_t formerEmbeddingDim_;
    uint64_t tailEmbeddingDim_;
    uint64_t curEmbeddingDim_;
};
}

#endif // EMBEDDING_DENSE_GRAD_V2_H_SCALE_H