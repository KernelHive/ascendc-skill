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
 * \file mse_loss_v2_base.h
 * \brief
 */


#ifndef MSE_LOSS_V2_BASE_H
#define MSE_LOSS_V2_BASE_H

#include "kernel_operator.h"

namespace MSELossV2Namespace {
template<typename T>
class MSELossV2Base {
public:
    __aicore__ inline MSELossV2Base(AscendC::TPipe *pipe, const MSELossV2TilingData *tilingData) :
        pipe(pipe),
        bufferNum(tilingData->bufferNum),
        epochs(tilingData->epochs),
        tileLength(tilingData->tileLength),
        tailTileLength(tilingData->tailTileLength)
    {
        this->globalOffset = tilingData->coreLength * MSELossV2Base<T>::HALF_SIZE * AscendC::GetBlockIdx();
        this->isLastCore = (AscendC::GetBlockIdx() == tilingData->coreNum - 1u);

        if (this->isLastCore) {
            this->epochs = tilingData->epochsForLastCore;
            this->tailTileLength = tilingData->tailTileLengthForLastCore;
            this->tailElems = tilingData->tailElems;
        }

        this->pipe->InitBuffer(this->inQue0, this->bufferNum, this->tileLength * MSELossV2Base<T>::HALF_SIZE);
        this->pipe->InitBuffer(this->inQue1, this->bufferNum, this->tileLength * MSELossV2Base<T>::HALF_SIZE);
        this->pipe->InitBuffer(this->castBuf0, this->tileLength * MSELossV2Base<T>::FLOAT_SIZE);
        this->pipe->InitBuffer(this->castBuf1, this->tileLength * MSELossV2Base<T>::FLOAT_SIZE);
    }

    __aicore__ inline void Init(GM_ADDR dst, GM_ADDR src0, GM_ADDR src1, GM_ADDR workspace)
    {
        this->src0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0 + this->globalOffset));
        this->src1Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src1 + this->globalOffset));
    }

    __aicore__ void Process();

protected:
    __aicore__ inline void CopyIn(const AscendC::LocalTensor<float> &cast0,
        const AscendC::LocalTensor<float> &cast1, uint64_t offset, uint64_t length)
    {
        AscendC::LocalTensor<T> src0Local = this->inQue0.template AllocTensor<T>();
        AscendC::DataCopy<T>(src0Local, this->src0Global[offset], length);
        this->inQue0.template EnQue<T>(src0Local);
        src0Local = this->inQue0.template DeQue<T>();
        AscendC::Cast<float, T>(cast0, src0Local, AscendC::RoundMode::CAST_NONE, length);
        this->inQue0.template FreeTensor<T>(src0Local);

        AscendC::LocalTensor<T> src1Local = this->inQue1.template AllocTensor<T>();
        AscendC::DataCopy<T>(src1Local, this->src1Global[offset], length);
        this->inQue1.template EnQue<T>(src1Local);
        src1Local = this->inQue1.template DeQue<T>();
        AscendC::Cast<float, T>(cast1, src1Local, AscendC::RoundMode::CAST_NONE, length);
        this->inQue1.template FreeTensor<T>(src1Local);
    }

protected:
    AscendC::TPipe *pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQue0;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQue1;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> castBuf0;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> castBuf1;

    AscendC::GlobalTensor<T> src0Global;
    AscendC::GlobalTensor<T> src1Global;
    AscendC::GlobalTensor<T> dstGlobal;

    bool isLastCore;
    uint64_t tailElems = 0u;
    uint64_t bufferNum = 0u;
    uint64_t epochs = 0u;
    uint64_t globalOffset = 0u;
    uint64_t tileLength = 0u;
    uint64_t tailTileLength = 0u;
    constexpr static uint64_t FLOAT_COUNT_PER_BLOCK = 16u;
    constexpr static uint64_t HALF_SIZE = sizeof(T);
    constexpr static uint64_t FLOAT_SIZE = sizeof(float);
};


template<>
class MSELossV2Base<float> {
public:
    __aicore__ inline MSELossV2Base(AscendC::TPipe *pipe, const MSELossV2TilingData *tilingData):
        pipe(pipe),
        bufferNum(tilingData->bufferNum),
        epochs(tilingData->epochs),
        tileLength(tilingData->tileLength),
        tailTileLength(tilingData->tailTileLength)
    {
        this->globalOffset = tilingData->coreLength * MSELossV2Base<float>::FLOAT_SIZE * AscendC::GetBlockIdx();
        this->isLastCore = (AscendC::GetBlockIdx() == tilingData->coreNum - 1u);

        if (this->isLastCore) {
            this->epochs = tilingData->epochsForLastCore;
            this->tailTileLength = tilingData->tailTileLengthForLastCore;
            this->tailElems = tilingData->tailElems;
        }

        this->pipe->InitBuffer(this->inQue0, this->bufferNum, this->tileLength * MSELossV2Base<float>::FLOAT_SIZE);
        this->pipe->InitBuffer(this->inQue1, this->bufferNum, this->tileLength * MSELossV2Base<float>::FLOAT_SIZE);
    }

    __aicore__ inline void Init(GM_ADDR dst, GM_ADDR src0, GM_ADDR src1, GM_ADDR workspace)
    {
        this->src0Global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src0 + this->globalOffset));
        this->src1Global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src1 + this->globalOffset));
    }

    __aicore__ void Process();

protected:
    __aicore__ inline void CopyIn(uint64_t offset, uint64_t length)
    {
        AscendC::LocalTensor<float> src0Local = this->inQue0.template AllocTensor<float>();
        AscendC::DataCopy<float>(src0Local, this->src0Global[offset], length);
        this->inQue0.template EnQue<float>(src0Local);

        AscendC::LocalTensor<float> src1Local = this->inQue1.template AllocTensor<float>();
        AscendC::DataCopy<float>(src1Local, this->src1Global[offset], length);
        this->inQue1.template EnQue<float>(src1Local);
    }

protected:
    AscendC::TPipe *pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQue0; 
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQue1;

    AscendC::GlobalTensor<float> src0Global;
    AscendC::GlobalTensor<float> src1Global;
    AscendC::GlobalTensor<float> dstGlobal;

    bool isLastCore;
    uint64_t tailElems = 0u;
    uint64_t bufferNum = 0u;
    uint64_t epochs = 0u;
    uint64_t globalOffset = 0u;
    uint64_t tileLength = 0u;
    uint64_t tailTileLength = 0u;
    constexpr static uint64_t FLOAT_COUNT_PER_BLOCK = 8u;
    constexpr static uint64_t FLOAT_SIZE = sizeof(float);
};
} // namespace MSELossV2Namespace

#endif