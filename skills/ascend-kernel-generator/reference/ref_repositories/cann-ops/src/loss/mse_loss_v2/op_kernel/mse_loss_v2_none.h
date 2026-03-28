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
 * \file mse_loss_v2_none.h
 * \brief
 */

#ifndef MSE_LOSS_V2_NONE_H
#define MSE_LOSS_V2_NONE_H

#include "mse_loss_v2_base.h"

namespace MSELossV2Namespace {
template<typename T>
class MSELossV2None : public MSELossV2Base<T> {
public:
    __aicore__ inline MSELossV2None(AscendC::TPipe *pipe, const MSELossV2TilingData *tilingData) :
        MSELossV2Base<T>(pipe, tilingData)
    {
        this->pipe->InitBuffer(this->dstQue, this->bufferNum, this->tileLength * MSELossV2Base<T>::HALF_SIZE);
    }

    __aicore__ inline void Init(GM_ADDR dst, GM_ADDR src0, GM_ADDR src1, GM_ADDR usrWorkspace)
    {
        MSELossV2Base<T>::Init(dst, src0, src1, usrWorkspace);
        this->dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dst + this->globalOffset));
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<float> cast0Local = this->castBuf0.template Get<float>();
        AscendC::LocalTensor<float> cast1Local = this->castBuf1.template Get<float>();

        for (uint64_t i = 0u; i < this->epochs; i++) {
            MSELossV2Base<T>::CopyIn(cast0Local, cast1Local, i * this->tileLength, this->tileLength);
            this->Compute(cast0Local, cast1Local, this->tileLength);
            this->CopyOut(i * this->tileLength, this->tileLength);
        }

        if (this->tailTileLength || (this->isLastCore && this->tailElems)) {
            if (this->isLastCore && this->tailElems) {
                this->tailTileLength = ((this->tailTileLength + this->tailElems) + 15u) & ~15u;
            }

            MSELossV2Base<T>::CopyIn(cast0Local, cast1Local, this->epochs * this->tileLength, this->tailTileLength);
            this->Compute(cast0Local, cast1Local, this->tailTileLength);
            this->CopyOut(this->epochs * this->tileLength, this->tailTileLength);
        }
    }

private:
    __aicore__ inline void Compute(const AscendC::LocalTensor<float> &cast0,
        const AscendC::LocalTensor<float> &cast1, const uint64_t length)
    {
        AscendC::Sub(cast0, cast0, cast1, length);
        AscendC::Mul(cast1, cast0, cast0, length);

        AscendC::LocalTensor<T> dstLocal = this->dstQue.template AllocTensor<T>();
#if __CCE_AICORE__ == 200
        AscendC::Cast(dstLocal, cast1, AscendC::RoundMode::CAST_NONE, length);
#else
        AscendC::Cast(dstLocal, cast1, AscendC::RoundMode::CAST_RINT, length);
#endif
        this->dstQue.template EnQue<T>(dstLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint64_t length)
    {
        AscendC::LocalTensor<T> dstLocal = this->dstQue.template DeQue<T>();
        AscendC::DataCopy<T>(this->dstGlobal[offset], dstLocal, length);
        this->dstQue.template FreeTensor<T>(dstLocal);
    }

private:
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> dstQue;
};


template<>
class MSELossV2None<float> : public MSELossV2Base<float> {
public:
    __aicore__ inline MSELossV2None(AscendC::TPipe *pipe, const MSELossV2TilingData *tilingData) :
        MSELossV2Base<float>(pipe, tilingData)
    {
        this->pipe->InitBuffer(this->dstQue, this->bufferNum, this->tileLength * MSELossV2Base<float>::FLOAT_SIZE);
    }

    __aicore__ inline void Init(GM_ADDR dst, GM_ADDR src0, GM_ADDR src1, GM_ADDR usrWorkspace)
    {
        MSELossV2Base<float>::Init(dst, src0, src1, usrWorkspace);
        this->dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dst + this->globalOffset));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0u; i < this->epochs; i++) {
            MSELossV2Base<float>::CopyIn(i * this->tileLength, this->tileLength);
            Compute(this->tileLength);
            CopyOut(i * this->tileLength, this->tileLength);
        }

        if (this->tailTileLength || (this->isLastCore && this->tailElems)) {
            if (this->isLastCore && this->tailElems) {
                this->tailTileLength = ((this->tailTileLength + this->tailElems) + 7u) & ~7u;
            }

            MSELossV2Base<float>::CopyIn(this->epochs * this->tileLength, this->tailTileLength);
            Compute(this->tailTileLength);
            CopyOut(this->epochs * this->tileLength, this->tailTileLength);
        }
    }

private:
    __aicore__ inline void Compute(const uint64_t length)
    {
        AscendC::LocalTensor<float> src0Local = this->inQue0.template DeQue<float>();
        AscendC::LocalTensor<float> src1Local = this->inQue1.template DeQue<float>();
        AscendC::Sub(src1Local, src0Local, src1Local, length);
        this->inQue0.template FreeTensor<float>(src0Local);

        AscendC::LocalTensor<float> dstLocal = this->dstQue.template AllocTensor<float>();
        AscendC::Mul(dstLocal, src1Local, src1Local, length);
        this->inQue1.template FreeTensor<float>(src1Local);
        this->dstQue.template EnQue<float>(dstLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint64_t length)
    {
        AscendC::LocalTensor<float> dstLocal = this->dstQue.template DeQue<float>();
        AscendC::DataCopy<float>(this->dstGlobal[offset], dstLocal, length);
        this->dstQue.template FreeTensor<float>(dstLocal);
    }

private:
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> dstQue;
};
} // namespace MSELossV2Namespace

#endif