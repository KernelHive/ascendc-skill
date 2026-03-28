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
 * \file mse_loss_v2_sum.h
 * \brief
 */

#ifndef MSE_LOSS_V2_SUM_H
#define MSE_LOSS_V2_SUM_H

#include "mse_loss_v2_base.h"

namespace MSELossV2Namespace {
template<typename T>
class MSELossV2Sum : public MSELossV2Base<T> {
public:
    __aicore__ inline MSELossV2Sum(AscendC::TPipe *pipe, const MSELossV2TilingData *tilingData) :
        MSELossV2Base<T>(pipe, tilingData)
    {
        this->pipe->InitBuffer(this->uploadQue, 1, this->BYTES_PER_BLOCK);
        this->pipe->InitBuffer(this->downloadQue, 1, this->BYTES_PER_BLOCK * AscendC::GetBlockNum());

        this->outLocal = this->uploadQue.template AllocTensor<float>();
        AscendC::Duplicate<float>(this->outLocal, 0.f, this->FLOATS_PER_BLOCK);
    }

    __aicore__ inline void Init(GM_ADDR dst, GM_ADDR src0, GM_ADDR src1, GM_ADDR usrWorkspace)
    {
        MSELossV2Base<T>::Init(dst, src0, src1, usrWorkspace);

        this->dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dst));
#if __CCE_AICORE__ == 200 // 310p
        this->syncGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(usrWorkspace));
#endif
        this->usrGlobal.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(
                usrWorkspace + this->BYTES_PER_BLOCK * (AscendC::GetBlockNum() + AscendC::GetBlockIdx())
            )
        );
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<float> cast0Local = this->castBuf0.template Get<float>();
        AscendC::LocalTensor<float> cast1Local = this->castBuf1.template Get<float>();

        for (uint64_t i = 0u; i < this->epochs; i++) {
            MSELossV2Base<T>::CopyIn(cast0Local, cast1Local, i * this->tileLength, this->tileLength);
            this->Compute(cast0Local, cast1Local, this->tileLength);
        }
        if (this->isLastCore && this->tailElems) {
            MSELossV2Base<T>::CopyIn(cast0Local, cast1Local, this->epochs * this->tileLength, 
                                     this->tailTileLength + MSELossV2Base<T>::FLOAT_COUNT_PER_BLOCK);
            this->Compute(cast0Local, cast1Local, this->tailTileLength + this->tailElems);
        } else if (this->tailTileLength) {
            MSELossV2Base<T>::CopyIn(cast0Local, cast1Local, this->epochs * this->tileLength, this->tailTileLength);
            this->Compute(cast0Local, cast1Local, this->tailTileLength);
        }
        this->CopyOut();
    }

private:
    __aicore__ inline void Compute(const AscendC::LocalTensor<float> &cast0,
        const AscendC::LocalTensor<float> &cast1, uint64_t tileLength)
    {
        AscendC::Sub<float>(cast0, cast0, cast1, tileLength);
        AscendC::Mul<float>(cast1, cast0, cast0, tileLength);
        this->ReduceSumBisect(cast1, tileLength, 1.0f);
    }

protected:
    __aicore__ inline void ReduceSumBisect(const AscendC::LocalTensor<float> &srcLocal, uint64_t tileLength, 
                                           const float scale)
    {
        uint64_t offset;
        while (tileLength > this->FLOATS_PER_BLOCK) {
            offset = (tileLength + 15u) >> 4 << 3; // Ceil(Ceil(tileLength, 8), 2) * 8
            AscendC::Add<float>(srcLocal, srcLocal, srcLocal[offset], tileLength - offset);
            tileLength = offset;
        }
        AscendC::Muls(srcLocal, srcLocal, scale, tileLength);
        AscendC::Add<float>(this->outLocal, srcLocal, this->outLocal, tileLength);
    }

    __aicore__ inline void CopyOut()
    {
        // MTE3 waits Vector
        this->uploadQue.template EnQue<float>(this->outLocal);
        this->outLocal = this->uploadQue.template DeQue<float>();
        AscendC::DataCopy(this->usrGlobal, this->outLocal, this->FLOATS_PER_BLOCK);
#if __CCE_AICORE__ == 200 // 310p
        AscendC::LocalTensor<int32_t> syncLocal = this->outLocal.template ReinterpretCast<int32_t>();
#endif
        this->uploadQue.template FreeTensor<float>(this->outLocal);
#if __CCE_AICORE__ == 200 // 310p
        AscendC::SyncAll(this->syncGlobal, syncLocal);
#else
        AscendC::SyncAll();
#endif

        if (AscendC::GetBlockIdx() == 0) {
            AscendC::LocalTensor<float> dstLocal = this->downloadQue.template AllocTensor<float>();
            AscendC::DataCopy(dstLocal, this->usrGlobal, AscendC::GetBlockNum() * this->FLOATS_PER_BLOCK);

            // Vector waits MTE2
            this->downloadQue.template EnQue<float>(dstLocal);
            dstLocal = this->downloadQue.template DeQue<float>();
            AscendC::ReduceSum<float>(dstLocal, dstLocal, dstLocal, AscendC::GetBlockNum() * this->FLOATS_PER_BLOCK);

            AscendC::LocalTensor<T> castLocal = this->uploadQue.template AllocTensor<T>();
#if __CCE_AICORE__ == 200 // 310p
            AscendC::Cast(castLocal, dstLocal, AscendC::RoundMode::CAST_NONE, this->FLOATS_PER_BLOCK);
            this->downloadQue.template FreeTensor<float>(dstLocal);
            this->uploadQue.template EnQue<T>(castLocal);
            castLocal = this->uploadQue.template DeQue<T>();
            AscendC::Duplicate<T>(castLocal[1], 0, this->HALF_PER_BLOCK - 1); // set others zero
            AscendC::DataCopy(this->dstGlobal, castLocal, this->HALF_PER_BLOCK);
#else  // 910b
            AscendC::Cast(castLocal, dstLocal, AscendC::RoundMode::CAST_RINT, 1);
            this->downloadQue.template FreeTensor<float>(dstLocal);
            this->uploadQue.template EnQue<T>(castLocal);
            castLocal = this->uploadQue.template DeQue<T>();
            AscendC::DataCopyExtParams copyParams{1, MSELossV2Base<T>::HALF_SIZE, 0, 0, 0}; // copy only first number to global
            AscendC::DataCopyPad(this->dstGlobal, castLocal, copyParams);
#endif
            this->uploadQue.template FreeTensor<T>(castLocal);
        }
    }

    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> uploadQue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> downloadQue;

    AscendC::LocalTensor<float> outLocal;
#if __CCE_AICORE__ == 200 // 310p
    AscendC::GlobalTensor<int32_t> syncGlobal;
#endif
    AscendC::GlobalTensor<float> usrGlobal;

    constexpr static uint64_t FLOATS_PER_BLOCK = 8ULL;
    constexpr static uint64_t HALF_PER_BLOCK = 16ULL;
    constexpr static uint64_t BYTES_PER_BLOCK = 32ULL;
};


template<>
class MSELossV2Sum<float> : public MSELossV2Base<float> {
public:
    __aicore__ inline MSELossV2Sum(AscendC::TPipe *pipe, const MSELossV2TilingData *tilingData) :
        MSELossV2Base<float>(pipe, tilingData)
    {
        this->pipe->InitBuffer(this->uploadQue, 1, this->BYTES_PER_BLOCK);
        this->pipe->InitBuffer(this->downloadQue, 1, this->BYTES_PER_BLOCK * AscendC::GetBlockNum());

        this->outLocal = this->uploadQue.template AllocTensor<float>();
        AscendC::Duplicate<float>(this->outLocal, 0.f, this->FLOATS_PER_BLOCK);
    }

    __aicore__ inline void Init(GM_ADDR dst, GM_ADDR src0, GM_ADDR src1, GM_ADDR usrWorkspace)
    {
        MSELossV2Base<float>::Init(dst, src0, src1, usrWorkspace);

        this->dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dst));
#if __CCE_AICORE__ == 200 // 310p
        this->syncGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(usrWorkspace));
#endif
        this->usrGlobal.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(
                usrWorkspace + this->BYTES_PER_BLOCK * (AscendC::GetBlockNum() + AscendC::GetBlockIdx())
            )
        );
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0u; i < this->epochs; i++) {
            MSELossV2Base<float>::CopyIn(i * this->tileLength, this->tileLength);
            this->Compute(this->tileLength);
        }

        if (this->isLastCore && this->tailElems) {
            MSELossV2Base<float>::CopyIn(this->epochs * this->tileLength, 
                                         this->tailTileLength + MSELossV2Base<float>::FLOAT_COUNT_PER_BLOCK);
            this->Compute(this->tailTileLength + this->tailElems);
        } else if (this->tailTileLength) {
            MSELossV2Base<float>::CopyIn(this->epochs * this->tileLength, this->tailTileLength);
            this->Compute(this->tailTileLength);
        }

        this->CopyOut();
    }

private:
    __aicore__ inline void Compute(uint64_t tileLength)
    {
        AscendC::LocalTensor<float> src0Local = this->inQue0.template DeQue<float>();
        AscendC::LocalTensor<float> src1Local = this->inQue1.template DeQue<float>();

        AscendC::Sub<float>(src0Local, src0Local, src1Local, tileLength);
        AscendC::Mul<float>(src1Local, src0Local, src0Local, tileLength);
        this->ReduceSumBisect(src1Local, tileLength, 1.0f);

        this->inQue0.template FreeTensor<float>(src0Local);
        this->inQue1.template FreeTensor<float>(src1Local);
    }

protected:
    __aicore__ inline void ReduceSumBisect(const AscendC::LocalTensor<float> &srcLocal, uint64_t tileLength,
                                           const float scale) 
    {
        uint64_t offset;
        while (tileLength > this->FLOATS_PER_BLOCK) {
            offset = (tileLength + 15u) >> 4 << 3; // Ceil(Ceil(tileLength, 8), 2) * 8
            AscendC::Add<float>(srcLocal, srcLocal, srcLocal[offset], tileLength - offset);
            tileLength = offset;
        }
        AscendC::Muls(srcLocal, srcLocal, scale, tileLength);
        AscendC::Add<float>(this->outLocal, srcLocal, this->outLocal, tileLength);
    }

    __aicore__ inline void CopyOut()
    {
        // MTE3 waits Vector
        this->uploadQue.template EnQue<float>(this->outLocal);
        this->outLocal = this->uploadQue.template DeQue<float>();
        AscendC::DataCopy(this->usrGlobal, this->outLocal, this->FLOATS_PER_BLOCK);
#if __CCE_AICORE__ == 200 // 310p
        AscendC::LocalTensor<int32_t> syncLocal = this->outLocal.template ReinterpretCast<int32_t>();
#endif
        this->uploadQue.template FreeTensor<float>(this->outLocal);
#if __CCE_AICORE__ == 200 // 310p
        AscendC::SyncAll(this->syncGlobal, syncLocal);
#else
        AscendC::SyncAll();
#endif

        if (AscendC::GetBlockIdx() == 0) {
            AscendC::LocalTensor<float> dstLocal = this->downloadQue.template AllocTensor<float>();
            AscendC::DataCopy(dstLocal, this->usrGlobal, AscendC::GetBlockNum() * this->FLOATS_PER_BLOCK);

            // Vector waits MTE2
            this->downloadQue.template EnQue<float>(dstLocal);
            dstLocal = this->downloadQue.template DeQue<float>();
            AscendC::ReduceSum<float>(dstLocal, dstLocal, dstLocal, AscendC::GetBlockNum() * this->FLOATS_PER_BLOCK);

            // Scalar waits Vector
            this->downloadQue.template EnQue<float>(dstLocal);
            dstLocal = this->downloadQue.template DeQue<float>();
#if __CCE_AICORE__ == 200 // 310p
            AscendC::Duplicate<float>(dstLocal[1], 0.f, this->FLOATS_PER_BLOCK - 1);
            AscendC::DataCopy(this->dstGlobal, dstLocal, this->FLOATS_PER_BLOCK);
#else
            AscendC::DataCopyExtParams copyParams{1, MSELossV2Base<float>::FLOAT_SIZE, 0, 0, 0};
            AscendC::DataCopyPad(this->dstGlobal, dstLocal, copyParams);
#endif
            this->downloadQue.template FreeTensor<float>(dstLocal);
        }
    }

    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> uploadQue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> downloadQue;

    AscendC::LocalTensor<float> outLocal;
#if __CCE_AICORE__ == 200 // 310p
    AscendC::GlobalTensor<int32_t> syncGlobal;
#endif
    AscendC::GlobalTensor<float> usrGlobal;

    constexpr static uint64_t FLOATS_PER_BLOCK = 8ULL;
    constexpr static uint64_t BYTES_PER_BLOCK = 32ULL;
};
} // namespace MSELossV2Namespace

#endif