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
 * \file gather_v3_tmpl_1.h
 * \brief
 */
#ifndef GATHER_V3_TMPL_1_H
#define GATHER_V3_TMPL_1_H

#include "gather_v3_base.h"

namespace GatherV3 {
using namespace AscendC;

constexpr int64_t KEY_CACHE_BASE = 20000;
constexpr int64_t KEY_CACHE_U_BASE = 30000;
constexpr int64_t KEY_ROBIN_BASE = 40000;
constexpr int64_t KEY_ROBIN_U_BASE = 50000;

constexpr int64_t KEY_ROBIN_GG_SP1 = 40331;
constexpr int64_t KEY_CACHE_BG_SP2 = 20132;
constexpr int64_t CMP_ALIGN_LEN = 128;

template <typename T_DATA, typename T_IDX>
class GatherV3Tmpl1 : public GatherV3Base<T_DATA, T_IDX> {
public:
    __aicore__ inline GatherV3Tmpl1(const GatherV3TilingData* tilingDataPtr)
        : GatherV3Base<T_DATA, T_IDX>(tilingDataPtr){};
    __aicore__ inline ~GatherV3Tmpl1() {
        if (xLoaded_) {
            PostCopy();
            FreeTensorX(xTensor_);
        }
    };
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR y, TPipe *pipe) {
        this->BaseInit(x, indices, y, pipe);

        PreCopy();

        if (this->tilingKey_ >= KEY_CACHE_U_BASE && this->tilingKey_ < KEY_CACHE_U_BASE + KEY_TYPE_RANGE ||
            this->tilingKey_ >= KEY_ROBIN_U_BASE && this->tilingKey_ < KEY_ROBIN_U_BASE + KEY_TYPE_RANGE) {
            this->doAlignInUB_ = true;
        } 

        if (this->tilingKey_ == KEY_ROBIN_GG_SP1) {
            this->pipe_->InitBuffer(xQue_, DB_BUF_CNT, this->xBufferSize_);
            this->pipe_->InitBuffer(yQue_, SG_BUF_CNT, this->yBufferSize_);
            this->pipe_->InitBuffer(this->idxQue_, SG_BUF_CNT, this->idxBufferSize_);
            this->pipe_->InitBuffer(idxFpBuf_,      this->CeilAlign(this->gTileSize_ * sizeof(float), BYTE_FULL_BLOCK));
            this->pipe_->InitBuffer(posBuf_,        this->gTileSize_ * sizeof(int32_t));
            this->pipe_->InitBuffer(cmpResBuf0_,    this->CeilAlign(this->gTileSize_ * sizeof(uint8_t), BYTE_BLOCK));
            this->pipe_->InitBuffer(cmpResBuf1_,    this->CeilAlign(this->gTileSize_ * sizeof(uint8_t), BYTE_BLOCK));
            this->pipe_->InitBuffer(idxReduceBuf_,  this->gTileSize_ * sizeof(int32_t));
            this->pipe_->InitBuffer(posReduceBuf_,  this->gTileSize_ * sizeof(int32_t));

            PostCopy();

            return;
        }

        if (this->tilingKey_ == KEY_CACHE_BG_SP2) {
            this->pipe_->InitBuffer(xQue_, SG_BUF_CNT, this->xBufferSize_);
            this->pipe_->InitBuffer(yQue_, this->bufferNum_, this->yBufferSize_);
            this->pipe_->InitBuffer(this->idxQue_, this->bufferNum_, this->idxBufferSize_);
            return;
        }

        if (this->doAlignInUB_) {
            this->pipe_->InitBuffer(xQue_, SG_BUF_CNT, this->xBufferSize_);
            this->pipe_->InitBuffer(xBuf_, this->xBufferSize_);
            this->pipe_->InitBuffer(yBuf_, this->yBufferSize_);
            this->pipe_->InitBuffer(yQue_, SG_BUF_CNT, this->yBufferSize_);
        } else {
            this->pipe_->InitBuffer(xQue_, this->bufferNum_, this->xBufferSize_);
            this->pipe_->InitBuffer(yQue_, this->bufferNum_, this->yBufferSize_);
        }

        this->pipe_->InitBuffer(this->idxQue_, this->bufferNum_, this->idxBufferSize_);
    };

    __aicore__ inline void ProcessSingleTile20330();
    __aicore__ inline void ProcessSingleTile20320();
    __aicore__ inline void ProcessSingleTile20310();
    __aicore__ inline void ProcessSingleTile20230();
    __aicore__ inline void ProcessSingleTile20220();
    __aicore__ inline void ProcessSingleTile20210();
    __aicore__ inline void ProcessSingleTile20130();
    __aicore__ inline void ProcessSingleTile20131();
    __aicore__ inline void ProcessSingleTile20132();
    __aicore__ inline void ProcessSingleTile20120();
    __aicore__ inline void ProcessSingleTile20110();

    __aicore__ inline void ProcessSingleTile30330();
    __aicore__ inline void ProcessSingleTile30320();
    __aicore__ inline void ProcessSingleTile30310();
    __aicore__ inline void ProcessSingleTile30230();
    __aicore__ inline void ProcessSingleTile30220();
    __aicore__ inline void ProcessSingleTile30210();
    __aicore__ inline void ProcessSingleTile30130();
    __aicore__ inline void ProcessSingleTile30131();
    __aicore__ inline void ProcessSingleTile30120();
    __aicore__ inline void ProcessSingleTile30110();

    __aicore__ inline void ProcessSingleTile40000();
    __aicore__ inline void ProcessSingleTile40331();
    __aicore__ inline void PreProcess40331();

    __aicore__ inline void ProcessSingleTile40330();
    __aicore__ inline void ProcessSingleTile40320();
    __aicore__ inline void ProcessSingleTile40310();
    __aicore__ inline void ProcessSingleTile50330();
    __aicore__ inline void ProcessSingleTile50320();
    __aicore__ inline void ProcessSingleTile50310();

protected:
    __aicore__ inline void PreCopy() {
        SetMaskCount();
        SetVectorMask<int16_t, MaskMode::COUNTER>(this->aBlockNum_ * BYTE_BLOCK / sizeof(int16_t));
    }

    __aicore__ inline void PostCopy() {
        SetMaskNorm();
        ResetMask();
    }

    __aicore__ inline void CopyUB2UB(const LocalTensor<T_DATA> &dstTensor, const LocalTensor<T_DATA> &srcTensor) {
        auto dst = dstTensor.template ReinterpretCast<int16_t>();
        auto src = srcTensor.template ReinterpretCast<int16_t>();

        Copy<int16_t, false>(dst, src, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
    };

    __aicore__ inline void CopyUB2UBRepeat(const LocalTensor<T_DATA> &dstTensor, const LocalTensor<T_DATA> &srcTensor, int64_t repeatTimes) {
        auto dst = dstTensor.template ReinterpretCast<int16_t>();
        auto src = srcTensor.template ReinterpretCast<int16_t>();

        CopyRepeatParams params = {
            1,
            1,
            static_cast<uint16_t>(this->gySize_ * this->aBlockNum_),
            static_cast<uint16_t>(this->gxSize_ * this->aBlockNum_)
        };

        int64_t loop = repeatTimes / VCOPY_MAX_REPEAT;
        int64_t tail = repeatTimes % VCOPY_MAX_REPEAT;
        int64_t dstOffset = 0;
        int64_t srcOffset = 0;

        if (loop > 0) {
            int64_t dstStep = VCOPY_MAX_REPEAT * this->gySize_ * BYTE_BLOCK / sizeof(int16_t) * this->aBlockNum_;
            int64_t srcStep = VCOPY_MAX_REPEAT * this->gxSize_ * BYTE_BLOCK / sizeof(int16_t) * this->aBlockNum_;
            for (int64_t i = 0; i < loop; i++) {
                Copy<int16_t, false>(dst[dstOffset], src[srcOffset], AscendC::MASK_PLACEHOLDER, VCOPY_MAX_REPEAT, params);
                dstOffset += dstStep;
                srcOffset += srcStep;
            }
        }

        if (tail > 0) {
            Copy<int16_t, false>(dst[dstOffset], src[srcOffset], AscendC::MASK_PLACEHOLDER, static_cast<uint8_t>(tail), params);
        }
    };

    __aicore__ inline LocalTensor<T_DATA> CopyInXDataAligned(int64_t offset, int64_t num, int64_t len) {
        auto xInTensor = this->xQue_.template AllocTensor<T_DATA>();

        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen   = (uint32_t)num * len * sizeof(T_DATA);
        params.srcStride  = 0;
        params.dstStride  = 0;
        DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0}; 

        auto dst = xInTensor.template ReinterpretCast<uint8_t>();
        DataCopyPad(dst, this->xGM_[offset * sizeof(T_DATA)], params, padParams);

        this->xQue_.EnQue(xInTensor);

        auto A = this->xQue_.template DeQue<T_DATA>();
        auto B = this->xBuf_.template Get<T_DATA>();

        PostCopy();
        AlignInUB(B, A, num, len);
        PreCopy();

        this->xQue_.template FreeTensor<T_DATA>(A);

        PipeBarrier<PIPE_V>();

        return B;
    };

    __aicore__ inline void PrefetchXData(int64_t offset, int64_t num, int64_t len) {
        LocalTensor<T_DATA> xInTensor = this->xQue_.template AllocTensor<T_DATA>();
        auto dst = xInTensor.template ReinterpretCast<uint8_t>();

        if (len * sizeof(T_DATA) % BYTE_BLOCK == 0) {
            DataCopy(dst, this->xGM_[offset * sizeof(T_DATA)], num * len * sizeof(T_DATA));
        } else {
            DataCopyExtParams params;
            params.blockCount = (uint16_t)num;
            params.blockLen   = (uint32_t)len * sizeof(T_DATA);
            params.srcStride  = 0;
            params.dstStride  = 0;
            DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0}; 
            DataCopyPad(dst, this->xGM_[offset * sizeof(T_DATA)], params, padParams);
        }

        this->xQue_.EnQue(xInTensor);
    };

    __aicore__ inline LocalTensor<T_DATA> GetCurXData(int64_t pos, int64_t num, int64_t totalNum, int64_t len) {
        LocalTensor<T_DATA> curTensor = this->xQue_.template DeQue<T_DATA>();
    
        int64_t nextNum = this->Min(totalNum - (pos + num), num);
        if (nextNum > 0) {
            PrefetchXData((pos + num) * len, nextNum, len);
        }

        return curTensor;
    };

    __aicore__ inline LocalTensor<T_DATA> CopyInXData(int64_t offset, int64_t num, int64_t len) {
        if (this->doAlignInUB_) {
            return CopyInXDataAligned(offset, num, len);
        }

        PrefetchXData(offset, num, len);

        return this->xQue_.template DeQue<T_DATA>();
    };

    __aicore__ inline void FreeTensorX(LocalTensor<T_DATA> &xTensor) {
        if (!this->doAlignInUB_) {
            this->xQue_.template FreeTensor<T_DATA>(xTensor);
        }
    };

    __aicore__ inline LocalTensor<T_DATA> GetTensorY() {
        if (this->doAlignInUB_) {
            return this->yBuf_.template Get<T_DATA>();
        } else {
            return this->yQue_.template AllocTensor<T_DATA>();
        }
    };

    __aicore__ inline void CopyOutYDataAligned(int64_t offset, LocalTensor<T_DATA> &yTensor, int64_t num, int64_t len) {
        auto yTensorForDeAlign = this->yQue_.template AllocTensor<T_DATA>();

        PipeBarrier<PIPE_V>();

        PostCopy();
        DeAlignInUB(yTensorForDeAlign, yTensor, num, len);
        PreCopy();

        this->yQue_.EnQue(yTensorForDeAlign);

        auto yOutTensor = this->yQue_.template DeQue<T_DATA>();
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen   = (uint32_t)num * len * sizeof(T_DATA);
        params.srcStride  = 0;
        params.dstStride  = 0;

        auto src = yOutTensor.template ReinterpretCast<uint8_t>();
        DataCopyPad(this->yGM_[offset * sizeof(T_DATA)], src, params);
        this->yQue_.template FreeTensor(yOutTensor);
    };

    __aicore__ inline void CopyOutYData(int64_t offset, LocalTensor<T_DATA> &yTensor, int64_t num, int64_t len) {
        if (this->doAlignInUB_) {
            CopyOutYDataAligned(offset, yTensor, num, len);
            return;
        } 

        this->yQue_.EnQue(yTensor);

        LocalTensor<T_DATA> yOutTensor = this->yQue_.template DeQue<T_DATA>();
        this->DoCopyUB2GM(offset, yOutTensor, num, len);
        this->yQue_.template FreeTensor(yOutTensor);
    };

    __aicore__ inline LocalTensor<int32_t> TypeNormIdx(LocalTensor<T_IDX> idxTensor) {
        auto idxFpTensor = this->idxFpBuf_.template Get<float>();
        Cast(idxFpTensor, idxTensor, RoundMode::CAST_CEIL, this->gyLength_);

        auto idxNormTensor = idxTensor.template ReinterpretCast<int32_t>();
        if constexpr (sizeof(T_IDX) == sizeof(int64_t)) {
            PipeBarrier<PIPE_V>();
            Cast(idxNormTensor, idxFpTensor, RoundMode::CAST_CEIL, this->gyLength_);
        }
        return idxNormTensor;
    }

    __aicore__ inline int64_t CalcIdxInRange(LocalTensor<int32_t> idxTensor, int64_t begin, int64_t end) {
        auto idxFpTensor = this->idxFpBuf_.template Get<float>();
        auto posTensor = this->posBuf_.template Get<int32_t>();
        auto cmpResTensor0 = this->cmpResBuf0_.template Get<uint8_t>();
        auto cmpResTensor1 = this->cmpResBuf1_.template Get<uint8_t>();
        auto idxReduceTensor_ = this->idxReduceBuf_.template Get<int32_t>();
        auto posReduceTensor_ = this->posReduceBuf_.template Get<int32_t>();

        CompareScalar(cmpResTensor0, idxFpTensor, (float)begin, CMPMODE::GE, this->CeilAlign(this->gyLength_, CMP_ALIGN_LEN));
        CompareScalar(cmpResTensor1, idxFpTensor, (float)end, CMPMODE::LT, this->CeilAlign(this->gyLength_, CMP_ALIGN_LEN));

        PipeBarrier<PIPE_V>();
        And(cmpResTensor0, cmpResTensor0, cmpResTensor1, this->gyLength_);
        PipeBarrier<PIPE_V>();

        auto pattern = cmpResTensor0.template ReinterpretCast<uint32_t>();

        uint64_t rsvdCnt = 0;
        GatherMask(idxReduceTensor_, idxTensor, pattern, true, this->gyLength_, {1, 1, 8, 0}, rsvdCnt);
        GatherMask(posReduceTensor_, posTensor, pattern, true, this->gyLength_, {1, 1, 8, 0}, rsvdCnt);
        PipeBarrier<PIPE_V>();
        Muls(posReduceTensor_, posReduceTensor_, (int32_t)this->aAlignSize_, rsvdCnt);
        Adds(idxReduceTensor_, idxReduceTensor_, -(int32_t)begin, rsvdCnt);
        PipeBarrier<PIPE_V>();
        Muls(idxReduceTensor_, idxReduceTensor_, (int32_t)this->aAlignSize_, rsvdCnt);

        return (int64_t)rsvdCnt;
    }

    template <bool srcHorizon = true>
    __aicore__ inline void TransposeB16(const LocalTensor<uint16_t>& D, const LocalTensor<uint16_t>& S, int64_t N);

    template <bool srcHorizon = true>
    __aicore__ inline void TransposeB8(const LocalTensor<uint8_t>& D, const LocalTensor<uint8_t>& S, int64_t N);

    __aicore__ inline void TransposeB8FoldH2V(const LocalTensor<uint8_t>& D, const LocalTensor<uint8_t>& S, int64_t N);

    __aicore__ inline void TransposeB8FoldV2H(const LocalTensor<uint8_t>& D, const LocalTensor<uint8_t>& S, int64_t N);

    __aicore__ inline void DeAlignByTransposeB8(const LocalTensor<T_DATA>& dst, const LocalTensor<T_DATA>& src,
                                                int64_t lineNum, int64_t elemNum);

    __aicore__ inline void DeAlignByReduceV2(const LocalTensor<T_DATA>& dst, const LocalTensor<T_DATA>& src, int64_t lineNum,
                                      int64_t elemNum);

    __aicore__ inline void DeAlignInUB(const LocalTensor<T_DATA>& dst, const LocalTensor<T_DATA>& src, 
                                       int64_t lineNum, int64_t elemNum);

    __aicore__ inline void AlignInUB(const LocalTensor<T_DATA>& dst, const LocalTensor<T_DATA>& src, 
                                       int64_t lineNum, int64_t elemNum);

    __aicore__ inline void AlignByTransposeB16(const LocalTensor<T_DATA>& dst, const LocalTensor<T_DATA> &src, int64_t m,
                                        int64_t n);

    __aicore__ inline void AlignByTransposeB8(const LocalTensor<T_DATA>& dst, const LocalTensor<T_DATA> &src, int64_t m,
                                              int64_t n);

protected:
    TQue<QuePosition::VECIN, 1>     xQue_;
    TQue<QuePosition::VECOUT, 1>    yQue_;

    TBuf<QuePosition::VECCALC>      xBuf_;
    TBuf<QuePosition::VECCALC>      yBuf_;
    TBuf<QuePosition::VECCALC>      idxFpBuf_;
    TBuf<QuePosition::VECCALC>      posBuf_;
    TBuf<QuePosition::VECCALC>      cmpResBuf0_;
    TBuf<QuePosition::VECCALC>      cmpResBuf1_;
    TBuf<QuePosition::VECCALC>      idxReduceBuf_;
    TBuf<QuePosition::VECCALC>      posReduceBuf_;

    bool doAlignInUB_ = false;

    bool xLoaded_ = false;
    LocalTensor<T_DATA> xTensor_;
};

// type size 2 4 8都ReinterpretCast到LocalTensor<int16_t>执行
template <typename T_DATA, typename T_IDX>
__aicore__ inline void GatherV3Tmpl1<T_DATA, T_IDX>::AlignByTransposeB16(const LocalTensor<T_DATA> &dst,  const LocalTensor<T_DATA> &src, int64_t m, int64_t n) {
    auto A = src.template ReinterpretCast<uint16_t>();
    auto B = dst.template ReinterpretCast<uint16_t>();
    
    int64_t nNorm = (n * sizeof(T_DATA) / sizeof(uint16_t));
    int64_t mAligned = this->CeilAlign(m, TRANSPOSE_H_2B);
    int64_t nAligned = this->CeilAlign(nNorm, TRANSPOSE_W_2B);

    // 1. A(m, n) reshape to A(16, align(m, 256)/16, n)
    //      Nothing to do with reshape. Just reserve enough memory.
    //      The size of A and B must be CeilAlign(m, 256) * CeilAlign(n, 16) * sizeof(T_DATA)

    // 2. A(16, align(m, 256)/16, n) vnchwconv to B(align(m, 256)/16, n, 16)
    TransposeB16<true>(B, A, mAligned / TRANSPOSE_H_2B * nNorm);

    PipeBarrier<PIPE_V>();

    // 3. B(align(m, 256)/16, n, 16) copy_ub_to_ub to A(align(m, 256)/16, align(n, 16), 16)
    SetMaskCount();
    SetVectorMask<uint16_t, MaskMode::COUNTER>(nNorm * BYTE_BLOCK / sizeof(uint16_t));
    Copy<uint16_t, false>(A, B, AscendC::MASK_PLACEHOLDER, 
                          mAligned / TRANSPOSE_W_2B, {1, 1, (uint16_t)nAligned, (uint16_t)nNorm});
    SetMaskNorm();
    ResetMask();

    PipeBarrier<PIPE_V>();

    // 4. A(align(m, 256)/16, align(n, 16), 16) vnchwconv to B(16, align(m, 256)/16, align(n, 16))
    TransposeB16<false>(B, A, mAligned / TRANSPOSE_H_2B * nAligned);
}

// type size 2 4 8都ReinterpretCast到LocalTensor<int16_t>执行
template <typename T_DATA, typename T_IDX>
__aicore__ inline void GatherV3Tmpl1<T_DATA, T_IDX>::AlignByTransposeB8(const LocalTensor<T_DATA> &dst,  const LocalTensor<T_DATA> &src, int64_t m, int64_t n) {
    auto A = src.template ReinterpretCast<uint8_t>();
    auto B = dst.template ReinterpretCast<uint8_t>();

    int64_t nNorm = (n * sizeof(T_DATA) / sizeof(uint8_t));
    int64_t mAligned = this->CeilAlign(m, TRANSPOSE_H_1B);
    int64_t nAligned = this->CeilAlign(nNorm, TRANSPOSE_W_1B);

    TransposeB8FoldH2V(B, A, mAligned / TRANSPOSE_H_1B * nNorm);

    PipeBarrier<PIPE_V>();

    // 完整block搬运，cast到uint16_t
    auto tA = A.template ReinterpretCast<uint16_t>();
    auto tB = B.template ReinterpretCast<uint16_t>(); 

    SetMaskCount();
    SetVectorMask<uint16_t, MaskMode::COUNTER>(nNorm * BYTE_BLOCK / sizeof(uint16_t));
    Copy<uint16_t, false>(tA, tB, AscendC::MASK_PLACEHOLDER, 
                          mAligned / TRANSPOSE_W_1B, {1, 1, (uint16_t)nAligned, (uint16_t)nNorm});
    SetMaskNorm();
    ResetMask();

    PipeBarrier<PIPE_V>();

    TransposeB8FoldV2H(B, A, mAligned / TRANSPOSE_H_1B * nAligned);
}

template <typename T_DATA, typename T_IDX>
template <bool srcHorizon>
__aicore__ inline void GatherV3Tmpl1<T_DATA, T_IDX>::TransposeB16(const LocalTensor<uint16_t> &D, const LocalTensor<uint16_t> &S, int64_t N) {
    uint64_t dstBase = (uint64_t)D.GetPhyAddr();
    uint64_t srcBase = (uint64_t)S.GetPhyAddr();

    uint64_t dstStride = srcHorizon ? 32 : 32 * N;
    uint64_t srcStride = srcHorizon ? 32 * N : 32;

    uint64_t dstRepeatStride = srcHorizon ? 16 : 1;
    uint64_t srcRepeatStride = srcHorizon ? 1 : 16;

    if (N == 1) {
        dstRepeatStride = 0;
        srcRepeatStride = 0;
    }

    uint64_t dst[16] = {};
    uint64_t src[16] = {};
    for (int i = 0; i < 16; i++) {
        dst[i] = dstBase + i * dstStride;
        src[i] = srcBase + i * srcStride;
    }

    TransDataTo5HD<uint16_t>(dst, src, {false, false, (uint8_t)N, (uint16_t)dstRepeatStride, (uint16_t)srcRepeatStride});
}


template <typename T_DATA, typename T_IDX>
__aicore__ inline void GatherV3Tmpl1<T_DATA, T_IDX>::DeAlignByReduceV2(const LocalTensor<T_DATA> &dst, const LocalTensor<T_DATA> &src,
                                                                 int64_t lineNum, int64_t elemNum) {
    auto dstTensor = dst.template ReinterpretCast<uint16_t>();
    auto srcTensor = src.template ReinterpretCast<uint16_t>();
    int64_t mask = elemNum * sizeof(T_DATA) / sizeof(uint16_t);
    uint16_t lineBlockNum = this->CeilDiv(mask * sizeof(uint16_t), BYTE_BLOCK);

    uint64_t totalCnt = 0;
    GatherMask(dstTensor, srcTensor, VREDUCE_MASK_ALL, true, mask, {1, (uint16_t)lineNum, lineBlockNum, 0}, totalCnt);                                                 
}

template <typename T_DATA, typename T_IDX>
template <bool srcHorizon>
__aicore__ inline void GatherV3Tmpl1<T_DATA, T_IDX>::TransposeB8(const LocalTensor<uint8_t>& D, const LocalTensor<uint8_t>& S, int64_t N) {
    TransDataTo5HDParams params;

    uint64_t dstBase = (uint64_t)D.GetPhyAddr();
    uint64_t srcBase = (uint64_t)S.GetPhyAddr();

    uint64_t dstStride;
    uint64_t srcStride;

    uint64_t dstRepeatStride;
    uint64_t srcRepeatStride;

    uint64_t dst[TRANSPOSE_BLOCK_NUM] = {};
    uint64_t src[TRANSPOSE_BLOCK_NUM] = {};

    dstStride = srcHorizon ? 32 : 32 * N;
    srcStride = srcHorizon ? 32 * N : 32;

    dstRepeatStride = srcHorizon ? 32 : 1;
    srcRepeatStride = srcHorizon ? 1 : 32;

    if (N == 1) {
        dstRepeatStride = 0;
        srcRepeatStride = 0;
    }

    // dst false src false
    for (int i = 0; i < TRANSPOSE_BLOCK_NUM; i++) {
        dst[i] = dstBase + i * dstStride;
        src[i] = srcBase + i * srcStride;
    }
    TransDataTo5HD<uint8_t>(dst, src, {false, false, (uint8_t)N, (uint16_t)dstRepeatStride, (uint16_t)srcRepeatStride});

    // dst false src true
    for (int i = 0; i < TRANSPOSE_BLOCK_NUM; i++) {
        dst[i] += dstStride * 16;
    }
    TransDataTo5HD<uint8_t>(dst, src, {false, true, (uint8_t)N, (uint16_t)dstRepeatStride, (uint16_t)srcRepeatStride});

    // dst true src true
    for (int i = 0; i < TRANSPOSE_BLOCK_NUM; i++) {
        src[i] += srcStride * 16;
    }
    TransDataTo5HD<uint8_t>(dst, src, {true, true, (uint8_t)N, (uint16_t)dstRepeatStride, (uint16_t)srcRepeatStride});

    // dst true src false
    for (int i = 0; i < TRANSPOSE_BLOCK_NUM; i++) {
        dst[i] -= dstStride * 16;
    }
    TransDataTo5HD<uint8_t>(dst, src, {true, false, (uint8_t)N, (uint16_t)dstRepeatStride, (uint16_t)srcRepeatStride});
}

template <typename T_DATA, typename T_IDX>
__aicore__ inline void GatherV3Tmpl1<T_DATA, T_IDX>::TransposeB8FoldH2V(const LocalTensor<uint8_t>& D, const LocalTensor<uint8_t>& S, int64_t N) {
    TransDataTo5HDParams params;

    uint64_t dstBase = (uint64_t)D.GetPhyAddr();
    uint64_t srcBase = (uint64_t)S.GetPhyAddr();

    uint64_t dst[TRANSPOSE_BLOCK_NUM] = {};
    uint64_t src[TRANSPOSE_BLOCK_NUM] = {};

    uint16_t dstRepeatStride = 32;
    uint16_t srcRepeatStride = 1;
    
    if (N <= 2) {
        dstRepeatStride = 0;
        srcRepeatStride = 0;
    }

    bool evenN = (N % 2 == 0);

    // dst false src false
    for (int i = 0; i < TRANSPOSE_BLOCK_NUM; i++) {
        dst[i] = dstBase + i * 32;
        src[i] = srcBase + i * 32 * N;
    }
    TransDataTo5HD<uint8_t>(dst, src, {false, false, (uint8_t)(N - N / 2), dstRepeatStride, srcRepeatStride});

    // dst false src true
    for (int i = 0; i < TRANSPOSE_BLOCK_NUM; i++) {
        dst[i] += 32 * 16; 
    }

    if (N / 2 == 1) {
        TransDataTo5HD<uint8_t>(dst, src, {false, true, (uint8_t)(N / 2), 0, 0});
    } else {
        TransDataTo5HD<uint8_t>(dst, src, {false, true, (uint8_t)(N / 2), dstRepeatStride, srcRepeatStride});
    }

    // dst true src false
    for (int i = 0; i < TRANSPOSE_BLOCK_NUM; i++) {
        src[i] += (N - N / 2) * 32;
    }

    if (N / 2 == 1) {
        TransDataTo5HD<uint8_t>(dst, src, {true, evenN, (uint8_t)(N / 2), 0, 0});
    } else {
        TransDataTo5HD<uint8_t>(dst, src, {true, evenN, (uint8_t)(N / 2), dstRepeatStride, srcRepeatStride});
    }

    // dst true src true
    if (!evenN) {
        for (int i = 0; i < TRANSPOSE_BLOCK_NUM; i++) {
            src[i] -= 32;
        }
    }

    for (int i = 0; i < TRANSPOSE_BLOCK_NUM; i++) {
        dst[i] -= 32 * 16;
    }
    TransDataTo5HD<uint8_t>(dst, src, {true, !evenN, (uint8_t)(N - N / 2), dstRepeatStride, srcRepeatStride});
}

template <typename T_DATA, typename T_IDX>
__aicore__ inline void GatherV3Tmpl1<T_DATA, T_IDX>::TransposeB8FoldV2H(const LocalTensor<uint8_t>& D, const LocalTensor<uint8_t>& S, int64_t N) {
    TransDataTo5HDParams params;

    uint64_t dstBase = (uint64_t)D.GetPhyAddr();
    uint64_t srcBase = (uint64_t)S.GetPhyAddr();

    uint64_t dst[TRANSPOSE_BLOCK_NUM] = {};
    uint64_t src[TRANSPOSE_BLOCK_NUM] = {};

    uint16_t dstRepeatStride = 1;
    uint16_t srcRepeatStride = 32;

    if (N <= 2) {
        dstRepeatStride = 0;
        srcRepeatStride = 0;
    }

    bool evenN = (N % 2 == 0);

    for (int i = 0; i < TRANSPOSE_BLOCK_NUM; i++) {
        dst[i] = dstBase + i * 32 * N;
        src[i] = srcBase + i * 32;
    }
    TransDataTo5HD<uint8_t>(dst, src, {false, false, (uint8_t)(N - N / 2), dstRepeatStride, srcRepeatStride});

    for (int i = 0; i < TRANSPOSE_BLOCK_NUM; i++) {
        src[i] += 32 * 16; 
    }

    if (N / 2 == 1) {
        TransDataTo5HD<uint8_t>(dst, src, {true, false, (uint8_t)(N / 2), 0, 0});
    } else {
        TransDataTo5HD<uint8_t>(dst, src, {true, false, (uint8_t)(N / 2), dstRepeatStride, srcRepeatStride});
    }

    for (int i = 0; i < TRANSPOSE_BLOCK_NUM; i++) {
        dst[i] += (N - N / 2) * 32;
    }

    if (N / 2 == 1) {
        TransDataTo5HD<uint8_t>(dst, src, {evenN, true, (uint8_t)(N / 2), 0, 0});
    } else {
        TransDataTo5HD<uint8_t>(dst, src, {evenN, true, (uint8_t)(N / 2), dstRepeatStride, srcRepeatStride});
    }

    // dst true src true
    if (!evenN) {
        for (int i = 0; i < TRANSPOSE_BLOCK_NUM; i++) {
            dst[i] -= 32;
        } 
    }

    for (int i = 0; i < TRANSPOSE_BLOCK_NUM; i++) {
        src[i] -= 32 * 16;
    }
    TransDataTo5HD<uint8_t>(dst, src, {!evenN, true, (uint8_t)(N - N / 2), dstRepeatStride, srcRepeatStride});
}

template <typename T_DATA, typename T_IDX>
__aicore__ inline void GatherV3Tmpl1<T_DATA, T_IDX>::DeAlignByTransposeB8(const LocalTensor<T_DATA> &dst, const LocalTensor<T_DATA> &src,
                                                                       int64_t m, int64_t n) {
    // B8数据类型UB内转置去对齐 
    auto A = src.template ReinterpretCast<uint8_t>();
    auto B = dst.template ReinterpretCast<uint8_t>();

    int64_t mAligned = this->CeilAlign(m, TRANSPOSE_H_1B);
    int64_t nAligned = this->CeilAlign((n * sizeof(T_DATA) / sizeof(uint8_t)), 32);

    TransposeB8FoldH2V(B, A, mAligned / TRANSPOSE_H_1B * nAligned);

    PipeBarrier<PIPE_V>();

    // 完整block搬运，cast到uint16_t
    auto tA = A.template ReinterpretCast<uint16_t>();
    auto tB = B.template ReinterpretCast<uint16_t>(); 
    uint64_t totalCnt     = 0;
    uint64_t mask         = n * 32 / sizeof(uint16_t);
    uint16_t repeatTimes  = (uint16_t)(mAligned / 32);
    uint16_t repeatStride = nAligned;
    GatherMask(tA, tB, VREDUCE_MASK_ALL, true, mask, {1, repeatTimes, repeatStride, 0}, totalCnt);

    PipeBarrier<PIPE_V>();

    TransposeB8FoldV2H(B, A, mAligned / TRANSPOSE_H_1B * n);
}

template <typename T_DATA, typename T_IDX>
__aicore__ inline void GatherV3Tmpl1<T_DATA, T_IDX>::DeAlignInUB(const LocalTensor<T_DATA> &dst, const LocalTensor<T_DATA> &src,
                                                                 int64_t lineNum, int64_t elemNum) {
    if constexpr (sizeof(T_DATA) >= sizeof(int16_t)) {
        DeAlignByReduceV2(dst, src, lineNum, elemNum);
    } else {
        if (elemNum % 2 == 0) {
            DeAlignByReduceV2(dst, src, lineNum, elemNum);
        } else {
            DeAlignByTransposeB8(dst, src, lineNum, elemNum);
        }
    }
}

template <typename T_DATA, typename T_IDX>
__aicore__ inline void GatherV3Tmpl1<T_DATA, T_IDX>::AlignInUB(const LocalTensor<T_DATA> &dst, const LocalTensor<T_DATA> &src,
                                                                 int64_t lineNum, int64_t elemNum) {
    if constexpr (sizeof(T_DATA) >= sizeof(int16_t)) {
        AlignByTransposeB16(dst, src, lineNum, elemNum);
    } else {
        if (elemNum % 2 == 0) {
            AlignByTransposeB16(dst, src, lineNum, elemNum);
        } else {
            AlignByTransposeB8(dst, src, lineNum, elemNum);
        }
    }
}

// key      x            y       core
// 20330    gx(entire)   g       [b, p]
template <typename T_DATA, typename T_IDX>
__aicore__ inline void GatherV3Tmpl1<T_DATA, T_IDX> :: ProcessSingleTile20330() {
    this->CalcBaseOffset();

    int64_t xOffset = this->xBaseOffset_;
    int64_t yOffset = this->yBaseOffset_;
    int64_t idxOffset = this->idxBaseOffset_;

    // 输入的整个ga轴 搬入时对齐
    auto xTensor = CopyInXData(xOffset, this->gxSize_, this->aSize_);

    int64_t lineNum = 0;
    for (int64_t gLoop = 0; gLoop < this->gyLength_; gLoop += this->ubLineLimit_) {
        lineNum = this->Min(this->ubLineLimit_, this->gyLength_ - gLoop);

        // 读入g轴切分范围内的索引
        auto idxTensor = this->CopyInIdx(idxOffset + gLoop, 1, lineNum);

        // gather多行
        auto yTensor = GetTensorY();

        this->SyncM2toS();
        for (int64_t idxLoop = 0; idxLoop < lineNum; idxLoop++) {
            T_IDX gxId = idxTensor.GetValue(idxLoop);
            this->CheckIdxValue(gxId);
            this->CopyUB2UB(yTensor[idxLoop * this->aAlignSize_], xTensor[gxId * this->aAlignSize_]);
        }

        this->FreeTensorIdx(idxTensor);

        // 整体搬出
        CopyOutYData(yOffset, yTensor, lineNum, this->aSize_);
        yOffset += lineNum * this->aSize_;
    }

    FreeTensorX(xTensor);
}

// key      x            y       core
// 20320    gx(entire)   p       [b, p_out]
template <typename T_DATA, typename T_IDX>
__aicore__ inline void GatherV3Tmpl1<T_DATA, T_IDX> :: ProcessSingleTile20320() {
    ProcessSingleTile20310();
}

// key      x            y       core
// 20310    gx(entire)   b       b_out
template <typename T_DATA, typename T_IDX>
__aicore__ inline void GatherV3Tmpl1<T_DATA, T_IDX> :: ProcessSingleTile20310() {
    this->CalcBaseOffset();

    int64_t xOffset = this->xBaseOffset_;
    int64_t yOffset = this->yBaseOffset_;
    int64_t idxOffset = this->idxBaseOffset_;

    // 读入b轴切分范围内的所有索引 搬入时整体对齐
    auto idxTensor = this->CopyInIdx(idxOffset, 1, this->bLength_ * this->gySize_);
    auto yTensor = GetTensorY();

    int64_t yOffsetInUB = 0;
    int64_t idxOffsetInUB = 0;
    this->SyncM2toS();
    for (int64_t bLoop = 0; bLoop < this->bLength_; bLoop++) {
        for (int64_t pLoop = 0; pLoop < this->pLength_; pLoop++) {
            // bp对应的整个ga轴 搬入时a轴对齐
            auto xTensor = CopyInXData(xOffset, this->gxSize_, this->aSize_);
            xOffset += this->xpElemSize_;

            // gather多行
            for (int64_t gLoop = 0; gLoop < this->gySize_; gLoop++) {
                T_IDX gxId = idxTensor.GetValue(idxOffsetInUB + gLoop);
                this->CheckIdxValue(gxId);
                this->CopyUB2UB(yTensor[yOffsetInUB], xTensor[gxId * this->aAlignSize_]);
                yOffsetInUB += this->aAlignSize_;
            }
            FreeTensorX(xTensor);
        }
        idxOffsetInUB += this->gySize_;
    }

    CopyOutYData(yOffset, yTensor, this->bLength_ * this->pLength_ * this->gySize_, this->aSize_);

    this->FreeTensorIdx(idxTensor);
}

// key      x       y       core
// 20230    p       g       [b, p_out]
template <typename T_DATA, typename T_IDX>
__aicore__ inline void GatherV3Tmpl1<T_DATA, T_IDX> :: ProcessSingleTile20230() {
    this->CalcBaseOffset();

    int64_t xOffset = this->xBaseOffset_;
    int64_t yOffset = this->yBaseOffset_;
    int64_t idxOffset = this->idxBaseOffset_;

    // 读入p轴切分范围内的所有ga轴 搬入时a轴对齐
    auto xTensor = CopyInXData(xOffset, this->pLength_ * this->gxSize_, this->aSize_);
    int64_t xOffsetInUB = 0;

    for (int64_t pLoop = 0; pLoop < this->pLength_; pLoop++) {
        idxOffset = this->idxBaseOffset_;
        for (int64_t gLoop = 0; gLoop < this->gySize_; gLoop += this->ubLineLimit_) {
            int64_t lineNum = this->Min(this->ubLineLimit_, this->gySize_ - gLoop);

            auto idxTensor = this->CopyInIdx(idxOffset, 1, lineNum);
            idxOffset += lineNum;

            auto yTensor = GetTensorY();

            int64_t yOffsetInUB = 0;

            this->SyncM2toS();
            for (int64_t idxLoop = 0; idxLoop < lineNum; idxLoop++) {
                T_IDX gxId = idxTensor.GetValue(idxLoop);
                this->CheckIdxValue(gxId);
                this->CopyUB2UB(yTensor[yOffsetInUB], xTensor[xOffsetInUB + gxId * this->aAlignSize_]);
                yOffsetInUB += this->aAlignSize_;
            }

            this->FreeTensorIdx(idxTensor);

            CopyOutYData(yOffset, yTensor, lineNum, this->aSize_);
            yOffset += lineNum * this->aSize_;
        }
        xOffsetInUB += this->gxSize_ * this->aAlignSize_;
    }

    FreeTensorX(xTensor);
}

// key      x       y       core
//  20220   p       p       [b, p_out]
template <typename T_DATA, typename T_IDX>
__aicore__ inline void GatherV3Tmpl1<T_DATA, T_IDX> :: ProcessSingleTile20220() {
    ProcessSingleTile20210();
}

// key      x       y       core
//  20210   p       b       b_out
template <typename T_DATA, typename T_IDX>
__aicore__ inline void GatherV3Tmpl1<T_DATA, T_IDX> :: ProcessSingleTile20210() {
    this->CalcBaseOffset();

    int64_t xOffset = this->xBaseOffset_;
    int64_t yOffset = this->yBaseOffset_;
    int64_t idxOffset = this->idxBaseOffset_;

    auto idxTensor = this->CopyInIdx(idxOffset, 1, this->bLength_ * this->gySize_);
    int64_t idxOffsetInUB = 0;

    auto yTensor = GetTensorY();
    int64_t yOffsetInUB = 0;

    this->SyncM2toS();
    for (int64_t bLoop = 0; bLoop < this->bLength_; bLoop++) {
        for (int64_t pLoop = 0; pLoop <  this->pLength_; pLoop += this->ubLineLimit_) {
            int64_t lineNum = this->Min(this->ubLineLimit_,  this->pLength_ - pLoop);

            yOffsetInUB = (bLoop * this->pLength_ + pLoop) * this->gySize_ * this->aAlignSize_;

            auto xTensor = CopyInXData(xOffset, lineNum * this->gxSize_, this->aSize_);
            xOffset += lineNum * this->gxSize_ * this->aSize_;

            for (int64_t gLoop = 0; gLoop < this->gySize_; gLoop++) {
                T_IDX gxId = idxTensor.GetValue(idxOffsetInUB + gLoop);
                this->CheckIdxValue(gxId);

                this->CopyUB2UBRepeat(yTensor[yOffsetInUB], xTensor[gxId * this->aAlignSize_], lineNum);
                yOffsetInUB += this->aAlignSize_;
            }

            FreeTensorX(xTensor);
        }
        idxOffsetInUB += this->gySize_;
    }

    this->FreeTensorIdx(idxTensor);

    CopyOutYData(yOffset, yTensor, this->bLength_ * this->pLength_ * this->gySize_, this->aSize_);
}

// key      x       y       core
//  20130   b       g       b_out
template <typename T_DATA, typename T_IDX>
__aicore__ inline void GatherV3Tmpl1<T_DATA, T_IDX> :: ProcessSingleTile20130() {
    this->CalcBaseOffset();

    int64_t xOffset = this->xBaseOffset_;
    int64_t yOffset = this->yBaseOffset_;
    int64_t idxOffset = this->idxBaseOffset_;

    // 搬入b轴切分范围所有pga
    auto xTensor = CopyInXData(xOffset, this->bLength_ *  this->pSize_ * this->gxSize_, this->aSize_);
    int64_t xOffsetInUB = 0;

    // 核内以g切分做循环 每次gather满搬出
    for (int64_t bLoop = 0; bLoop < this->bLength_; bLoop++) {
        for (int64_t gLoop = 0; gLoop < this->gySize_; gLoop += this->ubLineLimit_ ) {
            int64_t lineNum = this->Min(this->ubLineLimit_, this->gySize_ - gLoop);

            auto idxTensor = this->CopyInIdx(idxOffset + gLoop, 1, lineNum);

            this->SyncM2toS();
            for (int64_t pLoop = 0; pLoop <  this->pSize_; pLoop++) {
                auto yTensor = GetTensorY();
                int64_t yOffsetInUB = 0;

                for (int64_t i = 0; i < lineNum; i++) {
                    T_IDX gxId = idxTensor.GetValue(i);
                    this->CheckIdxValue(gxId);

                    this->CopyUB2UB(yTensor[yOffsetInUB], 
                                    xTensor[xOffsetInUB + (pLoop * this->gxSize_ + gxId) * this->aAlignSize_]);
                    yOffsetInUB += this->aAlignSize_;
                }

                yOffset = this->yBaseOffset_ + 
                          bLoop * this->ybElemSize_ + 
                          pLoop * this->ypElemSize_ + 
                          gLoop * this->aSize_;

                CopyOutYData(yOffset, yTensor, lineNum, this->aSize_);
            }
            this->FreeTensorIdx(idxTensor);
        }
        xOffsetInUB += this->pSize_ * this->gxSize_ * this->aAlignSize_;
        idxOffset += this->gySize_;
    }

    // 释放资源
    FreeTensorX(xTensor);
}

// key      x       y       core
//  20131   b       g       b_out
template <typename T_DATA, typename T_IDX>
__aicore__ inline void GatherV3Tmpl1<T_DATA, T_IDX> :: ProcessSingleTile20131() {
    this->CalcBaseOffset();

    int64_t yOffset = this->yBaseOffset_;
    int64_t idxOffset = this->idxBaseOffset_;

    if (!xLoaded_) {
        xTensor_ = CopyInXData(0, this->gxSize_, this->aSize_);
        xLoaded_ = true;
    }
    // 搬入b轴切分范围所有pga
    int64_t lineNum = this->gyLength_;

    auto idxTensor = this->CopyInIdx(idxOffset, 1, lineNum);
    auto yTensor = GetTensorY();
    int64_t yOffsetInUB = 0;
    int64_t loopNum = lineNum / 8;
    int64_t tailLen = lineNum % 8;

    int64_t idxPos = 0;

    T_IDX gxIdArray[8];

    this->SyncM2toS();

    for (int64_t loop = 0; loop < loopNum; loop++) {
        for (int64_t idxLoop = 0; idxLoop < 8; idxLoop++) {
            gxIdArray[idxLoop] = idxTensor.GetValue(idxPos++);
            this->CheckIdxValue(gxIdArray[idxLoop]);
        }

        this->SyncStoV();

        CopyUB2UB(yTensor[yOffsetInUB + 0 * this->aAlignSize_], xTensor_[gxIdArray[0] * this->aAlignSize_]);
        CopyUB2UB(yTensor[yOffsetInUB + 1 * this->aAlignSize_], xTensor_[gxIdArray[1] * this->aAlignSize_]);
        CopyUB2UB(yTensor[yOffsetInUB + 2 * this->aAlignSize_], xTensor_[gxIdArray[2] * this->aAlignSize_]);
        CopyUB2UB(yTensor[yOffsetInUB + 3 * this->aAlignSize_], xTensor_[gxIdArray[3] * this->aAlignSize_]);
        CopyUB2UB(yTensor[yOffsetInUB + 4 * this->aAlignSize_], xTensor_[gxIdArray[4] * this->aAlignSize_]);
        CopyUB2UB(yTensor[yOffsetInUB + 5 * this->aAlignSize_], xTensor_[gxIdArray[5] * this->aAlignSize_]);
        CopyUB2UB(yTensor[yOffsetInUB + 6 * this->aAlignSize_], xTensor_[gxIdArray[6] * this->aAlignSize_]);
        CopyUB2UB(yTensor[yOffsetInUB + 7 * this->aAlignSize_], xTensor_[gxIdArray[7] * this->aAlignSize_]);
        yOffsetInUB += 8 * this->aAlignSize_;
    }

    if (tailLen > 0) {
        for (int64_t idxLoop = 0; idxLoop < tailLen; idxLoop++) {
            gxIdArray[idxLoop] = idxTensor.GetValue(idxPos++);
            this->CheckIdxValue(gxIdArray[idxLoop]);
        }

        this->SyncStoV();

        for (int64_t idxLoop = 0; idxLoop < tailLen; idxLoop++) {
            this->CopyUB2UB(yTensor[yOffsetInUB], xTensor_[gxIdArray[idxLoop] * this->aAlignSize_]);
            yOffsetInUB += this->aAlignSize_;
        }
    }

    CopyOutYData(yOffset, yTensor, lineNum, this->aSize_);
    this->FreeTensorIdx(idxTensor);
}

// key      x       y       core
//  20131   b       g       b_out
template <typename T_DATA, typename T_IDX>
__aicore__ inline void GatherV3Tmpl1<T_DATA, T_IDX> :: ProcessSingleTile20132() {
    this->CalcBaseOffset();

    int64_t yOffset = this->yBaseOffset_;
    int64_t idxOffset = this->idxBaseOffset_;

    if (!xLoaded_) {
        xTensor_ = CopyInXData(0, 1, this->gxSize_ * this->aSize_);
        xLoaded_ = true;
    }
    // 搬入b轴切分范围所有pga
    int64_t lineNum = this->gyLength_;

    auto idxTensor = this->CopyInIdx(idxOffset, 1, lineNum);
    auto yTensor = GetTensorY();
    int64_t yOffsetInUB = 0;

    int64_t loopNum = lineNum / UNROLL_NUM;
    int64_t tailLen = lineNum % UNROLL_NUM;
    T_IDX gxIdArray[UNROLL_NUM];
    int64_t idxPos = 0;

    this->SyncM2toS();
    for (int64_t loop = 0; loop < loopNum; loop++) {
        for (int64_t idxLoop = 0; idxLoop < UNROLL_NUM; idxLoop++) {
            gxIdArray[idxLoop] = idxTensor.GetValue(idxPos++);
            this->CheckIdxValue(gxIdArray[idxLoop]);
        }

        for (int64_t idxLoop = 0; idxLoop < UNROLL_NUM; idxLoop++) {
            T_IDX gxId = gxIdArray[idxLoop];
            int64_t xOffsetInUB = gxId * this->aSize_;

            for (int64_t valPos = 0; valPos < this->aSize_; valPos++) {
                yTensor.SetValue(yOffsetInUB + valPos, xTensor_.GetValue(xOffsetInUB + valPos));
            }
            yOffsetInUB += this->aSize_;
        }
    }

    for (int64_t i = 0; i < tailLen; i++) {
        T_IDX gxId = idxTensor.GetValue(idxPos++);
        this->CheckIdxValue(gxId);
        int64_t xOffsetInUB = gxId * this->aSize_;
        
        for (int64_t valPos = 0; valPos < this->aSize_; valPos++) {
            yTensor.SetValue(yOffsetInUB + valPos, xTensor_.GetValue(xOffsetInUB + valPos));
        }
        yOffsetInUB += this->aSize_;
    }

    CopyOutYData(yOffset, yTensor, 1, lineNum * this->aSize_);
    this->FreeTensorIdx(idxTensor);
}

// key      x       y       core
// 20120    b       p       b_out
template <typename T_DATA, typename T_IDX>
__aicore__ inline void GatherV3Tmpl1<T_DATA, T_IDX> :: ProcessSingleTile20120() {
    this->CalcBaseOffset();

    int64_t xOffset = this->xBaseOffset_;
    int64_t yOffset = this->yBaseOffset_;
    int64_t idxOffset = this->idxBaseOffset_;

    // 搬入b轴切分范围所有pga
    auto xTensor = CopyInXData(xOffset, this->bLength_ *  this->pSize_ * this->gxSize_, this->aSize_);
    int64_t xOffsetInUB = 0;

    // 核内以p切分做循环 每次循环搬出
    for (int64_t bLoop = 0; bLoop < this->bLength_; bLoop++) {
        auto idxTensor = this->CopyInIdx(idxOffset, 1, this->gySize_);
        idxOffset += this->gySize_;

        this->SyncM2toS();
        for (int64_t pLoop = 0; pLoop <  this->pSize_; pLoop += this->ubLineLimit_) {
            int64_t lineNum = this->Min(this->ubLineLimit_,  this->pSize_ - pLoop);

            auto yTensor = GetTensorY();
            int64_t yOffsetInUB = 0;

            for (int64_t gLoop = 0; gLoop < this->gySize_; gLoop++) {
                T_IDX gxId = idxTensor.GetValue(gLoop);
                this->CheckIdxValue(gxId);

                this->CopyUB2UBRepeat(yTensor[yOffsetInUB], xTensor[xOffsetInUB + gxId * this->aAlignSize_], lineNum);
                yOffsetInUB += this->aAlignSize_;
            }
            xOffsetInUB += lineNum * this->gxSize_ * this->aAlignSize_;

            // 集中搬出
            CopyOutYData(yOffset, yTensor, lineNum * this->gySize_, this->aSize_);
            yOffset += lineNum * this->gySize_ * this->aSize_;
        }

        this->FreeTensorIdx(idxTensor);
    }

    // 释放资源
    FreeTensorX(xTensor);
}

// key      x       y       core
// 20110    b       b       b_out
template <typename T_DATA, typename T_IDX>
__aicore__ inline void GatherV3Tmpl1<T_DATA, T_IDX> :: ProcessSingleTile20110() {
    this->CalcBaseOffset();
    
    int64_t xOffset = this->xBaseOffset_;
    int64_t yOffset = this->yBaseOffset_;
    int64_t idxOffset = this->idxBaseOffset_;

    // 以输出b轴切分为准

    auto idxTensor = this->CopyInIdx(idxOffset, 1, this->bLength_ * this->gySize_);
    
    int64_t idxOffsetInUB = 0;

    auto yTensor = GetTensorY();
    int64_t yOffsetInUB = 0;

    this->SyncM2toS();
    for (int64_t bLoop = 0; bLoop < this->bLength_; bLoop += this->ubLineLimit_) {
        int64_t lineNum = this->Min(this->ubLineLimit_,  this->bLength_ - bLoop);

        // 搬入b轴切分范围所有pga
        auto xTensor =  CopyInXData(xOffset, lineNum *  this->pSize_ * this->gxSize_, this->aSize_);
        xOffset += lineNum *  this->pSize_ * this->gxSize_ * this->aSize_;

        int64_t xOffsetInUB = 0;
        int64_t yOffsetStep = this->pSize_ * this->gySize_ * this->aAlignSize_;
        int64_t xOffsetStep = this->pSize_ * this->gxSize_ * this->aAlignSize_;

        for (int i = 0; i < lineNum; i++) {
            yOffsetInUB = (bLoop + i) * yOffsetStep;

            idxOffsetInUB = (bLoop + i) * this->gySize_;
            for (int64_t gLoop = 0; gLoop < this->gySize_; gLoop++) {
                T_IDX gxId = idxTensor.GetValue(idxOffsetInUB++);
                this->CheckIdxValue(gxId);

                this->CopyUB2UBRepeat(yTensor[yOffsetInUB], xTensor[xOffsetInUB + gxId * this->aAlignSize_], this->pSize_);
                yOffsetInUB += this->aAlignSize_;
            }
            xOffsetInUB += xOffsetStep;
        }

        FreeTensorX(xTensor);
    }

    // 集中搬出
    CopyOutYData(yOffset, yTensor, this->bLength_ * this->pSize_ * this->gySize_, this->aSize_);

    this->FreeTensorIdx(idxTensor);
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl1<T_DATA, T_IDX>::ProcessSingleTile30330() {
    ProcessSingleTile20330();
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl1<T_DATA, T_IDX>::ProcessSingleTile30320() {
    ProcessSingleTile20320();
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl1<T_DATA, T_IDX>::ProcessSingleTile30310() {
    ProcessSingleTile20310();
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl1<T_DATA, T_IDX>::ProcessSingleTile30230() {
    ProcessSingleTile20230();
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl1<T_DATA, T_IDX>::ProcessSingleTile30220() {
    ProcessSingleTile20220();
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl1<T_DATA, T_IDX>::ProcessSingleTile30210() {
    ProcessSingleTile20210();
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl1<T_DATA, T_IDX>::ProcessSingleTile30130() {
    ProcessSingleTile20130();
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl1<T_DATA, T_IDX>::ProcessSingleTile30131() {
    ProcessSingleTile20131();
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl1<T_DATA, T_IDX>::ProcessSingleTile30120() {
    ProcessSingleTile20120();
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl1<T_DATA, T_IDX>::ProcessSingleTile30110() {
    ProcessSingleTile20110();
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl1<T_DATA, T_IDX>::ProcessSingleTile40000() {
    this->CalcBaseOffset();

    int64_t xOffset = this->xBaseOffset_;
    int64_t yOffset = this->yBaseOffset_;
    int64_t idxOffset = this->idxBaseOffset_;

    auto idxTensor = this->CopyInIdx(idxOffset, 1, this->bLength_ * this->gySize_);
    int64_t idxOffsetInUB = 0;

    auto yTensor = GetTensorY();
    int64_t yOffsetInUB = 0;

    this->SyncM2toS();
    for (int64_t bLoop = 0; bLoop < this->bLength_; bLoop++) {
        for (int64_t pLoop = 0; pLoop < this->pLength_; pLoop++) {
            for (int64_t gxLoop = 0; gxLoop < this->gxSize_; gxLoop += this->ubLineLimit_) {
                int64_t lineNum = this->Min(this->ubLineLimit_, this->gxSize_ - gxLoop);
                
                auto xTensor = CopyInXData(xOffset, lineNum, this->aSize_);
                xOffset += lineNum * this->aSize_;

                for (int64_t gyLoop = 0; gyLoop < this->gySize_; gyLoop++) {
                    int64_t gxId = idxTensor.GetValue(idxOffsetInUB + gyLoop);
                    this->CheckIdxValue(gxId);
                    if (gxId < gxLoop || gxId >= gxLoop + lineNum) {
                        continue;
                    }

                    this->CopyUB2UB(yTensor[yOffsetInUB + gyLoop * this->aAlignSize_], 
                                    xTensor[(gxId - gxLoop) * this->aAlignSize_]);
                }
                FreeTensorX(xTensor);
            }
            yOffsetInUB += this->gySize_ * this->aAlignSize_;
        }
        idxOffsetInUB += this->gySize_;
    }

    CopyOutYData(yOffset, yTensor, this->bLength_ * this->pLength_ * this->gySize_, this->aSize_);

    this->FreeTensorIdx(idxTensor);
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl1<T_DATA, T_IDX>::PreProcess40331() {
    auto posTensor = this->posBuf_.template Get<int32_t>();
    CreateVecIndex(posTensor, 0, (uint32_t)(this->gyLength_));

    PrefetchXData(0, this->Min(this->ubLineLimit_, this->gxSize_), this->aSize_);
    this->PrefetchIdx(this->idxBaseOffset_, 1, this->gyLength_);
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl1<T_DATA, T_IDX>::ProcessSingleTile40331() {
    this->CalcBaseOffset();

    PreProcess40331();

    auto idxReduceTensor_ = this->idxReduceBuf_.template Get<int32_t>();
    auto posReduceTensor_ = this->posReduceBuf_.template Get<int32_t>();

    auto yTensor = GetTensorY();    
    auto idxTensor = this->GetIdx();
    auto idxNormTensor = TypeNormIdx(idxTensor);

    this->SyncVtoS();
    this->SyncM2toS();

    for (int64_t gxLoop = 0; gxLoop < this->gxSize_; gxLoop += this->ubLineLimit_) {
        int64_t lineNum = this->Min(this->ubLineLimit_, this->gxSize_ - gxLoop);
        
        auto xTensor = GetCurXData(gxLoop, lineNum, this->gxSize_, this->aSize_);

        int64_t idxNum = CalcIdxInRange(idxNormTensor, gxLoop, gxLoop + lineNum);
        this->SyncVtoS();

        int64_t loopNum = idxNum / UNROLL_NUM;
        int64_t tailLen = idxNum % UNROLL_NUM;

        int32_t gxIdArray[UNROLL_NUM];
        int32_t gyIdArray[UNROLL_NUM];

        PreCopy();

        int64_t idxPos = 0;
        for (int64_t loop = 0; loop < loopNum; loop++) {
            for (int32_t idxLoop = 0; idxLoop < UNROLL_NUM; idxLoop++) {
                gxIdArray[idxLoop] = idxReduceTensor_.GetValue(idxPos);
                gyIdArray[idxLoop] = posReduceTensor_.GetValue(idxPos);
                idxPos++;
            }

            this->SyncStoV();

            for (int32_t idxLoop = 0; idxLoop < UNROLL_NUM; idxLoop++) {
                CopyUB2UB(yTensor[gyIdArray[idxLoop]], xTensor[gxIdArray[idxLoop]]);
            }
        }

        if (tailLen > 0) {
            for (int32_t idxLoop = 0; idxLoop < tailLen; idxLoop++) {
                gxIdArray[idxLoop] = idxReduceTensor_.GetValue(idxPos);
                gyIdArray[idxLoop] = posReduceTensor_.GetValue(idxPos);
                idxPos++;
            }

            this->SyncStoV();

            for (int32_t idxLoop = 0; idxLoop < tailLen; idxLoop++) {
                CopyUB2UB(yTensor[gyIdArray[idxLoop]], xTensor[gxIdArray[idxLoop]]);
            }
        }

        PostCopy();

        FreeTensorX(xTensor);
    }

    CopyOutYData(this->yBaseOffset_, yTensor, this->gyLength_, this->aSize_);
    this->FreeTensorIdx(idxTensor);
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl1<T_DATA, T_IDX>::ProcessSingleTile40330() {
    ProcessSingleTile40000();
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl1<T_DATA, T_IDX>::ProcessSingleTile40320() {
    ProcessSingleTile40000();
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl1<T_DATA, T_IDX>::ProcessSingleTile40310() {
    ProcessSingleTile40000();
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl1<T_DATA, T_IDX>::ProcessSingleTile50330() {
    ProcessSingleTile40000();
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl1<T_DATA, T_IDX>::ProcessSingleTile50320() {
    ProcessSingleTile40000();
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl1<T_DATA, T_IDX>::ProcessSingleTile50310() {
    ProcessSingleTile40000();
}

}  // namespace GatherV3

#endif  // GATHER_V3_TMPL_1_H