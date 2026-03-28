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
 * \file conv_bp_sub_func.h
 * \brief
 */

#ifndef CONV2D_BP_FILTER_SUB_FUNC_H
#define CONV2D_BP_FILTER_SUB_FUNC_H

namespace ConvolutionBackpropFunc {

template <class Intf>
static __aicore__ inline void CalcParamsL12L0a(Intf *self, uint16_t kPos)
{
    self->ctx.dstL12L0aOffset_ = 0;
    uint32_t kOffset = (kPos % self->ctx.tiling_->stepKa) * self->ctx.tiling_->baseK / self->ctx.tiling_->k0;
    self->ctx.srcL12L0aOffset_ = (kOffset)*16 * 16;
    self->ctx.load2d_.startIndex = 0;
    self->ctx.load2d_.repeatTimes = Ceil(self->ctx.baseUseK_, self->ctx.tiling_->k0);
    self->ctx.load2d_.srcStride = 1;
    self->ctx.load2d_.dstGap = 0;
    self->ctx.load2d_.ifTranspose = 1;
}

template <class Intf>
static __aicore__ inline void CalcParamsMmad(Intf *self, uint16_t kPos)
{
    self->ctx.dstL0cOffset_ = 0;
    self->ctx.srcL0aOffset_ = 0;
    self->ctx.srcL0bOffset_ = 0;
    self->ctx.mmad_.m = self->ctx.baseUseM_;
    self->ctx.mmad_.k = self->ctx.baseUseK_;
    self->ctx.mmad_.n = self->ctx.baseUseN_;
    self->ctx.mmad_.unitFlag = 0;
    self->ctx.mmad_.kDirectionAlign = 0;
    self->ctx.mmad_.cmatrixSource = 0;
    self->ctx.mmad_.cmatrixInitVal = kPos == 0;
}

template <class Intf>
static __aicore__ inline void LoadL12L0a(Intf *self, const LocalTensor<typename Intf::SrcT> &l1AMatrix, uint16_t kPos)
{
    self->ctx.a2_ = self->ctx.qidA2_.template AllocTensor<typename Intf::SrcT>();
    // Todo
    for (uint32_t m_iter = 0; m_iter < (self->ctx.tiling_->baseM / self->ctx.tiling_->m0); m_iter++) {
        uint64_t curStepKa = Ceil(static_cast<uint64_t>(self->ctx.singleShapeHo_) * self->ctx.tiling_->Wo, self->ctx.tiling_->baseK);
        uint64_t srcL12L0aOffset_ = m_iter * curStepKa * self->ctx.tiling_->baseK * 16 + self->ctx.srcL12L0aOffset_;
        uint64_t dstL12L0aOffset_ = static_cast<uint64_t>(m_iter) * self->ctx.tiling_->baseK * 16;
        LoadData(self->ctx.a2_[dstL12L0aOffset_], l1AMatrix[srcL12L0aOffset_], self->ctx.load2d_);
    }
    self->ctx.qidA2_.EnQue(self->ctx.a2_);
    self->ctx.qidA2_.template DeQue<typename Intf::SrcT>();
}

template <class Intf>
static __aicore__ inline void MmadLocal(Intf *self)
{
    MmadImpl(self->ctx.c1_[self->ctx.dstL0cOffset_],
        self->ctx.a2_[self->ctx.srcL0aOffset_],
        self->ctx.b2_[self->ctx.srcL0bOffset_],
        self->ctx.mmad_);
    self->ctx.qidA2_.FreeTensor(self->ctx.a2_);
    self->ctx.qidB2_.FreeTensor(self->ctx.b2_);
}

template <class Intf, class src0_T>
__aicore__ inline LocalTensor<typename Intf::SrcT> LoadToA1(Intf *self, bool cachePosA1, uint32_t kaIdx)
{
    LocalTensor<typename Intf::SrcT> useA1Buf;
    bool bufUsingFlag;
    if (self->ctx.tiling_->al1Pbuffer != 2) {
        useA1Buf = self->ctx.cacheA1BufPing_;
        bufUsingFlag = self->ctx.usingCacheA1Ping_;
        cachePosA1 = true;
    } else {
        useA1Buf = cachePosA1 ? self->ctx.cacheA1BufPing_ : self->ctx.cacheA1BufPong_;
        bufUsingFlag = cachePosA1 ? self->ctx.usingCacheA1Ping_ : self->ctx.usingCacheA1Pong_;
    }
    LocalTensor<src0_T> a1;
    if ((self->ctx.curML0Idx_ % self->ctx.tiling_->stepM == 0 && kaIdx % self->ctx.tiling_->stepKa == 0) &&
        ((self->ctx.tiling_->al1Pbuffer != 2) || (Ceil(self->ctx.kIter_, self->ctx.tiling_->stepKa) > 2) ||
            (likely(self->ctx.tiling_->iterateOrder)) ||
            (!likely(self->ctx.tiling_->iterateOrder) && Ceil(self->ctx.kIter_, self->ctx.tiling_->stepKa) <= 2 &&
                self->ctx.curNL1Idx_ / self->ctx.tiling_->stepN == 0))) {
        if (bufUsingFlag) {
            // 载入L1数据时，若所占buff已被使用过，需要释放
            self->ctx.qidA1_.FreeTensor(useA1Buf);
        }

        uint64_t out2A1SrcAddrOffset =
            static_cast<uint64_t>(self->ctx.curML1Idx_) * self->ctx.tiling_->baseM * self->ctx.tiling_->Ho * self->ctx.tiling_->Wo +
            static_cast<uint64_t>(kaIdx) / self->ctx.tiling_->stepKa * self->ctx.tiling_->stepKa * self->ctx.tiling_->baseK;
        a1 = self->ctx.qidA1_.template AllocTensor<src0_T>();

        DataCopyParams dataCopyParams;
        uint64_t srcStride =
            static_cast<uint64_t>(self->ctx.tiling_->Ho) * self->ctx.tiling_->Wo - static_cast<uint64_t>(self->ctx.singleShapeHo_) * self->ctx.tiling_->Wo;
        dataCopyParams.srcStride = static_cast<uint16_t>(srcStride);
        dataCopyParams.dstStride = 0;
        if ((self->ctx.kIter_ - 1) / self->ctx.tiling_->stepKa == kaIdx / self->ctx.tiling_->stepKa) {
            // 最后一块kAL1，考虑tailK, 32表示32Byte
            uint64_t blockLen =
                static_cast<uint64_t>(self->ctx.singleShapeHo_) * self->ctx.tiling_->Wo - static_cast<uint64_t>(kaIdx) * self->ctx.tiling_->baseK;
            dataCopyParams.blockLen = static_cast<uint16_t>(blockLen);
            uint64_t dstStride =
                Ceil(static_cast<uint64_t>(self->ctx.singleShapeHo_) * self->ctx.tiling_->Wo - static_cast<uint64_t>(kaIdx) * self->ctx.tiling_->baseK,
                     32 / sizeof(src0_T)) *
                32 / sizeof(src0_T) - static_cast<uint64_t>(self->ctx.singleShapeHo_) * self->ctx.tiling_->Wo;
            dataCopyParams.dstStride = static_cast<uint16_t>(dstStride);
        } else {
            dataCopyParams.blockLen = static_cast<uint16_t>(self->ctx.tiling_->stepKa * self->ctx.tiling_->baseK);
        }
        if ((self->ctx.mIter_ - 1) / self->ctx.tiling_->stepM == self->ctx.curML0Idx_ / self->ctx.tiling_->stepM) {
            // 最后一块mAL1，需要考虑tailM, 16 是一个Block 存放F16的元素个数
            dataCopyParams.blockCount = ((self->ctx.curStepM_ - 1) * self->ctx.tiling_->baseM + self->ctx.tailM_) / 16;
        } else {
            dataCopyParams.blockCount = self->ctx.curStepM_ * self->ctx.tiling_->baseM / 16;
        }
        DataCopy(a1, self->ctx.outBackPropGlobal_[out2A1SrcAddrOffset], dataCopyParams);

        self->ctx.qidA1_.EnQue(a1);
        self->ctx.qidA1_.DeQue();
        if (cachePosA1) {
            self->ctx.cacheA1BufPing_ = a1;
            self->ctx.usingCacheA1Ping_ = true;
        } else {
            self->ctx.cacheA1BufPong_ = a1;
            self->ctx.usingCacheA1Pong_ = true;
        }
        return a1;
    }
    a1 = useA1Buf;
    return a1;
}

template <class Intf, class src1_T>
__aicore__ inline LocalTensor<typename Intf::SrcT> LoadToB1(Intf *self, bool cachePosB1, uint16_t kbIdx)
{
    LocalTensor<typename Intf::SrcT> useB1Buf;
    bool bufUsingFlag;
    if (self->ctx.tiling_->bl1Pbuffer != 2) {
        useB1Buf = self->ctx.cacheB1BufPing_;
        bufUsingFlag = self->ctx.usingCacheB1Ping_;
        cachePosB1 = true;
    } else {
        useB1Buf = cachePosB1 ? self->ctx.cacheB1BufPing_ : self->ctx.cacheB1BufPong_;
        bufUsingFlag = cachePosB1 ? self->ctx.usingCacheB1Ping_ : self->ctx.usingCacheB1Pong_;
    }
    LocalTensor<src1_T> b1;
    // 需要载入BL1的条件为，被计算的BL0块是BL1上的第一块数据，一次载入完整BL1大小
    // 此时满足以下条件之一需要载入BL1：
    // 1.BL1上无db，每次都需要载入
    // 2.singleShapeK / stepKb > 2, 优先循环k方向，BL1上数据无法复用
    // 3.order_M时，L1上驻留AL1, BL1数据不复用
    // 4.order_N时，BL1驻留在L1上，且K <=
    // 2，即L1上可以栽下全部Kb，此时遍历M方向，BL1数据上数据不会被覆盖，只在M方向循环第一次时载入BL1
    if ((self->ctx.curNL0Idx_ % self->ctx.tiling_->stepN == 0 && kbIdx % self->ctx.tiling_->stepKb == 0) &&
        ((self->ctx.tiling_->bl1Pbuffer != 2) || (Ceil(self->ctx.kIter_, self->ctx.tiling_->stepKb) > 2) ||
            (!likely(self->ctx.tiling_->iterateOrder)) ||
            (likely(self->ctx.tiling_->iterateOrder) && Ceil(self->ctx.kIter_, self->ctx.tiling_->stepKb) <= 2 &&
                self->ctx.curML1Idx_ / self->ctx.tiling_->stepM == 0))) {

        if (bufUsingFlag) {
            // 载入L1数据时，若所占buff已被使用过，需要释放
            self->ctx.qidB1_.FreeTensor(useB1Buf);
        }

        // L0shape到orgShape的对应关系，L0和L1是16对齐的，orgShape是Wi对齐的,先算Wo对齐再算Wi对齐
        // 先算L0B所在BL1块的起始地址，16对齐的
        uint32_t b1SrcKAlign = kbIdx / self->ctx.tiling_->stepKb * self->ctx.tiling_->stepKb * self->ctx.tiling_->baseK;
        // load3d必须有完整Wo，做Wo对齐，计算起始地址所在的Ho
        uint32_t b1SrcHo = b1SrcKAlign / self->ctx.tiling_->Wo;
        // 计算Ho对应的Hi，根据卷积原理
        uint32_t b1SrcHi = b1SrcHo * self->ctx.tiling_->strideH - self->ctx.tiling_->padT;
        // 计算L0上cin, 去掉cin1HkWkCin里的HkWk
        uint32_t cin1L0 =
            self->ctx.tiling_->baseN / self->ctx.tiling_->Hk / self->ctx.tiling_->Wk / self->ctx.tiling_->channelSize;
        uint32_t b1SrCin = self->ctx.curNL1Idx_ / self->ctx.tiling_->stepN * cin1L0 * 16;
        // 得到gm的偏移量
        uint64_t out2B1SrcAddrOffset =
            (static_cast<uint64_t>(b1SrCin) * self->ctx.tiling_->Hi * self->ctx.tiling_->Wi + 
            static_cast<uint64_t>(b1SrcHi) * self->ctx.tiling_->Wi) * 16;

        LocalTensor<src1_T> b1 = self->ctx.qidB1_.template AllocTensor<src1_T>();
        DataCopyParams dataCopyParams;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        dataCopyParams.blockCount = 1;

        uint32_t kValue = 0;
        uint32_t nValue = 16;

        if ((self->ctx.kIter_ - 1) / self->ctx.tiling_->stepKb == kbIdx / self->ctx.tiling_->stepKb) {
            // 最后一块kBL1，需要考虑tailHoWo
            kValue = (self->ctx.singleShapeHo_ - kbIdx * self->ctx.tiling_->baseK / self->ctx.tiling_->Wo) *
                     self->ctx.tiling_->strideH * self->ctx.tiling_->Wi;
        } else {
            uint32_t b1KAlignEnd = b1SrcKAlign + self->ctx.tiling_->stepKb * self->ctx.tiling_->baseK;
            uint32_t b1HoEnd = b1KAlignEnd / self->ctx.tiling_->Wo;
            uint32_t b1HiEnd = b1HoEnd * self->ctx.tiling_->strideH - self->ctx.tiling_->padT + self->ctx.tiling_->Hk;
            kValue = (b1HiEnd - b1SrcHi) * self->ctx.tiling_->Wi;
        }

        uint64_t blockLen = static_cast<uint64_t>(kValue) * nValue / (32 / sizeof(src1_T));
        dataCopyParams.blockLen = static_cast<uint16_t>(blockLen);

        copy_gm_to_cbuf(((__cbuf__ void *)b1.GetPhyAddr()),
            (__gm__ void *)(self->ctx.fmapGlobal_.GetPhyAddr() + out2B1SrcAddrOffset),
            (int8_t)0,
            (uint16_t)dataCopyParams.blockCount,
            (uint16_t)dataCopyParams.blockLen,
            (uint16_t)dataCopyParams.srcStride,
            (uint16_t)dataCopyParams.dstStride,
            PAD_MODE7);
        self->ctx.qidB1_.EnQue(b1);
        self->ctx.qidB1_.DeQue();
        if (cachePosB1) {
            self->ctx.cacheB1BufPing_ = b1;
            self->ctx.usingCacheB1Ping_ = true;
        } else {
            self->ctx.cacheB1BufPong_ = b1;
            self->ctx.usingCacheB1Pong_ = true;
        }
        return b1;
    }
    b1 = useB1Buf;
    return b1;
}

template <class Intf>
static __aicore__ inline void LoadL0c2Gm(
    Intf *self, const GlobalTensor<typename Intf::DstT> &output, uint8_t enAtomic = 0, bool enSequentialWrite = false)
{
    if (enAtomic == 1) {
        SetAtomicAdd<typename Intf::DstT>();
    }
    if constexpr (Intf::Config::dType::format == ConvolutionBackprop::CubeFormat::NC1HWC0 ||
                  Intf::Config::dType::format == ConvolutionBackprop::CubeFormat::FRACTALZ_C04) {
        if (!enSequentialWrite) {
            uint64_t dstOffset =
                static_cast<uint64_t>(self->ctx.curNL0Idx_) % self->ctx.tiling_->stepN * self->ctx.tiling_->baseN * self->ctx.tiling_->Cout +
                static_cast<uint64_t>(self->ctx.curML0Idx_) % self->ctx.mIter_ * self->ctx.tiling_->baseM * 16;
            FixpipeParams<typename Intf::L0cT> fixpipeParams(static_cast<uint16_t>(Ceil(self->ctx.baseUseN_, 16)),
                static_cast<uint16_t>(self->ctx.baseUseM_ * BLOCK_CUBE * sizeof(typename Intf::L0cT) / 32),
                0,
                (self->ctx.tiling_->Cout - self->ctx.baseUseM_) * 16 * sizeof(typename Intf::DstT) / 32);
            Fixpipe(output[dstOffset], self->ctx.c1_, fixpipeParams);
        } else {
            return;
        }
    }
    if (enAtomic == 1) {
        SetAtomicNone();
    }
    self->ctx.qidCO1_.FreeTensor(self->ctx.c1_);
}
}  // namespace ConvolutionBackpropFunc

#endif