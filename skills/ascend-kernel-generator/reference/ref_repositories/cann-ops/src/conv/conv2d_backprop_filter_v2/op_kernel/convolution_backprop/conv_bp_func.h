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
 * \file conv_bp_func.h
 * \brief
 */

#ifndef CONV_BP_FUNC_H
#define CONV_BP_FUNC_H

#include "conv_bp_util.h"
#include "kernel_operator.h"

#if __CCE_AICORE__ == 220
#include "impl/dav_v220/conv_bp_sub_func.h"
#endif

DECLARE_CHECK_IMPL(Init);

DECLARE_CHECK_IMPL(SetFmap);
DECLARE_CHECK_IMPL(SetOutBackprop);
DECLARE_CHECK_IMPL(SetSingleShape);
DECLARE_CHECK_SYNC_IMPL(Iterate);
DECLARE_CHECK_SYNC_IMPL(IterateAll);
DECLARE_CHECK_SYNC_IMPL(GetTensorC);
DECLARE_CHECK_IMPL(End);
namespace ConvolutionBackpropFunc {

using TypeFalse = struct {
    __uint128_t _[1024];
};

enum class IterateOrder {
    ORDER_M = 0,
    ORDER_N,
    UNDEF,
};

template <class Intf>
__aicore__ inline void CheckTiling(Intf *self)
{
#ifdef __CCE_KT_TEST__
    ASCENDC_ASSERT((self->ctx.tiling_->N > 0),
        { KERNEL_LOG(KERNEL_ERROR, "orignal N is %d , which should be larger than 0", self->ctx.tiling_->N); });
    ASCENDC_ASSERT((self->ctx.tiling_->Cin > 0),
        { KERNEL_LOG(KERNEL_ERROR, "orignal Cin is %d , which should be larger than 0", self->ctx.tiling_->Cin); });
    ASCENDC_ASSERT((self->ctx.tiling_->Cout > 0),
        { KERNEL_LOG(KERNEL_ERROR, "orignal Cout is %d , which should be larger than 0", self->ctx.tiling_->Cout); });
    ASCENDC_ASSERT((self->ctx.tiling_->Ho > 0),
        { KERNEL_LOG(KERNEL_ERROR, "orignal Ho is %d , which should be larger than 0", self->ctx.tiling_->Ho); });
    ASCENDC_ASSERT((self->ctx.tiling_->Wo > 0),
        { KERNEL_LOG(KERNEL_ERROR, "orignal Wo is %d , which should be larger than 0", self->ctx.tiling_->Wo); });
    ASCENDC_ASSERT((self->ctx.tiling_->Hi > 0),
        { KERNEL_LOG(KERNEL_ERROR, "orignal Hi is %d , which should be larger than 0", self->ctx.tiling_->Hi); });
    ASCENDC_ASSERT((self->ctx.tiling_->Wi > 0),
        { KERNEL_LOG(KERNEL_ERROR, "orignal Wi is %d , which should be larger than 0", self->ctx.tiling_->Wi); });
    ASCENDC_ASSERT((self->ctx.tiling_->Hk > 0),
        { KERNEL_LOG(KERNEL_ERROR, "orignal Hk is %d , which should be larger than 0", self->ctx.tiling_->Hk); });
    ASCENDC_ASSERT((self->ctx.tiling_->Wk > 0),
        { KERNEL_LOG(KERNEL_ERROR, "orignal Wk is %d , which should be larger than 0", self->ctx.tiling_->Wk); });
    ASCENDC_ASSERT((self->ctx.tiling_->singleCoreBatch > 0), {
        KERNEL_LOG(
            KERNEL_ERROR, "singleCoreBatch is %d , which should be larger than 0", self->ctx.tiling_->singleCoreBatch);
    });
    ASCENDC_ASSERT((self->ctx.tiling_->singleCoreCout > 0), {
        KERNEL_LOG(
            KERNEL_ERROR, "singleCoreCout is %d , which should be larger than 0", self->ctx.tiling_->singleCoreCout);
    });
    ASCENDC_ASSERT((self->ctx.tiling_->singleCoreHo > 0), {
        KERNEL_LOG(KERNEL_ERROR, "singleCoreHo is %d , which should be larger than 0", self->ctx.tiling_->singleCoreHo);
    });
    ASCENDC_ASSERT((self->ctx.tiling_->singleCoreCin > 0), {
        KERNEL_LOG(
            KERNEL_ERROR, "singleCoreCin is %d , which should be larger than 0", self->ctx.tiling_->singleCoreCin);
    });
    ASCENDC_ASSERT((self->ctx.tiling_->baseM > 0),
        { KERNEL_LOG(KERNEL_ERROR, "baseM is %d , which should be larger than 0", self->ctx.tiling_->baseM); });
    ASCENDC_ASSERT((self->ctx.tiling_->baseK > 0),
        { KERNEL_LOG(KERNEL_ERROR, "baseK is %d , which should be larger than 0", self->ctx.tiling_->baseK); });
    ASCENDC_ASSERT((self->ctx.tiling_->baseN > 0),
        { KERNEL_LOG(KERNEL_ERROR, "baseN is %d , which should be larger than 0", self->ctx.tiling_->baseN); });
    ASCENDC_ASSERT((self->ctx.tiling_->stepM > 0),
        { KERNEL_LOG(KERNEL_ERROR, "stepM is %d , which should be larger than 0", self->ctx.tiling_->stepM); });
    ASCENDC_ASSERT((self->ctx.tiling_->stepN > 0),
        { KERNEL_LOG(KERNEL_ERROR, "stepN is %d , which should be larger than 0", self->ctx.tiling_->stepN); });
    ASCENDC_ASSERT((self->ctx.tiling_->stepKa > 0),
        { KERNEL_LOG(KERNEL_ERROR, "stepKa is %d , which should be larger than 0", self->ctx.tiling_->stepKa); });
    ASCENDC_ASSERT((self->ctx.tiling_->stepKb > 0),
        { KERNEL_LOG(KERNEL_ERROR, "stepKb is %d , which should be larger than 0", self->ctx.tiling_->stepKb); });
#endif
}

template <class Intf>
__aicore__ inline void InitStepMParams(Intf *self)
{
    self->ctx.mIter_ = Ceil(self->ctx.singleShapeCout_, self->ctx.tiling_->baseM);
    self->ctx.tailM_ = self->ctx.singleShapeCout_ % self->ctx.tiling_->baseM;
    if (self->ctx.tailM_ == 0) {
        self->ctx.tailM_ = self->ctx.tiling_->baseM;
    }
#ifdef __CCE_KT_TEST__
    ASCENDC_ASSERT((self->ctx.mIter_ > 0),
        { KERNEL_LOG(KERNEL_ERROR, "self->ctx.mIter_ is %d , which should be larger than 0", self->ctx.mIter_); });
#endif
}

template <class Intf>
__aicore__ inline void InitStepKParams(Intf *self)
{
    self->ctx.kIter_ = Ceil(static_cast<uint64_t>(self->ctx.singleShapeHo_) * self->ctx.tiling_->Wo, self->ctx.tiling_->baseK);
    self->ctx.tailK_ = self->ctx.singleShapeHo_ * self->ctx.tiling_->Wo % self->ctx.tiling_->baseK;
    if (self->ctx.tailK_ == 0) {
        self->ctx.tailK_ = self->ctx.tiling_->baseK;
    }
#ifdef __CCE_KT_TEST__
    ASCENDC_ASSERT((self->ctx.kIter_ > 0),
        { KERNEL_LOG(KERNEL_ERROR, "self->ctx.kIter_ is %d , which should be larger than 0", self->ctx.kIter_); });
#endif
}

template <class Intf>
__aicore__ inline void InitStepNParams(Intf *self)
{
    self->ctx.nIter_ = Ceil(Ceil(self->ctx.singleShapeCin_, self->ctx.tiling_->channelSize) *
                                self->ctx.tiling_->channelSize * self->ctx.tiling_->Hk * self->ctx.tiling_->Wk,
        self->ctx.tiling_->baseN);
    self->ctx.tailN_ =
        self->ctx.singleShapeCin_ * self->ctx.tiling_->Hk * self->ctx.tiling_->Wk % self->ctx.tiling_->baseN;
    if (self->ctx.tailN_ == 0) {
        self->ctx.tailN_ = self->ctx.tiling_->baseN;
    }
#ifdef __CCE_KT_TEST__
    ASCENDC_ASSERT((self->ctx.nIter_ > 0),
        { KERNEL_LOG(KERNEL_ERROR, "self->ctx.nIter_ is %d , which should be larger than 0", self->ctx.nIter_); });
#endif
}

template <class Intf>
__aicore__ inline void InitParams(Intf *self)
{
    self->ctx.baseMK_ = self->ctx.tiling_->baseM * self->ctx.tiling_->baseK;
    self->ctx.baseKN_ = self->ctx.tiling_->baseK * self->ctx.tiling_->baseN;
    self->ctx.baseMN_ = self->ctx.tiling_->baseM * self->ctx.tiling_->baseN;
    self->ctx.isFirstIter_ = true;
    self->ctx.usingCacheA1Ping_ = false;
    self->ctx.usingCacheA1Pong_ = false;
    self->ctx.usingCacheB1Ping_ = false;
    self->ctx.usingCacheB1Pong_ = false;
}

template <class Intf>
__aicore__ inline uint32_t CalHi(Intf *self)
{
    uint32_t hoCal;
    if (self->ctx.tiling_->baseK * self->ctx.tiling_->stepKb % self->ctx.tiling_->Wo == 0) {
        hoCal = self->ctx.tiling_->baseK * self->ctx.tiling_->stepKb / self->ctx.tiling_->Wo;
    } else if (self->ctx.tiling_->baseK * self->ctx.tiling_->stepKb > self->ctx.tiling_->Wo) {
        hoCal = self->ctx.tiling_->baseK * self->ctx.tiling_->stepKb / self->ctx.tiling_->Wo + 2;
    } else {
        hoCal = 2;
    }
    uint32_t khDilation = (self->ctx.tiling_->Hk - 1) * self->ctx.tiling_->dilationH + 1;
    return (hoCal - 1) * self->ctx.tiling_->strideH + khDilation;
}

template <class Intf>
__aicore__ inline void InitTque(Intf *self)
{
    uint64_t aMatrixByteSize = static_cast<uint64_t>(self->ctx.tiling_->stepM) * self->ctx.tiling_->baseM * self->ctx.tiling_->stepKa *
                               self->ctx.tiling_->baseK * sizeof(typename Intf::SrcT);
    uint32_t hiCal = CalHi<Intf>(self);
    uint64_t bMatrixByteSize;
    if (self->ctx.tiling_->baseN > (static_cast<uint64_t>(self->ctx.tiling_->Hk) * self->ctx.tiling_->Wk * 16)) {
        bMatrixByteSize = static_cast<uint64_t>(hiCal) * self->ctx.tiling_->Wi * self->ctx.tiling_->stepN *
                          (self->ctx.tiling_->baseN / (static_cast<uint64_t>(self->ctx.tiling_->Hk) * self->ctx.tiling_->Wk)) *
                          sizeof(typename Intf::SrcT);
    } else {
        bMatrixByteSize = static_cast<uint64_t>(hiCal) * self->ctx.tiling_->Wi * self->ctx.tiling_->channelSize * sizeof(typename Intf::SrcT);
    }
    self->ctx.pipe_.InitBuffer(self->ctx.qidA1_, self->ctx.tiling_->al1Pbuffer, aMatrixByteSize);
    self->ctx.pipe_.InitBuffer(self->ctx.qidB1_, self->ctx.tiling_->bl1Pbuffer, bMatrixByteSize);
    self->ctx.pipe_.InitBuffer(
        self->ctx.qidA2_, self->ctx.tiling_->al0Pbuffer, self->ctx.baseMK_ * sizeof(typename Intf::SrcT));
    self->ctx.pipe_.InitBuffer(
        self->ctx.qidB2_, self->ctx.tiling_->bl0Pbuffer, self->ctx.baseKN_ * sizeof(typename Intf::SrcT));
    self->ctx.pipe_.InitBuffer(
        self->ctx.qidCO1_, self->ctx.tiling_->cl0Pbuffer, self->ctx.baseMN_ * sizeof(typename Intf::L0cT));
}

template <class Intf>
static __aicore__ inline void CalcParamsL12L0b(Intf *self, uint16_t kPos)
{
    // load3dStepK
    self->ctx.load3d_.kExtension = self->ctx.baseUseN_;
    // load3dStepM
    self->ctx.load3d_.mExtension =
        Ceil(self->ctx.baseUseK_, 32 / sizeof(typename Intf::SrcT)) * 32 / sizeof(typename Intf::SrcT);
    // posK
    self->ctx.load3d_.kStartPt = (self->ctx.curNL0Idx_ % self->ctx.tiling_->stepN) * self->ctx.tiling_->baseN;
    // posM
    self->ctx.load3d_.mStartPt =
        (kPos - kPos % self->ctx.tiling_->stepKb) * self->ctx.tiling_->baseK % self->ctx.tiling_->Wo +
        kPos % self->ctx.tiling_->stepKb * self->ctx.tiling_->baseK;
    self->ctx.load3d_.strideW = self->ctx.tiling_->strideW;
    self->ctx.load3d_.strideH = self->ctx.tiling_->strideH;
    self->ctx.load3d_.filterW = self->ctx.tiling_->Wk;
    self->ctx.load3d_.filterH = self->ctx.tiling_->Hk;
    self->ctx.load3d_.dilationFilterW = self->ctx.tiling_->dilationW;
    self->ctx.load3d_.dilationFilterH = self->ctx.tiling_->dilationH;
    self->ctx.load3d_.filterSizeW = (self->ctx.tiling_->Wk >> 8) & 255;
    self->ctx.load3d_.filterSizeH = (self->ctx.tiling_->Hk >> 8) & 255;
    self->ctx.load3d_.enTranspose = 0;
    self->ctx.load3d_.fMatrixCtrl = 0;
    self->ctx.load3d_.channelSize = self->ctx.tiling_->channelSize;
}

template <class Intf>
static __aicore__ inline void CalcParamsSetFmatrix(Intf *self)
{
    // W
    self->ctx.load3d_.l1W = self->ctx.tiling_->Wi;
    // H
    uint64_t k = static_cast<uint64_t>(self->ctx.tiling_->stepKb) * self->ctx.tiling_->baseK;
    uint64_t ho = (k + self->ctx.tiling_->Wo - 1) / self->ctx.tiling_->Wo;
    if ((k % self->ctx.tiling_->Wo != 0) && (self->ctx.tiling_->Wo % k != 0)) {
        ho = ho + 1;
    }
    self->ctx.load3d_.l1H =
        (ho - 1) * self->ctx.tiling_->strideH + static_cast<uint64_t>(self->ctx.tiling_->Hk - 1) * self->ctx.tiling_->dilationH + 1;
    if (self->ctx.load3d_.l1H > self->ctx.tiling_->Hi) {
        self->ctx.load3d_.l1H = self->ctx.tiling_->Hi;
    }
    for (size_t i = 0; i < PAD_SIZE; i++) {
        self->ctx.load3d_.padList[i] = 0;
    }
}

template <class Intf>
static __aicore__ inline void LoadL12L0b(Intf *self, const LocalTensor<typename Intf::SrcT> &l1BMatrix)
{
    self->ctx.b2_ = self->ctx.qidB2_.template AllocTensor<typename Intf::SrcT>();
    LoadDataImpl(self->ctx.b2_, l1BMatrix, self->ctx.load3d_);
    self->ctx.qidB2_.EnQue(self->ctx.b2_);
    self->ctx.qidB2_.template DeQue<typename Intf::SrcT>();
}

template <class Intf>
static __aicore__ inline void Compute(Intf *self)
{
    self->ctx.c1_ = self->ctx.qidCO1_.template AllocTensor<typename Intf::L0cT>();
    for (uint32_t k = 0; k < self->ctx.kIter_; k++) {
        self->ctx.baseUseK_ = (k + 1 == self->ctx.kIter_) ? self->ctx.tailK_ : self->ctx.tiling_->baseK;
        /*
        通过M和K的奇偶判断load到L1A ping还是L1A pong, BL1同理
                    kL1Idx=0  kL1Idx=1 kL1Idx=2
                    ----------------------------
        mL1Idx=0    |  pong  |  ping  |  pong  |
                    ----------------------------
        mL1Idx=1    |  ping  |  pong  |  ping  |
                    ----------------------------
        mL1Idx=2    |  pong  |  ping  |  pong  |
                    ----------------------------
        */
        bool cachePosA1 = (self->ctx.curML1Idx_ / self->ctx.tiling_->stepM & 1) ^ (k / self->ctx.tiling_->stepKa & 1);
        auto a1 = ConvolutionBackpropFunc::LoadToA1<Intf, typename Intf::SrcT>(self, cachePosA1, k);

        bool cachePosB1 = (self->ctx.curNL1Idx_ / self->ctx.tiling_->stepN & 1) ^ (k / self->ctx.tiling_->stepKb & 1);
        auto b1 = ConvolutionBackpropFunc::LoadToB1<Intf, typename Intf::SrcT>(self, cachePosB1, k);

        CalcParamsL12L0a<Intf>(self, k);
        LoadL12L0a<Intf>(self, a1, k);

        CalcParamsSetFmatrix<Intf>(self);
        CalcParamsL12L0b<Intf>(self, k);
        LoadL12L0b<Intf>(self, b1);

        CalcParamsMmad<Intf>(self, k);
        MmadLocal<Intf>(self);
    }
    self->ctx.qidCO1_.EnQue(self->ctx.c1_);
    self->ctx.qidCO1_.template DeQue<typename Intf::L0cT>();
}

template <class Intf>
static __aicore__ inline void UpdateIdxAndStep(Intf *self)
{
    self->ctx.curML0Idx_ = self->ctx.curML1Idx_;
    self->ctx.curNL0Idx_ = self->ctx.curNL1Idx_;
    self->ctx.curStepM_ = (self->ctx.mIter_ - self->ctx.curML0Idx_) > self->ctx.tiling_->stepM
                                ? self->ctx.tiling_->stepM
                                : (self->ctx.mIter_ - self->ctx.curML1Idx_);
    self->ctx.curStepN_ = (self->ctx.nIter_ - self->ctx.curNL0Idx_) > self->ctx.tiling_->stepN
                                ? self->ctx.tiling_->stepN
                                : (self->ctx.nIter_ - self->ctx.curNL1Idx_);
}

template <class Intf>
struct Init {
    // 定义call函数的默认重载函数，支持任意类型任意数量的参数
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf *self, const TConvTiling *__restrict tiling)
    {
        self->ctx.tiling_ = tiling;
        CheckTiling<Intf>(self);
        InitParams<Intf>(self);
        InitTque<Intf>(self);
    }
};

template <class Intf>
struct SetFmap {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf *self, const GlobalTensor<typename Intf::SrcT> &fmap)
    {
        self->ctx.fmapGlobal_ = fmap;
    }
};

template <class Intf>
struct SetOutBackprop {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf *self, const GlobalTensor<typename Intf::SrcT> &outBackprop)
    {
        self->ctx.outBackPropGlobal_ = outBackprop;
    }
};

template <class Intf>
struct SetSingleShape {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf *self, uint32_t singleShapeM, uint64_t singleShapeN, uint64_t singleShapeK)
    {
        self->ctx.singleShapeCout_ = singleShapeM;
        self->ctx.singleShapeCin_ = singleShapeN / self->ctx.tiling_->Hk / self->ctx.tiling_->Wk;
        if (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NC1HWC0) {
            self->ctx.singleShapeCin_ =
                Ceil(self->ctx.singleShapeCin_, self->ctx.tiling_->channelSize) * self->ctx.tiling_->channelSize;
        }
        self->ctx.singleShapeHo_ = singleShapeK / self->ctx.tiling_->Wo;
        InitStepMParams<Intf>(self);
        InitStepKParams<Intf>(self);
        InitStepNParams<Intf>(self);
    }
};

template <class Intf, bool sync>
struct Iterate {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline bool call(Intf *self, bool enPartialSum)
    {
        /*
        |   <---------singleShapeM------->        |
        |  <---L1A_ping--->  |  <---L1A_pong--->  |
        |_L0A1_|_L0A2_|_L0A3_|_L0A4_|_L0A5_|_L0A6_|
        ↑  <--curStepM_-->    |                    ↑
        curML0Idx_            ↑                  mIter_
        curML1Idx_        next_curML1Idx

        |   <---------singleShapeN------->        |
        |  <---L1B_ping--->  |  <---L1B_pong--->  |
        |_L0B1_|_L0B2_|_L0B3_|_L0B4_|_L0B5_|_L0B6_|
        ↑  <--curStepN_-->    |                    ↑
        curNL0Idx_            ↑                   nIter_
        curNL1Idx_       next_curNL1Idx

        order_N表示L1上驻留B循环A，顺序为L1A_ping * L1B_ping, L1A_pong * L1B_ping，L1A_ping * L1B_pong，L1A_pong * L1B_pong
        L0上也是驻留B，循环A
        order_N: L0A1*L0B1, L0A2*L0B1, L0A3*L0B1, L0A1*L0B2 ………… L0A3*L0B3，L0A4*L0B1，L0A5*L0B1 …… L0A6*L0B6
        order_M: L0A1*L0B1, L0A1*L0B2, L0A1*L0B3, L0A2*L0B1 ………… L0A3*L0B3，L0A1*L0B4，L0A1*L0B5 …… L0A6*L0B6
        */
        // 更新idx，用L1、L1step、L0三个指针控制走位和计算offset，表示计算第几个mL0 * baseN
        if (unlikely(self->ctx.isFirstIter_)) {
            self->ctx.curML0Idx_ = 0;
            self->ctx.curNL0Idx_ = 0;
            self->ctx.curML1Idx_ = 0;
            self->ctx.curNL1Idx_ = 0;
            self->ctx.isFirstIter_ = false;
            self->ctx.curStepM_ = (self->ctx.mIter_ - self->ctx.curML0Idx_) > self->ctx.tiling_->stepM
                                        ? self->ctx.tiling_->stepM
                                        : (self->ctx.mIter_ - self->ctx.curML1Idx_);
            self->ctx.curStepN_ = (self->ctx.nIter_ - self->ctx.curNL0Idx_) > self->ctx.tiling_->stepN
                                        ? self->ctx.tiling_->stepN
                                        : (self->ctx.nIter_ - self->ctx.curNL1Idx_);
        } else if (likely(self->ctx.tiling_->iterateOrder == static_cast<int>(IterateOrder::ORDER_N))) {
            if (++self->ctx.curML0Idx_ >= self->ctx.curML1Idx_ + self->ctx.curStepM_) {
                self->ctx.curML0Idx_ = self->ctx.curML1Idx_;
                if (++self->ctx.curNL0Idx_ >= self->ctx.curNL1Idx_ + self->ctx.curStepN_) {
                    self->ctx.curML1Idx_ += self->ctx.curStepM_;
                    if (self->ctx.curNL0Idx_ >= self->ctx.nIter_ && self->ctx.curML1Idx_ >= self->ctx.mIter_) {
                        return false;
                    }
                    if (self->ctx.curML1Idx_ >= self->ctx.mIter_) {
                        self->ctx.curML1Idx_ = 0;
                        self->ctx.curNL1Idx_ += self->ctx.curStepN_;
                    }
                    UpdateIdxAndStep<Intf>(self);
                }
            }
        } else {  // order_M
            if (++self->ctx.curNL0Idx_ >= self->ctx.curNL1Idx_ + self->ctx.curStepN_) {
                self->ctx.curNL0Idx_ = self->ctx.curNL1Idx_;
                if (++self->ctx.curML0Idx_ >= self->ctx.curML1Idx_ + self->ctx.curStepM_) {
                    self->ctx.curNL1Idx_ += self->ctx.curStepN_;
                    if (self->ctx.curML0Idx_ >= self->ctx.mIter_ && self->ctx.curNL1Idx_ >= self->ctx.nIter_) {
                        return false;
                    }
                    if (self->ctx.curNL1Idx_ >= self->ctx.nIter_) {
                        self->ctx.curNL1Idx_ = 0;
                        self->ctx.curML1Idx_ += self->ctx.curStepM_;
                    }
                    UpdateIdxAndStep<Intf>(self);
                }
            }
        }
        self->ctx.baseUseM_ =
            (self->ctx.curML0Idx_ + 1 == self->ctx.mIter_) ? self->ctx.tailM_ : self->ctx.tiling_->baseM;
        self->ctx.baseUseN_ =
            (self->ctx.curNL0Idx_ + 1 == self->ctx.nIter_) ? self->ctx.tailN_ : self->ctx.tiling_->baseN;
        Compute<Intf>(self);
        return true;
    }
};

template <class Intf, bool sync>
struct IterateAll {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf *self, const GlobalTensor<typename Intf::DstT> &output, uint8_t enAtomic)
    {
        while (self->template Iterate<sync>()) {
            self->template GetTensorC<sync>(output, enAtomic);
        }
        self->ctx.isFirstIter_ = true;
    }
};

template <class Intf, bool sync>
struct GetTensorC {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
        uint8_t enAtomic = 0, bool enSequentialWrite = false)
    {
        LoadL0c2Gm<Intf>(self, output, enAtomic, enSequentialWrite);
    }
};

template <class Intf>
struct End {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf *self)
    {
        if (self->ctx.usingCacheA1Ping_) {
            self->ctx.qidA1_.FreeTensor(self->ctx.cacheA1BufPing_);
        }
        if (self->ctx.usingCacheA1Pong_) {
            self->ctx.qidA1_.FreeTensor(self->ctx.cacheA1BufPong_);
        }
        if (self->ctx.usingCacheB1Ping_) {
            self->ctx.qidB1_.FreeTensor(self->ctx.cacheB1BufPing_);
        }
        if (self->ctx.usingCacheB1Pong_) {
            self->ctx.qidB1_.FreeTensor(self->ctx.cacheB1BufPong_);
        }
        self->ctx.qidA1_.FreeAllEvent();
        self->ctx.qidB1_.FreeAllEvent();
        self->ctx.qidA2_.FreeAllEvent();
        self->ctx.qidB2_.FreeAllEvent();
        self->ctx.qidCO1_.FreeAllEvent();
    }
};
}  // namespace ConvolutionBackpropFunc
#endif
