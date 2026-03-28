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

#include "conv_bp_config_base.h"
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
    ASCENDC_ASSERT((self->ctx.tiling_->batch > 0),
        { KERNEL_LOG(KERNEL_ERROR, "orignal batch is %d , which should be larger than 0", self->ctx.tiling_->batch); });
    ASCENDC_ASSERT((self->ctx.tiling_->cin > 0), {
        KERNEL_LOG(KERNEL_ERROR, "orignal cin is %d , which should be larger than 0", self->ctx.tiling_->cin);
    });
    ASCENDC_ASSERT((self->ctx.tiling_->cout > 0), {
        KERNEL_LOG(KERNEL_ERROR, "orignal cout is %d , which should be larger than 0", self->ctx.tiling_->cout);
    });
    ASCENDC_ASSERT((self->ctx.tiling_->cin1G > 0), {
        KERNEL_LOG(KERNEL_ERROR, "orignal cin1G is %d , which should be larger than 0", self->ctx.tiling_->cin1G);
    });
    ASCENDC_ASSERT((self->ctx.tiling_->cout1G > 0), {
        KERNEL_LOG(KERNEL_ERROR, "orignal cout1G is %d , which should be larger than 0", self->ctx.tiling_->cout1G);
    });
    ASCENDC_ASSERT((self->ctx.tiling_->dout > 0),
        { KERNEL_LOG(KERNEL_ERROR, "orignal dout is %d , which should be larger than 0", self->ctx.tiling_->dout); });
    ASCENDC_ASSERT((self->ctx.tiling_->ho > 0),
        { KERNEL_LOG(KERNEL_ERROR, "orignal ho is %d , which should be larger than 0", self->ctx.tiling_->ho); });
    ASCENDC_ASSERT((self->ctx.tiling_->wo > 0),
        { KERNEL_LOG(KERNEL_ERROR, "orignal wo is %d , which should be larger than 0", self->ctx.tiling_->wo); });
    ASCENDC_ASSERT((self->ctx.tiling_->di > 0),
        { KERNEL_LOG(KERNEL_ERROR, "orignal di is %d , which should be larger than 0", self->ctx.tiling_->di); });
    ASCENDC_ASSERT((self->ctx.tiling_->hi > 0),
        { KERNEL_LOG(KERNEL_ERROR, "orignal hi is %d , which should be larger than 0", self->ctx.tiling_->hi); });
    ASCENDC_ASSERT((self->ctx.tiling_->wi > 0),
        { KERNEL_LOG(KERNEL_ERROR, "orignal wi is %d , which should be larger than 0", self->ctx.tiling_->wi); });
    ASCENDC_ASSERT((self->ctx.tiling_->dk > 0),
        { KERNEL_LOG(KERNEL_ERROR, "orignal dk is %d , which should be larger than 0", self->ctx.tiling_->dk); });
    ASCENDC_ASSERT((self->ctx.tiling_->hk > 0),
        { KERNEL_LOG(KERNEL_ERROR, "orignal hk is %d , which should be larger than 0", self->ctx.tiling_->hk); });
    ASCENDC_ASSERT((self->ctx.tiling_->wk > 0),
        { KERNEL_LOG(KERNEL_ERROR, "orignal wk is %d , which should be larger than 0", self->ctx.tiling_->wk); });
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
    self->ctx.tailM_ = self->ctx.singleShapeCout_ - self->ctx.tiling_->baseM * (self->ctx.mIter_ - 1);
#ifdef __CCE_KT_TEST__
    ASCENDC_ASSERT((self->ctx.mIter_ > 0),
        { KERNEL_LOG(KERNEL_ERROR, "self->ctx.mIter_ is %d , which should be larger than 0", self->ctx.mIter_); });
#endif
}

template <class Intf>
__aicore__ inline void InitStepKParams(Intf *self)
{
    uint64_t singleCoreHoWo = static_cast<uint64_t>(self->ctx.singleShapeHo_) * self->ctx.tiling_->wo;
    uint64_t kIter = Ceil(singleCoreHoWo, self->ctx.tiling_->baseK);
    self->ctx.kIter_ = kIter;
    self->ctx.tailK_ = singleCoreHoWo - self->ctx.tiling_->baseK * (kIter - 1);
    self->ctx.stepKaRound = Ceil(kIter, self->ctx.tiling_->stepKa);
    self->ctx.stepKbRound = Ceil(kIter, self->ctx.tiling_->stepKb);
#ifdef __CCE_KT_TEST__
    ASCENDC_ASSERT((self->ctx.kIter_ > 0),
        { KERNEL_LOG(KERNEL_ERROR, "self->ctx.kIter_ is %d , which should be larger than 0", self->ctx.kIter_); });
#endif
}

template <class Intf>
__aicore__ inline void InitStepNParams(Intf *self)
{
    self->ctx.nIter_ = Ceil(ShiftCeilChannelSize<Intf>(self->ctx.singleShapeCin_, self->ctx.tiling_->channelSize) *
        self->ctx.tiling_->channelSize * self->ctx.hwK_, self->ctx.tiling_->baseN);
    self->ctx.tailN_ =
        self->ctx.singleShapeCin_ * self->ctx.hwK_ - self->ctx.tiling_->baseN * (self->ctx.nIter_ - 1);
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
    self->ctx.hwK_ = self->ctx.tiling_->hk * self->ctx.tiling_->wk;
    self->ctx.hwO_ = static_cast<uint64_t>(self->ctx.tiling_->ho) * self->ctx.tiling_->wo;
    self->ctx.hwI_ = static_cast<uint64_t>(self->ctx.tiling_->hi) * self->ctx.tiling_->wi;
    self->ctx.kal1_ = self->ctx.tiling_->stepKa * self->ctx.tiling_->baseK;
    self->ctx.kbl1_ = self->ctx.tiling_->stepKb * self->ctx.tiling_->baseK;

    // simplify the calculation of hi = (ho - 1) * strideh + (hk - 1) * dilationh + 1
    self->ctx.strideKernelDilationH = static_cast<int64_t>(self->ctx.tiling_->hk - 1) * self->ctx.tiling_->dilationH
        + 1 - self->ctx.tiling_->strideH;

    self->ctx.isFirstIter_ = true;
    self->ctx.l0aPingPongFlag_ = 0;
    self->ctx.l0bPingPongFlag_ = 0;
    self->ctx.l0cPingPongFlag_ = 1;
    self->ctx.useL0PingPong_ = (self->ctx.tiling_->al0Pbuffer - 1) & (self->ctx.tiling_->bl0Pbuffer - 1);
    InitLoadToB2Params<Intf>(self);
    InitLoadToA2Params<Intf>(self);
    InitSetFmatrixParams<Intf>(self);
    InitMmadParams<Intf>(self);
}

template <class Intf>
__aicore__ inline void InitTque(Intf *self)
{
    uint32_t aMatrixByteSize = self->ctx.tiling_->stepM * self->ctx.tiling_->baseM *
        self->ctx.tiling_->stepKa * self->ctx.tiling_->baseK * sizeof(typename Intf::SrcT);
    uint32_t bl1BoundByteSize = self->ctx.tiling_->bl1Bound * sizeof(typename Intf::SrcT);
    self->ctx.pipe_.InitBuffer(self->ctx.a1Ping_, 1, aMatrixByteSize);
    self->ctx.pipe_.InitBuffer(self->ctx.b1Ping_, 1, bl1BoundByteSize);
    if (self->ctx.tiling_->al1Pbuffer > 1) {
        self->ctx.pipe_.InitBuffer(self->ctx.a1Pong_, 1, aMatrixByteSize);
    }
    if (self->ctx.tiling_->bl1Pbuffer > 1) {
        self->ctx.pipe_.InitBuffer(self->ctx.b1Pong_, 1, bl1BoundByteSize);
    }
    self->ctx.pipe_.InitBuffer(self->ctx.l0cPing_, 1, self->ctx.baseMN_ * sizeof(typename Intf::L0cT));
    if (self->ctx.tiling_->cl0Pbuffer > 1) {
        self->ctx.pipe_.InitBuffer(self->ctx.l0cPong_, 1, self->ctx.baseMN_ * sizeof(typename Intf::L0cT));
    }
    self->ctx.pipe_.InitBuffer(self->ctx.l0aBuf_, TOTAL_L0A_SIZE);
    self->ctx.pipe_.InitBuffer(self->ctx.l0bBuf_, TOTAL_L0B_SIZE);
}

template <class Intf>
__aicore__ inline void CalcParamsL12L0b(Intf *self, uint64_t kPos)
{
    // load3dStepK
    self->ctx.load3dB_.kExtension = self->ctx.baseUseN_;
    // posK
    uint32_t localN = ShiftDivChannelSize<Intf>(self->ctx.tiling_->baseN, self->ctx.tiling_->channelSize);
    uint32_t localUseN = ShiftDivChannelSize<Intf>(self->ctx.baseUseN_, self->ctx.tiling_->channelSize);
    uint32_t kStartLocal = RemainderOfHkWk(self->ctx.curNL1Idx_ * localN, self->ctx.hwK_) +
        RemainderStepN(self->ctx.curNL0Idx_, self->ctx.tiling_->stepN) * localN;
    self->ctx.load3dB_.kStartPt = kStartLocal * self->ctx.tiling_->channelSize;
    self->ctx.load3dB_.channelSize =
        CeilHkWk(kStartLocal + localUseN, self->ctx.hwK_) * self->ctx.tiling_->channelSize;
}

template <class Intf>
__aicore__ inline void CalcParamsL12L0a(Intf *self, uint64_t kPos)
{
    uint32_t alignedBaseUseM = ShiftCeilM0(self->ctx.baseUseM_, self->ctx.tiling_->m0) * self->ctx.tiling_->m0;
    self->ctx.load3dA_.kExtension = alignedBaseUseM;
    self->ctx.load3dA_.channelSize = alignedBaseUseM;
}

template <class Intf>
__aicore__ inline void LoadL12L0b(Intf *self, const LocalTensor<typename Intf::SrcT> &l1BMatrix,
                                  LocalTensor<typename Intf::SrcT> &l0b)
{
    static constexpr IsResetLoad3dConfig LOAD3D_CONFIG_220 = {false, true};
    SetFmatrix(self->ctx.load3dB_.l1H, self->ctx.load3dB_.l1W, self->ctx.load3dB_.padList,
               FmatrixMode::FMATRIX_RIGHT);
    LoadData<typename Intf::SrcT, LOAD3D_CONFIG_220>(l0b, l1BMatrix, self->ctx.load3dB_);
}

template <class Intf>
__aicore__ inline void FreeA1Tensor(Intf *self, bool a1PingPongFlag)
{
    if (a1PingPongFlag) {
        self->ctx.a1Ping_.FreeTensor(self->ctx.cacheA1BufPing_);
#ifdef ASCENDC_CPU_DEBUG
        self->ctx.cacheA1BufPing_.SetSize(0);
#endif
    } else {
        self->ctx.a1Pong_.FreeTensor(self->ctx.cacheA1BufPong_);
#ifdef ASCENDC_CPU_DEBUG
        self->ctx.cacheA1BufPong_.SetSize(0);
#endif
    }
}

template <class Intf>
__aicore__ inline void FreeB1Tensor(Intf *self, bool b1PingPongFlag)
{
    if (b1PingPongFlag) {
        self->ctx.b1Ping_.FreeTensor(self->ctx.cacheB1BufPing_);
#ifdef ASCENDC_CPU_DEBUG
        self->ctx.cacheB1BufPing_.SetSize(0);
#endif
    } else {
        self->ctx.b1Pong_.FreeTensor(self->ctx.cacheB1BufPong_);
#ifdef ASCENDC_CPU_DEBUG
        self->ctx.cacheB1BufPong_.SetSize(0);
#endif
    }
}

template <class Intf>
__aicore__ inline void Compute(Intf *self, Out2L1ScalarParams& out2L1Params)
{
    CalcParamsL12L0b<Intf>(self, 0);
    CalcParamsL12L0a<Intf>(self, 0);
    CalcParamsMmad<Intf>(self, 0);
    LocalTensor<typename Intf::SrcT> l0a;
    LocalTensor<typename Intf::SrcT> l0b;
    LocalTensor<typename Intf::L0cT> l0c;
    uint32_t curML0IdxModstepKaMulBaseM = RemainderStepM(self->ctx.curML0Idx_, self->ctx.tiling_->stepM) * self->ctx.tiling_->baseM;
    constexpr uint32_t l0aPingPongAddr = TOTAL_L0A_SIZE / 2 / sizeof(typename Intf::SrcT);
    constexpr uint32_t l0bPingPongAddr = TOTAL_L0B_SIZE / 2 / sizeof(typename Intf::SrcT);
    CalOut2L1ScalarParams(self, out2L1Params);

    if (self->ctx.l0cPingPongFlag_) {
        l0c = self->ctx.l0cPing_.template AllocTensor<typename Intf::L0cT>();
    } else {
        l0c = self->ctx.l0cPong_.template AllocTensor<typename Intf::L0cT>();
    }

    bool a1PingPongFlag = true;
    bool b1PingPongFlag = true;
    bool isAL1PingPong = self->ctx.tiling_->al1Pbuffer > 1;
    bool isBL1PingPong = self->ctx.tiling_->bl1Pbuffer > 1;
    uint32_t kaIdx = 0;
    uint32_t kbIdx = 0;
    uint64_t kaStepIdx = 0;
    uint64_t kbStepIdx = 0;
    uint64_t curMKL1Idx = self->ctx.stepKaRound * DivStepM(self->ctx.curML1Idx_, self->ctx.tiling_->stepM);
    uint64_t curNKL1Idx = self->ctx.stepKbRound * DivStepN(self->ctx.curNL1Idx_, self->ctx.tiling_->stepN);

    for (uint64_t k = 0; k < self->ctx.kIter_; k++) {
        bool isLastKIter = k + 1 == self->ctx.kIter_;
        bool isLastStepKa = kaIdx + 1 == self->ctx.tiling_->stepKa;
        bool isLastStepKb = kbIdx + 1 == self->ctx.tiling_->stepKb;
        bool isLoadA1 = kaIdx == 0;
        bool isLoadB1 = kbIdx == 0;
        self->ctx.baseUseK_ = isLastKIter ? self->ctx.tailK_ : self->ctx.tiling_->baseK;

        /*
        通过M*K的奇偶判断load到L1A ping还是L1A pong, BL1同理
                    kL1Idx=0  kL1Idx=1 kL1Idx=2
                    ----------------------------
        mL1Idx=0    |  ping  |  pong  |  ping  |
                    ----------------------------
        mL1Idx=1    |  pong  |  ping  |  pong  |
                    ----------------------------
        mL1Idx=2    |  ping  |  pong  |  ping  |
                    ----------------------------
        */
        if (self->ctx.tiling_->bl1Pbuffer > 1) {
            b1PingPongFlag = (curNKL1Idx + kbStepIdx + 1) & 1;
        }
        ConvolutionBackpropFunc::LoadToB1<Intf, typename Intf::SrcT>(self, b1PingPongFlag, k, out2L1Params, isLoadB1,
                                                                     kbStepIdx);

        if (self->ctx.tiling_->al1Pbuffer > 1) {
            a1PingPongFlag = (curMKL1Idx + kaStepIdx + 1) & 1;
        }
        ConvolutionBackpropFunc::LoadToA1<Intf, typename Intf::SrcT>(self, a1PingPongFlag, k, out2L1Params, isLoadA1,
                                                                     kaStepIdx);

        WaitFlag<HardEvent::M_MTE1>(self->ctx.l0aPingPongFlag_);

        uint32_t alignedBaseUseK =
            ShiftCeilChannelSize<Intf>(self->ctx.baseUseK_, self->ctx.tiling_->k0) * self->ctx.tiling_->k0;
        l0b = self->ctx.l0bBuf_.template Get<typename Intf::SrcT>();
        if (self->ctx.l0aPingPongFlag_) {
            l0b = l0b[l0bPingPongAddr];
        }
        // posM
        self->ctx.load3dB_.mStartPt = (k - kbIdx) * self->ctx.tiling_->baseK % self->ctx.tiling_->wo +
            kbIdx * self->ctx.tiling_->baseK;
        // load3dStepM
        self->ctx.load3dB_.mExtension = alignedBaseUseK;
        if (unlikely(out2L1Params.isLoad2L1B && isLoadB1)) {
            if (b1PingPongFlag) {
                self->ctx.cacheB1BufPing_ = self->ctx.b1Ping_.template DeQue<typename Intf::SrcT>();
            } else {
                self->ctx.cacheB1BufPong_ = self->ctx.b1Pong_.template DeQue<typename Intf::SrcT>();
            }
        }

        if (b1PingPongFlag) {
            self->ctx.load3dB_.l1H = self->ctx.bL1HiCopyLenPing;
            self->ctx.load3dB_.padList[2] = self->ctx.bL1PadUpPing;
            LoadL12L0b<Intf>(self, self->ctx.cacheB1BufPing_, l0b);
        } else {
            self->ctx.load3dB_.l1H = self->ctx.bL1HiCopyLenPong;
            self->ctx.load3dB_.padList[2] = self->ctx.bL1PadUpPong;
            LoadL12L0b<Intf>(self, self->ctx.cacheB1BufPong_, l0b);
        }

        if (out2L1Params.isFreeBL1 && (isLastStepKb || isLastKIter)) {
            FreeB1Tensor(self, b1PingPongFlag);
        }

        l0a = self->ctx.l0aBuf_.template Get<typename Intf::SrcT>();
        if (self->ctx.l0aPingPongFlag_) {
            l0a = l0a[l0aPingPongAddr];
        }
        uint32_t mOffset = curML0IdxModstepKaMulBaseM * self->ctx.curLoadKal1_;
        self->ctx.srcL12L0aOffset_ = kaIdx * self->ctx.tiling_->baseK * self->ctx.tiling_->channelSize + mOffset;
        self->ctx.load3dA_.mExtension = alignedBaseUseK;
        if (unlikely(out2L1Params.isLoad2L1A && isLoadA1)) {
            if (a1PingPongFlag) {
                self->ctx.cacheA1BufPing_ = self->ctx.a1Ping_.template DeQue<typename Intf::SrcT>();
            } else {
                self->ctx.cacheA1BufPong_ = self->ctx.a1Pong_.template DeQue<typename Intf::SrcT>();
            }
	    }

        if (a1PingPongFlag) {
            LoadL12L0a<Intf>(self, self->ctx.cacheA1BufPing_, k, l0a, out2L1Params, kaStepIdx);
        } else {
            LoadL12L0a<Intf>(self, self->ctx.cacheA1BufPong_, k, l0a, out2L1Params, kaStepIdx);
        }

        if (out2L1Params.isFreeAL1 && (isLastStepKa || isLastKIter)) {
            FreeA1Tensor(self, a1PingPongFlag);
        }

        SetFlag<HardEvent::MTE1_M>(self->ctx.l0aPingPongFlag_);
        WaitFlag<HardEvent::MTE1_M>(self->ctx.l0aPingPongFlag_);
        self->ctx.mmad_.cmatrixInitVal = k == 0;
        self->ctx.mmad_.k = self->ctx.baseUseK_;
        MmadLocal<Intf>(self, l0a, l0b, l0c);
        SetFlag<HardEvent::M_MTE1>(self->ctx.l0aPingPongFlag_);

        self->ctx.l0aPingPongFlag_ ^= self->ctx.useL0PingPong_;
        if (isLastStepKa) {
            ++kaStepIdx;
            kaIdx = 0;
        } else {
            ++kaIdx;
        }
        if (isLastStepKb) {
            ++kbStepIdx;
            kbIdx = 0;
        } else {
            ++kbIdx;
        }
    }

    if (self->ctx.l0cPingPongFlag_) {
        self->ctx.l0cPing_.EnQue(l0c);
    } else {
        self->ctx.l0cPong_.EnQue(l0c);
    }
}
template <class Intf>
__aicore__ inline void UpdateIdxAndStep(Intf *self)
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
    static __aicore__ inline void call(Intf *self, const TConv3DDwTiling *__restrict tiling)
    {
        self->ctx.tiling_ = tiling;
        CheckTiling<Intf>(self);
        InitParams<Intf>(self);
        InitTque<Intf>(self);
        SetHF32Mode(self->ctx.tiling_->hf32Flag);
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
    static __aicore__ inline void call(Intf *self, uint64_t singleShapeM, uint64_t singleShapeN, uint64_t singleShapeK,
                                                   uint32_t hoStartIdx)
    {
        self->ctx.singleShapeCout_ = singleShapeM;
        self->ctx.singleShapeCin_ = DivHkWk(singleShapeN, self->ctx.hwK_);
        // fp32场景下GM上8对齐
        self->ctx.singleShapeCin_ =
            Ceil(self->ctx.singleShapeCin_, self->ctx.tiling_->channelSize) * self->ctx.tiling_->channelSize;
        self->ctx.singleShapeHo_ = singleShapeK / self->ctx.tiling_->wo;
        self->ctx.hoStartIdx_ = hoStartIdx;
        self->ctx.hiStartIdx_ = hoStartIdx * self->ctx.tiling_->strideH - self->ctx.tiling_->padUp;
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

        // 以开DB为例，循环顺序NMK时，如果K方向需要两块buffer，M轴每循环到stepM最后一次时需要释放AL1上的ping pong buffer，第一次循环时载入
        // 如果循环顺序为MNK时，如果K方向需要两块buffer，M轴最后一次循环时需要释放AL1上的ping pong buffer，第一次循环时载入
        // 如果K方向需要的buffer数量大于bl1Pbuffer，当K循环到stepKa时就需要置换AL1
        // B矩阵计算思路同A矩阵，区别是MN反过来
        Out2L1ScalarParams out2L1Params;
        bool kIterCeilStepKaGreaterAl1Pbuffer =
            self->ctx.kIter_ > self->ctx.tiling_->stepKa * self->ctx.tiling_->al1Pbuffer;
        bool kIterCeilStepKbGreaterBl1Pbuffer =
            self->ctx.kIter_ > self->ctx.tiling_->stepKb * self->ctx.tiling_->bl1Pbuffer;
        out2L1Params.isLoad2L1A = kIterCeilStepKaGreaterAl1Pbuffer;
        out2L1Params.isFreeAL1 = kIterCeilStepKaGreaterAl1Pbuffer;
        out2L1Params.isLoad2L1B = kIterCeilStepKbGreaterBl1Pbuffer;
        out2L1Params.isFreeBL1 = kIterCeilStepKbGreaterBl1Pbuffer;
        if (unlikely(self->ctx.isFirstIter_)) {
            self->ctx.curML0Idx_ = 0;
            self->ctx.curNL0Idx_ = 0;
            self->ctx.curML1Idx_ = 0;
            self->ctx.curNL1Idx_ = 0;
            self->ctx.isFirstIter_ = false;
            self->ctx.curStepM_ = self->ctx.mIter_ > self->ctx.tiling_->stepM
                                        ? self->ctx.tiling_->stepM
                                        : self->ctx.mIter_;
            self->ctx.curStepN_ = self->ctx.nIter_ > self->ctx.tiling_->stepN
                                        ? self->ctx.tiling_->stepN
                                        : self->ctx.nIter_;
            bool isLastNLoop = self->ctx.nIter_ == 1;
            bool isLastMLoop = self->ctx.mIter_ == 1;
            bool isNLastStep = isLastNLoop || self->ctx.tiling_->stepN == 1;
            bool isMLastStep = isLastMLoop || self->ctx.tiling_->stepM == 1;
            out2L1Params.isLoad2L1A = true;
            out2L1Params.isFreeAL1 = kIterCeilStepKaGreaterAl1Pbuffer ||
                (self->ctx.tiling_->iterateOrder && isMLastStep) ||                  // OrderN
                (!(self->ctx.tiling_->iterateOrder) && isLastNLoop && isMLastStep);  // OrderM
            out2L1Params.isLoad2L1B = true;
            out2L1Params.isFreeBL1 = kIterCeilStepKbGreaterBl1Pbuffer ||
                (self->ctx.tiling_->iterateOrder && isLastMLoop && isNLastStep) ||    // OrderN
                (!(self->ctx.tiling_->iterateOrder) && isNLastStep);                  // OrderM
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
                    if (self->ctx.curML0Idx_ == 0) {
                        out2L1Params.isLoad2L1B = true; // OrderN, N轴循环结束，需要置换BL1
                    }
                }
                out2L1Params.isLoad2L1A = true; // OrderN, M轴循环结束，需要置换AL1
            }
            if (unlikely(self->ctx.curML0Idx_ == self->ctx.mIter_ - 1) ||
                self->ctx.curML0Idx_ == self->ctx.curML1Idx_ + self->ctx.curStepM_ - 1) {
                out2L1Params.isFreeAL1 = true; // OrderN, M轴最后一次循环，需要释放AL1
            }
            if (unlikely(self->ctx.curML0Idx_ == self->ctx.mIter_ - 1) &&
                (unlikely(self->ctx.curNL0Idx_ == self->ctx.nIter_ - 1) ||
                self->ctx.curNL0Idx_ == self->ctx.curNL1Idx_ + self->ctx.curStepN_ - 1)) {
                out2L1Params.isFreeBL1 = true; // OrderN, N轴最后一次循环，需要释放BL1
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
                    if (self->ctx.curNL0Idx_ == 0) {
                        out2L1Params.isLoad2L1A = true; // OrderM, M轴循环结束，需要置换AL1
                    }
                }
                out2L1Params.isLoad2L1B = true; // OrderM, N轴循环结束，需要置换BL1
            }
            if (unlikely(self->ctx.curNL0Idx_ == self->ctx.nIter_ - 1) &&
                (unlikely(self->ctx.curML0Idx_ == self->ctx.mIter_ - 1) ||
                self->ctx.curML0Idx_ == self->ctx.curML1Idx_ + self->ctx.curStepM_ - 1)) {
                out2L1Params.isFreeAL1 = true; // OrderM, M轴最后一次循环，需要释放AL1
            }
            if (unlikely(self->ctx.curNL0Idx_ == self->ctx.nIter_ - 1) ||
                self->ctx.curNL0Idx_ == self->ctx.curNL1Idx_ + self->ctx.curStepN_ - 1) {
                out2L1Params.isFreeBL1 = true; // OrderM, N轴最后一次循环，需要释放BL1
            }
        }
        self->ctx.baseUseM_ =
            (self->ctx.curML0Idx_ + 1 == self->ctx.mIter_) ? self->ctx.tailM_ : self->ctx.tiling_->baseM;
        self->ctx.baseUseN_ =
            (self->ctx.curNL0Idx_ + 1 == self->ctx.nIter_) ? self->ctx.tailN_ : self->ctx.tiling_->baseN;
        if ASCEND_IS_AIC {
            Compute<Intf>(self, out2L1Params);
        }
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
        self->ctx.a1Ping_.FreeAllEvent();
        if (self->ctx.tiling_->al1Pbuffer > 1) {
            self->ctx.a1Pong_.FreeAllEvent();
        }
        self->ctx.b1Ping_.FreeAllEvent();
        if (self->ctx.tiling_->bl1Pbuffer > 1) {
            self->ctx.b1Pong_.FreeAllEvent();
        }
        self->ctx.l0cPing_.FreeAllEvent();
        if (self->ctx.tiling_->cl0Pbuffer > 1) {
            self->ctx.l0cPong_.FreeAllEvent();
        }

        if (self->ctx.tiling_->hf32Flag) {
            SetHF32Mode(false);
        }
    }
};
}  // namespace ConvolutionBackpropFunc
#endif