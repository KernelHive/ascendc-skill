/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file conv3d_bp_func.h
 * \brief
 */

#ifndef CONV3D_BP_FUNC_H
#define CONV3D_BP_FUNC_H

#include "conv3d_bp_util.h"
#include "kernel_operator.h"

#if __CCE_AICORE__ == 220
#include "impl/dav_v220/conv_bp_sub_func.h"
#include "impl/dav_v220/conv3d_bp_kernel_split.h"
#endif

DECLARE_CHECK_IMPL(Init);

DECLARE_CHECK_IMPL(SetOutBackprop);
DECLARE_CHECK_IMPL(SetWeight);
DECLARE_CHECK_IMPL(SetSingleShape);
DECLARE_CHECK_IMPL(SetStartIdx);
DECLARE_CHECK_SYNC_IMPL(Iterate);
DECLARE_CHECK_SYNC_IMPL(IterateAll);
DECLARE_CHECK_SYNC_IMPL(GetTensorC);
DECLARE_CHECK_IMPL(End);
namespace Convolution3DBackpropFunc {

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
    ASCENDC_ASSERT((self->ctx.tiling_->batch > 0), {
        KERNEL_LOG(KERNEL_ERROR, "orignal batch is %d , which should be larger than 0", self->ctx.tiling_->batch);
    });
    ASCENDC_ASSERT((self->ctx.tiling_->cin > 0), {
        KERNEL_LOG(KERNEL_ERROR, "orignal cin is %d , which should be larger than 0", self->ctx.tiling_->cin);
    });
    ASCENDC_ASSERT((self->ctx.tiling_->cout > 0), {
        KERNEL_LOG(KERNEL_ERROR, "orignal cout is %d , which should be larger than 0", self->ctx.tiling_->cout);
    });
    ASCENDC_ASSERT((self->ctx.tiling_->ho > 0), {
        KERNEL_LOG(KERNEL_ERROR, "orignal ho is %d , which should be larger than 0", self->ctx.tiling_->ho);
    });
    ASCENDC_ASSERT((self->ctx.tiling_->wo > 0), {
        KERNEL_LOG(KERNEL_ERROR, "orignal wo is %d , which should be larger than 0", self->ctx.tiling_->wo);
    });
    ASCENDC_ASSERT((self->ctx.tiling_->hi > 0), {
        KERNEL_LOG(KERNEL_ERROR, "orignal hi is %d , which should be larger than 0", self->ctx.tiling_->hi);
    });
    ASCENDC_ASSERT((self->ctx.tiling_->wi > 0), {
        KERNEL_LOG(KERNEL_ERROR, "orignal wi is %d , which should be larger than 0", self->ctx.tiling_->wi);
    });
    ASCENDC_ASSERT((self->ctx.tiling_->hk > 0), {
        KERNEL_LOG(KERNEL_ERROR, "orignal hk is %d , which should be larger than 0", self->ctx.tiling_->hk);
    });
    ASCENDC_ASSERT((self->ctx.tiling_->wk > 0), {
        KERNEL_LOG(KERNEL_ERROR, "orignal wk is %d , which should be larger than 0", self->ctx.tiling_->wk);
    });
    ASCENDC_ASSERT((self->ctx.tiling_->stepM == 1 && self->ctx.tiling_->stepN == 1), {
        KERNEL_LOG(KERNEL_ERROR, "stepM or stepN is invalid.");
    });
    ASCENDC_ASSERT((self->ctx.tiling_->singleCoreBatch > 0), {
        KERNEL_LOG(KERNEL_ERROR, "singleCoreBatch is %d , which should be larger than 0",
                   self->ctx.tiling_->singleCoreBatch);
    });
    ASCENDC_ASSERT((self->ctx.tiling_->singleCoreCout > 0), {
        KERNEL_LOG(KERNEL_ERROR, "singleCoreCout is %d , which should be larger than 0",
                   self->ctx.tiling_->singleCoreCout);
    });
    ASCENDC_ASSERT((self->ctx.tiling_->singleCoreHo > 0), {
        KERNEL_LOG(KERNEL_ERROR, "singleCoreHo is %d , which should be larger than 0", self->ctx.tiling_->singleCoreHo);
    });
    ASCENDC_ASSERT((self->ctx.tiling_->singleCoreCin > 0), {
        KERNEL_LOG(KERNEL_ERROR, "singleCoreCin is %d , which should be larger than 0",
                   self->ctx.tiling_->singleCoreCin);
    });
    ASCENDC_ASSERT((self->ctx.tiling_->baseM > 0), {
        KERNEL_LOG(KERNEL_ERROR, "baseM is %d , which should be larger than 0", self->ctx.tiling_->baseM);
    });
    ASCENDC_ASSERT((self->ctx.tiling_->baseK > 0), {
        KERNEL_LOG(KERNEL_ERROR, "baseK is %d , which should be larger than 0", self->ctx.tiling_->baseK);
    });
    ASCENDC_ASSERT((self->ctx.tiling_->baseN > 0), {
        KERNEL_LOG(KERNEL_ERROR, "baseN is %d , which should be larger than 0", self->ctx.tiling_->baseN);
    });
    ASCENDC_ASSERT((self->ctx.tiling_->stepM > 0), {
        KERNEL_LOG(KERNEL_ERROR, "stepM is %d , which should be larger than 0", self->ctx.tiling_->stepM);
    });
    ASCENDC_ASSERT((self->ctx.tiling_->stepN > 0), {
        KERNEL_LOG(KERNEL_ERROR, "stepN is %d , which should be larger than 0", self->ctx.tiling_->stepN);
    });
    ASCENDC_ASSERT((self->ctx.tiling_->stepKa > 0), {
        KERNEL_LOG(KERNEL_ERROR, "stepKa is %d , which should be larger than 0", self->ctx.tiling_->stepKa);
    });
    ASCENDC_ASSERT((self->ctx.tiling_->stepKb > 0), {
        KERNEL_LOG(KERNEL_ERROR, "stepKb is %d , which should be larger than 0", self->ctx.tiling_->stepKb);
    });
#endif
}

template <class Intf>
__aicore__ inline void InitStepMParams(Intf *self)
{
    if constexpr (Intf::conv3dConfig.enableKernelSplit) {
        self->ctx.mIter_ = Ceil(self->ctx.splitSingleShapeM_, self->ctx.tiling_->baseM);
        self->ctx.tailM_ = self->ctx.splitSingleShapeM_ - (self->ctx.mIter_ - 1) * self->ctx.tiling_->baseM;
    } else {
        self->ctx.mIter_ = Ceil(self->ctx.singleShapeM_, self->ctx.tiling_->baseM);
        self->ctx.tailM_ = self->ctx.singleShapeM_ - (self->ctx.mIter_ - 1) * self->ctx.tiling_->baseM;
    }
#ifdef __CCE_KT_TEST__
    ASCENDC_ASSERT((self->ctx.mIter_ > 0), {
        KERNEL_LOG(KERNEL_ERROR, "self->ctx.mIter_ is %d , which should be larger than 0", self->ctx.mIter_);
    });
#endif
}

template <class Intf>
__aicore__ inline void InitStepKParams(Intf *self)
{
    uint64_t tmpSingleCoreK = static_cast<uint64_t>(self->ctx.singleShapeCout1_) * self->ctx.HkWkC0_;
    if constexpr (Intf::conv3dConfig.enableKernelSplit) {
        tmpSingleCoreK = static_cast<uint64_t>(self->ctx.tiling_->cout1G) * self->ctx.splitHkWkC0_;
    }
    uint64_t kIter = Ceil(tmpSingleCoreK, self->ctx.tiling_->baseK);
    self->ctx.kIter_ = kIter;
    self->ctx.tailK_ = tmpSingleCoreK - (kIter - 1) * self->ctx.tiling_->baseK;
#ifdef __CCE_KT_TEST__
    ASCENDC_ASSERT((self->ctx.kIter_ > 0), {
        KERNEL_LOG(KERNEL_ERROR, "self->ctx.kIter_ is %d , which should be larger than 0", self->ctx.kIter_);
    });
#endif
    self->ctx.stepKaRound_ = Ceil(kIter, self->ctx.tiling_->stepKa);
    self->ctx.stepKbRound_ = Ceil(kIter, self->ctx.tiling_->stepKb);
}

template <class Intf>
__aicore__ inline void InitStepNParams(Intf *self)
{
    uint64_t singleShapeCinAlign = self->ctx.singleShapeCin1_ * self->ctx.tiling_->c0;
    self->ctx.nIter_ = Ceil(singleShapeCinAlign, self->ctx.tiling_->baseN);
    self->ctx.tailN_ = singleShapeCinAlign - (self->ctx.nIter_ - 1) * self->ctx.tiling_->baseN;
#ifdef __CCE_KT_TEST__
    ASCENDC_ASSERT((self->ctx.nIter_ > 0), {
        KERNEL_LOG(KERNEL_ERROR, "self->ctx.nIter_ is %d , which should be larger than 0", self->ctx.nIter_);
    });
#endif
}

template <class Intf>
__aicore__ inline void InitParams(Intf *self)
{
    self->ctx.baseMN_ = self->ctx.tiling_->baseM * self->ctx.tiling_->baseN;
    self->ctx.isFirstIter_ = true;
    self->ctx.usingCacheC1Ping_ = true;
    self->ctx.HkWk_ = self->ctx.tiling_->hk * self->ctx.tiling_->wk;
    self->ctx.HkWkC0_ = self->ctx.tiling_->hk * self->ctx.tiling_->wk * self->ctx.tiling_->c0;
    self->ctx.DkHkWkC0_ = self->ctx.tiling_->dk * self->ctx.tiling_->hk * self->ctx.tiling_->wk *
        self->ctx.tiling_->c0;

    self->ctx.curCin1Size_ = self->ctx.tiling_->stepN * self->ctx.tiling_->baseN / self->ctx.tiling_->c0;
    self->ctx.isB1FullLoadFlag_ = (self->ctx.tiling_->dk == 1 && self->ctx.tiling_->bl1Pbuffer == 1 &&
        self->ctx.tiling_->baseK * self->ctx.tiling_->stepKb >=
        self->ctx.tiling_->singleCoreCout * self->ctx.HkWk_) &&
        (self->ctx.curCin1Size_ >= self->ctx.tiling_->singleCoreCin1);
    self->ctx.hwI_ = static_cast<uint64_t>(self->ctx.tiling_->hi) * self->ctx.tiling_->wi;
    self->ctx.hwO_ = static_cast<uint64_t>(self->ctx.tiling_->ho) * self->ctx.tiling_->wo;
    if constexpr (std::is_same<typename Intf::DstT, float>::value) {
        self->ctx.alignedCout1_ = DivCeil(self->ctx.tiling_->cout1G * self->ctx.tiling_->c0, BLOCK_CUBE);
        self->ctx.alignedCout_ = self->ctx.alignedCout1_ * BLOCK_CUBE;
    }
    if constexpr (Intf::conv3dConfig.enableKernelSplit) {
#ifdef __CCE_KT_TEST__
        ASCENDC_ASSERT((self->ctx.tiling_->hk >= self->ctx.tiling_->strideH),
                       { KERNEL_LOG(KERNEL_ERROR, "kernelH should be GE strideH"); });
        ASCENDC_ASSERT((self->ctx.tiling_->wk >= self->ctx.tiling_->strideW),
                       { KERNEL_LOG(KERNEL_ERROR, "kernelW should be GE strideW"); });
#endif
        // 泛化时需考虑不整除场景
        self->ctx.splitHk_ = self->ctx.tiling_->hk / self->ctx.tiling_->strideH;
        self->ctx.splitWk_ = self->ctx.tiling_->wk / self->ctx.tiling_->strideW;
        self->ctx.splitHkWk_ = self->ctx.splitHk_ * self->ctx.splitWk_;
        self->ctx.splitHkWkC0_ = self->ctx.splitHkWk_ * self->ctx.tiling_->c0;
        self->ctx.splitHi_ = self->ctx.tiling_->hi / self->ctx.tiling_->strideH;
        self->ctx.splitWi_ = self->ctx.tiling_->wi / self->ctx.tiling_->strideW;
        self->ctx.channelSize_ = (self->ctx.tiling_->stepKa * self->ctx.tiling_->baseK) / self->ctx.splitHkWk_;
        self->ctx.curHoSize_ = CalFmapHForKernelSplit<Intf>(self, self->ctx.tiling_->baseM * self->ctx.tiling_->stepM);
    } else {
        self->ctx.channelSize_ = (self->ctx.tiling_->stepKa * self->ctx.tiling_->baseK) / self->ctx.HkWk_;
        self->ctx.curHoSize_ = CalFmapH<Intf>(self, self->ctx.tiling_->baseM * self->ctx.tiling_->stepM);
    }
    self->ctx.l0aPingPongFlag_ = 0;
    self->ctx.useL0PingPong_ = (self->ctx.tiling_->al0Pbuffer - 1) & (self->ctx.tiling_->bl0Pbuffer - 1);
    InitLoadToA2Params<Intf>(self);
    if constexpr((std::is_same<typename Intf::SrcT, bfloat16_t>::value) || (std::is_same<typename Intf::SrcT, half>::value)) {
        InitLoadToB2Params<Intf>(self);
    }
}

template <class Intf>
__aicore__ inline void InitTque(Intf *self)
{
    // fp32场景下baseK可能为8的倍数，非16倍数，但是GM中K0一定是16倍数，但是实际K可能仅有8，额外为padding数据
    uint32_t bMatrixByteSize = 0;
    uint32_t aMatrixByteSize = 0;
    if constexpr (Intf::conv3dConfig.enableKernelSplit) {
        uint32_t hoSize = (self->ctx.curHoSize_ < self->ctx.tiling_->ho) ? self->ctx.curHoSize_ : self->ctx.tiling_->ho;
        // 泛化时，每个小kernel需要加载的wo大小可能不一样，可能是wo wo-1 wo-2 ...
        aMatrixByteSize = hoSize * (self->ctx.tiling_->wo - 1) *
                          self->ctx.tiling_->stepKa * self->ctx.tiling_->baseK / self->ctx.splitHkWk_ *
                          sizeof(typename Intf::SrcT);
    } else {
        uint32_t hoSize = self->ctx.curHoSize_;
        uint64_t hoExpand = static_cast<uint64_t>(self->ctx.tiling_->ho - 1) * self->ctx.tiling_->strideH + 1;
        if (hoExpand < static_cast<uint64_t>(self->ctx.curHoSize_)) {
            hoSize = static_cast<uint32_t>(hoExpand);
        }
        aMatrixByteSize = hoSize * self->ctx.tiling_->wo * self->ctx.tiling_->strideW *
                          self->ctx.tiling_->stepKa * self->ctx.tiling_->baseK / self->ctx.HkWk_ *
                          sizeof(typename Intf::SrcT);
    }

    if constexpr(std::is_same<typename Intf::SrcT, float>::value) {
        bMatrixByteSize = self->ctx.tiling_->stepN * self->ctx.tiling_->baseN *
                          DivCeil(self->ctx.tiling_->stepKb * self->ctx.tiling_->baseK / self->ctx.HkWkC0_, 2) *
                          self->ctx.HkWk_ * BLOCK_CUBE * sizeof(typename Intf::SrcT);
    } else {
        bMatrixByteSize = self->ctx.tiling_->stepN * self->ctx.tiling_->baseN * self->ctx.tiling_->stepKb *
                          self->ctx.tiling_->baseK * sizeof(typename Intf::SrcT);
    }


    self->ctx.pipe_.InitBuffer(self->ctx.a1Ping_, 1, aMatrixByteSize);
    self->ctx.pipe_.InitBuffer(self->ctx.b1Ping_, 1, bMatrixByteSize);
    if (self->ctx.tiling_->al1Pbuffer > 1) {
        self->ctx.pipe_.InitBuffer(self->ctx.a1Pong_, 1, aMatrixByteSize);
    }
    if (self->ctx.tiling_->bl1Pbuffer > 1) {
        self->ctx.pipe_.InitBuffer(self->ctx.b1Pong_, 1, bMatrixByteSize);
    }

    self->ctx.pipe_.InitBuffer(self->ctx.c1Ping_, 1, self->ctx.baseMN_ * sizeof(typename Intf::L0cT));
    if (self->ctx.tiling_->cl0Pbuffer > 1) {
        self->ctx.pipe_.InitBuffer(self->ctx.c1Pong_, 1, self->ctx.baseMN_ * sizeof(typename Intf::L0cT));
    }
    self->ctx.pipe_.InitBuffer(self->ctx.l0aBuf_, TOTAL_L0A_SIZE);
    self->ctx.pipe_.InitBuffer(self->ctx.l0bBuf_, TOTAL_L0B_SIZE);
}

template <class Intf>
static __aicore__ inline void Compute(Intf *self)
{
    // 先刷新h方向的值，方便判断是否为有效计算
    UpdateLoadToA2ParamsM<Intf>(self);

    // 在跳过计算逻辑中，如果有部分无需跳过的操作逻辑。后续如果有类似逻辑，可以在此处继续增加
    // 当前已存在的操作为isFreeB1_为true的情况(B1全加载且循环至最后一块运算空间，需要释放B1空间)。此时预期的只有ping空间被使用
    if (!self->ctx.needComputeFlag_) {
        if (self->ctx.isFreeB1_ && !self->ctx.isLoadB1_) {
            self->ctx.b1Ping_.FreeTensor(self->ctx.cacheB1BufPing_);
        }
        return;
    }

    if ASCEND_IS_AIV {
        return;
    }

    InitMmadParams<Intf>(self);
    if constexpr((std::is_same<typename Intf::SrcT, bfloat16_t>::value) || (std::is_same<typename Intf::SrcT, half>::value)) {
        if (unlikely(self->ctx.curNL0Idx_ == 0 || self->ctx.curNL0Idx_ + 1 == self->ctx.nIter_)) {
            UpdateLoadToB2ParamsN<Intf>(self);
        }
    }

    bool isFirstDk = true;
    bool a1PingPongFlag = true;
    bool b1PingPongFlag = true;
    LocalTensor<typename Intf::SrcT> l0a;
    LocalTensor<typename Intf::SrcT> l0b;
    LocalTensor<typename Intf::L0cT> l0c;
    uint8_t l0aPingPongFlag = self->ctx.l0aPingPongFlag_;
    constexpr uint32_t l0aPingPongAddr = TOTAL_L0A_SIZE / 2 / sizeof(typename Intf::SrcT);
    constexpr uint32_t l0bPingPongAddr = TOTAL_L0B_SIZE / 2 / sizeof(typename Intf::SrcT);

    if (self->ctx.usingCacheC1Ping_) {
        l0c = self->ctx.c1Ping_.template AllocTensor<typename Intf::L0cT>();
    } else {
        l0c = self->ctx.c1Pong_.template AllocTensor<typename Intf::L0cT>();
    }
    self->ctx.needComputeKFlag_ = false;
    for (uint64_t curKdIdx = 0; curKdIdx < self->ctx.tiling_->dk; curKdIdx++) {
        int64_t dTmp = 0;
        if (unlikely(self->ctx.tiling_->strideD > self->ctx.tiling_->dk)) {
            dTmp = self->ctx.curDinIdx_ + self->ctx.tiling_->padFront;
            if (CalcRemainder(dTmp, self->ctx.tiling_->strideD) >= self->ctx.tiling_->dk ||
                dTmp / self->ctx.tiling_->strideD >= self->ctx.tiling_->dout) {
                continue;
            }
        } else {
            // 由于膨胀卷积使dk的位置发生改变，求解dout_idx时，dk_idx需乘上膨胀系数再参与计算，才能求取正确的索引
            dTmp = self->ctx.curDinIdx_ + self->ctx.tiling_->padFront - curKdIdx * self->ctx.tiling_->dilationD;
            if (dTmp < 0 || CalcRemainder(dTmp, self->ctx.tiling_->strideD) > 0 ||
                dTmp >= self->ctx.tiling_->dout * self->ctx.tiling_->strideD) {
                continue;
            }
        }
        self->ctx.needComputeKFlag_ = true;
        int64_t curDoutIdx = dTmp;
        if (self->ctx.tiling_->strideD > 1) {
            curDoutIdx = dTmp / self->ctx.tiling_->strideD;
        }

        uint32_t kaIdx = 0;
        uint32_t kbIdx = 0;
        uint64_t kaStepIdx = 0;
        uint64_t kbStepIdx = 0;
        self->ctx.load3d_.kExtension = self->ctx.tiling_->baseK;
        self->ctx.mmad_.k = self->ctx.tiling_->baseK;
        self->ctx.curLoadKal1_ = self->ctx.tiling_->stepKa * self->ctx.tiling_->baseK;
        self->ctx.curLoadKbl1_ = self->ctx.tiling_->stepKb * self->ctx.tiling_->baseK;
        for (uint64_t kIdx = 0; kIdx < self->ctx.kIter_; kIdx++) {
            bool isLastKIdx = (kIdx + 1 == self->ctx.kIter_);
            if (isLastKIdx) {
                self->ctx.load3d_.kExtension = self->ctx.tailK_;
                self->ctx.mmad_.k = self->ctx.tailK_;
            }
            if (kIdx == self->ctx.kIterStepKaTail) {
                self->ctx.curLoadKal1_ = (self->ctx.stepKaTail - 1) * self->ctx.tiling_->baseK + self->ctx.tailK_;
            }
            if (kIdx == self->ctx.kIterStepKbTail) {
                self->ctx.curLoadKbl1_ = (self->ctx.stepKbTail - 1) * self->ctx.tiling_->baseK + self->ctx.tailK_;
            }
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
            bool isLoadA1 = kaIdx == 0;
            if (isLoadA1 && self->ctx.tiling_->al1Pbuffer > 1) {
                // 此处默认stepM = 1
                a1PingPongFlag = ((self->ctx.curML1Idx_ * self->ctx.stepKaRound_ + kaStepIdx + 1) & 1);
            }
            Convolution3DBackpropFunc::LoadToA1<Intf, typename Intf::SrcT>(self, kIdx, curDoutIdx, a1PingPongFlag,
                                                                           isLoadA1);

            bool isLoadB1 = kbIdx == 0;
            if (isLoadB1 && self->ctx.tiling_->bl1Pbuffer > 1) {
                // 此处默认stepN = 1
                b1PingPongFlag = ((self->ctx.curNL1Idx_ * self->ctx.stepKbRound_ + kbStepIdx + 1) & 1);
            }
            Convolution3DBackpropFunc::LoadToB1<Intf, typename Intf::SrcT>(self, kIdx, curKdIdx, b1PingPongFlag,
                                                                           isLoadB1);

            l0a = self->ctx.l0aBuf_.template Get<typename Intf::SrcT>();
            if (l0aPingPongFlag) {
                l0a = l0a[l0aPingPongAddr];
            }

            self->ctx.load3d_.kStartPt = kaIdx * self->ctx.tiling_->baseK;

            if (unlikely(isLoadA1)) {
                if (a1PingPongFlag) {
                    self->ctx.cacheA1BufPing_ = self->ctx.a1Ping_.template DeQue<typename Intf::SrcT>();
                } else {
                    self->ctx.cacheA1BufPong_ = self->ctx.a1Pong_.template DeQue<typename Intf::SrcT>();
                }
            }

            WaitFlag<HardEvent::M_MTE1>(l0aPingPongFlag);
            if (a1PingPongFlag) {
                LoadToA2<Intf>(self, self->ctx.cacheA1BufPing_, l0a);
            } else {
                LoadToA2<Intf>(self, self->ctx.cacheA1BufPong_, l0a);
            }

            bool isLastStepKa = kaIdx + 1 == self->ctx.tiling_->stepKa;
            if (isLastStepKa || isLastKIdx) {
                if (a1PingPongFlag) {
                    self->ctx.a1Ping_.FreeTensor(self->ctx.cacheA1BufPing_);
                } else {
                    self->ctx.a1Pong_.FreeTensor(self->ctx.cacheA1BufPong_);
                }
            }

            l0b = self->ctx.l0bBuf_.template Get<typename Intf::SrcT>();
            if (l0aPingPongFlag) {
                l0b = l0b[l0bPingPongAddr];
            }

            if (unlikely(isLoadB1 && (!self->ctx.isB1FullLoadFlag_ || (self->ctx.isB1FullLoadFlag_ &&
                self->ctx.isLoadB1_)))) {
                if (b1PingPongFlag) {
                    self->ctx.cacheB1BufPing_ = self->ctx.b1Ping_.template DeQue<typename Intf::SrcT>();
                } else {
                    self->ctx.cacheB1BufPong_ = self->ctx.b1Pong_.template DeQue<typename Intf::SrcT>();
                }
                if (self->ctx.isLoadB1_) {
                    self->ctx.isLoadB1_ = false;
                }
            }

            if constexpr((std::is_same<typename Intf::SrcT, bfloat16_t>::value) || (std::is_same<typename Intf::SrcT, half>::value)) {
                if (unlikely(kIdx == 0 || kIdx == self->ctx.kIterStepKbTail)) {
                    UpdateLoadToB2ParamsK<Intf>(self);
                }
            }
            if (b1PingPongFlag) {
                LoadToB2<Intf>(self, self->ctx.cacheB1BufPing_, kbIdx, kIdx, b1PingPongFlag, l0b);
            } else {
                LoadToB2<Intf>(self, self->ctx.cacheB1BufPong_, kbIdx, kIdx, b1PingPongFlag, l0b);
            }

            bool isLastStepKb = kbIdx + 1 == self->ctx.tiling_->stepKb;
            if ((isLastStepKb || isLastKIdx) && (!self->ctx.isB1FullLoadFlag_ ||
                (self->ctx.isB1FullLoadFlag_ && self->ctx.isFreeB1_))) {
                if (b1PingPongFlag) {
                    self->ctx.b1Ping_.FreeTensor(self->ctx.cacheB1BufPing_);
                } else {
                    self->ctx.b1Pong_.FreeTensor(self->ctx.cacheB1BufPong_);
                }
            }

            SetFlag<HardEvent::MTE1_M>(l0aPingPongFlag);
            WaitFlag<HardEvent::MTE1_M>(l0aPingPongFlag);

            MmadLocal<Intf>(self, l0a, l0b, l0c);
            SetFlag<HardEvent::M_MTE1>(l0aPingPongFlag);
            if (unlikely(isFirstDk && kIdx == 0)) {
                self->ctx.mmad_.cmatrixInitVal = 0;
            }

            l0aPingPongFlag ^= self->ctx.useL0PingPong_;
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
        isFirstDk = false;
    }
    if (self->ctx.usingCacheC1Ping_) {
        self->ctx.c1Ping_.EnQue(l0c);
    } else {
        self->ctx.c1Pong_.EnQue(l0c);
    }
    self->ctx.l0aPingPongFlag_ = l0aPingPongFlag;

    if constexpr (Intf::conv3dConfig.enableKernelSplit) {
        SetFlag<HardEvent::M_FIX>(EVENT_ID2);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID2);
        SetFlag<HardEvent::FIX_M>(EVENT_ID2);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID2);
    }
}

template <class Intf>
static __aicore__ inline void UpdateIdxAndStep(Intf *self)
{
    self->ctx.curML0Idx_ = self->ctx.curML1Idx_;
    self->ctx.curNL0Idx_ = self->ctx.curNL1Idx_;
    // 当前放大后Ho绝对坐标
    if constexpr (Intf::conv3dConfig.enableKernelSplit) {
        self->ctx.curHoIdx_ =
            self->ctx.curHoStartIdx_ + self->ctx.curML0Idx_ * self->ctx.tiling_->baseM / self->ctx.splitWi_ *
            self->ctx.tiling_->strideH;
    } else {
        self->ctx.curHoIdx_ =
            self->ctx.curHoStartIdx_ + self->ctx.curML0Idx_ * self->ctx.tiling_->baseM / self->ctx.tiling_->wi;
    }

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
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf *self, const TConv3DInputV2Tiling *__restrict tiling)
    {
        self->ctx.tiling_ = tiling;
        SetHF32Mode(self->ctx.tiling_->hf32Flag);
        CheckTiling<Intf>(self);
        InitParams<Intf>(self);
        InitTque<Intf>(self);
    }
};

template <class Intf>
struct SetWeight {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf *self, const GlobalTensor<typename Intf::SrcT> &weight)
    {
        self->ctx.weightGlobal_ = weight;
    }
};

template <class Intf>
struct SetOutBackprop {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf *self, const GlobalTensor<typename Intf::SrcT> &outBackprop)
    {
        self->ctx.outBackPropGlobal_ = outBackprop;
    }
};

template <class Intf>
struct SetSingleShape {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf *self, uint64_t singleShapeM, uint64_t singleShapeK, uint32_t singleShapeN,
                                       uint32_t singleShapeD)
    {
        self->ctx.singleShapeDin_ = singleShapeD;
        self->ctx.singleShapeM_ = singleShapeM;
        self->ctx.singleShapeCin1_ = (singleShapeN + self->ctx.tiling_->c0 -1) >> self->ctx.tiling_->c0Bits;
        self->ctx.singleShapeCout1_ = singleShapeK / self->ctx.DkHkWkC0_;
        if constexpr (Intf::conv3dConfig.enableKernelSplit) {
            self->ctx.splitSingleShapeM_ = singleShapeM / (self->ctx.tiling_->strideH * self->ctx.tiling_->strideW);
        }
        self->ctx.singleShapeHin_ = singleShapeM / self->ctx.tiling_->wi;
        self->ctx.singleShapeCin_ = singleShapeN;
        InitStepMParams<Intf>(self);
        InitStepKParams<Intf>(self);
        InitStepNParams<Intf>(self);

        self->ctx.kIterStepKaTail = (Ceil(self->ctx.kIter_, self->ctx.tiling_->stepKa) - 1) * self->ctx.tiling_->stepKa;
        self->ctx.kIterStepKbTail = (Ceil(self->ctx.kIter_, self->ctx.tiling_->stepKb) - 1) * self->ctx.tiling_->stepKb;
        self->ctx.stepKaTail = self->ctx.kIter_ - self->ctx.kIterStepKaTail;
        self->ctx.stepKbTail = self->ctx.kIter_ - self->ctx.kIterStepKbTail;
    }
};

template <class Intf>
struct SetStartIdx {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf *self, uint32_t curDinStartIdx, int32_t curHoStartIdx)
    {
        // 需要增加入参的startDinIdx判断，否则无法确定padding和补0问题
        // 增加入参，传参判断
        self->ctx.curDinStartIdx_ = curDinStartIdx;
        self->ctx.curHoStartIdx_ = curHoStartIdx;
    }
};

template <class Intf>
static __aicore__ inline void JudgeIterateSkip(Intf *self)
{
    self->ctx.needComputeFlag_ = true;
    UpdateCurHoSize<Intf>(self);

    // 使用needComputeFlag_来标记是否需要跳过curML0Idx、curNL0Idx、curDinIdx的计算
    uint32_t hDstDataSkipLine = CalcHDstDataSkipLine(self);

    if (self->ctx.curHoSize_ <= hDstDataSkipLine && self->ctx.tiling_->initOutputFlag == 1) {
        // 在跳过计算逻辑后，如果存在部分无需跳过的额外操作，则在compute逻辑中统一处理。当前已有的操作为isFreeB1
        self->ctx.needComputeFlag_ = false;        
    }
}

template <class Intf, bool sync>
struct Iterate {
    // 一次iterate计算（baseM, baseN, baseD)，当前baseD=1
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
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

        order_N表示L1上驻留B循环A，顺序为L1A_ping * L1B_ping, L1A_pong * L1B_ping，L1A_ping * L1B_pong，L1A_pong *
        L1B_pong L0上也是驻留B，循环A order_N: L0A1*L0B1, L0A2*L0B1, L0A3*L0B1, L0A1*L0B2 …………
        L0A3*L0B3，L0A4*L0B1，L0A5*L0B1 …… L0A6*L0B6 order_M: L0A1*L0B1, L0A1*L0B2, L0A1*L0B3, L0A2*L0B1 …………
        L0A3*L0B3，L0A1*L0B4，L0A1*L0B5 …… L0A6*L0B6
        */
        // 更新idx，用L1、L1step、L0三个指针控制走位和计算offset，表示计算第几个mL0 * baseN
        if (unlikely(self->ctx.isFirstIter_)) {
            self->ctx.curML0Idx_ = 0;
            self->ctx.curNL0Idx_ = 0;
            self->ctx.curML1Idx_ = 0;
            self->ctx.curNL1Idx_ = 0;
            self->ctx.curDinIdx_ = self->ctx.curDinStartIdx_;
            self->ctx.curHoIdx_ = self->ctx.curHoStartIdx_;
            if constexpr(std::is_same<typename Intf::SrcT, float>::value) {
                self->ctx.curPingCoutIdx_ = 0;
                self->ctx.curPongCoutIdx_ = 0;
            }
            self->ctx.isFirstIter_ = false;
            self->ctx.isLoadB1_ = true;
            self->ctx.isFreeB1_ = false;
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
                        self->ctx.curML1Idx_ = 0;
                        self->ctx.curNL1Idx_ = 0;
                        if (++self->ctx.curDinIdx_ >= self->ctx.curDinStartIdx_ + self->ctx.singleShapeDin_) {
                            return false;
                        }
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
                        self->ctx.curML1Idx_ = 0;
                        self->ctx.curNL1Idx_ = 0;
                        if (++self->ctx.curDinIdx_ >= self->ctx.curDinStartIdx_ + self->ctx.singleShapeDin_) {
                            return false;
                        }
                    }
                    if (self->ctx.curNL1Idx_ >= self->ctx.nIter_) {
                        self->ctx.curNL1Idx_ = 0;
                        self->ctx.curML1Idx_ += self->ctx.curStepM_;
                    }
                    UpdateIdxAndStep<Intf>(self);
                }
            }
        }
        self->ctx.isFreeB1_ = self->ctx.isB1FullLoadFlag_ && (self->ctx.curML0Idx_ == self->ctx.mIter_ - 1) &&
            (self->ctx.curNL0Idx_ == self->ctx.nIter_ - 1) &&
            (self->ctx.curDinIdx_ == self->ctx.curDinStartIdx_ + self->ctx.singleShapeDin_ - 1);
        if (self->ctx.curML0Idx_ + 1 == self->ctx.mIter_) {
            self->ctx.baseUseM_ = self->ctx.tailM_;
        } else if (self->ctx.curML0Idx_ == 0) {
            self->ctx.baseUseM_ = self->ctx.tiling_->baseM;
        }
        if (self->ctx.curNL0Idx_ + 1 == self->ctx.nIter_) {
            self->ctx.baseUseN_ = self->ctx.tailN_;
        } else if (self->ctx.curNL0Idx_ == 0) {
            self->ctx.baseUseN_ = self->ctx.tiling_->baseN;
        }
        if constexpr (std::is_same<typename Intf::DstT, float>::value) {
            // baseN可能非16对齐，为8对齐，此时L0B的仍然按照512B对齐的地址偏移计算
            self->ctx.baseUseAlignN_ = (self->ctx.baseUseN_ + BLOCK_CUBE - 1) / BLOCK_CUBE * BLOCK_CUBE;
        }
        JudgeIterateSkip<Intf>(self);
        Compute<Intf>(self);
        return true;
    }
};

template <class Intf, bool sync>
struct IterateAll {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
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
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
                                       uint8_t enAtomic = 0, bool enSequentialWrite = false)
    {
        LoadL0c2Gm<Intf>(self, output, enAtomic, enSequentialWrite);
    }
};

template <class Intf>
struct End {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
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
        self->ctx.c1Ping_.FreeAllEvent();
        if (self->ctx.tiling_->cl0Pbuffer > 1) {
            self->ctx.c1Pong_.FreeAllEvent();
        }
        if (self->ctx.tiling_->hf32Flag) {
            SetHF32Mode(false);
        }
    }
};
}  // namespace Convolution3DBackpropFunc
#endif
