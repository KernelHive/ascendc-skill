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
 * \file conv3d_common_init_func.h
 * \brief
 */

#ifndef CONV3D_COMMON_INIT_FUNC_H
#define CONV3D_COMMON_INIT_FUNC_H

#include "conv_framework_util.h"
#include "conv3d_common_sub_api.h"
#include "conv3d_config.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"

namespace Conv3dFunc {

template <class Intf>
__aicore__ inline void InitKDirectionBaseValue(
    Intf *self, uint64_t updateKAL1 = 0, uint64_t updateKBL1 = 0, uint64_t updateKL0 = 0)
{
    // K方向变量计算
    uint64_t currentKAL1 = updateKAL1 == 0 ? self->ctx.conv3dTiling->kAL1 : updateKAL1;
    uint64_t currentKBL1 = updateKBL1 == 0 ? self->ctx.conv3dTiling->kBL1 : updateKBL1;

    self->ctx.cin1 = CeilDIV(self->ctx.singleCoreCin, self->ctx.cin0);
    if constexpr (!Intf::bl1bypass && Intf::groupConvType) {
        self->ctx.singleCoreKL0 = updateKL0 == 0 ? self->ctx.conv3dTiling->kL0 : updateKL0;
    } else {
        self->ctx.singleCoreKL0 = self->ctx.conv3dTiling->kL0;
    }
    
    uint64_t alignCinKhKwKd =
        AlignB(self->ctx.singleCoreCin, GetInputkInOneC0Block<Intf>()) * self->ctx.kernelHxkernelWxkernelD;
    self->ctx.maxKAL1Iter = CeilDIV(alignCinKhKwKd, currentKAL1) - 1;
    self->ctx.ddr2l0LoopK = CeilDIV(alignCinKhKwKd, self->ctx.singleCoreKL0);
    self->ctx.maxKL0Iter = self->ctx.ddr2l0LoopK - 1;
    if constexpr (!Intf::bl1bypass) {
        self->ctx.maxKBL1Iter = CeilDIV(alignCinKhKwKd, currentKBL1) - 1;
        self->ctx.multiKBL1 = CeilDIV(currentKBL1, self->ctx.singleCoreKL0);
        self->ctx.kBL1fullload = (alignCinKhKwKd == currentKBL1);
        if (Intf::groupConvType) {
            self->ctx.groupKBL1Tail = alignCinKhKwKd % currentKBL1;
            self->ctx.groupKBL1Tail = self->ctx.groupKBL1Tail == 0 ? currentKBL1 : self->ctx.groupKBL1Tail;
        }
    } else {
        self->ctx.multiKBL1 = self->ctx.ddr2l0LoopK;
    }

    if constexpr (Intf::formatType == conv::ConvFormat::NCDHW) {
        self->ctx.kL0Tail = self->ctx.singleCoreCin % self->ctx.conv3dTiling->kL0;
    } else {
        self->ctx.kL0Tail = alignCinKhKwKd % self->ctx.singleCoreKL0;
    }
    
    self->ctx.kL0Tail = self->ctx.kL0Tail == 0 ? self->ctx.singleCoreKL0 : self->ctx.kL0Tail;
    self->ctx.multiKAL1 = CeilDIV(currentKAL1, self->ctx.singleCoreKL0);
    self->ctx.kAL1fullload = alignCinKhKwKd == currentKAL1;

    if (Intf::groupConvType) {
        self->ctx.groupKAL1Tail = alignCinKhKwKd % currentKAL1;
        self->ctx.groupKAL1Tail = self->ctx.groupKAL1Tail == 0 ? currentKAL1 : self->ctx.groupKAL1Tail;
        self->ctx.groupKAL1 = currentKAL1;
        self->ctx.groupKBL1 = currentKBL1;
    }
}

template <class Intf>
__aicore__ inline void InitMDirectionBaseValue(Intf *self)
{
    // M方向变量计算
    self->ctx.mAL1Tail = self->ctx.singleCoreM % self->ctx.conv3dTiling->mAL1;
    self->ctx.mAL1Tail = self->ctx.mAL1Tail == 0 ? self->ctx.conv3dTiling->mAL1 : self->ctx.mAL1Tail;
    self->ctx.ddr2l1LoopM = CeilDIV(self->ctx.singleCoreM, self->ctx.conv3dTiling->mAL1);
    self->ctx.maxMAL1Iter = self->ctx.ddr2l1LoopM - 1;
    self->ctx.mAL0Tail = self->ctx.mAL1Tail % self->ctx.conv3dTiling->mL0;
    self->ctx.mAL0Tail = self->ctx.mAL0Tail == 0 ? self->ctx.conv3dTiling->mL0 : self->ctx.mAL0Tail;
    self->ctx.l12l0LoopM = self->ctx.conv3dTiling->mAL1DivmL0;
    self->ctx.maxML0Iter = self->ctx.l12l0LoopM - 1;
}

template <class Intf>
__aicore__ inline void InitCoutDirectionBaseValue(Intf *self)
{
    // weight by pass
    if constexpr (Intf::bl1bypass) {
        self->ctx.nL0Tail = self->ctx.singleCoreCo % self->ctx.conv3dTiling->nL0;
        self->ctx.nL0Tail = self->ctx.nL0Tail == 0 ? self->ctx.conv3dTiling->nL0 : self->ctx.nL0Tail;
        self->ctx.maxNBL1Iter = 0;
        self->ctx.ddr2l1LoopN = 1;
        self->ctx.l12l0LoopN = CeilDIV(self->ctx.singleCoreCo, self->ctx.conv3dTiling->nL0);
        self->ctx.maxNL0Iter = self->ctx.l12l0LoopN - 1;
        return;
    }
    // Cout方向变量计算
    self->ctx.maxNBL1Iter = CeilDIV(self->ctx.singleCoreCo, self->ctx.conv3dTiling->nBL1) - 1;
    self->ctx.nBL1Tail = self->ctx.singleCoreCo % self->ctx.conv3dTiling->nBL1;
    self->ctx.nBL1Tail = self->ctx.nBL1Tail == 0 ? self->ctx.conv3dTiling->nBL1 : self->ctx.nBL1Tail;
    self->ctx.nBL1TailAlign = AlignB(self->ctx.nBL1Tail, BLOCK_L0_N);
    self->ctx.nL0Tail = self->ctx.nBL1Tail % self->ctx.conv3dTiling->nL0;
    self->ctx.nL0Tail = self->ctx.nL0Tail == 0 ? self->ctx.conv3dTiling->nL0 : self->ctx.nL0Tail;
    self->ctx.ddr2l1LoopN = self->ctx.maxNBL1Iter + 1;
    self->ctx.l12l0LoopN = self->ctx.conv3dTiling->nBL1DivnL0;
    self->ctx.maxNL0Iter = self->ctx.l12l0LoopN - 1;
}

template <class Intf>
__aicore__ inline void InitDoutDirectionBaseValue(Intf *self)
{
    // Do方向变量计算
    self->ctx.ddr2l1LoopD = self->ctx.singleCoreDo;
}

template <class Intf>
__aicore__ inline void InitGroupOptDirectionValue(Intf *self)
{
    // GroupOpt方向变量计算
    self->ctx.maxGroupOptIter = self->ctx.singleCoreGroupOpt;
}

template <class Intf>
__aicore__ inline void InitHoutDirectionValue(Intf *self)
{
    self->ctx.ddr2l1LoopHo = self->ctx.singleCoreHo;
}

template <class Intf, uint32_t ImplType>
struct Init {
    static __aicore__ inline bool call(Intf *self, const void *__restrict tiling)
    {
        InitParams(self, tiling);
        if ASCEND_IS_AIC { 
            if constexpr (Intf::formatType == conv::ConvFormat::NCDHW) {
                InitBufferPointWise(self);
            } else {
                InitBuffer(self);
            }
        }
        if ASCEND_IS_AIV { 
            if constexpr (Intf::quantType == static_cast<int8_t>(QuantType::PER_CHANNEL_NO_OFFSET)) {
                InitVecBuffer(self);
            }
        }

        if constexpr (Intf::outputOrder) {
            InitHoutDirectionValue(self);
        }
        InitKDirectionBaseValue(self);
        InitMDirectionBaseValue(self);
        InitCoutDirectionBaseValue(self);
        InitDoutDirectionBaseValue(self);
        InitGroupOptDirectionValue(self);
        InitHf32Mode(self);
        // preload case flag set: pbBL1, pbAL1 bit
        uint16_t preloadAL1Flag = (self->ctx.conv3dTiling->pBufferFlag & 0x18) >> 3;
        uint8_t L0ASet2dFlag = self->ctx.padHead != 0 || self->ctx.padTail != 0 ||
                self->ctx.padUp >= self->ctx.kernelH || self->ctx.padDown >= self->ctx.kernelH;
        bool kAL1TailCase = self->ctx.conv3dTiling->kAL1Tail != self->ctx.conv3dTiling->kAL1;
        self->ctx.preloadAL1DbFlag =
            preloadAL1Flag == 1 && !self->ctx.kAL1fullload && !L0ASet2dFlag && !kAL1TailCase;
        self->ctx.preloadABL1DbFlag = preloadAL1Flag == 3 && !self->ctx.kAL1fullload &&
                                    !self->ctx.kBL1fullload && !L0ASet2dFlag && !kAL1TailCase;

        self->ctx.totalBlockCount = self->ctx.conv3dTiling->nUB / BLOCK_L0_N;
        self->ctx.totalBlockLen = self->ctx.conv3dTiling->mUB * BLOCK_L0_N * sizeof(typename Intf::L0cT) / 32;                            
        return false;
    }

    static __aicore__ inline void InitParams(Intf *self, const void *__restrict tiling)
    {
        self->ctx.conv3dTiling = (TConv3DTiling *)tiling;
        self->ctx.kernelH = self->ctx.conv3dTiling->kernelH;
        self->ctx.kernelW = self->ctx.conv3dTiling->kernelW;
        self->ctx.kernelD = self->ctx.conv3dTiling->kernelD;
        self->ctx.dilatedKernelH = 1 + (self->ctx.kernelH - 1) * self->ctx.conv3dTiling->dilationH;
        self->ctx.dilatedKernelW = 1 + (self->ctx.kernelW - 1) * self->ctx.conv3dTiling->dilationW;
        self->ctx.dilatedKernelD = 1 + (self->ctx.kernelD - 1) * self->ctx.conv3dTiling->dilationD;
        self->ctx.kernelHxkernelW = self->ctx.kernelH * self->ctx.kernelW;
        self->ctx.kernelHxkernelWxkernelD = self->ctx.kernelHxkernelW * self->ctx.kernelD;
        self->ctx.orgHi = self->ctx.conv3dTiling->orgHi;
        self->ctx.orgWi = self->ctx.conv3dTiling->orgWi;
        self->ctx.orgDi = self->ctx.conv3dTiling->orgDi;
        self->ctx.orgHo = self->ctx.conv3dTiling->orgHo;
        self->ctx.orgWo = self->ctx.conv3dTiling->orgWo;
        self->ctx.orgCo = self->ctx.conv3dTiling->coutOpt;
        self->ctx.orgCi = self->ctx.conv3dTiling->cinOpt;
        self->ctx.orgDo = self->ctx.conv3dTiling->orgDo;
        self->ctx.singleCoreCo = self->ctx.conv3dTiling->singleCoreCo;
        self->ctx.singleCoreDo = self->ctx.conv3dTiling->singleCoreDo;
        self->ctx.singleCoreM = self->ctx.conv3dTiling->singleCoreM;
        if constexpr (Intf::outputOrder) {
            self->ctx.singleCoreM = self->ctx.conv3dTiling->orgWo;
            self->ctx.singleCoreHo = self->ctx.conv3dTiling->singleCoreM;
        }
        self->ctx.singleCoreCin = self->ctx.conv3dTiling->cinOpt;
        self->ctx.singleCoreGroupOpt = self->ctx.conv3dTiling->singleCoreGroupOpt;
        self->ctx.strideH = self->ctx.conv3dTiling->strideH;
        self->ctx.strideW = self->ctx.conv3dTiling->strideW;
        self->ctx.strideD = self->ctx.conv3dTiling->strideD;
        self->ctx.dilationH = self->ctx.conv3dTiling->dilationH;
        self->ctx.dilationW = self->ctx.conv3dTiling->dilationW;
        self->ctx.dilationD = self->ctx.conv3dTiling->dilationD;
        self->ctx.padHead = self->ctx.conv3dTiling->padHead;
        self->ctx.padTail = self->ctx.conv3dTiling->padTail;
        self->ctx.padUp = self->ctx.conv3dTiling->padUp;
        self->ctx.padDown = self->ctx.conv3dTiling->padDown;
        self->ctx.padLeft = self->ctx.conv3dTiling->padLeft;
        self->ctx.padRight = self->ctx.conv3dTiling->padRight;
        self->ctx.biasFullLoadFlag = self->ctx.conv3dTiling->biasFullLoadFlag;
        self->ctx.sizeOfFmap = sizeof(typename Intf::FmapT);
        self->ctx.sizeOfWeight = sizeof(typename Intf::WeightT);
        self->ctx.sizeOfBias = sizeof(typename Intf::BiasT);
        self->ctx.sizeOfL0c = sizeof(typename Intf::L0cT);
        self->ctx.sizeOfDst = sizeof(typename Intf::OutputT);
        self->ctx.cin0 = C0_SIZE / self->ctx.sizeOfFmap;
        self->ctx.cout0 = C0_SIZE / self->ctx.sizeOfDst;
        self->ctx.orgCoAlignK0 = AlignB(self->ctx.orgCo, self->ctx.cout0);
        self->ctx.orgCoAlignN0 = AlignB(self->ctx.orgCo, BLOCK_L0_N);
    }

    static __aicore__ inline void InitBuffer(Intf *self)
    {
        uint64_t cl0Spacesize = self->ctx.conv3dTiling->mL0 * self->ctx.conv3dTiling->nL0;
        uint64_t bl1Spacesize = self->ctx.conv3dTiling->nBL1 * self->ctx.conv3dTiling->kBL1;
        uint64_t biasl1Spacesize =
            self->ctx.conv3dTiling->nL0 * sizeof(typename Intf::BiasT);  // 非全载时bias伴随L0C切分
        if (self->ctx.biasFullLoadFlag) {
            biasl1Spacesize = AlignB(
                self->ctx.singleCoreCo * sizeof(typename Intf::BiasT), BLOCK_L0_N * sizeof(typename Intf::BiasT));
        }
        uint64_t biasBTSpacesize = self->ctx.conv3dTiling->nL0;

        uint64_t hoAL1Max = ((self->ctx.conv3dTiling->mAL1 < self->ctx.singleCoreM ? self->ctx.conv3dTiling->mAL1
                                                                                   : self->ctx.singleCoreM) /
                                self->ctx.orgWo) +
                            2;
        uint64_t hiAL1Max = (hoAL1Max - 1) * self->ctx.strideH + self->ctx.dilatedKernelH;
        uint64_t win = self->ctx.orgWi;
        if constexpr (Intf::outputOrder) {
            hiAL1Max = self->ctx.dilatedKernelH;
            win = (self->ctx.conv3dTiling->mAL1 - 1) * self->ctx.strideW + self->ctx.dilatedKernelW;
            win = win > self->ctx.orgWi ? self->ctx.orgWi : win;
        }
        hiAL1Max = hiAL1Max > self->ctx.orgHi ? self->ctx.orgHi : hiAL1Max;
        uint64_t al1Spacesize = self->ctx.conv3dTiling->cin1InAL1 * self->ctx.cin0 * hiAL1Max * win;
        InitBufferWithDoubleBuf(self, cl0Spacesize, al1Spacesize, bl1Spacesize);
        self->ctx.pipe.InitBuffer(self->ctx.queueBiasL1, 1, AlignB(biasl1Spacesize, C0_SIZE));
        self->ctx.pipe.InitBuffer(self->ctx.queueBiasBT, 1, AlignB(biasBTSpacesize * self->ctx.sizeOfL0c, BT_SIZE));
    }

    static __aicore__ inline void InitBufferPointWise(Intf *self)
    {
        uint64_t cl0Spacesize = self->ctx.conv3dTiling->mL0 * self->ctx.conv3dTiling->nL0;
        uint64_t bl1Spacesize = self->ctx.conv3dTiling->nBL1 * self->ctx.conv3dTiling->kBL1;
        uint64_t biasl1Spacesize =
            self->ctx.conv3dTiling->nL0 * sizeof(typename Intf::BiasT);  // 非全载时bias伴随L0C切分
        if (self->ctx.biasFullLoadFlag) {
            biasl1Spacesize = AlignB(
                self->ctx.singleCoreCo * sizeof(typename Intf::BiasT), BLOCK_L0_N * sizeof(typename Intf::BiasT));
        }
        biasl1Spacesize = biasl1Spacesize * K0_BIAS;
        uint64_t al1Spacesize = (self->ctx.conv3dTiling->mAL1 < self->ctx.singleCoreM ? self->ctx.conv3dTiling->mAL1
                                    : AlignB(self->ctx.singleCoreM, DATA_COPY_OP_LEN))
                                    * AlignB(self->ctx.conv3dTiling->kAL1, DATA_COPY_OP_LEN);
        InitBufferWithDoubleBuf(self, cl0Spacesize, al1Spacesize, bl1Spacesize);
        self->ctx.pipe.InitBuffer(self->ctx.queueBiasL1, 1, AlignB(biasl1Spacesize, C0_SIZE));
    }

    static __aicore__ inline void InitHf32Mode(Intf *self) 
    {
        SetHF32Mode(self->ctx.conv3dTiling->hf32Enable);
        SetHF32TransMode(false);
        ASC_OP_LOGD("[InitHf32Mode] hf32Mode: %d, hf32TransMode: %d \n", self->ctx.conv3dTiling->hf32Enable, self->ctx.conv3dTiling->hf32TransMode);
    }

    static __aicore__ inline void InitVecBuffer(Intf *self)
    {
        uint64_t ubinSpacesize = self->ctx.conv3dTiling->mUB * self->ctx.conv3dTiling->nUB;
        self->ctx.pipe.InitBuffer(self->ctx.queueUBin, 1, ubinSpacesize * self->ctx.sizeOfL0c);
        self->ctx.pipe.InitBuffer(self->ctx.queueUBout, 1, ubinSpacesize * self->ctx.sizeOfDst);
        
        uint64_t channelSize = 0;
        if constexpr (Intf::loadChannelType == static_cast<int8_t>(LoadChannelType::NORMAL)) {
            channelSize = self->ctx.conv3dTiling->nUB;
        }
        if constexpr (Intf::loadChannelType == static_cast<int8_t>(LoadChannelType::LOAD_TOTAL_LC0)) {
            channelSize = self->ctx.conv3dTiling->nL0;
        }
        if constexpr (Intf::loadChannelType == static_cast<int8_t>(LoadChannelType::LOAD_TOTAL_CORE)) {
            channelSize = AlignB(self->ctx.singleCoreCo, BLOCK_L0_N);
        }

        self->ctx.pipe.InitBuffer(self->ctx.queueUBbias, 1, channelSize * sizeof(typename Intf::BiasT));
        self->ctx.pipe.InitBuffer(self->ctx.queueUBscale, 1, channelSize * sizeof(typename Intf::FP32T));
        if constexpr (IsSameType<typename Intf::BiasT, bfloat16_t>::value || IsSameType<typename Intf::BiasT, half>::value) {
            self->ctx.pipe.InitBuffer(self->ctx.fp32BiasBuf, channelSize * sizeof(typename Intf::FP32T));
        }
    }

    static __aicore__ inline void InitBufferWithDoubleBuf(Intf *self, uint64_t cl0Spacesize, uint64_t al1Spacesize, uint64_t bl1Spacesize)
    {
        int8_t cl0db = (self->ctx.conv3dTiling->pBufferFlag & 0x04) >> 2;
        int8_t al1db = (self->ctx.conv3dTiling->pBufferFlag & 0x08) >> 3;
        int8_t bl1db = (self->ctx.conv3dTiling->pBufferFlag & 0x10) >> 4;

        self->ctx.pipe.InitBuffer(self->ctx.l0aBuf, conv3d::L0A_SIZE);
        self->ctx.pipe.InitBuffer(self->ctx.l0bBuf, conv3d::L0B_SIZE);

        self->ctx.al0Ping = self->ctx.l0aBuf.template Get<typename Intf::FmapT>();
        self->ctx.al0Pong = self->ctx.al0Ping[(L0A_SIZE / 2) / sizeof(typename Intf::FmapT)];
        self->ctx.bl0Ping = self->ctx.l0bBuf.template Get<typename Intf::WeightT>();
        self->ctx.bl0Pong = self->ctx.bl0Ping[(L0B_SIZE / 2) / sizeof(typename Intf::WeightT)];

        if constexpr (Intf::formatType == conv::ConvFormat::NCDHW) {
            // l0bBuf actual point to l0a memory, al0BiasB need to point to l0a, so al0BiasB -> l0bBuf;
            // as well as bl0BiasB.
            self->ctx.al0BiasB = self->ctx.l0bBuf.template Get<typename Intf::BiasT>();
            self->ctx.bl0BiasB = self->ctx.l0aBuf.template Get<typename Intf::BiasT>();
        }

        if (!cl0db) {
            self->ctx.pipe.InitBuffer(self->ctx.queueCL0, 1, cl0Spacesize * self->ctx.sizeOfL0c);
        } else {
            self->ctx.pipe.InitBuffer(self->ctx.queueCL0, 2, cl0Spacesize * self->ctx.sizeOfL0c);
        }
        if (!al1db) {
            self->ctx.pipe.InitBuffer(self->ctx.queueAL1, 1, AlignB(al1Spacesize * self->ctx.sizeOfFmap, C0_SIZE));
        } else {
            self->ctx.pipe.InitBuffer(self->ctx.queueAL1, 2, AlignB(al1Spacesize * self->ctx.sizeOfFmap, C0_SIZE));
        }
        if constexpr (!Intf::bl1bypass) {
            if (!bl1db) {
                self->ctx.pipe.InitBuffer(
                    self->ctx.queueBL1, 1, AlignB(bl1Spacesize * self->ctx.sizeOfWeight, C0_SIZE));
            } else {
                self->ctx.pipe.InitBuffer(
                    self->ctx.queueBL1, 2, AlignB(bl1Spacesize * self->ctx.sizeOfWeight, C0_SIZE));
            }
        }
    }
};

}  // namespace Conv3dFunc

#endif
