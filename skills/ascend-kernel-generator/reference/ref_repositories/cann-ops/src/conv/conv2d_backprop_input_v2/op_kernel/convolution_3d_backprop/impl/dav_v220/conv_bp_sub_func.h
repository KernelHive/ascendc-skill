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
 * \file conv_bp_sub_func.h
 * \brief
 */

#ifndef CONV3D_BP_INPUT_SUB_FUNC_H
#define CONV3D_BP_INPUT_SUB_FUNC_H

#include "conv3d_bp_kernel_split.h"

namespace Convolution3DBackpropFunc {
constexpr int32_t MAX_16BITS_STRIDE = 65535;

template <class Intf>
__aicore__ inline uint32_t CalFmapH(Intf *self, uint32_t mL1Size)
{
    uint32_t hiCal;
    if (mL1Size >= self->ctx.tiling_->wi) {
        hiCal = mL1Size / self->ctx.tiling_->wi;
        if (mL1Size != hiCal * self->ctx.tiling_->wi) {
            hiCal += 2;
        }
    } else {
        hiCal = self->ctx.tiling_->wi % mL1Size == 0 ? 1 : 2;
    }
    uint32_t khDilation = (self->ctx.tiling_->hk - 1) * self->ctx.tiling_->dilationH + 1;
    return (hiCal - 1) + khDilation;
}

template <class Intf>
__aicore__ inline void InitMmadParams(Intf *self)
{
    self->ctx.mmad_.m = self->ctx.baseUseM_;
    if (unlikely(self->ctx.curML0Idx_ == self->ctx.mIter_ - 1)) {
        // 4用来替换除法运算
        self->ctx.mmad_.m = ((self->ctx.baseUseM_ + BLOCK_CUBE - 1) >> 4) * BLOCK_CUBE;
    }
    self->ctx.mmad_.k = self->ctx.tiling_->baseK;
    self->ctx.mmad_.n = self->ctx.baseUseN_;
    self->ctx.mmad_.unitFlag = 0;
    self->ctx.mmad_.kDirectionAlign = 0;
    self->ctx.mmad_.cmatrixSource = 0;
    self->ctx.mmad_.cmatrixInitVal = 1;
}

template <class Intf>
__aicore__ inline void MmadLocal(Intf *self, const LocalTensor<typename Intf::SrcT> &l0a,
                                        const LocalTensor<typename Intf::SrcT> &l0b,
                                        LocalTensor<typename Intf::L0cT> &l0c)
{
    // MMAD计算量baseM*baseN小于一定阈值时需要添加PIPE_M同步,当前平台阈值为10*256
    constexpr int32_t mmadThreshold = 10 * 256;
    if (self->ctx.mmad_.m * self->ctx.mmad_.n <= mmadThreshold) {
        AscendC::PipeBarrier<PIPE_M>();
    }
    MmadImpl(l0c, l0a, l0b, self->ctx.mmad_);
}

// 设置fmatrix参数
template <class Intf>
__aicore__ inline void CalcSetFmatrixParams(Intf *self, uint32_t fmapH, uint32_t fmapW)
{
    // W
    self->ctx.load3d_.l1W = fmapW;
    // H
    self->ctx.load3d_.l1H = fmapH;
    // 设置pad, pad列表[left, right, top, down], 默认是0, 范围[0, 255]

    if (self->ctx.curHoIdx_ < 0) {
        self->ctx.load3d_.padList[2] = abs(self->ctx.curHoIdx_);
    } else {
        self->ctx.load3d_.padList[2] = 0;
    }
}

// 计算Load2A2的指令参数
template <class Intf>
__aicore__ inline void InitLoadToA2Params(Intf *self)
{
    // load3dStepM
    self->ctx.load3d_.mExtension = self->ctx.tiling_->baseM;
    // load3dStepK
    self->ctx.load3d_.kExtension = self->ctx.tiling_->baseK;
    // posM
    self->ctx.load3d_.mStartPt = 0;
    // posK
    self->ctx.load3d_.kStartPt = 0;
    // 前放大预处理，统一转换成stride=1的操作
    self->ctx.load3d_.strideW = 1;
    self->ctx.load3d_.strideH = 1;
    if constexpr (Intf::conv3dConfig.enableKernelSplit) {
        self->ctx.load3d_.filterW = self->ctx.splitWk_;
        self->ctx.load3d_.filterH = self->ctx.splitHk_;
        self->ctx.load3d_.filterSizeW = (self->ctx.splitWk_ >> 8) & 255;
        self->ctx.load3d_.filterSizeH = (self->ctx.splitHk_ >> 8) & 255;
    } else {
        self->ctx.load3d_.filterW = self->ctx.tiling_->wk;
        self->ctx.load3d_.filterH = self->ctx.tiling_->hk;
        self->ctx.load3d_.filterSizeW = (self->ctx.tiling_->wk >> 8) & 255;
        self->ctx.load3d_.filterSizeH = (self->ctx.tiling_->hk >> 8) & 255;
    }
    self->ctx.load3d_.dilationFilterW = self->ctx.tiling_->dilationW;
    self->ctx.load3d_.dilationFilterH = self->ctx.tiling_->dilationH;
    self->ctx.load3d_.enTranspose = 0;
    self->ctx.load3d_.fMatrixCtrl = 0;
    self->ctx.load3d_.channelSize = self->ctx.channelSize_;

    // 设置pad, pad列表[left, right, top, down], 默认是0, 范围[0, 255]
    self->ctx.load3d_.padList[0] = self->ctx.tiling_->backpropPadLeft;
    self->ctx.load3d_.padList[1] = self->ctx.tiling_->backpropPadRight;
    self->ctx.load3d_.padList[3] = 255;
}

template <class Intf>
__aicore__ inline void UpdateLoadToA2ParamsM(Intf *self)
{
    // load3dStepM
    self->ctx.load3d_.mExtension = self->ctx.baseUseM_;
    if (unlikely(self->ctx.curML0Idx_ == self->ctx.mIter_ - 1)) {
        // 4用来替换除法运算
        self->ctx.load3d_.mExtension = ((self->ctx.baseUseM_ + BLOCK_CUBE - 1) >> 4) * BLOCK_CUBE;
    }
    // posM: 当前默认stepM = 1
    if constexpr (Intf::conv3dConfig.enableKernelSplit) {
        self->ctx.load3d_.mStartPt = self->ctx.curML0Idx_ * self->ctx.tiling_->baseM % self->ctx.splitWi_;
    } else {
        self->ctx.load3d_.mStartPt = self->ctx.curML0Idx_ * self->ctx.tiling_->baseM % self->ctx.tiling_->wi;
    }
}

template <class Intf>
__aicore__ inline void UpdateCurHoSize(Intf *self)
{
    // posM: 当前默认stepM = 1
    uint32_t curHoSize = 0;
    if constexpr (Intf::conv3dConfig.enableKernelSplit) {
        curHoSize = CalFmapHForKernelSplit<Intf>(self, self->ctx.tiling_->baseM * self->ctx.tiling_->stepM);
    } else {
        curHoSize = CalFmapH<Intf>(self, self->ctx.tiling_->baseM * self->ctx.tiling_->stepM);
    }
    uint32_t hoExpand = (self->ctx.tiling_->ho - 1) * self->ctx.tiling_->strideH + 1;
    if (self->ctx.curHoIdx_ < 0) {
        curHoSize += self->ctx.curHoIdx_;
        curHoSize = curHoSize > hoExpand ? hoExpand : curHoSize;
    } else if (self->ctx.curHoIdx_ + curHoSize >= hoExpand) {
        if (self->ctx.curHoIdx_ < hoExpand) {
            curHoSize = hoExpand - self->ctx.curHoIdx_;
        } else {
            curHoSize = 0;
        }
    }
    self->ctx.curHoSize_ = curHoSize;
}

// 计算B2参数
template <class Intf>
__aicore__ inline void InitLoadToB2Params(Intf *self)
{
    self->ctx.load2d_.startIndex = 0;
    self->ctx.load2d_.repeatTimes = 0;
    self->ctx.load2d_.srcStride = 0;
    self->ctx.load2d_.dstGap = 0;
    self->ctx.load2d_.ifTranspose = 1;
    self->ctx.idxC1in_ = 0;
    self->ctx.baseB1Offset_ = 0;
}

template <class Intf>
__aicore__ inline void UpdateLoadToB2ParamsN(Intf *self)
{
    self->ctx.blockBaseN_ = self->ctx.baseUseN_ >> self->ctx.tiling_->c0Bits;
    self->ctx.load2d_.repeatTimes = self->ctx.blockBaseN_;
}

template <class Intf>
__aicore__ inline void UpdateLoadToB2ParamsK(Intf *self)
{
    if constexpr (Intf::conv3dConfig.enableKernelSplit) {
        self->ctx.load2d_.srcStride =
            self->ctx.curLoadKbl1_ / self->ctx.splitHkWkC0_ * self->ctx.splitHkWk_;
    } else if constexpr (Intf::conv3dConfig.loadB2Condition == Convolution3DBackprop::B2Condition::HKWK_EQ_ONE) {
            self->ctx.load2d_.srcStride = self->ctx.curLoadKbl1_ >> self->ctx.tiling_->c0Bits;
    } else {
        int64_t curBL1DivChannelSize = self->ctx.curLoadKbl1_ >> self->ctx.tiling_->c0Bits;
        self->ctx.load2d_.srcStride =
            CalcFloorAlign(curBL1DivChannelSize, self->ctx.HkWk_);
    }
}

template <class Intf>
__aicore__ inline void LoadToB2V1(Intf *self, const LocalTensor<typename Intf::SrcT> &l1B1Matrix, uint32_t kPos,
                                         LocalTensor<typename Intf::SrcT> &l0b)
{
    // 转置逆序
    uint32_t kRepeat = self->ctx.mmad_.k >> self->ctx.tiling_->c0Bits;
    uint32_t kBlockC0Num = kPos * (self->ctx.tiling_->baseK >> self->ctx.tiling_->c0Bits);
    constexpr uint32_t blockSize = 256; // 256为load2d最小搬运单位的元素数
    uint32_t dstB2Stride = self->ctx.blockBaseN_ * blockSize;

    if constexpr (Intf::conv3dConfig.enableKernelSplit) {
        LoadToB2ForKernelSplit<Intf>(self, l1B1Matrix, kRepeat, kBlockC0Num, blockSize, l0b);
    } else if constexpr (Intf::conv3dConfig.loadB2Condition == Convolution3DBackprop::B2Condition::HKWK_EQ_ONE) {
        uint64_t dstB2Offset = 0;
        uint32_t srcB1Offset = self->ctx.baseB1Offset_ + kBlockC0Num * blockSize;
        for (uint32_t i = 0; i < kRepeat; i++) {
            LoadData(l0b[dstB2Offset], l1B1Matrix[srcB1Offset], self->ctx.load2d_);
            dstB2Offset += dstB2Stride;
            srcB1Offset += blockSize;
        }
    } else if constexpr (Intf::conv3dConfig.loadB2Condition == Convolution3DBackprop::B2Condition::BASEK_LT_HKWK) {
        uint32_t baseC1outIdx = CalcDiv(kBlockC0Num, self->ctx.HkWk_);
        uint32_t curL1CoutC0 = CalcDiv(self->ctx.curLoadKbl1_, self->ctx.HkWk_) * self->ctx.tiling_->c0;
        uint32_t baseHkWkOffset = self->ctx.HkWk_ - 1 - kBlockC0Num + baseC1outIdx * self->ctx.HkWk_;
        uint32_t dstB2Offset = 0;
        uint32_t srcB1Offset = self->ctx.baseB1Offset_ + baseC1outIdx * blockSize + baseHkWkOffset * curL1CoutC0;

        for (uint32_t i = 0; i < kRepeat; i++) {
            LoadData(l0b[dstB2Offset], l1B1Matrix[srcB1Offset], self->ctx.load2d_);
            dstB2Offset += dstB2Stride;
            srcB1Offset -= curL1CoutC0;
        }
    } else {
        uint32_t baseC1outIdx = CalcDiv(kBlockC0Num, self->ctx.HkWk_);
        uint32_t curL1Cout = CalcDiv(self->ctx.curLoadKbl1_, self->ctx.HkWk_);
        uint32_t baseHkWkOffset = self->ctx.HkWk_ - 1 - kBlockC0Num + baseC1outIdx * self->ctx.HkWk_;
        uint64_t dstB2Offset = 0;

        for (uint32_t i = 0; i < kRepeat; i++) {
            uint32_t idxC1out = baseC1outIdx + CalcDiv(i, self->ctx.HkWk_);
            uint32_t srcB1Offset = self->ctx.baseB1Offset_ + idxC1out * blockSize +
                                (baseHkWkOffset - CalcRemainder(i, self->ctx.HkWk_)) * self->ctx.tiling_->c0 * curL1Cout;
            LoadData(l0b[dstB2Offset], l1B1Matrix[srcB1Offset], self->ctx.load2d_);
            dstB2Offset += dstB2Stride;
        }
    }
}

static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {false, false};
static constexpr uint8_t PADLIST_B[4] = {0, 0, 0, 0};

template <class Intf>
__aicore__ inline void LoadToB2Pro(Intf *self, const LocalTensor<typename Intf::SrcT> &l1B1Matrix,
    uint32_t kPos, uint32_t l0bKIdx, bool b1PingPongFlag, LocalTensor<typename Intf::SrcT> &l0b)
{
    // 转置逆序：load3dv2版本，当前进支持fp32，fp16场景可以简单适配
    // fp32场景，当baseK=8时，需要向上取整
    uint32_t kRepeat = self->ctx.mmad_.k >> self->ctx.tiling_->c0Bits;
    uint32_t kBlockC0Num = l0bKIdx * kRepeat;
    uint32_t baseHkOffset = kBlockC0Num / self->ctx.tiling_->wk % self->ctx.tiling_->hk;
    uint32_t baseWkOffset = kBlockC0Num % self->ctx.tiling_->wk;
    uint32_t baseHkWkOffset = self->ctx.HkWk_ - 1 - baseHkOffset * self->ctx.tiling_->wk - baseWkOffset;
    uint32_t curL1Cout = DivCeil(self->ctx.curLoadKbl1_ / self->ctx.HkWkC0_, 2) * BLOCK_CUBE;
    uint16_t wSize = static_cast<uint16_t>(curL1Cout * self->ctx.HkWk_);
    // 参数对应关系：H，W，pad，mode
    SetFmatrix(1, wSize, PADLIST_B, FmatrixMode::FMATRIX_RIGHT);
    uint16_t kStart = static_cast<uint16_t>(self->ctx.curNL0Idx_ % self->ctx.curStepN_ * self->ctx.baseUseN_);
    uint16_t channelSize = static_cast<uint16_t>(self->ctx.curStepN_ * self->ctx.baseUseN_);
    // fp32场景下，B矩阵是16*8的小分形，2个16*8小分形，变成2个8*16小分形
    uint16_t numHkWk = kPos * self->ctx.mmad_.k / self->ctx.HkWkC0_;
    // 根据是上半个分形，还是下半个分形，确定M方向是否产生偏移
    uint16_t mStartOffset = static_cast<uint16_t>((numHkWk & 1) * self->ctx.tiling_->c0);
    // 载入L1的数据，每2个HkWkC0对应1个C0out(16)，会出现只用一半的情况，此时load3dv2的M方向偏移要根据坐标推算
    uint32_t curCoutIdx = b1PingPongFlag ? self->ctx.curPingCoutIdx_ : self->ctx.curPongCoutIdx_;
    uint32_t coutOffset = (kPos * self->ctx.tiling_->baseK / (self->ctx.HkWk_ * BLOCK_CUBE) - curCoutIdx) * BLOCK_CUBE;
    for (uint32_t i = 0; i < kRepeat; i++) {
        // baseCoutOffset参数用于判断baseK是否需要载入多个HkWkC0，此时要进行Cout偏移计算
        uint32_t baseCoutOffset = i / self->ctx.HkWk_ * self->ctx.tiling_->c0;
        uint32_t dstB2Offset = i * self->ctx.baseUseAlignN_ * self->ctx.tiling_->c0;
        // (baseHkWkOffset - i % self->ctx.HkWk_) * BLOCK_CUBE表示HkWk上偏移，coutOffset表示Cout上面偏移，mStartOffset表一个小分型取上半或者下半数据
        uint16_t mStart = static_cast<uint16_t>((baseHkWkOffset - i % self->ctx.HkWk_) * curL1Cout + baseCoutOffset + coutOffset + mStartOffset);
        LoadData<typename Intf::SrcT, LOAD3DV2_CONFIG>(l0b[dstB2Offset], l1B1Matrix[0],
                                                    {
                                                        PADLIST_B,      // pad
                                                        1,              // L1_H
                                                        wSize,  // L1_W
                                                        channelSize,   // channelSize
                                                        static_cast<uint16_t>(self->ctx.baseUseN_),     // kExtension
                                                        BLOCK_CUBE,     // mExtension
                                                        kStart,      // kStartPt
                                                        mStart,      // mStartPt
                                                        1,      // strideW
                                                        1,      // strideH
                                                        1,      // filterW
                                                        1,      // filterH
                                                        1,      // dilationFilterW
                                                        1,      // dilationFilterH
                                                        true,   // enableTranspose，dst为L0B的情况下，硬件一定会使能tranpose能力，以满足L0B分型要求
                                                        false,  // enableSmallK
                                                        0,      // padValue
                                                        0,      // filterSizeWIn
                                                        0,      // filterSizeHIn
                                                        1       // fMatrixCtrlIn 使能set_fmatrix设置
                                                    }
                                                );
    }
}

template <class Intf>
__aicore__ inline void LoadToB2ProGemm(Intf *self, const LocalTensor<typename Intf::SrcT> &l1B1Matrix,
    uint32_t kPos, uint32_t l0bKIdx, LocalTensor<typename Intf::SrcT> &l0b)
{
    // 转置逆序：load3dv2版本，拆分出kernel=1*1模板
    uint32_t kRepeat = DivCeil(self->ctx.mmad_.k, BLOCK_CUBE);
    // 计算2块HkWkC0需要载入一份完整的C0out=16
    uint16_t wSize = static_cast<uint16_t>(DivCeil(self->ctx.curLoadKbl1_ / self->ctx.HkWkC0_, 2) * BLOCK_CUBE);
    // 参数对应关系：H，W，pad，mode
    SetFmatrix(1, wSize, PADLIST_B, FmatrixMode::FMATRIX_RIGHT);
    uint16_t kStart = static_cast<uint16_t>(self->ctx.curNL0Idx_ % self->ctx.curStepN_ * self->ctx.baseUseN_);
    uint16_t channelSize = static_cast<uint16_t>(self->ctx.curStepN_ * self->ctx.baseUseN_);
    uint32_t mStartL1Pos = l0bKIdx * self->ctx.mmad_.k;
    for (uint32_t i = 0; i < kRepeat; i++) {
        uint32_t dstB2Offset = i * self->ctx.baseUseAlignN_ * BLOCK_CUBE;
        uint16_t mStart = static_cast<uint16_t>(i * BLOCK_CUBE + mStartL1Pos);
        LoadData<typename Intf::SrcT, LOAD3DV2_CONFIG>(l0b[dstB2Offset], l1B1Matrix[0],
                                                    {
                                                        PADLIST_B,      // pad
                                                        1,              // L1_H
                                                        wSize,  // L1_W
                                                        channelSize,   // channelSize
                                                        static_cast<uint16_t>(self->ctx.baseUseN_),     // kExtension
                                                        BLOCK_CUBE,     // mExtension
                                                        kStart,      // kStartPt
                                                        mStart,      // mStartPt
                                                        1,      // strideW
                                                        1,      // strideH
                                                        1,      // filterW
                                                        1,      // filterH
                                                        1,      // dilationFilterW
                                                        1,      // dilationFilterH
                                                        true,   // enableTranspose，dst为L0B的情况下，硬件一定会使能tranpose能力，以满足L0B分型要求
                                                        false,  // enableSmallK
                                                        0,      // padValue
                                                        0,      // filterSizeWIn
                                                        0,      // filterSizeHIn
                                                        1       // fMatrixCtrlIn 使能set_fmatrix设置
                                                    }
                                                );
    }
}

template <class Intf>
__aicore__ inline void LoadToB2(Intf *self, const LocalTensor<typename Intf::SrcT> &l1B1Matrix,
    uint32_t l0bKIdx, uint32_t kPos, bool b1PingPongFlag, LocalTensor<typename Intf::SrcT> &l0b)
{
    if constexpr((std::is_same<typename Intf::SrcT, bfloat16_t>::value) || (std::is_same<typename Intf::SrcT, half>::value)) {
        LoadToB2V1<Intf>(self, l1B1Matrix, l0bKIdx, l0b);
    } else if constexpr(std::is_same<typename Intf::SrcT, float>::value) {
        if constexpr (Intf::conv3dConfig.loadB2Condition == Convolution3DBackprop::B2Condition::HKWK_EQ_ONE) {
            LoadToB2ProGemm<Intf>(self, l1B1Matrix, kPos, l0bKIdx, l0b);
        } else {
            LoadToB2Pro<Intf>(self, l1B1Matrix, kPos, l0bKIdx, b1PingPongFlag, l0b);
        }
    }
}

// 数据从A1加载到A2
template <class Intf>
__aicore__ inline void LoadToA2(Intf *self, const LocalTensor<typename Intf::SrcT> &l1A1Matrix,
                                       LocalTensor<typename Intf::SrcT> &l0a)
{
    LoadDataImpl(l0a, l1A1Matrix, self->ctx.load3d_);
}

template <class Intf>
static __aicore__ inline uint32_t CalcHDstDataSkipLine(Intf *self)
{
    uint32_t strideH = self->ctx.tiling_->strideH;
    uint32_t hDstDataSkipLine = 0;
    uint32_t actualDstDataStartIdx = (self->ctx.curHoIdx_ < 0) ? 0 : self->ctx.curHoIdx_;
    if (strideH > 1 && actualDstDataStartIdx % strideH) {
        hDstDataSkipLine = ((actualDstDataStartIdx + (strideH - 1)) / strideH) * strideH;
        hDstDataSkipLine = hDstDataSkipLine - actualDstDataStartIdx;
    }
    return hDstDataSkipLine;
}

template <class Intf, class src0_T>
__aicore__ inline void CalcLoadToA1DataCopyParams(Intf *self, DataCopyParams &dataCopyParams,
                                                         uint32_t &loadToA1Cout1Loop, uint32_t &loadToA1HLoop,
                                                         uint64_t &srcDataStride, uint32_t &dstDataStride,
                                                         uint32_t &padOffset, uint32_t &curHoSize, uint32_t curCout1Idx)
{
    uint32_t curCout1Size = 0;
    if constexpr (Intf::conv3dConfig.loadB2Condition == Convolution3DBackprop::B2Condition::HKWK_EQ_ONE) {
        curCout1Size = self->ctx.curLoadKal1_ >> self->ctx.tiling_->c0Bits;
    } else {
        curCout1Size = self->ctx.curLoadKal1_ / self->ctx.HkWkC0_;
    }

    curCout1Size = (curCout1Size < self->ctx.singleShapeCout1_ - curCout1Idx) ?
                    curCout1Size : self->ctx.singleShapeCout1_ - curCout1Idx;

    uint32_t hDstDataSkipLine = CalcHDstDataSkipLine(self);
    uint32_t strideH = self->ctx.tiling_->strideH;
    uint32_t woExpand = self->ctx.tiling_->wo;
    uint32_t strideW = self->ctx.tiling_->strideW;
    if (curHoSize <= hDstDataSkipLine) {
        dataCopyParams.blockCount = 0;
    } else if (strideW > 1) {
        // 一次datacopy， 拷贝wo个c0,每个c0在目的地址间隔strideW-1, 一共拷贝
        dataCopyParams.blockLen = 1;
        dataCopyParams.blockCount = self->ctx.tiling_->wo;
        dataCopyParams.dstStride = (strideW - 1);

        loadToA1Cout1Loop = curCout1Size;
        loadToA1HLoop = (curHoSize - hDstDataSkipLine + (strideH - 1)) / strideH;
        srcDataStride = (self->ctx.tiling_->wo * self->ctx.tiling_->c0);

        woExpand = ((woExpand - 1) * strideW) + 1;
        dstDataStride = woExpand * self->ctx.tiling_->c0;
        padOffset = hDstDataSkipLine * dstDataStride;
    } else {
        loadToA1HLoop = 1;
        if (strideH > 1) {  // 一次datacoy，拷贝curHo*Wo个c0，拷贝CurCout1次
            uint32_t curInputHoSize = (curHoSize - hDstDataSkipLine + (strideH - 1)) / strideH;
            dataCopyParams.srcStride = 0;
            // 当前不支持(strideH - 1) * self->ctx.tiling_->wo 超过uint16_max的场景
            dataCopyParams.dstStride = (strideH - 1) * self->ctx.tiling_->wo;
            dataCopyParams.blockLen = self->ctx.tiling_->wo;
            dataCopyParams.blockCount = curInputHoSize;

            uint32_t wDataStride = self->ctx.tiling_->wo * self->ctx.tiling_->c0;
            // 防止Ho较大，uint32溢出
            srcDataStride = self->ctx.hwO_ * self->ctx.tiling_->c0;
            dstDataStride = wDataStride;
            loadToA1Cout1Loop = curCout1Size;
            // 由于strideH 和Wo的限制，此处不会溢出
            padOffset = hDstDataSkipLine * wDataStride;
        } else {
            // 原数据尾和头的间隔
            uint64_t dataCoptSrcStride = static_cast<uint64_t>(self->ctx.tiling_->ho - curHoSize) * self->ctx.tiling_->wo;
            dataCopyParams.dstStride = 0;  // 目的数据尾和头的间隔
            dataCopyParams.blockLen = curHoSize * self->ctx.tiling_->wo;
            if (dataCoptSrcStride <= UINT16_MAX) {
                dataCopyParams.srcStride = static_cast<uint16_t>(dataCoptSrcStride);
                dataCopyParams.blockCount = curCout1Size;
                loadToA1Cout1Loop = 1;
            } else {
                // due to srcStride should not over uint16_max, so we have to move data one by one
                dataCopyParams.srcStride = 0;
                dataCopyParams.blockCount = 1;
                loadToA1Cout1Loop = curCout1Size;
                dstDataStride = self->ctx.tiling_->wo * self->ctx.tiling_->c0;
            }
        }
    }
    CalcSetFmatrixParams(self, curHoSize, woExpand);
}

template <class Intf, class src0_T>
__aicore__ inline void LoadToA1(Intf *self, uint32_t kIdx, uint32_t curDoutIdx, bool pingPongFlag, bool loadFlag)
{
    if (loadFlag) {
        LocalTensor<typename Intf::SrcT> useA1Buf;
        if (pingPongFlag) {
            useA1Buf = self->ctx.a1Ping_.template AllocTensor<typename Intf::SrcT>();
        } else {
            useA1Buf = self->ctx.a1Pong_.template AllocTensor<typename Intf::SrcT>();
        }

        if constexpr (!Intf::conv3dConfig.enableKernelSplit) {
            if (unlikely(self->ctx.tiling_->strideW * self->ctx.tiling_->strideH > 1)) {
                // block num 以32B为单位, 5用以替换除法运算
                InitConstValue(useA1Buf, {1, static_cast<uint16_t>(useA1Buf.GetLength() >> 5), 0, 0U});
                pipe_barrier(PIPE_MTE2);
            }
        }

        uint32_t curCout1Idx = 0;
        if constexpr (Intf::conv3dConfig.enableKernelSplit) {
            curCout1Idx = kIdx * self->ctx.tiling_->baseK / self->ctx.splitHkWkC0_;
        } else {
            curCout1Idx = kIdx * self->ctx.tiling_->baseK / self->ctx.HkWkC0_;
        }

        DataCopyParams dataCopyParams;
        uint32_t loadToA1Cout1Loop = 0;
        uint32_t loadToA1HLoop = 0;
        uint64_t srcDataStride = 0;
        uint32_t dstDataStride = 0;
        uint32_t padDataOffset = 0;
        uint32_t curHoSize = self->ctx.curHoSize_;

        if constexpr (Intf::conv3dConfig.enableKernelSplit) {
            CalcLoadToA1ParamsForKernelSplit<Intf, typename Intf::SrcT>(self, dataCopyParams, loadToA1Cout1Loop,
                loadToA1HLoop, srcDataStride, dstDataStride, padDataOffset, curHoSize, curCout1Idx);
            CalcSetFmatrixParams(self, curHoSize, self->ctx.tiling_->wo - 1);
        } else {
            CalcLoadToA1DataCopyParams<Intf, typename Intf::SrcT>(self, dataCopyParams, loadToA1Cout1Loop,
                loadToA1HLoop, srcDataStride, dstDataStride, padDataOffset, curHoSize, curCout1Idx);
        }
        if (dataCopyParams.blockCount > 0) {
            // GM的绝对地址已经在API外部计算，这里采用绝对坐标时，需要去除起始坐标
            int64_t curHoStartOffset = self->ctx.curHoStartIdx_ < 0 ? 0 : self->ctx.curHoStartIdx_;
            int64_t curHoIdx = self->ctx.curHoIdx_ < 0 ? 0 : (self->ctx.curHoIdx_ - curHoStartOffset);
            // 换算回放大前的相对Ho坐标（以单核HoStartIdx为原点）
            int64_t curOriHoIdx = curHoIdx;

            if (self->ctx.tiling_->strideH > 1) {
                int64_t skipHoSize = curHoStartOffset % self->ctx.tiling_->strideH;
                skipHoSize = skipHoSize > 0 ? (self->ctx.tiling_->strideH - skipHoSize) : skipHoSize;
                curHoIdx = self->ctx.curHoIdx_ < 0 ? 0 : (self->ctx.curHoIdx_ - curHoStartOffset);
                // 换算回放大前的相对Ho坐标（以单核HoStartIdx为原点）
                curOriHoIdx = (curHoIdx - skipHoSize + self->ctx.tiling_->strideH - 1) / self->ctx.tiling_->strideH;
            }

            int64_t hoStride = self->ctx.tiling_->wo * self->ctx.tiling_->c0;
            int64_t coutStride = self->ctx.tiling_->ho * hoStride;
            int64_t out2A1SrcAddrOffsetBase =
                (curDoutIdx * self->ctx.tiling_->cout1 + curCout1Idx) * coutStride + curOriHoIdx * hoStride;
            int64_t dstOffsetC = padDataOffset;
            int64_t dstDataStrideC = curHoSize * dstDataStride;

            for (int64_t j = 0; j < loadToA1Cout1Loop; j++) {
                int64_t out2A1SrcAddrOffset = out2A1SrcAddrOffsetBase;
                int64_t dstOffset = dstOffsetC;
                for (int64_t i = 0; i < loadToA1HLoop; i++) {
                    DataCopy(useA1Buf[dstOffset], self->ctx.outBackPropGlobal_[out2A1SrcAddrOffset], dataCopyParams);
                    out2A1SrcAddrOffset += srcDataStride;
                    dstOffset += (dstDataStride * self->ctx.tiling_->strideH);
                }
                dstOffsetC += dstDataStrideC;
                out2A1SrcAddrOffsetBase += coutStride;
            }
        }

        if (pingPongFlag) {
            self->ctx.a1Ping_.EnQue(useA1Buf);
        } else {
            self->ctx.a1Pong_.EnQue(useA1Buf);
        }
    }
}

template <class Intf, class src1_T>
__aicore__ inline void LoadToB1Fp32(Intf *self, const uint32_t kIdx, const uint32_t curDkIdx,
                                    LocalTensor<typename Intf::SrcT> &useB1Buf, bool b1PingPongFlag)
{
    // 此处为原始输入的C0in，与dataType相关：bf16为16，fp32为8
    uint32_t curCin1Idx = self->ctx.curNL1Idx_ * self->ctx.curCin1Size_;

    // 此处C0跟dataType无关，与NZ分型相关
    uint32_t curCout1Idx = kIdx * self->ctx.tiling_->baseK / (self->ctx.HkWk_ * BLOCK_CUBE);
    // 记录每次载入到L1中的绝对Cout坐标，用于计算load3dv2在M方向上的偏移，涉及到1:2的数据载入问题
    if (b1PingPongFlag) {
        self->ctx.curPingCoutIdx_ = curCout1Idx;
    } else {
        self->ctx.curPongCoutIdx_ = curCout1Idx;
    }

    // kernel shape: (dk * cin1 * hk * wk, cout1, cout0, cin0)
    // fp32场景Cout可能非16对齐，需要对齐到16
    uint64_t out2B1SrcAddrOffset = (static_cast<uint64_t>(curDkIdx) * self->ctx.tiling_->cin1G + curCin1Idx) *
                                    self->ctx.HkWkC0_ * self->ctx.alignedCout_ +   // 与dataType相关
                                    static_cast<uint64_t>(curCout1Idx) * BLOCK_CUBE * self->ctx.tiling_->c0;  // 与NZ分型相关

    // K方向非16对齐时，原始GM数据需要把K补齐到16对齐，可能含有padding数据
    // 搬移数据时，需要考虑padding的数据
    uint32_t curCin1Size = self->ctx.curCin1Size_ < (self->ctx.singleShapeCin1_ - curCin1Idx) ?
                           self->ctx.curCin1Size_ : (self->ctx.singleShapeCin1_ - curCin1Idx);
    uint32_t curCout1Size = DivCeil(self->ctx.curLoadKbl1_ / self->ctx.HkWkC0_, 2) * BLOCK_CUBE;
    // 由于L1B满载HkWKC0, 故tiling侧已保证HkWKC0 * curCin1Size * dataByte <= L1B, 假设L1B最大512KB
    // 则HkWK * curCin1Size <= 16384, 不会超出uint16_max
    uint16_t blockCount = curCin1Size * self->ctx.HkWk_;

    DataCopyParams dataCopyParams;
    dataCopyParams.blockLen = (curCout1Size < (self->ctx.alignedCout1_ - curCout1Idx) * BLOCK_CUBE) ?
                               curCout1Size : ((self->ctx.alignedCout1_ - curCout1Idx) * BLOCK_CUBE);
    dataCopyParams.dstStride = 0;
    uint64_t srcStride = self->ctx.alignedCout_ - dataCopyParams.blockLen;
    if (srcStride <= MAX_16BITS_STRIDE) {
        dataCopyParams.blockCount = blockCount;
        dataCopyParams.srcStride = static_cast<uint16_t>(srcStride);
        DataCopy(useB1Buf, self->ctx.weightGlobal_[out2B1SrcAddrOffset], dataCopyParams);
    } else {
        dataCopyParams.blockCount = 1;
        uint64_t srcOffsetInterval = self->ctx.alignedCout_ * self->ctx.tiling_->c0;
        uint64_t dstOffsetInterval = dataCopyParams.blockLen * self->ctx.tiling_->c0;
        uint64_t dstOffset = 0;
        for (uint32_t idx = 0; idx < blockCount; ++idx) {
            DataCopy(useB1Buf[dstOffset], self->ctx.weightGlobal_[out2B1SrcAddrOffset], dataCopyParams);
            out2B1SrcAddrOffset += srcOffsetInterval;
            dstOffset += dstOffsetInterval;
        }
    }
}

template <class Intf, class src1_T>
__aicore__ inline void LoadToB1BF16(Intf *self, const uint32_t kIdx, const uint32_t curDkIdx, LocalTensor<typename Intf::SrcT> &useB1Buf)
{
    uint32_t curCin1Idx = self->ctx.curNL1Idx_ * self->ctx.curCin1Size_;
    uint32_t curCout1Idx = 0;
    uint32_t curCout1Size = 0;
    if constexpr (Intf::conv3dConfig.loadB2Condition == Convolution3DBackprop::B2Condition::HKWK_EQ_ONE) {
        curCout1Idx = kIdx * self->ctx.tiling_->baseK >> self->ctx.tiling_->c0Bits;
        curCout1Size = self->ctx.curLoadKbl1_ >> self->ctx.tiling_->c0Bits;
    } else {
        curCout1Idx = kIdx * self->ctx.tiling_->baseK / self->ctx.HkWkC0_;
        curCout1Size = self->ctx.curLoadKbl1_ / self->ctx.HkWkC0_;
    }

    // kernel shape: (dk * cin1 * hk * wk, cout1, cout0, cin0)
    // group kernel shape: (group * dk * cin1G * hk * wk, cout1G, cout0, cin0)
    uint64_t out2B1SrcAddrOffset = (static_cast<uint64_t>(curDkIdx) * self->ctx.tiling_->cin1G + curCin1Idx) *
                                   self->ctx.HkWkC0_ * self->ctx.tiling_->cout1G * self->ctx.tiling_->c0 +
                                   static_cast<uint64_t>(curCout1Idx) * self->ctx.tiling_->c0 * self->ctx.tiling_->c0;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockLen = (curCout1Size < self->ctx.singleShapeCout1_ - curCout1Idx ?
                               curCout1Size : self->ctx.singleShapeCout1_ - curCout1Idx) * self->ctx.tiling_->c0;
    dataCopyParams.dstStride = 0;
    uint32_t curCin1Size = self->ctx.curCin1Size_ < (self->ctx.singleShapeCin1_ - curCin1Idx) ?
                           self->ctx.curCin1Size_ : (self->ctx.singleShapeCin1_ - curCin1Idx);
    uint16_t blockCount = curCin1Size * self->ctx.HkWk_;
    uint64_t srcStride = static_cast<uint64_t>(self->ctx.tiling_->cout1G) * self->ctx.tiling_->c0 -
                         dataCopyParams.blockLen;

    if (srcStride <= MAX_16BITS_STRIDE) {
        dataCopyParams.blockCount = blockCount;
        dataCopyParams.srcStride = static_cast<uint16_t>(srcStride);
        DataCopy(useB1Buf, self->ctx.weightGlobal_[out2B1SrcAddrOffset], dataCopyParams);
    } else {
        dataCopyParams.blockCount = 1;
        uint64_t srcOffsetInterval = static_cast<uint64_t>(self->ctx.tiling_->cout1G) * self->ctx.tiling_->c0 *
                                     self->ctx.tiling_->c0;
        uint32_t dstOffsetInterval = static_cast<uint32_t>(dataCopyParams.blockLen) * self->ctx.tiling_->c0;
        uint64_t dstOffset = 0;
        for (uint32_t idx = 0; idx < blockCount; ++idx) {
            DataCopy(useB1Buf[dstOffset], self->ctx.weightGlobal_[out2B1SrcAddrOffset], dataCopyParams);
            out2B1SrcAddrOffset += srcOffsetInterval;
            dstOffset += dstOffsetInterval;
        }
    }
}

template <class Intf, class src1_T>
__aicore__ inline void LoadToB1(Intf *self, uint32_t kIdx, uint32_t curDkIdx, bool pingPongFlag, bool loadFlag)
{
    if (unlikely(self->ctx.isB1FullLoadFlag_ && !self->ctx.isLoadB1_)) {
        return;
    }

    if (loadFlag) {
        LocalTensor<typename Intf::SrcT> useB1Buf;
        if (pingPongFlag) {
            useB1Buf = self->ctx.b1Ping_.template AllocTensor<typename Intf::SrcT>();
        } else {
            useB1Buf = self->ctx.b1Pong_.template AllocTensor<typename Intf::SrcT>();
        }

        if constexpr (Intf::conv3dConfig.enableKernelSplit) {
            LoadToB1ForKernelSplit<Intf>(self, kIdx, curDkIdx, useB1Buf);
        } else {
            if constexpr((std::is_same<typename Intf::SrcT, bfloat16_t>::value) || (std::is_same<typename Intf::SrcT, half>::value)) {
                LoadToB1BF16<Intf, src1_T>(self, kIdx, curDkIdx, useB1Buf); // FP16也复用该函数
            } else if constexpr(std::is_same<typename Intf::SrcT, float>::value) {
                LoadToB1Fp32<Intf, src1_T>(self, kIdx, curDkIdx, useB1Buf, pingPongFlag);
            }
        }

        if (pingPongFlag) {
            self->ctx.b1Ping_.EnQue(useB1Buf);
        } else {
            self->ctx.b1Pong_.EnQue(useB1Buf);
        }
    }
}

template <class Intf>
__aicore__ inline void CopyData2Gm(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
                                          LocalTensor<typename Intf::L0cT> &useC1Buf, QuantMode_t quantMode)
{
    uint64_t dstOffset = static_cast<uint64_t>(self->ctx.curNL0Idx_) * self->ctx.tiling_->baseN * self->ctx.hwI_ +
        (static_cast<uint64_t>(self->ctx.curML0Idx_) * self->ctx.tiling_->baseM +  // M方向偏移
        static_cast<uint64_t>(self->ctx.curDinIdx_) * self->ctx.hwI_ *
        self->ctx.tiling_->cin1) * self->ctx.tiling_->c0;  // D方向偏移
    bool enableChannelSplit = false;
    if constexpr (std::is_same<typename Intf::DstT, float>::value) {
        enableChannelSplit = true;
    }
    uint32_t alingedBaseUseM = ShiftCeilBlockCube(self->ctx.baseUseM_) * BLOCK_CUBE;
    if (self->ctx.hwI_ <= UINT32_MAX) {
        DataCopyCO12DstParams dataCopyParams(
            static_cast<uint16_t>(self->ctx.baseUseN_),         //nSize
            static_cast<uint16_t>(self->ctx.baseUseM_),         //mSize
            static_cast<uint32_t>(self->ctx.hwI_),
            alingedBaseUseM, //srcStride
            quantMode,
            0,
            enableChannelSplit,
            false
        );
        DataCopy(output[dstOffset], useC1Buf, dataCopyParams);
    } else {
        // 由于HF32/FP32需要channel split，暂时指令无法支持此场景，需在tiling侧进行拦截，此处只有BF16/FP16走进来
        uint16_t blockCnt = self->ctx.baseUseN_ / self->ctx.tiling_->c0;
        uint64_t dstStrideOffset = self->ctx.hwI_ * self->ctx.tiling_->c0;
        uint32_t srcStrideOffset = alingedBaseUseM;
        uint32_t srcOffset = 0;
        for (uint16_t i = 0; i < blockCnt; i++) {
            DataCopyCO12DstParams dataCopyParams(
                static_cast<uint16_t>(self->ctx.tiling_->c0),         //nSize
                static_cast<uint16_t>(self->ctx.baseUseM_),         //mSize
                static_cast<uint32_t>(self->ctx.baseUseM_),  // 目的数据头和头的间隔, 此处为了满足接口不可设0的要求，实际并不生效
                0, //原数据头和头的间隔， 此处只拷贝一次，故设置为0
                quantMode,
                0,
                enableChannelSplit,
                false
            );
            DataCopy(output[dstOffset], useC1Buf[srcOffset], dataCopyParams);
            dstOffset += dstStrideOffset;
            srcOffset += srcStrideOffset;
        }
    }
}

template <class Intf>
__aicore__ inline void CopyData2TmpWorkspace(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
    LocalTensor<typename Intf::L0cT> &useC1Buf)
{
    QuantMode_t quantMode = QuantMode_t::F322BF16;
    if constexpr(std::is_same<typename Intf::DstT, half>::value) {
        quantMode = QuantMode_t::F322F16;
    } else if constexpr(std::is_same<typename Intf::DstT, float>::value) {
        quantMode = NoQuant;
    }
    int64_t dstOffset = block_idx * self->ctx.tiling_->baseM * self->ctx.tiling_->baseN;
    if constexpr((std::is_same<typename Intf::SrcT, bfloat16_t>::value) || (std::is_same<typename Intf::SrcT, half>::value)) {
        FixpipeParams<typename Intf::L0cT> fixpipeParams(
            static_cast<uint16_t>(Ceil(self->ctx.baseUseN_, 16)),
            static_cast<uint16_t>(self->ctx.baseUseM_ * BLOCK_CUBE * sizeof(typename Intf::L0cT) / 32),
            0,
            0);
        fixpipeParams.quantParams.quantPre = quantMode;
        Fixpipe(output[dstOffset], useC1Buf, fixpipeParams);
    }
}

template <class Intf>
__aicore__ inline void FreeL0cTensor(Intf *self, LocalTensor<typename Intf::L0cT> &l0c)
{
    if (self->ctx.usingCacheC1Ping_) {
        self->ctx.c1Ping_.FreeTensor(l0c);
    } else {
        self->ctx.c1Pong_.FreeTensor(l0c);
    }
}

template <class Intf>
__aicore__ inline void LoadL0c2Gm(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
                                         uint8_t enAtomic = 0, bool enSequentialWrite = false)
{
    if (!self->ctx.needComputeFlag_) {
        return;
    }

    LocalTensor<typename Intf::L0cT> useC1Buf;
    if (self->ctx.usingCacheC1Ping_) {
        useC1Buf = self->ctx.c1Ping_.template DeQue<typename Intf::L0cT>();
    } else {
        useC1Buf = self->ctx.c1Pong_.template DeQue<typename Intf::L0cT>();
    }
    if (unlikely(!self->ctx.needComputeKFlag_)) {
        FreeL0cTensor<Intf>(self, useC1Buf);
        return;
    }

    if (enAtomic == 1) {
        SetAtomicAdd<typename Intf::DstT>();
    }
    if constexpr (Intf::Config::dType::format == Convolution3DBackprop::CubeFormat::NDC1HWC0) {
        if (!enSequentialWrite) {
            QuantMode_t quantMode = QuantMode_t::F322BF16;
            if constexpr(std::is_same<typename Intf::DstT, half>::value) {
                quantMode = QuantMode_t::F322F16;
            } else if constexpr(std::is_same<typename Intf::DstT, float>::value) {
                quantMode = NoQuant;
            }
            if constexpr (Intf::conv3dConfig.enableKernelSplit) {
                LoadL0c2GmForKernelSplit<Intf>(self, output, useC1Buf, quantMode);
            } else {
                CopyData2Gm<Intf>(self, output, useC1Buf, quantMode);
            }
        } else {
            return;
        }
    } else if constexpr (Intf::Config::dType::format == Convolution3DBackprop::CubeFormat::NCDHW) {
        if (!enSequentialWrite) {
            CopyData2TmpWorkspace<Intf>(self, output, useC1Buf);
        } else {
            return;
        }
    }
    if (enAtomic == 1) {
        SetAtomicNone();
    }

    FreeL0cTensor<Intf>(self, useC1Buf);
    if (unlikely(self->ctx.tiling_->cl0Pbuffer > 1)) {
        self->ctx.usingCacheC1Ping_ = !self->ctx.usingCacheC1Ping_;
    }
}
}  // namespace Convolution3DBackpropFunc

#endif
