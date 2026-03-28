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
 * \file conv3d_bp_kernel_split.h
 * \brief
 */

#ifndef CONV3D_BP_KERNEL_SPLIT_H
#define CONV3D_BP_KERNEL_SPLIT_H

#include "kernel_operator.h"

namespace Convolution3DBackpropFunc {

template <class Intf>
__aicore__ inline uint32_t CalFmapHForKernelSplit(Intf *self, uint32_t mL1Size)
{
    uint32_t hiCal;
    if (mL1Size % self->ctx.splitWi_ == 0) {
        hiCal = mL1Size / self->ctx.splitWi_;
    } else if (mL1Size > self->ctx.splitWi_) {
        hiCal = mL1Size / self->ctx.splitWi_ + 2;
    } else {
        hiCal = 2;
    }
    uint32_t khDilation = self->ctx.splitHk_;
    return (hiCal - 1) + khDilation;
}

template <class Intf, class src0_T>
__aicore__ inline void CalcLoadToA1ParamsForKernelSplit(Intf *self, DataCopyParams &dataCopyParams,
                                                               uint32_t &loadToA1Cout1Loop, uint32_t &loadToA1HLoop,
                                                               uint64_t &srcDataStride, uint32_t &dstDataStride,
                                                               uint32_t &padOffset, uint32_t &curHoSize,
                                                               uint32_t curCout1Idx)
{
    uint32_t strideH = self->ctx.tiling_->strideH;
    uint32_t curCout1Size = self->ctx.curLoadKal1_ / self->ctx.splitHkWkC0_;
    curCout1Size =
        curCout1Size < self->ctx.tiling_->cout1 - curCout1Idx ? curCout1Size : self->ctx.tiling_->cout1 - curCout1Idx;
    loadToA1Cout1Loop = curCout1Size;
    loadToA1HLoop = 1;
    dataCopyParams.srcStride = 1;
    dataCopyParams.dstStride = 0;
    dataCopyParams.blockLen = self->ctx.tiling_->wo - 1;
    dataCopyParams.blockCount = curHoSize;

    srcDataStride = 0;
    dstDataStride = (self->ctx.tiling_->wo - 1) * self->ctx.tiling_->c0;
    padOffset = 0;
}

template <class Intf>
static __aicore__ inline void LoadToB2ForKernelSplit(Intf *self, const LocalTensor<typename Intf::SrcT> &l1B1Matrix,
    uint32_t kRepeat, uint32_t kBlockC0Num, uint32_t blockSize, LocalTensor<typename Intf::SrcT> &l0b)
{
    uint32_t baseC1outIdx = kBlockC0Num / self->ctx.splitHkWk_;
    uint32_t curL1Cout = self->ctx.curLoadKbl1_ / self->ctx.splitHkWk_;
    uint32_t baseHkWkOffset = self->ctx.splitHkWk_ - 1 - kBlockC0Num + baseC1outIdx * self->ctx.splitHkWk_;
    uint32_t dstB2Offset = 0;
    uint32_t dstB2Stride = self->ctx.blockBaseN_ * blockSize;

    for (uint32_t i = 0; i < kRepeat; i++) {
        uint32_t idxC1out = baseC1outIdx + i / self->ctx.splitHkWk_;
        uint32_t srcB1Offset = self->ctx.baseB1Offset_ + idxC1out * blockSize +
                            (baseHkWkOffset - i % self->ctx.splitHkWk_) * self->ctx.tiling_->c0 * curL1Cout;
        LoadData(l0b[dstB2Offset], l1B1Matrix[srcB1Offset], self->ctx.load2d_);
        dstB2Offset += dstB2Stride;
    }
}

template <class Intf>
__aicore__ inline void LoadToB1ForKernelSplit(Intf *self, const uint32_t kIdx, const uint32_t curDkIdx,
                                              LocalTensor<typename Intf::SrcT> &useB1Buf)
{
    uint32_t dstOffset = 0;
    uint64_t cin1Offset = static_cast<uint64_t>(self->ctx.HkWkC0_) * self->ctx.tiling_->cout1 * self->ctx.tiling_->c0;
    uint32_t curCin1Idx = self->ctx.curNL1Idx_ * self->ctx.curCin1Size_;
    uint32_t curCout1Idx = kIdx * self->ctx.tiling_->baseK / self->ctx.splitHkWkC0_;

    // kernel shape: (dk * cin1 * hk * wk, cout1, cout0, cin0)
    uint64_t out2B1SrcAddrOffset = (static_cast<uint64_t>(curDkIdx) * self->ctx.tiling_->cin1 + curCin1Idx) *
                                    self->ctx.HkWkC0_ * self->ctx.tiling_->cout1 * self->ctx.tiling_->c0 +
                                    static_cast<uint64_t>(curCout1Idx) * self->ctx.tiling_->c0 * self->ctx.tiling_->c0;

    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = self->ctx.splitWk_;
    dataCopyParams.blockLen =
        self->ctx.curLoadKbl1_ / self->ctx.splitHkWk_;
    dataCopyParams.srcStride = (self->ctx.tiling_->cout1 * self->ctx.tiling_->c0) -
        dataCopyParams.blockLen + (self->ctx.tiling_->cout1 * self->ctx.tiling_->c0);
    dataCopyParams.dstStride = 0;
    uint32_t curCin1Size = self->ctx.curCin1Size_ < (self->ctx.tiling_->singleCoreCin1 - curCin1Idx) ?
                        self->ctx.curCin1Size_ : (self->ctx.tiling_->singleCoreCin1 - curCin1Idx);
    for (uint32_t cin1Idx = 0; cin1Idx < curCin1Size; ++cin1Idx) {
        uint64_t cin1AddrOffset = cin1Idx * cin1Offset;
        uint64_t srcAddrOffset = out2B1SrcAddrOffset + cin1AddrOffset;
        for (uint32_t khIdx = 0; khIdx < self->ctx.splitHk_; ++khIdx) {
            DataCopy(useB1Buf[dstOffset], self->ctx.weightGlobal_[srcAddrOffset], dataCopyParams);
            dstOffset += (self->ctx.splitWk_ * dataCopyParams.blockLen * 32 / sizeof(typename Intf::SrcT));
            srcAddrOffset += (self->ctx.tiling_->strideH * self->ctx.tiling_->wk * self->ctx.tiling_->c0 *
                self->ctx.tiling_->cout1 * self->ctx.tiling_->c0);
        }
    }
}

template <class Intf>
__aicore__ inline void LoadL0c2GmForKernelSplit(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
    LocalTensor<typename Intf::L0cT> &useC1Buf, QuantMode_t fixPipQuantMode)
{
    int64_t hiWi = static_cast<int64_t>(self->ctx.tiling_->hi) * self->ctx.tiling_->wi;
    int64_t outOneComputeDstOffset = static_cast<int64_t>(self->ctx.curML0Idx_) % self->ctx.mIter_ * self->ctx.tiling_->baseM *
        self->ctx.tiling_->strideH * self->ctx.tiling_->strideW * self->ctx.tiling_->c0 + // M方向偏移
        static_cast<int64_t>(self->ctx.curNL0Idx_) * self->ctx.tiling_->baseN * hiWi + // N方向偏移
        static_cast<int64_t>(self->ctx.curDinIdx_) * hiWi * self->ctx.tiling_->cin1 * self->ctx.tiling_->c0; // D方向偏移
    int64_t inOneComputeDstOffset = 0;
    int64_t dstOffset = 0;
    int srcOffset = 0;
    FixpipeParamsV220 fixpipeParams;
    if (self->ctx.splitWi_ % BLOCK_CUBE) {
        fixpipeParams.nSize = BLOCK_CUBE;
        fixpipeParams.mSize = self->ctx.splitWi_;
        fixpipeParams.srcStride = self->ctx.splitWi_;
        fixpipeParams.dstStride = self->ctx.tiling_->strideW * BLOCK_CUBE;
        fixpipeParams.quantPre = fixPipQuantMode;
        fixpipeParams.ndNum = 1;
        for (uint32_t nIdx = 0; nIdx < Ceil(self->ctx.baseUseN_, BLOCK_CUBE); ++nIdx) {
            inOneComputeDstOffset = nIdx * hiWi * BLOCK_CUBE;
            dstOffset = inOneComputeDstOffset + outOneComputeDstOffset;
            for (uint32_t hIdx = 0; hIdx < Ceil(self->ctx.baseUseM_, BLOCK_CUBE); ++hIdx) {
                Fixpipe(output[dstOffset], useC1Buf[srcOffset], fixpipeParams);
                srcOffset += (self->ctx.splitWi_ * BLOCK_CUBE);
                dstOffset += (self->ctx.tiling_->strideH * self->ctx.tiling_->wi * BLOCK_CUBE);
            }
        }
    } else {
        fixpipeParams.nSize = BLOCK_CUBE;
        fixpipeParams.mSize = self->ctx.splitWi_;
        fixpipeParams.srcStride = self->ctx.splitWi_;
        fixpipeParams.dstStride = self->ctx.tiling_->strideW * BLOCK_CUBE;
        fixpipeParams.srcNdStride = self->ctx.splitWi_ * BLOCK_CUBE * sizeof(typename Intf::L0cT) / 1024;
        fixpipeParams.dstNdStride = self->ctx.tiling_->strideH * self->ctx.tiling_->wi * BLOCK_CUBE;
        fixpipeParams.quantPre = fixPipQuantMode;
        fixpipeParams.ndNum = self->ctx.baseUseM_ / self->ctx.splitWi_;
        for (uint32_t nIdx = 0; nIdx < Ceil(self->ctx.baseUseN_, BLOCK_CUBE); ++nIdx) {
            inOneComputeDstOffset = nIdx * hiWi * BLOCK_CUBE;
            dstOffset = inOneComputeDstOffset + outOneComputeDstOffset;
            Fixpipe(output[dstOffset], useC1Buf[srcOffset], fixpipeParams);
            srcOffset += (self->ctx.baseUseM_ * BLOCK_CUBE);
        }
    }
}
}
#endif
