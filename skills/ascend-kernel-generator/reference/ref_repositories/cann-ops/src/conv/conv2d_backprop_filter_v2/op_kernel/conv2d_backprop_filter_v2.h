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
 * @file conv2d_backprop_filter_v2.h
 */
#ifndef CONV2D_BACKPROP_FILTER_H
#define CONV2D_BACKPROP_FILTER_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "conv2d_bp_filter.h"
#include "kernel_type.h"

constexpr int BLOCK_CUBE = 16;
constexpr int BLOCK_C04 = 4;

namespace AscendC {

__aicore__ inline constexpr ConvolutionBackprop::CubeFormat GetFormat(int format) {
    if (format == FORMAT_NCHW) {
        return ConvolutionBackprop::CubeFormat::NCHW;
    }else if(format == FORMAT_FRACTAL_Z_C04) {
        return ConvolutionBackprop::CubeFormat::FRACTALZ_C04;
    }
    return ConvolutionBackprop::CubeFormat::NC1HWC0;
}

template <typename xType, int xFormat, typename dedyType, int dedyFormat, typename yType, int yFormat>
class Conv2dDw {
public:
    __aicore__ inline Conv2dDw(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dedy,
                                GM_ADDR y, GM_ADDR workSpace,
                                const Conv2DBackpropFilterV2TilingData* tilingData) {
        InitTilingData(tilingData);
        // init global buffer
        xGm_.SetGlobalBuffer((__gm__ xType*)x);
        dedyGm_.SetGlobalBuffer((__gm__ dedyType*)dedy);
        yGm_.SetGlobalBuffer((__gm__ yType*)y);
        dw_.Init(&(tilingData->dwTiling));
    }

    /** main logical function
    */
    __aicore__ inline void Process() {
        if ASCEND_IS_AIV {
            return;
        }
        usedCoreNum_ = batchDim_ * mDim_ * nDim_ * kDim_;
        if (block_idx >= usedCoreNum_) {
            return;
        }

        CalSingleCoreShape();
        for (uint32_t i = 0; i < singleShapeBatch_; ++i) {
            CalcOffset(batchCoreIndx_ * singleCoreBatch_ + i);
            dw_.SetFmap(xGm_[offsetB_]);
            dw_.SetOutBackprop(dedyGm_[offsetA_]);
            dw_.SetSingleShape(singleShapeM_, singleShapeN_, singleShapeK_);
            dw_.IterateAll(yGm_[offsetC_], 1); // 1 means atomic add
        }
        dw_.End();
    }

protected:
    static constexpr ConvolutionBackprop::CubeFormat xCubeFormat = GetFormat(xFormat);
    static constexpr ConvolutionBackprop::CubeFormat dedyCubeFormat = GetFormat(dedyFormat);
    static constexpr ConvolutionBackprop::CubeFormat yCubeFormat = GetFormat(yFormat);
    using xDwType = ConvolutionBackprop::ConvType <TPosition::GM, xCubeFormat, xType>;
    using filterSizeDwType = ConvolutionBackprop::ConvType <TPosition::GM, ConvolutionBackprop::CubeFormat::ND, int32_t>;
    using dedyDwType = ConvolutionBackprop::ConvType <TPosition::GM, dedyCubeFormat, dedyType>;
    using yDwType = ConvolutionBackprop::ConvType <TPosition::GM, yCubeFormat, yType>;
    ConvolutionBackprop::Conv2DBackpropFilter <xDwType, filterSizeDwType, dedyDwType, yDwType> dw_;
    GlobalTensor<xType> xGm_;
    GlobalTensor<xType> dedyGm_;
    GlobalTensor<yType> yGm_;
    uint32_t batchOffset_;
    uint32_t groupOffset_;
    uint64_t offsetA_;
    uint64_t offsetB_;
    uint64_t offsetC_;
    uint32_t usedCoreNum_;
    uint32_t batchDim_;
    uint32_t kDim_;
    uint32_t mDim_;
    uint32_t nDim_;
    uint32_t batchCoreIndx_;
    uint32_t mCoreIndx_;
    uint32_t nCoreIndx_;
    uint32_t kCoreIndx_;
    uint32_t singleCoreBatch_;
    uint32_t singleCoreCout_;
    uint32_t singleCoreCin_;
    uint32_t singleCoreHo_;
    uint32_t batch_;
    uint32_t m_;
    uint64_t n_;
    uint64_t k_;
    uint32_t singleShapeBatch_;
    uint32_t singleShapeM_;
    uint64_t singleShapeN_;
    uint64_t singleShapeK_;
    uint32_t Cout_;
    uint32_t Cin_;
    uint32_t Ho_;
    uint32_t Wo_;
    uint32_t Hi_;
    uint32_t Wi_;
    uint32_t Hk_;
    uint32_t Wk_;
    uint32_t stride_h_;
    uint32_t padT_;

    __aicore__ inline void InitTilingData(const Conv2DBackpropFilterV2TilingData* tilingData) {
        batchDim_ = tilingData->params.batchDim;
        mDim_ = tilingData->params.mDim;
        nDim_ = tilingData->params.nDim;
        kDim_ = tilingData->params.kDim;
        batch_ = tilingData->dwTiling.N;
        m_ = tilingData->dwTiling.Cout;
        n_ = static_cast<uint64_t>(tilingData->dwTiling.Cin) * tilingData->dwTiling.Hk * tilingData->dwTiling.Wk;
        k_ = static_cast<uint64_t>(tilingData->dwTiling.Ho) * tilingData->dwTiling.Wo;
        singleCoreBatch_ = tilingData->dwTiling.singleCoreBatch;
        singleCoreCout_ = tilingData->dwTiling.singleCoreCout;
        singleCoreCin_ = tilingData->dwTiling.singleCoreCin;
        singleCoreHo_ = tilingData->dwTiling.singleCoreHo;
        Cout_ = tilingData->dwTiling.Cout;
        Cin_ = tilingData->dwTiling.Cin;
        Ho_ = tilingData->dwTiling.Ho;
        Wo_ = tilingData->dwTiling.Wo;
        Hi_ = tilingData->dwTiling.Hi;
        Wi_ = tilingData->dwTiling.Wi;
        Hk_ = tilingData->dwTiling.Hk;
        Wk_ = tilingData->dwTiling.Wk;
        stride_h_ = tilingData->dwTiling.strideH;
        padT_  = tilingData->dwTiling.padT;
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    __aicore__ inline void CalcOffset(uint32_t batchIdx) {
        uint64_t hoOffset = static_cast<uint64_t>(kCoreIndx_) * singleCoreHo_ * Wo_ * BLOCK_CUBE;
        uint64_t coutOffset = static_cast<uint64_t>(mCoreIndx_) * singleCoreCout_ * Ho_ * Wo_;
        uint64_t batchOffsetA = static_cast<uint64_t>(batchIdx) * Cout_ * Ho_ * Wo_;
        offsetA_ = batchOffsetA + coutOffset + hoOffset;

        uint64_t Hiidx = static_cast<uint64_t>(kCoreIndx_) * singleCoreHo_ * stride_h_ - padT_;
        uint64_t hiOffset = Hiidx * Wi_ * BLOCK_CUBE;
        uint64_t cinOffset = static_cast<uint64_t>(nCoreIndx_) * singleCoreCin_ * Hi_ * Wi_;
        uint64_t batchOffsetB = static_cast<uint64_t>(batchIdx) * BLOCK_CUBE * Hi_ * Wi_;
        offsetB_ = batchOffsetB + cinOffset + hiOffset;
        offsetC_ = static_cast<uint64_t>(nCoreIndx_) * singleCoreCin_ * Hk_ * Wk_ * Cout_ + static_cast<uint64_t>(mCoreIndx_) * singleCoreCout_ * BLOCK_C04;
    }
#else
    __aicore__ inline void CalcOffset(uint32_t batchIdx) {
        uint64_t hoOffset = static_cast<uint64_t>(kCoreIndx_) * singleCoreHo_ * Wo_;
        uint64_t coutOffset = static_cast<uint64_t>(mCoreIndx_) * singleCoreCout_ * Ho_ * Wo_;
        uint64_t batchOffsetA = static_cast<uint64_t>(batchIdx) * Cout_ * Ho_ * Wo_;
        offsetA_ = batchOffsetA + coutOffset + hoOffset;

        uint64_t Hiidx = static_cast<uint64_t>(kCoreIndx_) * singleCoreHo_ * stride_h_ - padT_;
        uint64_t hiOffset = Hiidx * Wi_;
        uint64_t cinOffset = static_cast<uint64_t>(nCoreIndx_) * singleCoreCin_ * Hi_ * Wi_;
        uint64_t batchOffsetB = static_cast<uint64_t>(batchIdx) * Cin_ * Hi_ * Wi_;
        offsetB_ = batchOffsetB + cinOffset + hiOffset;
        offsetC_ = static_cast<uint64_t>(nCoreIndx_) * singleCoreCin_ * Hk_ * Wk_ * Cout_ + static_cast<uint64_t>(mCoreIndx_) * singleCoreCout_* Cin_ * Hk_ * Wk_;
    }
#endif

    __aicore__ inline void CalSingleCoreShape() {
        kCoreIndx_ = block_idx % kDim_;
        nCoreIndx_ = (block_idx / kDim_) % nDim_;
        mCoreIndx_ = (block_idx / (kDim_ * nDim_)) % mDim_;
        batchCoreIndx_ = (block_idx / (kDim_ * nDim_ * mDim_)) % batchDim_;
        uint32_t batchRamin = batch_ - batchCoreIndx_ * singleCoreBatch_;
        uint32_t mRamin = m_ - mCoreIndx_ * singleCoreCout_;
        uint64_t nRamin = n_ - static_cast<uint64_t>(nCoreIndx_) * singleCoreCin_ * Hk_ * Wk_;
        uint64_t kRamin = k_ - static_cast<uint64_t>(kCoreIndx_) * singleCoreHo_ * Wo_;
        singleShapeBatch_ = batchRamin < singleCoreBatch_ ? batchRamin : singleCoreBatch_;
        singleShapeM_ = mRamin < singleCoreCout_ ? mRamin : singleCoreCout_;
        singleShapeN_ = nRamin < static_cast<uint64_t>(singleCoreCin_) * Hk_ * Wk_ ?
                        nRamin : static_cast<uint64_t>(singleCoreCin_) * Hk_ * Wk_;
        singleShapeK_ = kRamin < static_cast<uint64_t>(singleCoreHo_) * Wo_ ?
                        kRamin : static_cast<uint64_t>(singleCoreHo_) * Wo_;
    }
};
}

#endif // CONV2D_BACKPROP_FILTER_H