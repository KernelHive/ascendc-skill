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
 * \file conv3d_backprop_input_v2.h
 * \brief
 */
#ifndef CONV3D_BACKPROP_INPUT_V2_H
#define CONV3D_BACKPROP_INPUT_V2_H

#include "conv3d_bp_input.h"
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "kernel_type.h"
#include "lib/matmul_intf.h"

namespace AscendC {

using Conv3dConfig = typename Convolution3DBackprop::Conv3dConfig;

__aicore__ inline constexpr Convolution3DBackprop::CubeFormat GetFormat(int format)
{
    if (format == FORMAT_FRACTAL_Z_3D) {
        return Convolution3DBackprop::CubeFormat::FRACTALZ_3D;
    } else if (format == FORMAT_NCDHW) {
        return Convolution3DBackprop::CubeFormat::NCDHW;
    }
    return Convolution3DBackprop::CubeFormat::NDC1HWC0;
}

template <typename filterType, int filterFormat, typename dedyType, int dedyFormat, typename yType, int yFormat,
Convolution3DBackprop::B2Condition b2Condition, bool enableKernelSplit = false>
class Conv3dDx {
public:
    __aicore__ inline Conv3dDx(){};
    __aicore__ inline void Init(GM_ADDR filter, GM_ADDR dedy, GM_ADDR y, GM_ADDR workSpace,
                                const Conv3DBackpropInputV2TilingData *tilingData)
    {
        if constexpr (yFormat == FORMAT_NDC1HWC0) {
            if ASCEND_IS_AIV {
                return;
            }
        }
        InitTilingData(tilingData);

        filterGm_.SetGlobalBuffer((__gm__ filterType *)filter);
        dedyGm_.SetGlobalBuffer((__gm__ dedyType *)dedy);
        yGm_.SetGlobalBuffer((__gm__ yType *)y);
        dedx_.Init(&(tilingData->conv3DDxTiling));
#if __CCE_AICORE__ == 220
        InitMixCoreBuffer(workSpace);
#endif
    }

    __aicore__ inline void Process()
    {
        if constexpr (yFormat == FORMAT_NDC1HWC0) {
            if ASCEND_IS_AIV {
                return;
            }
        }
        CalSingleCoreShape();
        if (alignedHoStartIdx_ >= ho_) {
            dedx_.End();
            return;
        }
        dedx_.SetStartIdx(curDinStartIdx_, curHoStartIdx_);
        CalcBlockOffset(batchCoreIndx_ * singleCoreBatch_);

        if constexpr (enableKernelSplit) {
            uint64_t kernelSplitOffsetA = 0;
            uint64_t kernelSplitOffsetB = 0;
            uint64_t kernelSplitOffsetC = 0;
            dedx_.SetSingleShape(singleShapeM_, singleShapeK_, singleShapeN_, singleShapeDin_);
            for (uint32_t batchIdx = 0; batchIdx < singleShapeBatch_; ++batchIdx) {
                ProcessKernelSplit(kernelSplitOffsetA, kernelSplitOffsetB, kernelSplitOffsetC);
                CalcBatchOffset();
            }
        } else {
            if (likely(group_ == 1)) {
                dedx_.SetSingleShape(singleShapeM_, singleShapeK_, singleShapeN_, singleShapeDin_);
                for (uint32_t batchIdx = 0; batchIdx < singleShapeBatch_; ++batchIdx) {
                    dedx_.SetOutBackprop(dedyGm_[offsetA_]);
                    dedx_.SetWeight(filterGm_[offsetB_]);
                    DedxIterateAll();
                    CalcBatchOffset();
                }
            } else {
                ProcessGroup();
            }
        }

        dedx_.End();
    }

    __aicore__ inline void ProcessGroup()
    {
        for (uint32_t batchIdx = 0; batchIdx < singleShapeBatch_; ++batchIdx) {
            uint64_t backupOffsetA = offsetA_;
            uint64_t backupOffsetB = offsetB_;
            uint64_t backupOffsetC = offsetC_;
            for (uint32_t groupIdx = 0; groupIdx < singleShapeGroup_; ++groupIdx) {
                CalSingleCoreShapeInGroup(groupIdx);
                if (singleShapeKInGroup_ == 0 || singleShapeNInGroup_ == 0) {
                    continue;
                }
                dedx_.SetSingleShape(singleShapeM_, singleShapeKInGroup_, singleShapeNInGroup_, singleShapeDin_);
                dedx_.SetOutBackprop(dedyGm_[offsetA_]);
                dedx_.SetWeight(filterGm_[offsetB_]);
                DedxIterateAll();
                CalcGroupOffset();
            }
            CalcBatchOffset(backupOffsetA, backupOffsetB, backupOffsetC);
        }
    }

protected:
    static constexpr Convolution3DBackprop::CubeFormat filterCubeFormat = GetFormat(filterFormat);
    static constexpr Convolution3DBackprop::CubeFormat dedyCubeFormat = GetFormat(dedyFormat);
    static constexpr Convolution3DBackprop::CubeFormat yCubeFormat = GetFormat(yFormat);
    using filterDxType = Convolution3DBackprop::ConvType<TPosition::GM, filterCubeFormat, filterType>;
    using inputSizeDxType =
        Convolution3DBackprop::ConvType<TPosition::GM, Convolution3DBackprop::CubeFormat::ND, int32_t>;
    using dedyDxType = Convolution3DBackprop::ConvType<TPosition::GM, dedyCubeFormat, dedyType>;
    using yDxType = Convolution3DBackprop::ConvType<TPosition::GM, yCubeFormat, yType>;
    static constexpr Conv3dConfig conv3dConfig = {b2Condition, enableKernelSplit};
    Convolution3DBackprop::Conv3DBackpropInput<filterDxType, inputSizeDxType, dedyDxType, yDxType, conv3dConfig> dedx_;

    static constexpr uint8_t SYNC_MODE2 = 2;
    static constexpr uint16_t SYNC_AIC_AIV_FLAG = 6;
    static constexpr uint16_t SYNC_AIV_AIC_FLAG = 5;

    GlobalTensor<filterType> filterGm_;
    GlobalTensor<filterType> dedyGm_;
    GlobalTensor<yType> yGm_;
    GlobalTensor<yType> l0cOutGm_; // tmp workspace to store result data with format C1HWC0
    TQue<QuePosition::VECIN, 1> vecInQueue_;
    TQue<QuePosition::VECOUT, 1> vecOutQueue_;

    uint64_t cout1StrideA_ = 0;
    uint64_t groupStrideA_ = 0;
    uint64_t doStrideA_ = 0;
    uint64_t batchStrideA_ = 0;
    uint64_t groupStrideB_ = 0;
    uint64_t cin1StrideC_ = 0;
    uint64_t groupStrideC_ = 0;
    uint64_t diStrideC_ = 0;
    uint64_t batchStrideC_ = 0;
    uint64_t offsetA_ = 0;
    uint64_t offsetB_ = 0;
    uint64_t offsetC_ = 0;
    uint32_t usedCoreNum_ = 0;
    uint32_t batchDim_ = 0;
    uint32_t kDim_ = 0;
    uint32_t mDim_ = 0;
    uint32_t nDim_ = 0;
    uint32_t dDim_ = 0;
    uint32_t groupDim_ = 0;
    uint32_t batchCoreIndx_ = 0;
    uint32_t mCoreIndx_ = 0;
    uint32_t nCoreIndx_ = 0;
    uint32_t kCoreIndx_ = 0;
    uint32_t dCoreIndx_ = 0;
    uint32_t groupCoreIndx_ = 0;
    uint64_t singleCoreBatch_ = 0;
    uint32_t singleCoreCout_ = 0;
    uint32_t singleCoreCin_ = 0;
    uint32_t singleCoreCin1_ = 0;
    uint32_t singleCoreHi_ = 0;
    uint64_t singleCoreM_ = 0;
    uint32_t singleCoreDin_ = 0;
    uint32_t singleCoreGroup_ = 0;
    uint32_t curDinStartIdx_ = 0;
    int32_t curHoStartIdx_ = 0;
    uint32_t alignedHoStartIdx_ = 0;
    uint64_t batch_ = 0;
    uint64_t m_ = 0;
    uint32_t n_ = 0;
    uint32_t k_ = 0;
    uint64_t singleShapeBatch_ = 0;
    uint64_t singleShapeM_ = 0;
    uint32_t singleShapeN_ = 0;
    uint64_t singleShapeK_ = 0;
    uint32_t singleShapeCout1_ = 0;
    uint32_t singleShapeDin_ = 0;
    uint32_t singleShapeGroup_ = 0;
    uint32_t singleShapeNInGroup_ = 0;
    uint64_t singleShapeKInGroup_ = 0;
    uint32_t cin_ = 0;
    uint32_t cout1_ = 0;
    uint32_t cin1_ = 0;
    uint32_t cout1G_ = 0;
    uint32_t cin1G_ = 0;
    uint32_t c0_ = 0;
    uint32_t ho_ = 0;
    uint32_t wo_ = 0;
    uint32_t hi_ = 0;
    uint32_t wi_ = 0;
    uint32_t hk_ = 0;
    uint32_t wk_ = 0;
    uint32_t dk_ = 0;
    uint32_t di_ = 0;
    uint32_t do_ = 0;
    uint32_t group_ = 0;
    uint32_t dilationD_ = 0;
    uint32_t dilationH_ = 0;
    uint32_t strideH_ = 0;
    uint32_t strideW_ = 0;
    uint32_t padFront_ = 0;
    uint32_t backpropPadTail_ = 0;
    uint32_t backpropPadUp_ = 0;
    uint32_t backpropPadDown_ = 0;
    bool isFirstIter_ = true;

    __aicore__ inline void InitTilingData(const Conv3DBackpropInputV2TilingData *tilingData)
    {
        batchDim_ = tilingData->params.batchDim;
        mDim_ = tilingData->params.mDim;
        nDim_ = tilingData->params.nDim;
        kDim_ = tilingData->params.kDim;
        dDim_ = tilingData->params.dDim;
        groupDim_ = tilingData->params.groupDim;
        
        batch_ = tilingData->conv3DDxTiling.batch;
        m_ = static_cast<uint64_t>(tilingData->conv3DDxTiling.hi) * tilingData->conv3DDxTiling.wi;
        n_ = tilingData->conv3DDxTiling.cin1 * tilingData->conv3DDxTiling.c0;
        k_ = static_cast<uint64_t>(tilingData->conv3DDxTiling.dk) * tilingData->conv3DDxTiling.cout1 * tilingData->conv3DDxTiling.hk *
             tilingData->conv3DDxTiling.wk * tilingData->conv3DDxTiling.c0;
        group_ = tilingData->conv3DDxTiling.group;
        singleCoreBatch_ = tilingData->conv3DDxTiling.singleCoreBatch;
        singleCoreCout_ = tilingData->conv3DDxTiling.singleCoreCout;
        singleCoreCin_ = tilingData->conv3DDxTiling.singleCoreCin;
        singleCoreCin1_ = tilingData->conv3DDxTiling.singleCoreCin1;
        singleCoreM_ = tilingData->conv3DDxTiling.singleCoreM;
        singleCoreDin_ = tilingData->conv3DDxTiling.singleCoreDin;
        singleCoreGroup_ = tilingData->conv3DDxTiling.singleCoreGroup;
        cin_ = tilingData->conv3DDxTiling.cin;
        cout1_ = tilingData->conv3DDxTiling.cout1;
        cin1_ = tilingData->conv3DDxTiling.cin1;
        cout1G_ = tilingData->conv3DDxTiling.cout1G;
        cin1G_ = tilingData->conv3DDxTiling.cin1G;
        c0_ = tilingData->conv3DDxTiling.c0;
        ho_ = tilingData->conv3DDxTiling.ho;
        wo_ = tilingData->conv3DDxTiling.wo;
        do_ = tilingData->conv3DDxTiling.dout;
        hi_ = tilingData->conv3DDxTiling.hi;
        wi_ = tilingData->conv3DDxTiling.wi;
        di_ = tilingData->conv3DDxTiling.di;
        hk_ = tilingData->conv3DDxTiling.hk;
        wk_ = tilingData->conv3DDxTiling.wk;
        dk_ = tilingData->conv3DDxTiling.dk;
        dilationD_ = tilingData->conv3DDxTiling.dilationD;
        dilationH_ = tilingData->conv3DDxTiling.dilationH;
        strideH_ = tilingData->conv3DDxTiling.strideH;
        strideW_ = tilingData->conv3DDxTiling.strideW;
        padFront_ = tilingData->conv3DDxTiling.padFront;
        backpropPadTail_ = tilingData->conv3DDxTiling.backpropPadTail;
        backpropPadUp_ = tilingData->conv3DDxTiling.backpropPadUp;
        backpropPadDown_ = tilingData->conv3DDxTiling.backpropPadDown;
        singleCoreHi_ = singleCoreM_ / wi_;
#ifdef __CCE_KT_TEST__
        ASCENDC_ASSERT(singleCoreM_ % wi_ == 0, { KERNEL_LOG(KERNEL_ERROR, "singleCoreM_ % wi_ > 0"); });
#endif
    }

    __aicore__ inline void CalcBlockOffset(uint32_t batchIdx)
    {
        uint64_t groupOffsetNum = static_cast<uint64_t>(groupCoreIndx_) * static_cast<uint64_t>(singleCoreGroup_);
        cout1StrideA_ = static_cast<uint64_t>(ho_) * static_cast<uint64_t>(wo_) * static_cast<uint64_t>(c0_);
        groupStrideA_ = static_cast<uint64_t>(cout1G_) * cout1StrideA_;
        doStrideA_ = static_cast<uint64_t>(cout1_) * cout1StrideA_;
        batchStrideA_ = static_cast<uint64_t>(do_) * doStrideA_;
        uint64_t groupOffsetA = groupOffsetNum * groupStrideA_;
        uint64_t batchOffsetA = static_cast<uint64_t>(batchIdx) * batchStrideA_;
        uint64_t doutOffsetA = 0;
        uint64_t coutOffsetA = static_cast<uint64_t>(kCoreIndx_) * static_cast<uint64_t>(singleCoreCout_) *
                               static_cast<uint64_t>(ho_) * static_cast<uint64_t>(wo_);
        uint64_t mOffsetA = static_cast<uint64_t>(alignedHoStartIdx_) * static_cast<uint64_t>(wo_) *
                            static_cast<uint64_t>(c0_);
        offsetA_ = batchOffsetA + groupOffsetA + doutOffsetA + coutOffsetA + mOffsetA;

        // FP32场景下，cout1_ * c0_可能非16对齐，此时要补齐到16，原始数据GM中自动补0，计算偏移时要考虑对齐后的数据
        uint64_t cinStrideB = static_cast<uint64_t>(hk_ * wk_) * DivCeil(cout1G_ * c0_, BLOCK_CUBE) *
            static_cast<uint64_t>(BLOCK_CUBE * c0_);
        uint64_t cinOffsetB = static_cast<uint64_t>(nCoreIndx_) * static_cast<uint64_t>(singleCoreCin1_) * cinStrideB;
        groupStrideB_ = static_cast<uint64_t>(dk_) * static_cast<uint64_t>(cin1G_) * cinStrideB;
        uint64_t groupOffsetB = groupOffsetNum * groupStrideB_;
        uint64_t dkOffsetB = 0;
        offsetB_ = groupOffsetB + cinOffsetB + dkOffsetB;

        if (yFormat == FORMAT_NCDHW) {
            diStrideC_ = static_cast<uint64_t>(hi_) * static_cast<uint64_t>(wi_);
            uint64_t cinStrideC_ = static_cast<uint64_t>(di_) * diStrideC_;
            batchStrideC_ = static_cast<uint64_t>(cin_) * cinStrideC_;
            uint64_t batchOffsetC = static_cast<uint64_t>(batchIdx) * batchStrideC_;
            groupStrideC_ = static_cast<uint64_t>(cin1G_) * static_cast<uint64_t>(c0_) * cinStrideC_;
            uint64_t groupOffsetC = groupOffsetNum * groupStrideC_;
            uint64_t cinOffsetC = static_cast<uint64_t>(nCoreIndx_) * static_cast<uint64_t>(singleCoreCin_) * cinStrideC_;
            uint64_t dinOffsetC = 0;
            uint64_t mOffsetC = static_cast<uint64_t>(mCoreIndx_) * static_cast<uint64_t>(singleCoreHi_) *
                                static_cast<uint64_t>(wi_);
            offsetC_ = batchOffsetC + groupOffsetC + cinOffsetC + dinOffsetC + mOffsetC;
        } else {
            cin1StrideC_ = static_cast<uint64_t>(hi_) * static_cast<uint64_t>(wi_) * static_cast<uint64_t>(c0_);
            groupStrideC_ = static_cast<uint64_t>(cin1G_) * cin1StrideC_;
            diStrideC_ = static_cast<uint64_t>(cin1_) * cin1StrideC_;
            batchStrideC_ = static_cast<uint64_t>(di_) * diStrideC_;
            uint64_t groupOffsetC = groupOffsetNum * groupStrideC_;
            uint64_t batchOffsetC = static_cast<uint64_t>(batchIdx) * batchStrideC_;
            uint64_t dinOffsetC = 0;
            uint64_t cinOffsetC = static_cast<uint64_t>(nCoreIndx_) * static_cast<uint64_t>(singleCoreCin1_) * cin1StrideC_;
            uint64_t mOffsetC = static_cast<uint64_t>(mCoreIndx_) * static_cast<uint64_t>(singleCoreHi_) *
                                static_cast<uint64_t>(wi_) * static_cast<uint64_t>(c0_);
            offsetC_ = batchOffsetC + groupOffsetC + dinOffsetC + cinOffsetC + mOffsetC;
        }
    }

    __aicore__ inline void CalcBatchOffset()
    {
        offsetA_ += batchStrideA_;
        offsetC_ += batchStrideC_;
    }

    __aicore__ inline void CalcBatchOffset(uint64_t &backupOffsetA_, uint64_t &backupOffsetB_,
        uint64_t &backupOffsetC_)
    {
        offsetA_ = backupOffsetA_  + batchStrideA_;
        offsetB_ = backupOffsetB_;
        offsetC_ = backupOffsetC_  + batchStrideC_;
    }

    __aicore__ inline void CalcGroupOffset()
    {
        offsetA_ += groupStrideA_;
        offsetB_ += groupStrideB_;
        offsetC_ += groupStrideC_;
    }

    __aicore__ inline void ProcessKernelSplit(uint64_t &kernelSplitOffsetA, uint64_t &kernelSplitOffsetB,
        uint64_t &kernelSplitOffsetC)
    {
        for (uint32_t kernelHIdx = 0; kernelHIdx < strideH_; ++kernelHIdx) {
            for (uint32_t kernelWIdx = 0; kernelWIdx < strideW_; ++kernelWIdx) {
                CalcKernelSplitOffset(kernelHIdx, kernelWIdx, kernelSplitOffsetA,
                    kernelSplitOffsetB, kernelSplitOffsetC);
                dedx_.SetOutBackprop(dedyGm_[kernelSplitOffsetA]);
                dedx_.SetWeight(filterGm_[kernelSplitOffsetB]);
                dedx_.IterateAll(yGm_[kernelSplitOffsetC], 0);  // 1 means atomic add
            }
        }
    }

    __aicore__ inline void CalcKernelSplitOffset(uint32_t kernelHIdx, uint32_t kernelWIdx, uint64_t &kernelSplitOffsetA,
        uint64_t &kernelSplitOffsetB, uint64_t &kernelSplitOffsetC)
    {
        uint64_t kernelHOffsetA = (static_cast<uint64_t>(kernelHIdx) * wo_ * c0_);
        uint64_t kernelWOffsetA = (static_cast<uint64_t>(kernelWIdx) * c0_);
        kernelSplitOffsetA = (offsetA_ + kernelHOffsetA + kernelWOffsetA);

        uint64_t kernelHOffsetB = static_cast<uint64_t>(strideH_ - 1 - kernelHIdx) * wk_ * c0_ * cout1_ * c0_;
        uint64_t kernelWOffsetB = static_cast<uint64_t>(strideW_ - 1 - kernelWIdx) * c0_ * cout1_ * c0_;
        kernelSplitOffsetB = (offsetB_ + kernelHOffsetB + kernelWOffsetB);

        uint64_t kernelHOffsetC = (static_cast<uint64_t>(kernelHIdx) * wi_ * c0_);
        uint64_t kernelWOffsetC = (static_cast<uint64_t>(kernelWIdx) * c0_);
        kernelSplitOffsetC = (offsetC_ + kernelHOffsetC + kernelWOffsetC);
    }

    __aicore__ inline void ReCalSingleCoreShapeForOutputPad(uint64_t &mTail, uint32_t &dTail)
    {
        int32_t diRemain = static_cast<int32_t>(backpropPadTail_ - (dilationD_ * (dk_ - 1)));
        if (dTail > 0 && dTail > diRemain) {
            dTail = dTail - diRemain;
        }

        int32_t hiRemain = static_cast<int32_t>(backpropPadDown_ - (dilationH_ * (hk_ - 1)));
        if (mTail > 0 && mTail > hiRemain * wi_) {
            int32_t hiLen = hiRemain * wi_;
            mTail = (mTail > hiLen) ? mTail - hiLen : 0;
        }
    }

    __aicore__ inline void CalSingleCoreShape()
    {
        kCoreIndx_ = block_idx % kDim_;
        nCoreIndx_ = (block_idx / kDim_) % nDim_;
        mCoreIndx_ = (block_idx / (kDim_ * nDim_)) % mDim_;
        batchCoreIndx_ = (block_idx / (kDim_ * nDim_ * mDim_)) % batchDim_;
        dCoreIndx_ = (block_idx / (kDim_ * nDim_ * mDim_ * batchDim_)) % dDim_;
        groupCoreIndx_ = (block_idx / (kDim_ * nDim_ * mDim_ * batchDim_ * dDim_)) % groupDim_;
        // 放大后的绝对坐标
        curHoStartIdx_ = static_cast<int32_t>(mCoreIndx_ * singleCoreHi_) - backpropPadUp_;
        alignedHoStartIdx_ = curHoStartIdx_ < 0 ? 0 : ((curHoStartIdx_ + strideH_ -1) / strideH_);

        uint64_t batchTail = batch_ - batchCoreIndx_ * singleCoreBatch_;
        uint64_t mTail = m_ - mCoreIndx_ * singleCoreM_;
        uint32_t dTail = di_ - dCoreIndx_ * singleCoreDin_;
        uint32_t groupTail = group_ - groupCoreIndx_ * singleCoreGroup_;
        ReCalSingleCoreShapeForOutputPad(mTail, dTail);

        singleShapeBatch_ = Min(batchTail, singleCoreBatch_);
        singleShapeM_ = Min(mTail, singleCoreM_);
        singleShapeDin_ = Min(dTail, singleCoreDin_);
        singleShapeGroup_ = Min(groupTail, singleCoreGroup_);
        curDinStartIdx_ = dCoreIndx_ * singleCoreDin_;
        if constexpr (enableKernelSplit) {
            uint32_t nTail = cin_ - nCoreIndx_ * singleCoreCin1_ * c0_;
            singleShapeN_ = Min(nTail, singleCoreCin1_ * c0_);
            singleShapeK_ = k_;
        } else {
            if (group_ == 1) {
                uint32_t nTail = cin_ - nCoreIndx_ * singleCoreCin1_ * c0_;
                singleShapeN_ = Min(nTail, singleCoreCin1_ * c0_);
                singleShapeK_ = k_;
            } else {
                uint32_t nTail = cin_ - groupCoreIndx_ * singleCoreGroup_ * cin1G_ * c0_;
                uint32_t cout1Tail = cout1_ - groupCoreIndx_ * singleCoreGroup_ * cout1G_;
                singleShapeN_ = Min(nTail, singleCoreGroup_ * cin1G_ * c0_);
                singleShapeCout1_ = Min(cout1Tail, singleShapeGroup_ * cout1G_);
                singleShapeK_ = singleShapeCout1_ * dk_ * hk_ * wk_;
            }
        }
    }

    __aicore__ inline void CalSingleCoreShapeInGroup(uint32_t groupIdx)
    {
        uint32_t cout1Tail = singleShapeCout1_ - groupIdx * cout1G_;
        uint32_t nTail = singleShapeN_ - groupIdx * cin1G_ * c0_;
        if (cout1Tail == 0) {
            singleShapeKInGroup_ = 0;
            return;
        }
        if (nTail <= nCoreIndx_ * singleCoreCin1_ * c0_) {
            singleShapeNInGroup_ = 0;
            return;
        }
        uint32_t singleShapeCout1InGroup = Min(cout1Tail, cout1G_);
        singleShapeKInGroup_ = static_cast<uint64_t>(singleShapeCout1InGroup) * dk_ * hk_ * wk_ * c0_;
        nTail -= nCoreIndx_ * singleCoreCin1_ * c0_;
        singleShapeNInGroup_ = Min(nTail, singleCoreCin1_ * c0_);
    }

#if __CCE_AICORE__ == 220
    __aicore__ inline void InitMixCoreBuffer(GM_ADDR workSpace)
    {
        l0cOutGm_.SetGlobalBuffer((__gm__ yType *)workSpace);
        if ASCEND_IS_AIV {
            uint32_t halfUbSize = TOTAL_UB_SIZE / HALF_FACTOR;
            dedx_.ctx.pipe_.InitBuffer(vecInQueue_, 1, halfUbSize);
            dedx_.ctx.pipe_.InitBuffer(vecOutQueue_, 1, halfUbSize);
        }
    }

    __aicore__ inline void CopyInToUB()
    {
        if ASCEND_IS_AIC {
            return;
        }

        LocalTensor<yType> vecInBuf_ = vecInQueue_.template AllocTensor<yType>();
        int64_t srcOffset = block_idx * dedx_.ctx.tiling_->baseM * dedx_.ctx.tiling_->baseN;
        DataCopyParams loadGm2UbParams;
        loadGm2UbParams.srcStride = 0;
        loadGm2UbParams.dstStride = (dedx_.ctx.tiling_->baseM - dedx_.ctx.baseUseM_) * dedx_.ctx.tiling_->c0 *
            sizeof(yType) / ONE_BLK_SIZE;
        loadGm2UbParams.blockLen = static_cast<uint16_t>(dedx_.ctx.baseUseM_ * dedx_.ctx.tiling_->c0 *
            sizeof(yType) / ONE_BLK_SIZE);
        loadGm2UbParams.blockCount = static_cast<uint16_t>(Ceil(dedx_.ctx.baseUseN_, BLOCK_CUBE));
        DataCopy(vecInBuf_, l0cOutGm_[srcOffset], loadGm2UbParams);
        vecInQueue_.EnQue(vecInBuf_);
    }

    __aicore__ inline void TransFormat()
    {
        if ASCEND_IS_AIC {
            return;
        }

        LocalTensor<yType> vecInBuf_ = vecInQueue_.template DeQue<yType>();
        LocalTensor<yType> vecOutBuf_ = vecOutQueue_.template AllocTensor<yType>();
        TransDataTo5HDParams transDataParams;
        transDataParams.dstHighHalf = false;
        transDataParams.srcHighHalf = false;
        transDataParams.repeatTimes = static_cast<uint16_t>(Ceil(dedx_.ctx.baseUseM_, NCHW_CONV_ADDR_LIST_SIZE));
        transDataParams.dstRepStride = 1;
        transDataParams.srcRepStride = NCHW_CONV_ADDR_LIST_SIZE;
        // 参考AscendC API的使用说明，当repeatTimes为1时，repStride需要设置为0
        if (transDataParams.repeatTimes == 1) {
            transDataParams.dstRepStride = 0;
            transDataParams.srcRepStride = 0;
        }
        uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
        uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
        int64_t baseCount = 0;
        int64_t dstCount = 0;
        int64_t srcCount = 0;
        int loopCount = (dedx_.ctx.baseUseN_ >> dedx_.ctx.tiling_->c0Bits);
        uint64_t baseCountIncrement = (dedx_.ctx.tiling_->baseM << dedx_.ctx.tiling_->c0Bits);
        for (int j = 0; j < loopCount; j++) {
            dstCount = baseCount;
            for (int i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
                dstLocalList[i] = reinterpret_cast<uint64_t>(vecOutBuf_[dstCount].GetPhyAddr());
                dstCount += dedx_.ctx.tiling_->baseM;
            }
            srcCount = baseCount;
            for (int i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
                srcLocalList[i] = reinterpret_cast<uint64_t>(vecInBuf_[srcCount].GetPhyAddr());
                srcCount += dedx_.ctx.tiling_->c0;
            }
            TransDataTo5HD<half>(dstLocalList, srcLocalList, transDataParams);
            baseCount += baseCountIncrement;
        }
        vecOutQueue_.EnQue(vecOutBuf_);
        vecInQueue_.FreeTensor(vecInBuf_);
    }

    __aicore__ inline void CopyOutToGm(const GlobalTensor<yType> &output)
    {
        if ASCEND_IS_AIC {
            return;
        }

        LocalTensor<yType> vecOutBuf_ = vecOutQueue_.template DeQue<yType>();
        uint64_t diHiWi = static_cast<uint64_t>(dedx_.ctx.tiling_->di) * dedx_.ctx.tiling_->hi * dedx_.ctx.tiling_->wi;
        uint64_t dstStride = (diHiWi - dedx_.ctx.baseUseM_) * sizeof(yType);
        uint64_t dstOffset = static_cast<uint64_t>(dedx_.ctx.curNL0Idx_) * dedx_.ctx.tiling_->baseN * diHiWi +
            static_cast<uint64_t>(dedx_.ctx.curML0Idx_) * dedx_.ctx.tiling_->baseM +
            static_cast<uint64_t>(dedx_.ctx.curDinIdx_) * dedx_.ctx.tiling_->hi * dedx_.ctx.tiling_->wi;
        uint32_t curCinSize = dedx_.ctx.baseUseN_ < (dedx_.ctx.singleShapeCin_ - dedx_.ctx.curNL0Idx_ * dedx_.ctx.tiling_->baseN) ?
            dedx_.ctx.baseUseN_ : (dedx_.ctx.singleShapeCin_ - dedx_.ctx.curNL0Idx_ * dedx_.ctx.tiling_->baseN);
        DataCopyExtParams storeUb2GmParams;
        // 用&f 代替对16取余
        if (((dedx_.ctx.baseUseM_ & 0xf) == 0) && dstStride <= UINT32_MAX) {
            storeUb2GmParams.srcStride = (dedx_.ctx.tiling_->baseM - dedx_.ctx.baseUseM_) * sizeof(yType) / ONE_BLK_SIZE;
            storeUb2GmParams.dstStride = dstStride;
            storeUb2GmParams.blockLen = dedx_.ctx.baseUseM_ * sizeof(yType);
            storeUb2GmParams.blockCount = curCinSize;
            DataCopyPad(output[dstOffset], vecOutBuf_, storeUb2GmParams);
        } else {
            uint32_t ubOffset = 0;
            storeUb2GmParams.srcStride = 0;
            storeUb2GmParams.dstStride = 0;
            storeUb2GmParams.blockLen = dedx_.ctx.baseUseM_ * sizeof(yType);
            storeUb2GmParams.blockCount = 1;
            for (uint32_t n = 0; n < curCinSize; n++) {
                DataCopyPad(output[dstOffset], vecOutBuf_[ubOffset], storeUb2GmParams);
                dstOffset += diHiWi;
                ubOffset += dedx_.ctx.tiling_->baseM;
            }
        }
        vecOutQueue_.FreeTensor(vecOutBuf_);
    }

    __aicore__ inline bool JudgeComputeNecessary()
    {
        for (uint64_t curKdIdx = 0; curKdIdx < dedx_.ctx.tiling_->dk; curKdIdx++) {
            int64_t dTmp = 0;
            if (unlikely(dedx_.ctx.tiling_->strideD > dedx_.ctx.tiling_->dk)) {
                dTmp = dedx_.ctx.curDinIdx_ + dedx_.ctx.tiling_->padFront;
                if (CalcRemainder(dTmp, dedx_.ctx.tiling_->strideD) >= dedx_.ctx.tiling_->dk ||
                    dTmp / dedx_.ctx.tiling_->strideD >= dedx_.ctx.tiling_->dout) {
                    continue;
                }
            } else {
                dTmp = dedx_.ctx.curDinIdx_ + dedx_.ctx.tiling_->padFront - curKdIdx * dedx_.ctx.tiling_->dilationD;
                if (dTmp < 0 || CalcRemainder(dTmp, dedx_.ctx.tiling_->strideD) > 0 ||
                    dTmp >= dedx_.ctx.tiling_->dout * dedx_.ctx.tiling_->strideD) {
                    continue;
                }
            }
            return true;
        }
        return false;
    }

    __aicore__ inline void VecPostProcess(const GlobalTensor<yType> &output,
        uint8_t enAtomic = 0, bool enSequentialWrite = false)
    {
        if ASCEND_IS_AIC {
            return;
        }

        if (!JudgeComputeNecessary()) {
            CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG);
            return;
        }

        if (GetSubBlockIdx() > 0) {
            CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG);
            return;
        }
        if (unlikely(enAtomic == 1)) {
            SetAtomicAdd<yType>();
        }
        if constexpr (yFormat == FORMAT_NCDHW) {
            if (!enSequentialWrite) {
                CopyInToUB();
                CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG);
                TransFormat();
                CopyOutToGm(output);
            }
        }
        if (unlikely(enAtomic == 1)) {
            SetAtomicNone();
        }
    }

    __aicore__ inline void MergeOutputTransDataIterateAll(const GlobalTensor<yType> &output,
        uint8_t enAtomic = 0)
    {
        while (dedx_.Iterate()) {
            if ASCEND_IS_AIC {
                if (likely(!isFirstIter_)) {
                    CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG);
                }
                dedx_.GetTensorC(l0cOutGm_[0]);
                CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_AIV_FLAG);
            }

            if ASCEND_IS_AIV {
                CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG);
                VecPostProcess(output, enAtomic);
            }
            isFirstIter_ = false;
        }
        dedx_.ctx.isFirstIter_ = true;
    }
#endif

    __aicore__ inline void DedxIterateAll()
    {
#if __CCE_AICORE__ == 220
        if constexpr (yFormat == FORMAT_NDC1HWC0) {
            dedx_.IterateAll(yGm_[offsetC_], 0);  // 1 means atomic add
        } else if constexpr (yFormat == FORMAT_NCDHW) {
            MergeOutputTransDataIterateAll(yGm_[offsetC_], 0);
        }
#endif
    }

    template <typename T>
    __aicore__ inline T Max(T a, T b)
    {
        return a > b ? a : b;
    }

    template <typename T>
    __aicore__ inline T Min(T a, T b)
    {
        return a < b ? a : b;
    }
};
}  // namespace AscendC

#endif  // CONV3D_BACKPROP_INPUT_V2_H
