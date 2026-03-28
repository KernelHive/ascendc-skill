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
 * \file conv3d_mte1_sub_api.h
 * \brief
 */

#ifndef CONV3D_MTE1_SUB_API_H
#define CONV3D_MTE1_SUB_API_H

#include "conv3d_mte2_sub_api.h"

using namespace AscendC;
using namespace conv;
using namespace conv3d;

namespace Conv3dFunc {

static constexpr IsResetLoad3dConfig CONV3D_LOAD3DV2_DEFAULT_CONFIG = {false, false};
template <class Intf>
class LoadAL0Tools {
public:
    __aicore__ inline LoadAL0Tools()
    {}

    __aicore__ inline void SetParams(Intf *self, LoadAL1Tools<Intf> *aL1Tools)
    {
        self_ = self;
        aL1Tools_ = aL1Tools;
    }

    __aicore__ inline void SetM(uint64_t m)
    {
        currentML0_ = m;
    }

    __aicore__ inline void LoadAL0()
    {
        currentKL0_ = self_->ctx.kIter == self_->ctx.maxKL0Iter ? self_->ctx.kL0Tail : self_->ctx.singleCoreKL0;
        // get if set 2d flag from L1load tools
        if (aL1Tools_->GetSet2dFlag()) {
            SetAl02d();
            return;
        }

        // img2col fmap param info, which set in each ddr->L1
        LoadData3DParamsV2<typename Intf::FmapT> loadData3Dv2Params = aL1Tools_->GetLoadData3DParams();
        SetLoadData3DParamsV2(loadData3Dv2Params);

        LoadData<typename Intf::FmapT, CONV3D_LOAD3DV2_DEFAULT_CONFIG>(
            self_->ctx.al0, self_->ctx.al1, loadData3Dv2Params);
    };

    __aicore__ inline void SetAl02d()
    {
        // only support bf16 now, set pad value to be 0
        if constexpr (Intf::quantType == static_cast<int8_t>(QuantType::PER_CHANNEL_NO_OFFSET)) {
            LocalTensor<int16_t> clearLocal = self_->ctx.al0.template ReinterpretCast<int16_t>();
            al0Set2dSpacesize_ = currentML0_ * currentKL0_ * self_->ctx.sizeOfFmap / BLOCK_SIZE;
            InitConstValueParams<int16_t> initConstValueParams(1, (uint16_t)al0Set2dSpacesize_, 0, 0);
            InitConstValue<int16_t>(clearLocal, initConstValueParams);
        } else {
            al0Set2dSpacesize_ = currentML0_ * currentKL0_ * self_->ctx.sizeOfFmap / BLOCK_SIZE;
            InitConstValueParams<typename Intf::FmapT> initConstValueParams(1, (uint16_t)al0Set2dSpacesize_, 0, 0);
            InitConstValue<typename Intf::FmapT>(self_->ctx.al0, initConstValueParams);
        }
    }

private:
    __aicore__ inline void SetLoadData3DParamsV2(LoadData3DParamsV2<typename Intf::FmapT> &loadData3Dv2Params)
    {
        // params about k dicision
        loadData3Dv2Params.kExtension = currentKL0_;
        loadData3Dv2Params.kStartPt = self_->ctx.kAL0Iter * self_->ctx.singleCoreKL0;
        // params about m dicision
        loadData3Dv2Params.mExtension = currentML0_;
        loadData3Dv2Params.mStartPt = self_->ctx.mL0IsDivisibleByWo ?
            self_->ctx.mAL0Iter * self_->ctx.conv3dTiling->mL0 + self_->ctx.mStartPos % self_->ctx.orgWo :
            self_->ctx.mAL0Iter * self_->ctx.conv3dTiling->mL0 +
            (self_->ctx.mStartPos + self_->ctx.mAL1Iter * self_->ctx.conv3dTiling->mAL1) % self_->ctx.orgWo;
        ASC_OP_LOGD(
            "[LoadAL0] loadData3Dv2Params.channelSize %d, loadData3Dv2Params.kExtension %d, "
            "loadData3Dv2Params.kStartPt %d, loadData3Dv2Params.mExtension %d, loadData3Dv2Params.mStartPt %d.\n",
            loadData3Dv2Params.channelSize,
            loadData3Dv2Params.kExtension,
            loadData3Dv2Params.kStartPt,
            loadData3Dv2Params.mExtension,
            loadData3Dv2Params.mStartPt);
    }

private:
    Intf *self_ = nullptr;
    LoadAL1Tools<Intf> *aL1Tools_;
    uint64_t currentML0_ = 0;
    uint64_t currentKL0_ = 0;
    uint64_t al0Set2dSpacesize_ = 0;
};

template <class Intf>
class LoadBL0Tools {
public:
    __aicore__ inline LoadBL0Tools()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
        k0_ = ONE_BLK_SIZE / self_->ctx.sizeOfWeight;
    }

    __aicore__ inline void SetN(uint64_t n)
    {
        currentNL0_ = n;
        numNL0Block_ = currentNL0_ / BLOCK_L0_N;
    }

    __aicore__ inline void LoadBL0()
    {
        if constexpr (Intf::bl1bypass) {
            tilingNBSrc_ = self_->ctx.orgCoAlignN0;
        } else {
            tilingNBSrc_ = self_->ctx.nBL1Iter == self_->ctx.maxNBL1Iter ? self_->ctx.nBL1TailAlign
                                                                         : self_->ctx.conv3dTiling->nBL1;
        }
        uint64_t currentKL0 =
            self_->ctx.kIter == self_->ctx.maxKL0Iter ? self_->ctx.kL0Tail : self_->ctx.singleCoreKL0;
        // set LoadData2DParamsV2
        LoadData2DParams loadData2dParams;
        load2dSrcOffset = self_->ctx.kBL0Iter * self_->ctx.singleCoreKL0 * tilingNBSrc_ +
                          self_->ctx.nBL0Iter * self_->ctx.conv3dTiling->nL0xk0;
        ASC_OP_LOGD("[LoadBL0] currentKL0: %u\n", currentKL0);
        if constexpr (!Intf::bl1bypass) {
            if (self_->ctx.conv3dTiling->nBL1 == self_->ctx.conv3dTiling->nL0) {
                SetLoadData2DParams(loadData2dParams, numNL0Block_ * (currentKL0 / k0_));
                LoadData<typename Intf::WeightT>(self_->ctx.bl0, self_->ctx.bl1[load2dSrcOffset], loadData2dParams);
            } else {
                SetLoadData2DParams(loadData2dParams, numNL0Block_);
                uint64_t tmp1 = currentNL0_ * k0_;
                uint64_t tmp2 = tilingNBSrc_ * k0_;
                for (uint32_t copy_part = 0; copy_part < currentKL0 / k0_; ++copy_part) {
                    LoadData<typename Intf::WeightT>(self_->ctx.bl0[copy_part * tmp1],
                        self_->ctx.bl1[load2dSrcOffset + copy_part * tmp2],
                        loadData2dParams);
                }
            }
        } else {
            if (self_->ctx.conv3dTiling->nL0 >= self_->ctx.orgCo) {
                SetLoadData2DParams(loadData2dParams, numNL0Block_ * (currentKL0 / k0_));
                LoadData<typename Intf::WeightT>(self_->ctx.bl0, self_->ctx.bgm[load2dSrcOffset], loadData2dParams);
            } else {
                SetLoadData2DParams(loadData2dParams, numNL0Block_);
                uint64_t tmp1 = currentNL0_ * k0_;
                uint64_t tmp2 = tilingNBSrc_ * k0_;
                for (uint32_t copy_part = 0; copy_part < currentKL0 / k0_; ++copy_part) {
                    LoadData<typename Intf::WeightT>(self_->ctx.bl0[copy_part * tmp1],
                        self_->ctx.bgm[load2dSrcOffset + copy_part * tmp2],
                        loadData2dParams);
                }
            }
        }
    }

private:
    __aicore__ inline void SetLoadData2DParams(LoadData2DParams &loadData2dParams, const uint64_t &repeatTimes)
    {
        loadData2dParams.repeatTimes = repeatTimes;
        loadData2dParams.srcStride = 1;
        ASC_OP_LOGD("[LoadBL0] loadData2dParams.repeatTimes %d.\n", loadData2dParams.repeatTimes);
    }

private:
    Intf *self_ = nullptr;
    uint64_t tilingNBSrc_ = 0;
    uint64_t k0_ = 16;
    uint64_t currentNL0_ = 0;
    uint64_t numNL0Block_ = 0;
    uint64_t ratioOfNToN0_ = 0;
    uint64_t load2dSrcOffset = 0;
};

};  // namespace Conv3dFunc

#endif  // __CcONV3D_MTE1_SUB_API_H__
