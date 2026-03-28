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
 * \file conv3d_mte1_hw_mode_sub_api.h
 * \brief
 */

#ifndef CONV3D_MTE1_HW_MODE_SUB_API_H
#define CONV3D_MTE1_HW_MODE_SUB_API_H

#include "conv3d_mte2_hw_mode_sub_api.h"

using namespace AscendC;
using namespace conv;
using namespace conv3d;

namespace Conv3dFunc {

template <class Intf>
class LoadAL0WithHwModeTools {
public:
    __aicore__ inline LoadAL0WithHwModeTools() {}

    __aicore__ inline void SetParams(Intf *self, LoadAL1WithHwModeTools<Intf> *aL1Tools)
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
        al0Set2dSpacesize_ = currentML0_ * currentKL0_ * self_->ctx.sizeOfFmap / BLOCK_SIZE;
        InitConstValueParams<typename Intf::FmapT> initConstValueParams(1, (uint16_t)al0Set2dSpacesize_, 0, 0);
        InitConstValue<typename Intf::FmapT>(self_->ctx.al0, initConstValueParams);
    }

private:
    __aicore__ inline void SetLoadData3DParamsV2(LoadData3DParamsV2<typename Intf::FmapT> &loadData3Dv2Params)
    {
        // params about k dicision
        loadData3Dv2Params.kExtension = currentKL0_;
        loadData3Dv2Params.kStartPt = self_->ctx.kAL0Iter * self_->ctx.singleCoreKL0;
        // params about m dicision
        loadData3Dv2Params.mExtension = currentML0_;
        loadData3Dv2Params.mStartPt = self_->ctx.mAL0Iter * self_->ctx.conv3dTiling->mL0;
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
    LoadAL1WithHwModeTools<Intf> *aL1Tools_;
    uint64_t currentML0_ = 0;
    uint64_t currentKL0_ = 0;
    uint64_t al0Set2dSpacesize_ = 0;
};

};  // namespace Conv3dFunc

#endif  // __CONV3D_MTE1_HW_MODE_SUB_API_H__
