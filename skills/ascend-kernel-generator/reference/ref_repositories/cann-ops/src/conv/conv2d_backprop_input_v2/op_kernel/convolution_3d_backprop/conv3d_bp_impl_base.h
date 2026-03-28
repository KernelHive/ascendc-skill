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
 * \file conv3d_bp_impl_base.h
 * \brief
 */

#ifndef CONV3D_BP_IMPL_H
#define CONV3D_BP_IMPL_H

#include "conv3d_bp_config_base.h"
#include "conv3d_bp_func.h"
#include "conv3d_bp_util.h"
#include "kernel_operator.h"
#include "kernel_utils.h"

namespace Convolution3DBackprop {
template <typename Intf, class Config_>
struct ConvBpImpl {
public:
    using Config = Config_;

public:
    __aicore__ inline ConvBpImpl() {}

    DECLARE_IMPL(Config_, Convolution3DBackpropFunc, Init, Intf);
    DECLARE_IMPL(Config_, Convolution3DBackpropFunc, SetWeight, Intf);
    DECLARE_IMPL(Config_, Convolution3DBackpropFunc, SetOutBackprop, Intf);
    DECLARE_IMPL(Config_, Convolution3DBackpropFunc, SetSingleShape, Intf);
    DECLARE_IMPL(Config_, Convolution3DBackpropFunc, SetStartIdx, Intf);
    DECLARE_SYNC_IMPL(Config_, Convolution3DBackpropFunc, Iterate, Intf);
    DECLARE_SYNC_IMPL(Config_, Convolution3DBackpropFunc, IterateAll, Intf);
    DECLARE_SYNC_IMPL(Config_, Convolution3DBackpropFunc, GetTensorC, Intf);
    DECLARE_IMPL(Config_, Convolution3DBackpropFunc, End, Intf);
    struct ContextData : public Config::ContextData {
        __aicore__ inline ContextData(){};
        DEFINE_STUCT_FIELD(TPipe, pipe_);
        DEFINE_STUCT_FIELD(const TConv3DInputV2Tiling *__restrict, tiling_);
        DEFINE_STUCT_TEMPLATE_FIELD(TQue, c1Ping_, TPosition::CO1, 1);
        DEFINE_STUCT_TEMPLATE_FIELD(TQue, c1Pong_, TPosition::CO1, 1);
        DEFINE_STUCT_TEMPLATE_FIELD(TQue, a1Ping_, TPosition::A1, 1);
        DEFINE_STUCT_TEMPLATE_FIELD(TQue, a1Pong_, TPosition::A1, 1);
        DEFINE_STUCT_TEMPLATE_FIELD(TQue, b1Ping_, TPosition::B1, 1);
        DEFINE_STUCT_TEMPLATE_FIELD(TQue, b1Pong_, TPosition::B1, 1);
        DEFINE_STUCT_TEMPLATE_FIELD(TBuf, l0aBuf_, TPosition::A2);
        DEFINE_STUCT_TEMPLATE_FIELD(TBuf, l0bBuf_, TPosition::B2);
        DEFINE_STUCT_FIELD(uint8_t, isFirstIter_);
        DEFINE_STUCT_FIELD(uint8_t, needComputeFlag_);
        DEFINE_STUCT_FIELD(uint8_t, needComputeKFlag_);
        DEFINE_STUCT_FIELD(uint8_t, l0aPingPongFlag_);
        DEFINE_STUCT_FIELD(uint8_t, useL0PingPong_);
        DEFINE_STUCT_FIELD(uint8_t, usingCacheC1Ping_);
        DEFINE_STUCT_FIELD(bool, isB1FullLoadFlag_);
        DEFINE_STUCT_FIELD(bool, isLoadB1_);
        DEFINE_STUCT_FIELD(bool, isFreeB1_);
        DEFINE_STUCT_FIELD(uint32_t, curStepM_);
        DEFINE_STUCT_FIELD(uint32_t, curStepN_);
        DEFINE_STUCT_FIELD(uint32_t, curNL0Idx_);
        DEFINE_STUCT_FIELD(uint32_t, curNL1Idx_);
        DEFINE_STUCT_FIELD(uint32_t, curML0Idx_);
        DEFINE_STUCT_FIELD(uint32_t, curML1Idx_);
        DEFINE_STUCT_FIELD(uint32_t, curDinIdx_);
        DEFINE_STUCT_FIELD(uint32_t, curPingCoutIdx_);
        DEFINE_STUCT_FIELD(uint32_t, curPongCoutIdx_);
        DEFINE_STUCT_FIELD(uint32_t, channelSize_);
        DEFINE_STUCT_FIELD(uint32_t, curCin1Size_);
        DEFINE_STUCT_FIELD(uint32_t, curHoSize_);
        DEFINE_STUCT_FIELD(uint32_t, kIterStepKaTail);
        DEFINE_STUCT_FIELD(uint32_t, kIterStepKbTail);
        DEFINE_STUCT_FIELD(uint32_t, stepKaTail);
        DEFINE_STUCT_FIELD(uint32_t, stepKbTail);
        DEFINE_STUCT_FIELD(int32_t, curHoIdx_);
        DEFINE_STUCT_FIELD(uint32_t, idxC1in_);
        DEFINE_STUCT_FIELD(uint32_t, baseB1Offset_);
        DEFINE_STUCT_FIELD(uint32_t, blockBaseN_);
        DEFINE_STUCT_FIELD(uint32_t, baseUseM_);
        DEFINE_STUCT_FIELD(uint32_t, baseUseN_);
        DEFINE_STUCT_FIELD(uint32_t, baseUseK_);
        DEFINE_STUCT_FIELD(uint32_t, curLoadKal1_);
        DEFINE_STUCT_FIELD(uint32_t, curLoadKbl1_);
        DEFINE_STUCT_FIELD(uint32_t, baseUseAlignN_);
        DEFINE_STUCT_FIELD(uint32_t, baseMN_);
        DEFINE_STUCT_FIELD(uint32_t, HkWk_);
        DEFINE_STUCT_FIELD(uint32_t, HkWkC0_);
        DEFINE_STUCT_FIELD(uint32_t, DkHkWkC0_);
        DEFINE_STUCT_FIELD(uint32_t, singleShapeHin_);
        DEFINE_STUCT_FIELD(uint32_t, singleShapeCin1_);
        DEFINE_STUCT_FIELD(uint32_t, singleShapeCout1_);
        DEFINE_STUCT_FIELD(uint32_t, singleShapeCin_);
        DEFINE_STUCT_FIELD(uint32_t, singleShapeDin_);
        DEFINE_STUCT_FIELD(uint32_t, curDinStartIdx_);
        DEFINE_STUCT_FIELD(int32_t, curHoStartIdx_);
        DEFINE_STUCT_FIELD(uint32_t, alignedCout_);
        DEFINE_STUCT_FIELD(uint32_t, alignedCout1_);
        DEFINE_STUCT_FIELD(uint32_t, splitHk_);
        DEFINE_STUCT_FIELD(uint32_t, splitWk_);
        DEFINE_STUCT_FIELD(uint32_t, splitHkWk_);
        DEFINE_STUCT_FIELD(uint32_t, splitHkWkC0_);
        DEFINE_STUCT_FIELD(uint32_t, splitHi_);
        DEFINE_STUCT_FIELD(uint32_t, splitWi_);
        DEFINE_STUCT_FIELD(uint32_t, splitSingleShapeM_);
        DEFINE_STUCT_FIELD(uint32_t, stepKaRound_);
        DEFINE_STUCT_FIELD(uint32_t, stepKbRound_);
        DEFINE_STUCT_FIELD(uint32_t, nIter_);
        DEFINE_STUCT_FIELD(uint32_t, tailN_);
        DEFINE_STUCT_FIELD(uint64_t, singleShapeM_);
        DEFINE_STUCT_FIELD(uint64_t, mIter_);
        DEFINE_STUCT_FIELD(uint64_t, kIter_);
        DEFINE_STUCT_FIELD(uint64_t, tailM_);
        DEFINE_STUCT_FIELD(uint64_t, tailK_);
        DEFINE_STUCT_FIELD(uint64_t, hwI_);
        DEFINE_STUCT_FIELD(uint64_t, hwO_);

        DEFINE_STUCT_FIELD(MmadParams, mmad_);
        DEFINE_STUCT_FIELD(LoadData2DParams, load2d_);

        using LoadData3DParamsV2SrcT = LoadData3DParamsV2<typename Intf::SrcT>;
        DEFINE_STUCT_FIELD(LoadData3DParamsV2SrcT, load3d_);
        using srcLocalTensor = LocalTensor<typename Intf::SrcT>;
        DEFINE_STUCT_FIELD(srcLocalTensor, cacheA1BufPing_);
        DEFINE_STUCT_FIELD(srcLocalTensor, cacheA1BufPong_);
        DEFINE_STUCT_FIELD(srcLocalTensor, cacheB1BufPing_);
        DEFINE_STUCT_FIELD(srcLocalTensor, cacheB1BufPong_);
        using GlobalTnesor = GlobalTensor<typename Intf::SrcT>;
        DEFINE_STUCT_FIELD(GlobalTnesor, outBackPropGlobal_);
        DEFINE_STUCT_FIELD(GlobalTnesor, fmapGlobal_);
        DEFINE_STUCT_FIELD(GlobalTnesor, weightGlobal_);
    };
};

}  // namespace Convolution3DBackprop

#endif
