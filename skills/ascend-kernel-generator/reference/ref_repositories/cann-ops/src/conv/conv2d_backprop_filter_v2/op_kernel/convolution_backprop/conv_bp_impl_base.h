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
 * \file conv_bp_impl_base.h
 * \brief
 */

#ifndef CONV_BP_IMPL_H
#define CONV_BP_IMPL_H

#include "conv_bp_config_base.h"
#include "conv_bp_func.h"
#include "conv_bp_util.h"
#include "kernel_utils.h"
#include "kernel_operator.h"

namespace ConvolutionBackprop {
template <typename Intf, class Config_>
struct ConvBpImpl {
public:
    using Config = Config_;

public:
    __aicore__ inline ConvBpImpl()
    {}

    DECLARE_IMPL(Config_, ConvolutionBackpropFunc, Init, Intf);
    DECLARE_IMPL(Config_, ConvolutionBackpropFunc, SetFmap, Intf);
    DECLARE_IMPL(Config_, ConvolutionBackpropFunc, SetOutBackprop, Intf);
    DECLARE_IMPL(Config_, ConvolutionBackpropFunc, SetSingleShape, Intf);
    DECLARE_SYNC_IMPL(Config_, ConvolutionBackpropFunc, Iterate, Intf);
    DECLARE_SYNC_IMPL(Config_, ConvolutionBackpropFunc, IterateAll, Intf);
    DECLARE_SYNC_IMPL(Config_, ConvolutionBackpropFunc, GetTensorC, Intf);
    DECLARE_IMPL(Config_, ConvolutionBackpropFunc, End, Intf);
    struct ContextData : public Config::ContextData {
        __aicore__ inline ContextData(){};
        DEFINE_STUCT_FIELD(TPipe, pipe_);
        DEFINE_STUCT_FIELD(const TConvTiling *__restrict, tiling_);
        DEFINE_STUCT_FIELD(uint8_t, isFirstIter_);
        DEFINE_STUCT_FIELD(uint32_t, mIter_);
        DEFINE_STUCT_FIELD(uint32_t, nIter_);
        DEFINE_STUCT_FIELD(uint32_t, kIter_);
        DEFINE_STUCT_FIELD(uint32_t, tailM_);
        DEFINE_STUCT_FIELD(uint32_t, tailN_);
        DEFINE_STUCT_FIELD(uint32_t, tailK_);
        DEFINE_STUCT_TEMPLATE_FIELD(TQue, qidA2_, TPosition::A2, 2);
        DEFINE_STUCT_TEMPLATE_FIELD(TQue, qidB2_, TPosition::B2, 2);
        DEFINE_STUCT_TEMPLATE_FIELD(TQue, qidCO1_, TPosition::CO1, 2);
        DEFINE_STUCT_TEMPLATE_FIELD(TQue, qidB1_, TPosition::B1, 2);
        DEFINE_STUCT_TEMPLATE_FIELD(TQue, qidA1_, TPosition::A1, 2);
        DEFINE_STUCT_FIELD(uint32_t, curStepM_);
        DEFINE_STUCT_FIELD(uint32_t, curStepN_);
        DEFINE_STUCT_FIELD(uint32_t, curNL0Idx_);
        DEFINE_STUCT_FIELD(uint32_t, curNL1Idx_);
        DEFINE_STUCT_FIELD(uint32_t, curML0Idx_);
        DEFINE_STUCT_FIELD(uint32_t, curML1Idx_);
        DEFINE_STUCT_FIELD(uint32_t, baseUseM_);
        DEFINE_STUCT_FIELD(uint32_t, baseUseN_);
        DEFINE_STUCT_FIELD(uint32_t, baseUseK_);
        DEFINE_STUCT_FIELD(uint32_t, baseMK_);
        DEFINE_STUCT_FIELD(uint32_t, baseKN_);
        DEFINE_STUCT_FIELD(uint32_t, baseMN_);
        DEFINE_STUCT_FIELD(uint32_t, singleShapeHo_);
        DEFINE_STUCT_FIELD(uint64_t, singleShapeCin_);
        DEFINE_STUCT_FIELD(uint64_t, singleShapeCout_);
        DEFINE_STUCT_FIELD(uint64_t, dstL12L0aOffset_);
        DEFINE_STUCT_FIELD(uint64_t, srcL12L0aOffset_);
        DEFINE_STUCT_FIELD(uint64_t, srcL0aOffset_);
        DEFINE_STUCT_FIELD(uint64_t, srcL0bOffset_);
        DEFINE_STUCT_FIELD(uint64_t, dstL0cOffset_);
        DEFINE_STUCT_FIELD(MmadParams, mmad_);
        DEFINE_STUCT_FIELD(LoadData2DParams, load2d_);
        using LoadData3DParamsV2SrcT = LoadData3DParamsV2<typename Intf::SrcT>;
        DEFINE_STUCT_FIELD(LoadData3DParamsV2SrcT, load3d_);
        DEFINE_STUCT_FIELD(uint8_t, usingCacheA1Ping_);
        DEFINE_STUCT_FIELD(uint8_t, usingCacheA1Pong_);
        DEFINE_STUCT_FIELD(uint8_t, usingCacheB1Ping_);
        DEFINE_STUCT_FIELD(uint8_t, usingCacheB1Pong_);
        using LocalTnesor = LocalTensor<typename Intf::SrcT>;
        DEFINE_STUCT_FIELD(LocalTnesor, cacheA1BufPing_);
        DEFINE_STUCT_FIELD(LocalTnesor, cacheA1BufPong_);
        DEFINE_STUCT_FIELD(LocalTnesor, cacheB1BufPing_);
        DEFINE_STUCT_FIELD(LocalTnesor, cacheB1BufPong_);
        DEFINE_STUCT_FIELD(LocalTnesor, a2_);
        DEFINE_STUCT_FIELD(LocalTnesor, b2_);
        using locLocalTensor = LocalTensor<typename Intf::L0cT>;
        DEFINE_STUCT_FIELD(locLocalTensor, c1_);
        using GlobalTnesor = GlobalTensor<typename Intf::SrcT>;
        DEFINE_STUCT_FIELD(GlobalTnesor, outBackPropGlobal_);
        DEFINE_STUCT_FIELD(GlobalTnesor, fmapGlobal_);
    };
};

}  // namespace ConvolutionBackprop

#endif