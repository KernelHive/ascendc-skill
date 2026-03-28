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
 * \file conv3d_bp_filter_impl.h
 * \brief
 */

#ifndef CONV3D_BP_FILTER_IMPL_H
#define CONV3D_BP_FILTER_IMPL_H

#include "conv3d_bp_filter_config.h"
#include "conv3d_backprop/conv_bp_impl_base.h"
#include "conv3d_backprop/conv_bp_func.h"
#include "conv3d_backprop/conv_bp_util.h"
#include "kernel_utils.h"

namespace ConvolutionBackprop {
template <typename Intf_, class Config_>
struct Conv3DBpFilterImpl : public ConvBpImpl<Intf_, Config_> {
public:
    __aicore__ inline Conv3DBpFilterImpl() {}
    struct ContextData : public ConvBpImpl<Intf_, Config_>::ContextData {
        __aicore__ inline ContextData() {}
    };
};

}  // namespace ConvolutionBackprop

#endif