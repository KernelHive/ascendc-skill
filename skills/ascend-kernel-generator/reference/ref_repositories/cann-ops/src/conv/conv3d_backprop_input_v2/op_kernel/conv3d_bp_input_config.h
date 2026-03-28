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
 * \file conv3d_bp_input_config.h
 * \brief
 */

#ifndef CONV3D_BP_INPUT_CONFIG_H
#define CONV3D_BP_INPUT_CONFIG_H

#include "conv3d_bp_config_base.h"

namespace Convolution3DBackprop {

template <class A, class B, class C, class D, const Conv3dConfig& CONV3D_CONFIG = CONV3D_CFG_DEFAULT>
struct Conv3DBpInputCfg : public ConvBpContext<A, B, C, D> {
public:
    __aicore__ inline Conv3DBpInputCfg() {}

    using ContextData = struct _ : public ConvBpContext<A, B, C, D>::ContextData {
        __aicore__ inline _() {}
    };
    constexpr static Conv3dConfig conv3dConfig_ = CONV3D_CONFIG;
};

}  // namespace Convolution3DBackprop
#endif
