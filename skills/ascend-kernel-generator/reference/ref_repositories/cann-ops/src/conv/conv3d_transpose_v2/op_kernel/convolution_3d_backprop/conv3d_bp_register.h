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
 * \file conv3d_bp_register.h
 * \brief
 */

#ifndef CONV3D_BP_REGISTER_H
#define CONV3D_BP_REGISTER_H

#include "conv3d_bp_config_base.h"
#include "conv3d_bp_impl_base.h"
#include "conv3d_bp_intf_base.h"

namespace Convolution3DBackprop {
// 注册，通过别名定义用户接口
#define REGISTER_DX_IMPL(name, context, impl, intf)             \
    template <class X_T, class W_TYPE, class DEDY_T, class Y_T, \
    const Conv3dConfig& CONV3D_CONFIG = CONV3D_CFG_DEFAULT> \
    using name = intf<context<X_T, W_TYPE, DEDY_T, Y_T, CONV3D_CONFIG>, impl>
}  // namespace Convolution3DBackprop
#endif
