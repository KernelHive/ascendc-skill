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
 * @file conv2d_bp_filter.h
 */

#ifndef CONV2D_BP_FILTER_H
#define CONV2D_BP_FILTER_H

#include "convolution_backprop/conv_bp_register.h"
#include "conv2d_bp_filter_config.h"
#include "conv2d_bp_filter_impl.h"
#include "conv2d_bp_filter_intf.h"

namespace ConvolutionBackprop {

REGISTER_DW_IMPL(Conv2DBackpropFilter, Conv2DBpFilterCfg, Conv2DBpFilterImpl, Conv2DBpFilterIntf);

}  // namespace ConvolutionBackprop
#endif
