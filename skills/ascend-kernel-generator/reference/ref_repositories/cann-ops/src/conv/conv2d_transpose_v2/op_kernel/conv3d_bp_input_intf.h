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
 * \file conv3d_bp_input_intf.h
 * \brief
 */

#ifndef CONV3D_BP_INPUT_INTF_H
#define CONV3D_BP_INPUT_INTF_H

#include "convolution_3d_backprop/conv3d_bp_func.h"
#include "convolution_3d_backprop/conv3d_bp_util.h"

namespace Convolution3DBackprop {
template <class Config_, template <typename, class> class Impl>
struct Conv3DBpInputIntf : public ConvBpIntf<Config_, Impl> {
public:
    __aicore__ inline Conv3DBpInputIntf() {}
};

}  // namespace Convolution3DBackprop

#endif
