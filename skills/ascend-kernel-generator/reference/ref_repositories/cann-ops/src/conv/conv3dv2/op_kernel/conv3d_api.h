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
 * \file conv3d_api.h
 * \brief
 */

#ifndef CONV3D_API_H
#define CONV3D_API_H

#include "conv3d_intf.h"
#include "conv3d_config.h"
#include "conv3d_api_impl.h"

namespace conv3d {

template <class Config, template <typename, class, bool> class Impl = Conv3dApiImpl,
    template <class, template <typename, class, bool> class> class Intf = Conv3dIntf>
struct Conv3dIntfExt : public Intf<Config, Impl> {
    __aicore__ inline Conv3dIntfExt()
    {}
};

#define REGISTER_CONV3D_API(name, Config, Impl, Intf)                                                                \
    template <class INPUT_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class CONV_CFG = Conv3dParam> \
    using name =                                                                                                     \
        Conv3dIntfExt<Config<ConvDataType<INPUT_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, CONV_CFG>>, Impl, Intf>

REGISTER_CONV3D_API(Conv3d, Conv3dCfg, Conv3dApiImpl, Conv3dIntf);
}  // namespace conv3d
#endif
