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
 * \file conv3d_config.h
 * \brief
 */

#ifndef CONV3D_CONFIG_H
#define CONV3D_CONFIG_H

#include "conv_framework_util.h"
#include "conv_config.h"

using namespace conv;

namespace conv3d {

enum class ConvL0PingPong {
    ALL_CLOSE = 0,
    L0A_OPEN,
    L0B_OPEN,
    ALL_OPEN
};

enum class QuantType {
    NO_QUANT = 0,
    PER_TENSOR_NO_BIAS,
    PER_TENSOR_BIAS_INT32,
    PER_TENSOR_NO_OFFSET,
    PER_CHANNEL_NO_BIAS,
    PER_CHANNEL_BIAS_INT32,
    PER_CHANNEL_NO_OFFSET,
    PER_TOKEN_NO_BIAS,
    PER_TOKEN_BIAS_INT32,
    PER_TOKEN_NO_OFFSET
};

enum class LoadChannelType {
    NORMAL = 0,
    LOAD_TOTAL_CORE,
    LOAD_TOTAL_LC0
};

enum class VecComputeType {
    ALTERNATE,
    SPLIT
};

enum class ConvBL1ByPass {
    BYPASS_OFF = 0,
    BYPASS_ON = 1
};

enum class GroupConvType {
    NoGroup_Conv = 0, // 非group卷积
	GroupConv_Weight_Gfz // group卷积，weight数据为私有group_fractalz格式
};

enum class OutputOrder {
    M_Mode = 0,
    HW_Mode
};

struct Conv3dParam : public ConvParam {
    __aicore__ inline Conv3dParam(){};
};

template <class ConvDataType>
struct Conv3dCfg : public ConvConfig<ConvDataType> {
public:
    __aicore__ inline Conv3dCfg()
    {}

    using ContextData = struct _ : public ConvConfig<ConvDataType>::ContextData {
        __aicore__ inline _()
        {}
    };
};
}  // namespace conv3d

#endif