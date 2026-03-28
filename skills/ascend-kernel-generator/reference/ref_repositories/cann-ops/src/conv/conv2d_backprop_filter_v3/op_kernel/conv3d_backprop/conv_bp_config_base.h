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
 * \file conv_bp_config_base.h
 * \brief
 */

#ifndef CONV_BP_CONFIG_H
#define CONV_BP_CONFIG_H

#include "conv_bp_util.h"

using namespace AscendC;

namespace ConvolutionBackprop {

enum class CubeFormat {
    NC1HWC0,
    NCHW,
    NCDHW,
    NHWC,
    HWCN,
    FRACTALZ_C04,
    ND,
    NDC1HWC0,
    FRACTAL_Z_3D
};

template <typename T>
struct GetDstType {
    using Type = T;
};

template <>
struct GetDstType<float> {
    using Type = float;
};

template <>
struct GetDstType<half> {
    using Type = float;
};

template <>
struct GetDstType<int8_t> {
    using Type = int32_t;
};

template <>
struct GetDstType<bfloat16_t> {
    using Type = float;
};

// ConvType，定义卷积输入输出对象的属性
template <TPosition POSITION, CubeFormat FORMAT, typename T>
struct ConvType {
    constexpr static TPosition pos = POSITION;    // Convolution输入或输出时的scope
    constexpr static CubeFormat format = FORMAT;  // Convolution输入或者输出的format
    using Type = T;                               // Convolution输入或输出的数据类型
};

// 打包字段，内部实现的上下文，包含了用户构造的ConvBpParam
template <class A, class B, class C, class D>
struct ConvBpContext {
    using xType = A;
    using cType = C;
    using dType = D;
    using SrcT = typename A::Type;
    using SrcAT = typename A::Type;
    using SrcBT = typename C::Type;
    using DstT = typename D::Type;
    using L0cT = typename GetDstType<SrcT>::Type;

    constexpr static auto formatA = A::format;
    constexpr static auto formatB = B::format;
    constexpr static auto formatC = C::format;
    constexpr static auto formatD = D::format;

    constexpr static auto posA = A::pos;
    constexpr static auto posB = B::pos;
    constexpr static auto posC = C::pos;
    constexpr static auto posD = D::pos;

    using ContextData = struct _ {
        __aicore__ inline _() {}
    };
};
}  // namespace ConvolutionBackprop
#endif
