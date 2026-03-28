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
 * \file conv3d_bp_intf_base.h
 * \brief
 */

#ifndef CONV3D_BP_INTF_H
#define CONV3D_BP_INTF_H

#include "conv3d_bp_config_base.h"
#include "conv3d_bp_func.h"
#include "conv3d_bp_util.h"
#include "kernel_tiling/kernel_tiling.h"

namespace Convolution3DBackprop {
// 用户可见的api原型集合
template <class Config_, template <typename, class> class Impl>
struct ConvBpIntf {
    using Config = Config_;
    using Ext = Impl<ConvBpIntf, Config>;
    using SrcT = typename Config::SrcT;
    using DstT = typename Config::DstT;
    using L0cT = typename Config::L0cT;
    using ContextData = typename Ext::ContextData;

public:
    ContextData ctx;
    constexpr static Conv3dConfig conv3dConfig = Config::conv3dConfig_;

public:
    __aicore__ inline ConvBpIntf() {}

    __aicore__ inline void Init(const TConv3DInputV2Tiling *__restrict tiling)
    {
        using Local = typename Ext::Init;
        // CheckFun检查impl是否实现了Init的call函数
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this, tiling)) {
            Local::call(this, tiling);
        }
    }

    __aicore__ inline void SetFmap(const GlobalTensor<SrcT> &fmap)
    {
        using Local = typename Ext::SetFmap;
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this, fmap)) {
            Local::call(this, fmap);
        }
    }

    __aicore__ inline void SetWeight(const GlobalTensor<SrcT> &weight)
    {
        using Local = typename Ext::SetWeight;
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this, weight)) {
            Local::call(this, weight);
        }
    }

    __aicore__ inline void SetOutBackprop(const GlobalTensor<SrcT> &outBackprop)
    {
        using Local = typename Ext::SetOutBackprop;
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this, outBackprop)) {
            Local::call(this, outBackprop);
        }
    }

    __aicore__ inline void SetSingleShape(uint64_t singleShapeM, uint64_t singleShapeK, uint32_t singleShapeN,
                                          uint32_t singleShapeD)
    {
        using Local = typename Ext::SetSingleShape;
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this, singleShapeM, singleShapeK, singleShapeN,
                                singleShapeD)) {
            Local::call(this, singleShapeM, singleShapeK, singleShapeN, singleShapeD);
        }
    }

    __aicore__ inline void SetStartIdx(uint32_t curDinStartIdx, int32_t curHoStartIdx)
    {
        using Local = typename Ext::SetStartIdx;
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this, curDinStartIdx, curHoStartIdx)) {
            Local::call(this, curDinStartIdx, curHoStartIdx);
        }
    }

    template <bool sync = true>
    __aicore__ inline bool Iterate(bool enPartialSum = false)
    {
        using Local = typename Ext::template Iterate<sync>;
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this, enPartialSum)) {
            return Local::call(this, enPartialSum);
        }
    }

    template <bool sync = true>
    __aicore__ inline void IterateAll(const GlobalTensor<DstT> &output, uint8_t enAtomic = 0)
    {
        using Local = typename Ext::template IterateAll<sync>;
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this, output, enAtomic)) {
            Local::call(this, output, enAtomic);
        }
    }

    template <bool sync = true>
    __aicore__ inline void GetTensorC(const GlobalTensor<DstT> &output, uint8_t enAtomic = 0,
                                      bool enSequentialWrite = false)
    {
        using Local = typename Ext::template GetTensorC<sync>;
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this, output, enAtomic, enSequentialWrite)) {
            Local::call(this, output, enAtomic, enSequentialWrite);
        }
    }

    __aicore__ inline void End()
    {
        using Local = typename Ext::End;
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this)) {
            Local::call(this);
        }
    }
};

}  // namespace Convolution3DBackprop

#endif
