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
 * \file conv_common_func.h
 * \brief
 */

#ifndef CONV_COMMON_FUNC_H
#define CONV_COMMON_FUNC_H

#include "conv_util.h"
#include "conv_framework_util.h"

namespace ConvFunc {

CONV_DECLARE_REG_IMPL(SetFmap);
CONV_DECLARE_REG_IMPL(SetWeight);
CONV_DECLARE_REG_IMPL(SetBias);
CONV_DECLARE_REG_IMPL(End);

using TypeFalse = struct {
    __uint128_t _[1024];
};

template <class Intf, uint32_t ImplType>
struct SetFmap {
    static __aicore__ inline void call(Intf *self, const GlobalTensor<typename Intf::FmapT> &fmap)
    {
        self->ctx.agm.SetGlobalBuffer(fmap.GetPhyAddr(0), fmap.GetSize());
    }
};

template <class Intf, uint32_t ImplType>
struct SetWeight {
    static __aicore__ inline void call(Intf *self, const GlobalTensor<typename Intf::WeightT> &weight)
    {
        self->ctx.bgm.SetGlobalBuffer(weight.GetPhyAddr(0), weight.GetSize());
    }
};

template <class Intf, uint32_t ImplType>
struct SetBias {
    static __aicore__ inline void call(Intf *self, const GlobalTensor<typename Intf::BiasT> &bias)
    {
        self->ctx.biasgm.SetGlobalBuffer(bias.GetPhyAddr(0), bias.GetSize());
        if constexpr (Intf::quantType != static_cast<int8_t>(QuantType::PER_CHANNEL_NO_OFFSET)) {
            self->ctx.enableBias = true;
        }
    }
};

template <class Intf, uint32_t ImplType>
struct End {
    static __aicore__ inline void call(Intf *self)
    {
        if ASCEND_IS_AIC { 
            if (self->ctx.freeAL1TensorFlag) {
                self->ctx.queueAL1.FreeTensor(self->ctx.al1);
                self->ctx.freeAL1TensorFlag = false;
            }
            if (self->ctx.freeBL1TensorFlag) {
                self->ctx.queueBL1.FreeTensor(self->ctx.bl1);
                self->ctx.freeBL1TensorFlag = false;
            }
            self->ctx.queueAL1.FreeAllEvent();
            self->ctx.queueBL1.FreeAllEvent();
            self->ctx.queueBiasL1.FreeAllEvent();
            if constexpr (Intf::formatType != conv::ConvFormat::NCDHW) {
                self->ctx.queueBiasBT.FreeAllEvent();
            }
            self->ctx.queueCL0.FreeAllEvent();
        }
    }
};

}  // namespace ConvFunc
#endif
