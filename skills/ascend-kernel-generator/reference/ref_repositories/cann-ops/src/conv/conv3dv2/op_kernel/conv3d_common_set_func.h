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
 * \file conv3d_common_set_func.h
 * \brief
 */

#ifndef CONV3D_COMMON_SET_FUNC_H
#define CONV3D_COMMON_SET_FUNC_H

#include "conv_framework_util.h"
#include "conv3d_common_sub_api.h"
#include "conv3d_config.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"

namespace Conv3dFunc {

template <class Intf, uint32_t ImplType>
struct SetOrgFmapShape {
    static __aicore__ inline void call(Intf *self, uint64_t orgCi, uint64_t orgDi, uint64_t orgHi, uint64_t orgWi)
    {
        self->ctx.oriCi = orgCi;
        self->ctx.orgDi = orgDi;
        self->ctx.orgHi = orgHi;
        self->ctx.orgWi = orgWi;
    }
};

template <class Intf, uint32_t ImplType>
struct SetOrgWeightShape {
    static __aicore__ inline void call(
        Intf *self, uint64_t orgCo, uint64_t orgCi, uint64_t orgKd, uint64_t orgKh, uint64_t orgKw)
    {
        self->ctx.orgCo = orgCo;
        self->ctx.orgCi = orgCi;
        self->ctx.kernelD = orgKd;
        self->ctx.kernelH = orgKh;
        self->ctx.kernelW = orgKw;
    }
};

template <class Intf, uint32_t ImplType>
struct SetOrgOutputShape {
    static __aicore__ inline void call(Intf *self, uint64_t orgCo, uint64_t orgDo, uint64_t orgHo, uint64_t orgWo)
    {
        self->ctx.orgCo = orgCo;
        self->ctx.orgDo = orgDo;
        self->ctx.orgHo = orgHo;
        self->ctx.orgWo = orgWo;
    }
};

template <class Intf, uint32_t ImplType>
struct SetSingleFmapShape {
    static __aicore__ inline void call(
        Intf *self, uint64_t singleCi, uint64_t singleDi, uint64_t singleHi, uint64_t singleWi)
    {
        self->ctx.singleCoreCin = singleCi;
        InitKDirectionBaseValue<Intf>(self);
    }
};

template <class Intf, uint32_t ImplType>
struct SetSingleOutputShape {
    static __aicore__ inline void call(
        Intf *self, uint64_t singleCoreBatch, uint64_t singleCo, uint64_t singleDo, uint64_t singleHo,
        uint64_t singleWo, uint64_t singleGroupOpt)
    {
        self->ctx.singleCoreCo = singleCo;
        self->ctx.singleCoreDo = singleDo;
        self->ctx.singleCoreHo = singleHo;
        self->ctx.singleCoreGroupOpt = singleGroupOpt;
        InitHoutDirectionValue<Intf>(self);
        InitCoutDirectionBaseValue<Intf>(self);
        InitDoutDirectionBaseValue<Intf>(self);
        InitGroupOptDirectionValue<Intf>(self);
    }

    static __aicore__ inline void call(
        Intf *self, uint64_t singleCoreBatch, uint64_t singleCo, uint64_t singleDo, uint64_t singleCoreM,
        uint64_t singleGroupOpt)
    {
        self->ctx.singleCoreCo = singleCo;
        self->ctx.singleCoreDo = singleDo;
        self->ctx.singleCoreM = singleCoreM;
        self->ctx.singleCoreGroupOpt = singleGroupOpt;
        InitMDirectionBaseValue<Intf>(self);
        InitCoutDirectionBaseValue<Intf>(self);
        InitDoutDirectionBaseValue<Intf>(self);
        InitGroupOptDirectionValue<Intf>(self);
    }
};

template <class Intf, uint32_t ImplType>
struct SetFmapStartPosition {
    static __aicore__ inline void call(
        Intf *self, int64_t diStartPos, int64_t hiStartPos, int64_t wiStartPos, int64_t ciStartPos)
    {
        self->ctx.diStartPos = diStartPos;
        self->ctx.hiStartPos = hiStartPos;
    }

    static __aicore__ inline void call(Intf *self, int64_t diStartPos, int64_t mStartPos, int64_t ciStartPos)
    {
        self->ctx.diStartPos = diStartPos;
        self->ctx.mStartPos = mStartPos;
    }
};

template <class Intf, uint32_t ImplType>
struct BeginSetCrossFlag {
    static __aicore__ inline void call(Intf *self)
    {
        if constexpr (Intf::quantType == static_cast<int8_t>(QuantType::PER_CHANNEL_NO_OFFSET)) {
            if ASCEND_IS_AIV {  
                CrossCoreSetFlag<0x2, PIPE_MTE2>(self->ctx.V2CEvent);
                CrossCoreSetFlag<0x2, PIPE_MTE2>(self->ctx.V2CEvent + 1);
                CrossCoreSetFlag<0x2, PIPE_MTE2>(self->ctx.V2CEvent + 2);
                CrossCoreSetFlag<0x2, PIPE_MTE2>(self->ctx.V2CEvent + 3);
            }
        }
    }
};

template <class Intf, uint32_t ImplType>
struct EndWaitCrossFlag {
    static __aicore__ inline void call(Intf *self)
    {
        if constexpr (Intf::quantType == static_cast<int8_t>(QuantType::PER_CHANNEL_NO_OFFSET)) {
            if ASCEND_IS_AIC {  
                CrossCoreWaitFlag(self->ctx.V2CEvent);
                CrossCoreWaitFlag(self->ctx.V2CEvent + 1);
                CrossCoreWaitFlag(self->ctx.V2CEvent + 2);
                CrossCoreWaitFlag(self->ctx.V2CEvent + 3);
            }
        }
    }
};


template <class Intf, uint32_t ImplType>
struct SetWorkspace {
    static __aicore__ inline void call(Intf *self, const GlobalTensor<typename Intf::L0cT> &workspace)
    {
        self->ctx.workspacegm.SetGlobalBuffer(workspace.GetPhyAddr(0), workspace.GetSize());
    }
};

template <class Intf, uint32_t ImplType>
struct SetVecScale {
    static __aicore__ inline void call(Intf *self, const GlobalTensor<typename Intf::FP32T> &scale)
    {
        self->ctx.scalegm.SetGlobalBuffer(scale.GetPhyAddr(0), scale.GetSize());
    }
};

template <class Intf, uint32_t ImplType>
struct SetSubBlockIdx {
    static __aicore__ inline void call(Intf *self, uint32_t subblockIdx)
    {
        self->ctx.subblockIdx = subblockIdx;
    }
};

template <class Intf, uint32_t ImplType>
struct LoadTotalCoreChannel {
    static __aicore__ inline void call(Intf *self)
    {
        if constexpr (Intf::quantType == static_cast<int8_t>(QuantType::PER_CHANNEL_NO_OFFSET)) {
            if ASCEND_IS_AIV {  
                if constexpr (Intf::loadChannelType == static_cast<int8_t>(LoadChannelType::LOAD_TOTAL_CORE)) {
                    uint64_t singleco = AlignB(self->ctx.singleCoreCo, BLOCK_L0_N);
                    DataCopyParams copyinParams;
                    copyinParams.blockCount = 1;
                    copyinParams.blockLen =  singleco * sizeof(typename Intf::BiasT) / 32;
                    LocalTensor<typename Intf::BiasT> bias = self->ctx.queueUBbias.template AllocTensor<typename Intf::BiasT>();
                    DataCopy(bias, self->ctx.biasgm, copyinParams);     
                    
                    self->ctx.queueUBbias.EnQue(bias);

                    LocalTensor<typename Intf::BiasT> biasLocal = self->ctx.queueUBbias.template DeQue<typename Intf::BiasT>();

                    if constexpr (IsSameType<typename Intf::BiasT, bfloat16_t>::value || IsSameType<typename Intf::BiasT, half>::value) {
                        self->ctx.ubbias = self->ctx.fp32BiasBuf.template Get<typename Intf::FP32T>();
                        Cast(self->ctx.ubbias, biasLocal, RoundMode::CAST_NONE, singleco);
                        self->ctx.queueUBbias.FreeTensor(biasLocal);
                    } else {
                        self->ctx.ubbias = biasLocal;
                    }
        
                    copyinParams.blockLen =  singleco * sizeof(typename Intf::FP32T) / 32;
                    LocalTensor<typename Intf::FP32T> scale = self->ctx.queueUBscale.template AllocTensor<typename Intf::FP32T>();
                    DataCopy(scale, self->ctx.scalegm, copyinParams);
                    self->ctx.queueUBscale.EnQue(scale);
                    self->ctx.ubscale = self->ctx.queueUBscale.template DeQue<typename Intf::FP32T>();
                }
            }
        }
    }
};

template <class Intf, uint32_t ImplType>
struct FreeTotalCoreChannel {
    static __aicore__ inline void call(Intf *self)
    {
        if constexpr (Intf::quantType == static_cast<int8_t>(QuantType::PER_CHANNEL_NO_OFFSET)) {
            if ASCEND_IS_AIV {  
                if constexpr (Intf::loadChannelType == static_cast<int8_t>(LoadChannelType::LOAD_TOTAL_CORE)) {
                    if constexpr (!(IsSameType<typename Intf::BiasT, bfloat16_t>::value || IsSameType<typename Intf::BiasT, half>::value)) {
                        self->ctx.queueUBbias.FreeTensor(self->ctx.ubbias);
                    }
                    self->ctx.queueUBscale.FreeTensor(self->ctx.ubscale);
                }
            }
        }
    }
};

template <class Intf, uint32_t ImplType>
struct SetGroupOptInfo {
    static __aicore__ inline void call(
        Intf *self, uint64_t singleCoreCinTail, uint64_t singleCoreCoutTail, bool isGroupOptDimTail = false)
    {
        self->ctx.singleCoreCinTail = singleCoreCinTail;
        self->ctx.singleCoreCoutTail = singleCoreCoutTail;
        self->ctx.isGroupOptDimTail = isGroupOptDimTail;
    }
};

}  // namespace Conv3dFunc

#endif
