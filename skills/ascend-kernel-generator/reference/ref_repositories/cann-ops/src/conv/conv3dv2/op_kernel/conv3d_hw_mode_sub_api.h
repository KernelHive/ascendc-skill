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
 * \file conv3d_hw_mode_sub_api.h
 * \brief
 */

#ifndef CONV3D_HW_MODE_SUB_API_H
#define CONV3D_HW_MODE_SUB_API_H

#include "conv3d_mte1_hw_mode_sub_api.h"

using namespace AscendC;
using namespace conv;
using namespace conv3d;

namespace Conv3dFunc {

template <class Intf>
class CopyOutWithHwModeTools {
public:
    __aicore__ inline CopyOutWithHwModeTools() {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
        tilingNBL1_ = self_->ctx.conv3dTiling->nBL1;
        tilingMAL1_ = self_->ctx.conv3dTiling->mAL1;
        valueHoWo_ = self_->ctx.orgHo * self_->ctx.orgWo;
    }

    __aicore__ inline void SetMN(uint64_t m, uint64_t n)
    {
        currentML0_ = m;
        currentNL0_ = n;
    }

    __aicore__ inline void SetFixpipeIntriParams(FixpipeParamsV220 &intriParams)
    {
        if (self_->ctx.nBL1Iter == self_->ctx.maxNBL1Iter && self_->ctx.nBL0Iter == self_->ctx.maxNL0Iter) {
            intriParams.nSize = currentNL0_;
        } else {
            intriParams.nSize = self_->ctx.conv3dTiling->nL0;
        }

        if (self_->ctx.mAL1Iter == self_->ctx.maxMAL1Iter && self_->ctx.mAL0Iter == self_->ctx.maxML0Iter) {
            intriParams.mSize = self_->ctx.singleCoreM - self_->ctx.mAL1Iter * self_->ctx.conv3dTiling->mAL1 -
                                self_->ctx.mAL0Iter * self_->ctx.conv3dTiling->mL0;
            intriParams.srcStride = AlignB(self_->ctx.mAL0Tail, BLOCK_L0_M);
        } else {
            intriParams.mSize = self_->ctx.conv3dTiling->mL0;
            intriParams.srcStride = self_->ctx.conv3dTiling->mL0;
        }

        intriParams.dstStride = valueHoWo_;
        intriParams.quantPre = GetQuantPre();
        intriParams.reluEn = false;

        if constexpr (AscendC::IsSameType<typename Intf::L0cT, float>::value &&
                      AscendC::IsSameType<typename Intf::OutputT, float>::value) {
            intriParams.isChannelSplit = true;
        }

        ASC_OP_LOGD("[CopyOut] intriParams.nSize %d, intriParams.mSize %d, intriParams.srcStride %d, "
                    "intriParams.dstStride %d, intriParams.quantPre %d, intriParams.reluEn %d.\n",
            intriParams.nSize,
            intriParams.mSize,
            intriParams.srcStride,
            intriParams.dstStride,
            intriParams.quantPre,
            intriParams.reluEn);
    }

    __aicore__ inline void CopyOut(const GlobalTensor<typename Intf::OutputT> &output)
    {
        FixpipeParamsV220 intriParams;
        SetFixpipeIntriParams(intriParams);
        uint64_t offset = CalcFixpipeOffset();
        ASC_OP_LOGD("[CopyOut] offset %d.\n", offset);
        if constexpr (!(AscendC::IsSameType<typename Intf::L0cT, int32_t>::value &&
                AscendC::IsSameType<typename Intf::OutputT, bfloat16_t>::value)) {
            Fixpipe<typename Intf::OutputT, typename Intf::L0cT, CFG_NZ>(output[offset], self_->ctx.cl0, intriParams);
        }
    }

    __aicore__ inline void CopyOut2Workspace(const GlobalTensor<typename Intf::L0cT> &output)
    {
        FixpipeParamsV220 intriParams;
        SetFixpipeIntriParams(intriParams);
        intriParams.quantPre = QuantMode_t::NoQuant;
        intriParams.dstStride =self_->ctx.conv3dTiling->mL0;

        uint64_t offset = self_->ctx.workspaceDbFlag * self_->ctx.conv3dTiling->nL0 * self_->ctx.conv3dTiling->mL0;
        Fixpipe<typename Intf::L0cT, typename Intf::L0cT, CFG_NZ>(output[offset], self_->ctx.cl0, intriParams);
    }

    __aicore__ inline void CopyUBOut(const GlobalTensor<typename Intf::OutputT> &output, uint32_t mIter, uint32_t nIter, uint32_t m, uint32_t n,
                                     const LocalTensor<typename Intf::BF16T> &ubout)
    {
        // impl for hwMode
    }

    __aicore__ inline void GetL0CSize(uint64_t &mSize, uint64_t &nSize)
    {
        // impl for hwMode
    }

    __aicore__ inline uint64_t GetChannelOffset(uint64_t noffset)
    {
        return 0;
    }

private:
    __aicore__ inline uint64_t CalcFixpipeOffset()
    {
        uint64_t offsetCout = tilingNBL1_ * self_->ctx.nBL1Iter + self_->ctx.conv3dTiling->nL0 * self_->ctx.nBL0Iter;
        uint64_t offsetM = tilingMAL1_ * self_->ctx.mAL1Iter + self_->ctx.conv3dTiling->mL0 * self_->ctx.mAL0Iter;
        // 当前每次只出一个dout
        uint64_t offsetDout = self_->ctx.dOutIter;
        return offsetDout * self_->ctx.orgCoAlignK0 * valueHoWo_ + offsetCout * valueHoWo_ +
               self_->ctx.hoL1Iter * self_->ctx.orgWo * self_->ctx.cin0 + offsetM * self_->ctx.cin0;
    }

    __aicore__ inline QuantMode_t GetQuantPre()
    {
        if constexpr (AscendC::IsSameType<typename Intf::L0cT, float>::value &&
                      AscendC::IsSameType<typename Intf::OutputT, float>::value) {
            return QuantMode_t::NoQuant;
        }

        if constexpr (AscendC::IsSameType<typename Intf::L0cT, int32_t>::value &&
                      AscendC::IsSameType<typename Intf::OutputT, half>::value) {
            return QuantMode_t::VDEQF16;
        }

        if constexpr (AscendC::IsSameType<typename Intf::L0cT, float>::value &&
                      AscendC::IsSameType<typename Intf::OutputT, bfloat16_t>::value) {
            return QuantMode_t::F322BF16;
        }

        return QuantMode_t::F322F16;
    }

private:
    Intf *self_ = nullptr;
    uint64_t tilingNBL1_ = 0;
    uint64_t tilingMAL1_ = 0;
    uint64_t valueHoWo_ = 0;
    uint64_t currentML0_ = 0;
    uint64_t currentNL0_ = 0;
};

};  // namespace Conv3dFunc

#endif  // __CONV3D_HW_MODE_SUB_API_H__
