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
 * \file conv3d_pointwise_sub_api.h
 * \brief
 */

#ifndef CONV3D_POINTWISE_SUB_API_H
#define CONV3D_POINTWISE_SUB_API_H

#include "conv3d_mte1_pointwise_sub_api.h"

using namespace AscendC;
using namespace conv;
using namespace conv3d;

namespace Conv3dFunc {

template <class Intf, typename DataTypeT>
class LoadBiasL1WithPointWiseTools {
public:
    __aicore__ inline LoadBiasL1WithPointWiseTools()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
    }

    __aicore__ inline void SetN(uint64_t n)
    {
        currentNL0_ = n;
    }

    __aicore__ inline void LoadChannelWiseL1(const LocalTensor<DataTypeT> &tensorL1,
                                                const GlobalTensor<DataTypeT> &tensorGm)
    {
        PreProcess();
        uint64_t srcDValue = AlignB(currentNL0_, BLOCK_L0_M);

        InitConstValueParams<DataTypeT> initConstValueParams;
        initConstValueParams.repeatTimes = 1;
        initConstValueParams.blockNum = srcDValue;
        initConstValueParams.initValue = 0;
        InitConstValue<DataTypeT>(tensorL1, initConstValueParams);
        ASC_OP_LOGD("[LoadBiasL1WithPointWise] initConstValueParams.blockNum %d.\n", initConstValueParams.blockNum);

        SetNd2NzParams();
        DataCopy<DataTypeT>(tensorL1, tensorGm[tensorGmOffset], nd2NzParams);
    }

private:
    __aicore__ inline void PreProcess()
    {
        tensorGmOffset = self_->ctx.biasFullLoadFlag ? 0 :
            self_->ctx.nBL1Iter * self_->ctx.conv3dTiling->nBL1 + self_->ctx.nBL0Iter * self_->ctx.conv3dTiling->nL0;
        currentNL0_ = self_->ctx.biasFullLoadFlag ? self_->ctx.singleCoreCo : currentNL0_;
        ASC_OP_LOGD("[LoadBiasL1WithPointWise] tensorGmOffset %d currentNL0_ %d \n", tensorGmOffset, currentNL0_);
    }

    __aicore__ inline void SetNd2NzParams()
    {
        nd2NzParams.ndNum = 1;
        nd2NzParams.nValue = currentNL0_;
        nd2NzParams.dValue = 1;
        nd2NzParams.srcNdMatrixStride = 0;
        nd2NzParams.srcDValue = 1;
        nd2NzParams.dstNzC0Stride = 1;
        nd2NzParams.dstNzNStride = 1;
        nd2NzParams.dstNzMatrixStride = 1;
        ASC_OP_LOGD("[LoadBiasL1WithPointWise] nd2NzParams.nValue %d nd2NzParams.srcDValue %d.\n",
            nd2NzParams.nValue,
            nd2NzParams.srcDValue);
    }

private:
    Intf *self_ = nullptr;
    uint64_t tensorGmOffset = 0;
    uint64_t currentNL0_ = 0;
    Nd2NzParams nd2NzParams;
};

template <class Intf>
class LoadBiasL0WithBroadcastTools {
public:
    __aicore__ inline LoadBiasL0WithBroadcastTools()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
        tilingNBL1_ = self_->ctx.conv3dTiling->nBL1;
    }

    __aicore__ inline void SetMN(uint64_t m, uint64_t n)
    {
        currentML0_ = m;
        currentNL0_ = n;
    }

    __aicore__ inline void LoadBiasL0WithBroadcast()
    {
        uint32_t offset = 0;
        if (self_->ctx.conv3dTiling->biasFullLoadFlag) {
            offset = self_->ctx.nBL1Iter * tilingNBL1_ * K0_BIAS + 
                     self_->ctx.nBL0Iter * self_->ctx.conv3dTiling->nL0 * K0_BIAS;
        }
        LoadData2DParams loadData2dParams;
        SetLoadData2DParams(loadData2dParams, currentML0_ / BLOCK_L0_M);
        LoadData<typename Intf::BiasT>(self_->ctx.al0BiasB, self_->ctx.biasL1[offset], loadData2dParams);

        InitConstValueParams<typename Intf::BiasT> initConstValueParams;
        // 在B2中，最小处理单元是512B
        SetInitConstValueParams(initConstValueParams, 1, currentNL0_ / BLOCK_L0_N);
        InitConstValue<typename Intf::BiasT>(self_->ctx.bl0BiasB, initConstValueParams);
    }

private:
    __aicore__ inline void SetLoadData2DParams(LoadData2DParams &loadData2dParams, const uint64_t &repeatTimes)
    {
        loadData2dParams.repeatTimes = repeatTimes;
        loadData2dParams.srcStride = 1;
        ASC_OP_LOGD("[LoadBiasL0WithBroadcast] loadData2dParams.repeatTimes %d.\n", loadData2dParams.repeatTimes);
    }

    __aicore__ inline void SetInitConstValueParams(InitConstValueParams<typename Intf::BiasT> &initConstValueParams,
        const uint64_t repeatTimes, const uint64_t blockNum)
    {
        initConstValueParams.repeatTimes = repeatTimes;
        initConstValueParams.blockNum = blockNum;
        initConstValueParams.dstGap = 0;
        initConstValueParams.initValue = 1;
        ASC_OP_LOGD(
            "[LoadBiasL0WithBroadcast] initConstValueParams.repeatTimes %d, initConstValueParams.blockNum %d \n",
            initConstValueParams.repeatTimes,
            initConstValueParams.blockNum);
    }

private:
    Intf *self_ = nullptr;
    uint64_t tilingNBL1_ = 0;
    uint64_t currentML0_ = 0;
    uint64_t currentNL0_ = 0;
};

template <class Intf>
class MMadWithPointWiseTools {
public:
    __aicore__ inline MMadWithPointWiseTools ()
    {}

    __aicore__ inline void SetParams(Intf * self)
    {
        self_ = self;
        // for pointwise, kL0Tail is real tail. we need to align kL0Tail to mmad
        // for bf16/fp16, kL0Tail will align to 16
        // for fp32, due to limits of load2dTranspose, kL0Tail also align to 16.
        alignKL0Tail = AlignB(self_->ctx.kL0Tail, DATA_COPY_OP_LEN);
    }

    __aicore__ inline void SetMN(uint64_t m, uint64_t n)
    {
        currentML0_ = m;
        currentNL0_ = n;
    }

    __aicore__ inline void MadBias()
    {
        MmadParams mmadParams;
        mmadParams.m = currentML0_;
        mmadParams.n = currentNL0_;
        mmadParams.k = K0_BIAS;
        mmadParams.cmatrixInitVal = true;
        mmadParams.cmatrixSource = false;
        mmadParams.isBias = false;
        ASC_OP_LOGD(
            "[MMadWithPointWise] mmadParams.cmatrixInitVal %d, mmadParams.cmatrixSource %d, mmadParams.isBias %d, "
            "mmadParams.k %d, mmadParams.n %d, mmadParams.m %d.\n",
            mmadParams.cmatrixInitVal,
            mmadParams.cmatrixSource,
            mmadParams.isBias,
            mmadParams.k,
            mmadParams.n,
            mmadParams.m);
        Mmad<typename Intf::L0cT, typename Intf::L0cT, typename Intf::L0cT>(
            self_->ctx.cl0, self_->ctx.al0BiasB, self_->ctx.bl0BiasB, mmadParams);
    }

    __aicore__ inline void Mad()
    {
        MmadParams mmadParams;
        mmadParams.m = currentML0_;
        mmadParams.n = currentNL0_;

        if (self_->ctx.kIter != self_->ctx.maxKL0Iter) {
            mmadParams.k = self_->ctx.conv3dTiling->kL0;
        } else {
            mmadParams.k = alignKL0Tail;
        }

        if (self_->ctx.enableBias) {
            mmadParams.cmatrixInitVal = false;
        } else {
            if (self_->ctx.kIter == 0) {
                mmadParams.cmatrixInitVal = true;
            } else {
                mmadParams.cmatrixInitVal = false;
            }
        }
        mmadParams.cmatrixSource = false;
        mmadParams.isBias = false;

        ASC_OP_LOGD(
            "[MMadWithPointWise] mmadParams.cmatrixInitVal %d, mmadParams.cmatrixSource %d, mmadParams.isBias %d, "
            "mmadParams.k %d, mmadParams.n %d, mmadParams.m %d.\n",
            mmadParams.cmatrixInitVal,
            mmadParams.cmatrixSource,
            mmadParams.isBias,
            mmadParams.k,
            mmadParams.n,
            mmadParams.m);
        Mmad<typename Intf::L0cT, typename Intf::WeightT, typename Intf::FmapT>(
            self_->ctx.cl0, self_->ctx.bl0, self_->ctx.al0, mmadParams);
    }
private:
    Intf *self_ = nullptr;
    uint64_t currentML0_ = 0;
    uint64_t currentNL0_ = 0;
    uint64_t alignKL0Tail = 0;
};

template <class Intf>
class CopyOutWithPointWiseTools {
public:
    __aicore__ inline CopyOutWithPointWiseTools()
    {}

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
            intriParams.mSize = self_->ctx.singleCoreCo - self_->ctx.nBL1Iter * self_->ctx.conv3dTiling->nBL1 -
                                self_->ctx.nBL0Iter * self_->ctx.conv3dTiling->nL0;;
            intriParams.srcStride = AlignB(self_->ctx.nL0Tail, BLOCK_L0_M);
        } else {
            intriParams.mSize = self_->ctx.conv3dTiling->nL0;
            intriParams.srcStride = self_->ctx.conv3dTiling->nL0;
        }

        if (self_->ctx.mAL1Iter == self_->ctx.maxMAL1Iter && self_->ctx.mAL0Iter == self_->ctx.maxML0Iter) {
            intriParams.nSize = self_->ctx.singleCoreM - self_->ctx.mAL1Iter * self_->ctx.conv3dTiling->mAL1 -
                                self_->ctx.mAL0Iter * self_->ctx.conv3dTiling->mL0;;
        } else {
            intriParams.nSize = self_->ctx.conv3dTiling->mL0;
        }

        intriParams.dstStride = valueHoWo_ * self_->ctx.orgDo;
        intriParams.quantPre = GetQuantPre();
        intriParams.reluEn = false;

        ASC_OP_LOGD("[CopyOutWithPointWise] intriParams.nSize %d, intriParams.mSize %d, intriParams.srcStride %d, "
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
        ASC_OP_LOGD("[CopyOutWithPointWise] offset %d.\n", offset);
        if constexpr (!(AscendC::IsSameType<typename Intf::L0cT, int32_t>::value &&
                AscendC::IsSameType<typename Intf::OutputT, bfloat16_t>::value)) {
            Fixpipe<typename Intf::OutputT, typename Intf::L0cT, CFG_ROW_MAJOR>(output[offset], self_->ctx.cl0, intriParams);
        }
    }

    __aicore__ inline void CopyOut2Workspace(const GlobalTensor<typename Intf::L0cT> &output)
    {
        // impl for pointwise
    }

    __aicore__ inline void CopyUBOut(const GlobalTensor<typename Intf::OutputT> &output, uint32_t mIter,
                                     uint32_t nIter, uint32_t m, uint32_t n, const LocalTensor<typename Intf::BF16T> &ubout)
    {
        // impl for pointwise
    }

    __aicore__ inline void GetL0CSize(uint64_t &mSize, uint64_t &nSize)
    {
        // impl for pointwise
    }

    __aicore__ inline uint64_t GetChannelOffset(uint64_t noffset)
    {
        return 0;
    }

private:
    __aicore__ inline uint64_t CalcFixpipeOffset()
    {
        uint64_t offsetM = tilingMAL1_ * self_->ctx.mAL1Iter + self_->ctx.conv3dTiling->mL0 * self_->ctx.mAL0Iter;
        // 当前每次只出一个dout
        uint64_t offsetDout = self_->ctx.dOutIter;
        return (tilingNBL1_ * self_->ctx.nBL1Iter  + self_->ctx.conv3dTiling->nL0 * self_->ctx.nBL0Iter) *
                self_->ctx.orgDo * valueHoWo_ + offsetDout * valueHoWo_ + offsetM;
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

#endif  // __CONV3D_POINTWISE_SUB_API_H__