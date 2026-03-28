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
 * \file conv3d_mte2_pointwise_sub_api.h
 * \brief
 */

#ifndef CONV3D_MTE2_POINTWISE_SUB_API_H
#define CONV3D_MTE2_POINTWISE_SUB_API_H

#include "conv3d_util.h"
#include "kernel_tpipe.h"
#include "kernel_operator_data_copy_intf.h"
#include "kernel_operator_mm_intf.h"
#include "kernel_operator_fixpipe_intf.h"
#include "kernel_utils.h"

using namespace AscendC;
using namespace conv;
using namespace conv3d;

namespace Conv3dFunc {
template <class Intf>
class LoadAL1WithPointWiseTools {
public:
    __aicore__ inline LoadAL1WithPointWiseTools()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
        al1BlockNum = AL1_BLOCK_SIZE / self_->ctx.sizeOfFmap;
        dihiwi = self_->ctx.orgDi * self_->ctx.conv3dTiling->orgHixWi;
        alignMAL1Tail = AlignB(self_->ctx.mAL1Tail, BLOCK_L0_M);
        alignKAL1Tail = AlignB(self_->ctx.conv3dTiling->kAL1Tail, DATA_COPY_OP_LEN);
    }

    __aicore__ inline void LoadAL1()
    {
        PreProcess();
        if (IsMTail() || IsKAL1Tail()) {
            // L1 刷零，大小按照L0B对齐
            InitConstValueParams<typename Intf::FmapT> initConstValueParams;
            SetInitConstValueParams(initConstValueParams, 1, 
                alignCurrentMAL1 * alignCurrentKAL1 / al1BlockNum);
            InitConstValue<typename Intf::FmapT>(self_->ctx.al1, initConstValueParams);
            PipeBarrier<PIPE_MTE2>();
        }
         
        if constexpr (AscendC::IsSameType<typename Intf::FmapT, float>::value) {
            loadAL1ByKAlignTo16();
        } else {
            loadAL1ByNormal();
        }
        ASC_OP_LOGD("[LoadAL1WithPointWise] nd2NzParams.nValue %d nd2NzParams.dValue %d "
                    "nd2NzParams.srcDValue %d nd2NzParams.dstNzC0Stride %d.\n",
                    nd2NzParams.nValue, nd2NzParams.dValue, nd2NzParams.srcDValue, nd2NzParams.dstNzC0Stride);    
    }

private:
    __aicore__ inline void SetInitConstValueParams(InitConstValueParams<typename Intf::FmapT> &initConstValueParams,
        const uint64_t repeatTimes, const uint64_t blockNum)
    {
        initConstValueParams.repeatTimes = repeatTimes;
        initConstValueParams.blockNum = blockNum;
        initConstValueParams.dstGap = 0;
        initConstValueParams.initValue = 0;
        ASC_OP_LOGD(
            "[LoadAL1WithPointWise] initConstValueParams.repeatTimes %d, initConstValueParams.blockNum %d \n",
            initConstValueParams.repeatTimes,
            initConstValueParams.blockNum);
    }

    __aicore__ inline void SetNd2NzParams(uint16_t nValue, uint16_t srcDValue, uint16_t dstNzC0Stride)
    {
        nd2NzParams.ndNum = 1;
        nd2NzParams.nValue = nValue;
        nd2NzParams.dValue = currentMAL1;
        nd2NzParams.srcNdMatrixStride = 0;
        nd2NzParams.srcDValue = srcDValue;
        nd2NzParams.dstNzC0Stride = dstNzC0Stride;
        nd2NzParams.dstNzNStride = 1;
        nd2NzParams.dstNzMatrixStride = 1;
    }

    __aicore__ inline bool IsMTail()
    {
        return self_->ctx.mAL1Iter == self_->ctx.maxMAL1Iter;
    }

    __aicore__ inline bool IsKAL1Tail()
    {
        return self_->ctx.kAL1Iter == self_->ctx.maxKAL1Iter;
    }

    __aicore__ inline void PreProcess()
    {
        if (IsMTail()) {
            currentMAL1 = self_->ctx.mAL1Tail;
            alignCurrentMAL1 = alignMAL1Tail;
        } else {
            currentMAL1 = self_->ctx.conv3dTiling->mAL1;
            alignCurrentMAL1 = currentMAL1; // mAL1 has align to 16
        }

        if (IsKAL1Tail()) {
            currentKAL1 = self_->ctx.conv3dTiling->kAL1Tail;
            alignCurrentKAL1 = alignKAL1Tail;
        } else {
            currentKAL1 = self_->ctx.conv3dTiling->kAL1;
            alignCurrentKAL1 = currentKAL1; // kAL1 has align to 16 in tiling
        }
        
        aL1GmOffset = self_->ctx.kAL1Iter * self_->ctx.conv3dTiling->kAL1 * dihiwi +
                    self_->ctx.dOutIter * self_->ctx.conv3dTiling->orgHixWi +
                    self_->ctx.mAL1Iter * self_->ctx.conv3dTiling->mAL1;
        ASC_OP_LOGD("[LoadAL1WithPointWise] aL1GmOffset %u\n", aL1GmOffset);
    }

    __aicore__ inline void loadAL1ByNormal()
    {
        if (dihiwi > MAX_UINT16) {
            SetNd2NzParams(1, currentMAL1, alignCurrentKAL1);
            for (uint64_t i = 0; i < currentKAL1; ++i) {
                DataCopy<typename Intf::FmapT>(self_->ctx.al1[i * al1BlockNum],
                                            self_->ctx.agm[aL1GmOffset + i * dihiwi],
                                            nd2NzParams);
            }
        } else {
            SetNd2NzParams(currentKAL1, dihiwi, alignCurrentKAL1);
            DataCopy<typename Intf::FmapT>(self_->ctx.al1,
                                            self_->ctx.agm[aL1GmOffset],
                                            nd2NzParams);
        }
    }

    __aicore__ inline void loadAL1ByKAlignTo16()
    {
        uint64_t singleDataCopyElements = DATA_COPY_OP_LEN * alignCurrentMAL1;
        uint64_t dataCopyTimes = alignCurrentKAL1 / DATA_COPY_OP_LEN;
        uint64_t tailKAL1 = currentKAL1 % DATA_COPY_OP_LEN;

        if (dihiwi > MAX_UINT16) {
            SetNd2NzParams(1, currentMAL1, DATA_COPY_OP_LEN);
        } else {
            SetNd2NzParams(DATA_COPY_OP_LEN, dihiwi, DATA_COPY_OP_LEN);
        }
        uint64_t singleDataCopyOffset = 0;
        uint64_t currentDataCopyLines = 0;
        uint64_t upperLimits = tailKAL1 == 0 ? dataCopyTimes : dataCopyTimes - 1;
        for (uint64_t copyIdx = 0; copyIdx < upperLimits; ++copyIdx) {
            singleDataCopyOffset = copyIdx * singleDataCopyElements;
            currentDataCopyLines = copyIdx * DATA_COPY_OP_LEN;
            if (dihiwi > MAX_UINT16) {
                for (uint64_t innerTime = 0; innerTime < DATA_COPY_OP_LEN; ++innerTime) {
                    DataCopy<typename Intf::FmapT>(self_->ctx.al1[singleDataCopyOffset + innerTime * al1BlockNum],
                                        self_->ctx.agm[aL1GmOffset + dihiwi * (currentDataCopyLines + innerTime)],
                                        nd2NzParams);
                }
            } else {
                DataCopy<typename Intf::FmapT>(self_->ctx.al1[singleDataCopyOffset],
                                        self_->ctx.agm[aL1GmOffset + dihiwi * currentDataCopyLines],
                                        nd2NzParams);
            } 
        }

        if (tailKAL1) {
            singleDataCopyOffset = upperLimits * singleDataCopyElements;
            currentDataCopyLines = upperLimits * DATA_COPY_OP_LEN;
            if (dihiwi > MAX_UINT16) {
                for (uint64_t innerTime = 0; innerTime < tailKAL1; ++innerTime) {
                    DataCopy<typename Intf::FmapT>(self_->ctx.al1[singleDataCopyOffset + innerTime * al1BlockNum],
                                        self_->ctx.agm[aL1GmOffset + dihiwi * (currentDataCopyLines + innerTime)],
                                        nd2NzParams);
                }
            } else {
                nd2NzParams.nValue = tailKAL1;
                DataCopy<typename Intf::FmapT>(self_->ctx.al1[singleDataCopyOffset],
                                        self_->ctx.agm[aL1GmOffset + dihiwi * currentDataCopyLines],
                                        nd2NzParams);
            }
        }
    }

private:
    Intf *self_ = nullptr;
    uint64_t currentMAL1 = 0;
    uint64_t currentKAL1 = 0;
    uint64_t al1BlockNum = 16;
    uint64_t alignMAL1Tail = 0;
    uint64_t alignKAL1Tail = 0;
    uint64_t alignCurrentMAL1 = 0;
    uint64_t alignCurrentKAL1 = 0;
    uint64_t dihiwi = 0;
    uint64_t aL1GmOffset = 0;
    Nd2NzParams nd2NzParams;
};

template <class Intf>
class LoadBL1WithPointWiseTools {
public:
    __aicore__ inline LoadBL1WithPointWiseTools()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
        bl1BlockNum = ONE_BLK_SIZE / self_->ctx.sizeOfWeight;
        alignNBL1Tail = AlignB(self_->ctx.nBL1Tail, BLOCK_L0_M);
        alignKBL1Tail = AlignB(self_->ctx.conv3dTiling->kBL1Tail, DATA_COPY_OP_LEN);
    }

    __aicore__ inline void LoadBL1()
    {
        PreProcess();
        if (IsNTail() || IsKBL1Tail()) {
            // L1 刷零
            InitConstValueParams<typename Intf::WeightT> initConstValueParams;
            SetInitConstValueParams(initConstValueParams, 1, 
                alignCurrentNBL1 * alignCurrentKBL1 / bl1BlockNum);
            InitConstValue<typename Intf::WeightT>(self_->ctx.bl1, initConstValueParams);
            PipeBarrier<PIPE_MTE2>();
        }
    
        uint64_t bL1GmOffset = self_->ctx.nBL1Iter * self_->ctx.conv3dTiling->nBL1 * self_->ctx.orgCi +
                                self_->ctx.kBL1Iter * self_->ctx.conv3dTiling->kBL1;
        ASC_OP_LOGD("[LoadBL1WithPointWise] bL1GmOffset %u\n", bL1GmOffset);
        SetNd2NzParams();
        if (self_->ctx.orgCi > MAX_UINT16) {
            nd2NzParams.srcDValue = currentKBL1;
            nd2NzParams.nValue = 1;
            ASC_OP_LOGD("[LoadBL1WithPointWise] srcDValue Beyond MAX_UINT16. LoopTimes %d nd2NzParams.nValue %d "
                        "nd2NzParams.dValue %d nd2NzParams.srcDValue %d nd2NzParams.dstNzC0Stride %d.\n",
                        currentNBL1, nd2NzParams.nValue, nd2NzParams.dValue,
                        nd2NzParams.srcDValue, nd2NzParams.dstNzC0Stride);
            for (uint64_t i = 0; i < currentNBL1; ++i) {
                DataCopy<typename Intf::WeightT>(self_->ctx.bl1[i * bl1BlockNum],
                                                self_->ctx.bgm[bL1GmOffset + self_->ctx.orgCi * i], nd2NzParams);
            }
        } else {
            ASC_OP_LOGD("[LoadBL1WithPointWise] nd2NzParams.nValue %d "
                        "nd2NzParams.dValue %d nd2NzParams.srcDValue %d nd2NzParams.dstNzC0Stride %d.\n",
                        nd2NzParams.nValue, nd2NzParams.dValue,
                        nd2NzParams.srcDValue, nd2NzParams.dstNzC0Stride);
            DataCopy<typename Intf::WeightT>(self_->ctx.bl1, self_->ctx.bgm[bL1GmOffset], nd2NzParams);
        }
    }

private:
    __aicore__ inline void PreProcess()
    {
        if (IsNTail()) {
            currentNBL1 = self_->ctx.nBL1Tail;
            alignCurrentNBL1 = alignNBL1Tail;
        } else {
            currentNBL1 = self_->ctx.conv3dTiling->nBL1;
            alignCurrentNBL1 = currentNBL1;
        }

        if (IsKBL1Tail()) {
            currentKBL1 = self_->ctx.conv3dTiling->kBL1Tail;
            alignCurrentKBL1 = alignKBL1Tail;
        } else {
            currentKBL1 = self_->ctx.conv3dTiling->kBL1;
            alignCurrentKBL1 = currentKBL1;
        }
    }

        __aicore__ inline bool IsNTail()
    {
        return self_->ctx.nBL1Iter == self_->ctx.maxNBL1Iter;
    }

    __aicore__ inline bool IsKBL1Tail()
    {
        return self_->ctx.kBL1Iter == self_->ctx.maxKBL1Iter;
    }

    __aicore__ inline void SetInitConstValueParams(InitConstValueParams<typename Intf::WeightT> &initConstValueParams,
        const uint64_t repeatTimes, const uint64_t blockNum)
    {
        initConstValueParams.repeatTimes = repeatTimes;
        initConstValueParams.blockNum = blockNum;
        initConstValueParams.dstGap = 0;
        initConstValueParams.initValue = 0;
        ASC_OP_LOGD(
            "[LoadBL1WithPointWise] initConstValueParams.repeatTimes %d, initConstValueParams.blockNum %d \n",
            initConstValueParams.repeatTimes,
            initConstValueParams.blockNum);
    }

    __aicore__ inline void SetNd2NzParams()
    {
        nd2NzParams.ndNum = 1;
        nd2NzParams.nValue = currentNBL1;
        nd2NzParams.dValue = currentKBL1;
        nd2NzParams.srcNdMatrixStride = 0;
        nd2NzParams.srcDValue = self_->ctx.orgCi;
        nd2NzParams.dstNzC0Stride = alignCurrentNBL1;
        nd2NzParams.dstNzNStride = 1;
        nd2NzParams.dstNzMatrixStride = 1;
    }

private:
    Intf *self_ = nullptr;
    uint64_t bl1BlockNum = 16;
    uint64_t currentNBL1 = 0;
    uint64_t currentKBL1 = 0;
    uint64_t alignCurrentNBL1 = 0;
    uint64_t alignCurrentKBL1 = 0;
    uint64_t alignNBL1Tail = 0;
    uint64_t alignKBL1Tail = 0;
    Nd2NzParams nd2NzParams;
};

};  // namespace Conv3dFunc

#endif  // __CONV3D_MTE2_POINTWISE_SUB_API_H__