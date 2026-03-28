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
 * \file conv3d_groupopt_sub_api.h
 * \brief
 */

#ifndef CONV3D_GROUPOPT_SUB_API_H
#define CONV3D_GROUPOPT_SUB_API_H

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
class LoadBL1WithGroupOptTools {
public:
    __aicore__ inline LoadBL1WithGroupOptTools()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
    }

    __aicore__ inline void LoadBL1()
    {
        PreProcess();
        if (specialGroupDimTial()) {
            return;
        }
        if (currentNBL1 >= self_->ctx.orgCo) {
            ASC_OP_LOGD("[LoadBL1WithGroupOptTools] Process single instruction weight \n");
            uint64_t repeatTimes = (currentKBL1 * self_->ctx.conv3dTiling->nBL1) / (BLOCK_L0_N * self_->ctx.cin0);
            if (repeatTimes > LOAD2D_MAX_REPEAT_TIMES) {
                dataCopyParams.blockCount = 1;
                dataCopyParams.srcStride = 0;
                dataCopyParams.blockLen = CeilDIV(currentKBL1 * self_->ctx.conv3dTiling->nBL1, self_->ctx.cin0);
                DataCopy<typename Intf::WeightT>(self_->ctx.bl1, self_->ctx.bgm[bL1GmOffset], dataCopyParams);
            } else {
                SetLoadData2DParams(repeatTimes);
                LoadData<typename Intf::WeightT>(self_->ctx.bl1, self_->ctx.bgm[bL1GmOffset], loadData2dParams);
            }
        } else {
            ASC_OP_LOGD("[LoadBL1WithGroupOptTools] Process multi instruction weight \n");
            dataCopyParams.blockCount = currentKBL1 / self_->ctx.cin0;
            dataCopyParams.blockLen = currentNBL1;
            dataCopyParams.srcStride = self_->ctx.orgCoAlignN0 - currentNBL1;
            DataCopy<typename Intf::WeightT>(self_->ctx.bl1, self_->ctx.bgm[bL1GmOffset], dataCopyParams);
        }
        ASC_OP_LOGD("[LoadBL1WithGroupOptTools] bL1GmOffset %d self_->ctx.orgCo %d.\n", bL1GmOffset, self_->ctx.orgCo);
    }

private:
    __aicore__ inline bool IsNTail()
    {
        return self_->ctx.nBL1Iter == self_->ctx.maxNBL1Iter;
    }

    __aicore__ inline bool IsKBL1Tail()
    {
        return self_->ctx.kBL1Iter == self_->ctx.maxKBL1Iter;
    }

    __aicore__ inline void PreProcess()
    {
        bL1GmOffset = self_->ctx.kBL1Iter * self_->ctx.groupKBL1 * self_->ctx.orgCoAlignN0 +
                      self_->ctx.nBL1Iter * self_->ctx.conv3dTiling->nBL1 * self_->ctx.cin0;
        currentNBL1 = self_->ctx.conv3dTiling->nBL1;
        if (self_->ctx.isGroupOptDimTail) {
            currentNBL1 =
                self_->ctx.singleCoreCo >= currentNBL1 ? currentNBL1 : AlignB(self_->ctx.singleCoreCo, BLOCK_L0_N);
        }
        currentNBL1 = IsNTail() ? self_->ctx.nBL1TailAlign : currentNBL1;
        currentKBL1 = IsKBL1Tail() ? self_->ctx.groupKBL1Tail : self_->ctx.groupKBL1;
        ASC_OP_LOGD("[LoadBL1WithGroupOptTools] currentNBL1 %d currentKBL1 %d.\n", currentNBL1, currentKBL1);
        loadData2dParams.srcStride = 1;
    }

    __aicore__ inline void LoadData2dWithBeyond255(
        const uint64_t bL1Start, const uint64_t bL1GmStart, const uint64_t repeatTimes)
    {
        uint64_t bL1Offset = bL1Start;
        uint64_t bL1GmOffsetTmp = bL1GmStart;
        for (uint64_t i = 0; i < repeatTimes / LOAD2D_MAX_REPEAT_TIMES; i++) {
            SetLoadData2DParams(LOAD2D_MAX_REPEAT_TIMES);
            LoadData<typename Intf::WeightT>(
                self_->ctx.bl1[bL1Offset], self_->ctx.bgm[bL1GmOffsetTmp], loadData2dParams);
            bL1GmOffsetTmp += LOAD2D_MAX_REPEAT_TIMES * BLOCK_L0_N * self_->ctx.cin0;
            bL1Offset += LOAD2D_MAX_REPEAT_TIMES * BLOCK_L0_N * self_->ctx.cin0;
        }
        uint64_t tail = repeatTimes % LOAD2D_MAX_REPEAT_TIMES;
        if (tail != 0) {
            SetLoadData2DParams(tail);
            LoadData<typename Intf::WeightT>(
                self_->ctx.bl1[bL1Offset], self_->ctx.bgm[bL1GmOffsetTmp], loadData2dParams);
        }
        ASC_OP_LOGD("[LoadBL1WithGroupOptTools] bL1GmOffsetTmp %d.\n", bL1GmOffsetTmp);
    }

    __aicore__ inline bool specialGroupDimTial()
    {
        if (self_->ctx.isGroupOptDimTail && self_->ctx.groupOptIter >= self_->ctx.maxGroupOptIter - 1 &&
            AlignB(self_->ctx.orgCi, self_->ctx.cin0) * self_->ctx.conv3dTiling->groupOpt >
                AlignB(self_->ctx.conv3dTiling->orgCi, self_->ctx.cin0)) {
        } else {
            return false;
        }

        ASC_OP_LOGD("[LoadBL1WithGroupOptTools] Process groupdimtail weight \n");
        uint64_t repeatTimes = currentNBL1 / BLOCK_L0_N;
        uint64_t bL1GmOffsetTmp = 0;
        uint64_t bl1Offset = 0;
        uint64_t cin0xKhxKw = self_->ctx.cin0 * self_->ctx.kernelHxkernelW;
        uint64_t currentKd = CeilDIV(currentKBL1, self_->ctx.cin1 * cin0xKhxKw);
        uint64_t cin1xKhxKw = self_->ctx.cin1 * self_->ctx.kernelHxkernelW;
        if (currentKd == 1) {
            cin1xKhxKw = currentKBL1 / self_->ctx.cin0;
            uint64_t skipCinRepeats =
                (self_->ctx.kBL1Iter * self_->ctx.groupKBL1) / (self_->ctx.cin1 * cin0xKhxKw);
            uint64_t skipCinOffset = (AlignB(self_->ctx.orgCi, self_->ctx.cin0) - self_->ctx.cin1 * self_->ctx.cin0) *
                                     self_->ctx.kernelHxkernelW * self_->ctx.orgCoAlignN0;
            if (skipCinRepeats > 0) {
                bL1GmOffset += (skipCinOffset * skipCinRepeats);
            }
        } else {
            bL1GmOffset = self_->ctx.kBL1Iter * (self_->ctx.groupKBL1 / (self_->ctx.cin0 * self_->ctx.cin1)) *
                              self_->ctx.orgCi * self_->ctx.orgCoAlignN0 +
                          self_->ctx.nBL1Iter * self_->ctx.conv3dTiling->nBL1 * self_->ctx.cin0;
        }
        ASC_OP_LOGD("[LoadBL1WithGroupOptTools] currentKd %d cin1xKhxKw %d\n", currentKd, cin1xKhxKw);
        for (uint64_t i = 0; i < currentKd; i++) {
            bL1GmOffsetTmp = bL1GmOffset + i * AlignB(self_->ctx.orgCi, self_->ctx.cin0) *
                                               AlignB(self_->ctx.orgCo, self_->ctx.cin0) * self_->ctx.kernelHxkernelW;
            for (uint64_t j = 0; j < cin1xKhxKw; j++) {
                ASC_OP_LOGD("[LoadBL1WithGroupOptTools] bL1GmOffsetTmp, bl1Offset %d %d \n", bL1GmOffsetTmp, bl1Offset);
                if (repeatTimes > LOAD2D_MAX_REPEAT_TIMES) {
                    LoadData2dWithBeyond255(bl1Offset, bL1GmOffsetTmp, repeatTimes);
                } else {
                    SetLoadData2DParams(repeatTimes);
                    LoadData<typename Intf::WeightT>(
                        self_->ctx.bl1[bl1Offset], self_->ctx.bgm[bL1GmOffsetTmp], loadData2dParams);
                }
                bL1GmOffsetTmp += self_->ctx.orgCoAlignN0 * self_->ctx.cin0;
                bl1Offset += currentNBL1 * self_->ctx.cin0;
            }
        }

        return true;
    }

    __aicore__ inline void SetLoadData2DParams(const uint64_t &repeatTimes)
    {
        loadData2dParams.repeatTimes = repeatTimes;
        ASC_OP_LOGD("[LoadBL1WithGroupOptTools] loadData2dParams.repeatTimes %d.\n", loadData2dParams.repeatTimes);
    }

private:
    Intf *self_ = nullptr;
    uint64_t bL1GmOffset = 0;
    uint64_t currentNBL1 = 0;
    uint64_t currentKBL1 = 0;
    LoadData2DParams loadData2dParams;
    DataCopyParams dataCopyParams;
};

template <class Intf>
class LoadBL0WithGroupOptTools {
public:
    __aicore__ inline LoadBL0WithGroupOptTools()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
        k0_ = ONE_BLK_SIZE / self_->ctx.sizeOfWeight;
    }

    __aicore__ inline void SetN(uint64_t n)
    {
        currentNL0_ = n;
        numNL0Block_ = currentNL0_ / BLOCK_L0_N;
    }

    __aicore__ inline void LoadBL0()
    {
        if constexpr (Intf::bl1bypass) {
            tilingNBSrc_ = self_->ctx.orgCoAlignN0;
        } else {
            tilingNBSrc_ = self_->ctx.nBL1Iter == self_->ctx.maxNBL1Iter ? self_->ctx.nBL1TailAlign
                                                                         : self_->ctx.conv3dTiling->nBL1;
        }
        currentKL0 = self_->ctx.kIter == self_->ctx.maxKL0Iter ? self_->ctx.kL0Tail : self_->ctx.singleCoreKL0;
        // set LoadData2DParamsV2
        load2dSrcOffset = self_->ctx.kBL0Iter * self_->ctx.singleCoreKL0 * tilingNBSrc_ +
                          self_->ctx.nBL0Iter * self_->ctx.conv3dTiling->nL0 * k0_;
        ASC_OP_LOGD("[LoadBL0WithGroupOptTools] currentKL0: %u\n", currentKL0);
        if constexpr (!Intf::bl1bypass) {
            if (tilingNBSrc_ == currentNL0_) {
                SetLoadData2DParams(loadData2dParams, numNL0Block_ * (currentKL0 / k0_));
                LoadData<typename Intf::WeightT>(self_->ctx.bl0, self_->ctx.bl1[load2dSrcOffset], loadData2dParams);
            } else {
                SetLoadData2DParams(loadData2dParams, numNL0Block_);
                uint64_t tmp1 = currentNL0_ * k0_;
                uint64_t tmp2 = tilingNBSrc_ * k0_;
                for (uint32_t copy_part = 0; copy_part < currentKL0 / k0_; ++copy_part) {
                    LoadData<typename Intf::WeightT>(self_->ctx.bl0[copy_part * tmp1],
                        self_->ctx.bl1[load2dSrcOffset + copy_part * tmp2],
                        loadData2dParams);
                }
            }
        } else {
            if (specialGroupDimTial()) {
                return;
            }
            if (currentNL0_ >= self_->ctx.orgCo) {
                ASC_OP_LOGD("[LoadBL0WithGroupOptTools] Process single instruction weight \n");
                SetLoadData2DParams(loadData2dParams, numNL0Block_ * (currentKL0 / k0_));
                LoadData<typename Intf::WeightT>(self_->ctx.bl0, self_->ctx.bgm[load2dSrcOffset], loadData2dParams);
            } else {
                ASC_OP_LOGD("[LoadBL0WithGroupOptTools] Process multi instruction weight \n");
                SetLoadData2DParams(loadData2dParams, numNL0Block_);
                uint64_t tmp1 = currentNL0_ * k0_;
                uint64_t tmp2 = tilingNBSrc_ * k0_;
                for (uint32_t copy_part = 0; copy_part < currentKL0 / k0_; ++copy_part) {
                    LoadData<typename Intf::WeightT>(self_->ctx.bl0[copy_part * tmp1],
                        self_->ctx.bgm[load2dSrcOffset + copy_part * tmp2],
                        loadData2dParams);
                }
            }
        }
    }

private:
    __aicore__ inline void SetLoadData2DParams(LoadData2DParams &loadData2dParams, const uint64_t &repeatTimes)
    {
        loadData2dParams.repeatTimes = repeatTimes;
        loadData2dParams.srcStride = 1;
        ASC_OP_LOGD("[LoadBL0WithGroupOptTools] loadData2dParams.repeatTimes %d.\n", loadData2dParams.repeatTimes);
    }

    __aicore__ inline bool specialGroupDimTial()
    {
        if (self_->ctx.isGroupOptDimTail && self_->ctx.groupOptIter >= self_->ctx.maxGroupOptIter - 1 &&
            AlignB(self_->ctx.orgCi, self_->ctx.cin0) * self_->ctx.conv3dTiling->groupOpt >
                AlignB(self_->ctx.conv3dTiling->orgCi, self_->ctx.cin0)) {
        } else {
            return false;
        }

        ASC_OP_LOGD("[LoadBL0WithGroupOptTools] Process groupdimtail weight \n");
        SetLoadData2DParams(loadData2dParams, numNL0Block_);
        uint64_t tmp1 = currentNL0_ * k0_;
        uint64_t tmp2 = tilingNBSrc_ * k0_;
        uint64_t bgmOffset = 0;
        uint64_t bl0Offset = 0;
        uint64_t cin0xKhxKw = self_->ctx.cin0 * self_->ctx.kernelHxkernelW;
        uint64_t currentKd = CeilDIV(currentKL0, self_->ctx.cin1 * cin0xKhxKw);
        uint64_t cin1xKhxKw = self_->ctx.cin1 * self_->ctx.kernelHxkernelW;
        if (currentKd == 1) {
            cin1xKhxKw = currentKL0 / self_->ctx.cin0;
            uint64_t skipCinRepeats =
                (self_->ctx.kBL0Iter * self_->ctx.conv3dTiling->kL0) / (self_->ctx.cin1 * cin0xKhxKw);
            uint64_t skipCinOffset = (AlignB(self_->ctx.orgCi, self_->ctx.cin0) - self_->ctx.cin1 * self_->ctx.cin0) *
                                     self_->ctx.kernelHxkernelW * tilingNBSrc_;
            if (skipCinRepeats > 0) {
                load2dSrcOffset += (skipCinOffset * skipCinRepeats);
            }
        } else {
            load2dSrcOffset = self_->ctx.kBL0Iter * (self_->ctx.conv3dTiling->kL0 / (k0_ * self_->ctx.cin1)) *
                                  self_->ctx.orgCi * tilingNBSrc_ +
                              self_->ctx.nBL0Iter * self_->ctx.conv3dTiling->nL0 * k0_;
        }
        ASC_OP_LOGD("[LoadBL0WithGroupOptTools] currentKd %d cin1xKhxKw %d\n", currentKd, cin1xKhxKw);
        for (uint32_t copy_part = 0; copy_part < currentKd; copy_part++) {
            bgmOffset = load2dSrcOffset + copy_part * AlignB(self_->ctx.orgCi, self_->ctx.cin0) *
                                              AlignB(self_->ctx.orgCo, self_->ctx.cin0) * self_->ctx.kernelHxkernelW;
            for (uint64_t j = 0; j < cin1xKhxKw; j++) {
                LoadData<typename Intf::WeightT>(
                    self_->ctx.bl0[bl0Offset], self_->ctx.bgm[bgmOffset], loadData2dParams);
                ASC_OP_LOGD("[LoadBL0WithGroupOptTools] bgmOffset, bl0Offset %d %d \n", bgmOffset, bl0Offset);
                bgmOffset += tmp2;
                bl0Offset += tmp1;
            }
        }

        return true;
    }

private:
    Intf *self_ = nullptr;
    uint64_t tilingNBSrc_ = 0;
    uint64_t k0_ = 16;
    uint64_t currentNL0_ = 0;
    uint64_t currentKL0 = 0;
    uint64_t numNL0Block_ = 0;
    uint64_t ratioOfNToN0_ = 0;
    uint64_t load2dSrcOffset = 0;
    LoadData2DParams loadData2dParams;
};

};  // namespace Conv3dFunc

#endif  // __CONV3D_SUB_API_H__
