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
 * \file conv3d_mte2_hw_mode_sub_api.h
 * \brief
 */

#ifndef CONV3D_MTE2_HW_MODE_SUB_API_H
#define CONV3D_MTE2_HW_MODE_SUB_API_H

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
class LoadAL1WithHwModeTools {
public:
    __aicore__ inline LoadAL1WithHwModeTools() {}
 
    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
    }
 
    __aicore__ inline LoadData3DParamsV2<typename Intf::FmapT>& GetLoadData3DParams()
    {
        return loadData3Dv2Params;
    }
 
    __aicore__ inline void SetLoadData3DParams()
    {
        // img2col fmap param info, which set in each ddr->L1
        loadData3Dv2Params.strideW = self_->ctx.strideW;
        loadData3Dv2Params.strideH = self_->ctx.strideH;
        loadData3Dv2Params.filterW = self_->ctx.kernelW;
        loadData3Dv2Params.filterSizeW = self_->ctx.kernelW >> RIGHT_MOVE_8;
        loadData3Dv2Params.filterH = self_->ctx.kernelH;
        loadData3Dv2Params.filterSizeH = self_->ctx.kernelH >> RIGHT_MOVE_8;
        loadData3Dv2Params.dilationFilterW = self_->ctx.dilationW;
        loadData3Dv2Params.dilationFilterH = self_->ctx.dilationH;
    }
 
    __aicore__ inline bool GetSet2dFlag()
    {
        return aL1IsFullPad;
    }
 
    __aicore__ inline void LoadAL1()
    {
        PreProcess();
        ProcessWoutMiddleData();
        ProcessHoutMiddleData();
        ProcessDoutMiddleData();
        if (aL1IsFullPad) [[unlikely]] {
            ASC_OP_LOGD("[LoadAL1] aL1 is all pad, skip data copy!\n");
            return;
        }
        SetFmatrixForLoad3D();
        CopyDataGMToL1();
        ASC_OP_LOGD("[LoadAL1] hoStartIdx %d, hoEndIdx %d, hiLoadL1 %d, hiIdx %d, "
                "kdL1Idx %d, cin1L1Idx %d, diIdx %d, cin1LoadL1 %d, aL1GmOffset %d.\n",
            hoStartIdx,
            hoEndIdx,
            hiLoadL1,
            hiIdx,
            kdL1Idx,
            cin1L1Idx,
            diIdx,
            cin1LoadL1,
            aL1GmOffset);
    }
 
private:
    __aicore__ inline void SetFmatrixForLoad3D()
    {
        uint8_t padList[PAD_SIZE] = {0};
        padList[PAD_IDX_L] = padLeftL1;
        padList[PAD_IDX_R] = padRightL1;
        padList[PAD_IDX_T] = padTopL1;
        padList[PAD_IDX_B] = padBottomL1;
        SetFmatrix(hiLoadL1, wiLoadL1, padList, FmatrixMode::FMATRIX_LEFT);
    }
 
     __aicore__ inline void CopyDataGMToL1()
    {
        CalcAL1GmOffset();
        uint64_t blockCount = hiLoadL1;
        bool srcStrideBeyondMaxU16 = false;
        if (self_->ctx.orgWi - wiLoadL1 > MAX_UINT16) [[unlikely]] {
            blockCount = 1;
            srcStrideBeyondMaxU16 = true;
        }
        // todo din != 1
        uint64_t aL1Offset = 0;
        SetDataCopyParams(blockCount);
        for (uint64_t i = 0; i < cin1LoadL1; i++) {
            if (srcStrideBeyondMaxU16) [[unlikely]] {
                uint64_t aL1GmSubOffset = aL1GmOffset;
                uint64_t aL1SubOffset = aL1Offset;
                for (uint64_t j = 0; j < hiLoadL1; j++) {
                    DataCopy<typename Intf::FmapT>(
                        self_->ctx.al1[aL1SubOffset], self_->ctx.agm[aL1GmSubOffset], dataCopyParams);
                    aL1SubOffset += wiLoadL1 * self_->ctx.cin0;
                    aL1GmSubOffset += self_->ctx.orgWi * self_->ctx.cin0;
                }
            } else {
                SetDataCopyParams(hiLoadL1);
                DataCopy<typename Intf::FmapT>(self_->ctx.al1[aL1Offset], self_->ctx.agm[aL1GmOffset], dataCopyParams);
            }
            aL1Offset += hiLoadL1 * wiLoadL1 * self_->ctx.cin0;
            aL1GmOffset += self_->ctx.orgHi * self_->ctx.orgWi * self_->ctx.cin0;
        }
    }
 
    __aicore__ inline void CalcAL1GmOffset()
    {
        aL1GmOffset = diIdx * self_->ctx.cin1 * self_->ctx.orgHi * self_->ctx.orgWi * self_->ctx.cin0 +
                      cin1L1Idx * self_->ctx.orgHi * self_->ctx.orgWi * self_->ctx.cin0 +
                      hiIdx * self_->ctx.orgWi * self_->ctx.cin0 +
                      wiIdx * self_->ctx.cin0;
        ASC_OP_LOGD("[LoadAL1] aL1GmOffset %d.\n", aL1GmOffset);
    }
 
    __aicore__ inline void SetDataCopyParams(const uint64_t blockCount)
    {
        dataCopyParams.blockCount = blockCount;
        dataCopyParams.blockLen = wiLoadL1;
        dataCopyParams.srcStride = self_->ctx.orgWi - dataCopyParams.blockLen;
        dataCopyParams.dstStride = 0;
        ASC_OP_LOGD("[LoadAL1] dataCopyParams.blockCount %d, dataCopyParams.blockLen %d, "
                    "dataCopyParams.srcStride %d.\n",
            dataCopyParams.blockCount,
            dataCopyParams.blockLen,
            dataCopyParams.srcStride);
    }
 
    __aicore__ inline bool IsMTail()
    {
        return self_->ctx.mAL1Iter == self_->ctx.maxMAL1Iter;
    }
 
    __aicore__ inline bool IsKaL1Tail()
    {
        return self_->ctx.kAL1Iter == self_->ctx.maxKAL1Iter;
    }
 
    __aicore__ inline void PreProcess()
    {
        // todo ho != 1
        hoStartIdx = self_->ctx.hoL1Iter;
        hiLoadL1 = self_->ctx.dilatedKernelH;
        woStartIdx = self_->ctx.mAL1Iter * self_->ctx.conv3dTiling->mAL1;
        uint64_t woLoadL1 = IsMTail() ? self_->ctx.mAL1Tail : self_->ctx.conv3dTiling->mAL1;
        wiLoadL1 = (woLoadL1 - 1) * self_->ctx.strideW + self_->ctx.dilatedKernelW;
 
        cin1LoadL1 = self_->ctx.conv3dTiling->kAL1 / self_->ctx.kernelH / self_->ctx.kernelW / self_->ctx.cin0;
        uint64_t currentCin1LoadL1 = self_->ctx.kAL1Iter * cin1LoadL1;
        ASC_OP_LOGD("[LoadAL1WithHwMode] currentCin1LoadL1 %d.\n", currentCin1LoadL1);
        uint64_t kAL1Tmp = IsKaL1Tail() ? self_->ctx.conv3dTiling->kAL1Tail : self_->ctx.conv3dTiling->kAL1;
        loadData3Dv2Params.channelSize = kAL1Tmp / self_->ctx.kernelHxkernelW;
        cin1LoadL1 = kAL1Tmp / self_->ctx.kernelH / self_->ctx.kernelW / self_->ctx.cin0;
        kdL1Idx = (currentCin1LoadL1 / self_->ctx.cin1) % self_->ctx.kernelD;
        cin1L1Idx = currentCin1LoadL1 % self_->ctx.cin1;
 
        padLeftL1 = 0;
        padRightL1 = 0;
        padTopL1 = 0;
        padBottomL1 = 0;
        aL1IsFullPad = false;
        set2dFlagDHead = false;
        set2dFlagDTail = false;
    }
 
    __aicore__ inline bool ProcessWoutPad()
    {
        uint64_t wiStartIdxWithPad = woStartIdx * self_->ctx.strideW;
        uint64_t wiEndIdxWithPad = wiStartIdxWithPad + wiLoadL1;
        wiIdx = wiStartIdxWithPad - self_->ctx.padLeft;
        if (wiEndIdxWithPad <= self_->ctx.padLeft) {
            // wi开头全pad
            aL1IsFullPad = true;
            return true;
        }
        if (wiStartIdxWithPad < self_->ctx.padLeft) {
            // wi开头一部分是pad
            wiIdx = 0;
            wiLoadL1 = wiLoadL1 + wiStartIdxWithPad - self_->ctx.padLeft;
            padLeftL1 = self_->ctx.padLeft - wiStartIdxWithPad;
            if (wiEndIdxWithPad >= self_->ctx.orgWi + self_->ctx.padLeft) {
                // wi尾部部分是pad
                wiLoadL1 = self_->ctx.orgWi;
                padRightL1 = wiEndIdxWithPad - (self_->ctx.orgWi + self_->ctx.padLeft);
            }
            return true;
        }
        if (wiStartIdxWithPad >= self_->ctx.orgWi + self_->ctx.padLeft) {
            // wi尾部全是pad
            aL1IsFullPad = true;
            return true;
        }
        if (wiEndIdxWithPad > self_->ctx.orgWi + self_->ctx.padLeft) {
            // wi尾部部分是pad
            wiLoadL1 = self_->ctx.orgWi + self_->ctx.padLeft - wiStartIdxWithPad;
            padRightL1 = wiEndIdxWithPad - (self_->ctx.orgWi + self_->ctx.padLeft);
            return true;
        }
 
        // get wiLoadL1, wiIdx, padLeftL1, padRightL1
        return false;
    }
 
    __aicore__ inline void ProcessWoutMiddleData()
    {
        if (ProcessWoutPad()) [[unlikely]] {
            return;
        }
 
        uint64_t wiStartIdxWithPad = woStartIdx * self_->ctx.strideW;
        wiIdx = wiStartIdxWithPad - self_->ctx.padLeft;
        // get hiLoadL1, hiIdx
    }
 
    __aicore__ inline bool ProcessHoutPad()
    {
        uint64_t hiStartIdxWithPad = self_->ctx.hiStartPos + hoStartIdx * self_->ctx.strideH;
        uint64_t hiEndIdxWithPad = hiStartIdxWithPad + hiLoadL1;
        hiIdx = self_->ctx.hiStartPos <= self_->ctx.padUp ? hiStartIdxWithPad - self_->ctx.padUp
                                                            : hiStartIdxWithPad - self_->ctx.hiStartPos;
        if (hiEndIdxWithPad <= self_->ctx.padUp) {
            // hi开头全pad
            aL1IsFullPad = true;
            return true;
        }
        if (hiStartIdxWithPad < self_->ctx.padUp) {
            // hi开头一部分是pad
            hiIdx = 0;
            hiLoadL1 = hiLoadL1 + hiStartIdxWithPad - self_->ctx.padUp;
            padTopL1 = self_->ctx.padUp - hiStartIdxWithPad;
            if (hiEndIdxWithPad >= self_->ctx.orgHi + self_->ctx.padUp) {
                // hi尾部部分是pad
                hiLoadL1 = self_->ctx.orgHi;
                padBottomL1 = hiEndIdxWithPad - (self_->ctx.orgHi + self_->ctx.padUp);
            }
            return true;
        }
        if (hiStartIdxWithPad >= self_->ctx.orgHi + self_->ctx.padUp) {
            // hi尾部全是pad
            aL1IsFullPad = true;
            return true;
        }
        if (hiEndIdxWithPad > self_->ctx.orgHi + self_->ctx.padUp) {
            // hi尾部部分是pad
            hiLoadL1 = self_->ctx.orgHi + self_->ctx.padUp - hiStartIdxWithPad;
            padBottomL1 = hiEndIdxWithPad - (self_->ctx.orgHi + self_->ctx.padUp);
            return true;
        }
        // get hiLoadL1, hiIdx, padTopL1, padBottomL1
        return false;
    }
 
    __aicore__ inline void ProcessHoutMiddleData()
    {
        if (ProcessHoutPad()) [[unlikely]] {
            return;
        }
 
        uint64_t hiStartIdxWithPad = self_->ctx.hiStartPos + hoStartIdx * self_->ctx.strideH;
        hiIdx = self_->ctx.hiStartPos <= self_->ctx.padUp ? hiStartIdxWithPad - self_->ctx.padUp
                                                            : hiStartIdxWithPad - self_->ctx.hiStartPos;
        // get hiLoadL1, hiIdx
    }
    
    __aicore__ inline bool ProcessDoutPad()
    {
        uint64_t diStartWithPad =
            self_->ctx.diStartPos + self_->ctx.dOutIter * self_->ctx.strideD + kdL1Idx * self_->ctx.dilationD;
        uint64_t diEndWithPad = cin1LoadL1 <= self_->ctx.cin1
                                    ? diStartWithPad + 1
                                    : diStartWithPad + (cin1LoadL1 / self_->ctx.cin1 - 1) * self_->ctx.dilationD + 1;
        diIdx = self_->ctx.diStartPos <= self_->ctx.padHead ? diStartWithPad - self_->ctx.padHead
                                                            : diStartWithPad - self_->ctx.diStartPos;
        if (diEndWithPad <= self_->ctx.padHead) {
            // di开头全pad
            aL1IsFullPad = true;
            return true;
        }
        if (diStartWithPad < self_->ctx.padHead) {
            // di开头一部分是pad，只有cin1LoadL1 > self_->ctx.cin1情况会进
            set2dFlagDHead = true;
            uint64_t kdTmp = CeilDIV((self_->ctx.padHead - diStartWithPad), self_->ctx.dilationD);
            cin1LoadL1PadHead = kdTmp * self_->ctx.cin1;
            diIdx = self_->ctx.dilationD == 1 ? 0 : kdTmp * self_->ctx.dilationD - self_->ctx.padHead + diStartWithPad;
            cin1LoadL1 -= cin1LoadL1PadHead;
 
            if (diEndWithPad > self_->ctx.orgDi + self_->ctx.padHead) {
                // dout尾部部分是pad
                set2dFlagDTail = true;
                // 计算真实应该载入多少kdTmp
                kdTmp = CeilDIV((self_->ctx.orgDi - diIdx), self_->ctx.dilationD);
                cin1LoadL1PadTail = cin1LoadL1 - kdTmp * self_->ctx.cin1;
                cin1LoadL1 = kdTmp * self_->ctx.cin1;
            }
            return true;
        }
        if (diStartWithPad >= self_->ctx.orgDi + self_->ctx.padHead) {
            // dout尾部全是pad
            aL1IsFullPad = true;
            return true;
        }
        if (diEndWithPad > self_->ctx.orgDi + self_->ctx.padHead) {
            // dout尾部部分是pad，只有cin1LoadL1 > self_->ctx.cin1情况会进
            set2dFlagDTail = true;
            // 计算真实应该载入多少kdTmp
            uint64_t kdTmp = CeilDIV((self_->ctx.orgDi + self_->ctx.padHead - diStartWithPad), self_->ctx.dilationD);
            cin1LoadL1PadTail = cin1LoadL1 - kdTmp * self_->ctx.cin1;
            cin1LoadL1 = kdTmp * self_->ctx.cin1;
            return true;
        }
        // get cin1L1Idx, diIdx, cin1LoadL1
        return false;
    }
 
    __aicore__ inline void ProcessDoutMiddleData()
    {
        if (ProcessDoutPad()) [[unlikely]] {
            return;
        }
 
        uint64_t diStartWithPad =
            self_->ctx.diStartPos + self_->ctx.dOutIter * self_->ctx.strideD + kdL1Idx * self_->ctx.dilationD;
        diIdx = self_->ctx.diStartPos <= self_->ctx.padHead ? diStartWithPad - self_->ctx.padHead
                                                            : diStartWithPad - self_->ctx.diStartPos;
        // get cin1L1Idx, diIdx
    }
 
private:
    Intf *self_ = nullptr;
    uint64_t aL1GmOffset = 0;
    uint64_t wiLoadL1 = 0;
    uint64_t hiLoadL1 = 0;
    uint64_t cin1LoadL1 = 0;
    uint64_t cin1LoadL1PadHead = 0;
    uint64_t cin1LoadL1PadTail = 0;
    uint64_t diIdx = 0;
    uint64_t wiIdx = 0;
    uint64_t hiIdx = 0;
    uint64_t cin1L1Idx = 0;
    uint64_t woStartIdx = 0;
    uint64_t hoStartIdx = 0;
    uint64_t hoEndIdx = 0;
    uint64_t curCoreHiStartIdx = 0;
    uint64_t kdL1Idx = 0;
 
    uint64_t padTopL1 = 0;
    uint64_t padBottomL1 = 0;
    uint64_t padLeftL1 = 0;
    uint64_t padRightL1 = 0;

    uint8_t aL1IsFullPad = false;
    uint8_t set2dFlagDHead = false;
    uint8_t set2dFlagDTail = false;
    DataCopyParams dataCopyParams;
    LoadData3DParamsV2<typename Intf::FmapT> loadData3Dv2Params;
};

template <class Intf>
class LoadBL1WithHwModeTools {
public:
    __aicore__ inline LoadBL1WithHwModeTools() {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
    }

    __aicore__ inline void LoadBL1()
    {
        PreProcess();
        if (self_->ctx.conv3dTiling->nBL1 >= self_->ctx.orgCo) {
            dataCopyParams.blockCount = 1;
            dataCopyParams.srcStride = 0;
            dataCopyParams.blockLen = CeilDIV(currentKBL1 * self_->ctx.conv3dTiling->nBL1, self_->ctx.cin0);
        } else {
            dataCopyParams.blockCount = currentKBL1 / self_->ctx.cin0;
            dataCopyParams.blockLen = currentNBL1;
            dataCopyParams.srcStride = self_->ctx.orgCoAlignN0 - currentNBL1;
        }
        DataCopy<typename Intf::WeightT>(self_->ctx.bl1, self_->ctx.bgm[bL1GmOffset], dataCopyParams);
        ASC_OP_LOGD("[LoadBL1WithHwModeTools] bL1GmOffset %d self_->ctx.orgCo %d.\n", bL1GmOffset, self_->ctx.orgCo);
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
        bL1GmOffset = self_->ctx.kBL1Iter * self_->ctx.conv3dTiling->kBL1 * self_->ctx.orgCoAlignN0 +
                      self_->ctx.nBL1Iter * self_->ctx.conv3dTiling->nBL1 * self_->ctx.cin0;
        currentNBL1 = IsNTail() ? self_->ctx.nBL1TailAlign : self_->ctx.conv3dTiling->nBL1;
        currentKBL1 = IsKBL1Tail() ? self_->ctx.conv3dTiling->kBL1Tail : self_->ctx.conv3dTiling->kBL1;
    }

private:
    Intf *self_ = nullptr;
    uint64_t bL1GmOffset = 0;
    uint64_t currentNBL1 = 0;
    uint64_t currentKBL1 = 0;
    DataCopyParams dataCopyParams;
};

};  // namespace Conv3dFunc

#endif  // __CONV3D_MTE2_HW_MODE_SUB_API_H__
