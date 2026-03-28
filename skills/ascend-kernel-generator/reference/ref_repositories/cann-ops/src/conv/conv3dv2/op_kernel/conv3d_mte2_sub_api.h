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
 * \file conv3d_mte2_sub_api.h
 * \brief
 */

#ifndef CONV3D_MTE2_SUB_API_H
#define CONV3D_MTE2_SUB_API_H

#include "conv3d_util.h"
#include "conv3d_config.h"
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
class LoadAL1Tools {
public:
    __aicore__ inline LoadAL1Tools()
    {}

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
        ProcessMMiddleData();
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
        padList[PAD_IDX_L] = self_->ctx.padLeft;
        padList[PAD_IDX_R] = self_->ctx.padRight;
        padList[PAD_IDX_T] = padTopL1;
        padList[PAD_IDX_B] = padBottomL1;
        SetFmatrix(hiLoadL1, self_->ctx.orgWi, padList, FmatrixMode::FMATRIX_LEFT);
    }

     __aicore__ inline void CopyDataGMToL1()
    {
        CalcAL1GmOffset();
        uint64_t dataCopyLoop = CeilDIV(cin1LoadL1, self_->ctx.cin1);
        uint64_t dataCopySubLoop = 0;
        uint64_t blockCount = self_->ctx.cin1;
        bool srcStrideBeyondMaxU16 = false;
        if (self_->ctx.conv3dTiling->orgHixWi - hiLoadL1 * self_->ctx.orgWi > MAX_UINT16) [[unlikely]] {
            dataCopySubLoop = dataCopyLoop > 0 ? cin1LoadL1 / dataCopyLoop : 0;
            blockCount = 1;
            srcStrideBeyondMaxU16 = true;
        }
        uint64_t aL1Offset = 0;
        if (cin1LoadL1 > self_->ctx.cin1 || unlikely(srcStrideBeyondMaxU16)) {
            Set2DForDinPadHead(aL1Offset);
            SetDataCopyParams(blockCount);
            DataCopyWithLoop(aL1Offset, dataCopyLoop, dataCopySubLoop, srcStrideBeyondMaxU16);
            Set2DForDinPadTail(aL1Offset);
        } else {
            Set2DForDinPadHead(aL1Offset);
            SetDataCopyParams(cin1LoadL1);
            DataCopy<typename Intf::FmapT>(self_->ctx.al1[aL1Offset], self_->ctx.agm[aL1GmOffset], dataCopyParams);
            aL1Offset += cin1LoadL1 * hiLoadL1 * self_->ctx.conv3dTiling->oriWixk0;
            Set2DForDinPadTail(aL1Offset);
        }
    }

    __aicore__ inline void DataCopyWithLoop(uint64_t &aL1Offset, uint64_t dataCopyLoop,
                                            uint64_t dataCopySubLoop, bool srcStrideBeyondMaxU16)
    {
        for (uint64_t i = 0; i < dataCopyLoop; i++) {
            if (srcStrideBeyondMaxU16) {
                uint64_t aL1GmSubOffset = aL1GmOffset;
                uint64_t aL1SubOffset = aL1Offset;
                for (uint64_t j = 0; j < dataCopySubLoop; j++) {
                    DataCopy<typename Intf::FmapT>(
                        self_->ctx.al1[aL1SubOffset], self_->ctx.agm[aL1GmSubOffset], dataCopyParams);
                    aL1GmSubOffset += self_->ctx.conv3dTiling->oriHixOriWixk0;
                    aL1SubOffset += hiLoadL1 * self_->ctx.conv3dTiling->oriWixk0;
                }
            } else {
                DataCopy<typename Intf::FmapT>(self_->ctx.al1[aL1Offset],
                    self_->ctx.agm[aL1GmOffset], dataCopyParams);
            }
            if constexpr (Intf::groupConvType) {
                aL1GmOffset += self_->ctx.dilationD * CeilDIV(self_->ctx.conv3dTiling->orgCi, self_->ctx.cin0) *
                                self_->ctx.conv3dTiling->oriHixOriWixk0;
            } else {
                aL1GmOffset +=
                    self_->ctx.dilationD * self_->ctx.cin1 * self_->ctx.conv3dTiling->oriHixOriWixk0;
            }
            aL1Offset += self_->ctx.cin1 * hiLoadL1 * self_->ctx.conv3dTiling->oriWixk0;
        }
    }

    __aicore__ inline void Set2DForDinPadHead(uint64_t &aL1Offset)
    {
        if (set2dFlagDHead) {
            // 前di pad
            if constexpr (Intf::quantType == static_cast<int8_t>(QuantType::PER_CHANNEL_NO_OFFSET)) {
                LocalTensor<uint16_t> clearLocal = self_->ctx.al1.template ReinterpretCast<uint16_t>();
                InitConstValueParams<uint16_t> initConstValueParams;
                SetInitConstValueParams<uint16_t>(initConstValueParams,
                    cin1LoadL1PadHead / self_->ctx.cin1,
                    self_->ctx.cin1 * hiLoadL1 * self_->ctx.orgWi);
                InitConstValue<uint16_t>(clearLocal, initConstValueParams);
            } else {
                InitConstValueParams<typename Intf::FmapT> initConstValueParams;
                SetInitConstValueParams<typename Intf::FmapT>(initConstValueParams,
                    cin1LoadL1PadHead / self_->ctx.cin1,
                    self_->ctx.cin1 * hiLoadL1 * self_->ctx.orgWi);
                InitConstValue<typename Intf::FmapT>(self_->ctx.al1, initConstValueParams);
            }
            aL1Offset += cin1LoadL1PadHead * hiLoadL1 * self_->ctx.conv3dTiling->oriWixk0;
            set2dFlagDHead = false;
        }
    }

    __aicore__ inline void CalcAL1GmOffset()
    {
        if constexpr (Intf::groupConvType) {
            aL1GmOffset =
                diIdx * AlignB(self_->ctx.conv3dTiling->orgCi, self_->ctx.cin0) * self_->ctx.conv3dTiling->orgHixWi +
                self_->ctx.groupOptIter * self_->ctx.conv3dTiling->cin1xOriHixOriWixk0 +
                cin1L1Idx * self_->ctx.conv3dTiling->oriHixOriWixk0 +
                hiIdx * self_->ctx.conv3dTiling->oriWixk0;
        } else {
            aL1GmOffset = diIdx * self_->ctx.conv3dTiling->cin1xOriHixOriWixk0 +
                          cin1L1Idx * self_->ctx.conv3dTiling->oriHixOriWixk0 +
                          hiIdx * self_->ctx.conv3dTiling->oriWixk0;
        }
        ASC_OP_LOGD("[LoadAL1] aL1GmOffset %d.\n", aL1GmOffset);
    }

    __aicore__ inline void Set2DForDinPadTail(uint64_t &aL1Offset)
    {
        if (set2dFlagDTail) {
            // 后di pad
            if constexpr (Intf::quantType == static_cast<int8_t>(QuantType::PER_CHANNEL_NO_OFFSET)) {
                LocalTensor<uint16_t> clearLocal = self_->ctx.al1.template ReinterpretCast<uint16_t>();
                InitConstValueParams<uint16_t> initConstValueParams;
                SetInitConstValueParams<uint16_t>(initConstValueParams,
                    cin1LoadL1PadTail / self_->ctx.cin1,
                    self_->ctx.cin1 * hiLoadL1 * self_->ctx.orgWi);
                InitConstValue<uint16_t>(clearLocal[aL1Offset / INT16_DIV_INT8], initConstValueParams);
            } else {
                InitConstValueParams<typename Intf::FmapT> initConstValueParams;
                SetInitConstValueParams<typename Intf::FmapT>(initConstValueParams,
                    cin1LoadL1PadTail / self_->ctx.cin1,
                    self_->ctx.cin1 * hiLoadL1 * self_->ctx.orgWi);
                InitConstValue<typename Intf::FmapT>(self_->ctx.al1[aL1Offset], initConstValueParams);
            }
            set2dFlagDTail = false;
        }
    }

    template <typename T>
    __aicore__ inline void SetInitConstValueParams(InitConstValueParams<T> &initConstValueParams,
        const uint64_t repeatTimes, const uint64_t blockNum)
    {
        initConstValueParams.repeatTimes = repeatTimes;
        initConstValueParams.blockNum = blockNum;
        initConstValueParams.dstGap = 0;
        initConstValueParams.initValue = 0;
        ASC_OP_LOGD(
            "[LoadAL1] initConstValueParams.repeatTimes %d, initConstValueParams.blockNum %d \n",
            initConstValueParams.repeatTimes,
            initConstValueParams.blockNum);
    }

    __aicore__ inline void SetDataCopyParams(const uint64_t blockCount)
    {
        dataCopyParams.blockCount = blockCount;
        dataCopyParams.blockLen = hiLoadL1 * self_->ctx.orgWi;
        dataCopyParams.srcStride = self_->ctx.conv3dTiling->orgHixWi - dataCopyParams.blockLen;
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
        uint64_t currentML1 = IsMTail() ? self_->ctx.mAL1Tail : self_->ctx.conv3dTiling->mAL1;
        uint64_t currentM = self_->ctx.mStartPos + self_->ctx.mAL1Iter * self_->ctx.conv3dTiling->mAL1;
        hoStartIdx = currentM / self_->ctx.orgWo;
        hoEndIdx = CeilDIV(currentM + currentML1, self_->ctx.orgWo);
        hiLoadL1 = ((hoEndIdx - hoStartIdx) - 1) * self_->ctx.strideH + self_->ctx.dilatedKernelH;
        uint64_t tmpCurCoreHiStartIdx = (self_->ctx.mStartPos / self_->ctx.orgWo) * self_->ctx.strideH;
        curCoreHiStartIdx = tmpCurCoreHiStartIdx <= self_->ctx.padUp ? 0 : tmpCurCoreHiStartIdx - self_->ctx.padUp;

        uint64_t currentCin1LoadL1 = self_->ctx.kAL1Iter * self_->ctx.conv3dTiling->cin1InAL1;
        uint64_t kAL1Tmp = IsKaL1Tail() ? self_->ctx.conv3dTiling->kAL1Tail : self_->ctx.conv3dTiling->kAL1;
        if constexpr (Intf::groupConvType) {
            cin1LoadL1 = self_->ctx.groupKAL1 / self_->ctx.kernelH / self_->ctx.kernelW / self_->ctx.cin0;
            kAL1Tmp = IsKaL1Tail() ? self_->ctx.groupKAL1Tail : self_->ctx.groupKAL1;
            currentCin1LoadL1 = self_->ctx.kAL1Iter * cin1LoadL1;
        }
        ASC_OP_LOGD("[LoadAL1] currentCin1LoadL1 %d.\n", currentCin1LoadL1);
        loadData3Dv2Params.channelSize = kAL1Tmp / self_->ctx.kernelHxkernelW;
        cin1LoadL1 = kAL1Tmp / self_->ctx.kernelH / self_->ctx.kernelW / self_->ctx.cin0;
        kdL1Idx = (currentCin1LoadL1 / self_->ctx.cin1) % self_->ctx.kernelD;
        cin1L1Idx = currentCin1LoadL1 % self_->ctx.cin1;

        padTopL1 = 0;
        padBottomL1 = 0;
        aL1IsFullPad = false;
        set2dFlagDHead = false;
        set2dFlagDTail = false;
    }

    __aicore__ inline bool ProcessMPad()
    {
        uint64_t hiStartIdxWithPad = hoStartIdx * self_->ctx.strideH;
        uint64_t hiEndIdxWithPad = hiStartIdxWithPad + hiLoadL1;
        hiIdx = hiStartIdxWithPad - self_->ctx.padUp - curCoreHiStartIdx;
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
                // m尾部部分是pad
                hiLoadL1 = self_->ctx.orgHi - hiIdx;
                padBottomL1 = hiEndIdxWithPad - (self_->ctx.orgHi + self_->ctx.padUp);
            }
            return true;
        }
        if (hiStartIdxWithPad >= self_->ctx.orgHi + self_->ctx.padUp) {
            // m尾部全是pad
            aL1IsFullPad = true;
            return true;
        }
        if (hiEndIdxWithPad > self_->ctx.orgHi + self_->ctx.padUp) {
            // m尾部部分是pad
            hiLoadL1 = self_->ctx.orgHi + self_->ctx.padUp - hiStartIdxWithPad;
            padBottomL1 = hiEndIdxWithPad - (self_->ctx.orgHi + self_->ctx.padUp);
            return true;
        }
        // get hiLoadL1, hiIdx, padTopL1, padBottomL1
        return false;
    }

    __aicore__ inline void ProcessMMiddleData()
    {
        if (ProcessMPad()) {
            return;
        }

        uint64_t hiStartIdxWithPad = hoStartIdx * self_->ctx.strideH;
        hiIdx = hiStartIdxWithPad - self_->ctx.padUp - curCoreHiStartIdx;
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
        if (ProcessDoutPad()) {
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
    uint64_t hiLoadL1 = 0;
    uint64_t cin1LoadL1 = 0;
    uint64_t cin1LoadL1PadHead = 0;
    uint64_t cin1LoadL1PadTail = 0;
    uint64_t diIdx = 0;
    uint64_t hiIdx = 0;
    uint64_t cin1L1Idx = 0;
    uint64_t hoStartIdx = 0;
    uint64_t hoEndIdx = 0;
    uint64_t curCoreHiStartIdx = 0;
    uint64_t kdL1Idx = 0;

    uint64_t padTopL1 = 0;
    uint64_t padBottomL1 = 0;

    uint8_t aL1IsFullPad = false;
    uint8_t set2dFlagDHead = false;
    uint8_t set2dFlagDTail = false;
    DataCopyParams dataCopyParams;
    LoadData3DParamsV2<typename Intf::FmapT> loadData3Dv2Params;
};

template <class Intf>
class LoadBL1Tools {
public:
    __aicore__ inline LoadBL1Tools()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
    }

    __aicore__ inline void LoadBL1()
    {
        PreProcess();
        if (self_->ctx.conv3dTiling->nBL1 >= self_->ctx.orgCo) {
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
            dataCopyParams.blockCount = currentKBL1 / self_->ctx.cin0;
            dataCopyParams.blockLen = currentNBL1;
            dataCopyParams.srcStride = self_->ctx.orgCoAlignN0 - currentNBL1;
            DataCopy<typename Intf::WeightT>(self_->ctx.bl1, self_->ctx.bgm[bL1GmOffset], dataCopyParams);
        }
        ASC_OP_LOGD("[LoadBL1] bL1GmOffset %d self_->ctx.orgCo %d.\n", bL1GmOffset, self_->ctx.orgCo);
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
        
        loadData2dParams.srcStride = 1;
    }

    __aicore__ inline void SetLoadData2DParams(const uint64_t &repeatTimes)
    {
        loadData2dParams.repeatTimes = repeatTimes;
        ASC_OP_LOGD("[LoadBL1] loadData2dParams.repeatTimes %d.\n", loadData2dParams.repeatTimes);
    }

private:
    Intf *self_ = nullptr;
    uint64_t bL1GmOffset = 0;
    uint64_t currentNBL1 = 0;
    uint64_t currentKBL1 = 0;
    DataCopyParams dataCopyParams;
    LoadData2DParams loadData2dParams;
};

};  // namespace Conv3dFunc

#endif  // __CONV3D_MTE2_SUB_API_H__