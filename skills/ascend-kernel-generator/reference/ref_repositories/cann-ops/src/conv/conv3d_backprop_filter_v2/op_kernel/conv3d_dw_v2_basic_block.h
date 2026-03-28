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
 * \file conv3d_dw_v2_basic_block.h
 * \brief
 */
#ifndef CONV3D_BACKPROP_FILTER_BASIC_BLOCK_H
#define CONV3D_BACKPROP_FILTER_BASIC_BLOCK_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "conv3d_bp_filter.h"
#include "kernel_type.h"
#include "kernel_type.h"
#include "conv3d_backprop_filter_v2.h"

constexpr uint32_t ROW_FIRST = 1;
constexpr uint32_t COL_FIRST = 2;

namespace AscendC {
template <typename xType, int xFormat, typename dedyType, int dedyFormat, typename yType, int yFormat>
class Conv3dDwBasicBlock : public Conv3dDw<xType, xFormat, dedyType, dedyFormat, yType, yFormat> {
public:
    __aicore__ inline Conv3dDwBasicBlock() {};

    __aicore__ void InitCommonTilingData(const Conv3DBackpropFilterV2TilingData* tilingData) {
        Conv3dDw<xType, xFormat, dedyType, dedyFormat, yType, yFormat>::InitTilingData(tilingData);
        this->usedCoreNum_ = tilingData->basicBlockTiling.usedCoreNum;
        this->coreBindOrder_ = tilingData->basicBlockTiling.coreBindOrder;
#if __CCE_AICORE__ == 220
        if constexpr (xFormat == FORMAT_NCDHW) { // Transdata Merge
            // Initialize the size of each GM space.
            uint64_t singleCoreHo = tilingData->dwTiling.singleCoreHo;
            uint64_t singleCoreHi = (singleCoreHo - 1) * tilingData->dwTiling.strideH
                + (tilingData->dwTiling.hk - 1) * tilingData->dwTiling.dilationH + 1;
            singleCoreHi = (singleCoreHi < tilingData->dwTiling.hi) ? singleCoreHi : tilingData->dwTiling.hi;
            uint64_t singleCoreCin = tilingData->dwTiling.singleCoreCin;
            this->gmPongOffset = singleCoreCin * singleCoreHi * static_cast<uint64_t>(tilingData->dwTiling.wi);
        }
#endif
    }

    __aicore__ inline void InitTransdataBuffer() {
        this->dw_.ctx.pipe_.InitBuffer(this->dw_.ctx.transdataPing_, 1, SINGLE_UB_SIZE);
        this->dw_.ctx.pipe_.InitBuffer(this->dw_.ctx.transdataPong_, 1, SINGLE_UB_SIZE);
        this->dw_.ctx.pipe_.InitBuffer(this->dw_.ctx.transdataResultPing_, 1, SINGLE_UB_SIZE);
        this->dw_.ctx.pipe_.InitBuffer(this->dw_.ctx.transdataResultPong_, 1, SINGLE_UB_SIZE);
    }

    __aicore__ void TransDataTo6HD(const uint64_t batchIdx, const uint64_t doutIdx) {
        uint64_t dinIdx = GetDinIdx(doutIdx);
        uint64_t maxHiWiCount = SINGLE_UB_SIZE / (this->c0_ * sizeof(xType));
        int64_t singleShapeCin = static_cast<int64_t>(this->singleShapeN_) / (this->hk_ * this->wk_);
        uint64_t wsOffset = static_cast<uint64_t>(block_idx) * this->DOUBLE_BUFFER * this->gmPongOffset + (gmPingPongEventId_ ? this->gmPingOffset : this->gmPongOffset);
        uint64_t hiWi = static_cast<uint64_t>(this->hi_) * this->wi_;
        uint64_t batchOffset = batchIdx * this->cin_ * this->di_ * hiWi;
        uint64_t index = 0;
        uint64_t dkIdx = (static_cast<uint64_t>(this->nCoreIndx_) * this->singleCoreCin_) / this->cinG_;
        uint64_t cinIdx = (static_cast<uint64_t>(this->nCoreIndx_) * this->singleCoreCin_) - (dkIdx * this->cinG_);
        uint64_t hoOffset = static_cast<uint64_t>(this->kCoreIndx_) * this->singleShapeK_ / this->wo_;
        uint64_t hiIdx = hoOffset * this->strideH_ > this->padUp_ ? hoOffset * this->strideH_ - this->padUp_ : 0;
        uint64_t strideKernelDilationH = (static_cast<uint64_t>(this->hk_) - 1) * this->dilationH_ + 1 - this->strideH_;
        uint64_t singleCoreHi = this->singleCoreHo_ * this->strideH_ + strideKernelDilationH;
        singleCoreHi = (singleCoreHi < this->hi_) ? singleCoreHi : this->hi_;
        uint64_t singleShapeHi = hiIdx < 0 ? (singleCoreHi + hiIdx) : singleCoreHi;
        singleShapeHi = hiIdx + singleCoreHi > this->hi_ ? this->hi_ - hiIdx : singleCoreHi;
        uint64_t cin1Count = this->singleCoreCin_ / this->c0_;
        uint64_t transDataIdx = 0;
        for (uint64_t c1Idx = 0; c1Idx < cin1Count; c1Idx++) {
            if (dinIdx + dkIdx >= this->di_) {
                break;
            }
            uint64_t transDataCinCount = (static_cast<uint64_t>(this->cin_) - cinIdx) < this->c0_ ? (static_cast<uint64_t>(this->cin_) - cinIdx) : this->c0_;
            uint64_t hwCount = singleShapeHi * this->wi_;
            uint64_t hiTransDataCount = Ceil(hwCount, maxHiWiCount);
            uint64_t hiWiOffset = hiIdx * this->wi_;
            for (uint64_t hwIdx = 0; hwIdx < hiTransDataCount; hwIdx++) {
                uint64_t hwTransDataCount = hwCount < maxHiWiCount ? hwCount : maxHiWiCount;
                if (GetSubBlockIdx() == transDataIdx % blockFactor) {
                    uint64_t gmOffset = batchOffset + cinIdx * this->di_ * hiWi + (dinIdx + dkIdx) * hiWi + hiWiOffset;
                    CopyGmDataToVecin(gmOffset, transDataCinCount, hwTransDataCount, maxHiWiCount);
                    TransVecinDataTo5HDToVecout(maxHiWiCount);
                    CopyVecoutDataToWorkSpace(wsOffset, hwTransDataCount);
                    ubPingPongEventId_ &= 1;
                    ubPingPongEventId_ ^= 1;
                }
                wsOffset += this->c0_ * hwTransDataCount;
                hiWiOffset += hwTransDataCount;
                hwCount -= hwTransDataCount;
                transDataIdx += 1;
            }
            wsOffset += this->c0_ * (singleCoreHi - singleShapeHi) * this->wi_;
            cinIdx = (cinIdx + this->c0_) % this->cinG_;
            if (cinIdx == 0) {
                dkIdx += 1;
            }
        }
    }

protected:
    static constexpr uint64_t SINGLE_UB_SIZE = 49152;
    static constexpr uint64_t NCHW_CONV_ADDR_LIST_SIZE = 16;
    static constexpr uint64_t ONE_BLOCK_SIZE = 32;
    uint64_t usedCoreNum_;
    uint64_t coreBindOrder_;
    uint64_t gmPingOffset = 0;
    uint64_t gmPongOffset = 0;
    uint64_t blockFactor = 2;
    bool gmPingPongEventId_ = 1;
    bool ubPingPongEventId_ = 1;

    __aicore__ uint64_t GetDinIdx(uint64_t doutIdx) {
        uint64_t dinIdx = 0;
        if (doutIdx * this->strideD_ > this->padFront_) {
            dinIdx = doutIdx * this->strideD_ -  this->padFront_;
        }
        return dinIdx;
    }

    __aicore__ void CopyGmDataToVecin(const uint64_t gmOffset, uint64_t copyCinCount, uint64_t copyHiWiCount, uint64_t maxCopyHiWiCount) {
        LocalTensor<xType> vecin = ubPingPongEventId_ ? this->dw_.ctx.transdataPing_.template AllocTensor<xType>() : this->dw_.ctx.transdataPong_.template AllocTensor<xType>();
        xType initValue(0);
        Duplicate<xType>(vecin, initValue, this->c0_ * maxCopyHiWiCount);
        TEventID eventId = GetTPipePtr()->FetchEventID<HardEvent::V_MTE2>();
        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);
        DataCopyExtParams dataCopyParams;
        dataCopyParams.dstStride = 0;
        dataCopyParams.srcStride = 0;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = copyHiWiCount * sizeof(xType);
        DataCopyPadExtParams<xType> dataCopyPadParams;
        uint64_t srcOffset = 0;
        uint64_t dstOffset = 0;
        for (uint64_t copyCinIndex = 0; copyCinIndex < copyCinCount; copyCinIndex++) {
            DataCopyPad(vecin[dstOffset], this->xGm_[gmOffset + srcOffset], dataCopyParams, dataCopyPadParams);
            srcOffset += static_cast<uint64_t>(this->di_) * this->hi_ * this->wi_;
            dstOffset += maxCopyHiWiCount;
        }
        if (ubPingPongEventId_) {
            this->dw_.ctx.transdataPing_.template EnQue<xType>(vecin);
        } else {
            this->dw_.ctx.transdataPong_.template EnQue<xType>(vecin);
        }
    }

    __aicore__ void TransVecinDataTo5HDToVecout(const uint64_t maxCopyHiWiCount) {
        LocalTensor<xType> vecin;
        LocalTensor<xType> vecout;
        if (ubPingPongEventId_) {
            vecin = this->dw_.ctx.transdataPing_.template DeQue<xType>();
            vecout = this->dw_.ctx.transdataResultPing_.template AllocTensor<xType>();
        } else {
            vecin = this->dw_.ctx.transdataPong_.template DeQue<xType>();
            vecout = this->dw_.ctx.transdataResultPong_.template AllocTensor<xType>();
        }
        uint64_t srcLocalTensorAddrList[NCHW_CONV_ADDR_LIST_SIZE];
        uint64_t dstLocalTensorAddrList[NCHW_CONV_ADDR_LIST_SIZE];
        uint64_t srcOffset = 0;
        uint64_t dstOffset = 0;
        for (uint64_t index = 0; index < NCHW_CONV_ADDR_LIST_SIZE; index++) {
            srcLocalTensorAddrList[index] = reinterpret_cast<uint64_t>(vecin[srcOffset].GetPhyAddr());
            dstLocalTensorAddrList[index] = reinterpret_cast<uint64_t>(vecout[dstOffset].GetPhyAddr());
            srcOffset += maxCopyHiWiCount;
            dstOffset += this->c0_;
        }
        uint64_t srcRepStride = 1;
        TransDataTo5HDParams transDataParams;
        transDataParams.dstHighHalf = false;
        transDataParams.srcHighHalf = false;
        transDataParams.repeatTimes = maxCopyHiWiCount / this->c0_;
        transDataParams.dstRepStride = this->c0_;
        transDataParams.srcRepStride = srcRepStride;
        if (transDataParams.repeatTimes == 1) {
            transDataParams.dstRepStride = 0;
            transDataParams.srcRepStride = 0;
        }
        TransDataTo5HD<half>(dstLocalTensorAddrList, srcLocalTensorAddrList, transDataParams);
        if (ubPingPongEventId_) {
            this->dw_.ctx.transdataResultPing_.template EnQue<xType>(vecout);
            this->dw_.ctx.transdataPing_.template FreeTensor(vecin);
            this->dw_.ctx.transdataPing_.FreeAllEvent();
        } else {
            this->dw_.ctx.transdataResultPong_.template EnQue<xType>(vecout);
            this->dw_.ctx.transdataPong_.template FreeTensor(vecin);
            this->dw_.ctx.transdataPong_.FreeAllEvent();
        }
    }

    __aicore__ void CopyVecoutDataToWorkSpace(uint64_t wsOffset, uint64_t copyHiWiCount) {
        LocalTensor<xType> vecout;
        if (ubPingPongEventId_) {
            vecout = this->dw_.ctx.transdataResultPing_.template DeQue<xType>();
        } else {
            vecout = this->dw_.ctx.transdataResultPong_.template DeQue<xType>();
        }
        DataCopyParams dataCopyParams;
        dataCopyParams.dstStride = 0;
        dataCopyParams.srcStride = 0;
        dataCopyParams.blockCount = copyHiWiCount;
        dataCopyParams.blockLen = this->c0_ * sizeof(xType);
        DataCopyPad(this->workspaceXGm_[wsOffset], vecout, dataCopyParams);
        if (ubPingPongEventId_) {
            this->dw_.ctx.transdataResultPing_.template FreeTensor(vecout);
            this->dw_.ctx.transdataResultPing_.FreeAllEvent();
        } else {
            this->dw_.ctx.transdataResultPong_.template FreeTensor(vecout);
            this->dw_.ctx.transdataResultPong_.FreeAllEvent();
        }
    }
};

template <typename xType, int xFormat, typename dedyType, int dedyFormat, typename yType, int yFormat>
class Conv3dDwBasicBlockSplitMK : public Conv3dDwBasicBlock<xType, xFormat, dedyType, dedyFormat, yType, yFormat> {
public:
    __aicore__ inline Conv3dDwBasicBlockSplitMK() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dedy,
                                GM_ADDR y, GM_ADDR workSpace,
                                const Conv3DBackpropFilterV2TilingData* tilingData) {
        this->InitCommonTilingData(tilingData);
        InitSplitTilingData(tilingData);
        // init global buffer
        this->xGm_.SetGlobalBuffer((__gm__ xType*)x);
        this->dedyGm_.SetGlobalBuffer((__gm__ dedyType*)dedy);
        this->yGm_.SetGlobalBuffer((__gm__ yType*)y);
        this->dw_.Init(&(tilingData->dwTiling));
        this->workspaceXGm_.SetGlobalBuffer((__gm__ xType*)workSpace);
    }

    __aicore__ inline void Process() {
        if (block_idx >= this->usedCoreNum_) { // vector GetBlockIdx() 与 Cube不同，此处注意使用全局变量准确。
            return;
        }
        if ASCEND_IS_AIV {
            if constexpr (xFormat != FORMAT_NCDHW) {
                return ;
            }
            this->InitTransdataBuffer();
        }
        CalBasicBlock();
        this->dw_.End();
    }

protected:
    static constexpr uint8_t SYNC_MODE2 = 2;

    __aicore__ inline void InitSplitTilingData(const Conv3DBackpropFilterV2TilingData* tilingData) {
        this->singleShapeM_ = tilingData->basicBlockTiling.singleCoreM;
        this->singleShapeK_ = tilingData->basicBlockTiling.singleCoreK;
        this->singleShapeN_ = this->n_;
    }

    __aicore__ inline void CalBasicBlock() {
        uint32_t mCnt = DivCeil(this->m_, this->singleShapeM_);
        uint32_t mCoreTail = this->m_ - (mCnt - 1) * this->singleShapeM_;

        uint64_t kCnt = DivCeil(this->k_, this->singleShapeK_);
        uint64_t kCoreTail = this->k_ - (kCnt - 1) * this->singleShapeK_;
        kCoreTail = kCoreTail > this->wo_ ? kCoreTail : this->wo_;

        // 记录基本块的位置
        uint64_t batchDout = static_cast<uint64_t>(this->batch_) * this->dout_;
        uint64_t mkCnt = mCnt * kCnt;
        uint64_t totalCnt = batchDout * mkCnt;
        uint64_t calRound = totalCnt / this->usedCoreNum_;
        uint64_t tailCnt = totalCnt - calRound * this->usedCoreNum_;
        uint64_t basicBlockIdx = 0;

        // 拖尾的部分依次分配到前面的核计算，这些核会多算一轮
        if (block_idx < tailCnt) {
            basicBlockIdx = block_idx * calRound + block_idx;
            ++calRound;
        } else {
            basicBlockIdx = block_idx * calRound + tailCnt;
        }

        //1:M*K 行优先绑核，2:M*K 列优先绑核
        uint64_t batchDoutIndex = 0;
        uint64_t batchDoutMcnt = batchDout * mCnt;
        this->nCoreIndx_ = 0; // 默认不分核
        uint64_t syncTimes = 0;
        for (uint64_t j = 0; j < calRound; ++j) {
            if (this->coreBindOrder_ == ROW_FIRST) {
                // 行优先, NDC1HWC0的行方向是C0即M方向
                this->kCoreIndx_ = basicBlockIdx / batchDoutMcnt;
                batchDoutIndex = (basicBlockIdx - this->kCoreIndx_ * batchDoutMcnt) / mCnt;
                this->mCoreIndx_ = basicBlockIdx - this->kCoreIndx_ * batchDoutMcnt - batchDoutIndex * mCnt;
            } else {
                // 列优先, NDC1HWC0的列方向是HW即K方向
                uint64_t batchDoutMIndex = basicBlockIdx / kCnt;
                this->kCoreIndx_ = basicBlockIdx - batchDoutMIndex * kCnt;
                this->mCoreIndx_ = batchDoutMIndex % mCnt;
                batchDoutIndex = basicBlockIdx / mkCnt;
            }
            uint64_t batchIdx = batchDoutIndex / this->dout_;
            uint64_t doutIdx = batchDoutIndex - batchIdx * this->dout_;
            basicBlockIdx++;

            // 不可用totalCnt - 1作为尾块, totalCnt包含batch*dout
            uint64_t mCoreUse = (this->mCoreIndx_ == (mCnt - 1)) ? mCoreTail : this->singleShapeM_;
            uint64_t kCoreUse = (this->kCoreIndx_ == (kCnt - 1)) ? kCoreTail : this->singleShapeK_;

            this->CalcOffset(batchIdx, doutIdx, 0, 0, true);
            this->ReCalDkCinSingleCoreShape(batchIdx, doutIdx, 0, 0);
            if (this->singleShapeNInCurrentHo_ == 0 || this->singleShapeMInCurrentHo_ == 0) {
                continue;
            }
            this->dw_.SetOutBackprop(this->dedyGm_[this->offsetA_]);
            this->hoStartIdx_ = this->kCoreIndx_ * this->singleCoreHo_;
            this->dw_.SetSingleShape(mCoreUse, this->singleShapeNInCurrentHo_, kCoreUse, this->hoStartIdx_);
            this->dw_.SetFmap(this->xGm_[this->offsetB_]);
#if __CCE_AICORE__ == 220
            if constexpr (xFormat == FORMAT_NCDHW) { // Transdata Merge
                uint64_t wsOffset = block_idx * this->DOUBLE_BUFFER * this->gmPongOffset + (this->gmPingPongEventId_ ? this->gmPingOffset : this->gmPongOffset);
                this->dw_.SetFmap(this->workspaceXGm_[wsOffset]);
                if ASCEND_IS_AIV {
                    if (syncTimes > 1) {
                        CrossCoreWaitFlag(this->SYNC_AIC_AIV_FLAG + this->gmPingPongEventId_);
                    }
                    this->TransDataTo6HD(batchIdx, doutIdx);
                    CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(this->SYNC_AIV_AIC_FLAG + this->gmPingPongEventId_);
                    this->gmPingPongEventId_ &= 1;
                    this->gmPingPongEventId_ ^= 1;
                    syncTimes += 1;
                }
                if ASCEND_IS_AIC {
                    if constexpr (xFormat == FORMAT_NCDHW) { // Transdata Merge
                        CrossCoreWaitFlag(this->SYNC_AIV_AIC_FLAG + this->gmPingPongEventId_);
                        this->dw_.IterateAll(this->yGm_[this->offsetC_], 1);
                        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE2>(this->SYNC_AIC_AIV_FLAG + this->gmPingPongEventId_);
                        this->gmPingPongEventId_ &= 1;
                        this->gmPingPongEventId_ ^= 1;
                    }
                }
            } else {
                this->dw_.IterateAll(this->yGm_[this->offsetC_], 1);
            }
#else
            this->dw_.IterateAll(this->yGm_[this->offsetC_], 1);
#endif
            }
    }
};

template <typename xType, int xFormat, typename dedyType, int dedyFormat, typename yType, int yFormat>
class Conv3dDwBasicBlockSplitKN : public Conv3dDwBasicBlock<xType, xFormat, dedyType, dedyFormat, yType, yFormat> {
public:
    __aicore__ inline Conv3dDwBasicBlockSplitKN() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dedy,
                                GM_ADDR y, GM_ADDR workSpace,
                                const Conv3DBackpropFilterV2TilingData* tilingData) {
        this->InitCommonTilingData(tilingData);
        InitSplitTilingData(tilingData);
        // init global buffer
        this->xGm_.SetGlobalBuffer((__gm__ xType*)x);
        this->dedyGm_.SetGlobalBuffer((__gm__ dedyType*)dedy);
        this->yGm_.SetGlobalBuffer((__gm__ yType*)y);
        this->dw_.Init(&(tilingData->dwTiling));
        this->workspaceXGm_.SetGlobalBuffer((__gm__ xType*)workSpace);
    }

    __aicore__ inline void Process() {
        if (block_idx >= this->usedCoreNum_) {
            return;
        }
        if ASCEND_IS_AIV {
            if constexpr (xFormat != FORMAT_NCDHW) {
                return ;
            }
            this->InitTransdataBuffer();
        }
        CalBasicBlock();
        this->dw_.End();
    }

protected:
    static constexpr uint8_t SYNC_MODE2 = 2;

    __aicore__ inline void InitSplitTilingData(const Conv3DBackpropFilterV2TilingData* tilingData) {
        this->singleShapeM_ = this->m_;
        this->singleShapeK_ = tilingData->basicBlockTiling.singleCoreK;
        this->singleShapeN_ = tilingData->basicBlockTiling.singleCoreN;
    }

    __aicore__ inline void CalBasicBlock() {
        uint64_t nCnt = DivCeil(this->n_, this->singleShapeN_);
        uint64_t nCoreTail = this->n_ - (nCnt - 1) * this->singleShapeN_;

        uint64_t kCnt = DivCeil(this->k_, this->singleShapeK_);
        uint64_t kCoreTail = this->k_ - (kCnt - 1) * this->singleShapeK_;
        kCoreTail = kCoreTail > this->wo_ ? kCoreTail : this->wo_;

        // 记录基本块的位置
        uint64_t batchDout = static_cast<uint64_t>(this->batch_) * this->dout_;
        uint64_t nkCnt = nCnt * kCnt;
        uint64_t totalCnt = batchDout * nkCnt;
        uint64_t calRound = totalCnt / this->usedCoreNum_;
        uint64_t tailCnt = totalCnt - calRound * this->usedCoreNum_;
        uint64_t basicBlockIdx = 0;

        // 拖尾的部分依次分配到前面的核计算，这些核会多算一轮
        if (block_idx < tailCnt) {
            basicBlockIdx = block_idx * calRound + block_idx;
            ++calRound;
        } else {
            basicBlockIdx = block_idx * calRound + tailCnt;
        }

        //1:M*K 行优先绑核，2:M*K 列优先绑核
        uint64_t batchDoutIndex = 0;
        uint64_t batchDoutNcnt = batchDout * nCnt;
        this->mCoreIndx_ = 0; // 默认不分核
        uint64_t syncTimes = 0;
        for (uint64_t j = 0; j < calRound; ++j) {
            if (this->coreBindOrder_ == ROW_FIRST) {
                // 行优先, NDC1HWC0的行方向是C0即N方向
                this->kCoreIndx_ = basicBlockIdx / batchDoutNcnt;
                batchDoutIndex = (basicBlockIdx - this->kCoreIndx_ * batchDoutNcnt) / nCnt;
                this->nCoreIndx_ = basicBlockIdx - this->kCoreIndx_ * batchDoutNcnt - batchDoutIndex * nCnt;
            } else {
                // 列优先, NDC1HWC0的列方向是HW即K方向
                uint64_t batchDoutNIndex = basicBlockIdx / kCnt;
                this->kCoreIndx_ = basicBlockIdx - batchDoutNIndex * kCnt;
                this->nCoreIndx_ = batchDoutNIndex % nCnt;
                batchDoutIndex = basicBlockIdx / nkCnt;
            }
            uint64_t batchIdx = batchDoutIndex / this->dout_;
            uint64_t doutIdx = batchDoutIndex - batchIdx * this->dout_;
            basicBlockIdx++;

           // 不可用totalCnt - 1作为尾块, totalCnt包含batch*dout
            uint64_t nCoreUse = (this->nCoreIndx_ == (nCnt - 1)) ? nCoreTail : this->singleShapeN_;
            uint64_t kCoreUse = (this->kCoreIndx_ == (kCnt - 1)) ? kCoreTail : this->singleShapeK_;

            this->CalcOffset(batchIdx, doutIdx, 0, 0, true);
            this->ReCalDkCinSingleCoreShape(batchIdx, doutIdx, 0, 0);
            if (this->singleShapeNInCurrentHo_ == 0 || this->singleShapeMInCurrentHo_ == 0) {
                continue;
            }
            nCoreUse = nCoreUse > this->singleShapeNInCurrentHo_ ? this->singleShapeNInCurrentHo_ : nCoreUse;
            this->dw_.SetOutBackprop(this->dedyGm_[this->offsetA_]);
            this->hoStartIdx_ = this->kCoreIndx_ * this->singleCoreHo_;
            this->dw_.SetSingleShape(this->singleShapeM_, nCoreUse, kCoreUse, this->hoStartIdx_);
            this->dw_.SetFmap(this->xGm_[this->offsetB_]);
#if __CCE_AICORE__ == 220
            if constexpr (xFormat == FORMAT_NCDHW) { // Transdata Merge
                uint64_t wsOffset = block_idx * this->DOUBLE_BUFFER * this->gmPongOffset + (this->gmPingPongEventId_ ? this->gmPingOffset : this->gmPongOffset);
                this->dw_.SetFmap(this->workspaceXGm_[wsOffset]);
                if ASCEND_IS_AIV {
                    if (syncTimes > 1) {
                        CrossCoreWaitFlag(this->SYNC_AIC_AIV_FLAG + this->gmPingPongEventId_);
                    }
                    this->TransDataTo6HD(batchIdx, doutIdx);
                    CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(this->SYNC_AIV_AIC_FLAG + this->gmPingPongEventId_);
                    this->gmPingPongEventId_ &= 1;
                    this->gmPingPongEventId_ ^= 1;
                    syncTimes += 1;
                }
                if ASCEND_IS_AIC {
                    if constexpr (xFormat == FORMAT_NCDHW) { // Transdata Merge
                        CrossCoreWaitFlag(this->SYNC_AIV_AIC_FLAG + this->gmPingPongEventId_);
                        this->dw_.IterateAll(this->yGm_[this->offsetC_], 1);
                        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE2>(this->SYNC_AIC_AIV_FLAG + this->gmPingPongEventId_);
                        this->gmPingPongEventId_ &= 1;
                        this->gmPingPongEventId_ ^= 1;
                    }
                }
            } else {
                this->dw_.IterateAll(this->yGm_[this->offsetC_], 1);
            }
#else
            this->dw_.IterateAll(this->yGm_[this->offsetC_], 1);
#endif
            }
    }
};

template <typename xType, int xFormat, typename dedyType, int dedyFormat, typename yType, int yFormat>
class Conv3dDwBasicBlockSplitMN : public Conv3dDwBasicBlock<xType, xFormat, dedyType, dedyFormat, yType, yFormat> {
public:
    __aicore__ inline Conv3dDwBasicBlockSplitMN() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dedy,
                                GM_ADDR y, GM_ADDR workSpace,
                                const Conv3DBackpropFilterV2TilingData* tilingData) {
        this->InitCommonTilingData(tilingData);
        InitSplitTilingData(tilingData);
        // init global buffer
        this->xGm_.SetGlobalBuffer((__gm__ xType*)x);
        this->dedyGm_.SetGlobalBuffer((__gm__ dedyType*)dedy);
        this->yGm_.SetGlobalBuffer((__gm__ yType*)y);
        this->dw_.Init(&(tilingData->dwTiling));
        this->workspaceXGm_.SetGlobalBuffer((__gm__ xType*)workSpace);
    }

    __aicore__ inline void Process() {
        if (block_idx >= this->usedCoreNum_) {
            return;
        }
        if ASCEND_IS_AIV {
            if constexpr (xFormat != FORMAT_NCDHW) {
                return ;
            }
            this->InitTransdataBuffer();
        }
        CalBasicBlock();
        this->dw_.End();
    }

protected:
    static constexpr uint8_t SYNC_MODE2 = 2;

    __aicore__ inline void InitSplitTilingData(const Conv3DBackpropFilterV2TilingData* tilingData) {
        this->singleShapeM_ = tilingData->basicBlockTiling.singleCoreM;
        this->singleShapeK_ = this->k_;
        this->singleShapeN_ = tilingData->basicBlockTiling.singleCoreN;
    }

    __aicore__ inline void CalBasicBlock() {
        uint32_t mCnt = DivCeil(this->m_, this->singleShapeM_);
        uint32_t mCoreTail = this->m_ - (mCnt - 1) * this->singleShapeM_;

        uint64_t nCnt = DivCeil(this->n_, this->singleShapeN_);
        uint64_t nCoreTail = this->n_ - (nCnt - 1) * this->singleShapeN_;

        // 记录基本块的位置
        uint64_t batchDout = static_cast<uint64_t>(this->batch_) * this->dout_;
        uint64_t mnCnt = mCnt * nCnt;
        uint64_t totalCnt = batchDout * mnCnt;
        uint64_t calRound = totalCnt / this->usedCoreNum_;
        uint64_t tailCnt = totalCnt - calRound * this->usedCoreNum_;
        uint64_t basicBlockIdx = 0;

        // 拖尾的部分依次分配到前面的核计算，这些核会多算一轮
        if (block_idx < tailCnt) {
            basicBlockIdx = block_idx * calRound + block_idx;
            ++calRound;
        } else {
            basicBlockIdx = block_idx * calRound + tailCnt;
        }

        //1:M*K 行优先绑核，2:M*K 列优先绑核
        uint64_t batchDoutIndex = 0;
        uint64_t batchDoutNcnt = batchDout * nCnt;
        this->kCoreIndx_ = 0; // 默认不分核
        uint64_t syncTimes = 0;
        for (uint64_t j = 0; j < calRound; ++j) {
            if (this->coreBindOrder_ == ROW_FIRST) {
                // 行优先, NDC1HWC0的行方向是C0即N方向
                this->mCoreIndx_ = basicBlockIdx / batchDoutNcnt;
                batchDoutIndex = (basicBlockIdx - this->mCoreIndx_ * batchDoutNcnt) / nCnt;
                this->nCoreIndx_ = basicBlockIdx - this->mCoreIndx_ * batchDoutNcnt - batchDoutIndex * nCnt;
            } else {
                // 列优先, NDC1HWC0的列方向是Cout方向
                uint64_t batchDoutNIndex = basicBlockIdx / mCnt;
                this->mCoreIndx_ = basicBlockIdx - batchDoutNIndex * mCnt;
                this->nCoreIndx_ = batchDoutNIndex % nCnt;
                batchDoutIndex = basicBlockIdx / mnCnt;
            }
            uint64_t batchIdx = batchDoutIndex / this->dout_;
            uint64_t doutIdx = batchDoutIndex - batchIdx * this->dout_;

            basicBlockIdx++;

           // 不可用totalCnt - 1作为尾块, totalCnt包含batch*dout
            uint64_t mCoreUse = (this->mCoreIndx_ == (mCnt - 1)) ? mCoreTail : this->singleShapeM_;
            uint64_t nCoreUse = (this->nCoreIndx_ == (nCnt - 1)) ? nCoreTail : this->singleShapeN_;

            this->CalcOffset(batchIdx, doutIdx, 0, 0, true);
            this->ReCalDkCinSingleCoreShape(batchIdx, doutIdx, 0, 0);
            if (this->singleShapeNInCurrentHo_ == 0 || this->singleShapeMInCurrentHo_ == 0) {
                continue;
            }
            nCoreUse = nCoreUse > this->singleShapeNInCurrentHo_ ? this->singleShapeNInCurrentHo_ : nCoreUse;
            this->dw_.SetOutBackprop(this->dedyGm_[this->offsetA_]);
            this->hoStartIdx_ = this->kCoreIndx_ * this->singleCoreHo_;
            this->dw_.SetSingleShape(mCoreUse, nCoreUse, this->singleShapeK_, this->hoStartIdx_);
            this->dw_.SetFmap(this->xGm_[this->offsetB_]);
#if __CCE_AICORE__ == 220
            if constexpr (xFormat == FORMAT_NCDHW) { // Transdata Merge
                uint64_t wsOffset = block_idx * this->DOUBLE_BUFFER * this->gmPongOffset + (this->gmPingPongEventId_ ? this->gmPingOffset : this->gmPongOffset);
                this->dw_.SetFmap(this->workspaceXGm_[wsOffset]);
                if ASCEND_IS_AIV {
                    if (syncTimes > 1) {
                        CrossCoreWaitFlag(this->SYNC_AIC_AIV_FLAG + this->gmPingPongEventId_);
                    }
                    this->TransDataTo6HD(batchIdx, doutIdx);
                    CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(this->SYNC_AIV_AIC_FLAG + this->gmPingPongEventId_);
                    this->gmPingPongEventId_ &= 1;
                    this->gmPingPongEventId_ ^= 1;
                    syncTimes += 1;
                }
                if ASCEND_IS_AIC {
                    if constexpr (xFormat == FORMAT_NCDHW) { // Transdata Merge
                        CrossCoreWaitFlag(this->SYNC_AIV_AIC_FLAG + this->gmPingPongEventId_);
                        this->dw_.IterateAll(this->yGm_[this->offsetC_], 1);
                        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE2>(this->SYNC_AIC_AIV_FLAG + this->gmPingPongEventId_);
                        this->gmPingPongEventId_ &= 1;
                        this->gmPingPongEventId_ ^= 1;
                    }
                }
            } else {
                this->dw_.IterateAll(this->yGm_[this->offsetC_], 1);
            }
#else
            this->dw_.IterateAll(this->yGm_[this->offsetC_], 1);
#endif
            }
    }
};
}

#endif // CONV3D_BACKPROP_FILTER_BASIC_BLOCK_H