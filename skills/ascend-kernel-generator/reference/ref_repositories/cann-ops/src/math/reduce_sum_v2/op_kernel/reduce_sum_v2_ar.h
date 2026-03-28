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
 * \file reduce_sum_v2_ar.h
 * \brief
 */

#ifndef REDUCE_SUM_V2_AR_H
#define REDUCE_SUM_V2_AR_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "reduce_sum_v2_common.h"

namespace AscendC {

template<typename T_X, typename T_Y, uint8_t processId, uint8_t processNum>
class ReduceSumV2ARKernel : public ReduceSumV2KernelBase<T_X, T_Y> {
public:
    __aicore__ inline ReduceSumV2ARKernel() = delete;
    __aicore__ inline ReduceSumV2ARKernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                          const ReduceSumV2Process &processTiling, TPipe *pipe)
    {
        InitParams(processTiling);
        SetGmAddr(x, y, workspace);
        InitBuffers(pipe);
    }

    __aicore__ inline void Process()
    {   
        uint64_t xAOffset; // 输入地址偏移
        uint64_t yAOffset; // 输出地址偏移
        // former
        for (size_t a = 0; a < formerATimes; a++) {
            xAOffset = a * formerA * this->R;
            yAOffset = a * formerA;
            ProcessR(xAOffset, yAOffset, formerA);
        }
        // tail
        if (tailA) {
            xAOffset = formerATimes * formerA * this->R;
            yAOffset = formerATimes * formerA;
            ProcessR(xAOffset, yAOffset, tailA);
        }
        // group WorkspaceReduceSum
        if (needWorkspace) {
            WorkspaceReduceSum();
        }
    }

private:
    __aicore__ inline void ProcessR(const uint64_t &xAOffset, const uint64_t &yOffset, const uint64_t &rows)
    {
        uint64_t xOffset;
        rInCache_ = 0;
        for (size_t r = 0; r < formerRTimes; r++) {
            xOffset = xAOffset + r * formerR;
            DoReduce(xOffset, yOffset, rows, formerR);
            rInCache_++;
        }
        if (tailR) {
            xOffset = xAOffset + formerRTimes * formerR;
            DoReduce(xOffset, yOffset, rows, tailR);
            rInCache_++;
        }
        // R处理完，reduceCache
        if (needCache) {
            BinaryReduceSumInCache<ReducePattern::AR, true>(rows, rInCache_);
            if (needWorkspace) {
                CopyOutY2Ws(0, rows);
            } else {
                CopyOutY2Gm(yOffset, rows);
            }
        }
    }

    __aicore__ inline void InitParams(const ReduceSumV2Process &processTiling)
    {
        needInCast_ = !std::is_same<T_X, float>::value;
        needOutCast_ = !std::is_same<T_Y, float>::value;
        this->InitBasicParams(processTiling);
        blockAIdx_ = this->blockIdx_ / this->blockR.usedCoreNum;
        blockRIdx_ = this->blockIdx_ % this->blockR.usedCoreNum;

        // 0: 只切A；1: 切AR
        uint64_t ubInfosOffset = 0;

        if ((this->blockA.usedCoreNum && blockAIdx_ >= this->blockA.formerCoreNum) &&
            (this->blockR.usedCoreNum && blockRIdx_ >= this->blockR.formerCoreNum)) {
            ubInfosOffset = 1;
        }

        formerATimes = this->ubInfos[ubInfosOffset].formerATimes;
        formerA = this->ubInfos[ubInfosOffset].formerA;
        tailA = this->ubInfos[ubInfosOffset].tailA;
        formerRTimes = this->ubInfos[ubInfosOffset].formerRTimes;
        formerR = this->ubInfos[ubInfosOffset].formerR;
        tailR = this->ubInfos[ubInfosOffset].tailR;
        if (blockRIdx_ == this->blockR.usedCoreNum - 1) {
            formerRTimes = this->ubInfos[ubInfosOffset].formerRealTimes;
            tailR = this->ubInfos[ubInfosOffset].tailRealData;
        }
        inBufferSize_ = processTiling.inBufferSize;
        cacheBufferSize_ = processTiling.cacheBufferSize;
        outBufferSize_ = ops::CeilAlign(formerA, this->xFp32Align_) * sizeof(float);
        needCache = formerRTimes != 0;
        needWorkspace = this->blockR.usedCoreNum != 1;
        maxRInCache_ = cacheBufferSize_ / formerA / BLOCK_SIZE;
    }

    __aicore__ inline void InitBuffers(TPipe *pipe)
    {
        pipe->InitBuffer(dataInQue_, BUFFER_NUM, inBufferSize_);
        pipe->InitBuffer(dataOutQue_, BUFFER_NUM, outBufferSize_);
        pipe->InitBuffer(cacheBuf_, cacheBufferSize_);
        if (needInCast_) {
            pipe->InitBuffer(dataInCastQue_, BUFFER_NUM, inBufferSize_ / 2);
        }
    }

    __aicore__ inline void SetGmAddr(GM_ADDR x, GM_ADDR y, GM_ADDR workspace)
    {
        uint64_t xOffset = 0;
        uint64_t yOffset = 0;
        uint64_t baseWsOffset = 0;

        xOffset += this->GetOffset(blockAIdx_, this->R, this->blockA);
        xOffset += this->GetOffset(blockRIdx_, 1, this->blockR);

        yOffset += this->GetOffset(blockAIdx_, 1, this->blockA);

        if (processId % 2) {
            this->xGm_.SetGlobalBuffer((__gm__ T_X*)workspace + xOffset);
            baseWsOffset = this->A * this->R;
        } else {
            this->xGm_.SetGlobalBuffer((__gm__ T_X*)x + xOffset);
        }
        this->yGm_.SetGlobalBuffer((__gm__ T_Y*)y + yOffset);

        if (needWorkspace) {
            uint64_t syncOffset = this->blockA.usedCoreNum * this->blockR.usedCoreNum * BLOCK_SIZE / sizeof(float);
            uint64_t wsOffset = blockAIdx_ * this->blockR.usedCoreNum + blockRIdx_;
            
            syncGm_.SetGlobalBuffer((__gm__ int32_t*)workspace + ops::CeilDiv(baseWsOffset * sizeof(float), sizeof(int32_t))); // group
            wsGm_.SetGlobalBuffer((__gm__ float*)workspace + ops::CeilAlign(baseWsOffset * sizeof(float), sizeof(int32_t)) / sizeof(float)
                                                         + syncOffset + wsOffset); // group
            uint64_t syncDataNum = BLOCK_SIZE / sizeof(int32_t);
            GlobalTensor<int32_t> thisSyncGm = syncGm_[this->blockIdx_ * syncDataNum];
            InitGlobalMemory(thisSyncGm, syncDataNum, 0);
            this->MTE3ToMTE2Sync();
        }
    }

    __aicore__ inline void DoReduce(const uint64_t &xOffset, const uint64_t &yOffset, const uint64_t &rows, const uint64_t &cols)
    {
        CopyIn(xOffset, rows, cols);
        if (needCache) {
            if (rInCache_ == maxRInCache_) {
                BinaryReduceSumInCache<ReducePattern::AR, false>(rows, rInCache_);
                rInCache_ = 1;
            }
            ReduceSumProcess(rows, cols, rInCache_);
        } else {
            ReduceSumProcess(rows, cols);
            if (needWorkspace) {
                CopyOutY2Ws(0, rows);
            } else {
                CopyOutY2Gm(yOffset, rows);
            }
        }
    }

    __aicore__ inline void ReduceSumProcess(const uint64_t &rows, const uint64_t &cols, const uint64_t &cacheOffset = 0)
    {
        if (needCache) {
            LocalTensor<float> xLocal = dataInQue_.DeQue<float>();
            LocalTensor<float> cacheLocal = cacheBuf_.Get<float>();
            BinaryReduceSum<ReducePattern::AR>(cacheLocal[cacheOffset], xLocal, rows, cols);
            dataInQue_.FreeTensor<float>(xLocal);
            /////
        } else {
            LocalTensor<float> xLocal = dataInQue_.DeQue<float>();
            LocalTensor<float> yLocal = dataOutQue_.AllocTensor<float>();
            BinaryReduceSum<ReducePattern::AR>(yLocal, xLocal, rows, cols);
            dataInQue_.FreeTensor<float>(xLocal);
            dataOutQue_.EnQue<float>(yLocal);
        }
    }

    // 多行 - 连续 A,A 不需要cache直接输出到yGm
    template<ReducePattern pattern>
    __aicore__ inline void BinaryReduceSum(const LocalTensor<float> &dstLocal,
                                           const LocalTensor<float> &srcLocal,
                                           const uint64_t &rows,
                                           const uint64_t &cols)
    {
        uint64_t dstOffset;
        for (size_t row = 0; row < rows; row++) {
            uint64_t reduceNum = cols;
            uint64_t offset0 = row * ops::CeilAlign(cols, this->xAlign_); // 32B对齐
            while (reduceNum > wholeReduceNum) {
                uint64_t point = this->FindCutPoint(reduceNum);
                uint64_t remain = reduceNum - point;
                uint64_t calCount = this->Min(point, remain);                    
                uint64_t offset1 = offset0 + point;
                Add(srcLocal[offset0], srcLocal[offset0], srcLocal[offset1], calCount);
                reduceNum = point;
            }
            
            if (needCache) {
                dstOffset = row * ops::CeilAlign(formerRTimes + static_cast<uint64_t>(tailR != 0), this->xFp32Align_);
            } else {
                dstOffset = row;
            }
            UnitReduceSum(dstLocal[dstOffset], srcLocal[offset0], reduceNum);
        }
    }

    template<ReducePattern pattern, bool is2Gm>
    __aicore__ inline void BinaryReduceSumInCache(const uint64_t &rows, const uint64_t &reduceNum)
    {
        if (is2Gm) {
            LocalTensor<float> cacheLocal = cacheBuf_.Get<float>();
            LocalTensor<float> yLocal = dataOutQue_.AllocTensor<float>();
            uint64_t curReduceNum = reduceNum;
            for (size_t row = 0; row < rows; row++) {
                uint64_t offset0 = row * ops::CeilAlign(formerRTimes + static_cast<uint64_t>(tailR != 0), this->xFp32Align_); // 32B对齐
                while (curReduceNum > wholeReduceNum) {
                    uint64_t point = this->FindCutPoint(curReduceNum);
                    uint64_t remain = curReduceNum - point;
                    uint64_t calCount = this->Min(point, remain);                    
                    uint64_t offset1 = offset0 + point;
                    Add(cacheLocal[offset0], cacheLocal[offset0], cacheLocal[offset1], calCount);
                    curReduceNum = point;
                }
                UnitReduceSum(yLocal[row], cacheLocal[offset0], curReduceNum);
            }
            dataOutQue_.EnQue<float>(yLocal);
        }
    }
    
    __aicore__ inline void UnitReduceSum(const LocalTensor<float> &dstLocal, const LocalTensor<float> &srcLocal, const uint64_t &curReduceNum)
    {
        if (likely(curReduceNum > 1)) {
            WholeReduceSum<float>(dstLocal, srcLocal, curReduceNum, 1, 1, 1, 8);
        } else {
            dstLocal.SetValue(0, srcLocal.GetValue(0));
        }
    }

    __aicore__ inline void CopyIn(const uint64_t &xOffset, const uint64_t &rows, const uint64_t &curReduceNum)
    {
        LocalTensor<T_X> xLocal;
        if (needInCast_) {
            xLocal = dataInCastQue_.AllocTensor<T_X>();
        } else {
            xLocal = dataInQue_.AllocTensor<T_X>();
        }
        uint8_t padNum = (this->xAlign_ - curReduceNum % this->xAlign_) % this->xAlign_;
        // 非连续
        DataCopyExtParams copyParams{static_cast<uint16_t>(rows), static_cast<uint32_t>(sizeof(T_X) * curReduceNum),
                                     static_cast<uint32_t>((this->R - curReduceNum) * sizeof(T_X)), 0, 0};
        DataCopyPadExtParams<T_X> padParams{true, 0, padNum, 0};
        DataCopyPad(xLocal, this->xGm_[xOffset], copyParams, padParams);
        if (needInCast_) {
            dataInCastQue_.EnQue<T_X>(xLocal);
            DoInCast(rows, curReduceNum);
        } else {
            dataInQue_.EnQue<T_X>(xLocal);
        }
    }

    // group 连续搬出
    __aicore__ inline void CopyOutY2Ws(const uint64_t &wsOffset, const uint64_t &dataNum)
    {
        LocalTensor<float> wsLocal = dataOutQue_.DeQue<float>();
        DataCopyExtParams wsCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float) * dataNum), 0, 0, 0};
        DataCopyPad(wsGm_[wsOffset], wsLocal, wsCopyParams);
        dataOutQue_.FreeTensor<float>(wsLocal);
    }

    __aicore__ inline void CopyOutY2Gm(const uint64_t &yOffset, const uint64_t &dataNum)
    {
        if (needOutCast_) {
            DoOutCast(dataNum);
        }
        LocalTensor<T_Y> yLocal = dataOutQue_.DeQue<T_Y>();
        DataCopyExtParams yCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(T_Y) * dataNum), 0, 0, 0};
        DataCopyPad(this->yGm_[yOffset], yLocal, yCopyParams);
        dataOutQue_.FreeTensor<T_Y>(yLocal);
    }

    // group AR处理，连续搬出，规约核处理
    __aicore__ inline void WorkspaceReduceSum()
    {
        // 这行的其他核都算完了
        LocalTensor<int32_t> syncBuf = cacheBuf_.Get<int32_t>();
        IBSet(syncGm_, syncBuf, this->blockIdx_, 0);
        if (this->blockRIdx_ == 0) {
            for (size_t coreId = 0; coreId < this->blockR.usedCoreNum; coreId++) {
                IBWait(syncGm_, syncBuf, this->blockIdx_ + coreId, 0);
            }
            // 统一在这行的首个核规约
            CopyInFromWs(this->blockR.usedCoreNum);
            LocalTensor<float> xLocal = dataInQue_.DeQue<float>();
            LocalTensor<float> yLocal = dataOutQue_.AllocTensor<float>();
            UnitReduceSum(yLocal, xLocal, this->blockR.usedCoreNum);
            dataInQue_.FreeTensor<float>(xLocal);
            dataOutQue_.EnQue<float>(yLocal);
            CopyOutY2Gm(0, 1);
        }
    }

    __aicore__ inline void CopyInFromWs(const uint64_t &curReduceNum)
    {
        LocalTensor<float> xLocal = dataInQue_.AllocTensor<float>();
        uint8_t padNum = (this->xFp32Align_ - curReduceNum % this->xFp32Align_) % this->xFp32Align_;
        // 非连续
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float) * curReduceNum), 0, 0, 0};
        
        DataCopyPadExtParams<float> padParams{true, 0, padNum, 0};
        DataCopyPad(xLocal, wsGm_, copyParams, padParams);
        dataInQue_.EnQue<float>(xLocal);
    }

    __aicore__ inline void DoOutCast(const uint64_t &dataNum)
    {
        LocalTensor<float> yLocalFp32 = dataOutQue_.DeQue<float>();
        LocalTensor<T_Y> yLocal = yLocalFp32.template ReinterpretCast<T_Y>();
        Cast(yLocal, yLocalFp32, RoundMode::CAST_RINT, dataNum);
        dataOutQue_.EnQue<float>(yLocalFp32);
    }

    __aicore__ inline void DoInCast(const uint64_t &rows, const uint64_t &cols)
    {
        uint64_t dataNum = rows * ops::CeilAlign(cols, this->xAlign_);
        LocalTensor<T_X> xLocal = dataInCastQue_.DeQue<T_X>();
        LocalTensor<float> xLocalFp32 = dataInQue_.AllocTensor<float>();
        Cast(xLocalFp32, xLocal, RoundMode::CAST_NONE, dataNum);
        dataInCastQue_.FreeTensor<T_X>(xLocal);
        dataInQue_.EnQue<float>(xLocalFp32);
    }

private:
    uint64_t inBufferSize_;
    uint64_t cacheBufferSize_;
    uint64_t outBufferSize_;
    uint64_t wholeReduceNum = WHOLE_REDUCE_SIZE / sizeof(float);
    uint64_t reduceNumInCache_ = 0;
    uint64_t r2Cache;
    bool needCache = false; //切分R需要cache
    bool needWorkspace = false;
    bool needInCast_ = false;
    bool needOutCast_ = false;
    uint64_t rInCache_;

    TQue<TPosition::VECIN, BUFFER_NUM> dataInQue_;
    TQue<TPosition::VECIN, BUFFER_NUM> dataInCastQue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> dataOutQue_;
    TBuf<TPosition::VECCALC> cacheBuf_;

    uint64_t blockAIdx_;
    uint64_t blockRIdx_;
    uint64_t formerATimes;
    uint64_t formerA;
    uint64_t tailA;
    uint64_t formerRTimes;
    uint64_t formerR;
    uint64_t tailR;

    GlobalTensor<float> wsGm_;
    GlobalTensor<int32_t> syncGm_;
    uint64_t maxRInCache_;
};
}

#endif // REDUCE_SUM_V2_AR_H