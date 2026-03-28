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
 * \file reduce_sum_v2_ara.h
 * \brief
 */

#ifndef REDUCE_SUM_V2_ARA_H
#define REDUCE_SUM_V2_ARA_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "reduce_sum_v2_common.h"

namespace AscendC {

template<typename T_X, typename T_Y, uint8_t processId, uint8_t processNum>
class ReduceSumV2ARAKernel : public ReduceSumV2KernelBase<T_X, T_Y> {
public:
    __aicore__ inline ReduceSumV2ARAKernel() = delete;
    __aicore__ inline ReduceSumV2ARAKernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                          const ReduceSumV2Process &processTiling, TPipe *pipe)
    {
        InitParams(processTiling);
        SetGmAddr(x, y, workspace);
        InitBuffers(pipe);
    }

    // A -> A1 -> R
    __aicore__ inline void Process()
    {
        uint64_t xAOffset; // 输入地址偏移
        uint64_t yAOffset; // 输出地址偏移
        // former
        for (size_t a = 0; a < formerATimes; a++) {
            for (size_t i = 0; i < formerA; i++) {
                xAOffset = (a * formerA + i) * this->R * this->A1;
                yAOffset = (a * formerA + i) * this->A1;
                ProcessA1(xAOffset, yAOffset);
            }
        }
        // tail
        if (tailA) {
            for (size_t i = 0; i < tailA; i++) {
                xAOffset = (formerATimes * formerA + i) * this->R * this->A1;
                yAOffset = (formerATimes * formerA + i) * this->A1;
                ProcessA1(xAOffset, yAOffset);
            }
        }
        // group WorkspaceReduceSum
        if (needWorkspace) {
            WorkspaceReduceSum();
        }
    }

private:
    __aicore__ inline void InitParams(const ReduceSumV2Process &processTiling)
    {
        needInCast_ = !std::is_same<T_X, float>::value;
        needOutCast_ = !std::is_same<T_Y, float>::value;
        this->InitBasicParams(processTiling);
        this->InitBlockParams(this->blockA1, processTiling.blockA1);
        blockAIdx_ = this->blockIdx_ / (this->blockR.usedCoreNum * this->blockA1.usedCoreNum);
        blockRIdx_ = this->blockIdx_ % (this->blockR.usedCoreNum * this->blockA1.usedCoreNum) / this->blockA1.usedCoreNum;
        blockA1Idx_ = this->blockIdx_ % (this->blockR.usedCoreNum * this->blockA1.usedCoreNum) % this->blockA1.usedCoreNum;

        uint64_t ubInfosOffset = 0;
        if ((this->blockA.usedCoreNum && blockAIdx_ >= this->blockA.formerCoreNum) &&
            (this->blockR.usedCoreNum && blockRIdx_ >= this->blockR.formerCoreNum) &&
            (this->blockA1.usedCoreNum && blockA1Idx_ >= this->blockA1.formerCoreNum)) {
            ubInfosOffset = 1;
        }
        
        formerATimes = this->ubInfos[ubInfosOffset].formerATimes;
        formerA = this->ubInfos[ubInfosOffset].formerA;
        tailA = this->ubInfos[ubInfosOffset].tailA;
        formerRTimes = this->ubInfos[ubInfosOffset].formerRTimes;
        formerR = this->ubInfos[ubInfosOffset].formerR;
        tailR = this->ubInfos[ubInfosOffset].tailR;
        formerA1Times = this->ubInfos[ubInfosOffset].formerA1Times;
        formerA1 = this->ubInfos[ubInfosOffset].formerA1;
        tailA1 = this->ubInfos[ubInfosOffset].tailA1;
        if (blockA1Idx_ == this->blockA1.usedCoreNum - 1) {
            formerA1Times = this->ubInfos[ubInfosOffset].formerRealTimes;
            tailA1 = this->ubInfos[ubInfosOffset].tailRealData;
        }
        inBufferSize_ = processTiling.inBufferSize;
        cacheBufferSize_ = processTiling.cacheBufferSize;
        outBufferSize_ = ops::CeilAlign(formerA1, this->xFp32Align_) * sizeof(float);
        needCache = this->R > inBufferSize_ / CACHELINE;
        needWorkspace = this->blockR.usedCoreNum != 1;
    }

    __aicore__ inline void InitBuffers(TPipe *pipe)
    {
        pipe->InitBuffer(dataInQue_, BUFFER_NUM, inBufferSize_);
        pipe->InitBuffer(dataOutQue_, BUFFER_NUM, outBufferSize_);
        pipe->InitBuffer(cacheBuf_, cacheBufferSize_);
        if (needWorkspace) {
            pipe->InitBuffer(syncBuf_, BLOCK_SIZE);
        }
        if (needInCast_) {
            pipe->InitBuffer(dataInCastQue_, BUFFER_NUM, inBufferSize_ / 2);
        }
    }

    __aicore__ inline void SetGmAddr(GM_ADDR x, GM_ADDR y, GM_ADDR workspace)
    {
        uint64_t xOffset = 0;
        uint64_t yOffset = 0;
        uint64_t wsOffset = 0;
        uint64_t syncOffset = 0;
        uint64_t baseWsOffset = 0;

        xOffset += this->GetOffset(blockAIdx_, this->R * this->A1, this->blockA);
        xOffset += this->GetOffset(blockRIdx_, this->A1, this->blockR);
        xOffset += this->GetOffset(blockA1Idx_, 1, this->blockA1);

        yOffset += this->GetOffset(blockAIdx_, this->A1, this->blockA);
        yOffset += this->GetOffset(blockA1Idx_, 1, this->blockA1);

        if (processId == processNum - 1) {
            if (processId % 2) {
                this->xGm_.SetGlobalBuffer((__gm__ T_X*)workspace + xOffset);
                baseWsOffset = this->A * this->R * this->A1;
            } else {
                this->xGm_.SetGlobalBuffer((__gm__ T_X*)x + xOffset);
            }
            this->yGm_.SetGlobalBuffer((__gm__ T_Y*)y + yOffset);
        } else {
            if (processId % 2) {
                this->xGm_.SetGlobalBuffer((__gm__ T_X*)workspace + xOffset);
                this->yGm_.SetGlobalBuffer((__gm__ T_Y*)x + yOffset);
                baseWsOffset = this->A * this->R * this->A1;
            } else {
                this->xGm_.SetGlobalBuffer((__gm__ T_X*)x + xOffset);
                this->yGm_.SetGlobalBuffer((__gm__ T_Y*)workspace + yOffset);
                baseWsOffset = this->A * this->A1;
            }            
        }
        
        if (needWorkspace) {
            syncOffset = this->blockA.usedCoreNum * this->blockR.usedCoreNum * this->blockA1.usedCoreNum * BLOCK_SIZE / sizeof(float);
            wsOffset += blockAIdx_ * this->blockR.usedCoreNum * this->blockA1.usedCoreNum * this->blockA1.formerUnitDataLen;
            wsOffset += blockRIdx_ * this->blockA1.usedCoreNum * this->blockA1.formerUnitDataLen;
            wsOffset += blockA1Idx_ * this->blockA1.formerUnitDataLen;
            syncGm_.SetGlobalBuffer((__gm__ int32_t*)workspace + ops::CeilDiv(baseWsOffset * sizeof(float), sizeof(int32_t))); // group
            wsGm_.SetGlobalBuffer((__gm__ float*)workspace + ops::CeilAlign(baseWsOffset * sizeof(float), sizeof(int32_t)) / sizeof(float)
                                                         + syncOffset + wsOffset); // group
            uint64_t syncDataNum = BLOCK_SIZE / sizeof(int32_t);
            GlobalTensor<int32_t> thisSyncGm = syncGm_[this->blockIdx_ * syncDataNum];
            InitGlobalMemory(thisSyncGm, syncDataNum, 0);
            this->MTE3ToMTE2Sync();
        }
    }

    __aicore__ inline void ProcessA1(const uint64_t &xAOffset, const uint64_t &yAOffset)
    {
        uint64_t xAA1Offset;
        uint64_t yOffset;
        // former
        for (size_t a1 = 0; a1 < formerA1Times; a1++) {
            xAA1Offset = xAOffset + a1 * formerA1;
            yOffset = yAOffset + a1 * formerA1;
            ProcessR(xAA1Offset, yOffset, formerA1);
        }
        // tail
        if (tailA1) {
            xAA1Offset = xAOffset + formerA1Times * formerA1;
            yOffset = yAOffset + formerA1Times * formerA1;
            ProcessR(xAA1Offset, yOffset, tailA1);
        }
    }

    __aicore__ inline void ProcessR(const uint64_t &xAA1Offset, const uint64_t &yOffset, const uint64_t &cols)
    {
        uint64_t xOffset;
        uint64_t cacheOffset = 0;
        rInCache_ = 0;   // 初始化
        // calcMaxR
        maxRInCache_ = cacheBufferSize_ / (ops::CeilAlign(cols, this->xFp32Align_) * sizeof(float));
        // former
        for (size_t rTime = 0; rTime < formerRTimes; rTime++) {
            xOffset = xAA1Offset + rTime * formerR * this->A1;
            cacheOffset += static_cast<uint64_t>(rTime != 0) * ops::CeilAlign(cols, this->xFp32Align_);
            DoReduce(xOffset, yOffset, cacheOffset, formerR, cols);
            rInCache_++;
        }
        // tail
        if (tailR) {
            xOffset = xAA1Offset + formerRTimes * formerR * this->A1;
            cacheOffset += static_cast<uint64_t>(formerRTimes != 0) * ops::CeilAlign(cols, this->xFp32Align_);
            DoReduce(xOffset, yOffset, cacheOffset, tailR, cols);
            rInCache_++;
        }
        // 缓存内Reduce
        if (needCache) {
            BinaryReduceSumInCache<ReducePattern::RA, true>(rInCache_, cols);
            if (needWorkspace) {
                CopyOutY2Ws(0, cols);
            } else {
                CopyOutY2Gm(yOffset, cols);
            }
        }
    }

    __aicore__ inline void DoReduce(const uint64_t &xOffset, const uint64_t &yOffset, uint64_t &cacheOffset,
                                    const uint64_t &rows, const uint64_t &cols)
    {
        CopyIn(xOffset, rows, cols, this->A1);
        if (needCache) {
            // 如果缓存满了，需要缓存内Reduce
            if (rInCache_ == maxRInCache_) {
                BinaryReduceSumInCache<ReducePattern::RA, false>(rInCache_, cols);
                rInCache_ = 1;
                cacheOffset -= (maxRInCache_ - 1) * ops::CeilAlign(cols, this->xFp32Align_); // ??
            }
            ReduceSumProcess(rows, cols, cacheOffset);
        } else {
            ReduceSumProcess(rows, cols);
            if (needWorkspace) {
                CopyOutY2Ws(0, cols);
            } else {
                CopyOutY2Gm(yOffset, cols);
            }
        }
    }

    __aicore__ inline void ReduceSumProcess(const uint64_t &rows, const uint64_t &cols, const uint64_t &cacheOffset = 0)
    {
        if (needCache) {
            LocalTensor<float> xLocal = dataInQue_.DeQue<float>();
            LocalTensor<float> cacheLocal = cacheBuf_.Get<float>();
            BinaryReduceSum<ReducePattern::RA>(cacheLocal[cacheOffset], xLocal, rows, cols, this->xAlign_);
            dataInQue_.FreeTensor<float>(xLocal);
        } else {
            LocalTensor<float> xLocal = dataInQue_.DeQue<float>();
            LocalTensor<float> yLocal = dataOutQue_.AllocTensor<float>();
            BinaryReduceSum<ReducePattern::RA>(yLocal, xLocal, rows, cols, this->xAlign_);
            dataInQue_.FreeTensor<float>(xLocal);
            dataOutQue_.EnQue<float>(yLocal);
        }
    }

    __aicore__ inline void DoCopy(const LocalTensor<float> &dstLocal,
                                  const LocalTensor<float> &srcLocal,
                                  const uint64_t &cols)
    {
        uint64_t repeatNum = REPEAT_SIZE / sizeof(float);
        uint8_t maxRepeatTimes = REPEAT_SIZE - 1;
        uint64_t copyTimes = cols / (maxRepeatTimes * repeatNum);
        uint64_t remainCols = cols - copyTimes * maxRepeatTimes * repeatNum;
        uint64_t dataOffset = 0;
        for (size_t i = 0; i < copyTimes; i++) {
            dataOffset = i * maxRepeatTimes * repeatNum;
            Copy(dstLocal[dataOffset], srcLocal[dataOffset], repeatNum, maxRepeatTimes, {1, 1, REPEAT_STRIDE, REPEAT_STRIDE});
        }
        if (remainCols) {
            uint8_t repeatTimes = remainCols / repeatNum;
            uint64_t remainMask = remainCols - repeatNum * repeatTimes;
            dataOffset = copyTimes * maxRepeatTimes * repeatNum;
            Copy(dstLocal[dataOffset], srcLocal[dataOffset], repeatNum, repeatTimes, {1, 1, REPEAT_STRIDE, REPEAT_STRIDE});
            if (remainMask) {
                dataOffset = copyTimes * maxRepeatTimes * repeatNum + repeatTimes * repeatNum;
                Copy(dstLocal[dataOffset], srcLocal[dataOffset], repeatNum, maxRepeatTimes, {1, 1, REPEAT_STRIDE, REPEAT_STRIDE});
            }
        }
    }

    // 多行 - 连续 A,A 不需要cache直接输出到yGm
    template<ReducePattern pattern>
    __aicore__ inline void BinaryReduceSum(const LocalTensor<float> &dstLocal,
                                           const LocalTensor<float> &srcLocal,
                                           const uint64_t &rows,
                                           const uint64_t &cols,
                                           const uint64_t &xAlign)
    {
        if constexpr (pattern == ReducePattern::RA) {
            if (rows == 1) {
                DoCopy(dstLocal, srcLocal, cols);
                return;
            }
            uint64_t reduceNum = rows;
            uint64_t alignA = ops::CeilAlign(cols, xAlign);
            uint64_t calCount = cols;
            // 找到R的二分点
            while (reduceNum > 2) {
                uint64_t point = this->FindCutPoint(reduceNum);
                uint64_t remain = reduceNum - point;
                // 遍历折叠部分，依次累加
                for (size_t i = 0; i < remain; i++) {
                    uint64_t offset0 = i * alignA;
                    uint64_t offset1 = (point + i) * alignA;
                    Add(srcLocal[offset0], srcLocal[offset0], srcLocal[offset1], calCount);
                }
                reduceNum = point;
            }
            // 最后一次累加到vecout
            Add(dstLocal, srcLocal, srcLocal[alignA], calCount);
        }
    }

    template<ReducePattern pattern, bool is2Gm>
    __aicore__ inline void BinaryReduceSumInCache(const uint64_t &rows, const uint64_t &cols)
    {
        if constexpr (pattern == ReducePattern::RA) {
            LocalTensor<float> cacheLocal = cacheBuf_.Get<float>();
            if constexpr (is2Gm) {
                LocalTensor<float> yLocal = dataOutQue_.AllocTensor<float>();
                BinaryReduceSum<ReducePattern::RA>(yLocal, cacheLocal, rows, cols, this->xFp32Align_);
                dataOutQue_.EnQue<float>(yLocal);
            } else {
                BinaryReduceSum<ReducePattern::RA>(cacheLocal, cacheLocal, rows, cols, this->xFp32Align_);
            }
        }
    }

    __aicore__ inline void CopyIn(const uint64_t &xOffset, const uint64_t &rows, const uint64_t &cols, const uint64_t &originCols)
    {
        LocalTensor<T_X> xLocal;
        if (needInCast_) {
            xLocal = dataInCastQue_.AllocTensor<T_X>();
        } else {
            xLocal = dataInQue_.AllocTensor<T_X>();
        }
        uint8_t padNum = (this->xAlign_ - cols % this->xAlign_) % this->xAlign_;
        // 非连续
        DataCopyExtParams copyParams{static_cast<uint16_t>(rows), static_cast<uint32_t>(sizeof(T_X) * cols),
                                     static_cast<uint32_t>((originCols - cols) * sizeof(T_X)), 0, 0};
        DataCopyPadExtParams<T_X> padParams{true, 0, padNum, 0};
        DataCopyPad(xLocal, this->xGm_[xOffset], copyParams, padParams);
        if (needInCast_) {
            dataInCastQue_.EnQue<T_X>(xLocal);
            DoInCast(rows, cols);
        } else {
            dataInQue_.EnQue<T_X>(xLocal);
        }
    }

    __aicore__ inline void CopyOutY2Gm(const uint64_t &yOffset, const uint64_t &dataNums)
    {
        if (needOutCast_) {
            DoOutCast(dataNums);
        }
        LocalTensor<T_Y> yLocal = dataOutQue_.DeQue<T_Y>();
        DataCopyExtParams yCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(T_Y) * dataNums), 0, 0, 0};
        DataCopyPad(this->yGm_[yOffset], yLocal, yCopyParams);
        dataOutQue_.FreeTensor<T_Y>(yLocal);
    }

    __aicore__ inline void CopyOutY2Ws(const uint64_t &wsOffset, const uint64_t &dataNum)
    {
        LocalTensor<float> wsLocal = dataOutQue_.DeQue<float>();
        DataCopyExtParams wsCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float) * dataNum), 0, 0, 0};
        DataCopyPad(wsGm_[wsOffset], wsLocal, wsCopyParams);
        dataOutQue_.FreeTensor<float>(wsLocal);
    }

    // group AR处理，连续搬出，规约核处理
    __aicore__ inline void WorkspaceReduceSum()
    {
        // 这行的其他核都算完了
        LocalTensor<int32_t> syncBuf = syncBuf_.Get<int32_t>();
        IBSet(syncGm_, syncBuf, this->blockIdx_, 0);
        if (this->blockRIdx_ == 0) {
            for (size_t coreId = 0; coreId < this->blockR.usedCoreNum; coreId++) {
                IBWait(syncGm_, syncBuf, this->blockIdx_ + coreId * this->blockA1.usedCoreNum, 0);
            }
            // 统一在这行的首个核规约
            CopyInFromWs(this->blockR.usedCoreNum, tailA1, this->blockA1.usedCoreNum * this->blockA1.formerUnitDataLen);
            LocalTensor<float> xLocal = dataInQue_.DeQue<float>();
            LocalTensor<float> yLocal = dataOutQue_.AllocTensor<float>();
            BinaryReduceSum<ReducePattern::RA>(yLocal, xLocal, this->blockR.usedCoreNum, tailA1, this->xFp32Align_);
            dataInQue_.FreeTensor<float>(xLocal);
            dataOutQue_.EnQue<float>(yLocal);
            CopyOutY2Gm(0, tailA1);
        }
    }

    __aicore__ inline void CopyInFromWs(const uint64_t &rows, const uint64_t &cols, const uint64_t &originCols)
    {
        LocalTensor<float> xLocal = dataInQue_.AllocTensor<float>();
        uint8_t padNum = (this->xFp32Align_ - cols % this->xFp32Align_) % this->xFp32Align_;
        DataCopyExtParams copyParams{static_cast<uint16_t>(rows), static_cast<uint32_t>(sizeof(float) * cols),
                                     static_cast<uint32_t>((originCols - cols) * sizeof(float)), 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, padNum, 0};
        DataCopyPad(xLocal, wsGm_, copyParams, padParams);
        dataInQue_.EnQue<float>(xLocal);
    }

    __aicore__ inline void DoInCast(const uint64_t &rows, const uint64_t &cols)
    {
        uint64_t dataNums = rows * ops::CeilAlign(cols, this->xAlign_);
        LocalTensor<T_X> xLocal = dataInCastQue_.DeQue<T_X>();
        LocalTensor<float> xLocalFp32 = dataInQue_.AllocTensor<float>();
        Cast(xLocalFp32, xLocal, RoundMode::CAST_NONE, dataNums);
        dataInCastQue_.FreeTensor<T_X>(xLocal);
        dataInQue_.EnQue<float>(xLocalFp32);
    }

    __aicore__ inline void DoOutCast(const uint64_t &dataNum)
    {
        LocalTensor<float> yLocalFp32 = dataOutQue_.DeQue<float>();
        LocalTensor<T_Y> yLocal = yLocalFp32.template ReinterpretCast<T_Y>();
        Cast(yLocal, yLocalFp32, RoundMode::CAST_RINT, dataNum);
        dataOutQue_.EnQue<float>(yLocalFp32);
    }

private:
    uint64_t inBufferSize_;
    uint64_t cacheBufferSize_ ;
    uint64_t outBufferSize_;
    uint64_t maxRInCache_ = 0; // cache内r的最大值
    uint64_t rInCache_; // cache内r的数量。即有多少行
    uint64_t r2Cache;

    TQue<TPosition::VECIN, BUFFER_NUM> dataInQue_;
    TQue<TPosition::VECIN, BUFFER_NUM> dataInCastQue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> dataOutQue_;
    TBuf<TPosition::VECCALC> cacheBuf_;

    uint64_t formerATimes;
    uint64_t formerA;
    uint64_t tailA;
    uint64_t formerRTimes;
    uint64_t formerR;
    uint64_t tailR;
    uint64_t formerA1Times;
    uint64_t formerA1;
    uint64_t tailA1;

    uint64_t blockAIdx_;
    uint64_t blockRIdx_;
    uint64_t blockA1Idx_;

    bool needCache = false;
    bool needWorkspace = false;
    bool needInCast_ = false;
    bool needOutCast_ = false;
    GlobalTensor<float> wsGm_;
    GlobalTensor<int32_t> syncGm_;
    TBuf<TPosition::VECCALC> syncBuf_;
};
}

#endif // REDUCE_SUM_V2_ARA_H