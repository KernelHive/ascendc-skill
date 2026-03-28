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
 * \file circular_pad.h
 * \brief
 */
#ifndef CIRCULAR_PAD_H
#define CIRCULAR_PAD_H
#include "circular_pad_common.h"
using namespace AscendC;

struct LoopParams
{
    int64_t loopW{0};
    int64_t loopH{0};
};

template <typename T>
class CircularPad : public CircularPadCommon
{
public:
    __aicore__ inline CircularPad(TPipe* pipe) : CircularPadCommon(pipe) {};
    
    __aicore__ inline void Init(const CircularPadCommonTilingData& tiling_data)
    {
        TSize_ = sizeof(T);
        InitCommon(tiling_data, TSize_, TSize_);
        hasLR = (left_ > 0 || right_ > 0);
        inOutputH_ = inputH_ + nTop_ + nBottom_;
        inOutputW_ = inputW_ + nLeft_ + nRight_;
        inOutputWAlign_ = GetAlign(inOutputW_, TSize_);
        pipe_->InitBuffer(queBind_, BUFFER_NUM, UB_SIZE / BUFFER_NUM);
    }

    /************************************小shape***************************************************/
    template <bool HASLR>
    __aicore__ inline void PadLeftAndRightSmallShape()
    {
        if constexpr (HASLR) {
            DataCopyExtParams paramsIn;
            DataCopyExtParams paramsOut;
            DataCopyExtParams paramsRight;
            
            int64_t offsetIn = -nTop_ * inputW_ - nLeft_;
            int64_t offsetOut = leftAlign_;
            int64_t offsetRight = leftAlign_ + inOutputW_;

            // 搬移输入到workspace，并且搬移右pad
            for (uint32_t i = 0; i < perCoreTaskNum_; i++) {
                CalculateLeftAndRightParams(paramsIn, paramsOut, paramsRight);
                auto inLocal = queBind_.AllocTensor<T>();
                DataCopyPad(inLocal, xGM_[offsetIn], paramsIn, padParms);
                queBind_.EnQue(inLocal);
                inLocal = queBind_.DeQue<T>();
                DataCopyPad(workspaceGM_[offsetOut], inLocal, paramsOut);

                if (right_ > 0) {
                    PipeBarrier<PIPE_MTE3>();
                    DataCopyPad(workspaceGM_[offsetRight], inLocal, paramsRight);
                    offsetRight += workspaceLen_;
                }
                queBind_.FreeTensor(inLocal);
                offsetIn += inputLen_;
                offsetOut += workspaceLen_;
            }
            MTE3ToMTE2Sync();
            // 搬移左pad
            if (left_ > 0) {
                paramsIn.blockLen = static_cast<uint32_t>(leftAlign_ * TSize_);
                paramsIn.srcStride = static_cast<uint32_t>((rightAlign_ + inOutputWAlign_) * TSize_);
                paramsOut.blockLen = static_cast<uint32_t>(leftAlign_ * TSize_);
                paramsOut.dstStride = static_cast<uint32_t>((rightAlign_ + inOutputWAlign_) * TSize_);
                offsetIn = inOutputW_;
                offsetOut = 0;

                for (uint32_t i = 0; i < perCoreTaskNum_; i++) {
                    auto inLocal = queBind_.AllocTensor<T>();
                    DataCopyPad(inLocal, workspaceGM_[offsetIn], paramsIn, padParms);
                    queBind_.EnQue(inLocal);
                    inLocal = queBind_.DeQue<T>();
                    DataCopyPad(workspaceGM_[offsetOut], inLocal, paramsOut);
                    queBind_.FreeTensor(inLocal);
                    offsetIn += workspaceLen_;
                    offsetOut += workspaceLen_;
                }
            }
        }
    }

    __aicore__ inline void CopyToOutSmallShapeOnePage(GlobalTensor<T>& srcGM, int64_t pageIdxOut,
                                                      CopyParams& copyParamsIn, CopyParams& copyParamsOut)
    {
        auto inLocal = queBind_.AllocTensor<T>();
        DataCopyPad(inLocal, srcGM[copyParamsIn.offset], copyParamsIn.dcParams, padParms);
        queBind_.EnQue(inLocal);
        inLocal = queBind_.DeQue<T>();
        DataCopyPad(yGM_[copyParamsOut.offset], inLocal, copyParamsOut.dcParams);

        if (top_ > 0) {
            copyParamsOut.dcParams.blockCount = static_cast<uint16_t>(top_);
            DataCopyPad(yGM_[pageIdxOut * outputLen_], inLocal[(inOutputH_ - top_) * outputWAlign_],
                        copyParamsOut.dcParams);
        }
        if (bottom_ > 0) {
            copyParamsOut.dcParams.blockCount = static_cast<uint16_t>(bottom_);
            DataCopyPad(yGM_[pageIdxOut * outputLen_ + (outputH_ - bottom_) * outputW_],
                        inLocal, copyParamsOut.dcParams);
        }
        queBind_.FreeTensor(inLocal);
    }

    /************************************大shape***************************************************/
    template <bool HASLR>
    __aicore__ inline void PadLeftAndRightBigShape()
    {
        if constexpr (HASLR) {
            leftAlign_ = left_ > 0 ? leftAlign_ : Align_;
            rightAlign_ = right_ > 0 ? rightAlign_ : Align_;
            leftAlign_ = leftAlign_ > inputW_ ? pLeft_ : leftAlign_;
            rightAlign_ = rightAlign_ > inputW_ ? pRight_ : rightAlign_;

            LoopParams loopParams;
            loopParams.loopW = leftAlign_;
            loopParams.loopH = inOutputH_;
            CopyParams copyParamsIn;
            CopyParams copyParamsOut;
            uint32_t blockLen = static_cast<uint32_t>(leftAlign_ * TSize_);
            uint32_t srcStrideIn = static_cast<uint32_t>((inputW_ - leftAlign_) * TSize_);
            uint32_t dstStrideOut = static_cast<uint32_t>(rightAlign_ * TSize_);
            copyParamsIn.dcParams = {0, blockLen , srcStrideIn, 0, 0};
            copyParamsOut.dcParams = {0, blockLen, 0, dstStrideOut, 0};
            copyParamsIn.offset = inputW_ - leftAlign_ + nRight_  - nTop_ * inputW_;
            copyParamsOut.offset = 0;
            copyParamsIn.strideLoop = inputW_;
            copyParamsOut.strideLoop = leftAlign_ + rightAlign_;
            copyParamsIn.stridePage = inputLen_;
            copyParamsOut.stridePage = workspaceLen_;
            CopyLines(xGM_, workspaceGM_, loopParams, copyParamsIn, copyParamsOut);

            loopParams.loopW = rightAlign_;
            blockLen = static_cast<uint32_t>(rightAlign_ * TSize_);
            srcStrideIn = static_cast<uint32_t>((inputW_ - rightAlign_) * TSize_);
            dstStrideOut = static_cast<uint32_t>(leftAlign_ * TSize_);
            copyParamsIn.dcParams = {0, blockLen , srcStrideIn, 0, 0};
            copyParamsOut.dcParams = {0, blockLen, 0, dstStrideOut, 0};
            copyParamsIn.offset = -nLeft_ - nTop_ * inputW_ ;
            copyParamsOut.offset = leftAlign_;
            CopyLines(xGM_, workspaceGM_, loopParams, copyParamsIn, copyParamsOut);
        }
    }
    
    __aicore__ inline void CopyToOutBigShapeOnePage(int64_t pageIdxIn, int64_t pageIdxOut)
    {
        LoopParams loopParams;
        CopyParams copyParamsIn;
        CopyParams copyParamsOut;
        
        // workspace  ---> y
        CopyWSToOutOnce(pageIdxIn, pageIdxOut, loopParams, copyParamsIn, copyParamsOut);
        MTE3ToMTE2Sync();

        // x  ---> y
        CopyInToOutOnce(pageIdxIn, pageIdxOut, loopParams, copyParamsIn, copyParamsOut);
        MTE3ToMTE2Sync();

        // pad top and bottom
        PadTopAndBottomOnce(pageIdxIn, pageIdxOut, loopParams, copyParamsIn, copyParamsOut);
    }

    /************************************辅助函数***************************************************/
    __aicore__ inline void CalculateLeftAndRightParams(DataCopyExtParams& paramsIn, DataCopyExtParams& paramsOut,
                                                       DataCopyExtParams& paramsRight)
    {
        uint16_t blockCount = static_cast<uint16_t>(inOutputH_);
        uint32_t blockLenIn = static_cast<uint32_t>(inOutputW_ * TSize_);
        uint32_t srcStrideIn = static_cast<uint32_t>((-nLeft_ - nRight_) * TSize_);
        uint32_t blockLenOut = static_cast<uint32_t>(inOutputWAlign_ * TSize_);
        uint32_t dstStrideOut = static_cast<uint32_t>((leftAlign_ + rightAlign_) * TSize_);
        paramsIn = {blockCount, blockLenIn , srcStrideIn, 0, 0};
        paramsOut = {blockCount, blockLenOut, 0, dstStrideOut, 0};
        uint32_t blockLen = static_cast<uint32_t>(rightAlign_ * TSize_);
        uint32_t srcStride = static_cast<uint32_t>((inOutputWAlign_ - rightAlign_) * TSize_ / BLOCK_SIZE);
        uint32_t dstStride = static_cast<uint32_t>((leftAlign_ + inOutputWAlign_) * TSize_);
        paramsRight = {blockCount, blockLen, srcStride, dstStride, 0};
    }

    __aicore__ inline void CopyLines(GlobalTensor<T>& srcGM, GlobalTensor<T>& dstGM, LoopParams& loopParams,
                                     CopyParams& copyParamsIn, CopyParams& copyParamsOut)
    {
        uint16_t rowsNum = UB_SIZE / BUFFER_NUM / GetAlign(loopParams.loopW, TSize_) / TSize_;
        uint32_t loop = loopParams.loopH / rowsNum;
        uint16_t tail = loopParams.loopH % rowsNum;
        for (uint32_t i = 0; i < perCoreTaskNum_; i++) {
            for (uint32_t j = 0; j < loop; j++) {
                copyParamsIn.dcParams.blockCount = rowsNum;
                copyParamsOut.dcParams.blockCount = rowsNum;
                auto inLocal = queBind_.AllocTensor<T>();
                DataCopyPad(inLocal, srcGM[copyParamsIn.offset], copyParamsIn.dcParams, padParms);
                queBind_.EnQue(inLocal);
                inLocal = queBind_.DeQue<T>();
                DataCopyPad(dstGM[copyParamsOut.offset], inLocal, copyParamsOut.dcParams);
                queBind_.FreeTensor(inLocal);
                copyParamsIn.offset += rowsNum * copyParamsIn.strideLoop;
                copyParamsOut.offset += rowsNum * copyParamsOut.strideLoop;
            }
            if (tail > 0) {
                copyParamsIn.dcParams.blockCount = tail;
                copyParamsOut.dcParams.blockCount = tail;
                auto inLocal = queBind_.AllocTensor<T>();
                DataCopyPad(inLocal, srcGM[copyParamsIn.offset], copyParamsIn.dcParams, padParms);
                queBind_.EnQue(inLocal);
                inLocal = queBind_.DeQue<T>();
                DataCopyPad(dstGM[copyParamsOut.offset], inLocal, copyParamsOut.dcParams);
                queBind_.FreeTensor(inLocal);
            }
            copyParamsIn.offset += (copyParamsIn.stridePage - loop * rowsNum * copyParamsIn.strideLoop);
            copyParamsOut.offset += (copyParamsOut.stridePage - loop * rowsNum * copyParamsOut.strideLoop);
        }
    }

    __aicore__ inline void CopyWSToOutOnce(int64_t pageIdxIn, int64_t pageIdxOut, LoopParams& loopParams,
                                           CopyParams& copyParamsIn, CopyParams& copyParamsOut)
    {
        leftAlign_ = left_ > 0 ? leftAlign_ : Align_;
        rightAlign_ = right_ > 0 ? rightAlign_ : Align_;
        leftAlign_ = leftAlign_ > inputW_ ? pLeft_ : leftAlign_;
        rightAlign_ = rightAlign_ > inputW_ ? pRight_ : rightAlign_;
        loopParams.loopH = inOutputH_;
        copyParamsIn.strideLoop = leftAlign_ + rightAlign_;
        copyParamsOut.strideLoop = outputW_;
        copyParamsIn.stridePage = workspaceLen_;
        copyParamsOut.stridePage = outputLen_;
        if (left_ > 0) {
            loopParams.loopW = leftAlign_;
            uint32_t blockLen = static_cast<uint32_t>(leftAlign_ * TSize_);
            uint32_t srcStrideIn = static_cast<uint32_t>(rightAlign_ * TSize_);
            uint32_t dstStrideOut = static_cast<uint32_t>((outputW_ - leftAlign_) * TSize_);
            copyParamsIn.dcParams = {0, blockLen , srcStrideIn, 0, 0};
            copyParamsOut.dcParams = {0, blockLen, 0, dstStrideOut, 0};
            copyParamsIn.offset = pageIdxIn * workspaceLen_ + leftAlign_ - left_;
            copyParamsOut.offset = pageIdxOut * outputLen_ + pTop_ * outputW_;
            CopyOnePage(workspaceGM_, yGM_, loopParams, copyParamsIn, copyParamsOut);
        }
        if (right_ > 0) {
            loopParams.loopW = rightAlign_;
            uint32_t blockLen = static_cast<uint32_t>(rightAlign_ * TSize_);
            uint32_t srcStrideIn = static_cast<uint32_t>(leftAlign_ * TSize_);
            uint32_t dstStrideOut = static_cast<uint32_t>((outputW_ - rightAlign_) * TSize_);
            copyParamsIn.dcParams = {0, blockLen , srcStrideIn, 0, 0};
            copyParamsOut.dcParams = {0, blockLen, 0, dstStrideOut, 0};
            copyParamsIn.offset = pageIdxIn * workspaceLen_ + leftAlign_ + right_ - rightAlign_;
            copyParamsOut.offset = pageIdxOut * outputLen_ + pTop_ * outputW_ + outputW_ - rightAlign_;
            CopyOnePage(workspaceGM_, yGM_, loopParams, copyParamsIn, copyParamsOut);
        }
    }

    __aicore__ inline void CopyInToOutOnce(int64_t pageIdxIn, int64_t pageIdxOut, LoopParams& loopParams,
                                           CopyParams& copyParamsIn, CopyParams& copyParamsOut)
    {
        leftAlign_ = left_ > 0 ? leftAlign_ : 0;
        rightAlign_ = right_ > 0 ? rightAlign_ : 0;
        int64_t holeW = inOutputW_ - (leftAlign_ - pLeft_) - (rightAlign_- pRight_);
        if (holeW > 0) {
            loopParams.loopW = inOutputWAlign_;
            uint32_t blockLen = static_cast<uint32_t>(holeW * TSize_);
            uint32_t srcStrideIn = static_cast<uint32_t>((inputW_ - holeW) * TSize_);
            uint32_t dstStrideOut = static_cast<uint32_t>((leftAlign_ + rightAlign_) * TSize_);
            copyParamsIn.dcParams = {0, blockLen , srcStrideIn, 0, 0};
            copyParamsOut.dcParams = {0, blockLen, 0, dstStrideOut, 0};
            copyParamsIn.offset = pageIdxIn * inputLen_ + leftAlign_ - pLeft_ - nLeft_ - nTop_ * inputW_;
            copyParamsOut.offset = pageIdxOut * outputLen_ + pTop_ * outputW_ + leftAlign_;
            copyParamsIn.strideLoop = inputW_;
            CopyOnePage(xGM_, yGM_, loopParams, copyParamsIn, copyParamsOut);
        }
    }

    __aicore__ inline void PadTopAndBottomOnce(int64_t pageIdxIn, int64_t pageIdxOut, LoopParams& loopParams,
                                               CopyParams& copyParamsIn, CopyParams& copyParamsOut)
    {
        loopParams.loopW = outputWAlign_;
        copyParamsIn.strideLoop = outputW_;
        copyParamsOut.strideLoop = outputW_;
        copyParamsIn.stridePage = outputLen_;
        copyParamsOut.stridePage = outputLen_;
        if (top_ > 0) {
            loopParams.loopH = top_;
            uint32_t blockLen = static_cast<uint32_t>(outputW_ * TSize_);
            copyParamsIn.dcParams = {0, blockLen , 0, 0, 0};
            copyParamsOut.dcParams = {0, blockLen, 0, 0, 0};
            copyParamsIn.offset = pageIdxOut * outputLen_ + (outputH_ - pBottom_ - top_) * outputW_;
            copyParamsOut.offset = pageIdxOut * outputLen_;
            CopyOnePage(yGM_, yGM_, loopParams, copyParamsIn, copyParamsOut);
        }
        if (bottom_ > 0) {
            loopParams.loopH = bottom_;
            uint32_t blockLen = static_cast<uint32_t>(outputW_ * TSize_);
            copyParamsIn.dcParams = {0, blockLen , 0, 0, 0};
            copyParamsOut.dcParams = {0, blockLen, 0, 0, 0};
            copyParamsIn.offset = pageIdxOut * outputLen_ + pTop_ * outputW_;
            copyParamsOut.offset = pageIdxOut * outputLen_ + (outputH_ - bottom_) * outputW_;
            CopyOnePage(yGM_, yGM_, loopParams, copyParamsIn, copyParamsOut);
        }
    }
    
    __aicore__ inline void CopyOnePage(GlobalTensor<T>& srcGM, GlobalTensor<T>& dstGM, LoopParams loopParams,
                                       CopyParams& copyParamsIn, CopyParams& copyParamsOut)
    {
        uint16_t rowsNum = UB_SIZE / BUFFER_NUM / GetAlign(loopParams.loopW, TSize_) / TSize_;
        uint32_t loop = loopParams.loopH / rowsNum;
        uint16_t tail = loopParams.loopH % rowsNum;
        for (uint32_t j = 0; j < loop; j++) {
            copyParamsIn.dcParams.blockCount = rowsNum;
            copyParamsOut.dcParams.blockCount = rowsNum;
            auto inLocal = queBind_.AllocTensor<T>();
            DataCopyPad(inLocal, srcGM[copyParamsIn.offset], copyParamsIn.dcParams, padParms);
            queBind_.EnQue(inLocal);
            inLocal = queBind_.DeQue<T>();
            DataCopyPad(dstGM[copyParamsOut.offset], inLocal, copyParamsOut.dcParams);
            queBind_.FreeTensor(inLocal);
            copyParamsIn.offset += rowsNum * copyParamsIn.strideLoop;
            copyParamsOut.offset += rowsNum * copyParamsOut.strideLoop;
        }
        if (tail > 0) {
            copyParamsIn.dcParams.blockCount = tail;
            copyParamsOut.dcParams.blockCount = tail;
            auto inLocal = queBind_.AllocTensor<T>();
            DataCopyPad(inLocal, srcGM[copyParamsIn.offset], copyParamsIn.dcParams, padParms);
            queBind_.EnQue(inLocal);
            inLocal = queBind_.DeQue<T>();
            DataCopyPad(dstGM[copyParamsOut.offset], inLocal, copyParamsOut.dcParams);
            queBind_.FreeTensor(inLocal);
        }
    }

    __aicore__ inline void CopyGmToGm(int64_t pages, int64_t taskNum, int64_t offsetIn,
                                      int64_t offsetOut, int64_t stride)
    {
        int64_t loop = (outputLen_ * pages * TSize_) / (UB_SIZE / BUFFER_NUM);
        uint32_t tail = (outputLen_ * pages * TSize_) % (UB_SIZE / BUFFER_NUM);
        for (int64_t i = 0; i < taskNum; i++) {
            DataCopyExtParams paramsFront = {1, UB_SIZE / BUFFER_NUM, 0, 0, 0};
            for (int64_t j = 0; j < loop; j++) {
                auto inLocal = queBind_.AllocTensor<T>();
                DataCopyPad(inLocal, yGM_[offsetIn], paramsFront, padParms);
                queBind_.EnQue(inLocal);
                inLocal = queBind_.DeQue<T>();
                DataCopyPad(yGM_[offsetOut], inLocal, paramsFront);
                queBind_.FreeTensor(inLocal);
                offsetIn += (UB_SIZE / BUFFER_NUM / TSize_);
                offsetOut += (UB_SIZE / BUFFER_NUM / TSize_);
            }
            if (tail > 0) {
                paramsFront.blockLen = tail;
                auto inLocal = queBind_.AllocTensor<T>();
                DataCopyPad(inLocal, yGM_[offsetIn], paramsFront, padParms);
                queBind_.EnQue(inLocal);
                inLocal = queBind_.DeQue<T>();
                DataCopyPad(yGM_[offsetOut], inLocal, paramsFront);
                queBind_.FreeTensor(inLocal);
            }
            offsetIn += (stride - loop * (UB_SIZE / BUFFER_NUM / TSize_));
            offsetOut += (stride - loop * (UB_SIZE / BUFFER_NUM / TSize_));
        }
    }

protected:
    bool hasLR{true};
    uint8_t TSize_ {0};
    DataCopyPadExtParams<T> padParms = {false, 0, 0, 0};
    GlobalTensor<T> xGM_;
    GlobalTensor<T> yGM_;
    GlobalTensor<T> workspaceGM_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queBind_;
};
#endif