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
 * \file circular_pad_grad.h
 * \brief
 */
#ifndef CIRCULAR_PAD_GRAD_H
#define CIRCULAR_PAD_GRAD_H
#include "circular_pad_common.h"
using namespace AscendC;

constexpr int32_t DOUBLE = 2;
constexpr int32_t TRIPLE = 3;

template <typename T1, typename T2, bool ISCAST = false>
class CircularPadGrad : public CircularPadCommon
{
public:
    __aicore__ inline CircularPadGrad(TPipe* pipe) : CircularPadCommon(pipe) {};
    
    __aicore__ inline void Init(const CircularPadCommonTilingData& tiling_data)
    {
        T1Size_ = sizeof(T1);
        T2Size_ = sizeof(T2);
        InitCommon(tiling_data, T1Size_, T2Size_);
        
        topAndBottom_ = pTop_ + pBottom_;
        inOutputH_ = inputH_ - pTop_ - pBottom_;
        inOutputW_ = inputW_ - pLeft_ - pRight_;
        inOutputWAlign_ = GetAlign(inOutputW_, T1Size_);
        inOutputW32Align_ = GetAlign(inOutputW_, T2Size_);

        offsetH_ = -nTop_; 
        offsetW_ = -nLeft_;
        strideW_ = -(nLeft_ + nRight_);
        if constexpr (ISCAST) {
            T1UBSize_ = UB_SIZE / BUFFER_NUM / TRIPLE;
            T2UBSize_ = UB_SIZE / TRIPLE;
            pipe_->InitBuffer(queBindT1_, BUFFER_NUM, T1UBSize_);
            pipe_->InitBuffer(queBindT2_, BUFFER_NUM, T2UBSize_);
        } else {
            T1UBSize_ = UB_SIZE / BUFFER_NUM;
            T2UBSize_ = UB_SIZE / BUFFER_NUM;
            pipe_->InitBuffer(queBindT2_, BUFFER_NUM, T1UBSize_);
        }
    }

    /************************************小shape***************************************************/
    __aicore__ inline void AddTopAndBottomSmallShape()
    {
        sDataCopyExtParams params;
        CalculateTopAndBottomParms(params);
        if constexpr (ISCAST) {
            AddTopAndBottomSmallShapeCast(params);
        } else {
            AddTopAndBottomSmallShapeNoCast(params);
        }
    }

    __aicore__ inline void AddLeftAndRightSmallShape()
    {
        if (left_ > 0 || right_ > 0) {
            sDataCopyExtParams paramsLeft;
            sDataCopyExtParams paramsRight;
            CalculateLeftAndRightParms(paramsLeft, paramsRight);
            int64_t lInOffset = Align_;
            int64_t lOutOffset = Align_ + inOutputW_;
            int64_t rInOffset = Align_ + inputW_ - rightAlign_;
            int64_t rOutOffset = Align_ + pLeft_ + pRight_ - rightAlign_;

            CopyParams inCopyParamsl(lInOffset, 0, paramsLeft.paramsIn);
            CopyParams outCopyParamsl(lOutOffset, 0, paramsLeft.paramsOut);
            CopyParams inCopyParamsr(rInOffset, 0, paramsRight.paramsIn);
            CopyParams outCopyParamsr(rOutOffset, 0, paramsRight.paramsOut);

            for (int64_t i = 0; i < perCoreTaskNum_; i++) {
                AddLeftAndRightOnce(inCopyParamsl, outCopyParamsl, inCopyParamsr, outCopyParamsr);
                inCopyParamsl.offset += workspaceLen_;
                outCopyParamsl.offset += workspaceLen_;
                inCopyParamsr.offset += workspaceLen_;
                outCopyParamsr.offset += workspaceLen_;
            }
        }
    }

    __aicore__ inline void CopyToOutSmallShapeOnePage(int64_t inPageIdx, int64_t outPageIdx, 
                                                      sDataCopyExtParams& params)
    {
        int64_t inOffset = inPageIdx * workspaceLen_ + Align_ + pLeft_;
        int64_t outOffset = outPageIdx * outputLen_ + offsetH_ * outputW_ + offsetW_;
        CopyParams inCopyParams(inOffset, 0, params.paramsIn);
        CopyParams outCopyParams(outOffset, 0, params.paramsOut);
        if constexpr (ISCAST) {
            CopyGMToGMOnceCastFromFP32(workspaceGM_, yGM_, inCopyParams, outCopyParams);
        } else {
            CopyGMToGMOnceNoCast(workspaceGM_, yGM_, inCopyParams, outCopyParams);
        }
    }

    /************************************大shape**********************************************/
    __aicore__ inline void AddTopAndBottomBigShape()
    {
        sDataCopyExtParams params;
        CalculateTopAndBottomParms(params);
        CopyParams inCopyParams(pTop_ * inputW_, inputW_, params.paramsIn);
        CopyParams outCopyParams(Align_, inputWAlign_ + DOUBLE * Align_, params.paramsOut);
        
        CopyGMToGMLines(inOutputH_, inputWAlign_, inCopyParams, outCopyParams);
        MTE3ToMTE2Sync();

        // copy and add top
        if (top_ > 0) {
            inCopyParams.offset = 0;
            outCopyParams.offset = (inOutputH_ - top_) * (inputWAlign_ + DOUBLE * Align_) + Align_;
            SetAtomicAdd<T2>();
            CopyGMToGMLines(top_, inputWAlign_, inCopyParams, outCopyParams);
            SetAtomicNone();
        }
        
        MTE3ToMTE2Sync();
        if (bottom_ > 0) {
            inCopyParams.offset = (inputH_ - bottom_) * inputW_;
            outCopyParams.offset = Align_;
            SetAtomicAdd<T2>();
            CopyGMToGMLines(bottom_, inputWAlign_, inCopyParams, outCopyParams);
            SetAtomicNone();
        }
    }
    __aicore__ inline void AddLeftAndRightBigShape()
    {
        if (left_ > 0 || right_ > 0) {
            sDataCopyExtParams paramsLeft;
            sDataCopyExtParams paramsRight;
            CalculateLeftAndRightParms(paramsLeft, paramsRight);
            AddLeftAndRightLines(inOutputH_, paramsLeft, paramsRight);
        }
    }

    __aicore__ inline void CopyToOutBigShapeOnePage(int64_t inPageIdx, int64_t outPageIdx, 
                                                             sDataCopyExtParams& params)
    {
        int64_t inOffset = inPageIdx * workspaceLen_ + Align_ + pLeft_;
        int64_t outOffset = outPageIdx * outputLen_ + offsetH_ * outputW_ + offsetW_;
        CopyParams inCopyParams(inOffset, (inputWAlign_ + DOUBLE * Align_), params.paramsIn);
        CopyParams outCopyParams(outOffset, outputW_, params.paramsOut);
        CopyGMToGMLinesOnce(inOutputH_, inOutputWAlign_, inCopyParams, outCopyParams);
    }

    /************************************辅助函数**********************************************/
    __aicore__ inline void SetGMtoZero(int64_t len) {
        int64_t loop = len / (T1UBSize_ / T1Size_);
        int64_t tail = len % (T1UBSize_ / T1Size_);
        int64_t zeroNum = loop > 0 ? (T1UBSize_/ T1Size_) : tail;
        auto zeroLocal = queBindT2_.AllocTensor<T1>();
        Duplicate<T1>(zeroLocal, 0, zeroNum);
        uint32_t blockLen = static_cast<uint32_t>(T1UBSize_);
        DataCopyExtParams copyParams = {1, blockLen, 0, 0, 0};
        for(int i = 0; i < loop; i++){
            DataCopyPad(yGM_[i * zeroNum], zeroLocal, copyParams);
        }
        if(tail > 0){
            copyParams.blockLen = static_cast<uint32_t>(tail * T1Size_);
            DataCopyPad(yGM_[loop * zeroNum], zeroLocal, copyParams);
        }
        queBindT2_.FreeTensor(zeroLocal);
    }

    __aicore__ inline void AddTopAndBottomSmallShapeNoCast(sDataCopyExtParams& params)
    {
        for (int64_t i = 0; i < perCoreTaskNum_; i++) {
            auto inLocal = queBindT2_.AllocTensor<T2>();
            DataCopyPad(inLocal, xGM_[i * inputLen_], params.paramsIn, padParmsT1);
            queBindT2_.EnQue(inLocal);
            inLocal = queBindT2_.DeQue<T2>();
            params.paramsOut.blockCount = static_cast<uint16_t>(inOutputH_);
            DataCopyPad(workspaceGM_[i * workspaceLen_ + Align_], inLocal[pTop_ * inputWAlign_], params.paramsOut);
            PipeBarrier<PIPE_MTE3>();

            SetAtomicAdd<T2>();
            if (top_ > 0) {
                params.paramsOut.blockCount = static_cast<uint16_t>(top_);
                int64_t offset = i * workspaceLen_ + (inOutputH_ - top_) * (inputWAlign_ + DOUBLE * Align_) + Align_;
                DataCopyPad(workspaceGM_[offset], inLocal, params.paramsOut);
                PipeBarrier<PIPE_MTE3>();
            }
            if (bottom_ > 0) {
                params.paramsOut.blockCount = static_cast<uint16_t>(bottom_);
                DataCopyPad(workspaceGM_[i * workspaceLen_ + Align_], inLocal[(inputH_ - bottom_) * inputWAlign_],
                            params.paramsOut);
            }
            SetAtomicNone();
            queBindT2_.FreeTensor(inLocal);
        }
    }
    __aicore__ inline void AddTopAndBottomSmallShapeCast(sDataCopyExtParams& params)
    {
        auto blockCount = params.paramsOut.blockCount;
        for (int64_t i = 0; i < perCoreTaskNum_; i++) {
            // 输入搬入UB
            auto inLocal = queBindT1_.AllocTensor<T1>();
            DataCopyPad(inLocal, xGM_[i * inputLen_], params.paramsIn, padParmsT1);
            queBindT1_.EnQue<QuePosition::GM, QuePosition::VECIN, T1>(inLocal);

            // fp16转fp32
            inLocal = queBindT1_.DeQue<QuePosition::GM, QuePosition::VECIN, T1>();
            auto inLocalT2 = queBindT2_.AllocTensor<T2>();
            Cast(inLocalT2, inLocal, RoundMode::CAST_NONE, inputH_ * inputWAlign_);
            queBindT2_.EnQue<QuePosition::VECOUT, QuePosition::GM, T2>(inLocalT2);

            inLocalT2 = queBindT2_.DeQue<QuePosition::VECOUT, QuePosition::GM, T2>();
            DataCopyPad(workspaceGM_[i * workspaceLen_ + Align_], inLocalT2[pTop_ * inputWAlign_], params.paramsOut);
            PipeBarrier<PIPE_MTE3>();

            SetAtomicAdd<T2>();
            if (top_ > 0) {
                params.paramsOut.blockCount = static_cast<uint16_t>(top_);
                int64_t offset = i * workspaceLen_ + (inOutputH_ - top_) * (inputWAlign_ + DOUBLE * Align_) + Align_;
                DataCopyPad(workspaceGM_[offset], inLocalT2, params.paramsOut);
                PipeBarrier<PIPE_MTE3>();
            }
            if (bottom_ > 0) {
                params.paramsOut.blockCount = static_cast<uint16_t>(bottom_);
                DataCopyPad(workspaceGM_[i * workspaceLen_ + Align_], inLocalT2[(inputH_ - bottom_) * inputWAlign_],
                            params.paramsOut);
            }
            SetAtomicNone();
            queBindT1_.FreeTensor(inLocal);
            queBindT2_.FreeTensor(inLocalT2);
            params.paramsOut.blockCount = blockCount;
        }
    }

    __aicore__ inline void CopyGMToGMOnceNoCast(GlobalTensor<T1> srcGM, GlobalTensor<T1> dstGM,
                                                CopyParams& inCopyParams, CopyParams& outCopyParams)
    {
        auto inLocal = queBindT2_.AllocTensor<T1>();
        DataCopyPad(inLocal, srcGM[inCopyParams.offset], inCopyParams.dcParams, padParmsT1);
        queBindT2_.EnQue(inLocal);

        inLocal = queBindT2_.DeQue<T1>();
        DataCopyPad(dstGM[outCopyParams.offset], inLocal, outCopyParams.dcParams);
        queBindT2_.FreeTensor(inLocal);
    }

    __aicore__ inline void CopyGMToGMOnceCastToFP32(GlobalTensor<T1> srcGM, GlobalTensor<T2> dstGM,
                                                    CopyParams& inCopyParams, CopyParams& outCopyParams)
    {
        auto inLocal = queBindT1_.AllocTensor<T1>();
        DataCopyPad(inLocal, srcGM[inCopyParams.offset], inCopyParams.dcParams, padParmsT1);
        queBindT1_.EnQue<QuePosition::GM, QuePosition::VECIN, T1>(inLocal);

        // fp16转fp32
        inLocal = queBindT1_.DeQue<QuePosition::GM, QuePosition::VECIN, T1>();
        auto inLocalT2 = queBindT2_.AllocTensor<T2>();
        Cast(inLocalT2, inLocal, RoundMode::CAST_NONE, inCopyParams.dcParams.blockCount * inputWAlign_);
        queBindT2_.EnQue<QuePosition::VECOUT, QuePosition::GM, T2>(inLocalT2);
        
        inLocalT2 = queBindT2_.DeQue<QuePosition::VECOUT, QuePosition::GM, T2>();
        DataCopyPad(dstGM[outCopyParams.offset], inLocalT2, outCopyParams.dcParams);
        queBindT1_.FreeTensor(inLocal);
        queBindT2_.FreeTensor(inLocalT2);
    }

    __aicore__ inline void CopyGMToGMOnceCastFromFP32(GlobalTensor<T2> srcGM, GlobalTensor<T1> dstGM,
                                                      CopyParams& inCopyParams, CopyParams& outCopyParams)
    {
        auto inLocalT2 = queBindT2_.AllocTensor<T2>();
        DataCopyPad(inLocalT2, srcGM[inCopyParams.offset], inCopyParams.dcParams, padParmsT2);
        queBindT2_.EnQue<QuePosition::GM, QuePosition::VECIN, T2>(inLocalT2);

        inLocalT2 = queBindT2_.DeQue<QuePosition::GM, QuePosition::VECIN, T2>();
        auto inLocal = queBindT1_.AllocTensor<T1>();
        Cast(inLocal, inLocalT2, RoundMode::CAST_ROUND, 
             inCopyParams.dcParams.blockCount *
             (GetAlign((inCopyParams.dcParams.blockLen) / T2Size_, T2Size_) +
              inCopyParams.dcParams.dstStride * Align_));
        queBindT1_.EnQue<QuePosition::VECOUT, QuePosition::GM, T1>(inLocal);

        inLocal = queBindT1_.DeQue<QuePosition::VECOUT, QuePosition::GM, T1>();
        DataCopyPad(dstGM[outCopyParams.offset], inLocal, outCopyParams.dcParams);
        queBindT2_.FreeTensor(inLocalT2);
        queBindT1_.FreeTensor(inLocal);
    }

    __aicore__ inline void CalculateTopAndBottomParms(sDataCopyExtParams& params)
    {
        uint16_t inputHU16 = static_cast<uint16_t>(inputH_);
        uint32_t inputWSizeU32 = static_cast<uint32_t>(inputW_ * T1Size_);
        uint16_t inOutputHU16 = static_cast<uint16_t>(inOutputH_);
        uint32_t inputWAlignSizeU32 = static_cast<uint32_t>(inputWAlign_ * T2Size_);
        
        params.paramsIn = {inputHU16, inputWSizeU32, 0, 0, 0};
        params.paramsOut = {inOutputHU16, inputWAlignSizeU32, 0, DOUBLE * BLOCK_SIZE, 0};
    }

    __aicore__ inline void CalculateLeftAndRightParms(sDataCopyExtParams& paramsLeft, sDataCopyExtParams& paramsRight)
    {
        uint16_t inOutputHU16 = static_cast<uint16_t>(inOutputH_);
        uint32_t leftAlignSizeU32 = static_cast<uint32_t>(leftAlign_ * T2Size_);
        uint32_t rightAlignSizeU32 = static_cast<uint32_t>(rightAlign_ * T2Size_);
        uint32_t inputWAlignSizeU32 = static_cast<uint32_t>(inputWAlign_ * T2Size_);
        uint32_t wLeftStridU32 = inputWAlignSizeU32 - leftAlignSizeU32 + DOUBLE * BLOCK_SIZE;
        uint32_t wRightStridU32 = inputWAlignSizeU32 - rightAlignSizeU32 + DOUBLE * BLOCK_SIZE;
        
        paramsLeft.paramsIn = {inOutputHU16, leftAlignSizeU32, wLeftStridU32, 0, 0};
        paramsLeft.paramsOut = {inOutputHU16, leftAlignSizeU32, 0, wLeftStridU32, 0};
        paramsRight.paramsIn = {inOutputHU16, rightAlignSizeU32, wRightStridU32, 0, 0};
        paramsRight.paramsOut = {inOutputHU16, rightAlignSizeU32, 0, wRightStridU32, 0};
    }
    
    __aicore__ inline void CopyGMToGMLines(int64_t lines, int64_t lineW,
                                           CopyParams& inCopyParams, CopyParams& outCopyParams)
    {
        if (lineW == 0 || T1Size_ == 0) {
            return;
        }
        uint16_t rowsNum = T1UBSize_ / GetAlign(lineW, T1Size_) / T1Size_;
        uint32_t loop = lines / rowsNum;
        uint16_t tail = lines % rowsNum;
        for (int64_t i = 0; i < perCoreTaskNum_; i++) {
            for (uint32_t j = 0; j < loop; j++) {
                inCopyParams.dcParams.blockCount = rowsNum;
                outCopyParams.dcParams.blockCount = rowsNum;
                if constexpr (ISCAST) {
                    CopyGMToGMOnceCastToFP32(xGM_, workspaceGM_, inCopyParams, outCopyParams);
                } else {
                    CopyGMToGMOnceNoCast(xGM_, workspaceGM_, inCopyParams, outCopyParams);
                }
                inCopyParams.offset += rowsNum * inCopyParams.strideLoop;
                outCopyParams.offset += rowsNum * outCopyParams.strideLoop;
            }
            if (tail > 0) {
                inCopyParams.dcParams.blockCount = tail;
                outCopyParams.dcParams.blockCount = tail;
                if constexpr (ISCAST) {
                    CopyGMToGMOnceCastToFP32(xGM_, workspaceGM_, inCopyParams, outCopyParams);
                } else {
                    CopyGMToGMOnceNoCast(xGM_, workspaceGM_, inCopyParams, outCopyParams);
                }
            }
            inCopyParams.offset += (inputLen_ - loop * rowsNum * inCopyParams.strideLoop);
            outCopyParams.offset += (workspaceLen_ - loop * rowsNum * outCopyParams.strideLoop);
        }
    }

    __aicore__ inline void CopyGMToGMLinesOnce(int64_t lines, int64_t lineW,
                                               CopyParams& inCopyParams, CopyParams& outCopyParams)
    {
        if (lineW == 0 || T1Size_ == 0) {
            return;
        }
        uint16_t rowsNum = T1UBSize_ / GetAlign(lineW, T1Size_) / T1Size_;
        uint32_t loop = lines / rowsNum;
        uint16_t tail = lines % rowsNum;
        for (uint32_t j = 0; j < loop; j++) {
            inCopyParams.dcParams.blockCount = rowsNum;
            outCopyParams.dcParams.blockCount = rowsNum;
            if constexpr (ISCAST) {
                CopyGMToGMOnceCastFromFP32(workspaceGM_, yGM_, inCopyParams, outCopyParams);
            } else {
                CopyGMToGMOnceNoCast(workspaceGM_, yGM_, inCopyParams, outCopyParams);
            }
            inCopyParams.offset += rowsNum * inCopyParams.strideLoop;
            outCopyParams.offset += rowsNum * outCopyParams.strideLoop;
        }
        if (tail > 0) {
            inCopyParams.dcParams.blockCount = tail;
            outCopyParams.dcParams.blockCount = tail;
            if constexpr (ISCAST) {
                CopyGMToGMOnceCastFromFP32(workspaceGM_, yGM_, inCopyParams, outCopyParams);
            } else {
                CopyGMToGMOnceNoCast(workspaceGM_, yGM_, inCopyParams, outCopyParams);
            }
        }
    }

    __aicore__ inline void AddLeftAndRightOnce(CopyParams& inCopyParamsl, CopyParams& outCopyParamsl,
                                               CopyParams& inCopyParamsr, CopyParams& outCopyParamsr)
    {
        auto wLRLocal = queBindT2_.AllocTensor<T2>();
        if (right_ > 0) {
            DataCopyPad(wLRLocal, workspaceGM_[inCopyParamsr.offset], inCopyParamsr.dcParams, padParmsT2);
        }
        if (left_ > 0) {
            DataCopyPad(wLRLocal[inCopyParamsr.dcParams.blockCount * rightAlign_], workspaceGM_[inCopyParamsl.offset],
                        inCopyParamsl.dcParams, padParmsT2);
        }
        queBindT2_.EnQue(wLRLocal);
        wLRLocal = queBindT2_.DeQue<T2>();

        SetAtomicAdd<T2>();
        if (left_ > 0) {
            DataCopyPad(workspaceGM_[outCopyParamsl.offset], wLRLocal[inCopyParamsr.dcParams.blockCount * rightAlign_],
                        outCopyParamsl.dcParams);
            PipeBarrier<PIPE_MTE3>();
        }
        if (right_ > 0) {
            DataCopyPad(workspaceGM_[outCopyParamsr.offset], wLRLocal, outCopyParamsr.dcParams);
        }
        SetAtomicNone();
        queBindT2_.FreeTensor(wLRLocal);
    }

    __aicore__ inline void AddLeftAndRightLines(int64_t lines, sDataCopyExtParams& paramsLeft,
                                                sDataCopyExtParams& paramsRight)
    {
        int64_t lInOffset = Align_;
        int64_t lOutOffset = Align_ + inOutputW_;
        int64_t rInOffset = Align_ + inputW_ - rightAlign_;
        int64_t rOutOffset = Align_ + pLeft_ + pRight_ - rightAlign_;
        CopyParams inCopyParamsl(lInOffset, 0, paramsLeft.paramsIn);
        CopyParams outCopyParamsl(lOutOffset, 0, paramsLeft.paramsOut);
        CopyParams inCopyParamsr(rInOffset, 0, paramsRight.paramsIn);
        CopyParams outCopyParamsr(rOutOffset, 0, paramsRight.paramsOut);

        uint16_t rowsNum = T2UBSize_ / (leftAlign_ + rightAlign_) / T2Size_;
        uint32_t loop = lines / rowsNum;
        uint16_t tail = lines % rowsNum;
        for (int64_t i = 0; i < perCoreTaskNum_; i++) {
            for (uint32_t j = 0; j < loop; j++) {
                inCopyParamsl.dcParams.blockCount = rowsNum;
                outCopyParamsl.dcParams.blockCount = rowsNum;
                inCopyParamsr.dcParams.blockCount = rowsNum;
                outCopyParamsr.dcParams.blockCount = rowsNum;
                AddLeftAndRightOnce(inCopyParamsl, outCopyParamsl, inCopyParamsr, outCopyParamsr);
                inCopyParamsl.offset += rowsNum * (inputWAlign_ + DOUBLE * Align_);
                outCopyParamsl.offset += rowsNum * (inputWAlign_ + DOUBLE * Align_);
                inCopyParamsr.offset += rowsNum * (inputWAlign_ + DOUBLE * Align_);
                outCopyParamsr.offset += rowsNum * (inputWAlign_ + DOUBLE * Align_);
            }
            if (tail > 0) {
                inCopyParamsl.dcParams.blockCount = tail;
                outCopyParamsl.dcParams.blockCount = tail;
                inCopyParamsr.dcParams.blockCount = tail;
                outCopyParamsr.dcParams.blockCount = tail;
                AddLeftAndRightOnce(inCopyParamsl, outCopyParamsl, inCopyParamsr, outCopyParamsr);
            }
            inCopyParamsl.offset += (workspaceLen_ - loop * rowsNum * (inputWAlign_ + DOUBLE * Align_));
            outCopyParamsl.offset += (workspaceLen_ - loop * rowsNum * (inputWAlign_ + DOUBLE * Align_));
            inCopyParamsr.offset += (workspaceLen_ - loop * rowsNum * (inputWAlign_ + DOUBLE * Align_));
            outCopyParamsr.offset += (workspaceLen_ - loop * rowsNum * (inputWAlign_ + DOUBLE * Align_));
        }
    }

    __aicore__ inline void CopyGmToGm(int64_t pages, int64_t taskNum, int64_t offsetIn,
                                      int64_t offsetOut, int64_t stride)
    {
        int64_t loop = (workspaceLen_ * pages * T2Size_) / T2UBSize_;
        uint32_t tail = (workspaceLen_ * pages * T2Size_) % T2UBSize_;
        for (int64_t i = 0; i < taskNum; i++) {
            DataCopyExtParams paramsFront = {1, T2UBSize_, 0, 0, 0};
            for (int64_t j = 0; j < loop; j++) {
                auto inLocal = queBindT2_.AllocTensor<T2>();
                DataCopyPad(inLocal, workspaceGM_[offsetIn], paramsFront, padParmsT2);
                queBindT2_.EnQue(inLocal);
                inLocal = queBindT2_.DeQue<T2>();
                DataCopyPad(workspaceGM_[offsetOut], inLocal, paramsFront);
                queBindT2_.FreeTensor(inLocal);
                offsetIn += (T2UBSize_ / T2Size_);
                offsetOut += (T2UBSize_ / T2Size_);
            }
            if (tail > 0) {
                paramsFront.blockLen = tail;
                auto inLocal = queBindT2_.AllocTensor<T2>();
                DataCopyPad(inLocal, workspaceGM_[offsetIn], paramsFront, padParmsT2);
                queBindT2_.EnQue(inLocal);
                inLocal = queBindT2_.DeQue<T2>();
                DataCopyPad(workspaceGM_[offsetOut], inLocal, paramsFront);
                queBindT2_.FreeTensor(inLocal);
            }
            offsetIn += (stride - loop * (T2UBSize_ / T2Size_));
            offsetOut += (stride - loop * (T2UBSize_ / T2Size_));
        }
    }

    __aicore__ inline void CalculateOutParms(sDataCopyExtParams& params)
    {
        uint16_t blockCount = static_cast<uint16_t>(inOutputH_);
        uint32_t blockLenIn = static_cast<uint32_t>(inOutputW_ * T2Size_);
        uint32_t blockLenOut = static_cast<uint32_t>(inOutputW_ * T1Size_);
        uint32_t inoutStridU32 = static_cast<uint32_t>((inputWAlign_ - inOutputW_) * T2Size_ + DOUBLE * BLOCK_SIZE);
        uint32_t dstStride = static_cast<uint32_t>((inOutputWAlign_ - inOutputW32Align_) * T2Size_ / BLOCK_SIZE);
        uint32_t outputWSizeU32 = static_cast<uint32_t>(strideW_ * T1Size_);
        
        params.paramsIn = {blockCount, blockLenIn, inoutStridU32, dstStride, 0};
        params.paramsOut = {blockCount, blockLenOut, 0, outputWSizeU32, 0};
    }

    // 通过D、front和back计算outGm大的页偏移
    __aicore__ inline int64_t getNewBatchOffset(int64_t pageIdx, int64_t D, int64_t front, int64_t back)
    {
        int64_t outPageIdx = pageIdx;
        int64_t front_jump = 0;
        if (front > 0) {
            front_jump = front;
        }
        int64_t back_jump = 0;
        if (back > 0) {
            back_jump = back;
        }
        // 是front的部分，需要加回到后面
        if (pageIdx < front_jump) {
            outPageIdx = D - front_jump - back_jump - front_jump + pageIdx;
        }

        // 是back的部分，需要加回到前面
        if (pageIdx >= D - back_jump) {
            outPageIdx = pageIdx - (D - back_jump);
        }

        // 大间的普通页
        if (front_jump <= pageIdx && pageIdx < D - back_jump) {
            outPageIdx = pageIdx - front_jump;
        }

        // 若front小于0，则每一页需往后偏移front页
        if (front < 0) {
            outPageIdx -= front;
        }
        return outPageIdx;
    }

protected:
    uint8_t T1Size_{0};
    uint8_t T2Size_{0};
    uint32_t T1UBSize_{0};
    uint32_t T2UBSize_{0};
    int64_t topAndBottom_{0};
    int64_t offsetH_{0};
    int64_t offsetW_{0};
    int64_t strideW_{0};
    DataCopyPadExtParams<T1> padParmsT1 = {false, 0, 0, 0};
    DataCopyPadExtParams<T2> padParmsT2 = {false, 0, 0, 0};

    GlobalTensor<T1> xGM_;
    GlobalTensor<T1> yGM_;
    GlobalTensor<T2> workspaceGM_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queBindT1_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queBindT2_;
};
#endif