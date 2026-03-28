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
 * \file gather_v3_tmpl_2.h
 * \brief
 */
#ifndef GATHER_V3_TMPL_2_H
#define GATHER_V3_TMPL_2_H

#include "gather_v3_base.h"

namespace GatherV3 {
using namespace AscendC;

constexpr int64_t KEY_ONE_IN_N_BASE = 60000;
constexpr int64_t KEY_VGATHER_BASE = 70000;
constexpr int64_t FIX_PATTTERN_TYPE_A_LEN = 2;
constexpr int64_t FIX_PATTTERN_TYPE_B_LEN = 4;
constexpr int64_t KEY_VGATHER_BG_SP1 = 70131;
constexpr int64_t KEY_VGATHER_SCALAR = 70001;
constexpr int64_t REDUCEV2_MASK_UNIT = 64;
constexpr int64_t REDUCEV2_PATT_BASE_2 = 1;
constexpr int64_t REDUCEV2_PATT_BASE_4 = 3;

template <typename T_DATA, typename T_IDX>
class GatherV3Tmpl2 : public GatherV3Base<T_DATA, T_IDX> {
public:
    __aicore__ inline GatherV3Tmpl2(const GatherV3TilingData* tilingDataPtr)
        : GatherV3Base<T_DATA, T_IDX>(tilingDataPtr){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR y, TPipe *pipe) {
        this->BaseInit(x, indices, y, pipe);

        if (this->tilingKey_ == KEY_VGATHER_BG_SP1 || this->tilingKey_ == KEY_VGATHER_SCALAR) {
            this->pipe_->InitBuffer(xQue_, 1, this->xBufferSize_);
        } else {
            this->pipe_->InitBuffer(xQue_, this->bufferNum_, this->xBufferSize_);
        }
        this->pipe_->InitBuffer(yQue_, this->bufferNum_, this->yBufferSize_);
        this->pipe_->InitBuffer(this->idxQue_, this->bufferNum_, this->idxBufferSize_);
        
        if (this->tilingKey_ >= KEY_ONE_IN_N_BASE && this->tilingKey_ < KEY_ONE_IN_N_BASE + KEY_TYPE_RANGE) {
            if (this->gxSize_ != FIX_PATTTERN_TYPE_A_LEN && this->gxSize_ != FIX_PATTTERN_TYPE_B_LEN) {
                this->pipe_->InitBuffer(maskBuf_, BYTE_BLOCK);
            }
        }
    };

    __aicore__ inline void ProcessSingleTileInputLoop();
    __aicore__ inline void ProcessSingleTileOutputLoop();

    template <bool userDefPattern = true>
    __aicore__ inline void ProcessSingleTileOneInN();

    __aicore__ inline void ProcessSingleTile60020();
    __aicore__ inline void ProcessSingleTile60010();

    __aicore__ inline void ProcessSingleTile70330();
    __aicore__ inline void ProcessSingleTile70320();
    __aicore__ inline void ProcessSingleTile70310();
    __aicore__ inline void ProcessSingleTile70230();
    __aicore__ inline void ProcessSingleTile70220();
    __aicore__ inline void ProcessSingleTile70210();
    __aicore__ inline void ProcessSingleTile70130();
    __aicore__ inline void ProcessSingleTile70120();
    __aicore__ inline void ProcessSingleTile70110();
    __aicore__ inline void ProcessSingleTile70131();
    __aicore__ inline void ProcessSingleTile70001();

protected:
    __aicore__ inline LocalTensor<T_IDX> CopyInIdxPadZero(int64_t offset, int64_t num, int64_t len, T_IDX padValue) {
        LocalTensor<T_IDX> idxInTensor = this->idxQue_.template AllocTensor<T_IDX>();
        DataCopyExtParams params;
        params.blockCount = (uint16_t)num;
        params.blockLen   = (uint32_t)len * sizeof(T_IDX);
        params.srcStride  = 0;
        params.dstStride  = 0;

        // 补齐idx的元素个数，保证gather之后的"数据"是对齐到block的，方便搬出
        int64_t rightPadding = BYTE_BLOCK  - (len * sizeof(T_IDX)) % BYTE_BLOCK;

        DataCopyPadExtParams<uint8_t> padParams{true, 0, (uint8_t)rightPadding, (uint8_t)padValue};

        auto dst = idxInTensor.template ReinterpretCast<uint8_t>();
        DataCopyPad(dst, this->idxGM_[offset * sizeof(T_IDX)], params, padParams);
        this->idxQue_.EnQue(idxInTensor);

        return this->idxQue_.template DeQue<T_IDX>();
    };

    __aicore__ inline LocalTensor<T_DATA> CopyInXData(int64_t offset, int64_t num, int64_t len) {
        LocalTensor<T_DATA> xInTensor = this->xQue_.template AllocTensor<T_DATA>();
        DataCopyExtParams params;
        params.blockCount = (uint16_t)num;
        params.blockLen   = (uint32_t)len * sizeof(T_DATA);
        params.srcStride  = 0;
        params.dstStride  = 0;
        DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0}; 

        auto dst = xInTensor.template ReinterpretCast<uint8_t>();
        DataCopyPad(dst, this->xGM_[offset * sizeof(T_DATA)], params, padParams);

        this->xQue_.EnQue(xInTensor);
        return this->xQue_.template DeQue<T_DATA>();
    };

    __aicore__ inline void FreeTensorX(LocalTensor<T_DATA> &xTensor) {
        this->xQue_.template FreeTensor<T_DATA>(xTensor);
    };

    __aicore__ inline LocalTensor<T_DATA> GetTensorY() {
        return this->yQue_.template AllocTensor<T_DATA>();
    };

    __aicore__ inline LocalTensor<uint64_t> GetTensorMask() {
        return this->maskBuf_.template Get<uint64_t>();
    }

    template <bool userDefPattern>
    __aicore__ inline void CalcRepeatInfo(int64_t &lineNum, int64_t &lineLen, int64_t &repeatNum){
        if constexpr (userDefPattern) {
            // 自定义mask时，需要src1RepeatStride配置为0反复使用，p轴内需要做repeat，g轴间对齐。
            lineNum = this->bLength_ * this->pLength_;
            lineLen = this->gxSize_;
            repeatNum = this->pLength_;
        } else {
            // 2/4选1时，直接用参数配置模式，不必设置mask，p轴内不必做repeat，保持连续。
            lineNum = this->bLength_;
            lineLen = this->pLength_ * this->gxSize_;
            repeatNum = 1;
        }
    }

    __aicore__ inline void SetTensorMask(LocalTensor<uint64_t> &mask, int64_t gxId) {
        if (gxId >= REDUCEV2_MASK_UNIT) {
            mask.SetValue(0, 0);
            mask.SetValue(1, 1ULL << (gxId - REDUCEV2_MASK_UNIT));
        } else {
            mask.SetValue(0, 1ULL << gxId);
            mask.SetValue(1, 0);
        }
    }

    __aicore__ inline uint8_t GetPattern(int64_t gxId) {
        uint8_t pattern;
        if (this->gxSize_ == FIX_PATTTERN_TYPE_A_LEN) {
            pattern = gxId + REDUCEV2_PATT_BASE_2;
        } else {
            pattern = gxId + REDUCEV2_PATT_BASE_4;
        }
        return pattern;
    }

    __aicore__ inline void CopyOutYData(int64_t offset, LocalTensor<T_DATA> &yTensor, int64_t num, int64_t len) {
        this->yQue_.EnQue(yTensor);

        LocalTensor<T_DATA> yOutTensor = this->yQue_.template DeQue<T_DATA>();
        this->DoCopyUB2GM(offset, yOutTensor, num, len);
        this->yQue_.template FreeTensor(yOutTensor);
    };

    __aicore__ inline void CheckIdx(LocalTensor<T_IDX> idxTensor, LocalTensor<T_DATA> yTensor, int64_t idxNum) {
        auto checkBuffer = yTensor.template ReinterpretCast<float>();

        Cast(checkBuffer, idxTensor, RoundMode::CAST_CEIL, idxNum);
        PipeBarrier<PIPE_V>();
        ReduceMax(checkBuffer, checkBuffer, checkBuffer, idxNum);
        this->SyncVtoS();
        int64_t maxValue = (int64_t)checkBuffer.GetValue(0);
        this->CheckIdxValue(maxValue);
        PipeBarrier<PIPE_V>();
        Cast(checkBuffer, idxTensor, RoundMode::CAST_CEIL, idxNum);
        PipeBarrier<PIPE_V>();
        ReduceMin(checkBuffer, checkBuffer, checkBuffer, idxNum);
        this->SyncVtoS();
        int64_t minValue = (int64_t)checkBuffer.GetValue(0);
        this->CheckIdxValue(minValue);
    };

    __aicore__ inline void CastIdx(LocalTensor<T_IDX> idxTensor, LocalTensor<T_DATA> yTensor, int64_t len) {
        if constexpr (sizeof(T_IDX) == sizeof(int64_t)) {
            auto tBuffer = yTensor.template ReinterpretCast<int32_t>();
            auto idxBuffer = idxTensor.template ReinterpretCast<int32_t>();

            Cast(tBuffer, idxTensor, RoundMode::CAST_NONE, len);
            PipeBarrier<PIPE_V>();
            Adds(idxBuffer, tBuffer, 0, len);
        }
    };

    __aicore__ inline void BroadCastIdxPostProc(bool needReduce, LocalTensor<int32_t> &idxBuffer, LocalTensor<int32_t> &tBuffer, 
                                                int64_t lineNum, int64_t idxLen) {
        int64_t lenUnit = this->CeilAlign(idxLen, BYTE_BLOCK / sizeof(T_DATA)); 
        if (needReduce) {
            GatherMaskParams params;
            params.src0BlockStride = 1;
            params.repeatTimes = lineNum;
            params.src0RepeatStride = lenUnit * sizeof(int32_t) / BYTE_BLOCK;
            params.src1RepeatStride = 0;

            uint64_t totalCnt = 0;
            GatherMask(idxBuffer, tBuffer, VREDUCE_MASK_ALL, true, idxLen, params, totalCnt);  
        } else {
            SetMaskCount();
            SetVectorMask<int32_t, MaskMode::COUNTER>(lineNum * lenUnit);
            Copy<int32_t, false>(idxBuffer, tBuffer, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
            SetMaskNorm();
            ResetMask();
        }
    };

    __aicore__ inline void BroadCastIdx(LocalTensor<T_IDX> &idxTensor, LocalTensor<T_DATA> &yTensor, 
                                        int64_t bLength, int64_t pLength, int64_t idxLen, int64_t returnLen, 
                                        bool lineReturn = true, bool needReduce = true) {
        auto tBuffer = yTensor.template ReinterpretCast<int32_t>();
        auto idxBuffer = idxTensor.template ReinterpretCast<int32_t>();

        int64_t lenUnit = this->CeilAlign(idxLen, BYTE_BLOCK / sizeof(T_DATA)); 
        int64_t offset = 0;
        int64_t idxOffset = 0;
        int64_t step = 0;
        int64_t lineNum = 0;

        CheckIdx(idxTensor, yTensor, bLength * this->CeilAlign(idxLen, BYTE_BLOCK / sizeof(T_IDX)));
        for (int64_t bLoop = 0; bLoop < bLength; bLoop++) {
            PipeBarrier<PIPE_V>();

            if constexpr (sizeof(T_IDX) == sizeof(int64_t)) {
                Duplicate(tBuffer[offset], 0, lenUnit);
                PipeBarrier<PIPE_V>();
                Cast(tBuffer[offset], idxTensor[idxOffset], RoundMode::CAST_NONE, idxLen);
                PipeBarrier<PIPE_V>();
                Adds(tBuffer[offset], tBuffer[offset], (int32_t)step, lenUnit);
            } else {
                Duplicate(tBuffer[offset], 0, lenUnit);
                PipeBarrier<PIPE_V>();
                Adds(tBuffer[offset], idxTensor[idxOffset], (int32_t)step, idxLen);
            }
            PipeBarrier<PIPE_V>();

            int64_t baseOffset = offset;
            offset += lenUnit;
            idxOffset += this->CeilAlign(idxLen, BYTE_BLOCK / sizeof(T_IDX));

            step += this->gxSize_;
            lineNum++;
            if (lineReturn && lineNum >= returnLen) {
                step = 0;
                lineNum = 0;
            }

            for (int64_t pLoop = 1; pLoop < pLength; pLoop++) {
                Adds(tBuffer[offset], tBuffer[baseOffset], (int32_t)step, lenUnit);
                offset += lenUnit;

                step += this->gxSize_;
                lineNum++;
                if (lineReturn && lineNum >= returnLen) {
                    step = 0;
                    lineNum = 0;
                }
            }
        }
        PipeBarrier<PIPE_V>();

        BroadCastIdxPostProc(needReduce, idxBuffer, tBuffer, bLength * pLength, idxLen);
    };

protected:
    TQue<QuePosition::VECIN, 1> xQue_;
    TQue<QuePosition::VECOUT, 1> yQue_;
    TBuf<QuePosition::VECCALC> maskBuf_;
};

// 70320 g p
// 70310 g b
// 70220 p p
// 70210 p b
// 70110 b b
template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl2<T_DATA, T_IDX>::ProcessSingleTileInputLoop() {
    this->CalcBaseOffset();
    int64_t xOffset = this->xBaseOffset_;
    int64_t yOffset = this->yBaseOffset_;
    int64_t idxOffset = this->idxBaseOffset_;

    // 读入索引
    auto idxTensor = this->CopyInIdxPadZero(idxOffset, this->bLength_, this->gySize_, 0);
    auto yTensor = GetTensorY();
    int64_t yOffsetInUB = 0;
    int64_t idxNumNorm = this->CeilAlign(this->gySize_, BYTE_BLOCK / sizeof(T_DATA)); //以gather后数据的大小为准

    // 申请y，在y中做校验和broadcast，reduce回到idx中
    BroadCastIdx(idxTensor, yTensor, this->bLength_, this->pLength_, this->gySize_, this->ubLineLimit_, true, false);
    PipeBarrier<PIPE_V>();
    // 循环读入x，做gather
    auto idxTotal = idxTensor.template ReinterpretCast<uint32_t>();
    int64_t idxOffsetInUB = 0;

    // 1/4UB长度在合法范围内
    Muls(idxTotal.template ReinterpretCast<int32_t>(), idxTotal.template ReinterpretCast<int32_t>(), 
        (int32_t)sizeof(T_DATA), this->bLength_ * this->pLength_ * idxNumNorm);
    PipeBarrier<PIPE_V>();
    
    int64_t totalLen = this->bLength_ * this->pLength_;
    for (int64_t loop = 0; loop < totalLen; loop += this->ubLineLimit_) {
        int64_t lineNum = this->Min(this->ubLineLimit_, totalLen - loop);

        // 非对齐读入
        auto xTensor = CopyInXData(xOffset, 1, lineNum * this->gxSize_);
        xOffset += lineNum * this->gxSize_;

        Gather(yTensor[yOffsetInUB], xTensor, idxTotal[idxOffsetInUB], 0, lineNum * idxNumNorm);
        yOffsetInUB += lineNum * idxNumNorm;
        idxOffsetInUB += lineNum * idxNumNorm;

        FreeTensorX(xTensor);
    }

    CopyOutYData(yOffset, yTensor, this->bLength_ * this->pLength_, this->gySize_);
    this->FreeTensorIdx(idxTensor);
}

// 70330 g g
// 70230 p g
// 70130 b g

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl2<T_DATA, T_IDX>::ProcessSingleTileOutputLoop() {
    this->CalcBaseOffset();
    int64_t xOffset = this->xBaseOffset_;
    int64_t yOffset = this->yBaseOffset_;
    int64_t idxOffset = this->idxBaseOffset_;
    
    auto xTensor = CopyInXData(xOffset, 1, this->bLength_ * this->pLength_ * this->gxSize_);
    int64_t xOffsetInUB = 0;

    for (int64_t bLoop = 0; bLoop < this->bLength_; bLoop++) {
        for (int64_t gLoop = 0; gLoop < this->gySize_; gLoop += this->ubLineLimit_) {
            int64_t idxNum = this->Min(this->ubLineLimit_, this->gySize_ - gLoop);
            auto idxOriTensor = this->CopyInIdx(idxOffset, 1, idxNum);
            auto idxTensor = idxOriTensor.template ReinterpretCast<uint32_t>(); 
            idxOffset += idxNum;
            bool checked = false;

            for (int64_t pLoop = 0; pLoop < this->pLength_; pLoop++) {
                auto yTensor = GetTensorY();
                if (!checked) {
                    CheckIdx(idxOriTensor, yTensor, idxNum);
                    PipeBarrier<PIPE_V>();
                    CastIdx(idxOriTensor, yTensor, idxNum);
                    PipeBarrier<PIPE_V>();
                    Muls(idxTensor.template ReinterpretCast<int32_t>(), idxTensor.template ReinterpretCast<int32_t>(), 
                        (int32_t)sizeof(T_DATA), (int32_t)idxNum);
                    PipeBarrier<PIPE_V>();
                    checked = true;
                }

                int64_t baseAddr = (xOffsetInUB + pLoop * this->gxSize_) * sizeof(T_DATA);
                Gather(yTensor, xTensor, idxTensor, (uint32_t)baseAddr, idxNum);

                CopyOutYData(yOffset + pLoop * this->gySize_ + gLoop, yTensor, 1, idxNum);
            }

            this->FreeTensorIdx(idxOriTensor);
        }

        xOffsetInUB += this->xbElemSize_;
        yOffset += this->ybElemSize_;
    }

    FreeTensorX(xTensor);

    return;
}

template <typename T_DATA, typename T_IDX>
template <bool userDefPattern>
inline __aicore__ void GatherV3Tmpl2<T_DATA, T_IDX>::ProcessSingleTileOneInN() {
    this->CalcBaseOffset();

    auto yTensor = GetTensorY();
    int64_t yOffsetInUB = 0;

    int64_t lineNum = 0;
    int64_t lineLen = 0;
    int64_t repeatNum = 0;
    CalcRepeatInfo<userDefPattern>(lineNum, lineLen, repeatNum);

    auto xTensor = CopyInXData(this->xBaseOffset_, lineNum, lineLen);
    int64_t xOffsetInUB = 0;

    auto idxTensor = this->CopyInIdx(this->idxBaseOffset_, 1, this->bLength_);

    int64_t pxLineSize = repeatNum * this->CeilAlign(lineLen, BYTE_BLOCK / sizeof(T_DATA));
    int64_t pyLineSize = this->CeilAlign(this->pLength_, BYTE_BLOCK / sizeof(T_DATA));

    GatherMaskParams params;
    params.src0BlockStride = 1;
    params.repeatTimes = repeatNum;
    params.src0RepeatStride = this->CeilDiv(this->gxSize_, BYTE_BLOCK / sizeof(T_DATA));;
    params.src1RepeatStride = 0;

    this->SyncM2toS();

    for (int64_t bLoop = 0; bLoop < this->bLength_; bLoop++) {
        uint64_t count = 0;

        int64_t gxId = idxTensor.GetValue(bLoop);
        this->CheckIdxValue(gxId);

        if constexpr (userDefPattern) {
            auto mask = GetTensorMask();
            SetTensorMask(mask, gxId);

            this->SyncStoV();

            if constexpr (sizeof(T_DATA) == sizeof(uint32_t)) {
                GatherMask(yTensor[yOffsetInUB], xTensor[xOffsetInUB], 
                                   mask.template ReinterpretCast<uint32_t>(), true, lineLen, params, count);
            } else {
                GatherMask(yTensor[yOffsetInUB], xTensor[xOffsetInUB], 
                                   mask.template ReinterpretCast<uint16_t>(), true, lineLen, params, count);
            }
        } else {
            uint8_t pattern = GetPattern(gxId);
            GatherMask(yTensor[yOffsetInUB], xTensor[xOffsetInUB], pattern, true, lineLen, params, count);
        }

        xOffsetInUB += pxLineSize;
        yOffsetInUB += pyLineSize;
    }

    this->FreeTensorIdx(idxTensor);
    FreeTensorX(xTensor);
    CopyOutYData(this->yBaseOffset_, yTensor, this->bLength_, this->pLength_);
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl2<T_DATA, T_IDX>::ProcessSingleTile60020() {
    if (this->gxSize_ == 2 || this->gxSize_ == 4) {
        ProcessSingleTileOneInN<false>();
    } else {
        ProcessSingleTileOneInN<true>();
    }
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl2<T_DATA, T_IDX>::ProcessSingleTile60010() {
    if (this->gxSize_ == 2 || this->gxSize_ == 4) {
        ProcessSingleTileOneInN<false>();
    } else {
        ProcessSingleTileOneInN<true>();
    }
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl2<T_DATA, T_IDX>::ProcessSingleTile70330() {
    ProcessSingleTileOutputLoop();
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl2<T_DATA, T_IDX>::ProcessSingleTile70320() {
    ProcessSingleTileInputLoop();
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl2<T_DATA, T_IDX>::ProcessSingleTile70310() {
    ProcessSingleTileInputLoop();
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl2<T_DATA, T_IDX>::ProcessSingleTile70230() {
    ProcessSingleTileOutputLoop();
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl2<T_DATA, T_IDX>::ProcessSingleTile70220() {
    ProcessSingleTileInputLoop();
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl2<T_DATA, T_IDX>::ProcessSingleTile70210() {
    ProcessSingleTileInputLoop();
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl2<T_DATA, T_IDX>::ProcessSingleTile70130() {
    ProcessSingleTileOutputLoop();
}

// 70120 b p
template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl2<T_DATA, T_IDX>::ProcessSingleTile70120() {
    this->CalcBaseOffset();
    int64_t xOffset = this->xBaseOffset_;
    int64_t yOffset = this->yBaseOffset_;
    int64_t idxOffset = this->idxBaseOffset_;
    
    auto xTensor = CopyInXData(xOffset, 1, this->bLength_ * this->pLength_ * this->gxSize_);
    int64_t xOffsetInUB = 0;

    for (int64_t bLoop = 0; bLoop < this->bLength_; bLoop++) {
        auto idxTensor = this->CopyInIdx(idxOffset, 1, this->gySize_);
        idxOffset += this->gySize_;
        auto yTensor = GetTensorY();

        // 申请y，在y中做校验和broadcast，reduce回到idx中
        BroadCastIdx(idxTensor, yTensor, 1, this->ubLineLimit_, this->gySize_, 0, false);
        PipeBarrier<PIPE_V>();

        bool yExist = true;
        auto idxTotal = idxTensor.template ReinterpretCast<uint32_t>();
        Muls(idxTotal.template ReinterpretCast<int32_t>(), idxTotal.template ReinterpretCast<int32_t>(), 
            (int32_t)sizeof(T_DATA), this->ubLineLimit_ * this->gySize_);
        PipeBarrier<PIPE_V>();

        for (int64_t pLoop = 0; pLoop < this->pLength_; pLoop += this->ubLineLimit_) {
            int64_t lineNum = this->Min(this->ubLineLimit_, this->pLength_ - pLoop);
            
            if (!yExist) {
                yTensor = GetTensorY();
            }

            int64_t baseAddr = (xOffsetInUB + pLoop * this->gxSize_) * sizeof(T_DATA);
            int64_t idxNum = lineNum * this->gySize_;
            Gather(yTensor, xTensor, idxTotal, (uint32_t)baseAddr, idxNum);

            CopyOutYData(yOffset, yTensor, 1, idxNum);
            yOffset += idxNum;
            yExist = false;
        }
        xOffsetInUB += this->xbElemSize_;

        this->FreeTensorIdx(idxTensor);
    }

    FreeTensorX(xTensor);
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl2<T_DATA, T_IDX>::ProcessSingleTile70110() {
    ProcessSingleTileInputLoop();
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl2<T_DATA, T_IDX>::ProcessSingleTile70001() {
    this->CalcBaseOffset();

    int64_t idxOffset = this->idxBaseOffset_;

    auto yTensor = GetTensorY();

    for (int64_t gLoop = 0; gLoop < this->gyLength_; gLoop += this->ubLineLimit_) {
        int64_t idxNum = this->Min(this->ubLineLimit_, this->gyLength_ - gLoop);
        auto idxOriTensor = this->CopyInIdx(idxOffset, 1, idxNum);
        idxOffset += idxNum;

        CheckIdx(idxOriTensor, yTensor, idxNum);
        PipeBarrier<PIPE_V>();

        this->FreeTensorIdx(idxOriTensor);
    }

    if (this->tileIdx_ == 0) {
        auto xTensor = CopyInXData(0, 1, 1);
        this->SyncVtoS();
        this->SyncM2toS();
        yTensor.SetValue(0, xTensor.GetValue(0)); // 检查所有索引后，将x[0]直接赋给y[0]
        CopyOutYData(0, yTensor, 1, 1);
        FreeTensorX(xTensor);
    } else {
        this->yQue_.template FreeTensor(yTensor);
    }

    return;
}

template <typename T_DATA, typename T_IDX>
inline __aicore__ void GatherV3Tmpl2<T_DATA, T_IDX>::ProcessSingleTile70131() {
    this->CalcBaseOffset();

    int64_t xOffset = this->xBaseOffset_;
    int64_t yOffset = this->yBaseOffset_;
    int64_t idxOffset = this->idxBaseOffset_;

    auto xTensor = CopyInXData(xOffset, 1, this->gxSize_ * this->aSize_);

    for (int64_t gLoop = 0; gLoop < this->gyLength_; gLoop += this->ubLineLimit_) {
        int64_t idxNum = this->Min(this->ubLineLimit_, this->gyLength_ - gLoop);
        auto idxOriTensor = this->CopyInIdx(idxOffset, 1, idxNum);
        auto idxTensor = idxOriTensor.template ReinterpretCast<uint32_t>(); 
        idxOffset += idxNum;

        auto yTensor = GetTensorY();
        CheckIdx(idxOriTensor, yTensor, idxNum);
        PipeBarrier<PIPE_V>();
        CastIdx(idxOriTensor, yTensor, idxNum);
        PipeBarrier<PIPE_V>();
        Muls(idxTensor.template ReinterpretCast<int32_t>(), idxTensor.template ReinterpretCast<int32_t>(), 
            (int32_t)sizeof(T_DATA), (int32_t)idxNum);
        PipeBarrier<PIPE_V>();

        Gather(yTensor, xTensor, idxTensor, 0, idxNum);

        CopyOutYData(yOffset, yTensor, 1, idxNum);
        yOffset += idxNum;

        this->FreeTensorIdx(idxOriTensor);
    }

    FreeTensorX(xTensor);

    return;
}

}  // namespace GatherV3

#endif  // GATHER_V3_TMPL_0_H