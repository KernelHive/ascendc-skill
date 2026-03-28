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
 * \file gather_v3_base.h
 * \brief
 */
#ifndef GATHER_V3_BASE_H
#define GATHER_V3_BASE_H

#include "kernel_operator.h"

namespace GatherV3 {
using namespace AscendC;
constexpr int64_t BYTE_BLOCK = 32;
constexpr int64_t BYTE_FULL_BLOCK = 32 * 8;
constexpr int32_t UNROLL_NUM = 8;
constexpr int32_t VREDUCE_MASK_ALL = 7;
constexpr int32_t VREDUCEV2_SIZE = 2;
constexpr int32_t VCOPY_MAX_REPEAT = 255;
constexpr int32_t TRANSPOSE_H_2B = 256;
constexpr int32_t TRANSPOSE_W_2B = 16;
constexpr int32_t TRANSPOSE_H_1B = 512;
constexpr int32_t TRANSPOSE_W_1B = 32;
constexpr int32_t TRANSPOSE_BLOCK_NUM = 16;
constexpr int64_t KEY_TYPE_RANGE = 10000;
constexpr int64_t DB_BUF_CNT = 2;
constexpr int64_t SG_BUF_CNT = 1;

template <typename T_DATA, typename T_IDX>
class GatherV3Base {
public:
    __aicore__ inline GatherV3Base(const GatherV3TilingData* tilingDataPtr) {
        tilingKey_ = tilingDataPtr->tilingKey;
        
        // 各轴长度
        bSize_  = tilingDataPtr->bSize;
        pSize_  = tilingDataPtr->pSize;
        gxSize_ = tilingDataPtr->gxSize;
        gySize_ = tilingDataPtr->gySize;
        aSize_  = tilingDataPtr->aSize;

        // 各轴核切分值
        bTileNum_ = tilingDataPtr->bTileNum;
        pTileNum_ = tilingDataPtr->pTileNum;
        gTileNum_ = tilingDataPtr->gTileNum;
        aTileNum_ = tilingDataPtr->aTileNum;

        bTileSize_ = tilingDataPtr->bTileSize;
        pTileSize_ = tilingDataPtr->pTileSize;
        gTileSize_ = tilingDataPtr->gTileSize;
        aTileSize_ = tilingDataPtr->aTileSize;

        bTileHead_ = tilingDataPtr->bTileHead;
        pTileHead_ = tilingDataPtr->pTileHead;
        gTileHead_ = tilingDataPtr->gTileHead;
        aTileHead_ = tilingDataPtr->aTileHead;

        realCoreNum_ = tilingDataPtr->realCoreNum;
        xBufferSize_ = tilingDataPtr->xBufferSize;
        yBufferSize_ = tilingDataPtr->yBufferSize;
        idxBufferSize_ = tilingDataPtr->idxBufferSize;
        ubLineLimit_ = tilingDataPtr->ubLineLimit;
        bufferNum_ = tilingDataPtr->bufferNum;
    };

    template <typename CLS_NAME, void (CLS_NAME::*funSingleTile)()>
    __aicore__ inline void Process(CLS_NAME* objPtr) {
        int tileId = blockIdx_;

        while (tileId < tileTotalNum_) {
            this->tileIdx_ = tileId;
            CalcAxisRange(tileId);
            (objPtr->*funSingleTile)();
            tileId += realCoreNum_;
        }  
    };

protected:
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilDiv(T1 a, T2 b) {
        T1 bTemp(b);
        return bTemp == 0 ? a : (a + bTemp - 1) / bTemp;
    };

    template <typename T1, typename T2>
    __aicore__ inline T1 CeilAlign(T1 a, T2 b) {
        T1 bTemp(b);
        return bTemp == 0 ? a : CeilDiv(a, bTemp) * bTemp;
    };

    template <typename T1, typename T2>
    __aicore__ inline T1 Min(T1 a, T2 b) {
        T1 bTemp(b);
        return (a < b) ? a : b;
    }

    __aicore__ inline void SyncM2toS() {
        event_t eventId = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventId);
        WaitFlag<HardEvent::MTE2_S>(eventId);
    };

    __aicore__ inline void SyncStoM2() {
        event_t eventId = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::S_MTE2));
        SetFlag<HardEvent::S_MTE2>(eventId);
        WaitFlag<HardEvent::S_MTE2>(eventId);
    }

    __aicore__ inline void SyncStoV() {
        event_t eventId = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventId);
        WaitFlag<HardEvent::S_V>(eventId);
    };

    __aicore__ inline void SyncVtoS() {
        event_t eventId = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);
    };

    __aicore__ inline void CheckIdxValue(int64_t val) {
        assert((0 <= val && val < this->gxSize_), "Index %d out of range[%d %d)!\n", val, 0, this->gxSize_);
    };

    __aicore__ inline void BaseInit(GM_ADDR x, GM_ADDR indices, GM_ADDR y, TPipe *pipe) {
        this->pipe_ = pipe;

        SetMaskNorm();

        blockIdx_ = GetBlockIdx();

        tileTotalNum_ = bTileNum_ * pTileNum_ * gTileNum_ * aTileNum_;

        xbElemSize_ = pSize_ * gxSize_ * aSize_;
        ybElemSize_ = pSize_ * gySize_ * aSize_;

        xpElemSize_ = gxSize_ * aSize_;
        ypElemSize_ = gySize_ * aSize_;

        aAlignSize_ = CeilAlign(aSize_, (BYTE_BLOCK / sizeof(T_DATA)));
        aBlockNum_ = aAlignSize_ * sizeof(T_DATA) / BYTE_BLOCK;

        this->xGM_.SetGlobalBuffer((__gm__ uint8_t *)x);
        this->yGM_.SetGlobalBuffer((__gm__ uint8_t *)y);
        this->idxGM_.SetGlobalBuffer((__gm__ uint8_t *)indices);
    };

    __aicore__ inline void CalcAxisRange(int64_t tileId) {
        #define CALC_AXIS_OFFSET_LENGTH(m) do {\
            int64_t m##_idx = runId / m##ElemSize;\
            runId -= m##_idx * m##ElemSize;\
            if (m##_idx < m##TileHead_) {\
                m##Offset_ = m##_idx * m##TileSize_;\
                m##Length_ = m##TileSize_;\
            } else {\
                m##Offset_ = m##_idx * m##TileSize_ + m##TileHead_ - m##_idx;\
                m##Length_ = m##TileSize_ - 1;\
            }\
        } while(0)

        // i j k l
        int64_t aElemSize = 1;
        int64_t gElemSize = aTileNum_ * aElemSize;
        int64_t pElemSize = gTileNum_ * gElemSize;
        int64_t bElemSize = pTileNum_ * pElemSize; 

        int64_t runId = tileId;

        CALC_AXIS_OFFSET_LENGTH(b);
        CALC_AXIS_OFFSET_LENGTH(p);
        CALC_AXIS_OFFSET_LENGTH(g);
        CALC_AXIS_OFFSET_LENGTH(a);

        #undef CALC_AXIS_OFFSET_LENGTH
    };

    __aicore__ inline void PrefetchIdx(int64_t offset, int64_t num, int64_t len) {
        LocalTensor<T_IDX> idxInTensor = this->idxQue_.template AllocTensor<T_IDX>();
        DataCopyExtParams params;
        params.blockCount = (uint16_t)num;
        params.blockLen   = (uint32_t)len * sizeof(T_IDX);
        params.srcStride  = 0;
        params.dstStride  = 0;
        DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};

        auto dst = idxInTensor.template ReinterpretCast<uint8_t>();
        DataCopyPad(dst, this->idxGM_[offset * sizeof(T_IDX)], params, padParams);
        this->idxQue_.EnQue(idxInTensor);
    };

    __aicore__ inline LocalTensor<T_IDX> GetIdx() {
        return this->idxQue_.template DeQue<T_IDX>();
    };

    __aicore__ inline LocalTensor<T_IDX> CopyInIdx(int64_t offset, int64_t num, int64_t len) {
        PrefetchIdx(offset, num, len);

        return GetIdx();
    };

    __aicore__ inline void FreeTensorIdx(LocalTensor<T_IDX> &idxTensor) {
        return this->idxQue_.template FreeTensor<T_IDX>(idxTensor);
    }

    __aicore__ inline void DoCopyUB2GM(int64_t offset, const LocalTensor<T_DATA> &data, int64_t num, int64_t len){
        auto src = data.template ReinterpretCast<uint8_t>();

        if (len * sizeof(T_DATA) % BYTE_BLOCK == 0) {
            DataCopy(this->yGM_[offset * sizeof(T_DATA)], src, num * len * sizeof(T_DATA));
        } else {
            DataCopyExtParams params;
            params.blockCount = (uint16_t)num;
            params.blockLen   = (uint32_t)len * sizeof(T_DATA);
            params.srcStride  = 0;
            params.dstStride  = 0;
            DataCopyPad(this->yGM_[offset * sizeof(T_DATA)], src, params);
        }
    }

    __aicore__ inline void CalcBaseOffset() {
        this->gyOffset_ = this->gOffset_;
        this->gyLength_ = this->gLength_;

        this->xBaseOffset_ = this->bOffset_ * this->xbElemSize_ + 
                             this->pOffset_ * this->xpElemSize_ + 
                             this->aOffset_;

        this->yBaseOffset_ = this->bOffset_ * this->ybElemSize_ +
                             this->pOffset_ * this->ypElemSize_ +
                             this->gyOffset_ * this->aSize_ +
                             this->aOffset_;
        this->idxBaseOffset_ = this->bOffset_ * this->gySize_ + this->gyOffset_;
    };

protected:
    TPipe *pipe_;

    TQue<QuePosition::VECIN, 1>     idxQue_;

    GlobalTensor<uint8_t>           xGM_, yGM_;
    GlobalTensor<uint8_t>           idxGM_;

    TQueSync<PIPE_MTE2, PIPE_S>     syncM2toS_;
    TQueSync<PIPE_S, PIPE_MTE2>     syncStoM2_;
    TQueSync<PIPE_S, PIPE_V>        syncStoV_;
    TQueSync<PIPE_V, PIPE_S>        syncVtoS_;

    int64_t tilingKey_;
    int64_t blockIdx_;
    int64_t tileTotalNum_;
    int64_t realCoreNum_;
    int64_t bufferNum_;

    // 各轴元素个数
    int64_t bSize_;
    int64_t pSize_;
    int64_t gxSize_;
    int64_t gySize_;     
    int64_t aSize_;

    int64_t aAlignSize_;
    int64_t aBlockNum_;

    // 各轴核切分块数
    int64_t bTileNum_;
    int64_t pTileNum_;
    int64_t gTileNum_;
    int64_t aTileNum_;

    int64_t bTileSize_;
    int64_t pTileSize_;
    int64_t gTileSize_;
    int64_t aTileSize_;

    int64_t bTileHead_;
    int64_t pTileHead_;
    int64_t gTileHead_;
    int64_t aTileHead_;

    int64_t tileIdx_;

    // 各轴切分range
    int64_t bOffset_;
    int64_t bLength_;
  
    int64_t pOffset_;
    int64_t pLength_;

    int64_t gOffset_;
    int64_t gLength_;

    int64_t gxOffset_;
    int64_t gxLength_;

    int64_t gyOffset_;
    int64_t gyLength_;

    int64_t aOffset_;
    int64_t aLength_;

    int64_t xBaseOffset_;
    int64_t yBaseOffset_;
    int64_t idxBaseOffset_;

    // 切分同一个轴时，ub容量限制内循环的大小
    int64_t ubLineLimit_; 

    int64_t xBufferSize_;
    int64_t yBufferSize_;
    int64_t idxBufferSize_;

    // 内轴size聚合
    int64_t xbElemSize_;
    int64_t ybElemSize_;

    int64_t xpElemSize_;
    int64_t ypElemSize_;
};

}  // namespace GatherV3

#endif  // GATHER_V3_BASE_H