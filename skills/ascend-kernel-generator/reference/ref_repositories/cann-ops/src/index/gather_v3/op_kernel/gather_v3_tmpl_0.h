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
 * \file gather_v3_tmpl_0.h
 * \brief
 */
#ifndef GATHER_V3_TMPL_0_H
#define GATHER_V3_TMPL_0_H

#include "gather_v3_base.h"

namespace GatherV3 {
using namespace AscendC;

template <typename T_DATA, typename T_IDX>
class GatherV3Tmpl0 : public GatherV3Base<T_DATA, T_IDX> {
public:
    __aicore__ inline GatherV3Tmpl0(const GatherV3TilingData* tilingDataPtr)
        : GatherV3Base<T_DATA, T_IDX>(tilingDataPtr){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR y, TPipe *pipe) {
        this->BaseInit(x, indices, y, pipe);

        this->pipe_->InitBuffer(dataQue_, this->bufferNum_, this->yBufferSize_);
        this->pipe_->InitBuffer(this->idxQue_, this->bufferNum_, this->idxBufferSize_);
    };

    __aicore__ inline void ProcessSingleTile10430();
    __aicore__ inline void ProcessSingleTile10330();
    __aicore__ inline void ProcessSingleTile10200();
    __aicore__ inline void ProcessSingleTile12330();
    __aicore__ inline void ProcessSingleTile11100();

protected:
    __aicore__ inline LocalTensor<T_DATA> GetTensorData() {
        return this->dataQue_.template AllocTensor<T_DATA>();
    }

    __aicore__ inline void CopyInData(int64_t offset, int64_t num, int64_t len) {
        LocalTensor<T_DATA> dataTensor = this->dataQue_.template AllocTensor<T_DATA>();
        auto dst = dataTensor.template ReinterpretCast<uint8_t>();

        if (len * sizeof(T_DATA) % BYTE_BLOCK == 0) {
            DataCopy(dst, this->xGM_[offset * sizeof(T_DATA)], num * len * sizeof(T_DATA));
        } else {
            DataCopyExtParams params;
            params.blockCount = (uint16_t)num;
            params.blockLen   = (uint32_t)len * sizeof(T_DATA);
            params.srcStride  = 0;
            params.dstStride  = 0;
            DataCopyPadExtParams<uint8_t> padParams(false, 0, 0, 0);
            DataCopyPad(dst, this->xGM_[offset * sizeof(T_DATA)], params, padParams);
        }

        this->dataQue_.EnQue(dataTensor);
    };

    __aicore__ inline void CopyInData(const LocalTensor<T_DATA> &dataTensor, int64_t offsetInGM) {     
        DataCopyExtParams params;
        params.blockCount = (uint16_t)this->pLength_;
        params.blockLen   = (uint32_t)this->aSize_ * sizeof(T_DATA);
        params.srcStride  = (uint32_t)(this->xpElemSize_ - this->aSize_) * sizeof(T_DATA);
        params.dstStride  = (uint32_t)(this->gySize_ - 1) * this->aBlockNum_;

        auto dst = dataTensor.template ReinterpretCast<uint8_t>();
        DataCopyPadExtParams<uint8_t> padParams(false, 0, 0, 0);
        DataCopyPad(dst, this->xGM_[offsetInGM * sizeof(T_DATA)], params, padParams);
    }

    __aicore__ inline void CopyInDataLine(const LocalTensor<T_DATA> &dataTensor, int64_t offsetInGM, int64_t len) {
        auto dst = dataTensor.template ReinterpretCast<uint8_t>();
        
        if (len * sizeof(T_DATA) % BYTE_BLOCK == 0) {
            DataCopy(dst, this->xGM_[offsetInGM * sizeof(T_DATA)], len * sizeof(T_DATA));
        } else {
            DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen   = (uint32_t)len * sizeof(T_DATA);
            params.srcStride  = 0;
            params.dstStride  = 0;
            DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0}; 
            DataCopyPad(dst, this->xGM_[offsetInGM * sizeof(T_DATA)], params, padParams);
        }
    };

    __aicore__ inline void CopyOutData(int64_t offset, LocalTensor<T_DATA> &dataTensor, int64_t num, int64_t len) {
        this->dataQue_.EnQue(dataTensor);
        
        LocalTensor<T_DATA> outputTensor = this->dataQue_.template DeQue<T_DATA>();
        this->DoCopyUB2GM(offset, outputTensor, num, len);
        this->dataQue_.template FreeTensor<T_DATA>(outputTensor);
    };

    __aicore__ inline void CopyOutData(int64_t offset, int64_t num, int64_t len) {
        DataCopyExtParams params;
        params.blockCount = (uint16_t)num;
        params.blockLen   = (uint32_t)len * sizeof(T_DATA);
        params.srcStride  = 0;
        params.dstStride  = 0;
        LocalTensor<T_DATA> outputTensor = this->dataQue_.template DeQue<T_DATA>();

        auto src = outputTensor.template ReinterpretCast<uint8_t>();
        DataCopyPad(this->yGM_[offset * sizeof(T_DATA)], src, params);
        this->dataQue_.template FreeTensor<T_DATA>(outputTensor);
    };

protected:
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> dataQue_;
};

// key      out     idx     core
// 10430    a       g       [b, p, g_idx, a_out]
template <typename T_DATA, typename T_IDX>
__aicore__ inline void GatherV3Tmpl0<T_DATA, T_IDX> :: ProcessSingleTile10430() {
    this->CalcBaseOffset();

    auto idxTensor = this->CopyInIdx(this->idxBaseOffset_, 1, this->gyLength_);
    this->SyncM2toS();

    for (uint32_t i = 0; i < this->gyLength_; i++) {
        auto gxId = idxTensor.GetValue(i);

        this->CheckIdxValue(gxId);

        int64_t xOffset = this->xBaseOffset_ + gxId * this->aSize_;
        int64_t yOffset = this->yBaseOffset_ + i * this->aSize_;

        CopyInData(xOffset, 1, this->aLength_);
        CopyOutData(yOffset, 1, this->aLength_);
    }

    this->FreeTensorIdx(idxTensor);
}


// key      out     idx     core
// 10330    g       g       [b, p, g_idx]
template <typename T_DATA, typename T_IDX>
__aicore__ inline void GatherV3Tmpl0<T_DATA, T_IDX> :: ProcessSingleTile10330() {
    this->CalcBaseOffset();

    auto idxTensor = this->CopyInIdx(this->idxBaseOffset_, 1, this->gyLength_);

    this->SyncM2toS();

    int64_t yOffset = this->yBaseOffset_;
    for (int64_t gLoop = 0; gLoop < this->gyLength_; gLoop += this->ubLineLimit_) {
        auto dataTensor = GetTensorData();
        int64_t offsetInUB = 0;

        int64_t lineNum = this->Min(this->ubLineLimit_, this->gyLength_ - gLoop);

        for (int64_t i = 0; i < lineNum; i++) {
            T_IDX gxId = idxTensor.GetValue(gLoop + i);
            this->CheckIdxValue(gxId);

            int64_t xOffset = this->xBaseOffset_ + gxId * this->aSize_;

            CopyInDataLine(dataTensor[offsetInUB], xOffset, this->aSize_);
            offsetInUB += this->aAlignSize_;
        }

        CopyOutData(yOffset, dataTensor, lineNum, this->aSize_);
        yOffset += lineNum * this->aSize_;
    }

    this->FreeTensorIdx(idxTensor);
}

// key      out     idx     core
// 10200    p       b       [b_out, p_out]
template <typename T_DATA, typename T_IDX>
__aicore__ inline void GatherV3Tmpl0<T_DATA, T_IDX> :: ProcessSingleTile10200() {
    this->CalcBaseOffset();

    // b轴切分后的多个g轴一次读入，g轴不必对齐。
    auto idxTensor = this->CopyInIdx(this->idxBaseOffset_, 1, this->bLength_ * this->gySize_);
    int64_t idxOffset = 0;

    this->SyncM2toS();

    int64_t xOffset = this->xBaseOffset_;
    int64_t yOffset = this->yBaseOffset_;
    for (int64_t bLoop = 0; bLoop < this->bLength_; bLoop++) {
        auto dataTensor = GetTensorData();
        int64_t offsetInUB = 0;

        // g轴循环，p轴作为repeat，在p轴切分范围内，gather所有ga轴
        for (int64_t gLoop = 0; gLoop < this->gySize_; gLoop++) {
            T_IDX gxId = idxTensor.GetValue(idxOffset++);
            this->CheckIdxValue(gxId);

            CopyInData(dataTensor[offsetInUB], xOffset + gxId * this->aSize_);
            offsetInUB += this->aAlignSize_;
        }
        xOffset += this->xbElemSize_;

        CopyOutData(yOffset, dataTensor, this->pLength_ * this->gySize_, this->aSize_);

        yOffset += this->ybElemSize_;
    }

    this->FreeTensorIdx(idxTensor);
}

// key      out     idx     core
// 12330    [p, g]  g       [b, p_out, g_out]
template <typename T_DATA, typename T_IDX>
__aicore__ inline void GatherV3Tmpl0<T_DATA, T_IDX> :: ProcessSingleTile12330() {
    this->CalcBaseOffset();

    int64_t xOffset = this->xBaseOffset_;
    int64_t yOffset = this->yBaseOffset_;
    int64_t idxOffset = this->idxBaseOffset_;

    auto idxTensor = this->CopyInIdx(idxOffset, 1, this->gyLength_);
    auto dataTensor = GetTensorData();

    this->SyncM2toS();
    for (int64_t idxLoop = 0; idxLoop < this->gyLength_; idxLoop++) {
        T_IDX gxId = idxTensor.GetValue(idxLoop);
        this->CheckIdxValue(gxId);

        // p轴repeat
        CopyInData(dataTensor[idxLoop * this->aAlignSize_], xOffset + gxId * this->aSize_);
    }

    CopyOutData(yOffset, dataTensor, this->pLength_ * this->gyLength_, this->aSize_);

    this->FreeTensorIdx(idxTensor);
}

// key      out     idx     core
// 11100    b       b       b_out
template <typename T_DATA, typename T_IDX>
__aicore__ inline void GatherV3Tmpl0<T_DATA, T_IDX> :: ProcessSingleTile11100() {
    this->CalcBaseOffset();

    int64_t xOffset = this->xBaseOffset_;
    int64_t yOffset = this->yBaseOffset_;

    // b轴切分后的多个g轴一次读入，g轴不必对齐
    auto idxTensor = this->CopyInIdx(this->idxBaseOffset_, 1, this->bLength_ * this->gySize_);
    int64_t idxOffsetInUB = 0;

    this->SyncM2toS();

    for (int64_t bLoop = 0; bLoop < this->bLength_; bLoop += this->ubLineLimit_) {
        int64_t lineNum = this->Min(this->ubLineLimit_, this->bLength_ - bLoop);

        auto dataTensor = GetTensorData();
        int64_t yOffsetInUB = 0;

        for (int64_t i = 0; i < lineNum; i++) {
            // b循环，每次重新更新y的offset。因为p轴repeat，仅累加一个aAlignSize不够
            yOffsetInUB = (bLoop + i) * this->pSize_ * this->gySize_ * this->aAlignSize_;
            for (int64_t gLoop = 0; gLoop < this->gySize_; gLoop++) {
                T_IDX gxId = idxTensor.GetValue(idxOffsetInUB++);
                this->CheckIdxValue(gxId);

                // p轴repeat
                CopyInData(dataTensor[yOffsetInUB], xOffset + gxId * this->aSize_);
                yOffsetInUB += this->aAlignSize_;
            }
            xOffset += this->xbElemSize_;
        }

        CopyOutData(yOffset, dataTensor, lineNum * this->pSize_ * this->gySize_, this->aSize_);
        yOffset += lineNum * this->ybElemSize_;
    }

    this->FreeTensorIdx(idxTensor);
}

}  // namespace GatherV3

#endif  // GATHER_V3_TMPL_0_H