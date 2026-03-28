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
 * \file gather_v3_tmpl_3.h
 * \brief
 */
#ifndef GATHER_V3_TMPL_3_H
#define GATHER_V3_TMPL_3_H

#include "gather_v3_base.h"

namespace GatherV3 {
using namespace AscendC;

template <typename T_DATA, typename T_IDX>
class GatherV3Tmpl3 : public GatherV3Base<T_DATA, T_IDX> {
public:
    __aicore__ inline GatherV3Tmpl3(const GatherV3TilingData* tilingDataPtr)
        : GatherV3Base<T_DATA, T_IDX>(tilingDataPtr){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR y, TPipe *pipe) {
        this->BaseInit(x, indices, y, pipe);

        this->pipe_->InitBuffer(xQue_, 1, this->xBufferSize_);
        this->pipe_->InitBuffer(yQue_, 1, this->yBufferSize_);
        this->pipe_->InitBuffer(this->idxQue_, 1, this->idxBufferSize_);
    };

    __aicore__ inline void ProcessSingleTile80330();
    __aicore__ inline void ProcessSingleTile80220();

protected:
    __aicore__ inline void CopyInLine(LocalTensor<T_DATA> xTensor, int64_t offsetInUB, int64_t offsetInGM) {
        DataCopyExtParams params;
        params.blockCount = (uint16_t)this->pLength_;
        params.blockLen   = (uint32_t)this->aSize_ * sizeof(T_DATA);
        params.srcStride  = (uint32_t)(this->xpElemSize_ - this->aSize_) * sizeof(T_DATA);
        params.dstStride  = (uint32_t)(this->gySize_ - 1) * this->aBlockNum_;
        DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};

        auto dst = xTensor.template ReinterpretCast<uint8_t>();
        DataCopyPad(dst[offsetInUB * sizeof(T_DATA)], this->xGM_[offsetInGM * sizeof(T_DATA)], params, padParams);
    };

    __aicore__ inline void CopyIn(int64_t offsetInGM, LocalTensor<T_IDX> idxTensor, int64_t idxNum) {
        auto xTensor = this->xQue_.template AllocTensor<T_DATA>();

        int64_t loopNum = idxNum / UNROLL_NUM;
        int64_t tailLen = idxNum % UNROLL_NUM;
        int64_t idxPos  = 0;
        int64_t offsetInUB = 0;
        T_IDX gxIdArray[UNROLL_NUM];
        for (int64_t loop = 0; loop < loopNum; loop++) {
            for (int32_t idxLoop = 0; idxLoop < UNROLL_NUM; idxLoop++) {
                gxIdArray[idxLoop] = idxTensor.GetValue(idxPos++);
                this->CheckIdxValue(gxIdArray[idxLoop]);
            }

            this->SyncStoM2();

            for (int32_t idxLoop = 0; idxLoop < UNROLL_NUM; idxLoop++) {
                CopyInLine(xTensor, offsetInUB + idxLoop * this->aAlignSize_, offsetInGM + gxIdArray[idxLoop] * this->aSize_);
            }

            offsetInUB += UNROLL_NUM * this->aAlignSize_;
        }

        if (tailLen > 0) {
            for (int32_t idxLoop = 0; idxLoop < tailLen; idxLoop++) {
                gxIdArray[idxLoop] = idxTensor.GetValue(idxPos++);
                this->CheckIdxValue(gxIdArray[idxLoop]);
            }

            this->SyncStoM2();

            for (int32_t idxLoop = 0; idxLoop < tailLen; idxLoop++) {
                CopyInLine(xTensor, offsetInUB + idxLoop * this->aAlignSize_, offsetInGM + gxIdArray[idxLoop] * this->aSize_);
            }
        }

        this->xQue_.template EnQue(xTensor);
    };

    __aicore__ inline void DeAlignInUB(LocalTensor<T_DATA> dst, LocalTensor<T_DATA> src, int64_t elemNum, int64_t repeatNum) {
        auto dstTensor = dst.template ReinterpretCast<uint16_t>();
        auto srcTensor = src.template ReinterpretCast<uint16_t>();
        int64_t mask = elemNum * sizeof(T_DATA) / sizeof(uint16_t);
        uint64_t totalCnt = 0;

        GatherMask(dstTensor, srcTensor, VREDUCE_MASK_ALL, true, mask, {1, (uint16_t)repeatNum, (uint16_t)this->aBlockNum_, 0}, totalCnt);   
    };

    __aicore__ inline void DeAlignInUB4OddByte(LocalTensor<T_DATA> dst, LocalTensor<T_DATA> src, int64_t elemNum, int64_t repeatNum) {
        auto dst1B = dst.template ReinterpretCast<int8_t>();
        auto src1B = src.template ReinterpretCast<int8_t>();
        auto dst2B = dst.template ReinterpretCast<half>();
        auto src2B = src.template ReinterpretCast<half>();

        int64_t mask = elemNum;
        uint64_t totalCnt = 0;

        int64_t repeatStride2B = this->CeilDiv(mask * VREDUCEV2_SIZE, BYTE_BLOCK);
        int64_t repeatStride1B = this->aBlockNum_;

        // 上转换到float16, 应用vreducev2后，再下转换回int8_t。int8_t和float16之间无精度损失。
        if (mask > BYTE_FULL_BLOCK / VREDUCEV2_SIZE) {
            // 尾轴长时，连带pad一同做连续cast。1B<->2B的cast每个repeat128个元素，超过128，repeat就不能应用于g轴上。
            int64_t normCnt = this->aBlockNum_ * BYTE_BLOCK / sizeof(int8_t) * repeatNum;
            Cast<half, int8_t>(dst2B, src1B, RoundMode::CAST_NONE, normCnt);
            // 这时，vreducev2的stride是2倍
            repeatStride2B = this->aBlockNum_ * VREDUCEV2_SIZE;
        } else {
            // 尾轴短时，使用高维切分接口, repeat应用于g轴
            int64_t loopCnt = repeatNum / VCOPY_MAX_REPEAT;
            int64_t tailLen = repeatNum % VCOPY_MAX_REPEAT;
            int64_t dstPos = 0;
            int64_t srcPos = 0;
            int64_t dstStep = repeatStride2B * VCOPY_MAX_REPEAT * BYTE_BLOCK / sizeof(half);
            int64_t srcStep = repeatStride1B * VCOPY_MAX_REPEAT * BYTE_BLOCK / sizeof(int8_t);

            for (int64_t loop = 0; loop < loopCnt; loop++) {
                Cast<half, int8_t>(dst2B[dstPos], src1B[srcPos], RoundMode::CAST_NONE, 
                                mask, (uint8_t)VCOPY_MAX_REPEAT, {1, 1, (uint8_t)repeatStride2B, (uint8_t)repeatStride1B});
                dstPos += dstStep;
                srcPos += srcStep;
            }

            if (tailLen > 0) {
                Cast<half, int8_t>(dst2B[dstPos], src1B[srcPos], RoundMode::CAST_NONE, 
                                mask, (uint8_t)tailLen, {1, 1, (uint8_t)repeatStride2B, (uint8_t)repeatStride1B});
            }
        }

        PipeBarrier<PIPE_V>();
        GatherMask(src2B, dst2B, VREDUCE_MASK_ALL, true, mask, {1, (uint16_t)repeatNum, (uint16_t)repeatStride2B, 0}, totalCnt);   
        PipeBarrier<PIPE_V>();

        Cast<int8_t, half>(dst1B, src2B, RoundMode::CAST_NONE, totalCnt);
    };

    __aicore__ inline void Compute(int64_t lineNum) {
        LocalTensor<T_DATA> xTensor = this->xQue_.template DeQue<T_DATA>();
        LocalTensor<T_DATA> yTensor = this->yQue_.template AllocTensor<T_DATA>();

        if constexpr (sizeof(T_DATA) % VREDUCEV2_SIZE == 0) {
            DeAlignInUB(yTensor, xTensor, this->aSize_, lineNum);
        } else {
            if (this->aSize_ % VREDUCEV2_SIZE == 0) {
                DeAlignInUB(yTensor, xTensor, this->aSize_, lineNum);
            } else {
                DeAlignInUB4OddByte(yTensor, xTensor, this->aSize_, lineNum);
            }
        }

        this->yQue_.template EnQue(yTensor);
        this->xQue_.template FreeTensor(xTensor);
    };

    __aicore__ inline void CopyOut(int64_t offsetInGM, int64_t len) {
        LocalTensor<T_DATA> yTensor = this->yQue_.template DeQue<T_DATA>();
        this->DoCopyUB2GM(offsetInGM, yTensor, 1, len);
        this->yQue_.template FreeTensor(yTensor);
    };

protected:
    TQue<QuePosition::VECIN, 1>     xQue_;
    TQue<QuePosition::VECOUT, 1>    yQue_;
};

template <typename T_DATA, typename T_IDX>
__aicore__ inline void GatherV3Tmpl3<T_DATA, T_IDX> :: ProcessSingleTile80330() {
    ProcessSingleTile80220();
}

template <typename T_DATA, typename T_IDX>
__aicore__ inline void GatherV3Tmpl3<T_DATA, T_IDX> :: ProcessSingleTile80220() {
    this->CalcBaseOffset();

    int64_t xOffset = this->xBaseOffset_;
    int64_t yOffset = this->yBaseOffset_;
    int64_t idxOffset = this->idxBaseOffset_;

    auto idxTensor = this->CopyInIdx(idxOffset, 1, this->gyLength_);
    this->SyncM2toS();

    CopyIn(xOffset, idxTensor, this->gyLength_);
    this->FreeTensorIdx(idxTensor);

    Compute(this->pLength_ * this->gyLength_);

    CopyOut(yOffset, this->pLength_ * this->gyLength_ * this->aSize_);
}

}  // namespace GatherV3

#endif  // GATHER_V3_TMPL_3_H