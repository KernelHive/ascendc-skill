/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INPLACE_ATTN_SOFTMAX_LARGE_SHAPE_H
#define INPLACE_ATTN_SOFTMAX_LARGE_SHAPE_H

#include <typeinfo>
#include "kernel_operator.h"
#include "inplace_attn_softmax_bash.h"

namespace InplaceAttnSoftmaxOpt {
using namespace AscendC;

template <typename inType, typename outType, bool isCast, bool isBigshape>
class InplaceAttnSoftmax : public InplaceAttnSoftmaxBase<inType, outType, isCast, isBigshape> {
public:
    __aicore__ inline InplaceAttnSoftmax(TPipe *pipe)
    {
        this->pPipe = pipe;
    }

    __aicore__ inline void Init(GM_ADDR input_gm, GM_ADDR workspace, const InplaceAttnSoftmaxTilingData *__restrict tilingData)
    {
        this->ParseTilingData(tilingData);
        this->softmaxTilingData_ = tilingData->softmaxTilingData;
        this->InitParamsComm();
        InitAndSetBuffer(input_gm, workspace);
    }

    __aicore__ inline void Process()
    {
        ProcessCoreMultiUbMulti();
    }

private:

    __aicore__ inline void InitAndSetBuffer(GM_ADDR input_gm,GM_ADDR workspace_gm)
    {
        // gm数据
        xGm.SetGlobalBuffer((__gm__ inType *)input_gm,this->tilingData_.rowLen * this->tilingData_.colLen);
        this->pPipe->InitBuffer(inQueueA, BUFFER_NUM, this->tileLength * sizeof(inType));
        this->pPipe->InitBuffer(outQueueA, BUFFER_NUM, this->tileLength * sizeof(inType));
        // 若tilingKey为101, 201，则需要给精度转换留空间
        if constexpr(isCast) {
            this->pPipe->InitBuffer(sharedBTempBuf, this->tileLength * sizeof(float));
            tmpCLocal = sharedBTempBuf.Get<float>();
        }
    }

    __aicore__ inline void ProcessCoreMultiUbMulti()
    {
        for (uint32_t ridx = 0; ridx < this->rowLoop; ridx++) {
            this->basicRowLenCal =
                static_cast<uint32_t>((ridx == this->rowLoop - 1) ? (this->rowLenPerCore - (this->rowLoop - 1) * this->basicRowLen)
                                                            : this->basicRowLen);  // 每核处理的最后一个行循环单独处理
            ProcessCoreMultiUbMultiAlign(ridx);
        }
    }

    __aicore__ inline void ComputeVecInGmOffset(uint32_t ridx)
    {
        if (this->coreIdx < this->headCoreNum) {
            this->offsetParam.tmpVecGmOffset = this->coreIdx * this->rowLenPerCore * this->colLen + ridx * this->basicRowLen * this->basicColLen;
        } else {
            this->offsetParam.tmpVecGmOffset = this->headCoreNum * this->tilingData_.rowLenPerHeadCore * this->colLen +
                                         (this->coreIdx - this->headCoreNum) * this->rowLenPerCore * this->colLen +
                                         ridx * this->basicRowLen * this->basicColLen;
        }
    }

    __aicore__ inline void ProcessCoreMultiUbMultiAlign(uint32_t ridx)
    {
        DataCopyParams splitCopyinParams;
        DataCopyParams splitCopyoutParams;

        splitCopyinParams = {this->basicRowLenCal,(uint16_t)(this->basicColLen * sizeof(inType)),
                                0,0};
        splitCopyoutParams = {this->basicRowLenCal,(uint16_t)(this->basicColLen * sizeof(outType)),0,0};
        ComputeVecInGmOffset(ridx);
        CopyIn(this->offsetParam, splitCopyinParams,ridx);
        Compute(ridx);
        CopyOut(this->offsetParam, splitCopyoutParams,ridx);
    }

    __aicore__ inline void CopyIn(InplaceAttnSoftmaxOffsetParam &offsetParam,DataCopyParams &splitCopyinParams,uint32_t ridx)
    {
        LocalTensor<inType> aLocal = inQueueA.template AllocTensor<inType>();
        DataCopyPadParams padParams{true, 0, this->rightPadding, 0};
        DataCopyPad(aLocal, xGm[this->offsetParam.tmpVecGmOffset], splitCopyinParams, padParams);
        inQueueA.template EnQue(aLocal);
    }

    __aicore__ inline void CopyOut(InplaceAttnSoftmaxOffsetParam &offsetParam,DataCopyParams &splitCopyoutParams,uint32_t ridx)
    {
        LocalTensor<outType> outLocal = outQueueA.DeQue<outType>(); 
        DataCopyPad(xGm[this->offsetParam.tmpVecGmOffset], outLocal, splitCopyoutParams);
        outQueueA.FreeTensor(outLocal);
    }
    __aicore__ inline void Compute(uint32_t ridx)
    {
        LocalTensor<inType> aLocal = inQueueA.template DeQue<inType>(); 
        LocalTensor<inType> outLocal = outQueueA.template AllocTensor<inType>(); 
        SoftMaxShapeInfo srcShape = { this->basicRowLenCal, this->sizeHalfLen, this->basicRowLenCal, this->basicColLen};
        if constexpr(isCast) {
            AscendC::Cast(tmpCLocal, aLocal, AscendC::RoundMode::CAST_NONE, aLocal.GetSize());
            SoftMax<float>(tmpCLocal, tmpCLocal, this->softmaxTilingData_, srcShape);
            AscendC::Cast(outLocal, tmpCLocal, AscendC::RoundMode::CAST_RINT, aLocal.GetSize());
        } else {
            SoftMax<inType>(outLocal, aLocal, this->softmaxTilingData_, srcShape);
        }
        inQueueA.FreeTensor(aLocal);
        outQueueA.template EnQue<outType>(outLocal);
    }

private:
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueA;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueA;
    GlobalTensor<inType> xGm;
    TBuf<TPosition::VECCALC> sharedBTempBuf; 
    // quant
    LocalTensor<float> tmpCLocal;
};
}
#endif  