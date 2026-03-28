/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file select_reduce_max_d_sub_exp_reduce_sum_d_real_div.cpp
 */
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;
namespace AscendC {
template <typename condType, typename aType, typename bType, typename cType>
class SelectReduceMaxDSubExpReduceSumDRealDiv {
public:
    __aicore__ inline SelectReduceMaxDSubExpReduceSumDRealDiv(){}
    __aicore__ inline void Init(GM_ADDR cond, GM_ADDR a, GM_ADDR b, GM_ADDR c, const SelectReduceMaxDSubExpReduceSumDRealDivTilingData &tiling_data)
    {
        this->blockLength = tiling_data.totalLength / AscendC::GetBlockNum();   // 32   2 * 60 
        this->tileNum = tiling_data.tileNum;                                    // 2
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;      // 60
        this->tileLengthAlign = ((tileLength + 16- 1) / 16) * 16;               // 64

        condGlobal.SetGlobalBuffer((__gm__ condType*)cond + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        aGlobal.SetGlobalBuffer((__gm__ aType*)a + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        bGlobal.SetGlobalBuffer((__gm__ aType*)b + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        cGlobal.SetGlobalBuffer((__gm__ aType*)c + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueCond, BUFFER_NUM, tileLengthAlign * sizeof(condType));
        pipe.InitBuffer(inQueueA, BUFFER_NUM, tileLengthAlign * sizeof(aType));
        pipe.InitBuffer(inQueueB, BUFFER_NUM, tileLengthAlign * sizeof(aType));
        pipe.InitBuffer(outQueueC, BUFFER_NUM, tileLengthAlign * sizeof(aType));
        pipe.InitBuffer(tmpQueue0, BUFFER_NUM, tileLength * 2 * sizeof(uint8_t));
        pipe.InitBuffer(sel0TmpQueue, BUFFER_NUM, tileLength * 2 * sizeof(uint8_t));
        pipe.InitBuffer(sel1TmpQueue, BUFFER_NUM, tileLength * 2 * sizeof(uint8_t));
        pipe.InitBuffer(tmpQueue1, BUFFER_NUM, tileLengthAlign * sizeof(aType));
        pipe.InitBuffer(tmpQueue2, BUFFER_NUM, tileLengthAlign * sizeof(aType));
        pipe.InitBuffer(tmpQueue3, BUFFER_NUM, tileLengthAlign * sizeof(aType));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;             // 2
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<condType> condLocal = inQueueCond.AllocTensor<condType>();
        LocalTensor<aType> aLocal = inQueueA.AllocTensor<aType>();
        LocalTensor<aType> bLocal = inQueueB.AllocTensor<aType>();

        AscendC::DataCopyExtParams condCopyParams{1, (uint32_t)(this->tileLength * sizeof(condType)), 0, 0, 0};                      // 结构体DataCopyExtParams最后一个参数是rsv保留位
        AscendC::DataCopyExtParams copyParams{1, (uint32_t)(this->tileLength * sizeof(aType)), 0, 0, 0};                      // 结构体DataCopyExtParams最后一个参数是rsv保留位
        AscendC::DataCopyPadExtParams<condType> condpadParams{true, 0, 0, 0};
        AscendC::DataCopyPadExtParams<aType> padParams{true, 0, 0, 0};

        AscendC::DataCopyPad(condLocal, condGlobal[progress * this->tileLength], condCopyParams, condpadParams);    // 从GM->VECIN搬运
        AscendC::DataCopyPad(aLocal, aGlobal[progress * this->tileLength], copyParams, padParams);    // 从GM->VECIN搬运
        AscendC::DataCopyPad(bLocal, bGlobal[progress * this->tileLength], copyParams, padParams);    // 从GM->VECIN搬运

        inQueueCond.EnQue(condLocal);
        inQueueA.EnQue(aLocal);
        inQueueB.EnQue(bLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<condType> condLocal = inQueueCond.DeQue<condType>();    // 64 Bytes 64
        LocalTensor<aType> aLocal = inQueueA.DeQue<aType>();                // 128 Bytes 64
        LocalTensor<aType> bLocal = inQueueB.DeQue<aType>();                // 128 Bytes 64
        LocalTensor<aType> cLocal = outQueueC.AllocTensor<aType>();

        // --------------------------------------------------select
        LocalTensor<uint8_t> tmpTensor = tmpQueue0.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> sel0TmpTensor = sel0TmpQueue.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> sel1TmpTensor = sel1TmpQueue.AllocTensor<uint8_t>();
        LocalTensor<aType> sel0Local = tmpQueue1.AllocTensor<aType>();
        LocalTensor<aType> sel1Local = tmpQueue2.AllocTensor<aType>();
        aType scalar = 0.0f;

        AscendC::SelectWithBytesMaskShapeInfo shapeInfo;
        shapeInfo.firstAxis = 1;
        shapeInfo.srcLastAxis = tileLengthAlign;
        shapeInfo.maskLastAxis = tileLengthAlign;

        AscendC::SelectWithBytesMask(aLocal, scalar, aLocal, condLocal, sel0TmpTensor, shapeInfo);
        AscendC::SelectWithBytesMask(bLocal, bLocal, scalar, condLocal, sel1TmpTensor, shapeInfo);
        cLocal = aLocal + bLocal;

        tmpQueue0.FreeTensor(tmpTensor);
        sel0TmpQueue.FreeTensor(sel0TmpTensor);
        sel1TmpQueue.FreeTensor(sel1TmpTensor);
        tmpQueue1.FreeTensor(sel0Local);
        tmpQueue2.FreeTensor(sel1Local);

        // --------------------------------------------------reduceMaxD + broadcast + Sub
        LocalTensor<aType> maxBroadLocal = tmpQueue1.AllocTensor<aType>();
        LocalTensor<aType> maxLocal = tmpQueue2.AllocTensor<aType>();
        LocalTensor<aType> subLocal = tmpQueue3.AllocTensor<aType>();

        AscendC::WholeReduceMax<aType>(maxLocal, cLocal, 64, 1, 1, 1, 4, ReduceOrder::ORDER_ONLY_VALUE);    // [mnk]->[mn1]
        uint32_t dstShape[2] = {1, 64};
        uint32_t srcShape[2] = {1, 1};
        AscendC::BroadCast<aType, 2, 1>(maxBroadLocal, maxLocal, dstShape, srcShape);
        subLocal = cLocal - maxBroadLocal;

        tmpQueue1.FreeTensor(maxBroadLocal);
        tmpQueue2.FreeTensor(maxLocal);
        // Exp
        LocalTensor<aType> expLocal = tmpQueue1.AllocTensor<aType>();

        AscendC::Exp<aType, 15, false>(expLocal, subLocal, tileLengthAlign);

        tmpQueue3.FreeTensor(subLocal);
        // ReduceSumD + broadcast + RealDiv
        LocalTensor<aType> sumLocal = tmpQueue2.AllocTensor<aType>();
        LocalTensor<aType> sumBroadLocal = tmpQueue3.AllocTensor<aType>();
        
        AscendC::WholeReduceSum<aType>(sumLocal, expLocal, 60, 1, 1, 1, 4);
        AscendC::BroadCast<aType, 2, 1>(sumBroadLocal, sumLocal, dstShape, srcShape);
        Div(cLocal, expLocal, sumBroadLocal, (int32_t)expLocal.GetSize());

        tmpQueue1.FreeTensor(expLocal);
        tmpQueue2.FreeTensor(sumLocal);
        tmpQueue3.FreeTensor(sumBroadLocal);

        inQueueCond.FreeTensor(condLocal);
        inQueueA.FreeTensor(aLocal);
        inQueueB.FreeTensor(bLocal);
        outQueueC.EnQue<aType>(cLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<aType> cLocal = outQueueC.DeQue<aType>();
        DataCopyExtParams copyParams{1, (uint32_t)(this->tileLength * sizeof(aType)), 0, 0, 0};
        AscendC::DataCopyPad(cGlobal[progress * this->tileLength], cLocal, copyParams);
        outQueueC.FreeTensor(cLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueA, inQueueB, inQueueCond;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueC;
    TQue<QuePosition::VECCALC, BUFFER_NUM> tmpQueue0, sel0TmpQueue, sel1TmpQueue;
    TQue<QuePosition::VECCALC, BUFFER_NUM> tmpQueue1;
    TQue<QuePosition::VECCALC, BUFFER_NUM> tmpQueue2;
    TQue<QuePosition::VECCALC, BUFFER_NUM> tmpQueue3;

    GlobalTensor<aType> aGlobal;
    GlobalTensor<aType> bGlobal;
    GlobalTensor<aType> cGlobal;
    GlobalTensor<condType> condGlobal;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t tileLengthAlign = 0;
};

}
extern "C" __global__ __aicore__ void select_reduce_max_d_sub_exp_reduce_sum_d_real_div(GM_ADDR sel, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    SelectReduceMaxDSubExpReduceSumDRealDiv<bool, half, half, half> op;
    op.Init(sel, a, b, c, tiling_data);
    op.Process();
}