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
 * @file gelu_grad.cpp
 */

#include "kernel_operator.h"
#define GENERAL_OP_IMPL(templateClass,...)                                          \
  do{                                                                               \
      GET_TILING_DATA(tiling_data, tiling);                                         \
      templateClass<__VA_ARGS__>op;                                                 \
      op.Init(dy, x, y, z, tiling_data.smallCoreDataNum,                               \
            tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,                    \
            tiling_data.finalSmallTileNum, tiling_data.tileDataNum,                     \
            tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,                   \
            tiling_data.tailBlockNum, tiling_data.versionNum);                       \
      op.Process();                                                                 \
  }while(0)
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_DY, typename TYPE_X, typename TYPE_Z,bool Is0versionNum>
class KernelGeluGrad
{
    using T = TYPE_X;

public:
    __aicore__ inline KernelGeluGrad() {}
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum, uint32_t tileDataNum,
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum,
                                uint32_t tailBlockNum, uint32_t versionNum)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreNum = GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * GetBlockIdx();
        this->tileDataNum = tileDataNum;
        this->versionNum = versionNum;
        if (coreNum < tailBlockNum)
        {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        }
        else
        {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (GetBlockIdx() - tailBlockNum);
        }
        dyGm.SetGlobalBuffer((__gm__ TYPE_DY *)dy + globalBufferIndex, this->coreDataNum);
        xGm.SetGlobalBuffer((__gm__ TYPE_X *)x + globalBufferIndex, this->coreDataNum);
        zGm.SetGlobalBuffer((__gm__ TYPE_Z *)z + globalBufferIndex, this->coreDataNum);

        pipe.InitBuffer(inQueueDY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_DY));
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Z));

        pipe.InitBuffer(tmpBuffer, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(funBuffer, this->tileDataNum * sizeof(float));
        if constexpr(Is0versionNum)
        {
            if constexpr (!std::is_same_v<T, float32_t>)
            {
                pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(float));
                pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(float));
                pipe.InitBuffer(tmp3, this->tileDataNum * sizeof(float));
            }
        }
        else
        {
            if constexpr (!std::is_same_v<T, float32_t>)
            {
                pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(float));
                pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(float));
                pipe.InitBuffer(tmp3, this->tileDataNum * sizeof(float));
            }
            else
            {
                pipe.InitBuffer(tmpBuffer, this->tileDataNum * sizeof(float));
            }
        }      
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount-1; i++)
        {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        this->processDataNum = this->tailDataNum;
        CopyIn(loopCount-1);
        Compute(loopCount-1);
        CopyOut(loopCount-1);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        LocalTensor<TYPE_DY> dyLocal = inQueueDY.AllocTensor<TYPE_DY>();
        DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
        DataCopy(dyLocal, dyGm[progress * this->tileDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
        inQueueDY.EnQue(dyLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        LocalTensor<TYPE_DY> dyLocal = inQueueDY.DeQue<TYPE_DY>();
        LocalTensor<TYPE_Z> zLocal = outQueueZ.AllocTensor<TYPE_Z>();
        LocalTensor<float> fun = funBuffer.Get<float>();
        LocalTensor<float> tmp = tmpBuffer.Get<float>();
        if constexpr(Is0versionNum)
        {
            const float COEFF0 = -0.0713548162726002527220f;
            const float COEFF1 = -1.595769121605730711759f;
            const float COEFF2 = 0.2140644488178007f;
            const float COEFF3 = 1.595769121605730711759f;
            const float COEFF4 = 1.0f;
            if constexpr (!std::is_same_v<T, float32_t>)
            {
                LocalTensor<float> p1 = tmp1.Get<float>(); // xlocal
                LocalTensor<float> p2 = tmp2.Get<float>(); // dylocal
                LocalTensor<float> p3 = tmp3.Get<float>(); // zlocal
                Cast(p1, xLocal, RoundMode::CAST_NONE, this->processDataNum);
                Cast(p2, dyLocal, RoundMode::CAST_NONE, this->processDataNum);

                // 计算
                Mul(p3, p1, p1, this->processDataNum);      // x*2
                Muls(p3, p3, COEFF0, this->processDataNum); // x^2*COEFF0
                Adds(p3, p3, COEFF1, this->processDataNum); // x^2*COEFF0+COEFF1
                Mul(p3, p3, p1, this->processDataNum);      //( x^2*COEFF0+COEFF1 )*x
                Exp(p3, p3, this->processDataNum);          // exp()    px

                Mul(tmp, p1, p1, this->processDataNum);
                Muls(tmp, tmp, COEFF2, this->processDataNum); // x^2*COEFF2
                Adds(tmp, tmp, COEFF3, this->processDataNum); // x^2*COEFF2 +COEFF3
                Mul(tmp, tmp, p1, this->processDataNum);      // (x^2*COEFF2 +COEFF3)*x

                Adds(fun, p3, COEFF4, this->processDataNum); // t=tmp+1

                Duplicate(p1, COEFF4, this->processDataNum);
                Div(fun, p1, fun, this->processDataNum);

                Mul(p3, p3, fun, this->processDataNum); // z=tmp*t
                Mul(p3, p3, tmp, this->processDataNum); // z=z*primtmp
                Mul(p3, p3, fun, this->processDataNum); // z=z*t

                Add(p3, p3, fun, this->processDataNum); // z=z+t
                Mul(p3, p3, p2, this->processDataNum);  // z=z*dy

                Cast(zLocal, p3, RoundMode::CAST_RINT, this->processDataNum);
            }
            else
            {
                Mul(zLocal, xLocal, xLocal, this->processDataNum);  // x*2
                Muls(zLocal, zLocal, COEFF0, this->processDataNum); // x^2*COEFF0
                Adds(zLocal, zLocal, COEFF1, this->processDataNum); // x^2*COEFF0+COEFF1
                Mul(zLocal, zLocal, xLocal, this->processDataNum);  //( x^2*COEFF0+COEFF1 )*x
                Exp(zLocal, zLocal, this->processDataNum);          // exp()    px

                Mul(tmp, xLocal, xLocal, this->processDataNum);
                Muls(tmp, tmp, COEFF2, this->processDataNum); // x^2*COEFF2
                Adds(tmp, tmp, COEFF3, this->processDataNum); // x^2*COEFF2 +COEFF3
                Mul(tmp, tmp, xLocal, this->processDataNum);  // (x^2*COEFF2 +COEFF3)*x

                Adds(fun, zLocal, COEFF4, this->processDataNum); // t=tmp+1

                Duplicate(xLocal, COEFF4, this->processDataNum);
                Div(fun, xLocal, fun, this->processDataNum);

                Mul(zLocal, zLocal, fun, this->processDataNum); // z=tmp*t
                Mul(zLocal, zLocal, tmp, this->processDataNum); // z=z*primtmp
                Mul(zLocal, zLocal, fun, this->processDataNum); // z=z*t

                Add(zLocal, zLocal, fun, this->processDataNum);     // z=z+t
                Mul(zLocal, zLocal, dyLocal, this->processDataNum); // z=z*dy
            }
        }
        else
        {
            const float COEFF0 = -0.0713548162726002527220f;
            const float COEFF1 = -1.5957691216057308f;
            const float COEFF2 = -0.21406444881780074632901625683959062f;
            const float COEFF3 = -1.5957691216057307117597842397375274738f;
            const float COEFF4 = 1.0f;
            const float COEFF5 = -1.0f;

            if constexpr (!std::is_same_v<T, float32_t>)
            {
                LocalTensor<float> p1 = tmp1.Get<float>(); // xlocal
                LocalTensor<float> p2 = tmp2.Get<float>(); // dylocal
                LocalTensor<float> p3 = tmp3.Get<float>(); // zlocal
                Cast(p1, xLocal, RoundMode::CAST_NONE, this->processDataNum);
                Cast(p2, dyLocal, RoundMode::CAST_NONE, this->processDataNum);

                Mul(p3, p1, p1, this->processDataNum);
                Muls(p3, p3, COEFF0, this->processDataNum);
                Adds(p3, p3, COEFF1, this->processDataNum);
                Mul(p3, p3, p1, this->processDataNum);
                Exp(p3, p3, this->processDataNum); // exp()
                Adds(p3, p3, COEFF4, this->processDataNum);
                Duplicate(tmp, COEFF4, this->processDataNum);
                Div(p3, tmp, p3, this->processDataNum); // g1

                Adds(tmp, p3, COEFF5, this->processDataNum);
                Mul(tmp, tmp, p1, this->processDataNum); // res

                Mul(fun, p1, p1, this->processDataNum);
                Muls(fun, fun, COEFF2, this->processDataNum);
                Adds(fun, fun, COEFF3, this->processDataNum); // g2

                Mul(tmp, tmp, fun, this->processDataNum);
                Adds(tmp, tmp, COEFF4, this->processDataNum);
                Mul(tmp, tmp, p3, this->processDataNum);
                Mul(p3, tmp, p2, this->processDataNum);

                Cast(zLocal, p3, RoundMode::CAST_RINT, this->processDataNum);
            }
            else
            {
                LocalTensor<float> p1 = tmp1.Get<float>();
                Mul(zLocal, xLocal, xLocal, this->processDataNum);
                Muls(zLocal, zLocal, COEFF0, this->processDataNum);
                Adds(zLocal, zLocal, COEFF1, this->processDataNum);
                Mul(zLocal, zLocal, xLocal, this->processDataNum);
                Exp(zLocal, zLocal, this->processDataNum); // exp()
                Adds(zLocal, zLocal, COEFF4, this->processDataNum);
                Duplicate(p1, COEFF4, this->processDataNum);
                Div(zLocal, p1, zLocal, this->processDataNum); // g1

                Adds(tmp, zLocal, COEFF5, this->processDataNum);
                Mul(tmp, tmp, xLocal, this->processDataNum); // res

                Mul(fun, xLocal, xLocal, this->processDataNum);
                Muls(fun, fun, COEFF2, this->processDataNum);
                Adds(fun, fun, COEFF3, this->processDataNum); // g2

                Mul(tmp, tmp, fun, this->processDataNum);
                Adds(tmp, tmp, COEFF4, this->processDataNum);
                Mul(tmp, tmp, zLocal, this->processDataNum);
                Mul(zLocal, tmp, dyLocal, this->processDataNum);
            }
        }
        outQueueZ.EnQue<TYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueDY.FreeTensor(dyLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<TYPE_Z> zLocal = outQueueZ.DeQue<TYPE_Z>();
        DataCopy(zGm[progress * this->tileDataNum], zLocal, this->processDataNum);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueDY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    TBuf<QuePosition::VECCALC> tmp1, tmp2;
    TBuf<QuePosition::VECCALC> tmp3, funBuffer;
    TBuf<QuePosition::VECCALC> tmpBuffer;
    GlobalTensor<TYPE_X> xGm;
    GlobalTensor<TYPE_DY> dyGm;
    GlobalTensor<TYPE_Z> zGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    uint32_t versionNum;
};

extern "C" __global__ __aicore__ void gelu_grad(GM_ADDR dy, GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    if(TILING_KEY_IS(0))
    {
      GENERAL_OP_IMPL(KernelGeluGrad,DTYPE_DY, DTYPE_X, DTYPE_Z,true);
    }
     else if(TILING_KEY_IS(1))
    {
      GENERAL_OP_IMPL(KernelGeluGrad,DTYPE_DY, DTYPE_X, DTYPE_Z,false);
    }
}
