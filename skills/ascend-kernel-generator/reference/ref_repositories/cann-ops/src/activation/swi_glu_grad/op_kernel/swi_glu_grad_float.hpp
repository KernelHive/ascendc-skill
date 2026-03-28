/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file swi_glu_grad_float.hpp
 * \brief
 */
#ifndef OPP_SWI_GLU_GRAD_FLOAT_HPP
#define OPP_SWI_GLU_GRAD_FLOAT_HPP
#include "kernel_operator.h"

using namespace AscendC;
template<typename aType, typename bType, typename lType, typename mType, typename nType, uint16_t bufferNum>
class SwiGluGradVector {
public:
    __aicore__ inline SwiGluGradVector() {}

protected:
    __aicore__ inline void InitUbBuffer(uint64_t tileLength);
    __aicore__ inline void Compute(uint64_t curTileLen);

    float beta = 1.0;
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> inQueueA;
    TQue<QuePosition::VECIN, bufferNum> inQueueB;
    TQue<QuePosition::VECIN, bufferNum> inQueueL;
    TQue<QuePosition::VECOUT, bufferNum> outQueueM;
    TQue<QuePosition::VECOUT, bufferNum> outQueueN;
    TBuf<TPosition::VECCALC> tmpQueue;
    TBuf<TPosition::VECCALC> sigQueue;
    LocalTensor<float> tempLocal;
    LocalTensor<float> sigLocal;
    GlobalTensor<aType> aGm;
    GlobalTensor<bType> bGm;
    GlobalTensor<lType> lGm;
    GlobalTensor<mType> mGm;
    GlobalTensor<nType> nGm;
};

template<typename aType, typename bType, typename lType, typename mType, typename nType, uint16_t bufferNum>
__aicore__ inline void SwiGluGradVector<aType, bType, lType, mType, nType, bufferNum>::InitUbBuffer(uint64_t tileLength)
{
    // pipe alloc memory to queue, the unit is Bytes
    pipe.InitBuffer(inQueueA, bufferNum, tileLength * sizeof(aType));
    pipe.InitBuffer(inQueueB, bufferNum, tileLength * sizeof(bType));
    pipe.InitBuffer(inQueueL, bufferNum, tileLength * sizeof(lType));
    pipe.InitBuffer(outQueueM, bufferNum, tileLength * sizeof(mType)); // The length must be an integer multiple of 32
    pipe.InitBuffer(outQueueN, bufferNum, tileLength * sizeof(nType)); // The length must be an integer multiple of 32

    pipe.InitBuffer(tmpQueue, tileLength * sizeof(float));
    pipe.InitBuffer(sigQueue, tileLength * sizeof(float));
    tempLocal = tmpQueue.Get<float>();
    sigLocal = sigQueue.Get<float>();
}

template<typename aType, typename bType, typename lType, typename mType, typename nType, uint16_t bufferNum>
__aicore__ inline void SwiGluGradVector<aType, bType, lType, mType, nType, bufferNum>::Compute(uint64_t tileLength)
{
    // tbuf::templocaltensor
    // deque input tensors from VECIN queue
    //calc sigLocal
    LocalTensor<aType> aLocal = inQueueA.template DeQue<aType>(); //input a
    Muls(sigLocal, aLocal, beta, tileLength);
    PipeBarrier<PIPE_V>();
    Exp(sigLocal, sigLocal, tileLength);
    PipeBarrier<PIPE_V>();
    Adds(sigLocal, sigLocal, (mType)(1.0), tileLength);
    PipeBarrier<PIPE_V>();
    Duplicate<float>(tempLocal, (float)(1.0), tileLength);
    Div(sigLocal, tempLocal, sigLocal, tileLength);
    PipeBarrier<PIPE_V>();

    //----------------N
    LocalTensor<nType> nLocal = outQueueN.template AllocTensor<nType>(); // lb
    Mul(nLocal, sigLocal, aLocal, tileLength);
    LocalTensor<lType> lLocal = inQueueL.template DeQue<lType>(); // input l
    Mul(nLocal, nLocal, lLocal, tileLength);
    PipeBarrier<PIPE_V>();
    outQueueN.template EnQue<nType>(nLocal);

    //----------------M
    Muls(tempLocal, sigLocal, (mType)(-1.0), tileLength);
    PipeBarrier<PIPE_V>();
    Adds(tempLocal, tempLocal, (mType)(1.0), tileLength);
    PipeBarrier<PIPE_V>();
    LocalTensor<mType> mLocal = outQueueM.template AllocTensor<mType>(); // la
    Mul(mLocal, sigLocal, tempLocal, tileLength);
    Mul(mLocal, mLocal, aLocal, tileLength);
    inQueueA.template FreeTensor(aLocal);

    Muls(mLocal, mLocal, -beta, tileLength);
    Add(mLocal, mLocal, sigLocal, tileLength);
    LocalTensor<bType> bLocal = inQueueB.template DeQue<bType>(); //input b
    Mul(mLocal, mLocal, bLocal, tileLength);
    inQueueB.template FreeTensor(bLocal);

    Mul(mLocal, mLocal, lLocal, tileLength);
    PipeBarrier<PIPE_V>();
    // enque the output tensor to VECOUT queue
    outQueueM.template EnQue<mType>(mLocal);
    // free input tensors for reuse
    inQueueL.template FreeTensor(lLocal);
}
#endif // OPP_SWI_GLU_GRAD_FLOAT_HPP