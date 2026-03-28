/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file foreach_custom.h
 * \brief
 */
 
#ifndef FOREACH_CUSTOM_H
#define FOREACH_CUSTOM_H

#include "foreach_unary_v2.h"

namespace ForeachCustom{

constexpr int16_t BYTE_REPEATE = 256;

using namespace Common::OpKernel;
using namespace AscendC;
template<typename T, typename P, UnaryOp<P> *op, int32_t bufferNum=BUFFER_NUM, uint8_t paramsCount=INPUT_PARAMETER_COUNT,
         bool needCopyOut=NEED_COPY_OUT, bool needTempBuf=NEED_TEMP_BUF, typename Tiling=ForeachCommonTilingData>
class ForeachCustomNd : public ForeachUnaryV2<T, P, op, bufferNum, paramsCount, needCopyOut, needTempBuf, Tiling> {
public:
    using Unary = ForeachUnaryV2<T, P, op, bufferNum, paramsCount, needCopyOut, needTempBuf, Tiling>;
    __aicore__ inline ForeachCustomNd(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                const Tiling* tilingData);
};

template <typename T, typename P, UnaryOp<P> *op, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut, bool needTempBuf, typename Tiling>
__aicore__ inline void ForeachCustomNd<T, P, op, bufferNum, paramsCount, needCopyOut, needTempBuf, Tiling>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                const Tiling* tilingData) { 
    // tiling参数初始化         
    Unary::Base::Base::Init(tilingData);
    // 重新划分UB
    Unary::Base::inputsTensorUbSize -= BYTE_REPEATE;
    Unary::Base::inputsTensorUbSize = Unary::Base::inputsTensorUbSize / 32 * 32;    
    #if __CCE_AICORE__ == 220
        if (std::is_same_v<T, bfloat16_t>) {
            Unary::Base::totalTensorUbSize = Unary::Base::inputsTensorUbSize * COPY_SPACE_MULTIPLE;
            Unary::Base::maxDataCount = Unary::Base::totalTensorUbSize / sizeof(T);        
            Unary::Base::maxCastDataCount = Unary::Base::inputsTensorUbSize / sizeof(float);
        } else {
            Unary::Base::maxDataCount = Unary::Base::inputsTensorUbSize / sizeof(T);
        }
    #else 
        Unary::Base::maxDataCount = Unary::Base::inputsTensorUbSize / sizeof(T);
    #endif

    Unary::Base::inTensorsPtr = x;
    Unary::Base::outTensorsPtr = y;

    #if __CCE_AICORE__ == 220
        if (std::is_same_v<T, bfloat16_t>) {
            Unary::Base::pipe.InitBuffer(Unary::Base::dataQueue, bufferNum, Unary::Base::totalTensorUbSize);
            if (needCopyOut) {
                Unary::Base::pipe.InitBuffer(Unary::Base::outQueue, bufferNum, Unary::Base::totalTensorUbSize);
            }
            Unary::Base::pipe.InitBuffer(Unary::Base::float32Queue, 1, Unary::Base::inputsTensorUbSize * paramsCount);
            LocalTensor<float> float32Tensor = Unary::Base::float32Queue.template AllocTensor<float>();
            Unary::Base::float32Queue.EnQue(float32Tensor);
        } else {
            Unary::Base::pipe.InitBuffer(Unary::Base::dataQueue, bufferNum, Unary::Base::inputsTensorUbSize);
            if (needCopyOut) {
                Unary::Base::pipe.InitBuffer(Unary::Base::outQueue, bufferNum, Unary::Base::inputsTensorUbSize);
            }
        }
    #else 
        Unary::Base::pipe.InitBuffer(Unary::Base::dataQueue, bufferNum, Unary::Base::inputsTensorUbSize);
        if (needCopyOut) {
            Unary::Base::pipe.InitBuffer(Unary::Base::outQueue, bufferNum, Unary::Base::inputsTensorUbSize);
        }
    #endif

    // 给中间变量分配内存
    Unary::Base::pipe.InitBuffer(Unary::tempQueue, 1, BYTE_REPEATE);
    Unary::tempTensor = Unary::tempQueue.template AllocTensor<P>();
    Unary::tempQueue.EnQue(Unary::tempTensor);
}
}
#endif // FOREACH_CUSTOM_H
