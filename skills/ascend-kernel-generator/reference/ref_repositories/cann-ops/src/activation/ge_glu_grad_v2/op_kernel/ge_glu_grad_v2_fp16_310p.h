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
 * \file ge_glu_grad_v2_fp16_310p.h
 * \brief
 */
#ifndef GE_GLU_GRAD_V2_FP16_310P_H_
#define GE_GLU_GRAD_V2_FP16_310P_H_

#include "ge_glu_grad_v2_base_310p.h"

namespace GeGluGradV2For310P {
using namespace AscendC;

class GeGluGradV2FP16By310p : public GeGluGradV2Base310p<half> {
public:
    __aicore__ inline GeGluGradV2FP16By310p(GM_ADDR dy, GM_ADDR x, GM_ADDR gelu, GM_ADDR dx, GM_ADDR workspace,
                                            const GeGluGradV2TilingData* tilingDataPtr)
        : GeGluGradV2Base310p<half>(dy, x, gelu, dx, workspace, tilingDataPtr){};
    __aicore__ inline void Init();
    __aicore__ inline void Process() {
        if (valueM <= maxProcCount) {
            ProcessLessEqual<GeGluGradV2FP16By310p, &GeGluGradV2FP16By310p::ComputeLeftHalf,
                             &GeGluGradV2FP16By310p::ComputeRightHalf>(this);
        } else {
            ProcessGreater<GeGluGradV2FP16By310p, &GeGluGradV2FP16By310p::ComputeLeftHalf,
                           &GeGluGradV2FP16By310p::ComputeRightHalf>(this);
        }
    };

private:
    __aicore__ inline void ComputeLeftHalf(const int64_t& realProcCount);
    __aicore__ inline void ComputeRightHalf(const int64_t& realProcCount);
};

__aicore__ inline void GeGluGradV2FP16By310p::Init() {
    BaseInit();

    pipe.InitBuffer(inQueueDY, NO_DB_BUFFER, maxProcCount * sizeof(half));
    pipe.InitBuffer(inQueueGelu, NO_DB_BUFFER, maxProcCount * sizeof(half));
    pipe.InitBuffer(inQueueX1, NO_DB_BUFFER, maxProcCount * sizeof(half));
    pipe.InitBuffer(inQueueX2, NO_DB_BUFFER, maxProcCount * sizeof(half));
    pipe.InitBuffer(outQueueDX1, NO_DB_BUFFER, maxProcCount * sizeof(half));
    pipe.InitBuffer(outQueueDX2, NO_DB_BUFFER, maxProcCount * sizeof(half));

    pipe.InitBuffer(resultTempBuf1, maxProcCount * sizeof(float));
    pipe.InitBuffer(resultTempBuf2, maxProcCount * sizeof(float));
    pipe.InitBuffer(resultTempBuf3, maxProcCount * sizeof(float));
    pipe.InitBuffer(resultTempBuf4, maxProcCount * sizeof(float));
}

__aicore__ inline void GeGluGradV2FP16By310p::ComputeLeftHalf(const int64_t& realProcCount) {
    LocalTensor<half> ubDY = inQueueDY.DeQue<half>();
    LocalTensor<half> ubGelu = inQueueGelu.DeQue<half>();
    LocalTensor<half> outLocalLeft = outQueueDX1.AllocTensor<half>();
    Mul(outLocalLeft, ubGelu, ubDY, realProcCount);  // dx1 = gelu * dy
    outQueueDX1.EnQue(outLocalLeft);
    inQueueGelu.FreeTensor(ubGelu);
    LocalTensor<half> ubX1 = inQueueX1.DeQue<half>();
    PipeBarrier<PIPE_V>();
    Mul(ubX1, ubX1, ubDY, realProcCount);  // x1 = x1 * dy
    inQueueDY.FreeTensor(ubDY);
    LocalTensor<float> xBufLeft = resultTempBuf1.Get<float>();
    PipeBarrier<PIPE_V>();
    Cast(xBufLeft, ubX1, RoundMode::CAST_NONE, realProcCount);
    inQueueX1.FreeTensor(ubX1);
}

__aicore__ inline void GeGluGradV2FP16By310p::ComputeRightHalf(const int64_t& realProcCount) {
    LocalTensor<half> ubX2 = inQueueX2.DeQue<half>();
    LocalTensor<float> xBufRight = resultTempBuf2.Get<float>();
    PipeBarrier<PIPE_V>();
    Cast(xBufRight, ubX2, RoundMode::CAST_NONE, realProcCount);
    PipeBarrier<PIPE_V>();
    inQueueX2.FreeTensor(ubX2);

    LocalTensor<float> xBufLeft = resultTempBuf1.Get<float>();
    PipeBarrier<PIPE_V>();
    ComputeGeluGrad(xBufRight, xBufLeft, xBufRight, realProcCount);
    PipeBarrier<PIPE_V>();

    LocalTensor<half> outLocalRight = outQueueDX2.AllocTensor<half>();
    PipeBarrier<PIPE_V>();

    Cast(outLocalRight, xBufRight, RoundMode::CAST_NONE, realProcCount);
    PipeBarrier<PIPE_V>();
    outQueueDX2.EnQue(outLocalRight);
}

}  // namespace GeGluGradV2For310P

#endif  // GE_GLU_GRAD_V2_FP16_310P_H_