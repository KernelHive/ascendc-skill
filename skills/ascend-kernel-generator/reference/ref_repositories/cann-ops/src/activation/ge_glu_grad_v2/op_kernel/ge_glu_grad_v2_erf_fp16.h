/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
 * \file ge_glu_grad_v2_erf_fp16.h
 * \brief
 */
#ifndef GE_GLU_GRAD_V2_ERF_FP16_H_
#define GE_GLU_GRAD_V2_ERF_FP16_H_

#include "ge_glu_grad_v2_erf_base.h"

namespace GeGluGradV2Erf {
using namespace AscendC;

class GeGluGradV2ErfFP16 : public GeGluGradV2ErfBase<half> {
public:
    __aicore__ inline GeGluGradV2ErfFP16(GM_ADDR dy, GM_ADDR x, GM_ADDR gelu, GM_ADDR dx,
                                         const GeGluGradV2TilingData* tilingDataPtr)
        : GeGluGradV2ErfBase<half>(dy, x, gelu, dx, tilingDataPtr){};
    __aicore__ inline void Init();

    __aicore__ inline void Process(bool perfMode = false) {
        if (perfMode) {
            ProcessPerf<GeGluGradV2ErfFP16, &GeGluGradV2ErfFP16::ComputeLeftHalf,
                        &GeGluGradV2ErfFP16::ComputeRightHalf>(this);
            return;
        }

        if (valueM <= maxProcCount) {
            ProcessLessEqual<GeGluGradV2ErfFP16, &GeGluGradV2ErfFP16::ComputeLeftHalf,
                             &GeGluGradV2ErfFP16::ComputeRightHalf>(this);
        } else {
            ProcessGreater<GeGluGradV2ErfFP16, &GeGluGradV2ErfFP16::ComputeLeftHalf,
                           &GeGluGradV2ErfFP16::ComputeRightHalf>(this);
        }
    };

private:
    __aicore__ inline void ComputeLeftHalf(const int64_t& realProcCount);
    __aicore__ inline void ComputeRightHalf(const int64_t& realProcCount);
};

__aicore__ inline void GeGluGradV2ErfFP16::Init() {
    pipe.InitBuffer(inQueueDY, NO_DB_BUFFER, maxProcCount * sizeof(half));
    pipe.InitBuffer(inQueueGelu, NO_DB_BUFFER, maxProcCount * sizeof(half));
    pipe.InitBuffer(inQueueX1, NO_DB_BUFFER, maxProcCount * sizeof(half));
    pipe.InitBuffer(inQueueX2, NO_DB_BUFFER, maxProcCount * sizeof(half));

    pipe.InitBuffer(outQueueDX1, NO_DB_BUFFER, maxProcCount * sizeof(half));
    pipe.InitBuffer(outQueueDX2, NO_DB_BUFFER, maxProcCount * sizeof(half));

    pipe.InitBuffer(resultTempBuf, FP16_TEMP_BUF_CNT * maxProcCount * sizeof(float));
}

__aicore__ inline void GeGluGradV2ErfFP16::ComputeLeftHalf(const int64_t& realProcCount) {
    LocalTensor<half> ubDY = inQueueDY.DeQue<half>();
    LocalTensor<half> ubGelu = inQueueGelu.DeQue<half>();
    LocalTensor<half> outLocalLeft = outQueueDX1.AllocTensor<half>();

    Mul(outLocalLeft, ubGelu, ubDY, realProcCount);  // dx1 = gelu * dy
    outQueueDX1.EnQue(outLocalLeft);
    inQueueGelu.FreeTensor(ubGelu);

    LocalTensor<half> ubX1 = inQueueX1.DeQue<half>();
    Mul(ubX1, ubX1, ubDY, realProcCount);  // x1 = x1 * dy
    inQueueDY.FreeTensor(ubDY);
    LocalTensor<float> xBufLeft = GetTempBuf<float>(0);
    Cast(xBufLeft, ubX1, RoundMode::CAST_NONE, realProcCount);
    inQueueX1.FreeTensor(ubX1);
}

__aicore__ inline void GeGluGradV2ErfFP16::ComputeRightHalf(const int64_t& realProcCount) {
    LocalTensor<half> ubX2 = inQueueX2.DeQue<half>();
    LocalTensor<float> xBufRight = GetTempBuf<float>(4);
    Cast(xBufRight, ubX2, RoundMode::CAST_NONE, realProcCount);
    inQueueX2.FreeTensor(ubX2);

    LocalTensor<float> xBufLeft = GetTempBuf<float>(0);
    ComputeGeluGrad(xBufLeft, xBufLeft, xBufRight, realProcCount);

    LocalTensor<half> outLocalRight = outQueueDX2.AllocTensor<half>();
    Cast(outLocalRight, xBufLeft, RoundMode::CAST_RINT, realProcCount);
    outQueueDX2.EnQue(outLocalRight);
}

}  // namespace GeGluGradV2Erf

#endif  // GE_GLU_GRAD_V2_ERF_FP16_H_