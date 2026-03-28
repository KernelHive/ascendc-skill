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
 * \file ge_glu_grad_v2_tanh_bfp16.h
 * \brief
 */
#ifndef GE_GLU_GRAD_V2_TANH_BFP16_H_
#define GE_GLU_GRAD_V2_TANH_BFP16_H_

#include "ge_glu_grad_v2_tanh_base.h"

namespace GeGluGradV2Tanh {
using namespace AscendC;

class GeGluGradV2TanhBFP16 : public GeGluGradV2TanhBase<bfloat16_t> {
public:
    __aicore__ inline GeGluGradV2TanhBFP16(GM_ADDR dy, GM_ADDR x, GM_ADDR gelu, GM_ADDR dx,
                                           const GeGluGradV2TilingData* tilingDataPtr)
        : GeGluGradV2TanhBase<bfloat16_t>(dy, x, gelu, dx, tilingDataPtr){};
    __aicore__ inline void Init();

    __aicore__ inline void Process(bool perfMode = false) {
        if (perfMode) {
            ProcessPerf<GeGluGradV2TanhBFP16, &GeGluGradV2TanhBFP16::ComputeLeftHalf,
                        &GeGluGradV2TanhBFP16::ComputeRightHalf>(this);
            return;
        }

        if (valueM <= maxProcCount) {
            ProcessLessEqual<GeGluGradV2TanhBFP16, &GeGluGradV2TanhBFP16::ComputeLeftHalf,
                             &GeGluGradV2TanhBFP16::ComputeRightHalf>(this);
        } else {
            ProcessGreater<GeGluGradV2TanhBFP16, &GeGluGradV2TanhBFP16::ComputeLeftHalf,
                           &GeGluGradV2TanhBFP16::ComputeRightHalf>(this);
        }
    };

private:
    __aicore__ inline void ComputeLeftHalf(const int64_t& realProcCount);
    __aicore__ inline void ComputeRightHalf(const int64_t& realProcCount);
};

__aicore__ inline void GeGluGradV2TanhBFP16::Init() {
    pipe.InitBuffer(inQueueX1, NO_DB_BUFFER, maxProcCount * sizeof(bfloat16_t));
    pipe.InitBuffer(inQueueX2, NO_DB_BUFFER, maxProcCount * sizeof(bfloat16_t));
    pipe.InitBuffer(inQueueDY, NO_DB_BUFFER, maxProcCount * sizeof(float));
    pipe.InitBuffer(inQueueGelu, NO_DB_BUFFER, maxProcCount * sizeof(float));

    pipe.InitBuffer(outQueueDX1, NO_DB_BUFFER, maxProcCount * sizeof(bfloat16_t));
    pipe.InitBuffer(outQueueDX2, NO_DB_BUFFER, maxProcCount * sizeof(bfloat16_t));

    pipe.InitBuffer(resultTempBuf, BFP16_TEMP_BUF_CNT * maxProcCount * sizeof(float));
}

__aicore__ inline void GeGluGradV2TanhBFP16::ComputeLeftHalf(const int64_t& realProcCount) {
    LocalTensor<float> ubDY = inQueueDY.DeQue<float>();
    LocalTensor<bfloat16_t> ubDYbf16 = ubDY.ReinterpretCast<bfloat16_t>()[maxProcCount];
    Cast(ubDY, ubDYbf16, RoundMode::CAST_NONE, realProcCount);

    LocalTensor<float> ubGelu = inQueueGelu.DeQue<float>();
    LocalTensor<bfloat16_t> ubGelubf16 = ubGelu.ReinterpretCast<bfloat16_t>()[maxProcCount];
    Cast(ubGelu, ubGelubf16, RoundMode::CAST_NONE, realProcCount);

    LocalTensor<bfloat16_t> outLocalLeft = outQueueDX1.AllocTensor<bfloat16_t>();

    Mul(ubGelu, ubGelu, ubDY, realProcCount);  // dx1 = gelu * dy
    Cast(outLocalLeft, ubGelu, RoundMode::CAST_RINT, realProcCount);
    outQueueDX1.EnQue(outLocalLeft);
    inQueueGelu.FreeTensor(ubGelu);

    LocalTensor<bfloat16_t> ubX1 = inQueueX1.DeQue<bfloat16_t>();
    LocalTensor<float> xBufLeft = GetTempBuf<float>(0);
    Cast(xBufLeft, ubX1, RoundMode::CAST_NONE, realProcCount);
    inQueueX1.FreeTensor(ubX1);
    Mul(xBufLeft, xBufLeft, ubDY, realProcCount);  // x1 = x1 * dy
    inQueueDY.FreeTensor(ubDY);
}

__aicore__ inline void GeGluGradV2TanhBFP16::ComputeRightHalf(const int64_t& realProcCount) {
    LocalTensor<bfloat16_t> ubX2 = inQueueX2.DeQue<bfloat16_t>();
    LocalTensor<float> xBufRight = GetTempBuf<float>(4);
    Cast(xBufRight, ubX2, RoundMode::CAST_NONE, realProcCount);
    inQueueX2.FreeTensor(ubX2);

    LocalTensor<float> xBufLeft = GetTempBuf<float>(0);
    ComputeGeluGrad(xBufLeft, xBufLeft, xBufRight, realProcCount);

    LocalTensor<bfloat16_t> outLocalRight = outQueueDX2.AllocTensor<bfloat16_t>();
    Cast(outLocalRight, xBufLeft, RoundMode::CAST_RINT, realProcCount);
    outQueueDX2.EnQue(outLocalRight);
}

}  // namespace GeGluGradV2Tanh

#endif  // GE_GLU_GRAD_V2_TANH_BFP16_H_