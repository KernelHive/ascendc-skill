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
 * \file apply_fused_ema_adam_base.h
 * \brief
 */

#ifndef APPLY_FUSED_EMA_ADAM_BASE_H
#define APPLY_FUSED_EMA_ADAM_BASE_H

#include "kernel_operator.h"

namespace FusedEmaAdam {
using namespace AscendC;
constexpr int32_t BYTE_ONE_BLOCK = 32;
constexpr int32_t BLOCK_SIZE_FOR_FLOAT32 = 8;
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t INDEX_VAR = 0;
constexpr int32_t INDEX_M = 1;
constexpr int32_t INDEX_V = 2;
constexpr int32_t INDEX_S = 3;
constexpr int32_t INDEX_GRAD = 4;

template <typename T>
class FusedEmaAdamBase {
public:
    __aicore__ inline FusedEmaAdamBase(){};
    __aicore__ inline void InitData(const ApplyFusedEmaAdamTilingData& tiling);
    __aicore__ inline float ScalarPow(TBuf<QuePosition::VECCALC> powTBuf1, TBuf<QuePosition::VECCALC> powTBuf2,
                                      TQue<QuePosition::VECIN, 1> inQue, float x, float y);
    __aicore__ inline void PipeVM2();
    __aicore__ inline void PipeM2V();
    __aicore__ inline void PipeVM3();

protected:
    float lr;
    float emaDecay;
    float beta1;
    float beta2;
    float eps;
    uint64_t mode;
    uint64_t biasCorrection;
    float beta1Correction = 1.0;
    float beta2Correction = 1.0;
    float weightDecay;
    uint64_t frontCoreNum;
    uint64_t tailCoreNum;
    uint64_t coreCalcNum;
    uint64_t coreCalcMax;
    uint64_t loopNum;
    uint64_t frontCalcExtra;
    uint64_t tailCalcExtra;
};

template <typename T>
__aicore__ inline void FusedEmaAdamBase<T>::InitData(const ApplyFusedEmaAdamTilingData& tiling) {
    lr = tiling.lr;
    emaDecay = tiling.emaDecay;
    beta1 = tiling.beta1;
    beta2 = tiling.beta2;
    eps = tiling.eps;
    mode = tiling.mode;
    biasCorrection = tiling.biasCorrection;
    weightDecay = tiling.weightDecay;
    frontCoreNum = tiling.frontCoreNum;
    tailCoreNum = tiling.tailCoreNum;
    coreCalcNum = tiling.coreCalcNum;
    coreCalcMax = tiling.coreCalcMax;
    loopNum = tiling.loopNum;
    frontCalcExtra = tiling.frontCalcExtra;
    tailCalcExtra = tiling.tailCalcExtra;
}

template <typename T>
__aicore__ inline float FusedEmaAdamBase<T>::ScalarPow(TBuf<QuePosition::VECCALC> powTBuf1,
    TBuf<QuePosition::VECCALC> powTBuf2, TQue<QuePosition::VECIN, 1> inQue, float x, float y) {
    LocalTensor<T> inLocal = inQue.AllocTensor<T>();
    LocalTensor<uint8_t> interpreTensor = inLocal.template ReinterpretCast<uint8_t>();
    LocalTensor<float> baseLocal = powTBuf1.Get<float>();
    LocalTensor<float> outLocal = powTBuf2.Get<float>();
    Duplicate(baseLocal, x, BLOCK_SIZE_FOR_FLOAT32);
    Power<float, false>(outLocal, baseLocal, y, interpreTensor, BLOCK_SIZE_FOR_FLOAT32);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    float result = outLocal.GetValue(0);
    inQue.FreeTensor(inLocal);
    return result;
}

template <typename T>
__aicore__ inline void FusedEmaAdamBase<T>::PipeVM2() {
    event_t eventVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventVToMTE2);
    WaitFlag<HardEvent::V_MTE2>(eventVToMTE2);
}

template <typename T>
__aicore__ inline void FusedEmaAdamBase<T>::PipeM2V() {
    event_t eventMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2ToV);
}
    
template <typename T>
__aicore__ inline void FusedEmaAdamBase<T>::PipeVM3() {
    event_t eventVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventVToMTE3);
}

} // namespace FusedEmaAdam 

#endif  // APPLY_FUSED_EMA_ADAM_BASE_H