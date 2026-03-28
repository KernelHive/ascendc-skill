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
 * \file mse_loss_grad_v2_base.h
 * \brief
 */
#ifndef _MSE_LOSS_GRAD_V2_BASE_H
#define _MSE_LOSS_GRAD_V2_BASE_H
#pragma once
#include <cstdint>
#include "kernel_operator.h"
using namespace AscendC;

__aicore__ constexpr TQueConfig GetMyTQueConfig(bool nd2nzIn, bool nz2ndIn, bool scmBlockGroupIn,
    uint32_t bufferLenIn, uint32_t bufferNumberIn, uint32_t consumerSizeIn, const TPosition consumerIn[]) {
    return {
        nd2nzIn,
        nz2ndIn,
        scmBlockGroupIn,
        bufferLenIn,
        bufferNumberIn,
        consumerSizeIn,
        {consumerIn[1], consumerIn[2], consumerIn[3], consumerIn[4], consumerIn[5],
            consumerIn[6], consumerIn[7]}
    };
}

const constexpr TPosition tp[8] = {TPosition::MAX, TPosition::MAX, TPosition::MAX, TPosition::MAX,
    TPosition::MAX, TPosition::MAX, TPosition::MAX, TPosition::MAX};
constexpr TQueConfig conf = GetMyTQueConfig(false, false, false, 0, 1, 0, tp);

template <typename inType>
class KernelMseLossGradBase {
public:
    __aicore__ inline KernelMseLossGradBase(){}
    template <typename T1>
    __aicore__ inline T1 CeilAlign(T1 a) {
        return (a + 32 - 1) / 32 * 32;
    }
        
protected:
    float cof;
    bool pad = false;
    uint64_t totalLength;
    uint64_t gmSize;
    uint64_t blockLength;
    uint64_t tileNum;
    uint64_t tileLength;
    uint64_t tileLengthPtr;
    uint64_t tileLengthAlign;
    int32_t bufferNum;
};
#endif // _MSE_LOSS_GRAD_V2_BASE_H