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
 * \file quant_batch_matmul_v3_base.h
 * \brief
 */
#ifndef QUANT_BATCH_MATMUL_V3_BASE_H
#define QUANT_BATCH_MATMUL_V3_BASE_H

#include <cstdint>
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "kernel_type.h"
#include "lib/matmul_intf.h"

#define TemplateBasicType typename x1Type, typename x2Type, typename scaleType, typename yType, int x1Format, \
    int x2Format, bool aTrans, bool bTrans, class UPDATE_TYPE
#define TemplateBasicValue x1Type, x2Type, scaleType, yType, x1Format, x2Format, aTrans, bTrans, UPDATE_TYPE

constexpr uint32_t BMM_BLOCK_NUM = 16;
constexpr uint32_t K0_INT8 = 32;
constexpr uint32_t k0_FLOAT16 = 16;
constexpr uint32_t k0_FLOAT32 = 8;
constexpr int FORMAT_FRACTAL_NZ_INT = 29;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t M_N_TWO_DIMS = 2;

const uint32_t ROW_FIRST = 1;
const uint32_t COL_FIRST = 2;

constexpr uint16_t C2V_PING_FLAG = 0x4;
constexpr uint16_t C2V_PONG_FLAG = 0x5;
constexpr uint16_t V2C_PING_FLAG = 0x6;

constexpr MatmulConfig MM_CFG_VEC_ND2NZ = GetMDLConfig(false, false, false, true);
constexpr MatmulConfig MM_CFG_NO_PRELOAD{false, false, true, 0, 0, 0, false, false, false, false, false,
                                         0, 0, 0, 0, 0, 0, 0, true};
constexpr MatmulConfig MM_CFG_PRELOAD{false, false, true, 0, 0, 0, false, false, false, false, true,
                                      0, 0, 0, 0, 0, 0, 0, false};

struct L2CacheParam {
    uint32_t l2MCnt;
    uint32_t l2NCnt;
    uint32_t l2MCntTail;
    uint32_t l2NCntTail;
    uint32_t l2TotalTileCnt;
    uint32_t l2MCntUse;
    uint32_t l2NCntUse;
};

// 量化mm和mc2都用的输入输出地址结构体，不可随意修改
struct QBmmBlockOffset {
    uint64_t offsetA = 0;
    uint64_t offsetB = 0;
    uint64_t offsetScale = 0;
    uint64_t offsetBias = 0;
    uint64_t offsetPertoken = 0;
    uint64_t offsetC = 0;
};

// 量化mm和mc2都用的block输入信息结构体，不可随意修改
struct QBmmBaseBlockArgs {
    uint64_t index;
    uint64_t totalTileCnt;
    uint64_t singleCoreM; // 当前基本块计算大小
    uint64_t singleCoreN;
    uint64_t mTileCntL2;
    uint64_t nTileCntL2;
    uint64_t mTotalCnt;
    uint64_t nTotalCnt;
    uint64_t mCntUse;
    uint64_t nCntUse;
    uint64_t mTileAddrOffset;
    uint64_t nTileAddrOffset;
};

namespace DequantBmm {
template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return a > b ? a : b;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

__aicore__ inline uint64_t Align(uint64_t a, uint64_t b = 16)
{
    return (a + b - 1) / b * b;
}

__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline constexpr CubeFormat GetFormat(int format)
{
    if (format == FORMAT_FRACTAL_NZ_INT) {
        return CubeFormat::NZ;
    }
    return CubeFormat::ND;
}

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
__aicore__ inline void CalcDequantParams(uint32_t curAivM, uint32_t curAivN, AscendC::DequantParams &dequantParams,
                                         bool needUpdate = true)
{
    if (!needUpdate) {
        return;
    }
    uint32_t computedAivN = Align(curAivN, 8U);  // 8: 32B aligned for int32_t
    uint32_t ubResAlignedN = Align(curAivN);     // 16: sizeof(yType) is 2, 32B / 2
    if (computedAivN == ubResAlignedN) {
        // choose ddequat high performance
        dequantParams.m = 1;
        dequantParams.n = curAivM * computedAivN;
        dequantParams.calCount = computedAivN;
    } else {
        // general
        dequantParams.m = curAivM;
        dequantParams.n = computedAivN;
        dequantParams.calCount = curAivN;
    }
}

__aicore__ inline void SetGm2UbParams(AscendC::DataCopyParams &gm2UbParams, uint32_t curAivM, uint32_t curAivN)
{
    gm2UbParams.blockLen = curAivN * sizeof(int32_t);
    gm2UbParams.blockCount = curAivM;
    gm2UbParams.srcStride = 0;
}

template<typename yType>
__aicore__ inline void SetUb2GmParams(AscendC::DataCopyExtParams &ub2GmParams, uint32_t curAivM, uint32_t curAivN,
                                      uint32_t n)
{
    ub2GmParams.blockLen = curAivN * sizeof(yType);
    ub2GmParams.blockCount = curAivM;
    ub2GmParams.dstStride = (n - curAivN) * sizeof(yType);
}

__aicore__ inline void CopyMmOutToLocal(AscendC::LocalTensor<int32_t> &srcLocal, AscendC::GlobalTensor<int32_t> &curMmOutGm,
                                        AscendC::DataCopyParams &gm2UbParams, AscendC::DataCopyPadParams &padParams,
                                        uint32_t curAicAivOffset)
{
    DataCopyPad(srcLocal, curMmOutGm[curAicAivOffset], gm2UbParams, padParams);
    set_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(EVENT_ID0));
    wait_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(EVENT_ID0));
}

template<typename yType>
__aicore__ inline void CopyUbToGm(uint64_t yGmOffset, AscendC::DataCopyExtParams &ub2GmParams, AscendC::LocalTensor<yType> &dstLocal,
                                  AscendC::GlobalTensor<yType> &yGm, AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> &vecQueOut)
{
    DataCopyPad(yGm[yGmOffset], dstLocal, ub2GmParams);
    vecQueOut.FreeTensor(dstLocal);
}

template<typename scaleType>
__aicore__ inline void Bf16ScaleGm2Ub(AscendC::LocalTensor<scaleType> &scaleLocal, AscendC::GlobalTensor<scaleType> &scaleGm,
                                      AscendC::DataCopyPadParams &padParams, uint32_t curAivN, uint64_t offsetScale)
{
    AscendC::DataCopyParams scale2UbParams{1, 0, 0, 0};
    scale2UbParams.blockLen = curAivN * sizeof(scaleType);
    DataCopyPad(scaleLocal, scaleGm[offsetScale], scale2UbParams, padParams);
    set_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(EVENT_ID1));
    wait_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(EVENT_ID1));
}
#endif
}  // namespace DequantBmm

#endif  // QUANT_BATCH_MATMUL_V3_BASE_H