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
 * \file conv3dv2_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_CONV3D_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_CONV3D_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
    BEGIN_TILING_DATA_DEF(TConv3DTiling)
    TILING_DATA_FIELD_DEF(uint64_t, orgDo);
    TILING_DATA_FIELD_DEF(uint64_t, orgHo);
    TILING_DATA_FIELD_DEF(uint64_t, orgWo);
    TILING_DATA_FIELD_DEF(uint64_t, orgDi);
    TILING_DATA_FIELD_DEF(uint64_t, orgHi);
    TILING_DATA_FIELD_DEF(uint64_t, orgWi);
    TILING_DATA_FIELD_DEF(uint64_t, singleCoreDo);
    TILING_DATA_FIELD_DEF(uint64_t, singleCoreM);
    TILING_DATA_FIELD_DEF(uint64_t, singleCoreWo);
    TILING_DATA_FIELD_DEF(uint64_t, kL0xorgCoAlignN0);
    TILING_DATA_FIELD_DEF(uint64_t, kernelHxkernelW);
    TILING_DATA_FIELD_DEF(uint64_t, cin1xOriHixOriWixk0);
    TILING_DATA_FIELD_DEF(uint64_t, oriHixOriWixk0);
    TILING_DATA_FIELD_DEF(uint64_t, oriWixk0);
    TILING_DATA_FIELD_DEF(uint64_t, orgHixWi);
    TILING_DATA_FIELD_DEF(uint64_t, orgHoxWo);
    TILING_DATA_FIELD_DEF(uint32_t, orgCi);
    TILING_DATA_FIELD_DEF(uint32_t, kernelD);
    TILING_DATA_FIELD_DEF(uint32_t, kernelH);
    TILING_DATA_FIELD_DEF(uint32_t, kernelW);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreCo);
    TILING_DATA_FIELD_DEF(uint32_t, orgCo);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreGroups);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreGroupOpt);
    TILING_DATA_FIELD_DEF(uint32_t, groups);
    TILING_DATA_FIELD_DEF(uint32_t, enlarge);
    TILING_DATA_FIELD_DEF(uint32_t, strideH);
    TILING_DATA_FIELD_DEF(uint32_t, strideW);
    TILING_DATA_FIELD_DEF(uint32_t, strideD);
    TILING_DATA_FIELD_DEF(uint32_t, dilationH);
    TILING_DATA_FIELD_DEF(uint32_t, dilationW);
    TILING_DATA_FIELD_DEF(uint32_t, dilationD);
    TILING_DATA_FIELD_DEF(uint32_t, padHead);
    TILING_DATA_FIELD_DEF(uint32_t, padTail);
    TILING_DATA_FIELD_DEF(uint32_t, padUp);
    TILING_DATA_FIELD_DEF(uint32_t, padDown);
    TILING_DATA_FIELD_DEF(uint32_t, padLeft);
    TILING_DATA_FIELD_DEF(uint32_t, padRight);
    TILING_DATA_FIELD_DEF(uint32_t, mL0);
    TILING_DATA_FIELD_DEF(uint32_t, woL0);
    TILING_DATA_FIELD_DEF(uint32_t, kL0);
    TILING_DATA_FIELD_DEF(uint32_t, nL0);
    TILING_DATA_FIELD_DEF(uint32_t, kAL1);
    TILING_DATA_FIELD_DEF(uint32_t, kAL1Tail);
    TILING_DATA_FIELD_DEF(uint32_t, kBL1);
    TILING_DATA_FIELD_DEF(uint32_t, kBL1Tail);
    TILING_DATA_FIELD_DEF(uint32_t, nBL1);
    TILING_DATA_FIELD_DEF(uint32_t, mAL1);
    TILING_DATA_FIELD_DEF(uint32_t, woL1);
    TILING_DATA_FIELD_DEF(uint32_t, KBL1Divk0);
    TILING_DATA_FIELD_DEF(uint32_t, KBL1TailDivk0);
    TILING_DATA_FIELD_DEF(uint32_t, nBL1DivnL0);
    TILING_DATA_FIELD_DEF(uint32_t, mAL1DivmL0);
    TILING_DATA_FIELD_DEF(uint32_t, fmapKStride);
    TILING_DATA_FIELD_DEF(uint32_t, weightKStride);
    TILING_DATA_FIELD_DEF(uint32_t, cinOffsetBlockInGM);
    TILING_DATA_FIELD_DEF(uint32_t, coutOffsetBlock);
    TILING_DATA_FIELD_DEF(uint32_t, nL1DivBlockSize);
    TILING_DATA_FIELD_DEF(uint32_t, cin1InAL1);
    TILING_DATA_FIELD_DEF(uint32_t, cin1InAL1Tail);
    TILING_DATA_FIELD_DEF(uint32_t, cinBInCore);
    TILING_DATA_FIELD_DEF(uint32_t, cinBTailInCore);
    TILING_DATA_FIELD_DEF(uint32_t, nL0xk0);
    TILING_DATA_FIELD_DEF(uint32_t, mStep);
    TILING_DATA_FIELD_DEF(uint32_t, kStep);
    TILING_DATA_FIELD_DEF(uint32_t, nStep);
    TILING_DATA_FIELD_DEF(uint32_t, aL1SpaceSize);
    TILING_DATA_FIELD_DEF(uint32_t, multiNBL1);
    TILING_DATA_FIELD_DEF(uint32_t, pBufferFlag);
    TILING_DATA_FIELD_DEF(uint32_t, groupOpt);
    TILING_DATA_FIELD_DEF(uint32_t, cinOpt);
    TILING_DATA_FIELD_DEF(uint32_t, coutOpt);
    TILING_DATA_FIELD_DEF(uint32_t, mUB);
    TILING_DATA_FIELD_DEF(uint32_t, nUB);
    TILING_DATA_FIELD_DEF(uint32_t, scaleAndBiasLoadType);
    TILING_DATA_FIELD_DEF(uint32_t, workspaceSize);
    TILING_DATA_FIELD_DEF(int8_t, offsetx);
    TILING_DATA_FIELD_DEF(int8_t, roundMode);
    TILING_DATA_FIELD_DEF(uint8_t, hasBias);
    TILING_DATA_FIELD_DEF(uint8_t, hasScale);
    TILING_DATA_FIELD_DEF(uint8_t, bl1FullLoad);
    TILING_DATA_FIELD_DEF(uint8_t, al1FullLoad);
    TILING_DATA_FIELD_DEF(uint8_t, bl1BypassFlag);
    TILING_DATA_FIELD_DEF(uint8_t, iterateMNOrder);
    TILING_DATA_FIELD_DEF(uint8_t, biasFullLoadFlag);
    TILING_DATA_FIELD_DEF(uint8_t, fixpParamsFullLoadFlag);
    TILING_DATA_FIELD_DEF(uint8_t, hf32Enable);
    TILING_DATA_FIELD_DEF(uint8_t, hf32TransMode);
    TILING_DATA_FIELD_DEF(uint8_t, quantType);
    TILING_DATA_FIELD_DEF(uint8_t, resvered1);
    TILING_DATA_FIELD_DEF(uint8_t, resvered2);
    TILING_DATA_FIELD_DEF(uint8_t, resvered3);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(TConv3DTilingOp, TConv3DTiling);

    BEGIN_TILING_DATA_DEF(Conv3DRunInfo)
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, cin);
    TILING_DATA_FIELD_DEF(uint32_t, din);
    TILING_DATA_FIELD_DEF(uint64_t, hin);
    TILING_DATA_FIELD_DEF(uint64_t, win);
    TILING_DATA_FIELD_DEF(uint32_t, cout);
    TILING_DATA_FIELD_DEF(uint32_t, kd);
    TILING_DATA_FIELD_DEF(uint32_t, kh);
    TILING_DATA_FIELD_DEF(uint32_t, kw);
    TILING_DATA_FIELD_DEF(uint32_t, dout);
    TILING_DATA_FIELD_DEF(uint64_t, hout);
    TILING_DATA_FIELD_DEF(uint64_t, wout);
    TILING_DATA_FIELD_DEF(uint32_t, batchDim);
    TILING_DATA_FIELD_DEF(uint32_t, doDim);
    TILING_DATA_FIELD_DEF(uint32_t, mDim);
    TILING_DATA_FIELD_DEF(uint32_t, wDim);
    TILING_DATA_FIELD_DEF(uint32_t, nDim);
    TILING_DATA_FIELD_DEF(uint32_t, groupDim);
    TILING_DATA_FIELD_DEF(uint32_t, strideH);
    TILING_DATA_FIELD_DEF(uint32_t, strideW);
    TILING_DATA_FIELD_DEF(uint32_t, strideD);
    TILING_DATA_FIELD_DEF(uint32_t, dilationH);
    TILING_DATA_FIELD_DEF(uint32_t, dilationW);
    TILING_DATA_FIELD_DEF(uint32_t, dilationD);
    TILING_DATA_FIELD_DEF(uint32_t, padHead);
    TILING_DATA_FIELD_DEF(uint32_t, padTail);
    TILING_DATA_FIELD_DEF(uint32_t, padUp);
    TILING_DATA_FIELD_DEF(uint32_t, padDown);
    TILING_DATA_FIELD_DEF(uint32_t, padLeft);
    TILING_DATA_FIELD_DEF(uint32_t, padRight);
    TILING_DATA_FIELD_DEF(uint32_t, groups);
    TILING_DATA_FIELD_DEF(uint32_t, enlarge);
    TILING_DATA_FIELD_DEF(uint32_t, cinOpt);
    TILING_DATA_FIELD_DEF(uint32_t, coutOpt);
    TILING_DATA_FIELD_DEF(uint32_t, groupOpt);
    TILING_DATA_FIELD_DEF(uint8_t, hasBias);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(Conv3DRunInfoOp, Conv3DRunInfo)

    BEGIN_TILING_DATA_DEF(Conv3DTilingData)
    TILING_DATA_FIELD_DEF_STRUCT(TConv3DTiling, conv3dApiTiling);
    TILING_DATA_FIELD_DEF_STRUCT(Conv3DRunInfo, conv3dRunInfo);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(Conv3DV2, Conv3DTilingData)
}
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_CONV3D_TILING_H
