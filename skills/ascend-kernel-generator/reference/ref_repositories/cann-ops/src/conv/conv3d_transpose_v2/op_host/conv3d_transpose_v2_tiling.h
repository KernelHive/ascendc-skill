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
 * \file conv3d_transpose_v2_tiling.h
 * \brief
 */
#ifndef CONV3D_TRANSPOSE_V2_TILING_H
#define CONV3D_TRANSPOSE_V2_TILING_H

namespace optiling {

BEGIN_TILING_DATA_DEF(TConv3DInputV2Tiling)
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, cin);
TILING_DATA_FIELD_DEF(uint32_t, cout);
TILING_DATA_FIELD_DEF(uint32_t, cout1);
TILING_DATA_FIELD_DEF(uint32_t, cin1);
TILING_DATA_FIELD_DEF(uint32_t, cout1G);
TILING_DATA_FIELD_DEF(uint32_t, cin1G);
TILING_DATA_FIELD_DEF(uint32_t, c0);
TILING_DATA_FIELD_DEF(uint32_t, c0Bits);
TILING_DATA_FIELD_DEF(uint32_t, dout);
TILING_DATA_FIELD_DEF(uint32_t, ho);
TILING_DATA_FIELD_DEF(uint32_t, wo);
TILING_DATA_FIELD_DEF(uint32_t, di);
TILING_DATA_FIELD_DEF(uint32_t, hi);
TILING_DATA_FIELD_DEF(uint32_t, wi);
TILING_DATA_FIELD_DEF(uint32_t, dk);
TILING_DATA_FIELD_DEF(uint32_t, hk);
TILING_DATA_FIELD_DEF(uint32_t, wk);
TILING_DATA_FIELD_DEF(uint32_t, group);
TILING_DATA_FIELD_DEF(uint32_t, strideD);
TILING_DATA_FIELD_DEF(uint32_t, strideH);
TILING_DATA_FIELD_DEF(uint32_t, strideW);
TILING_DATA_FIELD_DEF(uint32_t, padFront);
TILING_DATA_FIELD_DEF(uint32_t, padBack);
TILING_DATA_FIELD_DEF(uint32_t, padUp);
TILING_DATA_FIELD_DEF(uint32_t, padDown);
TILING_DATA_FIELD_DEF(uint32_t, padLeft);
TILING_DATA_FIELD_DEF(uint32_t, padRight);
TILING_DATA_FIELD_DEF(uint32_t, backpropPadUp);
TILING_DATA_FIELD_DEF(uint32_t, backpropPadDown);
TILING_DATA_FIELD_DEF(uint32_t, backpropPadLeft);
TILING_DATA_FIELD_DEF(uint32_t, backpropPadRight);
TILING_DATA_FIELD_DEF(uint32_t, dilationD);
TILING_DATA_FIELD_DEF(uint32_t, dilationH);
TILING_DATA_FIELD_DEF(uint32_t, dilationW);
TILING_DATA_FIELD_DEF(uint32_t, al0Pbuffer);
TILING_DATA_FIELD_DEF(uint32_t, bl0Pbuffer);
TILING_DATA_FIELD_DEF(uint32_t, cl0Pbuffer);
TILING_DATA_FIELD_DEF(uint32_t, al1Pbuffer);
TILING_DATA_FIELD_DEF(uint32_t, bl1Pbuffer);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreGroup);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreCout);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreCout1);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreCin1);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreDin);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreHo);
TILING_DATA_FIELD_DEF(uint32_t, baseM);
TILING_DATA_FIELD_DEF(uint32_t, baseK);
TILING_DATA_FIELD_DEF(uint32_t, baseN);
TILING_DATA_FIELD_DEF(uint32_t, baseD);
TILING_DATA_FIELD_DEF(uint32_t, baseBatch);
TILING_DATA_FIELD_DEF(uint32_t, baseGroup);
TILING_DATA_FIELD_DEF(uint32_t, stepM);
TILING_DATA_FIELD_DEF(uint32_t, stepN);
TILING_DATA_FIELD_DEF(uint32_t, stepKa);
TILING_DATA_FIELD_DEF(uint32_t, stepKb);
TILING_DATA_FIELD_DEF(uint32_t, stepBatch);
TILING_DATA_FIELD_DEF(uint32_t, stepGroup);
TILING_DATA_FIELD_DEF(uint32_t, iterateOrder);
TILING_DATA_FIELD_DEF(int32_t, hf32Flag);
TILING_DATA_FIELD_DEF(uint64_t, singleCoreBatch);
TILING_DATA_FIELD_DEF(uint64_t, singleCoreM);
TILING_DATA_FIELD_DEF(uint64_t, singleCoreCin);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TConv3DInputV2TilingOp, TConv3DInputV2Tiling);

BEGIN_TILING_DATA_DEF(Conv3DBackpropInputV2Params)
TILING_DATA_FIELD_DEF(uint32_t, batchDim);
TILING_DATA_FIELD_DEF(uint32_t, groupDim);
TILING_DATA_FIELD_DEF(uint32_t, mDim);
TILING_DATA_FIELD_DEF(uint32_t, kDim);
TILING_DATA_FIELD_DEF(uint32_t, nDim);
TILING_DATA_FIELD_DEF(uint32_t, dDim);
TILING_DATA_FIELD_DEF(uint64_t, coreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv3DBackpropInputV2ParamsOp, Conv3DBackpropInputV2Params)

BEGIN_TILING_DATA_DEF(Conv3DBackpropInputV2TilingData)
TILING_DATA_FIELD_DEF_STRUCT(Conv3DBackpropInputV2Params, params);
TILING_DATA_FIELD_DEF_STRUCT(TConv3DInputV2Tiling, conv3DDxTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv3DBackpropInputV2, Conv3DBackpropInputV2TilingData)
REGISTER_TILING_DATA_CLASS(Conv3DTransposeV2, Conv3DBackpropInputV2TilingData)
}  // namespace optiling
#endif  // CONV3D_TRANSPOSE_V2_TILING_H