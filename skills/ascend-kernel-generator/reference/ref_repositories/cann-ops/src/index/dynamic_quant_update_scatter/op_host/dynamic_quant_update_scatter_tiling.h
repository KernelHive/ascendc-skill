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
 * \file dynamic_quant_update_scatter_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_QUANT_UPDATE_SCATTER_TILING_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_QUANT_UPDATE_SCATTER_TILING_H
#include <algorithm>
#include "tiling/tiling_api.h"
#include "tiling/tiling_base.h"
#include "register/op_def_registry.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(DynamicQuantUpdateScatterTilingData)
TILING_DATA_FIELD_DEF(int64_t, coreNum);        // ceil_div(updates_shape[0] * updates_shape[1], eachCoreBsNum)
TILING_DATA_FIELD_DEF(int64_t, eachCoreBsNum);  // ceil_div(updates_shape[0] * updates_shape[1], ori_core_num)
TILING_DATA_FIELD_DEF(int64_t, lastCoreBsNum);  // updates_shape[0] * updates_shape[1] - eachCoreBsNum * (core_num - 1)
TILING_DATA_FIELD_DEF(int64_t, updateAxisShape);    // updates_shape[axis]
TILING_DATA_FIELD_DEF(int64_t, srcBsStride);        // updates_shape[2] * updates_shape[3]
TILING_DATA_FIELD_DEF(int64_t, dstBsStride);        // data_shape[2] *  data_shape[3]
TILING_DATA_FIELD_DEF(int64_t, indexElements);      // index_shape[0]
TILING_DATA_FIELD_DEF(int64_t, numHead);            // data_shape[1]
TILING_DATA_FIELD_DEF(int64_t, sizePerHead);        // not axis shape, data_shape[2] or data_shape[3]
TILING_DATA_FIELD_DEF(int64_t, dataAxisShape);      // data_shape of axis
TILING_DATA_FIELD_DEF(int64_t, numOneBlock);        // 32 / dtype_size
TILING_DATA_FIELD_DEF(int64_t, innerLoopEle);       // nums of ele per inner loop
TILING_DATA_FIELD_DEF(int64_t, indicesShapeRank);   // rank of indices shape
TILING_DATA_FIELD_DEF(int64_t, srcFirBsStride);     // updates_shape[1] * updates_shape[2] * updates_shape[3]
TILING_DATA_FIELD_DEF(int64_t, dstFirSecBsStride);  // data_shape[1] * data_shape[2] *  data_shape[3]
TILING_DATA_FIELD_DEF(int64_t, updateDim0);         // updates_shape[0]
TILING_DATA_FIELD_DEF(int64_t, updateDim1);         // updates_shape[1]
TILING_DATA_FIELD_DEF(int64_t, varElements);
TILING_DATA_FIELD_DEF(int64_t, varScalesElements);
TILING_DATA_FIELD_DEF(int64_t, updatesElements);
TILING_DATA_FIELD_DEF(int64_t, quantReptNum);
TILING_DATA_FIELD_DEF(int64_t, varOrigLastDimSize);
TILING_DATA_FIELD_DEF(int64_t, sizeSrcPerHead);  // not axis shape, data_shape[2] or data_shape[3]
TILING_DATA_FIELD_DEF(int64_t, innerLoopFullRpt);
TILING_DATA_FIELD_DEF(int64_t, innerLoopTimes);  // inner loop time
TILING_DATA_FIELD_DEF(int64_t, innerLoopTail);   // nums of ele tail inner loop
TILING_DATA_FIELD_DEF(int64_t, innerLoopTimesLastCore);
TILING_DATA_FIELD_DEF(int64_t, innerLoopTailLastCore);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DynamicQuantUpdateScatter, DynamicQuantUpdateScatterTilingData)
struct DynamicQuantUpdateScatterCompileInfo {
  int64_t coreNum;
  int64_t ubSize;
};
}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_QUANT_TILING_H
