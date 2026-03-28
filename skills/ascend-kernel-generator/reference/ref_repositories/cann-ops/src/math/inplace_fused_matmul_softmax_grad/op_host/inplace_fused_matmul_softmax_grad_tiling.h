/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INPLACE_FUSED_MATMUL_SOFTMAX_GRAD_TILING_H
#define INPLACE_FUSED_MATMUL_SOFTMAX_GRAD_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"

namespace optiling {
// B*M*N
BEGIN_TILING_DATA_DEF(BaseTiling)
    TILING_DATA_FIELD_DEF(uint32_t, b);                    // batchLen
    TILING_DATA_FIELD_DEF(uint32_t, m);                    // rowLenPerBatch
    TILING_DATA_FIELD_DEF(uint32_t, n);                    // colLen
    TILING_DATA_FIELD_DEF(uint32_t, k);                    // k
    TILING_DATA_FIELD_DEF(uint32_t, rowLen);               // 行数
    TILING_DATA_FIELD_DEF(uint32_t, colLen);               // 列数
    TILING_DATA_FIELD_DEF(uint32_t, alignColLen);          // 每次计算的对齐列数
    TILING_DATA_FIELD_DEF(uint32_t, rowLenPerHeadCore);    // 头核处理行数
    TILING_DATA_FIELD_DEF(uint32_t, rowLenPerTailCore);    // 尾核处理行数
    TILING_DATA_FIELD_DEF(uint32_t, basicRowLenHeadCore);  // 头核每次计算的行数 类似于TILE_LENGTH
    TILING_DATA_FIELD_DEF(uint32_t, basicRowLenTailCore);  // 尾核每次计算的行数
    TILING_DATA_FIELD_DEF(uint32_t, realCoreNum);          // 实际使用的核数
    TILING_DATA_FIELD_DEF(uint32_t, headCoreNum);          // 使用的head核数
    TILING_DATA_FIELD_DEF(uint32_t, tailCoreNum);          // 使用的tail核数
    TILING_DATA_FIELD_DEF(uint32_t, blockNum);             // 32B块的元素个数
    // basicColLen
    TILING_DATA_FIELD_DEF(uint32_t, innerLoopTimes);       // 大shape下对colLen的循环次数
    TILING_DATA_FIELD_DEF(uint32_t, innerLoopHeadColLen);  // 大shape下对colLen的循环每次计算的列数
    TILING_DATA_FIELD_DEF(uint32_t, innerLoopTailColLen);  // 大shape下对colLen的循环的尾块大小
    TILING_DATA_FIELD_DEF(uint32_t, headLocalWorkSpaceSize);    // softmax高阶API所需的临时空间
    TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(BaseTilingOp, BaseTiling)

BEGIN_TILING_DATA_DEF(InplaceFusedMatmulSoftmaxGradTilingData)
    TILING_DATA_FIELD_DEF_STRUCT(BaseTiling, baseTilingData);
    TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, headSoftMaxGradTilingData);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(InplaceFusedMatmulSoftmaxGrad, InplaceFusedMatmulSoftmaxGradTilingData)

struct InplaceFusedMatmulSoftmaxGradCompileInfo {
    uint32_t inputDataByte = 2;
    ge::DataType inputDataType;

    uint32_t dataNumSingleUb = 1;  // UB空间可处理的最大数据量
    uint32_t blockNum = 1;        // 32B对齐使用 //
    uint32_t cacheLineLen = 1;     // 512B对齐使用
    
    uint32_t coreNum = 1; //
    uint32_t aivNum = 0; //
    uint32_t aicNum = 0; //
    uint64_t ubSize = 0; //
    uint64_t l1Size = 0;
    uint64_t l0ASize = 0;
    uint64_t l0BSize = 0;
    uint64_t l0CSize = 0;
    uint64_t sysWorkspaceSize = 0;
    platform_ascendc::SocVersion socVersion;
};     

struct InplaceFusedMatmulSoftmaxGradTilingParam {
    uint32_t optBaseRowLenHeadCore = 1;
    uint32_t optBaseRowLenTailCore = 1;
    uint32_t colLen = 1;
    uint32_t alignColLen = 1;

    uint32_t rowLenPerHeadCore = 0;
    uint32_t rowLenPerTailCore = 0;
    
    uint32_t coreNumUsed = 0;
    uint32_t headCoreNum = 0;
    uint32_t tailCoreNum = 0;
    uint32_t blockNum = 0;

    uint32_t innerLoopTimes = 0;
    uint32_t innerLoopHeadColLen = 1;
    uint32_t innerLoopTailColLen = 1;

    uint32_t b;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t mAligned; 
    uint32_t baseM; 
    uint32_t baseN;
    uint32_t headLocalWorkSpaceSize;
};

enum class InplaceFusedMatmulSoftmaxTilingKey : uint32_t {
    TILINGKEY_FP16_ALIGN = 11,
    TILINGKEY_FP16_UNALIGNED = 12,
    TILINGKEY_BF16_ALIGN = 21,
    TILINGKEY_BF16_UNALIGNED = 22,
    TILINGKEY_FP32_ALIGN = 31,
    TILINGKEY_FP32_UNALIGNED = 32,
};


}  // namespace optiling
#endif  // INPLACE_FUSED_MATMUL_SOFTMAX_GRAD_TILING_H