/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file mul_sigmoid_tiling.h
 */

#pragma once
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MulSigmoidTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, formerCoreNum); // 非尾核数据需要使用的核数
  TILING_DATA_FIELD_DEF(uint64_t, formerCoreRowLen); // 非尾核每个核负责的row数
  TILING_DATA_FIELD_DEF(uint64_t, tailCoreNum); // 尾核数据需要使用的核数
  TILING_DATA_FIELD_DEF(uint64_t, tailCoreRowLen); // 尾核每个核负责的row数
  TILING_DATA_FIELD_DEF(uint64_t, tileLen); // 核内循环次数，tileNum*tileLength=8k or 32k
  TILING_DATA_FIELD_DEF(uint64_t, tileNum); // 
  TILING_DATA_FIELD_DEF(float, t1); // 输入标量1
  TILING_DATA_FIELD_DEF(float, t2); // 输入标量2
  TILING_DATA_FIELD_DEF(float, t3); // 输入标量3
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MulSigmoid, MulSigmoidTilingData)

class MulSigmoidTiling {
public:
  explicit MulSigmoidTiling(gert::TilingContext* context) : context_(context) {}

  ge::graphStatus DoTiling() {
    auto ret = GetShapeAttrsInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = ComputeSetTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = PostTiling();
    return ret;
  }

private:
  gert::TilingContext* context_;
  MulSigmoidTilingData tiling;

  uint32_t row_len;
  uint32_t col_len;
  uint32_t used_core_num;
  float t1;
  float t2;
  float t3;
  uint32_t MAX_TILING_LEN = 16384;
  uint32_t REPEAT_SIZE = 256;
  uint32_t DTYPE_SIZE = 2; // fp16

private:

  ge::graphStatus GetShapeAttrsInfo()
  {
    auto x1_shape = context_->GetInputShape(0)->GetStorageShape();
    auto x2_shape = context_->GetInputShape(1)->GetStorageShape();
    
    this->row_len = x1_shape.GetDim(0);
    this->col_len = x1_shape.GetDim(1);

    if (this->col_len > this->MAX_TILING_LEN && this->col_len != 32768) {
      std::cout <<  "mul sigmoid input x1 does not support dimension 1 that is not smaller than 16k or exactly 32k, but receives " << this->col_len << std::endl;
      return ge::GRAPH_FAILED;
    }

    if (this->col_len * this->DTYPE_SIZE % this->REPEAT_SIZE) {
      std::cout << "mul sigmoid input x1 does not support dimension 1 that is not 256 bytes aligned (128 half)" << std::endl;
      return ge::GRAPH_FAILED;
    }
    
    this->t1 = *(context_->GetAttrs()->GetAttrPointer<float>(0));
    this->t2 = *(context_->GetAttrs()->GetAttrPointer<float>(1));
    this->t3 = *(context_->GetAttrs()->GetAttrPointer<float>(2));

    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus ComputeSetTiling()
  {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t maxCoreNum = ascendcPlatform.GetCoreNum();

    uint32_t tileLen;
    uint32_t tileNum;
    uint32_t rowLen = this->row_len;
    if (this->col_len <= this->MAX_TILING_LEN) {
      tileLen = this->col_len;
      tileNum = 1;
    } else {
      tileLen = this->MAX_TILING_LEN;
      tileNum = this->col_len / tileLen;
    }

    uint32_t formerCoreNum, formerCoreRowLen, tailCoreNum, tailCoreRowLen, usedCoreNum;

    if (rowLen * tileNum <= maxCoreNum) {
      formerCoreNum = rowLen;
      formerCoreRowLen = 1;
      tailCoreNum = 0;
      tailCoreRowLen = 0;
      this->used_core_num = rowLen * tileNum;
    } else {
      uint32_t doubleCoreNum = maxCoreNum / tileNum;
      formerCoreNum = rowLen % doubleCoreNum;
      formerCoreRowLen = (rowLen + doubleCoreNum - 1) / doubleCoreNum;
      tailCoreNum = doubleCoreNum - formerCoreNum;
      tailCoreRowLen = rowLen / doubleCoreNum;
      this->used_core_num = doubleCoreNum * tileNum;
    }

    tiling.set_formerCoreNum(formerCoreNum);
    tiling.set_formerCoreRowLen(formerCoreRowLen);
    tiling.set_tailCoreNum(tailCoreNum);
    tiling.set_tailCoreRowLen(tailCoreRowLen);
    tiling.set_tileLen(tileLen);
    tiling.set_tileNum(tileNum);
    tiling.set_t1(this->t1);
    tiling.set_t2(this->t2);
    tiling.set_t3(this->t3);

    uint32_t maxSize;
    uint32_t minSize;
    std::vector<int64_t> shape_vec = {tileLen};
    ge::Shape shape(shape_vec);
    AscendC::GetSigmoidMaxMinTmpSize(shape, 2, false, maxSize, minSize);
    uint32_t bufSize = 8 * tileLen + tileLen / 8;

    if ((bufSize + minSize) > ubSize) {
      std::cout << "tiling length too large, ub space size unavailable" << std::endl;
    }

    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus PostTiling()
  {
    context_->SetBlockDim(this->used_core_num);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = 0;
    tiling.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context_->SetTilingKey(1);
    
    return ge::GRAPH_SUCCESS;
  }
};
}