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
 * @file less_equal.cpp
 */
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "less_equal_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

constexpr uint32_t DATA_SIZE_4 = 4;
constexpr uint32_t DATA_SIZE_2 = 2;
constexpr uint32_t DATA_SIZE_1 = 1;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t TWO = 2;

static uint32_t GetDataTypeSize(ge::DataType dataType) {
  switch (dataType) {
    case ge::DT_FLOAT:
      return DATA_SIZE_4;
    case ge::DT_FLOAT16:
      return DATA_SIZE_2;
    case ge::DT_INT8:
      return DATA_SIZE_1;
    case ge::DT_INT32:
      return DATA_SIZE_4;
    default:
      return DATA_SIZE_4;
  }
}
struct LessEqualTilingParam {
  uint32_t totalLength;
  uint32_t pad32;
  uint32_t padMax;
  uint32_t padTemp;
  uint32_t maxBlockLength;
  uint32_t tileNumMean = 0;
  uint32_t tileNumEnd = 0;
  uint32_t tileLengthMean = 0;
  uint32_t tileLengthEnd = 0;
  uint32_t blockLengthMean = 0;
  uint32_t blockLengthEnd = 0;
};

static void ProcessLess32B(LessEqualTilingParam& param) {
  param.blockLengthMean = param.pad32;
  param.blockLengthEnd = param.totalLength;
  param.tileNumMean = static_cast<uint32_t>(1);
  param.tileNumEnd = static_cast<uint32_t>(1);
  param.tileLengthMean = param.totalLength;
  param.tileLengthEnd = param.totalLength;
}

static void ProcessLessEqual(LessEqualTilingParam& param, uint32_t coreNum) {
  if (param.maxBlockLength > param.padMax) {  // maxBlockLength大于padMax时对maxBlockLength进行判定
    param.padTemp = static_cast<uint32_t>(0);
    for (uint32_t i = param.padMax / static_cast<uint32_t>(2); i <= param.padMax; i += param.pad32) {
      param.padTemp = param.maxBlockLength % i == static_cast<uint32_t>(0) ? i : param.padTemp;
    }
    if (param.padTemp) {  // 如果maxBlockLength可以被PadTemp整除，那么padTemp就是tilelength
      param.blockLengthMean = param.maxBlockLength;
      param.blockLengthEnd = param.totalLength - param.blockLengthMean * (coreNum - static_cast<uint32_t>(1));
      param.tileNumMean = param.blockLengthMean / param.padTemp;
      param.tileNumEnd = param.tileNumMean;
      param.tileLengthMean = param.padTemp;
      param.tileLengthEnd = param.blockLengthEnd - param.padTemp * (param.tileNumEnd - static_cast<uint32_t>(1));
    } else {  // 如果maxBlockLength不能被PadTemp整除，那么padMax就是tilelength
      param.blockLengthMean = param.maxBlockLength - param.maxBlockLength % param.padMax;
      param.blockLengthEnd = param.totalLength - param.blockLengthMean * (coreNum - static_cast<uint32_t>(1));
      param.tileNumMean = param.blockLengthMean / param.padMax;
      param.tileNumEnd = param.blockLengthEnd % param.padMax
                         ? param.blockLengthEnd / param.padMax + static_cast<uint32_t>(1)
                         : (param.blockLengthEnd /
                            param.padMax);  // 计算最后一个核心会不会多一个尾数块
      if (param.padMax >= param.blockLengthEnd) {
        param.tileNumEnd = static_cast<uint32_t>(1);
      }
      param.tileLengthMean = param.padMax;
      param.tileLengthEnd = param.blockLengthEnd - param.padMax * (param.tileNumEnd - static_cast<uint32_t>(1));  // 计算最后一个核心的尾数块长度
    }
  } else {  // maxBlockLength小于padMax时直接取maxBlockLength中的最大Pad32倍数
    if (param.maxBlockLength >= param.pad32) {  // maxBlockLength大于pad32时
      param.blockLengthMean = param.maxBlockLength - param.maxBlockLength % param.pad32;
      param.blockLengthEnd = param.totalLength - param.blockLengthMean * (coreNum - static_cast<uint32_t>(1));
      param.tileNumMean = static_cast<uint32_t>(1);  // 只有一个tileNum
      param.tileNumEnd = param.blockLengthEnd % param.pad32
                         ? param.blockLengthEnd / param.pad32 + static_cast<uint32_t>(1)
                         : param.blockLengthEnd / param.blockLengthMean;  // 如果尾块不能32B对齐则多分配一个尾块
      if (param.blockLengthMean >= param.blockLengthEnd) {
        param.tileNumEnd = static_cast<uint32_t>(1);
      }
      param.tileLengthMean = param.blockLengthMean;
      param.tileLengthEnd = param.blockLengthEnd - param.tileLengthMean * (param.tileNumEnd - static_cast<uint32_t>(1));  // 将尾数彻底分给最后一个核心的最后一个tile
    } else {  // maxBlockLength小于pad32时，前面的block优先分配32B数据
      param.blockLengthMean = param.pad32;
      param.blockLengthEnd = param.totalLength - param.blockLengthMean * (coreNum - static_cast<uint32_t>(1));
      param.tileNumMean = static_cast<uint32_t>(1);
      param.tileNumEnd = static_cast<uint32_t>(1);
      param.tileLengthMean = param.pad32;
      param.tileLengthEnd = param.blockLengthEnd;
    }
  }
}

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  LessEqualTilingData tiling;
  uint64_t ubSize;
  uint32_t bufferNum = 16;
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  auto coreNum = ascendcPlatform.GetCoreNumAiv();
  
  ge::DataType dataType = context->GetInputDesc(0)->GetDataType();
  uint32_t dataSize = GetDataTypeSize(dataType);

  LessEqualTilingParam param;
  param.pad32 = BLOCK_SIZE;
  
  if (bufferNum == static_cast<uint32_t>(0)) return ge::GRAPH_FAILED;// non zero check
  param.padMax = static_cast<uint32_t>(ubSize) / bufferNum;
  if (dataSize == static_cast<uint32_t>(0)) return ge::GRAPH_FAILED;// non zero check
  param.padMax = param.padMax / dataSize;
  constexpr uint32_t TWO_BLOCKSIZE = TWO * BLOCK_SIZE;
  if (TWO_BLOCKSIZE == static_cast<uint32_t>(0) ) return ge::GRAPH_FAILED;// non zero check
  param.padMax = param.padMax / TWO_BLOCKSIZE * TWO_BLOCKSIZE;
  param.totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
  if (param.totalLength < param.pad32 * coreNum) {
    coreNum =param.totalLength % param.pad32 ? param.totalLength / param.pad32 + static_cast<uint32_t>(1) : param.totalLength / param.pad32;
  }

  context->SetBlockDim(coreNum);
  tiling.set_totalLength(param.totalLength);
  
  // 如果总数据比32B还小，直接当尾数处理
  if (param.totalLength < param.pad32) {
    ProcessLess32B(param);
  } else {  // 总数据至少比32B大时
    // 总数据至少比32B大时
    uint32_t realTotalLength =
        param.totalLength % (param.pad32 * coreNum)
            ?  // 补足totalLength到32B倍核心数的整数倍
            ((param.totalLength / (param.pad32 * coreNum)) + 1) * (param.pad32 * coreNum)
            : param.totalLength;
    if (coreNum == 0) {
      return ge::GRAPH_FAILED;
    }
    param.maxBlockLength = realTotalLength / coreNum;
    if (realTotalLength - param.totalLength > param.maxBlockLength) {
      param.maxBlockLength = param.totalLength / coreNum;
    }
    ProcessLessEqual(param, coreNum);
  }
  tiling.set_totalLength(param.totalLength);
  tiling.set_tileNumMean(param.tileNumMean);
  tiling.set_tileNumEnd(param.tileNumEnd);
  tiling.set_tileLengthMean(param.tileLengthMean);
  tiling.set_tileLengthEnd(param.tileLengthEnd);
  tiling.set_blockLengthMean(param.blockLengthMean);
  tiling.set_blockLengthEnd(param.blockLengthEnd);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  size_t* currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = static_cast<size_t>(0);
  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const gert::Shape* x1_shape = context->GetInputShape(0);
  gert::Shape* y_shape = context->GetOutputShape(0);
  *y_shape = *x1_shape;
  return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class LessEqual : public OpDef {
 public:
  explicit LessEqual(const char* name) : OpDef(name) {
    this->Input("x1")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat(
            {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("x2")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat(
            {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat(
            {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

    this->SetInferShape(ge::InferShape);

    this->AICore().SetTiling(optiling::TilingFunc);

    this->AICore().AddConfig("ascend310p")
                  .AddConfig("ascend310b")
                  .AddConfig("ascend910b");
  }
};

OP_ADD(LessEqual);
}  // namespace ops
