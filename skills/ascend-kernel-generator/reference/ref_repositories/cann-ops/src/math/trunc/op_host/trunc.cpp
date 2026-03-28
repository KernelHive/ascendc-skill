/**
 * Copyright (C) Henan KunLun Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
#include "trunc_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t REPEAT_SIZE = 256;
constexpr uint32_t DOUBLE_BUFFER = 2;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TruncTilingData tiling; // TilingData

    constexpr bool DOUBLE_BUFFER_ENABLE = true; // 是否启用 DoubleBuffer
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo()); // Ascend 平台信息
    uint64_t ubSize; // UB容量
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t CORE_NUM = 1; // 获取设备的核心数
    
    // 获取输入shape信息
    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize(); //输入数量
    uint32_t inputBytes = GetSizeByDataType(context->GetInputDesc(0)->GetDataType()); //输入类型
    uint32_t inputLength = inputBytes * inputNum; //输入长度

    // 可使用的ub空间 输入3输出1，手动考虑双缓存
    uint32_t ubDataNumber = 6;//(inputBytes == 2) ? 10 : 6;

    // The number of 32B data blocks that can be used for each data. DOUBLE BUFFER is already counted here
    uint32_t tileBlockNum = (ubSize / BLOCK_SIZE / DOUBLE_BUFFER) / ubDataNumber; //每个ub段可用的空间块数
    if (inputBytes == 0){
        return ge::GRAPH_FAILED;
    }
    uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / inputBytes; //每次处理的数据量

    // Input data for 32B alignment
    uint32_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE); //输入长度 对齐处理
    // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number of audits is the actual number of audits
    uint32_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE;// 输入数据需要多少空间块
    
    //  chunks are calculated and sliced several times using the number of data on each core
    uint32_t CoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes; //对齐空间后的输入数量
    uint32_t TileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? TileNum : TileNum + 1; //需要循环处理几次
    // Tail block calculation for  chunks of data
    uint32_t TailDataNum = CoreDataNum - (tileDataNum * TileNum);
    TailDataNum = TailDataNum == 0 ? tileDataNum : TailDataNum; //最后一次需要处理的数据量
    {
        // 核内切分数据
        tiling.set_Len(CoreDataNum); // 对齐空间后的输入数量
        tiling.set_fLen(TileNum); // 每次处理的数据量
        tiling.set_fNum(finalTileNum); // 需要循环处理几次
        tiling.set_tLen(TailDataNum); // 最后一次需要处理的数据量
    }
    // 保存Tiling数据
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

     // 设置任务核心数
    context->SetBlockDim(1);
    constexpr size_t userWorkspaceSize = 0;
    context->GetWorkspaceSizes(1)[0] = ascendcPlatform.GetLibApiWorkSpaceSize() + userWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

}

namespace ops {
class Trunc : public OpDef {
public:
    explicit Trunc(const char* name) : OpDef(name)
    {
        this->Input("input_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16,ge::DT_INT8,ge::DT_INT32,ge::DT_UINT8 })
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND })
            .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND });
        this->Output("output_y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16,ge::DT_INT8,ge::DT_INT32,ge::DT_UINT8 })
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND })
            .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND });

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b")
            .AddConfig("ascend310b");
    }
};

OP_ADD(Trunc);
}

