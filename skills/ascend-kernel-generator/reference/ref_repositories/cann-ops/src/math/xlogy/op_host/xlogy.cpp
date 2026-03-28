/*
* 版权所有 (c) 华为技术有限公司 2024
*/
// Copyright 2024 Huawei Technologies Co., Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License. 
#include <algorithm>
#include "xlogy_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

constexpr int32_t inputVarNum = 2;
constexpr int32_t maxDimNum = 64;
namespace optiling {
    const uint32_t BLOCK_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context) {    
    XlogyTilingData tiling;
    uint64_t ubSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize); //获取硬件平台存储空间 UB 的内存大小

    //获取输入shape信息
    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize(); //输入数量
    uint32_t inputBytes = GetSizeByDataType(context->GetInputDesc(0)->GetDataType()); //输入类型
    inputBytes = (inputBytes == 0) ? GetSizeByDataType(ge::DT_FLOAT) : inputBytes;
    uint32_t inputLength = inputBytes * inputNum; //输入长度

    //可使用的ub空间 输入3输出1，手动考虑双缓存
    uint32_t ubDataNumber = 10;//(inputBytes == 2) ? 10 : 10;

    // The number of 32B data blocks that can be used for each data. DOUBLE BUFFER is already counted here
    uint32_t tileBlockNum = (ubSize / BLOCK_SIZE) / ubDataNumber; //每个ub段可用的空间块数
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
    
    tiling.set_CoreDataNum(CoreDataNum);  //对齐空间后的输入数量
    tiling.set_finalTileNum(finalTileNum);//需要循环处理几次
    tiling.set_tileDataNum(tileDataNum); //每次处理的数据量
    tiling.set_TailDataNum(TailDataNum); //最后一次需要处理的数据量

    uint32_t x1_length = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t x2_length = context->GetInputShape(1)->GetStorageShape().GetShapeSize();
    int64_t numshapes = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    int64_t shape[maxDimNum * inputVarNum], shapefull[maxDimNum];
    for (int k = 0; k < inputVarNum; ++k) {
        int64_t *ss = &shape[k * maxDimNum];
        const gert::StorageShape* inputshape = context->GetInputShape(k);
        for (int i = 0; i < inputshape->GetStorageShape().GetDimNum(); i++) {
            ss[i] = inputshape->GetStorageShape().GetDim(i);
        }
    }
    for (int k = 0; k < numshapes; ++k) {
        int64_t *ss = &shape[0];
        int64_t *sf = &shapefull[0];
        sf[k] = (ss[k] > ss[k + maxDimNum]) ? ss[k] : ss[k + maxDimNum];   
    }
    tiling.set_numshapes(numshapes);
    tiling.set_shape(shape);
    tiling.set_shapefull(shapefull);
    tiling.set_x1_length(x1_length);
    tiling.set_x2_length(x2_length);
    
    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class Xlogy : public OpDef {
public:
    explicit Xlogy(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape);
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b")
                      .AddConfig("ascend910b");
    }
};

OP_ADD(Xlogy);
}
