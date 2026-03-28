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
 * @file log_host.cpp
 */
#include "log_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

#include <cmath>

namespace optiling {
const uint64_t BLOCK_SIZE = 32;
const uint64_t BUFFER_NUM = 2;
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    LogTilingData tiling;
    uint64_t ubLength = 0;
    uint64_t bigCoreDataNum = 0;
    uint64_t bigCoreLoopNum = 0;
    uint64_t bigCoreTailDataNum = 0;
    
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubLength);
    auto coreNum = ascendcPlatform.GetCoreNum();

    // 获取输入数据信息
    uint64_t inputDataNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t dataTypeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), dataTypeLength);
    uint64_t inputLength = inputDataNum * dataTypeLength;
    
    if (coreNum == 0 || BLOCK_SIZE == 0) {
        return ge::GRAPH_FAILED;
    }

    // 计算UB可用空间
    uint64_t ubPartNum = (context->GetInputDesc(0)->GetDataType() == ge::DT_BF16) ? 3 : 2;
    uint64_t ubPartLength = ubLength / ubPartNum / BUFFER_NUM;
    uint64_t ubPartBlockNum = ubPartLength / BLOCK_SIZE;
    uint64_t ubPartDataNum = (ubPartBlockNum * BLOCK_SIZE) / dataTypeLength;

    // 32B对齐处理
    uint64_t inputLengthAlign32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
   
    // 核心数分配逻辑
    if (ubPartDataNum >= inputDataNum) {
        coreNum = 1;
    } else {
        coreNum = (coreNum <  inputLengthAlign32 / BLOCK_SIZE) ? coreNum : inputLengthAlign32 / BLOCK_SIZE;
    }
    
    // 计算核心数据分配
    uint64_t everyCoreInputBlockNum = inputLengthAlign32 / BLOCK_SIZE / coreNum;
    uint64_t tailBlockNum = (inputLengthAlign32 / BLOCK_SIZE) % coreNum;
    
    // 小核数据计算
    uint64_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / dataTypeLength;
    uint64_t smallCoreLoopNum = smallCoreDataNum / ubPartDataNum;
    smallCoreLoopNum = (everyCoreInputBlockNum % ubPartBlockNum) == 0 
                     ? smallCoreLoopNum : smallCoreLoopNum + 1;
    uint64_t smallCoreTailDataNum = smallCoreDataNum - ubPartDataNum * (smallCoreLoopNum - 1);
    smallCoreTailDataNum = (smallCoreTailDataNum == 0) ? ubPartDataNum : smallCoreTailDataNum;

    // 大核数据计算
    if (tailBlockNum != 0) {
        everyCoreInputBlockNum += 1;
        bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / dataTypeLength;
        bigCoreLoopNum = bigCoreDataNum / ubPartDataNum;
        bigCoreLoopNum = (everyCoreInputBlockNum % ubPartBlockNum) == 0 
                       ? bigCoreLoopNum : bigCoreLoopNum + 1;
        bigCoreTailDataNum = bigCoreDataNum - ubPartDataNum * (bigCoreLoopNum - 1);
        bigCoreTailDataNum = (bigCoreTailDataNum == 0) ? ubPartDataNum : bigCoreTailDataNum;
        context->SetTilingKey(1);
    } else {
        context->SetTilingKey(0);
    }
    
    // 设置tiling参数
    tiling.set_smallCoreDataNum(smallCoreDataNum);
    tiling.set_bigCoreDataNum(bigCoreDataNum);
    tiling.set_ubPartDataNum(ubPartDataNum);
    tiling.set_smallCoreTailDataNum(smallCoreTailDataNum);
    tiling.set_bigCoreTailDataNum(bigCoreTailDataNum);
    tiling.set_smallCoreLoopNum(smallCoreLoopNum);
    tiling.set_bigCoreLoopNum(bigCoreLoopNum);
    tiling.set_tailBlockNum(tailBlockNum);
    context->SetBlockDim(coreNum);

    // 处理log特有属性
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const float* base = attrs->GetAttrPointer<float>(0);
    if (*base == -1.0f) {
        tiling.set_base(1.0f);
    } else if (*base > 0.0f && *base != 1.0f) {
        tiling.set_base(1.0f / std::log(*base));
    } else {
        tiling.set_base(1.0f);
    }
    
    const float* scale = attrs->GetAttrPointer<float>(1);
    tiling.set_scale(*scale);
    
    const float* shift = attrs->GetAttrPointer<float>(2);
    tiling.set_shift(*shift);

    // 保存tiling数据
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), 
                       context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Log : public OpDef {
public:
    explicit Log(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("base").AttrType(OPTIONAL).Float(-1.0);
        this->Attr("scale").AttrType(OPTIONAL).Float(1.0);
        this->Attr("shift").AttrType(OPTIONAL).Float(0.0);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Log);
} // namespace ops