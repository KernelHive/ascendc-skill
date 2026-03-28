/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <sstream>
#include <string>
#include "kl_div_target_backward_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")

const int64_t INPUT_VAR_NUM = 3;//输入变量
const int64_t MAX_SHAPE_DIM = 8;
const int64_t BLOCK_SIZE = 32;
const int64_t BUFFER_NUM = 2;
const int64_t ALIGNED_BLOCK_NUM = 8;
const uint32_t GRAD_OUTPUT_IDX = 0;
const uint32_t SELF_IDX = 1;
const uint32_t TARGET_IDX = 2;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    KlDivTargetBackwardTilingData tiling;
    uint64_t ubSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto coreNum = ascendcPlatform.GetCoreNum();
    
    // Based on the input length and the number of inputs, the number of bytes of the input data type is obtained
    uint32_t typeLength = 0;
    int64_t inputNum = context->GetInputShape(TARGET_IDX)->GetStorageShape().GetShapeSize();
    auto dtype = context->GetInputDesc(0)->GetDataType();
    ge::TypeUtils::GetDataTypeLength(dtype, typeLength);
    int64_t inputLength = inputNum * typeLength;
    int64_t inputBytes = inputLength / inputNum;

    // There are a total of 4 shared UB spaces in the input and output, 1 for tmp buf.
    int64_t ubDataNumber = 5;
    if (dtype == ge::DT_BF16) {
        ubDataNumber = 10; // 10 = 4 input and output(bf16) + 4 input and output(fp32) + 2 tmp buf
    }

    // 因为kernel用到了CompareScalar指令要求输入按256B对齐
    int64_t tileBlockNum = (static_cast<int64_t>(ubSize) / BLOCK_SIZE / BUFFER_NUM) /
        ubDataNumber / ALIGNED_BLOCK_NUM * ALIGNED_BLOCK_NUM;
    int64_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / inputBytes;

    // Input data for 32B alignment
    int64_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number of audits is the actual number of audits
    coreNum = (coreNum <  inputLengthAlgin32 / BLOCK_SIZE) ? coreNum : inputLengthAlgin32 / BLOCK_SIZE;
    coreNum = (coreNum >= 1) ? coreNum : 1;
    int64_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
    int64_t tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;
    
    // Small chunks are calculated and sliced several times using the number of data on each core
    int64_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    int64_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    int64_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    // Tail block calculation for small chunks of data
    int64_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;
    
    // The total length of a large block of data is 32B larger than that of a small block of data
    everyCoreInputBlockNum += 1;
    int64_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    int64_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    int64_t finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    int64_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    auto attrs = context->GetAttrs();
    uint32_t reduction = static_cast<uint32_t>(*attrs->GetAttrPointer<int32_t>(0));
    uint32_t logTarget = static_cast<uint32_t>(*attrs->GetAttrPointer<bool>(1));
    
    tiling.set_smallCoreDataNum(smallCoreDataNum);
    tiling.set_bigCoreDataNum(bigCoreDataNum);
    tiling.set_tileDataNum(tileDataNum);
    tiling.set_smallTailDataNum(smallTailDataNum);
    tiling.set_bigTailDataNum(bigTailDataNum);
    tiling.set_finalSmallTileNum(finalSmallTileNum);
    tiling.set_finalBigTileNum(finalBigTileNum);
    tiling.set_tailBlockNum(tailBlockNum);
    tiling.set_inputNum(inputNum);
    tiling.set_reduction(reduction);
    tiling.set_logTarget(logTarget);

    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    currentWorkspace[0] = sysWorkspaceSize;

    OP_LOGD(context->GetNodeName(), "smallCoreDataNum = %lu.", smallCoreDataNum);
    OP_LOGD(context->GetNodeName(), "bigCoreDataNum = %lu.", bigCoreDataNum);
    OP_LOGD(context->GetNodeName(), "finalBigTileNum = %lu.", finalBigTileNum);
    OP_LOGD(context->GetNodeName(), "finalSmallTileNum = %lu.", finalSmallTileNum);
    OP_LOGD(context->GetNodeName(), "tileDataNum = %lu.", tileDataNum);
    OP_LOGD(context->GetNodeName(), "smallTailDataNum = %lu.", smallTailDataNum);
    OP_LOGD(context->GetNodeName(), "bigTailDataNum = %lu.", bigTailDataNum);
    OP_LOGD(context->GetNodeName(), "tailBlockNum = %lu.", tailBlockNum);
    OP_LOGD(context->GetNodeName(), "inputNum = %lu.", inputNum);
    OP_LOGD(context->GetNodeName(), "reduction = %u.", reduction);
    OP_LOGD(context->GetNodeName(), "logTarget = %u.", logTarget);

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
template <typename T>
std::string Shape2String(const T& shape) {
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; i++) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

static graphStatus InferShape(gert::InferShapeContext* context)
{
    int64_t numshapes0 = context->GetInputTensor(GRAD_OUTPUT_IDX)->GetOriginShape().GetDimNum();
    int64_t numshapes1 = context->GetInputTensor(SELF_IDX)->GetOriginShape().GetDimNum();
    int64_t numshapes2 = context->GetInputTensor(TARGET_IDX)->GetOriginShape().GetDimNum();
    // 广播后维度数
    int64_t maxShapeDim = std::max(numshapes0, std::max(numshapes1, numshapes2));
    // 记录每个输入原始shape
    int64_t shape[INPUT_VAR_NUM * MAX_SHAPE_DIM];
    // 记录广播后shape
    int64_t shapefull[maxShapeDim];
    for (int64_t i = 0; i < INPUT_VAR_NUM; i++) {
        int64_t *ss = &shape[i * MAX_SHAPE_DIM];
        auto inputshape = context->GetInputTensor(i);
        int64_t dim_num =  inputshape->GetOriginShape().GetDimNum();
        for (int64_t j = 0; j < maxShapeDim; j++) {
            if(j < maxShapeDim - dim_num){
                ss[j] = 1;
            } else {
                ss[j] = inputshape->GetStorageShape().GetDim(j - (maxShapeDim - dim_num));
            }
        }
    }
    for (int64_t i = 0; i < maxShapeDim; i++) {
        int64_t maxDim = 1;
        int64_t *sf = &shapefull[0];
        for (int64_t j = 0; j < INPUT_VAR_NUM; j++) {
            int64_t *ss = &shape[j * MAX_SHAPE_DIM];
            if (ss[i] > maxDim) {
                maxDim = ss[i];
            }
        }
        sf[i] = maxDim;  // 广播后的形状
    }
    const gert::Shape *x1_shape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    yShape->SetDimNum(maxShapeDim);
    for (int64_t i = 0; i < maxShapeDim; i++) {
        yShape->SetDim(i, shapefull[i]);
    }
    OP_LOGD(context->GetNodeName(), "yShape: %s", Shape2String(*yShape).c_str());
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class KlDivTargetBackward : public OpDef {
public:
    explicit KlDivTargetBackward(const char* name) : OpDef(name)
    {
        this->Input("grad_output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("self")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("target")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("grad_target")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("reduction").Int(0);
        this->Attr("log_target").Bool(false);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(KlDivTargetBackward);
}