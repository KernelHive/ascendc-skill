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
 * @file coalesce_sparse.cpp
 */
#include "coalesce_sparse_tiling.h"
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"
#include "platform/platform_info.h"

#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)

namespace optiling {
constexpr uint64_t INT64_ID  = 0;
constexpr uint64_t INT32_ID  = 1;
constexpr uint64_t FP32_ID  = 0;
constexpr uint64_t FP16_ID  = 2;
constexpr uint64_t KEY_MODE1  = 6;
constexpr uint64_t KEY_MODE2  = 3;
constexpr uint64_t AILGN256 = 256;
constexpr uint64_t AILGN32 = 32;
constexpr uint64_t MAXRPTIME = 4096;
constexpr size_t INPUT_UNIQUE_INDICES_IDX = 1;
constexpr size_t INPUT_INDICES_IDX = 2;
constexpr size_t INPUT_VALUES_IDX = 3;
class CoalesceSparseTiling {
    public:
        explicit CoalesceSparseTiling(gert::TilingContext* context) : TilingContext(context){};
        ge::graphStatus Init();
        ge::graphStatus RunKernelTiling();
        void SetTilingKey(ge::DataType uniqueIndicesDtype, ge::DataType indicesDtype, ge::DataType valuesDtype);
        void TilingDataPrint() const;
    private:
        CoalesceSparseTilingData TilingData;
        gert::TilingContext* TilingContext = nullptr;
        uint64_t tiling_key = 0;
        uint64_t n = 0;
        uint64_t m = 0;
        uint64_t valueSize = 0;
        uint64_t taskNum = 0;
        uint64_t usedCoreNum = 0;
        uint64_t taskTail = 0;
        uint64_t moveOneSize = 0;
        uint64_t taskRepeatTimes = 0;
        uint64_t taskRepeatTail = 0;
        uint64_t taskTailRepeatTimes = 0;
        uint64_t taskTailRepeatTail = 0;
        uint64_t moveValueTimes = 0;
        uint64_t moveValueLen = 0;
        uint64_t moveValueTail = 0;
};

void CoalesceSparseTiling::SetTilingKey(ge::DataType uniqueIndicesDtype, ge::DataType indicesDtype, ge::DataType valuesDtype)
{
    if (uniqueIndicesDtype == ge::DT_INT64) {
        tiling_key = INT64_ID * KEY_MODE1;
    } else {
        tiling_key = INT32_ID * KEY_MODE1;
    }
    if (indicesDtype == ge::DT_INT64) {
        tiling_key += INT64_ID * KEY_MODE2;
    } else {
        tiling_key += INT32_ID * KEY_MODE2;
    }
    if (valuesDtype == ge::DT_FLOAT) {
        tiling_key += FP32_ID;
    } else if (valuesDtype == ge::DT_INT32) {
        tiling_key += INT32_ID;
    } else {
        tiling_key += FP16_ID;
    }
}
ge::graphStatus CoalesceSparseTiling::Init()
{
    OP_LOGD(TilingContext->GetNodeName(), "Tiling initing.");
    auto platformInfo = TilingContext->GetPlatformInfo();
    auto indicesShape = TilingContext->GetInputShape(INPUT_INDICES_IDX)->GetStorageShape();
    auto valueShape = TilingContext->GetInputShape(INPUT_VALUES_IDX)->GetStorageShape();

    auto uniqueIndicesDtype = TilingContext->GetInputDesc(INPUT_UNIQUE_INDICES_IDX)->GetDataType();
    auto indicesDtype = TilingContext->GetInputDesc(INPUT_INDICES_IDX)->GetDataType();
    auto valuesDtype = TilingContext->GetInputDesc(INPUT_VALUES_IDX)->GetDataType();

    uint64_t uniqueIndiceTypeSize = GetSizeByDataType(uniqueIndicesDtype);
    uint64_t indiceTypeSize = GetSizeByDataType(indicesDtype);
    uint64_t valueTypeSize = GetSizeByDataType(valuesDtype);

    uint64_t ubSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(TilingContext->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t coreNum = platformInfo->GetCoreNum();
    SetTilingKey(uniqueIndicesDtype, indicesDtype, valuesDtype);

    n = indicesShape.GetDim(0);
    m = indicesShape.GetDim(1);
    valueSize = 1;
    uint64_t valueShapeSize = valueShape.GetDimNum();
    for(uint64_t i = 1; i < valueShapeSize; i++) {
        valueSize *= valueShape.GetDim(i);
    }
    taskNum = coreNum > 0 ? ((n - 1) / coreNum / AILGN256 + 1) * AILGN256 : 0;
    usedCoreNum = (n - 1) / taskNum + 1;
    taskTail = n % taskNum;
    if(taskTail == 0) {
        taskTail = taskNum;
    }

    moveValueLen = valueSize;
    if (moveValueLen > MAXRPTIME) {
        moveValueLen = MAXRPTIME;
    }
    uint64_t indexUb = ubSize - (moveValueLen * valueTypeSize / AILGN32 + 1) * AILGN32;

    uint64_t alignDivIndicesTypeSize = indiceTypeSize > 0 ? AILGN32 / indiceTypeSize : 0;
    uint64_t indicesSizeAlign32 = alignDivIndicesTypeSize > 0 ? ((m - 1) / alignDivIndicesTypeSize + 1) * AILGN32 : 0;
    moveOneSize = (indexUb / (indicesSizeAlign32 + uniqueIndiceTypeSize)) / AILGN32 * AILGN32;
    if(moveOneSize >= MAXRPTIME) {
        moveOneSize = MAXRPTIME - AILGN32;
    }
    taskRepeatTimes = taskNum / moveOneSize;
    taskRepeatTail = taskNum % moveOneSize;
    taskTailRepeatTimes = taskTail / moveOneSize;
    taskTailRepeatTail = taskTail % moveOneSize;

    moveValueTimes = valueSize / moveValueLen;
    moveValueTail = valueSize % moveValueLen;

    OP_LOGD(TilingContext->GetNodeName(), "Tiling inited.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CoalesceSparseTiling::RunKernelTiling(){
    OP_LOGD(TilingContext->GetNodeName(), "Tiling start.");
    TilingContext->SetBlockDim(usedCoreNum);
    TilingContext->SetTilingKey(tiling_key);
    TilingData.set_usedCoreNum(usedCoreNum);
    TilingData.set_m(m);
    TilingData.set_n(n);
    TilingData.set_valueSize(valueSize);
    TilingData.set_taskNum(taskNum);
    TilingData.set_taskTail(taskTail);
    TilingData.set_moveOneSize(moveOneSize);
    TilingData.set_taskRepeatTimes(taskRepeatTimes);
    TilingData.set_taskRepeatTail(taskRepeatTail);
    TilingData.set_taskTailRepeatTimes(taskTailRepeatTimes);
    TilingData.set_taskTailRepeatTail(taskTailRepeatTail);
    TilingData.set_moveValueTimes(moveValueTimes);
    TilingData.set_moveValueLen(moveValueLen);
    TilingData.set_moveValueTail(moveValueTail);

    TilingData.SaveToBuffer(TilingContext->GetRawTilingData()->GetData(), TilingContext->GetRawTilingData()->GetCapacity());
    TilingContext->GetRawTilingData()->SetDataSize(TilingData.GetDataSize());
    TilingDataPrint();
    OP_LOGD(TilingContext->GetNodeName(), "Tiling end.");
    return ge::GRAPH_SUCCESS;
}

void CoalesceSparseTiling::TilingDataPrint() const {
    OP_LOGD(TilingContext->GetNodeName(), "usedCoreNum: %lu.", usedCoreNum);
    OP_LOGD(TilingContext->GetNodeName(), "m: %lu.", m);
    OP_LOGD(TilingContext->GetNodeName(), "n: %lu.", n);
    OP_LOGD(TilingContext->GetNodeName(), "valueSize: %lu.", valueSize);
    OP_LOGD(TilingContext->GetNodeName(), "taskNum: %lu.", taskNum);
    OP_LOGD(TilingContext->GetNodeName(), "taskTail: %lu.", taskTail);
    OP_LOGD(TilingContext->GetNodeName(), "moveOneSize: %lu.", moveOneSize);
    OP_LOGD(TilingContext->GetNodeName(), "taskRepeatTimes: %lu.", taskRepeatTimes);
    OP_LOGD(TilingContext->GetNodeName(), "taskRepeatTail: %lu.", taskRepeatTail);
    OP_LOGD(TilingContext->GetNodeName(), "taskTailRepeatTimes: %lu.", taskTailRepeatTimes);
    OP_LOGD(TilingContext->GetNodeName(), "taskTailRepeatTail: %lu.", taskTailRepeatTail);
    OP_LOGD(TilingContext->GetNodeName(), "moveValueTimes: %lu.", moveValueTimes);
    OP_LOGD(TilingContext->GetNodeName(), "moveValueLen: %lu.", moveValueLen);
    OP_LOGD(TilingContext->GetNodeName(), "moveValueTail: %lu.", moveValueTail);
}

static ge::graphStatus TilingCoalesceSparse(gert::TilingContext *context)
{
    CoalesceSparseTiling tilingObject(context);
    tilingObject.Init();
    return tilingObject.RunKernelTiling();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForCoalesceSparse(gert::TilingParseContext* context) {
    OP_LOGD("CoalesceSparse", "TilingPrepareForCoalesceSparse start.");
    return ge::GRAPH_SUCCESS;
}

struct CoalesceSparseCompileInfo {};
IMPL_OP_OPTILING(CoalesceSparse)
    .Tiling(TilingCoalesceSparse)
    .TilingParse<CoalesceSparseCompileInfo>(TilingPrepareForCoalesceSparse);
} // namespace optiling

namespace ge {
constexpr size_t INPUT_UNIQUE_LEN_IDX = 0;
constexpr size_t INPUT_UNIQUE_INDICES_IDX = 1;
constexpr size_t INPUT_INDICES_IDX = 2;
constexpr size_t INPUT_VALUES_IDX = 3;

#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)            \
    if ((ptr) == nullptr)                                    \
    {                                                        \
        std::printf("nullptr error!");                       \
        return ge::GRAPH_SUCCESS;                            \
    }                                                        \

static graphStatus CoalesceSparseInferShape(gert::InferShapeContext *context)
{
    const gert::Shape* uniqueLenShape = context->GetInputShape(INPUT_UNIQUE_LEN_IDX);
    const gert::Shape* uniqueShape = context->GetInputShape(INPUT_UNIQUE_INDICES_IDX);
    const gert::Shape* indicesShape = context->GetInputShape(INPUT_INDICES_IDX);
    const gert::Shape* valueShape = context->GetInputShape(INPUT_VALUES_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, uniqueLenShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, uniqueShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, indicesShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, valueShape);

    gert::Shape* newIndicesShape = context->GetOutputShape(0);
    gert::Shape* newValueShape = context->GetOutputShape(1);

    uint64_t len = uniqueLenShape->GetDim(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, newIndicesShape);
    newIndicesShape->AppendDim(len);
    newIndicesShape->AppendDim(indicesShape->GetDim(1));

    uint64_t valueShapeSize = valueShape->GetDimNum();
    OPS_CHECK_NULL_WITH_CONTEXT(context, newValueShape);
    newValueShape->AppendDim(len);
    for(uint64_t i = 1; i < valueShapeSize; i++) {
        newValueShape->AppendDim(valueShape->GetDim(i));
    }
}

static graphStatus CoalesceSparseInferDataType(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4CoalesceSparse");

    auto indicesDataType = context->GetInputDataType(INPUT_INDICES_IDX);
    auto valueDataType = context->GetInputDataType(INPUT_VALUES_IDX);
    context->SetOutputDataType(0, indicesDataType);
    context->SetOutputDataType(1, valueDataType); 

    OP_LOGD(context->GetNodeName(), "End to do InferDataType4CoalesceSparse end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CoalesceSparse).InferShape(CoalesceSparseInferShape).InferDataType(CoalesceSparseInferDataType);
} // namespace ge

namespace ops {
class CoalesceSparse : public OpDef {
public:
    explicit CoalesceSparse(const char *name) : OpDef(name)
    {
        this->Input("unique_len")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("unique_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("values")
            .DataType({ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("new_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("new_values")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true);
        this->AICore()
            .AddConfig("ascend910b", aicore_config);
    }
};
OP_ADD(CoalesceSparse);
} // namespace ops
