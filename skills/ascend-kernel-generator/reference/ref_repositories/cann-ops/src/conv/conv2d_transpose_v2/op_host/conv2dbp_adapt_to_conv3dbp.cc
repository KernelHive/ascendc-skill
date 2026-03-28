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
 * \file conv2dbp_adapt_to_conv3dbp.cc
 * \brief
 */
#include "conv2dbp_adapt_to_conv3dbp.h"
#
#include <vector>
#include <string>
#include "conv3d_backprop_input_v2_tiling.h"

using namespace optiling;
using namespace std;
namespace optiling {
#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)
struct tagTilingContextMakerParam {
    // dataformat
    ge::Format input0Format3D;
    ge::Format input1Format3D;
    ge::Format input2Format3D;
    ge::Format output0Format3D;
    ge::Format input0OralFormat3D = ge::FORMAT_NCDHW;
    ge::Format input1OralFormat3D = ge::FORMAT_NCDHW;
    ge::Format input2OralFormat3D = ge::FORMAT_NCDHW;
    ge::Format output0OralFormat3D = ge::FORMAT_NCDHW;
    // datatype
    ge::DataType in0DataType3D;
    ge::DataType in1DataType3D;
    ge::DataType in2DataType3D;
    ge::DataType out0DataType3D;
    int32_t offsetX;
    int32_t group;
    string dataFormatIn3D;
    vector<int64_t> strides = vector<int64_t>(5, 0); // num of strides
    vector<int64_t> pads = vector<int64_t>(6, 0); // num of pads
    vector<int64_t> dilations = vector<int64_t>(5, 0); // num of dilations
    vector<int64_t> outputPadding = vector<int64_t>(5, 0); // num of outputPadding
    gert::StorageShape input0Shape3D;
    gert::StorageShape input1Shape3D;
    gert::StorageShape input2Shape3D;
    vector<gert::StorageShape> outputShapes = vector<gert::StorageShape>(1);
    optiling::Conv3DCompileInfo compileInfo;
    vector<void *> refShapeOutputs = vector<void *>(1);
};

using TilingContextMakerParam = struct tagTilingContextMakerParam;
constexpr uint32_t INPUT0_INDEX = 0;
constexpr uint32_t INPUT1_INDEX = 1;
constexpr uint32_t INPUT2_INDEX = 2;
constexpr uint32_t OUTPUT0_INDEX = 0;
constexpr uint32_t STRIDE_INDEX = 0;
constexpr uint32_t PAD_INDEX = 1;
constexpr uint32_t DILATION_INDEX = 2;
constexpr uint32_t GROUP_INDEX = 3;
constexpr uint32_t DATA_FORMAT_INDEX = 4;
constexpr uint32_t OUTPUT_PADDING_INDEX = 5;
constexpr uint32_t OFFSET_X = 6;

static ge::graphStatus GetDataFormat(const gert::TilingContext *context, TilingContextMakerParam &param,
                            string formatIn2D, string &formatIn3D, string opType)
{
    if (formatIn2D == string("NCHW")) {
        formatIn3D.assign(string("NCDHW"));
        OP_TILING_CHECK(context->GetInputDesc(INPUT0_INDEX) == nullptr ||
            context->GetInputDesc(INPUT1_INDEX) == nullptr || context->GetOutputDesc(OUTPUT0_INDEX) == nullptr,
            CUBE_INNER_ERR_REPORT(opType, "fail to get input/y tensor desc from context"),
            return ge::GRAPH_FAILED;
        );
        // 设置Format
        if (opType == string("Conv2DBackpropFilterV3")) {
            param.input0Format3D = ge::FORMAT_NDC1HWC0;
            param.input1Format3D = context->GetInputDesc(INPUT1_INDEX)->GetStorageFormat(); // ND
            param.input1OralFormat3D = context->GetInputDesc(INPUT1_INDEX)->GetOriginFormat(); // ND
            param.input2Format3D = ge::FORMAT_NDC1HWC0;
        } else if (opType == string("Conv2DBackpropInputV2")) {
            param.input0Format3D = context->GetInputDesc(INPUT0_INDEX)->GetStorageFormat(); // ND
            param.input0OralFormat3D = context->GetInputDesc(INPUT0_INDEX)->GetOriginFormat(); // ND
            param.input1Format3D = ge::FORMAT_FRACTAL_Z_3D;
            param.input2Format3D = ge::FORMAT_NDC1HWC0;
        } else if (opType == string("Conv2DTransposeV2")) {
            param.input0Format3D = context->GetInputDesc(INPUT0_INDEX)->GetStorageFormat(); // ND
            param.input0OralFormat3D = context->GetInputDesc(INPUT0_INDEX)->GetOriginFormat(); // ND
            param.input1Format3D = ge::FORMAT_NDC1HWC0;
            param.input2Format3D = ge::FORMAT_FRACTAL_Z_3D;
        } else {
            return ge::GRAPH_FAILED;
        }
        auto outFormat2D =
            static_cast<ge::Format>(ge::GetPrimaryFormat(context->GetOutputDesc(OUTPUT0_INDEX)->GetStorageFormat()));
        if (outFormat2D == ge::FORMAT_NCHW) {
            param.output0Format3D = ge::FORMAT_NCDHW;
        } else if (outFormat2D == ge::FORMAT_NC1HWC0) {
            param.output0Format3D = ge::FORMAT_NDC1HWC0;
        } else if (outFormat2D == ge::FORMAT_FRACTAL_Z) {
            param.output0Format3D = ge::FORMAT_FRACTAL_Z_3D;
        }
    } else {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetTilingContextAttr(gert::TilingContext *context, TilingContextMakerParam &param, string opType)
{
    // 1 获取属性
    auto attrs = context->GetAttrs();
    auto strides = attrs->GetListInt(STRIDE_INDEX);
    auto pads = attrs->GetListInt(PAD_INDEX);
    auto dilations = attrs->GetListInt(DILATION_INDEX);
    auto attrPtrGroup = attrs->GetInt(GROUP_INDEX);
    param.group = static_cast<int32_t>(*attrPtrGroup);
    const char *dataFormat = attrs->GetStr(DATA_FORMAT_INDEX);
    auto ret = GetDataFormat(context, param, dataFormat, param.dataFormatIn3D, opType);
    // 2 设置属性
    vector<int64_t> conv3dPads({0, 0, pads->GetData()[0], pads->GetData()[1], pads->GetData()[2],
                                pads->GetData()[3]});
    vector<int64_t> conv3dStrides({strides->GetData()[0], strides->GetData()[1], 1,
                                   strides->GetData()[2], strides->GetData()[3]});
    vector<int64_t> conv3dDilations({dilations->GetData()[0], dilations->GetData()[1], 1,
                                    dilations->GetData()[2], dilations->GetData()[3]});
    param.strides = conv3dStrides;
    param.pads = conv3dPads;
    param.dilations = conv3dDilations;

    if (opType == string("Conv2DTransposeV2")) {
        auto outputPadding = attrs->GetListInt(OUTPUT_PADDING_INDEX);
        vector<int64_t> conv3dOutputpadding({outputPadding->GetData()[0], outputPadding->GetData()[1], 0,
                                        outputPadding->GetData()[2], outputPadding->GetData()[3]});
        param.outputPadding = conv3dOutputpadding;
        auto offsetX = attrs->GetInt(OFFSET_X);
        param.offsetX = static_cast<int32_t>(*offsetX);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetTilingContextInStorageShape(gert::TilingContext *context, TilingContextMakerParam &param, string opType)
{
    // 2 获取输入输出shape
    const gert::Shape &in0StorageShape = context->GetInputShape(INPUT0_INDEX)->GetStorageShape();
    const gert::Shape &in1StorageShape = context->GetInputShape(INPUT1_INDEX)->GetStorageShape();
    const gert::Shape &in2StorageShape = context->GetInputShape(INPUT2_INDEX)->GetStorageShape();
    const gert::Shape &in0OriginShape = context->GetInputShape(INPUT0_INDEX)->GetOriginShape();
    const gert::Shape &in1OriginShape = context->GetInputShape(INPUT1_INDEX)->GetOriginShape();
    const gert::Shape &in2OriginShape = context->GetInputShape(INPUT2_INDEX)->GetOriginShape();
    // 设置输入Shape
    gert::StorageShape input0Shape3D;
    gert::StorageShape input1Shape3D;
    gert::StorageShape input2Shape3D;
    if (opType == string("Conv2DBackpropFilterV3")) {
        input0Shape3D = {
            {in0OriginShape.GetDim(0), in0OriginShape.GetDim(1), 1, in0OriginShape.GetDim(2), in0OriginShape.GetDim(3)}, // NCDHW
            {in0StorageShape.GetDim(0), 1, in0StorageShape.GetDim(1), in0StorageShape.GetDim(2), in0StorageShape.GetDim(3), in0StorageShape.GetDim(4)}}; // NDC1HWC0
        input1Shape3D = {{5}, {5}}; // 5dim
        input2Shape3D = {
            {in2OriginShape.GetDim(0), in2OriginShape.GetDim(1), 1, in2OriginShape.GetDim(2), in2OriginShape.GetDim(3)}, // NCDHW
            {in2StorageShape.GetDim(0), 1, in2StorageShape.GetDim(1), in2StorageShape.GetDim(2), in2StorageShape.GetDim(3), in2StorageShape.GetDim(4)}}; // NDC1HWC0
    } else if (opType == string("Conv2DBackpropInputV2")) {
        input0Shape3D = {{5}, {5}}; // 5dim
        input1Shape3D = {
            {in1OriginShape.GetDim(0), in1OriginShape.GetDim(1), 1, in1OriginShape.GetDim(2), in1OriginShape.GetDim(3)}, // NCDHW
            {in1StorageShape.GetDim(0), in1StorageShape.GetDim(1), in1StorageShape.GetDim(2), in1StorageShape.GetDim(3)}}; // FRACTAL_Z_3D
        input2Shape3D = {
            {in2OriginShape.GetDim(0), in2OriginShape.GetDim(1), 1, in2OriginShape.GetDim(2), in2OriginShape.GetDim(3)}, // NCDHW
            {in2StorageShape.GetDim(0), 1, in2StorageShape.GetDim(1), in2StorageShape.GetDim(2), in2StorageShape.GetDim(3), in2StorageShape.GetDim(4)}}; // NDC1HWC0
    } else if (opType == string("Conv2DTransposeV2")) {
        input0Shape3D = {{5}, {5}}; // 5dim
        input1Shape3D = {
            {in1OriginShape.GetDim(0), in1OriginShape.GetDim(1), 1, in1OriginShape.GetDim(2), in1OriginShape.GetDim(3)}, // NCDHW
            {in1StorageShape.GetDim(0), 1, in1StorageShape.GetDim(1), in1StorageShape.GetDim(2), in1StorageShape.GetDim(3), in1StorageShape.GetDim(4)}}; // NDC1HWC0
        input2Shape3D = {
            {in2OriginShape.GetDim(0), in2OriginShape.GetDim(1), 1, in2OriginShape.GetDim(2), in2OriginShape.GetDim(3)}, // NCDHW
            {in2StorageShape.GetDim(0), in2StorageShape.GetDim(1), in2StorageShape.GetDim(2), in2StorageShape.GetDim(3)}}; // FRACTAL_Z_3D
    } else {
        return ge::GRAPH_FAILED;
    }
    param.input0Shape3D.MutableStorageShape() = input0Shape3D.MutableStorageShape();
    param.input0Shape3D.MutableOriginShape() = input0Shape3D.MutableOriginShape();
    param.input1Shape3D.MutableStorageShape() = input1Shape3D.MutableStorageShape();
    param.input1Shape3D.MutableOriginShape() = input1Shape3D.MutableOriginShape();
    param.input2Shape3D.MutableStorageShape() = input2Shape3D.MutableStorageShape();
    param.input2Shape3D.MutableOriginShape() = input2Shape3D.MutableOriginShape();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetTilingContextOutStorageShape(gert::TilingContext *context, TilingContextMakerParam &param, string opType)
{
    // 设置输出shape
    OP_TILING_CHECK(context->GetOutputDesc(OUTPUT0_INDEX) == nullptr,
        CUBE_INNER_ERR_REPORT(opType, "fail to get output tensor desc from context"),
        return ge::GRAPH_FAILED;
    );
    const gert::Shape &outStorageShape = context->GetOutputShape(OUTPUT0_INDEX)->GetStorageShape();
    const gert::Shape &outOriginShape = context->GetOutputShape(OUTPUT0_INDEX)->GetOriginShape();
    auto outFormat2D = 
        static_cast<ge::Format>(ge::GetPrimaryFormat(context->GetOutputDesc(OUTPUT0_INDEX)->GetStorageFormat()));
    gert::StorageShape outputStorage3D;
    if (outFormat2D == ge::FORMAT_NCHW) { // dx nchw
        outputStorage3D = {
            {outOriginShape.GetDim(0), outOriginShape.GetDim(1), 1, outOriginShape.GetDim(2), outOriginShape.GetDim(3)}, // NCDHW
            {outStorageShape.GetDim(0), outStorageShape.GetDim(1), 1, outStorageShape.GetDim(2), outStorageShape.GetDim(3)}}; // NCDHW
    } else if (outFormat2D == ge::FORMAT_NC1HWC0) { // dx nc1hwc0, dtranspose
        outputStorage3D = {
            {outOriginShape.GetDim(0), outOriginShape.GetDim(1), 1, outOriginShape.GetDim(2), outOriginShape.GetDim(3)}, // NCDHW
            {outStorageShape.GetDim(0), 1, outStorageShape.GetDim(1), outStorageShape.GetDim(2), outStorageShape.GetDim(3), outStorageShape.GetDim(4)}}; // NDC1HWC0
    } else if (outFormat2D == ge::FORMAT_FRACTAL_Z) { // dw
        outputStorage3D = {
            {outOriginShape.GetDim(0), outOriginShape.GetDim(1), 1, outOriginShape.GetDim(2), outOriginShape.GetDim(3)}, // NCDHW
            {outStorageShape.GetDim(0), outStorageShape.GetDim(1), outStorageShape.GetDim(2), outStorageShape.GetDim(3)}}; // FRACTAL_Z_3D
    } else {
        return ge::GRAPH_FAILED;
    }
    param.outputShapes[0] = outputStorage3D;
    for (size_t i = 0; i < param.outputShapes.size(); ++i) {
        param.refShapeOutputs[i] = &param.outputShapes[i];
    }
    return ge::GRAPH_SUCCESS;
}

static void GetTilingContextDataType(const gert::TilingContext *context, TilingContextMakerParam &param, const string opType)
{
    // 3 获取输入输出 datatype
    OP_TILING_CHECK(context->GetInputDesc(INPUT0_INDEX) == nullptr,
        CUBE_INNER_ERR_REPORT(opType, "fail to get 0th input tensor desc from context"),
        return;
    );
    param.in0DataType3D = context->GetInputDesc(INPUT0_INDEX)->GetDataType();
    OP_TILING_CHECK(context->GetInputDesc(INPUT1_INDEX) == nullptr,
        CUBE_INNER_ERR_REPORT(opType, "fail to get 1st input tensor desc from context"),
        return;
    );
    param.in1DataType3D = context->GetInputDesc(INPUT1_INDEX)->GetDataType();
    OP_TILING_CHECK(context->GetInputDesc(INPUT2_INDEX) == nullptr,
        CUBE_INNER_ERR_REPORT(opType, "fail to get 2nd input tensor desc from context"),
        return;
    );
    param.in2DataType3D = context->GetInputDesc(INPUT2_INDEX)->GetDataType();
    OP_TILING_CHECK(context->GetOutputDesc(OUTPUT0_INDEX) == nullptr,
        CUBE_INNER_ERR_REPORT(opType, "fail to get y tensor desc from context"),
        return;
    );
    param.out0DataType3D = context->GetOutputDesc(OUTPUT0_INDEX)->GetDataType();
}

static ge::graphStatus GetTilingContextInfo(gert::TilingContext *context, TilingContextMakerParam &param, string opType)
{
    // 1 获取3D属性和format
    ge::graphStatus ret = GetTilingContextAttr(context, param, opType);
    // 2 获取输入/输出 shape
    ret = GetTilingContextInStorageShape(context, param, opType);
    ret = GetTilingContextOutStorageShape(context, param, opType);
    // 3 获取输入/输出 dataType
    GetTilingContextDataType(context, param, opType);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AdaptConv3DBpTiliingAndCopy(gert::TilingContext *tilingContext3D,
    gert::TilingContext *context, string opType)
{
    // 调用3D的tiling模板
    auto ret = TilingRegistry::GetInstance().DoTilingImpl(tilingContext3D);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE("AdaptConv3DBpTiliingAndCopy: DoTilingImpl failed\n");
        return ge::GRAPH_FAILED;
    }
    // 将3DContext的TilingData信息拷贝至context
    context->GetRawTilingData()->SetDataSize(tilingContext3D->GetRawTilingData()->GetDataSize());
    auto cpRet = memcpy_s(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetDataSize(),
                         tilingContext3D->GetRawTilingData()->GetData(),
                         tilingContext3D->GetRawTilingData()->GetDataSize());
    if (cpRet != EOK) {
        OP_LOGE("AdaptConv3DBpTiliingAndCopy: memcpy_s failed\n");
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(tilingContext3D->GetBlockDim());
    context->SetTilingKey(tilingContext3D->GetTilingKey());
    size_t *workspaces = context->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = tilingContext3D->GetWorkspaceSizes(1)[0];
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptTilingToConv3DBp(gert::TilingContext *context, string opType)
{
    OP_LOGD("Converting %s to Conv3DBackprop/Conv3DTranspose in tiling.\n", opType.c_str());
    // 构造3D TilingContext的输入/输出shape、datatype、attr、compileInfo等信息
    TilingContextMakerParam param;
    ge::graphStatus ret = GetTilingContextInfo(context, param, opType);
    int capSize = sizeof(optiling::Conv3DBackpropInputV2TilingData);
    auto rawTilingData = gert::TilingData::CreateCap(capSize);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096); // 4k default;
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    vector<pair<string, ge::AnyValue>> keysToValue = {
        {"strides", ge::AnyValue::CreateFrom<vector<int64_t>>(param.strides)},
        {"pads", ge::AnyValue::CreateFrom<vector<int64_t>>(param.pads)},
        {"dilations", ge::AnyValue::CreateFrom<vector<int64_t>>(param.dilations)},
        {"groups", ge::AnyValue::CreateFrom<int64_t>(param.group)},
        {"data_format", ge::AnyValue::CreateFrom<string>(param.dataFormatIn3D)}};
    string opTypeForSet = opType;
    if (opType == string("Conv2DTransposeV2")) {
        keysToValue.push_back({"output_padding", ge::AnyValue::CreateFrom<vector<int64_t>>(param.outputPadding)});
        keysToValue.push_back({"offset_x", ge::AnyValue::CreateFrom<int64_t>(param.offsetX)});
        opTypeForSet = string("Conv3DTransposeV2");
    }
    
    auto tilingContextHolder =
    optiling::TilingContextMaker()
            .NodeIoNum(3, 1) // 3 input, 1 output
            .IrInstanceNum({1, 1, 1})
            .InputShapes({&param.input0Shape3D, &param.input1Shape3D, &param.input2Shape3D})
            .OutputShapes({param.refShapeOutputs})
            .CompileInfo(const_cast<void*>(context->GetCompileInfo()))
            .NodeInputTd(INPUT0_INDEX, param.in0DataType3D, param.input0OralFormat3D, param.input0Format3D)
            .NodeInputTd(INPUT1_INDEX, param.in1DataType3D, param.input1OralFormat3D, param.input1Format3D)
            .NodeInputTd(INPUT2_INDEX, param.in2DataType3D, param.input2OralFormat3D, param.input2Format3D)
            .NodeOutputTd(OUTPUT0_INDEX, param.out0DataType3D, param.output0OralFormat3D, param.output0Format3D)
            .NodeAttrs(keysToValue)
            .TilingData(static_cast<void *>(rawTilingData.get()))
            .Workspace(workspace)
            .SetOpType(opTypeForSet)
            .Build();
    
    auto *tilingContext3D = tilingContextHolder.GetContext<gert::TilingContext>();

    ret = AdaptConv3DBpTiliingAndCopy(tilingContext3D, context, opType);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling