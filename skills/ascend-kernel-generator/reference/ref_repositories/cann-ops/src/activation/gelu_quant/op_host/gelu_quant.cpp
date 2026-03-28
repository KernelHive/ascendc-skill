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
 * @file gelu_quant.cpp
 */
#include <cstdint>
#include <cstdio>
#include <sstream>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "gelu_quant_tiling.h"
#include "gelu_quant_tiling_def.h"
#include "gelu_quant_tiling_base.h"
namespace optiling {
ge::graphStatus GeluQuantTiling::RunGeluQuantTiling()
{
    OP_LOGD(nodeName_, "[GeluQuant] RunGeluQuantTiling start running");
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    ret = GetInputInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = GetPlatformInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = DoTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = PostTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    DumpTilingInfo();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluQuantTiling::GetInputInfo()
{
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    ret = ProcessAttrsInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = ProcessRequiredInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = ProcessOptionalScaleInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = ProcessOptionalOffsetInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    OP_LOGD(nodeName_, "[GeluQuant] GetInputInfo run completed.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluQuantTiling::ProcessOptionalScaleInfo()
{
    OP_LOGD(nodeName_, "[GeluQuant] ProcessOptionalScaleInfo start running.");

    auto scaleInputShapePtr = context_->GetOptionalInputShape(SCALE_INPUT_INDEX);
    if (scaleInputShapePtr == nullptr) {
        baseInfoOp.inputScaleType = EMPTY_TENSOR;
        OP_TILING_CHECK((baseInfoOp.quantMode == STATIC_QUANT_MODE),
            VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "the input_scale should be required when quant_mode is static."),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    auto scaleInputDesc = context_->GetOptionalInputDesc(SCALE_INPUT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, scaleInputDesc);
    baseInfoOp.scaleInputDtype = scaleInputDesc->GetDataType();
    OP_TILING_CHECK((baseInfoOp.xInputDtype == ge::DT_FLOAT && baseInfoOp.scaleInputDtype != ge::DT_FLOAT),
        VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "the dtype of x should be float when the dtype of scale is float."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((baseInfoOp.xInputDtype == ge::DT_FLOAT16 && baseInfoOp.scaleInputDtype == ge::DT_BF16),
        VECTOR_INNER_ERR_REPORT_TILIING(nodeName_,
        "the dtype of x should not be half when the dtype of scale is bfloat16."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((baseInfoOp.xInputDtype == ge::DT_BF16 && baseInfoOp.scaleInputDtype == ge::DT_FLOAT16),
        VECTOR_INNER_ERR_REPORT_TILIING(nodeName_,
        "the dtype of x should not be bfloat16 when the dtype of scale is half."),
        return ge::GRAPH_FAILED);

    if (scaleInputShapePtr->GetStorageShape().GetShapeSize() == 1) {
        baseInfoOp.inputScaleType = SCALAR_TENSOR;
    } else {
        baseInfoOp.inputScaleType = NORMAL_TENSOR;
        auto scaleInputDimNum = scaleInputShapePtr->GetStorageShape().GetDimNum();
        OP_TILING_CHECK((scaleInputDimNum != 1),
            VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "the shape dim of input_scale should be 1 or 0,but got %zu .",
            scaleInputDimNum),
            return ge::GRAPH_FAILED);
        auto scaleInputDim0 = scaleInputShapePtr->GetStorageShape().GetDim(0);
        OP_TILING_CHECK((scaleInputDim0 != baseInfoOp.endAxisLen),
            VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "the shape of input_scale should be [%ld] ,but got [%ld].",
            baseInfoOp.endAxisLen, scaleInputDim0),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluQuantTiling::ProcessOptionalOffsetInfo()
{
    OP_LOGD(nodeName_, "[GeluQuant] ProcessOptionalOffsetInfo start running.");

    if (baseInfoOp.quantMode == DYNAMIC_QUANT_MODE) {
        return ge::GRAPH_SUCCESS;
    }

    auto offsetInputShapePtr = context_->GetOptionalInputShape(OFFSET_INPUT_INDEX);
    if (offsetInputShapePtr == nullptr) {
        baseInfoOp.inputOffsetType = EMPTY_TENSOR;
        return ge::GRAPH_SUCCESS;
    }

    auto offsetInputDesc = context_->GetOptionalInputDesc(OFFSET_INPUT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, offsetInputDesc);
    baseInfoOp.offsetInputDtype = offsetInputDesc->GetDataType();
    OP_TILING_CHECK((baseInfoOp.scaleInputDtype != baseInfoOp.offsetInputDtype),
        VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "the dtype of input_scale should be same with input_offset."),
        return ge::GRAPH_FAILED);

    if (offsetInputShapePtr->GetStorageShape().GetShapeSize() == 1) {
        baseInfoOp.inputOffsetType = SCALAR_TENSOR;
        OP_TILING_CHECK((baseInfoOp.inputScaleType != SCALAR_TENSOR),
            VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "the shape of input_scale should be same with input_offset."),
            return ge::GRAPH_FAILED);
    } else {
        baseInfoOp.inputOffsetType = NORMAL_TENSOR;
        OP_TILING_CHECK((baseInfoOp.inputScaleType != NORMAL_TENSOR),
            VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "the shape of input_scale should be same with input_offset."),
            return ge::GRAPH_FAILED);
        auto offsetInputDimNum = offsetInputShapePtr->GetStorageShape().GetDimNum();
        OP_TILING_CHECK((offsetInputDimNum != 1),
            VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "the shape dim of input_offset should be 1 or 0,but got %zu .",
            offsetInputDimNum),
            return ge::GRAPH_FAILED);
        auto offsetInputDim0 = offsetInputShapePtr->GetStorageShape().GetDim(0);
        OP_TILING_CHECK((offsetInputDim0 != baseInfoOp.endAxisLen),
            VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "the shape of input_offset should be [%ld] ,but got [%ld].",
            baseInfoOp.endAxisLen, offsetInputDim0),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluQuantTiling::ProcessRequiredInfo()
{
    OP_LOGD(nodeName_, "[GeluQuant] ProcessRequiredInfo start running.");
    // x
    auto xInputDesc = context_->GetInputDesc(X_INPUT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, xInputDesc);
    baseInfoOp.xInputDtype = xInputDesc->GetDataType();
    OP_TILING_CHECK((baseInfoOp.xInputDtype != ge::DT_FLOAT && baseInfoOp.xInputDtype != ge::DT_FLOAT16 &&
        baseInfoOp.xInputDtype != ge::DT_BF16),
        VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "the dtype of input x should be one of FP32/FP16/BF16 ."),
        return ge::GRAPH_FAILED);

    auto xInputShapePtr = context_->GetInputShape(X_INPUT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, xInputShapePtr);
    auto xInputShape = xInputShapePtr->GetStorageShape();
    baseInfoOp.xDimNum = xInputShape.GetDimNum();
    OP_TILING_CHECK((baseInfoOp.xDimNum > INPUT_MAX_DIMENSIONS),
        VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "the input of x should be no more than 8 dimensions."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((baseInfoOp.xDimNum < INPUT_MIN_DIMENSIONS && baseInfoOp.quantMode == DYNAMIC_QUANT_MODE),
        VECTOR_INNER_ERR_REPORT_TILIING(nodeName_,
        "the input of x should be at least 2 dimensions when quant_mode is dynamic."),
        return ge::GRAPH_FAILED);

    baseInfoOp.endAxisLen = xInputShape.GetDim(baseInfoOp.xDimNum - 1);
    baseInfoOp.endAxisLenAligned = AlignToCeil(baseInfoOp.endAxisLen, FP32_BLOCK_NUM);
    for (int64_t i = 0; i < baseInfoOp.xDimNum - 1; i++) {
        baseInfoOp.fusedFrontAxis *= xInputShape.GetDim(i);
    }
    baseInfoOp.fusedAllAxis = baseInfoOp.fusedFrontAxis * baseInfoOp.endAxisLen;
    baseInfoOp.elementNumAlign = AlignToCeil(baseInfoOp.fusedAllAxis, FP32_BLOCK_NUM);

    // y
    auto yOutputShapePtr = context_->GetOutputShape(Y_OUTPUT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, yOutputShapePtr);
    auto yOutputShape = yOutputShapePtr->GetStorageShape();
    OP_TILING_CHECK((xInputShape != yOutputShape),
        VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "the shape of y should be same as x."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluQuantTiling::ProcessAttrsInfo()
{
    OP_LOGD(nodeName_, "[GeluQuant] ProcessAttrsInfo start running.");
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const char *approximate = attrs->GetAttrPointer<char>(APPROXIMATE_ATTR_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, approximate);
    if (strcmp(approximate, "none") == 0) {
        baseInfoOp.approximate = APPROXIMATE_NONE;
    } else if (strcmp(approximate, "tanh") == 0) {
        baseInfoOp.approximate = APPROXIMATE_TANH;
    } else {
        OP_TILING_CHECK((true),
            VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "the attr of approximate should be none or tanh."),
            return ge::GRAPH_FAILED);
    }

    const char *quantMode = attrs->GetAttrPointer<char>(QUANT_MODE_ATTR_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, quantMode);
    if (strcmp(quantMode, "static") == 0) {
        baseInfoOp.quantMode = STATIC_QUANT_MODE;
    } else if (strcmp(quantMode, "dynamic") == 0) {
        baseInfoOp.quantMode = DYNAMIC_QUANT_MODE;
    } else {
        OP_TILING_CHECK((true),
            VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "the attr of quant mode should be static or dynamic."),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluQuantTiling::GetPlatformInfo()
{
    OP_LOGD(nodeName_, "[GeluQuant] GetPlatformInfo start running.");
    auto platformInfo = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint32_t totalCore = platformInfo.GetCoreNumAiv();
    baseInfoOp.vectorCoreNum = totalCore;
    OP_TILING_CHECK((baseInfoOp.vectorCoreNum <= 0),
        VECTOR_INNER_ERR_REPORT_TILIING(nodeName_,
        "GeluQuantTiling get num of vector core is less than or equal to 0."),
        return ge::GRAPH_FAILED);

    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, baseInfoOp.ubSize);
    OP_TILING_CHECK((baseInfoOp.ubSize <= 0),
        VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "GeluQuantTiling get ub size is less than or equal to 0."),
        return ge::GRAPH_FAILED);

    baseInfoOp.ubSize -= RESERVED_UB_SIZE; // 可用UB空间

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluQuantTiling::DoStaticQuantPerTensorTiling()
{
    OP_LOGD(nodeName_, "[GeluQuant] DoStaticQuantPerTensorTiling start running.");
    splitCoreOp.coexistentNodeNum = STATIC_QUANT_PER_TENSOR_COEXISTING_QUANTITY;
    splitCoreOp.coexistentNodeElementNum = AlignToFloor((SafeDivide(static_cast<int64_t>(baseInfoOp.ubSize),
        static_cast<int64_t>(sizeof(float)) * splitCoreOp.coexistentNodeNum)),
        FP32_BLOCK_NUM);

    if (baseInfoOp.fusedAllAxis <= SINGLE_CORE_PROCESS_MIN_NUM) {
        splitCoreOp.usedCoreNum = 1;
        splitCoreOp.normalCoreProcessNum = baseInfoOp.fusedAllAxis;
        splitCoreOp.tailCoreProcessNum = baseInfoOp.fusedAllAxis;
    } else {
        splitCoreOp.normalCoreProcessNum = CeilDivide(baseInfoOp.fusedAllAxis, baseInfoOp.vectorCoreNum);
        splitCoreOp.normalCoreProcessNum = splitCoreOp.normalCoreProcessNum < SINGLE_CORE_PROCESS_MIN_NUM ?
            SINGLE_CORE_PROCESS_MIN_NUM :
            splitCoreOp.normalCoreProcessNum;
        splitCoreOp.usedCoreNum = CeilDivide(baseInfoOp.fusedAllAxis, splitCoreOp.normalCoreProcessNum);
        splitCoreOp.tailCoreProcessNum =
            baseInfoOp.fusedAllAxis - splitCoreOp.normalCoreProcessNum * (splitCoreOp.usedCoreNum - 1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluQuantTiling::DoStaticQuantFullKernelSmallEndAxis()
{
    int64_t mulRowsInUb = splitCoreOp.coexistentNodeElementNum / baseInfoOp.endAxisLenAligned;
    while (mulRowsInUb >= TWO_END_AXIS) {
        int64_t ubNum = CeilDivide(baseInfoOp.fusedFrontAxis, mulRowsInUb);
        if (ubNum >= baseInfoOp.vectorCoreNum) {
            break;
        } else {
            mulRowsInUb--;
        }
    }
    if (mulRowsInUb == 1) {
        splitCoreOp.templateMode = STATIC_FUNCTION_TEMPLATE;
        return ge::GRAPH_SUCCESS;
    }

    splitCoreOp.rowInner = mulRowsInUb;
    splitCoreOp.rowOuter = CeilDivide(baseInfoOp.fusedFrontAxis, mulRowsInUb);
    int64_t rowTailTmp = GetMod(baseInfoOp.fusedFrontAxis, mulRowsInUb);
    splitCoreOp.rowTail = rowTailTmp == 0 ? splitCoreOp.rowInner : rowTailTmp;

    splitCoreOp.colInner = baseInfoOp.endAxisLen;
    splitCoreOp.colOuter = 1;
    splitCoreOp.colTail = baseInfoOp.endAxisLen;

    splitCoreOp.normalCoreProcessNum = CeilDivide(splitCoreOp.rowOuter, baseInfoOp.vectorCoreNum);
    splitCoreOp.usedCoreNum = CeilDivide(splitCoreOp.rowOuter, splitCoreOp.normalCoreProcessNum);
    splitCoreOp.tailCoreProcessNum =
        splitCoreOp.rowOuter - splitCoreOp.normalCoreProcessNum * (splitCoreOp.usedCoreNum - 1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluQuantTiling::DoStaticQuantNotFullKernelSplitEndAxis()
{
    splitCoreOp.rowInner = 1;
    splitCoreOp.rowOuter = baseInfoOp.fusedFrontAxis;
    splitCoreOp.rowTail = 1;

    int64_t colSplitNum = CeilDivide(baseInfoOp.vectorCoreNum, baseInfoOp.fusedFrontAxis);
    int64_t colInnerTmp = CeilDivide(baseInfoOp.endAxisLen, colSplitNum);
    colInnerTmp = colInnerTmp < SINGLE_CORE_PROCESS_MIN_NUM ? SINGLE_CORE_PROCESS_MIN_NUM :
                                                              AlignToFloor(colInnerTmp, SINGLE_CORE_PROCESS_MIN_NUM);
    if (colInnerTmp > splitCoreOp.coexistentNodeElementNum) {
        colInnerTmp = splitCoreOp.coexistentNodeElementNum;
    }

    splitCoreOp.colInner = colInnerTmp;
    splitCoreOp.colOuter = CeilDivide(baseInfoOp.endAxisLen, colInnerTmp);
    int64_t colTailTmp = GetMod(baseInfoOp.endAxisLen, colInnerTmp);
    splitCoreOp.colTail = colTailTmp == 0 ? splitCoreOp.colInner : colTailTmp;

    splitCoreOp.normalCoreProcessNum =
        CeilDivide(splitCoreOp.rowOuter * splitCoreOp.colOuter, baseInfoOp.vectorCoreNum);
    splitCoreOp.usedCoreNum = CeilDivide(splitCoreOp.rowOuter * splitCoreOp.colOuter, splitCoreOp.normalCoreProcessNum);
    splitCoreOp.tailCoreProcessNum =
        splitCoreOp.rowOuter * splitCoreOp.colOuter - splitCoreOp.normalCoreProcessNum * (splitCoreOp.usedCoreNum - 1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluQuantTiling::DoStaticQuantTiling()
{
    OP_LOGD(nodeName_, "[GeluQuant] DoStaticQuantTiling start running.");
    if (baseInfoOp.inputScaleType == SCALAR_TENSOR) {
        splitCoreOp.templateMode = STATIC_PER_TENSOR_TEMPLATE;
        return DoStaticQuantPerTensorTiling();
    }

    splitCoreOp.coexistentNodeNum = STATIC_QUANT_COEXISTING_QUANTITY;
    splitCoreOp.coexistentNodeElementNum = AlignToFloor((SafeDivide(static_cast<int64_t>(baseInfoOp.ubSize),
        static_cast<int64_t>(sizeof(float)) * splitCoreOp.coexistentNodeNum)),
        FP32_BLOCK_NUM);

    splitCoreOp.normalCoreProcessNum = CeilDivide(baseInfoOp.fusedFrontAxis, baseInfoOp.vectorCoreNum);
    splitCoreOp.usedCoreNum = CeilDivide(baseInfoOp.fusedFrontAxis, splitCoreOp.normalCoreProcessNum);
    splitCoreOp.tailCoreProcessNum =
        baseInfoOp.fusedFrontAxis - splitCoreOp.normalCoreProcessNum * (splitCoreOp.usedCoreNum - 1);

    int64_t mulRowsInUb = splitCoreOp.coexistentNodeElementNum / baseInfoOp.endAxisLenAligned;
    if (baseInfoOp.fusedFrontAxis >= baseInfoOp.vectorCoreNum && mulRowsInUb < TWO_END_AXIS) {
        splitCoreOp.templateMode = STATIC_FUNCTION_TEMPLATE; // 满核模式 正常尾轴或者大尾轴
        return ge::GRAPH_SUCCESS;
    }

    splitCoreOp.templateMode = STATIC_PERFORMANCE_TEMPLATE;
    if (baseInfoOp.fusedFrontAxis >= baseInfoOp.vectorCoreNum) { // 满核模式 小尾轴
        return DoStaticQuantFullKernelSmallEndAxis();
    } else {
        return DoStaticQuantNotFullKernelSplitEndAxis();
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluQuantTiling::DoDynamicQuantTiling()
{
    OP_LOGD(nodeName_, "[GeluQuant] DoDynamicQuantTiling start running.");
    splitCoreOp.normalCoreProcessNum = CeilDivide(baseInfoOp.fusedFrontAxis, baseInfoOp.vectorCoreNum);
    splitCoreOp.usedCoreNum = CeilDivide(baseInfoOp.fusedFrontAxis, splitCoreOp.normalCoreProcessNum);
    splitCoreOp.tailCoreProcessNum =
        baseInfoOp.fusedFrontAxis - splitCoreOp.normalCoreProcessNum * (splitCoreOp.usedCoreNum - 1);

    splitCoreOp.coexistentNodeNum = DYNAMIC_QUANT_COEXISTING_QUANTITY;
    splitCoreOp.coexistentNodeElementNum = AlignToFloor((SafeDivide(static_cast<int64_t>(baseInfoOp.ubSize),
        static_cast<int64_t>(sizeof(float)) * splitCoreOp.coexistentNodeNum)),
        FP32_BLOCK_NUM);

    int64_t mulRowsInUb = splitCoreOp.coexistentNodeElementNum / baseInfoOp.endAxisLenAligned;

    if (mulRowsInUb == 0) {
        splitCoreOp.templateMode = DYNAMIC_WORKSPACE_TEMPLATE;
        splitCoreOp.coexistentNodeNum = DYNAMIC_QUANT_WORKSPACE_COEXISTING_QUANTITY;
        splitCoreOp.coexistentNodeElementNum = AlignToFloor(SafeDivide(static_cast<int64_t>(baseInfoOp.ubSize),
            static_cast<int64_t>(sizeof(float)) * splitCoreOp.coexistentNodeNum),
            FP32_BLOCK_NUM);
    } else {
        splitCoreOp.templateMode = DYNAMIC_NORMAL_TEMPLATE;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluQuantTiling::DoTiling()
{
    OP_LOGD(nodeName_, "[GeluQuant] DoTiling start running.");
    if (baseInfoOp.quantMode == STATIC_QUANT_MODE) {
        DoStaticQuantTiling();
    } else {
        DoDynamicQuantTiling();
    }

    OP_LOGD(nodeName_, "[GeluQuant] DoTiling run completed.");
    return ge::GRAPH_SUCCESS;
}

void GeluQuantTiling::SaveToTilingData()
{
    tilingData.set_usedCoreNum(splitCoreOp.usedCoreNum);
    tilingData.set_normalCoreProcessNum(splitCoreOp.normalCoreProcessNum);
    tilingData.set_tailCoreProcessNum(splitCoreOp.tailCoreProcessNum);
    tilingData.set_coexistentNodeNum(splitCoreOp.coexistentNodeNum);
    tilingData.set_coexistentNodeElementNum(splitCoreOp.coexistentNodeElementNum);
    tilingData.set_rowInner(splitCoreOp.rowInner);
    tilingData.set_rowOuter(splitCoreOp.rowOuter);
    tilingData.set_rowTail(splitCoreOp.rowTail);
    tilingData.set_colInner(splitCoreOp.colInner);
    tilingData.set_colOuter(splitCoreOp.colOuter);
    tilingData.set_colTail(splitCoreOp.colTail);
    tilingData.set_tilingKey(splitCoreOp.tilingKey);

    tilingData.set_endAxisLen(baseInfoOp.endAxisLen);
    tilingData.set_endAxisLenAligned(baseInfoOp.endAxisLenAligned);
    tilingData.set_quantMode(baseInfoOp.quantMode);
    tilingData.set_approximate(baseInfoOp.approximate);
    tilingData.set_inputScaleType(baseInfoOp.inputScaleType);
    tilingData.set_inputOffsetType(baseInfoOp.inputOffsetType);
}

ge::graphStatus GeluQuantTiling::PostTiling()
{
    size_t *userWorkspaceSize = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, userWorkspaceSize);
    size_t workspaceSize = WORKSPACE_BUFFER;
    if (splitCoreOp.templateMode == DYNAMIC_WORKSPACE_TEMPLATE) {
        workspaceSize += baseInfoOp.endAxisLen * sizeof(float) * splitCoreOp.usedCoreNum;
    }

    userWorkspaceSize[0] = workspaceSize;

    splitCoreOp.tilingKey = GetTilingKey();
    SaveToTilingData();

    context_->SetBlockDim(splitCoreOp.usedCoreNum);
    if (tilingData.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }

    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    context_->SetTilingKey(splitCoreOp.tilingKey);
    OP_LOGD(nodeName_, "[GeluQuant] PostTiling run completed");
    return ge::GRAPH_SUCCESS;
}

uint64_t ComputeTilingKey(const std::vector<int64_t> &args)
{
    uint64_t result = 1000UL;
    uint64_t startValue = 1;
    constexpr uint64_t incrementCoefficient = 10;
    for (auto it = args.rbegin(); it != args.rend(); ++it) {
        result += *it * startValue;
        startValue *= incrementCoefficient;
    }
    return result;
}

uint64_t GeluQuantTiling::GetTilingKey() const
{
    InputDataType inputDataType = InputDataType::FLOAT_FLOAT;
    if (baseInfoOp.scaleInputDtype == ge::DT_FLOAT16) {
        inputDataType = InputDataType::HALF_HALF;
    } else if (baseInfoOp.scaleInputDtype == ge::DT_BF16) {
        inputDataType = InputDataType::BF16_BF16;
    } else if (baseInfoOp.xInputDtype == ge::DT_FLOAT) {
        inputDataType = InputDataType::FLOAT_FLOAT;
    } else if (baseInfoOp.xInputDtype == ge::DT_FLOAT16) {
        inputDataType = InputDataType::HALF_FLOAT;
    } else {
        inputDataType = InputDataType::BF16_FLOAT;
    }
    std::vector<int64_t> args{ splitCoreOp.templateMode, static_cast<int64_t>(inputDataType) };
    auto tilingKey = ComputeTilingKey(args);
    OP_LOGD(nodeName_, "[GeluQuant] GetTilingKey [%lu].", tilingKey);
    return tilingKey;
}

void GeluQuantTiling::DumpTilingInfo() const
{
    OP_LOGD(nodeName_, "[GeluQuant] DumpTilingInfo start running");

    std::ostringstream info;
    info << "GeluQuantTiling input info: " << std::endl;
    info << "baseInfoOp.vectorCoreNum: " << baseInfoOp.vectorCoreNum << std::endl;
    info << "baseInfoOp.ubSize: " << baseInfoOp.ubSize << std::endl;
    info << "baseInfoOp.xDimNum: " << baseInfoOp.xDimNum << std::endl;
    info << "baseInfoOp.endAxisLen: " << baseInfoOp.endAxisLen << std::endl;
    info << "baseInfoOp.endAxisLenAligned: " << baseInfoOp.endAxisLenAligned << std::endl;
    info << "baseInfoOp.fusedFrontAxis: " << baseInfoOp.fusedFrontAxis << std::endl;
    info << "baseInfoOp.fusedAllAxis: " << baseInfoOp.fusedAllAxis << std::endl;
    info << "baseInfoOp.elementNumAlign: " << baseInfoOp.elementNumAlign << std::endl;
    info << "dtype map: 0 [float]  1 [float16]  27 [bf16]  " << std::endl;
    info << "baseInfoOp.xInputDtype: " << baseInfoOp.xInputDtype << std::endl;
    info << "baseInfoOp.scaleInputDtype: " << baseInfoOp.scaleInputDtype << std::endl;
    info << "baseInfoOp.offsetInputDtype: " << baseInfoOp.offsetInputDtype << std::endl;
    info << "baseInfoOp.quantMode: " << baseInfoOp.quantMode << " [0:static  1:dynamic] " << std::endl;
    info << "baseInfoOp.approximate: " << baseInfoOp.approximate << " [0:none  1:tanh] " << std::endl;
    info << "input type map: 0 [empty]  1 [scalar]  2 [normal]  " << std::endl;
    info << "baseInfoOp.inputScaleType: " << baseInfoOp.inputScaleType << std::endl;
    info << "baseInfoOp.inputOffsetType: " << baseInfoOp.inputOffsetType << std::endl;
    OP_LOGD(nodeName_, "%s", info.str().c_str());
    info.str("");

    info << "GeluQuantTiling split info: " << std::endl;
    info << "splitCoreOp.usedCoreNum: " << splitCoreOp.usedCoreNum << std::endl;
    info << "splitCoreOp.normalCoreProcessNum: " << splitCoreOp.normalCoreProcessNum << std::endl;
    info << "splitCoreOp.tailCoreProcessNum: " << splitCoreOp.tailCoreProcessNum << std::endl;
    info << "splitCoreOp.coexistentNodeNum: " << splitCoreOp.coexistentNodeNum << std::endl;
    info << "splitCoreOp.coexistentNodeElementNum: " << splitCoreOp.coexistentNodeElementNum << std::endl;
    info << "templateMode: 0 [static_per_tensor]  1 [static_function]  2 [static_performance]  3 [dynamic_normal]  4 "
        "[dynamic_workspace] " <<
        std::endl;
    info << "splitCoreOp.templateMode: " << splitCoreOp.templateMode << std::endl;
    info << "splitCoreOp.rowInner: " << splitCoreOp.rowInner << std::endl;
    info << "splitCoreOp.rowOuter: " << splitCoreOp.rowOuter << std::endl;
    info << "splitCoreOp.rowTail: " << splitCoreOp.rowTail << std::endl;
    info << "splitCoreOp.colInner: " << splitCoreOp.colInner << std::endl;
    info << "splitCoreOp.colOuter: " << splitCoreOp.colOuter << std::endl;
    info << "splitCoreOp.colTail: " << splitCoreOp.colTail << std::endl;
    info << "splitCoreOp.tilingKey: " << splitCoreOp.tilingKey << std::endl;

    OP_LOGD(nodeName_, "%s", info.str().c_str());
}

static ge::graphStatus TilingForGeluQuant(gert::TilingContext *context)
{
    OP_TILING_CHECK(context == nullptr, VECTOR_INNER_ERR_REPORT_TILIING("GeluQuant", "context should not be nullptr."),
        return ge::GRAPH_FAILED);

    GeluQuantTiling tiling(context);
    auto ret = tiling.RunGeluQuantTiling();
    return ret;
}

ge::graphStatus TilingPrepareForGeluQuant(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}


IMPL_OP_OPTILING(GeluQuant).Tiling(TilingForGeluQuant).TilingParse<GeluQuantCompileInfo>(TilingPrepareForGeluQuant);
} // namespace optiling