/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <numeric>
#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "tiling/tiling_api.h"
#include "scatter_list_tiling.h"

// tools api
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
namespace ops {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }
}  // namespace ops
namespace optiling {
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)
}  // namespace optiling
namespace optiling {

static const int64_t VAR_INDEX = 0;
static const int64_t MASK_INDEX = 3;
static const int64_t DIM_0 = 0;
static const int64_t DIM_1 = 1;
static const int64_t DIM_2 = 2;
static const int64_t DIM_3 = 3;
static const int64_t DIM_4 = 4;
static const int64_t DIM_NEG_2 = -2;
static const int64_t DIM_NEG_1 = -1;
static const int64_t DOUBLE_SIZE = 2;
static const int64_t BLOCK_SIZE = 32;
static const int64_t INT64_SIZE = 8;
static const int64_t REPEAT_NUM = 16;
static const int64_t MAX_COUNT = 4095;
static const int64_t RESERVED_UB_SIZE = 4 * BLOCK_SIZE;
static const int64_t WORK_SPACE_SIZE = 32;
static const int64_t NUM_ONE = 1;

class ScatterListTiling {
public:
    explicit ScatterListTiling(gert::TilingContext* tilingContext) : context(tilingContext){};
    ge::graphStatus GetPlatformData();
    ge::graphStatus GetVarTensorNum();
    ge::graphStatus GetVarAndUpdateData();
    ge::graphStatus ReShape(const gert::Shape& updatesShape, const gert::Shape& varTensorShape);
    ge::graphStatus GetIndiceData();
    ge::graphStatus GetMaskData();
    bool ChooseGetOtherData() const;
    ge::graphStatus GetOtherData();
    ge::graphStatus CalculateParams();
    ge::graphStatus GetOtherDataNeg1();
    ge::graphStatus GetDataNeg(const int64_t &maxUbSize, const int64_t &preCoreBatchUbSize,
                               const int64_t &updateSizeMore);
    ge::graphStatus GetOtherDataNeg2();
    ge::graphStatus GetRSBSEData();
    ge::graphStatus GetRLBSEData();
    ge::graphStatus GetRSBLEData();
    ge::graphStatus GetRLBLEData();
    ge::graphStatus GetRLBSEPadData();
    ge::graphStatus GetRLBLEPadData();
    ge::graphStatus SetTilingData();
    void PrintTilingData();
    ge::graphStatus GetOtherDataNegMovePad(const int64_t& maxUbSize, const int64_t& preDim23UbSize);
    int64_t CeilDiv(const int64_t& value, const int64_t& factor) const;
    int64_t CeilDivMul(const int64_t& value, const int64_t& factor) const;

private:
    ScatterListTilingData tilingData;
    gert::TilingContext* context = nullptr;
    bool isNeg1 = false;
    bool supportMovePad = false;
    bool isAscend910 = false;

    int64_t totalCoreNum = 0;
    int64_t totalUbSize = 0;
    int64_t varTensorNum = 0;
    int64_t varTensorDims = 0;
    int64_t updatesDims = 0;
    int64_t newAxis = 0;
    vector<int64_t> updateShape;
    vector<int64_t> varShape;
    int64_t indiceIndex = 0;
    int64_t updatesIndex = 0;
    ge::DataType updatesDtype = ge::DT_UNDEFINED;
    int64_t updatesDtypeSize = 0;
    int64_t indiceOneBlock = 0;
    int64_t maxUpdatesUbCount = 0;
    int64_t preCoreSrcCount = 0;
    int64_t dim01 = 0;
    int64_t dim012 = 0;

    int64_t dim0Count = 0;
    int64_t dim1Count = 0;
    int64_t dim2Count = 0;
    int64_t dim3Count = 0;
    int64_t varDim1Count = 0;
    int64_t varDim2Count = 0;
    int64_t dim3CountAlign = 0;
    int64_t updatesOneBlock = 0;
    int64_t indiceDims = 0;
    int64_t indiceCount = 0;
    int64_t indiceUbSize = 0;
    int64_t maskCount = 0;
    int64_t maskUbSize = 0;
    int64_t srcBatchStride = 0;
    int64_t srcBatchStrideAlign = 0;
    int64_t dstBatchStride = 0;
    int64_t useCoreNum = 0;
    int64_t preCoreBatchNum = 0;
    int64_t lastCoreBatchNum = 0;
    int64_t eachLoopNum = 0;
    int64_t eachPreLoopEle = 0;
    int64_t eachLastLoopEle = 0;
    int64_t eachLastLoopEleAlign = 0;
    int64_t updatesCount = 0;
    int64_t updatesUbSize = 0;
    int64_t dataUbSize = 0;
    int64_t transposeUbSize = 0;
    int64_t transRepeatTimes = 0;
    int64_t transRepeatTimesTail = 0;
    int64_t updateDim23Align = 0;
    int64_t preCoreUpdateDim23 = 0;
    int64_t varDim3Stride = 0;
    int64_t varDim3Count = 0;
    int64_t dim3CountSize = 0;
    int64_t eachLastSize = 0;
    int64_t tilingKey = 0;
};

ge::graphStatus ScatterListTiling::GetPlatformData() {
    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    platform_ascendc::SocVersion socVersion = ascendcPlatform.GetSocVersion();
    supportMovePad = (socVersion == platform_ascendc::SocVersion::ASCEND910B);
    isAscend910 = (socVersion == platform_ascendc::SocVersion::ASCEND910);

    totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_TILING_CHECK(totalCoreNum <= 0, OP_LOGE(context->GetNodeName(), "PrepareTiling fail to get core num."),
                    return ge::GRAPH_FAILED);
    uint64_t platformUbSize = 0;
    platformInfo->GetLocalMemSize(fe::LocalMemType::UB, platformUbSize);
    totalUbSize = platformUbSize;
    OP_TILING_CHECK(totalUbSize <= 0, OP_LOGE(context->GetNodeName(), "PrepareTiling fail to get ub size."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

int64_t ScatterListTiling::CeilDiv(const int64_t& value, const int64_t& factor) const {
    if (factor == 0) {
        return value;
    }
    return (value + factor - 1) / factor;
}

int64_t ScatterListTiling::CeilDivMul(const int64_t &value, const int64_t &factor) const {
    if (factor == 0) {
        return value;
    }
    return CeilDiv(value, factor) * factor;
}

ge::graphStatus ScatterListTiling::GetVarTensorNum() {
    auto computeNodeInfoPtr = context->GetComputeNodeInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, computeNodeInfoPtr);
    auto anchorInstanceInfoPtr = computeNodeInfoPtr->GetInputInstanceInfo(VAR_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, anchorInstanceInfoPtr);
    varTensorNum = anchorInstanceInfoPtr->GetInstanceNum();
    OP_TILING_CHECK(varTensorNum == 0, OP_LOGE(context->GetNodeName(), "var can not be a empty tensor list"),
                    return ge::GRAPH_FAILED);
    auto varTensorShapePtr = context->GetDynamicInputShape(VAR_INDEX, 0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, varTensorShapePtr);
    auto varTensorShape = varTensorShapePtr->GetStorageShape();
    for (int64_t i = 1; i < varTensorNum; i++) {
        auto varTensorShape2Ptr = context->GetDynamicInputShape(VAR_INDEX, i);
        OPS_CHECK_NULL_WITH_CONTEXT(context, varTensorShape2Ptr);
        auto varTensorShape2 = varTensorShape2Ptr->GetStorageShape();
        OP_TILING_CHECK(varTensorShape2 != varTensorShape,
                        OP_LOGE(context->GetNodeName(), "all var tensor shape should be equal"),
                        return ge::GRAPH_FAILED);
    }
    indiceIndex = varTensorNum;
    updatesIndex = varTensorNum + 1;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterListTiling::ReShape(const gert::Shape& updatesShape, const gert::Shape& varTensorShape) {
    for (int64_t i = 0; i < updatesDims; ++i) {
        OP_TILING_CHECK(updatesShape.GetDim(i) <= 0, OP_LOGE(context->GetNodeName(), "updates can't be empty tensor"),
                        return ge::GRAPH_FAILED);
        updateShape.push_back(updatesShape.GetDim(i));
        if (i < (updatesDims - 1)) {
            OP_TILING_CHECK(varTensorShape.GetDim(i) <= 0,
                            OP_LOGE(context->GetNodeName(), "var tensor can't be empty tensor"),
                            return ge::GRAPH_FAILED);
            varShape.push_back(varTensorShape.GetDim(i));
        }
    }
    isNeg1 = (newAxis == (updatesDims - 1));
    dim2Count = updatesShape.GetDim(newAxis);
    varDim2Count = varTensorShape.GetDim(newAxis - 1);
    dim1Count = std::accumulate(updateShape.begin() + 1, updateShape.begin() + newAxis, NUM_ONE,
                                std::multiplies<int64_t>());
    varDim1Count = std::accumulate(varShape.begin(), varShape.begin() + newAxis - 1, NUM_ONE,
                                   std::multiplies<int64_t>());
    dim3Count = std::accumulate(updateShape.begin() + newAxis + 1, updateShape.end(), NUM_ONE,
                                std::multiplies<int64_t>());
    varDim3Count = std::accumulate(varShape.begin() + newAxis, varShape.end(), NUM_ONE, std::multiplies<int64_t>());

    OP_TILING_CHECK(varDim1Count != dim1Count,
                    OP_LOGE(context->GetNodeName(), "var shape1 must be equal update shape1"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(varDim3Count != dim3Count,
                    OP_LOGE(context->GetNodeName(), "var shape3 must be equal update shape3"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(dim2Count > varDim2Count, OP_LOGE(context->GetNodeName(), "update shape2 must < var shape2"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterListTiling::GetVarAndUpdateData() {
    auto varTensorShapePtr = context->GetDynamicInputShape(VAR_INDEX, 0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, varTensorShapePtr);
    auto varTensorShape = varTensorShapePtr->GetStorageShape();
    varTensorDims = varTensorShape.GetDimNum();
    OP_TILING_CHECK(varTensorDims < DIM_1, OP_LOGE(context->GetNodeName(), "The dimension of var must > 1"),
                    return ge::GRAPH_FAILED);
    auto updatesShapePtr = context->GetInputShape(updatesIndex);
    OPS_CHECK_NULL_WITH_CONTEXT(context, updatesShapePtr);
    auto updatesShape = updatesShapePtr->GetStorageShape();
    updatesDims = updatesShape.GetDimNum();
    dim0Count = updatesShape.GetDim(DIM_0);
    OP_TILING_CHECK(updatesDims < 2, OP_LOGE(context->GetNodeName(), "The dimension of updates must >= 2"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(varTensorNum != dim0Count,
                    OP_LOGE(context->GetNodeName(), "The tensor num of var must equal to update first dim"),
                    return ge::GRAPH_FAILED);
    const int64_t* axis = context->GetAttrs()->GetAttrPointer<int64_t>(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, axis);
    newAxis = *axis;
    newAxis = newAxis < 0 ? (updatesDims + newAxis) : newAxis;
    OP_TILING_CHECK((newAxis <= 0 || newAxis >= updatesDims), OP_LOGE(context->GetNodeName(), "axis is error"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(ReShape(updatesShape, varTensorShape) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "get var and update failed"), return ge::GRAPH_FAILED);
    auto varTensorDescPtr = context->GetDynamicInputDesc(VAR_INDEX, 0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, varTensorDescPtr);
    auto varTensorDtype = varTensorDescPtr->GetDataType();
    auto updatesDescPtr = context->GetInputDesc(updatesIndex);
    OPS_CHECK_NULL_WITH_CONTEXT(context, updatesDescPtr);
    updatesDtype = updatesDescPtr->GetDataType();
    updatesDtypeSize = ge::GetSizeByDataType(updatesDtype);
    OP_TILING_CHECK(updatesDtypeSize <= 0,
                    OP_LOGE(context->GetNodeName(), "typeSize is invalid %ld, please check.", updatesDtypeSize),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(varTensorDtype != updatesDtype,
                    OP_LOGE(context->GetNodeName(),
                            "The dtype of all elements of var should be euqal to the dtype of updates"),
                    return ge::GRAPH_FAILED);
    updatesOneBlock = BLOCK_SIZE / updatesDtypeSize;
    dim3CountAlign = CeilDivMul(dim3Count, updatesOneBlock);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterListTiling::GetIndiceData() {
    auto indiceShapePtr = context->GetInputShape(indiceIndex);
    OPS_CHECK_NULL_WITH_CONTEXT(context, indiceShapePtr);
    auto indiceShape = indiceShapePtr->GetStorageShape();
    indiceDims = indiceShape.GetDimNum();
    OP_TILING_CHECK(indiceDims != DIM_1 && indiceDims != DIM_2,
                    OP_LOGE(context->GetNodeName(), "The dimension of indice only support 1 or 2 now"),
                    return ge::GRAPH_FAILED);
    auto indiceDim0 = indiceShape.GetDim(DIM_0);
    OP_TILING_CHECK(indiceDim0 != dim0Count,
                    OP_LOGE(context->GetNodeName(),
                            "The first dim of indice should be euqal to the first dim of updates"),
                    return ge::GRAPH_FAILED);
    indiceCount = dim0Count;
    if (indiceDims == DIM_2) {
        OP_TILING_CHECK(indiceShape.GetDim(DIM_1) != DOUBLE_SIZE,
                        OP_LOGE(context->GetNodeName(), "The second dim of indice should be 2"),
                        return ge::GRAPH_FAILED);
        indiceCount = indiceCount * DOUBLE_SIZE;
    }
    auto indiceDescPtr = context->GetInputDesc(indiceIndex);
    OPS_CHECK_NULL_WITH_CONTEXT(context, indiceDescPtr);
    auto indiceDtype = indiceDescPtr->GetDataType();
    int64_t indiceDtypeSize = ge::GetSizeByDataType(indiceDtype);
    OP_TILING_CHECK(indiceDtypeSize <= 0,
                    OP_LOGE(context->GetNodeName(), "typeSize is invalid %ld, please check.", indiceDtypeSize),
                    return ge::GRAPH_FAILED);
    indiceOneBlock = BLOCK_SIZE / indiceDtypeSize;
    indiceCount = CeilDivMul(indiceCount, indiceOneBlock);
    indiceUbSize = indiceCount * indiceDtypeSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterListTiling::GetMaskData() {
    auto mask = context->GetOptionalInputTensor(MASK_INDEX);
    if (mask == nullptr) {
        OP_LOGD(context->GetNodeName(), "mask is null, ScatterList GetMaskData running end"); return ge::GRAPH_SUCCESS;
    }

    auto maskShapePtr = context->GetOptionalInputShape(MASK_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, maskShapePtr);
    auto maskShape = maskShapePtr->GetStorageShape();
    auto maskDims = maskShape.GetDimNum();
    OP_TILING_CHECK(maskDims != DIM_1, OP_LOGE(context->GetNodeName(), "The dimension of mask only support 1 now"),
                    return ge::GRAPH_FAILED);
    auto maskDim0 = maskShape.GetDim(DIM_0);
    OP_TILING_CHECK(maskDim0 != dim0Count,
                    OP_LOGE(context->GetNodeName(), "The first dim of mask must be euqal to the first dim of updates"),
                    return ge::GRAPH_FAILED);
    maskCount = dim0Count;
    auto maskDescPtr = context->GetOptionalInputDesc(MASK_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, maskDescPtr);
    auto maskDtype = maskDescPtr->GetDataType();
    int64_t maskDtypeSize = ge::GetSizeByDataType(maskDtype);
    OP_TILING_CHECK(maskDtypeSize <= 0,
                    OP_LOGE(context->GetNodeName(), "typeSize is invalid %ld, please check.", maskDtypeSize),
                    return ge::GRAPH_FAILED);
    auto maskOneBlock = BLOCK_SIZE / maskDtypeSize;
    maskCount = CeilDivMul(maskCount, maskOneBlock);
    maskUbSize = maskCount * maskDtypeSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterListTiling::GetOtherDataNegMovePad(const int64_t& maxUbSize, const int64_t& preDim23UbSize) {
    dim3CountAlign = CeilDiv(dim3Count, updatesOneBlock);
    int64_t eSizeAlign = dim3CountAlign * updatesOneBlock;
    updatesUbSize = preDim23UbSize;
    varDim3Stride = (varDim3Count - dim3Count) * updatesDtypeSize;
    dim3CountSize = dim3Count * updatesDtypeSize;
    if (!supportMovePad) {
       varDim3Stride = (varDim3Count - dim3Count) / updatesOneBlock;
       dim3CountSize = dim3Count / updatesOneBlock;
    }
    int64_t updatesUbSizeLarge = dim2Count * eSizeAlign * updatesDtypeSize;
    int64_t updatesUbSizeLargeE = eSizeAlign * updatesDtypeSize;
    if ((dim3Count % updatesOneBlock == 0) && (maxUbSize >= updatesUbSize)) {
        tilingKey = static_cast<int64_t>(ScatterListTilingKey::TILINGKEY_PSMALL);  // e is align and ub > n*c*e
        preCoreUpdateDim23 = preCoreBatchNum * srcBatchStride;
        return ge::GRAPH_SUCCESS;
    }
    if ((maxUbSize >= updatesUbSizeLarge) && (updatesUbSizeLarge / eSizeAlign <= MAX_COUNT)) {
        tilingKey = static_cast<int64_t>(ScatterListTilingKey::TILINGKEY_PMORE);  // ub >= c*e or e is not align
        updatesUbSize = updatesUbSizeLarge;
        return ge::GRAPH_SUCCESS;
    }
    if (maxUbSize >= updatesUbSizeLargeE) {
        tilingKey = static_cast<int64_t>(ScatterListTilingKey::TILINGKEY_PLARGE);  // ub < c*e and ub > e
        int64_t preNum = maxUbSize / (eSizeAlign * updatesDtypeSize);
        eachPreLoopEle = preNum > MAX_COUNT ? MAX_COUNT : preNum;
        updatesUbSize = eachPreLoopEle * eSizeAlign * updatesDtypeSize;
        eachLoopNum = CeilDiv(dim2Count, eachPreLoopEle);
        eachLastLoopEle = dim2Count - (eachLoopNum - 1) * eachPreLoopEle;
        eachLoopNum = eachLoopNum - 1;
        preCoreUpdateDim23 = eachPreLoopEle * dim3Count;
        eachLastSize = eachLastLoopEle * dim3Count;
        return ge::GRAPH_SUCCESS;
    }
    tilingKey = static_cast<int64_t>(ScatterListTilingKey::TILINGKEY_PLE);  // ub < e
    dim1Count = dim1Count * dim2Count;
    dim01 = dim0Count * dim1Count;
    preCoreBatchNum = CeilDiv(dim01, totalCoreNum);
    useCoreNum = CeilDiv(dim01, preCoreBatchNum);
    lastCoreBatchNum = dim01 - preCoreBatchNum * (useCoreNum - 1);
    updatesUbSize = ((maxUbSize - RESERVED_UB_SIZE) / BLOCK_SIZE) * BLOCK_SIZE;
    eachPreLoopEle = (updatesUbSize / updatesDtypeSize / updatesOneBlock) * updatesOneBlock;
    eachLoopNum = CeilDiv(dim3Count, eachPreLoopEle);
    eachLastLoopEle = dim3Count - (eachLoopNum - 1) * eachPreLoopEle;
    eachLoopNum = eachLoopNum - 1;
    eachLastSize = eachLastLoopEle * updatesDtypeSize;
    eachLastLoopEleAlign = CeilDivMul(eachLastLoopEle, updatesOneBlock);
    tilingKey = indiceDims == DIM_2 ? static_cast<int64_t>(ScatterListTilingKey::TILINGKEY_PLEDIM2) : tilingKey;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterListTiling::CalculateParams() {
    dim3Count = dim2Count;
    varDim3Count = varDim2Count;
    if (updatesDims == DIM_2) {
        dim1Count = DIM_1;
        dim2Count = DIM_1;
    } else if (updatesDims == DIM_3) {
        dim1Count = DIM_1;
        dim2Count = updateShape[1];
    } else {
        dim1Count = updateShape[1];
        dim2Count = std::accumulate(updateShape.begin() + DIM_2, updateShape.end() - 1, NUM_ONE,
                                    std::multiplies<int64_t>());
    }
    varDim1Count = dim1Count;
    varDim2Count = dim2Count;
    srcBatchStride = dim2Count * dim3Count;
    dstBatchStride = varDim2Count * varDim3Count;

    dim01 = dim0Count * dim1Count;
    preCoreBatchNum = CeilDiv(dim01, totalCoreNum);
    useCoreNum = CeilDiv(dim01, preCoreBatchNum);
    lastCoreBatchNum = dim01 - preCoreBatchNum * (useCoreNum - 1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterListTiling::GetDataNeg(const int64_t &maxUbSize, const int64_t &preCoreBatchUbSize,
                                              const int64_t &updateSizeMore) {
    bool isAlign = (varDim3Count % updatesOneBlock == 0);
    bool isAlignDim2 = (dim2Count % updatesOneBlock == 0);
    bool istransSmall = (maxUbSize >= (dataUbSize + updatesUbSize + transposeUbSize));
    bool istransMore = (maxUbSize >= (dataUbSize + updateSizeMore + transposeUbSize));
    bool dimIsOne = (dim3Count == DIM_1 && indiceDims == 1);

    if (isAlign && isAlignDim2 && dimIsOne && (updatesDtypeSize != INT64_SIZE) && istransSmall) {
        tilingKey = static_cast<int64_t>(ScatterListTilingKey::TILINGKEY_TSMALL);  // ub >= n*c*e and e = 1
        return ge::GRAPH_SUCCESS;
    }
    if (isAlign && dimIsOne && (updatesDtypeSize != INT64_SIZE) && istransMore) {
        tilingKey = static_cast<int64_t>(ScatterListTilingKey::TILINGKEY_TMORE);  // ub >= c*e and e = 1
        updatesUbSize = updateSizeMore;
        return ge::GRAPH_SUCCESS;
    }
    if (isAlign && dimIsOne && (updatesDtypeSize != INT64_SIZE) && (!istransMore)) {
        tilingKey = static_cast<int64_t>(ScatterListTilingKey::TILINGKEY_TLARGE);  // ub < c*e
        eachPreLoopEle = ((maxUbSize / updatesDtypeSize) / (DIM_2 * updatesOneBlock + 1)) / updatesOneBlock *
                         updatesOneBlock;
        eachPreLoopEle = (eachPreLoopEle / REPEAT_NUM) * REPEAT_NUM;

        eachPreLoopEle = updatesDtypeSize != 1 ? eachPreLoopEle : (eachPreLoopEle / BLOCK_SIZE) * BLOCK_SIZE;
        updatesUbSize = eachPreLoopEle * updatesDtypeSize;
        eachLoopNum = CeilDiv(varDim2Count, eachPreLoopEle);
        eachLastLoopEle = varDim2Count - (eachLoopNum - 1) * eachPreLoopEle;
        eachLastSize = CeilDivMul(eachLastLoopEle, REPEAT_NUM);
        eachLastSize = updatesDtypeSize != 1 ? eachLastSize : CeilDivMul(eachLastLoopEle, BLOCK_SIZE);
        dataUbSize = eachPreLoopEle * updatesOneBlock * updatesDtypeSize;
        transposeUbSize = dataUbSize;
        transRepeatTimes = eachPreLoopEle / REPEAT_NUM;
        transRepeatTimesTail = eachLastSize / REPEAT_NUM;
        eachLoopNum = eachLoopNum - 1;
        updateDim23Align = CeilDivMul(eachPreLoopEle, updatesOneBlock);
        srcBatchStrideAlign = CeilDivMul(eachLastLoopEle, updatesOneBlock);
        if (updatesDtypeSize == 1) {
            transRepeatTimes = (eachPreLoopEle * BLOCK_SIZE) / (REPEAT_NUM * REPEAT_NUM) / DIM_4;
            transRepeatTimesTail = (eachLastSize * BLOCK_SIZE) / (REPEAT_NUM * REPEAT_NUM) / DIM_4;
        }
        return ge::GRAPH_SUCCESS;
    }
    return GetOtherDataNegMovePad(maxUbSize, preCoreBatchUbSize);
}

ge::graphStatus ScatterListTiling::GetOtherDataNeg1() {
    int64_t maxUbSize = totalUbSize - indiceUbSize - maskUbSize;
    updateDim23Align = CeilDivMul(srcBatchStride, updatesOneBlock);
    preCoreUpdateDim23 = preCoreBatchNum * updateDim23Align;
    int64_t preCoreBatchUbSize = preCoreUpdateDim23 * updatesDtypeSize;
    eachLastSize = CeilDivMul(varDim2Count, REPEAT_NUM);
    dataUbSize = eachLastSize * updatesOneBlock * updatesDtypeSize;
    transposeUbSize = dataUbSize;
    updatesUbSize = preCoreBatchUbSize;
    int64_t updateSizeMore = updateDim23Align * updatesDtypeSize;
    varDim3Stride = varDim3Count * updatesDtypeSize / BLOCK_SIZE - 1;
    transRepeatTimes = eachLastSize / REPEAT_NUM;

    if (updatesDtypeSize == 1) {
        eachLastSize = CeilDivMul(varDim2Count, updatesOneBlock);
        dataUbSize = eachLastSize * updatesOneBlock * updatesDtypeSize;
        transposeUbSize = dataUbSize;
        transRepeatTimes = (eachLastSize * BLOCK_SIZE) / (REPEAT_NUM * REPEAT_NUM) / DIM_4;
    }
    return GetDataNeg(maxUbSize, preCoreBatchUbSize, updateSizeMore);
}

ge::graphStatus ScatterListTiling::GetRSBSEData() {
    updatesCount = preCoreSrcCount;
    updatesUbSize = updatesCount * updatesDtypeSize;
    tilingKey = static_cast<int64_t>(ScatterListTilingKey::TILINGKEY_RSBSE);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterListTiling::GetRLBSEData() {
    updatesCount = srcBatchStride;
    updatesUbSize = updatesCount * updatesDtypeSize;
    tilingKey = static_cast<int64_t>(ScatterListTilingKey::TILINGKEY_RLBSE);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterListTiling::GetRSBLEData() {
    updatesCount = preCoreBatchNum * dim3Count;
    updatesUbSize = updatesCount * updatesDtypeSize;
    tilingKey = static_cast<int64_t>(ScatterListTilingKey::TILINGKEY_RSBLE);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterListTiling::GetRLBLEData() {
    eachLoopNum = CeilDiv(srcBatchStride, maxUpdatesUbCount);
    eachPreLoopEle = CeilDiv(srcBatchStride, eachLoopNum);
    eachPreLoopEle = CeilDivMul(eachPreLoopEle, updatesOneBlock);
    eachLastLoopEle = srcBatchStride - (eachLoopNum - 1) * eachPreLoopEle;
    updatesCount = eachPreLoopEle;
    updatesUbSize = updatesCount * updatesDtypeSize;
    tilingKey = static_cast<int64_t>(ScatterListTilingKey::TILINGKEY_RLBLE);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterListTiling::GetRLBSEPadData() {
    updatesCount = srcBatchStrideAlign;
    updatesUbSize = updatesCount * updatesDtypeSize;
    tilingKey = static_cast<int64_t>(ScatterListTilingKey::TILINGKEY_RLBSE_PAD);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterListTiling::GetRLBLEPadData() {
    eachLoopNum = CeilDiv(srcBatchStride, maxUpdatesUbCount);
    eachPreLoopEle = CeilDiv(srcBatchStride, eachLoopNum);
    eachPreLoopEle = CeilDivMul(eachPreLoopEle, updatesOneBlock);
    eachLastLoopEle = srcBatchStride - (eachLoopNum - 1) * eachPreLoopEle;
    eachLastLoopEleAlign = CeilDivMul(eachLastLoopEle, updatesOneBlock);
    updatesCount = eachPreLoopEle;
    updatesUbSize = updatesCount * updatesDtypeSize;
    tilingKey = static_cast<int64_t>(ScatterListTilingKey::TILINGKEY_RLBLE_PAD);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterListTiling::GetOtherDataNeg2() {
    maxUpdatesUbCount = (totalUbSize - indiceUbSize - maskUbSize - RESERVED_UB_SIZE) / updatesDtypeSize;
    preCoreSrcCount = preCoreBatchNum * srcBatchStride;

    if (dim3Count == dim3CountAlign) {
        if (preCoreSrcCount <= maxUpdatesUbCount) {
            return GetRSBSEData();
        }
        if (srcBatchStride <= maxUpdatesUbCount) {
            return GetRLBSEData();
        }
        dim012 = dim01 * dim2Count;
        auto newPreCoreBatchNum = CeilDiv(dim012, totalCoreNum);
        if (dim01 < totalCoreNum && newPreCoreBatchNum * dim3Count <= maxUpdatesUbCount) {
            preCoreBatchNum = newPreCoreBatchNum;
            useCoreNum = CeilDiv(dim012, preCoreBatchNum);
            lastCoreBatchNum = dim012 - preCoreBatchNum * (useCoreNum - 1);
            return GetRSBLEData();
        }
        return GetRLBLEData();
    } else {
        if (srcBatchStrideAlign <= maxUpdatesUbCount) {
            return GetRLBSEPadData();
        }
        return GetRLBLEPadData();
    }
}

bool ScatterListTiling::ChooseGetOtherData() const {
    bool alignE = (dim2Count % updatesOneBlock == 0);
    bool alignD = (varDim2Count % updatesOneBlock == 0);
    bool supportTrans = (isNeg1 && dim2Count == 1 && alignD && indiceDims == 1);
    bool fp32Trans = (isAscend910 && (updatesDtypeSize == DIM_4));
    if (supportMovePad && isNeg1) {
        return true;
    }
    if (!supportMovePad && isNeg1 && alignE && alignD && indiceDims == 1) {
        return true;
    }
    if (!supportMovePad && supportTrans && (!fp32Trans)) {
        return true;
    }
    return false;
}

ge::graphStatus ScatterListTiling::GetOtherData() {
    const char* reduce = context->GetAttrs()->GetAttrPointer<char>(0);
    OP_TILING_CHECK(strcmp(reduce, "update") != 0, OP_LOGE(context->GetNodeName(), "reduce only support update now"),
                    return ge::GRAPH_FAILED);

    if (ChooseGetOtherData()) {
        CalculateParams();
        return GetOtherDataNeg1();
    }

    srcBatchStride = dim2Count * dim3Count;
    srcBatchStrideAlign = CeilDivMul(srcBatchStride, updatesOneBlock);
    dstBatchStride = varDim2Count * varDim3Count;

    // one block dim
    if (dim3Count != dim3CountAlign && !supportMovePad) {
       if ((indiceDims == DIM_1 && srcBatchStride < updatesOneBlock) ||
           (indiceDims == DIM_2 && dim3Count < updatesOneBlock)) {
          totalCoreNum = 1;
       }
    }

    dim01 = dim0Count * dim1Count;
    preCoreBatchNum = CeilDiv(dim01, totalCoreNum);
    useCoreNum = CeilDiv(dim01, preCoreBatchNum);
    lastCoreBatchNum = dim01 - preCoreBatchNum * (useCoreNum - 1);
    return GetOtherDataNeg2();
}

void ScatterListTiling::PrintTilingData() {
    OP_LOGD(context->GetNodeName(),
            "ScatterList tilingData is dim0Count:%ld, dim1Count:%ld, varDim2Count:%ld, dim2Count:%ld, dim3Count:%ld, "
            "dim3CountAlign:%ld, updatesOneBlock:%ld, indiceDims:%ld, indiceCount:%ld, indiceUbSize:%ld, "
            "maskCount:%ld, maskUbSize:%ld, srcBatchStride:%ld, srcBatchStrideAlign:%ld, dstBatchStride:%ld, "
            "useCoreNum:%ld, preCoreBatchNum:%ld, lastCoreBatchNum:%ld, eachLoopNum:%ld, eachPreLoopEle:%ld, "
            "eachLastLoopEle:%ld, eachLastLoopEleAlign:%ld, updatesCount:%ld, updatesUbSize:%ld, dataUbSize:%ld, "
            "transposeUbSize:%ld, transRepeatTimes:%ld, transRepeatTimesTail:%ld, updateDim23Align:%ld, "
            "preCoreUpdateDim23:%ld, varDim3Stride:%ld, varDim3Count:%ld, dim3CountSize:%ld, eachLastSize:%ld, "
            "tilingKey:%ld",
            tilingData.get_dim0Count(), tilingData.get_dim1Count(), tilingData.get_varDim2Count(),
            tilingData.get_dim2Count(), tilingData.get_dim3Count(), tilingData.get_dim3CountAlign(),
            tilingData.get_updatesOneBlock(), tilingData.get_indiceDims(), tilingData.get_indiceCount(),
            tilingData.get_indiceUbSize(), tilingData.get_maskCount(), tilingData.get_maskUbSize(),
            tilingData.get_srcBatchStride(), tilingData.get_srcBatchStrideAlign(), tilingData.get_dstBatchStride(),
            tilingData.get_useCoreNum(), tilingData.get_preCoreBatchNum(), tilingData.get_lastCoreBatchNum(),
            tilingData.get_eachLoopNum(), tilingData.get_eachPreLoopEle(), tilingData.get_eachLastLoopEle(),
            tilingData.get_eachLastLoopEleAlign(), tilingData.get_updatesCount(), tilingData.get_updatesUbSize(),
            tilingData.get_dataUbSize(), tilingData.get_transposeUbSize(), tilingData.get_transRepeatTimes(),
            tilingData.get_transRepeatTimesTail(), tilingData.get_updateDim23Align(),
            tilingData.get_preCoreUpdateDim23(), tilingData.get_varDim3Stride(), tilingData.get_varDim3Count(),
            tilingData.get_dim3CountSize(), tilingData.get_eachLastSize(), tilingData.get_tilingKey());
    return ;
}

ge::graphStatus ScatterListTiling::SetTilingData() {
    tilingData.set_dim0Count(dim0Count);
    tilingData.set_dim1Count(dim1Count);
    tilingData.set_varDim2Count(varDim2Count);
    tilingData.set_dim2Count(dim2Count);
    tilingData.set_dim3Count(dim3Count);
    tilingData.set_dim3CountAlign(dim3CountAlign);
    tilingData.set_updatesOneBlock(updatesOneBlock);
    tilingData.set_indiceDims(indiceDims);
    tilingData.set_indiceCount(indiceCount);
    tilingData.set_indiceUbSize(indiceUbSize);
    tilingData.set_maskCount(maskCount);
    tilingData.set_maskUbSize(maskUbSize);
    tilingData.set_srcBatchStride(srcBatchStride);
    tilingData.set_srcBatchStrideAlign(srcBatchStrideAlign);
    tilingData.set_dstBatchStride(dstBatchStride);
    tilingData.set_useCoreNum(useCoreNum);
    tilingData.set_preCoreBatchNum(preCoreBatchNum);
    tilingData.set_lastCoreBatchNum(lastCoreBatchNum);
    tilingData.set_eachLoopNum(eachLoopNum);
    tilingData.set_eachPreLoopEle(eachPreLoopEle);
    tilingData.set_eachLastLoopEle(eachLastLoopEle);
    tilingData.set_eachLastLoopEleAlign(eachLastLoopEleAlign);
    tilingData.set_updatesCount(updatesCount);
    tilingData.set_updatesUbSize(updatesUbSize);
    tilingData.set_dataUbSize(dataUbSize);
    tilingData.set_transposeUbSize(transposeUbSize);
    tilingData.set_transRepeatTimes(transRepeatTimes);
    tilingData.set_transRepeatTimesTail(transRepeatTimesTail);
    tilingData.set_updateDim23Align(updateDim23Align);
    tilingData.set_preCoreUpdateDim23(preCoreUpdateDim23);
    tilingData.set_varDim3Stride(varDim3Stride);
    tilingData.set_varDim3Count(varDim3Count);
    tilingData.set_dim3CountSize(dim3CountSize);
    tilingData.set_eachLastSize(eachLastSize);
    tilingData.set_tilingKey(tilingKey);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());

    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    context->SetBlockDim(tilingData.get_useCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());

    size_t* workspaces = context->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = WORK_SPACE_SIZE;
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4ScatterList(gert::TilingContext* context) {
    ScatterListTiling tilingObject(context);
    if (tilingObject.GetPlatformData() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "GetPlatformData return failed.");
        return ge::GRAPH_FAILED;
    }
    if (tilingObject.GetVarTensorNum() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "GetVarTensorNum return failed.");
        return ge::GRAPH_FAILED;
    }
    if (tilingObject.GetVarAndUpdateData() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "GetVarAndUpdateData return failed.");
        return ge::GRAPH_FAILED;
    }
    if (tilingObject.GetIndiceData() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "GetIndiceData return failed.");
        return ge::GRAPH_FAILED;
    }
    if (tilingObject.GetMaskData() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "GetMaskData return failed.");
        return ge::GRAPH_FAILED;
    }
    if (tilingObject.GetOtherData() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "GetOtherData return failed.");
        return ge::GRAPH_FAILED;
    }
    if (tilingObject.SetTilingData() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "SetTilingData return failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4ScatterList(gert::TilingParseContext* context) {
    OP_LOGD(context->GetNodeName(), "TilingPrepare is running.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ScatterList).Tiling(Tiling4ScatterList);
}  // namespace optiling

namespace ge {
namespace ops {
static ge::graphStatus InferShape4ScatterList(gert::InferShapeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do Infershape of ScatterList.");
  auto extend_kernel_context = reinterpret_cast<gert::ExtendedKernelContext*>(context);
  auto output_num = extend_kernel_context->GetComputeNodeOutputNum();
  for (size_t i = 0U; i < output_num; ++i) {
    auto x_shape = context->GetInputShape(i);
    auto y_shape = context->GetOutputShape(i);
    if ((x_shape == nullptr) || (y_shape == nullptr)) {
      return ge::GRAPH_FAILED;
    }
    *y_shape = *x_shape;
  }
  OP_LOGD(context->GetNodeName(), "End to do Infershape of ScatterList.");
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(ScatterList).InferShape(InferShape4ScatterList);
}  // namespace ops
} // namespace ge