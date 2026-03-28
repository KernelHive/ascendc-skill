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
 * \file weight_quant_batch_matmul_v2_tiling.cpp
 * \brief
 */

#include "weight_quant_batch_matmul_v2_tiling.h"

namespace optiling {

constexpr size_t BIAS_INDEX = 6UL;
constexpr size_t MM_SHAPE_LEN_ND = 2UL;
constexpr size_t MM_SHAPE_LEN_NZ = 4UL;
constexpr uint64_t MAX_SHAPE_DIM = 65535UL;
constexpr uint64_t MIN_GROUP_SIZE = 32UL;
constexpr uint64_t MAX_INT32 = 2147483647UL;
constexpr uint64_t INT4_IN_INT32_NUMS = 8UL;
static const std::initializer_list<ge::DataType> WEIGHT_DTYPE_LIST = {
    ge::DT_INT8, ge::DT_INT4, ge::DT_INT32, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8
};

void GetDtype(WeightQuantBatchMatmulInfo &matmulInfo, const gert::TilingContext *context)
{
    size_t idx = 0;
    matmulInfo.aDtype = context->GetInputDesc(idx++)->GetDataType();
    matmulInfo.bDtype = context->GetInputDesc(idx++)->GetDataType();
    matmulInfo.antiQuantScaleDtype = context->GetInputDesc(idx++)->GetDataType();
    matmulInfo.cDtype = context->GetOutputDesc(0)->GetDataType();
    auto biasDesc = context->GetOptionalInputDesc(BIAS_INDEX);
    if (biasDesc != nullptr) {
        matmulInfo.biasDtype = biasDesc->GetDataType();
    }
}

void GetAttrs(WeightQuantBatchMatmulInfo &matmulInfo, const gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    size_t idx = 0;
    auto transposeX = attrs->GetAttrPointer<bool>(idx++);
    auto transposeWeight = attrs->GetAttrPointer<bool>(idx++);
    const int64_t *groupSizePtr = nullptr;
    if (attrs->GetAttrNum() > idx) {
        groupSizePtr = attrs->GetAttrPointer<int64_t>(idx++);
    }
    if (groupSizePtr != nullptr) {
        matmulInfo.groupSize = static_cast<uint64_t>(*groupSizePtr);
    }

    idx++; // 跳过dtype属性
    const int64_t *innerPrecisePtr = nullptr;
    if (attrs->GetAttrNum() > idx) {
        innerPrecisePtr = attrs->GetAttrPointer<int64_t>(idx++);
    }
    if (innerPrecisePtr != nullptr) {
        // 参数校验阶段已校验innerPrecise只能为0或1， 可转为uint64_t类型
        matmulInfo.innerPrecise = static_cast<uint64_t>(*innerPrecisePtr);
    }

    matmulInfo.transA = transposeX != nullptr && *transposeX;
    matmulInfo.transB = transposeWeight != nullptr && *transposeWeight;
}

void GetInputs(WeightQuantBatchMatmulInfo &matmulInfo, const gert::TilingContext *context)
{
    size_t idx = 0;
    auto xShape = context->GetInputShape(idx++);
    auto weightShape = context->GetInputShape(idx++);
    auto antiQuantScaleShape = context->GetInputShape(idx++);
    auto antiQuantOffsetShape = context->GetOptionalInputShape(idx++);
    auto quantScaleShape = context->GetOptionalInputShape(idx++);
    auto biasShape = context->GetOptionalInputShape(BIAS_INDEX);
    matmulInfo.bFormat = GetInputStorageFormat(context, 1);
    uint64_t weightLastDim = weightShape->GetOriginShape().GetDim(1);
    if (matmulInfo.bDtype == ge::DT_INT32) {
        weightLastDim *= INT4_IN_INT32_NUMS;
    }
    matmulInfo.hasBias = biasShape != nullptr && biasShape->GetStorageShape().GetShapeSize() != 0;
    matmulInfo.mSize = static_cast<uint64_t>(matmulInfo.transA ? xShape->GetOriginShape().GetDim(1)
                                                               : xShape->GetOriginShape().GetDim(0));
    matmulInfo.kSize = static_cast<uint64_t>(matmulInfo.transA ? xShape->GetOriginShape().GetDim(0)
                                                               : xShape->GetOriginShape().GetDim(1));
    matmulInfo.nSize = static_cast<uint64_t>(matmulInfo.transB ? weightShape->GetOriginShape().GetDim(0)
                                                               : weightLastDim);

    if (CheckOptionalInputByShape(antiQuantOffsetShape)) {
        matmulInfo.hasAntiQuantOffset = true;
    }
    size_t antiQuantScaleShapeSize = static_cast<size_t>(antiQuantScaleShape->GetStorageShape().GetShapeSize());
    if (antiQuantScaleShapeSize == 1) {
        matmulInfo.antiQuantType = QuantType::PER_TENSOR;
    } else if (matmulInfo.groupSize > 0) {
        matmulInfo.antiQuantType = QuantType::PER_GROUP;
    } else {
        matmulInfo.antiQuantType = QuantType::PER_CHANNEL;
    }
    if (CheckOptionalInputByShape(quantScaleShape)) {
        size_t quantScaleShapeSize = static_cast<size_t>(quantScaleShape->GetStorageShape().GetShapeSize());
        if (quantScaleShapeSize == 0) {
            matmulInfo.quantType = QuantType::NONE;
        } else if (quantScaleShapeSize == 1) {
            matmulInfo.quantType = QuantType::PER_TENSOR;
        } else {
            matmulInfo.quantType = QuantType::PER_CHANNEL;
        }
    }
}

ge::graphStatus WeightQuantBatchMatmulV2Tiling::GetShapeAttrsInfo()
{
    try {
        matmulInfoPtr_ = std::make_unique<WeightQuantBatchMatmulInfo>();
    } catch (const std::bad_alloc& e) {
        OP_LOGE(context_->GetNodeName(), "failed to instantiate matmul info");
        return ge::GRAPH_FAILED;
    }

    GetDtype(*matmulInfoPtr_, context_);
    GetAttrs(*matmulInfoPtr_, context_);
    GetInputs(*matmulInfoPtr_, context_);
    opName_ = context_->GetNodeName();
    // int4pack输入场景修正dtype为int4
    if (matmulInfoPtr_->bDtype == ge::DT_INT32) {
        matmulInfoPtr_->bDtype = ge::DT_INT4;
        OP_LOGI(opName_, "The conversion of weight from int32 to int4 is completed.");
    }
    OP_LOGD(opName_,
            "input params: MKN[%lu, %lu, %lu], transA[%s], transB[%s], bias[%s], "
            "group size[%lu]",
            matmulInfoPtr_->mSize, matmulInfoPtr_->kSize, matmulInfoPtr_->nSize,
            matmulInfoPtr_->transA ? "true" : "false", matmulInfoPtr_->transB ? "true" : "false",
            matmulInfoPtr_->hasBias ? "true" : "false", matmulInfoPtr_->groupSize);
    return ge::GRAPH_SUCCESS;
}

void WeightQuantBatchMatmulV2Tiling::InitCompileInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null");
        return;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    try {
        compileInfoPtr_ = std::make_unique<WeightQuantBatchMatmulV2CompileInfo>();
    } catch (const std::bad_alloc& e) {
        OP_LOGE(context_->GetNodeName(), "failed to instantiate compile info");
        return;
    }

    compileInfoPtr_->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr_->aivNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr_->l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr_->l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr_->l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr_->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr_->ubSize);
    compileInfoPtr_->workspaceNum = ascendcPlatform.GetLibApiWorkSpaceSize();
    compileInfoPtr_->socVersion = ascendcPlatform.GetSocVersion();

    gert::GemmCompileInfo gmmcompileInfo;
    gmmcompileInfo.ParseRuntimePlatformInfo(context_->GetNodeName(), *platformInfoPtr);
    gmmcompileInfo.core_num = compileInfoPtr_->aicNum;
    optiling::PlatformInfo::GetInstance().SetInstance(gmmcompileInfo);
    OP_LOGD(context_->GetNodeName(), "MatmulAllReduce Init Quant Tiling Compile Info Success");
}

void WeightQuantBatchMatmulV2Tiling::SetCommonTilingKeyElement(TilingKeyConfigure &tilingKeyConfigure) const
{
    tilingKeyConfigure.socVersionType = static_cast<uint8_t>(SocVersionType::SUPPORT_L0C_TO_OUT) * 10; // 10:乘10第0位
    tilingKeyConfigure.quantizationScenario = static_cast<uint8_t>(QuantizationScenario::DEFAULT);
    tilingKeyConfigure.transposeSituation =
        (static_cast<uint16_t>(matmulInfoPtr_->transA) << 1) | static_cast<uint16_t>(matmulInfoPtr_->transB);
    tilingKeyConfigure.antiquantType = static_cast<uint8_t>(matmulInfoPtr_->antiQuantType);
    tilingKeyConfigure.quantType = static_cast<uint8_t>(QuantType::NONE);
    tilingKeyConfigure.optionInputSituation = (static_cast<uint16_t>(matmulInfoPtr_->hasAntiQuantOffset) << 1);
    tilingKeyConfigure.weightFormat = static_cast<uint8_t>(matmulInfoPtr_->bFormat == ge::FORMAT_ND ?
            WeightFormat::ND : WeightFormat::FRACTAL_NZ);
    tilingKeyConfigure.apiConstexpr = 0;
}

ge::graphStatus WeightQuantBatchMatmulV2Tiling::GetPlatformInfo()
{
    auto compileInfoPtr =
        compileInfoPtr_ ? compileInfoPtr_.get()
                        : reinterpret_cast<const WeightQuantBatchMatmulV2CompileInfo *>(context_->GetCompileInfo());
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context_->GetNodeName(), "compileInfoPtr is null");

    if (compileInfoPtr_ == nullptr) {
        compileInfoPtr_ = std::unique_ptr<WeightQuantBatchMatmulV2CompileInfo>(
            new (std::nothrow) WeightQuantBatchMatmulV2CompileInfo());
        OP_LOGE_IF(compileInfoPtr_ == nullptr, ge::GRAPH_FAILED, opName_, "compileInfoPtr_ is null");
        compileInfoPtr_->ubSize = compileInfoPtr->ubSize;
        compileInfoPtr_->l1Size = compileInfoPtr->l1Size;
        compileInfoPtr_->l0cSize = compileInfoPtr->l0cSize;
        compileInfoPtr_->l0aSize = compileInfoPtr->l0aSize;
        compileInfoPtr_->l0bSize = compileInfoPtr->l0bSize;
        compileInfoPtr_->workspaceNum = compileInfoPtr->workspaceNum;
        compileInfoPtr_->aivNum = compileInfoPtr->aivNum;
        compileInfoPtr_->aicNum = compileInfoPtr->aicNum;
        compileInfoPtr_->socVersion = compileInfoPtr->socVersion;
    }

    aicoreParams_.blockDim = 0;
    aicoreParams_.aicNum = compileInfoPtr->aicNum;
    aicoreParams_.ubSize = compileInfoPtr->ubSize;
    aicoreParams_.l1Size = compileInfoPtr->l1Size;
    aicoreParams_.l0cSize = compileInfoPtr->l0cSize;
    aicoreParams_.l0aSize = compileInfoPtr->l0aSize;
    aicoreParams_.l0bSize = compileInfoPtr->l0bSize;

    OP_TILING_CHECK(compileInfoPtr->aivNum == 0 || compileInfoPtr->aicNum == 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        opName_, "aivNum and aicNum >= 0 but they are %u and %u",
                        compileInfoPtr->aivNum, compileInfoPtr->aicNum),
                    return ge::GRAPH_FAILED);

    OP_LOGI(opName_,
            "get platform: aivNum(%u) aicNum(%u) ubSize(%lu) l1Size(%lu) "
            "l0cSize(%lu)  l0aSize(%lu) l0bSize(%lu)",
            compileInfoPtr->aivNum, compileInfoPtr->aicNum, aicoreParams_.ubSize, aicoreParams_.l1Size,
            aicoreParams_.l0cSize, aicoreParams_.l0aSize, aicoreParams_.l0bSize);

    return ge::GRAPH_SUCCESS;
}

bool CheckInputShape(WeightQuantBatchMatmulInfo *inputParams, const gert::StorageShape *xShape,
                     const gert::StorageShape *weightShape) {
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    size_t xOriDimNum = xShape->GetOriginShape().GetDimNum();
    size_t weigthDimNum = weightShape->GetStorageShape().GetDimNum();
    size_t weightOriDimNum = weightShape->GetOriginShape().GetDimNum();
    OP_TILING_CHECK(xOriDimNum != MM_SHAPE_LEN_ND,
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        inputParams->opName, "OriginalShape of x must be 2, but is [%zu]", xOriDimNum),
                    return false);
    OP_TILING_CHECK(xDimNum != MM_SHAPE_LEN_ND,
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        inputParams->opName, "StorageShape of x must be 2, but is [%zu]", xDimNum),
                    return false);
    OP_TILING_CHECK(weightOriDimNum != MM_SHAPE_LEN_ND,
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        inputParams->opName, "OriginalShape of weight must be 2, but is [%zu]", weightOriDimNum),
                    return false);
    if (inputParams->bFormat != ge::FORMAT_FRACTAL_NZ) {
        OP_TILING_CHECK(weigthDimNum != MM_SHAPE_LEN_ND,
                        VECTOR_INNER_ERR_REPORT_TILIING(
                            inputParams->opName, "StorageShape of weight must be 2 in ND, but is [%zu]", weigthDimNum),
                        return false);
    } else {
        OP_TILING_CHECK(weigthDimNum != MM_SHAPE_LEN_NZ,
                        VECTOR_INNER_ERR_REPORT_TILIING(
                            inputParams->opName, "StorageShape of weight must be 4 in NZ, but is [%zu]", weigthDimNum),
                        return false);
    }
    uint64_t weightLastDim = weightShape->GetOriginShape().GetDim(1);
    if (inputParams->bDtype == ge::DT_INT32) {
        weightLastDim *= INT4_IN_INT32_NUMS;
    }
    inputParams->mSize = static_cast<uint64_t>(inputParams->transA ? xShape->GetOriginShape().GetDim(1)
                                                                   : xShape->GetOriginShape().GetDim(0));
    inputParams->kSize = static_cast<uint64_t>(inputParams->transA ? xShape->GetOriginShape().GetDim(0)
                                                                   : xShape->GetOriginShape().GetDim(1));
    inputParams->nSize = static_cast<uint64_t>(inputParams->transB ? weightShape->GetOriginShape().GetDim(0)
                                                                   : weightLastDim);
    auto kBSize = static_cast<uint64_t>(inputParams->transB ? weightLastDim
                                                            : weightShape->GetOriginShape().GetDim(0));
    OP_TILING_CHECK(inputParams->kSize != kBSize,
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        inputParams->opName, "K dim of x and weight must equal, but they are [%lu] and [%ld]",
                        inputParams->kSize, kBSize),
                    return false);
    if (inputParams->bDtype == ge::DT_INT4 || inputParams->bDtype == ge::DT_INT32) {
        uint64_t innerAxisDim = static_cast<uint64_t>(weightLastDim);
        OP_TILING_CHECK((innerAxisDim & 1) != 0,
                        VECTOR_INNER_ERR_REPORT_TILIING(
                            inputParams->opName, "Inner dim of weight must be even when dtype is int4, but is [%lu]",
                            innerAxisDim),
                        return false);
    }
    return true;
}

bool CheckAntiQuantShape(WeightQuantBatchMatmulInfo *inputParams, const gert::StorageShape *antiQuantScaleShape,
                         const gert::StorageShape *antiQuantOffsetShape) {
    size_t antiQuantScaleDimNum = antiQuantScaleShape->GetStorageShape().GetDimNum();
    size_t antiQuantScaleShapeSize = static_cast<size_t>(antiQuantScaleShape->GetStorageShape().GetShapeSize());
    OP_TILING_CHECK(antiQuantScaleDimNum > MM_SHAPE_LEN_ND,
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        inputParams->opName, "antiquant scale shape size should not be more than 2, but is [%zu]",
                        antiQuantScaleDimNum),
                    return false);
    if (antiQuantScaleShapeSize != 1) {
        if (antiQuantScaleDimNum == MM_SHAPE_LEN_ND) {
            gert::Shape expectShape;
            uint64_t kNum = inputParams->groupSize > 0 ? ops::CeilDiv(inputParams->kSize, inputParams->groupSize) : 1;
            if (inputParams->transB) {
                expectShape.AppendDim(static_cast<int64_t>(inputParams->nSize));
                expectShape.AppendDim(static_cast<int64_t>(kNum));
            } else {
                expectShape.AppendDim(static_cast<int64_t>(kNum));
                expectShape.AppendDim(static_cast<int64_t>(inputParams->nSize));
            }
            OP_TILING_CHECK(
                expectShape != antiQuantScaleShape->GetStorageShape(),
                VECTOR_INNER_ERR_REPORT_TILIING(
                    inputParams->opName, "Antiquant shape expect %s, but is %s",
                    ge::Shape2String(expectShape).c_str(),
                    ge::Shape2String(antiQuantScaleShape->GetStorageShape()).c_str()),
                return false);
            inputParams->antiQuantType = inputParams->groupSize > 0 ? QuantType::PER_GROUP : QuantType::PER_CHANNEL;
        } else {
            OP_TILING_CHECK(antiQuantScaleShapeSize != inputParams->nSize,
                            VECTOR_INNER_ERR_REPORT_TILIING(
                                inputParams->opName, "Antiquant size should be n size when perchannel, but is %s",
                                ge::Shape2String(antiQuantScaleShape->GetStorageShape()).c_str()),
                            return false);
            inputParams->antiQuantType = QuantType::PER_CHANNEL;
        }
    } else {
        inputParams->antiQuantType = QuantType::PER_TENSOR;
    }
    if (CheckOptionalInputByShape(antiQuantOffsetShape)) {
        OP_TILING_CHECK(antiQuantScaleShape->GetStorageShape() != antiQuantOffsetShape->GetStorageShape(),
                        VECTOR_INNER_ERR_REPORT_TILIING(
                            inputParams->opName, "Antiquant scale and offset should be same, but are %s and %s",
                            ge::Shape2String(antiQuantScaleShape->GetStorageShape()).c_str(),
                            ge::Shape2String(antiQuantOffsetShape->GetStorageShape()).c_str()),
                        return false);
    }
    return true;
}

bool CheckQuantShape(WeightQuantBatchMatmulInfo *inputParams, const gert::StorageShape *quantScaleShape,
                     const gert::StorageShape *quantOffsetShape) {
    if (!CheckOptionalInputByShape(quantScaleShape)) {
        if (CheckOptionalInputByShape(quantOffsetShape)) {
            VECTOR_INNER_ERR_REPORT_TILIING(inputParams->opName, "Quant offset should exist with scale");
            return false;
        }
        if (inputParams->cDtype != inputParams->aDtype) {
            VECTOR_INNER_ERR_REPORT_TILIING(
                inputParams->opName, "Output dtype should be x dtype without quant scale, but is [%s] and x is [%s]",
                ge::TypeUtils::DataTypeToAscendString(inputParams->cDtype).GetString(),
                ge::TypeUtils::DataTypeToAscendString(inputParams->aDtype).GetString());
            return false;
        }
        return true;
    } else {
        if (inputParams->cDtype != ge::DT_INT8) {
            VECTOR_INNER_ERR_REPORT_TILIING(inputParams->opName,
                                            "Output dtype should be int8 with quant scale, but is [%s]",
                                            ge::TypeUtils::DataTypeToAscendString(inputParams->cDtype).GetString());
            return false;
        }
    }
    size_t quantScaleDimNum = quantScaleShape->GetStorageShape().GetDimNum();
    size_t quantScaleShapeSize = static_cast<size_t>(quantScaleShape->GetStorageShape().GetShapeSize());
    OP_TILING_CHECK(
        quantScaleDimNum > MM_SHAPE_LEN_ND,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams->opName, "quant scale shape size should not be more than 2, but is [%zu]", quantScaleDimNum),
        return false);
    if (quantScaleDimNum == MM_SHAPE_LEN_ND) {
        OP_TILING_CHECK(
            quantScaleShape->GetStorageShape().GetDim(0) != 1 ||
                static_cast<uint64_t>(quantScaleShape->GetStorageShape().GetDim(1)) != inputParams->nSize,
            VECTOR_INNER_ERR_REPORT_TILIING(inputParams->opName, "quant shape should be (1, n), but is %s",
                                            ge::Shape2String(quantScaleShape->GetStorageShape()).c_str()),
            return false);
    } else {
        OP_TILING_CHECK(quantScaleShapeSize > 1 && quantScaleShapeSize != inputParams->nSize,
                        VECTOR_INNER_ERR_REPORT_TILIING(
                            inputParams->opName, "Quant size should be n size when perchannel, but is %s",
                            ge::Shape2String(quantScaleShape->GetStorageShape()).c_str()),
                        return false);
    }
    return true;
}

bool CheckBiasShape(WeightQuantBatchMatmulInfo *inputParams, const gert::StorageShape *biasShape) {
    if (biasShape != nullptr) {
        auto biasShapeDimNum = static_cast<uint64_t>(biasShape->GetStorageShape().GetDimNum());
        if (biasShapeDimNum == 1) {
            OP_TILING_CHECK(biasShape->GetStorageShape().GetDim(0) != 0 &&
                                static_cast<uint64_t>(biasShape->GetStorageShape().GetDim(0)) != inputParams->nSize,
                            VECTOR_INNER_ERR_REPORT_TILIING(inputParams->opName, "Bias should be 0 or n, but is %s",
                                                            ge::Shape2String(biasShape->GetStorageShape()).c_str()),
                            return false);
        } else if (biasShapeDimNum == MM_SHAPE_LEN_ND) {
            OP_TILING_CHECK(biasShape->GetStorageShape().GetDim(0) != 1 ||
                                static_cast<uint64_t>(biasShape->GetStorageShape().GetDim(1)) != inputParams->nSize,
                            VECTOR_INNER_ERR_REPORT_TILIING(inputParams->opName, "Bias should be [1, n], but is %s",
                                                            ge::Shape2String(biasShape->GetStorageShape()).c_str()),
                            return false);
        } else {
            VECTOR_INNER_ERR_REPORT_TILIING(inputParams->opName, "Input shape of bias should be 1 or 2, but is [%zu]",
                                            biasShapeDimNum);
            return false;
        }
        return true;
    }
    return true;
}

bool CheckShapeDims(WeightQuantBatchMatmulInfo *inputParams) {
    OP_TILING_CHECK(inputParams->kSize > MAX_SHAPE_DIM || inputParams->nSize > MAX_SHAPE_DIM,
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        inputParams->opName, "Dim of k or n should not more than 65535, but they are [%lu] and [%lu]",
                        inputParams->kSize, inputParams->nSize),
                    return false);
    uint64_t batchMax = inputParams->transA ? MAX_SHAPE_DIM : MAX_INT32;
    OP_TILING_CHECK(
        inputParams->mSize > batchMax,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams->opName, "Dim of m should not more than [%lu], but is [%lu]",
                                        batchMax, inputParams->mSize),
        return false);
    OP_TILING_CHECK(inputParams->groupSize >= inputParams->kSize || inputParams->groupSize % MIN_GROUP_SIZE != 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        inputParams->opName, "Group sizes should not more than [%lu] and align to 32, but is [%lu]",
                        inputParams->kSize, inputParams->groupSize),
                    return false);

    return true;
}

/*
The function is check the shape limit:
    1. the input x shape dims should be 2, weight shape dims should be 2(ND) or 4(NZ)
    2. the k of x and weight must be same; the inner axis must be even when weight is int4
    3. antiquant shape must be:
        per_group:  not trans_b: (ceil(k, group_size), n); trans_b: (n, ceil(k, group_size))
        per_channel: not trans_n: (1, n) or (n); trans_b: (n, 1) or (n)
        per_tensor: (1,)
    4. quant shape must be (1,), (1, n), (n) or empty tensor
    5. bias shape must be (n,), (1, n) or empty tensor
    6. nk must <= 65535, m <= 65535(trans_a) or int32_max(not trans_a);
    7. group_size < k, align to 32
*/
bool CheckShape(gert::TilingContext *context, WeightQuantBatchMatmulInfo *inputParams) {
    size_t idx = 0;
    auto xShape = context->GetInputShape(idx++);
    auto weightShape = context->GetInputShape(idx++);
    auto antiQuantScaleShape = context->GetInputShape(idx++);
    auto antiQuantOffsetShape = context->GetOptionalInputShape(idx++);
    auto quantScaleShape = context->GetOptionalInputShape(idx++);
    auto quantOffsetShape = context->GetOptionalInputShape(idx++);
    auto biasShape = context->GetOptionalInputShape(idx++);
    auto outputShape = context->GetOutputShape(0);

    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, weightShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, antiQuantScaleShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    // not yet support empty tensor for input
    OP_TILING_CHECK(xShape->GetStorageShape().GetShapeSize() == 0 ||
                    weightShape->GetStorageShape().GetShapeSize() == 0 ||
                    antiQuantScaleShape->GetStorageShape().GetShapeSize() == 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams->opName, "Not yet support empty tensor"),
                    return false);
    inputParams->bFormat = GetInputStorageFormat(context, 1);
    OP_TILING_CHECK(inputParams->bFormat == ge::FORMAT_NULL,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams->opName, "Input weight format is null"),
                    return false);
    OP_TILING_CHECK(!CheckInputShape(inputParams, xShape, weightShape),
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams->opName, "Check input x and weight shape failed"),
                    return false);
    OP_TILING_CHECK(!CheckAntiQuantShape(inputParams, antiQuantScaleShape, antiQuantOffsetShape),
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams->opName, "Check antiquant shape failed"),
                    return false);
    OP_TILING_CHECK(!CheckQuantShape(inputParams, quantScaleShape, quantOffsetShape),
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams->opName, "Check quant shape failed"),
                    return false);
    OP_TILING_CHECK(!CheckBiasShape(inputParams, biasShape),
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams->opName, "Check bias shape failed"),
                    return false);
    OP_TILING_CHECK(!CheckShapeDims(inputParams),
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams->opName, "Check shape dims failed"),
                    return false);
    return true;
}

bool CheckInputDtype(gert::TilingContext *context, WeightQuantBatchMatmulInfo *inputParams,
                     platform_ascendc::SocVersion socVersion) {
    OP_TILING_CHECK(
        inputParams->aDtype != ge::DT_FLOAT16 && inputParams->aDtype != ge::DT_BF16,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams->opName, "Input x dtype must be fp16 or bf16, but is [%s]",
            ge::TypeUtils::DataTypeToAscendString(inputParams->aDtype).GetString()),
        return false);
    OP_TILING_CHECK(
        std::find(WEIGHT_DTYPE_LIST.begin(), WEIGHT_DTYPE_LIST.end(), inputParams->bDtype) == WEIGHT_DTYPE_LIST.end(),
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams->opName,
            "Input weight dtype must be int8, int4, float8_e5m2, float8_e4m3fn, hifloat8, but is [%s]",
            ge::TypeUtils::DataTypeToAscendString(inputParams->bDtype).GetString()),
        return false);
    // the bias is the 6th input
    auto biasDesc = context->GetOptionalInputDesc(6);
    if (biasDesc != nullptr) {
        auto biasDtype = biasDesc->GetDataType();
        if (inputParams->aDtype == ge::DT_BF16) {
            if (socVersion == platform_ascendc::SocVersion::ASCEND910B) {
                OP_TILING_CHECK(biasDtype != ge::DT_FLOAT,
                                VECTOR_INNER_ERR_REPORT_TILIING(
                                    inputParams->opName, "Bias dtype must be fp32 when x is bf16, but is [%s]",
                                    ge::TypeUtils::DataTypeToAscendString(biasDtype).GetString()),
                                return false);
            } else {
                OP_TILING_CHECK(biasDtype != ge::DT_FLOAT && biasDtype != ge::DT_BF16,
                                VECTOR_INNER_ERR_REPORT_TILIING(
                                    inputParams->opName, "Bias dtype must be fp32 or bf16 when x is bf16, but is [%s]",
                                    ge::TypeUtils::DataTypeToAscendString(biasDtype).GetString()),
                                return false);
            }
        } else {
            OP_TILING_CHECK(biasDtype != inputParams->aDtype,
                            VECTOR_INNER_ERR_REPORT_TILIING(
                                inputParams->opName, "Bias dtype must be fp16 when x dtype is fp16, but is [%s]",
                                ge::TypeUtils::DataTypeToAscendString(biasDtype).GetString()),
                            return false);
        }
    }
    return true;
}

bool CheckAntiQuantDtype(gert::TilingContext *context, WeightQuantBatchMatmulInfo *inputParams,
                         platform_ascendc::SocVersion socVersion) {
    // the antiquant scale is the 3nd input
    auto antiQuantOffsetDesc = context->GetOptionalInputDesc(3);
    if (inputParams->antiQuantScaleDtype == ge::DT_UINT64 || inputParams->antiQuantScaleDtype == ge::DT_INT64) {
        OP_TILING_CHECK(socVersion != platform_ascendc::SocVersion::ASCEND910B,
                        VECTOR_INNER_ERR_REPORT_TILIING(
                            inputParams->opName, "This socversion not support antiquant scale dtype int64/uint64"),
                        return false);
        OP_TILING_CHECK(inputParams->aDtype != ge::DT_FLOAT16,
                        VECTOR_INNER_ERR_REPORT_TILIING(
                            inputParams->opName,
                            "X dtype must be fp16 when antiquant scale dtype is int64/uint64, but is [%s]",
                            ge::TypeUtils::DataTypeToAscendString(inputParams->aDtype).GetString()),
                        return false);
        if (antiQuantOffsetDesc != nullptr) {
            auto antiQuantOffsetDtype = antiQuantOffsetDesc->GetDataType();
            OP_TILING_CHECK(antiQuantOffsetDtype != ge::DT_INT32,
                            VECTOR_INNER_ERR_REPORT_TILIING(
                                inputParams->opName,
                                "Antiquant offset dtype must be int32 when scale dtype is int64/uint64, but is [%s]",
                                ge::TypeUtils::DataTypeToAscendString(antiQuantOffsetDtype).GetString()),
                            return false);
        }
    } else if (inputParams->antiQuantScaleDtype == inputParams->aDtype) {
        if (antiQuantOffsetDesc != nullptr) {
            auto antiQuantOffsetDtype = antiQuantOffsetDesc->GetDataType();
            OP_TILING_CHECK(antiQuantOffsetDtype != inputParams->aDtype,
                            VECTOR_INNER_ERR_REPORT_TILIING(
                                inputParams->opName,
                                "Antiquant offset dtype must be x dtype when scale dtype is x dtype, but is [%s]",
                                ge::TypeUtils::DataTypeToAscendString(antiQuantOffsetDtype).GetString()),
                            return false);
        }
    } else {
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams->opName,
            "Antiquant scale dtype must be int64/uint64 or x dtype, but is [%s] and xType is [%s]",
            ge::TypeUtils::DataTypeToAscendString(inputParams->antiQuantScaleDtype).GetString(),
            ge::TypeUtils::DataTypeToAscendString(inputParams->aDtype).GetString());
        return false;
    }
    return true;
}

bool CheckQuantDtype(gert::TilingContext *context, WeightQuantBatchMatmulInfo *inputParams) {
    // the quant_scale is the 4th input
    auto quantScaleDesc = context->GetOptionalInputDesc(4);
    if (quantScaleDesc != nullptr) {
        auto quantScaleDtype = quantScaleDesc->GetDataType();
        OP_TILING_CHECK(quantScaleDtype != ge::DT_UINT64,
                        VECTOR_INNER_ERR_REPORT_TILIING(
                            inputParams->opName, "Quant scale dtype should be uint64, but is [%s]",
                            ge::TypeUtils::DataTypeToAscendString(quantScaleDtype).GetString()),
                        return false);
    } else {
        OP_TILING_CHECK(inputParams->cDtype != inputParams->aDtype,
                        VECTOR_INNER_ERR_REPORT_TILIING(
                            inputParams->opName,
                            "Output dtype should be x dtype without quant, but is [%s] and x is [%s]",
                            ge::TypeUtils::DataTypeToAscendString(inputParams->cDtype).GetString(),
                            ge::TypeUtils::DataTypeToAscendString(inputParams->aDtype).GetString()),
                        return false);
    }
    return true;
}

/*
The function is check the dtype limit:
    1. Input x dtype should be fp16 or bf16, weight should be int8 or int4
    2. Output dtype should be same with x dtype without quant
    3. bias dtype should be fp16 when x dtype is fp16, fp32 when x dype is bf16
    4. antiquant scale dtype should be x dtype or int64/uint64, antiquant offset dtype should be x dtype or int32;
    5. quant scale dtype should be uint64
*/
bool CheckDtype(gert::TilingContext *context, WeightQuantBatchMatmulInfo *inputParams,
                platform_ascendc::SocVersion socVersion) {
    size_t idx = 0;
    inputParams->aDtype = context->GetInputDesc(idx++)->GetDataType();
    inputParams->bDtype = context->GetInputDesc(idx++)->GetDataType();
    inputParams->antiQuantScaleDtype = context->GetInputDesc(idx++)->GetDataType();
    inputParams->cDtype = context->GetOutputDesc(0)->GetDataType();
    OP_TILING_CHECK(!CheckInputDtype(context, inputParams, socVersion),
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams->opName, "Check a, weight and bias dtype failed"),
                    return false);
    OP_TILING_CHECK(!CheckAntiQuantDtype(context, inputParams, socVersion),
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams->opName, "Check antiquant dtype failed"),
                    return false);
    OP_TILING_CHECK(!CheckQuantDtype(context, inputParams),
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams->opName, "Check quant dtype failed"),
                    return false);
    return true;
}

bool CheckAttr(gert::TilingContext *context, WeightQuantBatchMatmulInfo *inputParams) {
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    size_t idx = 0;
    auto transposeX = attrs->GetAttrPointer<bool>(idx++);
    auto transposeWeight = attrs->GetAttrPointer<bool>(idx++);
    inputParams->transA = transposeX != nullptr && *transposeX;
    inputParams->transB = transposeWeight != nullptr && *transposeWeight;
    const int64_t *groupSizePtr = nullptr;
    if (attrs->GetAttrNum() > idx) {
        groupSizePtr = attrs->GetAttrPointer<int64_t>(idx++);
    }
    // the group size not less than 0
    if (groupSizePtr != nullptr) {
        OP_TILING_CHECK(*groupSizePtr < 0,
                        VECTOR_INNER_ERR_REPORT_TILIING(
                            inputParams->opName, "Group size should not less than 0, but is [%ld]", *groupSizePtr),
                        return false);
        inputParams->groupSize = static_cast<uint64_t>(*groupSizePtr);
    }
    idx++; // 跳过dtype属性
    const int64_t *innerPrecisePtr = nullptr;
    if (attrs->GetAttrNum() > idx) {
        innerPrecisePtr = attrs->GetAttrPointer<int64_t>(idx++);
    }
    if (innerPrecisePtr != nullptr) {
        OP_TILING_CHECK((*innerPrecisePtr != 0) && (*innerPrecisePtr != 1) ,
                        VECTOR_INNER_ERR_REPORT_TILIING(
                            inputParams->opName, "innerPrecise only support 0 or 1, but is [%ld]", *innerPrecisePtr),
                        return false);
        inputParams->innerPrecise = static_cast<uint64_t>(*innerPrecisePtr);
    }
    return true;
}

ge::graphStatus CheckPara(gert::TilingContext *context, platform_ascendc::SocVersion socVersion) {
    // check Raw TilingData
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetRawTilingData()->GetData());
    // check the Required input and output desc
    size_t idx = 0;
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(idx++));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(idx++));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(idx++));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetOutputDesc(0));
    WeightQuantBatchMatmulInfo inputParams;
    inputParams.opName = context->GetNodeName();
    // check the input and output dtype
    OP_TILING_CHECK(!CheckDtype(context, &inputParams, socVersion),
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams.opName, "Check input dtype failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckAttr(context, &inputParams),
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams.opName, "Check attr failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckShape(context, &inputParams),
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams.opName, "Check shape failed"),
                    return ge::GRAPH_FAILED);
    // weightNZ support A16W8/W4C16, perchannel, pergroup without transA
    if (inputParams.bFormat == ge::FORMAT_FRACTAL_NZ) {
        OP_TILING_CHECK(
            (inputParams.antiQuantType == QuantType::PER_GROUP && inputParams.transA &&
             inputParams.bDtype != ge::DT_INT4) ||
             inputParams.cDtype == ge::DT_INT8 || inputParams.antiQuantType == QuantType::PER_TENSOR,
            VECTOR_INNER_ERR_REPORT_TILIING(inputParams.opName,
            "WeightNZ cannot support per-group with transA int8, cannot support int8 output or per-tensor"),
            return ge::GRAPH_FAILED);
    }
    if (inputParams.bDtype == ge::DT_FLOAT8_E5M2 || inputParams.bDtype == ge::DT_FLOAT8_E4M3FN ||
        inputParams.bDtype == ge::DT_HIFLOAT8) {
        OP_TILING_CHECK(
            inputParams.transA || inputParams.cDtype == ge::DT_INT8 || inputParams.bFormat == ge::FORMAT_FRACTAL_NZ,
            VECTOR_INNER_ERR_REPORT_TILIING(inputParams.opName,
            "Weight F8 input cannot support transA, int8 output and weightNz"),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling
