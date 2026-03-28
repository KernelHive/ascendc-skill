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
 * \file weight_quant_batch_matmul_v2_weight_nz_tiling.cpp
 * \brief
 */
#include "weight_quant_batch_matmul_v2_weight_nz_tiling.h"

namespace {
constexpr uint64_t MAX_SHAPE_DIM = 65535UL;
constexpr size_t VALID_INPUT_DIM_NUM = 2UL;
constexpr uint64_t BLOCK_REDUCE_INT8 = 32UL;
constexpr size_t MM_SHAPE_LEN_ND = 2UL;
constexpr size_t MM_SHAPE_LEN_NZ = 4UL;
static const int32_t ANTI_QUANT_TENSOR = 2;
static const int32_t DB_BUFFER = 2;
constexpr size_t QUANT_SCALE_INDEX = 4UL;
constexpr size_t QUANT_OFFSET_INDEX = 5UL;
constexpr size_t BIAS_INDEX = 6UL;
static const uint64_t MAX_BLOCK_STRIDE = 256;
static const uint64_t MAX_NBUB_SIZE = 128;
static const uint64_t ANTIQUANT_BUF_NUM = 74;
static const uint64_t BUB_SIZE_ALIGN = 32;
static const uint64_t BUB_BUF_NUM = 6;
static const uint64_t AUB_SIZE_ALIGN = 128 * 128;
static const uint64_t AUB_BUF_NUM = 8;

class WhiteListShape {
public:
    bool operator<(const WhiteListShape &right) const { return memcmp(this, &right, sizeof(WhiteListShape)) < 0; }

    uint64_t mSize_;
    uint64_t nSize_;
    uint64_t kSize_;
};

class WeightNzTilingCache {
public:
    uint8_t cubeBlockDimN_;
    uint8_t cubeBlockDimM_;
    uint16_t aL1Pingpong_;
    uint16_t bL1Pingpong_;
    uint64_t kAlign_;
    uint64_t mSize_;
    uint64_t kSize_;
    uint64_t nSize_;
    uint64_t mAubSize_;
    uint64_t kAubSize_;
    uint64_t nBubSize_;
    uint64_t kBubSize_;
    uint64_t mCubSize_;
    uint64_t nCubSize_;
    uint64_t mAL1Size_;
    uint64_t kAL1Size_;
    uint64_t nBL1Size_;
    uint64_t kBL1Size_;
    uint64_t groupSize_;

    // matmul字段
    int32_t m_;
    int32_t n_;
    int32_t ka_;
    int32_t kb_;
    int32_t singleCoreM_;
    int32_t singleCoreN_;
    int32_t singleCoreK_;
    int32_t baseM_;
    int32_t baseN_;
    int32_t baseK_;
    int32_t depthA1_;
    int32_t depthB1_;
    int32_t stepM_;
    int32_t stepN_;
    int32_t transLength_;
    int32_t iterateOrder_;
    int32_t shareMode_;
    int32_t shareL1Size_;
    int32_t shareL0CSize_;
    int32_t stepKa_;
    int32_t stepKb_;
    int32_t dbL0A_;
    int32_t dbL0B_;
    int32_t dbL0C_;
};

const std::map<WhiteListShape, WeightNzTilingCache> WEIGHT_NZ_TILING_CACHE = {
    // 125.6
    {{4, 2048, 4096},
     {8,   1,  2,   1,   4096, 4,   4096, 2048, 64, 256, 128, 128,   64, 256, 64, 256, 256, 256, 0, 64, 256, 256,
      256, 64, 256, 256, 64,   256, 64,   64,   64, 1,   1,   32768, 0,  0,   0,  0,   64,  64,  2, 2,  2}},

    // 337.9
    {{4, 5504, 4096},
     {8,   1,  2,   1,   4096, 4,   4096, 5504, 16, 256, 128, 128,   16, 176, 16, 256, 704, 256, 0, 16, 704, 256,
      256, 16, 704, 256, 16,   704, 16,   32,   32, 1,   1,   45056, 0,  0,   0,  0,   32,  32,  2, 2,  2}},

    // 123.3
    {{4, 4096, 2048},
     {8,   1,  2,   1,   2048, 4,   2048, 4096, 16, 256, 128, 128,   16, 256, 16, 256, 512, 256, 0, 16, 512, 256,
      256, 16, 512, 256, 16,   512, 32,   16,   16, 1,   1,   32768, 0,  0,   0,  0,   16,  16,  2, 2,  2}},

    // 3530
    {{4, 65000, 4096},
     {8,   1,  2,   1,   4096, 4,   4096, 65000, 16, 128, 128, 128,   16, 256, 16, 256, 512, 256, 0, 16, 512, 256,
      256, 16, 512, 256, 16,   512, 32,   16,    16, 1,   1,   32768, 0,  0,   0,  0,   16,  16,  2, 2,  2}},

    // 298.3
    {{4, 4096, 5504},
     {8,   1,  2,   2,   5504, 4,   5504, 4096, 16, 256, 128, 128,   16, 256, 16, 256, 512, 256, 0, 16, 512, 256,
      256, 16, 512, 256, 16,   512, 32,   8,    8,  1,   1,   32768, 0,  0,   0,  0,   8,   8,   2, 2,  2}},

    // 2590.5
    {{4096, 2048, 4096},
     {4,   2,   2,   2,   4096, 4096, 4096, 2048, 128, 128, 128, 128,    512, 64, 512, 128, 128, 128, 0, 512, 128, 128,
      128, 512, 128, 128, 512,  128,  32,   4,    4,   1,   1,   262144, 1,   0,  0,   0,   4,   4,   2, 2,   1}},

    // 7930
    {{4096, 5504, 4096},
     {4,   2,   2,   2,   4096, 4096, 4096, 5504, 128, 128, 128, 128,    512, 64, 512, 128, 128, 128, 0, 512, 128, 128,
      128, 512, 128, 128, 512,  128,  32,   4,    4,   1,   1,   262144, 1,   0,  0,   0,   4,   4,   2, 2,   1}},

    // 81465
    {{4096, 65000, 4096},
     {8,   1,   2,   2,   4096, 4096, 4096, 65000, 128, 128, 128, 128,    512, 64, 512, 128, 128, 128, 0, 512, 128, 128,
      128, 512, 128, 128, 512,  128,  32,   4,     4,   1,   1,   262144, 1,   0,  0,   0,   4,   4,   2, 2,   1}},

    // 425
    {{512, 4096, 2048},
     {8,   1,   2,   2,   2048, 512, 2048, 4096, 128, 128, 128, 128,    256, 128, 256, 256, 256, 256, 0, 256, 256, 256,
      256, 256, 256, 256, 256,  256, 64,   4,    4,   1,   1,   262144, 0,   0,   0,   0,   4,   4,   2, 2,   1}},

    // 791
    {{896, 4096, 2048},
     {8,   1,   2,   2,   2048, 896, 2048, 4096, 112, 128, 128, 128,    224, 256, 224, 256, 256, 256, 0, 224, 256, 256,
      256, 224, 256, 256, 224,  256, 64,   4,    4,   1,   1,   229376, 0,   0,   0,   0,   4,   4,   2, 2,   1}},

    // 1021
    {{512, 4096, 5504},
     {8,   1,   2,   2,   5504, 512, 5504, 4096, 128, 128, 128, 128,    256, 128, 256, 256, 256, 256, 0, 256, 256, 256,
      256, 256, 256, 256, 256,  256, 64,   4,    4,   1,   1,   262144, 0,   0,   0,   0,   4,   4,   2, 2,   1}},

    // 1931
    {{896, 4096, 5504},
     {8,   1,   2,   2,   5504, 896, 5504, 4096, 112, 128, 128, 128,    224, 256, 224, 256, 256, 256, 0, 224, 256, 256,
      256, 224, 256, 256, 224,  256, 64,   4,    4,   1,   1,   229376, 0,   0,   0,   0,   4,   4,   2, 2,   1}},

    {{4, 11008, 4096},
     {8,   1,  2,   1,   4096, 4,   4096, 11008, 16, 256, 128, 128,   16, 16, 16, 512, 688, 512, 0, 16, 688, 512,
      512, 16, 688, 512, 16,   688, 16,   32,    32, 1,   1,   44032, 0,  0,  0,  0,   32,  32,  2, 2,  1}},

    {{4812, 9612, 2462},
     {1,   1,   2,   2,   2464, 4812, 2462, 9612, 128, 128, 128, 32,     128, 256, 128, 128, 256, 128, 0, 128, 256, 128,
      128, 128, 256, 128, 128,  256,  64,   2,    2,   1,   1,   131072, 0,   0,   0,   0,   2,   2,   2, 2,   1}},

    {{1811, 7925, 29},
     {1,  1,   2,   2,  32,  1811, 29, 7925, 128, 64, 128, 32, 128, 256, 128, 64, 256, 64, 0, 128, 256, 64,
      64, 128, 256, 64, 128, 256,  32, 2,    2,   1,  1,   0,  1,   0,   0,   0,  2,   2,  2, 2,   1}},

    {{14512, 3376, 48},
     {1,   1,   2,   2,   64,  14512, 48, 3376, 128, 128, 128, 32, 128, 256, 128, 576, 256, 576, 0, 128, 256, 576,
      576, 128, 256, 576, 128, 256,   48, 12,   12,  1,   1,   0,  0,   0,   0,   0,   12,  12,  2, 2,   1}},

    {{5232, 3872, 6736},
     {1,   1,   2,   2,   6752, 5232, 6736, 3872, 128, 128, 128, 32, 128, 256, 128, 256, 256, 256, 0, 128, 256, 256,
      256, 128, 256, 256, 128,  256,  64,   4,    4,   1,   1,   0,  0,   0,   0,   0,   4,   4,   2, 2,   1}},

    {{5236, 13521, 7310},
     {1,   1,   2,   2,   7328, 5236, 7310, 13521, 128, 128, 128, 32, 128, 256, 128, 128, 256, 128, 0, 128, 256, 128,
      128, 128, 256, 128, 128,  256,  64,   2,     2,   1,   1,   0,  0,   0,   0,   0,   2,   2,   2, 2,   1}}};

}  // namespace
namespace optiling {

bool WeightQuantBatchMatmulV2WeightNz::IsCapable()
{
    // 判断weight矩阵是否是4维
    auto weightShape = context_->GetInputShape(1);
    auto weightDimNum = weightShape->GetStorageShape().GetDimNum();
    OP_LOGI_IF_RETURN(weightDimNum != MM_SHAPE_LEN_NZ, false, inputParams_.opName,
                      "weightNz template only support right matrix weight 4 dims Tensor,now is %ld", weightDimNum);

    // weightNz模板只支持weight矩阵是转置的场景
    OP_LOGI_IF_RETURN(!inputParams_.transB, false, inputParams_.opName,
                      "weightNz template only support right matrix is transposed.");

    // weightNz模板只支持weight矩阵的format是FRACTAL_NZ的
    auto weightFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(1)->GetStorageFormat()));
    OP_LOGI_IF_RETURN(weightFormat != ge::FORMAT_FRACTAL_NZ, false, inputParams_.opName,
                      "weightNz template only support right matrix format is NZ.");

    return true;
}

void WeightQuantBatchMatmulV2WeightNz::SetMatmulTiling()
{
    auto &mmtiling = tilingData_->matmulTiling;
    uint32_t minKL1 = std::min(tilingData_->get_kBL1Size(), tilingData_->get_kAL1Size());
    uint32_t depthK = ops::CeilDiv(minKL1, static_cast<uint32_t>(mmtiling.get_baseK()));
    mmtiling.set_M(mmtiling.get_baseM());
    mmtiling.set_Ka(minKL1);
    mmtiling.set_N(mmtiling.get_baseN());
    mmtiling.set_Kb(minKL1);
    mmtiling.set_singleCoreM(mmtiling.get_baseM());
    mmtiling.set_singleCoreK(minKL1);
    mmtiling.set_singleCoreN(mmtiling.get_baseN());
    mmtiling.set_depthA1(depthK);
    mmtiling.set_depthB1(depthK);
    mmtiling.set_stepKa(depthK);
    mmtiling.set_stepKb(depthK);
    mmtiling.set_stepM(1);
    mmtiling.set_stepN(1);
    mmtiling.set_shareL0CSize(0);
    mmtiling.set_shareL1Size(0);
    uint64_t singleMLoop = ops::CeilDiv(ops::CeilDiv(tilingData_->get_mSize(), tilingData_->get_mAL1Size()),
                                        static_cast<uint64_t>(tilingData_->get_cubeBlockDimM()));
    auto mDim = ops::CeilDiv(tilingData_->get_mSize(), singleMLoop * tilingData_->get_mAL1Size());
    uint64_t singleNLoop = ops::CeilDiv(ops::CeilDiv(tilingData_->get_nSize(), tilingData_->get_nBL1Size()),
                                        static_cast<uint64_t>(tilingData_->get_cubeBlockDimN()));
    auto nDim = ops::CeilDiv(tilingData_->get_nSize(), singleNLoop * tilingData_->get_nBL1Size());
    tilingData_->set_cubeBlockDimN(nDim);
    tilingData_->set_cubeBlockDimM(mDim);
}

void WeightQuantBatchMatmulV2WeightNz::GetubFactorTiling()
{
    auto &mmtiling = tilingData_->matmulTiling;
    uint64_t mmUsedUbSize = mmtiling.get_baseM() * mmtiling.get_baseN() * GetSizeByDataType(ge::DT_FLOAT16);
    uint64_t ubLength =
        inputParams_.hasBias ? aicoreParams_.ubSize - mmtiling.get_baseN() * sizeof(float) : aicoreParams_.ubSize;
    ubLength -= ANTIQUANT_BUF_NUM * tilingData_->matmulTiling.get_baseN();
    uint64_t bubUsedSize = CalBubFactorTiling(ubLength);
    uint64_t aubUsedSize = CalAubFactorTiling(ubLength - bubUsedSize);
    uint64_t cubNz2NdCanUseSize = CalCubFactorTiling(ubLength - mmUsedUbSize);
    OP_LOGD("WeightQuantBatchmatmul", "cubUseSize %lu, bubusedsize %lu, aubusedsize %lu, cubnd2nzusedsize %lu",
            mmUsedUbSize, bubUsedSize, aubUsedSize, cubNz2NdCanUseSize);
}

ge::graphStatus WeightQuantBatchMatmulV2WeightNz::GetPlatformInfo()
{
    auto compileInfoPtr =
        compileInfoPtr_ ? compileInfoPtr_.get()
                        : reinterpret_cast<const WeightQuantBatchMatmulV2CompileInfo *>(context_->GetCompileInfo());
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context_->GetNodeName(), "compileInfoPtr is null");
    if (compileInfoPtr_ == nullptr) {
        compileInfoPtr_ = std::unique_ptr<WeightQuantBatchMatmulV2CompileInfo>(
            new (std::nothrow) WeightQuantBatchMatmulV2CompileInfo());
        OP_LOGE_IF(compileInfoPtr_ == nullptr, ge::GRAPH_FAILED, context_->GetNodeName(), "compileInfoPtr_ is null");
        compileInfoPtr_->workspaceNum = compileInfoPtr->workspaceNum;
    }

    aicoreParams_.blockDim = 0;
    aicoreParams_.ubSize = compileInfoPtr->ubSize;
    aicoreParams_.l1Size = compileInfoPtr->l1Size;
    aicoreParams_.l0cSize = compileInfoPtr->l0cSize;

    OP_LOGI(inputParams_.opName, "get platform: aivNum(%u) aicNum(%u) ubSize(%lu) l1Size(%lu) l0cSize(%lu)",
            compileInfoPtr->aivNum, compileInfoPtr->aicNum, aicoreParams_.ubSize, aicoreParams_.l1Size,
            aicoreParams_.l0cSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantBatchMatmulV2WeightNz::CheckContext() const
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    size_t idx = 0;
    auto xShape = context_->GetInputShape(idx);
    auto xDesc = context_->GetInputDesc(idx++);
    auto weightShape = context_->GetInputShape(idx);
    auto weightDesc = context_->GetInputDesc(idx++);
    auto antiQuantScaleShape = context_->GetInputShape(idx);
    auto antiQuantScaleDesc = context_->GetInputDesc(idx++);
    auto quantScaleShape = context_->GetOptionalInputShape(QUANT_SCALE_INDEX);
    auto quantOffsetShape = context_->GetOptionalInputShape(QUANT_OFFSET_INDEX);
    auto outputShape = context_->GetOutputShape(0);
    auto outputDesc = context_->GetOutputDesc(0);

    OPS_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, weightShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, weightDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, antiQuantScaleShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, antiQuantScaleDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
    // 不支持C8场景
    OP_LOGE_IF(quantScaleShape != nullptr, ge::GRAPH_FAILED, context_->GetNodeName(), "quant scale is not supported.");
    OP_LOGE_IF(quantOffsetShape != nullptr, ge::GRAPH_FAILED, context_->GetNodeName(),
               "quant offset is not supported.");

    return ge::GRAPH_SUCCESS;
}

bool WeightQuantBatchMatmulV2WeightNz::AnalyzeBiasDtype(const gert::CompileTimeTensorDesc *biasDesc)
{
    if (biasDesc != nullptr) {
        inputParams_.biasDtype = biasDesc->GetDataType();

        OP_TILING_CHECK(
            (inputParams_.biasDtype != inputParams_.aDtype),
            VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "bias[%s] dtype should same as x[%s]",
                                            ge::TypeUtils::DataTypeToAscendString(inputParams_.biasDtype).GetString(),
                                            ge::TypeUtils::DataTypeToAscendString(inputParams_.aDtype).GetString()),
            return false);
    }

    return true;
}

bool WeightQuantBatchMatmulV2WeightNz::AnalyzeDtype()
{
    size_t idx = 0;
    inputParams_.aDtype = context_->GetInputDesc(idx++)->GetDataType();
    inputParams_.bDtype = context_->GetInputDesc(idx++)->GetDataType();
    auto antiQuantScaleDtype = context_->GetInputDesc(idx++)->GetDataType();
    auto antiQuantOffsetDesc = context_->GetOptionalInputDesc(idx++);
    auto biasDesc = context_->GetOptionalInputDesc(BIAS_INDEX);
    OP_TILING_CHECK(
        (inputParams_.aDtype != ge::DT_FLOAT16),
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "x only support DT_FLOAT16 dtype, get actual dtype[%s]",
                                        ge::TypeUtils::DataTypeToAscendString(inputParams_.aDtype).GetString()),
        return false);

    inputParams_.cDtype = context_->GetOutputDesc(0)->GetDataType();
    OP_TILING_CHECK(
        (inputParams_.bDtype != ge::DT_INT8),
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "weight only support DT_INT8 dtype, get actual dtype[%s]",
                                        ge::TypeUtils::DataTypeToAscendString(inputParams_.bDtype).GetString()),
        return false);

    OP_TILING_CHECK((inputParams_.cDtype != inputParams_.aDtype),
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        inputParams_.opName,
                        "y dtype should same as x or be DT_INT8 if quant param is provided, get actual dtype[%s]",
                        ge::TypeUtils::DataTypeToAscendString(inputParams_.cDtype).GetString()),
                    return false);
    return AnalyzeBiasDtype(biasDesc) && AnalyzeAntiQuantDtype(antiQuantScaleDtype, antiQuantOffsetDesc);
}

bool WeightQuantBatchMatmulV2WeightNz::AnalyzeAntiQuantDtype(
    ge::DataType antiQuantScaleDtype, const gert::CompileTimeTensorDesc *antiQuantOffsetDesc) const
{
    OP_TILING_CHECK(
        antiQuantScaleDtype != inputParams_.aDtype,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "antiquant_scale[%s] dtype should same as x[%s]",
                                        ge::TypeUtils::DataTypeToAscendString(antiQuantScaleDtype).GetString(),
                                        ge::TypeUtils::DataTypeToAscendString(inputParams_.aDtype).GetString()),
        return false);
    if (antiQuantOffsetDesc != nullptr) {
        auto antiQuantOffsetDtype = antiQuantOffsetDesc->GetDataType();
        OP_TILING_CHECK(
            antiQuantOffsetDtype != inputParams_.aDtype,
            VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "antiquant_offset[%s] dtype should same as x[%s]",
                                            ge::TypeUtils::DataTypeToAscendString(antiQuantOffsetDtype).GetString(),
                                            ge::TypeUtils::DataTypeToAscendString(inputParams_.aDtype).GetString()),
            return false);
    }

    return true;
}

bool WeightQuantBatchMatmulV2WeightNz::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    size_t idx = 0;
    auto transposeX = attrs->GetAttrPointer<bool>(idx++);
    auto transposeWeight = attrs->GetAttrPointer<bool>(idx++);
    const int64_t *groupSizePtr = nullptr;
    if (attrs->GetAttrNum() > idx) {
        groupSizePtr = attrs->GetAttrPointer<int64_t>(idx++);
    }

    OP_TILING_CHECK(
        groupSizePtr != nullptr && *groupSizePtr != 0,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
                                        "group antiquant is not supported, current group size: [%ld]", *groupSizePtr),
        return false);

    // OP_LOG_FULL(DLOG_WARN, inputParams_.opName, "current attr param num is [%lu]", attrs->GetAttrNum());
    inputParams_.transA = transposeX != nullptr && *transposeX;
    inputParams_.transB = transposeWeight != nullptr && *transposeWeight;

    return true;
}

bool WeightQuantBatchMatmulV2WeightNz::SetAntiQuantType(const gert::StorageShape *antiQuantScaleShape)
{
    // antiQuantScaleShape nullptr is impossible
    size_t antiQuantScaleShapeSize = static_cast<size_t>(antiQuantScaleShape->GetStorageShape().GetShapeSize());
    OP_TILING_CHECK(antiQuantScaleShapeSize == 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "antiquant_scale %s cannot be empty tensor",
                                                    ge::Shape2String(antiQuantScaleShape->GetStorageShape()).c_str()),
                    return false);

    if (antiQuantScaleShapeSize == 1) {
        inputParams_.antiQuantType = QuantType::PER_TENSOR;
    } else {
        OP_TILING_CHECK(antiQuantScaleShapeSize != inputParams_.nSize,
                        VECTOR_INNER_ERR_REPORT_TILIING(
                            inputParams_.opName, "antiquant_scale %s shape size should same as N[%lu]",
                            ge::Shape2String(antiQuantScaleShape->GetStorageShape()).c_str(), inputParams_.nSize),
                        return false);
        inputParams_.antiQuantType = QuantType::PER_CHANNEL;
    }
    return true;
}

bool WeightQuantBatchMatmulV2WeightNz::AnalyzeBiasShape() const
{
    auto biasShape = context_->GetOptionalInputShape(BIAS_INDEX);
    if (biasShape != nullptr) {
        auto biasShapeSize = static_cast<uint64_t>(biasShape->GetStorageShape().GetShapeSize());
        OP_TILING_CHECK(
            biasShapeSize != inputParams_.nSize && biasShapeSize != 0,
            VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "bias shape size[%lu] is not equal nSize[%lu] or zero",
                                            biasShape->GetStorageShape().GetShapeSize(), inputParams_.nSize),
            return false);
        OP_TILING_CHECK(
            biasShape->GetStorageShape().GetDimNum() > 1 && biasShape->GetStorageShape().GetDim(0) != 1,
            VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "bias shape only support [1, n] or [n,], input is %s",
                                            ge::Shape2String(biasShape->GetStorageShape()).c_str()),
            return false);
    }
    return true;
}

bool WeightQuantBatchMatmulV2WeightNz::AnalyzeInputs()
{
    size_t idx = 0;
    auto xShape = context_->GetInputShape(idx++);
    auto weightShape = context_->GetInputShape(idx++);
    auto antiQuantScaleShape = context_->GetInputShape(idx++);
    auto antiQuantOffsetShape = context_->GetOptionalInputShape(idx++);
    auto biasShape = context_->GetOptionalInputShape(BIAS_INDEX);
    auto outShape = context_->GetOutputShape(0)->GetStorageShape();
    inputParams_.hasBias = biasShape != nullptr && biasShape->GetStorageShape().GetShapeSize() != 0;
    OP_TILING_CHECK(xShape->GetStorageShape().GetShapeSize() == 0 || weightShape->GetStorageShape().GetShapeSize() == 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "not support empty tensor, x %s, weight %s",
                                                    ge::Shape2String(xShape->GetStorageShape()).c_str(),
                                                    ge::Shape2String(weightShape->GetStorageShape()).c_str()),
                    return false);
    auto shapeSizeA = xShape->GetStorageShape().GetDimNum();
    auto shapeSizeB = weightShape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(
        (shapeSizeA != MM_SHAPE_LEN_ND),
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "left matrix X only support 2 dims Tensor,now is %s",
                                        ge::Shape2String(xShape->GetStorageShape()).c_str()),
        return false);
    inputParams_.mSize = static_cast<uint64_t>(
        inputParams_.transA ? xShape->GetStorageShape().GetDim(shapeSizeA - 1)
                            : xShape->GetStorageShape().GetDim(shapeSizeA - MM_SHAPE_LEN_ND));
    inputParams_.kSize =
        static_cast<uint64_t>(inputParams_.transA ? xShape->GetStorageShape().GetDim(shapeSizeA - MM_SHAPE_LEN_ND)
                                                  : xShape->GetStorageShape().GetDim(shapeSizeA - 1));
    inputParams_.nSize = static_cast<uint64_t>(outShape.GetDim(1));
    uint64_t kBSize = static_cast<uint64_t>(weightShape->GetStorageShape().GetDim(0) *
                                            weightShape->GetStorageShape().GetDim(shapeSizeB - 1));

    return AnalyzeAntiQuantShape(antiQuantScaleShape, antiQuantOffsetShape) && SetAntiQuantType(antiQuantScaleShape) &&
           AnalyzeBiasShape();
}

bool WeightQuantBatchMatmulV2WeightNz::AnalyzeAntiQuantShape(const gert::StorageShape *antiQuantScaleShape,
                                                             const gert::StorageShape *antiQuantOffsetShape)
{
    OP_TILING_CHECK(antiQuantScaleShape->GetStorageShape().GetDimNum() > VALID_INPUT_DIM_NUM ||
                        (antiQuantScaleShape->GetStorageShape().GetDimNum() == VALID_INPUT_DIM_NUM &&
                         antiQuantScaleShape->GetStorageShape().GetDim(1) != 1),
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        inputParams_.opName,
                        "per-channel antiquant_scale shape only support [n, 1] or [n,], actual input shape is %s",
                        ge::Shape2String(antiQuantScaleShape->GetStorageShape()).c_str()),
                    return false);

    if (CheckOptionalInputByShape(antiQuantOffsetShape)) {
        OP_TILING_CHECK(antiQuantScaleShape->GetStorageShape() != antiQuantOffsetShape->GetStorageShape(),
                        VECTOR_INNER_ERR_REPORT_TILIING(
                            inputParams_.opName, "antiquant_scale %s and antiquant_offset %s should have same shape",
                            ge::Shape2String(antiQuantScaleShape->GetStorageShape()).c_str(),
                            ge::Shape2String(antiQuantOffsetShape->GetStorageShape()).c_str()),
                        return false);
        inputParams_.hasAntiQuantOffset = true;
    }
    return true;
}

ge::graphStatus WeightQuantBatchMatmulV2WeightNz::GetShapeAttrsInfo()
{
    inputParams_.opName = context_->GetNodeName();
    auto compileInfoPtr =
        compileInfoPtr_ ? compileInfoPtr_.get()
                        : reinterpret_cast<const WeightQuantBatchMatmulV2CompileInfo *>(context_->GetCompileInfo());
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context_->GetNodeName(), "compileInfoPtr is null");
    OP_TILING_CHECK(CheckContext() != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "invalid context"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(!AnalyzeDtype() || !AnalyzeAttrs() || !AnalyzeInputs(),
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "fail to analyze context info"),
                    return ge::GRAPH_FAILED);

    bool maxDimCheck = inputParams_.kSize > MAX_SHAPE_DIM || inputParams_.nSize > MAX_SHAPE_DIM;
    // A矩阵转置场景，m需要小于65535
    if (inputParams_.transA) {
        maxDimCheck |= inputParams_.mSize > MAX_SHAPE_DIM;
    }
    OP_TILING_CHECK(maxDimCheck,
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        inputParams_.opName, "only support MKN in range [1, %lu], get actual value[%lu, %lu, %lu]",
                        MAX_SHAPE_DIM, inputParams_.mSize, inputParams_.kSize, inputParams_.nSize),
                    return ge::GRAPH_FAILED);

    OP_LOGD(inputParams_.opName, "input params: MKN[%lu, %lu, %lu], transA[%s], transB[%s], bias[%s]",
            inputParams_.mSize, inputParams_.kSize, inputParams_.nSize, inputParams_.transA ? "true" : "false",
            inputParams_.transB ? "true" : "false", inputParams_.hasBias ? "true" : "false");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantBatchMatmulV2WeightNz::InstantiateTilingData()
{
    if (tilingData_ == nullptr) {
        tilingDataManager_ = std::unique_ptr<WeightQuantBatchMatmulV2NzTilingData>(
            new (std::nothrow) WeightQuantBatchMatmulV2NzTilingData());
        tilingData_ = tilingDataManager_.get();
    }
    OP_TILING_CHECK(tilingData_ == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "failed to instantiate tilingData"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        context_->GetRawTilingData()->GetCapacity() < tilingData_->GetDataSize(),
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "tiling data capacity %zu < actual tiling data size %zu",
                                        context_->GetRawTilingData()->GetCapacity(), tilingData_->GetDataSize()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantBatchMatmulV2WeightNz::DoOpTiling()
{
    OP_TILING_CHECK(InstantiateTilingData() == ge::GRAPH_FAILED,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "unable to get pointer of tiling data"),
                    return ge::GRAPH_FAILED);

    // 优先从cache中获取Tiling
    if (GetTilingFromCache()) {
        return ge::GRAPH_SUCCESS;
    }

    uint64_t weightBlockAlignSize = GetBlockAlignSizeByDataType(inputParams_.bDtype);
    if (inputParams_.transB) {
        tilingData_->set_kAlign(ops::CeilAlign(inputParams_.kSize, weightBlockAlignSize));
        tilingData_->set_nAlign(inputParams_.nSize);
    } else {
        tilingData_->set_kAlign(inputParams_.kSize);
        tilingData_->set_nAlign(ops::CeilAlign(inputParams_.nSize, weightBlockAlignSize));
    }
    tilingData_->set_kSize(inputParams_.kSize);
    tilingData_->set_nSize(inputParams_.nSize);
    tilingData_->set_mSize(inputParams_.mSize);
    OP_CHECK(!GetMatMulTiling(),
             VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "failed to get mm tiling for mnk[%lu, %lu, %lu]",
                                             inputParams_.mSize, inputParams_.nSize, inputParams_.kSize),
             return ge::GRAPH_FAILED);
    GetL1tiling();
    GetL1Pingpong();  // 判断A/B在L1上是否开DB_BUFFER
    GetubFactorTiling();
    SetMatmulTiling();
    return ge::GRAPH_SUCCESS;
}

uint64_t WeightQuantBatchMatmulV2WeightNz::GetTilingKey() const
{
    return RecursiveSum(inputParams_.transA, inputParams_.antiQuantType, inputParams_.hasAntiQuantOffset,
                        L1FullloadMode_, KernelTemplateType::WEIGHT_NZ);
}

void WeightQuantBatchMatmulV2WeightNz::Reset()
{
    inputParams_.transA = false;
    inputParams_.transB = false;
    inputParams_.hasBias = false;
    inputParams_.hasAntiQuantOffset = false;
    inputParams_.groupSize = 0UL;
    inputParams_.mSize = 0L;
    inputParams_.kSize = 0L;
    inputParams_.nSize = 0L;

    inputParams_.aDtype = ge::DT_FLOAT16;
    inputParams_.bDtype = ge::DT_INT8;
    inputParams_.cDtype = ge::DT_FLOAT16;
    inputParams_.biasDtype = ge::DT_FLOAT16;
    aFormat = ge::FORMAT_ND;
    bFormat = ge::FORMAT_ND;
    inputParams_.antiQuantType = QuantType::NONE;
    inputParams_.quantType =
        QuantType::PER_TENSOR;  // set default value as per tensor, kernel can detect ydtype firstly
    inputParams_.opName = nullptr;
}

void WeightQuantBatchMatmulV2WeightNz::InitCompileInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null");
        return;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr_ =
        std::unique_ptr<WeightQuantBatchMatmulV2CompileInfo>(new (std::nothrow) WeightQuantBatchMatmulV2CompileInfo());
    OP_TILING_CHECK(compileInfoPtr_ == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "failed to instantiate compile info"),
                    return );

    compileInfoPtr_->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr_->aicNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr_->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr_->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr_->l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr_->l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr_->l0bSize);
    compileInfoPtr_->workspaceNum = ascendcPlatform.GetLibApiWorkSpaceSize();
    compileInfoPtr_->socVersion = ascendcPlatform.GetSocVersion();

    gert::GemmCompileInfo gmmcompileInfo;
    gmmcompileInfo.ParseRuntimePlatformInfo(context_->GetNodeName(), *platformInfoPtr);
    gmmcompileInfo.core_num = compileInfoPtr_->aicNum;
    optiling::PlatformInfo::GetInstance().SetInstance(gmmcompileInfo);
    OP_LOGD(context_->GetNodeName(), "MatmulAllReduce Init Quant Tiling Compile Info Success");
}

static void GetMatmulTilingFromCache(TCubeTiling &matmulTiling, const WeightNzTilingCache &weightNzTilingCache)
{
    matmulTiling.set_M(weightNzTilingCache.m_);
    matmulTiling.set_N(weightNzTilingCache.n_);
    matmulTiling.set_Ka(weightNzTilingCache.ka_);
    matmulTiling.set_Kb(weightNzTilingCache.kb_);
    matmulTiling.set_singleCoreM(weightNzTilingCache.singleCoreM_);
    matmulTiling.set_singleCoreN(weightNzTilingCache.singleCoreN_);
    matmulTiling.set_singleCoreK(weightNzTilingCache.singleCoreK_);
    matmulTiling.set_baseM(weightNzTilingCache.baseM_);
    matmulTiling.set_baseN(weightNzTilingCache.baseN_);
    matmulTiling.set_baseK(weightNzTilingCache.baseK_);
    matmulTiling.set_depthA1(weightNzTilingCache.depthA1_);
    matmulTiling.set_depthB1(weightNzTilingCache.depthB1_);
    matmulTiling.set_stepM(weightNzTilingCache.stepM_);
    matmulTiling.set_stepN(weightNzTilingCache.stepN_);
    matmulTiling.set_stepKa(weightNzTilingCache.stepKa_);
    matmulTiling.set_stepKb(weightNzTilingCache.stepKb_);
    matmulTiling.set_transLength(weightNzTilingCache.transLength_);
    matmulTiling.set_iterateOrder(weightNzTilingCache.iterateOrder_);
    matmulTiling.set_shareL1Size(weightNzTilingCache.shareL1Size_);
    matmulTiling.set_shareL0CSize(weightNzTilingCache.shareL0CSize_);
    matmulTiling.set_dbL0A(weightNzTilingCache.dbL0A_);
    matmulTiling.set_dbL0B(weightNzTilingCache.dbL0B_);
    matmulTiling.set_dbL0C(weightNzTilingCache.dbL0C_);
    matmulTiling.set_usedCoreNum(1);
    matmulTiling.set_batchM(1);
    matmulTiling.set_batchN(1);
    matmulTiling.set_singleBatchM(1);
    matmulTiling.set_singleBatchN(1);
}

bool WeightQuantBatchMatmulV2WeightNz::GetTilingFromCache()
{
    WhiteListShape shape({inputParams_.mSize, inputParams_.nSize, inputParams_.kSize});
    auto it = WEIGHT_NZ_TILING_CACHE.find(shape);
    if (it == WEIGHT_NZ_TILING_CACHE.end()) {
        OP_LOGI(inputParams_.opName, "not find weightNz tiling from cache");
        return false;
    }

    OP_LOGD(inputParams_.opName, "get weightNz tiling from cache");

    auto &weightNzTilingCache = it->second;
    tilingData_->set_cubeBlockDimN(weightNzTilingCache.cubeBlockDimN_);
    tilingData_->set_cubeBlockDimM(weightNzTilingCache.cubeBlockDimM_);
    tilingData_->set_AL1Pingpong(weightNzTilingCache.aL1Pingpong_);
    tilingData_->set_BL1Pingpong(weightNzTilingCache.bL1Pingpong_);
    tilingData_->set_kAlign(weightNzTilingCache.kAlign_);

    tilingData_->set_mSize(weightNzTilingCache.mSize_);
    tilingData_->set_kSize(weightNzTilingCache.kSize_);
    tilingData_->set_nSize(weightNzTilingCache.nSize_);
    tilingData_->set_nAlign(weightNzTilingCache.nSize_);
    tilingData_->set_mAubSize(weightNzTilingCache.mAubSize_);
    tilingData_->set_kAubSize(weightNzTilingCache.kAubSize_);
    tilingData_->set_nBubSize(weightNzTilingCache.nBubSize_);
    tilingData_->set_kBubSize(weightNzTilingCache.kBubSize_);
    tilingData_->set_mCubSize(weightNzTilingCache.mCubSize_);
    tilingData_->set_nCubSize(weightNzTilingCache.nCubSize_);
    tilingData_->set_mAL1Size(weightNzTilingCache.mAL1Size_);
    tilingData_->set_kAL1Size(weightNzTilingCache.kAL1Size_);
    tilingData_->set_nBL1Size(weightNzTilingCache.nBL1Size_);
    tilingData_->set_kBL1Size(weightNzTilingCache.kBL1Size_);
    tilingData_->set_groupSize(weightNzTilingCache.groupSize_);

    auto &matmulTiling = tilingData_->matmulTiling;
    matmulTiling.set_isBias(static_cast<int32_t>(inputParams_.hasBias));
    GetMatmulTilingFromCache(matmulTiling, weightNzTilingCache);

    return true;
}

ge::graphStatus WeightQuantBatchMatmulV2WeightNz::GetWorkspaceSize()
{
    workspaceSize_ = compileInfoPtr_->workspaceNum;
    return ge::GRAPH_SUCCESS;
}

bool WeightQuantBatchMatmulV2WeightNz::GetMmTilingInput(MmTilingInput &mmTilingInput)
{
    mmTilingInput.aDtype = GetMatmulTilingDtype(inputParams_.aDtype);
    mmTilingInput.bDtype = GetMatmulTilingDtype(inputParams_.aDtype);  // same to a
    if (aFormat == ge::FORMAT_ND || aFormat == ge::FORMAT_FRACTAL_NZ) {
        mmTilingInput.aPosition = matmul_tiling::TPosition::GM;
    } else {
        OP_LOGE(inputParams_.opName, "The format of input a is unsupported.");
        return false;
    }
    if (bFormat == ge::FORMAT_ND || bFormat == ge::FORMAT_FRACTAL_NZ) {
        mmTilingInput.aPosition = matmul_tiling::TPosition::GM;
    } else {
        OP_LOGE(inputParams_.opName, "The format of input b is unsupported.");
        return false;
    }
    mmTilingInput.bPosition = matmul_tiling::TPosition::GM;
    mmTilingInput.biasPosition =
        inputParams_.biasDtype == ge::DT_FLOAT ? matmul_tiling::TPosition::GM : matmul_tiling::TPosition::VECCALC;
    return true;
}

bool WeightQuantBatchMatmulV2WeightNz::CheckUBSize()
{
    auto &mmtiling = tilingData_->matmulTiling;
    uint64_t mmUsedUbSize = mmtiling.get_baseM() * mmtiling.get_baseN() * GetSizeByDataType(ge::DT_FLOAT16);
    if (inputParams_.hasBias) {
        if (inputParams_.biasDtype == ge::DT_FLOAT16) {
            mmUsedUbSize +=
                mmtiling.get_baseN() * (GetSizeByDataType(ge::DT_FLOAT16) + GetSizeByDataType(ge::DT_FLOAT));
        } else {
            mmUsedUbSize += mmtiling.get_baseN() * GetSizeByDataType(ge::DT_FLOAT);
        }
    }
    // cub 至少占用的空间
    uint64_t cubNz2NdCanUseSize =
        std::min(mmtiling.get_baseM(), mmtiling.get_baseN()) * GetSizeByDataType(ge::DT_FLOAT16) * BLOCK_CUBE;
    return mmUsedUbSize + cubNz2NdCanUseSize < aicoreParams_.ubSize;
}

MatrixTraverse WeightQuantBatchMatmulV2WeightNz::GetIteratorOrder(const Tiling &tbeTiling, int32_t singleCoreM,
                                                                  int32_t singleCoreN, int32_t singleCoreK) const
{
    int32_t reduceSize = static_cast<int32_t>((inputParams_.aDtype));
    bool fullkAL1Load = singleCoreK <= (tbeTiling.kal1_16 * reduceSize);
    bool fullkBL1Load = singleCoreK <= (tbeTiling.kbl1_16 * reduceSize);

    // if KAL1 and KBL1 both can not be full loaded, then select m or n which is no matter
    if (!fullkAL1Load && !fullkBL1Load) {
        return MatrixTraverse::FIRSTM;
    } else if (fullkAL1Load && !fullkBL1Load) {  // if KAL1 is full loaded, then select the order N first
        return MatrixTraverse::FIRSTN;
    } else if (!fullkAL1Load && fullkBL1Load) {  // if KBL1 is full loaded, then select the order M first
        return MatrixTraverse::FIRSTM;
    } else {
        // if AL1LoadSize less than BL1LoadSize, then select order N first, vice versa.
        int32_t mLoop = ops::CeilDiv(singleCoreM, static_cast<int32_t>(tbeTiling.m_al1 * tbeTiling.m_l0 * BLOCK_CUBE));
        int32_t nLoop = ops::CeilDiv(singleCoreN, static_cast<int32_t>(tbeTiling.n_bl1 * tbeTiling.n_l0 * BLOCK_CUBE));
        int32_t aL1LoadSize = singleCoreM + singleCoreN * mLoop;
        int32_t bL1LoadSize = singleCoreN + singleCoreM * nLoop;
        return aL1LoadSize < bL1LoadSize ? MatrixTraverse::FIRSTN : MatrixTraverse::FIRSTM;
    }
}

void WeightQuantBatchMatmulV2WeightNz::Convert2AscendCTiling(const Tiling &tbeTiling, TCubeTiling &matmulTiling)
{
    auto mDim =
        ops::CeilDiv(inputParams_.mSize, ops::CeilDiv(inputParams_.mSize, static_cast<uint64_t>(tbeTiling.m_dim)));
    auto nDim =
        ops::CeilDiv(inputParams_.nSize, ops::CeilDiv(inputParams_.nSize, static_cast<uint64_t>(tbeTiling.n_dim)));
    tilingData_->set_cubeBlockDimM(static_cast<uint64_t>(mDim));
    tilingData_->set_cubeBlockDimN(static_cast<uint64_t>(nDim));

    // 内轴需256对齐以提高nd2nz效率
    matmulTiling.set_Kb(ops::CeilAlign(tilingData_->get_kSize(), static_cast<uint64_t>(256)));

    matmulTiling.set_singleCoreM(static_cast<int32_t>(ops::CeilDiv(inputParams_.mSize, mDim)));
    matmulTiling.set_singleCoreN(static_cast<int32_t>(ops::CeilDiv(inputParams_.nSize, nDim)));
    matmulTiling.set_baseN(tbeTiling.n_l0 * BLOCK_CUBE);
    matmulTiling.set_baseM(tbeTiling.m_l0 * BLOCK_CUBE);
    int32_t reduceSize = static_cast<int32_t>(GetBlockAlignSizeByDataType(inputParams_.aDtype));
    matmulTiling.set_baseK(tbeTiling.k_l0 * reduceSize);
    auto depthA1 = std::max(ops::CeilDiv(tbeTiling.kal1_16, tbeTiling.k_l0) * tbeTiling.m_al1 * tbeTiling.db_al1, 2L);
    auto depthB1 = std::max(ops::CeilDiv(tbeTiling.kbl1_16, tbeTiling.k_l0) * tbeTiling.n_bl1 * tbeTiling.db_bl1, 2L);
    matmulTiling.set_depthA1(depthA1);
    matmulTiling.set_depthB1(depthB1);
    matmulTiling.set_stepM(tbeTiling.m_al1);
    matmulTiling.set_stepN(tbeTiling.n_bl1);
    matmulTiling.set_stepKa(ops::CeilDiv(tbeTiling.kal1_16, tbeTiling.k_l0));
    matmulTiling.set_stepKb(ops::CeilDiv(tbeTiling.kbl1_16, tbeTiling.k_l0));
    int32_t a1Length = static_cast<int32_t>(
        GetShapeSizeWithDataType(matmulTiling.get_baseM() * matmulTiling.get_baseK(), inputParams_.aDtype));
    int32_t b1Length = static_cast<int32_t>(
        GetShapeSizeWithDataType(matmulTiling.get_baseN() * matmulTiling.get_baseK(), inputParams_.aDtype));
    int32_t c1Length = matmulTiling.get_baseN() * matmulTiling.get_baseM() * sizeof(float);  // L0C

    matmulTiling.set_transLength(std::max(std::max(a1Length, b1Length), c1Length));
    // MatrixTraverse枚举值和matmul api使用的枚举值相差1
    matmulTiling.set_iterateOrder(
        static_cast<int32_t>(GetIteratorOrder(tbeTiling, matmulTiling.get_singleCoreM(), matmulTiling.get_singleCoreN(),
                                              matmulTiling.get_singleCoreK())) -
        1);
    matmulTiling.set_dbL0C(tbeTiling.db_l0c);
    int32_t aL1Size = a1Length * matmulTiling.get_depthA1();
    int32_t biasL1Size =
        inputParams_.hasBias
            ? GetShapeSizeWithDataType(matmulTiling.get_baseN(), inputParams_.biasDtype) * tbeTiling.n_bl1
            : 0;
    int32_t bL1Size = b1Length * matmulTiling.get_depthB1();
    matmulTiling.set_shareL1Size(aL1Size + bL1Size + biasL1Size);
    matmulTiling.set_shareL0CSize(c1Length);

    SetAscendCTiling(matmulTiling);
}

void WeightQuantBatchMatmulV2WeightNz::SetAscendCTiling(TCubeTiling &matmulTiling)
{
    matmulTiling.set_singleCoreK(inputParams_.kSize);
    matmulTiling.set_M(inputParams_.mSize);
    matmulTiling.set_N(inputParams_.nSize);
    matmulTiling.set_Ka(inputParams_.kSize);

    matmulTiling.set_shareMode(0);
    matmulTiling.set_dbL0A(2);  // db switch, 1: off, 2: on
    matmulTiling.set_dbL0B(2);  // db switch, 1: off, 2: on

    matmulTiling.set_shareUbSize(0);
    matmulTiling.set_batchM(1);
    matmulTiling.set_batchN(1);
    matmulTiling.set_singleBatchM(1);
    matmulTiling.set_singleBatchN(1);
    matmulTiling.set_isBias(inputParams_.hasBias ? 1 : 0);

    matmulTiling.set_usedCoreNum(1);
    matmulTiling.set_Kb(tilingData_->get_kAlign());
}

bool WeightQuantBatchMatmulV2WeightNz::InvokeCacheTiling()
{
    BatchmatmulCompileParas compileParams;
    compileParams.binary_mode_flag = true;
    compileParams.bias_flag = inputParams_.hasBias;
    compileParams.pattern_flag = true;

    bool alignedMKN = inputParams_.mSize % BLOCK_CUBE == 0 && inputParams_.kSize % BLOCK_CUBE == 0 &&
                      inputParams_.nSize % BLOCK_CUBE == 0;
    BatchmatmulRunParas runParams;
    runParams.trans_a_flag = inputParams_.transA;
    runParams.trans_b_flag = inputParams_.transB;
    runParams.format_a_nd = true;
    runParams.format_out_nd = true;
    runParams.nd_flag = runParams.format_a_nd && runParams.format_b_nd;
    runParams.used_aligned_pattern = alignedMKN && runParams.nd_flag;
    runParams.bias_flag = inputParams_.hasBias;
    runParams.pattern_flag = !inputParams_.hasBias;
    runParams.unaligned_flag = !alignedMKN;
    runParams.zero_flag = compileParams.zero_flag;
    runParams.weight_nz_flag = true;
    runParams.hf32_flag = 0;
    runParams.dtype_a = static_cast<int32_t>(inputParams_.aDtype);
    runParams.dtype_b = runParams.dtype_a;
    runParams.dtype_out = runParams.dtype_a;
    runParams.dtype_bias = ge::GetSizeByDataType(inputParams_.biasDtype);
    runParams.m = ops::CeilDiv(inputParams_.mSize, static_cast<uint64_t>(BLOCK_CUBE));
    runParams.k = ops::CeilDiv(inputParams_.kSize, static_cast<uint64_t>(BLOCK_CUBE));
    runParams.n = ops::CeilDiv(inputParams_.nSize, static_cast<uint64_t>(BLOCK_CUBE));
    runParams.batch = 1;
    runParams.ori_shape_m = inputParams_.mSize;
    runParams.ori_shape_k = inputParams_.kSize;
    runParams.ori_shape_n = inputParams_.nSize;
    runParams.m_quant_check = inputParams_.transA && inputParams_.cDtype == ge::DT_INT8;
    runParams.n_quant_check = !inputParams_.transB;
    runParams.bias_dtype = inputParams_.biasDtype;
    runParams.vector_pre_conv_mode = inputParams_.cDtype == ge::DT_INT8;
    Tiling tiling;
    tiling.tiling_id = std::numeric_limits<uint64_t>::max();
    GenTiling("WeightQuantBatchMatmulV2", compileParams, runParams, tiling, context_);
    OP_LOGI_IF_RETURN(tiling.tiling_id == std::numeric_limits<uint64_t>::max(), false, inputParams_.opName,
                      "cannot get tiling from cachetiling, mnk[%lu, %lu, %lu]", inputParams_.mSize, inputParams_.nSize,
                      inputParams_.kSize);
    Convert2AscendCTiling(tiling, tilingData_->matmulTiling);

    return true;
}

bool WeightQuantBatchMatmulV2WeightNz::GetMatMulTiling()
{
    if (InvokeCacheTiling() && CheckUBSize()) {
        OP_LOGD(inputParams_.opName, "invoke cache tiling success");
        return true;
    }
    return false;
}

void WeightQuantBatchMatmulV2WeightNz::GetBaseMKNByTrans(matmul_tiling::MatmulApiTiling &mmTiling) const
{
    if (aFormat != ge::FORMAT_ND && bFormat != ge::FORMAT_ND) {
        return;
    }
    mmTiling.SetFixSplit(cubeBaseM_, cubeBaseN_);
    mmTiling.SetSplitRange(defaultValue_, defaultValue_, defaultValue_, defaultValue_, defaultValue_, cubeSingleMinK_);
}

bool WeightQuantBatchMatmulV2WeightNz::GetLoopOrder()
{
    uint64_t mteWeight = 1;
    uint64_t vecWeight = 1;
    uint64_t singleCoreM = ops::CeilDiv(inputParams_.mSize, static_cast<uint64_t>(tilingData_->get_cubeBlockDimM()));

    uint64_t mLoop = ops::CeilDiv(singleCoreM, static_cast<uint64_t>(tilingData_->get_mAL1Size()));
    uint64_t singleCoreN = ops::CeilDiv(inputParams_.nSize, static_cast<uint64_t>(tilingData_->get_cubeBlockDimN()));
    uint64_t nLoop = ops::CeilDiv(singleCoreN, static_cast<uint64_t>(tilingData_->get_nBL1Size()));
    // 额外重复载入及重复计算代价，搬运加权暂定为1，计算加权按照指令条数。
    uint64_t extraAub =
        (nLoop - 1) * (mteWeight * singleCoreN * inputParams_.kSize * GetSizeByDataType(inputParams_.aDtype) +
                       vecWeight * singleCoreM * inputParams_.kSize);

    // cast (add) mul nd2nz 4条指令， add可选
    uint64_t extraBub =
        (mLoop - 1) * (mteWeight * singleCoreN * inputParams_.kSize * GetSizeByDataType(inputParams_.bDtype) +
                       vecWeight * singleCoreN * inputParams_.kSize * ANTI_QUANT_TENSOR);

    return extraAub >= extraBub;
}

// 判断A/B在L1是否开DB_BUFFER
void WeightQuantBatchMatmulV2WeightNz::GetL1Pingpong()
{
    auto mmtiling = tilingData_->matmulTiling;
    uint64_t loopM = static_cast<uint64_t>(ops::CeilDiv(mmtiling.get_singleCoreM(), mmtiling.get_baseM()));
    uint64_t loopN = static_cast<uint64_t>(ops::CeilDiv(mmtiling.get_singleCoreN(), mmtiling.get_baseN()));
    uint64_t depthAk = static_cast<uint64_t>(ops::CeilDiv(tilingData_->get_kAlign(), tilingData_->get_kAL1Size()));
    uint64_t depthBk = static_cast<uint64_t>(ops::CeilDiv(tilingData_->get_kAlign(), tilingData_->get_kBL1Size()));
    // if L1fullload both loopM, depthAk is 1
    uint64_t pingPongA = std::min(static_cast<uint64_t>(DB_BUFFER), loopM * depthAk);
    // if L1fullload both loopN, depthBk is 1
    uint64_t pingPongB = std::min(static_cast<uint64_t>(DB_BUFFER), loopN * depthBk);
    uint64_t aL1Size =
        tilingData_->get_mAL1Size() * tilingData_->get_kAL1Size() * GetSizeByDataType(inputParams_.aDtype);
    uint64_t bL1Size =
        tilingData_->get_kBL1Size() * tilingData_->get_nBL1Size() * GetSizeByDataType(inputParams_.aDtype);
    if (pingPongA * aL1Size + pingPongB * bL1Size <= aicoreParams_.l1Size) {
        tilingData_->set_AL1Pingpong(pingPongA);
        tilingData_->set_BL1Pingpong(pingPongB);
    } else if (pingPongA * aL1Size + bL1Size <= aicoreParams_.l1Size &&
               aL1Size + pingPongB * bL1Size > aicoreParams_.l1Size) {
        tilingData_->set_AL1Pingpong(pingPongA);
        tilingData_->set_BL1Pingpong(1);
    } else if (aL1Size + pingPongB * bL1Size <= aicoreParams_.l1Size &&
               pingPongA * aL1Size + bL1Size > aicoreParams_.l1Size) {
        tilingData_->set_AL1Pingpong(1);
        tilingData_->set_BL1Pingpong(pingPongB);
    } else if (pingPongA * aL1Size + bL1Size <= aicoreParams_.l1Size &&
               aL1Size + pingPongB * bL1Size <= aicoreParams_.l1Size) {
        // according M/N choose pingPong
        tilingData_->set_AL1Pingpong(mmtiling.get_singleCoreN() >= mmtiling.get_singleCoreM() ? 1 : pingPongA);
        tilingData_->set_BL1Pingpong(mmtiling.get_singleCoreN() >= mmtiling.get_singleCoreM() ? pingPongB : 1);
    } else {
        tilingData_->set_AL1Pingpong(1);
        tilingData_->set_BL1Pingpong(1);
    }
}

void WeightQuantBatchMatmulV2WeightNz::GetL1tiling()
{
    auto mmtiling = tilingData_->matmulTiling;
    tilingData_->set_mAL1Size(tilingData_->matmulTiling.get_baseM());
    tilingData_->set_nBL1Size(tilingData_->matmulTiling.get_baseN());
    auto minKL1 = std::min(mmtiling.get_depthA1(), mmtiling.get_depthB1());
    tilingData_->set_kBL1Size(minKL1 * mmtiling.get_baseK());
    tilingData_->set_kAL1Size(minKL1 * mmtiling.get_baseK());
    L1FullloadMode_ = FullLoadMode::NONE_AB_K;
}

uint64_t WeightQuantBatchMatmulV2WeightNz::GetBubSize(uint64_t bubN, uint64_t bubD) const
{
    return BUB_BUF_NUM * bubN * ops::CeilAlign(bubD, BUB_SIZE_ALIGN);
}

uint64_t WeightQuantBatchMatmulV2WeightNz::GetAubSize(uint64_t aubN, uint64_t aubD) const
{
    return AUB_BUF_NUM * ops::CeilAlign(aubN * aubD, AUB_SIZE_ALIGN);
}

uint64_t WeightQuantBatchMatmulV2WeightNz::CalBubFactorTiling(uint64_t bubCanUseUbSize)
{
    auto mmtiling = tilingData_->matmulTiling;
    auto minKL1 = std::min(tilingData_->get_kBL1Size(), tilingData_->get_kAL1Size());
    uint64_t bL1Ddim = minKL1;
    uint64_t bL1Ndim = inputParams_.transB ? mmtiling.get_baseN() : minKL1;
    uint64_t bShapeDdim = inputParams_.transB ? inputParams_.kSize : inputParams_.nSize;

    uint64_t bubDdim = bL1Ddim;
    uint64_t bubNdim = bL1Ndim;
    uint64_t bubUsedSize = 0;
    uint64_t bubTempDdim = std::min(bL1Ddim, static_cast<uint64_t>(MAX_BLOCK_STRIDE));
    for (; bubTempDdim >= BLOCK_CUBE; bubTempDdim -= BLOCK_CUBE) {
        if (bubTempDdim < bShapeDdim && bubTempDdim % BLOCK_REDUCE_INT8 != 0) {
            continue;
        }
        for (uint64_t bubTempNdim = MAX_NBUB_SIZE; bubTempNdim >= BLOCK_CUBE; bubTempNdim -= BLOCK_CUBE) {
            if (bL1Ndim % bubTempNdim != 0) {
                continue;
            }
            bubUsedSize = GetBubSize(bubTempNdim, bubTempDdim);
            if (bubUsedSize < bubCanUseUbSize) {
                bubDdim = bubTempDdim;
                bubNdim = bubTempNdim;
                break;
            }
        }
    }
    tilingData_->set_kBubSize(bubDdim);
    tilingData_->set_nBubSize(bubNdim);
    return bubUsedSize;
}

uint64_t WeightQuantBatchMatmulV2WeightNz::CalAubFactorTiling(uint64_t aubCanUseUbSize)
{
    auto mmtiling = tilingData_->matmulTiling;
    auto minKL1 = std::min(tilingData_->get_kBL1Size(), tilingData_->get_kAL1Size());
    uint64_t aL1Ddim = inputParams_.transA ? mmtiling.get_baseM() : minKL1;
    uint64_t aL1Ndim = inputParams_.transA ? minKL1 : mmtiling.get_baseM();
    uint64_t aubDdim = aL1Ddim;
    uint64_t aubNdim = aL1Ndim;
    uint64_t aubUsedSize = 0;
    uint64_t aubTempDdim = std::min(aL1Ddim, static_cast<uint64_t>(MAX_BLOCK_STRIDE));
    for (; aubTempDdim >= BLOCK_CUBE; aubTempDdim -= BLOCK_CUBE) {
        if (aubTempDdim > 0 && aL1Ddim % aubTempDdim != 0) {
            continue;
        }
        for (uint64_t aubTempNdim = aubNdim; aubTempNdim >= BLOCK_CUBE; aubTempNdim -= BLOCK_CUBE) {
            if (aL1Ndim % aubTempNdim != 0) {
                continue;
            }
            aubUsedSize = GetAubSize(aubTempNdim, aubTempDdim);
            if (aubUsedSize <= aubCanUseUbSize) {
                aubDdim = aubTempDdim;
                aubNdim = aubTempNdim;
                break;
            }
        }
        if (aubUsedSize <= aubCanUseUbSize) {
            break;
        }
    }
    tilingData_->set_kAubSize(inputParams_.transA ? aubNdim : aubDdim);
    tilingData_->set_mAubSize(inputParams_.transA ? aubDdim : aubNdim);
    return aubUsedSize;
}

uint64_t WeightQuantBatchMatmulV2WeightNz::CalCubFactorTiling(uint64_t cubNz2NdCanUseSize)
{
    uint64_t cubM = tilingData_->matmulTiling.get_baseM();
    uint64_t cubN = tilingData_->matmulTiling.get_baseN();
    uint64_t cubUsedSize = 0;
    // cubM is factor of baseM, cubN is factor of baseN
    uint64_t cubTempN = std::min(cubN, static_cast<uint64_t>(MAX_BLOCK_STRIDE));
    for (uint64_t tempM = cubM; tempM >= BLOCK_CUBE; tempM = tempM - BLOCK_CUBE) {
        if (cubM % tempM != 0) {
            continue;
        }
        for (uint64_t tempN = cubTempN; tempN >= BLOCK_CUBE; tempN = tempN - BLOCK_CUBE) {
            if (cubN % tempN != 0) {
                continue;
            }
            cubUsedSize = tempM * tempN * GetSizeByDataType(ge::DT_FLOAT16);
            if (cubUsedSize < cubNz2NdCanUseSize) {
                tilingData_->set_nCubSize(tempN);
                break;
            }
        }
        if (cubUsedSize < cubNz2NdCanUseSize) {
            tilingData_->set_mCubSize(tempM);
            break;
        }
    }
    return cubUsedSize;
}

void WeightQuantBatchMatmulV2WeightNz::PrintCVTilingData(bool debugLevel)
{
    if (debugLevel && AlogCheckDebugLevel(OP, DLOG_DEBUG) != 1) {
        return;
    }

    std::stringstream ss;
    ss << "kAlign: " << tilingData_->get_kAlign() << " kSize: " << tilingData_->get_kSize()
       << " nSize: " << tilingData_->get_nSize() << " mSize: " << tilingData_->get_mSize()
       << " cubeBlockDimN: " << static_cast<uint32_t>(tilingData_->get_cubeBlockDimN())
       << " cubeBlockDimM: " << static_cast<uint32_t>(tilingData_->get_cubeBlockDimM())
       << " mAubSize: " << tilingData_->get_mAubSize() << " kAubSize: " << tilingData_->get_kAubSize()
       << " nBubSize: " << tilingData_->get_nBubSize() << " kBubSize: " << tilingData_->get_kBubSize()
       << " mCubSize: " << tilingData_->get_mCubSize() << " nCubSize: " << tilingData_->get_nCubSize()
       << " mAL1Size: " << tilingData_->get_mAL1Size() << " kAL1Size: " << tilingData_->get_kAL1Size()
       << " nBL1Size: " << tilingData_->get_nBL1Size() << " kBL1Size: " << tilingData_->get_kBL1Size()
       << " AL1Pingpong: " << tilingData_->get_AL1Pingpong() << " BL1Pingpong: " << tilingData_->get_BL1Pingpong();
    int32_t logLevel = debugLevel ? DLOG_DEBUG : DLOG_ERROR;
}

ge::graphStatus WeightQuantBatchMatmulV2WeightNz::PostTiling()
{
    OP_LOGD(inputParams_.opName, "final tiling data size: %zu", tilingData_->GetDataSize());
    OP_CHECK(tilingData_->GetDataSize() % sizeof(uint64_t) != 0,
             VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "tiling data size[%zu] not aligned to 8",
                                             tilingData_->GetDataSize()),
             return ge::GRAPH_FAILED);

    context_->GetRawTilingData()->SetDataSize(tilingData_->GetDataSize());
    uint32_t usedAicNum = tilingData_->get_cubeBlockDimM() * tilingData_->get_cubeBlockDimN();
    context_->SetBlockDim(usedAicNum);
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    tilingData_->SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    PrintCVTilingData(true);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling
