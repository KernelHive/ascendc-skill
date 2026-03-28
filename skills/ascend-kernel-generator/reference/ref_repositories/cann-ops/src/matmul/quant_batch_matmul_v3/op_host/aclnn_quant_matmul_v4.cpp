/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_quant_matmul_v4.h"
#include <dlfcn.h>
#include "aclnn_quant_matmul_v3.h"
#include "aclnn_quant_matmul_weight_nz.h"
#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "matmul_util.h"
#include "quant_matmul_v3_l0.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "op_math_util.h"

using namespace op;
using TupleTensor = std::tuple<const aclTensor *, const aclTensor *, const aclTensor *>;
using TupleAttr = std::tuple<bool, bool>;

namespace {
static constexpr int INDEX_X1_IN_MANDTORY_TUPLE = 0;
static constexpr int INDEX_X2_IN_MANDTORY_TUPLE = 1;
static constexpr int INDEX_SCALE_IN_MANDTORY_TUPLE = 2;
static constexpr int INDEX_OFFSET_IN_OPTIONAL_TUPLE = 0;
static constexpr int INDEX_PERTOKEN_IN_OPTIONAL_TUPLE = 1;
static constexpr int INDEX_BIAS_IN_OPTIONAL_TUPLE = 2;
static constexpr int INDEX_OUT_IN_TUPLE = 2;
static constexpr int INDEX_ISA4W4_IN_BOOL_TUPLE = 2;
static constexpr int LAST_SECOND_DIM_INDEX = 2;

static const int MIN_DIM_NUM_ND = 2;
static const int MAX_DIM_NUM_ND = 6;
static const int MIN_DIM_NUM_NZ = 4;
static const int MAX_DIM_NUM_NZ = 8;
static const int PENULTIMATE_DIM = 2;
static const int NZ_K1_INDEX = 3;
static const int NZ_K1_INDEX_TRANS = 4;
static const int NZ_STORAGE_PENULTIMATE_DIM = 16;
static const int NZ_STORAGE_LAST_DIM = 32;
static const int NZ_K0_VALUE_INT8 = 16;
static const int NZ_K0_VALUE_INT8_TRANS = 32;
static constexpr int64_t OUTPUT_INFER_FAIL = -1;
static const int64_t LAST_AXIS_LIMIT = 65535;
static const int X2_FIXED_DIM_NUM_A4W4 = 2;
static const int64_t INT4_NUMS_IN_INT8 = 2;
static const int64_t INT4_NUMS_IN_INT32 = 8;
static const int64_t INNER_SIZE_MULTIPLE = 64;
static const int64_t K_VALUE = 3696;
static const int64_t N_VALUE = 8192;
static const int64_t M_RANGE1_LEFT = 128;
static const int64_t M_RANGE1_RIGHT = 512;
static const int32_t CORE_NUM_20 = 20;

static const std::initializer_list<op::DataType> IN_TYPE_SUPPORT_LIST = {op::DataType::DT_INT4,
                                                                         op::DataType::DT_INT8};
static const std::initializer_list<op::DataType> OUT_TYPE_SUPPORT_LIST = {op::DataType::DT_INT8,
                                                                          op::DataType::DT_FLOAT16,
                                                                          op::DataType::DT_BF16,
                                                                          op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> SCALE_TYPE_SUPPORT_LIST = {op::DataType::DT_UINT64,
                                                                            op::DataType::DT_BF16,
                                                                            op::DataType::DT_INT64,
                                                                            op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> BIAS_TYPE_SUPPORT_LIST = {op::DataType::DT_INT32,
                                                                           op::DataType::DT_BF16,
                                                                           op::DataType::DT_FLOAT16,
                                                                           op::DataType::DT_FLOAT};

static inline bool CheckNotNull(TupleTensor mandatoryTensors, const aclTensor *out) {
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(scale, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckDtypeValidOnOnlyL0c2ub(TupleTensor mandatoryTensors, TupleTensor optionalTensors,
                                               const aclTensor *out)
{
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto pertokenScaleOptional = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);

    if (x1->GetDataType() != op::DataType::DT_INT8 || x2->GetDataType() != op::DataType::DT_INT8) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input x1 and x2 dtype should be INT8, actual dtype are %s and %s",
                op::ToString(x1->GetDataType()).GetString(), op::ToString(x2->GetDataType()).GetString());
        return false;
    }
    if (!(scale->GetDataType() == op::DataType::DT_UINT64 || scale->GetDataType() == op::DataType::DT_INT64)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Scale dtype should be UINT64 or INT64, actual dtype is %s",
                op::ToString(scale->GetDataType()).GetString());
        return false;
    }
    if (bias != nullptr && bias->GetDataType() != op::DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias dtype should be INT32, actual dtype is %s",
                op::ToString(bias->GetDataType()).GetString());
        return false;
    }
    if (pertokenScaleOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "PertokenScale should be null");
        return false;
    }
    if (!(out->GetDataType() == op::DataType::DT_INT8 || out->GetDataType() == op::DataType::DT_FLOAT16)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output dtype should be INT8 or FLOAT16, actual dtype is %s",
                op::ToString(out->GetDataType()).GetString());
        return false;
    }
    return true;
}

static inline bool CheckDtypeValidOnOnlyL0c2outForA4W4(TupleTensor mandatoryTensors, TupleTensor optionalTensors,
                                                       const aclTensor *out)
{
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto pertokenScaleOptional = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    if (x1->GetDataType() != op::DataType::DT_INT4 || x2->GetDataType() != op::DataType::DT_INT4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Iutput x1 x2 dtype should be INT4 in a4w4 scenario, actual dtype is %s %s.",
                op::ToString(x1->GetDataType()).GetString(), op::ToString(x2->GetDataType()).GetString());
        return false;
    }
    if (pertokenScaleOptional == nullptr) {
        if (scale->GetDataType() != op::DataType::DT_UINT64 && scale->GetDataType() != op::DataType::DT_INT64) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Scale dtype should be UINT64 or INT64 in a4w4 without pertoken scale scenario, actual dtype is %s.",
                    op::ToString(scale->GetDataType()).GetString());
            return false;
        }
        if (bias != nullptr && bias->GetDataType() != op::DataType::DT_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias dtype should be INT32 in a4w4 without pertoken scale scenario, actual dtype is %s",
                    op::ToString(bias->GetDataType()).GetString());
            return false;
        }
    }
    if (out->GetDataType() != op::DataType::DT_FLOAT16 && out->GetDataType() != op::DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output dtype should be FLOAT16 or BF16 in a4w4 scenario, actual dtype is %s.",
                op::ToString(out->GetDataType()).GetString());
        return false;
    }
    return true;
}

static inline bool CheckDtypeValidOnOnlyL0c2outForPertoken(TupleTensor mandatoryTensors, TupleTensor optionalTensors,
                                                           const aclTensor *out)
{
    auto scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto pertokenScaleOptional = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    if (pertokenScaleOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(pertokenScaleOptional, op::DataType::DT_FLOAT, return false);
        if (bias != nullptr && bias->GetDataType() == op::DataType::DT_FLOAT16 &&
            out->GetDataType() != op::DataType::DT_FLOAT16) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "When bias dtype is FLOAT16 with pertokenScale, out dtype should be FLOAT16, actual dtype is %s.",
                    op::ToString(out->GetDataType()).GetString());
            return false;
        }
        if (out->GetDataType() != op::DataType::DT_FLOAT16 && out->GetDataType() != op::DataType::DT_BF16) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "When pertokenScale is not null, out dtype should be FLOAT16 or BF16, actual dtype is %s.",
                    op::ToString(out->GetDataType()).GetString());
            return false;
        }
        if (out->GetDataType() == op::DataType::DT_FLOAT16 && scale->GetDataType() != op::DataType::DT_FLOAT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "When out dtype is FLOAT16 with pertokenScale, scale dtype should be FLOAT, actual dtype is %s.",
                    op::ToString(scale->GetDataType()).GetString());
            return false;
        }
    } else {
        if (bias != nullptr && bias->GetDataType() == op::DataType::DT_FLOAT16) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When pertokenScale is not null, Bias dtype should not be FLOAT16.");
            return false;
        }
        if (bias != nullptr && bias->GetDataType() == op::DataType::DT_FLOAT &&
            out->GetDataType() != op::DataType::DT_BF16) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "When bias dtype is FLOAT without pertokenScale, out dtype should be BF16, actual dtype is %s.",
                    op::ToString(out->GetDataType()).GetString());
            return false;
        }
        if ((out->GetDataType() == op::DataType::DT_INT8 || out->GetDataType() == op::DataType::DT_FLOAT16) &&
            scale->GetDataType() == op::DataType::DT_FLOAT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Scale dtype should not be FLOAT when output dtype is INT8 or FLOAT16 without pertokenScale");
            return false;
        }
    }
    return true;
}

static inline bool CheckDtypeValidOnOnlyL0c2outForUnclassified(TupleTensor mandatoryTensors,
                                                               TupleTensor optionalTensors, const aclTensor *out)
{
    auto scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto pertokenScaleOptional = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    if (bias != nullptr && bias->GetDataType() == op::DataType::DT_BF16 &&
        out->GetDataType() != op::DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When bias dtype is BF16, out dtype should be BF16, actual dtype is %s.",
                op::ToString(out->GetDataType()).GetString());
        return false;
    }
    if (scale->GetDataType() == op::DataType::DT_BF16 && out->GetDataType() != op::DataType::DT_BF16 &&
        out->GetDataType() != op::DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "When scale dtype is BF16, out dtype should be BF16 or INT32, actual dtype is %s",
                op::ToString(out->GetDataType()).GetString());
        return false;
    }
    if (out->GetDataType() == op::DataType::DT_BF16 &&
        !(scale->GetDataType() == op::DataType::DT_BF16 || scale->GetDataType() == op::DataType::DT_FLOAT)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "When out dtype is BF16, scale dtype should be BF16 or FLOAT, actual dtype is %s.",
                op::ToString(scale->GetDataType()).GetString());
        return false;
    }
    if (out->GetDataType() == op::DataType::DT_INT8 &&
        !(scale->GetDataType() == op::DataType::DT_INT64 || scale->GetDataType() == op::DataType::DT_UINT64)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "When out dtype is INT8, scale dtype should be INT64 or UINT64, actual dtype is %s.",
                op::ToString(scale->GetDataType()).GetString());
        return false;
    }
    if (out->GetDataType() == op::DataType::DT_INT32 &&
        !(scale->GetDataType() == op::DataType::DT_FLOAT || scale->GetDataType() == op::DataType::DT_BF16)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "When out dtype is INT32, scale dtype should be FLOAT or BF16, actual dtype is %s.",
                op::ToString(scale->GetDataType()).GetString());
        return false;
    }
    if (out->GetDataType() == op::DataType::DT_INT32 && bias != nullptr &&
        bias->GetDataType() != op::DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When out dtype is INT32, bias dtype should be INT32, actual dtype is %s.",
                op::ToString(bias->GetDataType()).GetString());
        return false;
    }
    return true;
}

static inline bool CheckDtypeValidOnOnlyL0c2out(TupleTensor mandatoryTensors, TupleTensor optionalTensors,
                                                const aclTensor *out, bool isA4W4)
{
    // 对A4W4场景/非A4W4场景进行校验
    if (isA4W4 && !CheckDtypeValidOnOnlyL0c2outForA4W4(mandatoryTensors, optionalTensors, out)) {
        return false;
    }
    if (!CheckDtypeValidOnOnlyL0c2outForUnclassified(mandatoryTensors, optionalTensors, out)) {
        return false;
    }
    if (!CheckDtypeValidOnOnlyL0c2outForPertoken(mandatoryTensors, optionalTensors, out)) {
        return false;
    }
    return true;
}

static inline bool CheckDtypeValid(TupleTensor mandatoryTensors, TupleTensor optionalTensors, const aclTensor *out,
                                   bool isA4W4)
{
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto offset = std::get<INDEX_OFFSET_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto pertokenScaleOptional = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, IN_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, IN_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(scale, SCALE_TYPE_SUPPORT_LIST, return false);
    if (bias != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, BIAS_TYPE_SUPPORT_LIST, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(out, OUT_TYPE_SUPPORT_LIST, return false);

    // 无芯片差异的公共校验
    if (x1->GetDataType() != x2->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x1 and x2 dtype should be same, actual x1 dtype is %s and x2 dtype is %s.",
                op::ToString(x1->GetDataType()).GetString(), op::ToString(x2->GetDataType()).GetString());
        return false;
    }

    if (offset != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(offset, op::DataType::DT_FLOAT, return false);
        // offset only exists if out is int8
        if (out->GetDataType() != op::DataType::DT_INT8) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Offset only exists when out dtype is INT8, actual dtype is %s.",
                    op::ToString(out->GetDataType()).GetString());
            return false;
        }
    }

    // 区分芯片校验
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P &&
        !CheckDtypeValidOnOnlyL0c2ub(mandatoryTensors, optionalTensors, out)) {
        return false;
    } else if ((GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
                GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) &&
               !CheckDtypeValidOnOnlyL0c2out(mandatoryTensors, optionalTensors, out, isA4W4)) {
        return false;
    }
    return true;
}

static inline bool CheckFormatInt4(const aclTensor *x1, const aclTensor *x2) {
    if (x1->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x1 only support ND format in a4w4 scenario, but now is %s.",
                op::ToString(x1->GetStorageFormat()).GetString());
        return false;
    }
    if (x2->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x2 only support ND in a4w4 scenario, but now is %s.",
                op::ToString(x2->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static inline bool CheckFormat(TupleTensor mandatoryTensors, TupleTensor optionalTensors, bool isA4W4) {
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto offset = std::get<INDEX_OFFSET_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto pertokenScaleOptional = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto x2StorageFormat = ge::GetPrimaryFormat(x2->GetStorageFormat());
    bool formatValid = x1->GetStorageFormat() == op::Format::FORMAT_ND &&
                       (x2StorageFormat == op::Format::FORMAT_ND || x2StorageFormat == op::Format::FORMAT_FRACTAL_NZ) &&
                        scale->GetStorageFormat() == op::Format::FORMAT_ND;
    if (offset != nullptr) {
        formatValid = formatValid && offset->GetStorageFormat() == op::Format::FORMAT_ND;
    }
    if (pertokenScaleOptional != nullptr) {
        formatValid = formatValid && pertokenScaleOptional->GetStorageFormat() == op::Format::FORMAT_ND;
    }
    if (bias != nullptr) {
        formatValid = formatValid && bias->GetStorageFormat() == op::Format::FORMAT_ND;
    }
    if (isA4W4) {
        formatValid = formatValid && CheckFormatInt4(x1, x2);
    }
    return formatValid;
}

static inline bool CheckDimRange(const aclTensor *x1, const aclTensor *x2, const aclTensor *scale,
                                 const aclTensor *out) {
    auto x2StorageFormat = ge::GetPrimaryFormat(x2->GetStorageFormat());
    int64_t x2MaxDimNum = x2StorageFormat == op::Format::FORMAT_FRACTAL_NZ ? MAX_DIM_NUM_NZ : MAX_DIM_NUM_ND;
    int64_t x2MinDimNum = x2StorageFormat == op::Format::FORMAT_FRACTAL_NZ ? MIN_DIM_NUM_NZ : MIN_DIM_NUM_ND;
    int64_t x2DimNum = x2->GetStorageShape().GetDimNum();
    CHECK_RET(x2DimNum >= x2MinDimNum && x2DimNum <= x2MaxDimNum, false);
    OP_CHECK_MIN_DIM(x1, MIN_DIM_NUM_ND, return false);
    OP_CHECK_MIN_DIM(out, MIN_DIM_NUM_ND, return false);
    OP_CHECK_MAX_DIM(x1, MAX_DIM_NUM_ND, return false);
    OP_CHECK_MAX_DIM(out, MAX_DIM_NUM_ND, return false);
    OP_CHECK_WRONG_DIMENSION(scale, 1, return false);
    OP_LOGD("QuantMatmul check dim-num range success");
    return true;
}

static int64_t InferOutputShape(const aclTensor *x1, const aclTensor *x2, std::vector<int64_t> &batchRecord) {
    int64_t inferedOutbatchValue = 1;
    auto x1DimNum = x1->GetViewShape().GetDimNum();
    auto x2DimNum = x2->GetViewShape().GetDimNum();
    auto outDimNum = std::max(x1DimNum, x2DimNum);
    auto &longShapeTensor = x1DimNum > x2DimNum ? x1 : x2;
    auto &shortShapeTensor = x1DimNum > x2DimNum ? x2 : x1;
    size_t vaildOffset = outDimNum - std::min(x1DimNum, x2DimNum);
    for (size_t i = 0; i < outDimNum - PENULTIMATE_DIM; i++) {
        auto shortDimValue = i < vaildOffset ? 1 : shortShapeTensor->GetViewShape().GetDim(i - vaildOffset);
        auto longDimValue = longShapeTensor->GetViewShape().GetDim(i);
        if (shortDimValue > 1 && longDimValue > 1 && shortDimValue != longDimValue) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Current short dim value %ld and long dim value %ld are not supported for broadcasting.",
                    shortDimValue, longDimValue);
            return OUTPUT_INFER_FAIL;
        }
        int64_t curBatchValue = static_cast<int64_t>(std::max(shortDimValue, longDimValue));
        inferedOutbatchValue = inferedOutbatchValue * curBatchValue;
        batchRecord.push_back(curBatchValue);
    }
    return inferedOutbatchValue;
}

static inline bool CheckBiasShape(const aclTensor *bias, int64_t x2NDim, const std::vector<int64_t> batchRecord,
                                  int64_t inferedOutbatchValue) {
    auto biasDimNum = bias->GetViewShape().GetDimNum();
    // 3 is bias with batch dim-num
    if (biasDimNum != 3 && biasDimNum != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Bias dim-num should equal 3 or 1, but it is %zu.", biasDimNum);
        return false;
    }
    auto biasFirstDim = bias->GetViewShape().GetDim(0);
    if (biasDimNum == 1) {
        OP_CHECK(biasFirstDim == x2NDim,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias 1st dim should be equal to x2 n dim %ld, but it is %ld.",
                         x2NDim, biasFirstDim),
                 return false);
        return true;
    }
    auto biasSecondDim = bias->GetViewShape().GetDim(1);
    // 2 is bias last dim index
    auto biasThirdDim = bias->GetViewShape().GetDim(2);
    // output batch need to be only 1 dim when bias dim is 3
    if (batchRecord.size() != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "When bias dim-num is 3, infered out batch dim-num should be 1, but infered out batch dim-num is %zu.",
                batchRecord.size());
                return false;
    }
    OP_CHECK(biasFirstDim == inferedOutbatchValue,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Bias 1st dim should be equal to out batch dim, but it is %ld and infered out batch dim is %ld.",
                biasFirstDim, inferedOutbatchValue), return false);
    OP_CHECK(biasSecondDim == 1,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias 2nd dim should be equal to 1, but it is %ld.", biasFirstDim),
             return false);
    OP_CHECK(biasThirdDim == x2NDim,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias last dim should be equal to x2 n dim %ld, but actually is %ld.",
                     x2NDim, biasThirdDim), return false);
    return true;
}

static inline bool CheckOutShape(const aclTensor *out, bool twoDimMatmulCaseFlag, int64_t x1MDim,
                                 int64_t x2NDim, const std::vector<int64_t> &batchRecord) {
    auto outDimNum = out->GetViewShape().GetDimNum();
    int64_t outMDim = out->GetViewShape().GetDim(outDimNum - PENULTIMATE_DIM);
    int64_t outNDim = out->GetViewShape().GetDim(outDimNum - 1);
    size_t inferedOutDimNum = batchRecord.size() + 2;
    // x1 and x2 are 2 dim and out is 3 dim speical case
    if (outMDim == 1 && inferedOutDimNum == 2 && outDimNum == 3 && twoDimMatmulCaseFlag) {
        outDimNum -= 1;
        outMDim = out->GetViewShape().GetDim(0);
    }
    if (inferedOutDimNum != outDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Infered output dim-num %zu is not equal to actual out dim-num %zu.",
                inferedOutDimNum, outDimNum);
        return false;
    }
    OP_CHECK(outMDim == x1MDim,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Out 1st dim should be equal to x1 m dim, but out 1st dim is %ld, x1 m dim is %ld.",
                     outMDim, x1MDim), return false);
    OP_CHECK(outNDim == x2NDim,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Out 2nd dim should be equal to x2 n dim, but out 2nd dim is %ld, x2 n dim is %ld.",
                     outNDim, x2NDim), return false);
    for (size_t i = 0; i < outDimNum - PENULTIMATE_DIM; i++) {
        OP_CHECK(out->GetViewShape().GetDim(i) == batchRecord[i],
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "Output dim %ld is not equal to infered output dim %ld at shape index %zu.",
                         out->GetViewShape().GetDim(i), batchRecord[i], i), return false);
    }
    return true;
}

static inline std::tuple<int64_t, int64_t, int64_t, int64_t> GetX1X2DimValue(const aclTensor *x1, const aclTensor *x2,
                                                                             bool transposeX1, bool transposeX2) {
    auto x1DimNum = x1->GetViewShape().GetDimNum();
    auto x2DimNum = x2->GetViewShape().GetDimNum();
    const op::Shape x1Shape = x1->GetViewShape();
    const op::Shape x2Shape = x2->GetViewShape();
    int64_t x1KDim = transposeX1 ? x1Shape[x1DimNum - PENULTIMATE_DIM] : x1Shape[x1DimNum - 1];
    int64_t x1MDim = transposeX1 ? x1Shape[x1DimNum - 1] : x1Shape[x1DimNum - PENULTIMATE_DIM];
    int64_t x2KDim = transposeX2 ? x2Shape[x2DimNum - 1] : x2Shape[x2DimNum - PENULTIMATE_DIM];
    int64_t x2NDim = transposeX2 ? x2Shape[x2DimNum - PENULTIMATE_DIM] : x2Shape[x2DimNum - 1];
    return std::tie(x1KDim, x1MDim, x2KDim, x2NDim);
}

static inline bool CheckDimValue(const aclTensor *scale, const aclTensor *offset,
                                 const aclTensor *pertokenScaleOptional, int64_t x2NDim, int64_t x1MDim) {
    if (scale->GetViewShape().GetDim(0) != x2NDim && scale->GetViewShape().GetDim(0) != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Scale last dim should equal to x2 n dim %ld or 1, but actual is %ld.",
                x2NDim, scale->GetViewShape().GetDim(0));
        return false;
    }

    if (offset != nullptr) {
        OP_CHECK_WRONG_DIMENSION(offset, 1, return false);
        if (offset->GetViewShape().GetDim(0) != x2NDim && offset->GetViewShape().GetDim(0) != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Offset 1st dim should equal to x2 n dim %ld or 1, but actual is %ld.",
                    x2NDim, offset->GetViewShape().GetDim(0));
            return false;
        }
    }

    if (pertokenScaleOptional != nullptr) {
        OP_CHECK_WRONG_DIMENSION(pertokenScaleOptional, 1, return false);
        if (pertokenScaleOptional->GetViewShape().GetDim(0) != x1MDim) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "PertokenScaleOptional 1st dim should be equal to x1 m dim %ld or 1, but actually is %ld.",
                    x1MDim, pertokenScaleOptional->GetViewShape().GetDim(0));
            return false;
        }
    }
    return true;
}

static inline bool MaxDimCheck(int64_t x1DimNum, int64_t x2DimNum, const op::Shape x1Shape, const op::Shape x2Shape) {
    OP_CHECK(x1Shape[x1DimNum - 1] <= LAST_AXIS_LIMIT && x2Shape[x2DimNum - 1] <= LAST_AXIS_LIMIT,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x1 last dim or x2 last dim is larger than 65535, x1 is %ld, x2 is %ld.",
                x1Shape[x1DimNum - 1], x2Shape[x2DimNum - 1]),
        return false);
    return true;
}

static inline bool CheckShapeForWeightNz(const aclTensor *x1, const aclTensor *x2, bool transposeX1, bool transposeX2) {
    const op::Shape x1Shape = x1->GetViewShape();
    const op::Shape x2Shape = x2->GetStorageShape();
    auto x1DimNum = x1->GetViewShape().GetDimNum();
    auto x2DimNum = x2->GetStorageShape().GetDimNum();
    int64_t x1KDim = transposeX1 ? x1Shape[x1DimNum - PENULTIMATE_DIM] : x1Shape[x1DimNum - 1];
    int64_t x2K1Dim = transposeX2 ? x2Shape[x2DimNum - NZ_K1_INDEX_TRANS] : x2Shape[x2DimNum - NZ_K1_INDEX];
    int64_t aligneValue = transposeX2 ? NZ_K0_VALUE_INT8_TRANS : NZ_K0_VALUE_INT8;
    int64_t alignedX1K = ((x1KDim + aligneValue - 1) / aligneValue) * aligneValue;
    if (alignedX1K != x2K1Dim * aligneValue) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "AlignedK1 value %ld is not matched with k1 value times aligneValue, which is %ld.",
                alignedX1K, x2K1Dim * aligneValue);
        return false;
    }
    return true;
}

template <typename T>
static inline bool IsAligned(T num, T factor)
{
    if (factor == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The divisor cannot be zero.");
        return false;
    }
    return num > 0 && num % factor == 0;
}

static inline bool CheckShapeInt4(const aclTensor *x1, const aclTensor *x2, bool transposeX1, bool transposeX2,
                                  const aclTensor *bias)
{
    int64_t x1KDim, x1MDim, x2KDim, x2NDim;
    std::tie(x1KDim, x1MDim, x2KDim, x2NDim) = GetX1X2DimValue(x1, x2, transposeX1, transposeX2);
    if (!IsAligned<int64_t>(x1KDim, INT4_NUMS_IN_INT8)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "x1_k should be a positive even number in a4w4 senario, but now x1_k is %ld.", x1KDim);
        return false;
    }
    if (transposeX2 && !IsAligned<int64_t>(x2KDim, INT4_NUMS_IN_INT8)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "x2_k should be a positive even number when transposeX2 is true in a4w4 senario, but now x2_k is %ld.", x2KDim);
        return false;
    }
    if (!transposeX2 && !IsAligned<int64_t>(x2NDim, INT4_NUMS_IN_INT8)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "x2_n should be a positive even number when transposeX2 is false in a4w4 senario, but now x2_n is %ld.", x2NDim);
        return false;
    }
    if (transposeX1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "TransposeX1 should be false in a4w4 senario, but now is true.");
        return false;
    }
    if (x2->GetViewShape().GetDimNum() != X2_FIXED_DIM_NUM_A4W4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x2 should be 2-d in a4w4, but is %zu.", x2->GetViewShape().GetDimNum());
        return false;
    }
    if (bias != nullptr && bias->GetViewShape().GetDimNum() != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias should be 1-d in a4w4, but is %zu.", bias->GetViewShape().GetDimNum());
        return false;
    }
    return true;
}

static inline bool CheckShape(TupleTensor &mandatoryTensors, TupleTensor &optionalTensors,
                             TupleAttr &boolsTrans, bool isA4W4, const aclTensor *out) {
    auto transposeX1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(boolsTrans);
    auto transposeX2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(boolsTrans);
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto offset = std::get<INDEX_OFFSET_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto pertokenScaleOptional = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto x1DimNum = x1->GetViewShape().GetDimNum();
    auto x2DimNum = x2->GetViewShape().GetDimNum();
    const op::Shape x1Shape = x1->GetViewShape();
    const op::Shape x2Shape = x2->GetViewShape();
    int64_t x1KDim, x1MDim, x2KDim, x2NDim;
    std::tie(x1KDim, x1MDim, x2KDim, x2NDim) = GetX1X2DimValue(x1, x2, transposeX1, transposeX2);

    if (isA4W4 && !CheckShapeInt4(x1, x2, transposeX1, transposeX2, bias)) {
        return false;
    }

    OP_CHECK(x1KDim == x2KDim,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x1 k dim and x2 k dim should be same, but x1 is %ld, x2 is %ld.",
                     x1KDim, x2KDim), return false);

    CHECK_RET(MaxDimCheck(x1DimNum, x2DimNum, x1Shape, x2Shape), false);

    if (ge::GetPrimaryFormat(x2->GetStorageFormat()) == Format::FORMAT_FRACTAL_NZ) {
        CHECK_RET(CheckShapeForWeightNz(x1, x2, transposeX1, transposeX2), false);
    }

    CHECK_RET(CheckDimValue(scale, offset, pertokenScaleOptional, x2NDim, x1MDim), false);

    std::vector<int64_t> batchRecord;
    int64_t inferedOutbatchValue = InferOutputShape(x1, x2, batchRecord);
    if (inferedOutbatchValue == OUTPUT_INFER_FAIL) {
        return false;
    }
    if (bias != nullptr) {
        if (!CheckBiasShape(bias, x2NDim, batchRecord, inferedOutbatchValue)) {
            return false;
        }
    }
    bool twoDimMatmulCaseFlag = x1DimNum == x2DimNum && x2DimNum == 2;
    CHECK_RET(CheckOutShape(out, twoDimMatmulCaseFlag, x1MDim, x2NDim, batchRecord), false);
    return true;
}

static inline bool CheckEmptyTensor(TupleTensor mandatoryTensors) {
    // scale, out和可选参数已在CheckShape函数校验，无需再次校验空tensor场景。
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    if (x1->IsEmpty() || x2->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "QuantMatmul not support to process empty tensor currently.");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(TupleTensor mandatoryTensors, TupleTensor optionalTensors, TupleAttr boolsTrans,
                               bool isA4W4, const aclTensor *out) {
    // 1. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(mandatoryTensors, optionalTensors, out, isA4W4), ACLNN_ERR_PARAM_INVALID);

    // 2. 检查shape是否符合要求
    CHECK_RET(CheckShape(mandatoryTensors, optionalTensors, boolsTrans, isA4W4, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查format是否符合要求
    CHECK_RET(CheckFormat(mandatoryTensors, optionalTensors, isA4W4), ACLNN_ERR_PARAM_INVALID);

    // 4. 空Tensor处理逻辑
    CHECK_RET(CheckEmptyTensor(mandatoryTensors), ACLNN_ERR_PARAM_INVALID);

    OP_LOGD("QuantMatmul check params success.");
    return ACLNN_SUCCESS;
}

static bool CheckSpecialCase(const aclTensor *tensor, int64_t firstLastDim, int64_t secondLastDim) {
    if (tensor->GetViewShape().GetDim(firstLastDim) == tensor->GetViewShape().GetDim(secondLastDim) == 1) {
        OP_LOGD("QuantMatmul special case, no need to set transpose attr value.");
        return true;
    }
    return false;
}

static bool GetTransposeAttrValue(const aclTensor *tensor, bool transpose, bool checkSpecialCase = true) {
    int64_t dim1 = tensor->GetViewShape().GetDimNum() - 1;
    int64_t dim2 = tensor->GetViewShape().GetDimNum() - PENULTIMATE_DIM;
    // check if tensor is contiguous layout
    if (tensor->GetViewStrides()[dim2] == 1 && tensor->GetViewStrides()[dim1] == tensor->GetViewShape().GetDim(dim2)) {
        OP_LOGD("QuantMatmul GetTransposeAttrValue, find tensor is not contiguous.");
        const_cast<aclTensor *>(tensor)->SetViewShape(SwapLastTwoDimValue(tensor->GetViewShape()));
        // 如果不需要校验特殊case，则直接返回
        if(!checkSpecialCase) {
            return !transpose;
        }
        if (!CheckSpecialCase(tensor, dim1, dim2)) {
            return !transpose;
        }
    }
    return transpose;
}

static const aclTensor* SetTensorToNDFormat(const aclTensor *input) {
    OP_LOGD("QuantMatmul set tensor to ND format.");
    auto formatTensor = const_cast<aclTensor *>(input);
    if (input->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ) {
        formatTensor->SetViewFormat(op::Format::FORMAT_ND);
        formatTensor->SetOriginalFormat(op::Format::FORMAT_ND);
        formatTensor->SetStorageFormat(op::Format::FORMAT_ND);
    }
    return formatTensor;
}

static aclIntArray* GetPerm(int64_t dim, aclOpExecutor* executor) {
    CHECK_RET(dim >= MIN_DIM_NUM_ND, nullptr);
    std::vector<int64_t> valuePerm(dim);
    for (int64_t i = 0; i < dim; i++) {
        valuePerm[i] = i;
    }
    std::swap(valuePerm[dim - 1], valuePerm[dim - PENULTIMATE_DIM]);
    return executor->AllocIntArray(valuePerm.data(), dim);
}

static aclnnStatus TransposeAndTransDataForInputs(const aclTensor *&x1, const aclTensor *&x2, bool& transposeX1,
                                                  bool& transposeX2, aclOpExecutor* executor) {
    if (transposeX1) {
        auto perm = GetPerm(x1->GetViewShape().GetDimNum(), executor);
        CHECK_RET(perm != nullptr, ACLNN_ERR_INNER_NULLPTR);
        x1 = l0op::Transpose(x1, perm, executor);
        CHECK_RET(x1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
        transposeX1 = !transposeX1;
    }
    if (ge::GetPrimaryFormat(x2->GetStorageFormat()) == Format::FORMAT_FRACTAL_NZ) {
        return ACLNN_SUCCESS;
    }
    x2 = SetTensorToNDFormat(x2);
    if (!transposeX2) {
        auto perm = GetPerm(x2->GetViewShape().GetDimNum(), executor);
        CHECK_RET(perm != nullptr, ACLNN_ERR_INNER_NULLPTR);
        x2 = l0op::Transpose(x2, perm, executor);
        CHECK_RET(x2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
        transposeX2 = !transposeX2;
    }
    x2 = l0op::TransData(x2, Format::FORMAT_FRACTAL_NZ, 0, executor);
    CHECK_RET(x2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus TransdataForX1(const aclTensor *&inputTensor, aclOpExecutor* executor)
{
    OP_LOGD("QuantMatmul enter TransdataForX1 func.");
    inputTensor = l0op::Contiguous(inputTensor, executor);
    OP_CHECK(inputTensor != nullptr,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
            "The function Contiguous() return nullptr, which causes function TransdataForX1() to fail."),
        return ACLNN_ERR_INNER_NULLPTR);
    inputTensor = l0op::TransData(inputTensor, Format::FORMAT_FRACTAL_NZ, 0, executor);
    OP_CHECK(inputTensor != nullptr,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                "The function TransData() return nullptr, which causes function TransdataForX1() to fail."),
            return ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static inline bool TensorContiguousProcess(const aclTensor *&contiguousTensor, bool &transpose, aclOpExecutor *executor) {
    if (contiguousTensor == nullptr) {
        OP_LOGD("QuantMatmul no need to do contiguous process.");
        return true;
    }
    auto transposeFlag = IsTransposeLastTwoDims(contiguousTensor);
    // swap tensor if its viewshape not satisfy request shape without adding a transpose node
    if (transposeFlag) {
        contiguousTensor = executor->CreateView(contiguousTensor, SwapLastTwoDimValue(contiguousTensor->GetViewShape()),
                                                contiguousTensor->GetViewOffset());
        transpose = !transpose;
    } else {
        contiguousTensor = l0op::Contiguous(contiguousTensor, executor);
    }
    CHECK_RET(contiguousTensor != nullptr, false);
    return true;
}

static aclnnStatus SpecialOutputProcess(const aclTensor *x1, const aclTensor *x2, const aclTensor *out,
                                        const aclTensor *&matmulRet, aclOpExecutor* executor) {
    // we have to reshape for case which x1 and x2 are 2 dims and out is 3 dims, otherwise, viewcopy will fail
    OP_LOGD("QuantMatmul enter SpecialOutputProcess func.");
    auto x1DimNum = x1->GetViewShape().GetDimNum();
    auto x2DimNum = x2->GetViewShape().GetDimNum();
    auto outShape = out->GetViewShape();
    auto outDimNum = outShape.GetDimNum();
    int64_t outMDim = outShape.GetDim(outDimNum - 2);
    // speical case : x1 and x2 are 2 dim, output is 3 dim, have to reshape matmul result, otherwise viewcopy will fail.
    if (x1DimNum == 2 && outDimNum == 3 && outMDim == 1 && x2DimNum == 2) {
        matmulRet = l0op::Reshape(matmulRet, outShape, executor);
    }
    CHECK_RET(matmulRet != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckSupportSocVersion(bool isA4W4) {
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (isA4W4) {
        // a4w4 support 910B 910_93，其余暂不支持
        switch (socVersion) {
            case SocVersion::ASCEND910B:
            case SocVersion::ASCEND910_93:
                break;
            default: {
                OP_LOGE(ACLNN_ERR_RUNTIME_ERROR,
                        "QuantBatchMatmul support for %s is not implemented in a4w4 senario.",
                        op::ToString(socVersion).GetString());
                return ACLNN_ERR_RUNTIME_ERROR;
            }
        }
    } else {
        // a8w8 support 910B 910_93 310P，其余暂不支持
        switch (socVersion) {
            case SocVersion::ASCEND910B:
            case SocVersion::ASCEND910_93:
            case SocVersion::ASCEND310P:
                break;
            default: {
                OP_LOGE(ACLNN_ERR_RUNTIME_ERROR,
                        "QuantBatchMatmul support for %s is not implemented in a8w8 senario.",
                        op::ToString(socVersion).GetString());
                return ACLNN_ERR_RUNTIME_ERROR;
            }
        }
    }
    return ACLNN_SUCCESS;
}

static const aclTensor* GetNDFormat(const aclTensor *input) {
    const aclTensor* reformatedInput = input;
    if (input != nullptr) {
        reformatedInput = SetTensorToNDFormat(input);
    }
    return reformatedInput;
}

static aclTensor* ConvertTensorToInt4(const aclTensor* input, aclOpExecutor* executor)
{
    // 将int32的输入dtype修改为int4, 同时ViewShape和ViewStrides也从int32修改为int4所对应的。
    auto viewShape = input->GetViewShape();
    auto viewShapeDim = viewShape.GetDimNum();
    viewShape[viewShapeDim - 1] = viewShape[viewShapeDim - 1] * INT4_NUMS_IN_INT32;
    auto inputTemp = executor->CreateView(input, viewShape, input->GetViewOffset());
    inputTemp->SetDataType(DataType::DT_INT4);
    OP_LOGD("The conversion from int32 to int4 is completed.");
    return inputTemp;
}

static void InputPreProcessA4W4(const aclTensor *&x1, const aclTensor *&x2, bool &isA4W4, aclOpExecutor *executor)
{
    if (x1->GetDataType() == DataType::DT_INT32) {
        isA4W4 = true;
        x1 = ConvertTensorToInt4(x1, executor);
    }
    if (x2->GetDataType() == DataType::DT_INT32) {
        isA4W4 = true;
        x2 = ConvertTensorToInt4(x2, executor);
    }
    isA4W4 = isA4W4 || x1->GetDataType() == DataType::DT_INT4 || x2->GetDataType() == DataType::DT_INT4;
}

static aclnnStatus WeightNZCaseProcess(const aclTensor *&x2, bool &transposeX2, aclOpExecutor *executor) {
    // if weight is already in nz format, no need to set contiguous
    if (ge::GetPrimaryFormat(x2->GetStorageFormat()) == op::Format::FORMAT_FRACTAL_NZ) {
        x2->SetOriginalShape(x2->GetViewShape());
    } else {
        CHECK_RET(TensorContiguousProcess(x2, transposeX2, executor), ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus A4W4CaseProcess(const aclTensor *&x1, const aclTensor *&x2, bool &isA4W4, aclOpExecutor *executor) {
    InputPreProcessA4W4(x1, x2, isA4W4, executor);
    CHECK_RET(CheckSupportSocVersion(isA4W4) != ACLNN_ERR_RUNTIME_ERROR, ACLNN_ERR_RUNTIME_ERROR);
    return ACLNN_SUCCESS;
}

static aclnnStatus PostMatmulCalcProcess(const aclTensor *matmulRet, TupleTensor mandatoryTensors,
                                         aclOpExecutor *executor) {
    CHECK_RET(matmulRet != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto out = std::get<INDEX_OUT_IN_TUPLE>(mandatoryTensors);
    CHECK_RET(SpecialOutputProcess(x1, x2, out, matmulRet, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(matmulRet, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus PreMatmulCalcProcess(TupleTensor &mandatoryTensors, TupleAttr &boolsTrans, bool &isA4W4,
                                        const aclTensor *out, aclOpExecutor *executor) {
    auto &x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto &x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto &scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    bool &transposeX1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(boolsTrans);
    bool &transposeX2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(boolsTrans);
    CHECK_RET(executor != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    CHECK_RET(CheckNotNull(std::tie(x1, x2, scale), out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(TensorContiguousProcess(x1, transposeX1, executor), ACLNN_ERR_INNER_NULLPTR);
    auto ret = WeightNZCaseProcess(x2, transposeX2, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    CHECK_RET(CheckDimRange(x1, x2, scale, out), ACLNN_ERR_PARAM_INVALID);
    A4W4CaseProcess(x1, x2, isA4W4, executor);
    return ACLNN_SUCCESS;
}

static void GetDtypeAndTranspose(TupleTensor mandatoryTensors, int64_t &dtype, bool &transposeX1,
                                 bool &transposeX2) {
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto out = std::get<INDEX_OUT_IN_TUPLE>(mandatoryTensors);
    dtype = static_cast<int64_t> (out->GetDataType());
    transposeX1 = GetTransposeAttrValue(x1, transposeX1, true);
    transposeX2 = GetTransposeAttrValue(x2, transposeX2, true);
    OP_LOGD("QuantMatmul attr transposeX1 is %d, transposeX2 is %d.", transposeX1, transposeX2);
}

static aclTensor* ProcessScaleTensor(const aclTensor *scale) {
    auto castedScale = const_cast<aclTensor*>(scale);
    if (castedScale->GetDataType() == op::DataType::DT_INT64) {
        castedScale->SetDataType(op::DataType::DT_UINT64);
    }
    return castedScale;
}

static bool IsX1Transdata(const aclTensor *x1, const aclTensor *x2, int64_t dtype, bool transposeX1, bool transposeX2)
{
    if (transposeX1 == true || transposeX2 == true) {
        return false;
    }
    if (x1->GetStorageFormat() != op::Format::FORMAT_ND || x2->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ) {
        return false;
    }
    if (x1->GetDataType() != op::DataType::DT_INT8) {
        return false;
    }
    if (dtype != static_cast<int>(op::DataType::DT_FLOAT16) && dtype != static_cast<int>(op::DataType::DT_BF16) &&
        dtype != static_cast<int>(op::DataType::DT_INT32)) {
        return false;
    }
    // innersize待校验
    Shape x1Shape = x1->GetViewShape();
    int64_t x1DimNum = x1Shape.GetDimNum();
    Shape x2Shape = x2->GetOriginalShape();
    int64_t x2DimNum = x2Shape.GetDimNum();
    if (x1DimNum != MIN_DIM_NUM_ND || x2DimNum != MIN_DIM_NUM_ND) {
        return false;
    }
    int64_t m = transposeX1 ? x1Shape.GetDim(x1DimNum - 1) : x1Shape.GetDim(x1DimNum - 2);
    int64_t k = transposeX1 ? x1Shape.GetDim(x1DimNum - 2) : x1Shape.GetDim(x1DimNum - 1);
    int64_t n = transposeX2 ? x2Shape.GetDim(x2DimNum - 2) : x2Shape.GetDim(x2DimNum - 1);
    int64_t innerSize = x1Shape.GetDim(x1DimNum - 1);
    // m校验
    bool isSupportedM = false;
    if ((m > M_RANGE1_LEFT && m <= M_RANGE1_RIGHT)) {
        isSupportedM = true;
    }
    if (innerSize % INNER_SIZE_MULTIPLE == 0 || !isSupportedM || k != K_VALUE || n != N_VALUE) {
        return false;
    }
    return true;
}

static aclnnStatus aclnnQuantMatmulGetWorkspaceSizeCommonProcess(TupleTensor mandatoryTensors,
                                                                 TupleTensor optionalTensors,
                                                                 TupleAttr boolsTrans, const aclTensor *out,
                                                                 aclOpExecutor *executor) {
    auto &x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto &x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto &scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto &offset = std::get<INDEX_OFFSET_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto &pertokenScaleOptional = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto &bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    bool &transposeX1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(boolsTrans);
    bool &transposeX2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(boolsTrans);
    bool isA4W4 = false;
    auto ret = PreMatmulCalcProcess(mandatoryTensors, boolsTrans, isA4W4, out, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    bool biasTransposeValue = false;
    CHECK_RET(TensorContiguousProcess(bias, biasTransposeValue, executor), ACLNN_ERR_INNER_NULLPTR);
    auto reformatedX1 = SetTensorToNDFormat(x1);
    const aclTensor* reformatedX2 = x2;
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P) {
        ret = TransposeAndTransDataForInputs(reformatedX1, reformatedX2, transposeX1, transposeX2,
                                             executor);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    } else {
        reformatedX2 = SetTensorToNDFormat(x2);
    }

    const aclTensor* reformatedpertokenScaleOptional = GetNDFormat(pertokenScaleOptional);
    const aclTensor* reformatedBias = GetNDFormat(bias);

    ret = CheckParams(std::tie(reformatedX1, reformatedX2, scale),
                      std::tie(offset, reformatedpertokenScaleOptional, reformatedBias),
                      std::tie(transposeX1, transposeX2), isA4W4, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    auto castedScale = ProcessScaleTensor(scale);
    int64_t dtype = 0;
    GetDtypeAndTranspose(std::tie(reformatedX1, reformatedX2, out), dtype, transposeX1, transposeX2);
    bool isX1TransdataFlag = IsX1Transdata(reformatedX1, reformatedX2, dtype, transposeX1, transposeX2);
    auto socLongVersion = GetCurrentPlatformInfo().GetSocLongVersion();
    bool checkSocLongVersion =
        (socLongVersion == "Ascend910B3" || socLongVersion == "Ascend910B4" || socLongVersion == "Ascend910B4-1");
    auto coreNum = static_cast<int32_t>(GetCurrentPlatformInfo().GetCubeCoreNum());
    if (isX1TransdataFlag && checkSocLongVersion && coreNum == CORE_NUM_20) {
        ret = TransdataForX1(reformatedX1, executor);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }

    // 调用l0算子QuantBatchMatmulV3进行计算
    auto matmulRet = l0op::QuantBatchMatmulV3(reformatedX1, reformatedX2, castedScale, offset, reformatedBias,
                                              reformatedpertokenScaleOptional, dtype, transposeX1, transposeX2,
                                              executor);
    CHECK_RET(PostMatmulCalcProcess(matmulRet, std::tie(x1, x2, out), executor) == ACLNN_SUCCESS, ret);
    return ACLNN_SUCCESS;
}
}

aclnnStatus aclnnQuantMatmulV3GetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *scale,
                                               const aclTensor *offset, const aclTensor *bias, bool transposeX1,
                                               bool transposeX2, const aclTensor *out, uint64_t *workspaceSize,
                                               aclOpExecutor **executor) {
    L2_DFX_PHASE_1(aclnnQuantMatmulV3, DFX_IN(x1, x2, scale, offset, bias), DFX_OUT(out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    const aclTensor *tempPtr = nullptr;
    auto ret = aclnnQuantMatmulGetWorkspaceSizeCommonProcess(std::tie(x1, x2, scale),
                                                             std::tie(offset, tempPtr, bias),
                                                             std::tie(transposeX1, transposeX2),
                                                             out, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantMatmulV4GetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *scale,
                                               const aclTensor *offset, const aclTensor *pertokenScaleOptional,
                                               const aclTensor *bias, bool transposeX1, bool transposeX2,
                                               const aclTensor *out, uint64_t *workspaceSize,
                                               aclOpExecutor **executor) {
    L2_DFX_PHASE_1(aclnnQuantMatmulV4, DFX_IN(x1, x2, scale, offset, pertokenScaleOptional, bias), DFX_OUT(out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    auto ret = aclnnQuantMatmulGetWorkspaceSizeCommonProcess(std::tie(x1, x2, scale),
                                                             std::tie(offset, pertokenScaleOptional, bias),
                                                             std::tie(transposeX1, transposeX2),
                                                             out, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

namespace {
static op::Shape GetWeightNzShape(const aclTensor *input, bool transpose)
{
    int64_t viewDimNum = input->GetViewShape().GetDimNum();
    uint64_t k = transpose ? input->GetViewShape().GetDim(viewDimNum - 1)
                           : input->GetViewShape().GetDim(viewDimNum - LAST_SECOND_DIM_INDEX);
    uint64_t n = transpose ? input->GetViewShape().GetDim(viewDimNum - LAST_SECOND_DIM_INDEX)
                           : input->GetViewShape().GetDim(viewDimNum - 1);

    uint64_t k1 = transpose ? CeilDiv(k, NZ_K0_VALUE_INT8_TRANS) : CeilDiv(k, NZ_K0_VALUE_INT8);
    uint64_t n1 = transpose ? CeilDiv(n, NZ_K0_VALUE_INT8) : CeilDiv(n, NZ_K0_VALUE_INT8_TRANS);

    op::Shape weightNzShape;
    for (uint64_t i = 0; i < viewDimNum - LAST_SECOND_DIM_INDEX; i++) {
        weightNzShape.AppendDim(input->GetViewShape().GetDim(i));
    }
    if (transpose) {
        weightNzShape.AppendDim(k1);
        weightNzShape.AppendDim(n1);
    } else {
        weightNzShape.AppendDim(n1);
        weightNzShape.AppendDim(k1);
    }
    weightNzShape.AppendDim(NZ_STORAGE_PENULTIMATE_DIM);
    weightNzShape.AppendDim(NZ_STORAGE_LAST_DIM);
    return weightNzShape;
}

static bool CheckWeightNzStorageShape(const op::Shape &nzShape, const op::Shape &storageShape)
{
    uint64_t nzDimMultiply = 1;
    uint64_t nzDimNum = nzShape.GetDimNum();
    for (uint64_t i = 0; i < nzDimNum; i++) {
        nzDimMultiply *= nzShape[i];
    }

    uint64_t storageDimMultiply = 1;
    uint64_t storageDimNum = storageShape.GetDimNum();
    for (uint64_t i = 0; i < storageDimNum; i++) {
        storageDimMultiply *= storageShape[i];
    }

    return nzDimMultiply == storageDimMultiply;
}

static const aclTensor *SetTensorToNZFormat(const aclTensor *input, op::Shape &shape, aclOpExecutor *executor)
{
    auto formatTensor = executor->CreateView(input, shape, input->GetViewOffset());
    formatTensor->SetStorageFormat(op::Format::FORMAT_FRACTAL_NZ);
    formatTensor->SetOriginalFormat(op::Format::FORMAT_ND);
    formatTensor->SetViewShape(input->GetViewShape());
    return formatTensor;
}

bool checkNotSupportParam(const aclTensor *yScale, const aclTensor *x1Offset, const aclTensor *yOffset, int64_t groupSize)
{
    if (yScale != nullptr && yScale->GetViewShape().GetShapeSize() != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Current version do not support yScale.");
        return false;
    }

    if (x1Offset != nullptr && x1Offset->GetViewShape().GetShapeSize() != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Current version do not support x1Offset.");
        return false;
    }

    if (yOffset != nullptr && yOffset->GetViewShape().GetShapeSize() != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Current version do not support yOffset.");
        return false;
    }

    if (groupSize != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Current version do not support groupSize.");
        return false;
    }

    return true;
}
}

aclnnStatus aclnnQuantMatmulWeightNzGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *x1Scale,
                                                     const aclTensor *x2Scale, const aclTensor *yScale,
                                                     const aclTensor *x1Offset, const aclTensor *x2Offset,
                                                     const aclTensor *yOffset, const aclTensor *bias, bool transposeX1,
                                                     bool transposeX2, int64_t groupSize, aclTensor *out,
                                                     uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnQuantMatmulWeightNz,
                   DFX_IN(x1, x2, x1Scale, x2Scale, yScale, x1Offset, x2Offset, yOffset, bias, transposeX1, transposeX2,
                          groupSize),
                   DFX_OUT(out));

    if (!checkNotSupportParam(yScale, x1Offset, yOffset, groupSize)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (x2 == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "QuantMatmul WeightNz do not support x2 is nullptr.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    int64_t viewDimNum = x2->GetViewShape().GetDimNum();
    if(viewDimNum < MIN_DIM_NUM_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x2's view dimNum should greater than 1, but is %ld.", viewDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }

    transposeX2 = GetTransposeAttrValue(x2, transposeX2, false);

    op::Shape weightNzShape = GetWeightNzShape(x2, transposeX2);
    if (!CheckWeightNzStorageShape(weightNzShape, x2->GetStorageShape())) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
              "x2'format only support NZ, but now x2's format is not NZ(Ascend affinity format). \
aclnnCalculateMatmulWeightSizeV2 and aclnnTransMatmulWeight can be used to convert the input format from ND to Ascend \
affinity format.");
      return ACLNN_ERR_PARAM_INVALID;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    x2 = SetTensorToNZFormat(x2, weightNzShape, uniqueExecutor.get());
    auto ret =
        aclnnQuantMatmulGetWorkspaceSizeCommonProcess(std::tie(x1, x2, x2Scale), std::tie(x2Offset, x1Scale, bias),
                                                      std::tie(transposeX1, transposeX2), out, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantMatmulV3(void *workspace, uint64_t workspaceSize,
                               aclOpExecutor *executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnQuantMatmulV3);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnQuantMatmulV4(void *workspace, uint64_t workspaceSize,
                               aclOpExecutor *executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnQuantMatmulV4);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnQuantMatmulWeightNz(void *workspace, uint64_t workspaceSize,
                               aclOpExecutor *executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnQuantMatmulWeightNz);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
