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
 * \file quant_batch_matmul_v3_tiling.cc
 * \brief
 */

#include "quant_batch_matmul_v3_tiling.h"

#include <map>
#include <numeric>

#include "tiling/tiling_templates_registry.h"
#include "tiling/tiling_type.h"
#include "cube_tiling_runtime.h"
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "quant_batch_matmul_info_factory.h"

namespace optiling {
#define CUBE_INNER_ERR_REPORT(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)
}  // namespace optiling

using AscendC::BLOCK_CUBE;    // uint32_t 16
using AscendC::ONE_BLK_SIZE;  // uint32_t 32

namespace {
constexpr uint64_t INT_REDUCE_FACTOR = 32;
constexpr size_t LAST_FIRST_DIM_INDEX = 1;
constexpr size_t LAST_SECOND_DIM_INDEX = 2;
constexpr size_t LAST_BATCH_DIM_INDEX = 3;
constexpr size_t MIN_DIM_NUM_ND = 2;
constexpr size_t MAX_DIM_NUM_ND = 6;
constexpr size_t A4W4_X2_LEN = 2;
constexpr size_t BIAS_THREE_DIM = 3;
constexpr uint32_t WORKSPACE_LIMIT = 50 * 1024 * 1024; // workspaca limit 50M
constexpr int32_t IDX_K_LOW = 2;
constexpr int32_t IDX_K_HIGH = 3;
constexpr int32_t IDX_N_LOW = 4;
constexpr int32_t IDX_N_HIGH = 5;
constexpr int32_t IDX_B_LOW = 6;
constexpr int32_t BANK_LEN = 512;
constexpr int64_t LAST_AXIS_LIMIT = 65535;

// QuantBatchMatmulV3 input index, mc2 is not same
constexpr uint32_t X1_INDEX = 0;
constexpr uint32_t X2_INDEX = 1;
constexpr uint32_t SCALE_INDEX = 2;
constexpr uint32_t OFFSET_INDEX = 3;
constexpr uint32_t BIAS_INDEX = 4;
constexpr uint32_t PERTOKEN_SCALE_INDEX = 5;
using QuantBatchMatmulV3CompileInfo = gert::GemmCompileInfo;

constexpr uint64_t BIAS_TABLE_NUM = 256;
constexpr uint64_t DATA_SIZE_FP32 = 4;
constexpr uint64_t UB_EXTRE_BYTE = 8;

constexpr uint64_t L2_REAL_SIZE = 168;  // B4真实的L2Size大小
constexpr uint64_t L2_FAKE_SIZE = 96;   // B4被上层修改后的L2Size大小


static constexpr int8_t OUTPUT_INFER_FAIL = -1;
static constexpr int8_t OUTPUT_INFER_SUCCESS = 1;
const std::map<ge::DataType, matmul_tiling::DataType> DTYPE_MAP =
{
    {ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16},
    {ge::DT_FLOAT, matmul_tiling::DataType::DT_FLOAT},
    {ge::DT_BF16, matmul_tiling::DataType::DT_BF16},
    {ge::DT_INT8, matmul_tiling::DataType::DT_INT8},
    {ge::DT_INT4, matmul_tiling::DataType::DT_INT4},
    {ge::DT_INT32, matmul_tiling::DataType::DT_INT32},
};

matmul_tiling::DataType GetMatmulTilingDtype(ge::DataType dtype)
{
    auto it = DTYPE_MAP.find(dtype);
    return it != DTYPE_MAP.end() ? it->second : matmul_tiling::DataType::DT_UNDEFINED;
}

template <typename T>
static T CalcTailSize(T num1, T num2)
{
    OP_TILING_CHECK(num2 == 0, CUBE_INNER_ERR_REPORT("Nil", "cannot divide by zero"), return 0);

    T mod = num1 % num2;
    if (mod != 0) {
        return mod;
    } else {
        return num2;
    }
}

// 对{K,N}满足以下组合可走增量优化模板，后续应撤销白名单
const std::vector<std::pair<uint64_t, uint64_t>> WHITE_LIST_X2_KN {
    {11264, 6912}, {11264, 1664}, {1408, 11264}, {6912, 11264},
    {8192, 2560}, {2048, 8192}, {5504, 8192}, {8192, 11008}
};

static optiling::QuantBatchMatmulInfoFactory g_quantBatchMatmulInfoFactory;

std::string DType2Str(const ge::DataType dataType)
{
    std::string serialString = ge::TypeUtils::DataTypeToSerialString(dataType);
    std::string prefix = "DT_";
    size_t pos = serialString.find(prefix);
    if (pos != std::string::npos) {
        serialString.erase(pos, prefix.length());
    }
    return serialString;
}
}  // namespace

namespace optiling {
uint64_t QuantBatchMatmulInfo::GetMatmulApiMSize() const
{
    return mSizePerNpu > 0 ? mSizePerNpu : mSize;
}

uint64_t QuantBatchMatmulInfo::GetTotalMatmulApiMSize(uint64_t baseM) const
{
    if (mSizePerNpu > 0) {
        return ops::CeilAlign(mSizePerNpu, baseM) * ops::CeilDiv(mSize, mSizePerNpu);
    } else {
        return mSize;
    }
}

uint64_t QuantBatchMatmulInfo::GetTotalBaseMCnt(uint64_t baseM) const
{
    return ops::CeilDiv(GetTotalMatmulApiMSize(baseM), baseM); // m方向需要的轮数
}

void QuantBatchMatmulInfo::Reset()
{
    initFlag = false;
    transA = false;
    transB = false;
    hasBias = false;
    mSize = 0UL;
    mSizePerNpu = 0UL;
    kSize = 0UL;
    nSize = 0UL;
    batchA = 0UL;
    batchA1 = 0UL;
    batchA2 = 0UL;
    batchA3 = 0UL;
    batchA4 = 0UL;
    batchB = 0UL;
    batchB1 = 0UL;
    batchB2 = 0UL;
    batchB3 = 0UL;
    batchB4 = 0UL;
    batchC = 0UL;
    batchBias = 0UL;
    aDtype = ge::DT_INT8;
    bDtype = ge::DT_INT8;
    cDtype = ge::DT_FLOAT16;
    biasDtype = ge::DT_INT32;
    scaleDtype = ge::DT_UINT64;
    isPerTensor = false;
    isPertoken = false;
    outDtype = 0L;
    libApiWorkSpaceSize = 0U;
    bf16ExtreWorkSpaceSize = 0UL;
    opName = nullptr;
    aFormat = ge::FORMAT_ND;
    bFormat = ge::FORMAT_ND;
    cFormat = ge::FORMAT_ND;
}

QuantBatchMatmulV3Tiling::QuantBatchMatmulV3Tiling(gert::TilingContext *context)
    : TilingBaseClass(context), inputParams_(*(g_quantBatchMatmulInfoFactory.Get())), tilingData_(tilingDataSelf_)
{
    Reset();
}

QuantBatchMatmulV3Tiling::QuantBatchMatmulV3Tiling(gert::TilingContext *context, QuantBatchMatmulV3TilingData *out)
    : TilingBaseClass(context),
      inputParams_(*(g_quantBatchMatmulInfoFactory.Get())),
      tilingData_(*out),
      isTilingOut_(true)
{
    Reset();
    InitCompileInfo();
    inputParams_.Reset();
}

void QuantBatchMatmulV3Tiling::InitCompileInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null");
        return;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    gert::GemmCompileInfo compileInfo;
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfo.l1_size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfo.l0c_size);
    compileInfo.ub_size = ubSize;
    compileInfo.workspace_num = ascendcPlatform.GetLibApiWorkSpaceSize();
    compileInfo.ParseRuntimePlatformInfo(context_->GetNodeName(), *platformInfoPtr);
    compileInfo.core_num = ascendcPlatform.GetCoreNumAic();
    optiling::PlatformInfo &plaformInstance = optiling::PlatformInfo::GetInstance();
    plaformInstance.SetInstance(compileInfo);
    if (plaformInstance.core_num <= 0) {
        OP_LOGE(context_->GetNodeName(), "coreNum <= 0");
        return;
    }
    OP_LOGD(context_->GetNodeName(), "MatmulAllReduce Init Quant Tiling Compile Info Success");
    compileInfoInit_ = true;
    compileInfo_ = compileInfo;
}

void QuantBatchMatmulV3Tiling::Reset()
{
    isBf16Opt_ = false;
    isUbQuant_ = false;

    if (!isTilingOut_) {
        tilingData_.SetDataPtr(context_->GetRawTilingData()->GetData());
        OP_TILING_CHECK(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                                 0, context_->GetRawTilingData()->GetCapacity()) != EOK,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName, "fail to clear tiling data"), return);
    }
}

ge::graphStatus QuantBatchMatmulV3Tiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulV3Tiling::GetShapeAttrsInfo()
{
    if (inputParams_.initFlag) {
        OP_LOGD(inputParams_.opName, "no need to get shape and attrs from tiling context again");
        return ge::GRAPH_SUCCESS;
    }

    inputParams_.opName = context_->GetNodeName();
    OP_TILING_CHECK(CheckContext() != ge::GRAPH_SUCCESS, CUBE_INNER_ERR_REPORT(inputParams_.opName, "invalid context"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(!AnalyzeAttrs() || !AnalyzeDtype() || !AnalyzeInputs(),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "fail to analyze context info"),
                    return ge::GRAPH_FAILED);

    OP_LOGD(inputParams_.opName, "input params: MKN[%ld, %ld, %ld], transA[%s], transB[%s], bias[%s]",
            inputParams_.mSize, inputParams_.kSize, inputParams_.nSize, inputParams_.transA ? "true" : "false",
            inputParams_.transB ? "true" : "false", inputParams_.hasBias ? "true" : "false");

    inputParams_.initFlag = true;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulV3Tiling::CheckContext()
{
    auto x1Shape = context_->GetInputShape(X1_INDEX);
    auto x1Desc = context_->GetInputDesc(X1_INDEX);
    auto x2Shape = context_->GetInputShape(X2_INDEX);
    auto x2Desc = context_->GetInputDesc(X2_INDEX);
    auto scaleShape = context_->GetInputShape(SCALE_INDEX);
    auto scaleDesc = context_->GetInputDesc(SCALE_INDEX);
    auto outputShape = context_->GetOutputShape(0);
    auto outputDesc = context_->GetOutputDesc(0);
    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "context_->GetAttrs() failed"),
                    return ge::GRAPH_FAILED);
    auto dtypeAttr = attrs->GetAttrPointer<int64_t>(0);

    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1Desc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2Desc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, scaleShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, scaleDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, dtypeAttr);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
    OP_TILING_CHECK(
        context_->GetRawTilingData()->GetCapacity() < tilingData_.GetDataSize(),
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "context tiling data capacity %zu < actual tiling data size %zu",
                              context_->GetRawTilingData()->GetCapacity(), tilingData_.GetDataSize()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool QuantBatchMatmulV3Tiling::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    // if op calls QuantBatchMatmulV3 with attrs, it could rewrite as GetXXXAttr()
    if (attrs) {
        size_t idx = 0;
        auto dtypePtr = attrs->GetAttrPointer<int64_t>(idx++);
        auto transposeX1Ptr = attrs->GetAttrPointer<bool>(idx++);
        auto transposeX2Ptr = attrs->GetAttrPointer<bool>(idx++);
        OP_TILING_CHECK(!dtypePtr,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName, "There should be at least the required dtype attr"),
                        return false);
        inputParams_.outDtype = *dtypePtr;
        inputParams_.transA = transposeX1Ptr ? *transposeX1Ptr : false;
        inputParams_.transB = transposeX2Ptr ? *transposeX2Ptr : false;
    }

    QuantBatchMatmulV3Trans trans = QuantBatchMatmulV3Trans::NO_TRANS;
    SetTransAttr(trans);
    if (!optiling::PlatformInfo::GetInstance().support_l0c2out()) {
        OP_TILING_CHECK(
            trans != QuantBatchMatmulV3Trans::B_TRANS,
            CUBE_INNER_ERR_REPORT(
                inputParams_.opName,
                "this platform only support output transpose_x1 false and transpose_x2 true, actual [%s, %s]",
                inputParams_.transA ? "true" : "false", inputParams_.transB ? "true" : "false"),
            return false);
    }
    return true;
}

bool QuantBatchMatmulV3Tiling::CheckDtypeOnOnlyL0c2ub() const
{
    OP_TILING_CHECK(
        inputParams_.aDtype != DT_INT8 || inputParams_.bDtype != DT_INT8,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Input x1 and x2 dtype should be INT8, actual dtype are %s and %s",
                              DType2Str(inputParams_.aDtype).c_str(), DType2Str(inputParams_.bDtype).c_str()),
        return false);
    OP_TILING_CHECK(
        inputParams_.scaleDtype != DT_UINT64 && inputParams_.scaleDtype != DT_INT64,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Scale dtype should be UINT64 or INT64, actual dtype is %s",
                              DType2Str(inputParams_.scaleDtype).c_str()),
        return false);
    OP_TILING_CHECK(
        inputParams_.cDtype != DT_INT8 && inputParams_.cDtype != DT_FLOAT16,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Output dtype should be INT8 or FLOAT16, actual dtype is %s",
                              DType2Str(inputParams_.cDtype).c_str()),
        return false);
    OP_TILING_CHECK(context_->GetOptionalInputDesc(BIAS_INDEX) != nullptr && inputParams_.biasDtype != DT_INT32,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "Bias dtype should be INT32, actual dtype is %s",
                                          DType2Str(inputParams_.biasDtype).c_str()),
                    return false);
    auto x1Desc = context_->GetInputDesc(X1_INDEX);
    auto x1Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x1Desc->GetStorageFormat()));
    OP_TILING_CHECK(x1Format != Format::FORMAT_ND,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "X1 format should be ND, actual format is %s",
                                          TypeUtils::FormatToSerialString(x1Format).c_str()),
                    return false);
    auto x2Desc = context_->GetInputDesc(X2_INDEX);
    auto x2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x2Desc->GetStorageFormat()));
    OP_TILING_CHECK(x2Format != Format::FORMAT_FRACTAL_NZ,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "X2 format should be FRACTAL_NZ , actual format is %s",
                                          TypeUtils::FormatToSerialString(x2Format).c_str()),
                    return false);
    OP_TILING_CHECK(context_->GetOptionalInputDesc(PERTOKEN_SCALE_INDEX) != nullptr &&
                        context_->GetOptionalInputShape(PERTOKEN_SCALE_INDEX) != nullptr,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "PertokenScale should be null"), return false);
    return true;
}

bool QuantBatchMatmulV3Tiling::CheckDtypeOnOnlyL0c2outForSupportedList() const
{
    // x1 x2 scale bias y 合法dtype
    OP_TILING_CHECK(
        !(inputParams_.aDtype == DT_INT8 || inputParams_.aDtype == DT_INT4) ||
            !(inputParams_.bDtype == DT_INT8 || inputParams_.bDtype == DT_INT4),
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Input dtype should be INT8 or DT_INT4, actual dtype are %s and %s",
                              DType2Str(inputParams_.aDtype).c_str(),
                              DType2Str(inputParams_.bDtype).c_str()),
        return false);
    OP_TILING_CHECK(!(inputParams_.scaleDtype == DT_UINT64 || inputParams_.scaleDtype == DT_BF16 ||
                      inputParams_.scaleDtype == DT_INT64 || inputParams_.scaleDtype == DT_FLOAT),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "Scale dtype should be UINT64, BF16, INT64 or FLOAT, actual dtype is %s",
                                          DType2Str(inputParams_.scaleDtype).c_str()),
                    return false);
    OP_TILING_CHECK(context_->GetOptionalInputDesc(BIAS_INDEX) != nullptr &&
                        !(inputParams_.biasDtype == DT_INT32 || inputParams_.biasDtype == DT_BF16 ||
                          inputParams_.biasDtype == DT_FLOAT16 || inputParams_.biasDtype == DT_FLOAT),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "Bias dtype should be INT32, BF16, FLOAT16 or FLOAT, actual dtype is %s",
                                          DType2Str(inputParams_.biasDtype).c_str()),
                    return false);
    OP_TILING_CHECK(!(inputParams_.cDtype == DT_INT8 || inputParams_.cDtype == DT_FLOAT16 ||
                      inputParams_.cDtype == DT_BF16 || inputParams_.cDtype == DT_INT32),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "Output dtype should be INT8, FLOAT16, BF16 or INT32, actual dtype is %s",
                                          DType2Str(inputParams_.cDtype).c_str()),
                    return false);
    return true;
}

bool QuantBatchMatmulV3Tiling::CheckDtypeOnOnlyL0c2outForA4W4() const
{
    // int4
    if (inputParams_.aDtype == DT_INT4) {
        OP_TILING_CHECK(
            inputParams_.cDtype != DT_FLOAT16 && inputParams_.cDtype != DT_BF16,
            CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                  "When input dtype is int4, output dtype should be FLOAT16 or BF16, actual dtype is %s",
                                  DType2Str(inputParams_.cDtype).c_str()),
            return false);
        // a4w4场景，x1必须为ND
        auto x1Desc = context_->GetInputDesc(X1_INDEX);
        auto x1Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x1Desc->GetStorageFormat()));
        OP_TILING_CHECK(x1Format != Format::FORMAT_ND,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName, "X1 format should be ND, actual format is %s",
                                              TypeUtils::FormatToSerialString(x1Format).c_str()),
                        return false);
        if (context_->GetOptionalInputDesc(PERTOKEN_SCALE_INDEX) == nullptr ||
            context_->GetOptionalInputShape(PERTOKEN_SCALE_INDEX) == nullptr) {
            OP_TILING_CHECK(
                inputParams_.scaleDtype != DT_UINT64 && inputParams_.scaleDtype != DT_INT64,
                CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                    "When input dtype is int4 without pertoken scale, scale dtype should be UINT64 or INT64, actual dtype is %s",
                                    DType2Str(inputParams_.scaleDtype).c_str()),
                return false);
            OP_TILING_CHECK(
                context_->GetOptionalInputDesc(BIAS_INDEX) != nullptr && inputParams_.biasDtype != DT_INT32,
                CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                    "When input dtype is int4 without pertoken scale, bias dtype should be INT32, actual dtype is %s",
                                    DType2Str(inputParams_.biasDtype).c_str()),
                return false);
            } else if (!CheckDtypeOnOnlyL0c2outForPertoken()) {
                return false;
            }
    }
    return true;
}

bool QuantBatchMatmulV3Tiling::CheckDtypeOnOnlyL0c2outForPertoken() const
{
    if (context_->GetOptionalInputDesc(PERTOKEN_SCALE_INDEX) != nullptr &&
        context_->GetOptionalInputShape(PERTOKEN_SCALE_INDEX) != nullptr) {
        // 当bias为FLOAT16,并且有pertoken时，y必须是FLOAT16
        OP_TILING_CHECK(
            context_->GetOptionalInputDesc(BIAS_INDEX) != nullptr && inputParams_.biasDtype == DT_FLOAT16 &&
                inputParams_.cDtype != DT_FLOAT16,
            CUBE_INNER_ERR_REPORT(
                inputParams_.opName,
                "When bias dtype is FLOAT16 with pertokenScale, output dtype should be FLOAT16, actual dtype is %s",
                DType2Str(inputParams_.cDtype).c_str()),
            return false);
        // 有pertoken时，y必须是FLOAT16/BF16
        OP_TILING_CHECK(
            !(inputParams_.cDtype == DT_FLOAT16 || inputParams_.cDtype == DT_BF16),
            CUBE_INNER_ERR_REPORT(
                inputParams_.opName,
                "When pertokenScale is not null, output dtype should be FLOAT16 or BF16, actual dtype is %s",
                DType2Str(inputParams_.cDtype).c_str()),
            return false);
        // 当y为FLOAT16,并且有pertoken时，scale必须是FLOAT
        OP_TILING_CHECK(
            inputParams_.cDtype == DT_FLOAT16 && inputParams_.scaleDtype != DT_FLOAT,
            CUBE_INNER_ERR_REPORT(
                inputParams_.opName,
                "When output dtype is FLOAT16 with pertokenScale, scale dtype should be FLOAT, actual dtype is %s",
                DType2Str(inputParams_.scaleDtype).c_str()),
            return false);
    } else {
        // 当无pertoken时，bias不能是FLOAT16
        OP_TILING_CHECK(
            context_->GetOptionalInputDesc(BIAS_INDEX) != nullptr && inputParams_.biasDtype == DT_FLOAT16,
            CUBE_INNER_ERR_REPORT(inputParams_.opName, "When pertokenScale is null, bias dtype can not be FLOAT16"),
            return false);
        // 当bias为FLOAT,并且无pertoken时，y必须是BF16
        OP_TILING_CHECK(
            context_->GetOptionalInputDesc(BIAS_INDEX) != nullptr && inputParams_.biasDtype == DT_FLOAT &&
                inputParams_.cDtype != DT_BF16,
            CUBE_INNER_ERR_REPORT(
                inputParams_.opName,
                "When bias dtype is FLOAT without pertokenScale, output dtype should be BF16, actual dtype is %s",
                DType2Str(inputParams_.cDtype).c_str()),
            return false);
        // 当y为INT8或FLOAT16,并且无pertoken时，scale不能为FLOAT
        OP_TILING_CHECK(
            (inputParams_.cDtype == DT_INT8 || inputParams_.cDtype == DT_FLOAT16) &&
                inputParams_.scaleDtype == ge::DT_FLOAT,
            CUBE_INNER_ERR_REPORT(
                inputParams_.opName,
                "When output dtype is int8 or float16 without pertokenScale, scale dtype should not be float"),
            return false);
    }
    return true;
}

bool QuantBatchMatmulV3Tiling::CheckDtypeOnOnlyL0c2outForX1NZ() const
{
    // 当y为int8时，x1必须为ND
    auto x1Desc = context_->GetInputDesc(X1_INDEX);
    auto x1Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x1Desc->GetStorageFormat()));
    OP_TILING_CHECK(inputParams_.cDtype == DT_INT8 && x1Format != Format::FORMAT_ND,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "When out dtype is INT8, X1 format should be ND, actual format is %s",
                                          TypeUtils::FormatToSerialString(x1Format).c_str()),
                    return false);
    // 当x2为ND时，x1必须为ND
    auto x2Desc = context_->GetInputDesc(X2_INDEX);
    auto x2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x2Desc->GetStorageFormat()));
    OP_TILING_CHECK(
        x2Format == Format::FORMAT_ND && x1Format != Format::FORMAT_ND,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "When X2 format is ND, X1 format should be ND, actual format is %s",
                              TypeUtils::FormatToSerialString(x1Format).c_str()),
        return false);
    return true;
}

bool QuantBatchMatmulV3Tiling::CheckDtypeOnOnlyL0c2outForUnclassified() const
{
    // dtype 约束条件
    // 当bias为BF16时，y必须为BF16
    OP_TILING_CHECK(context_->GetOptionalInputDesc(BIAS_INDEX) != nullptr && inputParams_.biasDtype == DT_BF16 &&
                        inputParams_.cDtype != DT_BF16,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "When bias dtype is BF16, output dtype should be BF16, actual dtype is %s",
                                          DType2Str(inputParams_.cDtype).c_str()),
                    return false);

    // 当scale为BF16时，y必须为BF16或INT32
    OP_TILING_CHECK(
        inputParams_.scaleDtype == DT_BF16 && !(inputParams_.cDtype == DT_BF16 || inputParams_.cDtype == DT_INT32),
        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                              "When scale dtype is BF16, output dtype should be BF16 or INT32, actual dtype is %s",
                              DType2Str(inputParams_.cDtype).c_str()),
        return false);
    // 当y为BF16时，scale必须为BF16或FLOAT
    OP_TILING_CHECK(
        inputParams_.cDtype == DT_BF16 && !(inputParams_.scaleDtype == DT_BF16 || inputParams_.scaleDtype == DT_FLOAT),
        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                              "When out dtype is BF16, scale dtype should be BF16 or FLOAT, actual dtype is %s",
                              DType2Str(inputParams_.scaleDtype).c_str()),
        return false);
    // 当y为INT8时，scale必须为INT64或UINT64
    OP_TILING_CHECK(
        inputParams_.cDtype == DT_INT8 &&
            !(inputParams_.scaleDtype == DT_UINT64 || inputParams_.scaleDtype == DT_INT64),
        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                              "When out dtype is INT8, scale dtype should be UINT64 or INT64, actual dtype is %s",
                              DType2Str(inputParams_.scaleDtype).c_str()),
        return false);

    // 当y为INT32时，bias必须为INT32
    OP_TILING_CHECK(inputParams_.cDtype == DT_INT32 && context_->GetOptionalInputDesc(BIAS_INDEX) != nullptr &&
                        inputParams_.biasDtype != DT_INT32,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "When out dtype is INT32, bias dtype should be INT32, actual dtype is %s",
                                          DType2Str(inputParams_.biasDtype).c_str()),
                    return false);
    // 当y为INT32时，scale必须为FLOAT或BF16
    OP_TILING_CHECK(
        inputParams_.cDtype == DT_INT32 && !(inputParams_.scaleDtype == DT_FLOAT || inputParams_.scaleDtype == DT_BF16),
        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                              "When out dtype is INT32, scale dtype should be FLOAT or BF16, actual dtype is %s",
                              DType2Str(inputParams_.scaleDtype).c_str()),
        return false);
    return true;
}

bool QuantBatchMatmulV3Tiling::CheckDtypeOnOnlyL0c2out() const
{
    // 对可选类型进行校验
    if (!CheckDtypeOnOnlyL0c2outForSupportedList()) {
        return false;
    }
    // 对A4W4场景/非A4W4场景进行校验
    if (inputParams_.aDtype == DT_INT4) {
        if (!CheckDtypeOnOnlyL0c2outForA4W4()) {
            return false;
        }
    } else {
        if (!CheckDtypeOnOnlyL0c2outForX1NZ() || !CheckDtypeOnOnlyL0c2outForUnclassified() || !CheckDtypeOnOnlyL0c2outForPertoken()) {
            return false;
        }
    }
    return true;
}

bool QuantBatchMatmulV3Tiling::AnalyzeDtype()
{
    inputParams_.aDtype = context_->GetInputDesc(X1_INDEX)->GetDataType();
    auto x2Desc = context_->GetInputDesc(X2_INDEX);
    inputParams_.bDtype = x2Desc->GetDataType();
    inputParams_.scaleDtype = context_->GetInputDesc(SCALE_INDEX)->GetDataType();
    auto pertokenScaleDesc = context_->GetOptionalInputDesc(PERTOKEN_SCALE_INDEX);
    auto biasDesc = context_->GetOptionalInputDesc(BIAS_INDEX);
    inputParams_.biasDtype = biasDesc != nullptr ? biasDesc->GetDataType() : ge::DT_INT32;
    inputParams_.cDtype = context_->GetOutputDesc(0)->GetDataType();
    isUbQuant_ = inputParams_.cDtype == ge::DT_BF16 || pertokenScaleDesc != nullptr;
    SetFormat();

    //无芯片差异的公共校验
    OP_TILING_CHECK(
        inputParams_.aDtype != inputParams_.bDtype,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Input dtype of x1 and x2 must be same, actual dtype are %s and %s",
                              DType2Str(inputParams_.aDtype).c_str(),
                              DType2Str(inputParams_.bDtype).c_str()),
        return false);

    //区分芯片校验
    if (!optiling::PlatformInfo::GetInstance().support_l0c2out() && !CheckDtypeOnOnlyL0c2ub()) {
        return false;
    } else if (optiling::PlatformInfo::GetInstance().support_l0c2out() && !CheckDtypeOnOnlyL0c2out()) {
        return false;
    }
    return true;
}

uint64_t QuantBatchMatmulV3Tiling::GetBatchSize(const gert::Shape &shape) const
{
    uint64_t batch = 1;
    auto shapeLen = shape.GetDimNum();
    for (size_t i = LAST_BATCH_DIM_INDEX; i <= shape.GetDimNum(); i++) {
        batch = batch * shape.GetDim(shapeLen - i);
    }
    return batch;
}

bool QuantBatchMatmulV3Tiling::InferOutBatchDim(const gert::Shape &x1Shape, const gert::Shape &x2Shape)
{
    inputParams_.batchC = 1;
    auto x1DimNum = x1Shape.GetDimNum();
    auto x2DimNum = x2Shape.GetDimNum();
    auto outDimNum = std::max(x1DimNum, x2DimNum);
    const gert::Shape &shapeLong = x1DimNum > x2DimNum ? x1Shape : x2Shape;
    const gert::Shape &shapeShort = x1DimNum > x2DimNum ? x2Shape : x1Shape;
    size_t validOffset = outDimNum - std::min(x1DimNum, x2DimNum);
    for (size_t i = 0; i < outDimNum - LAST_SECOND_DIM_INDEX; i++) {
        auto shortDim = i < validOffset ? 1 : shapeShort.GetDim(i - validOffset);
        auto longDim = shapeLong.GetDim(i);
        if (shortDim > 1 && longDim > 1 && shortDim != longDim) {
            return false;
        }
        inputParams_.batchC = inputParams_.batchC * static_cast<uint64_t>(std::max(shortDim, longDim));
    }
    return true;
}

int8_t QuantBatchMatmulV3Tiling::CheckFusionBatchA(const gert::Shape &x1Shape, const gert::Shape &x2Shape,
                                                 const gert::Shape &biasShape, uint64_t fusedDimValue) const {
    auto x1ShapeLen = x1Shape.GetDimNum();
    auto x2ShapeLen = x2Shape.GetDimNum();
    auto x1BatchLen = x1ShapeLen - 2;
    if (inputParams_.isPertoken) { // pertoken dim equals to original m dim, cannot fuse
        return 0;
    }
    if (inputParams_.hasBias && biasShape.GetDimNum() > 1) { // 3 dim batch need original batch dim value, cannot fuse
        return 0;
    }
    if (fusedDimValue > static_cast<int64_t>(INT32_MAX)) { // exceed axis limit, cannot fuse
        if (inputParams_.aDtype == DT_INT4) {
            return OUTPUT_INFER_FAIL;
        }
        return 0;
    }
    // only fusion when x2 shape length is 2
    if (x2ShapeLen == 2 && inputParams_.transA == false && x1BatchLen >= 1) {
        OP_LOGD("QuantBatchMatmulV3", "CheckFusionBatchA success, start fusion batch A to m dim");
        return OUTPUT_INFER_SUCCESS;
    }
    return 0;
}

void QuantBatchMatmulV3Tiling::DoBatchFusion(uint64_t fusedDimValue)
{
    inputParams_.mSize = fusedDimValue;
    inputParams_.batchC = 1;
    inputParams_.batchA = 1;
}

bool QuantBatchMatmulV3Tiling::CheckShapeInRangeForMandtoryInputs(size_t x1ShapeLen, size_t x2ShapeLen,
                                                                  size_t scaleShapeLen) const
{
    OP_TILING_CHECK(x1ShapeLen < MIN_DIM_NUM_ND || x2ShapeLen < MIN_DIM_NUM_ND,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "x1 len and x2 len should be greater than 1, but x1 len: %zu, x2 len: %zu",
                                          x1ShapeLen, x2ShapeLen), return false);
    OP_TILING_CHECK(x1ShapeLen > MAX_DIM_NUM_ND || x2ShapeLen > MAX_DIM_NUM_ND,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "x1 len and x2 len should be less than 7, but x1 len: %zu, x2 len: %zu",
                                          x1ShapeLen, x2ShapeLen), return false);
    OP_TILING_CHECK(scaleShapeLen != 1,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "only support for scale dim equals to 1, but the actually value is : %zu",
                                          scaleShapeLen), return false);

    if (inputParams_.aDtype == ge::DT_INT4 && !inputParams_.isPertoken) {
        OP_TILING_CHECK(x2ShapeLen != A4W4_X2_LEN,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                              "if input dtype is int4 and without pertoken scale, the dim of x2 must be 2."), return false);
    }
    return true;
}

bool QuantBatchMatmulV3Tiling::CheckShapeInRangeForOptionalInputs(const gert::StorageShape *biasShape,
                                                                  const gert::StorageShape *pertokenShape) const
{
    if (biasShape != nullptr) {
        auto biasDimNum = biasShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(!(biasDimNum == 1 || biasDimNum == BIAS_THREE_DIM),
                        CUBE_INNER_ERR_REPORT(inputParams_.opName, "bias dim should equal to 1 or 3, but it is %zu",
                                              biasDimNum), return false);
    }
    if (pertokenShape != nullptr) {
        auto pertokenDimNum = pertokenShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(pertokenDimNum != 1,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName, "pertoken dim should equal to 1, but it is %zu",
                                              pertokenDimNum), return false);
    }
    return true;
}

bool QuantBatchMatmulV3Tiling::BiasShapeCheck(const gert::Shape &biasShape) const
{
    auto biasDimNum = biasShape.GetDimNum();
    if (biasDimNum == 1) {
        OP_TILING_CHECK(static_cast<uint64_t>(biasShape.GetDim(0)) != inputParams_.nSize,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                              "bias dim value should equal n, but it is %zu while n is %lu",
                                              biasShape.GetDim(0), inputParams_.nSize),
                                              return false);
    }
    // 3 dim bias case
    if (biasDimNum == BIAS_THREE_DIM) {
        auto biasFirstDim = static_cast<uint64_t>(biasShape.GetDim(0)); // using index 0 to get bias first dim value
        auto biasSecondDim = static_cast<uint64_t>(biasShape.GetDim(1)); // using index 1 to get bias second dim value
        auto biasThirdDim = static_cast<uint64_t>(biasShape.GetDim(2)); // using index 2 to get bias third dim value
        OP_TILING_CHECK(biasFirstDim != inputParams_.batchC,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                              "bias 1st dim value should equal to batchC, but it is %zu",
                                              biasFirstDim), return false);
        OP_TILING_CHECK(biasSecondDim != 1,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                              "bias 2nd dim value should equal to 1, but it is %zu",
                                              biasSecondDim), return false);
        OP_TILING_CHECK(biasThirdDim != inputParams_.nSize,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                              "bias 3rd dim value should equal to inputParams_.nSize, but it is %zu \
                                              while n is %lu",
                                              biasThirdDim, inputParams_.nSize), return false);
    }
    return true;
}

bool QuantBatchMatmulV3Tiling::CheckDimValue(const gert::Shape & scaleShape, const gert::StorageShape *biasShape,
                                             const gert::StorageShape *pertokenShape,
                                             const std::vector<int64_t> &dimValueOfMKN) const
{
    auto x1Inner = dimValueOfMKN[0]; // using index 0 to get x1Inner
    auto x2Inner = dimValueOfMKN[2]; // using index 2 to get x2Inner
    auto x2Outer = dimValueOfMKN[3]; // using index 3 to get x2Outer
    auto kBSize = static_cast<uint64_t>(inputParams_.transB ? x2Inner : x2Outer);
    auto scaleDimValue = static_cast<uint64_t>(scaleShape.GetDim(0));
    OP_TILING_CHECK(inputParams_.kSize != kBSize,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "kA[%lu] is not equal to kB[%lu]",
                                          inputParams_.kSize, kBSize),
                    return false);
    OP_TILING_CHECK(scaleDimValue != 1 && scaleDimValue != inputParams_.nSize,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "scale dim value should equal to 1 or n, but it is %zu",
                                          scaleDimValue),
                    return false);
    if (biasShape != nullptr && !BiasShapeCheck(biasShape->GetStorageShape())) {
        return false;
    }
    if (inputParams_.isPertoken && !isTilingOut_) {
        auto pertoken = pertokenShape->GetStorageShape();
        OP_TILING_CHECK(static_cast<uint64_t>(pertoken.GetDim(0)) != inputParams_.mSize,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                              "pertoken shape should be equal to m[%lu] but atcual is [%zu]",
                                              inputParams_.mSize, pertoken.GetDim(0)), return false);
    }
    if (inputParams_.aDtype == ge::DT_INT4) {
        // remainder by 2 to check if it is a even number
        OP_TILING_CHECK(x1Inner < 0 || x1Inner % 2 != 0 || x2Inner < 0 || x2Inner % 2 != 0,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName, "if input dtype is int4, \
                                              last axis of input x1 and x2 has to be a positive even number, \
                                              but atcually last axis of x1 is [%lu], last axis of x2 is [%lu].",
                                              x1Inner, x2Inner), return false);
    }
    return true;
}

bool QuantBatchMatmulV3Tiling::CheckShape(const std::vector<gert::Shape *> &mandtoryShape,
                                          const gert::StorageShape *biasShape,
                                          const gert::StorageShape *pertokenShape,
                                          const std::vector<int64_t> &dimValueOfMKN) const
{
    auto x1Shape = *mandtoryShape[0]; // using index 0 to get x1Shape
    auto x2Shape = *mandtoryShape[1]; // using index 1 to get x2Shape
    auto scaleShape = *mandtoryShape[2]; // using index 2 to get scaleShape
    if (!CheckShapeInRangeForOptionalInputs(biasShape, pertokenShape)){
        return false;
    }
    if (!CheckDimValue(scaleShape, biasShape, pertokenShape, dimValueOfMKN)){
        return false;
    }
    if (!CheckShapeInBoundary(x1Shape, X1_INDEX) || !CheckShapeInBoundary(x2Shape, X2_INDEX)) {
        return false;
    }
    return true;
}

// Notice: 修改此函数可能会影响mc2功能，使用isTilingOut_变量判断是否为mc2场景
bool QuantBatchMatmulV3Tiling::AnalyzeInputs()
{
    auto x1Shape = GetX1Shape(X1_INDEX);
    auto x2Shape = GetX2Shape(X2_INDEX);
    auto scaleShape = GetScaleShape(SCALE_INDEX);
    auto pertokenShape = GetPertokenShape(PERTOKEN_SCALE_INDEX);
    inputParams_.isPertoken = pertokenShape != nullptr;
    auto biasShape = GetBiasShape(BIAS_INDEX);
    inputParams_.hasBias = biasShape != nullptr;
    inputParams_.batchBias = inputParams_.hasBias ? GetBatchSize(biasShape->GetStorageShape()) : 1;
    auto x1ShapeLen = x1Shape.GetDimNum();
    auto x2ShapeLen = x2Shape.GetDimNum();
    auto scaleShapeLen = scaleShape.GetDimNum();
    if (!isTilingOut_ && !CheckShapeInRangeForMandtoryInputs(x1ShapeLen, x2ShapeLen, scaleShapeLen)){
        return false;
    }
    auto x1Inner = x1Shape.GetDim(x1ShapeLen - LAST_FIRST_DIM_INDEX);
    auto x1Outer = x1Shape.GetDim(x1ShapeLen - LAST_SECOND_DIM_INDEX);
    auto x2Inner = x2Shape.GetDim(x2ShapeLen - LAST_FIRST_DIM_INDEX);
    auto x2Outer = x2Shape.GetDim(x2ShapeLen - LAST_SECOND_DIM_INDEX);
    const std::vector<int64_t> dimValueOfMKN = {x1Inner, x1Outer, x2Inner, x2Outer};
    inputParams_.mSize = static_cast<uint64_t>(inputParams_.transA ? x1Inner : x1Outer);
    inputParams_.kSize = static_cast<uint64_t>(inputParams_.transA ? x1Outer : x1Inner);
    inputParams_.nSize = static_cast<uint64_t>(inputParams_.transB ? x2Outer : x2Inner);
    const std::vector<gert::Shape *> mandtoryShape = {&x1Shape, &x2Shape, &scaleShape};

    inputParams_.batchA = GetBatchSize(x1Shape);
    inputParams_.batchB = GetBatchSize(x2Shape);
    AnalyzeBatchInfo(context_->GetInputShape(0)->GetOriginShape(), context_->GetInputShape(1)->GetOriginShape());
    OP_TILING_CHECK(!InferOutBatchDim(x1Shape, x2Shape),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "batch dim can not be broadcasted"), return false);
    if (!isTilingOut_ && !CheckShape(mandtoryShape, biasShape, pertokenShape, dimValueOfMKN)) {
        return false;
    }
    inputParams_.isPerTensor = scaleShape.GetDim(0) == 1;
    OP_TILING_CHECK(!CheckOutputShapeAvailable(),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "multiple of output shape dims should be in boundary of INT64_MAX"),
                                          return false);
    uint64_t fusedDimValue = inputParams_.mSize * inputParams_.batchA;
    int8_t resultCheckFusionBatchA = CheckFusionBatchA(x1Shape, x2Shape, biasShape->GetStorageShape(), fusedDimValue);
    OP_TILING_CHECK(resultCheckFusionBatchA == OUTPUT_INFER_FAIL,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                    "The fused M [%lu] exceed INT32_MAX [%d] in a4w4 case",
                    fusedDimValue, INT32_MAX), return false);
    if (resultCheckFusionBatchA == OUTPUT_INFER_SUCCESS) {
        DoBatchFusion(fusedDimValue);
    }
    OP_LOGD("QuantBatchMatmulV3", "batchA: %lu, batchB: %lu, batchC: %lu, isPerTensor: %s, isPertoken: %s",
            inputParams_.batchA, inputParams_.batchB, inputParams_.batchC, inputParams_.isPerTensor ? "true" : "false",
            inputParams_.isPertoken ? "true" : "false");
    return true;
}

bool QuantBatchMatmulV3Tiling::CheckShapeInBoundary(const gert::Shape &shape, uint32_t shapeIdx) const
{
    int64_t mul = 1;
    int64_t mulBound = 1;
    const char* dimName = shapeIdx == X1_INDEX ? "x1" : "x2";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        int64_t curDim = shape.GetDim(i);

        OP_TILING_CHECK(i == shape.GetDimNum() - LAST_FIRST_DIM_INDEX && curDim > LAST_AXIS_LIMIT,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                              "last dim of %s should not be larger than 65535 but atcual is %ld",
                                              dimName, curDim),
                        return false);

        OP_TILING_CHECK(curDim <= 0 || curDim > static_cast<int64_t>(INT32_MAX),
                        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                              "shape must be within the range [1, %d] but atcual %zu dim of %s is %ld",
                                              INT32_MAX, i, dimName, curDim),
                        return false);

        mulBound = curDim * mul;
        OP_TILING_CHECK(mulBound / curDim != mul,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                              "multiple of %s shape dims should be in boundary of INT64_MAX",
                                              dimName),
                        return false);
        mul = mulBound;
    }
    return true;
}

bool QuantBatchMatmulV3Tiling::CheckOutputShapeAvailable() const
{
    int64_t m = static_cast<int64_t>(inputParams_.mSize);
    int64_t n = static_cast<int64_t>(inputParams_.nSize);
    int64_t b = static_cast<int64_t>(inputParams_.batchC);
    int64_t mul = m * n;
    if (mul / m != n) {
        return false;
    }
    mul *= b;
    return mul / b == m * n;
}

bool QuantBatchMatmulV3Tiling::IsCapable() { return true; }

void QuantBatchMatmulV3Tiling::ProcessMSmall()
{
    // mix增量优化模板：只优化量化MM的白名单增量用例，不支持batch。后续泛化功能性能。
    std::pair<uint64_t, uint64_t> dimPair{inputParams_.kSize, inputParams_.nSize};
    auto it = std::find(WHITE_LIST_X2_KN.begin(), WHITE_LIST_X2_KN.end(), dimPair);
    bool isInWhiteList = it != WHITE_LIST_X2_KN.end();
    bool isAllMix = isInWhiteList && isUbQuant_ && inputParams_.batchC == 1;

    uint64_t baseM = static_cast<uint64_t>(tbeTiling_.m_l0) * BLOCK_CUBE;
    uint64_t baseN = static_cast<uint64_t>(tbeTiling_.n_l0) * BLOCK_CUBE;
    uint64_t needWorkspace =
        ops::CeilAlign(ops::CeilDiv(inputParams_.nSize, static_cast<uint64_t>(tbeTiling_.n_dim)), baseN) * baseM *
        sizeof(int32_t) * aicoreParams_.aicNum;
    // 控制m方向无循环的进入增量优化模板且workspace不能超过50M限制。
    bool isPertoken = (needWorkspace < WORKSPACE_LIMIT) && inputParams_.isPertoken;
    bool isDecode = inputParams_.mSize <= baseM;
    isBf16Opt_ = isDecode && (isAllMix || isPertoken) && !isTilingOut_;
    if (isBf16Opt_ && inputParams_.isPertoken) {
        // cv并行，每次base块
        tbeTiling_.m_al1 = 1;
        tbeTiling_.n_bl1 = 1;
    }
}

ge::graphStatus QuantBatchMatmulV3Tiling::DoOpTiling()
{
    isUbQuant_ = inputParams_.cDtype == ge::DT_BF16 || inputParams_.isPertoken;
    // 需要给aicoreParams_ 和libApiWorkSpaceSize赋值
    OP_LOGE_IF(!SetPlatformInfoForTiling(), ge::GRAPH_FAILED, inputParams_.opName, "SetPlatformInfoForTiling fail");
    if (!GetTbeTiling()) {
        OP_LOGE(inputParams_.opName, "GetTbeTiling fail");
        return ge::GRAPH_FAILED;
    }
    PrintTbeTiling();
    ProcessMSmall();
    tilingData_.params.set_batchA(inputParams_.batchA);
    tilingData_.params.set_batchB(inputParams_.batchB);
    tilingData_.params.set_batchC(inputParams_.batchC);
    tilingData_.params.set_singleCoreBatch(
        ops::CeilDiv(inputParams_.batchC, static_cast<uint64_t>(tbeTiling_.batch_dim)));
    tilingData_.params.set_biasThreeDim(static_cast<uint32_t>(inputParams_.batchBias > 1));
    tilingData_.params.set_isPerTensor(static_cast<uint32_t>(inputParams_.isPerTensor));
    tilingData_.params.set_isPertoken(static_cast<uint32_t>(inputParams_.isPertoken));
    tilingData_.params.set_biasDtype(static_cast<uint32_t>(inputParams_.biasDtype));
    if (isUbQuant_) {
        return CalcUbTiling();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulV3Tiling::InitTilingData(matmul_tiling::MatmulApiTilingBase &mm, bool fallback)
{
    optiling::PlatformInfo& plaformInstance = optiling::PlatformInfo::GetInstance();
    auto aFormat = inputParams_.aFormat == ge::FORMAT_ND ? matmul_tiling::CubeFormat::ND : matmul_tiling::CubeFormat::NZ;
    auto bFormat = inputParams_.bFormat == ge::FORMAT_ND ? matmul_tiling::CubeFormat::ND : matmul_tiling::CubeFormat::NZ;
    auto cFormat = inputParams_.cFormat == ge::FORMAT_ND ? matmul_tiling::CubeFormat::ND : matmul_tiling::CubeFormat::NZ;
    auto aDtype = GetMatmulTilingDtype(inputParams_.aDtype);
    auto bDtype = GetMatmulTilingDtype(inputParams_.bDtype);
    auto cDtype = GetMatmulTilingDtype(inputParams_.cDtype);
    OP_LOGE_IF(aDtype == matmul_tiling::DataType::DT_UNDEFINED, ge::GRAPH_FAILED, inputParams_.opName,
               "invalid A dtype %s", DType2Str(inputParams_.aDtype).c_str());
    OP_LOGE_IF(bDtype == matmul_tiling::DataType::DT_UNDEFINED, ge::GRAPH_FAILED, inputParams_.opName,
               "invalid B dtype %s", DType2Str(inputParams_.bDtype).c_str());
    OP_LOGE_IF(cDtype == matmul_tiling::DataType::DT_UNDEFINED, ge::GRAPH_FAILED, inputParams_.opName,
               "invalid C dtype %s", DType2Str(inputParams_.cDtype).c_str());
    mm.SetAType(matmul_tiling::TPosition::GM, aFormat, aDtype, inputParams_.transA);
    mm.SetBType(matmul_tiling::TPosition::GM, bFormat, bDtype, inputParams_.transB);
    mm.SetCType(matmul_tiling::TPosition::GM, cFormat, cDtype);
    mm.SetShape(inputParams_.mSize, inputParams_.nSize, inputParams_.kSize);
    if (fallback) {
        mm.SetShape(ops::CeilDiv<uint64_t>(inputParams_.mSize, tbeTiling_.m_dim),
                    ops::CeilDiv<uint64_t>(inputParams_.nSize, tbeTiling_.n_dim),
                    ops::CeilDiv<uint64_t>(inputParams_.kSize, tbeTiling_.k_dim));
        mm.SetMatmulConfigParams(1, false, matmul_tiling::ScheduleType::INNER_PRODUCT, matmul_tiling::MatrixTraverse::NOSET, true);
    }
    mm.SetOrgShape(inputParams_.mSize, inputParams_.nSize, inputParams_.kSize);
    if (inputParams_.hasBias) {
        mm.SetBias(true);
        auto biasDtype = GetMatmulTilingDtype(inputParams_.biasDtype);
        OP_LOGE_IF(biasDtype == matmul_tiling::DataType::DT_UNDEFINED, ge::GRAPH_FAILED, inputParams_.opName,
                "invalid bias dtype %s", DType2Str(inputParams_.biasDtype).c_str());
        mm.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, biasDtype);
    }
    mm.SetBufferSpace(plaformInstance.l1_size, plaformInstance.l0c_size, plaformInstance.ub_size);
    if (mm.GetTiling(tilingData_.matmulTiling) == -1) {
        OP_LOGE(inputParams_.opName, "Quant MatmulV3 Get Tiling Failed!");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void QuantBatchMatmulV3Tiling::ConstructCacheParams(BatchmatmulCompileParas &compileParams,
                                                    BatchmatmulRunParas &runParams) const
{
    compileParams.binary_mode_flag = true;
    compileParams.bias_flag = inputParams_.hasBias && (inputParams_.biasDtype == ge::DT_INT32);
    compileParams.pattern_flag = optiling::PlatformInfo::GetInstance().support_l0c2out();
    compileParams.zero_flag = false;
    compileParams.aub_double_num = 1;
    compileParams.bub_double_num = 1;

    runParams.trans_a_flag = inputParams_.transA;
    runParams.trans_b_flag = inputParams_.transB;
    // 知识库的匹配用真实的format a和format b, 后面tiling计算全部使用FORMAT_ND
    runParams.format_a_nd = (inputParams_.aFormat == ge::FORMAT_ND);
    runParams.format_b_nd = (inputParams_.bFormat == ge::FORMAT_ND);
    runParams.format_out_nd = true;
    runParams.format_a = inputParams_.aFormat;
    runParams.format_b = inputParams_.bFormat;
    runParams.format_out = ge::FORMAT_ND;
    runParams.reserved_bool = !inputParams_.isPerTensor;
    runParams.nd_flag = runParams.format_a_nd && runParams.format_b_nd;
    runParams.use_pre_ub = runParams.nd_flag && !optiling::PlatformInfo::GetInstance().support_l0c2out();
    runParams.weight_nz_flag = !runParams.format_b_nd;
    runParams.batch_a1 = inputParams_.batchA1;
    runParams.batch_a2 = inputParams_.batchA2;
    runParams.batch_a3 = inputParams_.batchA3;
    runParams.batch_a4 = inputParams_.batchA4;
    runParams.batch_b1 = inputParams_.batchB1;
    runParams.batch_b2 = inputParams_.batchB2;
    runParams.batch_b3 = inputParams_.batchB3;
    runParams.batch_b4 = inputParams_.batchB4;

    runParams.b_have_batch = inputParams_.batchB != 1 && inputParams_.batchC > 1;
    runParams.is_batch_matmul_mode = inputParams_.batchC > 1;
    runParams.is_batch_matmul_op = inputParams_.batchC > 1;
    bool alignedMKN = inputParams_.mSize % BLOCK_CUBE == 0 && inputParams_.kSize % INT_REDUCE_FACTOR == 0 &&
                      inputParams_.nSize % BLOCK_CUBE == 0;
    runParams.used_aligned_pattern = alignedMKN && runParams.nd_flag;
    runParams.bias_flag = inputParams_.hasBias && (inputParams_.biasDtype == ge::DT_INT32);
    runParams.pattern_flag = compileParams.pattern_flag;
    runParams.unaligned_flag = !alignedMKN;
    runParams.zero_flag = compileParams.zero_flag;
    runParams.hf32_flag = 0;
    runParams.dtype_a = static_cast<int32_t>(inputParams_.aDtype);
    runParams.dtype_b = static_cast<int32_t>(inputParams_.bDtype);
    // scale为必选输入，因scale导致mix和纯cube场景的baseN可能不同，需要区分
    // 后面tiling计算全部使用fp16，特别的：int32场景下要用int32匹配知识库，用fp16去计算tiling
    runParams.dtype_out = static_cast<int32_t>(inputParams_.cDtype);
    runParams.dtype_bias = ge::GetSizeByDataType(inputParams_.biasDtype);
    runParams.m = ops::CeilDiv(inputParams_.mSize, static_cast<uint64_t>(BLOCK_CUBE));
    runParams.k = ops::CeilDiv(inputParams_.kSize, INT_REDUCE_FACTOR);
    runParams.n = ops::CeilDiv(inputParams_.nSize, static_cast<uint64_t>(BLOCK_CUBE));
    runParams.batch = static_cast<int64_t>(inputParams_.batchC);
    runParams.ori_shape_m = inputParams_.mSize;
    runParams.ori_shape_k = inputParams_.kSize;
    runParams.ori_shape_n = inputParams_.nSize;
    runParams.m_quant_check = inputParams_.transA;
    runParams.n_quant_check = !inputParams_.transB;
    runParams.bias_dtype = inputParams_.biasDtype;
    runParams.vector_pre_conv_mode = !inputParams_.isPerTensor && !isUbQuant_;
    runParams.is_quant_batch_matmul_v3 = true;
    runParams.is_pertoken = inputParams_.isPertoken;
    ModifyCacheParams(runParams);
}

// 当输入的M轴或N轴dim大小16对齐后超过INT32_MAX
// 需要在传入TBE Tiling前对batch轴解除多核的绑定
// 使得M/N轴上能有更多的核数参与tiling
// 避免singleCoreM/N超过INT32_MAX
void QuantBatchMatmulV3Tiling::ModifyCacheParams(BatchmatmulRunParas &runParams) const
{
    if (runParams.m * static_cast<int64_t>(BLOCK_CUBE) > static_cast<int64_t>(INT32_MAX) ||
        runParams.n * static_cast<int64_t>(BLOCK_CUBE) > static_cast<int64_t>(INT32_MAX)) {
        runParams.batch = 1;
        runParams.batch_a1 = 1;
        runParams.batch_a2 = 1;
        runParams.batch_a3 = 1;
        runParams.batch_a4 = 1;
        runParams.batch_b1 = 1;
        runParams.batch_b2 = 1;
        runParams.batch_b3 = 1;
        runParams.batch_b4 = 1;
    }
}

ge::graphStatus QuantBatchMatmulV3Tiling::DoLibApiTiling()
{
    OP_TILING_CHECK(!SetMatmulTilingFromTbeTiling(),
                CUBE_INNER_ERR_REPORT(inputParams_.opName, "failed to get tbe tiling"), return ge::GRAPH_FAILED);
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

/**
 * 1.原则：同一版本修改tilingKey需tiling和kernel匹配
 * 2.mc2涉及tilingkey的文件(目前mc2调用量化MM的tilingKey跟随该设置方法)有：
 *  2.1 matmul_all_reduce_add_rms_norm.cpp
 *  2.2 inplace_matmul_all_reduce_add_rms_norm.cpp
 *  2.3 matmul_all_reduce.cpp
 *  2.4 quant_matmul_all_reduce_tiling.h
 * 3.如何搜索：tiling文件搜索：quant_batch_matmul_v3_tiling.h, matmul_all_reduce_tiling.h
 */
uint64_t QuantBatchMatmulV3Tiling::GetTilingKey(bool isBasicTiling) const
{
    // 新增特性应往后添加,相同特性应在同bit位
    if (inputParams_.cDtype == ge::DT_BF16) {
        return RecursiveSum(inputParams_.transB, inputParams_.transA, isBasicTiling,
                            isBf16Opt_, inputParams_.isPertoken, false);
    } else {
        return RecursiveSum(inputParams_.transB, inputParams_.transA, isBasicTiling,
                            isBf16Opt_, inputParams_.isPertoken, NeedAtomiClean());
    }
}

uint64_t QuantBatchMatmulV3Tiling::GetTilingKey() const
{
    return GetTilingKey(false);
}

ge::graphStatus QuantBatchMatmulV3Tiling::GetWorkspaceSize()
{
    workspaceSize_ = inputParams_.libApiWorkSpaceSize;
    if (isUbQuant_) {
        auto ret = GetUbDequantExtreSpace();
        OP_TILING_CHECK(!ret, CUBE_CALL_ERR_REPORT(inputParams_.opName, "GetUbDequantExtreSpace is failed"),
                        return ge::GRAPH_FAILED);
        workspaceSize_ += inputParams_.bf16ExtreWorkSpaceSize;
    }

    if (NeedAtomiClean() && !optiling::PlatformInfo::GetInstance().support_l0c2out()) {
        workspaceSize_ += aicoreParams_.aicNum * ONE_BLK_SIZE;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulV3Tiling::PostTiling()
{
    OP_LOGD(inputParams_.opName, "final tiling data size: %zu", tilingData_.GetDataSize());

    OP_TILING_CHECK(tilingData_.GetDataSize() % sizeof(uint64_t) != 0,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "tiling data size[%zu] is not aligned to 8",
                                          tilingData_.GetDataSize()),
                    return ge::GRAPH_FAILED);
    auto blockDim = tilingData_.matmulTiling.get_usedCoreNum();
    context_->SetBlockDim(blockDim);
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    if (NeedAtomiClean() && optiling::PlatformInfo::GetInstance().support_l0c2out()) {
        context_->SetScheduleMode(1); // 独占全核，设置以后会让所有核空闲以后才启动，有多核同步指令需要做此设置避免影响整网其他算子
    }
    size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    PrintTilingParams();
    return ge::GRAPH_SUCCESS;
}

bool QuantBatchMatmulV3Tiling::SetPlatformInfoForTiling()
{
    // mc2和qbmm把compileInfo都赋值给compileInfo_，后续硬件信息可以直接从compileInfo_中获取
    if (!compileInfoInit_) {
        auto mmCompileInfo =  reinterpret_cast<const QuantBatchMatmulV3CompileInfo *>(context_->GetCompileInfo());
        OP_TILING_CHECK(mmCompileInfo == nullptr,
                        CUBE_CALL_ERR_REPORT(inputParams_.opName, "quant_batch_matmul_v3_tiling GetCompileInfo is null"),
                        return false);
        compileInfo_ = *mmCompileInfo;
    }
    OP_LOGE_IF(compileInfo_.core_num <= 0, false, inputParams_.opName, "coreNum <= 0");
    aicoreParams_.aicNum = compileInfo_.core_num;
    OP_LOGE_IF(compileInfo_.l2_size <= 0, false, inputParams_.opName, "l2Size <= 0");
    // 纠正L2实际物理大小
    compileInfo_.l2_size =
        compileInfo_.l2_size == L2_FAKE_SIZE * MB_SIZE ? L2_REAL_SIZE * MB_SIZE : compileInfo_.l2_size;
    inputParams_.libApiWorkSpaceSize = compileInfo_.workspace_num;
    aicoreParams_.ubSize = compileInfo_.ub_size;
    aicoreParams_.l1Size = compileInfo_.l1_size;
    aicoreParams_.l0cSize = compileInfo_.l0c_size;
    aicoreParams_.blockDim = 0;
    return true;
}

bool QuantBatchMatmulV3Tiling::GetTbeTiling()
{
    // 在TilingParse回调函数中初始化的(编译态传过来的compile_info): GemmParseFunc->AnalyzeCompileInfo->AnalyzeExtendInfo
    BatchmatmulCompileParas compileParams;
    BatchmatmulRunParas runParams;
    ConstructCacheParams(compileParams, runParams);

    tbeTiling_.tiling_id = std::numeric_limits<uint64_t>::max();
    return GenTiling("QuantBatchMatmulV3", compileParams, runParams, tbeTiling_, context_);
}

int32_t QuantBatchMatmulV3Tiling::GetIteratorOrder()
{
    const int32_t singleCoreM = tilingData_.matmulTiling.get_singleCoreM();
    const int32_t singleCoreN = tilingData_.matmulTiling.get_singleCoreN();
    const int32_t singleCoreK = tilingData_.matmulTiling.get_singleCoreK();
    int32_t reduceSize = GetShapeWithDataType(ONE_BLK_SIZE, inputParams_.aDtype);
    OP_TILING_CHECK(tbeTiling_.kal1_16 * reduceSize == 0 || tbeTiling_.kbl1_16 == 0,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "kal1(%ld), kbl1(%ld) or reduceSize(%d) is 0.",
                                              tbeTiling_.kal1_16, tbeTiling_.kbl1_16, reduceSize),
                    return -1);
    bool fullkAL1Load = (static_cast<float>(singleCoreK) / (tbeTiling_.kal1_16 * reduceSize)) > 1.0 ? false : true;
    bool fullkBL1Load = (static_cast<float>(singleCoreK) / (tbeTiling_.kbl1_16 * reduceSize)) > 1.0 ? false : true;

    // if KAL1 and KBL1 both can not be full loaded, then select m or n which is no matter
    if (!fullkAL1Load && !fullkBL1Load) {
        return 0;
    } else if (fullkAL1Load && !fullkBL1Load) {  // if KAL1 is full loaded, then select the order N fist
        return 1;
    } else if (!fullkAL1Load && fullkBL1Load) {  // if KBL1 is full loaded, then select the order M fist
        return 0;
    } else {
        // if AL1LoadSize less then BL1LoadSize, then select order N first, vice versa.
        int64_t mLoop =
            ops::CeilDiv(static_cast<int64_t>(singleCoreM), tbeTiling_.m_al1 * tbeTiling_.m_l0 * BLOCK_CUBE);
        int64_t nLoop =
            ops::CeilDiv(static_cast<int64_t>(singleCoreN), tbeTiling_.n_bl1 * tbeTiling_.n_l0 * BLOCK_CUBE);
        int64_t aL1LoadSize = singleCoreM + singleCoreN * mLoop;
        int64_t bL1LoadSize = singleCoreN + singleCoreM * nLoop;
        return aL1LoadSize < bL1LoadSize ? 1 : 0;
    }
}

bool QuantBatchMatmulV3Tiling::SetBlockDimsAndSingleCore(TCubeTiling &mt)
{
    auto mFactor = (inputParams_.transA && inputParams_.mSize >= static_cast<uint64_t>(mt.get_baseM())
                        ? static_cast<uint64_t>(mt.get_baseM())
                        : BLOCK_CUBE);
    auto nFactor = (!inputParams_.transB && inputParams_.nSize >= static_cast<uint64_t>(mt.get_baseN())
                        ? static_cast<uint64_t>(mt.get_baseN())
                        : BLOCK_CUBE);
    auto singleCoreM =
        ops::CeilDiv(ops::CeilDiv(inputParams_.mSize, static_cast<uint64_t>(tbeTiling_.m_dim)), mFactor) * mFactor;
    auto singleCoreN =
        ops::CeilDiv(ops::CeilDiv(inputParams_.nSize, static_cast<uint64_t>(tbeTiling_.n_dim)), nFactor) * nFactor;
    auto singleCoreK = ops::CeilDiv(inputParams_.kSize, static_cast<uint64_t>(tbeTiling_.k_dim));
    singleCoreK =
        tbeTiling_.k_dim == 1 ? singleCoreK : ops::CeilAlign(singleCoreK, static_cast<uint64_t>(ONE_BLK_SIZE));
    OP_TILING_CHECK(singleCoreM > static_cast<uint64_t>(std::numeric_limits<int>::max()),
                    CUBE_INNER_ERR_REPORT(
                        inputParams_.opName,
                        "cache tiling inner error: singleCoreM exceeds the expression range of the int"), return false);

    OP_TILING_CHECK(singleCoreN > static_cast<uint64_t>(std::numeric_limits<int>::max()),
                    CUBE_INNER_ERR_REPORT(
                        inputParams_.opName,
                        "cache tiling inner error: singleCoreN exceeds the expression range of the int"), return false);
    mt.set_singleCoreM(singleCoreM);
    mt.set_singleCoreN(singleCoreN);
    mt.set_singleCoreK(singleCoreK);
    if (isBf16Opt_ && inputParams_.isPertoken) {
        mt.set_singleCoreM(mt.get_baseM());
        mt.set_singleCoreN(mt.get_baseN());
    }
    if (isUbQuant_) {
        tilingData_.params.set_realSingleCoreM(singleCoreM);
        tilingData_.params.set_realSingleCoreN(singleCoreN);
    }

    auto mDim = ops::CeilDiv(inputParams_.mSize, static_cast<uint64_t>(singleCoreM));
    auto nDim = ops::CeilDiv(inputParams_.nSize, static_cast<uint64_t>(singleCoreN));
    auto batchDim = ops::CeilDiv(inputParams_.batchC, static_cast<uint64_t>(tilingData_.params.get_singleCoreBatch()));
    auto kDim = ops::CeilDiv(inputParams_.kSize, static_cast<uint64_t>(tilingData_.matmulTiling.get_singleCoreK()));
    auto blockDim = mDim * nDim * batchDim * kDim;
    OP_TILING_CHECK(blockDim > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "cache tiling inner error: blockDim exceeds the expression range of the int"),
                    return false);
    mt.set_usedCoreNum(std::min(blockDim, aicoreParams_.aicNum));
    return true;
}

bool QuantBatchMatmulV3Tiling::SetMatmulTilingFromTbeTiling()
{
    TCubeTiling &mt = tilingData_.matmulTiling;
    mt.set_M(inputParams_.mSize);
    mt.set_N(inputParams_.nSize);
    mt.set_Ka(inputParams_.kSize);
    mt.set_Kb(inputParams_.kSize);
    mt.set_baseM(tbeTiling_.m_l0 * BLOCK_CUBE);
    mt.set_baseN(tbeTiling_.n_l0 * BLOCK_CUBE);
    OP_TILING_CHECK(!SetBlockDimsAndSingleCore(mt),
                    CUBE_INNER_ERR_REPORT(
                        inputParams_.opName, "Set usedCoreNum or singleCoreM/N faild when m(%lu) and n(%lu).",
                            inputParams_.mSize, inputParams_.nSize),
                    return false);

    int32_t reduceSize = ONE_BLK_SIZE / ge::GetSizeByDataType(inputParams_.aDtype);
    mt.set_baseK(tbeTiling_.k_l0 * reduceSize);

    mt.set_depthA1(ops::CeilDiv(tbeTiling_.kal1_16, tbeTiling_.k_l0) * tbeTiling_.m_al1 * tbeTiling_.db_al1);
    mt.set_depthB1(ops::CeilDiv(tbeTiling_.kbl1_16, tbeTiling_.k_l0) * tbeTiling_.n_bl1 * tbeTiling_.db_bl1);
    mt.set_stepM(tbeTiling_.m_al1);
    mt.set_stepN(tbeTiling_.n_bl1);
    mt.set_stepKa(ops::CeilDiv(tbeTiling_.kal1_16, tbeTiling_.k_l0));
    mt.set_stepKb(ops::CeilDiv(tbeTiling_.kbl1_16, tbeTiling_.k_l0));

    mt.set_isBias(inputParams_.hasBias ? 1 : 0);

    int32_t a1Length = mt.get_baseM() * mt.get_baseK() * ge::GetSizeByDataType(inputParams_.aDtype);
    int32_t b1Length = mt.get_baseN() * mt.get_baseK() * ge::GetSizeByDataType(inputParams_.bDtype);
    int32_t c1Length = mt.get_baseN() * mt.get_baseM() * 4;  // L0C
    mt.set_transLength(std::max(std::max(a1Length, b1Length), c1Length));
    auto iteratorOrder = GetIteratorOrder();
    OP_TILING_CHECK(iteratorOrder < 0,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "get iteratorOrder failed."), return false);
    mt.set_iterateOrder(iteratorOrder);
    mt.set_shareMode(0);
    mt.set_dbL0A(2);  // db switch, 1: off, 2: on
    mt.set_dbL0B(2);  // db switch, 1: off, 2: on
    mt.set_dbL0C(tbeTiling_.db_l0c);

    bool fallback = false;
    OP_TILING_CHECK(!CalcUsedL1AndUBSize(a1Length * mt.get_depthA1(), b1Length * mt.get_depthB1(), fallback),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "cache tiling inner error"), return false);
    if (!fallback) {
        OP_LOGD(inputParams_.opName, "SetMatmulTilingFromTbeTiling !fallback");
        mt.set_shareL0CSize(c1Length);
        mt.set_batchM(1);
        mt.set_batchN(1);
        mt.set_singleBatchM(1);
        mt.set_singleBatchN(1);
    }

    tilingData_.tileL2cacheTiling.set_isBasicTiling(0U); // kernel分支控制
    return true;
}

uint32_t QuantBatchMatmulV3Tiling::GetABankConflictSize() {
    TCubeTiling &mt = tilingData_.matmulTiling;
    uint32_t ret = 0;
    if (inputParams_.transA) {
        bool isABankConflict = ops::CeilDiv<uint64_t>(mt.get_stepM() * mt.get_baseM(), ONE_BLK_SIZE) * 32 % 512 == 0;
        ret = isABankConflict ? mt.get_baseK() * ONE_BLK_SIZE * mt.get_stepKa() : 0;
    } else {
        bool isABankConflict = ops::CeilDiv<uint64_t>(mt.get_stepKa() * mt.get_baseK(), ONE_BLK_SIZE) * 32 % 512 == 0;
        ret = isABankConflict ? mt.get_baseM() * ONE_BLK_SIZE * mt.get_stepM() : 0;
    }
    return ret;
}

bool QuantBatchMatmulV3Tiling::CalcUsedL1AndUBSize(int32_t aL1Size, int32_t bL1Size, bool &fallback)
{
    TCubeTiling &mt = tilingData_.matmulTiling;
    int32_t biasL1Size = (inputParams_.hasBias && (inputParams_.biasDtype == ge::DT_INT32))
                             ? mt.get_baseN() * ge::GetSizeByDataType(inputParams_.biasDtype) * mt.get_stepN()
                             : 0;
    uint32_t ubSize = 0;
    if (!optiling::PlatformInfo::GetInstance().support_l0c2out()) {
        biasL1Size = 0;
        ubSize = static_cast<uint32_t>(aicoreParams_.ubSize);
        // ND/NZ trans tensor, bias and scale tensor can reuse UB, and they use tow buffer in UB at the same time
        // so set transLength as half of UB
        mt.set_transLength(static_cast<int32_t>(ubSize >> 1));

        while (CalcND2NZSpace() > mt.get_transLength()) {
            OP_LOGD(inputParams_.opName, "baseM*baseK*stepKa*stepM > half of UB, decrease stepM");
            mt.set_stepM(mt.get_stepM() - 1);
            mt.set_depthA1(mt.get_stepKa() * mt.get_stepM() * tbeTiling_.db_al1);
            aL1Size = mt.get_baseM() * mt.get_baseK() * ge::GetSizeByDataType(inputParams_.aDtype) * mt.get_depthA1();
        }

        OP_TILING_CHECK(mt.get_stepM() <= 0,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName, "get invalid tiling(stepM <= 0)"), return false);
        uint32_t aBankConflictSize = GetABankConflictSize();
        uint32_t rqdL1Size = aL1Size + bL1Size + aBankConflictSize * tbeTiling_.db_al1;
        fallback = rqdL1Size > optiling::PlatformInfo::GetInstance().l1_size;
        OP_LOGD(inputParams_.opName, "aBankConflictSize: %d", aBankConflictSize);
        OP_LOGD(inputParams_.opName, "rqdL1Size: %d", rqdL1Size);
        auto usedCoreNum = mt.get_usedCoreNum();
        if (fallback) {
            auto &platformInstance = optiling::PlatformInfo::GetInstance();
            matmul_tiling::PlatformInfo platformInfo = {platform_ascendc::SocVersion::ASCEND310P,
                static_cast<uint64_t>(platformInstance.l1_size), static_cast<uint64_t>(platformInstance.l0c_size),
                static_cast<uint64_t>(platformInstance.ub_size), static_cast<uint64_t>(platformInstance.l0a_size),
                static_cast<uint64_t>(platformInstance.l0b_size)};
            matmul_tiling::MatmulApiTiling mm(platformInfo);
            OP_TILING_CHECK(InitTilingData(mm, fallback) != ge::GRAPH_SUCCESS,
                            CUBE_INNER_ERR_REPORT(inputParams_.opName, "init tilingdata (fallback) failed."),
                            return false);
            tilingData_.params.set_ubSize(ubSize);
            mt.set_transLength(static_cast<int32_t>(ubSize >> 1));
            mt.set_usedCoreNum(usedCoreNum);
            return true;
        }
    }

    mt.set_shareL1Size(aL1Size + bL1Size + biasL1Size);
    mt.set_shareUbSize(0);
    tilingData_.params.set_ubSize(ubSize);
    return true;
}

int32_t QuantBatchMatmulV3Tiling::CalcND2NZSpace() const {
    TCubeTiling &mt = tilingData_.matmulTiling;
    auto aDtypeSize = ge::GetSizeByDataType(inputParams_.aDtype);
    int32_t nd2nzSpace = mt.get_baseM() * mt.get_baseK() * mt.get_stepM() * mt.get_stepKa() * aDtypeSize;
    // ub bank confict
    if (ops::CeilAlign(mt.get_baseK()* mt.get_stepKa(), static_cast<int32_t>(ONE_BLK_SIZE)) % BANK_LEN == 0 &&
        ops::CeilDiv(static_cast<uint32_t>(mt.get_baseK()* mt.get_stepKa()), ONE_BLK_SIZE) < ONE_BLK_SIZE) {
        nd2nzSpace += mt.get_baseM() * mt.get_stepM() * aDtypeSize;
    }

    return nd2nzSpace;
}

void QuantBatchMatmulV3Tiling::PrintTilingData()
{
    if (AlogCheckDebugLevel(OP, DLOG_DEBUG) != 1) {
        return;
    }

    optiling::TCubeTiling &tiling = tilingData_.matmulTiling;
    std::stringstream ss;
    ss << " usedCoreNum: " << tiling.get_usedCoreNum() << " M: " << tiling.get_M() << " N: " << tiling.get_N()
       << " Ka: " << tiling.get_Ka() << " Kb: " << tiling.get_Kb() << " singleCoreM: " << tiling.get_singleCoreM()
       << " singleCoreN: " << tiling.get_singleCoreN() << " singleCoreK: " << tiling.get_singleCoreK()
       << " baseM: " << tiling.get_baseM() << " baseN: " << tiling.get_baseN() << " baseK: " << tiling.get_baseK()
       << " depthA1: " << tiling.get_depthA1() << " depthB1: " << tiling.get_depthB1()
       << " stepM: " << tiling.get_stepM() << " stepN: " << tiling.get_stepN() << " stepka: " << tiling.get_stepKa()
       << " stepkb: " << tiling.get_stepKb() << " isBias: " << tiling.get_isBias()
       << " transLength: " << tiling.get_transLength()
       << " iterateOrder: " << ((tiling.get_iterateOrder() == 1) ? "orderM" : "orderN")
       << " shareMode: " << tiling.get_shareMode() << " dbL0A: " << tiling.get_dbL0A()
       << " dbL0B: " << tiling.get_dbL0B() << " dbL0C: " << tiling.get_dbL0C()
       << " usedL1Size: " << tiling.get_shareL1Size() << " usedL0CSize: " << tiling.get_shareL0CSize()
       << " usedUBSize: " << tiling.get_shareUbSize() << " batchM: " << tiling.get_batchM()
       << " batchN: " << tiling.get_batchN() << " singleBatchM: " << tiling.get_singleBatchM()
       << " singleBatchN: " << tiling.get_singleBatchN();
}

void QuantBatchMatmulV3Tiling::PrintTbeTiling()
{
    if (AlogCheckDebugLevel(OP, DLOG_DEBUG) != 1) {
        return;
    }

    optiling::Tiling &tiling = tbeTiling_;
    std::stringstream ss;
    ss << "tiling_id: " << tiling.tiling_id << " n_cub: " << tiling.n_cub << " db_cub: " << tiling.db_cub
       << " m_l0: " << tiling.m_l0 << " k_l0: " << tiling.k_l0 << " n_l0: " << tiling.n_l0
       << " batch_dim: " << tiling.batch_dim << " n_dim: " << tiling.n_dim << " m_dim: " << tiling.m_dim
       << " k_dim: " << tiling.k_dim << " kal1_16: " << tiling.kal1_16 << " kbl1_16: " << tiling.kbl1_16
       << " kal1_factor: " << tiling.kal1_factor << " kbl1_factor: " << tiling.kbl1_factor << " m_al1: " << tiling.m_al1
       << " n_bl1: " << tiling.n_bl1 << " db_al1: " << tiling.db_al1 << " db_bl1: " << tiling.db_bl1
       << " k_aub: " << tiling.k_aub << " m_aub: " << tiling.m_aub << " db_aub: " << tiling.db_aub
       << " k_bub: " << tiling.k_bub << " n_bub: " << tiling.n_bub << " db_bub: " << tiling.db_bub
       << " aub_dim: " << tiling.aub_dim << " bub_dim: " << tiling.bub_dim << " m1_aub: " << tiling.m1_aub
       << " n1_bub: " << tiling.n1_bub << " k1_aub: " << tiling.k1_aub << " k1_bub: " << tiling.k1_bub
       << " m_aub_dim: " << tiling.m_aub_dim << " n_bub_dim: " << tiling.n_bub_dim << " k_aub_dim: " << tiling.k_aub_dim
       << " k_bub_dim: " << tiling.k_bub_dim << " k_org_dim: " << tiling.k_org_dim << " db_l0c: " << tiling.db_l0c
       << " batch_l0: " << tiling.batch_l0 << " batch_aub: " << tiling.batch_aub << " batch_bub: " << tiling.batch_bub
       << " batch_cub: " << tiling.batch_cub << " out_branch_flag: " << tiling.out_branch_flag
       << " bias_flag: " << tiling.bias_flag << " aub_multi_flag: " << tiling.aub_multi_flag
       << " bub_multi_flag: " << tiling.bub_multi_flag << " a_align_value: " << tiling.a_align_value
       << " b_align_value: " << tiling.b_align_value << " aub_align_bound: " << tiling.aub_align_bound
       << " bub_align_bound: " << tiling.bub_align_bound << " min_kl1_cmp_kl0: " << tiling.min_kl1_cmp_kl0
       << " al1_attach_flag: " << tiling.al1_attach_flag << " bl1_attach_flag: " << tiling.bl1_attach_flag
       << " abkl1_attach_flag: " << tiling.abkl1_attach_flag << " l0c_multi_batch: " << tiling.l0c_multi_batch
       << " m_single_core: " << tiling.m_single_core << " n_single_core: " << tiling.n_single_core
       << " flag_cub_solving_bank_conflict: " << tiling.flag_cub_solving_bank_conflict
       << " al1_full_load: " << tiling.al1_full_load << " bl1_full_load: " << tiling.bl1_full_load
       << " hf32_flag: " << tiling.hf32_flag << " zero_flag: " << tiling.zero_flag
       << " datatype_bf16: " << tiling.datatype_bf16 << " deq_scale_var: " << tiling.deq_scale_var;
}

void QuantBatchMatmulV3Tiling::PrintTilingParams() const
{
    if (AlogCheckDebugLevel(OP, DLOG_DEBUG) != 1) {
        return;
    }

    optiling::QuantBatchMatmulV3Params& params = tilingData_.params;
    std::stringstream ss;
    ss << " batchA: " << params.get_batchA() << " batchB: " << params.get_batchB() << " batchC: " << params.get_batchC()
        << " singleCoreBatch: " << params.get_singleCoreBatch() << " isPerTensor: " << params.get_isPerTensor()
        << " isPertoken: " << params.get_isPertoken() << " biasThreeDim: " << params.get_biasThreeDim()
        << " ubCalcM: " << params.get_ubCalcM() << " ubCalcN: " << params.get_ubCalcN()
        << " needUbBuffer: " << params.get_needUbBuffer() << " realSingleCoreM: " << params.get_realSingleCoreM()
        << " realSingleCoreN: " << params.get_realSingleCoreN() << " biasDtype: " << params.get_biasDtype()
        << " ubSize: " << params.get_ubSize();
}

void QuantBatchMatmulV3Tiling::SpiltSingleCore(int32_t &singleCoreM, int32_t &singleCoreN)
{
    // 任意m,n方向无循环，KFC mm计算分区内不会S型计算，可以确定每次计算的起始点
    if (tilingData_.matmulTiling.get_baseM() >= singleCoreM || tilingData_.matmulTiling.get_baseN() >= singleCoreN) {
        return;
    }
    bool spiltM = ops::CeilDiv(singleCoreM, tilingData_.matmulTiling.get_baseM()) <=
                    ops::CeilDiv(singleCoreN, tilingData_.matmulTiling.get_baseN());
    //  spilt singleCore down to baseM/N in one direction
    if (spiltM) {
        tilingData_.matmulTiling.set_singleCoreM(tilingData_.matmulTiling.get_baseM());
        singleCoreM = tilingData_.matmulTiling.get_baseM();
        tbeTiling_.m_al1 = 1;
        tilingData_.matmulTiling.set_depthA1(ops::CeilDiv(tbeTiling_.kal1_16, tbeTiling_.k_l0) * tbeTiling_.m_al1 *
                                                tbeTiling_.db_al1);
        tilingData_.matmulTiling.set_stepM(tbeTiling_.m_al1);
    } else {
        tilingData_.matmulTiling.set_singleCoreN(tilingData_.matmulTiling.get_baseN());
        singleCoreN = tilingData_.matmulTiling.get_baseN();
        tbeTiling_.n_bl1 = 1;
        tilingData_.matmulTiling.set_depthB1(ops::CeilDiv(tbeTiling_.kbl1_16, tbeTiling_.k_l0) * tbeTiling_.n_bl1 *
                                                tbeTiling_.db_bl1);
        tilingData_.matmulTiling.set_stepN(tbeTiling_.n_bl1);
    }
}


void QuantBatchMatmulV3Tiling::SpiltForWorkSpaceLimit(int32_t singleCoreM, int32_t singleCoreN, int32_t blockDim)
{
    int32_t maxSingleCoreM = std::max(singleCoreM, tilingData_.matmulTiling.get_baseM());
    int32_t maxSingleCoreN = std::max(singleCoreN, tilingData_.matmulTiling.get_baseN());
    if (isBf16Opt_ && inputParams_.isPertoken) {
        maxSingleCoreN = ops::CeilAlign(singleCoreN, tilingData_.matmulTiling.get_baseN());
    }
    inputParams_.bf16ExtreWorkSpaceSize = static_cast<uint64_t>(maxSingleCoreM) * maxSingleCoreN *
                                              sizeof(int32_t) * blockDim;
    if (inputParams_.bf16ExtreWorkSpaceSize <= WORKSPACE_LIMIT) {
        return;
    }
    uint32_t singleCoreLimit = WORKSPACE_LIMIT / blockDim;
    uint64_t singleCoreShapeLimit = static_cast<uint64_t>(singleCoreLimit) / sizeof(int32_t);
    // N after M, minimum is baseM/baseN
    uint64_t spiltFactor = ops::CeilDiv(static_cast<uint64_t>(maxSingleCoreM) * maxSingleCoreN, singleCoreShapeLimit);
    uint64_t newSingleM = (ops::CeilDiv(static_cast<uint64_t>(maxSingleCoreM), static_cast<uint64_t>(BLOCK_CUBE))
                            / spiltFactor) * BLOCK_CUBE;
    newSingleM = std::max(newSingleM, static_cast<uint64_t>(tilingData_.matmulTiling.get_baseM()));
    spiltFactor = ops::CeilDiv(static_cast<uint64_t>(newSingleM) * maxSingleCoreN,  singleCoreShapeLimit);
    uint64_t newSingleN = static_cast<uint64_t>
                            (ops::CeilDiv(ops::CeilDiv(maxSingleCoreN, tilingData_.matmulTiling.get_baseN()),
                            static_cast<int32_t>(spiltFactor))) * tilingData_.matmulTiling.get_baseN();
    while (newSingleM * newSingleN > singleCoreShapeLimit) {
        newSingleN -= tilingData_.matmulTiling.get_baseN();
    }
    newSingleN = std::max(newSingleN, static_cast<uint64_t>(tilingData_.matmulTiling.get_baseN()));
    tilingData_.matmulTiling.set_singleCoreM(newSingleM);
    tilingData_.matmulTiling.set_singleCoreN(newSingleN);
    if (static_cast<uint32_t>(tilingData_.matmulTiling.get_baseM() * tbeTiling_.m_al1) > newSingleM) {
        tbeTiling_.m_al1 = 1;
        tilingData_.matmulTiling.set_depthA1(ops::CeilDiv(tbeTiling_.kal1_16, tbeTiling_.k_l0) * tbeTiling_.m_al1 *
                                                tbeTiling_.db_al1);
        tilingData_.matmulTiling.set_stepM(tbeTiling_.m_al1);
    }
    if (static_cast<uint32_t>(tilingData_.matmulTiling.get_baseN() * tbeTiling_.n_bl1) > newSingleN) {
        tbeTiling_.n_bl1 = 1;
        tilingData_.matmulTiling.set_depthB1(ops::CeilDiv(tbeTiling_.kbl1_16, tbeTiling_.k_l0) * tbeTiling_.n_bl1 *
                                                tbeTiling_.db_bl1);

        tilingData_.matmulTiling.set_stepN(tbeTiling_.n_bl1);
    }
    inputParams_.bf16ExtreWorkSpaceSize = newSingleM * newSingleN * sizeof(int32_t) * blockDim;
}

bool QuantBatchMatmulV3Tiling::GetUbDequantExtreSpace()
{
    int32_t singleCoreM = tilingData_.params.get_realSingleCoreM();
    int32_t singleCoreN = tilingData_.params.get_realSingleCoreN();

    // when M, N both have loops in singlecore/base, fixpipe order would be complex, split singlecore to
    // have only one direction with loop for now
    SpiltSingleCore(singleCoreM, singleCoreN);

    SpiltForWorkSpaceLimit(singleCoreM, singleCoreN, static_cast<int32_t>(tilingData_.matmulTiling.get_usedCoreNum()));
    return true;
}

ge::graphStatus QuantBatchMatmulV3Tiling::CalcPertokenOptUbTiling()
{
    uint64_t ubSize = aicoreParams_.ubSize;
    uint32_t ubCalcN = static_cast<uint32_t>(tbeTiling_.n_l0) * BLOCK_CUBE;
    // input and ub out: mm out int32, ub out bf16/fp16
    uint64_t ubCalc = NUM_DB * ubCalcN * (sizeof(int32_t) + sizeof(int16_t));
    // BroadCast需要的临时空间，最小为256b，最大为：baseM * 32, baseM不会超过2048，不需要乘法溢出校验
    uint64_t needUbSize = tbeTiling_.m_l0 * BLOCK_CUBE * ONE_BLK_SIZE;
    // veccalc: pertokenScale * scale -> (M * baseN) fp32, maxSize: 128KB, in ub all the time
    needUbSize += inputParams_.mSize * ubCalcN * sizeof(float32_t);
    // input: pertokenScale, once, in ub all the time
    needUbSize += ops::CeilAlign(inputParams_.mSize, 8UL) * sizeof(float32_t); // 8: 32 / sizeof(fp32)
    if (!inputParams_.isPerTensor) {
        // input: scale, fp32: db * sizeof(fp32); bf16：db * sizeof(bf16) + veccalc sizeof(fp32) -> 2 * 4
        needUbSize += NUM_DB * ubCalcN * sizeof(float32_t);
    }
    if (inputParams_.biasDtype != ge::DT_INT32) {
        // input: bias bf16 fp16 fp32
        needUbSize += NUM_DB * ubCalcN * ge::GetSizeByDataType(inputParams_.biasDtype);
        // veccalc: bias fp32
        needUbSize += ubCalcN * sizeof(float32_t);
    }
    OP_TILING_CHECK(needUbSize >= ubSize,
                    CUBE_INNER_ERR_REPORT(
                        inputParams_.opName, "there is no proper ub tiling when m(%lu) pertoken opt", inputParams_.mSize),
                    return ge::GRAPH_FAILED);

    ubSize -= needUbSize;
    // 已知ubCalcN, 求解还能放得下的ubCalcM
    // veccalc: int32 -> fp32
    ubCalc += ubCalcN * sizeof(float32_t);
    uint32_t ubCalcM =
        std::min(std::min(ubSize / ubCalc, static_cast<uint64_t>(tbeTiling_.m_l0) * BLOCK_CUBE), inputParams_.mSize);
    OP_TILING_CHECK(ubCalcM == 0,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "failed to calc ubCalcM(0) with ubCalcN(%u)", ubCalcN),
                    return ge::GRAPH_FAILED);
    tilingData_.params.set_ubCalcM(ubCalcM);
    tilingData_.params.set_ubCalcN(ubCalcN);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulV3Tiling::CalcUbTiling()
{
    if (isBf16Opt_ && inputParams_.isPertoken) {
        return CalcPertokenOptUbTiling();
    }
    return CalcUbTiling(static_cast<uint32_t>(tbeTiling_.n_l0) * BLOCK_CUBE,
                        static_cast<uint32_t>(tbeTiling_.m_l0) * BLOCK_CUBE);
}

ge::graphStatus QuantBatchMatmulV3Tiling::CalcUbTiling(uint32_t baseN, uint32_t baseM)
{
    uint64_t ubSize = aicoreParams_.ubSize;
    uint64_t needUbSize = 0;
    uint32_t ubCalcN = baseN;
    // src(int32) + scale(fp32/bf16) + pertoken(fp32) + out(fp16/bf16) + veccalc, in and out need double buffer
    // int16_t reprersent bf16, input src + output dst + veccalc dequant api
    uint64_t ubCalc = (NUM_DB * (sizeof(int32_t) + sizeof(int16_t)) + UB_EXTRE_BYTE) * ubCalcN;
    // input: scale perchannel
    if (!inputParams_.isPerTensor) {
        ubCalc += NUM_DB * ge::GetSizeByDataType(inputParams_.scaleDtype)* ubCalcN;
    }
    if (inputParams_.isPertoken || (inputParams_.biasDtype != ge::DT_INT32)) {
        // veccalc: dequant api dst fp32
        ubCalc += sizeof(float) * ubCalcN;
    }
    if (inputParams_.isPertoken) {
        // veccalc: BroadCast需要的临时空间，最小为256b，最大为align(ubM, 8) * 32b, 按照baseM先算
        // baseM不会超过2048，不需要乘法溢出校验
        needUbSize += baseM * ONE_BLK_SIZE;
    }
    if (inputParams_.isPertoken) {
        // input: pertokenScale fp32
        ubCalc += NUM_DB * sizeof(float);
        // 7: to comfirm that pertokenScale 32B(8, fp32) aligned, up to 7, eg: 1->8
        needUbSize += NUM_DB * sizeof(float) * 7;
        // veccalc: mul(* pertokenScale) fp32 m * n, res of broadcast
        ubCalc += sizeof(float) * ubCalcN;
    }
    if (inputParams_.biasDtype != ge::DT_INT32) {
        // veccalc: fp32 out muls fp32 bias
        ubCalc += sizeof(float) * ubCalcN;
        // input: bias bf16/fp16/fp32, veccalc: bias fp32
        needUbSize += NUM_DB * ge::GetSizeByDataType(inputParams_.biasDtype) * ubCalcN + sizeof(float) * ubCalcN;
    }
    OP_TILING_CHECK(needUbSize >= ubSize,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "there is no proper ub tiling when m(%lu) n(%lu) baseM(%u) baseN(%u)",
                                          inputParams_.mSize, inputParams_.nSize, baseM, baseN),
                    return ge::GRAPH_FAILED);
    ubSize -= needUbSize;
    uint32_t ubCalcM = std::min(std::min(ubSize / ubCalc, static_cast<uint64_t>(baseM)), inputParams_.mSize);
    OP_TILING_CHECK(ubCalcM == 0,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "failed to calc ubCalcM(0) with ubCalcN(%u)", ubCalcN),
                    return ge::GRAPH_FAILED);
    tilingData_.params.set_ubCalcN(ubCalcN);
    tilingData_.params.set_ubCalcM(ubCalcM);
    tilingData_.params.set_needUbBuffer(ubCalcN * ubCalcM * UB_EXTRE_BYTE);
    return ge::GRAPH_SUCCESS;
}

void QuantBatchMatmulV3Tiling::AnalyzeBatchInfo(const gert::Shape &oriShapeA, const gert::Shape &oriShapeB)
{
    int32_t numDimA = static_cast<int32_t>(oriShapeA.GetDimNum());
    int32_t numDimB = static_cast<int32_t>(oriShapeB.GetDimNum());
    inputParams_.batchA4 = numDimA > IDX_K_LOW ? oriShapeA.GetDim(numDimA - IDX_K_HIGH) : 1;
    inputParams_.batchA3 = numDimA > IDX_K_HIGH ? oriShapeA.GetDim(numDimA - IDX_N_LOW) : 1;
    inputParams_.batchA2 = numDimA > IDX_N_LOW ? oriShapeA.GetDim(numDimA - IDX_N_HIGH) : 1;
    inputParams_.batchA1 = numDimA > IDX_N_HIGH ? oriShapeA.GetDim(numDimA - IDX_B_LOW) : 1;
    inputParams_.batchB4 = numDimB > IDX_K_LOW ? oriShapeB.GetDim(numDimB - IDX_K_HIGH) : 1;
    inputParams_.batchB3 = numDimB > IDX_K_HIGH ? oriShapeB.GetDim(numDimB - IDX_N_LOW) : 1;
    inputParams_.batchB2 = numDimB > IDX_N_LOW ? oriShapeB.GetDim(numDimB - IDX_N_HIGH) : 1;
    inputParams_.batchB1 = numDimB > IDX_N_HIGH ? oriShapeB.GetDim(numDimB - IDX_B_LOW) : 1;
}

// Notice: 修改此函数可能会影响mc2功能，使用isTilingOut_变量判断是否为mc2场景
const gert::Shape &QuantBatchMatmulV3Tiling::GetScaleShape(const size_t index)
{
    return context_->GetInputShape(index)->GetStorageShape();
}

// Notice: 修改此函数可能会影响mc2功能，使用isTilingOut_变量判断是否为mc2场景
const gert::Shape QuantBatchMatmulV3Tiling::GetX1Shape(const size_t index)
{
    auto x1Desc = context_->GetInputDesc(index);
    auto x1Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x1Desc->GetStorageFormat()));
    auto x1Shape = context_->GetInputShape(index)->GetStorageShape();
    if (x1Format == Format::FORMAT_FRACTAL_NZ) {
        x1Shape = context_->GetInputShape(index)->GetOriginShape();
    }
    return x1Shape;
}

// Notice: 修改此函数可能会影响mc2功能，使用isTilingOut_变量判断是否为mc2场景
const gert::Shape QuantBatchMatmulV3Tiling::GetX2Shape(const size_t index)
{
    auto x2Desc = context_->GetInputDesc(index);
    auto x2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x2Desc->GetStorageFormat()));
    auto x2Shape = context_->GetInputShape(index)->GetStorageShape();
    if (x2Format == Format::FORMAT_FRACTAL_NZ) {
        x2Shape = context_->GetInputShape(index)->GetOriginShape();
    }
    return x2Shape;
}

// Notice: 修改此函数可能会影响mc2功能，使用isTilingOut_变量判断是否为mc2场景
const gert::StorageShape *QuantBatchMatmulV3Tiling::GetPertokenShape(const size_t index)
{
    return context_->GetOptionalInputShape(index);
}

// Notice: 修改此函数可能会影响mc2功能，使用isTilingOut_变量判断是否为mc2场景
const gert::StorageShape *QuantBatchMatmulV3Tiling::GetBiasShape(const size_t index)
{
    return context_->GetOptionalInputShape(index);
}

void QuantBatchMatmulV3Tiling::SetFormat()
{
    inputParams_.aFormat = ge::FORMAT_ND;
    inputParams_.bFormat = ge::FORMAT_ND;
    inputParams_.cFormat = ge::FORMAT_ND;
    auto x1Desc = context_->GetInputDesc(X1_INDEX);
    if (static_cast<ge::Format>(ge::GetPrimaryFormat(x1Desc->GetStorageFormat())) == Format::FORMAT_FRACTAL_NZ) {
        inputParams_.aFormat = ge::FORMAT_FRACTAL_NZ;
    }
    auto x2Desc = context_->GetInputDesc(X2_INDEX);
    if (static_cast<ge::Format>(ge::GetPrimaryFormat(x2Desc->GetStorageFormat())) == Format::FORMAT_FRACTAL_NZ) {
        inputParams_.bFormat = ge::FORMAT_FRACTAL_NZ;
    }
}

bool QuantBatchMatmulV3Tiling::NeedAtomiClean() const {
    if (!optiling::PlatformInfo::GetInstance().support_l0c2out()) {
        uint32_t alignSize = ONE_BLK_SIZE / ge::GetSizeByDataType(inputParams_.cDtype);

        uint32_t baseN = static_cast<uint32_t>(tilingData_.matmulTiling.get_baseN());
        uint32_t singleCoreN = static_cast<uint32_t>(tilingData_.matmulTiling.get_singleCoreN());
        if (baseN < alignSize || CalcTailSize(singleCoreN, baseN) < alignSize) {
            return true;
        }

        uint32_t nDim = ops::CeilDiv(static_cast<uint32_t>(inputParams_.nSize), singleCoreN);
        uint32_t tailSingleCoreN = inputParams_.nSize - (nDim - 1) * singleCoreN;
        return CalcTailSize(tailSingleCoreN, baseN) < alignSize;
    } else {
        uint32_t singleCoreK = static_cast<uint32_t>(tilingData_.matmulTiling.get_singleCoreK());
        return singleCoreK < static_cast<uint32_t>(inputParams_.kSize);
    }
}

void QuantBatchMatmulV3Tiling::SetTransAttr(QuantBatchMatmulV3Trans &trans) const
{
    if (!inputParams_.transA && inputParams_.transB) { // most for ND
        trans = QuantBatchMatmulV3Trans::B_TRANS;
    } else if (!inputParams_.transA && !inputParams_.transB) { // most for weight NZ
        trans = QuantBatchMatmulV3Trans::NO_TRANS;
    } else if (inputParams_.transA && !inputParams_.transB) {
        trans = QuantBatchMatmulV3Trans::A_TRANS;
    } else {
        trans = QuantBatchMatmulV3Trans::AB_TRANS;
    }
}

REGISTER_TILING_TEMPLATE("QuantBatchMatmulV3", QuantBatchMatmulV3Tiling, 1);

static ge::graphStatus QuantBatchMatmulV3TilingFunc(gert::TilingContext *context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingParseForQuantBatchMatmulV3(gert::TilingParseContext *context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "platformInfoPtr is null");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    auto compileInfoPtr = context->GetCompiledInfo<QuantBatchMatmulV3CompileInfo>();
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "compileInfoPtr is null");
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1_size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0c_size);
    compileInfoPtr->ub_size = ubSize;
    compileInfoPtr->workspace_num = ascendcPlatform.GetLibApiWorkSpaceSize();
    compileInfoPtr->ParseRuntimePlatformInfo(context->GetNodeName(), *platformInfoPtr);
    compileInfoPtr->core_num = ascendcPlatform.GetCoreNumAic();
    optiling::PlatformInfo& plaformInstance = optiling::PlatformInfo::GetInstance();
    OP_LOGE_IF(plaformInstance.core_num <= 0, ge::GRAPH_FAILED, context->GetNodeName(), "coreNum <= 0");
    plaformInstance.SetInstance(*compileInfoPtr);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QuantBatchMatmulV3)
    .Tiling(QuantBatchMatmulV3TilingFunc)
    .TilingParse<QuantBatchMatmulV3CompileInfo>(TilingParseForQuantBatchMatmulV3);
}  // namespace optiling
