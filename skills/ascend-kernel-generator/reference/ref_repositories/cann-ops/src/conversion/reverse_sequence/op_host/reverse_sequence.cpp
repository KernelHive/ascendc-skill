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
 * @file reverse_sequence.cpp
 */
#include <sstream>
#include "register/op_def_registry.h"
#include "reverse_sequence_tiling.h"
#include "tiling/tiling_api.h"
#include "strided_slice.h"

namespace ops{

#define OP_TILING_CHECK(cond, log_func, expr)  \
    do {                                       \
        if (cond) {                            \
            std::printf(log_func);             \
            expr;                              \
        }                                      \
    } while (0)

#define OP_CHECK(cond, log_func, return_expr)    \
    if (cond) {                                  \
        std::printf(log_func);                   \
        return_expr;                             \
    }

#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                     \
    if ((ptr) == nullptr) {                                                                           \
        const char* name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName();  \
        std::printf("name is nullptr!");                                                              \
        return ge::GRAPH_FAILED;                                                                      \
    }

#define OP_LOGE(opname, ...)

#define ADD_VALUE_TO_STR(str, value) (str) = ((str) + #value + "=" + std::to_string(value) + ", ")
} // namespace ops

template <typename T>
inline T* GetCompileInfoPtr(gert::TilingParseContext* context) {
    return context->GetCompiledInfo<T>();
}

namespace optiling {

constexpr int64_t X_INDEX = 0;
constexpr int64_t SEQ_LENGTHS_INDEX = 1;
constexpr int64_t Y_INDEX = 0;

constexpr int64_t SEQ_DIM_ATTR_INDEX = 0;
constexpr int64_t BATCH_DIM_ATTR_INDEX = 1;

constexpr int64_t X_MIN_DIM_CNT = 3;
constexpr int64_t SEQ_LENGTHS_DIM_CNT = 1;
constexpr int64_t UB_BUF_CNT = 2;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t WORK_SPACE_SIZE = 1;

static const std::vector<ge::DataType> X_DTYPES = {ge::DT_FLOAT16, ge::DT_FLOAT,    ge::DT_INT8,  ge::DT_UINT8,
                                                   ge::DT_INT16,   ge::DT_UINT16,   ge::DT_INT32, ge::DT_INT64,
                                                   ge::DT_BOOL,    ge::DT_COMPLEX64};
static const std::string X_DTYPES_STR = "[float16, float, int8, uint8, int16, uint16, int32, int64, bool, complex64]";
static const std::vector<ge::DataType> SEQ_LENGTHS_DTYPES = {ge::DT_INT32, ge::DT_INT64};
static const std::string SEQ_LENGTHS_DTYPES_STR = "[int32, int64]";

class ReverseSequenceTiling {
public:
    explicit ReverseSequenceTiling(gert::TilingContext* context)
        : context_(context), nodeName_(context->GetNodeName()){};
    ge::graphStatus RunTiling4ReverseSequence();

private:
    ge::graphStatus CheckParams();
    ge::graphStatus CheckInputParams();
    ge::graphStatus CheckOutputParams();
    ge::graphStatus CheckAttrParams();
    ge::graphStatus FillCompileInfo();

    void MergeAxes();
    void ProcessTilingCore();
    void FillTilingData();

private:
    gert::TilingContext* context_ = nullptr;
    std::string nodeName_ = "ReverseSequence";
    ReverseSequenceTilingData tilingData_;
    ReverseSequenceCompileInfo compileInfo_;
    ReverseSequenceTilingKey tilingKey_;

    gert::Shape xShape_;
    gert::Shape seqLengthsShape_;
    ge::DataType xDtype_ = ge::DT_UNDEFINED;
    int64_t realCoreNum_ = 0;
    int64_t xDimCount_ = 0;
    int64_t seqDimAttr_ = 0;
    int64_t batchDimAttr_ = 0;

    // tiling params
    int64_t batchDimValue_ = 0;
    int64_t seqDimValue_ = 0;
    int64_t xDtypeSize_ = 0;
    int64_t batchSize_ = 1;
    int64_t seqSize_ = 1;
    int64_t cSize_ = 1;
    int64_t maxProcCount_ = 0;
    int64_t loopTimePerCore_ = 0;
    int64_t tailCoreNum_ = 0;
    int64_t innerLoopTime_ = 0;
    int64_t innerTailCount_ = 0;
};

ge::graphStatus ReverseSequenceTiling::RunTiling4ReverseSequence() {
    OP_TILING_CHECK(CheckParams() != ge::GRAPH_SUCCESS, "CheckParams failed.",
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(FillCompileInfo() != ge::GRAPH_SUCCESS, "FillCompileInfo failed.",
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(compileInfo_.totalCoreNum == 0 || compileInfo_.ubSizePlatForm == 0,
                    "Invalid compile info.", return ge::GRAPH_FAILED);
    maxProcCount_ = (compileInfo_.ubSizePlatForm / UB_BUF_CNT / BLOCK_SIZE) * BLOCK_SIZE / xDtypeSize_;
    MergeAxes();
    OP_TILING_CHECK(cSize_ * xDtypeSize_ < BLOCK_SIZE, "MTE carries less than 32 Byte.",
                    return ge::GRAPH_FAILED);
    ProcessTilingCore();

    context_->SetBlockDim(realCoreNum_);
    context_->SetTilingKey(static_cast<uint64_t>(tilingKey_));
    FillTilingData();

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = WORK_SPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReverseSequenceTiling::CheckParams() {
    OP_TILING_CHECK(CheckInputParams() != ge::GRAPH_SUCCESS, "CheckInputParams failed.",
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOutputParams() != ge::GRAPH_SUCCESS, "CheckOutputParams failed.",
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAttrParams() != ge::GRAPH_SUCCESS, "CheckAttrParams failed.",
                    return ge::GRAPH_FAILED);

    batchDimValue_ = xShape_.GetDim(batchDimAttr_);
    seqDimValue_ = xShape_.GetDim(seqDimAttr_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReverseSequenceTiling::CheckInputParams() {
    auto xDescPtr = context_->GetInputDesc(X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, xDescPtr);
    xDtype_ = xDescPtr->GetDataType();
    OP_TILING_CHECK(std::find(X_DTYPES.begin(), X_DTYPES.end(), xDtype_) == X_DTYPES.end(),
                    "Input x dtype only supports", return ge::GRAPH_FAILED);

    xDtypeSize_ = ge::GetSizeByDataType(xDtype_);
    OP_TILING_CHECK(xDtypeSize_ <= 0, "Get xDtypeSize failed.", return ge::GRAPH_FAILED);

    auto xShapePtr = context_->GetInputShape(X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    xShape_ = xShapePtr->GetStorageShape();
    xDimCount_ = xShape_.GetDimNum();
    OP_TILING_CHECK(xDimCount_ < X_MIN_DIM_CNT, "Input x dim invaild", return ge::GRAPH_FAILED);

    auto seqLengthsDescPtr = context_->GetInputDesc(SEQ_LENGTHS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, seqLengthsDescPtr);
    auto seqLengthsDtype = seqLengthsDescPtr->GetDataType();
    OP_TILING_CHECK(
        std::find(SEQ_LENGTHS_DTYPES.begin(), SEQ_LENGTHS_DTYPES.end(), seqLengthsDtype) == SEQ_LENGTHS_DTYPES.end(),
        "Input seqLengths dtype invaild.", return ge::GRAPH_FAILED);
    auto seqLengthsShapePtr = context_->GetInputShape(SEQ_LENGTHS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, seqLengthsShapePtr);
    seqLengthsShape_ = seqLengthsShapePtr->GetStorageShape();
    OP_TILING_CHECK(seqLengthsShape_.GetDimNum() != SEQ_LENGTHS_DIM_CNT,
                    "Input seqLengths dim invaild.", return ge::GRAPH_FAILED);

    OP_TILING_CHECK(xShape_.GetShapeSize() == 0 || seqLengthsShape_.GetShapeSize() == 0,
                    "Input x or seqLengths not support empty tensor.", return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReverseSequenceTiling::CheckOutputParams() {
    auto yDescPtr = context_->GetInputDesc(Y_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, yDescPtr);
    auto yDtype = yDescPtr->GetDataType();
    OP_TILING_CHECK(yDtype != xDtype_, "The dtype of y and x must be the same.",
                    return ge::GRAPH_FAILED);

    auto yShapePtr = context_->GetInputShape(Y_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();

    OP_TILING_CHECK(yShape != xShape_, "The shape of y and x must be the same.",
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReverseSequenceTiling::CheckAttrParams() {
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int64_t* ptrSeqDim = attrs->GetAttrPointer<int64_t>(SEQ_DIM_ATTR_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, ptrSeqDim);
    seqDimAttr_ = *ptrSeqDim;
    seqDimAttr_ = seqDimAttr_ < 0 ? xDimCount_ + seqDimAttr_ : seqDimAttr_;
    const int64_t* ptrBatchDim = attrs->GetAttrPointer<int64_t>(BATCH_DIM_ATTR_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, ptrBatchDim);
    batchDimAttr_ = *ptrBatchDim;
    batchDimAttr_ = batchDimAttr_ < 0 ? xDimCount_ + batchDimAttr_ : batchDimAttr_;
    OP_TILING_CHECK(seqDimAttr_ < 0 || seqDimAttr_ > xDimCount_ - 1,
                    "Invalid attr, seqDimAttr_.", return ge::GRAPH_FAILED);
    OP_TILING_CHECK(batchDimAttr_ < 0 || batchDimAttr_ > xDimCount_ - 1,
                    "Invalid attr, batchDimAttr_.", return ge::GRAPH_FAILED);

    OP_TILING_CHECK(seqDimAttr_ == batchDimAttr_, "Attr seqDim and batchDim cannot be the same.",
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(seqDimAttr_ == xDimCount_ - 1 || batchDimAttr_ == xDimCount_ - 1,
                   "seqDim or batchDim cannot be the tail axis.", return ge::GRAPH_FAILED);
    OP_TILING_CHECK(seqLengthsShape_.GetDim(0) != xShape_.GetDim(batchDimAttr_),
                    "seqLengthsShape[0] is not equal to xShape[batchDim].",
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReverseSequenceTiling::FillCompileInfo() {
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto ptrCompileInfo = context_->GetCompileInfo<ReverseSequenceCompileInfo>();
        if (ptrCompileInfo == nullptr) {
            std::printf("GetCompileInfo is also nullptr.");
            return ge::GRAPH_FAILED;
        }
        compileInfo_ = *ptrCompileInfo;
        return ge::GRAPH_SUCCESS;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo_.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ubSizePlatForm);
    compileInfo_.ubSizePlatForm = ubSizePlatForm;
    return ge::GRAPH_SUCCESS;
}

void ReverseSequenceTiling::MergeAxes() {
    if (batchDimAttr_ < seqDimAttr_) {
        for (int64_t i = 0; i < batchDimAttr_ + 1; ++i) {
            batchSize_ *= xShape_.GetDim(i);
        }
        for (int64_t i = batchDimAttr_ + 1; i < seqDimAttr_ + 1; ++i) {
            seqSize_ *= xShape_.GetDim(i);
        }
        for (int64_t i = seqDimAttr_ + 1; i < xDimCount_; ++i) {
            cSize_ *= xShape_.GetDim(i);
        }
        if (cSize_ > maxProcCount_) {
            tilingKey_ = ReverseSequenceTilingKey::BATCH_DIM_0_C_BIG;
        } else {
            tilingKey_ = ReverseSequenceTilingKey::BATCH_DIM_0_C_SMALL;
        }
    } else {
        for (int64_t i = 0; i < seqDimAttr_ + 1; ++i) {
            seqSize_ *= xShape_.GetDim(i);
        }
        for (int64_t i = seqDimAttr_ + 1; i < batchDimAttr_ + 1; ++i) {
            batchSize_ *= xShape_.GetDim(i);
        }
        for (int64_t i = batchDimAttr_ + 1; i < xDimCount_; ++i) {
            cSize_ *= xShape_.GetDim(i);
        }
        if (cSize_ > maxProcCount_) {
            tilingKey_ = ReverseSequenceTilingKey::BATCH_DIM_1_C_BIG;
        } else {
            tilingKey_ = ReverseSequenceTilingKey::BATCH_DIM_1_C_SMALL;
        }
    }
}

void ReverseSequenceTiling::ProcessTilingCore() {
    if (batchSize_ < compileInfo_.totalCoreNum) {
        realCoreNum_ = batchSize_;
        loopTimePerCore_ = 1;
        tailCoreNum_ = 0;
    } else {
        realCoreNum_ = compileInfo_.totalCoreNum;
        loopTimePerCore_ = batchSize_ / realCoreNum_;
        tailCoreNum_ = batchSize_ % realCoreNum_;
    }
    if (cSize_ > maxProcCount_) {
        innerLoopTime_ = cSize_ / maxProcCount_;
        innerTailCount_ = cSize_ % maxProcCount_;
    }
}

void ReverseSequenceTiling::FillTilingData() {
    tilingData_.set_tilingKey(static_cast<int64_t>(tilingKey_));
    tilingData_.set_batchDimValue(batchDimValue_);
    tilingData_.set_seqDimValue(seqDimValue_);
    tilingData_.set_xDtypeSize(xDtypeSize_);
    tilingData_.set_batchSize(batchSize_);
    tilingData_.set_seqSize(seqSize_);
    tilingData_.set_cSize(cSize_);
    tilingData_.set_maxProcCount(maxProcCount_);
    tilingData_.set_loopTimePerCore(loopTimePerCore_);
    tilingData_.set_tailCoreNum(tailCoreNum_);
    tilingData_.set_innerLoopTime(innerLoopTime_);
    tilingData_.set_innerTailCount(innerTailCount_);

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
}

ge::graphStatus Tiling4ReverseSequence(gert::TilingContext* context) {
    ReverseSequenceTiling tilingObject(context);
    OP_TILING_CHECK(tilingObject.RunTiling4ReverseSequence() != ge::GRAPH_SUCCESS,
                    "RunTiling4ReverseSequence failed.", return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4ReverseSequence(gert::TilingParseContext* context) {
    auto compileInfo = GetCompileInfoPtr<ReverseSequenceCompileInfo>(context);
    OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_TILING_CHECK(
        compileInfo->totalCoreNum == 0,
       "TilingPrepare4ReverseSequence get core num failed.", return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSizePlatForm);
    OP_TILING_CHECK(
        compileInfo->ubSizePlatForm == 0,
        "TilingPrepare4ReverseSequence get ub size failed.", return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ReverseSequence)
    .Tiling(Tiling4ReverseSequence)
    .TilingParse<ReverseSequenceCompileInfo>(TilingPrepare4ReverseSequence);

}  // namespace optiling

namespace ge {
constexpr int64_t UNKNOWN_RANK_DIM_VALUE_ = -2;
inline bool IsUnknownRank(const gert::Shape* check_shape) {
    return check_shape->GetDimNum() == 1 && check_shape->GetDim(0) == UNKNOWN_RANK_DIM_VALUE_;
}

inline ge::graphStatus SetUnknownRank(gert::Shape* output_shape) {
    OP_CHECK(output_shape == nullptr, "the output_shape is nullptr, return unsuccess",
        return ge::GRAPH_FAILED);
    output_shape->SetDimNum(0);
    output_shape->AppendDim(UNKNOWN_RANK_DIM_VALUE_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShape4Elewise(gert::InferShapeContext* context) {
    auto in_shape = context->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto out_shape = context->GetOutputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, out_shape);

    if (IsUnknownRank(in_shape)) {
        return SetUnknownRank(out_shape);
    }

    *out_shape = *in_shape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ReverseSequence).InferShape(InferShape4Elewise);

} // namespace ge


namespace ops {

constexpr int64_t X_MIN_DIM_CNT = 3;
constexpr int64_t SEQ_LENGTHS_DIM_CNT = 1;

constexpr int64_t DYN_SHAPE = -1;
constexpr int64_t UNKNOW_SHAPE = -2;

constexpr int64_t BLOCK_SIZE = 32;

static const std::vector<ge::DataType> g_xDtypes = {
    ge::DT_FLOAT16, ge::DT_FLOAT,  ge::DT_INT8,      ge::DT_UINT8,   ge::DT_INT16, ge::DT_UINT16,   ge::DT_INT32,
    ge::DT_INT64,   ge::DT_BOOL,   ge::DT_COMPLEX64, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8,     ge::DT_UINT8,
    ge::DT_INT16,   ge::DT_UINT16, ge::DT_INT32,     ge::DT_INT64,   ge::DT_BOOL,  ge::DT_COMPLEX64};

static const std::vector<ge::DataType> g_yDtypes = g_xDtypes;

static const std::vector<ge::DataType> g_seqLengthsDtypes = {
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};

static const std::vector<ge::Format> g_formats = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class ReverseSequenceCheckSupported {
public:
    explicit ReverseSequenceCheckSupported(const ge::Operator& op) : op_(op){};

    ge::graphStatus CheckParams() {
        ge::graphStatus retStatus = CheckInputParams();
        if (retStatus != ge::GRAPH_SUCCESS) {
            return retStatus;
        };
        retStatus = CheckOutputParams();
        if (retStatus != ge::GRAPH_SUCCESS) {
            return retStatus;
        }
        retStatus = CheckAttrParams();
        if (retStatus != ge::GRAPH_SUCCESS) {
            return retStatus;
        }
        int64_t max_value = seqDimAttr_ > batchDimAttr_ ? seqDimAttr_ : batchDimAttr_;
        int64_t cSzie = 1;
        for (int64_t i = max_value + 1; i < xDimCount_; ++i) {
            cSzie *= xShape_.GetDim(i);
        }
        if (cSzie * xDtypeSize_ < BLOCK_SIZE) {
            reasonOSS_ << "MTE carries less than 32 Byte.";
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    std::string GetReasonStr() {
        return reasonOSS_.str();
    }

private:
    ge::graphStatus CheckInputParams() {
        auto xDesc = op_.GetInputDescByName("x");
        xShape_ = xDesc.GetShape();
        std::vector<int64_t> xDims = xShape_.GetDims();
        for (const auto& dim : xDims) {
            if (dim == DYN_SHAPE || dim == UNKNOW_SHAPE) {
                reasonOSS_ << "Input x is dynamic shape or unknow shape.";
                return ge::GRAPH_NOT_CHANGED;
            }
        }

        seqLengthsShape_ = op_.GetInputDescByName("seq_lengths").GetShape();
        std::vector<int64_t> seqLengthsDims = seqLengthsShape_.GetDims();
        for (const auto& dim : seqLengthsDims) {
            if (dim == DYN_SHAPE || dim == UNKNOW_SHAPE) {
                reasonOSS_ << "Input seqLengths is dynamic shape or unknow shape.";
                return ge::GRAPH_NOT_CHANGED;
            }
        }

        xDimCount_ = xShape_.GetDimNum();
        if (xDimCount_ < X_MIN_DIM_CNT) {
            reasonOSS_ << "Input x dim count must >= " << X_MIN_DIM_CNT << ", cur is " << xDimCount_ << ".";
            return ge::GRAPH_FAILED;
        }
        if (seqLengthsShape_.GetDimNum() != SEQ_LENGTHS_DIM_CNT) {
            reasonOSS_ << "Input seqLengths dim count must be " << SEQ_LENGTHS_DIM_CNT << ".";
            return ge::GRAPH_FAILED;
        }

        xDtypeSize_ = ge::GetSizeByDataType(xDesc.GetDataType());
        if (xDtypeSize_ <= 0) {
            reasonOSS_ << "Get xDtype Size failed.";
            return ge::GRAPH_FAILED;
        }

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckOutputParams() {
        auto yShape = op_.GetOutputDescByName("y").GetShape();
        if (yShape.GetDims() != xShape_.GetDims()) {
            reasonOSS_ << "The shape of y and x must be the same.";
            return ge::GRAPH_FAILED;
        }

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckAttrParams() {
        ge::graphStatus retStatus = op_.GetAttr("seq_dim", seqDimAttr_);
        if (retStatus != ge::GRAPH_SUCCESS) {
            reasonOSS_ << "Get attr seq_dim failed.";
            return ge::GRAPH_FAILED;
        }
        seqDimAttr_ = seqDimAttr_ < 0 ? xDimCount_ + seqDimAttr_ : seqDimAttr_;

        retStatus = op_.GetAttr("batch_dim", batchDimAttr_);
        if (retStatus != ge::GRAPH_SUCCESS) {
            reasonOSS_ << "Get attr batch_dim failed.";
            return ge::GRAPH_FAILED;
        }
        batchDimAttr_ = batchDimAttr_ < 0 ? xDimCount_ + batchDimAttr_ : batchDimAttr_;

        if (seqDimAttr_ < 0 || seqDimAttr_ > xDimCount_ - 1) {
            reasonOSS_ << "Invalid attr, seqDimAttr_: " << seqDimAttr_ << ".";
            return ge::GRAPH_FAILED;
        }
        if (batchDimAttr_ < 0 || batchDimAttr_ > xDimCount_ - 1) {
            reasonOSS_ << "Invalid attr, batchDimAttr_: " << batchDimAttr_ << ".";
            return ge::GRAPH_FAILED;
        }
        if (seqDimAttr_ == batchDimAttr_) {
            reasonOSS_ << "Attr seqDim and batchDim cannot be the same.";
            return ge::GRAPH_FAILED;
        }
        if (seqDimAttr_ == xDimCount_ - 1 || batchDimAttr_ == xDimCount_ - 1) {
            reasonOSS_ << "seqDim or batchDim cannot be the tail axis.";
            return ge::GRAPH_FAILED;
        }
        if (seqLengthsShape_.GetDim(0) != xShape_.GetDim(batchDimAttr_)) {
            reasonOSS_ << "seqLengthsShape[0] is not equal to xShape[batchDim].";
            return ge::GRAPH_FAILED;
        }

        return ge::GRAPH_SUCCESS;
    }

private:
    const ge::Operator& op_;
    std::ostringstream reasonOSS_;
    ge::Shape xShape_;
    ge::Shape seqLengthsShape_;
    int64_t xDtypeSize_ = 0;
    int64_t xDimCount_ = 0;
    int64_t seqDimAttr_ = 0;
    int64_t batchDimAttr_ = 0;
};

static ge::graphStatus CheckSupported(const ge::Operator& op, ge::AscendString& result) {
    std::string resultJsonStr;
    ReverseSequenceCheckSupported checkSupportedObj(op);
    ge::graphStatus ret = checkSupportedObj.CheckParams();
    if (ret == ge::GRAPH_SUCCESS) {
        resultJsonStr = R"({"isSupported": "True", "dynamicCompileStatic": "True", "reason": ""})";
        result = ge::AscendString(resultJsonStr.c_str());
        return ge::GRAPH_SUCCESS;
    } else if (ret == ge::GRAPH_FAILED) {
        resultJsonStr = R"({"isSupported": "False", "dynamicCompileStatic": "True", "reason": ")" +
                        checkSupportedObj.GetReasonStr() + R"("})";
        result = ge::AscendString(resultJsonStr.c_str());
        return ge::GRAPH_FAILED;
    } else {
        resultJsonStr = R"({"isSupported": "Unknown", "dynamicCompileStatic": "True", "reason": ")" +
                        checkSupportedObj.GetReasonStr() + R"("})";
        result = ge::AscendString(resultJsonStr.c_str());
        return ge::GRAPH_SUCCESS;
    }
}

class ReverseSequence : public OpDef {
public:
    explicit ReverseSequence(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType(g_xDtypes).Format(g_formats);
        this->Input("seq_lengths").ParamType(REQUIRED).DataType(g_seqLengthsDtypes).Format(g_formats);
        this->Output("y").ParamType(REQUIRED).DataType(g_yDtypes).Format(g_formats);
        this->Attr("seq_dim").Int();
        this->Attr("batch_dim").AttrType(OPTIONAL).Int(0);

        this->AICore().SetCheckSupport(CheckSupported);

        OpAICoreConfig aicConfig;
        aicConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(true)
            .PrecisionReduceFlag(false);

        this->AICore().AddConfig("ascend310p", aicConfig);
    }
};

OP_ADD(ReverseSequence);
}  // namespace ops
