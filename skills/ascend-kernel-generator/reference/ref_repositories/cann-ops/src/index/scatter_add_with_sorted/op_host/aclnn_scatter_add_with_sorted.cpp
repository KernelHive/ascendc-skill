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
 * \file aclnn_scatter_add_with_sorted.cpp
 * \brief
 */
#include "aclnn_scatter_add_with_sorted.h"
#include "broadcast_to.h"
#include "aclnn_kernels/contiguous.h"
#include "scatter_elements.h"
#include "maximum.h"
#include "minimum.h"
#include "squeeze.h"
#include "unsqueeze.h"
#include "sort.h"
#include "arange.h"
#include "tensor_move.h"
#include "scatter_add_with_sorted_l0.h"
#include "linear_index.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/reshape.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_UINT8, op::DataType::DT_INT8, op::DataType::DT_INT16, op::DataType::DT_INT32,
    op::DataType::DT_INT64, op::DataType::DT_BOOL, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
    op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_UINT8, op::DataType::DT_INT8, op::DataType::DT_INT16, op::DataType::DT_INT32,
    op::DataType::DT_INT64, op::DataType::DT_BOOL, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
    op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_INT8,
    op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> AICORE_910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> AICORE_910B_DTYPE_FLOAT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_INDICES = {op::DataType::DT_INT64,
    op::DataType::DT_INT32};

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
    return ASCEND910B_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910_DTYPE_SUPPORT_LIST;
  }
}

static const int64_t REDUCE_ADD = 1;
static const int64_t REDUCE_MUL = 2;
static const int64_t REDUCE_MAX = 3;
static const int64_t REDUCE_MIN = 4;
static const int64_t NEG_ONE = -1;
static const int64_t NEG_TWO = -2;
static const int64_t TWO_DIM = 2;
static const int64_t THREE_DIM = 3;
static const int64_t MAX_SUPPORT_DIMS_NUMS = 8;
static const int64_t MAX_EXACT_FLOAT = 16777216;
static const std::string REDUCTION_NONE = "none";
static const std::string REDUCTION_ADD = "add";
static const std::string REDUCTION_MUL = "mul";
static const std::string REDUCTION_MAX = "max";
static const std::string REDUCTION_MIN = "min";

static const std::string &GetReduceStr(int64_t reduce)
{
    if (reduce == REDUCE_ADD) {
        return REDUCTION_ADD;
    } else if (reduce == REDUCE_MUL) {
        return REDUCTION_MUL;
    } else if (reduce == REDUCE_MAX) {
        return REDUCTION_MAX;
    } else if (reduce == REDUCE_MIN) {
        return REDUCTION_MIN;
    }
    return REDUCTION_NONE;
}

// 除了dim维度以外的每个维度 a <= b 返回true, 否则返回false。 a, b的维度数一样
static bool CompareTensorShape(const aclTensor *a, const aclTensor *b, int64_t dim = -1)
{
    auto aDimSize = a->GetViewShape().GetDimNum();
    auto bDimSize = b->GetViewShape().GetDimNum();
    aDimSize = aDimSize < bDimSize ? aDimSize : bDimSize;
    for (int64_t i = 0; i < aDimSize; i++) {
        if (i != dim) {
            auto aDim = (a->GetViewShape())[i];
            auto bDim = (b->GetViewShape())[i];
            if (bDim < aDim) {
                return false;
            }
        }
    }
    return true;
}

static inline bool CheckNotNull(const aclTensor *self, const aclTensor *index, const aclTensor *src,
    const aclTensor *out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(index, return false);
    OP_CHECK_NULL(src, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckValueDtypeValid(const aclTensor *self, const aclTensor *index, const aclTensor *out)
{
    auto supportList = GetDtypeSupportList();

    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);
    // 当index为空时，不对其dtype进行校验
    if (!index->IsEmpty()) {
        OP_CHECK_DTYPE_NOT_SUPPORT(index, DTYPE_SUPPORT_LIST_INDICES, return false);
    }
    return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *index, const aclTensor *src, const aclTensor *out)
{
    auto supportList = GetDtypeSupportList();

    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
    OP_CHECK_DTYPE_NOT_MATCH(src, self->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);
    // 当index为空时，不对其dtype进行校验
    if (!index->IsEmpty()) {
        OP_CHECK_DTYPE_NOT_SUPPORT(index, DTYPE_SUPPORT_LIST_INDICES, return false);
    }
    return true;
}

// dim num: src = index = self
static bool CheckTensorDim(const aclTensor *self, const aclTensor *index, const aclTensor *src)
{
    auto selfDimSize = self->GetViewShape().GetDimNum();
    auto indexDimSize = index->GetViewShape().GetDimNum();
    auto srcDimSize = src->GetViewShape().GetDimNum();
    if (selfDimSize != indexDimSize || selfDimSize != srcDimSize) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expect self, index, src have same number of dimensions.");
        return false;
    }

    return true;
}

static inline int64_t GetFinalDim(int64_t dim, int64_t dimSize)
{
    int64_t dimFinal = dim;
    if (dim < 0) {
        dimFinal = ((dim + dimSize) >= 0) ? (dim + dimSize) : 0;
    } else {
        dimFinal = (dim < dimSize) ? dim : (dimSize - 1);
    }

    return dimFinal;
}

static bool CheckShape(const aclTensor *self, const aclTensor *index, const aclTensor *src, const aclTensor *out,
    int64_t dim)
{
    // 检查 self 与 out的 shape是否相同
    if (out->GetViewShape().GetDimNum() != 0) {
        OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
    }

    int64_t dimSize = static_cast<int64_t>(self->GetViewShape().GetDimNum());
    int64_t finalDim = GetFinalDim(dim, dimSize);

    // index.shape <= src.shape
    if (!CompareTensorShape(index, src)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expect index shape %s should be smaller than src shape %s in each dimension.",
            op::ToString(index->GetViewShape()).GetString(), op::ToString(src->GetViewShape()).GetString());
        return false;
    }

    // index除dim的维度长 <= self.shape
    if (!CompareTensorShape(index, self, finalDim)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Apart from dimension %ld, index shape %s should be smaller than self shape %s in each dimension.",
            dim, op::ToString(index->GetViewShape()).GetString(), op::ToString(self->GetViewShape()).GetString());
        return false;
    }

    // 检查不超过8维
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);

    return true;
}

static bool CheckDimRange(const aclTensor *self, int64_t dim)
{
    auto dimSize = static_cast<int64_t>(self->GetViewShape().GetDimNum());
    auto dimMax = std::max(-1 * dimSize, dimSize - 1);
    auto dimMin = std::min(-1 * dimSize, dimSize - 1);
    if (dim > dimMax || dim < dimMin) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dim must be in range [%ld, %ld].", dimMin, dimMax);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *index, const aclTensor *src,
    const aclTensor *out)
{
    // 1. 检查参数是否为空指针
    CHECK_COND(CheckNotNull(self, index, src, out), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed!");

    // 2. 检查参数的数据类型是否符合预期
    CHECK_COND(CheckDtypeValid(self, index, src, out), ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed!");

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams2(const aclTensor *self, const aclTensor *index, const aclTensor *src,
    const aclTensor *out, int64_t dim)
{
    // 3. self/index/src的维度数应该一致
    CHECK_COND(CheckTensorDim(self, index, src), ACLNN_ERR_PARAM_INVALID, "CheckTensorDim failed!");

    // 4. self.shape = out.shape + 除了dim以外的维度，index.shape <= self.shape + index.shape <= src.shape
    CHECK_COND(CheckShape(self, index, src, out, dim), ACLNN_ERR_PARAM_INVALID, "checkshape failed!");

    // 5. 检查dim是否为 [-N, N-1] 范围内
    CHECK_COND(CheckDimRange(self, dim), ACLNN_ERR_PARAM_INVALID, "dim must in [-N, N-1]!");

    return ACLNN_SUCCESS;
}

static aclIntArray* GetTensorShape(const aclTensor* self, aclOpExecutor* executor)
{
    auto tensorSize = self->GetViewShape().GetDimNum();
    std::vector<int64_t> tensorShape(tensorSize);
    for (size_t i = 0; i < tensorSize; i++) {
        tensorShape[i] = (self->GetViewShape())[i];
    }
    auto res = executor->AllocIntArray(tensorShape.data(), tensorSize);
    return res;
}

// 如果为0维tensor，那么转换为1维tensor。其余情况转成连续tensor
static const aclTensor* InitializeTensor(const aclTensor* x, aclOpExecutor* executor)
{
    auto xContiguous = l0op::Contiguous(x, executor);
    CHECK_RET(xContiguous != nullptr, nullptr);
    // 如果tensor为0维，则转换为1维tensor
    int64_t squeezeDim = 0;
    if (xContiguous->GetViewShape().GetDimNum() == 0) {
        xContiguous = l0op::UnsqueezeNd(xContiguous, squeezeDim, executor);
    }
    return xContiguous;
}

static void ViewDataType(const aclTensor *input, const op::DataType dtype)
{
    auto tmpTensor = const_cast<aclTensor *>(input);
    tmpTensor->SetDataType(dtype);
    input = tmpTensor;
}

// index[x, y] => index[x,]
static const aclTensor* DoSliceIndex(const aclTensor *index, aclOpExecutor* executor)
{
    auto indexViewShape = index->GetViewShape();
    // slice index
    FVector<int64_t> reshapeVector;
    int64_t size = 1;
    for (size_t i = 0; i < indexViewShape.GetDimNum() - 1; ++i) {
        int64_t dim = indexViewShape.GetDim(i);
        size *= dim;
        reshapeVector.emplace_back(dim);
    }
    aclIntArray *reshapeArray = executor->AllocIntArray(reshapeVector.data(), reshapeVector.size());
    CHECK_RET(reshapeArray != nullptr, nullptr);

    FVector<int64_t> offsetVector(1, 0);    
    aclIntArray *offsetArray = executor->AllocIntArray(offsetVector.data(), offsetVector.size());
    CHECK_RET(offsetArray != nullptr, nullptr);

    FVector<int64_t> sizeVector(1, size);
    aclIntArray *sizeArray = executor->AllocIntArray(sizeVector.data(), sizeVector.size());
    CHECK_RET(sizeArray != nullptr, nullptr);

    auto indexSlice = l0op::Slice(index, offsetArray, sizeArray, executor);
    CHECK_RET(indexSlice != nullptr, nullptr);
    auto indexReshape = l0op::Reshape(indexSlice, reshapeArray, executor);
    return indexReshape;
}

static const aclTensor* DoScatterAddWithSorted(const aclTensor *self, const aclTensor *index, const aclTensor *src,
                                               aclTensor *out, const std::string &reduction,
                                               int64_t dim, aclOpExecutor* executor)
{
    // slice index
    auto indexSlice = DoSliceIndex(index, executor);
    CHECK_RET(indexSlice != nullptr, nullptr);

    // linear index，转换负数索引并输出int32索引，进行合轴
    auto indexLinear = l0op::LinearIndex(indexSlice, self, dim, true, executor);
    CHECK_RET(indexLinear != nullptr, nullptr);
    auto linearSize = indexLinear->GetViewShape().GetDim(0);

    const aclTensor *scatterRes = nullptr;
    // scatter_add and not in float,float16,bf16
    if (!CheckType(self->GetDataType(), AICORE_910B_DTYPE_FLOAT_LIST) && reduction == REDUCTION_ADD) {
        // scatter add index
        scatterRes = l0op::ScatterAddWithSorted(self, src, indexLinear, nullptr, reduction, executor);
    } else {
        // sort index
        const aclTensor *indiceViewFloat = executor->CreateView(indexLinear, {linearSize}, indexLinear->GetViewOffset());
        ViewDataType(indiceViewFloat, op::DataType::DT_FLOAT);
        auto sortResult = l0op::Sort(indiceViewFloat, -1, false, true, executor);
        auto sortIdxOut = std::get<0>(sortResult);
        auto posIdx = std::get<1>(sortResult);
        CHECK_RET(sortIdxOut != nullptr && posIdx != nullptr, nullptr);
        auto sortIndice = executor->CreateView(sortIdxOut, {linearSize}, sortIdxOut->GetViewOffset());
        sortIndice->SetDataType(op::DataType::DT_INT32);

        // scatter add index
        scatterRes = l0op::ScatterAddWithSorted(self, src, sortIndice, posIdx, reduction, executor);
    }
    CHECK_RET(scatterRes != nullptr, nullptr);

    auto viewCopyResult = l0op::ViewCopy(scatterRes, out, executor);
    CHECK_RET(viewCopyResult != nullptr, nullptr);

    return viewCopyResult;
}

static const aclTensor* DoScatterElements(const aclTensor *self, const aclTensor *index, const aclTensor *src,
                                          int64_t dimInput, bool isCopy, const std::string &reduction, aclOpExecutor* executor)
{
    const aclTensor *scatterRes = nullptr;
    std::string newReduction = (reduction == REDUCTION_MIN || reduction == REDUCTION_MAX) ? REDUCTION_NONE : reduction;

    // 910B走ScatterElementsV2且self 和 out 地址不同时，需要拷贝
    if (isCopy) {
        aclTensor *selfCopy = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
        CHECK_RET(selfCopy != nullptr, nullptr);
        auto copyResult = l0op::TensorMove(self, selfCopy, executor);
        CHECK_RET(copyResult != nullptr, nullptr);
        scatterRes = l0op::ScatterElements(selfCopy, index, src, dimInput, newReduction, executor);
    } else {
        scatterRes = l0op::ScatterElements(self, index, src, dimInput, newReduction, executor);
    }
    CHECK_RET(scatterRes != nullptr, nullptr);

    if (reduction == REDUCTION_MAX) {
        scatterRes = l0op::Maximum(self, scatterRes, executor);
        CHECK_RET(scatterRes != nullptr, nullptr);
    } else if (reduction == REDUCTION_MIN) {
        scatterRes = l0op::Minimum(self, scatterRes, executor);
        CHECK_RET(scatterRes != nullptr, nullptr);
    }

    return scatterRes;
}

static aclnnStatus ExecScatterBase(const aclTensor *self, int64_t dim, const aclTensor *index,
    const aclTensor *src, int64_t reduce, aclTensor *out, aclOpExecutor* executor)
{
    const std::string& reduction = GetReduceStr(reduce);

    auto ret = CheckParams(self, index, src, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 如果是0维，需要扩维成1维进行计算
    bool needUnsqueeze = (self->GetViewShape().GetDimNum() == 0);
    int64_t squeezeDim = 0;

    auto selfContiguous = InitializeTensor(self, executor);
    CHECK_COND(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR, "InitializeTensor self failed!");
    auto srcContiguous = InitializeTensor(src, executor);
    CHECK_COND(srcContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR, "InitializeTensor src failed!");

    if (index->IsEmpty()) {
        auto viewCopyResult = l0op::ViewCopy(selfContiguous, out, executor);
        CHECK_COND(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR, "viewCopy self failed!");
        return ACLNN_SUCCESS;
    }

    ret = CheckParams2(self, index, src, out, dim);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    bool aicoreSupport = true;
    bool flag910b = false;
    auto selfType = selfContiguous->GetDataType();
    if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93) {
        aicoreSupport = CheckType(selfType, AICORE_910B_DTYPE_SUPPORT_LIST);
        flag910b = true;
    } else {
        // 转换BF16数据类型为FP32，因算子暂不支持BF16
        if (selfType == op::DataType::DT_BF16 || srcContiguous->GetDataType() == op::DataType::DT_BF16) {
            selfContiguous = l0op::Cast(selfContiguous, op::DataType::DT_FLOAT, executor);
            CHECK_COND(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR, "cast self failed!");
            srcContiguous = l0op::Cast(srcContiguous, op::DataType::DT_FLOAT, executor);
            CHECK_COND(srcContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR, "cast src failed!");
        }
        aicoreSupport = CheckType(selfContiguous->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
    }

    aicoreSupport = aicoreSupport && (reduction != REDUCTION_MUL) && (socVersion != SocVersion::ASCEND310P);

    int64_t selfDimNum = static_cast<int64_t>(selfContiguous->GetViewShape().GetDimNum());
    int64_t dimFinal = GetFinalDim(dim, selfDimNum);

    auto strides = index->GetViewStrides();
    auto indexShape = index->GetViewShape(); 
    auto shape = index->GetStorageShape();

    bool aicore910b = aicoreSupport && flag910b;
    // index的步长为[1,0]或者[x,1,0]，是index expand场景，走scatteraddwithsorted, 超过16777216的FP32无法精准表示整数
    bool expandFlag = aicore910b && ((selfDimNum == TWO_DIM && dimFinal != 1)
                      || (selfDimNum == THREE_DIM && indexShape[0] * indexShape[1] < MAX_EXACT_FLOAT && dimFinal != TWO_DIM))
                      && strides[selfDimNum + NEG_TWO] == 1 && strides[selfDimNum + NEG_ONE] == 0 && shape.GetDimNum() == 1;
    if (expandFlag) {
        auto scatterRes = DoScatterAddWithSorted(selfContiguous, index, srcContiguous, out, reduction, dimFinal, executor);
        CHECK_COND(scatterRes != nullptr, ACLNN_ERR_INNER_NULLPTR, "DoScatterAddWithSorted failed!");
        return ACLNN_SUCCESS;
    }

    const aclTensor *indexContiguous = InitializeTensor(index, executor);
    CHECK_COND(indexContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR, "InitializeTensor index failed!");

    std::vector<int64_t> perm(selfDimNum);
    for (size_t i = 0; i < selfDimNum; ++i) {
        perm[i] = i;
    }
    CHECK_COND(selfDimNum - 1 >= 0, ACLNN_ERR_PARAM_INVALID, "self dim num must greater than 0");
    std::swap(perm[dimFinal], perm[selfDimNum - 1]);
    auto valuePerm = executor->AllocIntArray(perm.data(), selfDimNum);
    CHECK_COND(valuePerm != nullptr, ACLNN_ERR_INNER_NULLPTR, "alloc intarray failed!");

    int64_t dimInput = dimFinal;
    // 310P统一走aicpu,无需Transpose
    bool needTranspose = (dimFinal != (selfDimNum - 1)) && aicoreSupport;
    if (needTranspose) {
        selfContiguous = l0op::Transpose(selfContiguous, valuePerm, executor);
        CHECK_COND(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR, "transpose self failed!");
        indexContiguous = l0op::Transpose(indexContiguous, valuePerm, executor);
        CHECK_COND(indexContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR, "transpose index failed!");
        srcContiguous = l0op::Transpose(srcContiguous, valuePerm, executor);
        CHECK_COND(srcContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR, "transpose src failed!");
        dimInput = selfDimNum - 1;
    }

    if (aicore910b) {
        // linear index，转换负数索引并输出int32索引
        indexContiguous = l0op::LinearIndex(indexContiguous, selfContiguous, -1, false, executor);
        CHECK_COND(indexContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR, "LinearIndex failed!");
    }

    bool isCopy = aicore910b && self->GetData() != out->GetData();
    const aclTensor *scatterRes = DoScatterElements(selfContiguous, indexContiguous, srcContiguous,
                                                    dimInput, isCopy, reduction, executor);
    CHECK_COND(scatterRes != nullptr, ACLNN_ERR_INNER_NULLPTR, "DoScatterElements failed!");

    scatterRes = needTranspose ? l0op::Transpose(scatterRes, valuePerm, executor) : scatterRes;
    CHECK_COND(scatterRes != nullptr, ACLNN_ERR_INNER_NULLPTR, "transpose output failed!");

    scatterRes = needUnsqueeze ? l0op::SqueezeNd(scatterRes, squeezeDim, executor) : scatterRes;
    CHECK_COND(scatterRes != nullptr, ACLNN_ERR_INNER_NULLPTR, "squeeze output failed!");

    // 转换scatter数据类型和out保持一致
    scatterRes = l0op::Cast(scatterRes, out->GetDataType(), executor);
    CHECK_COND(scatterRes != nullptr, ACLNN_ERR_INNER_NULLPTR, "cast output failed!");

    auto viewCopyResult = l0op::ViewCopy(scatterRes, out, executor);
    CHECK_COND(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR, "viewCopy output failed!");

    return ACLNN_SUCCESS;
}

static bool isScalarComplex(const aclScalar* data){
    return data->GetDataType() == op::DataType::DT_COMPLEX64 ||
           data->GetDataType() == op::DataType::DT_COMPLEX128;
}

static bool isTensorComplex(const aclTensor* data){
    return data->GetDataType() == op::DataType::DT_COMPLEX64 ||
           data->GetDataType() == op::DataType::DT_COMPLEX128;
}

static aclnnStatus ExecScatterGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index,
    const aclTensor *src, int64_t reduce, aclTensor *out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_COND(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR, "CREATE_EXECUTOR failed!");

    auto ret = ExecScatterBase(self, dim, index, src, reduce, out, uniqueExecutor.get());

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ret;
}

static aclnnStatus ExecScatterValueGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index,
    const aclScalar *value, int64_t reduce, aclTensor *out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_COND(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR, "CREATE_EXECUTOR failed!");

    // index为nullptr直接报错返回
    CHECK_COND(((self != nullptr) && (index != nullptr) && (value != nullptr) && (out != nullptr)),
               ACLNN_ERR_PARAM_NULLPTR, "self, index, value, out cannot be nullptr!");

    CHECK_COND(CheckValueDtypeValid(self, index, out), ACLNN_ERR_PARAM_INVALID, "CheckValueDtypeValid failed!");

    CHECK_COND(isTensorComplex(self) || !isScalarComplex(value), ACLNN_ERR_PARAM_INVALID,
               "When value is COMPLEX, the data type of self must be COMPLEX.");
    auto src = uniqueExecutor.get()->ConvertToTensor(value, self->GetDataType());
    if (self->GetViewShape().GetDimNum() != 0) {
        auto indexShape = GetTensorShape(index, uniqueExecutor.get());
        CHECK_COND(indexShape != nullptr, ACLNN_ERR_INNER_NULLPTR, "get indexShape failed!");
        src = l0op::BroadcastTo(src, indexShape, uniqueExecutor.get());
        CHECK_COND(src != nullptr, ACLNN_ERR_INNER_NULLPTR, "value broadcast failed!");
    }

    auto ret = ExecScatterBase(self, dim, index, src, reduce, out, uniqueExecutor.get());

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ret;
}

// 非inplace
aclnnStatus aclnnScatterGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index,
    const aclTensor *src, int64_t reduce, aclTensor *out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnScatter, DFX_IN(self, dim, index, src, reduce), DFX_OUT(out));
    return ExecScatterGetWorkspaceSize(self, dim, index, src, reduce, out, workspaceSize, executor);
}

aclnnStatus aclnnScatterValueGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index,
    const aclScalar *value, int64_t reduce, aclTensor *out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnScatterValue, DFX_IN(self, dim, index, value, reduce), DFX_OUT(out));
    return ExecScatterValueGetWorkspaceSize(self, dim, index, value, reduce, out, workspaceSize, executor);
}

aclnnStatus aclnnScatterAddGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index,
    const aclTensor *src, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnScatterAdd, DFX_IN(self, dim, index, src), DFX_OUT(out));
    return ExecScatterGetWorkspaceSize(self, dim, index, src, REDUCE_ADD, out, workspaceSize, executor);
}

aclnnStatus aclnnScatter(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnScatter);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnScatterValue(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnScatterValue);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnScatterAdd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnScatterAdd);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

// inplace
aclnnStatus aclnnInplaceScatterGetWorkspaceSize(aclTensor *selfRef, int64_t dim, const aclTensor *index,
    const aclTensor *src, int64_t reduce, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnInplaceScatter, DFX_IN(selfRef, dim, index, src, reduce), DFX_OUT(selfRef));
    auto out = const_cast<aclTensor*>(selfRef);
    return ExecScatterGetWorkspaceSize(selfRef, dim, index, src, reduce, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceScatterValueGetWorkspaceSize(aclTensor *selfRef, int64_t dim, const aclTensor *index,
    const aclScalar *value, int64_t reduce, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnInplaceScatterValue, DFX_IN(selfRef, dim, index, value, reduce), DFX_OUT(selfRef));
    auto out = const_cast<aclTensor*>(selfRef);
    return ExecScatterValueGetWorkspaceSize(selfRef, dim, index, value, reduce, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceScatter(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceScatter);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceScatterValue(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceScatterValue);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
