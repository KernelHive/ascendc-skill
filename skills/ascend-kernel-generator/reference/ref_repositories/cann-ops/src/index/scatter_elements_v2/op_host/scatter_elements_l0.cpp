/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file scatter_elements.cpp
 * \brief
 */
#include "scatter_elements_l0.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"


using namespace op;

static const int64_t AXIS_LIMIT = 8;                 // L0算子最大支持8维
static const int64_t DATA_MAX_AICORE = 2097152;      // data总数据量小于该值才可走aicore
static const int64_t INDICE_MAX_COMB = 30000;        // 可合轴情况下，indices总数据量小于该值可走aicore
static const int64_t INDICE_MAX_NO_COMB = 5500;      // 非可合轴情况下，indices总数据量小于该值可走aicore

static const int32_t FLOAT_CAST = 4;
static const int32_t HALF_UB = 2;
static const int32_t V2_MAX_LOOPS = 50;
// 一个32的block里面能装多少个元素   32 / sizeof(dtype) bytes   Ex: fp32 4 bytes -> block最多8个元素
static const int32_t BLOCK_8 = 32;
static const int32_t BLOCK_16 = 16;
static const int32_t BLOCK_32 = 8;

constexpr size_t UB_DATA_RATIO_LAST_AXIS = 2;
constexpr size_t BYTES_PER_BLOCK = 32;

static const int32_t UBSIZE_910BC = 192 * 1024;  // ASCEND910B 和 ASCEND910_93的UB size
static const int32_t UBSIZE_NORMAL = 256 * 1024; // 其余芯片的UB size

static map<const op::DataType, const int32_t> DATA_BLOCK_LEN = {
  {op::DataType::DT_INT8, BLOCK_8},   {op::DataType::DT_UINT8, BLOCK_8}, {op::DataType::DT_FLOAT16, BLOCK_16},
  {op::DataType::DT_FLOAT, BLOCK_32}, {op::DataType::DT_INT32, BLOCK_32}, {op::DataType::DT_BF16, BLOCK_16}
};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_INT8,
    op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> AICORE_910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_BF16};

namespace l0op {
OP_TYPE_REGISTER(ScatterElements);
OP_TYPE_REGISTER(ScatterElementsV2);

// 判断是否可以合轴： 当data_shape和indices_shape仅在axis取值不一致时可以合轴
bool CanCombineAxis(const aclTensor *data, const aclTensor *indices, int64_t axis)
{
    auto size_data = data->GetViewShape().GetDimNum();
    auto size_indices = indices->GetViewShape().GetDimNum();
    auto size = size_data > size_indices ? size_indices : size_data;
    for (size_t i = 0; i < size; i++) {
        if ((data->GetViewShape())[i] != (indices->GetViewShape())[i]) {
            if (i != axis) {
                return false;
            }
        }
    }
    return true;
}

bool CanCombineAxisV2(const aclTensor *data, const aclTensor *indices, const aclTensor * updates,
                      int64_t axis, int64_t dataDimSize)
{
    auto dataShape = data->GetViewShape();
    auto size_data = dataShape.GetDimNum();
    auto size_indices = indices->GetViewShape().GetDimNum();
    auto size_updates = updates->GetViewShape().GetDimNum();
    size_indices = size_indices > size_updates ? size_updates : size_indices;
    auto size = size_indices > size_data ? size_data : size_indices;
    for (size_t i = 0; i < size; i++) {
        if (dataShape[i] != (indices->GetViewShape())[i] || dataShape[i] != (updates->GetViewShape())[i]) {
            if (i != axis && i != 0 && i != dataDimSize - 1) {
                return false;
            }
        }
    }
    return true;
}

static int32_t GetUBSizeBytes() {
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    bool is910BC = (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93);
    int32_t ubSize = is910BC ? UBSIZE_910BC : UBSIZE_NORMAL;
    return ubSize;
}

// 每次data循环搬入次数不超过50时，ScatterElementsV2的性能占优，循环搬入次数 = 每段data使用核数 * 每个核的搬入次数
static bool IsScatterElementsV2Better(const aclTensor *data, const aclTensor *indices, int64_t axis) {
    auto dataDimNum = data->GetViewShape().GetDimNum();
    uint32_t tasks = 1;
    for (size_t i = 0; i < dataDimNum - 1; i++) {
        tasks *= indices->GetViewShape().GetDim(i);
    }
    uint32_t pieces = 1;
    uint32_t coreNum = GetCurrentPlatformInfo().GetVectorCoreNum();
    if (tasks < coreNum && coreNum > 0) {
        pieces = (tasks - 1) / coreNum + 1;
    }

    op::DataType dataType = data->GetDataType();
    int bytes = BYTES_PER_BLOCK / DATA_BLOCK_LEN[data->GetDataType()];
    if (dataType == op::DataType::DT_FLOAT16 || dataType == op::DataType::DT_BF16) {
        bytes += FLOAT_CAST;
    }
    int dataNum = GetUBSizeBytes() / HALF_UB / bytes; // 一个核UB上限是多少个数
    int inputOnePiece = (data->GetViewShape().GetDim(axis) - 1) / pieces + 1;
    int inputLoop = (inputOnePiece - 1) / dataNum + 1;
    OP_LOGD("ScatterElementsV2 checked: inputLoop=%u, pieces=%u.", inputLoop, pieces);
    if (pieces * inputLoop > V2_MAX_LOOPS) {
        OP_LOGD("Loops more than 50, do not use ScatterElementsV2.");
        return false;
    }
    return true;
}

// 仅在dtype为aicore支持(前面校验过) + 轴为末尾轴(-1 / size-1)进行校验
// 特定场景下校验是否走aicore更好。return True，继续往下走是否判断aicore    return False,确定走aicpu
static bool IsAicoreBetter(const aclTensor *data, const aclTensor *indices, int64_t axis, bool *aiCpuWithUbDataFlag) {
    int64_t dataDimNum = static_cast<int64_t>(data->GetViewShape().GetDimNum());
    // 判断是否为末尾轴
    *aiCpuWithUbDataFlag = false;
    if ((axis != -1 && axis != dataDimNum - 1)) {
        return false;
    }
    int32_t block = DATA_BLOCK_LEN[data->GetDataType()];             // 这个block中最多含有多少个元素
    int32_t dataWidth = data->GetViewShape().GetDim(axis);           // axis轴的数据量
    int32_t ubForData = GetUBSizeBytes() / UB_DATA_RATIO_LAST_AXIS;  // ub space for data

    // dataWidth <= maxDataShapeInDim
    int32_t maxDataShapeInDim = ubForData / BYTES_PER_BLOCK * block;  // max data shape that ub can hold
    OP_CHECK(dataWidth <= maxDataShapeInDim,
             OP_LOGD("dataShape[axis](%d) > maxDataShapeInDim(%d)", dataWidth, maxDataShapeInDim), return false);

    if (!CanCombineAxis(data, indices, axis)) {
        OP_CHECK(dataWidth >= block, OP_LOGD("dataShape[axis](%d) < block(%d)", dataWidth, block), return false);
        return true;
    } else {
        int32_t dataBlock = (dataWidth + block - 1) / block * block;  // datawidth向上取整, axis轴每block个一行要取多少行
        int32_t repeat = dataBlock / block;                           // 一行几个block
        int32_t repeatPerCore = 1;
        if (dataWidth != dataBlock) {
            while ((repeatPerCore * dataWidth) % block != 0) {
                repeatPerCore++;
            }
            int32_t actualDataInUB = repeatPerCore * repeat * BYTES_PER_BLOCK;  // actual ub space that data uses
            if (actualDataInUB > ubForData) {
                OP_LOGD("actualDataInUb(%d) > ubForData(%d) in hostapi ScatterElements.", actualDataInUB, ubForData);
                *aiCpuWithUbDataFlag = true;
                return false;
            }
        }
        return true;
    }
}

// 存在一个维度 shape a[i] < shape b[i], 返回True
static bool CompareTensorShape(const aclTensor *a, const aclTensor *b)
{
    auto aDimSize = a->GetViewShape().GetDimNum();
    auto bDimSize = b->GetViewShape().GetDimNum();
    aDimSize = aDimSize < bDimSize ? aDimSize : bDimSize;
    for (size_t i = 0; i < aDimSize; i++) {
        auto aDim = (a->GetViewShape())[i];
        auto bDim = (b->GetViewShape())[i];
        if (aDim < bDim) {
            return true;
        }
    }
    return false;
}

// 判断是否走ScatterElementsV2
static bool UseScatterElementsV2(const aclTensor *data, const aclTensor *indices, const aclTensor * updates, int64_t axis,
                                 int64_t dataDimSize, const std::string& reduction)
{
    // Ascend310P无二进制，统一走acipu
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    OP_CHECK(socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93,
        OP_LOGD("ScatterElementsV2 only support ASCEND910B"), return false);

    // 如果dtype不支持
    if (!CheckType(data->GetDataType(), AICORE_910B_DTYPE_SUPPORT_LIST)) {
        return false;
    }

    if (reduction == "mul") {
        return false;
    }

    if (axis != dataDimSize - 1 && axis != 0) {
        return false;
    }

    if (!CanCombineAxisV2(data, indices, updates, axis, dataDimSize)) {
        return false;
    }

    return IsScatterElementsV2Better(data, indices, axis);
}

// 判断是否走ScatterElements
static bool UseScatterElements(const aclTensor *data, const aclTensor *indices, const aclTensor * updates, int64_t axis,
                               const std::string& reduction)
{
    // Ascend310P无二进制，统一走acipu, ScatterElements目前仅在910A上使用
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    OP_CHECK(socVersion == SocVersion::ASCEND910,
        OP_LOGD("ScatterElements only support Aicore for ASCEND910"), return false);

    // 如果dtype不支持
    if (!CheckType(data->GetDataType(), AICORE_DTYPE_SUPPORT_LIST)) {
        return false;
    }

    // 如果index.shape < src.shape  aicore功能不支持
    if (CompareTensorShape(indices, updates)) {
        return false;
    }

    if (reduction == "mul") {
        return false;
    }

    // 判断是否为特殊场景时可以使用aicore IsAicoreBetter返回true，说明走aicore
    bool aiCpuWithUbDataFlag;
    if (IsAicoreBetter(data, indices, axis, &aiCpuWithUbDataFlag)) {
        return true;
    }

    // actualDataInUB > ubForData use aicpu
    if (aiCpuWithUbDataFlag) {
        return false;
    }

    // data总数据量 <= 该值才可走aicore
    if (data->Size() > DATA_MAX_AICORE) {
        return false;
    }

    int64_t indiceSize = indices->Size();
    if (CanCombineAxis(data, indices, axis)) {
        return indiceSize <= INDICE_MAX_COMB;
    } else {
        return indiceSize <= INDICE_MAX_NO_COMB;
    }
}

const aclTensor *ScatterElements(const aclTensor *data, const aclTensor *indices, const aclTensor *updates,
    int64_t axis, const std::string &reduction, aclOpExecutor *executor)
{
    L0_DFX(ScatterElements, data, indices, updates, axis, reduction);
    auto dataDimSize = static_cast<int64_t>(data->GetViewShape().GetDimNum());
    axis = axis >= 0 ? axis : axis + dataDimSize;   // axis保证为非负数
    if (dataDimSize > AXIS_LIMIT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Tensor data dimension size must be in range [0, 8]. Current size is %ld.",
            dataDimSize);
    }
    if (reduction != "none" && reduction != "add" && reduction != "mul") {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "reduction must be one of [none, add, mul]");
    }

    auto out = executor->AllocTensor(data->GetViewShape(), data->GetDataType(), data->GetViewFormat());
    CHECK_RET(out != nullptr, nullptr);
    if (UseScatterElementsV2(data, indices, updates, axis, dataDimSize, reduction)) {
        OP_LOGD("Use AICORE for ScatterElementsV2.");
        auto selfOut = const_cast<aclTensor*>(data);
        auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ScatterElementsV2, OP_INPUT(data, indices, updates), OP_OUTPUT(selfOut),
                                               OP_ATTR(axis, reduction));
        OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ScatterElementsV2AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
            return nullptr);
        return selfOut;
    } else if (UseScatterElements(data, indices, updates, axis, reduction)) {
        OP_LOGD("Use AICORE for ScatterElements.");
        auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ScatterElements, OP_INPUT(data, indices, updates), OP_OUTPUT(out),
                                               OP_ATTR(axis, reduction));
        OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ScatterElementsAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
            return nullptr);
    } else {
        OP_LOGD("not supported!");
        return nullptr;
    }

    return out;
}
}
