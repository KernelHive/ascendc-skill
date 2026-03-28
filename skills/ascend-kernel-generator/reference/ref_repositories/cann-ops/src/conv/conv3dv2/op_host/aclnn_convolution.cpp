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
 * \file aclnn_convolution.cpp
 * \brief
 */

#include "aclnn_convolution.h"

#include <map>
#include <memory>
#include <vector>
#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "convolution_l0.h"
#include "aclnn_kernels/transdata.h"
#include "cube_util_l2.h"

using namespace std;
using namespace op;
using namespace ge;
using namespace l0op;
namespace op {
const size_t BN_MIN_SUPPORT_DIMS_NUMS = 2;
const size_t MAX_SUPPORT_DIMS_NUMS = 8;
const int8_t FP16FP32_KEEP_DTYPE = -1;
const int8_t KEEP_DTYPE = 0;
const int8_t ALLOW_FP32_DOWN_PRECISION = 1;
const int8_t USE_FP16 = 2;
const int8_t USE_HF32 = 3;
static inline ge::AscendString ToString(const std::int64_t value) {
  return ge::AscendString(to_string(value).c_str());
}
struct ConvolutionOpInfo {
  op::DataType inputDtype;
  op::Format inputFormat;
  op::DataType weightDtype;
  op::Format weightFormat;
  op::DataType biasDtype;
  op::Format biasFormat;
  op::DataType outputDtype;
  op::Format outputFormat;
};
static const int64_t specialStride = 63;
static const int64_t specialChannelIndex = 3;
static const int64_t SMALL_CHANNEL = 4;
static const string REFLECTION_MODE = "constant";

// Conv3d
const size_t CONV_3D_DIM_SIZE = 5;
const size_t CONST_VALUE_TWO = 2;
const static uint64_t MAX_UINT16 = 65536;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> BIAS_SUPPORT_LIST = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> BIAS_SUPPORT_LIST_ASCEND310P = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

/**
* --------------------------------------L0函数注册机制start------------------------------------------------
* 以下逻辑支持将L0函数注册到一个map里，在各convXX类中实例化这个map
* L0FUNCTION类型代表通用的L0函数定义，作为map的value类型, 逻辑上相当于一个占位符
* XXX_FUNCTION类型代表不同L0类别的函数指针
* GET_FUNC_ID宏通过输入输出的类别、format确定一个唯一的ID，作为函数map的key
* REG_L0_FUNCTION宏，可将function注册进map
* FUNCTION_CALL进行实际的函数调用，此处调用了L0函数的适配器ConvL0Warper
*/
typedef void (*L0FUNCTION)();

typedef const aclTensor *(*CONV_FUNCTION)(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                          const aclIntArray *stride, const aclIntArray *padding,
                                          const aclIntArray *dilation, int groups, aclOpExecutor *executor);

typedef const aclTensor *(*CONV_WITHFLAG_FUNCTION)(const aclTensor *input, const aclTensor *weight,
                                                  const aclTensor *bias, const aclIntArray *stride,
                                                  const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                                  bool useHf32, aclOpExecutor *executor);

std::string CharToString(const char *a) {
  return std::string(a);
}

#define GET_FUNC_ID(inputDtype, inputFormat, outputDtype, outputFormat)           \
  (CharToString(op::ToString(inputDtype).GetString()) + CharToString(op::ToString(inputFormat).GetString()) + \
  CharToString(op::ToString(outputDtype).GetString()) + CharToString(op::ToString(outputFormat).GetString()))

#define REG_L0_FUNCTION(map, function, inputDtype, inputFormat, outputDtype, outputFormat) \
  ((map).emplace((GET_FUNC_ID((inputDtype), (inputFormat), (outputDtype), (outputFormat))), (L0FUNCTION(&(function)))))

#define FUNCTION_CALL(l0Functions, opInfo, input, weight, bias, stride, padding, dilation, transposed, outputPadding,  \
                      groups, useHf32, executor)                                                                       \
  ConvL0Warper(l0Functions, opInfo, input, weight, bias, stride, padding, dilation, transposed, outputPadding, groups, \
              useHf32, executor)

static const aclTensor *ConvL0Warper(map<string, L0FUNCTION> l0Functions, ConvolutionOpInfo &opInfo,
                                    const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                    const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                                    const bool transposed, const aclIntArray *outputPadding, const int64_t groups,
                                    bool useHf32, aclOpExecutor *executor) {
  const aclTensor *result = nullptr;

  std::string funcId = GET_FUNC_ID(opInfo.inputFormat, opInfo.inputDtype, opInfo.outputFormat, opInfo.outputDtype);
  if (l0Functions.find(funcId) == l0Functions.end()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Not support the given data type combination: "
        "inputDtype: %s, outputDtype: %s",
        op::ToString(opInfo.inputDtype).GetString(), op::ToString(opInfo.outputDtype).GetString());

    return result;
  }

  L0FUNCTION fn = l0Functions.at(funcId);

  OP_LOGI("The opInfo.inputDtype is %s", op::ToString(opInfo.inputDtype).GetString());
  if (opInfo.inputDtype == op::DataType::DT_FLOAT16 || opInfo.inputDtype == op::DataType::DT_BF16) {
    result = ((CONV_FUNCTION)fn)(input, weight, bias, stride, padding, dilation, groups, executor);
  } else {
    result = ((CONV_WITHFLAG_FUNCTION)fn)(input, weight, bias, stride, padding, dilation, groups, useHf32, executor);
  }
  return result;
}
/* --------------------------------------L0函数注册机制end------------------------------------------------ */

/* --------------------------------------公共check能力start------------------------------------------------ */

template <typename T>
static inline bool Equal(T a, T b) {
  return a == b;
}

template <typename T>
static inline bool Greater(T a, T b) {
  return a > b;
}

template <typename T>
static inline bool LessEqual(T a, T b) {
  return a <= b;
}

template <typename T>
static inline bool Less(T a, T b) {
  return a < b;
}

template <typename T, typename Func>
static inline bool Any(T value, Func f) {
  return false;
}
// 参数仅需满足任一参数列表判断
template <typename T, typename Func, typename... LIST>
static inline bool Any(T value, Func f, T compare, LIST... list) {
  bool result = f(value, compare);
  if (!result) {
    return Any(value, f, list...);
  }
  return true;
}

template <typename T, typename Func>
static inline bool All(T value, Func f) {
  return true;
}
// 参数需要满足所有参数列表判断
template <typename T, typename Func, typename... LIST>
static inline bool All(T value, Func f, T compare, LIST... list) {
  bool result = f(value, compare);
  if (result) {
    return Any(value, f, list...);
  }
  return false;
}

// param必须等于给定值
#define CHECK_PARAM_EQ(param, value)                                                                    \
  do {                                                                                                  \
    if ((param) != (value)) {                                                                           \
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected %s = %s, get %s", #param, op::ToString(value).GetString(), \
              op::ToString(param).GetString());                                                             \
      return ACLNN_ERR_PARAM_INVALID;                                                                   \
    }                                                                                                   \
  } while (0)

// param必须大于给定值
#define CHECK_PARAM_GT(param, boundary)                                                                    \
  do {                                                                                                     \
    if ((param) <= (boundary)) {                                                                           \
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected %s > %s, get %s", #param, op::ToString(boundary).GetString(), \
              op::ToString(param).GetString());                                                                \
      return ACLNN_ERR_PARAM_INVALID;                                                                      \
    }                                                                                                      \
  } while (0)

// param必须小于给定值
#define CHECK_PARAM_LT(param, boundary)                                                                    \
  do {                                                                                                     \
    if ((param) >= (boundary)) {                                                                           \
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected %s < %s, get %s", #param, op::ToString(boundary).GetString(), \
              op::ToString(param).GetString());                                                                \
      return ACLNN_ERR_PARAM_INVALID;                                                                      \
    }                                                                                                      \
  } while (0)

// param必须大于等于给定值
#define CHECK_PARAM_GTE(param, boundary)                                                                    \
  do {                                                                                                      \
    if ((param) < (boundary)) {                                                                             \
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected %s >= %s, get %s", #param, op::ToString(boundary).GetString(), \
              op::ToString(param).GetString());                                                                 \
      return ACLNN_ERR_PARAM_INVALID;                                                                       \
    }                                                                                                       \
  } while (0)
/**
* 定义CHECK系列宏，支持任意param和变长参数的比较
*/
// param满足等于参数列表之一
#define CHECK_PARAM_EQ_ONE(param, type, ...)                                                         \
  do {                                                                                               \
    if (!Any((param), Equal<type>, __VA_ARGS__)) {                                                   \
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected %s equals one of %s, get %s", #param, #__VA_ARGS__, \
              op::ToString(param).GetString());                                                          \
      return ACLNN_ERR_PARAM_INVALID;                                                                \
    }                                                                                                \
  } while (0)

// 参数Dtype在支持列表里
#define CHECK_PARAM_DTYPE_VALID(tensorDtype, supportList, retExpr)                                                         \
  if (!CheckType(tensorDtype, supportList)) { \
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s %s is not supported, should be in dtype support list %s.", \
      #tensorDtype, op::ToString(tensorDtype).GetString(), op::ToString(supportList).GetString());          \
    retExpr; \
  }

// 参数列表满足都等于value
#define CHECK_PARAM_ALL_EQ(value, type, ...)                                                                      \
  do {                                                                                                            \
    if (!All((value), Equal<type>, __VA_ARGS__)) {                                                                \
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected all of %s equal %s", #__VA_ARGS__, op::ToString(value).GetString()); \
      return ACLNN_ERR_PARAM_INVALID;                                                                             \
    }                                                                                                             \
  } while (0)

// 参数列表满足都大于等于value
#define CHECK_PARAM_ALL_GTE(value, type, ...)                                                                  \
  do {                                                                                                         \
    if (!All((value), LessEqual<type>, __VA_ARGS__)) {                                                         \
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected all of %s >= %s", #__VA_ARGS__, op::ToString(value).GetString()); \
      return ACLNN_ERR_PARAM_INVALID;                                                                          \
    }                                                                                                          \
  } while (0)

// param满足小于所有参数列表
#define CHECK_PARAM_LT_ALL(param, type, ...)                                                            \
  do {                                                                                                  \
    if (!All((param), Less<type>, __VA_ARGS__)) {                                                       \
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected %s less than all of %s, get %s", #param, #__VA_ARGS__, \
              op::ToString(param).GetString());                                                             \
      return ACLNN_ERR_PARAM_INVALID;                                                                   \
    }                                                                                                   \
  } while (0)

// param满足大于所有参数列表
#define CHECK_PARAM_GT_ALL(param, type, ...)                                                          \
  do {                                                                                                \
    if (!All((param), Greater<type>, __VA_ARGS__)) {                                                  \
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected %s greater all of %s, get %s", #param, #__VA_ARGS__, \
              op::ToString(param).GetString());                                                           \
      return ACLNN_ERR_PARAM_INVALID;                                                                 \
    }                                                                                                 \
  } while (0)

#define CHECK_RET_RELEASE(condition, impl, ret_value)           \
  do {                                                          \
    if (!(condition)) {                                         \
      OP_LOGE(ACLNN_ERR_INNER, "check the condition which is \"%s\" failed, except the condition is true.",\
              #condition);                                      \
      delete (impl);                                            \
      return ret_value;                                         \
    }                                                           \
  } while (false)

struct ConvParams {
  const aclTensor *input;
  const aclTensor *weight;
  const aclTensor *bias;
  const aclIntArray *stride;
  const aclIntArray *padding;
  const aclIntArray *dilation;
  const bool transposed;
  const aclIntArray *outputPadding;
  const int64_t groups;
  aclTensor *output;
  int8_t cubeMathType;
  uint64_t *workspaceSize;
  aclOpExecutor **executor;
};

struct TensorMeta {
public:
  op::Format format;
  op::DataType dataType;
  FVector<int64_t> shape;
  op::Shape tensorShape;
  TensorMeta() {}
  void SetFromTensor(const aclTensor *tensor) {
    if (tensor == nullptr) {
      return;
    }
    format = tensor->GetViewFormat();
    dataType = tensor->GetDataType();

    string formatStr = op::ToString(format).GetString();
    tensorShape = tensor->GetViewShape();
    shape = ToFVector(tensorShape);

    // 未定义的shape，默认为1，实际不会被用到，将所哟类型包括一起查询，未找到的为npos
    auto len = shape.size();
    auto npos = formatStr.npos;
    auto index = formatStr.find('N');
    n_ = (index == npos || index >= len) ? 1 : shape[index];

    index = formatStr.find('C');
    c_ = (index == npos || index >= len) ? 1 : shape[index];

    index = formatStr.find('D');
    d_ = (index == npos || index >= len) ? 1 : shape[index];

    index = formatStr.find('H');
    h_ = (index == npos || index >= len) ? 1 : shape[index];

    index = formatStr.find('W');
    w_ = (index == npos || index >= len) ? 1 : shape[index];

    index = formatStr.find('L');
    l_ = (index == npos || index >= len) ? 1 : shape[index];

    // formatStr.endswith('C')
    channelLast_ = formatStr.find('C') == formatStr.length() - 1;
  }
  explicit TensorMeta(const aclTensor *tensor) { this->SetFromTensor(tensor); }
  // candy access functions
  int64_t N() { return n_; }
  int64_t C() { return c_; }
  int64_t D() { return d_; }
  int64_t H() { return h_; }
  int64_t W() { return w_; }
  int64_t L() { return l_; }
  bool ChannelLast() { return channelLast_; }
  FVector<int64_t> ToFVector(op::Shape &shapeT) {
    FVector<int64_t> vShape;
    if (shapeT.GetDimNum() != 0) {
      size_t dimNum = shapeT.GetDimNum();
      for (size_t idx = 0; idx < dimNum; idx++) {
        int64_t tmpVal = shapeT.GetDim(idx);
        vShape.push_back(tmpVal);
      }
    }
    return vShape;
  }

private:
  int64_t n_ = 0;
  int64_t c_ = 0;
  int64_t d_ = 0;
  int64_t h_ = 0;
  int64_t w_ = 0;
  int64_t l_ = 0;
  bool channelLast_ = false;
};
op::DataType GetUpperFloatDataType(op::DataType a, op::DataType b) {
  OP_LOGD("The input dtype is %s and %s", op::ToString(a).GetString(), op::ToString(b).GetString());
  if (a == op::DataType::DT_DOUBLE || b == op::DataType::DT_DOUBLE) {
    return op::DataType::DT_DOUBLE;
  }
  if (a == op::DataType::DT_FLOAT || b == op::DataType::DT_FLOAT) {
    return op::DataType::DT_FLOAT;
  }
  if (a == op::DataType::DT_BF16 && b == op::DataType::DT_BF16) {
    return op::DataType::DT_BF16;
  }
  if (a == op::DataType::DT_FLOAT16 && b == op::DataType::DT_FLOAT16) {
    return op::DataType::DT_FLOAT16;
  }

  return op::DataType::DT_FLOAT;  // 注意，原则上a,b都是float类型，若不是，则默认用FP32计算
}
struct ConvMeta {
public:
  TensorMeta input;
  TensorMeta weight;
  TensorMeta bias;
  TensorMeta output;
  // stride、dilation 按照空间分布，3维DHW
  FVector<int64_t> stride;
  FVector<int64_t> dilation;
  FVector<int64_t> padding;
  FVector<int64_t> outputPadding;
  op::DataType calculatDataType;
  void FromParams(ConvParams &params) {
    input.SetFromTensor(params.input);
    weight.SetFromTensor(params.weight);
    output.SetFromTensor(params.output);
    if (params.bias) {
      bias.format = params.bias->GetViewFormat();
      bias.dataType = params.bias->GetDataType();
      bias.tensorShape = params.bias->GetViewShape();
      bias.shape = bias.ToFVector(bias.tensorShape);
    }

    stride = ToVector(params.stride);
    dilation = ToVector(params.dilation);
    padding = ToVector(params.padding);
    if (params.transposed) {
      outputPadding = ToVector(params.outputPadding);
    }
    calculatDataType = GetUpperFloatDataType(input.dataType, weight.dataType);
  }

private:
  FVector<int64_t> ToVector(const aclIntArray *array) {
    FVector<int64_t> v;
    if (array != nullptr) {
      for (uint64_t i = 0; i < array->Size(); ++i) {
        v.push_back((*array)[i]);
      }
    }
    return v;
  }
};

struct ConvEngine {
public:
  ConvParams params;
  // 存储输入输出的元数据，可被直接访问，避免多次调用Get函数带来性能损失
  ConvMeta meta;
  explicit ConvEngine(ConvParams &convParams) : params(convParams) { meta.FromParams(params); }
  FVector<int64_t> CalcOutputShape() { return InferShape(); }

private:
  FVector<int64_t> InferShape() {
    FVector<int64_t> output;
    FVector<int64_t> inputShape = meta.input.shape;
    int64_t inputSpaceDimIndex = meta.input.ChannelLast() ? 1 : 2;  // 空间维度在shape中的起始位置，C维度后置时为1，否则为2
    int64_t inputSpaceDimNum = meta.input.shape.size() - 2;  // 空间维度大小，3d为3
    FVector<int64_t> weightShape = meta.weight.shape;
    int64_t weightSpaceDimIndex = meta.weight.ChannelLast() ? 1 : 2;  // 空间维度在shape中的起始位置，C维度后置时为1，否则为2
    // step 1: put nOut in the first place of shape; for conv and transpose mode
    output.push_back(meta.input.N());
    int64_t cOut = meta.weight.N();
    // step 2: calc spaceDim size and push back to shape
    for (int64_t i = 0; i < inputSpaceDimNum; ++i) {
        int64_t xOut = (inputShape[i + inputSpaceDimIndex] + CONST_VALUE_TWO * meta.padding[i] - meta.dilation[i] *
                        (weightShape[i + weightSpaceDimIndex] - 1) - 1) / meta.stride[i] + 1;
        output.push_back(xOut);
    }
    // last step : put cOut in right place
    if (meta.input.ChannelLast()) {
      output.push_back(cOut);
    } else {
      output.insert(output.begin() + 1, cOut);
    }
    return output;
  }
};

/* --------------------------------------公共check能力end------------------------------------------------ */

static const std::initializer_list<op::DataType>& GetBiasDtypeSupportListBySocVersion()
{
  SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();

  return BIAS_SUPPORT_LIST;
}

static bool CheckPointWise(const aclIntArray *array, int64_t value)                          
  {
    for (uint64_t i = 0; i < array->Size(); ++i) {
      if ((*array)[i] != value) {
        return false;
      }
    }
    return true;
  }


static bool NeedPointWiseKernel(const aclTensor *weight, const aclIntArray *stride, const aclIntArray *padding, 
                                                            const aclIntArray *dilation, const int64_t groups)
{
  if (groups != 1) {
    return false;
  }
  if (!CheckPointWise(dilation, 1) || !CheckPointWise(stride, 1) || !CheckPointWise(padding, 0)) {
    return false;
  }

  auto weightShape = weight->GetViewShape();
  size_t dimNum = weightShape.GetDimNum();
  for (size_t idx = CONST_VALUE_TWO; idx < dimNum; ++idx) {
    if (weightShape.GetDim(idx) != 1) {
      return false;
    }
  }
  return true;
}

static bool PointWiseKernelBeyondLimits(const aclTensor *fmap)
{
  auto fmapShape = fmap->GetViewShape();
  uint64_t dihiwi = 1;
  for (size_t idx = CONST_VALUE_TWO; idx < CONV_3D_DIM_SIZE; ++idx) {
    dihiwi = dihiwi * fmapShape.GetDim(idx);
  }
  return dihiwi >= MAX_UINT16;
}

class ConvolutionChecker {
public:
  ConvolutionChecker() = default;
  virtual ~ConvolutionChecker() = default;
  virtual aclnnStatus Check(ConvEngine &engine) = 0;
};

class NullptrChecker : public ConvolutionChecker {
public:
  NullptrChecker() = default;
  ~NullptrChecker() override = default;
  aclnnStatus Check(ConvEngine &engine) {
    CHECK_RET(engine.params.input != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(engine.params.weight != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(engine.params.stride != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(engine.params.padding != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(engine.params.dilation != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(engine.params.output != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(engine.params.workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(engine.params.executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
  }
};

class DtypeChecker : public ConvolutionChecker {
public:
  DtypeChecker() = default;
  ~DtypeChecker() override = default;
  aclnnStatus Check(ConvEngine &engine) {
    DataType inputDtype = engine.meta.input.dataType;
    DataType weightDtype = engine.meta.weight.dataType;
    DataType biasDtype = inputDtype;
    if (engine.params.bias != nullptr) {
      biasDtype = engine.meta.bias.dataType;
      CHECK_PARAM_DTYPE_VALID(biasDtype, GetBiasDtypeSupportListBySocVersion(), return ACLNN_ERR_PARAM_INVALID);
    }
    DataType outputDtype = engine.meta.output.dataType;

    auto dtypeSupportList = GetDtypeSupportListBySocVersion();
    CHECK_PARAM_DTYPE_VALID(inputDtype, dtypeSupportList, return ACLNN_ERR_PARAM_INVALID);
    CHECK_PARAM_DTYPE_VALID(weightDtype, dtypeSupportList, return ACLNN_ERR_PARAM_INVALID);
    CHECK_PARAM_DTYPE_VALID(outputDtype, dtypeSupportList, return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
  }
};

class DimChecker : public ConvolutionChecker {
public:
  DimChecker() = default;
  ~DimChecker() override = default;
  aclnnStatus CheckDim(const string &inStr, size_t inDim) {
    if (inDim != CONV_3D_DIM_SIZE) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expect %s equals 5 for conv3d, get %s",
              inStr.c_str(), to_string(inDim).c_str());
      return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
  }

  aclnnStatus Check(ConvEngine &engine) {
    size_t inputDim = engine.meta.input.shape.size();
    const string inputDimStr = "inputDim";
    aclnnStatus ret = CheckDim(inputDimStr, inputDim);
    if (ret != ACLNN_SUCCESS) {
      return ret;
    }

    size_t weightDim = engine.meta.weight.shape.size();
    const string weightDimStr = "weightDim";
    ret = CheckDim(weightDimStr, weightDim);
    if (ret != ACLNN_SUCCESS) {
      return ret;
    }

    size_t outputDim = engine.meta.output.shape.size();
    const string outputDimStr = "outputDim";
    ret = CheckDim(outputDimStr, outputDim);
    if (ret != ACLNN_SUCCESS) {
      return ret;
    }

    CHECK_PARAM_EQ(weightDim, inputDim);
    CHECK_PARAM_EQ(outputDim, inputDim);

    if (engine.params.bias != nullptr) {
      size_t biasDim = engine.meta.bias.shape.size();
      size_t biasSize = 0;
      size_t groupsValue = engine.params.groups;
      size_t weightNValue = engine.meta.weight.N();
      size_t weightCValue = engine.meta.weight.C();
      if (biasDim > 0) {
        biasSize = engine.meta.bias.shape[0];
      }
      // 如果是非transpose场景, bias的维度数必须为1维, 维度大小必须为 weight N
      if (biasDim != 1 || biasSize != weightNValue) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Given weight of size %s, expected bias to be 1-dimensional with %ld elements, "
                "but got bias of size %s instead",
                op::ToString(engine.meta.weight.tensorShape).GetString(), weightNValue,
                op::ToString(engine.meta.bias.tensorShape).GetString());
        return ACLNN_ERR_PARAM_INVALID;
      }
    }

    auto strideSize = engine.meta.stride.size();
    CHECK_PARAM_EQ(strideSize, inputDim - 2);

    auto dilationSize = engine.meta.dilation.size();
    CHECK_PARAM_EQ(dilationSize, inputDim - 2);

    auto paddingSize = engine.meta.padding.size();
    CHECK_PARAM_EQ(paddingSize, inputDim - 2);

    return ACLNN_SUCCESS;
  };
};

class FormatChecker : public ConvolutionChecker {
public:
  FormatChecker() = default;
  ~FormatChecker() override = default;
  aclnnStatus Check(ConvEngine &engine) {
    size_t inputDimNum = engine.meta.input.shape.size();
    auto inputFormat = engine.meta.input.format;
    auto weightFormat = engine.meta.weight.format;
    auto outputFormat = engine.meta.output.format;

    // conv3d convtranspose3d, input weight output 支持 NCDHW NDHWC
    CHECK_PARAM_EQ_ONE(inputFormat, op::Format, Format::FORMAT_NCDHW, Format::FORMAT_NDHWC);
    CHECK_PARAM_EQ_ONE(weightFormat, op::Format, Format::FORMAT_NCDHW, Format::FORMAT_NDHWC);
    CHECK_PARAM_EQ_ONE(outputFormat, op::Format, Format::FORMAT_NCDHW, Format::FORMAT_NDHWC);

    // 输入和输出format要求必须一致
    if (inputFormat != outputFormat) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
              "expected input format the same as output format. but get input format: %s, output format: %s",
              op::ToString(inputFormat).GetString(), op::ToString(outputFormat).GetString());
      return ACLNN_ERR_PARAM_INVALID;
    }
    // bias format不支持 NHWC
    if (engine.params.bias != nullptr) {
        auto biasFormat = engine.meta.bias.format;
        CHECK_PARAM_EQ_ONE(biasFormat, op::Format, Format::FORMAT_NCL, Format::FORMAT_NCHW,
            Format::FORMAT_NCDHW, Format::FORMAT_ND);
    }

    return ACLNN_SUCCESS;
  };
};

class ValueChecker : public ConvolutionChecker {
public:
  ValueChecker() = default;
  ~ValueChecker() override = default;

  aclnnStatus Check(ConvEngine &engine) {
    if (CheckShape(engine.meta.input, engine.meta.weight) != ACLNN_SUCCESS) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "shape check failed");
      return ACLNN_ERR_PARAM_INVALID;
    }
    // check stride
    if (CheckVectorValueGt0(engine.meta.stride) != ACLNN_SUCCESS) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "stride check failed");
      return ACLNN_ERR_PARAM_INVALID;
    }
    // check dilation
    if (CheckVectorValueGt0(engine.meta.dilation) != ACLNN_SUCCESS) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dilation check failed");
      return ACLNN_ERR_PARAM_INVALID;
    }
    // check pad
    if (CheckPad(engine.meta.input, engine.meta.weight, engine.meta.stride, engine.meta.dilation, engine.meta.padding) 
                != ACLNN_SUCCESS) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "pad check failed");
      return ACLNN_ERR_PARAM_INVALID;
    }
    // check channel_value (bias, groups)
    if (engine.params.groups <= 0) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected groups > 0, get %ld.", engine.params.groups);
      return ACLNN_ERR_PARAM_INVALID;
    }

    // check channel and groups
    int64_t inChannel = engine.meta.input.C();
    int64_t outChannel = -1L;

    outChannel = engine.meta.weight.N();
    if (engine.meta.weight.C() * engine.params.groups != inChannel) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "expected input channel equal to filter channel * groups. "
            "get input channel %ld, filter channel %ld, groups %ld.",
            inChannel, engine.meta.weight.C(), engine.params.groups);
        return ACLNN_ERR_PARAM_INVALID;
    }
    

    if (engine.meta.weight.N() % engine.params.groups != 0) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
              "expected weight 1st dim divisible by groups (including transpose mode), get weight %ld, groups %ld",
              engine.meta.weight.N(), engine.params.groups);
      return ACLNN_ERR_PARAM_INVALID;
    }

    if (engine.params.bias != nullptr) {
      if (CheckConvBias(engine.meta.bias, engine.meta.input, outChannel) != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "check bias failed");
        return ACLNN_ERR_PARAM_INVALID;
      }
    }

    return ACLNN_SUCCESS;
  };

private:
  /** 空tensor判断逻辑
  * input:
  * 在ValueChecker时，保证加上pad后，空间维度也大于0
  * weight: Cout和K不为0，在ValueChecker已完成校验
  */

  static aclnnStatus CheckShape(TensorMeta &input, TensorMeta &weight) {
    // check n c
    int64_t inputShapeN = input.N();
    int64_t inputShapeC = input.C();
    int64_t weightShapeN = weight.N();
    int64_t weightShapeC = weight.C();
    CHECK_PARAM_ALL_GTE(0L, int64_t, inputShapeN, inputShapeC, weightShapeC);
    CHECK_PARAM_GT(weightShapeN, 0L);

    // check space(d h w or l)
    FVector<int64_t> inputShape = input.shape;
    bool inputChannleLast = input.ChannelLast();
    int64_t inputSpaceDimIndex = inputChannleLast ? 1 : 2;  // 空间维度在shape中的起始位置，C维度后置时为1，否则为2
    size_t inputSpaceDimNum = input.shape.size() - 2;  // 空间维度大小，3d为3
    if (inputSpaceDimNum > 1) {
      CHECK_PARAM_GT(weightShapeC, 0L); // NCHW NCDHW 不支持weight的空tensor输入
    }
    FVector<int64_t> weightShape = weight.shape;
    bool weightChannleLast = weight.ChannelLast();
    int64_t weightSpaceDimIndex = weightChannleLast ? 1 : 2;  // 空间维度在shape中的起始位置，C维度后置时为1，否则为2

    // 如果N、C其中有1个为0，则返回空tensor。允许L、H、W存在0
    if (inputShapeN == 0 || inputShapeC == 0) {
      return ACLNN_SUCCESS;
    }

    // 假设是NCL，判断L的值。假设是NCHW，判断HW的值
    for (size_t i = 0; i < inputSpaceDimNum; ++i) {
      int64_t inputShapeSpace = inputShape[i + inputSpaceDimIndex];  // 空间维度的值
      if (inputShapeSpace <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected input %ldth dim > 0, but get %ld", i + inputSpaceDimIndex + 1,
                inputShapeSpace);
        return ACLNN_ERR_PARAM_INVALID;
      }
      int64_t weightShapeSpace = weightShape[i + weightSpaceDimIndex];  // 空间维度的值
      if (weightShapeSpace <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected weight %ldth dim > 0, but get %ld", i + weightSpaceDimIndex + 1,
                weightShapeSpace);
        return ACLNN_ERR_PARAM_INVALID;
      }
    }
    return ACLNN_SUCCESS;
  }

  static inline aclnnStatus CheckVectorValueGt0(FVector<int64_t> &param) {
    for (size_t i = 0; i < param.size(); ++i) {
      if (param[i] <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected %ldth value > 0, but get %ld", i + 1, param[i]);
        return ACLNN_ERR_PARAM_INVALID;
      }
    }
    return ACLNN_SUCCESS;
  }

  static aclnnStatus CheckPad(TensorMeta &input, TensorMeta &weight, FVector<int64_t> &stride,
                              FVector<int64_t> &dilation, FVector<int64_t> &padding) {
    FVector<int64_t> inputShape = input.shape;
    bool inputChannleLast = input.ChannelLast();
    int64_t inputSpaceDimIndex = inputChannleLast ? 1 : 2;  // 空间维度在shape中的起始位置，C维度后置时为1，否则为2
    size_t inputSpaceDimNum = input.shape.size() - 2;  // 空间维度大小，3d为3
    FVector<int64_t> weightShape = weight.shape;
    bool weightChannleLast = weight.ChannelLast();
    int64_t weightSpaceDimIndex = weightChannleLast ? 1 : 2;  // 空间维度在shape中的起始位置，C维度后置时为1，否则为2

    auto newpad = padding;
    for (size_t i = 0; i < inputSpaceDimNum; ++i) {
      auto inputShapeValue = inputShape[i + inputSpaceDimIndex];
      auto weightShapeValue = weightShape[i + weightSpaceDimIndex];
      auto strideValue = stride[i];
      auto paddingValueFront = padding[i];
      auto dilationValue = dilation[i];

      // check input shape after pad only for conv
      int64_t inputShapeValueAfterPad =
          (inputShapeValue + paddingValueFront * 2 - dilationValue * (weightShapeValue - 1) - 1);
          CHECK_PARAM_GTE(inputShapeValueAfterPad, 0);
    }

    return ACLNN_SUCCESS;
  }

  static aclnnStatus CheckConvBias(TensorMeta &bias, TensorMeta &input, int64_t outChannel) {
    auto biasShape = bias.shape;
    size_t biasDimNum = biasShape.size();

    // the index of C in Bias
    size_t idx_c = 0;
    if (biasDimNum != 1) {
      std::string str(op::ToString(input.format).GetString());
      idx_c = str.find('C');
    }

    for (size_t i = 0; i < biasDimNum; i++) {
      if (i == idx_c) {
        auto biasCout = biasShape[i];
        if (biasCout != outChannel) {
          OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected input bias size of dim_c[%ld] equal to out_channels[%ld].",
                  biasCout, outChannel);
          return ACLNN_ERR_PARAM_INVALID;
        }
      } else {
        if (biasShape[i] != 1) {
          OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected input bias size of none channel equal to 1.");
          return ACLNN_ERR_PARAM_INVALID;
        }
      }
    }

    return ACLNN_SUCCESS;
  }
};

class ConvXdChecker : public ConvolutionChecker {
public:
  ConvXdChecker() = default;
  ~ConvXdChecker() override = default;
  aclnnStatus Check(ConvEngine &engine) {
    FVector<int64_t> outputShape = engine.CalcOutputShape();
    for (size_t i = 0; i < outputShape.size(); i++) {
      if (outputShape[i] != engine.meta.output.shape[i]) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected output %ldth dim equal %ld, get %ld", i + 1, outputShape[i],
                engine.meta.output.shape[i]);
        return ACLNN_ERR_PARAM_INVALID;
      }
    }

    return ACLNN_SUCCESS;
  }
};

class HardwareLimitChecker : public ConvolutionChecker {
public:
  HardwareLimitChecker() = default;
  ~HardwareLimitChecker() override = default;
  aclnnStatus Check(ConvEngine &engine) {
    DataType outputDtype = engine.meta.output.dataType;
    DataType upperDtype = engine.meta.calculatDataType;
    CHECK_RET(CheckCubeMathType(upperDtype, engine.params.cubeMathType), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
  }
};


class TemporarySoftwareLimitChecker : public ConvolutionChecker {
public:
  TemporarySoftwareLimitChecker() = default;
  ~TemporarySoftwareLimitChecker() override = default;
  aclnnStatus Check(ConvEngine &engine) {
    size_t inputDim = engine.meta.input.shape.size();
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    switch (socVersion) {
      case SocVersion::ASCEND910B:
        break;
      default: {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "support for %s is not implemented", op::ToString(socVersion).GetString());
        return ACLNN_ERR_PARAM_INVALID;
      }
    }
    return ACLNN_SUCCESS;
  }
};

static aclnnStatus CheckConvParams(ConvEngine &engine) {
  std::vector<unique_ptr<ConvolutionChecker>> checkList;
  // math level check
  // common checkers: nullptr, dims, format
  checkList.push_back(make_unique<NullptrChecker>());
  checkList.push_back(make_unique<DtypeChecker>());
  checkList.push_back(make_unique<DimChecker>());
  checkList.push_back(make_unique<FormatChecker>());
  checkList.push_back(make_unique<ValueChecker>());
  // different conv checkers: infershape and so on
  checkList.push_back(make_unique<ConvXdChecker>());

  // implement level check
  // hardware limit checkers:double conv, fp32 conv in 1980...
  checkList.push_back(make_unique<HardwareLimitChecker>());
  // temporary software limitation checkers: 3d conv
  checkList.push_back(make_unique<TemporarySoftwareLimitChecker>());

  for (auto &checker : checkList) {
    aclnnStatus ret = checker->Check(engine);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
  }
  return ACLNN_SUCCESS;
}

// 实现公共数据预处理，将数据准备为L0可接受的形式
static aclnnStatus CommonPreProcess(const aclTensor *&input, const aclTensor *&weight, const aclTensor *&bias,
                                    const int64_t groups, const bool transposed, const ConvolutionOpInfo &opInfo,
                                    bool changeFormat, bool contiguous, aclOpExecutor *executor) {
  // 非连续转连续 + cast + transdata
  // input
  auto contiguousInput = input;
  auto contiguousWeight = weight;
  auto contiguousBias = bias;
  if (contiguous) {
    contiguousInput = l0op::Contiguous(input, executor);
    CHECK_RET(contiguousInput != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (op::GetPrimaryFormat(weight->GetStorageFormat()) != op::Format::FORMAT_FRACTAL_Z) {
        contiguousWeight = l0op::Contiguous(weight, executor);
        CHECK_RET(contiguousWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (bias != nullptr) {
      contiguousBias = l0op::Contiguous(bias, executor);
      CHECK_RET(contiguousBias != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
  }
  // cast
  auto castedInput = l0op::Cast(contiguousInput, opInfo.inputDtype, executor);
  CHECK_RET(castedInput != nullptr, ACLNN_ERR_INNER_NULLPTR);

  if (changeFormat) {
    // input format transdata
    input = l0op::TransData(castedInput, opInfo.inputFormat, groups, executor);
    CHECK_RET(input != nullptr, ACLNN_ERR_INNER_NULLPTR);
  } else {
    input = castedInput;
  }
  // weight
  // cast
  auto castedWeight = l0op::Cast(contiguousWeight, opInfo.weightDtype, executor);
  CHECK_RET(castedWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);

  if (changeFormat) {
    // weight format transdata
    weight = l0op::TransData(castedWeight, opInfo.weightFormat, groups, executor);
    CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
  } else {
    weight = castedWeight;
  }

  // bias
  if (bias != nullptr) {
    // cast
    auto castBias = l0op::Cast(contiguousBias, opInfo.biasDtype, executor);
    CHECK_RET(castBias != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // transdata
    bias = castBias;
  }

  return ACLNN_SUCCESS;
}

namespace {
  const int64_t DIM_DHW_NUM = 3;
  const int64_t CI_DIM_CO_CI_DHW_INDEX = 1;
  const int64_t D_DIM_NCDHW_INDEX = 2;
  const int64_t H_DIM_NCDHW_INDEX = 3;
  const int64_t W_DIM_NCDHW_INDEX = 4;
  struct BatchMatmulInput {
    const aclTensor *leftData;
    const aclTensor *rightData;
    const aclTensor *biasData;
    const aclTensor *outputData;
    bool isLeftTranspose;
    bool isRightTranspose;
  };
}

// 实现公共数据后处理，将数据转换为L2输出，但并不做viewcopy
static aclnnStatus CommonPostProcess(const int64_t groups, bool changeFormat, const aclTensor *output,
                                    const aclTensor *&convOut, aclOpExecutor *executor) {
  // output format transdata
  auto formatOutput = changeFormat ? l0op::TransData(convOut, output->GetStorageFormat(), groups, executor) : convOut;
  CHECK_RET(formatOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // output cast
  auto castedOutput = l0op::Cast(formatOutput, output->GetDataType(), executor);
  CHECK_RET(castedOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);

  convOut = castedOutput;

  return ACLNN_SUCCESS;
}

void GetConvolutionOpDtype(const aclTensor *input, const aclTensor *weight, const aclTensor *bias, aclTensor *output,
                          struct ConvolutionOpInfo &opInfo, const bool transposed, int8_t cubeMathType) {
  SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
  DataType upperDtype = GetUpperFloatDataType(input->GetDataType(), weight->GetDataType());
  upperDtype = CalcPromoteTypeCubemathtype(upperDtype, cubeMathType);

  opInfo.outputDtype = upperDtype;

  // USE_FP16场景必走FP16， 因此必须转为fp16实现
  if (cubeMathType == USE_FP16) {
    opInfo.outputDtype = op::DataType::DT_FLOAT16; // 目前底层二进制暂不支持16进32出的场景，故设为FP16运算
  }

  opInfo.inputDtype = upperDtype;
  opInfo.weightDtype = upperDtype;
  if (bias != nullptr) {
    opInfo.biasDtype = upperDtype;
    // 因为bias二进制不支持为BF16，所以得转成FP32
    if (upperDtype == op::DataType::DT_BF16) {
      OP_LOGD("Since bias does not support BF16, change the dtype of bias to fp32.");
      opInfo.biasDtype = op::DataType::DT_FLOAT;
    }
  }
}

void GetConvolution3dOpFormat(struct ConvolutionOpInfo &opInfo) {
  opInfo.weightFormat = Format::FORMAT_FRACTAL_Z_3D;
  opInfo.inputFormat = Format::FORMAT_NDC1HWC0;
  opInfo.outputFormat = Format::FORMAT_NDC1HWC0;
  opInfo.biasFormat = Format::FORMAT_ND;
}

void GetConv3dOpInfo(const aclTensor *input, const aclTensor *weight, const aclTensor *bias, aclTensor *output,
                  struct ConvolutionOpInfo &opInfo, const bool transposed, int64_t groups, const aclIntArray *stride,
                  const aclIntArray *padding, const aclIntArray *dilation, int8_t cubeMathType) {
  GetConvolutionOpDtype(input, weight, bias, output, opInfo, transposed, cubeMathType);
  GetConvolution3dOpFormat(opInfo);
}

static inline const aclTensor *ViewWithShape(const aclTensor *tensor, const op::Shape &shape, aclOpExecutor *executor) {
  if (shape == tensor->GetViewShape() && shape == tensor->GetStorageShape()) {
    return tensor;
  }
  return executor->CreateView(tensor, shape, tensor->GetViewOffset());
}

bool EmptyTensor(const aclTensor *inputTensor, const aclTensor *outputTensor) {
  TensorMeta inputTensorMeta(inputTensor);
  TensorMeta outputTensorMeta(outputTensor);
  return (inputTensorMeta.N() * inputTensorMeta.C() == 0) ||
        (outputTensorMeta.D() * outputTensorMeta.H() * outputTensorMeta.W() * outputTensorMeta.L() == 0);
}

class ConvolutionImpl {
public:
  virtual aclnnStatus PreProcess() = 0;
  virtual aclnnStatus Impl() = 0;
  virtual aclnnStatus PostProcess() = 0;
  ConvolutionImpl(const aclTensor *inputParam, const aclTensor *weightParam, const aclTensor *biasParam,
                  const aclIntArray *strideParam, const aclIntArray *paddingParam, const aclIntArray *dilationParam,
                  const bool transposedParam, const aclIntArray *outputPaddingParam, const int64_t groupsParam,
                  aclTensor *outputParam, bool useHf32Param, int8_t cubeMathTypeParam, aclOpExecutor *executorParam)
      : input(inputParam),
        weight(weightParam),
        bias(biasParam),
        stride(strideParam),
        padding(paddingParam),
        dilation(dilationParam),
        transposed(transposedParam),
        outputPadding(outputPaddingParam),
        groups(groupsParam),
        output(outputParam),
        useHf32(useHf32Param),
        cubeMathType(cubeMathTypeParam),
        executor(executorParam){};
  virtual ~ConvolutionImpl() {
    input = nullptr;
    weight = nullptr;
    bias = nullptr;
    stride = nullptr;
    padding = nullptr;
    dilation = nullptr;
    outputPadding = nullptr;
    output = nullptr;
    executor = nullptr;
  };

protected:
  const aclTensor *input;
  const aclTensor *weight;
  const aclTensor *bias;
  const aclIntArray *stride;
  const aclIntArray *padding;
  const aclIntArray *dilation;
  const bool transposed;
  const aclIntArray *outputPadding;
  const int64_t groups;
  aclTensor *output;
  const bool useHf32;
  int8_t cubeMathType;
  uint64_t *workspaceSize = nullptr;
  aclOpExecutor *executor;
  const aclTensor *convOut = nullptr;
  ConvolutionOpInfo opInfo = {};  // 用于提前计算所有前后处理相关的format、dtype等信息
  map<string, L0FUNCTION> l0Functions;
};

#define CONV_CONSTRUCTOR(type)                                                                                        \
  Conv##type##Impl(const aclTensor *inputParam, const aclTensor *weightParam, const aclTensor *biasParam,             \
                  const aclIntArray *strideParam, const aclIntArray *paddingParam, const aclIntArray *dilationParam, \
                  const bool transposedParam, const aclIntArray *outputPaddingParam, const int64_t groupsParam,      \
                  aclTensor *outputParam, bool useHf32Param, int8_t cubeMathTypeParam, aclOpExecutor *executorParam) \
      : ConvolutionImpl(inputParam, weightParam, biasParam, strideParam, paddingParam, dilationParam,                 \
                        transposedParam, outputPaddingParam, groupsParam, outputParam, useHf32Param,                  \
                        cubeMathTypeParam, executorParam) {}


class Conv3dImpl : public ConvolutionImpl {
public:
  CONV_CONSTRUCTOR(3d)

  aclnnStatus PreProcess() {
    REG_L0_FUNCTION(l0Functions, Conv3dv26HdFp16, op::Format::FORMAT_NDC1HWC0, op::DataType::DT_FLOAT16,
                    op::Format::FORMAT_NDC1HWC0, op::DataType::DT_FLOAT16);
    REG_L0_FUNCTION(l0Functions, Conv3dv26HdFp32, op::Format::FORMAT_NDC1HWC0, op::DataType::DT_FLOAT,
                    op::Format::FORMAT_NDC1HWC0, op::DataType::DT_FLOAT);
    REG_L0_FUNCTION(l0Functions, Conv3dv26HdBf16, op::Format::FORMAT_NDC1HWC0, op::DataType::DT_BF16,
                    op::Format::FORMAT_NDC1HWC0, op::DataType::DT_BF16);
    REG_L0_FUNCTION(l0Functions, Conv3dv2NCDHWFp16, op::Format::FORMAT_NCDHW, op::DataType::DT_FLOAT16,
                    op::Format::FORMAT_NCDHW, op::DataType::DT_FLOAT16);
    REG_L0_FUNCTION(l0Functions, Conv3dv2NCDHWFp32, op::Format::FORMAT_NCDHW, op::DataType::DT_FLOAT,
                    op::Format::FORMAT_NCDHW, op::DataType::DT_FLOAT);
    REG_L0_FUNCTION(l0Functions, Conv3dv2NCDHWBf16, op::Format::FORMAT_NCDHW, op::DataType::DT_BF16,
                    op::Format::FORMAT_NCDHW, op::DataType::DT_BF16);
    GetConv3dOpInfo(input, weight, bias, output, opInfo, transposed, groups, stride, padding, dilation, cubeMathType);

    // 判断是否是PointWise卷积
    isPointWiseKernelFlag = IsSupportConv3DToConv3DV2() &&
                            NeedPointWiseKernel(weight, stride, padding, dilation, groups) &&
                            !PointWiseKernelBeyondLimits(input);
    // PointWise卷积，biasDtype只能为FLOAT32
    if (isPointWiseKernelFlag) {
      opInfo.biasDtype = op::DataType::DT_FLOAT;
      opInfo.weightFormat = Format::FORMAT_NCDHW;
      opInfo.inputFormat = Format::FORMAT_NCDHW;
      opInfo.outputFormat = Format::FORMAT_NCDHW;
      OP_LOGD("Entering PointWise branch.");
    }
    OP_LOGD("convolution aclnn op inputDtype: %s, outputDtype: %s, biasDtype: %s, useHf32: %d.",
            op::ToString(opInfo.inputDtype).GetString(), op::ToString(opInfo.outputDtype).GetString(),
            op::ToString(opInfo.biasDtype).GetString(), useHf32);
    // 调用静态函数PreProcess
    return CommonPreProcess(input, weight, bias, groups, transposed, opInfo, !isPointWiseKernelFlag, true, executor);
  };

  aclnnStatus Impl() {
    convOut = FUNCTION_CALL(l0Functions, opInfo, input, weight, bias, stride, padding, dilation, transposed,
                            outputPadding, groups, useHf32, executor);
    if (convOut == nullptr) {
      OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "conv3d raise an unknown error");
      return ACLNN_ERR_RUNTIME_ERROR;
    }
    return ACLNN_SUCCESS;
  };

  aclnnStatus PostProcess() {
    auto res = CommonPostProcess(groups, !isPointWiseKernelFlag, output, convOut, executor);
    CHECK_RET(res == ACLNN_SUCCESS, res);

    auto result = l0op::ViewCopy(convOut, output, executor);
    CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    return ACLNN_SUCCESS;
  };
  ~Conv3dImpl(){};

private:
  bool isPointWiseKernelFlag = false;
};

ConvolutionImpl *CreateConvolutionImpl(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                      const aclIntArray *stride, const aclIntArray *padding,
                                      const aclIntArray *dilation, const bool transposed, const bool tbc,
                                      const aclIntArray *outputPadding, const int64_t groups, int8_t cubeMathType,
                                      aclTensor *output, aclOpExecutor *executor) {
  // 存疑：是否按照原来的只看input dtype
  auto promoteType = op::PromoteType(input->GetDataType(), weight->GetDataType());
  if (bias != nullptr) {
    promoteType = op::PromoteType(promoteType, bias->GetDataType());
  }
  bool useHf32 = NeedCubeGoHF32(promoteType, cubeMathType);

  size_t inputDim = input->GetViewShape().GetDimNum();

  return new (std::nothrow) Conv3dImpl(input, weight, bias, stride, padding, dilation, transposed, outputPadding,
                                            groups, output, useHf32, cubeMathType, executor);
}
}  // namespace op

#ifdef __cplusplus
extern "C" {
#endif
aclnnStatus aclnnConvolutionGetWorkspaceSize(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                            const aclIntArray *stride, const aclIntArray *padding,
                                            const aclIntArray *dilation, bool transposed,
                                            const aclIntArray *outputPadding, const int64_t groups, aclTensor *output,
                                            int8_t cubeMathType, uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(
      aclnnConvolution,
      DFX_IN(input, weight, bias, stride, padding, dilation, transposed, outputPadding, groups, cubeMathType),
      DFX_OUT(output));
  // construct param and convolution engine
  ConvParams params = {input,         weight, bias,   stride,       padding,       dilation, transposed,
                      outputPadding, groups, output, cubeMathType, workspaceSize, executor};
  ConvEngine convEngine(params);
  // check param
  auto ret = CheckConvParams(convEngine);
  CHECK_RET_CODE(ret, "Check Param failed");

  auto uniqueExecutor = CREATE_EXECUTOR();
  // 创建OpExecutor
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  if (!EmptyTensor(input, output)) {
    ConvolutionImpl *convImpl =
        CreateConvolutionImpl(input, weight, bias, stride, padding, dilation, transposed, false, outputPadding, groups,
                              cubeMathType, output, uniqueExecutor.get());

    CHECK_RET_RELEASE(convImpl != nullptr, convImpl, ACLNN_ERR_INNER);

    ret = convImpl->PreProcess();
    CHECK_RET_RELEASE(ret == ACLNN_SUCCESS, convImpl, ret);

    ret = convImpl->Impl();
    CHECK_RET_RELEASE(ret == ACLNN_SUCCESS, convImpl, ret);

    ret = convImpl->PostProcess();
    CHECK_RET_RELEASE(ret == ACLNN_SUCCESS, convImpl, ret);

    delete convImpl;
  }

  *workspaceSize = (uniqueExecutor.get())->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnConvolution(void *workspace, const uint64_t workspaceSize, aclOpExecutor *executor,
                            aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnConvolution);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
