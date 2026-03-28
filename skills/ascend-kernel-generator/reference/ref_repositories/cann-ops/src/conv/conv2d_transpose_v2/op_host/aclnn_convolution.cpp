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
#include "convolution_util.h"

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
#include "op_api_def.h"
#include "add.h"
#include "broadcast_to.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "convolution.h"
#include "padv3.h"
#include "aclnn_kernels/reshape.h"
#include "squeeze.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "unsqueeze.h"
#include "cube_util_l2.h"
#include "matmul_util.h"

using namespace std;
using namespace op;
using namespace ge;
using namespace l0op;
namespace op {
static inline ge::AscendString ToString(const std::int64_t value) {
  return ge::AscendString(to_string(value).c_str());
}
}  // namespace op

static const int64_t specialStride = 63;
static const int64_t specialChannelIndex = 3;
static const int64_t SMALL_CHANNEL = 4;

static const string REFLECTION_MODE = "constant";

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

typedef const aclTensor *(*CONVTRANSPOSE_FUNCTION)(const aclTensor *input, const aclTensor *weight,
                                                   const aclTensor *bias, const aclIntArray *stride,
                                                   const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                                   const aclIntArray *outputPadding, aclOpExecutor *executor);

typedef const aclTensor *(*CONVTRANSPOSE_WITHFLAG_FUNCTION)(const aclTensor *input, const aclTensor *weight,
                                                            const aclTensor *bias, const aclIntArray *stride,
                                                            const aclIntArray *padding, const aclIntArray *dilation,
                                                            int groups, const aclIntArray *outputPadding, bool useHf32,
                                                            aclOpExecutor *executor);
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
    if (!transposed) {
      result = ((CONV_FUNCTION)fn)(input, weight, bias, stride, padding, dilation, groups, executor);
    } else {
      result =
          ((CONVTRANSPOSE_FUNCTION)fn)(input, weight, bias, stride, padding, dilation, groups, outputPadding, executor);
    }
  } else {
    if (!transposed) {
      result = ((CONV_WITHFLAG_FUNCTION)fn)(input, weight, bias, stride, padding, dilation, groups, useHf32, executor);
    } else {
      result = ((CONVTRANSPOSE_WITHFLAG_FUNCTION)fn)(input, weight, bias, stride, padding, dilation, groups,
                                                     outputPadding, useHf32, executor);
    }
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

// Conv1d, 2d, 3d
const size_t CONV_1D_DIM_SIZE = 3;
const size_t CONV_2D_DIM_SIZE = 4;
const size_t CONV_3D_DIM_SIZE = 5;
const size_t CONST_VALUE_TWO = 2;
const static uint64_t MAX_UINT16 = 65536;

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
  // stride、dilation 按照空间分布，3维DHW，2维HW，1维L
  FVector<int64_t> stride;
  FVector<int64_t> dilation;
  // padding outputpadding 按照方向维度分布，3维3个值，代表前后、上下、左右，2维度上下、左右，1维度左右
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

namespace {
  const size_t CONV_1D_PAD_DIM = 1;
  const size_t CONV_2D_PAD_DIM = 2;
  const size_t CONV_4D_PAD_DIM = 4;
  const size_t PAD_TOP_INDEX = 0;
  const size_t PAD_BOTTOM_INDEX = 1;
  const size_t PAD_LEFT_INDEX = 2;
  const size_t PAD_RIGHT_INDEX = 3;
}

// 本函数的目的是给conv1d制造1维的pad数组，给conv2d制造2维的pad数组，其他类型的conv保留原数组不变
static FVector<int64_t> ConstructPad(FVector<int64_t> &oldPad, FVector<int64_t> &inputShape)
{
  FVector<int64_t> newPad;
  if (inputShape.size() == CONV_1D_DIM_SIZE) {
    if (oldPad.size() == 1) {
      newPad = {oldPad[0] + oldPad[0]};
    } else if (oldPad.size() == CONV_2D_PAD_DIM) {
      newPad = {oldPad[0] + oldPad[1]};
    } else {
      newPad = {0};
    }
  } else if (inputShape.size() == CONV_2D_DIM_SIZE) {
      if (oldPad.size() == CONV_2D_PAD_DIM) {
        newPad = {(oldPad[0] + oldPad[0]), (oldPad[1] + oldPad[1])};
      } else if (oldPad.size() == CONV_4D_PAD_DIM) {
        newPad = {(oldPad[PAD_TOP_INDEX] + oldPad[PAD_BOTTOM_INDEX]),
                  (oldPad[PAD_LEFT_INDEX] + oldPad[PAD_RIGHT_INDEX])};
      } else {
        newPad = {0, 0};
      }
  } else {
    return oldPad;
  }
  return newPad;
}

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
    int64_t inputSpaceDimNum = meta.input.shape.size() - 2;  // 空间维度大小，1d卷积时为1，2d为2，3d为3
    FVector<int64_t> weightShape = meta.weight.shape;
    int64_t weightSpaceDimIndex = meta.weight.ChannelLast() ? 1 : 2;  // 空间维度在shape中的起始位置，C维度后置时为1，否则为2
    // step 1: put nOut in the first place of shape; for conv and transpose mode
    output.push_back(meta.input.N());
    int64_t cOut = meta.weight.N();
    // step 2: calc spaceDim size and push back to shape
    if (!params.transposed) {
      if (inputShape.size() == CONV_1D_DIM_SIZE) {
        cOut = (meta.weight.C() != 0) ? cOut : 0; // conv1D场景，若weight的cin为0，特殊处理cout也为0。3对应的是NCL的维度大小
      }
      if (inputShape.size() == CONV_1D_DIM_SIZE || inputShape.size() == CONV_2D_DIM_SIZE) {
        auto newPad = ConstructPad(meta.padding, inputShape);
        for (int64_t i = 0; i < inputSpaceDimNum; ++i) {
          int64_t xOut = (inputShape[i + inputSpaceDimIndex] + newPad[i] - meta.dilation[i] *
                         (weightShape[i + weightSpaceDimIndex] - 1) - 1) / meta.stride[i] +  1;
          output.push_back(xOut);
        }
      } else {
        for (int64_t i = 0; i < inputSpaceDimNum; ++i) {
          int64_t xOut = (inputShape[i + inputSpaceDimIndex] + CONV_2D_PAD_DIM * meta.padding[i] - meta.dilation[i] *
                         (weightShape[i + weightSpaceDimIndex] - 1) - 1) / meta.stride[i] + 1;
          output.push_back(xOut);
        }
      }
    } else {
      cOut = meta.weight.C() * params.groups;
      if (inputShape.size() == CONV_2D_DIM_SIZE) {
        auto newPad = ConstructPad(meta.padding, inputShape);
        for (int64_t i = 0; i < inputSpaceDimNum; ++i) {
          int64_t xOut = (inputShape[i + inputSpaceDimIndex] - 1) * meta.stride[i] - newPad[i] +
                         meta.dilation[i] * (weightShape[i + weightSpaceDimIndex] - 1) + meta.outputPadding[i] + 1;
          output.push_back(xOut);
        }
      } else {
        for (int64_t i = 0; i < inputSpaceDimNum; ++i) {
          int64_t xOut = (inputShape[i + inputSpaceDimIndex] - 1) * meta.stride[i] - 2 * meta.padding[i] +
                         meta.dilation[i] * (weightShape[i + weightSpaceDimIndex] - 1) + meta.outputPadding[i] + 1;
          output.push_back(xOut);
        }
      }
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
  if (socVersion == SocVersion::ASCEND310P) {
    return BIAS_SUPPORT_LIST_ASCEND310P;
  }

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
    if (engine.params.transposed) {
      CHECK_RET(engine.params.outputPadding != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
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

class DtypeCheckerTbc : public ConvolutionChecker {
 public:
  DtypeCheckerTbc() = default;
  ~DtypeCheckerTbc() override = default;
  aclnnStatus Check(ConvEngine &engine) {
    DataType inputDtype = engine.meta.input.dataType;
    DataType weightDtype = engine.meta.weight.dataType;
    DataType outputDtype = engine.meta.output.dataType;
    DataType biasDtype = engine.meta.bias.dataType;

    if (engine.params.bias != nullptr) {
      biasDtype = engine.meta.bias.dataType;
      CHECK_PARAM_DTYPE_VALID(biasDtype, GetBiasDtypeSupportListBySocVersion(), return ACLNN_ERR_PARAM_INVALID);
    }

    auto dtypeSupportList = GetDtypeSupportListBySocVersion();
    CHECK_PARAM_DTYPE_VALID(inputDtype, dtypeSupportList, return ACLNN_ERR_PARAM_INVALID);
    CHECK_PARAM_EQ(inputDtype, weightDtype);
    CHECK_PARAM_EQ(weightDtype, outputDtype);

    return ACLNN_SUCCESS;
  }
};

class DtypeCheckerDepthwise2d : public ConvolutionChecker {
 public:
  DtypeCheckerDepthwise2d() = default;
  ~DtypeCheckerDepthwise2d() override = default;
  aclnnStatus Check(ConvEngine &engine) {
    DataType inputDtype = engine.meta.input.dataType;
    DataType weightDtype = engine.meta.weight.dataType;
    DataType biasDtype = inputDtype;
    if (engine.params.bias != nullptr) {
      biasDtype = engine.meta.bias.dataType;
      CHECK_PARAM_DTYPE_VALID(biasDtype, GetBiasDtypeSupportListBySocVersion(), return ACLNN_ERR_PARAM_INVALID);
      CHECK_PARAM_EQ(inputDtype, biasDtype);
    }
    DataType outputDtype = engine.meta.output.dataType;

    auto dtypeSupportList = GetDtypeSupportListBySocVersion();
    CHECK_PARAM_DTYPE_VALID(inputDtype, dtypeSupportList, return ACLNN_ERR_PARAM_INVALID);
    CHECK_PARAM_DTYPE_VALID(weightDtype, dtypeSupportList, return ACLNN_ERR_PARAM_INVALID);
    CHECK_PARAM_DTYPE_VALID(outputDtype, dtypeSupportList, return ACLNN_ERR_PARAM_INVALID);

    CHECK_PARAM_EQ(inputDtype, weightDtype);
    CHECK_PARAM_EQ(weightDtype, outputDtype);

    return ACLNN_SUCCESS;
  }
};

class DimChecker : public ConvolutionChecker {
 public:
  DimChecker() = default;
  ~DimChecker() override = default;
  aclnnStatus CheckDim(const string &inStr, size_t inDim) {
    if (inDim != CONV_1D_DIM_SIZE && inDim != CONV_2D_DIM_SIZE && inDim != CONV_3D_DIM_SIZE) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expect %s equals 3 for conv1d, 4 for conv2d or 5 for conv3d, get %s",
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
      // 如果是transpose场景, bias的维度数必须为1维, 维度大小必须为 weight C * groups
      if (engine.params.transposed && (biasDim != 1 || biasSize != weightCValue * groupsValue)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Given transposed=1, weight of size %s, expected bias to be 1-dimensional with %ld elements, "
                "but got bias of size %s instead",
                op::ToString(engine.meta.weight.tensorShape).GetString(), weightCValue * groupsValue,
                op::ToString(engine.meta.bias.tensorShape).GetString());
        return ACLNN_ERR_PARAM_INVALID;
      }
      // 如果是非transpose场景, bias的维度数必须为1维, 维度大小必须为 weight N
      if (!engine.params.transposed && (biasDim != 1 || biasSize != weightNValue)) {
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
    if (((inputDim == CONV_1D_DIM_SIZE || inputDim == CONV_2D_DIM_SIZE) && !engine.params.transposed) ||
        (inputDim == CONV_2D_DIM_SIZE && engine.params.transposed)) {
      CHECK_PARAM_EQ_ONE(paddingSize, size_t, inputDim - 2, inputDim * 2 - 4);
    } else {
      CHECK_PARAM_EQ(paddingSize, inputDim - 2);
    }

    if (engine.params.transposed) {
      auto outputPaddingSize = engine.meta.outputPadding.size();
      CHECK_PARAM_EQ(outputPaddingSize, inputDim - 2);
    }

    return ACLNN_SUCCESS;
  };
};

class DimCheckerTbc : public ConvolutionChecker {
 public:
  DimCheckerTbc() = default;
  ~DimCheckerTbc() override = default;
  aclnnStatus Check(ConvEngine &engine) {
    size_t inputDim = engine.meta.input.shape.size();
    CHECK_PARAM_EQ_ONE(inputDim, size_t, CONV_1D_DIM_SIZE);

    size_t weightDim = engine.meta.weight.shape.size();
    CHECK_PARAM_EQ_ONE(weightDim, size_t, CONV_1D_DIM_SIZE);

    size_t outputDim = engine.meta.output.shape.size();
    CHECK_PARAM_EQ_ONE(outputDim, size_t, CONV_1D_DIM_SIZE);

    const size_t biasDimAllowTbc = 1;
    size_t biasDim = engine.meta.bias.shape.size();
    CHECK_PARAM_EQ_ONE(biasDim, size_t, biasDimAllowTbc);

    return ACLNN_SUCCESS;
  };
};

class DimCheckerDepthwise2d : public ConvolutionChecker {
 public:
  DimCheckerDepthwise2d() = default;
  ~DimCheckerDepthwise2d() override = default;
  aclnnStatus Check(ConvEngine &engine) {
    size_t inputDim = engine.meta.input.shape.size();
    CHECK_PARAM_EQ_ONE(inputDim, size_t, CONV_2D_DIM_SIZE);

    size_t weightDim = engine.meta.weight.shape.size();
    CHECK_PARAM_EQ_ONE(weightDim, size_t, CONV_2D_DIM_SIZE);

    size_t outputDim = engine.meta.output.shape.size();
    CHECK_PARAM_EQ_ONE(outputDim, size_t, CONV_2D_DIM_SIZE);

    if (engine.params.bias != nullptr) {
      size_t biasDim = engine.meta.bias.shape.size();
      const size_t biasDimAllow = 1;
      CHECK_PARAM_EQ_ONE(biasDim, size_t, biasDimAllow);
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

    switch (inputDimNum) {
      case CONV_1D_DIM_SIZE: {
        // conv1d convtranspose1d，input weight output format都应是NCL
        CHECK_PARAM_ALL_EQ(Format::FORMAT_NCL, op::Format, inputFormat, weightFormat, outputFormat);
        break;
      }
      case CONV_2D_DIM_SIZE: {
        // conv2d input weight output format 支持NCHW. conv2d transpose支持 NCHW
        OP_LOGD("conv2d transpose: [%d]", engine.params.transposed);
        if (!engine.params.transposed) {
          CHECK_PARAM_EQ_ONE(inputFormat, op::Format, Format::FORMAT_NCHW);
          CHECK_PARAM_EQ_ONE(weightFormat, op::Format, Format::FORMAT_NCHW, Format::FORMAT_FRACTAL_Z);
          CHECK_PARAM_EQ_ONE(outputFormat, op::Format, Format::FORMAT_NCHW);
        } else {
          CHECK_PARAM_EQ_ONE(inputFormat, op::Format, Format::FORMAT_NCHW);
          CHECK_PARAM_EQ_ONE(weightFormat, op::Format, Format::FORMAT_NCHW);
          CHECK_PARAM_EQ_ONE(outputFormat, op::Format, Format::FORMAT_NCHW);
        }
        break;
      }
      case CONV_3D_DIM_SIZE: {
        // conv3d convtranspose3d, input weight output 支持 NCDHW NDHWC
        CHECK_PARAM_EQ_ONE(inputFormat, op::Format, Format::FORMAT_NCDHW, Format::FORMAT_NDHWC);
        CHECK_PARAM_EQ_ONE(weightFormat, op::Format, Format::FORMAT_NCDHW, Format::FORMAT_NDHWC);
        CHECK_PARAM_EQ_ONE(outputFormat, op::Format, Format::FORMAT_NCDHW, Format::FORMAT_NDHWC);
        break;
      }
      default:
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "invalid input dim: %ld", inputDimNum);
        break;
    }
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
          
        if (engine.params.transposed && biasFormat != Format::FORMAT_ND) {
          OP_LOGW("Please set bias format to %s, other formats may cause precision issues.", 
                  op::ToString(Format::FORMAT_ND).GetString());
        }
    }

    return ACLNN_SUCCESS;
  };
};

class FormatCheckerTbc : public ConvolutionChecker {
 public:
  FormatCheckerTbc() = default;
  ~FormatCheckerTbc() override = default;
  aclnnStatus Check(ConvEngine &engine) {
    size_t inputDimNum = engine.meta.input.shape.size();
    auto inputFormat = engine.meta.input.format;
    auto weightFormat = engine.meta.weight.format;
    auto outputFormat = engine.meta.output.format;

    // conv_tbc，input weight output format都应是ND或者NCL
    CHECK_PARAM_EQ_ONE(inputFormat, op::Format, Format::FORMAT_ND, Format::FORMAT_NCL);
    CHECK_PARAM_EQ_ONE(weightFormat, op::Format, Format::FORMAT_ND, Format::FORMAT_NCL);
    CHECK_PARAM_EQ_ONE(outputFormat, op::Format, Format::FORMAT_ND, Format::FORMAT_NCL);

    return ACLNN_SUCCESS;
  };
};

class FormatCheckerDepthwise2d : public ConvolutionChecker {
 public:
  FormatCheckerDepthwise2d() = default;
  ~FormatCheckerDepthwise2d() override = default;
  aclnnStatus Check(ConvEngine &engine) {
    auto inputFormat = engine.meta.input.format;
    auto weightFormat = engine.meta.weight.format;
    auto outputFormat = engine.meta.output.format;

    CHECK_PARAM_EQ_ONE(inputFormat, op::Format, Format::FORMAT_NCHW);
    CHECK_PARAM_EQ_ONE(weightFormat, op::Format, Format::FORMAT_NCHW);
    CHECK_PARAM_EQ_ONE(outputFormat, op::Format, Format::FORMAT_NCHW);

    // 输入和输出format要求必须一致
    if (inputFormat != outputFormat) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
              "expected input format the same as output format. but get input format: %s, output format: %s",
              op::ToString(inputFormat).GetString(), op::ToString(outputFormat).GetString());
      return ACLNN_ERR_PARAM_INVALID;
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
    if (CheckPad(engine.meta.input, engine.meta.weight, engine.meta.stride, engine.meta.dilation, engine.meta.padding,
                 engine.params.transposed) != ACLNN_SUCCESS) {
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
    int64_t outChannel = -1;
    if (engine.params.transposed) {
      outChannel = engine.meta.weight.C() * engine.params.groups;
      if (engine.meta.weight.N() != inChannel) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected weight 1st dim equal to input channel in transpose mode");
        return ACLNN_ERR_PARAM_INVALID;
      }
      // output_padding value check  output_padding参数不支持负数
      for (size_t i = 0; i < engine.meta.outputPadding.size(); i++) {
        auto outputPaddingValue = engine.meta.outputPadding[i];
        if (outputPaddingValue >= engine.meta.stride[i] && outputPaddingValue >= engine.meta.dilation[i]) {
          OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                  "expected outputPadding < dilation or stride, get outputPadding %ld, dilation %ld stride %ld.",
                  outputPaddingValue, engine.meta.dilation[i], engine.meta.stride[i]);
          return ACLNN_ERR_PARAM_INVALID;
        }

        OP_CHECK(outputPaddingValue >= 0, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "negative output_padding[%ld] is not supported",
          outputPaddingValue), return ACLNN_ERR_PARAM_INVALID);
      }
    } else {
      outChannel = engine.meta.weight.N();
      if (engine.meta.weight.C() * engine.params.groups != inChannel) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "expected input channel equal to filter channel * groups. "
                "get input channel %ld, filter channel %ld, groups %ld.",
                inChannel, engine.meta.weight.C(), engine.params.groups);
        return ACLNN_ERR_PARAM_INVALID;
      }
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

    if (ACLNN_SUCCESS != CheckEmptyTensor(engine.meta.input, engine.params.transposed)) {
      return ACLNN_ERR_PARAM_INVALID;
    }

    // 针对 2d transpose error msg is: backprop pad value invalid 提前拦截
    if (!(padBinaryValid(engine))) {
      return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
  };

 private:
  /** 空tensor判断逻辑
   * input:
   * 在ValueChecker时，保证加上pad后，空间维度也大于0
   * 此处校验针对transpose的情况，仅支持输入的n为0，因此仅需要校验C维度是否为0
   * weight: Cout和K不为0，在ValueChecker已完成校验
   */
  static aclnnStatus CheckEmptyTensor(TensorMeta &input, bool transposed) {
    if (transposed && input.C() == 0) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input c should not be zero in transpose mode");
      return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
  }

  // 针对 卷积2d transpose： error msg is: backprop pad value invalid 提前拦截
  // 反向pad:  (weightW - 1) * dilationW + 1 - padLeft/Right <=255   (weightH - 1) * dilationH + 1 - padUp/down
  static bool padBinaryValid(ConvEngine &engine) {
    if (!engine.params.transposed) {
      return true;
    }

    // 255是目前阈值，二进制不支持该值大于255进行计算
    int64_t padBinValue = 255;
    int64_t dilationH = engine.meta.dilation[0];
    int64_t dilationW = (engine.meta.dilation.size() == 1) ? engine.meta.dilation[0] : engine.meta.dilation[1];
    int64_t padTop = engine.meta.padding[0];
    int64_t padLeft = (engine.meta.padding.size() == 1) ? engine.meta.padding[0] : engine.meta.padding[1];
    auto weightShape = engine.meta.weight.tensorShape;
    int64_t weightW = engine.meta.weight.W();
    int64_t weightH = engine.meta.weight.H();
    int64_t weightL = engine.meta.weight.L();
    bool padValueValid = false;
    // 3为weight为NCL场景
    if (weightShape.GetDimNum() == 3) {
      padValueValid = ((weightL - 1) * dilationW - padLeft) <= padBinValue;
      OP_CHECK(padValueValid, OP_LOGE(ACLNN_ERR_PARAM_INVALID, 
        "Current case is not supported. ((weightL - 1) * dilationW - padLeft)=%ld "
        "should not be larger than 255.", (weightL - 1) * dilationW - padLeft),
        return false);
    } else if (weightShape.GetDimNum() == 4) { // 4为weight为NCHW / NHWC场景
      padValueValid = (((weightW - 1) * dilationW - padLeft) <= padBinValue) &&
        (((weightH - 1) * dilationH - padTop) <= padBinValue);
      OP_CHECK(padValueValid, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Current case is not supported. "
        "((weight_w - 1) * dilation_w - padding_left)=%ld or ((weight_h - 1) * dilation_h - padding_top)=%ld "
        "should not be larger than 255", (weightW - 1) * dilationW - padLeft, (weightH - 1) * dilationH - padTop),
        return false);
    } else if (weightShape.GetDimNum() == CONV_3D_DIM_SIZE) {
      int64_t dilationLast = (engine.meta.dilation.size() == 1) ? engine.meta.dilation[0] : engine.meta.dilation[2];
      int64_t padRight = (engine.meta.padding.size() == 1) ? engine.meta.padding[0] : engine.meta.padding[2];
      padValueValid = (((weightW - 1) * dilationLast - padRight) <= padBinValue) &&
        (((weightW - 1) * dilationLast - padRight) >= 0) && 
        (((weightH - 1) * dilationW - padLeft) <= padBinValue) &&
        (((weightH - 1) * dilationW - padLeft) >= 0);
      OP_CHECK(padValueValid, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Current case is not supported. "
        "((weight_w - 1) * dilation_w - padding_left)=%ld or ((weight_h - 1) * dilation_h - padding_top)=%ld or "
        "should not be larger than 255 or less than 0",
        (weightW - 1) * dilationLast - padRight, (weightH - 1) * dilationW - padLeft),
        return false);
    }

    return true;
  }

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
    size_t inputSpaceDimNum = input.shape.size() - 2;  // 空间维度大小，1d卷积时为1，2d为2，3d为3
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
                              FVector<int64_t> &dilation, FVector<int64_t> &padding, bool transposed) {
    FVector<int64_t> inputShape = input.shape;
    bool inputChannleLast = input.ChannelLast();
    int64_t inputSpaceDimIndex = inputChannleLast ? 1 : 2;  // 空间维度在shape中的起始位置，C维度后置时为1，否则为2
    size_t inputSpaceDimNum = input.shape.size() - 2;  // 空间维度大小，1d卷积时为1，2d为2，3d为3
    FVector<int64_t> weightShape = weight.shape;
    bool weightChannleLast = weight.ChannelLast();
    int64_t weightSpaceDimIndex = weightChannleLast ? 1 : 2;  // 空间维度在shape中的起始位置，C维度后置时为1，否则为2

    auto newpad = ConstructPad(padding, inputShape);
    for (size_t i = 0; i < inputSpaceDimNum; ++i) {
      auto inputShapeValue = inputShape[i + inputSpaceDimIndex];
      auto weightShapeValue = weightShape[i + weightSpaceDimIndex];
      auto strideValue = stride[i];
      auto paddingValueFront = (input.shape.size() == CONV_1D_DIM_SIZE
        || input.shape.size() == CONV_2D_DIM_SIZE) ? newpad[i] : padding[i];
      auto dilationValue = dilation[i];

      // check input shape after pad only for conv
      if (!transposed) {
        if (input.shape.size() == CONV_1D_DIM_SIZE || input.shape.size() == CONV_2D_DIM_SIZE) {
          int64_t inputShapeValueAfterPad =
              (inputShapeValue + paddingValueFront - dilationValue * (weightShapeValue - 1) - 1);
          CHECK_PARAM_GTE(inputShapeValueAfterPad, 0);
        } else {
          int64_t inputShapeValueAfterPad =
              (inputShapeValue + paddingValueFront * 2 - dilationValue * (weightShapeValue - 1) - 1);
          CHECK_PARAM_GTE(inputShapeValueAfterPad, 0);
        }
      }
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

class ValueCheckerTbc : public ConvolutionChecker {
 public:
  ValueCheckerTbc() = default;
  ~ValueCheckerTbc() override = default;
  aclnnStatus Check(ConvEngine &engine) {
    if (CheckShapeTbc(engine.meta.input, engine.meta.weight, engine.meta.output) != ACLNN_SUCCESS) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "check conv_tbc shape failed");
      return ACLNN_ERR_PARAM_INVALID;
    }
    int64_t outChannel = engine.meta.weight.N();
    if (CheckConvBiasTbc(engine.meta.bias, outChannel) != ACLNN_SUCCESS) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "check conv_tbc bias failed");
      return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
  };

 private:
  /*
  input weight output 的shape均瑶大于等于0
  bias（一维）的值要等于channel_out
  */
  aclnnStatus CheckShapeTbc(TensorMeta &input, TensorMeta &weight, TensorMeta &output) {
    // check shape >= 0
    int64_t inputShapeN = input.N();
    int64_t inputShapeC = input.C();
    int64_t inputShapeL = input.L();
    int64_t weightShapeN = weight.N();
    int64_t weightShapeC = weight.C();
    int64_t weightShapeL = weight.L();
    int64_t outputShapeN = output.N();
    int64_t outputShapeC = output.C();
    int64_t outputShapeL = output.L();

    CHECK_PARAM_ALL_GTE(0L, int64_t, inputShapeN, inputShapeC, inputShapeL, weightShapeN, weightShapeC, weightShapeL,
                        outputShapeN, outputShapeC, outputShapeL);

    return ACLNN_SUCCESS;
  }

  aclnnStatus CheckConvBiasTbc(TensorMeta &bias, int64_t outChannel) {
    auto biasShape = bias.shape;
    size_t biasDimNum = biasShape.size();
    if (biasShape[0] != outChannel) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected input bias size of dim_c[%ld] equal to out_channels[%ld].",
              biasShape[0], outChannel);
      return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
  }
};

class ValueCheckerDepthwise2d : public ConvolutionChecker {
 public:
  ValueCheckerDepthwise2d() = default;
  ~ValueCheckerDepthwise2d() override = default;
  aclnnStatus Check(ConvEngine &engine) {
      if (ACLNN_SUCCESS != CheckEmptyTensorDepthwise2d(engine.meta.input)) {
        return ACLNN_ERR_PARAM_INVALID;
      }
    if (CheckShapeDepthwise2d(engine.meta.input, engine.meta.weight) != ACLNN_SUCCESS) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "shape check failed");
      return ACLNN_ERR_PARAM_INVALID;
    }
    // check stride
    if (CheckVectorValueGt0Depthwise2d(engine.meta.stride) != ACLNN_SUCCESS) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "stride check failed");
      return ACLNN_ERR_PARAM_INVALID;
    }
    // check dilation
    if (CheckVectorValueGt0Depthwise2d(engine.meta.dilation) != ACLNN_SUCCESS) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dilation check failed");
      return ACLNN_ERR_PARAM_INVALID;
    }
    // check pad
    if (CheckPadDepthwise2d(engine.meta.input, engine.meta.weight, engine.meta.stride, engine.meta.dilation,
                            engine.meta.padding, engine.params.transposed) != ACLNN_SUCCESS) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "pad check failed");
      return ACLNN_ERR_PARAM_INVALID;
    }
    // check channel
    int64_t inChannel = engine.meta.input.C();
    int64_t outChannel = -1;
    outChannel = engine.meta.weight.N();
    if (engine.meta.weight.C() != 1) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
              "expected filter channel equal to 1. "
              "get filter channel %ld.",
              engine.meta.weight.C());
      return ACLNN_ERR_PARAM_INVALID;
    }

    if (outChannel % inChannel != 0) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
              "expected outChannel divisible by inChannel, get outChannel %ld, inChannel %ld",
              outChannel, inChannel);
      return ACLNN_ERR_PARAM_INVALID;
    }

    if (engine.params.bias != nullptr) {
      if (CheckConvBiasDepthwise2d(engine.meta.bias, outChannel) != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "check bias failed");
        return ACLNN_ERR_PARAM_INVALID;
      }
    }

    return ACLNN_SUCCESS;
  };

 private:
  static aclnnStatus CheckEmptyTensorDepthwise2d(TensorMeta &input) {
    if (input.N() == 0 || input.C() == 0) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input not support empty tensor");
      return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
  }

  static aclnnStatus CheckShapeDepthwise2d(TensorMeta &input, TensorMeta &weight) {
    // check n c
    int64_t inputShapeN = input.N();
    int64_t inputShapeC = input.C();
    int64_t weightShapeN = weight.N();
    int64_t weightShapeC = weight.C();
    CHECK_PARAM_ALL_GTE(0L, int64_t, inputShapeN, inputShapeC, weightShapeN, weightShapeC);
    CHECK_PARAM_GT(weightShapeN, 0L);

    // check space(d h w or l)
    FVector<int64_t> inputShape = input.shape;
    bool inputChannleLast = input.ChannelLast();
    int64_t inputSpaceDimIndex = inputChannleLast ? 1 : 2;  // 空间维度在shape中的起始位置，C维度后置时为1，否则为2
    size_t inputSpaceDimNum = input.shape.size() - 2;  // 空间维度大小，1d卷积时为1，2d为2，3d为3
    FVector<int64_t> weightShape = weight.shape;
    bool weightChannleLast = weight.ChannelLast();
    int64_t weightSpaceDimIndex = weightChannleLast ? 1 : 2;  // 空间维度在shape中的起始位置，C维度后置时为1，否则为2
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

  static inline aclnnStatus CheckVectorValueGt0Depthwise2d(FVector<int64_t> &param) {
    for (size_t i = 0; i < param.size(); ++i) {
      if (param[i] <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected %ldth value > 0, but get %ld", i + 1, param[i]);
        return ACLNN_ERR_PARAM_INVALID;
      }
    }
    return ACLNN_SUCCESS;
  }

  static aclnnStatus CheckPadDepthwise2d(TensorMeta &input, TensorMeta &weight, FVector<int64_t> &stride,
                                         FVector<int64_t> &dilation, FVector<int64_t> &padding, bool transposed) {
    FVector<int64_t> inputShape = input.shape;
    bool inputChannleLast = input.ChannelLast();
    int64_t inputSpaceDimIndex = inputChannleLast ? 1 : 2;  // 空间维度在shape中的起始位置，C维度后置时为1，否则为2
    size_t inputSpaceDimNum = input.shape.size() - 2;  // 空间维度大小，1d卷积时为1，2d为2，3d为3
    FVector<int64_t> weightShape = weight.shape;
    bool weightChannleLast = weight.ChannelLast();
    int64_t weightSpaceDimIndex = weightChannleLast ? 1 : 2;  // 空间维度在shape中的起始位置，C维度后置时为1，否则为2

    for (size_t i = 0; i < inputSpaceDimNum; ++i) {
      auto inputShapeValue = inputShape[i + inputSpaceDimIndex];
      auto weightShapeValue = weightShape[i + weightSpaceDimIndex];
      auto strideValue = stride[i];
      auto paddingValueFront = padding[i];
      auto dilationValue = dilation[i];

      // check input shape after pad only for conv
      if (!transposed) {
        int64_t inputShapeValueAfterPad =
            (inputShapeValue + paddingValueFront * 2 - dilationValue * (weightShapeValue - 1) - 1);
        CHECK_PARAM_GTE(inputShapeValueAfterPad, 0);
      }
    }

    return ACLNN_SUCCESS;
  }

  static aclnnStatus CheckConvBiasDepthwise2d(TensorMeta &bias, int64_t outChannel) {
    auto biasShape = bias.shape;
    if (biasShape[0] != outChannel) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected bias shape %ld equal to out_channels %ld.",
              biasShape[0], outChannel);
      return ACLNN_ERR_PARAM_INVALID;
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
    // CHECK_RET(CheckCubeMathType(upperDtype, engine.params.cubeMathType), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
  }
};

class HardwareLimitCheckerTbc : public ConvolutionChecker {
 public:
  HardwareLimitCheckerTbc() = default;
  ~HardwareLimitCheckerTbc() override = default;
  aclnnStatus Check(ConvEngine &engine) {
    DataType inputDtype = engine.meta.input.dataType; // input和weight应该是一个Dtype
    // CHECK_RET(CheckCubeMathType(inputDtype, engine.params.cubeMathType), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
  }
};

class TemporarySoftwareLimitChecker : public ConvolutionChecker {
 public:
  TemporarySoftwareLimitChecker() = default;
  ~TemporarySoftwareLimitChecker() override = default;
  aclnnStatus Check(ConvEngine &engine) {
    size_t inputDim = engine.meta.input.shape.size();
    // 除了910A 910B 310P，其余暂不支持
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    switch (socVersion) {
      case SocVersion::ASCEND910:
      case SocVersion::ASCEND910B:
      case SocVersion::ASCEND910_93:
      case SocVersion::ASCEND310P:
        break;
      default: {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "support for %s is not implemented", op::ToString(socVersion).GetString());
        return ACLNN_ERR_PARAM_INVALID;
      }
    }
    return ACLNN_SUCCESS;
  }
};

class TemporarySoftwareLimitCheckerTbc : public ConvolutionChecker {
 public:
  TemporarySoftwareLimitCheckerTbc() = default;
  ~TemporarySoftwareLimitCheckerTbc() override = default;
  aclnnStatus Check(ConvEngine &engine) {  // 3D暂不支持
    // 除了910A 910B 310P，其余暂不支持
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    switch (socVersion) {
      case SocVersion::ASCEND910:
      case SocVersion::ASCEND910B:
      case SocVersion::ASCEND910_93:
      case SocVersion::ASCEND310P:
        break;
      default: {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "support for %s is not implemented", op::ToString(socVersion).GetString());
        return ACLNN_ERR_PARAM_INVALID;
      }
    }
    return ACLNN_SUCCESS;
  }
};

static inline bool CheckNotNull(const aclTensor *self, const aclTensor *weight, const aclTensor *bias,
                                const aclTensor *output) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(weight, return false);
  OP_CHECK_NULL(bias, return false);
  OP_CHECK_NULL(output, return false);
  return true;
}

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

static aclnnStatus CheckConvTbcParams(ConvEngine &engine) {
  std::vector<unique_ptr<ConvolutionChecker>> checkList;
  // math level check
  // common checkers: nullptr, dims, format
  checkList.push_back(make_unique<DtypeCheckerTbc>());
  checkList.push_back(make_unique<DimCheckerTbc>());
  checkList.push_back(make_unique<FormatCheckerTbc>());
  checkList.push_back(make_unique<ValueCheckerTbc>());
  // different conv checkers: infershape and so on
  checkList.push_back(make_unique<ConvXdChecker>());

  // implement level check
  // hardware limit checkers:double conv, fp32 conv in 1980...
  checkList.push_back(make_unique<HardwareLimitCheckerTbc>());
  // temporary software limitation checkers: 3d conv
  checkList.push_back(make_unique<TemporarySoftwareLimitCheckerTbc>());

  for (auto &checker : checkList) {
    aclnnStatus ret = checker->Check(engine);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckConvDepthwise2dParams(ConvEngine &engine) {
  std::vector<unique_ptr<ConvolutionChecker>> checkList;
  // math level check
  // common checkers: nullptr, dims, format
  checkList.push_back(make_unique<NullptrChecker>());
  checkList.push_back(make_unique<DtypeCheckerDepthwise2d>());
  checkList.push_back(make_unique<DimCheckerDepthwise2d>());
  checkList.push_back(make_unique<FormatCheckerDepthwise2d>());
  checkList.push_back(make_unique<ValueCheckerDepthwise2d>());
  // different conv checkers: infershape and so on
  checkList.push_back(make_unique<ConvXdChecker>());

  // implement level check
  // hardware limit checkers:double conv, fp32 conv in 1980...
  checkList.push_back(make_unique<HardwareLimitChecker>());
  // temporary software limitation checkers: 3d conv
  checkList.push_back(make_unique<TemporarySoftwareLimitCheckerTbc>());

  for (auto &checker : checkList) {
    aclnnStatus ret = checker->Check(engine);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
  }
  return ACLNN_SUCCESS;
}

static inline aclnnStatus CheckParamsNullptrTbc(const aclTensor *self, const aclTensor *weight, const aclTensor *bias,
                                                const aclTensor *output) {
  // 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, weight, bias, output), ACLNN_ERR_PARAM_NULLPTR);

  return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutputBiasShape(const aclTensor *output, const aclTensor *bias) {
  size_t outputDimNum = output->GetViewShape().GetDimNum();
  OP_CHECK_WRONG_DIMENSION(output, CONV_1D_DIM_SIZE, return false);

  for (size_t i = 0; i < outputDimNum; i++) {
    if (output->GetViewShape()[i] < 0) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "output for tbc expected %ldth value > 0, but get [%ld].", i + 1,
              output->GetViewShape()[i]);
      return false;
    }
  }
  OP_CHECK_WRONG_DIMENSION(bias, 1, return false);

  if (bias->GetViewShape()[0] != output->GetViewShape()[2]) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "bias for tbc [%ld] should be equal to output's last dim [%ld].",
            bias->GetViewShape()[0], output->GetViewShape()[2]);
    return false;
  }
  return true;
}

static aclnnStatus CheckOutputBiasDtype(const aclTensor *output, const aclTensor *bias) {
  auto dtypeSupportList = GetDtypeSupportListBySocVersion();
  OP_CHECK_DTYPE_NOT_SUPPORT(output, dtypeSupportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(bias, dtypeSupportList, return false);
  return true;
}

static aclnnStatus CheckOutputBiasFormat(const aclTensor *output, const aclTensor *bias) {
  if ((output->GetViewFormat() != op::Format::FORMAT_ND && output->GetViewFormat() != op::Format::FORMAT_NCL)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "output for tbc format should be NCL or ND, actual [%s].",
            op::ToString(output->GetViewFormat()).GetString());
    return false;
  }

  if (bias->GetViewFormat() != op::Format::FORMAT_ND) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "bias for tbc format should be ND, actual [%s].",
            op::ToString(bias->GetViewFormat()).GetString());
    return false;
  }
  return true;
}

static inline aclnnStatus CheckParamsEmpty(const aclTensor *output, const aclTensor *bias) {
  CHECK_RET(CheckOutputBiasShape(output, bias), ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(CheckOutputBiasDtype(output, bias), ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(CheckOutputBiasFormat(output, bias), ACLNN_ERR_PARAM_INVALID);
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
    if (!transposed) {
      bias = castBias;
    } else {
      bias = l0op::ReFormat(castBias, opInfo.biasFormat);
      CHECK_RET(bias != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
  }

  return ACLNN_SUCCESS;
}

// 实现公共数据预处理，将数据准备为L0可接受的形式  C04特殊分支
static aclnnStatus CommonPreProcessC04(const aclTensor *&input, const aclTensor *&weight, const aclTensor *&bias,
                                       const int64_t groups, const bool transposed, const ConvolutionOpInfo &opInfo,
                                       bool changeFormat, bool contiguous, aclOpExecutor *executor) {
  auto contiguousInput = input;
  auto contiguousWeight = weight;
  auto contiguousBias = bias;
  if (contiguous) {
    contiguousInput = l0op::Contiguous(input, executor);
    CHECK_RET(contiguousInput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    contiguousWeight = l0op::Contiguous(weight, executor);
    CHECK_RET(contiguousWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (bias != nullptr) {
      contiguousBias = l0op::Contiguous(bias, executor);
      CHECK_RET(contiguousBias != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
  }

  auto castedInput = l0op::Cast(contiguousInput, opInfo.inputDtype, executor);
  CHECK_RET(castedInput != nullptr, ACLNN_ERR_INNER_NULLPTR);
  if (changeFormat) {
    // input format transdata
    input = l0op::TransData(castedInput, opInfo.inputFormat, groups, executor);
    CHECK_RET(input != nullptr, ACLNN_ERR_INNER_NULLPTR);
  } else {
    input = castedInput;
  }

  // weight特殊操作规避：NCHW(1, 1, 3, 3) -> pad -> NCHW(1, 4, 3, 3) -> transpose -> NHWC(1, 3, 3, 4)
  //                   -> ND(1, 36) -> transdata -> FZ_NZ -> setformat(FZC04)
  auto castedWeight = l0op::Cast(contiguousWeight, opInfo.weightDtype, executor);
  CHECK_RET(castedWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
  OP_LOGD("Before c04 weight is %s", weight->ToString().GetString());
  auto initialWeightViewShape = weight->GetViewShape();
  if (changeFormat) {
    // NCHW (a, b, c, d) -> padV3 -> NCHW(a, 4, c, d)  保证C维度pad到4
    if (weight->GetViewShape().GetDim(1) != 4) {
      const_cast<aclTensor*> (weight)->SetStorageShape(weight->GetViewShape());
      const_cast<aclTensor*> (weight)->SetOriginalShape(weight->GetViewShape());
      OP_LOGD("Before padv3 weight is %s", weight->ToString().GetString());
      // [0, 0, 0, 4 - b, 0, 0, 0, 0] padding  padding总长度为8
      int64_t paddingArray[8] = {};
      // padding的总长度是8
      for (int64_t i = 0; i < 8; i++) {
        paddingArray[i] = 0;
      }
      // CO4需求，因此需要补充维度到4。预计维度数是4，padding中的第3位要pad到4，具体取1指代NCHW中的C
      paddingArray[3] = 4 - weight->GetViewShape().GetDim(1);
      aclIntArray *paddingArrayRes = executor->AllocIntArray(paddingArray, 8);
      CHECK_RET(paddingArrayRes != nullptr, ACLNN_ERR_INNER_NULLPTR);
      auto paddingTensor = executor->ConvertToTensor(paddingArrayRes, DataType::DT_INT32);
      auto constantValues = executor->ConvertToTensor(executor->AllocScalar(0), weight->GetDataType());
      weight = l0op::PadV3(weight, paddingTensor, constantValues, REFLECTION_MODE, true, executor);
      CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
      OP_LOGD("After padv3 weight is %s", weight->ToString().GetString());
    }

    // NCHW(a, b, c, d) -> transpose -> NHWC(a, c, d, b)  因为NCHW，所以长度为4
    int64_t valuePerm[4] = {};
    //  因为NCHW，所以长度为4
    for (int64_t i = 0; i < 4; i++) {
        valuePerm[i] = i;
    }
    // 1表示C，3表示W
    std::swap(valuePerm[1], valuePerm[3]);  // b, c, d -> d, c, b
    // 1表示W，2表示H
    std::swap(valuePerm[1], valuePerm[2]);  // d, c, b -> c, d, b
    auto perm = executor->AllocIntArray(valuePerm, 4);
    CHECK_RET(perm != nullptr, ACLNN_ERR_INNER_NULLPTR);
    weight = l0op::Transpose(weight, perm, executor);
    CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const_cast<aclTensor*> (weight)->SetViewFormat(Format::FORMAT_NHWC);
    const_cast<aclTensor*> (weight)->SetStorageFormat(Format::FORMAT_NHWC);
    const_cast<aclTensor*> (weight)->SetOriginalFormat(Format::FORMAT_NHWC);
    OP_LOGD("After transpose weight is %s", weight->ToString().GetString());

    // NHWC (a, b, c, d) 转成 ND (a, b * c * d)
    auto weightShape = weight->GetViewShape();
    int64_t newFormatRes = weightShape.GetDim(1) * weightShape.GetDim(2) * weightShape.GetDim(3);
    op::Shape ndShape = op::Shape({weightShape.GetDim(0), newFormatRes});
    const_cast<aclTensor*> (weight)->SetStorageFormat(Format::FORMAT_ND);
    const_cast<aclTensor*> (weight)->SetOriginalFormat(Format::FORMAT_ND);
    const_cast<aclTensor*> (weight)->SetStorageShape(ndShape);
    const_cast<aclTensor*> (weight)->SetOriginalShape(ndShape);
    OP_LOGD("After update format weight is %s", weight->ToString().GetString());

    // transdata: NHWC -> FZ_NZ
    weight = l0op::TransData(weight, FORMAT_FRACTAL_NZ, groups, executor);
    CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    OP_LOGD("After transdata weight is %s", weight->ToString().GetString());
    const_cast<aclTensor*> (weight)->SetOriginalShape(initialWeightViewShape);
    const_cast<aclTensor*> (weight)->SetOriginalFormat(Format::FORMAT_NCHW);
    auto storageShape = weight->GetStorageShape();

    // reformat
    auto weightFormatC04 = executor->CreateView(weight, weight->GetViewShape(), weight->GetViewOffset());
    weight = weightFormatC04;
    const_cast<aclTensor*> (weight)->SetStorageFormat(Format::FORMAT_FRACTAL_Z_C04);
    const_cast<aclTensor*> (weight)->SetStorageShape(storageShape);
    const_cast<aclTensor*> (weight)->SetOriginalShape(initialWeightViewShape);
    const_cast<aclTensor*> (weight)->SetOriginalFormat(Format::FORMAT_NCHW);
    OP_LOGD("After reformat weight is %s", weight->ToString().GetString());
  } else {
    weight = castedWeight;
  }

  // bias
  if (bias != nullptr) {
    // cast
    auto castBias = l0op::Cast(contiguousBias, opInfo.biasDtype, executor);
    CHECK_RET(castBias != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // transdata
    if (!transposed) {
      bias = castBias;
    } else {
      bias = l0op::ReFormat(castBias, opInfo.biasFormat);
      CHECK_RET(bias != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
  }

  return ACLNN_SUCCESS;
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
  if (!transposed) {
    opInfo.outputDtype = upperDtype; // 目前conv2d算子底层二进制仅支持输入输出相同，暂不支持16进32出的场景
  } else {
    opInfo.outputDtype = (output->GetDataType() == op::DataType::DT_FLOAT) ? output->GetDataType() : upperDtype;
  }
  // ASCEND910 + ASCEND310P 仅支持fp16的卷积，或者USE_FP16场景必走FP16， 因此必须转为fp16实现
  if (socVersion == SocVersion::ASCEND910 || socVersion == SocVersion::ASCEND310P || cubeMathType == USE_FP16) {
    opInfo.outputDtype = op::DataType::DT_FLOAT16; // 目前底层二进制暂不支持16进32出的场景，故设为FP16运算
  }

  opInfo.inputDtype = upperDtype;
  opInfo.weightDtype = upperDtype;
  if (bias != nullptr) {
    if (transposed ) { // transpose拆AclnnADD，bias与output满足互推导关系计算，更高精度
      upperDtype = GetUpperFloatDataType(opInfo.outputDtype, bias->GetDataType());
    }
    opInfo.biasDtype = upperDtype;
    // 因为bias二进制不支持为BF16，所以得转成FP32
    if (upperDtype == op::DataType::DT_BF16 &&
        (!(transposed && input->GetViewShape().GetDimNum() == CONV_3D_DIM_SIZE))) {
      OP_LOGD("Since bias does not support BF16, change the dtype of bias to fp32.");
      opInfo.biasDtype = op::DataType::DT_FLOAT;
    }
  }
}

namespace {
  const int STRIDEH_DMA = 63;
  const int DILATION_DMA = 255;
  const int PAD_DMA = 255;
  const int weight_DMA = 511;
}
static bool isNotDMAFromPad(bool isDMASpec, const aclIntArray *padding)
{
  if (padding->Size() == CONV_2D_PAD_DIM) {
    isDMASpec =  isDMASpec || ((*padding)[0] > PAD_DMA) || ((*padding)[1] > PAD_DMA);
  } else if (padding->Size() == CONV_4D_PAD_DIM) {
    isDMASpec = isDMASpec || ((*padding)[0] > PAD_DMA) || ((*padding)[1] > PAD_DMA) ||
      ((*padding)[PAD_LEFT_INDEX] > PAD_DMA) || ((*padding)[PAD_RIGHT_INDEX] > PAD_DMA);
  }
  return isDMASpec;
}
// padding = [pad_top, pad_bottom, pad_left, pad_right]
// 1. 不满足DMA的规格   2. load3d L1最小切分要在L1能够放下
static bool isNotDMA(const aclTensor *input, const aclTensor *weight, const aclTensor *bias, aclTensor *output,
                     const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation) {
  int64_t inputHeight = (int64_t) input->GetViewShape().GetDim(2);
  int64_t inputWidth = (int64_t) input->GetViewShape().GetDim(3);
  int64_t weightH = (int64_t) weight->GetViewShape().GetDim(2);
  int64_t weightW = (int64_t) weight->GetViewShape().GetDim(3);
  int64_t outputSize = (int64_t) output->GetViewShape().GetDimNum();
  int64_t outputW = (outputSize == 4) ? (int64_t) output->GetViewShape().GetDim(3) :
    (int64_t) output->GetViewShape().GetDim(2);

  // CUBE_FP16的M, K, N
  int64_t BLK_M = 16;
  int64_t BLK_K = 16;
  int64_t BLK_N = 16;
  int64_t BIT_L12BT_MIN_BIAS = 64;

  // 1. 不满足DMA的规格
  int64_t strideH = (*stride)[0];
  int64_t strideW = (*stride)[1];
  int64_t dilationH = (*dilation)[0];
  int64_t dilationW = (*dilation)[1];
  bool alignResult = ((weightH * weightW * 4 + BLK_K - 1) / BLK_K * BLK_K) <= 65535;
  OP_LOGD("alignResult is %d", alignResult);

  // stride wh <=63, dilation wh<=255, padding <=255, weight wh<=511, align(weightH * weightW * 4, BLK_K) <= 65535
  // 以上条件同时满足表示不满足DMA规格
  bool isDMASpec = (strideH > STRIDEH_DMA) || (strideW > STRIDEH_DMA) || (dilationH > DILATION_DMA)
      || (dilationW > DILATION_DMA) || (weightH > weight_DMA) || (weightW > weight_DMA) || !alignResult;
  isDMASpec = isNotDMAFromPad(isDMASpec, padding);
  if (isDMASpec) {
    OP_LOGD("Fulfill DMA requirement，return False");
    return false;
  }

  // 2. load3d L1最小切分要在L1能够放下
  int64_t hoNum = BLK_M / outputW + 2;
  int64_t hkDilation = (weightH - 1) * dilationH + 1;
  int64_t hiNum = std::min(((hoNum - 1) * strideH + hkDilation), inputHeight);
  int64_t wiL1 = (int64_t) input->GetViewShape().GetDim(3);
  int64_t hiL1 = hiNum;

  // input_height = 1 & weight_height = 1 & pad_top = 0 & pad bottom = 0
  bool isConv1d = (inputHeight == 1) && (weightH == 1) && ((*padding)[0] == 0) && ((*padding)[1] == 0);
  OP_LOGD("isConv1d is %d", isConv1d);
  if (isConv1d) {
    int64_t woNum = BLK_M;
    int64_t wkDilation = (weightW - 1) * dilationW + 1;
    wiL1 = std::min(((woNum - 1) * strideW + wkDilation), inputWidth);
  }

  // 非Conv1d时width <= 32767才能走C04
  int64_t WIDTH_THRESSHOLD = 32767;
  if (!isConv1d && inputWidth > WIDTH_THRESSHOLD) {
    OP_LOGD("when not conv1d scene, inputWidth[%ld] > 32767", inputWidth);
    return false;
  }
  int64_t hiwiMul = wiL1 * hiL1;
  int64_t c0OnL1 = 4;
  uint64_t maxL1Size = hiwiMul * c0OnL1 * 2;  // dataTypeToByte(FP16)为2
  maxL1Size = (bias != nullptr)? maxL1Size + BIT_L12BT_MIN_BIAS: maxL1Size;
  OP_LOGD("maxL1Size is %ld", maxL1Size);

  // hardwareInfo.l1size 910B 为524288
  return maxL1Size <= 524288;
}

static bool CanSwitchC04InBF16Scene(const struct ConvolutionOpInfo &opInfo)
{
  if (opInfo.weightDtype == op::DataType::DT_BF16 && (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93
     || GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B)) {
    return true;
  }
  return false;
}

static bool CanSwitchC04InF16Scene(const struct ConvolutionOpInfo &opInfo)
{
  // datatype为float16，且socversion为910B和910_93
  if (opInfo.weightDtype == op::DataType::DT_FLOAT16 && IsCubeSupportFp32()) {
    return true;
  }
  return false;
}

// 1. groups为1  2. Cin<=4  3. dtype为FP16 4. 必须是910B芯片 5. 非DMA场景   同时满足才能走c04
static bool CanSwitchC04(const aclTensor *input, const aclTensor *weight, const aclTensor *bias, aclTensor *output,
                         const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                         int64_t groups, bool transposed) {
  // 必须为非transpose场景 + format为NCHW才行
  if (transposed || input->GetViewFormat() != Format::FORMAT_NCHW) {
    OP_LOGD("input is not NCHW or is transposed, thus no C04");
    return false;
  }

  int64_t cin = input->GetViewShape().GetDim(1);
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();

  // groups数量必须为1， 并且C04场景必须Cin为4,非DMA可直接切c04

  if ((groups == 1) && (cin <= SMALL_CHANNEL && cin > 0)) {
    return isNotDMA(input, weight, bias, output, stride, padding, dilation);
  }

  OP_LOGD("Not fulfill the requirements for C04");
  return false;
}

// 非C04场景的更新 卷积format
void GetConvolutionOpFormat(struct ConvolutionOpInfo &opInfo) {
  opInfo.weightFormat = Format::FORMAT_FRACTAL_Z;
  opInfo.inputFormat = Format::FORMAT_NC1HWC0;
  opInfo.outputFormat = Format::FORMAT_NC1HWC0;
  opInfo.biasFormat = Format::FORMAT_ND;
}

void GetConvolution3dOpFormat(struct ConvolutionOpInfo &opInfo) {
  opInfo.weightFormat = Format::FORMAT_FRACTAL_Z_3D;
  opInfo.inputFormat = Format::FORMAT_NDC1HWC0;
  opInfo.outputFormat = Format::FORMAT_NDC1HWC0;
  opInfo.biasFormat = Format::FORMAT_ND;
}

void GetConvolutionOpFormatC04(struct ConvolutionOpInfo &opInfo) {
  opInfo.weightFormat = Format::FORMAT_FRACTAL_Z_C04;
  opInfo.inputFormat = Format::FORMAT_NC1HWC0;
  opInfo.outputFormat = Format::FORMAT_NC1HWC0;
  opInfo.biasFormat = Format::FORMAT_ND;
}

// 更新convolution所需要的dtype format
void GetConvOpInfo(const aclTensor *input, const aclTensor *weight, const aclTensor *bias, aclTensor *output,
                   struct ConvolutionOpInfo &opInfo, const bool transposed, int64_t groups, const aclIntArray *stride,
                   const aclIntArray *padding, const aclIntArray *dilation, int8_t cubeMathType) {
  GetConvolutionOpDtype(input, weight, bias, output, opInfo, transposed, cubeMathType);
  // 支持C04 + NCHW + 非transposed的场景
  op::Shape inputSpecialShape = op::Shape({320, 3, 224, 224});  // 客户专用场景
  op::Shape weightSpecialShape = op::Shape({768, 3, 32, 32});  // 客户专用场景
  if ((weight->GetViewShape() == weightSpecialShape) && (input->GetViewShape() == inputSpecialShape) &&
    CanSwitchC04InF16Scene(opInfo) &&
    CanSwitchC04(input, weight, bias, output, stride, padding, dilation, groups, transposed)) {
    OP_LOGD("Entering float16 C04 branch");
    GetConvolutionOpFormatC04(opInfo);
  } else if (CanSwitchC04InBF16Scene(opInfo) &&
            CanSwitchC04(input, weight, bias, output, stride, padding, dilation, groups, transposed)) {
    OP_LOGD("Entering bfloat16 C04 branch");
    GetConvolutionOpFormatC04(opInfo);
  } else {
    Conv2DSplitWInfo conv2dInfo;
    conv2dInfo.InitConv2DSplitWInfo(input, weight, stride, padding, dilation);
    if (conv2dInfo.CanSwitchSplitW(bias, output, groups, opInfo)){
      OP_LOGD("Entering splitW branch");
      GetConvolution3dOpFormat(opInfo);
    } else {
      OP_LOGD("Entering normal C04 branch");
      GetConvolutionOpFormat(opInfo);
    }
  }
}

void GetConv3dOpInfo(const aclTensor *input, const aclTensor *weight, const aclTensor *bias, aclTensor *output,
                   struct ConvolutionOpInfo &opInfo, const bool transposed, int64_t groups, const aclIntArray *stride,
                   const aclIntArray *padding, const aclIntArray *dilation, int8_t cubeMathType) {
  GetConvolutionOpDtype(input, weight, bias, output, opInfo, transposed, cubeMathType);
  GetConvolution3dOpFormat(opInfo);
}

static aclIntArray *ViewConv1dPad1dAs4d(const aclIntArray *intArray, aclOpExecutor *executor) {
  const uint64_t newDimSize = 4;
  int64_t data[newDimSize];
  uint64_t size = intArray->Size();
  data[0] = 0;
  data[1] = 0;
  data[PAD_LEFT_INDEX] = (*intArray)[0];
  data[PAD_RIGHT_INDEX] = (*intArray)[0];
  aclIntArray *newArray = executor->AllocIntArray(data, newDimSize);
  return newArray;
}

static aclIntArray *ViewConv1dPad2dAs4d(const aclIntArray *intArray, aclOpExecutor *executor) {
  const uint64_t newDimSize = 4;
  int64_t data[newDimSize];
  uint64_t size = intArray->Size();
  data[0] = 0;
  data[1] = 0;
  data[PAD_LEFT_INDEX] = (*intArray)[0];
  data[PAD_RIGHT_INDEX] = (*intArray)[1];
  aclIntArray *newArray = executor->AllocIntArray(data, newDimSize);
  return newArray;
}

static aclIntArray *ViewConv2dPad2dAs4d(const aclIntArray *intArray, aclOpExecutor *executor) {
  const uint64_t newDimSize = 4;
  int64_t data[newDimSize];
  uint64_t size = intArray->Size();
  data[0] = (*intArray)[0];
  data[1] = (*intArray)[0];
  data[PAD_LEFT_INDEX] = (*intArray)[1];
  data[PAD_RIGHT_INDEX] = (*intArray)[1];
  aclIntArray *newArray = executor->AllocIntArray(data, newDimSize);
  return newArray;
}

aclIntArray *View1dAs2d(const aclIntArray *intArray, int64_t expendValue, aclOpExecutor *executor) {
  const uint64_t newDimSize = 2;
  int64_t data[newDimSize];
  uint64_t size = intArray->Size();
  if (size != 1) {
    return nullptr;
  }
  data[0] = expendValue;
  data[1] = (*intArray)[0];
  aclIntArray *newArray = executor->AllocIntArray(data, newDimSize);
  return newArray;
}

aclIntArray *ViewValueAs1d(const int64_t value, aclOpExecutor *executor) {
  int64_t data[1];
  data[0] = value;
  aclIntArray *newArray = executor->AllocIntArray(data, 1);
  return newArray;
}

const aclTensor *View1dAs4d(const aclTensor *input, aclOpExecutor *executor) {
  // input NCL->contigious->unsqueeze(2)->reformat NCHW
  // 非连续转连续contigious
  auto contiguousInput = l0op::Contiguous(input, executor);
  CHECK_RET(contiguousInput != nullptr, nullptr);

  // unsqeeze(2)
  const int64_t appendDim[] = {0, 2, 3};
  aclIntArray *dim = executor->AllocIntArray(appendDim, 3);
  auto unsqueezedInput = l0op::UnsqueezeNd(contiguousInput, dim, executor);
  CHECK_RET(unsqueezedInput != nullptr, nullptr);

  // reformat
  auto reformatInput = l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCHW);
  CHECK_RET(reformatInput != nullptr, nullptr);

  return reformatInput;
}

static const aclTensor *View3dAs4d(const aclTensor *input, aclOpExecutor *executor) {
  // input NCL->contigious->unsqueeze(2)->reformat NCHW
  // 非连续转连续contigious
  auto contiguousInput = l0op::Contiguous(input, executor);
  CHECK_RET(contiguousInput != nullptr, nullptr);

  // unsqeeze(2)
  const int64_t appendDim[] = {2};
  aclIntArray *dim = executor->AllocIntArray(appendDim, 1);
  auto unsqueezedInput = l0op::UnsqueezeNd(contiguousInput, dim, executor);
  CHECK_RET(unsqueezedInput != nullptr, nullptr);

  // reformat
  auto reformatInput = l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCHW);
  CHECK_RET(reformatInput != nullptr, nullptr);

  return reformatInput;
}

static const aclTensor *View4dAs3d(const aclTensor *input, aclOpExecutor *executor) {
  // input NCL->contigious->unsqueeze(2)->reformat NCHW
  // 非连续转连续contigious
  auto contiguousInput = l0op::Contiguous(input, executor);
  CHECK_RET(contiguousInput != nullptr, nullptr);
  // unsqeeze(2)
  const int64_t appendDim[] = {2};
  aclIntArray *dim = executor->AllocIntArray(appendDim, 1);
  auto squeezedInput = l0op::SqueezeNd(contiguousInput, dim, executor);
  CHECK_RET(squeezedInput != nullptr, nullptr);

  // reformat
  auto reformatInput = l0op::ReFormat(squeezedInput, op::Format::FORMAT_NCL);
  CHECK_RET(reformatInput != nullptr, nullptr);

  return reformatInput;
}

static const aclTensor *Permute(const aclTensor *input, FVector<int64_t> dims, aclOpExecutor *executor) {
  // contigious
  auto contiguousInput = l0op::Contiguous(input, executor);
  CHECK_RET(contiguousInput != nullptr, nullptr);
  // Transpose
  auto *perm = executor->AllocIntArray(dims.data(), dims.size());
  CHECK_RET(perm != nullptr, nullptr);

  auto *result = l0op::Transpose(contiguousInput, perm, executor);
  CHECK_RET(result != nullptr, nullptr);

  return result;
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

static aclnnStatus CheckConv2dWithWeightFZ(const aclTensor *input, const aclTensor *weight)
{
    if (weight->GetStorageFormat() != Format::FORMAT_FRACTAL_Z) {
        return ACLNN_SUCCESS;
    }
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND310P) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Current weight format is Internal Format, only support soc 310P! Current soc is %s.",
                op::ToString(GetCurrentPlatformInfo().GetSocVersion()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (input->GetDataType() != weight->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Current weight format is Internal Format, fmap should be same dtype with weight! "
                "input dtype: %s, weight dtype: %s.",
                op::ToString(input->GetDataType()).GetString(),
                op::ToString(weight->GetDataType()).GetString());
            return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
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

class ConvTbcImpl : public ConvolutionImpl {
 public:
  CONV_CONSTRUCTOR(Tbc)

  aclnnStatus PreProcess() {
    REG_L0_FUNCTION(l0Functions, Conv2d5HdFp16, op::Format::FORMAT_NC1HWC0, op::DataType::DT_FLOAT16,
                    op::Format::FORMAT_NC1HWC0, op::DataType::DT_FLOAT16);
    REG_L0_FUNCTION(l0Functions, Conv2d5HdFp32, op::Format::FORMAT_NC1HWC0, op::DataType::DT_FLOAT,
                    op::Format::FORMAT_NC1HWC0, op::DataType::DT_FLOAT);
    REG_L0_FUNCTION(l0Functions, Conv2d5HdFp1625HdFp32, op::Format::FORMAT_NC1HWC0, op::DataType::DT_FLOAT16,
                    op::Format::FORMAT_NC1HWC0, op::DataType::DT_FLOAT);
    REG_L0_FUNCTION(l0Functions, Conv2d5HdBf16, op::Format::FORMAT_NC1HWC0, op::DataType::DT_BF16,
                    op::Format::FORMAT_NC1HWC0, op::DataType::DT_BF16);
    // conv1d is implemented by 2d, so first change view of input, weight, bias
    stride = View1dAs2d(stride, 1, executor);
    CHECK_RET(stride != nullptr, ACLNN_ERR_INNER_NULLPTR);

    padding = View1dAs2d(padding, 0, executor);
    CHECK_RET(padding != nullptr, ACLNN_ERR_INNER_NULLPTR);

    dilation = View1dAs2d(dilation, 1, executor);
    CHECK_RET(dilation != nullptr, ACLNN_ERR_INNER_NULLPTR);

    input = View3dAs4d(input, executor);
    CHECK_RET(input != nullptr, ACLNN_ERR_INNER_NULLPTR);

    weight = View3dAs4d(weight, executor);
    CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);

    bias = View1dAs4d(bias, executor);
    CHECK_RET(bias != nullptr, ACLNN_ERR_INNER_NULLPTR);

    GetConvOpInfo(input, weight, bias, output, opInfo, transposed, groups, stride, padding, dilation, cubeMathType);
    // 调用静态函数PreProcess
    return CommonPreProcess(input, weight, bias, groups, transposed, opInfo, true, false, executor);
  };

  aclnnStatus Impl() {
    // conv1d is implement by conv2d
    convOut = FUNCTION_CALL(l0Functions, opInfo, input, weight, bias, stride, padding, dilation, transposed,
                            outputPadding, groups, useHf32, executor);
    if (convOut == nullptr) {
      OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "convTbc raise an unknown error");
      return ACLNN_ERR_RUNTIME_ERROR;
    }
    return ACLNN_SUCCESS;
  };

  aclnnStatus PostProcess() {
    // 因仅支持NCL格式的conv1d，所以转为conv2d的format默认为HCHW
    auto fakeOutput2d = executor->AllocTensor(output->GetDataType(), op::Format::FORMAT_NCHW, op::Format::FORMAT_NCHW);

    // 调用静态函数PostProcess
    auto res = CommonPostProcess(groups, true, fakeOutput2d, convOut, executor);
    CHECK_RET(res == ACLNN_SUCCESS, res);
    // 现在Conv1d转为conv2d来做，所以需要转换输出
    convOut = View4dAs3d(convOut, executor);
    CHECK_RET(convOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    // permute to [T, B, C]
    FVector<int64_t> permuteDim {2, 0, 1};
    auto permuteConvTbc = Permute(convOut, permuteDim, executor);
    CHECK_RET(permuteConvTbc != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    // view copy
    auto castConvTbc = l0op::ReFormat(permuteConvTbc, Format::FORMAT_ND);
    auto result = l0op::ViewCopy(castConvTbc, output, executor);
    CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    return ACLNN_SUCCESS;
  };
  ~ConvTbcImpl(){};
};

class Conv1dImpl : public ConvolutionImpl {
 public:
  CONV_CONSTRUCTOR(1d)

  aclnnStatus PreProcess() {
    REG_L0_FUNCTION(l0Functions, Conv2d5HdFp16, op::Format::FORMAT_NC1HWC0, op::DataType::DT_FLOAT16,
                    op::Format::FORMAT_NC1HWC0, op::DataType::DT_FLOAT16);
    REG_L0_FUNCTION(l0Functions, Conv2d5HdFp32, op::Format::FORMAT_NC1HWC0, op::DataType::DT_FLOAT,
                    op::Format::FORMAT_NC1HWC0, op::DataType::DT_FLOAT);
    REG_L0_FUNCTION(l0Functions, Conv2d5HdFp1625HdFp32, op::Format::FORMAT_NC1HWC0, op::DataType::DT_FLOAT16,
                    op::Format::FORMAT_NC1HWC0, op::DataType::DT_FLOAT);
    REG_L0_FUNCTION(l0Functions, Conv2d5HdBf16, op::Format::FORMAT_NC1HWC0, op::DataType::DT_BF16,
                    op::Format::FORMAT_NC1HWC0, op::DataType::DT_BF16);

    // conv1d is implemented by 2d, so first change view of input, weight, bias
    stride = View1dAs2d(stride, 1, executor);
    CHECK_RET(stride != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (padding->Size() == 1) {
      padding = ViewConv1dPad1dAs4d(padding, executor);
      CHECK_RET(padding != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else if (padding->Size() == CONV_2D_PAD_DIM) {
      padding = ViewConv1dPad2dAs4d(padding, executor);
      CHECK_RET(padding != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
      OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "conv1d pad dim not equal to 1 or 2");
      return ACLNN_ERR_INNER_NULLPTR;
    }

    dilation = View1dAs2d(dilation, 1, executor);
    CHECK_RET(dilation != nullptr, ACLNN_ERR_INNER_NULLPTR);

    input = View3dAs4d(input, executor);
    CHECK_RET(input != nullptr, ACLNN_ERR_INNER_NULLPTR);

    weight = View3dAs4d(weight, executor);
    CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (bias != nullptr) {
      if (bias->GetViewShape().GetDimNum() == 3) {  // 输入维度为3
        bias = View3dAs4d(bias, executor);
      } else {
        // bias dim = 1, 其他dim在check时候返回
        bias = View1dAs4d(bias, executor);
      }
      CHECK_RET(bias != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    GetConvOpInfo(input, weight, bias, output, opInfo, transposed, groups, stride, padding, dilation, cubeMathType);
    specialConv1d = isSpecialConv1d(input, weight, stride, padding, dilation) && (groups == 1);
    // 调用静态函数PreProcess
    return CommonPreProcess(input, weight, bias, groups, transposed, opInfo, !specialConv1d, false, executor);
  };

  aclnnStatus Impl() {
    if (specialConv1d) {
      // assert x and weight format = NCHW, C=H=1
      // x view to shape n*w/s s, the batch dim is fold to the dim 1,
      op::Shape inputShape2d =
          op::Shape({input->GetViewShape()[0] * input->GetViewShape()[3] / (*stride)[1], (*stride)[1]});
      auto input2d = ViewWithShape(input, inputShape2d, executor);
      CHECK_RET(input2d != nullptr, ACLNN_ERR_INNER_NULLPTR);

      // weight reshape to shape Cout s
      op::Shape weightShape2d = op::Shape({weight->GetViewShape()[0], (*stride)[1]});
      auto weight2d = ViewWithShape(weight, weightShape2d, executor);
      CHECK_RET(weight2d != nullptr, ACLNN_ERR_INNER_NULLPTR);

      // weight perpute to shape s Cout
      FVector<int64_t> dims {1, 0};
      auto permutWeight = Permute(weight2d, dims, executor);
      CHECK_RET(permutWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);

      auto input2dND = l0op::ReFormat(input2d, op::Format::FORMAT_ND);
      auto permutWeightND = l0op::ReFormat(permutWeight, op::Format::FORMAT_ND);
      // matmul (x,weight) to shape n*w/s Cout
      auto mmOut = ExecMmOp(input2dND, permutWeightND, 0, executor);
      CHECK_RET(mmOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

      // matmul output reshape to shape n w/s Cout
      op::Shape mmOut3dShape =
          op::Shape({input->GetViewShape()[0], input->GetViewShape()[3] / (*stride)[1], weight->GetViewShape()[0]});
      auto mmOut3d = ViewWithShape(mmOut, mmOut3dShape, executor);
      CHECK_RET(mmOut3d != nullptr, ACLNN_ERR_INNER_NULLPTR);
      auto mmOut3dNCL = l0op::ReFormat(mmOut3d, op::Format::FORMAT_NCL);

      // matmul output contiguous
      auto contiguousMmOut3d = l0op::Contiguous(mmOut3dNCL, executor);
      CHECK_RET(contiguousMmOut3d != nullptr, ACLNN_ERR_INNER_NULLPTR);

      // matmul output permut to shape n Cout w/s
      dims = {0, 2, 1};
      auto permutMmOut3d = Permute(contiguousMmOut3d, dims, executor);
      CHECK_RET(permutMmOut3d != nullptr, ACLNN_ERR_INNER_NULLPTR);

      auto output3dNCL = l0op::ReFormat(permutMmOut3d, op::Format::FORMAT_NCL);
      convOut = output3dNCL;

      return ACLNN_SUCCESS;
    }
    // conv1d is implement by conv2d
    convOut = FUNCTION_CALL(l0Functions, opInfo, input, weight, bias, stride, padding, dilation, transposed,
                            outputPadding, groups, useHf32, executor);
    if (convOut == nullptr) {
      OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "conv1d raise an unknown error");
      return ACLNN_ERR_RUNTIME_ERROR;
    }
    return ACLNN_SUCCESS;
  };

  aclnnStatus PostProcess() {
    // conv1d 转换为conv2d做，所以后处理先按照conv2d的处理方式处理输出
    // 因仅支持NCL格式的conv1d，所以转为conv2d的format默认为HCHW
    auto fakeOutput2d = executor->AllocTensor(output->GetDataType(), op::Format::FORMAT_NCHW, op::Format::FORMAT_NCHW);

    // 调用静态函数PostProcess
    auto res = CommonPostProcess(groups, !specialConv1d, fakeOutput2d, convOut, executor);
    CHECK_RET(res == ACLNN_SUCCESS, res);
    // 现在Conv1d转为conv2d来做，所以需要转换输出
    if (!specialConv1d) {
      convOut = View4dAs3d(convOut, executor);
      CHECK_RET(convOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }

    auto result = l0op::ViewCopy(convOut, output, executor);
    CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    return ACLNN_SUCCESS;
  };

  ~Conv1dImpl(){};

 private:
  bool isSpecialConv1d(const aclTensor *inputParam, const aclTensor *weightParam, const aclIntArray *strideParam,
                       const aclIntArray *paddingParam, const aclIntArray *dilationParam) {
    if ((*strideParam)[1] > specialStride && (*strideParam)[1] == weightParam->GetViewShape()[specialChannelIndex] &&
        (*paddingParam)[PAD_LEFT_INDEX] == 0 && (*paddingParam)[PAD_RIGHT_INDEX] == 0 && (*dilationParam)[1] == 1 &&
        inputParam->GetViewShape()[1] == 1) {
      return true;
    } else {
      return false;
    }
  }
  bool specialConv1d = false;
};

class Conv2dImpl : public ConvolutionImpl {
 public:
  CONV_CONSTRUCTOR(2d)

  aclnnStatus PreProcess() {
    REG_L0_FUNCTION(l0Functions, Conv2d5HdFp16, op::Format::FORMAT_NC1HWC0, op::DataType::DT_FLOAT16,
                    op::Format::FORMAT_NC1HWC0, op::DataType::DT_FLOAT16);
    REG_L0_FUNCTION(l0Functions, Conv2d5HdFp32, op::Format::FORMAT_NC1HWC0, op::DataType::DT_FLOAT,
                    op::Format::FORMAT_NC1HWC0, op::DataType::DT_FLOAT);
    REG_L0_FUNCTION(l0Functions, Conv2d5HdFp1625HdFp32, op::Format::FORMAT_NC1HWC0, op::DataType::DT_FLOAT16,
                    op::Format::FORMAT_NC1HWC0, op::DataType::DT_FLOAT);
    REG_L0_FUNCTION(l0Functions, Conv2d5HdBf16, op::Format::FORMAT_NC1HWC0, op::DataType::DT_BF16,
                    op::Format::FORMAT_NC1HWC0, op::DataType::DT_BF16);
    if (padding->Size() != CONV_2D_PAD_DIM && padding->Size() != CONV_4D_PAD_DIM) {
      OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "conv2d pad size is not in 2 or 4");
      return ACLNN_ERR_RUNTIME_ERROR;
    }
    if (padding->Size() == CONV_2D_PAD_DIM) {
      padding = ViewConv2dPad2dAs4d(padding, executor);
      CHECK_RET(padding != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    auto ret = CheckConv2dWithWeightFZ(input, weight);
    CHECK_RET(ret == ACL_SUCCESS, ret);

    GetConvOpInfo(input, weight, bias, output, opInfo, transposed, groups, stride, padding, dilation, cubeMathType);

    // 需要切C04分支卷积
    if (opInfo.weightFormat == Format::FORMAT_FRACTAL_Z_C04 && weight->GetDataType() == op::DataType::DT_FLOAT16) {
      OP_LOGD("Conv2d entering float16 C04 branch");
      return CommonPreProcessC04(input, weight, bias, groups, transposed, opInfo, true, true, executor);
    }
    if (opInfo.weightFormat == Format::FORMAT_FRACTAL_Z_C04 && weight->GetDataType() == op::DataType::DT_BF16) {
      OP_LOGD("Conv2d entering bfloat16 C04 branch");
    } else if (opInfo.inputFormat == Format::FORMAT_NDC1HWC0) {
      OP_LOGD("Conv2d entering splitW branch");
      auto changeRes = ChangeConv2dAttrToConv3d(stride, padding, dilation, executor);
      CHECK_RET(changeRes == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
      changeRes = ChangeConv2dInputToConv3d(input, weight, executor);
      CHECK_RET(changeRes == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
      REG_L0_FUNCTION(l0Functions, Conv3dv26HdFp16, op::Format::FORMAT_NDC1HWC0, op::DataType::DT_FLOAT16,
                    op::Format::FORMAT_NDC1HWC0, op::DataType::DT_FLOAT16);
      OP_LOGD("convolution aclnn op inputDtype: %s, outputDtype: %s, biasDtype: %s, useHf32: %d.",
            op::ToString(opInfo.inputDtype).GetString(), op::ToString(opInfo.outputDtype).GetString(),
            op::ToString(opInfo.biasDtype).GetString(), useHf32);
    } else {
      OP_LOGD("Conv2d entering normal branch");
    }
    // 调用静态函数PreProcess
    return CommonPreProcess(input, weight, bias, groups, transposed, opInfo, true, true, executor);
  };

  aclnnStatus Impl() {
    convOut = FUNCTION_CALL(l0Functions, opInfo, input, weight, bias, stride, padding, dilation, transposed,
                            outputPadding, groups, useHf32, executor);
    if (convOut == nullptr) {
      OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "conv2d raise an unknown error");
      return ACLNN_ERR_RUNTIME_ERROR;
    }
    return ACLNN_SUCCESS;
  };

  aclnnStatus PostProcess() {
    if (opInfo.inputFormat != Format::FORMAT_NDC1HWC0) {
      auto res = CommonPostProcess(groups, true, output, convOut, executor);
      CHECK_RET(res == ACLNN_SUCCESS, res);
    } else {
      // splitw模式，会使得conv2d转为conv3d做，所以后处理先按照conv3d的处理方式输出
      auto fakeOutput3d = executor->AllocTensor(output->GetDataType(),
                                            op::Format::FORMAT_NCDHW, op::Format::FORMAT_NCDHW);
      CHECK_RET(fakeOutput3d != nullptr, ACLNN_ERR_INNER_NULLPTR);
      auto res = CommonPostProcess(groups, true, fakeOutput3d, convOut, executor);
      CHECK_RET(res == ACLNN_SUCCESS, res);
      convOut = View5dAs4dForOutput(convOut, executor);
      CHECK_RET(convOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    auto result = l0op::ViewCopy(convOut, output, executor);
    CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    return ACLNN_SUCCESS;
  };
  ~Conv2dImpl(){};
};

class Conv3dImpl : public ConvolutionImpl {
 public:
  CONV_CONSTRUCTOR(3d)

  aclnnStatus PreProcess() {
    REG_L0_FUNCTION(l0Functions, Conv3d6HdFp16, op::Format::FORMAT_NDC1HWC0, op::DataType::DT_FLOAT16,
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
class ConvTransposed1dImpl : public ConvolutionImpl {
 public:
  CONV_CONSTRUCTOR(Transposed1d)

  aclnnStatus PreProcess() {
    REG_L0_FUNCTION(l0Functions, ConvTranspose2d5HdFp16, op::Format::FORMAT_NC1HWC0, op::DataType::DT_FLOAT16,
                    op::Format::FORMAT_NC1HWC0, op::DataType::DT_FLOAT16);
    REG_L0_FUNCTION(l0Functions, ConvTranspose2d5HdFp32, op::Format::FORMAT_NC1HWC0, op::DataType::DT_FLOAT,
                    op::Format::FORMAT_NC1HWC0, op::DataType::DT_FLOAT);
    REG_L0_FUNCTION(l0Functions, ConvTranspose2d5HdBf16, op::Format::FORMAT_NC1HWC0, op::DataType::DT_BF16,
                    op::Format::FORMAT_NC1HWC0, op::DataType::DT_BF16);

    stride = View1dAs2d(stride, 1, executor);
    CHECK_RET(stride != nullptr, ACLNN_ERR_INNER_NULLPTR);

    padding = View1dAs2d(padding, 0, executor);
    CHECK_RET(padding != nullptr, ACLNN_ERR_INNER_NULLPTR);

    dilation = View1dAs2d(dilation, 1, executor);
    CHECK_RET(dilation != nullptr, ACLNN_ERR_INNER_NULLPTR);

    outputPadding = View1dAs2d(outputPadding, 0, executor);
    CHECK_RET(outputPadding != nullptr, ACLNN_ERR_INNER_NULLPTR);

    input = View3dAs4d(input, executor);
    CHECK_RET(input != nullptr, ACLNN_ERR_INNER_NULLPTR);

    weight = View3dAs4d(weight, executor);
    CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (bias != nullptr) {
      if (bias->GetViewShape().GetDimNum() == 3) {  // 输入维度为3
        bias = View3dAs4d(bias, executor);
      } else {
        // bias dim = 1, 其他dim在check时候返回
        bias = View1dAs4d(bias, executor);
      }
      CHECK_RET(bias != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    GetConvOpInfo(input, weight, bias, output, opInfo, transposed, groups, stride, padding, dilation, cubeMathType);
    // 调用静态函数PreProcess
    return CommonPreProcess(input, weight, bias, groups, transposed, opInfo, true, false, executor);
  };

  aclnnStatus Impl() {
    // conv1d is implement by conv2d
    convOut = FUNCTION_CALL(l0Functions, opInfo, input, weight, bias, stride, padding, dilation, transposed,
                            outputPadding, groups, useHf32, executor);
    if (convOut == nullptr) {
      OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "conv1d raise an unknown error");
      return ACLNN_ERR_RUNTIME_ERROR;
    }
    return ACLNN_SUCCESS;
  };

  aclnnStatus PostProcess() {
    // conv1d 转换为conv2d做，所以后处理先按照conv2d的处理方式处理输出
    // 因仅支持NCL格式的conv1d，所以转为conv2d的format默认为HCHW
    auto fakeOutput2d = executor->AllocTensor(output->GetDataType(), op::Format::FORMAT_NCHW, op::Format::FORMAT_NCHW);

    // 调用静态函数PostProcess
    auto res = CommonPostProcess(groups, true, fakeOutput2d, convOut, executor);
    CHECK_RET(res == ACLNN_SUCCESS, res);
    // 现在Conv1d转为conv2d来做，所以需要转换输出
    convOut = View4dAs3d(convOut, executor);
    CHECK_RET(convOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto result = l0op::ViewCopy(convOut, output, executor);
    CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    return ACLNN_SUCCESS;
  };
  ~ConvTransposed1dImpl(){};
};
class ConvTransposed2dImpl : public ConvolutionImpl {
 public:
  CONV_CONSTRUCTOR(Transposed2d)

  aclnnStatus PreProcess() {
    REG_L0_FUNCTION(l0Functions, ConvTranspose2d5HdFp16, op::Format::FORMAT_NC1HWC0, op::DataType::DT_FLOAT16,
                    op::Format::FORMAT_NC1HWC0, op::DataType::DT_FLOAT16);
    REG_L0_FUNCTION(l0Functions, ConvTranspose2d5HdFp32, op::Format::FORMAT_NC1HWC0, op::DataType::DT_FLOAT,
                    op::Format::FORMAT_NC1HWC0, op::DataType::DT_FLOAT);
    REG_L0_FUNCTION(l0Functions, ConvTranspose2d5HdBf16, op::Format::FORMAT_NC1HWC0, op::DataType::DT_BF16,
                    op::Format::FORMAT_NC1HWC0, op::DataType::DT_BF16);

    GetConvOpInfo(input, weight, bias, output, opInfo, transposed, groups, stride, padding, dilation, cubeMathType);
    // 调用静态函数PreProcess
    return CommonPreProcess(input, weight, bias, groups, transposed, opInfo, true, true, executor);
  };

  aclnnStatus Impl() {
    convOut = FUNCTION_CALL(l0Functions, opInfo, input, weight, bias, stride, padding, dilation, transposed,
                            outputPadding, groups, useHf32, executor);
    if (convOut == nullptr) {
      OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "convTranspose2d raise an unknown error");
      return ACLNN_ERR_RUNTIME_ERROR;
    }
    return ACLNN_SUCCESS;
  };

  aclnnStatus PostProcess() {
    auto res = CommonPostProcess(groups, true, output, convOut, executor);
    CHECK_RET(res == ACLNN_SUCCESS, res);

    auto result = l0op::ViewCopy(convOut, output, executor);
    CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    return ACLNN_SUCCESS;
  };
  ~ConvTransposed2dImpl(){};
};
class ConvTransposed3dImpl : public ConvolutionImpl {
 public:
  CONV_CONSTRUCTOR(Transposed3d)

  aclnnStatus PreProcess() {
    REG_L0_FUNCTION(l0Functions, ConvTranspose3d6HdFp16, op::Format::FORMAT_NDC1HWC0, op::DataType::DT_FLOAT16,
                    op::Format::FORMAT_NDC1HWC0, op::DataType::DT_FLOAT16);
    REG_L0_FUNCTION(l0Functions, ConvTranspose3d6HdFp32, op::Format::FORMAT_NDC1HWC0, op::DataType::DT_FLOAT,
                    op::Format::FORMAT_NDC1HWC0, op::DataType::DT_FLOAT);
    REG_L0_FUNCTION(l0Functions, ConvTranspose3d6HdBf16, op::Format::FORMAT_NDC1HWC0, op::DataType::DT_BF16,
                    op::Format::FORMAT_NDC1HWC0, op::DataType::DT_BF16);
    GetConv3dOpInfo(input, weight, bias, output, opInfo, transposed, groups, stride, padding, dilation, cubeMathType);
    // 调用静态函数PreProcess
    return CommonPreProcess(input, weight, bias, groups, transposed, opInfo, true, true, executor);
  };

  aclnnStatus Impl() {
    convOut = FUNCTION_CALL(l0Functions, opInfo, input, weight, bias, stride, padding, dilation, transposed,
                            outputPadding, groups, useHf32, executor);
    if (convOut == nullptr) {
      OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "convTranspose3d raise an unknown error");
      return ACLNN_ERR_RUNTIME_ERROR;
    }
    return ACLNN_SUCCESS;
  };

  aclnnStatus PostProcess() {
     // output format transdata
    convOut = l0op::TransData(convOut, output->GetStorageFormat(), groups, executor);
    CHECK_RET(convOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (bias) {
      op::Shape biasShape = bias->GetViewShape();
      int64_t biasLength = biasShape.GetDim(0);
      bias = l0op::Reshape(bias, {1, biasLength, 1, 1, 1}, executor);
      CHECK_RET(bias != nullptr, ACLNN_ERR_INNER_NULLPTR);

      convOut = l0op::Add(convOut, bias, executor);
      CHECK_RET(convOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // output cast
    convOut = l0op::Cast(convOut, output->GetDataType(), executor);
    CHECK_RET(convOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    
    auto result = l0op::ViewCopy(convOut, output, executor);
    CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    return ACLNN_SUCCESS;
  };

  ~ConvTransposed3dImpl(){};
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
  if (tbc) {
    return new ConvTbcImpl(input, weight, bias, stride, padding, dilation, transposed, outputPadding, groups, output,
                           useHf32, cubeMathType, executor);
  }
  if (!transposed) {
    switch (inputDim) {
      case CONV_1D_DIM_SIZE: {
        return new (std::nothrow) Conv1dImpl(input, weight, bias, stride, padding, dilation, transposed, outputPadding,
                                             groups, output, useHf32, cubeMathType, executor);
      }
      case CONV_2D_DIM_SIZE: {
        return new (std::nothrow) Conv2dImpl(input, weight, bias, stride, padding, dilation, transposed, outputPadding,
                                             groups, output, useHf32, cubeMathType, executor);
      }
      case CONV_3D_DIM_SIZE: {
        return new (std::nothrow) Conv3dImpl(input, weight, bias, stride, padding, dilation, transposed, outputPadding,
                                             groups, output, useHf32, cubeMathType, executor);
      }
      default:
        return nullptr;
    }
  }
  switch (inputDim) {
    case CONV_1D_DIM_SIZE: {
      return new (std::nothrow) ConvTransposed1dImpl(input, weight, bias, stride, padding, dilation, transposed,
                                                     outputPadding, groups, output, useHf32, cubeMathType, executor);
    }
    case CONV_2D_DIM_SIZE: {
      return new (std::nothrow) ConvTransposed2dImpl(input, weight, bias, stride, padding, dilation, transposed,
                                                     outputPadding, groups, output, useHf32, cubeMathType, executor);
    }
    case CONV_3D_DIM_SIZE: {
      return new (std::nothrow) ConvTransposed3dImpl(input, weight, bias, stride, padding, dilation, transposed,
                                                     outputPadding, groups, output, useHf32, cubeMathType, executor);
    }
    default:
      return nullptr;
  }
}

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

  // 空tensor的情况，由于外部已经将output的shape,type,format设置好，故不需要做任何操作，直接返回
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
