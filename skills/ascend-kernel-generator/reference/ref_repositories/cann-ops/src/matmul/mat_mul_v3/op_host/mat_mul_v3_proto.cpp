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
 * \file matmul_infer_fns.cc
 * \brief
 */
#include <map>
#include <string>
#include <sstream>
#include <vector>
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"

using namespace std;

namespace ops {
#define CUBE_INNER_ERR_REPORT(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_LOGI(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_LOGW(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_LOGE(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_LOGD(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)
}  // namespace ops

namespace ge {
/*
 * get debug string of vector
 * param[in] v vector
 * return vector's debug string
 */
template <typename T>
std::string DebugString(const std::vector<T>& v) {
  std::ostringstream oss;
  oss << "[";
  if (v.size() > 0) {
    for (size_t i = 0; i < v.size() - 1; ++i) {
      oss << v[i] << ", ";
    }
    oss << v[v.size() - 1];
  }
  oss << "]";
  return oss.str();
}

template <typename T>
std::string DebugString(const std::vector<std::pair<T, T>>& v) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    oss << "(" << v[i].first << ", " <<v[i].second << ")";
  }
  oss << "]";
  return oss.str();
}
}

namespace ops {
/*
 * @brief: trans the gert::Shape to vector<int64_t>
 * @param [in] format: gert::Shape
 * @return vector<int64_t>: the vector shape
 */
std::vector<int64_t> ToVector(const gert::Shape& shape);

/*
 * @brief: trans the gert::TypedContinuousVector<T> to vector<T>
 * @param [in] vec: reference of gert::TypedContinuousVector<T>
 * @return vector<T>: the vector of T
 */
template <typename T>
std::vector<T> ToVector(const gert::TypedContinuousVector<T>& vec) {
  size_t vec_size = vec.GetSize();
  std::vector<T> vec_t(vec_size, 0);

  for (size_t i = 0; i < vec_size; i++) {
    vec_t[i] = *(vec.GetData() + i);
  }
  return vec_t;
}

std::string ToString(const ge::DataType& type);

/*
 * @brief: get format string from enum
 * @param [in] format: enum format
 * @return string: format string
 */
std::string ToString(const ge::Format& format);

/*
 * @brief: get shape string from gert::Shape, for debug
 * @param [in] shape: reference of gert::Shape
 * @return string: shape string
 */
std::string ToString(const gert::Shape& shape);

/*
 * @brief: get shape string from gert::Shape, for debug
 * @param [in] shape: ptr of gert::Shape
 * @return string: shape string
 */
std::string ToString(const gert::Shape* shape);

std::string ToString(const std::vector<int64_t>& shape);
std::string ToString(const std::vector<gert::Shape>& shapes);

/*
 * @brief: get TypedContinuousVector string from gert::TypedContinuousVector&, for debug
 * @param [in] vec: reference of gert::TypedContinuousVector<T>
 * @return string: TypedContinuousVector string
 */
template <typename T>
std::string ToString(const gert::TypedContinuousVector<T>& vec) {
  return ge::DebugString(ToVector(vec));
}

/*
 * @brief: get TypedContinuousVector string from gert::TypedContinuousVector*, for debug
 * @param [in] vec: ptr of gert::TypedContinuousVector<T>
 * @return string: TypedContinuousVector string
 */
template <typename T>
std::string ToString(const gert::TypedContinuousVector<T>* vec) {
  return ge::DebugString(ToVector(*vec));
}

template <typename T>
std::string ToString(const T* value, size_t size) {
  std::string r = "[";
  for (size_t i = 0; i < size; i++) {
    r = r + std::to_string(value[i]) + ", ";
  }
  r = r + "]";
  return r;
}
}

namespace {
constexpr int64_t kFraczK1Index = 0;
constexpr int64_t kFraczN1Index = 1;
constexpr int64_t kFraczN0Index = 2;
constexpr int64_t kFraczK0Index = 3;
constexpr int64_t kK1CopressRatio = 2;
const size_t kCompressFcOpShapeSize = 1;
const size_t kCompressFcOpMaxShapeSize = 4;
const size_t KCompressInfoValue = 3;
const size_t kMatmulV2MinShapeSize = 2;
const size_t kMatmulV2MaxShapeSize = 4;
const size_t kBatchMatmulMinShapeSize = 2;
const size_t kBatchMatmulMaxShapeSize = 8;
const size_t kBatchMatMulFixpipeBiasIdx = 3;
const size_t kTransposeBatchMatMulScaleIdx = 3;
const int64_t kBlockSize = 16;
const size_t kBatchMatMulBiasIdx = 2; // input_bias index is different with op_type
std::map<string, size_t> OptypeBiasIndex = {
  {"WeightQuantBatchmatmul", 5},
  {"MatMulV2CompressDequant", 4},
  {"QuantBatchMatmul", 3},
  {"WeightQuantBatchmatmulV3", 8},
  {"QuantBatchMatmulV3", 4},
};
std::map<string, size_t> OptypeDeqScaleIndex = {
  {"WeightQuantBatchmatmul", 4},
  {"MatMulV2CompressDequant", 3},
  {"QuantBatchMatmul", 2},
  {"BatchMatmulFixpipe", 2},
};
}  // namespace

using namespace gert;
namespace ops {
static void InferComplementedInput(Shape &shape_x1_new, Shape &shape_x2_new, bool &shape_x1_reshape_flag,
                                   bool &shape_x2_reshape_flag) {
  if (shape_x1_new.GetDimNum() == 1 && shape_x1_new.GetDim(0) > 0) {
    shape_x1_reshape_flag = true;
    int64_t ori_dim = shape_x1_new.GetDim(0);
    shape_x1_new.SetDimNum(kBatchMatmulMinShapeSize);
    shape_x1_new.SetDim(0, 1);
    shape_x1_new.SetDim(1, ori_dim);
  }

  if (shape_x2_new.GetDimNum() == 1 && shape_x2_new.GetDim(0) > 0) {
    shape_x2_reshape_flag = true;
    int64_t ori_dim = shape_x2_new.GetDim(0);
    shape_x2_new.SetDimNum(kBatchMatmulMinShapeSize);
    shape_x2_new.SetDim(0, ori_dim);
    shape_x2_new.SetDim(1, 1);
  }
}

static void InferComplementedOutput(bool shape_x1_reshape_flag, bool shape_x2_reshape_flag, Shape &shape_out) {
  size_t dim_num = shape_out.GetDimNum();
  if (dim_num >= kBatchMatmulMinShapeSize) {
    if (shape_x1_reshape_flag && !shape_x2_reshape_flag) {
      shape_out.SetDim(dim_num - kBatchMatmulMinShapeSize, shape_out.GetDim(dim_num - 1));
      shape_out.SetDimNum(dim_num - 1);
    }

    if (!shape_x1_reshape_flag && shape_x2_reshape_flag) {
      shape_out.SetDimNum(dim_num - 1);
    }
  }
}

static bool TransposeShape(const Shape &src, const TypedContinuousVector<int64_t> &perm, Shape &dst) {
  if (perm.GetSize() == 0) {
    dst = src;
    return true;
  }

  if (src.GetDimNum() != perm.GetSize() || dst.GetDimNum() != 0) {
    return false;
  }

  for (size_t idx_dst = 0; idx_dst < perm.GetSize(); ++idx_dst) {
    dst.AppendDim(src.GetDim(*(perm.GetData() + idx_dst)));
  }

  return true;
}

static ge::graphStatus InferDataTypeFixedOutputType(ge::DataType output_data_type_, gert::InferDataTypeContext *context_) {
  ge::graphStatus ret = context_->SetOutputDataType(0, output_data_type_);
  auto op_name = context_->GetNodeName();
  CHECK(ret != ge::GRAPH_SUCCESS, CUBE_INNER_ERR_REPORT(op_name, "[InferDataType] Failed."),
                                                                    return ge::GRAPH_FAILED);
  OP_LOGD(op_name, "set y dtype %d", output_data_type_);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForMatMul(InferShapeContext *context, bool is_matmul_v2) {
  auto op_name = context->GetNodeName();
  auto shape_x1 = context->GetInputShape(0);
  auto shape_x2 = context->GetInputShape(1);
  auto shape_out = context->GetOutputShape(0);
  auto tensor_x1 = context->GetInputDesc(0);
  auto attrs = context->GetAttrs();
  CHECK(shape_x1 == nullptr || shape_x2 == nullptr || shape_out == nullptr || tensor_x1 == nullptr || attrs == nullptr,
        CUBE_INNER_ERR_REPORT(op_name, "shape or attrs is null"), return ge::GRAPH_FAILED);

  const bool *trans_a = attrs->GetAttrPointer<bool>(0);
  const bool *trans_b = attrs->GetAttrPointer<bool>(1);
  CHECK(trans_a == nullptr || trans_b == nullptr, CUBE_INNER_ERR_REPORT(op_name, "attribute is null"),
        return ge::GRAPH_FAILED);

  ge::DataType dtype = tensor_x1->GetDataType();
  if (dtype == ge::DT_FLOAT) {
    OP_LOGW(op_name, "%s fp32 op has poor performance!", context->GetNodeName());
  }

  Shape shape_x1_new(*shape_x1);
  Shape shape_x2_new(*shape_x2);
  bool shape_x1_reshape_flag = false;
  bool shape_x2_reshape_flag = false;
  InferComplementedInput(shape_x1_new, shape_x2_new, shape_x1_reshape_flag, shape_x2_reshape_flag);

  const Shape *shape_bias = nullptr;
  if (is_matmul_v2) {
    if (attrs->GetAttrNum() >= 5) {  // 5 attrs at least: transpose_x1, transpose_x2, offset_x, input_size, hidden_size
      auto input_size = attrs->GetAttrPointer<int64_t>(3);   // 3: input_size
      auto hidden_size = attrs->GetAttrPointer<int64_t>(4);  // 4: hidden_size
      if (input_size != nullptr && hidden_size != nullptr && *input_size > 0 && *hidden_size > 0) {
        OP_LOGD(op_name, "get private attr, input_size: %ld, hidden_size: %ld", *input_size, *hidden_size);
        shape_x2_new.SetDim(1, shape_x1_new.GetDim(1));
        int64_t align_dim = (*input_size + kBlockSize - 1) / kBlockSize * kBlockSize +
                            (*hidden_size + kBlockSize) / kBlockSize * kBlockSize;
        shape_x2_new.SetDim(0, align_dim);
      }
    }

    shape_bias = context->GetOptionalInputShape(kBatchMatMulBiasIdx);
    OP_LOGD(op_name, "check the input shape length.");
    if (shape_x1_new.GetDimNum() != kMatmulV2MinShapeSize && shape_x1_new.GetDimNum() != kMatmulV2MaxShapeSize) {
      CUBE_INNER_ERR_REPORT(op_name, "first input dim num[%zu] is not 2 or 4!", shape_x1_new.GetDimNum());
      return ge::GRAPH_FAILED;
    }
  }

  size_t idx_m = 0;
  size_t idx_k_a = 1;
  size_t idx_k_b = 0;
  size_t idx_n = 1;
  if (*trans_a) {
    idx_m = 1;
    idx_k_a = 0;
  }

  if (*trans_b) {
    idx_k_b = 1;
    idx_n = 0;
  }

  CHECK(shape_x1_new.GetDim(idx_k_a) != shape_x2_new.GetDim(idx_k_b),
        CUBE_INNER_ERR_REPORT(op_name, "The k-axis of a(%ld) and b(%ld) tensors must be the same",
                              shape_x1_new.GetDim(idx_k_a), shape_x2_new.GetDim(idx_k_b)),
        return ge::GRAPH_FAILED);

  shape_out->SetDimNum(kBatchMatmulMinShapeSize);
  shape_out->SetDim(0, shape_x1_new.GetDim(idx_m));
  shape_out->SetDim(1, shape_x2_new.GetDim(idx_n));
  if (shape_bias != nullptr && shape_bias->GetDimNum() > 0) {
    int64_t bias_dim = shape_bias->GetDimNum();
    CHECK(shape_bias->GetDim(bias_dim - 1) != shape_out->GetDim(1),
          OP_LOGE(op_name, "The dimension of n [%ld] and bias [%ld] tensors must be the same", shape_out->GetDim(1),
                  shape_bias->GetDim(bias_dim - 1)),
          return ge::GRAPH_FAILED);
  }

  InferComplementedOutput(shape_x1_reshape_flag, shape_x2_reshape_flag, *shape_out);

  OP_LOGD(op_name, "end infershape.");
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForMatMul(InferShapeContext *context) {
  CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("MatMul", "context is null"), return ge::GRAPH_FAILED);
  return InferShapeForMatMul(context, false);
}

static ge::graphStatus InferShapeForMatMulV2(InferShapeContext *context) {
  CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("MatMulV2", "context is null"), return ge::GRAPH_FAILED);
  return InferShapeForMatMul(context, true);
}

class InferShapeBatchMatMul {
 public:
  InferShapeBatchMatMul(InferShapeContext *context, const Shape &input_shape_a, const Shape &input_shape_b,
                        bool input_trans_a, bool input_trans_b, size_t batch_matmul_bias_index = kBatchMatMulBiasIdx)
      : op_name(context->GetNodeName()),
        shape_a(input_shape_a),
        shape_b(input_shape_b),
        trans_a(input_trans_a),
        trans_b(input_trans_b),
        shape_out(*(context->GetOutputShape(0))),
        shape_bias(context->GetOptionalInputShape(batch_matmul_bias_index)) {
    num_dima = shape_a.GetDimNum();
    num_dimb = shape_b.GetDimNum();
    num_dim = std::max(num_dima, num_dimb);
    num_dim_bias = 0;
    if (shape_bias != nullptr) {
      num_dim_bias = shape_bias->GetDimNum();
      num_dim = std::max(num_dim, num_dim_bias);
    }
    shape_out.SetDimNum(num_dim);
  };

  InferShapeBatchMatMul(InferShapeContext *context, const Shape &input_shape_a, const Shape &input_shape_b)
      : op_name(context->GetNodeName()),
        shape_a(input_shape_a),
        shape_b(input_shape_b),
        shape_out(*(context->GetOutputShape(0))) {
    shape_bias = context->GetOptionalInputShape(kBatchMatMulFixpipeBiasIdx);
    num_dima = shape_a.GetDimNum();
    num_dimb = shape_b.GetDimNum();
    num_dim = std::max(num_dima, num_dimb);
    auto attrs = context->GetAttrs();
    trans_a = *(attrs->GetAttrPointer<bool>(0));
    trans_b = *(attrs->GetAttrPointer<bool>(1));
    num_dim_bias = 0;
    if (shape_bias != nullptr) {
      num_dim_bias = shape_bias->GetDimNum();
      num_dim = std::max(num_dim, num_dim_bias);
    }
    shape_out.SetDimNum(num_dim);
  };

  ~InferShapeBatchMatMul(){};
  bool InferShape();

 protected:
  bool InferBatch() const;
  bool InferBias();

  size_t num_dim;
  size_t num_dima;
  size_t num_dimb;
  size_t num_dim_bias;

  const char *op_name;
  const Shape &shape_a;
  const Shape &shape_b;
  bool trans_a;
  bool trans_b;
  Shape &shape_out;
  const Shape *shape_bias;
};

static void CopyOutShapeFromInputShape(const Shape &shape_in, Shape &shape_out, int64_t valid_offset) {
  for (auto i = 0; i < valid_offset; ++i) {
    shape_out.SetDim(i, shape_in.GetDim(i));
  }
}

bool InferShapeBatchMatMul::InferBatch() const {
  auto valid_offset = num_dim - std::min(num_dima, num_dimb);
  const Shape &shape_long = num_dima < num_dimb ? shape_b : shape_a;
  const Shape &shape_short = num_dima < num_dimb ? shape_a : shape_b;
  int64_t shape_value_long;
  int64_t shape_value_short;

  CopyOutShapeFromInputShape(shape_long, shape_out, valid_offset);
  // use index - 2 to get index of m
  for (auto i = valid_offset; i < num_dim - 2; ++i) {
    shape_value_short = shape_short.GetDim(i - valid_offset);
    shape_value_long = shape_long.GetDim(i);
    if (shape_value_short > 1 && shape_value_long > 1 && shape_value_short != shape_value_long) {
      return false;
    }
    shape_out.SetDim(i, std::max(shape_value_short, shape_value_long));
  }
  return true;
}

static bool BroadcastBatchDim(const char *op_name, const int64_t dim_a, const int64_t dim_b, int64_t &dim) {
  if (dim_a > 1 && dim_b > 1) {
    CHECK(dim_a != dim_b,
          CUBE_INNER_ERR_REPORT(op_name, "[InferShape] dimensions a(%ld) and b(%ld) must be equal", dim_a, dim_b),
          return false);

    dim = dim_a;
    return true;
  }

  dim = std::max(dim_a, dim_b);
  return true;
}

static bool InferNDimWithBias(const char *op_name, const int64_t dim_a, const int64_t dim_b, int64_t &dim) {
  // shape_bias_n > 0 && n > 0
  if (dim_a > 0 && dim_b > 0) {
    CHECK(dim_a != dim_b,
          CUBE_INNER_ERR_REPORT(op_name, "[InferShape] dimensions a(%ld) and b(%ld) must be equal", dim_a, dim_b),
          return false);
    dim = dim_a;
    return true;
  }

  return false;
}

bool InferShapeBatchMatMul::InferBias() {
  int64_t shape_value_out = shape_out.GetDim(num_dim - 1);
  // 1) shape_bias = {}
  CHECK(num_dim_bias == 0, CUBE_INNER_ERR_REPORT(op_name, "[InferShape] bias dims number is zero"), return true);
  CHECK(shape_bias->GetShapeSize() == 0, OP_LOGI(op_name, "[InferShape] bias shape size is zero"), return true);

  // 2) infer n with bias
  CHECK(
      !InferNDimWithBias(op_name, shape_bias->GetDim(num_dim_bias - 1), shape_out.GetDim(num_dim - 1), shape_value_out),
      CUBE_INNER_ERR_REPORT(op_name, "[InferShape] failed to infer N dim with bias"), return false);

  shape_out.SetDim(num_dim - 1, shape_value_out);

  // 3) infer batch with bias
  auto valid_offset = num_dim - std::min(num_dim_bias, std::max(num_dima, num_dimb));
  if (num_dim_bias < num_dim) {
    // stop before num_dim - 2 so as to avoid traversing axis m, n
    for (auto i = valid_offset; i < num_dim - 2; ++i) {
      CHECK(!BroadcastBatchDim(op_name, shape_bias->GetDim(i - valid_offset), shape_out.GetDim(i), shape_value_out),
            CUBE_INNER_ERR_REPORT(op_name, "[InferShape] failed to broadcast batch dim"), return false);

      shape_out.SetDim(i, shape_value_out);
    }
    return true;
  }
  CopyOutShapeFromInputShape(*shape_bias, shape_out, valid_offset);
  // stop before num_dim - 2 so as to avoid traversing axis m, n
  for (auto i = valid_offset; i < num_dim - 2; ++i) {
    CHECK(!BroadcastBatchDim(op_name, shape_bias->GetDim(i), shape_out.GetDim(i - valid_offset), shape_value_out),
          CUBE_INNER_ERR_REPORT(op_name, "[InferShape] failed to broadcast batch dim"), return false);

    shape_out.SetDim(i, shape_value_out);
  }
  return true;
}

bool InferShapeBatchMatMul::InferShape() {
  if (shape_a.GetDimNum() < kBatchMatmulMinShapeSize || shape_b.GetDimNum() < kBatchMatmulMinShapeSize) {
    CHECK(!InferBatch(), CUBE_INNER_ERR_REPORT(op_name, "[InferShape] Failed to x1/x2 dim num less than 2."),
          return false);
    return false;
  }
  // using index - 2 to get m_dim
  size_t idx_m = num_dima - 2;
  size_t idx_k_a = num_dima - 1;
  // using index - 2 to get k_dim
  size_t idx_k_b = num_dimb - 2;
  size_t idx_n = num_dimb - 1;
  if (trans_a) {
    idx_m = num_dima - 1;
    // using index - 2 to get k_dim
    idx_k_a = num_dima - 2;
  }
  if (trans_b) {
    idx_k_b = num_dimb - 1;
    // using index - 2 to get n_dim
    idx_n = num_dimb - 2;
  }

  if (shape_a.GetDim(idx_k_a) != shape_b.GetDim(idx_k_b)) {
    CUBE_INNER_ERR_REPORT(op_name, "[InferShape] The k-axis of a(%ld) and b(%ld) tensors must be the same",
                          shape_a.GetDim(idx_k_a), shape_b.GetDim(idx_k_b));
    return false;
  }
  CHECK(!InferBatch(), CUBE_INNER_ERR_REPORT(op_name, "[InferShape] Failed to infer Batch."), return false);

  // using index - 2 to get m_dim in shape_out
  shape_out.SetDim((num_dim - 2), shape_a.GetDim(idx_m));
  shape_out.SetDim((num_dim - 1), shape_b.GetDim(idx_n));
  if (shape_bias != nullptr) {
    CHECK(!InferBias(), CUBE_INNER_ERR_REPORT(op_name, "[InferShape] Infer bias failed."), return false);
  }

  return true;
}

static ge::graphStatus InferShapeForBatchMatMulV2(InferShapeContext *context, string& op_type) {
  auto shape_x1 = context->GetInputShape(0);
  auto shape_x2 = context->GetInputShape(1);
  auto shape_out = context->GetOutputShape(0);
  auto attrs = context->GetAttrs();
  auto op_name = context->GetNodeName();
  size_t batch_matmul_bias_index = kBatchMatMulBiasIdx;
  CHECK(shape_x1 == nullptr || shape_x2 == nullptr || shape_out == nullptr || attrs == nullptr,
        CUBE_INNER_ERR_REPORT(op_name, "[Infershape]shape is null"), return ge::GRAPH_FAILED);
  int attr_adj_idx = 0;
  if (op_type == "QuantBatchMatmulV3") {
    attr_adj_idx++;
  }
  const bool *adj_x1 = attrs->GetAttrPointer<bool>(attr_adj_idx);
  const bool *adj_x2 = attrs->GetAttrPointer<bool>(++attr_adj_idx);

  CHECK(adj_x1 == nullptr || adj_x2 == nullptr, CUBE_INNER_ERR_REPORT(op_name, "[Infershape]attribute is null"),
        return ge::GRAPH_FAILED);

  auto dim_num = std::max(shape_x1->GetDimNum(), shape_x2->GetDimNum());
  if (dim_num < 1 || dim_num > kBatchMatmulMaxShapeSize) {
    CUBE_INNER_ERR_REPORT(op_name, "[Infershape]The shape can only be in the range of 1 to 8.");
  }

  Shape shape_x2_new(*shape_x2);
  bool shape_x2_reshape_flag = false;
  if (shape_x2_new.GetDimNum() == 1 && shape_x2_new.GetDim(0) > 0) {
    shape_x2_reshape_flag = true;
    int64_t ori_dim = shape_x2_new.GetDim(0);
    shape_x2_new.SetDimNum(kBatchMatmulMinShapeSize);
    shape_x2_new.SetDim(0, ori_dim);
    shape_x2_new.SetDim(1, 1);
  }

  if (OptypeBiasIndex.find(op_type) != OptypeBiasIndex.end()) {
    batch_matmul_bias_index = OptypeBiasIndex.find(op_type)->second;
  }

  if (op_type == "MatMulV2CompressDequant") {
    shape_x2_reshape_flag = false;
    auto compress_info = attrs->GetListInt(2);
    int64_t shape_ori_k = compress_info->GetData()[2];
    int64_t shape_ori_n = compress_info->GetData()[3];
    shape_x2_new.SetDim(0, shape_ori_k);
    shape_x2_new.SetDim(1, shape_ori_n);
  }

  Shape shape_x1_new(*shape_x1);
  bool shape_x1_reshape_flag = false;
  if (shape_x1_new.GetDimNum() == 1 && shape_x1_new.GetDim(0) > 0) {
    shape_x1_reshape_flag = true;
    int64_t ori_dim = shape_x1_new.GetDim(0);
    shape_x1_new.SetDimNum(kBatchMatmulMinShapeSize);
    shape_x1_new.SetDim(0, 1);
    shape_x1_new.SetDim(1, ori_dim);
  }

  if (op_type == "BatchMatmulFixpipe") {
    InferShapeBatchMatMul BatchMatMulV2Infer(context, shape_x1_new, shape_x2_new);
    CHECK(!BatchMatMulV2Infer.InferShape(), CUBE_INNER_ERR_REPORT(op_name, "[InferShape] Failed to infer output shape"),
          return ge::GRAPH_FAILED);
  } else {
    InferShapeBatchMatMul BatchMatMulV2Infer(context, shape_x1_new, shape_x2_new, *adj_x1, *adj_x2, batch_matmul_bias_index);
    CHECK(!BatchMatMulV2Infer.InferShape(), CUBE_INNER_ERR_REPORT(op_name, "[InferShape] Failed to infer output shape"),
          return ge::GRAPH_FAILED);
  }

  InferComplementedOutput(shape_x1_reshape_flag, shape_x2_reshape_flag, *shape_out);
  // no need to SetDataType in runtime
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForBatchMatMulV2(InferShapeContext *context) {
  string op_type = "BatchMatMulV2";
  ge::graphStatus res = InferShapeForBatchMatMulV2(context, op_type);
  return res;
}


static ge::graphStatus ChechScaleForTransposeBatchMatMul(InferShapeContext *context, const Shape *shape_x2) {
  const Shape *shape_scale = context->GetOptionalInputShape(kTransposeBatchMatMulScaleIdx);
  if (shape_scale != nullptr && shape_scale->GetDimNum() == 1) {
    int64_t scale_dim = shape_scale->GetDimNum();
    CHECK(shape_scale->GetDim(scale_dim - 1) != shape_x2->GetDim(0) * shape_x2->GetDim(2),  // scale = batch * n
          OP_LOGE(context->GetNodeName(), "The dimension of n * b [%ld] and scale [%ld] tensors must be the same", shape_x2->GetDim(0) * shape_x2->GetDim(2),
                  shape_scale->GetDim(scale_dim - 1)),
          return ge::GRAPH_FAILED);
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForTransposeBatchMatMul(InferShapeContext *context) {
  CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("TransposeBatchMatMul", "context is null"), return ge::GRAPH_FAILED);

  auto shape_x1 = context->GetInputShape(0);
  auto shape_x2 = context->GetInputShape(1);
  auto shape_y = context->GetOutputShape(0);
  auto attrs = context->GetAttrs();
  auto name_op = context->GetNodeName();
  CHECK(shape_x1 == nullptr || shape_x2 == nullptr || shape_y == nullptr || attrs == nullptr,
        CUBE_INNER_ERR_REPORT(name_op, "[Infershape]shape or attrs is null"), return ge::GRAPH_FAILED);

  CHECK(shape_x1->GetDimNum() == 3 && shape_x2->GetDimNum() == 3 && shape_y->GetDimNum() == 3 ,
        CUBE_INNER_ERR_REPORT(name_op, "[Infershape] x1, x2, and y's dim should be 3"), return ge::GRAPH_FAILED);
  const auto perm_x1 = attrs->GetListInt(0);
  const auto perm_x2 = attrs->GetListInt(1);
  const auto perm_y = attrs->GetListInt(2);
  CHECK(perm_x1 == nullptr || perm_x2 == nullptr || perm_y == nullptr,
        CUBE_INNER_ERR_REPORT(name_op, "[Infershape] null"), return ge::GRAPH_FAILED);

  OP_LOGD(name_op, "x1_shape: %s, x2_shape: %s, perm_x1: %s, perm_x2: %s, perm_y: %s", ops::ToString(shape_x1).c_str(),
          ops::ToString(shape_x2).c_str(), ops::ToString(perm_x1).c_str(), ops::ToString(perm_x2).c_str(),
          ops::ToString(perm_y).c_str());

  Shape shape_x1_transposed;
  Shape shape_x2_transposed;
  CHECK(!TransposeShape(*shape_x1, *perm_x1, shape_x1_transposed),
        CUBE_INNER_ERR_REPORT(name_op, "[InferShape] Failed to transpose shape of x1"), return ge::GRAPH_FAILED);
  CHECK(!TransposeShape(*shape_x2, *perm_x2, shape_x2_transposed),
        CUBE_INNER_ERR_REPORT(name_op, "[InferShape] Failed to transpose shape of x2"), return ge::GRAPH_FAILED);

  const auto perm_x1_data = perm_x1->GetData();
  CHECK(perm_x1_data[0] == 1 && perm_x1_data[1] == 0 && perm_x1_data[2] == 2,
        CUBE_INNER_ERR_REPORT(name_op, "[InferShape] perm_x1 should {1, 0, 2}"), return ge::GRAPH_FAILED);
        
  const auto perm_y_data = perm_y->GetData();
  CHECK(perm_y_data[0] == 1 && perm_y_data[1] == 0 && perm_y_data[2] == 2,
        CUBE_INNER_ERR_REPORT(name_op, "[InferShape] perm_y should {1, 0, 2}"), return ge::GRAPH_FAILED);

  CHECK(shape_x1_transposed.GetDim(0) != shape_x2_transposed.GetDim(0),
        CUBE_INNER_ERR_REPORT(name_op, "[InferShape] batch must be equal, transposed shape of x1 and x2 is %s, %s.",
                              ops::ToString(shape_x1_transposed).c_str(), ops::ToString(shape_x2_transposed).c_str()),
        return ge::GRAPH_FAILED);

  Shape shape_y_transposed;
  CHECK(!TransposeShape(*shape_y, *perm_y, shape_y_transposed),
        CUBE_INNER_ERR_REPORT(name_op, "[InferShape] Failed to transpose shape of y"), return ge::GRAPH_FAILED);
  for (size_t i = 0; i < shape_y->GetDimNum(); ++i) {
    shape_y->SetDim(i, shape_y_transposed.GetDim(i));
  }
  const Shape *shape_bias = context->GetOptionalInputShape(kBatchMatMulBiasIdx);
  CHECK(shape_bias == nullptr,
        CUBE_INNER_ERR_REPORT(name_op, "[InferShape] bias is not support in TBMM."), return ge::GRAPH_FAILED);

  InferShapeBatchMatMul BatchMatMulV2Infer(context, shape_x1_transposed, shape_x2_transposed, false, false);
  CHECK(!BatchMatMulV2Infer.InferShape(), CUBE_INNER_ERR_REPORT(name_op, "[InferShape] Failed to infer output shape"),
        return ge::GRAPH_FAILED);
  CHECK(ChechScaleForTransposeBatchMatMul(context, shape_x2), CUBE_INNER_ERR_REPORT(name_op, "[InferShape] scale shape check failed"),
        return ge::GRAPH_FAILED);
  // no need to SetDataType in runtime
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForQuantBatchMatmul(InferShapeContext *context) {
  string op_type = "QuantBatchMatmul";
  CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("QuantBatchMatmul", "context is null"), return ge::GRAPH_FAILED);
  auto op_name = context->GetNodeName();
  ge::graphStatus res = InferShapeForBatchMatMulV2(context, op_type);
  CHECK(res != ge::GRAPH_SUCCESS,
        CUBE_INNER_ERR_REPORT(op_name, "[InferShape] Failed."), return ge::GRAPH_FAILED);
  size_t deq_idx = 2;
  if (OptypeDeqScaleIndex.find(op_type) != OptypeDeqScaleIndex.end()) {
    deq_idx = OptypeDeqScaleIndex.find(op_type)->second;
  }
  auto shape_deq = context->GetInputShape(deq_idx);
  CHECK(shape_deq == nullptr,
        CUBE_INNER_ERR_REPORT(op_name, "[Infershape] input deq_scale can not be null"), return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForQuantBatchMatmul(gert::InferDataTypeContext *context) {
  CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("QuantBatchMatmul", "context is null"), return ge::GRAPH_FAILED);
  const ge::DataType output_data_type = ge::DT_FLOAT16;
  ge::graphStatus ret = InferDataTypeFixedOutputType(output_data_type, context);
  return ret;
}

static ge::graphStatus InferShapeForWeightQuantBatchMatmul(InferShapeContext *context) {
  string op_type = "WeightQuantBatchmatmul";
  CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("WeightQuantBatchMatmul", "context is null"),
        return ge::GRAPH_FAILED);
  auto op_name = context->GetNodeName();
  ge::graphStatus res = InferShapeForBatchMatMulV2(context, op_type);
  CHECK(res != ge::GRAPH_SUCCESS,
        CUBE_INNER_ERR_REPORT(op_name, "[InferShape] Failed."), return ge::GRAPH_FAILED);
  size_t deq_idx = 4;
  if (OptypeDeqScaleIndex.find(op_type) != OptypeDeqScaleIndex.end()) {
    deq_idx = OptypeDeqScaleIndex.find(op_type)->second;
  }
  auto shape_deq = context->GetInputShape(deq_idx);
  CHECK(shape_deq == nullptr,
        CUBE_INNER_ERR_REPORT(op_name, "[Infershape] input deq_scale can not be null"), return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForMatMulV2CompressDequant(InferShapeContext *context) {
  string op_type = "MatMulV2CompressDequant";
  CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("MatMulV2CompressDequant", "context is null"),
        return ge::GRAPH_FAILED);
  auto op_name = context->GetNodeName();
  ge::graphStatus res = InferShapeForBatchMatMulV2(context, op_type);
  CHECK(res != ge::GRAPH_SUCCESS,
        CUBE_INNER_ERR_REPORT(op_name, "[InferShape] Failed."), return ge::GRAPH_FAILED);
  auto compress_index = context->GetInputShape(2);
  size_t deq_idx = 3;
  if (OptypeDeqScaleIndex.find(op_type) != OptypeDeqScaleIndex.end()) {
    deq_idx = OptypeDeqScaleIndex.find(op_type)->second;
  }
  auto shape_deq = context->GetInputShape(deq_idx);
  CHECK(compress_index == nullptr,
        CUBE_INNER_ERR_REPORT(op_name, "input compress_index can not be null is null"), return ge::GRAPH_FAILED);
  CHECK(shape_deq == nullptr,
        CUBE_INNER_ERR_REPORT(op_name, "input shape_deq can not be null is null"), return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMatMulV2CompressDequant(gert::InferDataTypeContext *context) {
  CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("MatMulV2CompressDequant", "context is null"), return ge::GRAPH_FAILED);
  const ge::DataType output_data_type = ge::DT_FLOAT16;
  ge::graphStatus ret = InferDataTypeFixedOutputType(output_data_type, context);
  return ret;
}

static ge::graphStatus InferShapeForBatchMatmulFixpipe(InferShapeContext *context) {
  string op_type = "BatchMatmulFixpipe";
  CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("BatchMatmulFixpipe", "context is null"), return ge::GRAPH_FAILED);
  auto op_name = context->GetNodeName();
  ge::graphStatus res = InferShapeForBatchMatMulV2(context, op_type);
  CHECK(res != ge::GRAPH_SUCCESS,
        CUBE_INNER_ERR_REPORT(op_name, "[InferShape] Failed."), return ge::GRAPH_FAILED);
  size_t deq_idx = 2;
  if (OptypeDeqScaleIndex.find(op_type) != OptypeDeqScaleIndex.end()) {
    deq_idx = OptypeDeqScaleIndex.find(op_type)->second;
  }
  auto shape_quant_pre = context->GetInputShape(deq_idx);
  CHECK(shape_quant_pre == nullptr,
        CUBE_INNER_ERR_REPORT(op_name, "[Infershape] input shape_quant_pre can not be null"), return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForBatchMatmulFixpipe(gert::InferDataTypeContext *context) {
    CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("BatchMatmulFixpipe", "context is null"), return ge::GRAPH_FAILED);
    const ge::DataType output_data_type = ge::DT_INT8;
    ge::graphStatus ret = InferDataTypeFixedOutputType(output_data_type, context);
    return ret;
}

static ge::graphStatus InferShapeForWeightQuantBatchmatmulV3(InferShapeContext *context) {
  string op_type = "WeightQuantBatchmatmulV3";
  CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("WeightQuantBatchmatmulV3",
                                                  "context is null"), return ge::GRAPH_FAILED);
  auto op_name = context->GetNodeName();
  ge::graphStatus res = InferShapeForBatchMatMulV2(context, op_type);
  CHECK(res != ge::GRAPH_SUCCESS,
        CUBE_INNER_ERR_REPORT(op_name, "[InferShape] Failed."), return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForQuantBatchMatmulV3(InferShapeContext *context) {
  CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("QuantBatchMatmulV3",
                                                  "context is null"), return ge::GRAPH_FAILED);
  string op_type = context->GetNodeType();
  auto op_name = context->GetNodeName();
  ge::graphStatus res = InferShapeForBatchMatMulV2(context, op_type);
  CHECK(res != ge::GRAPH_SUCCESS,
        CUBE_INNER_ERR_REPORT(op_name, "[InferShape] Failed."), return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeQuantBatchMatmulV3(gert::InferDataTypeContext *context) {
  const int64_t *dtype_num = context->GetAttrs()->GetAttrPointer<int64_t>(0);
  ge::DataType out_dtype = static_cast<ge::DataType> (*dtype_num);
  context->SetOutputDataType(0, out_dtype);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForGemmV2(InferShapeContext *context)
{
  auto op_name = context->GetNodeName();
  auto shape_a = context->GetInputShape(0);
  auto shape_b = context->GetInputShape(1);
  auto shape_c = context->GetInputShape(4);
  auto shape_out = context->GetOutputShape(0);
  auto attrs = context->GetAttrs();
  CHECK(shape_a == nullptr || shape_b == nullptr || shape_c == nullptr || shape_out == nullptr || attrs == nullptr,
        CUBE_INNER_ERR_REPORT(op_name, "shape or attrs is null"), return ge::GRAPH_FAILED);

  const bool *trans_a = attrs->GetAttrPointer<bool>(0);
  const bool *trans_b = attrs->GetAttrPointer<bool>(1);
  CHECK(trans_a == nullptr || trans_b == nullptr, CUBE_INNER_ERR_REPORT(op_name, "attribute is null"),
        return ge::GRAPH_FAILED);

  OP_LOGD(op_name, "check the input shape length.");
  CHECK((shape_a->GetDimNum() != kMatmulV2MinShapeSize || shape_b->GetDimNum() != kMatmulV2MinShapeSize ||
      shape_c->GetDimNum() != kMatmulV2MinShapeSize),
      CUBE_INNER_ERR_REPORT(op_name, "input dim num[%zu] [%zu] [%zu]is not 2!",
                            shape_a->GetDimNum(), shape_b->GetDimNum(), shape_c->GetDimNum()), return ge::GRAPH_FAILED);

  size_t idx_m = *trans_a ? 1 : 0;
  size_t idx_k_a = *trans_a ? 0 : 1;
  size_t idx_k_b = *trans_b ? 1 : 0;
  size_t idx_n = *trans_b ? 0 : 1;

  CHECK(shape_a->GetDim(idx_k_a) != shape_b->GetDim(idx_k_b),
      CUBE_INNER_ERR_REPORT(op_name, "The k-axis of a(%ld) and b(%ld) tensors must be the same",
                            shape_a->GetDim(idx_k_a), shape_b->GetDim(idx_k_b)),
      return ge::GRAPH_FAILED);

  CHECK(shape_a->GetDim(idx_m) != shape_c->GetDim(0) || shape_b->GetDim(idx_n) != shape_c->GetDim(1),
      CUBE_INNER_ERR_REPORT(op_name, "The m(%ld), n(%ld) tensors must be the same c(%ld, %ld)",
                            shape_a->GetDim(idx_m), shape_b->GetDim(idx_n), shape_c->GetDim(0), shape_c->GetDim(1)),
      return ge::GRAPH_FAILED);

  shape_out->SetDimNum(kMatmulV2MinShapeSize);
  shape_out->SetDim(0, shape_a->GetDim(idx_m));
  shape_out->SetDim(1, shape_b->GetDim(idx_n));

  OP_LOGI(op_name, "end infershape.");
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForGemmV2(gert::InferDataTypeContext *context)
{
  auto op_name = context->GetNodeName();
  const ge::DataType a_data_type = context->GetInputDataType(0);
  const ge::DataType b_data_type = context->GetInputDataType(1);
  const ge::DataType c_data_type = context->GetInputDataType(4);
  CHECK(((a_data_type != ge::DT_FLOAT16 && a_data_type != ge::DT_BF16) ||
        (b_data_type != ge::DT_FLOAT16 && b_data_type != ge::DT_BF16) ||
        c_data_type != ge::DT_FLOAT),
        CUBE_INNER_ERR_REPORT(op_name, "input dtype not support"),
        return ge::GRAPH_FAILED);
  ge::graphStatus ret = context->SetOutputDataType(0, ge::DT_FLOAT);
  return ret;
}

static ge::graphStatus InferDataTypeForTransposeBatchMatMul(gert::InferDataTypeContext *context)
{
  auto op_name = context->GetNodeName();
  const ge::DataType a_data_type = context->GetInputDataType(0);
  const ge::DataType b_data_type = context->GetInputDataType(1);
  CHECK(((a_data_type != ge::DT_FLOAT16 && a_data_type != ge::DT_BF16) ||
        (b_data_type != ge::DT_FLOAT16 && b_data_type != ge::DT_BF16)),
        CUBE_INNER_ERR_REPORT(op_name, "input dtype not support"),
        return ge::GRAPH_FAILED);
        
  ge::graphStatus ret;
  if (context->GetOptionalInputDataType(kTransposeBatchMatMulScaleIdx) != ge::DT_UNDEFINED) {
    ret = context->SetOutputDataType(0, ge::DT_INT8);
  } else {
    ret = context->SetOutputDataType(0, a_data_type);
  }
  return ret;
}

// HostCPU operator. input and output shapes of the original image has already set.
// GE has added the infershape verification function, so additional infer-related functions are needed.
static ge::graphStatus SparseInferShapeForCompressFcOp(InferShapeContext *context) {
  auto op_name = context->GetNodeName();
  OP_LOGD(op_name, "start sparse infershape.");
  auto weightShape = context->GetInputShape(0);
  CHECK(weightShape == nullptr, OP_LOGE(op_name, "weightShape is null"), return ge::GRAPH_FAILED);
  CHECK(weightShape->GetDimNum() < 4,
        OP_LOGE(op_name, "weightShape->GetDimNum() should be less than 4, but got %zu", weightShape->GetDimNum()),
        return ge::GRAPH_FAILED);
  int k1 = (*weightShape)[kFraczK1Index];
  int64_t k1Pad = ((k1 + 1) / 2) * 2;
  auto shape_out_0 = context->GetOutputShape(0);
  auto shape_out_1 = context->GetOutputShape(1);
  CHECK(shape_out_0 == nullptr || shape_out_1 == nullptr, OP_LOGE(op_name, "outputshape is null"),
        return ge::GRAPH_FAILED);
  shape_out_0->SetDimNum(kCompressFcOpMaxShapeSize);
  constexpr int FIRST_OUTPUT_DIM = 0;
  constexpr int SECOND_OUTPUT_DIM = 1;
  constexpr int THIRD_OUTPUT_DIM = 2;
  constexpr int FOURTH_OUTPUT_DIM = 3;
  constexpr int FOUR_DENSE_ELEMENT = 4; // 4 means 8 bit for 4 dense int8 elements

  shape_out_0->SetDim(FIRST_OUTPUT_DIM, int64_t(k1Pad / kK1CopressRatio));
  shape_out_0->SetDim(SECOND_OUTPUT_DIM, int64_t((*weightShape)[kFraczN1Index]));
  shape_out_0->SetDim(THIRD_OUTPUT_DIM, int64_t((*weightShape)[kFraczN0Index]));
  shape_out_0->SetDim(FOURTH_OUTPUT_DIM, int64_t((*weightShape)[kFraczK0Index]));

  shape_out_1->SetDimNum(kCompressFcOpMaxShapeSize);
  shape_out_1->SetDim(FIRST_OUTPUT_DIM, int64_t(k1Pad / kK1CopressRatio));
  shape_out_1->SetDim(SECOND_OUTPUT_DIM, int64_t((*weightShape)[kFraczN1Index]));
  shape_out_1->SetDim(THIRD_OUTPUT_DIM, int64_t((*weightShape)[kFraczN0Index]));
  shape_out_1->SetDim(FOURTH_OUTPUT_DIM, int64_t((*weightShape)[kFraczK0Index] / FOUR_DENSE_ELEMENT));
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus UnzipInferShapeForCompressFcOp(InferShapeContext *context) {
  auto op_name = context->GetNodeName();
  auto weightShape = context->GetInputShape(0);
  CHECK(weightShape == nullptr, OP_LOGE(op_name, "weightShape is null"), return ge::GRAPH_FAILED);
  auto compress_params = context->GetAttrs()->GetListInt(0);
  CHECK(compress_params == nullptr, OP_LOGE(op_name, "compress_params is null"), return ge::GRAPH_FAILED);
  int64_t compress_info_shape = compress_params->GetData()[1];
  size_t weightSizeTotal = weightShape->GetShapeSize();
  OP_LOGD(op_name, "start Unzip infershape.");
  auto shape_out_0 = context->GetOutputShape(0);
  auto shape_out_1 = context->GetOutputShape(1);
  auto shape_out_2 = context->GetOutputShape(2);
  CHECK(shape_out_0 == nullptr || shape_out_1 == nullptr || shape_out_2 == nullptr,
        OP_LOGE(op_name, "unzip outputshape is null"), return ge::GRAPH_FAILED);

  shape_out_0->SetDimNum(kCompressFcOpShapeSize);
  shape_out_0->SetDim(0, weightSizeTotal);
  shape_out_1->SetDimNum(kCompressFcOpShapeSize);
  shape_out_1->SetDim(0, compress_info_shape);
  shape_out_2->SetDimNum(kCompressFcOpShapeSize);
  shape_out_2->SetDim(0, KCompressInfoValue);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForCompressFcOp(InferShapeContext *context) {
  OP_LOGD("CompressFcOp", "start infershape.");
  CHECK(context == nullptr, OP_LOGE("CompressFcOp", "CompressFcOp context is null"), return ge::GRAPH_FAILED);
  auto op_name = context->GetNodeName();
  auto attrs = context->GetAttrs();
  const char *compress_flag = "weight_unzip";
  constexpr int ATTR_FLAG_NUM = 2;
  if (attrs->GetAttrNum() >= ATTR_FLAG_NUM) {
    compress_flag = attrs->GetAttrPointer<char>(1);
  }
  if (strcmp(compress_flag, "weight_sparse_4_2") == 0) {
    ge::graphStatus sparse_res = SparseInferShapeForCompressFcOp(context);
    CHECK(sparse_res != ge::GRAPH_SUCCESS, OP_LOGE(op_name, "[InferShape] Failed."), return ge::GRAPH_FAILED);
  } else {
    ge::graphStatus unzip_res = UnzipInferShapeForCompressFcOp(context);
    CHECK(unzip_res != ge::GRAPH_SUCCESS, OP_LOGE(op_name, "[InferShape] Failed."), return ge::GRAPH_FAILED);
  }
  OP_LOGD(op_name, "end infershape.");
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForCompressFcOp(gert::InferDataTypeContext *context) {
    CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("CompressFcOp", "context is null"), return ge::GRAPH_FAILED);
    auto op_name = context->GetNodeName();
    const ge::DataType output_weight_compress = ge::DT_INT8;
    const ge::DataType output_compress_index = ge::DT_INT8;
    const ge::DataType output_compress_info = ge::DT_UINT32;
    ge::graphStatus a_ret = context->SetOutputDataType(0, output_weight_compress);
    ge::graphStatus b_ret = context->SetOutputDataType(1, output_compress_index);
    ge::graphStatus c_ret = context->SetOutputDataType(2, output_compress_info);
    CHECK((a_ret != ge::GRAPH_SUCCESS || b_ret != ge::GRAPH_SUCCESS || c_ret != ge::GRAPH_SUCCESS),
        CUBE_INNER_ERR_REPORT(op_name, "[InferDataType] Failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BatchMatMul)
    .InferShape(InferShapeForBatchMatMulV2)
    .PrivateAttr("_op_impl_mode_enum", (int64_t)-1)
    .PrivateAttr("enable_pad", (int64_t)0);

IMPL_OP_INFERSHAPE(BatchMatMulV2)
    .InferShape(InferShapeForBatchMatMulV2)
    .PrivateAttr("_op_impl_mode_enum", (int64_t)-1)
    .PrivateAttr("enable_pad", (int64_t)0);

IMPL_OP_INFERSHAPE(MatMul)
    .InferShape(InferShapeForMatMul)
    .PrivateAttr("_op_impl_mode_enum", (int64_t)-1)
    .PrivateAttr("enable_pad", (int64_t)0);

IMPL_OP_INFERSHAPE(MatMulV2)
    .InferShape(InferShapeForMatMulV2)
    .PrivateAttr("input_size", -1L)
    .PrivateAttr("hidden_size", -1L)
    .PrivateAttr("_op_impl_mode_enum", (int64_t)-1)
    .PrivateAttr("enable_pad", (int64_t)0);

IMPL_OP_INFERSHAPE(QuantBatchMatmul)
    .InferShape(InferShapeForQuantBatchMatmul)
    .InferDataType(InferDataTypeForQuantBatchMatmul)
    .PrivateAttr("_op_impl_mode_enum", (int64_t)-1)
    .PrivateAttr("enable_pad", (int64_t)0);

IMPL_OP_INFERSHAPE(WeightQuantBatchmatmul)
    .InferShape(InferShapeForWeightQuantBatchMatmul)
    .InferOutDataTypeSameWithFirstInput()
    .InputsDataDependency({4})
    .PrivateAttr("_op_impl_mode_enum", (int64_t)-1)
    .PrivateAttr("enable_pad", (int64_t)0);

IMPL_OP_INFERSHAPE(MatMulV2CompressDequant)
    .InferShape(InferShapeForMatMulV2CompressDequant)
    .InferDataType(InferDataTypeForMatMulV2CompressDequant)
    .PrivateAttr("_op_impl_mode_enum", (int64_t)-1)
    .PrivateAttr("enable_pad", (int64_t)0);

IMPL_OP_INFERSHAPE(BatchMatmulFixpipe)
    .InferShape(InferShapeForBatchMatmulFixpipe)
    .InferDataType(InferDataTypeForBatchMatmulFixpipe)
    .InputsDataDependency({2})
    .PrivateAttr("_op_impl_mode_enum", (int64_t)-1);

IMPL_OP_INFERSHAPE(WeightQuantBatchmatmulV3)
    .InferShape(InferShapeForWeightQuantBatchmatmulV3)
    .InferOutDataTypeSameWithFirstInput()
    .InputsDataDependency({7})
    .PrivateAttr("_op_impl_mode_enum", (int64_t)-1);

IMPL_OP_INFERSHAPE(MatMulV3)
    .InferShape(InferShapeForMatMulV2);

IMPL_OP_INFERSHAPE(BatchMatMulV3)
    .InferShape(InferShapeForBatchMatMulV2);

IMPL_OP_INFERSHAPE(QuantBatchMatmulV3)
    .InferShape(InferShapeForQuantBatchMatmulV3)
    .InferDataType(InferDataTypeQuantBatchMatmulV3);

IMPL_OP_INFERSHAPE(GemmV2)
    .InferShape(InferShapeForGemmV2)
    .InferDataType(InferDataTypeForGemmV2);

IMPL_OP_INFERSHAPE(CompressFcOp)
    .InferShape(InferShapeForCompressFcOp)
    .InferDataType(InferDataTypeForCompressFcOp)
    .PrivateAttr("alg", "weight_unzip");

IMPL_OP_INFERSHAPE(TransposeBatchMatMul)
    .InferShape(InferShapeForTransposeBatchMatMul)
    .InferDataType(InferDataTypeForTransposeBatchMatMul);
}  // namespace gert

