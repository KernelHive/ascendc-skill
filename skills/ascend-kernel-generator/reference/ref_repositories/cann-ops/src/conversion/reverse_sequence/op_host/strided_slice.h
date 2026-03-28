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
 * \file strided_slice.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_RUNTIME_STRIDED_SLICE_H_
#define OPS_BUILT_IN_OP_PROTO_RUNTIME_STRIDED_SLICE_H_

#include <string>
#include "op_common_util.h"

namespace ops {
static const int32_t kStridedSliceNewAxis = -2;
static const std::string OP_NAME = "StridedSlice";
using QuickVector = gert::Shape;

struct StridedSliceParams {
  gert::Shape input_shape;
  QuickVector begin;
  QuickVector end;
  QuickVector strides;
  uint64_t begin_mask;
  uint64_t end_mask;
  uint64_t ellipsis_mask;
  uint64_t new_axis_mask;
  uint64_t shrink_axis_mask;
  bool begin_valid;
  bool end_valid;
  bool stride_valid;
};

struct ProcessingData {
  gert::Shape processing_shape;
  QuickVector processing_begin;
  QuickVector processing_end;
  QuickVector processing_strides;
};

struct InputParamUnit {
  int64_t begin;
  int64_t end;
  int64_t stride;
  int64_t dim;
  bool shrink;
};

struct StridedSliceSparseSpec {
  int64_t dims;
  int32_t num_add_axis_after_ellipsis;
  const QuickVector begin;
  const QuickVector end;
  const QuickVector strides;
  const uint64_t begin_mask;
  const uint64_t end_mask;
  uint64_t ellipsis_mask;
  const uint64_t new_axis_mask;
  const uint64_t shrink_axis_mask;
};

struct StridedSliceDenseSpec {
  const int64_t dims;
  uint64_t begin_mask;
  uint64_t end_mask;
  bool begin_valid;
  bool end_valid;
  QuickVector begin;
  QuickVector end;
  QuickVector strides;

  // This vector helps construct the final shape of the slice.
  // The final tensor is reduced in rank whenever a single index e.g. foo[3]
  // is called for. The final tensor increases in rank with tf.newaxis
  // entries. If an index in this array is positive, the size of the dimension
  // is obtained from canonical end-begin. Otherwise, if it is a kNewAxis,
  // it will be 1. A shrunk dimension is skipped.
  gert::Shape final_shape_gather_indices;

  // The dense indexed shrink mask is which processing dimensions
  // should be shrunk. For example, if foo.shape = (10,10,10,10)
  // foo[3, ..., 5] has sparse_shrink_axis_mask of 0x5 and
  // dense_shrink_axis_mask of 0x9, yielding a final shape (10,10).
  uint64_t shrink_axis_mask;
};

static inline uint64_t bit1value(int i) {
  const uint64_t bit_i = static_cast<uint64_t>(1) << static_cast<uint64_t>(i);
  return bit_i;
}

static bool FwdOutOfBound(int64_t fwd, int64_t lower, int64_t upper) {
  return (fwd < lower) || (fwd >= upper);
}

static void BuildSparseSpec(StridedSliceParams& params, StridedSliceSparseSpec& sparse_spec) {
  sparse_spec.dims = static_cast<int64_t>(params.strides.GetDimNum());
  bool ellipsis_seen = false;
  for (int32_t i = 0; i < sparse_spec.dims; i++) {
    const uint64_t bit_i = bit1value(i);
    if (ellipsis_seen && (bit_i & params.new_axis_mask) != 0) {
      sparse_spec.num_add_axis_after_ellipsis++;
    }
    if ((bit_i & params.ellipsis_mask) != 0) {
      ellipsis_seen = true;
    }
  }
  // If no ellipsis insert one at the end
  if (!ellipsis_seen) {
    sparse_spec.ellipsis_mask |= bit1value(sparse_spec.dims);
    sparse_spec.dims++;  // this effects loop iteration below
  }
}

static bool BuildDenseSpec(const StridedSliceSparseSpec& sparse, StridedSliceDenseSpec* dense) {
  constexpr int32_t kShrinkAxis = -1;
  // Build expanded begin, end, strides, begin_mask, end_mask
  // to remove any ellipsis
  dense->begin.SetDimNum(dense->dims);
  dense->end.SetDimNum(dense->dims);
  dense->strides.SetDimNum(dense->dims);

  // What indices to get the final shape from.
  dense->begin_mask = 0;
  dense->end_mask = 0;
  dense->shrink_axis_mask = 0;

  int full_index = 0;
  for (int i = 0; i < sparse.dims; i++) {
    const uint64_t bit_i = bit1value(i);
    if ((bit_i & sparse.ellipsis_mask) != 0) {
      // Expand the ellipsis into the appropriate indices
      // NOTE: this only works because we guaranteed one ellipsis
      int32_t next_index =
          std::min(dense->dims - (sparse.dims - i) + 1 + sparse.num_add_axis_after_ellipsis, dense->dims);
      for (; full_index < next_index; full_index++) {
        // new_axis' aren't real axis so you have to skip
        dense->begin[full_index] = dense->end[full_index] = 0;
        dense->strides[full_index] = 1;
        dense->begin_mask |= bit1value(full_index);
        dense->end_mask |= bit1value(full_index);
        dense->final_shape_gather_indices.AppendDim(full_index);
      }
    } else if ((bit_i & sparse.new_axis_mask) != 0) {
      dense->final_shape_gather_indices.AppendDim(kStridedSliceNewAxis);
    } else {
      if (static_cast<size_t>(full_index) == dense->begin.GetDimNum()) {
        return false;
      }

      // Gather slicing spec into appropriate index
      dense->begin[full_index] = sparse.begin[i];
      dense->end[full_index] = sparse.end[i];
      dense->strides[full_index] = sparse.strides[i];

      if ((sparse.begin_mask & bit_i) != 0) {
        dense->begin_mask |= bit1value(full_index);
      }
      if ((sparse.end_mask & bit_i) != 0) {
        dense->end_mask |= bit1value(full_index);
      }

      // If shrink, record where to get the dimensionality from (i.e.
      // new_axis creates a fake 1 size dimension. Also remember shrink
      // axis (now in dense form) so we can ignore dense->end below.
      if ((sparse.shrink_axis_mask & bit_i) != 0) {
        dense->final_shape_gather_indices.AppendDim(kShrinkAxis);
        dense->shrink_axis_mask |= bit1value(full_index);
      } else {
        dense->final_shape_gather_indices.AppendDim(full_index);
      }
      full_index++;
    }
  }

  return true;
}

static void BuildProcessingShape(StridedSliceDenseSpec& dense_spec,
                                 InputParamUnit& input_param_unit,
                                 const bool begin_and_end_masked,
                                 gert::Shape& processing_shape) {
  int64_t interval_length;
  bool known_interval = false;
  if (dense_spec.begin_valid && dense_spec.end_valid) {
    interval_length = input_param_unit.end - input_param_unit.begin;
    known_interval = true;
  } else if (input_param_unit.shrink) {
    // The dimension is still known as 1 for the processing_shape, but will be
    // discarded for the final shape.
    interval_length = 1;
    known_interval = true;
  } else if (begin_and_end_masked) {
    // Even if we don't have values for begin or end, we do know that this
    // dimension covers the whole interval. If we have shape information for
    // this dimension, that tells us the interval length.
    if (input_param_unit.dim >= 0) {
      if (input_param_unit.stride < 0) {
        interval_length = -input_param_unit.dim;
      } else {
        interval_length = input_param_unit.dim;
      }
      known_interval = true;
    }
  }
  if (known_interval) {
    int64_t size_i;
    // Hold zero if the interval is degenerate, otherwise account for
    // remainder
    if (interval_length == 0 || ((interval_length < 0) != (input_param_unit.stride < 0))) {
      size_i = 0;
    } else {
      size_i = interval_length / input_param_unit.stride + (interval_length % input_param_unit.stride != 0 ? 1 : 0);
    }
    processing_shape.AppendDim(size_i);
  } else {
    processing_shape.AppendDim(-1);
  }
}

static bool BuildProcessingData(StridedSliceDenseSpec& dense_spec,
                                StridedSliceParams& params,
                                ProcessingData& processing_data) {
  bool is_identity = true;
  bool slice_dim0 = true;
  bool is_simple_slice = true;
  for (int i = 0; i < static_cast<int>(params.input_shape.GetDimNum()); ++i) {
    auto& begin_i = params.begin[i];
    auto& end_i = params.end[i];
    auto& stride_i = params.strides[i];
    auto dim_i = params.input_shape.GetDim(i);
    if (stride_i == 0) {
      return false;
    }

    const uint64_t bit_i = bit1value(i);
    bool shrink_i = (dense_spec.shrink_axis_mask & bit_i);
    const std::array<uint64_t, 2> masks = {{dense_spec.begin_mask & bit_i, dense_spec.end_mask & bit_i}};
    if (dim_i == -1) {
      processing_data.processing_shape.AppendDim(shrink_i ? 1 : -1);
      processing_data.processing_begin.AppendDim(begin_i);
      processing_data.processing_end.AppendDim(shrink_i ? (begin_i + 1) : end_i);
      processing_data.processing_strides.AppendDim(shrink_i ? 1 : stride_i);
      continue;
    }

    const std::array<int64_t, 2> valid_range = {{stride_i > 0 ? 0 : -1, stride_i > 0 ? dim_i : dim_i - 1}};

    auto canonical = [stride_i, dim_i, masks, valid_range](int64_t x, int c) {
      if (masks[c]) {
        return stride_i > 0 ? valid_range[c] : valid_range[static_cast<uint64_t>(c + 1) & static_cast<uint64_t>(1)];
      } else {
        int64_t x_fwd = x < 0 ? dim_i + x : x;  // make negative indices positive
        return x_fwd < valid_range[0] ? valid_range[0] : std::min(x_fwd, valid_range[1]);
      }
    };

    if (shrink_i && stride_i <= 0) {
      return false;
    }
    is_simple_slice = is_simple_slice && (stride_i == 1);

    const bool begin_and_end_masked = ((dense_spec.begin_mask & bit_i) != 0) && ((dense_spec.end_mask & bit_i) != 0);
    if (dense_spec.begin_valid && dense_spec.end_valid) {
      if (shrink_i) {
        // If we are shrinking, the end index is now possibly incorrect. In
        // particular foo[-1] produces sparse_begin = -1, sparse_end = 0.
        // and canonical puts these to n-1 and 0, which implies a degenerate
        // interval. Fortunately, it is now safe to re-create end as begin+1.
        int64_t x_fwd = begin_i < 0 ? dim_i + begin_i : begin_i;
        begin_i = x_fwd;
        end_i = begin_i + 1;
        if (FwdOutOfBound(x_fwd, 0, dim_i)) {
          return false;
        }
      } else {
        begin_i = canonical(begin_i, 0);
        end_i = canonical(end_i, 1);
      }

      processing_data.processing_begin.AppendDim(begin_i);
      processing_data.processing_end.AppendDim(end_i);
      processing_data.processing_strides.AppendDim(stride_i);

      // Update optimization values
      bool take_all_in_dimension = stride_i == 1 && begin_i == 0 && end_i == dim_i;
      is_identity = is_identity && take_all_in_dimension;
      slice_dim0 = slice_dim0 && ((i == 0 && stride_i == 1) || take_all_in_dimension);
    } else {
      is_identity = is_identity && (stride_i == 1 && begin_and_end_masked);
      slice_dim0 = slice_dim0 && ((i == 0 && stride_i == 1) || begin_and_end_masked);
      processing_data.processing_begin.AppendDim(begin_i);
      processing_data.processing_end.AppendDim(end_i);
      processing_data.processing_strides.AppendDim(1);
    }

    // Compute the processing shape (the intermediate Eigen will produce)
    InputParamUnit input_param_unit = {begin_i, end_i, stride_i, dim_i, shrink_i};
    BuildProcessingShape(dense_spec, input_param_unit, begin_and_end_masked, processing_data.processing_shape);
  }
  return true;
}

static void BuildFinalShape(ProcessingData& processing_data,
                            StridedSliceDenseSpec& dense_spec,
                            StridedSliceParams& params,
                            gert::Shape* out_shape) {
  params.begin.SetDimNum(0);
  params.end.SetDimNum(0);
  params.strides.SetDimNum(0);
  out_shape->SetDimNum(0);
  gert::Shape final_shape_input;
  int shrink_gather_index = 0;
  for (size_t i = 0; i < dense_spec.final_shape_gather_indices.GetDimNum(); i++) {
    auto gather_index = dense_spec.final_shape_gather_indices.GetDim(i);
    if (gather_index >= 0) {
      const auto dim_gather_i = processing_data.processing_shape[gather_index];
      out_shape->AppendDim(dim_gather_i);
      final_shape_input.AppendDim(params.input_shape.GetDim(gather_index));
      params.begin.AppendDim(processing_data.processing_begin[gather_index]);
      params.end.AppendDim(processing_data.processing_end[gather_index]);
      params.strides.AppendDim(processing_data.processing_strides[gather_index]);
      shrink_gather_index = gather_index + 1;
    } else if (gather_index == kStridedSliceNewAxis) {
      out_shape->AppendDim(1);
      // input is scalar
      if (params.input_shape.IsScalar()) {
        final_shape_input.AppendDim(1);
        params.begin.AppendDim(0);
        params.end.AppendDim(1);
        params.strides.AppendDim(1);
      }
    } else {
      final_shape_input.AppendDim(params.input_shape.GetDim(shrink_gather_index));
      params.begin.AppendDim(processing_data.processing_begin[shrink_gather_index]);
      params.end.AppendDim(processing_data.processing_begin[shrink_gather_index] + 1);
      params.strides.AppendDim(1);
      shrink_gather_index += 1;
    }
  }

  params.input_shape = final_shape_input;
}

static bool InferShape(StridedSliceParams& params, gert::Shape* out_shape) {
  // Use bit compares to ensure ellipsis_mask is 0 or a power of 2
  // i.e. there exists only no more than one ellipsis
  auto& ellipsis_mask = params.ellipsis_mask;
  if ((ellipsis_mask != 0) && ((ellipsis_mask & (ellipsis_mask - 1)) != 0)) {
    return false;
  }

  // Step 1: Account for ellipsis and new axis
  //
  // Check for ellipses and count how many non-newaxis' there are after
  StridedSliceSparseSpec sparse_spec = {0, 0, params.begin, params.end, params.strides,
                                        params.begin_mask, params.end_mask, params.ellipsis_mask,
                                        params.new_axis_mask, params.shrink_axis_mask};
  BuildSparseSpec(params, sparse_spec);

  // Step 2: Make a sparse spec into a full index spec
  //
  // The sparse spec does not correspond to the number of dimensions
  // Make a dense spec that corresponds to the number of dimensions
  //
  // For example suppose foo[...,3:] on foo.shape=(2,2,3) then
  // we need to produce the missing begin_mask for the first two
  // dimensions i.e. from begin_mask_spec=0, end_mask_spec=2
  // we achieve begin_mask=6, end_mask=7
  StridedSliceDenseSpec dense_spec = {static_cast<int64_t>(params.input_shape.GetDimNum()), 0, 0,
                                      params.begin_valid, params.end_valid, params.begin, params.end, params.strides};
  if (!BuildDenseSpec(sparse_spec, &dense_spec)) {
    return false;
  }

  // Step 3: Make implicit ranges (non-zero begin_masks and end_masks) explicit
  //         and bounds check!
  ProcessingData processing_data;
  params.begin = dense_spec.begin;
  params.end = dense_spec.end;
  params.strides = dense_spec.strides;
  if (!BuildProcessingData(dense_spec, params, processing_data)) {
    return false;
  }

  // Step 4: Compute the final shape
  //
  // new_axis will increase dimension by 1 (with a one-size dimension)
  // slices like foo[3,...] will reduce dimension by 1.
  // This cannot be done earlier, because it depends on Step 3.
  BuildFinalShape(processing_data, dense_spec, params, out_shape);
  return true;
}
} // namespace ops
#endif  // OPS_BUILT_IN_OP_PROTO_RUNTIME_STRIDED_SLICE_H_