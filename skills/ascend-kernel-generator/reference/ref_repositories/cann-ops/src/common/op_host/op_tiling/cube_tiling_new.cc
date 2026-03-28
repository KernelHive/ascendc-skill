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
 * \file cube_tiling.cpp
 * \brief
 */
#include <cstdlib>
#include <string>

#include "error_util.h"
#include "cube_tiling_new.h"

namespace {
  #define CUBE_INNER_ERR_REPORT(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
  #define OP_LOGD(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
  static const int kRangeDimLen = 2;

  const std::vector<std::string> kConv3DVarNames = {"batch_n", "fmap_d", "fmap_h", "fmap_w"};
  const std::vector<std::string> kConv3DBpInputVarNames = {"batch_n", "dedy_d", "dedy_h", "dedy_w"};
  const std::vector<std::string> kConv3DTransposeVarNames = {"batch_n", "dedy_d", "dedy_h", "dedy_w"};
  const std::vector<std::string> kAvgPool3DGradVarNames = {"batch_n", "dedy_d", "dedy_h", "dedy_w"};
  const std::map<std::string, std::vector<std::string>> kOpVarNamesMap = {
      {"Conv3D", kConv3DVarNames},
      {"Conv3DBackpropFilter", kConv3DVarNames},
      {"Conv3DBackpropInput", kConv3DBpInputVarNames},
      {"Conv3DTranspose", kConv3DTransposeVarNames},
      {"AvgPool3D", kConv3DVarNames},
      {"AvgPool3DGrad", kAvgPool3DGradVarNames}
  };


  static bool is_shape_in_range_cube(const std::vector<int64_t> &shape, const std::vector<int64_t> &range) {
    const std::vector<int32_t> shape_dim = {0, 2, 3};
    const std::vector<int32_t> range_dim = {0, 1, 2, 3, 4, 5};
    if (range.size() == range_dim.size()) {
      for (size_t i = 0; i < shape_dim.size(); ++i) {
        if (shape[shape_dim[i]] < range[range_dim[i * kRangeDimLen]] ||
            shape[shape_dim[i]] > range[range_dim[i * kRangeDimLen + 1]]) {
          return false;
        }
      }
    } else if (range.size() == kRangeDimLen) {
      if (shape[shape_dim[0]] < range[0] || shape[shape_dim[0]] > range[1]) {
        return false;
      }
    } else {
      return false;
    }

    return true;
  }

  string cube_tiling_batch(const std::string& op_type, const std::vector<int64_t>& cur_shape,
                           const nlohmann::json& compile_info, string tiling_id) {
    if (cur_shape.empty()) {
      return tiling_id;
    }

    if (!compile_info.contains("tiling_range")) {
      CUBE_INNER_ERR_REPORT(op_type.c_str(), "compile_info does not contain the key value of the tiling_range");
      return tiling_id;
    }
    auto& tiling_range = compile_info.at("tiling_range");
    for (auto it = tiling_range.begin(); it != tiling_range.end(); it++) {
      auto& range = it.value();
      if (is_shape_in_range_cube(cur_shape, range)) {
        tiling_id = it.key();
      }
    }
    return tiling_id;
  }

  string cube_tiling_nhw(const std::string& op_type, const std::vector<int64_t>& cur_shape,
                         const nlohmann::json& compile_info, string tiling_id) {
    if (!compile_info.contains("repo_seeds") || !compile_info.contains("repo_range")) {
      CUBE_INNER_ERR_REPORT(op_type.c_str(),
                            "compile_info does not contain the key value of the repo_sends or repo_range");
      return tiling_id;
    }

    int32_t seedHDim = 1;
    int32_t seedWDim = 2;
    int32_t hDim = 2;
    int32_t wDim = 3;

    const auto& repo_range = compile_info.at("repo_range");
    auto& tiling_seeds = compile_info.at("repo_seeds");
    int64_t min_dist = std::numeric_limits<int64_t>::max();
    for (auto it = tiling_seeds.begin(); it != tiling_seeds.end(); it++) {
      std::vector<int32_t> seed = it.value().get<std::vector<int32_t>>();
      auto& range = repo_range[it.key()];

      if (is_shape_in_range_cube(cur_shape, range)) {
        int32_t dist = abs(cur_shape[hDim] - seed[seedHDim]) + abs(cur_shape[wDim] - seed[seedWDim]);
        if (dist < min_dist) {
          tiling_id = it.key();
          min_dist = dist;
        }
      }
    }
    if (tiling_id.empty()) {
      if (!compile_info.contains("cost_range")) {
        CUBE_INNER_ERR_REPORT(op_type.c_str(), "compile_info does not contain the key value of the cost_range");
        return tiling_id;
      }

      auto& cost_range = compile_info.at("cost_range");
      for (auto it = cost_range.begin(); it != cost_range.end(); it++) {
        auto& range = it.value();
        if (is_shape_in_range_cube(cur_shape, range)) {
          tiling_id = it.key();
          break;
        }
      }
    }
    return tiling_id;
  }
}

namespace optiling {
  bool cube_tiling(const std::string& op_type,
                   const std::vector<int64_t>& input_shape,
                   const std::vector<int64_t>& var_value,
                   const nlohmann::json& compile_info) {
    try {
      OP_LOGD(op_type.c_str(), "input compile info: %s", compile_info.dump().c_str());
      std::vector<std::string> vars = compile_info.at("_vars").begin().value().get<std::vector<std::string>>();
      std::string tiling_id("");

      if (compile_info["tiling_type"] == "default_tiling") {
        std::vector<int64_t> default_range = compile_info["default_range"].begin().value().get<std::vector<int64_t>>();
        if (is_shape_in_range_cube(input_shape, default_range)) {
          tiling_id = compile_info["default_range"].begin().key();
        }
      } else if (vars.size() != 1) {
        tiling_id = cube_tiling_nhw(op_type, input_shape, compile_info, tiling_id);
      } else {
        tiling_id = cube_tiling_batch(op_type, input_shape, compile_info, tiling_id);
      }

      if (tiling_id.empty()) {
          if (compile_info.contains("correct_range_flag") && compile_info["correct_range_flag"]) {
              CUBE_INNER_ERR_REPORT(op_type.c_str(), "The original range does not meet requirements,"
                "new range is generated during op compile, but the shape is not covered by new range");
          }
          CUBE_INNER_ERR_REPORT(op_type.c_str(), "This shape is not covered by any tiling,"
                                                 "please modify range and recompile");
          return false;
      }

      if (!compile_info.contains("block_dim")) {
          CUBE_INNER_ERR_REPORT(op_type.c_str(), "compile_info does not contain the key value of the block_dim");
          return false;
      }

      OP_LOGD(op_type.c_str(), "get tiling_id: %s", tiling_id.c_str());
      return true;
    } catch (...) {
        CUBE_INNER_ERR_REPORT(op_type.c_str(), "get unknown exception, please check compile info json.");
      return false;
    }
  }
}  // namespace optiling
