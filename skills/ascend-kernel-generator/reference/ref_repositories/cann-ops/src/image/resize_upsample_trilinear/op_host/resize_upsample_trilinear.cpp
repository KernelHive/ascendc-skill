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
 * \file resize_upsample_trilinear.cpp
 * \brief
 */
#include <vector>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "resize_upsample_trilinear_tiling.h"

namespace optiling {
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
    do {                                      \
        if (cond) {                           \
            log_func;                         \
            expr;                             \
        }                                     \
    } while (0)
}

namespace optiling {

static constexpr int OUTPUT_SIZE_IDX = 0;
static constexpr int SCALES_IDX = 1;
static constexpr int ALIGN_CORNERS_IDX = 2;
static constexpr int INPUT_X_IDX = 0;
static constexpr int OUTPUT_Y_IDX = 0;

static constexpr int OUTPUT_WEIGHT_IDX = 2;
static constexpr int OUTPUT_HEIGHT_IDX = 1;
static constexpr int OUTPUT_DEPTH_IDX = 0;

static constexpr int SCALE_W_IDX = 4;
static constexpr int SCALE_H_IDX = 3;
static constexpr int SCALE_D_IDX = 2;

static constexpr int DIREC_WIDTH = 0;
static constexpr int DIREC_HEIGHT = 1;
static constexpr int DIREC_DEEPTH = 2;
static constexpr int REDUNDANCE = 2;

static constexpr int FLOAT_BYTE_LENGTH = 4;
static constexpr int FLOAT16_BYTE_LENGTH = 2;
static constexpr int BYTE_SIZE = 8;

static constexpr int TILING_DATATYPE_FP16 = 1000;
static constexpr int TILING_DATATYPE_FP32 = 3000;
static constexpr int TILING_DATATYPE_BF16 = 2000;

static constexpr int SLIDE_SIZE = 16;       // best matmul paradigm.bf16,f16:16*16 x 16*16 ,f32:16*8 x 8*16  value of N
static constexpr uint32_t BATCH_SIZE = 16;  // value of M
static constexpr int RESERVED_WORKSPACE_SIZE = 32 * 1024 * 1024;

constexpr float MAX_SUPPORT_SCALE = 50.0f;
constexpr float BEST_PERFORMANCE_SCALE_1 = 27.0f;
constexpr float BEST_PERFORMANCE_SCALE_2 = 11.0f;
constexpr float BEST_PERFORMANCE_SCALE_3 = 3.0f;

constexpr uint32_t MIN_BATCH_SIZE = 128;
constexpr uint32_t BEST_PERFORMANCE_BATCH_SIZE_1 = 256;
constexpr uint32_t BEST_PERFORMANCE_BATCH_SIZE_2 = 512;
constexpr uint32_t BEST_PERFORMANCE_BATCH_SIZE_3 = 1024;

constexpr uint32_t MAX_DATA_NUM = 8192;
constexpr uint32_t RESERVED_NUM = 4;

static constexpr uint16_t SOC_VERSION_310 = 200;
static constexpr uint16_t SOC_VERSION_910 = 220;

uint16_t totalCoreNum = 0;
uint16_t socVersionType = SOC_VERSION_910;

template <typename T>
inline uint32_t CEIL(T x) {
  int32_t floor_v = int32_t(x);
  if (x == floor_v) {
    return floor_v;
  }
  return floor_v + 1;
}

template <typename T1, typename T2>
inline uint32_t CEILA2B(T1 a, T2 b) {
  if (b != 0) {
    return (a + b - 1) / b;
  } else {
    return a;
  }
}

inline bool allTrue(bool a, bool b) {
  return a && b;
}

inline uint32_t GET_DATATYPE_SIZE(ge::DataType dataType) {
  if (dataType == ge::DT_FLOAT) {
    return FLOAT_BYTE_LENGTH;
  } else if (dataType == ge::DT_FLOAT16) {
    return FLOAT16_BYTE_LENGTH;
  } else if (dataType == ge::DT_BF16) {
    return FLOAT16_BYTE_LENGTH;
  } else {
    return FLOAT_BYTE_LENGTH;
  }
}

template <typename T>
T GetOptionalAttr(const gert::RuntimeAttrs* attrs, const int idx, const T& defaultValue) {
  const T* attrPtr = attrs->GetAttrPointer<T>(idx);
  T outValue = (nullptr == attrPtr) ? defaultValue : (*attrPtr);
  return outValue;
}

static float ComputeScales(float scale, uint32_t input_size, uint32_t output_size) {
  auto zero = static_cast<float>(0.);
  if (scale > zero) {
    return scale;
  } else if (output_size == 0) {
    return zero;
  } else {
    return (static_cast<float>(input_size) / output_size);
  }
}

static inline bool FloatEqual(float a, float b) {
  float closeTo0 = float(1e-6);
  if (a > b) {
    return a - b < closeTo0;
  } else {
    return b - a < closeTo0;
  }
}

static float AreaPixelComputeScale(bool align_corners, int64_t input_size, int64_t output_size, float scale) {
  if (output_size == input_size) {
    return static_cast<float>(1);
  }
  if (align_corners) {
    if (output_size > 1) {
      return static_cast<float>(input_size - 1) / (output_size - 1);
    } else {
      return static_cast<float>(0);
    }
  } else {
    return ComputeScales(scale, input_size, output_size);
  }
}

static bool CheckShapes(gert::TilingContext* context, int64_t w, int64_t h, int64_t d) {
  OP_TILING_CHECK(
      !(w > 0 && h > 0 && d > 0),
      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                      "shape should bigger than 0, but got shape (D: %ld,H: %ld, W: %ld)", d, h, w),
      return false);
  return true;
}

static bool CheckScales(gert::TilingContext* context, float scalesW, float scalesH, float scalesD) {
  OP_TILING_CHECK((scalesW > MAX_SUPPORT_SCALE || scalesH > MAX_SUPPORT_SCALE || scalesD > MAX_SUPPORT_SCALE),
                  VECTOR_INNER_ERR_REPORT_TILIING(
                      context->GetNodeName(),
                      "Scales should less than 50, but got scale (scalesW: %f, scalesH: %f, scalesD: %f) ", scalesW,
                      scalesH, scalesD),
                  return false);
  return true;
}

static bool SetTCubeTilingD(gert::TilingContext* context, uint32_t batches, UpsampleTrilinearTilingData& tiling,
                            ge::DataType dataType) {
  auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
  matmul_tiling::MatmulApiTiling mmTiling_d;
  mmTiling_d.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
  mmTiling_d.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
  mmTiling_d.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
  // 这里设置的k不能比kernel里面设置的小，否则会出问题
  mmTiling_d.SetShape(tiling.get_output_d(), tiling.get_output_w() * tiling.get_output_h(),
                      CEIL(SLIDE_SIZE * tiling.get_scale_d()) + REDUNDANCE);
  mmTiling_d.SetOrgShape(tiling.get_output_d(), tiling.get_output_w() * tiling.get_output_h(),
                         batches * tiling.get_input_d());
  OP_TILING_CHECK((mmTiling_d.GetTiling(tiling.matmul_tiling_d) == -1),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "MatmulApiTiling_d set failed"),
                  return false);
  return true;
}

static bool SetTCubeTilingH(gert::TilingContext* context, uint32_t batches, UpsampleTrilinearTilingData& tiling,
                            ge::DataType dataType) {
  auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
  matmul_tiling::MatmulApiTiling mmTiling_h;
  mmTiling_h.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
  mmTiling_h.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
  mmTiling_h.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
  mmTiling_h.SetShape(tiling.get_output_h(), tiling.get_output_w(),
                      CEIL(SLIDE_SIZE * tiling.get_scale_h()) + REDUNDANCE);
  mmTiling_h.SetOrgShape(tiling.get_output_h(), tiling.get_output_w(),
                         batches * tiling.get_input_d() * tiling.get_input_h());
  OP_TILING_CHECK((mmTiling_h.GetTiling(tiling.matmul_tiling_h) == -1),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "MatmulApiTiling_h set failed"),
                  return false);
  return true;
}

static bool SetTCubeTilingW(gert::TilingContext* context, uint32_t batches, UpsampleTrilinearTilingData& tiling,
                            ge::DataType dataType) {
  auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
  matmul_tiling::MatmulApiTiling mmTiling_w;
  mmTiling_w.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
  mmTiling_w.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
  mmTiling_w.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
  mmTiling_w.SetShape(batches * tiling.get_input_d() * tiling.get_input_h(), SLIDE_SIZE,
                      CEIL(SLIDE_SIZE * tiling.get_scale_w()) + REDUNDANCE);
  mmTiling_w.SetOrgShape(batches * tiling.get_input_d() * tiling.get_input_h(), tiling.get_output_w(),
                         tiling.get_input_w());
  OP_TILING_CHECK((mmTiling_w.GetTiling(tiling.matmul_tiling_w) == -1),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "MatmulApiTiling_w set failed"),
                  return false);
  return true;
}

static void SetTailGroupList(UpsampleTrilinearTilingData& tiling, int direction, uint32_t* tailGroupSlideStartInxList,
                             uint32_t* tailGroupSlideEndInxList) {
  if (DIREC_WIDTH == direction) {
    tiling.set_tail_group_slide_start_inx_w_list(tailGroupSlideStartInxList);
    tiling.set_tail_group_slide_end_inx_w_list(tailGroupSlideEndInxList);
  } else if (DIREC_HEIGHT == direction) {
    // h方向按照slide切分的逻辑
    tiling.set_tail_group_start_inx_h_list(tailGroupSlideStartInxList);
    tiling.set_tail_group_end_inx_h_list(tailGroupSlideEndInxList);
  } else {
    tiling.set_tail_group_start_inx_d_list(tailGroupSlideStartInxList);
    tiling.set_tail_group_end_inx_d_list(tailGroupSlideEndInxList);
  }
}

static void SetTailBatchList(UpsampleTrilinearTilingData& tiling, int direction, uint32_t* tailGroupBlockStartInxList,
                             uint32_t* tailGroupBlockEndInxList) {
  if (DIREC_WIDTH == direction) {
    tiling.set_tail_group_start_inx_w_list(tailGroupBlockStartInxList);
    tiling.set_tail_group_end_inx_w_list(tailGroupBlockEndInxList);
  } else if (DIREC_HEIGHT == direction) {
    // batch方向按照个数切分的逻辑
    tiling.set_tail_group_batch_start_inx_h_list(tailGroupBlockStartInxList);
    tiling.set_tail_group_batch_end_inx_h_list(tailGroupBlockEndInxList);
  } else {
    tiling.set_tail_group_batch_start_inx_d_list(tailGroupBlockStartInxList);
    tiling.set_tail_group_batch_end_inx_d_list(tailGroupBlockEndInxList);
  }
}

static uint32_t calcIndxPerCoreInTailBlockW(UpsampleTrilinearTilingData& tiling, uint32_t core_num, uint32_t group_size,
                                            uint32_t* core_per_group, uint32_t group_tail_w) {
  uint32_t batch = tiling.get_batches() * tiling.get_input_d() * tiling.get_input_h();
  uint32_t tailGroupBlockStartInxList[core_num] = {0};
  uint32_t tailGroupBlockEndInxList[core_num] = {0};
  uint32_t tailGroupSlideStartInxList[core_num] = {0};
  uint32_t tailGroupSlideEndInxList[core_num] = {0};
  uint32_t slide_num = tiling.get_output_w() / (SLIDE_SIZE * core_num);
  uint32_t tail_block_start;
  uint32_t last_slide_idx = slide_num > 0 ? core_num * slide_num * SLIDE_SIZE - 1 : 0;
  uint32_t core_num_cur_group, coreRealNeedNum = 0;
  size_t idx = 0;
  for (size_t i = 0; i < group_size; i++) {
    if (i == 0) {
      core_num_cur_group = core_per_group[i] + 1;
    } else {
      core_num_cur_group = core_per_group[i] - core_per_group[i - 1];
    }
    // singleM每组至少是16.否则数据量太小不够乘
    uint32_t avg_blocks_per_core = std::max(CEILA2B(batch, core_num_cur_group), BATCH_SIZE);
    for (size_t j = 0; j < core_num_cur_group; j++) {
      tail_block_start = j * avg_blocks_per_core;
      if (tail_block_start >= batch) {
        break;
      }
      tailGroupBlockStartInxList[idx] = tail_block_start;
      tailGroupBlockEndInxList[idx] = std::min(batch - 1, tailGroupBlockStartInxList[idx] + avg_blocks_per_core - 1);
      if (i == 0) {
        tailGroupSlideStartInxList[idx] = last_slide_idx == 0 ? 0 : last_slide_idx + i * SLIDE_SIZE + 1;
      } else {
        tailGroupSlideStartInxList[idx] = last_slide_idx == 0 ? i * SLIDE_SIZE : last_slide_idx + i * SLIDE_SIZE + 1;
      }
      if (i == group_size - 1) {
        uint32_t group_width = group_tail_w == 0 ? SLIDE_SIZE : group_tail_w;
        tailGroupSlideEndInxList[idx] = tailGroupSlideStartInxList[idx] + group_width - 1;
      } else {
        tailGroupSlideEndInxList[idx] = tailGroupSlideStartInxList[idx] + SLIDE_SIZE - 1;
      }
      coreRealNeedNum++;
      idx++;
    }
  }
  SetTailGroupList(tiling, DIREC_WIDTH, tailGroupSlideStartInxList, tailGroupSlideEndInxList);
  SetTailBatchList(tiling, DIREC_WIDTH, tailGroupBlockStartInxList, tailGroupBlockEndInxList);
  return coreRealNeedNum;
}

static uint32_t calcIndxPerCoreInTailBlock(UpsampleTrilinearTilingData& tiling, uint32_t core_num, int direction,
                                           uint32_t group_size, uint32_t* core_per_group, uint32_t group_tail) {
  uint32_t batch, last_slide_index, slide_num;
  if (DIREC_HEIGHT == direction) {
    batch = tiling.get_batches() * tiling.get_input_d();
    slide_num = tiling.get_output_h() / (SLIDE_SIZE * core_num);
  } else {
    batch = tiling.get_batches();
    slide_num = tiling.get_output_d() / (SLIDE_SIZE * core_num);
  }
  last_slide_index = slide_num > 0 ? core_num * slide_num * SLIDE_SIZE - 1 : 0;
  uint32_t tailGroupBlockStartInxList[core_num] = {0};
  uint32_t tailGroupBlockEndInxList[core_num] = {0};
  uint32_t tailGroupSlideStartInxList[core_num] = {0};
  uint32_t tailGroupSlideEndInxList[core_num] = {0};
  uint32_t core_num_cur_group, coreRealNeedNum = 0;
  size_t free_core = 0;
  size_t idx = 0;
  for (size_t i = 0; i < group_size; i++) {
    uint32_t tail, tail_temp;
    if (i == 0) {
      core_num_cur_group = core_per_group[i] + 1;
    } else {
      core_num_cur_group = core_per_group[i] - core_per_group[i - 1];
    }
    if (core_num_cur_group == 0) {
      continue;
    }
    // batch侧分组最少可到1个batch一组
    uint32_t avg_batch_per_core = batch / core_num_cur_group;
    tail = batch % core_num_cur_group;
    uint32_t real_core = avg_batch_per_core > 0 ? core_num_cur_group : tail;
    tail_temp = tail;
    idx = idx - free_core;
    free_core = 0;
    for (size_t j = 0; j < core_num_cur_group; j++) {
      // 计算按照slide切分的方向
      if (i == 0) {
        tailGroupSlideStartInxList[idx] = last_slide_index == 0 ? 0 : last_slide_index + i * SLIDE_SIZE + 1;
      } else {
        tailGroupSlideStartInxList[idx] =
            last_slide_index == 0 ? i * SLIDE_SIZE : last_slide_index + i * SLIDE_SIZE + 1;
      }
      if (i == group_size - 1) {
        // 当前核处理最后一组，但是要考虑到如果尾块是0，说明最后一组处理地也是一个完整滑块
        uint32_t group_width = group_tail == 0 ? SLIDE_SIZE : group_tail;
        tailGroupSlideEndInxList[idx] = tailGroupSlideStartInxList[idx] + group_width - 1;
      } else {
        tailGroupSlideEndInxList[idx] = tailGroupSlideStartInxList[idx] + SLIDE_SIZE - 1;
      }
      // 注意存在1的场景，start和end按照左闭右开设置
      if (j < real_core) {
        if (tail_temp > 0) {
          tail_temp--;
          if (avg_batch_per_core == 0) {
            tailGroupBlockStartInxList[idx] = j;
            tailGroupBlockEndInxList[idx] = tailGroupBlockStartInxList[idx] + 1;
          } else {
            tailGroupBlockStartInxList[idx] = j == 0 ? 0 : j * avg_batch_per_core + tail - tail_temp - 1;
            tailGroupBlockEndInxList[idx] = tailGroupBlockStartInxList[idx] + avg_batch_per_core + 1;
          }
        } else {
          tailGroupBlockStartInxList[idx] = j == 0 ? 0 : j * avg_batch_per_core + tail;
          tailGroupBlockEndInxList[idx] = tailGroupBlockStartInxList[idx] + avg_batch_per_core;
        }
        coreRealNeedNum++;
      } else {
        free_core++;
      }
      idx++;
    }
  }
  SetTailGroupList(tiling, direction, tailGroupSlideStartInxList, tailGroupSlideEndInxList);
  SetTailBatchList(tiling, direction, tailGroupBlockStartInxList, tailGroupBlockEndInxList);
  return coreRealNeedNum;
}

static uint32_t calcIndxPerCore(UpsampleTrilinearTilingData& tiling, uint32_t core_num, int64_t batches,
                                int direction) {
  int64_t direc_length;
  uint32_t tailGroupBlockStartInxList[core_num] = {0};
  uint32_t tailGroupBlockEndInxList[core_num] = {0};
  uint32_t tailGroupSlideStartInxList[core_num] = {0};
  uint32_t tailGroupSlideEndInxList[core_num] = {0};
  if ((DIREC_WIDTH == direction && FloatEqual(tiling.get_scale_w(), 1.0)) ||
      (DIREC_HEIGHT == direction && FloatEqual(tiling.get_scale_h(), 1.0))) {
    SetTailGroupList(tiling, direction, tailGroupSlideStartInxList, tailGroupSlideEndInxList);
    SetTailBatchList(tiling, direction, tailGroupBlockStartInxList, tailGroupBlockEndInxList);
    // 至少需要一个核计算
    return 1;
  }
  if (allTrue(DIREC_DEEPTH == direction, FloatEqual(tiling.get_scale_d(), 1.0)) &&
      !allTrue(FloatEqual(tiling.get_scale_w(), 1.0), FloatEqual(tiling.get_scale_h(), 1.0))) {
    SetTailGroupList(tiling, direction, tailGroupSlideStartInxList, tailGroupSlideEndInxList);
    SetTailBatchList(tiling, direction, tailGroupBlockStartInxList, tailGroupBlockEndInxList);
    // 全1场景也需要计算下
    return 1;
  }
  if (DIREC_WIDTH == direction) {
    direc_length = tiling.get_output_w();
  } else if (DIREC_HEIGHT == direction) {
    direc_length = tiling.get_output_h();
  } else {
    direc_length = tiling.get_output_d();
  }
  // for example:50*3295
  uint32_t slide_num = direc_length / (SLIDE_SIZE * core_num);
  uint32_t slide_tail_size = direc_length % (SLIDE_SIZE * core_num);
  uint32_t group_size = CEILA2B(slide_tail_size, SLIDE_SIZE);
  uint32_t group_tail_size = slide_tail_size % SLIDE_SIZE;
  uint32_t core_num_per_group = 0;
  uint32_t core_num_per_group_tail = 0;
  uint32_t core_real_need_num = core_num;
  if (group_size != 0) {
    core_num_per_group = core_num / group_size;
    core_num_per_group_tail = core_num % group_size;
  }
  uint32_t core_num_per_group_tail_temp = core_num_per_group_tail;
  uint32_t core_per_group[group_size] = {0};
  for (size_t i = 0; i < group_size; i++) {
    if (core_num_per_group_tail_temp > 0) {
      core_num_per_group_tail_temp--;
      core_per_group[i] = (i + 1) * core_num_per_group + core_num_per_group_tail - core_num_per_group_tail_temp - 1;
    } else {
      core_per_group[i] = (i + 1) * core_num_per_group + core_num_per_group_tail - 1;
    }
  }
  if (DIREC_WIDTH == direction) {
    core_real_need_num = calcIndxPerCoreInTailBlockW(tiling, core_num, group_size, core_per_group, group_tail_size);
  } else {
    core_real_need_num =
        calcIndxPerCoreInTailBlock(tiling, core_num, direction, group_size, core_per_group, group_tail_size);
  }
  return slide_num > 0 ? core_num : core_real_need_num;
}

static uint32_t GetBatchSize(float scale) {
  uint32_t batchSize = MIN_BATCH_SIZE;
  if (scale <= BEST_PERFORMANCE_SCALE_3) {
    batchSize = BEST_PERFORMANCE_BATCH_SIZE_3;
  } else if (scale <= BEST_PERFORMANCE_SCALE_2) {
    batchSize = BEST_PERFORMANCE_BATCH_SIZE_2;
  } else if (scale <= BEST_PERFORMANCE_SCALE_1) {
    batchSize = BEST_PERFORMANCE_BATCH_SIZE_1;
  }
  return batchSize;
}

static uint32_t GetNeedCoreNum(UpsampleTrilinearTilingData& tiling, uint32_t coreNumPlatform) {
  uint32_t batchSize = GetBatchSize(tiling.get_scale_w());
  uint32_t slideSize = 1;
  if (tiling.get_batches() <= batchSize) {
    batchSize = tiling.get_batches();
    if ((CEIL(tiling.get_scale_w()) + 1) != 0 && batchSize != 0) {
      slideSize = (MAX_DATA_NUM / batchSize - RESERVED_NUM) / (CEIL(tiling.get_scale_w()) + 1);
    }
  }

  int64_t slideNum = CEILA2B(tiling.get_output_w(), slideSize) * tiling.get_output_h() * tiling.get_output_d();
  int64_t eachCoreSlideNum = coreNumPlatform > 0 ? slideNum / coreNumPlatform : 0;
  int64_t remainder = coreNumPlatform > 0 ? slideNum % coreNumPlatform : 0;
  int64_t tensorSize = CEIL(slideSize * tiling.get_scale_w()) + RESERVED_NUM;
  tiling.set_each_core_slide_num(eachCoreSlideNum);
  tiling.set_remainder(remainder);
  tiling.set_tail_start_slide_num(eachCoreSlideNum * coreNumPlatform);
  tiling.set_slide_size(slideSize);
  tiling.set_batch_size(batchSize);
  tiling.set_tensor_size(tensorSize);

  uint32_t needCoreNum = coreNumPlatform;
  if(eachCoreSlideNum == 0 && remainder > 0) {
    needCoreNum = remainder;
  }
  return needCoreNum;
}

static uint64_t GetWorkSpaceSize(UpsampleTrilinearTilingData& tiling, uint32_t batches, uint32_t coreRealNeedNum,
                                 uint32_t data_type_size) {
  uint64_t temp_result_w_size, temp_result_h_size, ratio_size_w, ratio_size_h, ratio_size_d;
  // 存放w右乘的临时空间大小,此处是元素大小
  temp_result_w_size = FloatEqual(tiling.get_scale_w(), 1.0)
                           ? 0
                           : batches * tiling.get_input_d() * tiling.get_input_h() * tiling.get_output_w();
  // 存放h左乘的临时空间大小
  temp_result_h_size = FloatEqual(tiling.get_scale_h(), 1.0)
                           ? 0
                           : batches * tiling.get_input_d() * tiling.get_output_h() * tiling.get_output_w();
  // 存放系数矩阵临时空间大小,要求下各个方向最大值，保险起见防止踩踏内存，加个2
  ratio_size_w = (CEIL(SLIDE_SIZE * tiling.get_scale_w()) + REDUNDANCE) * SLIDE_SIZE;
  ratio_size_h = (CEIL(SLIDE_SIZE * tiling.get_scale_h()) + REDUNDANCE) * SLIDE_SIZE;
  ratio_size_d = (CEIL(SLIDE_SIZE * tiling.get_scale_d()) + REDUNDANCE) * SLIDE_SIZE;
  uint64_t max_ratio_size = std::max(ratio_size_w, std::max(ratio_size_h, ratio_size_d));

  tiling.set_ratio_metrix_size(max_ratio_size);
  return (temp_result_w_size + temp_result_h_size + max_ratio_size * coreRealNeedNum) * data_type_size;
}

static ge::graphStatus Tiling4UpsampleTrilinear(gert::TilingContext* context) {
  int batchIdx = OUTPUT_DEPTH_IDX;
  int channelIdx = OUTPUT_HEIGHT_IDX;
  int weightIdx = SCALE_W_IDX;
  int heightIdx = SCALE_H_IDX;
  int depthIdx = SCALE_D_IDX;
  int outwIdx = SCALE_W_IDX;
  int outhIdx = SCALE_H_IDX;
  int outdIdx = SCALE_D_IDX;
  
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  auto socVersion = ascendcPlatform.GetSocVersion();
  totalCoreNum = ascendcPlatform.GetCoreNumAic();
  if (socVersion == platform_ascendc::SocVersion::ASCEND310P) {
      totalCoreNum = ascendcPlatform.GetCoreNumAiv();
      socVersionType = SOC_VERSION_310;
  }
  if (socVersionType == SOC_VERSION_310) {
    batchIdx = SCALE_H_IDX;
    channelIdx = SCALE_W_IDX;
    weightIdx = SCALE_D_IDX;
    heightIdx = OUTPUT_HEIGHT_IDX;
    depthIdx = OUTPUT_DEPTH_IDX;
    outwIdx = SCALE_D_IDX;
    outhIdx = OUTPUT_HEIGHT_IDX;
    outdIdx = OUTPUT_DEPTH_IDX;
  }

  UpsampleTrilinearTilingData tiling;

  const gert::RuntimeAttrs* attrs = context->GetAttrs();
  const auto align_corners = GetOptionalAttr<bool>(attrs, SCALES_IDX, false);

  auto output_storage_shape = context->GetOutputShape(0)->GetStorageShape();
  if (!CheckShapes(context, output_storage_shape.GetDim(outwIdx), output_storage_shape.GetDim(outhIdx),
                   output_storage_shape.GetDim(outdIdx))) {
    return ge::GRAPH_FAILED;
  }
  tiling.set_output_w(output_storage_shape.GetDim(outwIdx));
  tiling.set_output_h(output_storage_shape.GetDim(outhIdx));
  tiling.set_output_d(output_storage_shape.GetDim(outdIdx));
  auto input_storage_shape = context->GetInputShape(0)->GetStorageShape();
  tiling.set_scale_w(AreaPixelComputeScale(align_corners, input_storage_shape.GetDim(weightIdx),
                                           output_storage_shape.GetDim(outwIdx),
                                           GetOptionalAttr<float>(attrs, SCALE_W_IDX, static_cast<float>(1.0))));
  tiling.set_scale_h(AreaPixelComputeScale(align_corners, input_storage_shape.GetDim(heightIdx),
                                           output_storage_shape.GetDim(outhIdx),
                                           GetOptionalAttr<float>(attrs, SCALE_H_IDX, static_cast<float>(1.0))));
  tiling.set_scale_d(AreaPixelComputeScale(align_corners, input_storage_shape.GetDim(depthIdx),
                                           output_storage_shape.GetDim(outdIdx),
                                           GetOptionalAttr<float>(attrs, SCALE_D_IDX, static_cast<float>(1.0))));
  if (!CheckScales(context, tiling.get_scale_w(), tiling.get_scale_h(), tiling.get_scale_d())) {
    return ge::GRAPH_FAILED;
  }
  tiling.set_align_corners(align_corners ? 1 : 0);

  tiling.set_input_w(input_storage_shape.GetDim(weightIdx));
  tiling.set_input_h(input_storage_shape.GetDim(heightIdx));
  tiling.set_input_d(input_storage_shape.GetDim(depthIdx));
  if (!CheckShapes(context, tiling.get_input_w(), tiling.get_input_h(), tiling.get_input_d())) {
    return ge::GRAPH_FAILED;
  }
  // data type:fp32,fp16,bf16
  auto dataType = context->GetInputDesc(0)->GetDataType();
  uint64_t tilingKey = 0;
  if (dataType == ge::DT_FLOAT16) {
    tilingKey = TILING_DATATYPE_FP16;
  } else if (dataType == ge::DT_FLOAT) {
    tilingKey = TILING_DATATYPE_FP32;
  } else if (dataType == ge::DT_BF16) {
    tilingKey = TILING_DATATYPE_BF16;
  }
  context->SetTilingKey(tilingKey);
  uint32_t data_type_size = GET_DATATYPE_SIZE(dataType);

  // Block Tiling
  uint16_t maxCoreNum = totalCoreNum;
  tiling.set_total_core_num(maxCoreNum);
  const int64_t batch = input_storage_shape.GetDim(batchIdx);
  const int64_t channel = input_storage_shape.GetDim(channelIdx);
  tiling.set_batches(batch * channel);

  uint32_t coreRealNeedNum = 0;
  if (socVersionType == SOC_VERSION_310) {
    coreRealNeedNum = GetNeedCoreNum(tiling, maxCoreNum);
    coreRealNeedNum = coreRealNeedNum < 1 ? 1 : coreRealNeedNum;
    context->SetBlockDim(maxCoreNum);
    tiling.set_real_core_num(coreRealNeedNum);
  } else {
    uint32_t core_real_need_num_w =
        calcIndxPerCore(tiling, maxCoreNum, batch * channel * tiling.get_input_d() * tiling.get_input_h(), DIREC_WIDTH);
    uint32_t core_real_need_num_h =
        calcIndxPerCore(tiling, maxCoreNum, batch * channel * tiling.get_input_d(), DIREC_HEIGHT);
    uint32_t core_real_need_num_d = calcIndxPerCore(tiling, maxCoreNum, batch * channel, DIREC_DEEPTH);
    coreRealNeedNum = std::max(core_real_need_num_w, std::max(core_real_need_num_h, core_real_need_num_d));

    if (!SetTCubeTilingW(context, batch * channel, tiling, dataType)) {
      return ge::GRAPH_FAILED;
    }
    if (!SetTCubeTilingH(context, batch * channel, tiling, dataType)) {
      return ge::GRAPH_FAILED;
    }
    if (!SetTCubeTilingD(context, batch * channel, tiling, dataType)) {
      return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(coreRealNeedNum);
  }

  size_t* currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] =
      GetWorkSpaceSize(tiling, batch * channel, coreRealNeedNum, data_type_size) + RESERVED_WORKSPACE_SIZE;

  // savetobuffer会把数据加载到缓存里面待发送，所以对tiling的所有设值在在此之前执行
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ResizeUpsampleTrilinear).Tiling(Tiling4UpsampleTrilinear);
}  // namespace optiling
