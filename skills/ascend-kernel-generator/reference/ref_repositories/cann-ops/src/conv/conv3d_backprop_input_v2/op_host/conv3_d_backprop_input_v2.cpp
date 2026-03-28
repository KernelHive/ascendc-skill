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
 * \file conv3d_backprop_input_v2_tiling.cpp
 * \brief
 */

 #include "conv3d_backprop_input_v2_tiling.h"

 #include <map>
 #include <numeric>
 #include <cinttypes>
 
 #include "cube_tiling_runtime.h"
 #include "graph/utils/type_utils.h"
 #include "op_log.h"
 #include "register/op_impl_registry.h"
 #include "tiling/tiling_templates_registry.h"
 #include "tiling/tiling_type.h"
 #include "cube/util/math_util.h"
 
 using namespace optiling::cachetiling;
 
namespace {
constexpr size_t OUTPUT_PADDING_DIM = 5;
constexpr int32_t BYTE_BLOCK = 32;
constexpr uint32_t DB_ON = 2;
constexpr uint32_t B16_BITS = 4;
constexpr uint32_t FP32_BITS = 3;
constexpr uint32_t FP32_DATA_SIZE = 4;
constexpr uint32_t F16_DATA_SIZE = 2; // BF16和FP16共用
constexpr uint32_t NUM_FP32_C1OUT = 2;
constexpr int32_t BLOCK_CUBE = 16;
constexpr int32_t CORE_NUM_910B3 = 20;
constexpr int32_t CORE_NUM_910B2 = 24;
constexpr int64_t L0_SIZE = 65536;
constexpr int64_t WORKSIZE = 16 * 1024 * 1024;
constexpr uint64_t L1_SIZE = 512 * 1024 - 128;
constexpr uint32_t BEST_BASE_M = 128;
constexpr uint32_t BEST_BASE_K = 128;
constexpr uint32_t BEST_BASE_N = 256;
constexpr int32_t FACTOR_910B3 = 5;
constexpr int32_t DIM_THRESHOLD = 4;
constexpr int32_t DIM_FACTOR = 2;
constexpr int32_t FMAP_H_NUM = 2;
constexpr int32_t BUFFER_NUM_DB = 2;
constexpr int32_t BUFFER_NUM_L1 = 4;
constexpr float CORE_USED_THRESHOLD = 0.6f;

const size_t Y_INDEX = 0;
const size_t FILTER_INDEX = 1;
const size_t OUTPUT_BP_INDEX = 2;
const size_t BAIS_INDEX = 3;
const size_t OFFSET_W_INDEX = 4;
const size_t OUTPUT_PADDING_INDEX = 5;
const size_t OFFSET_X_INDEX = 6;

const int32_t DIM_LOW = 1;
const int32_t PAD_DIM_LOW = 0;
const int32_t PAD_DIM_UP = 255;
const int32_t STRIDES_DIM_HW_UP = 63;
const int32_t STRIDES_DIM_DEPTH_UP = 255;
const int32_t GROUPS_LOW = 1;
const int32_t GROUPS_UP = 65535;
const int32_t K_START_POSITION_MAX = 65535;

const std::vector<int32_t> CORE_FACTOR_910B3 = {20, 10, 5, 4, 2};
const std::vector<int32_t> CORE_FACTOR_910B2 = {24, 12, 8, 6, 4, 3, 2};

const std::map<std::string, optiling::TilingValue> TILING_DATA_MAP_B2 {
     {"1_4_15_60_60_10_15_122_122_4_4_4_2_2_2_0_0_0_0_0_0_1_1_1", // vqvae_videogpt_net_ID_3, 2268us
     {24, 1, 1, 4, 1, 3, 2, 1, 1, 3782, 240, 15, 80, 5, 5, 1, 2, 2, 1, 2, 2,
     256, 64, 80, 1, 1, 1, 1, 1, 12, 12, 1, 1, 1}},
 
     {"16_20_86_32_32_20_16_32_32_1_1_1_1_1_1_0_0_0_0_0_0_1_1_1",  // magvit_bs16_net_ID_9, 2573us
     {24, 3, 1, 8, 1, 1, 1, 6, 1, 128, 1364, 86, 256, 16, 20, 1, 2, 2, 1, 2, 2,
     128, 16, 256, 1, 1, 1, 1, 1, 30, 15, 1, 1, 1}},
 
     {"8_20_86_32_32_20_16_32_32_1_1_1_1_1_1_0_0_0_0_0_0_1_1_1",  // magvit_bs8_net_ID_9, 1289us
     {24, 3, 1, 8, 1, 1, 1, 3, 1, 128, 1364, 86, 256, 16, 20, 1, 2, 2, 1, 2, 2,
     128, 16, 256, 1, 1, 1, 1, 1, 30, 15, 1, 1, 1}},
 
     {"8_5_171_16_16_5_32_16_16_1_1_1_1_1_1_0_0_0_0_0_0_1_1_1",  // magvit_bs8_net_ID_19, 613us
     {24, 3, 1, 4, 1, 2, 1, 3, 1, 64, 2730, 171, 256, 16, 5, 1, 2, 2, 1, 2, 2,
     64, 16, 256, 1, 1, 1, 1, 1, 64, 16, 1, 1, 1}},
 
     {"8_5_32_16_16_5_86_16_16_1_1_1_1_1_1_0_0_0_0_0_0_1_1_1",  // magvit_bs8_net_ID_18, 113us
     {16, 8, 1, 1, 1, 2, 1, 1, 1, 256, 512, 32, 688, 43, 5, 1, 2, 2, 1, 2, 2,
     128, 64, 256, 1, 1, 1, 1, 1, 8, 4, 1, 1, 1}},
 
     {"4_4_4_256_256_6_4_258_258_3_3_3_1_1_1_0_0_0_0_0_0_1_1_1",  // vqvae_magvit_net_ID_21, 1992us
     {24, 2, 1, 12, 1, 1, 1, 2, 1, 5676, 64, 4, 64, 4, 6, 1, 2, 2, 1, 2, 2,
     256, 48, 64, 2, 1, 1, 1, 1, 12, 12, 1, 1, 1}},
 
     {"4_4_16_64_64_6_16_66_66_3_3_3_1_1_1_0_0_0_0_0_0_1_1_1", // vqvae_magvit_net_ID_5
     {22, 1, 1, 22, 1, 1, 1, 4, 1, 198, 256, 16, 256, 16, 6, 1, 2, 2, 1, 2, 2,
     128, 48, 256, 2, 1, 1, 1, 1, 45, 3, 1, 1, 1}},
 
     {"4_4_1_256_256_6_4_258_258_3_3_3_1_1_1_0_0_0_0_0_0_1_1_1", // vqvae_magvit_net_ID_9
     {24, 1, 1, 24, 1, 1, 1, 4, 1, 2838, 3, 1, 64, 4, 6, 1, 2, 2, 1, 2, 2,
     256, 48, 64, 3, 1, 1, 1, 1, 3, 3, 1, 1, 1}},
 
     {"16_20_16_32_32_20_16_32_32_1_1_1_1_1_1_0_0_0_0_0_0_1_1_1", // magvit_bs16_net_ID_6
     {24, 6, 1, 4, 1, 1, 1, 3, 1, 256, 256, 16, 256, 16, 20 ,1, 2, 2, 1, 2, 2,
     128, 64, 256, 1, 1, 1, 1, 1, 4, 4, 1, 1, 1}},
 
     {"4_4_86_64_64_4_16_64_64_1_1_1_1_1_1_0_0_0_0_0_0_1_1_1",
     {24, 2, 1, 3, 1, 1, 4, 2, 1, 1408, 1364, 86, 256, 16, 1, 1, 2, 2, 1, 2, 2,
     256, 48, 128, 1, 1, 1, 1, 1, 4, 2, 1, 1, 1}},
 
     {"1_62_16_66_66_120_16_128_128_4_4_4_2_2_2_3_3_3_3_3_3_1_1_1",
      {24, 1, 1, 8, 1, 1, 3, 1, 1, 2048, 256, 16, 256, 16, 40, 1, 2, 2,
       2, 2, 2, 128, 64, 128, 1, 1, 1, 1, 1, 8, 8, 1, 1, 1}},  // conv3d_transpose_videogpt_f240_h256_net_ID_1
 
     {"1_122_16_130_130_240_1_256_256_4_4_4_2_2_2_3_3_3_3_3_3_1_1_1",
      {24, 1, 1, 8, 1, 1, 3, 1, 1, 8192, 256, 16, 16, 1, 80, 1, 2, 2,
       2, 1, 2, 256, 64, 16, 1, 1, 1, 1, 1, 16, 32, 1, 1, 1}},  // conv3d_transpose_videogpt_f240_h256_net_ID_2
 
     {"1_9_16_64_64_9_16_129_129_1_3_3_1_2_2_0_0_0_0_0_0_1_1_1",
      {24, 1, 1, 8, 1, 1, 3, 1, 1, 2193, 256, 16, 256, 16, 3, 1, 2, 2,
       1, 2, 2, 256, 48, 128, 1, 1, 1, 1, 1, 12, 12, 1, 1, 1}},  // x1_13
 
     {"1_5_32_32_32_7_32_32_32_3_3_3_1_1_1_0_0_1_1_1_1_1_1_1",
      {22, 1, 1, 11, 1, 2, 1, 1, 1, 96, 512, 32, 256, 16, 7, 1, 2, 2,
       1, 2, 2, 96, 48, 256, 1, 1, 1, 1, 1, 75, 3, 1, 1, 1}},  // x1_19
 };

const std::map<std::string, optiling::TilingValue> TILING_DATA_MAP_B3{
     // key:
     // "N_Do_Co1_Ho_Wo_Di_Ci1_Hi_Wi_Dk_Hk_Wk_strideD_strideH_strideW_
     // _padFront_padBack_padUp_padDown_padLeft_padRight_dilationD_dilationH_dilationW"
     // value:
     // {coreNum, batchDim, groupDim, mDim, kDim, nDim, dDim,
     // singleCoreBatch, singleCoreGroup, singleCoreM, singleCoreCout,
     // singleCoreCout1, singleCoreCin, singleCoreCin1, singleCoreDin,
     // singleCoreHo,
     // al0Pbuffer, bl0Pbuffer, cl0Pbuffer, al1Pbuffer, bl1Pbuffer, baseM, baseK,
     // baseN, baseD, baseBatch, baseGroup,
     // stepM, stepN, stepKa, stepKb, stepBatch, stepGroup, iterateOrder}
     {"1_17_8_256_256_17_16_256_256_1_1_1_1_1_1_0_0_0_0_0_0_1_1_1",
      {20, 1, 1, 20, 1,   1,  1,   1, 1, 3328, 128, 8, 256, 16, 17, 1, 2,
       2,  1, 2, 2,  256, 64, 128, 1, 1, 1,    1,   1, 2,   2,  1,  1, 1}},  // x1_03
     {"1_17_8_256_256_19_1_256_256_3_3_3_1_1_1_0_0_1_1_1_1_1_1_1",
      {20, 1, 1, 20, 1,   1,  1,   1, 1, 3328, 128, 8, 16, 1,  19, 1, 2,
       2,  1, 1, 1,  256, 48, 128, 1, 1, 1,    1,   1, 24, 12, 1,  1, 1}},  // x1_05
     {"1_17_16_128_128_17_32_128_128_1_1_1_1_1_1_0_0_0_0_0_0_1_1_1",
      {16, 1, 1, 16, 1,   1,  1,   1, 1, 1024, 256, 16, 512, 32, 17, 1, 2,
       2,  1, 2, 2,  128, 32, 256, 1, 1, 1,    1,   1,  4,   4,  1,  1, 1}},  // x1_07
     {"1_17_16_128_128_19_32_128_128_3_3_3_1_1_1_0_0_1_1_1_1_1_1_1",
      {20, 1, 1, 10, 1,   2,  1,   1, 1, 1664, 256, 16, 256, 16, 19, 1, 2,
       2,  1, 2, 2,  256, 48, 128, 1, 1, 1,    1,   1,  12,  12, 1,  1, 1}},  // x1_08
     {"1_9_16_128_128_9_8_128_128_1_1_1_1_1_1_0_0_0_0_0_0_1_1_1",
      {19, 1, 1, 19, 1,   1,  1,   1, 1, 896, 256, 16, 128, 8, 9, 1, 2,
       2,  1, 2, 2,  256, 64, 128, 1, 1, 1,   1,   1,  4,   4, 1, 1, 1}},  // x1_11
     {"1_17_1_256_256_19_8_256_256_3_3_3_1_1_1_0_0_1_1_1_1_1_1_1",
      {20, 1, 1, 20, 1,   1,  1,   1, 1, 3328, 3, 1, 128, 8,  19, 1, 2,
       2,  1, 2, 2,  256, 48, 128, 1, 1, 1,    1, 1, 12,  12, 1,  1, 1}},  // x1_14
     {"1_5_32_32_32_5_32_32_32_1_1_1_1_1_1_0_0_0_0_0_0_1_1_1",
      {16, 1, 1, 8, 1,   2,  1,   1, 1, 128, 512, 32, 256, 16, 5, 1, 2,
       2,  1, 2, 2, 128, 64, 256, 1, 1, 1,   1,   1,  2,   2,  1, 1, 1}},  // x1_17
     {"1_5_32_64_64_5_16_64_64_1_1_1_1_1_1_0_0_0_0_0_0_1_1_1",
      {16, 1, 1, 16, 1,   1,  1,   1, 1, 256, 512, 32, 256, 16, 5, 1, 2,
       2,  1, 2, 2,  256, 64, 128, 1, 1, 1,   1,   1,  2,   2,  1, 1, 1}},  // x1_20
     {"1_5_32_64_64_7_16_64_64_3_3_3_1_1_1_0_0_1_1_1_1_1_1_1",
      {16, 1, 1, 16, 1,   1,  1,   1, 1, 256, 512, 32, 256, 16, 7, 1, 2,
       2,  1, 2, 2,  256, 48, 128, 1, 1, 1,   1,   1,  12,  12, 1, 1, 1}},  // x1_21
     {"1_5_32_64_64_5_32_64_64_1_3_3_1_1_1_0_0_1_1_1_1_1_1_1",
      {16, 1, 1, 16, 1,   1,  1,   1, 1, 256, 512, 32, 512, 32, 5, 1, 2,
       2,  1, 2, 2,  256, 48, 128, 1, 1, 1,   1,   1,  12,  12, 1, 1, 1}},  // x1_22
     {"1_5_32_64_64_7_32_64_64_3_3_3_1_1_1_0_0_1_1_1_1_1_1_1",
      {16, 1, 1, 16, 1,   1,  1,   1, 1, 256, 512, 32, 512, 32, 7, 1, 2,
       2,  1, 2, 2,  256, 48, 128, 1, 1, 1,   1,   1,  12,  12, 1, 1, 1}},  // x1_23
     {"1_9_32_128_128_9_32_128_128_1_3_3_1_1_1_0_0_1_1_1_1_1_1_1",
      {20, 1, 1, 10, 1,   2,  1,   1, 1, 1664, 512, 32, 256, 16, 9, 1, 2,
       2,  1, 2, 2,  256, 48, 128, 1, 1, 1,    1,   1,  12,  12, 1, 1, 1}},  // x1_24
     {"1_5_1_32_32_7_32_32_32_3_3_3_1_1_1_0_0_1_1_1_1_1_1_1",
      {16, 1, 1, 16, 1,  1,  1,   1, 1, 64, 8, 1, 512, 32, 7, 1, 2,
       2,  1, 1, 1,  64, 48, 256, 1, 1, 1,  1, 1, 3,   3,  1, 1, 1}},  // x1_26
     {"1_62_16_66_66_120_16_128_128_4_4_4_2_2_2_3_3_3_3_3_3_1_1_1",
      {20, 1, 1, 4, 1, 1, 5, 1, 1, 4096, 256, 16, 256, 16, 24, 1, 2, 2,
       2, 2, 2, 128, 64, 128, 1, 1, 1, 1, 1, 8, 8, 1, 1, 1}},  // conv3d_transpose_videogpt_f240_h256_net_ID_1
     {"1_122_16_130_130_240_1_256_256_4_4_4_2_2_2_3_3_3_3_3_3_1_1_1",
      {20, 1, 1, 4, 1, 1, 5, 1, 1, 16384, 256, 16, 16, 1, 48, 1, 2, 2,
       2, 1, 2, 256, 64, 16, 1, 1, 1, 1, 1, 16, 32, 1, 1, 1}},  // conv3d_transpose_videogpt_f240_h256_net_ID_2
     {"16_20_16_32_32_22_16_34_34_3_3_3_1_1_1_0_0_0_0_0_0_1_1_1",
      {20, 2, 1, 5, 1,   1,  2,   8, 1, 238, 256, 16, 256, 16,  11, 1, 2,
       2,  1,  2, 2, 128, 48, 256, 1, 1, 1,    1,   1, 24,  6, 1,  1, 1}},  // conv3d_dx_magvit_bs16_net_ID_7
     {"16_20_8_64_64_22_8_66_66_3_3_3_1_1_1_0_0_0_0_0_0_1_1_1",
      {20, 2, 1, 5, 1,   1,  2,   8, 1, 924, 128, 8, 128, 8, 11, 1, 2,
       2,  1,  2, 2, 256, 48, 128, 1, 1, 1,    1,   1,  12,  12, 1,  1, 1}},  // conv3d_dx_magvit_bs16_net_ID_5
     {"1_60_16_64_64_122_16_130_130_4_4_4_2_2_2_0_0_0_0_0_0_1_1_1",
      {20, 1, 1, 5, 1,   2,  2,   1, 1, 3380, 256, 16, 128, 8, 61, 1, 2,
       2,  1, 2, 2, 256, 64, 128, 1, 1, 1,    1,   1,  12,  8, 1,  1, 1}},  // conv3d_dx_x1_videogpt_f240_h256_net_ID_6
     {"1_60_8_64_64_62_16_66_66_3_3_3_1_1_1_0_0_0_0_0_0_1_1_1",
      {20, 1, 1, 5, 1,   2,  2,   1, 1, 924, 128, 8, 128, 8, 31, 1, 2,
       2,  1, 2, 2, 256, 48, 128, 1, 1, 1,   1,   1, 6,   3, 1,  1, 1}},  // conv3d_dx_x1_videogpt_f240_h256_net_ID_2
     {"1_5_32_32_32_7_32_32_32_3_3_3_1_1_1_0_0_1_1_1_1_1_1_1",
      {16, 1, 1, 4, 1,   4,  1,   1, 1, 256, 512, 32, 128, 8,  7, 1, 2,
       2,  1, 2, 2, 256, 48, 128, 1, 1, 1,   1,   1,  12,  12, 1, 1, 1}},  // x1_19
     {"1_9_32_64_64_11_32_64_64_3_3_3_1_1_1_0_0_1_1_1_1_1_1_1",
      {20, 1, 1, 10, 1,   2,  1,   1, 1, 448, 512, 32, 256, 16, 11, 1, 2,
       2,  1, 2, 2,  256, 48, 128, 1, 1, 1,   1,   1,  12,  12, 1,  1, 1}},  // x1_25
     {"16_20_86_32_32_20_16_32_32_1_1_1_1_1_1_0_0_0_0_0_0_1_1_1",
      {20, 1, 1, 1, 1,   1,  20,  16, 1, 1024, 1364, 86, 256, 16, 1, 1, 2,
       2,  1, 1, 1, 128, 64, 256, 1,  1, 1,    1,    1,  3,   3,  1, 1, 1}},  // conv3d_dx_magvit_bs16_net_ID_9
     {"16_20_171_16_16_20_32_16_16_1_1_1_1_1_1_0_0_0_0_0_0_1_1_1",
      {20, 1, 1, 1, 1,   1,  20,  16, 1, 256, 2730, 171, 512, 32, 1, 1, 2,
       2,  1, 2, 2, 128, 64, 256, 1,  1, 1,   1,    1,   3,   3,  1, 1, 1}},  // conv3d_dx_magvit_bs16_net_ID_13
     {"16_5_171_16_16_5_32_16_16_1_1_1_1_1_1_0_0_0_0_0_0_1_1_1",
      {20, 4, 1, 1, 1,   1,  5,   4, 1, 256, 2730, 171, 512, 32, 1, 1, 2,
       2,  1, 2, 2, 128, 64, 256, 1, 1, 1,   1,    1,   3,   3,  1, 1, 1}},  // conv3d_dx_magvit_bs16_net_ID_19
     {"4_4_171_32_32_4_32_32_32_1_1_1_1_1_1_0_0_0_0_0_0_1_1_1",
      {16, 4, 1, 2, 1,   2,  1,   1, 1, 512, 2730, 171, 256, 16, 4, 1, 2,
       2,  1, 2, 2, 256, 16, 128, 1, 1, 1,   1,    1,   4,   4,  1, 1, 1}},
     {"1_5_32_32_32_7_1_32_32_3_3_3_1_1_1_0_0_1_1_1_1_1_1_1",
      {14, 1, 1, 2, 1,   1,  7,  1, 1, 512, 512, 32, 4,  1,  1, 1, 2,
       2,  1, 2, 2, 256, 48, 16, 1, 1, 1,   1,   1,  48, 24, 1, 1, 1}},  // x1_16
     {"1_25_20_40_64_25_20_40_64_3_1_1_1_1_1_1_1_0_0_0_0_1_1_1",
      {20, 1, 1, 4, 1,   1,  5,   1, 1, 640, 320, 20, 320, 20, 5, 1, 2,
       2,  1, 2, 2, 256, 64, 128, 1, 1, 1,   1,   1,  4,   2,  1, 1, 1}},
     {"4_4_86_64_64_4_16_64_64_1_1_1_1_1_1_0_0_0_0_0_0_1_1_1",
      {16, 4, 1, 1, 1,   1,  4,   1, 1, 4096, 1364, 86, 256, 16, 1, 1, 2,
       2,  1, 2, 2, 256, 48, 128, 1, 1, 1,    1,    1,  4,   2,  1, 1, 1}},
     {"8_20_86_32_32_20_16_32_32_1_1_1_1_1_1_0_0_0_0_0_0_1_1_1",
      {20, 1, 1, 1, 1,   1,  20,  8, 1, 1024, 1364, 86, 256, 16, 1, 1, 2,
       2,  1, 2, 2, 256, 48, 128, 1, 1, 1,    1,    1,  4,   4,  1, 1, 1}},
     {"1_17_16_128_128_19_16_128_128_3_3_3_1_1_1_0_0_1_1_1_1_1_1_1",
      {20, 1, 1, 10, 1,   2,  1,   1, 1, 1664, 256, 16, 128, 8,  19, 1, 2,
       2,  1, 2, 2, 256, 48, 128, 1, 1, 1,   1,   1,  21,  3, 1, 1, 1}},  // x1_06
     {"1_9_16_128_128_11_16_128_128_3_3_3_1_1_1_0_0_1_1_1_1_1_1_1",
      {20, 1, 1, 10, 1,   2,  1,   1, 1, 1664, 256, 16, 128, 8, 11, 1, 2,
       2,  1, 2, 2, 256, 48, 128, 1, 1, 1,   1,   1,  21,  3, 1, 1, 1}},  // x1_12
     {"4_5_32_32_32_9_32_64_64_3_3_3_2_2_2_1_1_1_1_1_1_1_1_1",
      {20, 4, 1, 5, 1,   1,  1,  1, 1, 832, 512, 32, 512, 32, 9, 1, 2,
       2,  1, 2, 2, 208, 48, 128, 1, 1, 1,   1,   1,  12, 12, 1,  1, 1}},  // conv3d_dx_AIGC_net_ID_11
     {"16_20_1_128_128_22_4_130_130_3_3_3_1_1_1_0_0_0_0_0_0_1_1_1",
      {20, 2, 1, 5, 1,   1,  2,  8, 1, 3380, 16, 1, 64, 4, 11, 1, 2,
       2,  1, 1, 1, 336, 48, 64, 1, 1, 1,    1, 1, 3, 3, 1,  1, 1}},  // conv3d_dx_magvit_bs16_net_ID_1
     {"16_20_4_128_128_20_4_128_128_1_1_1_1_1_1_0_0_0_0_0_0_1_1_1",
      {20, 1, 1, 1, 1,   1,  20,  16, 1, 16384, 64, 4, 64, 4, 1, 1, 2,
       2,  1, 1, 1, 512, 32, 64, 1, 1, 1,    1, 1, 2, 2, 1,  1, 1}},  // conv3d_dx_magvit_bs16_net_ID_2
     {"16_20_4_128_128_22_4_130_130_3_3_3_1_1_1_0_0_0_0_0_0_1_1_1",
      {20, 2, 1, 5, 1,   1,  2,  8, 1, 3380, 64, 4, 64, 4, 11, 1, 2,
       2,  1, 2, 1, 336, 48, 64, 1,  1, 1,    1,    1,  6,   12,  1, 1, 1}},  // conv3d_dx_magvit_bs16_net_ID_3
     {"16_20_8_64_64_20_4_128_128_1_3_3_1_2_2_0_0_1_1_1_1_1_1_1",
      {20, 1, 1, 1, 1,   1,  20,  16, 1, 16384, 128, 8, 64, 4, 1, 1, 2,
       2,  1, 2, 1, 336, 48, 64, 1,  1, 1,    1,    1,  6,   24,  1, 1, 1}},  // conv3d_dx_magvit_bs16_net_ID_110
 };

const std::map<std::string, optiling::TilingValue> FP32_TILING_DATA_MAP_B2{
     {"2_4_64_18_18_3_32_17_17_4_4_4_1_1_1_2_2_2_2_2_2_1_1_1",
      {24, 2, 1, 1, 1, 4, 3, 1, 1, 289, 512, 64, 64, 8, 1, 1, 2, 2, 1, 2, 2, 256,
       32, 64, 3, 1, 1, 1, 1, 48, 8, 1, 1, 1}  // TX_AI_net_ID_16
     }
 };

const std::map<std::string, optiling::TilingValue> FP32_TILING_DATA_MAP_B3{
    {"2_4_64_18_18_3_32_17_17_4_4_4_1_1_1_2_2_2_2_2_2_1_1_1",
     {12, 2, 1, 1, 1, 2, 3, 1, 1, 289, 512, 64, 128, 16, 1, 1, 2, 2, 1, 2, 2, 256,
      32, 128, 3, 1, 1, 1, 1, 48, 8, 1, 1, 1}}  // TX_AI_net_ID_16
};

const std::map<std::string, optiling::TilingValue> FP32_BASIC_BLOCK_MAP_B3 {
    {"4_6_8_112_112_17_1_229_229_7_7_7_2_2_2_0_0_0_0_0_0_1_1_1",
     {20, 1, 1, 1, 1, 1, 1, 1, 1, 10534, 64, 8, 8, 1, 1, 1,
      2, 2, 1, 2, 2, 1024, 8, 16, 1, 1, 1, 1, 1, 49, 196, 1, 1, 1}},  // ASKJ_050
};

const std::map<std::string, optiling::TilingValue> FP32_BASIC_BLOCK_MAP_B2 {
    {"4_6_8_112_112_17_1_229_229_7_7_7_2_2_2_0_0_0_0_0_0_1_1_1",
     {24, 1, 1, 1, 1, 1, 1, 1, 1, 10534, 64, 8, 8, 1, 1, 1,
      2, 2, 1, 2, 2, 1024, 8, 16, 1, 1, 1, 1, 1, 49, 196, 1, 1, 1}},  // ASKJ_050
};

using Conv3DBackpropInputV2CompileInfo = optiling::Conv3DBackPropInputCompileInfo;

#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
#define OP_LOGW(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__);  \
    std::printf("\n")
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)       \
    if ((ptr) == nullptr) {                               \
        std::printf("nullptr error!");                  \
        return ge::GRAPH_FAILED;                        \
    }

inline const char* get_cstr(const std::string& str) {
    return str.c_str();
  }

#define OPPROTO_SUBMOD_NAME "OP_COMMON"

class OpLog {
public:
  static uint64_t GetTid() {
        #ifdef __GNUC__
            const uint64_t tid = static_cast<uint64_t>(syscall(__NR_gettid));
        #else
            const uint64_t tid = static_cast<uint64_t>(GetCurrentThreadId());
        #endif
            return tid;
  }
};

#define OpLogSub(moduleId, level, op_info, fmt, ...)                                                                   \
    do {                                                                                                                \
      if (AlogCheckDebugLevel(static_cast<int>(moduleId), level) == 1) {                                                \
                  AlogRecord(static_cast<int>(moduleId), DLOG_TYPE_DEBUG, level,                                        \
                      "[%s:%d][%s][%s][%" PRIu64 "] OpName:[%s] " #fmt,                                                 \
                      __FILE__, __LINE__, OPPROTO_SUBMOD_NAME,                                                          \
                      __FUNCTION__, OpLog::GetTid(), get_cstr(op_info), ##__VA_ARGS__);                                 \
        }                                                                                                               \
  } while (0)

#define MSG_LENGTH 1024
constexpr const int OP_MAX_LOG_SIZE = 16000;
constexpr const int OP_MSG_HEADER_LEN = 200;

int32_t CheckLogLevel1(int32_t moduleId, int32_t logLevel) {
    (void)moduleId;
    (void)logLevel;
    return 1;
}

#define OP_LOG_FULL(level, opname, format, ...)                              \
do {                                                                         \
  if (0 == CheckLogLevel1(OP, level)) {                                       \
    break;                                                                   \
  }                                                                          \
  char msgbufxyz[OP_MAX_LOG_SIZE];                                           \
  size_t msgmaxlen = (MSG_LENGTH - OP_MSG_HEADER_LEN);                       \
  int rettmp = snprintf_s(msgbufxyz, sizeof(msgbufxyz), sizeof(msgbufxyz) - 1, format, ##__VA_ARGS__); \
  if (rettmp == -1) {                                                        \
    msgbufxyz[sizeof(msgbufxyz) - 1] = '\0';                                 \
  }                                                                          \
  size_t msglength = std::strlen(msgbufxyz);                                 \
  if (msglength < msgmaxlen) {                                               \
    OpLogSub(OP, level, opname, "%s", msgbufxyz);                            \
    break;                                                                   \
  }                                                                          \
  char *msgchunkbegin = msgbufxyz;                                           \
  char *msgchunkend = nullptr;                                               \
  while (msgchunkbegin < msgbufxyz + msglength) {                            \
    if (msgchunkbegin[0] == '\n') {                                          \
      OpLogSub(OP, level, opname, "");                                       \
      msgchunkbegin += 1;                                                    \
      continue;                                                              \
    }                                                                        \
    msgchunkend = std::strchr(msgchunkbegin, '\n');                          \
    if (msgchunkend == nullptr) {                                            \
      msgchunkend = msgchunkbegin + std::strlen(msgchunkbegin);              \
    }                                                                        \
    while (msgchunkend > msgchunkbegin) {                                    \
      std::string msgchunk(msgchunkbegin, std::min(msgmaxlen, static_cast<size_t>(msgchunkend - msgchunkbegin))); \
      OpLogSub(OP, level, opname, "%s", msgchunk.c_str());                   \
      msgchunkbegin += msgchunk.size();                                      \
    }                                                                        \
    msgchunkbegin += 1;                                                      \
  }                                                                          \
} while (0)
}  // namespace

 namespace optiling {
 static inline bool CheckRange(int32_t value, int32_t valueLow, int32_t valueUp)
 {
     if (value < valueLow || value > valueUp) {
         return false;
     }
     return true;
 }
 
 static inline uint32_t GetMaxDivisor(uint32_t a, uint32_t b, uint32_t step)
 {
     while (b >= step) {
         if (a % b == 0) {
             return b;
         }
         b -= step;
     }
     return 0;
 }
 
 bool Conv3DBackpropInputV2Tiling::CheckL0Size(uint32_t baseM, uint32_t baseN, uint32_t baseK, uint32_t byteSize)
 {
     int64_t l0aSize = static_cast<int64_t>(baseM) * baseK * byteSize * DB_ON;
     int64_t l0bSize = static_cast<int64_t>(baseN) * baseK * byteSize * DB_ON;
     if (byteSize == FP32_DATA_SIZE && runInfo_.kernel_h * runInfo_.kernel_w > 1) {
         // fp32场景下，当HkWk>1时，L0B需要预留额外空间进行转置处理
         l0bSize = static_cast<int64_t>(baseN) * (baseK + blockSize_) * byteSize * DB_ON;
     }
 
     return l0aSize <= L0_SIZE && l0bSize <= L0_SIZE;
 }
 
 void Conv3DBackpropInputV2Tiling::AlignCout1(uint32_t &cout1A, uint32_t &cout1B, bool adaptFP32)
 {
     if (cout1A == cout1B) {
         return;
     } else if (cout1B > cout1A) {
         cout1A = GetMaxDivisor(cout1B, cout1A, 1);
         return;
     }
 
     if (!adaptFP32) {
         cout1B = GetMaxDivisor(cout1A, cout1B, 1);
         return;
     }
 
     uint32_t tempCout1A = cout1A;
     while (tempCout1A % cout1B > 0) {
         tempCout1A--;
     }
     uint64_t cout1AB = static_cast<uint64_t>(tempCout1A) * cout1B;
     uint32_t step = BLOCK_CUBE / blockSize_;
     uint32_t tempCout1B = GetMaxDivisor(cout1A, cout1B, step);
     if (tempCout1B == 0) {
         cout1A = tempCout1A;
         return;
     }
 
     uint64_t cout1ABSmallerB = tempCout1B * cout1A;
     if (cout1ABSmallerB > cout1AB) {
         cout1B = tempCout1B;
     } else {
         cout1A = tempCout1A;
     }
 }
 
 void Conv3DBackpropInputV2Tiling::Reset()
 {
     tilingData_.SetDataPtr(context_->GetRawTilingData()->GetData());
     opName_ = nullptr;
 }
 
 ge::graphStatus Conv3DBackpropInputV2Tiling::GetPlatformInfo()
 {
     return ge::GRAPH_SUCCESS;
 }
 
 ge::graphStatus Conv3DBackpropInputV2Tiling::GetShapeAttrsInfo()
 {
     opName_ = context_->GetNodeName();
     OP_LOG_FULL(DLOG_INFO, opName_, "TilingContext: %s", optiling::DebugTilingContext(context_).c_str());
     OP_TILING_CHECK(!AnalyzeDtype(), CUBE_INNER_ERR_REPORT(opName_, "fail to analyze context info"),
                     return ge::GRAPH_FAILED);
     return ge::GRAPH_SUCCESS;
 }
 
 bool Conv3DBackpropInputV2Tiling::IsCapable()
 {
     return true;
 }
 
 ge::graphStatus Conv3DBackpropInputV2Tiling::DoOpTiling()
 {
     if (!GetTbeTiling()) {
         OP_LOGE(context_->GetNodeName(), "GetTbeTiling failed");
         return ge::GRAPH_FAILED;
     }
 
     return ge::GRAPH_SUCCESS;
 }
 
 ge::graphStatus Conv3DBackpropInputV2Tiling::DoLibApiTiling()
 {
     SetDxTilingFromTbeTiling();
     PrintTilingData();
     return ge::GRAPH_SUCCESS;
 }
 
 uint64_t Conv3DBackpropInputV2Tiling::GetTilingKey() const {
     return RecursiveSum(loadB2Condition_, enableKernelSplit_, useBasicBlock_);
 }
 
 ge::graphStatus Conv3DBackpropInputV2Tiling::GetWorkspaceSize()
 {
     size_t *workspaces = context_->GetWorkspaceSizes(1);
     OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
     // 框架预留16M
     workspaces[0] = WORKSIZE;
 
     // 仅在支持TBE的架构上，dx融合输出transdata时，需要分配临时GM空间
     OP_TILING_CHECK(context_->GetOutputDesc(Y_INDEX) == nullptr,
         CUBE_INNER_ERR_REPORT(opName_, "failed to get y tensor desc from context"),
         return ge::GRAPH_FAILED
     );
     if (context_->GetOutputDesc(Y_INDEX)->GetStorageFormat() == ge::FORMAT_NCDHW) {
         optiling::PlatformInfo &plaformInstance = optiling::PlatformInfo::GetInstance();
         size_t singleCoreUsrSpaceSize = plaformInstance.l0c_size;
         size_t usrSpaceSize = plaformInstance.core_num * singleCoreUsrSpaceSize;
         workspaces[0] += usrSpaceSize;
         OP_LOGD(opName_, "output storage format is FORMAT_NCDHW, usrSpaceSize = %ld", usrSpaceSize);
     }
 
     return ge::GRAPH_SUCCESS;
 }
 
 ge::graphStatus Conv3DBackpropInputV2Tiling::PostTiling()
 {
     OP_LOGD(opName_, "final tiling data size: %zu", tilingData_.GetDataSize());
 
     OP_TILING_CHECK(
         tilingData_.GetDataSize() % sizeof(uint64_t) != 0,
         CUBE_INNER_ERR_REPORT(opName_, "tiling data size[%zu] not aligned to 8",
                                 tilingData_.GetDataSize()),
         return ge::GRAPH_FAILED);
     context_->SetBlockDim(tilingData_.params.get_coreNum());
     context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
 
     return ge::GRAPH_SUCCESS;
 }
 
 bool Conv3DBackpropInputV2Tiling::AnalyzeDtype() const
 {
     size_t aMatrixesIndex = OUTPUT_BP_INDEX;
     size_t bMatrixesIndex = FILTER_INDEX;
 
     if (opType_ == cachetiling::kConv3DTranspose) {
         aMatrixesIndex = FILTER_INDEX;
         bMatrixesIndex = OUTPUT_BP_INDEX;
     }
     OP_TILING_CHECK(
         context_->GetInputDesc(aMatrixesIndex) == nullptr || context_->GetInputDesc(bMatrixesIndex) == nullptr
         || context_->GetOutputDesc(Y_INDEX) == nullptr,
         CUBE_INNER_ERR_REPORT(opName_, "failed to get out_backprop/filter/y tensor desc from context"),
         return false
     );
     ge::DataType aDtype = context_->GetInputDesc(aMatrixesIndex)->GetDataType();
     ge::DataType bDtype = context_->GetInputDesc(bMatrixesIndex)->GetDataType();
 
     ge::DataType cDtype = context_->GetOutputDesc(Y_INDEX)->GetDataType();
 
     OP_TILING_CHECK(!((aDtype == ge::DT_BF16 && bDtype == ge::DT_BF16 && cDtype == ge::DT_BF16) ||
                     (aDtype == ge::DT_FLOAT && bDtype == ge::DT_FLOAT && cDtype == ge::DT_FLOAT) ||
                     (aDtype == ge::DT_FLOAT16 && bDtype == ge::DT_FLOAT16 && cDtype == ge::DT_FLOAT16)),
                     CUBE_INNER_ERR_REPORT(
                         opName_, "x/weight/y only support DT_BF16/DT_FLOAT16/DT_FLOAT dtype, get actual aDtype[%s] bDtype[%s] cDtype[%s]",
                         ge::TypeUtils::DataTypeToAscendString(aDtype).GetString(),
                         ge::TypeUtils::DataTypeToAscendString(bDtype).GetString(),
                         ge::TypeUtils::DataTypeToAscendString(cDtype).GetString()),
                     return false);
     return true;
 }
 
 bool Conv3DBackpropInputV2Tiling::CheckAttrs(Conv3DDxParas &conv3ddxParas)
 {
     OP_TILING_CHECK(
         conv3ddxParas.tiling_param.stride_d > conv3ddxParas.tiling_param.b_shape.d,
         CUBE_INNER_ERR_REPORT(opName_, "cannot support stride_d: %d > kernel_d: %ld",
                                         conv3ddxParas.tiling_param.stride_d, conv3ddxParas.tiling_param.b_shape.d),
         return false);
 
     OP_TILING_CHECK(CheckRange(conv3ddxParas.tiling_param.stride_h, DIM_LOW, STRIDES_DIM_HW_UP) == false,
                     CUBE_INNER_ERR_REPORT(opName_, "stride_h: %d is invaild, support range [%d, %d]",
                                                     conv3ddxParas.tiling_param.stride_h, DIM_LOW, STRIDES_DIM_HW_UP),
                     return false);
 
     OP_TILING_CHECK(CheckRange(conv3ddxParas.tiling_param.stride_w, DIM_LOW, STRIDES_DIM_HW_UP) == false,
                     CUBE_INNER_ERR_REPORT(opName_, "stride_w: %d is invaild, support range [%d, %d]",
                                                     conv3ddxParas.tiling_param.stride_w, DIM_LOW, STRIDES_DIM_HW_UP),
                     return false);
 
     OP_TILING_CHECK(CheckRange(conv3ddxParas.tiling_param.stride_d, DIM_LOW, STRIDES_DIM_DEPTH_UP) == false,
                     CUBE_INNER_ERR_REPORT(opName_, "stride_d: %d is invaild, support range [%d, %d]",
                                                     conv3ddxParas.tiling_param.stride_d, DIM_LOW, STRIDES_DIM_DEPTH_UP),
                     return false);
     uint64_t curL0CDstStride = static_cast<uint64_t>(conv3ddxParas.tiling_param.c_shape.h) *
         conv3ddxParas.tiling_param.c_shape.w;
     OP_TILING_CHECK(curL0CDstStride > UINT32_MAX,
         CUBE_INNER_ERR_REPORT(opName_, "cannot support hi * wi=%lu over %u", curL0CDstStride, UINT32_MAX),
         return false);
 
     OP_TILING_CHECK(
         CheckRange(conv3ddxParas.tiling_param.groups, GROUPS_LOW, GROUPS_UP) == false,
         CUBE_INNER_ERR_REPORT(opName_, "only support groups(%d) in range [%d, %d]",
             conv3ddxParas.tiling_param.groups, GROUPS_LOW, GROUPS_UP),
         return false);
     return true;
 }
 
 bool Conv3DBackpropInputV2Tiling::CheckPadRange(Conv3DDxParas &conv3ddxParas)
 {
     int32_t padHDimUp = std::min(PAD_DIM_UP, static_cast<int32_t>(conv3ddxParas.tiling_param.b_shape.d - 1));
     int32_t padUDimUp = std::min(PAD_DIM_UP, static_cast<int32_t>(conv3ddxParas.tiling_param.b_shape.h - 1));
     int32_t padLDimUp = std::min(PAD_DIM_UP, static_cast<int32_t>(conv3ddxParas.tiling_param.b_shape.w - 1));
     OP_TILING_CHECK(CheckRange(conv3ddxParas.tiling_param.pad_h, PAD_DIM_LOW, padHDimUp) == false,
                     CUBE_INNER_ERR_REPORT(opName_, "pad head: %d invaild, it should be in [%d, %d]",
                                           conv3ddxParas.tiling_param.pad_h, PAD_DIM_LOW, padHDimUp),
                     return false);
     OP_TILING_CHECK(CheckRange(conv3ddxParas.tiling_param.pad_t, PAD_DIM_LOW, padHDimUp) == false,
                     CUBE_INNER_ERR_REPORT(opName_, "pad tail: %d invaild, it should be in [%d, %d]",
                                           conv3ddxParas.tiling_param.pad_t, PAD_DIM_LOW, padHDimUp),
                     return false);
     OP_TILING_CHECK(CheckRange(conv3ddxParas.tiling_param.pad_u, PAD_DIM_LOW, padUDimUp) == false,
                     CUBE_INNER_ERR_REPORT(opName_, "pad up: %d invaild, it should be in [%d, %d]",
                                           conv3ddxParas.tiling_param.pad_u, PAD_DIM_LOW, padUDimUp),
                     return false);
     OP_TILING_CHECK(CheckRange(conv3ddxParas.tiling_param.pad_d, PAD_DIM_LOW, padUDimUp) == false,
                     CUBE_INNER_ERR_REPORT(opName_, "pad down: %d invaild, it should be in [%d, %d]",
                                           conv3ddxParas.tiling_param.pad_d, PAD_DIM_LOW, padUDimUp),
                     return false);
     OP_TILING_CHECK(CheckRange(conv3ddxParas.tiling_param.pad_l, PAD_DIM_LOW, padLDimUp) == false,
                     CUBE_INNER_ERR_REPORT(opName_, "pad left: %d invaild, it should be in [%d, %d]",
                                           conv3ddxParas.tiling_param.pad_l, PAD_DIM_LOW, padLDimUp),
                     return false);
     OP_TILING_CHECK(CheckRange(conv3ddxParas.tiling_param.pad_r, PAD_DIM_LOW, padLDimUp) == false,
                     CUBE_INNER_ERR_REPORT(opName_, "pad right: %d invaild, it should be in [%d, %d]",
                                           conv3ddxParas.tiling_param.pad_r, PAD_DIM_LOW, padLDimUp),
                     return false);
     return true;
 }
 
 bool Conv3DBackpropInputV2Tiling::CheckTranspose() const
 {
     auto biasShape = context_->GetOptionalInputShape(BAIS_INDEX);
     auto offsetWShape = context_->GetOptionalInputShape(OFFSET_W_INDEX);
 
     auto attrs = context_->GetAttrs();
     OP_TILING_CHECK(attrs == nullptr, CUBE_INNER_ERR_REPORT(opName_, "attrs is null"), return false);
 
     const auto offsetX = attrs->GetAttrPointer<int64_t>(OFFSET_X_INDEX);
     auto outputPadding = attrs->GetAttrPointer<gert::ContinuousVector>(OUTPUT_PADDING_INDEX);
 
     if (outputPadding != nullptr) {
         OP_TILING_CHECK(outputPadding->GetData() == nullptr,
                     CUBE_INNER_ERR_REPORT(opName_, "output_padding GetData is null"), return false);
         OP_TILING_CHECK(outputPadding->GetSize() != OUTPUT_PADDING_DIM,
                     CUBE_INNER_ERR_REPORT(opName_, "the output padding:%zu is invalid, it should be 5d",
                     outputPadding->GetSize()), return false);
     }
     OP_TILING_CHECK(offsetX != nullptr && *offsetX != 0,
                     CUBE_INNER_ERR_REPORT(opName_, "cannot support offset_x attribute parameters"),
                     return false);
     OP_TILING_CHECK(offsetWShape != nullptr && offsetWShape->GetStorageShape().GetShapeSize() != 0,
                     CUBE_INNER_ERR_REPORT(opName_,"cannot support offset_w input parameters"),
                     return false);
     OP_TILING_CHECK(biasShape != nullptr && biasShape->GetStorageShape().GetShapeSize() != 0,
                     CUBE_INNER_ERR_REPORT(opName_, "cannot support bias"), return false);
 
     return true;
 }
 
 void Conv3DBackpropInputV2Tiling::SetInitOutput(const Conv3DDxParas &conv3ddxParas)
 {
     int32_t doModulo = (conv3ddxParas.fmap_d_padding - conv3ddxParas.tiling_param.filter_d_dilation) %
                         conv3ddxParas.tiling_param.stride_d;
     int32_t hoModulo = (conv3ddxParas.fmap_h_padding - conv3ddxParas.tiling_param.filter_h_dilation) %
                         conv3ddxParas.tiling_param.stride_h;
     if (doModulo > conv3ddxParas.tiling_param.pad_t ||
         hoModulo > conv3ddxParas.tiling_param.pad_d ||
         runInfo_.stride_h > runInfo_.kernel_h ||
         (opType_ == cachetiling::kConv3DTranspose && (conv3ddxParas.output_padding_d > 0 || conv3ddxParas.output_padding_h > 0)) ||
         conv3ddxParas.tiling_param.dilation_d > 1) {
         // 1 is init output with l0C, 2 is init output with l1, defualt is 0 means not init output
         initOutputFlag = 1;
     }
 }
 
 // currently only supports conv3d_transpose_videogpt_f240_h256_net_ID_1 and conv3d_transpose_videogpt_f240_h256_net_ID_2
 bool Conv3DBackpropInputV2Tiling::CheckKernelSplitEnable() const
 {
     return ((runInfo_.real_g == 1 && runInfo_.batch_n == 1 && runInfo_.dedy_d == 62 && runInfo_.dedy_cout1 == 16 && runInfo_.dedy_h == 66 &&
         runInfo_.dedy_w == 66 && runInfo_.dedx_d == 120 && runInfo_.dedx_cin1 == 16 && runInfo_.dedx_h == 128 &&
         runInfo_.dedx_w == 128 && runInfo_.kernel_d == 4 && runInfo_.kernel_h == 4 && runInfo_.kernel_w == 4 &&
         runInfo_.stride_d == 2 && runInfo_.stride_h == 2 && runInfo_.stride_w == 2 && runInfo_.pad_h == 3 &&
         runInfo_.pad_t == 3 && runInfo_.pad_u == 3 && runInfo_.pad_d == 3 && runInfo_.pad_l == 3 &&
         runInfo_.pad_r == 3 && runInfo_.dilation_d == 1 && runInfo_.dilation_h == 1 && runInfo_.dilation_w == 1) ||
         (runInfo_.batch_n == 1 && runInfo_.dedy_d == 122 && runInfo_.dedy_cout1 == 16 && runInfo_.dedy_h == 130 &&
         runInfo_.dedy_w == 130 && runInfo_.dedx_d == 240 && runInfo_.dedx_cin1 == 1 && runInfo_.dedx_h == 256 &&
         runInfo_.dedx_w == 256 && runInfo_.kernel_d == 4 && runInfo_.kernel_h == 4 && runInfo_.kernel_w == 4 &&
         runInfo_.stride_d == 2 && runInfo_.stride_h == 2 && runInfo_.stride_w == 2 && runInfo_.pad_h == 3 &&
         runInfo_.pad_t == 3 && runInfo_.pad_u == 3 && runInfo_.pad_d == 3 && runInfo_.pad_l == 3 &&
         runInfo_.pad_r == 3 && runInfo_.dilation_d == 1 && runInfo_.dilation_h == 1 && runInfo_.dilation_w == 1));
 }
 
 bool Conv3DBackpropInputV2Tiling::GetImplMode(Conv3DDxParas &conv3ddxParas)
 {
     auto attrs = context_->GetAttrs();
     OP_TILING_CHECK(attrs == nullptr, CUBE_INNER_ERR_REPORT(context_, "failed to get runtime attrs"), return false);
     auto inputDesc = context_->GetInputDesc(OUTPUT_BP_INDEX);
     OP_TILING_CHECK(inputDesc == nullptr,
         CUBE_INNER_ERR_REPORT(opName_, "failed to get out_backprop tensor desc from context"),
         return false
     );
     ge::DataType aDtype = inputDesc->GetDataType();
     size_t idx = 0;
     if (opType_ == cachetiling::kConv3DTranspose) {
         idx = 8; // transpose impl mode idx is 8
     } else {
         idx = 6; // dx impl mode idx is 6
     }
     if (aDtype == ge::DT_FLOAT && idx < attrs->GetAttrNum()) {
         const int32_t *precisionMode = attrs->GetAttrPointer<int32_t>(idx);
         if (precisionMode != nullptr && *precisionMode != -1) {
             conv3ddxParas.tiling_param.hf32_flag = (*precisionMode & 0x40) ? 1:0;
         } else {
             OP_LOGW(opName_, "impl mode is not support, so we set hf32 flag with 0 as default");
             conv3ddxParas.tiling_param.hf32_flag = 0;
         }
     }
     return true;
 }
 
 bool Conv3DBackpropInputV2Tiling::GetTbeTiling()
 {
     auto compileInfoPtr =
         reinterpret_cast<const Conv3DBackpropInputV2CompileInfo *>(context_->GetCompileInfo());
     OP_TILING_CHECK(compileInfoPtr == nullptr, CUBE_INNER_ERR_REPORT("Conv3DBackpropInputV2", "compile_info is null"),
                     return false);
     Conv3DDxParas conv3ddxParas(cachetiling::kConv3DBackpropInput);
     cachetiling::Conv3DBpInputTilingParam &conv3ddxTilingParam = conv3ddxParas.tiling_param;
     conv3ddxTilingParam.binary_mode = 1;
     conv3ddxTilingParam.platform_info.SetRuntimePlatformInfo(*compileInfoPtr);
 
     OP_TILING_CHECK(!Conv3DBackpropInputParseFunc(context_, opType_, conv3ddxParas, true),
                     CUBE_INNER_ERR_REPORT(context_, "failed to parse context"), return false);
     OP_TILING_CHECK(!GetImplMode(conv3ddxParas),
                     CUBE_INNER_ERR_REPORT(context_, "failed to get impl mode"), return false);
     blockSize_ = BYTE_BLOCK / conv3ddxTilingParam.a_dtype_bytes;
     dtypeByte_ = conv3ddxTilingParam.a_dtype_bytes;
 
     if (!CheckCalPads(context_, conv3ddxParas) || !CheckParams(conv3ddxParas, context_) ||
         !CheckAttrs(conv3ddxParas) || !CheckPadRange(conv3ddxParas)) {
         OP_LOGE(context_, "params is invalid");
         return false;
     }
     if (opType_ == cachetiling::kConv3DTranspose && !CheckTranspose()) {
         OP_LOGE(context_, "params is invalid");
         return false;
     }
 
     if (!cachetiling::GetTiling<cachetiling::Conv3DBpInputTilingParam, cachetiling::Conv3DBpInputTiling,
                                 cachetiling::Conv3DBpInputHashParam, cachetiling::Conv3DBpInputHashItem>(
             conv3ddxTilingParam, tbeTiling_, context_, opType_)) {
         OP_LOGE(opName_, "GetTiling interface failed");
         return false;
     }
 
     OP_TILING_CHECK(!SetRunInfoConv3DDx(conv3ddxParas, tbeTiling_, runInfo_, context_),
                     CUBE_INNER_ERR_REPORT(context_, "failed to set run info"), return false);
     if (runInfo_.real_g == 1) {
         runInfo_.dedy_cout1_g = runInfo_.dedy_cout1;
         runInfo_.dedx_cin1_g = runInfo_.dedx_cin1;
     }
     SetInitOutput(conv3ddxParas);
     return true;
 }
 
 int32_t Conv3DBackpropInputV2Tiling::GetDimFactor(const int64_t& value, const std::vector<int32_t>& factorLits) const
 {
     int32_t dimFactor = 1;
     for (uint32_t i = 0; i < factorLits.size();i++) {
         if (value % factorLits[i] == 0) {
             dimFactor = factorLits[i];
             break;
         }
     }
     return dimFactor;
 }
 
 void Conv3DBackpropInputV2Tiling::GetCoreDim(int32_t& batchDim, int32_t& dDim, int32_t& mDim, int32_t& nDim,
     const int32_t curCoreNum)
 {
     if (curCoreNum == 0 || curCoreNum < static_cast<int32_t>(coreNum_ * CORE_USED_THRESHOLD)) {
         return;
     }
     int64_t maxM = static_cast<int64_t>(runInfo_.dedx_h) * static_cast<int64_t>(runInfo_.dedx_w);
     int64_t maxN = static_cast<int64_t>(runInfo_.dedx_cin1);
     int64_t maxNBytes = maxN * blockSize_;
     // 分核目标：
     // (1) 保证singleCoreM和singleCoreN足够大, 可以使能128/64/256的基本块tiling;
     // (2) 数据读取的方向连续且顺序访问，提高cache复用率
     // (3) 所有核因子中间变量以Factor后缀命名，公约数和除法只操作因子返回因子，避免除0
 
     std::vector<int32_t> coreFactors = {};
     optiling::cachetiling::MathUtil::GetFactors(coreFactors, curCoreNum, curCoreNum);
     std::sort(coreFactors.rbegin(), coreFactors.rend());
 
     // B和D方向的最大公因子乘积是核的倍数，直接均匀分核，结束
     int32_t dMaxFactor = GetDimFactor(static_cast<int64_t>(runInfo_.dedx_d), coreFactors);
     int32_t bMaxFactor = GetDimFactor(static_cast<int64_t>(runInfo_.batch_n), coreFactors);
     if ((dMaxFactor * bMaxFactor) % curCoreNum == 0) {
         dDim = dMaxFactor;
         batchDim = curCoreNum / dMaxFactor;
         return;
     }
 
     // B和D分不完，找B*D方向的最大公因子，尽可能在B和D多分
     int32_t batchDepthMaxFactor = GetDimFactor(static_cast<int64_t>(dMaxFactor * bMaxFactor), coreFactors);
     int32_t remainFactor = curCoreNum / batchDepthMaxFactor;
 
     // 剩余的在M, N方向如果能均匀分核且切块粒度不小于基本块，也结束
     int32_t mMaxFactor = GetDimFactor(maxM, coreFactors);
     int32_t nMaxFactor = GetDimFactor(maxN, coreFactors);
     if ((mMaxFactor * nMaxFactor) % remainFactor == 0) {
         // 先从M方向考虑且粒度合适，结束
         int32_t mFactor = optiling::cachetiling::MathUtil::GetGcd(mMaxFactor, remainFactor);
         int32_t nFactor = remainFactor / mFactor;
         if ((nFactor == 1 || maxNBytes >= (nFactor * BEST_BASE_N)) && (mFactor == 1 || maxM >= (mFactor * BEST_BASE_M))) {
             dDim = optiling::cachetiling::MathUtil::GetGcd(dMaxFactor, batchDepthMaxFactor);
             batchDim = batchDepthMaxFactor / dDim;
             mDim = mFactor;
             nDim = nFactor;
             return;
         }
 
         // 再从N方向考虑且粒度合适，结束
         nFactor = optiling::cachetiling::MathUtil::GetGcd(nMaxFactor, remainFactor);
         mFactor = remainFactor / nFactor;
         if ((nFactor == 1 || maxNBytes >= nFactor * BEST_BASE_N) && (mFactor == 1 || maxM >= mFactor * BEST_BASE_M)) {
             dDim = optiling::cachetiling::MathUtil::GetGcd(dMaxFactor, batchDepthMaxFactor);
             batchDim = batchDepthMaxFactor / dDim;
             mDim = mFactor;
             nDim = nFactor;
             return;
         }
 
         // 这里假设M和N的范围差异是比较大的, 不再从M和N里面找其他形式的均衡
     }
 
     // m的粒度合适，执行M的非因子分核
     if (maxM >= (remainFactor * BEST_BASE_M)) {
         dDim = optiling::cachetiling::MathUtil::GetGcd(dMaxFactor, batchDepthMaxFactor);
         batchDim = batchDepthMaxFactor / dDim;
         mDim = remainFactor;
         return;
     }
 
     // 当前核数无法分核，尝试[coreNum_ - 1, coreNum_ * 60%]
     GetCoreDim(batchDim, dDim, mDim, nDim, curCoreNum - 1);
 
     // 都不满足，do nothing, 继承TBE
     return;
 }
 
 void Conv3DBackpropInputV2Tiling::SetTilingParamByDimInfo(TilingValue& tilingParams,
         const int32_t batchDim, const int32_t dDim, const int32_t mDim, const int32_t nDim)
 {
     tilingParams.coreNum =
         static_cast<uint64_t>(batchDim) * tbeTiling_.group_dim * mDim * tbeTiling_.k_dim * nDim * dDim;
     tilingParams.batchDim = batchDim;
     tilingParams.groupDim = tbeTiling_.group_dim;
     tilingParams.mDim = mDim;
     tilingParams.kDim = tbeTiling_.k_dim;
     tilingParams.nDim = nDim;
     tilingParams.dDim = dDim;
     tilingParams.singleCoreBatch = ops::CeilDiv(static_cast<int32_t>(runInfo_.batch_n), batchDim);
     tilingParams.singleCoreGroup = ops::CeilDiv(static_cast<int32_t>(runInfo_.real_g), tbeTiling_.group_dim);
     tilingParams.singleCoreM = static_cast<uint64_t>(ops::CeilDiv(runInfo_.dedx_h, mDim)) * runInfo_.dedx_w;
     tilingParams.singleCoreCout = runInfo_.dedy_cout1_g * blockSize_;
     tilingParams.singleCoreCout1 = runInfo_.dedy_cout1_g;
     tilingParams.singleCoreCin =
         static_cast<uint64_t>(ops::CeilDiv(static_cast<int32_t>(runInfo_.dedx_cin1_g), nDim)) * blockSize_;
     tilingParams.singleCoreCin1 = ops::CeilDiv(static_cast<int32_t>(runInfo_.dedx_cin1_g), nDim);
     tilingParams.singleCoreDin = ops::CeilDiv(static_cast<int32_t>(runInfo_.dedx_d), dDim);
     tilingParams.singleCoreHo = 1;
 }
 
 void Conv3DBackpropInputV2Tiling::CalCoreDimTiling(TilingValue& tilingParams, const uint32_t coreNum,
     bool& enableTbeBlock)
 {
     //4根轴可以分核：其中batch和m支持非因子分核
     int32_t batchDim = 1;
     int32_t dDim = 1;
     int32_t mDim = 1;
     int32_t nDim = 1;
 
     bool enableTBEtiling = false;
     // 优先从Depth进行分核，主要考虑增加D维度分核
     GetCoreDim(batchDim, dDim, mDim, nDim, coreNum);
     // 至少可以用满60%的核
     int64_t coreNumUsed = static_cast<int64_t>(batchDim) * dDim * mDim * nDim;
     enableTBEtiling = coreNumUsed < (coreNum * CORE_USED_THRESHOLD) || coreNumUsed > coreNum ||
         runInfo_.real_g > 1;
 
     if (enableTBEtiling) {
         // 采用TBE分核策略
         batchDim = tbeTiling_.batch_dim;
         dDim = tbeTiling_.d_dim;
         mDim = ops::CeilDiv(runInfo_.dedx_h, ops::CeilDiv(runInfo_.dedx_h, tbeTiling_.m_dim));
         nDim = tbeTiling_.n_dim;
         enableTbeBlock = true;
     } else {
         // 因M轴非因子切分可能导致实际使用的mDim小于原始mDim，此处需要修正
         mDim = ops::CeilDiv(runInfo_.dedx_h, ops::CeilDiv(runInfo_.dedx_h, mDim));
     }
 
     SetTilingParamByDimInfo(tilingParams, batchDim, dDim, mDim, nDim);
 }
 
 void Conv3DBackpropInputV2Tiling::UpdateBaseBlock(uint32_t& baseM, uint32_t& baseK, uint32_t& baseN, const TilingValue& tilingParams)
 {
     uint64_t singleC0BaseK = static_cast<uint64_t>(runInfo_.kernel_w) * runInfo_.kernel_h * blockSize_;
     if (baseK % singleC0BaseK == 0) {
         baseK = BEST_BASE_K / dtypeByte_;
     } else if (singleC0BaseK < baseK) {
         baseK = runInfo_.kernel_h * runInfo_.kernel_w * blockSize_;
     } else if (singleC0BaseK > baseK) {
         baseK = runInfo_.kernel_w * blockSize_;
     }
     //调换基本块tiling的M和N方向，确保singleCoreM和singleCoreN方向够用
     if (tilingParams.singleCoreM > tilingParams.singleCoreCin) {
         baseM = BEST_BASE_N;
         baseN = BEST_BASE_M;
     }
     // 超限处理
     if (baseM > tilingParams.singleCoreM) {
         baseM = ops::CeilDiv(static_cast<int32_t>(tilingParams.singleCoreM), BLOCK_CUBE) * BLOCK_CUBE;
     }
     if (baseN > tilingParams.singleCoreCin) {
         baseN = ops::CeilDiv(static_cast<int32_t>(tilingParams.singleCoreCin), BLOCK_CUBE) * BLOCK_CUBE;
     }
     uint64_t singleDepthMaxK =
         static_cast<uint64_t>(tilingParams.singleCoreCout1) * blockSize_ * runInfo_.kernel_h * runInfo_.kernel_w;
     // kernel侧应该保证下述条件的功能正确，当前在tiling侧进行约束
     if (baseK > singleDepthMaxK) {
         baseK = singleDepthMaxK;
     } else if (singleDepthMaxK % baseK != 0) {
         baseK = runInfo_.kernel_w * blockSize_;
     }
 }
 
 int32_t Conv3DBackpropInputV2Tiling::CalFmapH(const int32_t& mL1Size) const
 {
     int32_t hiCal;
     if (mL1Size % runInfo_.dedx_w == 0 || runInfo_.dedx_w % mL1Size == 0) {
         hiCal = ops::CeilDiv(mL1Size, runInfo_.dedx_w);
     } else if (mL1Size > runInfo_.dedx_w) {
         hiCal = mL1Size / runInfo_.dedx_w + FMAP_H_NUM;
     } else {
         hiCal = FMAP_H_NUM;
     }
     int32_t khDilation = (runInfo_.kernel_h - 1) * runInfo_.dilation_h + 1;
     int32_t hoCal = (hiCal - 1) + khDilation;
     int64_t hoExpand = static_cast<int64_t>(runInfo_.dedy_h - 1) * runInfo_.stride_h + 1;
     return static_cast<int32_t>(std::min(static_cast<int64_t>(hoCal), hoExpand));
 }
 
 void Conv3DBackpropInputV2Tiling::UpdateStepFp32(uint32_t &stepKa, uint32_t &stepKb, TilingValue &tilingParams)
 {
     // 调整fp32场景先L1中缓存step参数，当前有3个场景需要约束：
     // (1)B1不开double buffer场景，此时不使能preload功能，导致尾块处理逻辑变复杂,通过tiling约束;
     // (2)Cout1*C0是8对齐但是非16对齐场景，此时尾块可能存在跨越2个C0out=16，但是只取一个C0out=16的情况;
     // (3)Hk*Wk==1场景，此时无逆序操作，LoadToB2可以连续转换;
     // 如果不完整载入C0out=16，会导致跨C0out场景，kernel逻辑调整复杂，对性能无收益
     // tiling约束载入C0out=16的倍数，可以有效防止冗余数据载入
     uint64_t lenHkWkC0out = static_cast<uint64_t>(runInfo_.kernel_h) * runInfo_.kernel_w * BLOCK_CUBE;
     if ((tilingParams.bl1Pbuffer == 1) ||
         (runInfo_.dedy_cout1_g * blockSize_ % BLOCK_CUBE != 0) ||
         (runInfo_.kernel_h * runInfo_.kernel_w == 1 && tilingParams.baseK < BLOCK_CUBE)) {
         stepKb = (stepKb * tilingParams.baseK + lenHkWkC0out - 1) / lenHkWkC0out * lenHkWkC0out / tilingParams.baseK;
     }
     // 调整A矩阵载入量，不影响上述stepKb的计算结果，让stepKa和stepKb满足整数倍关系
     uint64_t lenHkWkC0 = lenHkWkC0out / 2;
     if (stepKa > stepKb) {
         while (stepKa % stepKb > 0) {
             stepKa--;
         }
     } else {
         while (stepKb % stepKa > 0) {
             stepKa--;
         }
     }
     // 调整stepKa可能导致非整份载入HkWkC0
     stepKa = stepKa * tilingParams.baseK / lenHkWkC0 * lenHkWkC0 / tilingParams.baseK;
 }
 
 void Conv3DBackpropInputV2Tiling::UpdateBaseStep(uint32_t &stepKa, uint32_t &stepKb, TilingValue &tilingParams)
 {
     uint32_t hoCal = CalFmapH(tilingParams.baseM);  // 此处默认stepM=1
     uint64_t lenHkWkC0 = runInfo_.kernel_h * runInfo_.kernel_w * blockSize_;
     uint64_t baseNHkWkC0Size = lenHkWkC0 * tilingParams.baseN * dtypeByte_;
     uint64_t l1BSize = L1_SIZE / BUFFER_NUM_L1;
     uint64_t l1ASize = l1BSize;
     bool adaptFP32 = (dtypeByte_ == FP32_DATA_SIZE && runInfo_.kernel_h * runInfo_.kernel_w > 1);
     // fp32场景下Cout0为16，c0为8，而tiling中的Cout1是以C0对其，因此需保证加载的cout1要为2的倍数
     uint32_t fp32Cout1Factor = 2;
     uint32_t cout1B1 = std::max(static_cast<uint64_t>(1), l1BSize / baseNHkWkC0Size);
     if (adaptFP32) {
         if (cout1B1 == 1) {
             cout1B1 = fp32Cout1Factor;
             l1ASize = l1ASize - (baseNHkWkC0Size * cout1B1 - l1BSize);
         } else {
             cout1B1 = cout1B1 / fp32Cout1Factor * fp32Cout1Factor; // fp32场景下确保cout1为2的倍数
         }
     }
 
     uint64_t curHiWiSize = static_cast<uint64_t>(dtypeByte_) * hoCal * runInfo_.dedy_w * runInfo_.stride_w * blockSize_;
     uint32_t cout1A1 = std::max(static_cast<uint64_t>(1), l1ASize / curHiWiSize);
     if (cout1A1 >= tilingParams.singleCoreCout1) {
         cout1A1 = tilingParams.singleCoreCout1;
         tilingParams.al1Pbuffer = 1;
     }
 
     if (cout1B1 >= tilingParams.singleCoreCout1) {
         cout1B1 = tilingParams.singleCoreCout1;
         tilingParams.bl1Pbuffer = 1;
     }
     AlignCout1(cout1A1, cout1B1, adaptFP32);
 
     stepKa = std::max(static_cast<uint64_t>(1),
         ops::CeilDiv(static_cast<uint64_t>(cout1A1) * runInfo_.kernel_h * runInfo_.kernel_w * blockSize_,
                      static_cast<uint64_t>(tilingParams.baseK)));
     stepKa = std::min(stepKa, K_START_POSITION_MAX / tilingParams.baseK);
     stepKb = std::max(static_cast<uint64_t>(1),
         ops::CeilDiv(static_cast<uint64_t>(cout1B1) * runInfo_.kernel_h * runInfo_.kernel_w * blockSize_,
                      static_cast<uint64_t>(tilingParams.baseK)));
     if (stepKa > stepKb) {
         stepKa = ops::FloorAlign(stepKa, stepKb);
     } else {
         stepKb = ops::FloorAlign(stepKb, stepKa);
     }
     // fp32场景下，对tiling进行修改，以符合fp32场景要求
     if (dtypeByte_ == FP32_DATA_SIZE) {
         UpdateStepFp32(stepKa, stepKb, tilingParams);
     }
 }
 
 bool Conv3DBackpropInputV2Tiling::CalBaseBlockTiling(TilingValue& tilingParams)
 {
     // 默认开启double buffer
     tilingParams.al1Pbuffer = DB_ON;
     tilingParams.bl1Pbuffer = DB_ON;
     tilingParams.al0Pbuffer = DB_ON;
     tilingParams.bl0Pbuffer = DB_ON;
     tilingParams.cl0Pbuffer = 1;
 
     // 默认采用最优基本块tiling
     // 910B最优基本块tiling
     uint32_t stepKa = 1;
     uint32_t stepKb = 1;
     uint32_t baseM = BEST_BASE_M;
     uint32_t baseK = BEST_BASE_K / dtypeByte_;
     uint32_t baseN = BEST_BASE_N;
 
     // 更新并设置基本块tiling
     UpdateBaseBlock(baseM, baseK, baseN, tilingParams);
     tilingParams.baseM = baseM;
     tilingParams.baseK = baseK;
     tilingParams.baseN = baseN;
     tilingParams.baseD = runInfo_.d_cl0;
     tilingParams.baseBatch = 1;
     tilingParams.baseGroup = 1;
     // 更新并设置step tiling
     UpdateBaseStep(stepKa, stepKb, tilingParams);
     tilingParams.stepKa = stepKa;
     tilingParams.stepKb = stepKb;
     tilingParams.stepM = 1;
     tilingParams.stepN = 1;
     tilingParams.stepBatch = 1;
     tilingParams.stepGroup = 1;
     tilingParams.iterateOrder = 1;
 
     uint64_t b1Size = 0;
     uint64_t lenHkWkC0 = static_cast<uint64_t>(runInfo_.kernel_h) * runInfo_.kernel_w * blockSize_;
     if (dtypeByte_ == FP32_DATA_SIZE) {
         uint64_t numHkWkC0 = ops::CeilDiv(static_cast<uint64_t>(tilingParams.stepKb * tilingParams.baseK), lenHkWkC0);
         b1Size = tilingParams.bl1Pbuffer * dtypeByte_ * baseN *
                 ops::CeilDiv(numHkWkC0, static_cast<uint64_t>(NUM_FP32_C1OUT)) * NUM_FP32_C1OUT * lenHkWkC0;
     } else {
         b1Size = tilingParams.bl1Pbuffer * dtypeByte_ * stepKb * baseN * baseK;
     }
     uint64_t coutNum = static_cast<uint64_t>(stepKa) * baseK / (runInfo_.kernel_h * runInfo_.kernel_w);
     uint64_t a1PixelNum = static_cast<uint64_t>(CalFmapH(baseM)) * runInfo_.dedy_w * runInfo_.stride_w * coutNum;
     uint64_t l1UsedSize = a1PixelNum * dtypeByte_ * tilingParams.al1Pbuffer + b1Size;
     if (stepKa * stepKb == 0) {
         return false;
     }
 
     // 校验是否满足整数倍关系
     uint32_t stepParaCheck = (stepKa > stepKb) ? (stepKa % stepKb) : (stepKb % stepKa);
     // 校验是否满足整数倍HkWkC0关系
     bool stepParaCheck2 = ((static_cast<uint64_t>(tilingParams.stepKa * tilingParams.baseK) % lenHkWkC0 == 0) &&
                             (static_cast<uint64_t>(tilingParams.stepKb * tilingParams.baseK) % lenHkWkC0 == 0));
     // 校验能否采用基本块tiling策略，否则切换到TBE tiling策略
     // 为保证A和B矩阵使能double buffer，两者加和要小于L1 size的一半
     if (CheckL0Size(baseM, baseN, baseK, dtypeByte_) && (l1UsedSize <= L1_SIZE) && stepParaCheck == 0 && stepParaCheck2 == 1) {
         OP_LOGD(opName_, "Use basic-block tiling.");
         return true;
     }
     return false;
 }
 
 void Conv3DBackpropInputV2Tiling::CalTbeBlockTiling(TilingValue& tilingParams)
 {
     tilingParams.al0Pbuffer = DB_ON;
     tilingParams.bl0Pbuffer = DB_ON;
     tilingParams.cl0Pbuffer = static_cast<uint32_t>(tbeTiling_.db_l0c);
     uint32_t baseM = tbeTiling_.m_l0 * BLOCK_CUBE;
     uint32_t baseK = tbeTiling_.k_l0 * blockSize_;
     uint32_t baseN = tbeTiling_.n_l0 * BLOCK_CUBE;
     uint32_t tmpBaseKMax = std::max(runInfo_.kernel_h * blockSize_, runInfo_.kernel_w * blockSize_);
     uint32_t tmpBaseKMin = std::min(runInfo_.kernel_h * blockSize_, runInfo_.kernel_w * blockSize_);
 
     if (CheckL0Size(baseM, baseN, runInfo_.kernel_h * runInfo_.kernel_w * blockSize_, dtypeByte_)) {
         baseK = runInfo_.kernel_h * runInfo_.kernel_w * blockSize_;
     } else if (CheckL0Size(baseM, baseN, tmpBaseKMax, dtypeByte_)) {
         baseK = tmpBaseKMax;
     } else if (CheckL0Size(baseM, baseN, tmpBaseKMin, dtypeByte_)) {
         baseK = tmpBaseKMin;
     } else {
         baseK = blockSize_;
     }
 
     tilingParams.baseM = baseM;
     tilingParams.baseK = baseK;
     tilingParams.baseN = baseN;
     tilingParams.baseD = runInfo_.d_cl0;
     tilingParams.baseBatch = 1;
     tilingParams.baseGroup = 1;
 
     tilingParams.al1Pbuffer = static_cast<uint32_t>(tbeTiling_.db_al1);
     tilingParams.bl1Pbuffer = static_cast<uint32_t>(tbeTiling_.db_bl1);
     tilingParams.stepKa = ops::CeilDiv(static_cast<uint64_t>(tbeTiling_.k_al1) * runInfo_.kernel_h * runInfo_.kernel_w,
                                 static_cast<uint64_t>(baseK / blockSize_));
     tilingParams.stepKb = ops::CeilDiv(static_cast<uint64_t>(tbeTiling_.k_bl1) * runInfo_.kernel_h * runInfo_.kernel_w,
                                 static_cast<uint64_t>(baseK / blockSize_));
 
     if (dtypeByte_ == FP32_DATA_SIZE) {
         UpdateStepFp32(tilingParams.stepKa, tilingParams.stepKb, tilingParams);
     }
 
     tilingParams.stepM = 1;
     tilingParams.stepN = 1;
     tilingParams.stepBatch = 1;
     tilingParams.stepGroup = 1;
     tilingParams.iterateOrder = 1;
 }
 
 void Conv3DBackpropInputV2Tiling::InitTilingValue(TilingValue& tilingParams, const uint32_t coreNum)
 {
     bool enableTbeBlock = false;
     CalCoreDimTiling(tilingParams, coreNum, enableTbeBlock);
     if (enableTbeBlock || !CalBaseBlockTiling(tilingParams)) {
         OP_LOGD(opName_, "Use tbe cache tiling.");
         int32_t mDim = ops::CeilDiv(runInfo_.dedx_h, ops::CeilDiv(runInfo_.dedx_h, tbeTiling_.m_dim));
         SetTilingParamByDimInfo(tilingParams, tbeTiling_.batch_dim, tbeTiling_.d_dim, mDim, tbeTiling_.n_dim);
         CalTbeBlockTiling(tilingParams);
     }
 }
 
 void Conv3DBackpropInputV2Tiling::SetTilingValue(TConv3DInputV2Tiling& dxt, const TilingValue& tilingParams)
 {
     // singleCore
     dxt.set_singleCoreBatch(tilingParams.singleCoreBatch);
     dxt.set_singleCoreGroup(tilingParams.singleCoreGroup);
     dxt.set_singleCoreM(tilingParams.singleCoreM);
     dxt.set_singleCoreCout(tilingParams.singleCoreCout);
     dxt.set_singleCoreCout1(tilingParams.singleCoreCout1);
     dxt.set_singleCoreCin1(tilingParams.singleCoreCin1);
     dxt.set_singleCoreCin(tilingParams.singleCoreCin);
     dxt.set_singleCoreDin(tilingParams.singleCoreDin);
     dxt.set_singleCoreHo(tilingParams.singleCoreHo);
 
     tilingData_.params.set_batchDim(tilingParams.batchDim);
     tilingData_.params.set_groupDim(tilingParams.groupDim);
     tilingData_.params.set_mDim(tilingParams.mDim);
     tilingData_.params.set_kDim(tilingParams.kDim);
     tilingData_.params.set_nDim(tilingParams.nDim);
     tilingData_.params.set_dDim(tilingParams.dDim);
     tilingData_.params.set_coreNum(tilingParams.coreNum);
 
     dxt.set_baseM(tilingParams.baseM);
     dxt.set_baseK(tilingParams.baseK);
     dxt.set_baseN(tilingParams.baseN);
     dxt.set_baseD(tilingParams.baseD);
     dxt.set_baseBatch(tilingParams.baseBatch);
     dxt.set_baseGroup(tilingParams.baseGroup);
 
     dxt.set_stepM(tilingParams.stepM);
     dxt.set_stepN(tilingParams.stepN);
 
     dxt.set_stepKa(tilingParams.stepKa);
     dxt.set_stepKb(tilingParams.stepKb);
     dxt.set_stepBatch(tilingParams.stepBatch);
     dxt.set_stepGroup(tilingParams.stepGroup);
 
     dxt.set_al0Pbuffer(tilingParams.al0Pbuffer);  // 默认开
     dxt.set_bl0Pbuffer(tilingParams.bl0Pbuffer);  // 默认开
     dxt.set_cl0Pbuffer(tilingParams.cl0Pbuffer);
     dxt.set_al1Pbuffer(tilingParams.al1Pbuffer);
     dxt.set_bl1Pbuffer(tilingParams.bl1Pbuffer);
 
     dxt.set_iterateOrder(tilingParams.iterateOrder);
 
     if (runInfo_.kernel_h * runInfo_.kernel_w == 1) {
         loadB2Condition_ = 2; // 2表示Hk*Wk = 1的情况
     } else if (tilingParams.baseK / blockSize_ >= static_cast<uint32_t>(runInfo_.kernel_h * runInfo_.kernel_w)) {
         loadB2Condition_ = 1;
     } else {
         loadB2Condition_ = 0;
     }
     if (coreNum_ == CORE_NUM_910B2 || coreNum_ == CORE_NUM_910B3) {
         enableKernelSplit_ = CheckKernelSplitEnable();
     }
 }
 
 void Conv3DBackpropInputV2Tiling::SetBackpropPadInfo(TConv3DInputV2Tiling& dxt)
 {
     int64_t bpPadTail = runInfo_.dedx_d - (static_cast<int64_t>(runInfo_.dedy_d - 1) * runInfo_.stride_d + 1) +
                         (runInfo_.kernel_d - 1) * runInfo_.dilation_d - runInfo_.backprop_pad_h;
     if (bpPadTail < PAD_DIM_LOW || bpPadTail > PAD_DIM_UP) {
         dxt.set_backpropPadTail(runInfo_.backprop_pad_t);
     } else {
         dxt.set_backpropPadTail(static_cast<uint32_t>(bpPadTail));
     }
     OP_LOGD(opName_, "backprop tail pad: %ld, origin backprop_pad_t: %d", bpPadTail, runInfo_.backprop_pad_t);
 
     dxt.set_backpropPadUp(runInfo_.backprop_pad_u);
     int64_t bpPadDown = runInfo_.dedx_h - (static_cast<int64_t>(runInfo_.dedy_h - 1) * runInfo_.stride_h + 1) +
                         (runInfo_.kernel_h - 1) * runInfo_.dilation_h - runInfo_.backprop_pad_u;
     if (bpPadDown < PAD_DIM_LOW || bpPadDown > PAD_DIM_UP) {
         dxt.set_backpropPadDown(runInfo_.backprop_pad_d);
     } else {
         dxt.set_backpropPadDown(static_cast<uint32_t>(bpPadDown));
     }
     OP_LOGD(opName_, "backprop down pad: %ld, origin backprop_pad_d: %d", bpPadDown, runInfo_.backprop_pad_d);
 
     dxt.set_backpropPadLeft(runInfo_.backprop_pad_l);
     int64_t bpPadRight = runInfo_.dedx_w - (static_cast<int64_t>(runInfo_.dedy_w - 1) * runInfo_.stride_w + 1) +
                         (runInfo_.kernel_w - 1) * runInfo_.dilation_w - runInfo_.backprop_pad_l;
     if (bpPadRight < PAD_DIM_LOW || bpPadRight > PAD_DIM_UP) {
         dxt.set_backpropPadRight(runInfo_.backprop_pad_r);
     } else {
         dxt.set_backpropPadRight(static_cast<uint32_t>(bpPadRight));
     }
     OP_LOGD(opName_, "backprop right pad: %ld, origin backprop_pad_r: %d", bpPadRight, runInfo_.backprop_pad_r);
 }
 
 void Conv3DBackpropInputV2Tiling::SetRunInfoTiling(TConv3DInputV2Tiling& dxt)
 {
     // shape
     dxt.set_batch(runInfo_.batch_n);
     dxt.set_cin(runInfo_.dedx_cin);
     dxt.set_cout(runInfo_.dedy_cout);
     dxt.set_cin1G(runInfo_.dedx_cin1_g);
     dxt.set_cout1G(runInfo_.dedy_cout1_g);
     dxt.set_cin1(runInfo_.dedx_cin1);
     dxt.set_cout1(runInfo_.dedy_cout1);
     dxt.set_c0(blockSize_);
     if (dtypeByte_ == F16_DATA_SIZE) {
         dxt.set_c0Bits(B16_BITS);
     } else if (dtypeByte_ == FP32_DATA_SIZE) {
         dxt.set_c0Bits(FP32_BITS);
     }
     dxt.set_ho(runInfo_.dedy_h);
     dxt.set_wo(runInfo_.dedy_w);
     dxt.set_dout(runInfo_.dedy_d);
     dxt.set_di(runInfo_.dedx_d);
     dxt.set_hi(runInfo_.dedx_h);
     dxt.set_wi(runInfo_.dedx_w);
     dxt.set_hk(runInfo_.kernel_h);
     dxt.set_wk(runInfo_.kernel_w);
     dxt.set_dk(runInfo_.kernel_d);
 
     dxt.set_group(runInfo_.real_g);
     dxt.set_strideH(runInfo_.stride_h);
     dxt.set_strideW(runInfo_.stride_w);
     dxt.set_strideD(runInfo_.stride_d);
     dxt.set_padFront(runInfo_.pad_h);
     dxt.set_padBack(runInfo_.pad_t);
     dxt.set_padUp(runInfo_.pad_u);
     dxt.set_padDown(runInfo_.pad_d);
     dxt.set_padLeft(runInfo_.pad_l);
     dxt.set_padRight(runInfo_.pad_r);
     SetBackpropPadInfo(dxt);
 
     dxt.set_dilationH(runInfo_.dilation_h);
     dxt.set_dilationW(runInfo_.dilation_w);
     dxt.set_dilationD(runInfo_.dilation_d);
     dxt.set_hf32Flag(runInfo_.hf32_flag);
     dxt.set_initOutputFlag(initOutputFlag);
 }
 
 void Conv3DBackpropInputV2Tiling::SetDxTilingFromTbeTiling()
 {
     TConv3DInputV2Tiling &dxt = tilingData_.conv3DDxTiling;
     TilingValue tilingParams;
     // key:
     // "N_Do_Co1_Ho_Wo_Di_Ci1_Hi_Wi_Dk_Hk_Wk_strideD_strideH_strideW_
     // _padFront_padBack_padUp_padDown_padLeft_padRight_dilationD_dilationH_dilationW"
     std::string key =
         std::to_string(runInfo_.batch_n) + "_" +
         std::to_string(runInfo_.dedy_d) + "_" +
         std::to_string(runInfo_.dedy_cout1) + "_" +
         std::to_string(runInfo_.dedy_h) + "_" +
         std::to_string(runInfo_.dedy_w) + "_" +
         std::to_string(runInfo_.dedx_d) + "_" +
         std::to_string(runInfo_.dedx_cin1) + "_" +
         std::to_string(runInfo_.dedx_h) + "_" +
         std::to_string(runInfo_.dedx_w) + "_" +
         std::to_string(runInfo_.kernel_d) + "_" +
         std::to_string(runInfo_.kernel_h) + "_" +
         std::to_string(runInfo_.kernel_w) + "_" +
         std::to_string(runInfo_.stride_d) + "_" +
         std::to_string(runInfo_.stride_h) + "_" +
         std::to_string(runInfo_.stride_w) + "_" +
         std::to_string(runInfo_.pad_h) + "_" + std::to_string(runInfo_.pad_t) +
         "_" + std::to_string(runInfo_.pad_u) + "_" +
         std::to_string(runInfo_.pad_d) + "_" + std::to_string(runInfo_.pad_l) +
         "_" + std::to_string(runInfo_.pad_r) + "_" +
         std::to_string(runInfo_.dilation_d) + "_" +
         std::to_string(runInfo_.dilation_h) + "_" +
         std::to_string(runInfo_.dilation_w);
     optiling::PlatformInfo &plaformInstance = optiling::PlatformInfo::GetInstance();
     coreNum_ = plaformInstance.core_num;
     OP_TILING_CHECK(coreNum_ <= 0, CUBE_INNER_ERR_REPORT(opName_, "Failed to get valid core number from platform information. core num: %d", coreNum_), return);
     bool isHitKnowledgeMap = false;
     if (runInfo_.real_g == 1) {
         if (dtypeByte_ == F16_DATA_SIZE) {
             if (coreNum_ == CORE_NUM_910B3 && TILING_DATA_MAP_B3.find(key) != TILING_DATA_MAP_B3.end()) {
                 tilingParams = TILING_DATA_MAP_B3.at(key);
                 isHitKnowledgeMap = true;
             } else if (coreNum_ == CORE_NUM_910B2 && TILING_DATA_MAP_B2.find(key) != TILING_DATA_MAP_B2.end()) {
                 tilingParams = TILING_DATA_MAP_B2.at(key);
                 isHitKnowledgeMap = true;
             }
         } else if (dtypeByte_ == FP32_DATA_SIZE) {
             if (coreNum_ == CORE_NUM_910B2 && FP32_TILING_DATA_MAP_B2.find(key) != FP32_TILING_DATA_MAP_B2.end()) {
                 tilingParams = FP32_TILING_DATA_MAP_B2.at(key);
                 isHitKnowledgeMap = true;
             } else if (coreNum_ == CORE_NUM_910B3 && FP32_TILING_DATA_MAP_B3.find(key) != FP32_TILING_DATA_MAP_B3.end()) {
                 tilingParams = FP32_TILING_DATA_MAP_B3.at(key);
                 isHitKnowledgeMap = true;
             } else if (coreNum_ == CORE_NUM_910B2 && FP32_BASIC_BLOCK_MAP_B2.count(key)) {
                 tilingParams = FP32_BASIC_BLOCK_MAP_B2.at(key);
                 useBasicBlock_ = isHitKnowledgeMap = true;
             } else if (coreNum_ == CORE_NUM_910B3 && FP32_BASIC_BLOCK_MAP_B3.count(key)) {
                 tilingParams = FP32_BASIC_BLOCK_MAP_B3.at(key);
                 useBasicBlock_ = isHitKnowledgeMap = true;
             }
         }
     }
     if (isHitKnowledgeMap) {
         OP_LOGD(opName_, "hit tiling knowledge map");
     } else {
         InitTilingValue(tilingParams, coreNum_);
     }
     SetRunInfoTiling(dxt);
     SetTilingValue(dxt, tilingParams);
 }
 
 void Conv3DBackpropInputV2Tiling::PrintTilingData()
 {
   TConv3DInputV2Tiling &tiling = tilingData_.conv3DDxTiling;
   std::stringstream ss;
   ss << " batch: " << tiling.get_batch() << " cin: " << tiling.get_cin()
      << " cout: " << tiling.get_cout() << " di: " << tiling.get_di()
      << " dout: " << tiling.get_dout() << " dk: " << tiling.get_dk()
      << " ho: " << tiling.get_ho() << " wo: " << tiling.get_wo()
      << " hi: " << tiling.get_hi() << " wi: " << tiling.get_wi()
      << " hk: " << tiling.get_hk() << " wk: " << tiling.get_wk()
      << " singleCoreBatch: " << tiling.get_singleCoreBatch()
      << " singleCoreGroup: " << tiling.get_singleCoreGroup()
      << " singleCoreCout: " << tiling.get_singleCoreCout()
      << " singleCoreCin: " << tiling.get_singleCoreCin()
      << " singleCoreHo: " << tiling.get_singleCoreHo()
      << " baseM: " << tiling.get_baseM() << " baseK: " << tiling.get_baseK()
      << " baseN: " << tiling.get_baseN()
      << " baseBatch: " << tiling.get_baseBatch()
      << " baseGroup: " << tiling.get_baseGroup()
      << " stepM: " << tiling.get_stepM() << " stepN: " << tiling.get_stepN()
      << " stepKa: " << tiling.get_stepKa() << " stepKb: " << tiling.get_stepKb()
      << " stepBatch: " << tiling.get_stepBatch()
      << " stepGroup: " << tiling.get_stepGroup()
      << " al0Pbuffer: " << tiling.get_al0Pbuffer()
      << " bl0Pbuffer: " << tiling.get_bl0Pbuffer()
      << " cl0Pbuffer: " << tiling.get_cl0Pbuffer()
      << " al1Pbuffer: " << tiling.get_al1Pbuffer()
      << " bl1Pbuffer: " << tiling.get_bl1Pbuffer()
      << " group: " << tiling.get_group() << " strideH: " << tiling.get_strideH()
      << " strideW: " << tiling.get_strideW()
      << " strideD: " << tiling.get_strideD()
      << " padFront: " << tiling.get_padFront()
      << " padBack: " << tiling.get_padBack() << " padUp: " << tiling.get_padUp()
      << " padDown: " << tiling.get_padDown()
      << " padLeft: " << tiling.get_padLeft()
      << " padRight: " << tiling.get_padRight()
      << " dilationH: " << tiling.get_dilationH()
      << " dilationW: " << tiling.get_dilationW()
      << " dilationD: " << tiling.get_dilationD()
      << " iterateOrder: " << tiling.get_iterateOrder()
      << " hf32Flag: " << tiling.get_hf32Flag();
   OP_LOG_FULL(DLOG_DEBUG, opName_, "api tiling: %s", ss.str().c_str());
 }
 
 REGISTER_TILING_TEMPLATE("Conv3DBackpropInputV2", Conv3DBackpropInputV2Tiling, 1);
 
 static ge::graphStatus Conv3DBackpropInputV2TilingFunc(gert::TilingContext *context)
 {
     return TilingRegistry::GetInstance().DoTilingImpl(context);
 }
 
 static ge::graphStatus TilingParseForConv3DBackpropInputV2(gert::TilingParseContext *context)
 {
     auto platformInfoPtr = context->GetPlatformInfo();
     OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "platformInfoPtr is null");
     auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
 
     auto compileInfoPtr = context->GetCompiledInfo<Conv3DBackpropInputV2CompileInfo>();
     OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "compileInfo is null");
     compileInfoPtr->ParseRuntimePlatformInfo(context->GetNodeName(), *platformInfoPtr);
     compileInfoPtr->core_num = ascendcPlatform.GetCoreNumAic();
 
     optiling::PlatformInfo &plaformInstance = optiling::PlatformInfo::GetInstance();
     plaformInstance.SetInstance(*compileInfoPtr);
 
     return ge::GRAPH_SUCCESS;
 }
 
 IMPL_OP_OPTILING(Conv3DBackpropInputV2)
     .Tiling(Conv3DBackpropInputV2TilingFunc)
     .TilingParse<Conv3DBackpropInputV2CompileInfo>(TilingParseForConv3DBackpropInputV2);
 }  // namespace optiling
 