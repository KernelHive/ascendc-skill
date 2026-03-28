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
 * \file weight_quant_batch_matmul_v2_tiling_custom.cpp
 * \brief
 */

#include "weight_quant_batch_matmul_v2_tiling_custom.h"

#include "weight_quant_batch_matmul_v2_compute_matmul_tiling.h"
#include "weight_quant_batch_matmul_v2_white_list.h"

namespace optiling {

constexpr uint64_t MAX_SHAPE_DIM = 65535UL;
constexpr uint64_t MIN_GROUP_SIZE = 32UL;
constexpr int32_t MAX_REPEAT_TIMES = 255;
constexpr uint32_t CUSTOM_NZ_TRANS_BASE_N = 64;
constexpr uint32_t CUSTOM_NZ_NO_TRANS_BASE_N = 32;
constexpr uint32_t CUSTOM_NZ_TRANS_BF16_BASE_K = 256;
constexpr uint32_t CUSTOM_NZ_NO_TRANS_BF16_BASE_N = 544;
constexpr uint32_t CUSTOM_NZ_TRANS_FP16_BASE_K = 384;
constexpr uint32_t CUSTOM_NZ_NO_TRANS_FP16_BASE_K = 864;
constexpr int32_t TILING_COMPENSATION_FACTOR = 2;
constexpr uint32_t CUSTOM_NZ_GROUP_BASE_N = 48U;

const std::map<WhiteListShape, MatMulTilingCache> MM_TILING_CACHE = {
    {{6, 11264, 1664, false, false, true, 24},
     {3, 6, 96, 1664, 11264, 11264, 16, 80, 11264, 16, 80, 128, 8, 8, 1, 1, 4, 4, 30720, 0, 180224, 30720, 2, 2, 1}},
    {{64, 11264, 1664, false, false, true, 24}, {3, 8, 1024, 1664, 11264, 11264, 128, 192,    11264, 128, 192, 64, 16,
                                                 8, 1, 1,    8,    4,     98304, 0,   458752, 98304, 2,   2,   1}},
    {{6, 1408, 11264, false, false, true, 24}, {8,  3, 96, 11264, 1408, 1536,  32, 1408,   1408,  32, 128, 128, 11,
                                                11, 1, 1,  11,    11,   32768, 1,  450560, 16384, 2,  2,   2}},
    {{10240, 6848, 4096, false, false, true, 8},
     {1, 8, 256, 128, 512, 512, 256, 128, 512, 256, 128, 64, 8, 8, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2}},
    {{1, 4096, 27392, false, false, false, 20},
     {18, 1, 1, 27392, 4096, 4096, 1, 1522, 4096, 16, 128, 128, 32, 2, 1, 1, 32, 1, 32768, 1, 196608, 8192, 2, 2, 2}},
    {{1, 13696, 4096, false, false, false, 20},
     {16, 1, 1, 4096, 13696, 13696, 1, 256, 13696, 16, 64, 128, 107, 2, 1, 1, 107, 1, 16384, 1, 471040, 4096, 2, 2, 2}},
    {{1, 4096, 4096, false, false, false, 20},
     {16, 1, 1, 4096, 4096, 4096, 1, 256, 4096, 16, 128, 128, 64, 2, 1, 1, 32, 1, 32768, 1, 327680, 8192, 2, 2, 2}},
    {{1, 10240, 8640, false, false, false, 20},
     {17, 1, 1, 8640, 10240, 10240, 1, 509, 10240, 16, 64, 256, 40, 2, 1, 1, 40, 1, 32768, 1, 393216, 4096, 2, 2, 2}},
    {{1, 1280, 10240, false, false, false, 20},
     {20, 1, 1, 10240, 1280, 1280, 1, 512, 1280, 16, 256, 64, 20, 2, 1, 1, 20, 1, 32768, 1, 106496, 16384, 2, 2, 2}},
    {{1, 4320, 10240, false, false, false, 20},
     {20, 1, 1, 10240, 4320, 4320, 1, 512, 4320, 16, 128, 128, 68, 2, 1, 1, 34, 1, 32768, 1, 344064, 8192, 2, 2, 2}},
    {{2, 13696, 4096, false, false, false, 20},
     {8, 2, 32, 4096, 13696, 13696, 16, 512, 13696, 16, 128, 128, 32, 2, 1, 1, 16, 1, 32768, 0, 196608, 8192, 2, 2, 2}},
    {{2, 4096, 27392, false, false, false, 20}, {18, 1, 32, 27392, 4096, 4096,  32, 1522,   4096,  32, 128, 128, 32,
                                                 2,  1, 1,  16,    1,    32768, 1,  327680, 16384, 2,  2,   2}},
    {{2, 10240, 8640, false, false, false, 20}, {8, 2, 32, 8640, 10240, 10240, 16, 1080,   10240, 16, 128, 128, 80,
                                                 2, 1, 1,  80,   1,     32768, 0,  393216, 8192,  2,  2,   2}},
    {{2, 4320, 10240, false, false, false, 20},
     {8, 2, 32, 10240, 4320, 4320, 16, 1280, 4320, 16, 128, 128, 68, 2, 1, 1, 34, 1, 32768, 1, 344064, 8192, 2, 2, 2}},
    {{4, 13696, 4096, false, false, false, 20}, {8, 2, 64, 4096, 13696, 13696, 32, 512,    13696, 32, 128, 128, 54,
                                                 2, 1, 1,  27,   1,     32768, 0,  507904, 16384, 2,  2,   2}},
    {{4, 4096, 27392, false, false, false, 20},
     {8, 2, 64, 27392, 4096, 4096, 32, 3424, 4096, 32, 256, 64, 64, 2, 1, 1, 64, 1, 32768, 0, 327680, 32768, 2, 2, 2}},
    {{4, 10240, 8640, false, false, false, 20}, {8, 2, 64, 8640, 10240, 10240, 32, 1080,   10240, 32, 128, 128, 40,
                                                 2, 1, 1,  20,   1,     32768, 0,  393216, 16384, 2,  2,   2}},
    {{4, 4320, 10240, false, false, false, 20},
     {8, 2, 64, 10240, 4320, 4320, 32, 1280, 4320, 32, 256, 64, 68, 2, 1, 1, 68, 1, 32768, 0, 344064, 32768, 2, 2, 2}},
    {{5, 13696, 4096, false, false, false, 20}, {6, 3, 80, 4096, 13696, 13696, 27, 683,    13696, 32, 128, 128, 54,
                                                 2, 1, 1,  27,   1,     32768, 0,  507904, 16384, 2,  2,   1}},
    {{5, 4096, 27392, false, false, false, 20},
     {8, 2, 80, 27392, 4096, 4096, 40, 3424, 4096, 48, 256, 64, 64, 2, 1, 1, 64, 1, 49152, 0, 458752, 49152, 2, 2, 1}},
    {{5, 10240, 8640, false, false, false, 20}, {6, 3, 80, 8640, 10240, 10240, 27, 1440,   10240, 32, 128, 128, 40,
                                                 2, 1, 1,  20,   1,     32768, 0,  393216, 16384, 2,  2,   1}},
    {{5, 4320, 10240, false, false, false, 20},
     {6, 3, 80, 10240, 4320, 4320, 27, 1707, 4320, 32, 256, 64, 68, 2, 1, 1, 68, 1, 32768, 0, 344064, 32768, 2, 2, 1}},
    {{6, 13696, 4096, false, false, false, 20},
     {8, 2, 96, 4096, 13696, 13696, 48, 512, 13696, 48, 128, 64, 8, 2, 1, 1, 4, 1, 24576, 0, 81920, 24576, 2, 2, 1}},
    {{6, 4320, 10240, false, false, false, 20},
     {10, 2, 96, 10240, 4320, 4320, 48, 1024, 4320, 48, 256, 64, 8, 2, 1, 1, 4, 1, 49152, 0, 114688, 49152, 2, 2, 1}},
    {{6, 4096, 27392, false, false, false, 20},
     {9, 2, 96, 27392, 4096, 4096, 48, 3044, 4096, 48, 256, 64, 8, 2, 1, 1, 4, 1, 49152, 0, 114688, 49152, 2, 2, 1}},
    {{6, 10240, 8640, false, false, false, 20},
     {6, 3, 96, 8640, 10240, 10240, 32, 1440, 10240, 32, 256, 64, 8, 2, 1, 1, 4, 1, 32768, 0, 98304, 32768, 2, 2, 1}},
    {{6, 10240, 1536, true, false, false, 20},
     {6, 3, 96, 1536, 10240, 10240, 32, 256, 10240, 32, 256, 64, 80, 2, 1, 1, 40, 1, 32768, 0, 393728, 32768, 2, 2, 2}},
    {{3, 13696, 4096, false, false, false, 20},
     {8, 2, 48, 4096, 13696, 13696, 24, 512, 13696, 32, 128, 64, 8, 2, 1, 1, 4, 1, 16384, 0, 65536, 16384, 2, 2, 2}},
    {{3, 4320, 10240, false, false, false, 20},
     {10, 2, 48, 10240, 4320, 4320, 24, 1024, 4320, 32, 256, 48, 32, 8, 1, 1, 16, 4, 32768, 0, 294912, 32768, 2, 2, 2}},
    {{3, 10240, 8640, false, false, false, 20},
     {8, 2, 48, 8640, 10240, 10240, 24, 1080, 10240, 32, 128, 64, 16, 4, 1, 1, 8, 2, 16384, 0, 131072, 16384, 2, 2, 2}},
    {{3, 4096, 27392, false, false, false, 20},
     {9, 2, 48, 27392, 4096, 4096, 24, 3044, 4096, 32, 256, 64, 32, 8, 1, 1, 16, 4, 32768, 0, 393216, 32768, 2, 2, 2}},
    {{256, 11088, 4096, false, false, true, 20},
     {4, 5, 4096, 4096, 11088, 11264,  832, 1024,   11088,  128, 256, 64, 16,
      8, 1, 1,    8,    4,     131072, 0,   524288, 131072, 2,   2,   1}},
    {{1, 5120, 27648, false, false, false, 20},
     {18, 1, 1, 27648, 5120, 5120, 16, 1536, 5120, 16, 128, 64, 160, 8, 1, 1, 80, 4, 16384, 1, 458752, 8192, 2, 2, 2}},
    {{1, 5120, 15360, false, false, false, 20},
     {20, 1, 1, 15360, 5120, 5120, 16, 768, 5120, 16, 64, 64, 80, 8, 1, 1, 40, 4, 8192, 0, 229376, 4096, 2, 2, 2}},
    {{1, 13824, 5120, false, false, false, 20},
     {20, 1, 1, 5120, 13824, 13824, 16, 256, 13824, 16, 64, 128, 108, 4, 1, 1, 54, 2, 16384, 0, 507904, 4096, 2, 2, 2}},
    {{1, 12288, 7808, false, false, true, 20},
     {19, 1, 1, 7808, 12288, 12288, 16, 411, 12288, 16, 32, 64, 64, 16, 1, 1, 32, 8, 4096, 0, 196608, 2048, 2, 2, 2}},
    {{1, 8192, 4096, true, false, false, 20},
     {20, 1, 1, 4096, 8192, 8192, 16, 205, 8192, 16, 256, 64, 16, 8, 1, 1, 8, 4, 32768, 0, 295424, 16384, 2, 2, 2}},
    {{1536, 10240, 8640, false, false, false, 20},
     {5, 4, 24576, 8640, 10240, 10240,  6144, 1728,   10240,  128, 256, 64, 16,
      8, 1, 1,     8,    4,     131072, 0,    524288, 131072, 2,   2,   1}},
    {{1536, 10240, 1536, false, false, false, 20},
     {3, 6, 24576, 1536, 10240, 10240,  4096, 512,    10240,  128, 256, 64, 16,
      8, 1, 1,     8,    4,     131072, 0,    524288, 131072, 2,   2,   1}},
    {{50, 6656, 1920, true, false, true, 20},
     {4, 5, 800, 1920, 6656, 6656, 160, 128, 6656, 160, 128, 64, 16, 8, 1, 1, 8, 4, 81920, 0, 491520, 81920, 2, 2, 1}},
    {{50, 6656, 11136, false, false, true, 20}, {4, 5, 800, 11136, 6656, 6656,   160, 192,    6656,   160, 192, 64, 16,
                                                 8, 1, 1,   8,     4,    122880, 0,   524288, 122880, 2,   2,   1}},
    {{50, 5568, 6656, false, false, true, 20}, {4, 5, 800, 6656, 5568, 5632,   160, 192,    5568,   160, 192, 64, 16,
                                                8, 1, 1,   8,    4,    122880, 0,   524288, 122880, 2,   2,   1}},
    {{75, 6656, 1920, true, false, true, 20}, {4, 5, 1200, 1920, 6656, 6656,   240, 128,    6656,   240, 128, 64, 8,
                                               8, 1, 1,    4,    4,    122880, 0,   409600, 122880, 2,   2,   1}},
    {{75, 6656, 11136, false, false, true, 20}, {4, 5, 1200, 11136, 6656, 6656,   240, 128,    6656,   240, 128, 64, 8,
                                                 8, 1, 1,    4,     4,    122880, 0,   376832, 122880, 2,   2,   1}},
    {{100, 1664, 6656, false, false, true, 20}, {4, 5, 1600, 6656, 1664, 1792,   320, 192,    1664,   160, 192, 64, 8,
                                                 8, 1, 1,    4,    4,    122880, 0,   360448, 122880, 2,   2,   1}},
    {{100, 6656, 11136, false, false, true, 20}, {4, 5, 1600, 11136, 6656, 6656,   320, 192,    6656,   160, 192, 64, 8,
                                                  8, 1, 1,    4,     4,    122880, 0,   360448, 122880, 2,   2,   1}},
    {{100, 5568, 6656, false, false, true, 20}, {4, 5, 1600, 6656, 5568, 5632,   320, 192,    5568,   160, 192, 64, 16,
                                                 8, 1, 1,    8,    4,    122880, 0,   524288, 122880, 2,   2,   1}},
};

const std::map<WhiteListShape, MatMulTilingCache> MM_NZ_TILING_CACHE = {
    {{64, 6912, 11264, false, false, true, 24}, {3, 8, 1024, 11264, 6912, 6912,   128, 3755,   6912,   128, 256, 64, 8,
                                                 8, 1, 1,    4,     4,    131072, 0,   393216, 131072, 2,   2,   1}},
    {{512, 2560, 5120, false, false, false, 24}, {3, 8, 8192, 5120, 2560, 2560,   1024, 1707,   2560,   128, 256, 64, 8,
                                                  8, 1, 1,    4,    4,    131072, 0,    393216, 131072, 2,   2,   1}},
    {{512, 5120, 2560, true, false, false, 24}, {3, 8, 8192, 2560, 5120, 5120,   1024, 854,    5120,   128, 256, 64, 8,
                                                 8, 1, 1,    4,    4,    131072, 0,    393728, 131072, 2,   2,   1}},
    {{256, 3584, 8192, false, false, true, 24}, {3, 8, 4096, 8192, 3584, 3584,   512, 2731,   3584,   128, 256, 64, 8,
                                                 8, 1, 1,    4,    4,    131072, 0,   393216, 131072, 2,   2,   1}},
    {{512, 640, 5120, false, false, false, 24},
     {4, 6, 8192, 5120, 640, 640, 1366, 1280, 640, 128, 256, 64, 8, 8, 1, 1, 4, 4, 131072, 1, 393216, 131072, 2, 2, 1}},
    {{64, 11264, 1664, true, false, true, 24}, {3, 8, 1024, 1664, 11264, 11264, 128, 555,    11264, 128, 192, 64, 8,
                                                8, 1, 1,    4,    4,     98304, 1,   328448, 98304, 2,   2,   1}},
    {{64, 11264, 6912, true, false, true, 24}, {3, 8, 1024, 6912, 11264, 11264,  128, 2304,   11264,  128, 256, 64, 8,
                                                8, 1, 1,    4,    4,     131072, 0,   394240, 131072, 2,   2,   1}},
    {{512, 5120, 1920, true, false, true, 24}, {4, 6, 8192, 1920, 5120, 5120,   1366, 480,    5120,   128, 240, 64, 8,
                                                8, 1, 1,    4,    4,    122880, 0,    376832, 122880, 2,   2,   1}},
    {{256, 8192, 32000, false, false, true, 24}, {3, 8, 4096, 32000, 8192, 8192,   512, 10667,  8192,   128, 256, 64, 8,
                                                  8, 1, 1,    4,     4,    131072, 0,   393216, 131072, 2,   2,   1}},
    {{256, 8192, 3584, false, false, true, 24}, {3, 8, 4096, 3584, 8192, 8192,   512, 1195,   8192,   128, 240, 64, 8,
                                                 8, 1, 1,    4,    4,    122880, 0,   376832, 122880, 2,   2,   1}},
    {{64, 1408, 11264, false, false, true, 20}, {5, 4, 1024, 11264, 1408, 1536,   256, 2253,   1408,   128, 256, 64, 8,
                                                 8, 1, 1,    4,     4,    131072, 0,   393216, 131072, 2,   2,   1}},
    {{64, 11264, 6912, true, false, true, 20}, {5, 4, 1024, 6912, 11264, 11264,  256, 1383,   11264,  128, 240, 64, 8,
                                                8, 1, 1,    4,    4,     122880, 0,   377792, 122880, 2,   2,   1}},
    {{64, 6912, 11264, false, false, true, 20}, {5, 4, 1024, 11264, 6912, 6912,   256, 2253,   6912,   128, 256, 64, 8,
                                                 8, 1, 1,    4,     4,    131072, 0,   393216, 131072, 2,   2,   1}},
    {{64, 6912, 11264, false, false, true, 20}, {5, 4, 1024, 11264, 6912, 6912,   256, 2253,   6912,   128, 256, 64, 8,
                                                 8, 1, 1,    4,     4,    131072, 0,   393216, 131072, 2,   2,   1}},
    {{256, 3584, 8192, false, false, true, 20}, {5, 4, 4096, 8192, 3584, 3584,   1024, 1639,   3584,   128, 240, 64, 8,
                                                 8, 1, 1,    4,    4,    122880, 0,    376832, 122880, 2,   2,   1}},
    {{512, 2560, 5120, false, false, false, 20}, {5, 4, 8192, 5120, 2560, 2560,   2048, 1024,   2560,   128, 256, 64, 8,
                                                  8, 1, 1,    4,    4,    131072, 0,    393216, 131072, 2,   2,   1}},
    {{512, 5120, 2560, true, false, false, 20}, {5, 4, 8192, 2560, 5120, 5120,   2048, 512,    5120,   128, 256, 64, 8,
                                                 8, 1, 1,    4,    4,    131072, 0,    393728, 131072, 2,   2,   1}},
    {{512, 5120, 1920, true, false, true, 20}, {4, 5, 8192, 1920, 5120, 5120,   1639, 480,    5120,   128, 240, 64, 8,
                                                8, 1, 1,    4,    4,    122880, 0,    377312, 122880, 2,   2,   1}},
    {{200, 2672, 11856, false, false, true, 24}, {8, 3, 3200, 11856, 2672, 2816,  1067, 1482,   2672,  192, 128, 64, 16,
                                                  8, 1, 1,    8,     4,    98304, 0,    524288, 98304, 2,   2,   1}},
    {{848, 3698, 14422, true, false, false, 24},
     {2, 12, 13566, 14422, 3698, 3712,   1131, 7211,   3698,   128, 256, 64, 16,
      4, 1,  1,     8,     2,    131072, 0,    393728, 131072, 2,   2,   1}},
    {{1003, 1415, 2991, false, false, false, 24},
     {12, 2, 16043, 2991, 1415, 1440,   8022, 250,    1415,   128, 256, 64, 16,
      4,  1, 1,     8,    2,    131072, 0,    393216, 131072, 2,   2,   1}},
    {{848, 3698, 14422, true, false, false, 20},
     {10, 2, 13566, 14422, 3698, 3712,   6783, 1443,   3698,   128, 256, 64, 16,
      2,  1, 1,     8,     1,    131072, 0,    328192, 131072, 2,   2,   1}},
    {{1003, 1415, 2991, false, false, false, 20},
     {2, 10, 16043, 2991, 1415, 1440,   1605, 1496,   1415,   128, 256, 64, 16,
      2, 1,  1,     8,    1,    131072, 0,    327680, 131072, 2,   2,   1}},
    {{256, 3584, 8192, false, false, false, 24}, {3, 8, 4096, 8192, 3584, 3584,   512, 2731,   3584,   128, 256, 64, 8,
                                                  8, 1, 1,    4,    4,    131072, 0,   393216, 131072, 2,   2,   1}},
    {{256, 8192, 32000, false, false, false, 24},
     {3, 8, 4096, 32000, 8192, 8192,   512, 10667,  8192,   128, 256, 64, 8,
      8, 1, 1,    4,     4,    131072, 0,   393216, 131072, 2,   2,   1}},
    {{256, 8192, 7168, false, false, false, 24}, {4, 6, 4096, 7168, 8192, 8192,   683, 1792,   8192,   128, 256, 64, 8,
                                                  8, 1, 1,    4,    4,    131072, 0,   393216, 313072, 2,   2,   1}},
    {{256, 1024, 8192, false, false, false, 24}, {3, 8, 4096, 8192, 1024, 1024,   512, 2731,   1024,   128, 256, 64, 8,
                                                  8, 1, 1,    4,    4,    131072, 0,   393216, 131072, 2,   2,   1}},
    {{256, 8192, 1280, false, false, false, 24}, {3, 8, 4096, 1280, 8192, 8192,   512, 427,    8192,   128, 256, 64, 8,
                                                  8, 1, 1,    4,    4,    131072, 1,   393216, 131072, 2,   2,   1}},
    {{256, 8192, 3584, false, false, false, 24}, {3, 8, 4096, 3584, 8192, 8192,   512, 1195,   8192,   128, 256, 64, 8,
                                                  8, 1, 1,    4,    4,    131072, 0,   393216, 131072, 2,   2,   1}},
    {{64, 11264, 1664, false, false, false, 24}, {3, 8, 1024, 1664, 11264, 11264,  128, 555,    11264,  128, 256, 64, 8,
                                                  8, 1, 1,    4,    4,     131072, 0,   393216, 131072, 2,   2,   1}},
    {{64, 11264, 6912, false, false, false, 24}, {3, 8, 1024, 6912, 11264, 11264,  128, 2304,   11264,  128, 256, 64, 8,
                                                  8, 1, 1,    4,    4,     131072, 0,   393216, 131072, 2,   2,   1}},
    {{64, 6912, 11264, false, false, false, 24}, {3, 8, 1024, 11264, 6912, 6912,   128, 3755,   6912,   128, 256, 64, 8,
                                                  8, 1, 1,    4,     4,    131072, 0,   393216, 131072, 2,   2,   1}},
    {{64, 1408, 11264, false, false, false, 24}, {6, 4, 1024, 11264, 1408, 1408,   256, 1878,   1408,   256, 128, 64, 8,
                                                  8, 1, 1,    4,     4,    131072, 0,   393216, 131072, 2,   2,   1}},
};

void WeightQuantBatchMatmulV2TilingCustom::Reset()
{
    cubeBaseN_ = static_cast<uint64_t>(BLOCK_CUBE);
}

/*
The function is limite of custom
1. not support antiquant scale dtype is uint64/int64
*/
bool WeightQuantBatchMatmulV2TilingCustom::IsCapable()
{
    OP_LOGI(opName_, "Begin check custom");
    OP_TILING_CHECK(((matmulInfoPtr_->antiQuantScaleDtype == ge::DT_UINT64) ||
                     (matmulInfoPtr_->antiQuantScaleDtype == ge::DT_INT64)),
                    OP_LOGI(opName_, "Custom do not support antiquant scale dtype is uint64 and int64"),
                    return false);
    if (matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ && matmulInfoPtr_->antiQuantType == QuantType::PER_GROUP) {
        OP_TILING_CHECK(matmulInfoPtr_->groupSize != 64 && matmulInfoPtr_->groupSize != 128,
                        OP_LOGI(opName_,
                        "Custom Nz only support group_size = 64 or 128 for per-group scene, but is [%lu]",
                        matmulInfoPtr_->groupSize),
                        return false);
        OP_TILING_CHECK(matmulInfoPtr_->kSize % matmulInfoPtr_->groupSize != 0,
                        OP_LOGI(opName_, "Custom Nz only support kSize align to group_size for per-group scene, "
                        "but kSize is [%lu], group_size is [%lu]", matmulInfoPtr_->kSize, matmulInfoPtr_->groupSize),
                        return false);
        OP_TILING_CHECK(matmulInfoPtr_->kSize % 64 != 0 && matmulInfoPtr_->nSize % 64 != 0,
                        OP_LOGI(opName_, "Custom Nz only support kSize and nSize align to 64 for per-group scene, "
                        "but kSize is [%lu], nSize is [%lu]", matmulInfoPtr_->kSize, matmulInfoPtr_->nSize),
                        return false);
        OP_TILING_CHECK(matmulInfoPtr_->transB,
                        OP_LOGI(opName_, "Custom Nz cannot support weight transpose for per-group scene"),
                        return false);
        OP_TILING_CHECK(matmulInfoPtr_->kSize > MAX_SHAPE_DIM || matmulInfoPtr_->nSize > MAX_SHAPE_DIM,
                        OP_LOGI(opName_, "Custom Nz only support and n < 65536 and k < 65536"),
                        return false);
    }
    OP_LOGI(opName_, "Check custom succ");
    return true;
}

ge::graphStatus WeightQuantBatchMatmulV2TilingCustom::DoOpTiling()
{
    OP_TILING_CHECK(InstantiateTilingData() == ge::GRAPH_FAILED,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "unable to get pointer of tiling data"),
                    return ge::GRAPH_FAILED);
    // Set shape dim and pad of tiling date
    SetShapeSize();
    OP_TILING_CHECK(
        !GetMatMulTiling(),
        VECTOR_INNER_ERR_REPORT_TILIING(opName_, "failed to get mm tiling for mnk[%ld, %ld, %ld]",
                                        matmulInfoPtr_->mSize, matmulInfoPtr_->nSize, matmulInfoPtr_->kSize),
        return ge::GRAPH_FAILED);

    uint64_t defaultVecSingleN = 0;
    uint64_t defaultVecSingleK = 0;
    if (matmulInfoPtr_->groupSize > 0) {
        ComputeGroupDefaultBlock(defaultVecSingleK, defaultVecSingleN);
    } else {
        ComputeDefaultBlock(defaultVecSingleK, defaultVecSingleN);
    }

    uint64_t vecSingleN = std::min(defaultVecSingleN, tilingData_->get_nAlign());
    uint64_t vecSingleK = std::min(defaultVecSingleK, tilingData_->get_kAlign());

    tilingData_->set_vecSingleN(static_cast<uint32_t>(vecSingleN));
    tilingData_->set_vecSingleK(static_cast<uint32_t>(vecSingleK));

    uint64_t totalCubeSingleN = cubeBaseN_ * tilingData_->get_cubeBlockDimN();
    totalCubeSingleN = std::min(totalCubeSingleN, tilingData_->get_nAlign());
    tilingData_->set_vecSingleNLoop(ops::CeilDiv(totalCubeSingleN, vecSingleN));
    tilingData_->set_vecSingleNTailLoop(
        ops::CeilDiv(CalcTailSize(matmulInfoPtr_->nSize, cubeBaseN_ * tilingData_->get_cubeBlockDimN()), vecSingleN));
    tilingData_->set_vecSingleKLoop(ops::CeilDiv(matmulInfoPtr_->kSize, vecSingleK));

    tilingData_->set_vecBlockDimK(1);
    uint64_t taskNum = tilingData_->get_vecSingleNLoop() * tilingData_->get_vecSingleKLoop();
    uint64_t singleCoreVecLoop = ops::CeilDiv(taskNum, static_cast<uint64_t>(compileInfoPtr_->aivNum));
    tilingData_->set_vecBlockDimN(ops::CeilDiv(taskNum, singleCoreVecLoop));
    return ge::GRAPH_SUCCESS;
}

void WeightQuantBatchMatmulV2TilingCustom::SetShapeSize()
{
    tilingData_->set_groupSize(matmulInfoPtr_->groupSize);
    uint64_t weightBlockAlignSize = GetBlockAlignSizeByDataType(matmulInfoPtr_->bDtype);
    if (matmulInfoPtr_->transB) {
        tilingData_->set_kAlign(ops::CeilAlign(matmulInfoPtr_->kSize, weightBlockAlignSize));
        if (matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) {
            tilingData_->set_nAlign(ops::CeilAlign(matmulInfoPtr_->nSize, static_cast<uint64_t>(BLOCK_CUBE)));
        } else {
            tilingData_->set_nAlign(matmulInfoPtr_->nSize);
        }
        tilingData_->set_kPadSize(static_cast<uint8_t>(tilingData_->get_kAlign() - matmulInfoPtr_->kSize));
    } else {
        if (matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) {
            tilingData_->set_kAlign(ops::CeilAlign(matmulInfoPtr_->kSize, static_cast<uint64_t>(BLOCK_CUBE)));
        } else {
            tilingData_->set_kAlign(matmulInfoPtr_->kSize);
        }
        tilingData_->set_nAlign(ops::CeilAlign(matmulInfoPtr_->nSize, weightBlockAlignSize));
        tilingData_->set_nPadSize(static_cast<uint8_t>(tilingData_->get_nAlign() - matmulInfoPtr_->nSize));
    }
    tilingData_->set_mSize(matmulInfoPtr_->mSize);
    tilingData_->set_nSize(matmulInfoPtr_->nSize);
    tilingData_->set_kSize(matmulInfoPtr_->kSize);
    // weightquantbmmv2 not support batch dims
    tilingData_->set_haveBatchA(0);
    tilingData_->set_haveBatchB(0);
    tilingData_->set_shapeBatch(1);
}

ge::graphStatus WeightQuantBatchMatmulV2TilingCustom::InstantiateTilingData()
{
    if (tilingData_ == nullptr) {
        OP_TILING_CHECK(isOutTilingData_,
                        VECTOR_INNER_ERR_REPORT_TILIING(opName_, "The out incoming tilingData is nullptr"),
                        return ge::GRAPH_FAILED);
        tilingDataManager_ = std::unique_ptr<WeightQuantBatchMatmulV2TilingData>(
            new (std::nothrow) WeightQuantBatchMatmulV2TilingData());
        tilingData_ = tilingDataManager_.get();
    }
    OP_TILING_CHECK(tilingData_ == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "failed to instantiate tilingData"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        context_->GetRawTilingData()->GetCapacity() < tilingData_->GetDataSize(),
        VECTOR_INNER_ERR_REPORT_TILIING(opName_, "tiling data capacity %zu < actual tiling data size %zu",
                                        context_->GetRawTilingData()->GetCapacity(), tilingData_->GetDataSize()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool WeightQuantBatchMatmulV2TilingCustom::GetMatMulTiling()
{
    if (!GetTilingFromCache() && !InvokeCacheTiling()) {
        auto mmInputDtype = GetMatmulTilingDtype(matmulInfoPtr_->aDtype);
        auto mmOutputDtype = GetMatmulTilingDtype(matmulInfoPtr_->cDtype);
        matmul_tiling::MultiCoreMatmulTiling mmTiling;
        matmul_tiling::CubeFormat bCubeFormat = (matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ)
                                                ? matmul_tiling::CubeFormat::NZ
                                                : matmul_tiling::CubeFormat::ND;
        mmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmInputDtype,
                          matmulInfoPtr_->transA);
        mmTiling.SetBType(matmul_tiling::TPosition::GM, bCubeFormat, mmInputDtype, matmulInfoPtr_->transB);
        mmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmOutputDtype);
        mmTiling.SetBias(matmulInfoPtr_->hasBias);
        if (matmulInfoPtr_->hasBias) {
            auto mmBiasDtype = GetMatmulTilingDtype(matmulInfoPtr_->biasDtype);
            mmTiling.SetBiasType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, mmBiasDtype);
        }
        mmTiling.SetDim(compileInfoPtr_->aicNum);
        // 转置场景内轴256对齐
        uint64_t kAlignSize = !matmulInfoPtr_->transB ? tilingData_->get_kAlign()
                              : ops::CeilAlign(tilingData_->get_kSize(), static_cast<uint64_t>(256));
        if (kAlignSize >= MAX_SHAPE_DIM) {
            kAlignSize = tilingData_->get_kSize();
        }
        mmTiling.SetOrgShape(matmulInfoPtr_->mSize, matmulInfoPtr_->nSize, matmulInfoPtr_->kSize, kAlignSize);
        mmTiling.SetShape(matmulInfoPtr_->mSize, matmulInfoPtr_->nSize, matmulInfoPtr_->kSize);
        mmTiling.SetSingleRange(-1, -1, -1, -1, -1, matmulInfoPtr_->kSize);
        mmTiling.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize);
        OP_TILING_CHECK(mmTiling.GetTiling(tilingData_->matmulTiling) == -1,
                        VECTOR_INNER_ERR_REPORT_TILIING(matmulInfoPtr_->opName, "failed to get matmul tiling"),
                        return false);

        auto mDim =
            ops::CeilDiv(matmulInfoPtr_->mSize, static_cast<uint64_t>(tilingData_->matmulTiling.get_singleCoreM()));
        auto nDim =
            ops::CeilDiv(matmulInfoPtr_->nSize, static_cast<uint64_t>(tilingData_->matmulTiling.get_singleCoreN()));
        OP_TILING_CHECK(mDim * nDim != static_cast<uint64_t>(tilingData_->matmulTiling.get_usedCoreNum()),
                        VECTOR_INNER_ERR_REPORT_TILIING(
                            matmulInfoPtr_->opName, "mDim(%lu) * nDim(%lu) != usedCoreNum(%d)",
                            mDim, nDim, tilingData_->matmulTiling.get_usedCoreNum()),
                        return false);
        tilingData_->set_cubeBlockDimN(static_cast<uint8_t>(nDim));
        tilingData_->set_cubeBlockDimM(static_cast<uint8_t>(mDim));
    }
    AdjustMatmulTiling();

    uint64_t singleCoreN = ops::CeilDiv(matmulInfoPtr_->nSize, static_cast<uint64_t>(tilingData_->get_cubeBlockDimN()));
    tilingData_->matmulTiling.set_singleCoreN(tilingData_->matmulTiling.get_baseN());
    cubeBaseN_ = static_cast<uint64_t>(tilingData_->matmulTiling.get_baseN());
    auto nDim = ops::CeilDiv(matmulInfoPtr_->nSize, ops::CeilAlign(singleCoreN, cubeBaseN_));
    tilingData_->set_cubeBlockDimN(static_cast<uint8_t>(nDim));
    return true;
}

void WeightQuantBatchMatmulV2TilingCustom::AdjustMatmulTiling() const
{
    if (matmulInfoPtr_->bFormat != ge::FORMAT_FRACTAL_NZ || matmulInfoPtr_->transB) {
        return;
    }
    int32_t baseN = tilingData_->matmulTiling.get_baseN();
    int32_t minCubeBaseN = ONE_BLK_SIZE;
    if (matmulInfoPtr_->bDtype == ge::DT_INT4) {
        minCubeBaseN = minCubeBaseN << 1;
    }
    if ((baseN * tilingData_->get_cubeBlockDimN()) % minCubeBaseN != 0) {
        tilingData_->matmulTiling.set_baseN(std::max(ops::FloorAlign(baseN, minCubeBaseN), minCubeBaseN));
        int32_t baseK = tilingData_->matmulTiling.get_baseK();
        if (tilingData_->matmulTiling.get_baseN() > baseN) {
            // baseN小于32，被向上对齐了，K要相应缩小并且向下对齐到16
            tilingData_->matmulTiling.set_baseK(
                std::max(ops::FloorAlign(tilingData_->matmulTiling.get_baseK() / TILING_COMPENSATION_FACTOR,
                                         static_cast<int32_t>(BLOCK_CUBE)), static_cast<int32_t>(BLOCK_CUBE)));
        }
        if (baseK == tilingData_->matmulTiling.get_baseK()) {
            // kl0没有缩小，就要缩小kL1; 如果stepKb为1时无法调整stepKb，改成调整stepN
            if (tilingData_->matmulTiling.get_stepKb() == 1) {
                tilingData_->matmulTiling.set_stepN(
                    std::max(tilingData_->matmulTiling.get_stepN() / TILING_COMPENSATION_FACTOR, 1));
            } else {
                tilingData_->matmulTiling.set_stepKb(
                    std::max(tilingData_->matmulTiling.get_stepKb() / TILING_COMPENSATION_FACTOR, 1));
            }
            tilingData_->matmulTiling.set_depthB1(
                std::max(tilingData_->matmulTiling.get_depthB1() / TILING_COMPENSATION_FACTOR, 1));
            if (tilingData_->matmulTiling.get_stepKb() > tilingData_->matmulTiling.get_stepKa() &&
                tilingData_->matmulTiling.get_stepKb() % tilingData_->matmulTiling.get_stepKa() != 0 &&
                tilingData_->matmulTiling.get_stepKb() * baseK < static_cast<int32_t>(tilingData_->get_kSize())) {
                tilingData_->matmulTiling.set_stepKb(
                    ops::FloorAlign(tilingData_->matmulTiling.get_stepKb(), tilingData_->matmulTiling.get_stepKa()));
            }
            if (tilingData_->matmulTiling.get_stepKa() > tilingData_->matmulTiling.get_stepKb() &&
                tilingData_->matmulTiling.get_stepKa() % tilingData_->matmulTiling.get_stepKb() != 0 &&
                tilingData_->matmulTiling.get_stepKa() * baseK < static_cast<int32_t>(tilingData_->get_kSize())) {
                tilingData_->matmulTiling.set_stepKa(
                    ops::FloorAlign(tilingData_->matmulTiling.get_stepKa(), tilingData_->matmulTiling.get_stepKb()));
            }
        } else {
            // kl0缩小了，相应的L1上k一定没全载，stepM和stepN只能为1
            tilingData_->matmulTiling.set_depthB1(tilingData_->matmulTiling.get_depthB1() /
                                                  tilingData_->matmulTiling.get_stepN());
            tilingData_->matmulTiling.set_depthA1(tilingData_->matmulTiling.get_depthA1() /
                                                  tilingData_->matmulTiling.get_stepM());
            tilingData_->matmulTiling.set_stepM(1);
            tilingData_->matmulTiling.set_stepN(1);
        }
        AdjustL1Size();
    }
}

void WeightQuantBatchMatmulV2TilingCustom::AdjustL1Size() const
{
    // 如果调整完之后l1size还是大于l1空间，则缩小stepM和depthA1
    uint64_t a1Length = static_cast<uint64_t>(GetShapeSizeWithDataType(
        tilingData_->matmulTiling.get_baseM() * tilingData_->matmulTiling.get_baseK(), matmulInfoPtr_->aDtype));
    uint64_t b1Length = static_cast<uint64_t>(GetShapeSizeWithDataType(
        tilingData_->matmulTiling.get_baseN() * tilingData_->matmulTiling.get_baseK(), matmulInfoPtr_->aDtype));
    uint64_t aL1Size = a1Length * tilingData_->matmulTiling.get_depthA1();
    uint64_t bL1Size = b1Length * tilingData_->matmulTiling.get_depthB1();
    uint64_t biasL1Size = matmulInfoPtr_->hasBias ? GetShapeSizeWithDataType(tilingData_->matmulTiling.get_baseN(),
                                                                             matmulInfoPtr_->biasDtype) : 0;
    uint64_t l1Size = aL1Size + bL1Size + biasL1Size;
    if (l1Size > aicoreParams_.l1Size) {
        tilingData_->matmulTiling.set_stepM(tilingData_->matmulTiling.get_stepM() / TILING_COMPENSATION_FACTOR);
        tilingData_->matmulTiling.set_depthA1(tilingData_->matmulTiling.get_depthA1() / TILING_COMPENSATION_FACTOR);
    }
}

void WeightQuantBatchMatmulV2TilingCustom::ComputeDefaultBlock(uint64_t &defaultVecSingleK, uint64_t &defaultVecSingleN)
{
    uint64_t defaultInnerAxis = matmulInfoPtr_->bDtype == ge::DT_INT8 ? 512 : 1024;
    uint64_t defaultOutterAxis = 32;

    // 非group场景，一次求解即可
    if (matmulInfoPtr_->transB) {
        defaultVecSingleN = defaultOutterAxis;
        // 保证mte2的带宽，根据weight的数据类型，默认载入量取512和1024
        defaultVecSingleK = matmulInfoPtr_->bDtype == ge::DT_INT8 ? 512 : 1024;
    } else {
        // weight不转置场景，n轴取值为cube一轮计算的n轴
        uint64_t weightInnerAxisAlignSize = ONE_BLK_SIZE / sizeof(matmulInfoPtr_->bDtype);

        if (matmulInfoPtr_->bDtype == ge::DT_INT4) {
            // int4场景, 内轴shape按照2倍的ONE_BLK_SIZE对齐
            weightInnerAxisAlignSize = ONE_BLK_SIZE * 2;
        }
        defaultVecSingleN = std::min(
            defaultInnerAxis, ops::CeilAlign(cubeBaseN_ * tilingData_->get_cubeBlockDimN(), weightInnerAxisAlignSize));
        defaultVecSingleK = defaultOutterAxis;
    }
    ComputeVectorDefaultBlock(defaultVecSingleK, defaultVecSingleN);
}

void WeightQuantBatchMatmulV2TilingCustom::ComputeGroupDefaultBlock(uint64_t &defaultVecSingleK,
                                                                    uint64_t &defaultVecSingleN)
{
    if (matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) {
        defaultVecSingleK = CUSTOM_NZ_TRANS_BF16_BASE_K;
        if (!matmulInfoPtr_->transB && tilingData_->get_nSize() > INT16_MAX) {
            defaultVecSingleK = matmulInfoPtr_->groupSize;
        }
        if (matmulInfoPtr_->aDtype == ge::DT_BF16 && matmulInfoPtr_->transB) {
            defaultVecSingleN = CUSTOM_NZ_NO_TRANS_BASE_N;
        } else {
            defaultVecSingleN = CUSTOM_NZ_TRANS_BASE_N;
        }
        return;
    }
    uint64_t defaultInnerAxis = matmulInfoPtr_->bDtype == ge::DT_INT8 ? 512 : 1024;
    uint64_t defaultOutterAxis = 32;
    if (matmulInfoPtr_->transB) {
        uint32_t repeatStrideMax = 255;
        uint32_t repeatAxisMax = repeatStrideMax * (ONE_BLK_SIZE / sizeof(matmulInfoPtr_->aDtype));
        if ( matmulInfoPtr_->aDtype == ge::DT_BF16) {
            repeatAxisMax = repeatStrideMax * (ONE_BLK_SIZE / sizeof(float));
        }
        tilingData_->set_repeatAxisMax(repeatAxisMax);
        if (tilingData_->get_kAlign() <= repeatAxisMax ||
            (tilingData_->get_kAlign() > repeatAxisMax && matmulInfoPtr_->groupSize <= repeatAxisMax &&
             tilingData_->get_kAlign() % matmulInfoPtr_->groupSize == 0)) {
            // k轴不会导致repeatStride超过限制，或者kAlign满足groupSize对齐的限制。考虑k全载，避免复杂尾块处理
            defaultVecSingleK = tilingData_->get_kAlign();
            ComputeVectorDefaultBlock(defaultVecSingleK, defaultVecSingleN);
        }
        if (defaultVecSingleN == 0) {
            OP_LOGD(opName_, "the K axis cannot full load, current defaultVecSingleK: [%lu], groupSize: [%lu].",
                    defaultVecSingleK, matmulInfoPtr_->groupSize);
            // k无法全载的情况下，需重新设置k轴载入量, 同时保证mte2的带宽，根据weight的数据类型，默认载入量取512和1024
            defaultVecSingleK = matmulInfoPtr_->bDtype == ge::DT_INT8 ? 512 : 1024;
            if (defaultVecSingleK >= matmulInfoPtr_->groupSize) {
                defaultVecSingleK = defaultVecSingleK / matmulInfoPtr_->groupSize * matmulInfoPtr_->groupSize;
            }
            ReviseGroupDefaultBlockWithTrans(defaultVecSingleK, defaultVecSingleN);
        }
    } else {
        // weight不转置场景，n轴取值为cube一轮计算的n轴
        uint64_t weightInnerAxisAlignSize = ONE_BLK_SIZE / sizeof(matmulInfoPtr_->bDtype);

        if (matmulInfoPtr_->bDtype == ge::DT_INT4) {
            // int4场景, 内轴shape按照32Byte的2倍对齐
            weightInnerAxisAlignSize = ONE_BLK_SIZE * 2;
        }
        defaultVecSingleN = std::min(
            defaultInnerAxis, ops::CeilAlign(cubeBaseN_ * tilingData_->get_cubeBlockDimN(), weightInnerAxisAlignSize));
        defaultVecSingleK = ops::CeilDiv(defaultOutterAxis, matmulInfoPtr_->groupSize) * matmulInfoPtr_->groupSize;
        ComputeVectorDefaultBlock(defaultVecSingleK, defaultVecSingleN);
        ReviseGroupDefaultBlockWithoutTrans(defaultVecSingleK, defaultVecSingleN);
    }
}

void WeightQuantBatchMatmulV2TilingCustom::ReviseGroupDefaultBlockWithTrans(uint64_t &defaultVecSingleK,
                                                                            uint64_t &defaultVecSingleN)
{
    for (; defaultVecSingleK > matmulInfoPtr_->groupSize; defaultVecSingleK -= matmulInfoPtr_->groupSize) {
        ComputeVectorDefaultBlock(defaultVecSingleK, defaultVecSingleN);
        if (defaultVecSingleN > 0) {
            // n轴大于0,表明该解为合法解，提前退出
            return;
        }
    }

    for (; defaultVecSingleK >= MIN_GROUP_SIZE; defaultVecSingleK -= MIN_GROUP_SIZE) {
        if (matmulInfoPtr_->groupSize % defaultVecSingleK != 0) {
            // 合适的k轴必须满足groupSize_因子的关系
            continue;
        }
        ComputeVectorDefaultBlock(defaultVecSingleK, defaultVecSingleN);
        if (defaultVecSingleN > 0) {
            // 求得一个合法解，提前退出
            return;
        }
    }
}

void WeightQuantBatchMatmulV2TilingCustom::ReviseGroupDefaultBlockWithoutTrans(uint64_t &defaultVecSingleK,
                                                                               uint64_t &defaultVecSingleN)
{
    while (defaultVecSingleN > 0) {
        // 若groupSize比MAX_REPEAT_TIMES大，则k对齐到groupSize后必定不满足小于MAX_REPEAT_TIMES的要求，
        // 因此排除这种情况下对k的修正
        if (matmulInfoPtr_->groupSize < MAX_REPEAT_TIMES && defaultVecSingleK >= matmulInfoPtr_->groupSize) {
            // 不转置场景下，k在向groupSize取整后应保证小于MAX_REPEAT_TIMES
            defaultVecSingleK =
                std::min(MAX_REPEAT_TIMES / matmulInfoPtr_->groupSize, defaultVecSingleK / matmulInfoPtr_->groupSize) *
                matmulInfoPtr_->groupSize;
            return;
        }
        for (uint32_t targetK = matmulInfoPtr_->groupSize; targetK >= MIN_GROUP_SIZE; targetK -= MIN_GROUP_SIZE) {
            // 合法的k值在不转置场景下应满足小于MAX_REPEAT_TIMES的限制
            if (targetK > MAX_REPEAT_TIMES) {
                continue;
            }

            // 合法的k值需要满足为groupSize的因子
            if (matmulInfoPtr_->groupSize % targetK != 0) {
                continue;
            }
            if (targetK <= defaultVecSingleK) {
                defaultVecSingleK = targetK;
                return;
            }
        }

        // 无法搜索到满足条件的k值，尝试缩小n重新搜索
        defaultVecSingleN = defaultVecSingleN >> 1;
        ComputeVectorDefaultBlock(defaultVecSingleK, defaultVecSingleN);
    }
}

void WeightQuantBatchMatmulV2TilingCustom::ComputeVectorDefaultBlock(uint64_t &defaultVecSingleK,
                                                                     uint64_t &defaultVecSingleN)
{
    /*
        整体vec处理的基本块推导应该满足如下公式：antiquantBufferSize + weightBufferSize < ubSize
        group场景，固定k轴求n，基本公式化简为：n = ubSize * gs / (antiquantCoefficient * k + weightCoefficient * k *
        gs) 非group场景，固定k轴求n，基本公式进一步化简为：n = ub / (antiquantCoefficient + weightCoefficient * k)
        int4场景，weightCoefficient涉及除2操作，因此先放大weightCoefficient 2倍再除2。避免浮点数的系数影响
    */
    if (matmulInfoPtr_->groupSize > 0 || matmulInfoPtr_->bDtype == ge::DT_INT4) {
        ComputeInt4VectorDefaultBlock(defaultVecSingleK, defaultVecSingleN);
    } else {
        ComputeInt8VectorDefaultBlock(defaultVecSingleK, defaultVecSingleN);
    }
}

void WeightQuantBatchMatmulV2TilingCustom::ComputeInt4VectorDefaultBlock(uint64_t &defaultVecSingleK,
                                                                         uint64_t &defaultVecSingleN)
{
    if (matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) {
        if (matmulInfoPtr_->aDtype == ge::DT_BF16) {
            defaultVecSingleN = 64; // 固定切分，n取64
            defaultVecSingleK = 256; // 固定切分，k取256
        } else {
            if (matmulInfoPtr_->transB) {
                defaultVecSingleN = 64; // 固定切分，n取64
                defaultVecSingleK = 384; // 固定切分，k取384
            } else {
                defaultVecSingleN = 64; // 固定切分，n取64
                defaultVecSingleK = 256; // 固定切分，k取256
            }
        }
        return;
    }
    uint64_t start = 0;
    uint64_t length = matmulInfoPtr_->kSize + 1;
    if (matmulInfoPtr_->transB) {
        length = matmulInfoPtr_->nSize + 1;
    }

    // 固定内轴的情况下，二分求解最大的外轴是多少
    while (length > 0) {
        uint64_t mid = start + (length >> 1);
        uint64_t antiquantBuffer;
        uint64_t weightBuffer;
        if (matmulInfoPtr_->transB) {
            antiquantBuffer = ComputeAntiquantBuffer(defaultVecSingleK, mid);
            weightBuffer = ComputeWeightBuffer(defaultVecSingleK, mid);
        } else {
            antiquantBuffer = ComputeAntiquantBuffer(mid, defaultVecSingleN);
            weightBuffer = ComputeWeightBuffer(mid, defaultVecSingleN);
        }

        if (aicoreParams_.ubSize < antiquantBuffer + weightBuffer) {
            length = length >> 1;
        } else {
            start = mid + 1;
            length = length - (length >> 1) - 1;
        }
    }

    // start是不满足条件的最小值，因此最终结果需要-1
    if (matmulInfoPtr_->transB) {
        defaultVecSingleN = start - 1;
    } else {
        defaultVecSingleK = start - 1;
    }
}

uint64_t WeightQuantBatchMatmulV2TilingCustom::ComputeAntiquantBuffer(uint64_t &defaultVecSingleK,
                                                                      uint64_t &defaultVecSingleN)
{
    uint64_t aDtypeBlockSize = GetBlockAlignSizeByDataType(matmulInfoPtr_->aDtype);
    uint64_t antiquantSize = ops::CeilAlign(defaultVecSingleN, aDtypeBlockSize);
    if (matmulInfoPtr_->groupSize > 0) {
        if (matmulInfoPtr_->transB) {
            if (defaultVecSingleK >= matmulInfoPtr_->kSize) {
                // 全载场景，antiquant的n*gourpCount合并成一根轴计算
                antiquantSize = ops::CeilAlign(
                    ops::CeilDiv(defaultVecSingleK, matmulInfoPtr_->groupSize) * defaultVecSingleN, aDtypeBlockSize);
            } else {
                // 非全载场景，antiquant的shape只能当作(n, gourpCount)计算，同时考虑内轴对齐
                antiquantSize =
                    defaultVecSingleN *
                    ops::CeilAlign(ops::CeilDiv(defaultVecSingleK, matmulInfoPtr_->groupSize), aDtypeBlockSize);
            }
        } else {
            // 不转置场景，antiquant的shape只能当作(gourpCount，n)计算，同时考虑内轴对齐
            antiquantSize = ops::CeilDiv(defaultVecSingleK, matmulInfoPtr_->groupSize) *
                            ops::CeilAlign(defaultVecSingleN, aDtypeBlockSize);
        }
    }

    // scale和offset两个入参，需要占用2份空间
    uint64_t antiquantParamsCount = 2;
    uint64_t antiquantInQueSize = antiquantParamsCount * antiquantSize * sizeof(matmulInfoPtr_->aDtype);
    if (matmulInfoPtr_->transB) {
        if (matmulInfoPtr_->aDtype == ge::DT_BF16) {
            return antiquantInQueSize + antiquantSize * sizeof(float) +
                   antiquantParamsCount * antiquantSize * ONE_BLK_SIZE;
        } else {
            return antiquantInQueSize + antiquantParamsCount * antiquantSize * ONE_BLK_SIZE;
        }
    } else {
        if (matmulInfoPtr_->aDtype == ge::DT_BF16) {
            return antiquantInQueSize + antiquantSize * sizeof(float) +
                   antiquantParamsCount * antiquantSize * sizeof(float);
        } else {
            return antiquantInQueSize + antiquantParamsCount * antiquantSize * sizeof(matmulInfoPtr_->aDtype);
        }
    }
}

uint64_t WeightQuantBatchMatmulV2TilingCustom::ComputeWeightBuffer(uint64_t defaultVecSingleK,
                                                                   uint64_t defaultVecSingleN)
{
    uint64_t originWeightAlignAxis = ONE_BLK_SIZE / sizeof(matmulInfoPtr_->bDtype);
    if (matmulInfoPtr_->bDtype == ge::DT_INT4) {
        // int4场景，内轴的长度是32Byte的2倍
        originWeightAlignAxis = ONE_BLK_SIZE * 2;
    }

    uint64_t weightShape;
    if (matmulInfoPtr_->transB) {
        weightShape = defaultVecSingleN * ops::CeilAlign(defaultVecSingleK, originWeightAlignAxis);
    } else {
        weightShape = defaultVecSingleK * ops::CeilAlign(defaultVecSingleN, originWeightAlignAxis);
    }
    uint64_t originWeightSize = weightShape * sizeof(matmulInfoPtr_->bDtype);
    if (matmulInfoPtr_->bDtype == ge::DT_INT4) {
        originWeightSize = originWeightSize >> 1;
    }
    uint64_t weight16Size = weightShape * sizeof(matmulInfoPtr_->aDtype);
    uint64_t weight32Size = weightShape * sizeof(float);
    // 输出的buffer共有2份，方便开db
    uint64_t weightOutSize = 2 * weight16Size;

    if (matmulInfoPtr_->aDtype == ge::DT_BF16) {
        return originWeightSize + weight16Size + weight32Size + weightOutSize;
    } else {
        return originWeightSize + weight16Size + weightOutSize;
    }
}

void WeightQuantBatchMatmulV2TilingCustom::ComputeInt8VectorDefaultBlock(uint64_t &defaultVecSingleK,
                                                                         uint64_t &defaultVecSingleN) const
{
    if (matmulInfoPtr_->bFormat != ge::FORMAT_FRACTAL_NZ) {
        if (matmulInfoPtr_->aDtype == ge::DT_BF16) {
            if (matmulInfoPtr_->transB) {
                // 需要11nk + 76n空间，内轴优先512B对齐，根据k=512和总UB空间，计算出n=32
                defaultVecSingleN = 32;
                defaultVecSingleK = BASIC_BLOCK;
            } else {
                // 需要11nk + 12n空间，内轴优先512B对齐，根据k=512和总UB空间，计算出n=32
                defaultVecSingleK = 32;
                defaultVecSingleN = BASIC_BLOCK;
            }
        } else {
            if (matmulInfoPtr_->transB) {
                // 需要7nk + 68n空间，内轴优先512B对齐，根据k=512和总UB空间，计算出n=40
                defaultVecSingleN = 40;
                defaultVecSingleK = BASIC_BLOCK;
            } else {
                // 需要7nk + 4n空间，内轴优先512B对齐，根据k=512和总UB空间，计算出n=48
                defaultVecSingleK = 48;
                defaultVecSingleN = BASIC_BLOCK;
            }
        }
    } else {
        if (matmulInfoPtr_->aDtype == ge::DT_BF16) {
            if (matmulInfoPtr_->transB) {
                defaultVecSingleN = CUSTOM_NZ_TRANS_BASE_N;
                defaultVecSingleK = CUSTOM_NZ_TRANS_BF16_BASE_K;
            } else {
                defaultVecSingleN = CUSTOM_NZ_NO_TRANS_BASE_N;
                defaultVecSingleK = CUSTOM_NZ_NO_TRANS_BF16_BASE_N;
            }
        } else {
            if (matmulInfoPtr_->transB) {
                defaultVecSingleN = CUSTOM_NZ_TRANS_BASE_N;
                defaultVecSingleK = CUSTOM_NZ_TRANS_FP16_BASE_K;
            } else {
                defaultVecSingleN = CUSTOM_NZ_NO_TRANS_BASE_N;
                defaultVecSingleK = CUSTOM_NZ_NO_TRANS_FP16_BASE_K;
            }
        }
    }
}

ge::graphStatus WeightQuantBatchMatmulV2TilingCustom::DoLibApiTiling()
{
    uint64_t cubeBlockDimN = static_cast<uint64_t>(tilingData_->get_cubeBlockDimN());
    uint64_t cubeEachCoreN = ops::CeilAlign(ops::CeilDiv(matmulInfoPtr_->nSize, cubeBlockDimN), cubeBaseN_);
    tilingData_->set_cubeSingleNLoop(ops::CeilDiv(cubeEachCoreN, cubeBaseN_));
    tilingData_->set_cubeSingleNTailLoop(
        ops::CeilDiv(matmulInfoPtr_->nSize - cubeEachCoreN * (cubeBlockDimN - 1), cubeBaseN_));
    tilingData_->set_cubeTailM(
        CalcTailSize(matmulInfoPtr_->mSize, static_cast<uint64_t>(tilingData_->matmulTiling.get_singleCoreM())));
    tilingData_->set_cubeTailN(
        CalcTailSize(matmulInfoPtr_->nSize, static_cast<uint64_t>(tilingData_->matmulTiling.get_baseN())));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantBatchMatmulV2TilingCustom::GetWorkspaceSize()
{
    // weight的缓存最多使用3份空间，实际划分少于3时以实际划分为准
    uint64_t weightCacheCount = std::min(static_cast<uint32_t>(3), tilingData_->get_cubeSingleNLoop());
    uint64_t weightCacheNSize = tilingData_->matmulTiling.get_singleCoreN() * tilingData_->get_cubeBlockDimN();
    if (!matmulInfoPtr_->transB && matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) {
        weightCacheNSize = ops::CeilAlign(weightCacheNSize, static_cast<uint64_t>(ONE_BLK_SIZE));
    }
    uint64_t weightCacheSize = tilingData_->get_kAlign() * weightCacheNSize;
    if (matmulInfoPtr_->transB) {
        // 内轴需256对齐以提高nd2nz效率
        weightCacheSize = ops::CeilAlign(tilingData_->get_kSize(), static_cast<uint64_t>(256)) *
                          tilingData_->matmulTiling.get_singleCoreN() * tilingData_->get_cubeBlockDimN();
    }
    // 向256对齐，可以保证workspace起始地址保证512B对齐，提升mte3性能
    uint64_t weightCacheAlignSize = ops::CeilDiv(weightCacheSize, static_cast<uint64_t>(256)) * 256;
    workspaceSize_ = weightCacheAlignSize * weightCacheCount * ge::GetSizeByDataType(matmulInfoPtr_->aDtype) +
                     compileInfoPtr_->workspaceNum;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantBatchMatmulV2TilingCustom::PostTiling()
{
    OP_LOGD(opName_, "final tiling data size: %zu", tilingData_->GetDataSize());

    OP_TILING_CHECK(tilingData_->GetDataSize() % sizeof(uint64_t) != 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "tiling data size[%zu] not aligned to 8",
                                                    tilingData_->GetDataSize()),
                    return ge::GRAPH_FAILED);
    context_->GetRawTilingData()->SetDataSize(tilingData_->GetDataSize());
    uint32_t usedAicNum = tilingData_->get_cubeBlockDimM() * tilingData_->get_cubeBlockDimN();
    uint32_t usedAivNum = tilingData_->get_vecBlockDimK() * tilingData_->get_vecBlockDimN();
    context_->SetBlockDim(std::max(usedAicNum, CalcTschBlockDim(
        usedAivNum, compileInfoPtr_->aicNum, compileInfoPtr_->aivNum)));

    size_t *workspaces = context_->GetWorkspaceSizes(1);  // set workspace
    workspaces[0] = workspaceSize_;

    tilingData_->SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    return ge::GRAPH_SUCCESS;
}

uint64_t WeightQuantBatchMatmulV2TilingCustom::GetTilingKey() const
{
    KernelTemplateType templateType = matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ
                                          ? KernelTemplateType::WEIGHT_NZ
                                          : KernelTemplateType::CUSTOM_ANTIQUANT;
    return RecursiveSum(matmulInfoPtr_->transA, matmulInfoPtr_->transB, matmulInfoPtr_->antiQuantType,
                        matmulInfoPtr_->hasAntiQuantOffset, matmulInfoPtr_->quantType, templateType);
}

bool WeightQuantBatchMatmulV2TilingCustom::GetTilingFromCache()
{
    uint64_t mMatchSize = ops::CeilDiv(matmulInfoPtr_->mSize, static_cast<uint64_t>(BLOCK_CUBE));
    WhiteListShape shape({mMatchSize, matmulInfoPtr_->kSize, matmulInfoPtr_->nSize, matmulInfoPtr_->hasBias,
                          matmulInfoPtr_->transA, matmulInfoPtr_->transB, compileInfoPtr_->aicNum});
    auto mmCache = matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ ? MM_NZ_TILING_CACHE : MM_TILING_CACHE;
    auto it = mmCache.find(shape);
    if (it == mmCache.end()) {
        OP_LOGD(opName_, "not find mm tiling from cache");
        return false;
    }

    OP_LOGD(opName_, "get mm tiling from cache");
    auto &matmulTilingCache = it->second;
    matmulTilingCache.SetMatmulTilingFromCacheData(
        tilingData_->matmulTiling, matmulInfoPtr_->mSize, matmulInfoPtr_->nSize,
        static_cast<int32_t>(matmulInfoPtr_->hasBias));
    tilingData_->set_cubeBlockDimM(matmulTilingCache.mDim_);
    tilingData_->set_cubeBlockDimN(matmulTilingCache.nDim_);
    return true;
}

bool WeightQuantBatchMatmulV2TilingCustom::CheckCacheTiling()
{
    if (matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) {
        int32_t kAL1Loop = ops::CeilDiv(tilingData_->matmulTiling.get_singleCoreK(),
                                        tilingData_->matmulTiling.get_baseK() * tilingData_->matmulTiling.get_stepKa());
        int32_t kBL1Loop = ops::CeilDiv(tilingData_->matmulTiling.get_singleCoreK(),
                                        tilingData_->matmulTiling.get_baseK() * tilingData_->matmulTiling.get_stepKb());
        if (kAL1Loop == 0 || kBL1Loop == 0) {
            return false;
        }
        if (kAL1Loop % kBL1Loop != 0 && kBL1Loop % kAL1Loop != 0) {
            return false;
        }
    }
    // 拦截分核数小于0.5倍总核数的解
    OP_TILING_CHECK(tilingData_->get_cubeBlockDimM() * tilingData_->get_cubeBlockDimN() < 0.5 * compileInfoPtr_->aicNum,
                    OP_LOGI(opName_, "Current cache tiling result is aborted for insufficient core use"),
                    return false);

    OP_LOGD(opName_, "get and convert cache tiling success");
    return true;
}

bool WeightQuantBatchMatmulV2TilingCustom::InvokeCacheTiling()
{
    MatmulMultiCoreResult multiCoreResult;
    bool result = ComputeMatmulTiling::GetTiling(
        tilingData_->matmulTiling, multiCoreResult,
        {matmulInfoPtr_->mSize, matmulInfoPtr_->kSize, matmulInfoPtr_->nSize, matmulInfoPtr_->aDtype,
         matmulInfoPtr_->bDtype, matmulInfoPtr_->cDtype, matmulInfoPtr_->biasDtype, matmulInfoPtr_->transA,
         matmulInfoPtr_->transB, matmulInfoPtr_->hasBias, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
         matmulInfoPtr_->quantType, true},
        aicoreParams_, context_);

    OP_LOGI_IF_RETURN(!result, false, opName_, "cannot get tiling from cachetiling, mnk[%lu, %lu, %lu]",
                      matmulInfoPtr_->mSize, matmulInfoPtr_->kSize, matmulInfoPtr_->nSize);

    tilingData_->set_cubeBlockDimM(static_cast<uint8_t>(multiCoreResult.mDim));
    tilingData_->set_cubeBlockDimN(static_cast<uint8_t>(multiCoreResult.nDim));
    tilingData_->set_blockBatch(static_cast<uint8_t>(multiCoreResult.batchDim));

    return CheckCacheTiling();
}
}  // namespace optiling

