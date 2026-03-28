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
 * \file conv3d_tiling_util.h
 * \brief
 */

#ifndef ASCENDC_TIKCFW_TILING_CONV3D_TILING_UTIL_H
#define ASCENDC_TIKCFW_TILING_CONV3D_TILING_UTIL_H

#include <iostream>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include "tiling/platform/platform_ascendc.h"

namespace conv3d_tiling {

#ifdef ASC_OP_DEBUG_TEST
#define TILING_CUBE_LOG(level, format, ...)                               \
    do {                                                                \
        fprintf(stdout, "[LOG] %s " format "\n", level, ##__VA_ARGS__); \
    } while (0)
#else
#define TILING_CUBE_LOG(level, format, ...)
#endif

#define TILING_DEBUG_LOG(format, ...) TILING_CUBE_LOG("[DEBUG]", format, ##__VA_ARGS__)
#define TILING_INFO_LOG(format, ...) TILING_CUBE_LOG("[INFO] ", format, ##__VA_ARGS__)
#define TILING_WARNING_LOG(format, ...) TILING_CUBE_LOG("[WARN] ", format, ##__VA_ARGS__)
#define TILING_ERROR_LOG(format, ...) TILING_CUBE_LOG("[ERROR]", format, ##__VA_ARGS__)

enum class ConvDtype {
    FLOAT16 = 0,
    FLOAT32,
    BF16,
    INT4,
    INT8,
    UINT8,
    INT32,
    INT64,
    UINT64,
    CONVDTYPEMAX
};

enum class ConvFormat {
    ND = 0,
    NCHW,
    NHWC,
    HWCN,
    DHWNC,
    DHWCN,
    NDHWC,
    NCDHW,
    NC1HWC0,
    NDC1HWC0,
    FRACTAL_Z_3D
};

enum class TPosition {
    GM = 0,
    A1,
    A2,
    B1,
    B2,
    C1,
    C2,
    CO1,
    CO2,
    VECIN,
    VECOUT,
    VECCALC,
    LCM = VECCALC,
    SPM,
    SHM = SPM,
    TSCM,
    MAX
};

enum class IterateMNOrder {
    ITER_M_FST = 0,
    ITER_N_FST,
    INVALID
};

const std::map<ConvDtype, std::string> g_dtypeToStr = {
    {ConvDtype::FLOAT16, "float16"},
    {ConvDtype::FLOAT32, "float32"},
    {ConvDtype::BF16, "bfloat16"},
    {ConvDtype::INT4, "int4"},
    {ConvDtype::INT8, "int8"},
    {ConvDtype::UINT8, "uint8"},
    {ConvDtype::INT64, "int64"},
    {ConvDtype::UINT64, "uint64"},
    {ConvDtype::INT32, "int32"},
};

const std::map<ConvFormat, std::string> g_formatToStr = {
    {ConvFormat::NCHW, "NCHW"},
    {ConvFormat::NHWC, "NHWC"},
    {ConvFormat::HWCN, "HWCN"},
    {ConvFormat::DHWNC, "DHWNC"},
    {ConvFormat::DHWCN, "DHWCN"},
    {ConvFormat::NDHWC, "NDHWC"},
    {ConvFormat::NCDHW, "NCDHW"},
    {ConvFormat::NC1HWC0, "NC1HWC0"},
    {ConvFormat::ND, "ND"}
};

const std::map<ConvDtype, uint32_t> g_dtypeSizeTab = {
    {ConvDtype::FLOAT16, 2},
    {ConvDtype::FLOAT32, 4},
    {ConvDtype::BF16, 2},
    {ConvDtype::INT8, 1},
    {ConvDtype::UINT8, 1},
    {ConvDtype::INT64, 8},
    {ConvDtype::UINT64, 8},
    {ConvDtype::INT32, 4}
};

constexpr uint32_t COUNT_PARAMS_BIAS_SCALE = 5; // [fmap, weight, bias, scale, output]
constexpr uint32_t COUNT_PARAMS_BIAS = 4; // [fmap, weight, bias, output]
constexpr uint32_t COUNT_PARAMS_NO_BIAS = 3; // [fmap, weight, output]

const std::vector<std::vector<ConvDtype>> SUPPORTED_TYPES_WITH_BIAS_SCALE = {
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::BF16},
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::BF16, ConvDtype::FLOAT32, ConvDtype::BF16}
};

const std::vector<std::vector<ConvDtype>> SUPPORTED_TYPES_WITH_BIAS = {
    {ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16},
    {ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::FLOAT32},
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::INT32, ConvDtype::FLOAT16},
    {ConvDtype::BF16, ConvDtype::BF16, ConvDtype::FLOAT32, ConvDtype::BF16}
};
const std::vector<std::vector<ConvDtype>> SUPPORTED_TYPES_WITHOUT_BIAS = {
    {ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16},
    {ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::FLOAT32},
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::FLOAT16},
    {ConvDtype::BF16, ConvDtype::BF16, ConvDtype::BF16}
};

const std::vector<std::vector<ConvDtype>> POINTWISE_SUPPORTED_TYPES_WITH_BIAS = {
    {ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT32, ConvDtype::FLOAT16},
    {ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::FLOAT32},
    {ConvDtype::BF16, ConvDtype::BF16, ConvDtype::FLOAT32, ConvDtype::BF16}
};
const std::vector<std::vector<ConvDtype>> POINTWISE_SUPPORTED_TYPES_WITHOUT_BIAS = {
    {ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16},
    {ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::FLOAT32},
    {ConvDtype::BF16, ConvDtype::BF16, ConvDtype::BF16}
};

struct ConvType {
    ConvFormat format;
    ConvDtype dtype;
    TPosition pos;
};

struct PlatformInfo {
    platform_ascendc::SocVersion socVersion;
    uint64_t l1Size = 0;
    uint64_t l0CSize = 0;
    uint64_t ubSize = 0;
    uint64_t l0ASize = 0;
    uint64_t l0BSize = 0;
    uint64_t btSize = 0;
    uint64_t fbSize = 0;
};

struct Conv3DOriGroupInfo {
    int64_t groups = -1L;
    int64_t cin = -1L;
    int64_t cout = -1L;
    ConvDtype dtype = ConvDtype::CONVDTYPEMAX;
};

struct Conv3DGroupOptInfo {
    int64_t groupOpt = -1L;
    int64_t cinOpt = -1L;
    int64_t coutOpt = -1L;
};

constexpr uint64_t DOUBLE_BUFFER_NUM = 2;
constexpr int8_t M_Mode = 0;
constexpr int8_t HW_Mode = 1;

constexpr uint32_t L1_SIZE = 524288;
constexpr uint32_t L0A_SIZE = 65536;
constexpr uint32_t L0B_SIZE = 65536;
constexpr uint32_t L0C_SIZE = 131072;
constexpr uint32_t UB_SIZE = 262144;
constexpr uint32_t BT_SIZE = 1024;
constexpr uint32_t FB_SIZE = 2048;

constexpr uint32_t C0_BYTE_SIZE = 32;
constexpr uint32_t MIN_BURST_SIZE = 128;
constexpr uint32_t LOAD3D_MAX_STRIDE_H_W = 63;
constexpr uint32_t LOAD3D_MAX_DILATION_H_W = 255;
constexpr uint32_t LOAD3D_MAX_PAD = 255;
constexpr uint32_t LOAD3D_MAX_FILTER_H_W = 511;
constexpr uint32_t LOAD3D_MAX_DDR2L1_SIZE = 65535;

constexpr uint32_t CUBE_UNIT = 16;
constexpr uint32_t FP16_CUBE_UNIT = 16;
constexpr uint32_t FP32_CUBE_UNIT = 8;
constexpr uint32_t INT8_CUBE_UNIT = 32;
constexpr uint32_t MKN_MAX_SIZE = 3;
constexpr uint32_t MKN_M_INDEX = 0;
constexpr uint32_t MKN_K_INDEX = 1;
constexpr uint32_t MKN_N_INDEX = 2;
constexpr uint32_t MKN_VALUE_DEFAULT = 16;

constexpr uint64_t MAX_64_BIT_NUM = 0xFFFFFFFFFFFFFFFFU;

constexpr uint32_t CONST_VALUE_2 = 2;
constexpr uint32_t CONST_HO_1 = 1;
constexpr uint32_t C0_SIZE = 32;

// load3dv2 postk limit
constexpr uint32_t POSTK_LIMIT = 65535;

struct AscendApiCubeTypeMap {
public:
    struct {
        ConvDtype madType;
        ConvDtype biasType;
    } typeMaps[static_cast<uint8_t>(ConvDtype::CONVDTYPEMAX) + 1] = {
        {ConvDtype::CONVDTYPEMAX, ConvDtype::CONVDTYPEMAX}
    };
    
    ConvDtype ToBiasType(ConvDtype type) const
    {
        return typeMaps[static_cast<uint8_t>(type)].biasType;
    }
    ConvDtype ToMadType(ConvDtype type) const
    {
        return typeMaps[static_cast<uint8_t>(type)].madType;
    }

    AscendApiCubeTypeMap()
    {
        SetMapping(ConvDtype::INT4, ConvDtype::INT32, ConvDtype::INT32);
        SetMapping(ConvDtype::INT8, ConvDtype::INT32, ConvDtype::INT32);
        SetMapping(ConvDtype::UINT8, ConvDtype::INT32, ConvDtype::INT32);
        SetMapping(ConvDtype::FLOAT16, ConvDtype::FLOAT32, ConvDtype::FLOAT32);
        SetMapping(ConvDtype::BF16, ConvDtype::FLOAT32, ConvDtype::FLOAT32);
        SetMapping(ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::FLOAT32);
    }

private:
    void SetMapping(ConvDtype key, ConvDtype madType, ConvDtype biasType)
    {
        typeMaps[static_cast<uint8_t>(key)].madType = madType;
        typeMaps[static_cast<uint8_t>(key)].biasType = biasType;
    }
};

struct AscendApiMknMap {
    int32_t GetByIndex(uint32_t idx) const
    {
        if (idx > MKN_MAX_SIZE - 1) {
        return MKN_VALUE_DEFAULT;
        }
        return map[idx];
    }
    int8_t map[MKN_MAX_SIZE];
};
 
struct AscendApiCubeMkn {
    int8_t toIdx[static_cast<uint8_t>(ConvDtype::CONVDTYPEMAX) + 1] = {0};
    AscendApiMknMap maps[3] = {{CUBE_UNIT, FP16_CUBE_UNIT, CUBE_UNIT}, // fp16
                               {CUBE_UNIT, FP32_CUBE_UNIT, CUBE_UNIT}, // fp32
                               {CUBE_UNIT, INT8_CUBE_UNIT, CUBE_UNIT}}; // int8
    uint32_t GetMKN(ConvDtype dType, uint32_t idx) const
    {
        return maps[toIdx[static_cast<uint8_t>(dType)]].GetByIndex(idx);
    }
    AscendApiCubeMkn()
    {
        toIdx[static_cast<uint8_t>(ConvDtype::FLOAT16)] = 0;
        toIdx[static_cast<uint8_t>(ConvDtype::FLOAT32)] = 1;
        toIdx[static_cast<uint8_t>(ConvDtype::BF16)] = 0;
        toIdx[static_cast<uint8_t>(ConvDtype::INT8)] = CONST_VALUE_2;
    }
};

const AscendApiCubeTypeMap CUBE_TYPE_TAB = AscendApiCubeTypeMap();
const AscendApiCubeMkn CUBE_MKN_TAB = AscendApiCubeMkn();

int64_t LCM(int64_t numL, int64_t numR);
uint64_t CeilDiv(uint64_t a, uint64_t b);
uint64_t AlignB(uint64_t a, uint64_t b);
uint64_t Gcd(uint64_t a, uint64_t b);
void CalcCommFactorWithPowerOfTwo(const uint64_t num, const uint64_t numMax, std::vector<uint64_t> &resList);
void CalcCommFactor(const uint64_t num, const uint64_t numMax, std::vector<uint64_t> &resList);
void CalcFactorPointWise(uint64_t numMax, std::vector<uint64_t> &resList);
void VectorElementMultip(std::vector<uint64_t> &range, const uint64_t value);
bool IsArrayEqual(const std::vector<ConvDtype>& arr1, const std::vector<ConvDtype>& arr2, uint32_t size);

template <typename T>
bool AddWithOverflowCheck(T &res, T a, T b)
{
    T tmpRes = a + b;
    if (tmpRes < a || tmpRes < b) {
        return true;
    }
    res = tmpRes;
    return false;
}

template <typename T>
bool MulWithOverflowCheck(T &res, T a, T b)
{
    if (a == 0 || b == 0) {
        res = 0;
        return false;
    }
    T tmpRes = a * b;
    if (tmpRes / a != b) {
        return true;
    }
    res = tmpRes;
    return false;
}

// 调用时控制传参个数，避免栈溢出
template <typename T, typename... Args>
bool MulWithOverflowCheck(T &res, T a, T b, Args... args)
{
    T tmp;
    return MulWithOverflowCheck(tmp, a, b) || MulWithOverflowCheck(res, tmp, args...);
}

} // namespace conv3d_tiling

#endif