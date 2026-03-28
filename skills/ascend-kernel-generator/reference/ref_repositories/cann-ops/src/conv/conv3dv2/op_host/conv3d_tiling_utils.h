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
 * \file conv3d_tiling_utils.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_CONV3D_TILING_UTILS_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_CONV3D_TILING_UTILS_H

#include <vector>
#include "tiling/tiling_api.h"
#include "conv3d_tiling.h"
#include "conv3d_tiling_util.h"

namespace optiling {
namespace conv3d_ops_tiling {
constexpr uint32_t INPUT_FMAP_INDEX = 0;
constexpr uint32_t INPUT_WEIGHT_INDEX = 1;
constexpr uint32_t INPUT_BIAS_INDEX = 2;
constexpr uint32_t INPUT_SCALE_INDEX = 3;
constexpr uint32_t INPUT_OFFSET_INDEX = 4;
constexpr uint32_t INPUT_PERTOKEN_SCALE_INDEX = 5;
constexpr uint32_t INPUT_OFFSET_W_INDEX = 6;
constexpr uint32_t OUTPUT_INDEX = 0;

constexpr uint32_t FORMAT_NCDHW_N_INDEX = 0;
constexpr uint32_t FORMAT_NCDHW_C_INDEX = 1;
constexpr uint32_t FORMAT_NCDHW_D_INDEX = 2;
constexpr uint32_t FORMAT_NCDHW_H_INDEX = 3;
constexpr uint32_t FORMAT_NCDHW_W_INDEX = 4;
constexpr uint32_t FORMAT_NCDHW_DIM = 5;

constexpr uint32_t FORMAT_NDC1HWC0_N_INDEX = 0;
constexpr uint32_t FORMAT_NDC1HWC0_D_INDEX = 1;
constexpr uint32_t FORMAT_NDC1HWC0_C1_INDEX = 2;
constexpr uint32_t FORMAT_NDC1HWC0_H_INDEX = 3;
constexpr uint32_t FORMAT_NDC1HWC0_W_INDEX = 4;
constexpr uint32_t FORMAT_NDC1HWC0_C0_INDEX = 5;
constexpr uint32_t FORMAT_NDC1HWC0_DIM = 6;

constexpr uint32_t FORMAT_ND_DIM = 1;

constexpr uint32_t FORMAT_FRACTAL_3D_DKCIN1KHKW_INDEX = 0;
constexpr uint32_t FORMAT_FRACTAL_3D_N1_INDEX = 1;
constexpr uint32_t FORMAT_FRACTAL_3D_N0_INDEX = 2;
constexpr uint32_t FORMAT_FRACTAL_3D_C0_INDEX = 3;
constexpr uint32_t FORMAT_FRACTAL_3D_DIM = 4;

constexpr uint32_t ATTR_STRIDE_INDEX = 0;
constexpr uint32_t ATTR_PAD_INDEX = 1;
constexpr uint32_t ATTR_DILATION_INDEX = 2;
constexpr uint32_t ATTR_GROUP_INDEX = 3;
constexpr uint32_t ATTR_DATA_FORMAT_INDEX = 4;
constexpr uint32_t ATTR_HF32_FLAG_INDEX = 6;
constexpr uint32_t ATTR_OP_IMPL_MODE_INDEX = 8;

constexpr uint32_t PAD_HEAD_INDEX = 0;
constexpr uint32_t PAD_TAIL_INDEX = 1;
constexpr uint32_t PAD_UP_INDEX = 2;
constexpr uint32_t PAD_DOWN_INDEX = 3;
constexpr uint32_t PAD_LEFT_INDEX = 4;
constexpr uint32_t PAD_RIGHT_INDEX = 5;
constexpr uint32_t FORMAT_PAD_DIM = 6;

constexpr uint32_t LOAD3D_MAX_STRIDE_H_W = 63;
constexpr uint32_t LOAD3D_MAX_DILATION_H_W = 255;
constexpr uint32_t LOAD3D_MAX_PAD = 255;
constexpr uint32_t LOAD3D_MAX_FILTER_H_W = 511;

constexpr uint32_t MAX_16_BIT_NUM = 65535;
 
constexpr uint32_t MIN_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t WORKSPACE_NUM = 4;
constexpr uint32_t C0_BYTE_SIZE = 32;

constexpr uint32_t CUBE_UNIT = 16;
constexpr uint32_t FP16_CUBE_UNIT = 16;
constexpr uint32_t FP32_CUBE_UNIT = 8;
constexpr uint32_t INT8_CUBE_UNIT = 32;
constexpr uint32_t MKN_MAX_SIZE = 3;
constexpr uint32_t MKN_M_INDEX = 0;
constexpr uint32_t MKN_K_INDEX = 1;
constexpr uint32_t MKN_N_INDEX = 2;
constexpr uint32_t MKN_VALUE_DEFAULT = 16;

constexpr uint32_t CONST_VALUE_2 = 2;

constexpr uint32_t BATCH_AICORE_COF = 2;
constexpr uint32_t C0_SIZE = 32;

//blockdim: [batchDim, mdim, nDim, doDim, groupDim]
constexpr uint32_t BLOCKDIM_DEC_NUM = 5;
constexpr uint32_t BLOCKDIM_BATCH_IDX = 0;
constexpr uint32_t BLOCKDIM_M_IDX = 1;
constexpr uint32_t BLOCKDIM_N_IDX = 2;
constexpr uint32_t BLOCKDIM_DO_IDX = 3;
constexpr uint32_t BLOCKDIM_GROUP_IDX = 4;

constexpr uint32_t MKN_M_IDX = 0;
constexpr uint32_t MKN_K_IDX = 1;
constexpr uint32_t MKN_N_IDX = 2;

constexpr uint32_t COUNT_PARAMS_WITH_BIAS = 4; // [fmap, weight, bias, output]
constexpr uint32_t COUNT_PARAMS_WITHOUT_BIAS = 3; // [fmap, weight, output]

constexpr int8_t M_Mode = 0;
constexpr int8_t HW_Mode = 1;

constexpr uint64_t TILING_KEY_0 = 0;
constexpr uint64_t TILING_KEY_10 = 10;
constexpr uint64_t TILING_KEY_200 = 200;
constexpr uint64_t TILING_KEY_210 = 210;
constexpr uint64_t TILING_KEY_400 = 400;
constexpr uint64_t TILING_KEY_410 = 410;
constexpr uint64_t TILING_KEY_600 = 600;
constexpr uint64_t TILING_KEY_610 = 610;
constexpr uint64_t TILING_KEY_10010 = 10010;
constexpr uint64_t TILING_KEY_10210 = 10210;
constexpr uint64_t TILING_KEY_10400 = 10400;
constexpr uint64_t TILING_KEY_10410 = 10410;
constexpr uint64_t TILING_KEY_10600 = 10600;
constexpr uint64_t TILING_KEY_10610 = 10610;
constexpr uint64_t TILING_KEY_100000 = 100000;
constexpr uint64_t TILING_KEY_100010 = 100010;
constexpr uint64_t TILING_KEY_100200 = 100200;
constexpr uint64_t TILING_KEY_100210 = 100210;
constexpr uint64_t TILING_KEY_100400 = 100400;
constexpr uint64_t TILING_KEY_100410 = 100410;
constexpr uint64_t TILING_KEY_100600 = 100600;
constexpr uint64_t TILING_KEY_100610 = 100610;

constexpr uint32_t TILING_KEY_BYPASS_BASE = 10;
constexpr uint32_t TILING_KEY_L0PINGPONG_BASE = 100;
constexpr uint32_t TILING_KEY_GROUPS_BASE = 10000;
constexpr uint32_t TILING_KEY_HW_MODE_BASE = 100000;

constexpr uint64_t MAX_ORI_ONE_DIM_SIZE = 1000000;
constexpr uint64_t MAX_ORI_FMAP_SIZE = 10000000;

constexpr uint32_t TILING_KEY_L0PINGPONG_L0C_POS_INDEX = 0;
constexpr uint32_t TILING_KEY_L0PINGPONG_L0A_POS_INDEX = 1;
constexpr uint32_t TILING_KEY_L0PINGPONG_L0B_POS_INDEX = 2;

constexpr uint32_t PBUFFERFLAG_L0A_MASK = 1;
constexpr uint32_t PBUFFERFLAG_L0B_MASK = 2;
constexpr uint32_t PBUFFERFLAG_L0C_MASK = 4;

const std::vector<std::vector<conv3d_tiling::ConvDtype>> SUPPORTED_TYPES_WITH_BIAS = {
    {conv3d_tiling::ConvDtype::FLOAT16, conv3d_tiling::ConvDtype::FLOAT16,
     conv3d_tiling::ConvDtype::FLOAT32, conv3d_tiling::ConvDtype::FLOAT16},
    {conv3d_tiling::ConvDtype::FLOAT32, conv3d_tiling::ConvDtype::FLOAT32,
     conv3d_tiling::ConvDtype::FLOAT32, conv3d_tiling::ConvDtype::FLOAT32},
    {conv3d_tiling::ConvDtype::INT8, conv3d_tiling::ConvDtype::INT8,
     conv3d_tiling::ConvDtype::INT32, conv3d_tiling::ConvDtype::INT32},
    {conv3d_tiling::ConvDtype::BF16, conv3d_tiling::ConvDtype::BF16,
     conv3d_tiling::ConvDtype::FLOAT32, conv3d_tiling::ConvDtype::BF16}
};
const std::vector<std::vector<conv3d_tiling::ConvDtype>> SUPPORTED_TYPES_WITHOUT_BIAS = {
    {conv3d_tiling::ConvDtype::FLOAT16, conv3d_tiling::ConvDtype::FLOAT16, conv3d_tiling::ConvDtype::FLOAT16},
    {conv3d_tiling::ConvDtype::FLOAT32, conv3d_tiling::ConvDtype::FLOAT32, conv3d_tiling::ConvDtype::FLOAT32},
    {conv3d_tiling::ConvDtype::INT8, conv3d_tiling::ConvDtype::INT8, conv3d_tiling::ConvDtype::INT32},
    {conv3d_tiling::ConvDtype::BF16, conv3d_tiling::ConvDtype::BF16, conv3d_tiling::ConvDtype::BF16},
};

struct Conv3DAscendcShapesInfo {
    uint32_t batch = 1;
    uint32_t cIn = 1;
    uint32_t di = 1;
    uint64_t hi = 1;
    uint64_t wi = 1;
    uint32_t cOut = 1;
    uint32_t kd = 1;
    uint32_t kh = 1;
    uint32_t kw = 1;
    uint32_t dOut = 1;
    uint64_t ho = 1;
    uint64_t wo = 1;
    uint64_t cinOpt = 1;
    uint64_t coutOpt = 1;
};

struct Conv3DDescInfo {
    ge::DataType weightDtype = ge::DT_BF16;
    ge::DataType fMapDtype = ge::DT_BF16;
    ge::DataType biasDtype = ge::DT_FLOAT;
    ge::DataType scaleDtype = ge::DT_FLOAT;    
    ge::DataType outDtype = ge::DT_BF16;

    ge::Format weightFormat = ge::FORMAT_FRACTAL_Z_3D;
    ge::Format fMapFormat = ge::FORMAT_NDC1HWC0;
    ge::Format biasFormat = ge::FORMAT_ND;
    ge::Format scaleFormat = ge::FORMAT_ND;
    ge::Format outFormat = ge::FORMAT_NDC1HWC0;
};

struct Conv3DTilingFlag {
    bool hasBias = false;
    bool hasScale = false;
};

struct BlockDimRange {
  std::vector<uint32_t> aicNumRange;
  std::vector<uint32_t> batchRange;
  std::vector<uint32_t> mRange;
  std::vector<uint32_t> nRange;
  std::vector<uint32_t> doRange;
  std::vector<uint32_t> groupRange;
};

struct BlockDimConstParas {
  uint64_t m0;
  uint64_t n0;
  uint64_t k0;
  uint64_t ci1;
  uint64_t co1;
};

struct BlockDimRes {
  uint32_t batchDim = 0;
  uint32_t mDim = 0;
  uint32_t nDim = 0;
  uint32_t doDim = 0;
  uint32_t groupDim = 0;
  uint64_t minCost = 0;
};

static std::map<ge::DataType, std::string> g_dtypeToStrTab = {
    {ge::DataType::DT_FLOAT16, "float16"}, {ge::DataType::DT_FLOAT, "float32"}, {ge::DataType::DT_BF16, "bfloat16"},
    {ge::DataType::DT_INT8, "int8"}, {ge::DataType::DT_UINT8, "uint8"}, {ge::DataType::DT_INT64, "int64"},
    {ge::DataType::DT_UINT64, "uint64"}, {ge::DataType::DT_INT32, "int32"}};

static std::map<conv3d_tiling::ConvDtype, std::string> g_convDtypeToStr = {
    {conv3d_tiling::ConvDtype::FLOAT16, "float16"},
    {conv3d_tiling::ConvDtype::FLOAT32, "float32"},
    {conv3d_tiling::ConvDtype::BF16, "bfloat16"},
    {conv3d_tiling::ConvDtype::INT4, "int4"},
    {conv3d_tiling::ConvDtype::INT8, "int8"},
    {conv3d_tiling::ConvDtype::UINT8, "uint8"},
    {conv3d_tiling::ConvDtype::INT64, "int64"},
    {conv3d_tiling::ConvDtype::UINT64, "uint64"},
    {conv3d_tiling::ConvDtype::INT32, "int32"},
};

static std::map<ge::DataType, uint32_t> g_dataTypeSizeTab = {
    {ge::DataType::DT_FLOAT16, 2}, {ge::DataType::DT_FLOAT, 4}, {ge::DataType::DT_BF16, 2}, {ge::DataType::DT_INT8, 1},
    {ge::DataType::DT_UINT8, 1}, {ge::DataType::DT_INT64, 8}, {ge::DataType::DT_UINT64, 8}, {ge::DataType::DT_INT32, 4}};

static std::map<ge::DataType, conv3d_tiling::ConvDtype> g_dtypeMap = {
    {ge::DT_FLOAT16, conv3d_tiling::ConvDtype::FLOAT16},
    {ge::DT_FLOAT, conv3d_tiling::ConvDtype::FLOAT32},
    {ge::DT_BF16, conv3d_tiling::ConvDtype::BF16},
    {ge::DT_INT8, conv3d_tiling::ConvDtype::INT8},
    {ge::DT_UINT8, conv3d_tiling::ConvDtype::UINT8},
    {ge::DT_INT64, conv3d_tiling::ConvDtype::INT64},
    {ge::DT_UINT64, conv3d_tiling::ConvDtype::UINT64},
    {ge::DT_INT32, conv3d_tiling::ConvDtype::INT32}
};

static std::map<ge::Format, std::string> g_formatToStrTab = {
    {ge::FORMAT_NCHW, "NCHW"}, {ge::FORMAT_NHWC, "NHWC"}, {ge::FORMAT_HWCN, "HWCN"}, {ge::FORMAT_DHWNC, "DHWNC"},
    {ge::FORMAT_DHWCN, "DHWCN"}, {ge::FORMAT_NDHWC, "NDHWC"}, {ge::FORMAT_NCDHW, "NCDHW"},
    {ge::FORMAT_NC1HWC0, "NC1HWC0"}, {ge::FORMAT_ND, "ND"}, {ge::FORMAT_NDC1HWC0, "NDC1HWC0"},
    {ge::FORMAT_FRACTAL_Z_3D, "FRACTAL_Z_3D"}};

static std::map<ge::Format, conv3d_tiling::ConvFormat> g_formatMap = {
    {ge::FORMAT_ND, conv3d_tiling::ConvFormat::ND},
    {ge::FORMAT_NCHW, conv3d_tiling::ConvFormat::NCHW},
    {ge::FORMAT_NHWC, conv3d_tiling::ConvFormat::NHWC},
    {ge::FORMAT_HWCN, conv3d_tiling::ConvFormat::HWCN},
    {ge::FORMAT_DHWNC, conv3d_tiling::ConvFormat::DHWNC},
    {ge::FORMAT_DHWCN, conv3d_tiling::ConvFormat::DHWCN},
    {ge::FORMAT_NDHWC, conv3d_tiling::ConvFormat::NDHWC},
    {ge::FORMAT_NCDHW, conv3d_tiling::ConvFormat::NCDHW},
    {ge::FORMAT_NC1HWC0, conv3d_tiling::ConvFormat::NC1HWC0},
    {ge::FORMAT_NDC1HWC0, conv3d_tiling::ConvFormat::NDC1HWC0},
    {ge::FORMAT_FRACTAL_Z_3D, conv3d_tiling::ConvFormat::FRACTAL_Z_3D}
};

struct AscendOpsCubeTypeMap {
    struct {
        conv3d_tiling::ConvDtype madType;
        conv3d_tiling::ConvDtype biasType;
    } typeMaps[static_cast<uint8_t>(conv3d_tiling::ConvDtype::CONVDTYPEMAX) + 1] =
        {{conv3d_tiling::ConvDtype::CONVDTYPEMAX, conv3d_tiling::ConvDtype::CONVDTYPEMAX}};
    
    conv3d_tiling::ConvDtype ToBiasType(conv3d_tiling::ConvDtype type) const {
        return typeMaps[static_cast<uint8_t>(type)].biasType;
    }
    conv3d_tiling::ConvDtype ToMadType(conv3d_tiling::ConvDtype type) const {
        return typeMaps[static_cast<uint8_t>(type)].madType;
    }
    
    AscendOpsCubeTypeMap() {
        SetMapping(conv3d_tiling::ConvDtype::INT4, conv3d_tiling::ConvDtype::INT32, conv3d_tiling::ConvDtype::INT32);
        SetMapping(conv3d_tiling::ConvDtype::INT8, conv3d_tiling::ConvDtype::INT32, conv3d_tiling::ConvDtype::INT32);
        SetMapping(conv3d_tiling::ConvDtype::UINT8, conv3d_tiling::ConvDtype::INT32, conv3d_tiling::ConvDtype::INT32);
        SetMapping(conv3d_tiling::ConvDtype::FLOAT16, conv3d_tiling::ConvDtype::FLOAT32, conv3d_tiling::ConvDtype::FLOAT32);
        SetMapping(conv3d_tiling::ConvDtype::BF16, conv3d_tiling::ConvDtype::FLOAT32, conv3d_tiling::ConvDtype::FLOAT32);
        SetMapping(conv3d_tiling::ConvDtype::FLOAT32, conv3d_tiling::ConvDtype::FLOAT32, conv3d_tiling::ConvDtype::FLOAT32);
    }
    
    private:
    void SetMapping(conv3d_tiling::ConvDtype key, conv3d_tiling::ConvDtype madType, conv3d_tiling::ConvDtype biasType) {
        typeMaps[static_cast<uint8_t>(key)].madType = madType;
        typeMaps[static_cast<uint8_t>(key)].biasType = biasType;    // bias dtype in bt
    }
};

const AscendOpsCubeTypeMap CUBE_TYPE_MAP = AscendOpsCubeTypeMap();

struct AscendOpsMknMap {
    int32_t GetByIndex(uint32_t idx) const {
        if (idx > MKN_MAX_SIZE - 1) {
        return MKN_VALUE_DEFAULT;
        }
        return map[idx];
    }
    int8_t map[MKN_MAX_SIZE];
};
 
struct AscendOpsCubeMkn {
    int8_t toIdx[static_cast<uint8_t>(conv3d_tiling::ConvDtype::CONVDTYPEMAX) + 1] = {0};
    AscendOpsMknMap maps[3] = {{CUBE_UNIT, FP16_CUBE_UNIT, CUBE_UNIT}, // fp16
                        {CUBE_UNIT, FP32_CUBE_UNIT, CUBE_UNIT}, // fp32
                        {CUBE_UNIT, INT8_CUBE_UNIT, CUBE_UNIT}}; // int8
    uint32_t GetMKN(conv3d_tiling::ConvDtype dType, uint32_t idx) {
        return maps[toIdx[static_cast<uint8_t>(dType)]].GetByIndex(idx);
    }
    AscendOpsCubeMkn() {
        toIdx[static_cast<uint8_t>(conv3d_tiling::ConvDtype::FLOAT16)] = 0;
        toIdx[static_cast<uint8_t>(conv3d_tiling::ConvDtype::FLOAT32)] = 1;
        toIdx[static_cast<uint8_t>(conv3d_tiling::ConvDtype::BF16)] = 0;
        toIdx[static_cast<uint8_t>(conv3d_tiling::ConvDtype::INT8)] = CONST_VALUE_2;
    }
};

static AscendOpsCubeMkn g_cubeMknMap = AscendOpsCubeMkn();

uint64_t CeilDiv(uint64_t a, uint64_t b);
void CalcCommFactor(const uint64_t num, const uint32_t numMax, std::vector<uint32_t> &reslist);
bool IsArrayEqual(std::vector<conv3d_tiling::ConvDtype>& arr1,
                  const std::vector<conv3d_tiling::ConvDtype>& arr2,
                  uint32_t size);
uint64_t AlignB(uint64_t a, uint64_t b);
uint64_t InferHiL1(uint64_t inputHoL1, uint64_t hi, uint64_t singlekH, uint32_t dilationH, uint32_t strideH);
uint64_t InferWiL1(uint64_t inputWoL1, uint64_t wi, uint64_t singlekW, uint32_t dilationW, uint32_t strideW);

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

}
}

#endif