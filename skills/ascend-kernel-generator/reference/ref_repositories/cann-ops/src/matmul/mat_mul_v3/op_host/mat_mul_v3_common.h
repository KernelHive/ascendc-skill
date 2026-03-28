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
 * \file matmul_v3_common.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_COMMON_H__
#define __OP_HOST_MATMUL_V3_COMMON_H__

#define OP_LOGI(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGD(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGW(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGE(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)

namespace optiling {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }
}  // namespace optiling

namespace optiling {
#define CUBE_INNER_ERR_REPORT(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)
}  // namespace optiling

#include "register/op_def_registry.h"
#include "op_util.h"

class MathUtil {
 public:
  template <typename T>
  static T CeilDivision(T num1, T num2) {
    return ops::CeilDiv(num1, num2);
  }

  static int64_t CeilDivision(int64_t num1, int32_t num2) {
    return ops::CeilDiv(num1, static_cast<int64_t>(num2));
  }
  static int64_t Align(int32_t num1, int32_t num2) { return static_cast<int64_t>(CeilDivision(num1, num2)) * num2; }

  static uint64_t Align(uint32_t num1, uint32_t num2) {
    return static_cast<uint64_t>(CeilDivision(num1, num2)) * num2;
  }

  static int64_t Align(int64_t num1, int32_t num2) { return CeilDivision(num1, static_cast<int64_t>(num2)) * num2; }

  static int64_t Align(int64_t num1, int64_t num2) { return CeilDivision(num1, num2) * num2; }

  static uint64_t Align(uint64_t num1, uint64_t num2) { return CeilDivision(num1, num2) * num2; }

  static int64_t Align(int64_t num1, uint32_t num2) { return CeilDivision(num1, static_cast<int64_t>(num2)) * num2; }

  static int64_t Align(int32_t num1, int64_t num2) { return CeilDivision(static_cast<int64_t>(num1), num2) * num2; }
};

namespace optiling {
namespace matmul_v3 {

constexpr uint64_t BASIC_ALIGN_16 = 16;
constexpr uint64_t BASIC_ALIGN_256 = 256;
constexpr uint64_t BASIC_ALIGN_512 = 512;
constexpr uint64_t MB_SIZE = 1024 * 1024;
constexpr uint64_t KB_SIZE = 1024;
constexpr uint64_t DB_SIZE = 2;
constexpr uint64_t DB_OFF_SIZE = 1;
constexpr uint64_t LOC_DATA_SIZE = 4;
constexpr uint64_t L2_TILE_LENGTH = 3072;
constexpr uint64_t L2_TILE_LENGTH_L2_128 = 1536;
constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint64_t BASIC_BLOCK_SIZE_64 = 64;
constexpr uint64_t BASIC_BLOCK_SIZE_16 = 16;
constexpr uint64_t BASIC_BLOCK_K_256_BYTE = 256;
constexpr uint64_t BASIC_BLOCK_K_128_BYTE = 128;
constexpr uint64_t BIAS_TABLE_NUM = 256;
constexpr uint64_t BASIC_ALIGN_BLK = 256;
constexpr uint64_t DATA_SIZE_FP32 = 4;
constexpr uint64_t DATA_SIZE_FP16 = 2;
constexpr uint64_t L0C_SIZE_256_KB = 262144;
constexpr uint64_t NUM_HALF = 2;
constexpr uint64_t BASE_STEP = 1;
constexpr uint64_t ITER_COL_FIRST = 0;
constexpr uint64_t ITER_ROW_FIRST = 1;
constexpr uint64_t N_FIRST_COEFF = 10;
constexpr uint64_t SMALL_SHAPE_THRES = 256;
constexpr uint64_t IF_MULTIK_THRES = 8;
constexpr uint64_t FP16_BF16_DTYPE_SIZE = 2;
constexpr uint64_t FP32_HF32_DTYPE_SIZE = 4;
constexpr uint64_t SPLIT_K_THRES = 27392;
constexpr uint64_t ND2NZ_ON_THE_FLY_LIMIT = 65535;
constexpr uint64_t DETER_THRES_OUT_SIZE = 4;
constexpr uint64_t ALIGN_128 = 128;
constexpr uint64_t ALIGN_OUTER = 32;
constexpr uint64_t ALIGN_INNER = 256;
constexpr uint64_t RPC_WORKSIZE = 20;
constexpr size_t BIAS_IDX = 2;

enum class CalcType {
  M_BY_BASE_NK,
  MN_BY_BASE_K
};

enum class MatmulV3Trans
{
    NO_TRANS = 0,
    A_TRANS = 1,
    B_TRANS = 2,
    AB_TRANS = 3
};

enum class TilingCalcSelect  //选择不同的计算Tiling的方法
{
    ALL = 0,
    BASE = 1,
    SINGLE_CORE_SPLIT_K = 2,
    DETERMINISTIC_SPLIT_K = 3
};

enum class TilingEnableSplitCore // 互斥flag, 对应不同切K模板选择
{
    BASE = 0,
    SINGLE_CORE_SPLIT_K = 2,
    DETERMINISTIC_SPLIT_K = 3,
    MAX = 10 //模板类别不能超过10个
};

enum class TilingEnableFullLoad// 互斥flag, 对应不同全载模板选择
{
    BASE = 0,
    AL1_FULL_LOAD = 1,
    BL1_FULL_LOAD = 2,
    MAX = 10 //模板类别不能超过10个
};

enum class TilingEnableFixOpti// 互斥flag, 对应不同输出优化模板选择
{
    BASE = 0,
    BASE_ENABLE_ALIGNOUT = 1,
    VEC_NZ2ND_UNALIGNOUT = 2,
    MAX = 10 //模板类别不能超过10个
};

struct TilingEnable
{
    TilingEnableSplitCore tilingEnableSplitCore = TilingEnableSplitCore::BASE; //aoetilingenable的个位
    TilingEnableFullLoad tilingEnableFullLoad = TilingEnableFullLoad::BASE; //aoetilingenable的十位
    TilingEnableFixOpti tilingEnableFixOpti = TilingEnableFixOpti::BASE; //aoetilingenable的千位
};

struct MatmulV3Args
{
    const char *opName = nullptr;
    bool isATrans = false;
    bool isBTrans = false;
    bool isHf32 = false;
    bool hasBias = false;
    bool nd2nzA = false;
    bool nd2nzB = false;
    ge::DataType aType = ge::DT_FLOAT16;
    ge::DataType bType = ge::DT_FLOAT16;
    ge::DataType cType = ge::DT_FLOAT16;
    ge::DataType biasType = ge::DT_FLOAT16;
    ge::Format aFormat = ge::FORMAT_ND;
    ge::Format bFormat = ge::FORMAT_ND;
    ge::Format outFormat = ge::FORMAT_ND;
    uint64_t mValue = 0L;
    uint64_t mOriValue = 0L;
    uint64_t nOriValue = 0L;
    uint64_t kValue = 0L;
    uint64_t nValue = 0L;
    double l2Ratio = 0;
};

struct MatmulV3L2RunInfo
{
    uint64_t mTile = 1;
    uint64_t nTile = 1;
    uint64_t mTileBlock = 0;
    uint64_t nTileBlock = 0;
    uint64_t calOrder = 0;
};

struct MatmulV3RunInfo
{
    bool needUpdate = false;
    uint64_t usedCoreNum = 1;
    uint64_t singleCoreM = 1;
    uint64_t singleCoreN = 1;
    uint64_t singleCoreK = 1;
    uint64_t baseM = 1;
    uint64_t baseN = 1;
    uint64_t baseK = 1;
    uint64_t stepKa = 1;
    uint64_t stepKb = 1;
    uint64_t depthA1 = 1;
    uint64_t depthB1 = 1;
    uint64_t stepM = 1;
    uint64_t stepN = 1;
    uint64_t iterateOrder = 0;
    uint64_t dbL0c = 0;
    uint64_t baseAN = 0;
    uint64_t baseAD = 0;
    uint64_t baseBN = 0;
    uint64_t baseBD = 0;
    MatmulV3L2RunInfo l2Info;
};

struct MatmulV3L2SplitParams
{
    uint64_t outBase = 0;
    uint64_t innerBase = 0;
    uint64_t outValue = 0;
    uint64_t innerValue = 0;
    uint64_t outDtypeSize = 0;
    uint64_t innerDtypeSize = 0;
    uint64_t maxConflictDim = 0;
    uint64_t minConflictDim = 0;
    uint64_t outTailCnt = 0;
    uint64_t innerTailCnt = 0;
};

const static std::map<ge::DataType, matmul_tiling::DataType> DTYPE_MAP =
{
    {ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16},
    {ge::DT_FLOAT, matmul_tiling::DataType::DT_FLOAT},
    {ge::DT_BF16, matmul_tiling::DataType::DT_BF16},
    {ge::DT_INT8, matmul_tiling::DataType::DT_INT8},
};

template<typename T>
inline bool Is256BAlign(T base, uint64_t dTypeSize) {
    if (base * dTypeSize % 256 == 0) { // 256: align byte size
        return true;
    }
    return false;
};

template<typename T>
inline bool Is32BAlign(T base, uint64_t dTypeSize) {
    if (base * dTypeSize % 32 == 0) { // 32: align byte size
        return true;
    }
    return false;
};

template<typename T>
inline bool CheckNumberIsVaild(const T &number, const std::string &opName, const std::string &description) {
    if (number > static_cast<uint64_t>(UINT32_MAX)) {
        return true;
    }
    return false;
};

// 需要对齐16的参数需要判断是否大于floorAlign(uint32_max, 16)
template<typename T>
inline bool CheckNumberIsVaild2(const T &number, const std::string &opName, const std::string &description) {
    if (number > ops::FloorAlign(static_cast<uint64_t>(UINT32_MAX), BASIC_ALIGN_16)) {
        return true;
    }
    return false;
};

inline ge::graphStatus GenSimplifiedKey(gert::TilingContext *context, ge::char_t *simplifiedKey) {
    static const size_t DEST_MAX = 100;
    static const size_t MAX_LEN_SIMPLIFIED_KEY = 256;
    static const uint32_t INPUT0_INDEX = 0;
    static const uint32_t INPUT1_INDEX = 1;
    static const uint32_t BIAS_INDEX = 2;
    OP_LOGI(context->GetNodeName(), "Enter GenSimplifiedKey.");
    OP_TILING_CHECK(simplifiedKey == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "simplifiedKey is null"),
                    return ge::GRAPH_FAILED);
    std::string simpKeyTemp = "";
    strcat_s(simplifiedKey, DEST_MAX, "diy,");
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(INPUT0_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(INPUT1_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetOutputDesc(0));
    if (context->GetInputDesc(BIAS_INDEX) != nullptr) {
        simpKeyTemp = std::to_string(context->GetInputDesc(INPUT0_INDEX)->GetStorageFormat()) + "/" +
                      std::to_string(context->GetInputDesc(INPUT1_INDEX)->GetStorageFormat()) + "/" +
                      std::to_string(ge::FORMAT_ND) + "/" + // bias的format均为FormatND，因此约束为仅通过FORMAT_ND参与匹配
                      std::to_string(context->GetOutputDesc(0)->GetStorageFormat()) + "/" +
                      std::to_string(context->GetInputDesc(INPUT0_INDEX)->GetDataType()) + "/" +
                      std::to_string(context->GetInputDesc(INPUT1_INDEX)->GetDataType()) + "/" +
                      std::to_string(context->GetInputDesc(BIAS_INDEX)->GetDataType()) + "/" +
                      std::to_string(context->GetOutputDesc(0)->GetDataType());
    } else {
        // 二进制发布json有无bias时合并为同一个json发布，当无法获取bias信息时，当前约定使用input0的信息代替
        simpKeyTemp = std::to_string(context->GetInputDesc(INPUT0_INDEX)->GetStorageFormat()) + "/" +
                      std::to_string(context->GetInputDesc(INPUT1_INDEX)->GetStorageFormat()) + "/" +
                      std::to_string(ge::FORMAT_ND) + "/" +
                      std::to_string(context->GetOutputDesc(0)->GetStorageFormat()) + "/" +
                      std::to_string(context->GetInputDesc(INPUT0_INDEX)->GetDataType()) + "/" +
                      std::to_string(context->GetInputDesc(INPUT1_INDEX)->GetDataType()) + "/" +
                      std::to_string(context->GetInputDesc(INPUT0_INDEX)->GetDataType()) + "/" +
                      std::to_string(context->GetOutputDesc(0)->GetDataType());
    }
    errno_t err = strcat_s(simplifiedKey, DEST_MAX, simpKeyTemp.c_str());
    if (err != 0) {
        std::cerr << "Error: strcat_s failed with error code " << err << std::endl;
        return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK(strlen(simplifiedKey) > MAX_LEN_SIMPLIFIED_KEY,
                           CUBE_INNER_ERR_REPORT(context->GetNodeName(), "len of simplifiedKey exceeds max length."),
                           return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

}
}
#endif // __OP_HOST_MATMUL_V3_COMMON_H__
