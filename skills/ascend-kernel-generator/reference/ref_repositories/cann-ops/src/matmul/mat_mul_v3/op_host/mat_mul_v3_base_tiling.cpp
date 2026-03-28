/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file matmul_v3_base_tiling.cc
 * \brief
 */
#include "mat_mul_v3_base_tiling.h"
#include "mat_mul_v3_l2_cache.h"
#include "tiling/tiling_type.h"
#include "aoe/runtime_kb/runtime_bank_manager.h"
#include "cache_tiling.h"
typedef float float32_t;
using namespace optiling::matmul_v3;


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

namespace {
constexpr double CORE_RATIO = 0.9;
// 基本块二维矩阵定义{baseMShape, baseNShape, baseKByte}
// 参数一代表基本块中M的大小, 参数二代表N的大小
// 参数三代表K的Byte大小（需注意baseKByte = KShape * KDtypeSize), 后续输出baseK时，须在矩阵基础上除以dtypeSize
const std::vector<std::vector<uint64_t>> CALC_M_BASIC = {{128, 256, 128}, {128, 128, 256}, {64, 512, 64}};
const std::vector<std::vector<uint64_t>> CALC_MN_BASIC = {{64, 64, 512}, {128, 128, 256}, {128, 256, 128}};
const std::vector<std::vector<uint64_t>> CALC_M_BASIC_L0C_256 = {{256, 256, 128}, {128, 128, 256}, {128, 512, 64}};
const std::vector<std::vector<uint64_t>> CALC_MN_BASIC_L0C_256 = {{16, 16, 2048}, {32, 32, 1024}, {64, 64, 512},
                                                          {128, 128, 256}, {256, 256, 128}};
const std::vector<uint64_t> CALC_ND_BASIC = {6144, 4096, 2048};
const std::vector<uint64_t> SUPPORT_ND2NZ_GM2L0 = {32, 64, 96, 128, 160, 192, 224, 256, 384};
constexpr uint64_t SMALL_SHAPE_LOWER_THRES = 128;
constexpr uint64_t L0C_THRES = 32768;
constexpr uint64_t NUMBER_TWO = 2;
constexpr uint64_t BLOCK_BYTE_SIZE = 32;
constexpr uint64_t N_ALIGNED = 16;
constexpr uint64_t NCALC_THRES = 16;
constexpr uint64_t MIN_TAIL = 512;
constexpr uint64_t MULTI_CORE_SINGLE_K = 384;
constexpr uint64_t CACHELINE = 512;
constexpr uint64_t NUMBER_SIXTEEN = 16;
constexpr uint64_t ND2NZ_THRES = 384;
constexpr uint64_t VNCHW_UP_THRES = 72368;
constexpr uint64_t VECTOR_D_BASE = 2048;
constexpr size_t HF32_ATTR_NUM = 4;
constexpr size_t HF32_ATTR_INDEX = 3;
constexpr uint64_t ONE_BATCH_DIM = 1;
constexpr uint64_t TWO_BATCH_DIM = 2;
constexpr uint64_t THREE_BATCH_DIM = 3;
constexpr uint64_t FOUR_BATCH_DIM = 4;
constexpr uint64_t LAST_DIM = 1;
constexpr uint64_t LAST_SECOND_DIM = 2;
constexpr uint64_t L2_SIZE_2 = 192 * 1024 * 1024;
constexpr uint64_t BLOCK = 128;
constexpr uint64_t N_THRESHOLD = 256;
constexpr uint64_t M_THRESHOLD = 256;
constexpr uint64_t MIX_BLOCK_NUM = 24;
constexpr uint64_t MIN_SPLITK_K = 1000;
constexpr uint64_t MAX_SPLITK_K = 4608;

#define DO_CACL_TILING_ENABLE(func) if (func) { break; }

inline uint64_t CalBaseSize(uint64_t cnt, uint64_t totalCoreNum, uint64_t size, uint64_t maxBase)
{
    uint64_t curCnt = totalCoreNum / cnt;
    curCnt = std::max(curCnt, 1UL);
    uint64_t base = ops::CeilAlign(std::max(size / curCnt, 1UL), BASIC_ALIGN_16);
    base = std::min(base, maxBase);
    return base;
}
}

namespace optiling {
namespace matmul_v3 {

bool MatmulV3BaseTiling::CheckAoeTilingEnable(uint32_t aoeTilingEnable, const std::string &opName)
{
    if (aoeTilingEnable <= 1) {
        return true;
    }
    uint32_t tilingSplitK = aoeTilingEnable % 10; // aoe 的tilingEnable的个位
    if (tilingSplitK == 1 || tilingSplitK > static_cast<uint32_t>(TilingEnableSplitCore::DETERMINISTIC_SPLIT_K)) {
        return false;
    }
    tilingEnable_.tilingEnableSplitCore = static_cast<TilingEnableSplitCore>(tilingSplitK);
    uint32_t tilingFullLoad = (aoeTilingEnable / 10) % 10; // aoe 的tilingEnable的十位
    if (tilingFullLoad > static_cast<uint32_t>(TilingEnableFullLoad::BL1_FULL_LOAD)) {
        return false;
    }
    tilingEnable_.tilingEnableFullLoad = static_cast<TilingEnableFullLoad>(tilingFullLoad);
    uint32_t tilingFixOpti = (aoeTilingEnable / 1000) % 10; // aoe 的tilingEnable的千位
    if (tilingFixOpti > static_cast<uint32_t>(TilingEnableFixOpti::VEC_NZ2ND_UNALIGNOUT)) {
        return false;
    }
    tilingEnable_.tilingEnableFixOpti = static_cast<TilingEnableFixOpti>(tilingFixOpti);
    return true;
}

ge::graphStatus MatmulV3BaseTiling::GetPlatformInfo() // 检查平台信息是否支持
{
    if (!compileInfoInit_) {
        auto compileInfoPtr = reinterpret_cast<const MatmulV3CompileInfo *>(context_->GetCompileInfo());
        OPS_CHECK_NULL_WITH_CONTEXT(context_, compileInfoPtr);
        compileInfo_ = *compileInfoPtr;
    }
    if (compileInfo_.aicNum == 0) {
        OP_LOGE(context_->GetNodeName(), "compileInfo.aicNum is zero.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void MatmulV3BaseTiling::InitCompileInfo() // 检查输入属性是否支持
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGW(context_->GetNodeName(), "platformInfo is null");
        return;
    }
    MatmulV3CompileInfo compileInfo;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    platformInfo->GetPlatformRes("version", "SoC_version", compileInfo.socVersionStr);
    std::string val;
    std::string dataMoveL12Bt;
    platformInfo->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_fix_pipe_l0c2out", val);
    platformInfo->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_data_move_l12bt", dataMoveL12Bt);
    compileInfo.supportL0c2out = !val.empty();
    compileInfo.supportL12BtBf16 = (dataMoveL12Bt.find("bf16") != string::npos);
    compileInfo.aicNum = static_cast<uint64_t>(ascendcPlatform.GetCoreNumAic());
    compileInfo.aivNum = static_cast<uint64_t>(ascendcPlatform.GetCoreNumAiv());
    compileInfo.socVersion = ascendcPlatform.GetSocVersion();
    compileInfo.btSize = compileInfo.supportL0c2out ? 1024 : 0;                    // 1024 is btSize
    compileInfo.btSize = compileInfo.supportL12BtBf16 ? 4096 : compileInfo.btSize; // 4096 is btSize
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfo.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfo.l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfo.l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfo.l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfo.l2Size);

    gert::GemmCompileInfo tbeCompileInfo;
    tbeCompileInfo.ParseRuntimePlatformInfo(context_->GetNodeName(), *platformInfo);
    tbeCompileInfo.core_num = compileInfo.aicNum;
    optiling::PlatformInfo::GetInstance().SetInstance(tbeCompileInfo);
    OP_LOGI(context_->GetNodeName(),
        "parse compile info soc:%d, l1Size:%lu, l2Size:%lu, coreNum:%lu, supportL0c2out:%d, supportL12BtBf16:%d",
        static_cast<int32_t>(compileInfo.socVersion), compileInfo.l1Size, compileInfo.l2Size, compileInfo.aicNum,
        compileInfo.supportL0c2out, compileInfo.supportL12BtBf16);
    compileInfoInit_ = true;
    compileInfo_ = compileInfo;
}


ge::graphStatus MatmulV3BaseTiling::GetShapeAttrsInfo() // 检查输入属性是否支持
{
    args_.opName = context_->GetNodeName();
    OP_TILING_CHECK(args_.opName == nullptr, CUBE_INNER_ERR_REPORT("matmul", "get op name invalid"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((CheckArgs() != ge::GRAPH_SUCCESS) || (GetArgs() != ge::GRAPH_SUCCESS),
        CUBE_INNER_ERR_REPORT(args_.opName, "invalid context"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulV3BaseTiling::CheckArgs()
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    size_t idx = 0;
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(idx));
    idx++;
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(idx));
    idx++;
    // 区分Matmul和GemmV2，只有3个输入的为Matmul，并设置bias标志
    if (std::string(context_->GetNodeType()) != "TransposeBatchMatMul" && 
        context_->GetInputDesc(idx) != nullptr && context_->GetInputDesc(idx + 1) == nullptr) {
        args_.hasBias = true;
    }

    if (attrs->GetAttrNum() >= HF32_ATTR_NUM) {
        OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<int32_t>(HF32_ATTR_INDEX - 1));
        OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(HF32_ATTR_INDEX));
    }
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(0));
    return ge::GRAPH_SUCCESS;
}

static inline void GetFormat(const gert::TilingContext &context, MatmulV3Args &args)
{
    ge::Format formatA = static_cast<ge::Format>(ge::GetPrimaryFormat(context.GetInputDesc(0)->GetStorageFormat()));
    ge::Format formatB = static_cast<ge::Format>(ge::GetPrimaryFormat(context.GetInputDesc(1)->GetStorageFormat()));
    ge::Format formatOut = static_cast<ge::Format>(ge::GetPrimaryFormat(context.GetOutputDesc(0)->GetStorageFormat()));
    args.aFormat = (formatA != ge::FORMAT_FRACTAL_NZ) ? ge::FORMAT_ND : formatA;
    args.bFormat = (formatB != ge::FORMAT_FRACTAL_NZ) ? ge::FORMAT_ND : formatB;
    args.outFormat = (formatOut != ge::FORMAT_FRACTAL_NZ) ? ge::FORMAT_ND : formatOut;
}

static inline void GetDtype(const gert::TilingContext &context, MatmulV3Args &args)
{
    // op_impl_mode_enum: 0x1: default 0x2: high_performance 0x4: high_precision 0x8: super_performance
    // 0x10: support_of_bound_index 0x20: enable_float_32_execution 0x40: enable_hi_float_32_execution
    if (context.GetAttrs()->GetAttrNum() == HF32_ATTR_NUM) {
        args.isHf32 = *context.GetAttrs()->GetAttrPointer<bool>(HF32_ATTR_INDEX) ? 1 : 0;
    }
    OP_LOGD(args.opName, "Hf32 flag is: %d", args.isHf32);

    args.aType = context.GetInputDesc(0)->GetDataType();
    args.bType = context.GetInputDesc(1)->GetDataType();
    args.cType = context.GetOutputDesc(0)->GetDataType();
    if (args.hasBias) {
        args.biasType = context.GetInputDesc(BIAS_IDX)->GetDataType();
    }
}

static ge::graphStatus GetInputDims(const gert::Shape &storageShape, ge::Format format, int64_t (&dims)[TWO_BATCH_DIM])
{
    const size_t dimNum = storageShape.GetDimNum();
    if (format == ge::FORMAT_ND) {
        if (dimNum < TWO_BATCH_DIM) {
            return ge::GRAPH_FAILED;
        }
        dims[0] = storageShape[dimNum - TWO_BATCH_DIM];
        dims[1] = storageShape[dimNum - ONE_BATCH_DIM];
    } else {
        if (dimNum < FOUR_BATCH_DIM) {
            return ge::GRAPH_FAILED;
        }
        dims[0] = storageShape[dimNum - THREE_BATCH_DIM] * storageShape[dimNum - TWO_BATCH_DIM];
        dims[1] = storageShape[dimNum - FOUR_BATCH_DIM] * storageShape[dimNum - ONE_BATCH_DIM];
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus OpSpecificCheck(const gert::TilingContext &context, const MatmulV3Args &args)
{
    const bool isMatMulV3 = (strcmp(context.GetNodeType(), "MatMulV3") == 0);
    const bool isBatchMatMulV3 = (strcmp(context.GetNodeType(), "BatchMatMulV3") == 0);
    if (!isBatchMatMulV3 && !isMatMulV3) {
        // apply no additional checks for ops other than MMV3, BMMV3, for now
        return ge::GRAPH_SUCCESS;
    }

    // check input dim num equals to 2
    if (isMatMulV3) {
        auto isMatrix = [](const gert::Shape &oriShape) {
            return oriShape.GetDimNum() == TWO_BATCH_DIM;
        };
        if (!isMatrix(context.GetInputShape(0)->GetOriginShape()) ||
            !isMatrix(context.GetInputShape(1)->GetOriginShape())) {
            OP_LOGE(args.opName, "invalid input dim num");
            return ge::GRAPH_FAILED;
        }
    }

    // check bias
    if (args.hasBias) {
        const gert::Shape &biasShape = context.GetInputShape(BIAS_IDX)->GetOriginShape();
        const int64_t biasValue = biasShape[biasShape.GetDimNum() - 1];
        if (args.nOriValue != static_cast<uint64_t>(biasValue)) {
            OP_LOGE(args.opName, "illegal value: bias[%ld], n[%lu]", biasValue, args.nOriValue);
            return ge::GRAPH_FAILED;
        }
    }

    // dtype check
    std::vector<ge::DataType> dtype = {args.aType, args.bType, args.cType};
    if (args.hasBias) { dtype.push_back(args.biasType); }
    auto isValidDtype = [&args](const std::vector<ge::DataType> &dtypeList) -> ge::graphStatus {
        const std::vector<std::vector<ge::DataType> > dtypeSuportList = {
            // x1,              x2,             y,              bias
            {ge::DT_FLOAT16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
            {ge::DT_FLOAT16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT},
            {ge::DT_FLOAT,      ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT},
            {ge::DT_BF16,       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT}
        };
        for (auto &supported : dtypeSuportList) {
            if (std::equal(dtypeList.begin(), dtypeList.end(), supported.begin())) {
                return ge::GRAPH_SUCCESS;
            }
        }
        OP_LOGE(args.opName, "Unsupported data type: x1[%d], x2[%d], y[%d], bias[%d], hasBias[%d]",
                args.aType, args.bType, args.cType, args.biasType, args.hasBias);
        return ge::GRAPH_FAILED;
    };
    return isValidDtype(dtype);
}

static ge::graphStatus GetShape(const gert::TilingContext &context, MatmulV3Args &args)
{
    // get transpose
    args.isATrans = *context.GetAttrs()->GetAttrPointer<bool>(0);
    args.isBTrans = *context.GetAttrs()->GetAttrPointer<bool>(1);

    // get (m, k, n)
    int64_t mkDims[TWO_BATCH_DIM];
    int64_t knDims[TWO_BATCH_DIM];
    if ((GetInputDims(context.GetInputShape(0)->GetStorageShape(), args.aFormat, mkDims) != ge::GRAPH_SUCCESS) ||
        (GetInputDims(context.GetInputShape(1)->GetStorageShape(), args.bFormat, knDims) != ge::GRAPH_SUCCESS)) {
        OP_LOGE(args.opName, "invalid input dim num");
        return ge::GRAPH_FAILED;
    }
    int64_t k = mkDims[args.isATrans ? 0 : 1];
    if (k != knDims[args.isBTrans ? 1 : 0]) {
        OP_LOGE(args.opName, "unequal input kDim values: k_left[%ld], k_right[%ld]", k, knDims[args.isBTrans ? 1 : 0]);
        return ge::GRAPH_FAILED;
    }
    int64_t m = mkDims[args.isATrans ? 1 : 0];
    int64_t n = knDims[args.isBTrans ? 0 : 1];
    auto isValidDimValue = [](int64_t dim) -> bool { return (dim > 0) && (dim <= INT32_MAX); };
    if (!isValidDimValue(m) || !isValidDimValue(k) || !isValidDimValue(n)) {
        OP_LOGE(args.opName, "illegal value: m[%ld], k[%ld], n[%ld]", m, k, n);
        return ge::GRAPH_FAILED;
    }
    args.mValue = static_cast<uint64_t>(m);
    args.kValue = static_cast<uint64_t>(k);
    args.nValue = static_cast<uint64_t>(n);

    // get origin (m, n)
    const gert::Shape &cShape = context.GetOutputShape(0)->GetOriginShape();
    const size_t cDimNum = cShape.GetDimNum();
    if (cDimNum < TWO_BATCH_DIM) {
        OP_LOGE(args.opName, "illegal value: output dim num (%zu)", cDimNum);
        return ge::GRAPH_FAILED;
    }
    args.nOriValue = cShape[cDimNum - LAST_DIM];
    args.mOriValue = cShape[cDimNum - LAST_SECOND_DIM];

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulV3BaseTiling::GetArgs()
{
    GetFormat(*context_, args_);
    GetDtype(*context_, args_);
    if (GetShape(*context_, args_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return OpSpecificCheck(*context_, args_);
}

ge::graphStatus MatmulV3BaseTiling::GetMoreArgs()
{
    aDtypeSize_ = GetSizeByDataType(args_.aType);
    bDtypeSize_ = GetSizeByDataType(args_.bType);
    cDtypeSize_ = GetSizeByDataType(args_.cType);
    m256Align_ = Is256BAlign(args_.mValue, aDtypeSize_); // A矩阵 m轴256B对齐
    kA256Align_ = Is256BAlign(args_.kValue, aDtypeSize_); // A矩阵 k轴256B对齐
    kB256Align_ = Is256BAlign(args_.kValue, bDtypeSize_); // B矩阵 k轴256B对齐
    n256Align_ = Is256BAlign(args_.nValue, bDtypeSize_);  // B矩阵 n轴256B对齐
    bool innerAlignA = kA256Align_;
    bool innerAlignB = n256Align_;
    uint64_t innerSizeA = args_.kValue;
    uint64_t innerSizeB = args_.nValue;
    uint64_t outerSizeA = args_.mValue;
    uint64_t outerSizeB = args_.kValue;
    trans_ = MatmulV3Trans::NO_TRANS;
    if (args_.isATrans) {
        trans_ = MatmulV3Trans::A_TRANS;
        innerAlignA = m256Align_;
        innerSizeA = args_.mValue;
        outerSizeA = args_.kValue;
    }
    if (args_.isBTrans) {
        trans_ = MatmulV3Trans::B_TRANS;
        innerAlignB = kB256Align_;
        innerSizeB = args_.kValue;
        outerSizeB = args_.nValue;
    }
    if (args_.isATrans && args_.isBTrans) {
        trans_ = MatmulV3Trans::AB_TRANS;
    }
    calcMBasic_ = compileInfo_.l0CSize == L0C_SIZE_256_KB ? CALC_MN_BASIC_L0C_256 : CALC_M_BASIC;
    calcMNBasic_ = compileInfo_.l0CSize == L0C_SIZE_256_KB ? CALC_MN_BASIC_L0C_256 : CALC_MN_BASIC;
    l2TileLength_ = L2_TILE_LENGTH;
    OP_TILING_CHECK(!compileInfo_.supportL0c2out, tilingSelect_ = TilingCalcSelect::BASE, return ge::GRAPH_SUCCESS);

    // check the size is equaled to {32, 64, 96, 128, 160, 192, 224, 256, 384}.
    bool supportNd2NzOnTheWayA = IsOnTheWay(args_.aFormat, innerSizeA, aDtypeSize_, SUPPORT_ND2NZ_GM2L0);
    bool supportNd2NzOnTheWayB = IsOnTheWay(args_.bFormat, innerSizeB, bDtypeSize_, SUPPORT_ND2NZ_GM2L0);
    args_.nd2nzA = ((!innerAlignA || innerSizeA > ND2NZ_ON_THE_FLY_LIMIT) && (args_.aFormat == ge::FORMAT_ND) &&
        (!supportNd2NzOnTheWayA) &&
        !(args_.aType == ge::DT_FLOAT && !args_.isHf32 && innerSizeA < ND2NZ_ON_THE_FLY_LIMIT) &&
        !(args_.aType == ge::DT_FLOAT && args_.isHf32 && innerSizeA * aDtypeSize_ < CACHELINE));
    args_.nd2nzB = ((!innerAlignB || innerSizeB > ND2NZ_ON_THE_FLY_LIMIT) && (args_.bFormat == ge::FORMAT_ND) &&
        (!supportNd2NzOnTheWayB) &&
        !(args_.bType == ge::DT_FLOAT && !args_.isHf32 && innerSizeB < ND2NZ_ON_THE_FLY_LIMIT) &&
        !(args_.bType == ge::DT_FLOAT && args_.isHf32 && innerSizeB * bDtypeSize_ < CACHELINE));

    OP_LOGD(args_.opName, "After judging nd2nz tiling condition, matrix A need normal mode nd2nz = %u, matrix B = %u.",
            static_cast<uint32_t>(args_.nd2nzA), static_cast<uint32_t>(args_.nd2nzB));
    args_.nd2nzA = args_.nd2nzA || NeedNd2NzVnchw(outerSizeA, innerSizeA, supportNd2NzOnTheWayA,
                                                  aDtypeSize_, args_.aFormat);
    args_.nd2nzB = args_.nd2nzB || NeedNd2NzVnchw(outerSizeB, innerSizeB, supportNd2NzOnTheWayB,
                                                  bDtypeSize_, args_.bFormat);
    // (k, n) n为16384的倍数时，mata冲突严重，m越大，右矩阵重复载入越多，冲突影响越大，将右矩阵先做nd2nz
    // 限制为fp16、bf16场景
    constexpr uint64_t nMataThread = 16384;
    constexpr uint64_t mMataThread = 4096;
    constexpr uint64_t kMataThread = 6656;
    bool mataConflictFlag = !args_.isBTrans && args_.nValue % nMataThread == 0 && args_.mValue > mMataThread &&
                            args_.kValue >= kMataThread && args_.bFormat == ge::FORMAT_ND &&
                            (args_.aType == ge::DT_FLOAT16 || args_.aType == ge::DT_BF16);
    args_.nd2nzB = args_.nd2nzB || mataConflictFlag;
    OP_LOGD(args_.opName, "After judging nd2nz tiling condition, matrix A need vnchw mode nd2nz = %u, matrix B = %u.",
            static_cast<uint32_t>(args_.nd2nzA), static_cast<uint32_t>(args_.nd2nzB));
    return ge::GRAPH_SUCCESS;
}

inline void ResetBase(MatmulV3RunInfo &runInfo, const uint64_t l0CSize, const uint64_t dTypeSize = 2)
{
    runInfo.baseM = (l0CSize == L0C_SIZE_256_KB) ? BASIC_BLOCK_SIZE_256 : BASIC_BLOCK_SIZE_128;
    runInfo.baseN = BASIC_BLOCK_SIZE_256; // 256 is better base
    runInfo.baseK = BASIC_BLOCK_K_128_BYTE / dTypeSize;
    runInfo.stepM = BASE_STEP;
    runInfo.stepN = BASE_STEP;
    runInfo.iterateOrder = ITER_COL_FIRST;
    runInfo.dbL0c = DB_OFF_SIZE;
}

bool MatmulV3BaseTiling::IsOnTheWay(ge::Format matFormat, uint64_t innerSize, uint64_t dtypeSize,
                                    const std::vector<uint64_t> supportNd2nzList) const
{
    if (matFormat == ge::FORMAT_ND) {
        return (std::find(supportNd2nzList.begin(), supportNd2nzList.end(), innerSize * dtypeSize) !=
                supportNd2nzList.end());
    }
    return false;
}

bool MatmulV3BaseTiling::NeedNd2NzVnchw(uint64_t outerSize, uint64_t innerSize, bool supportNd2NzOnTheWay,
                                        uint64_t dtypeSize, ge::Format matFormat) const
{
    if (matFormat == ge::FORMAT_ND) {
        bool innerAlign = Is256BAlign(innerSize, dtypeSize);
        // 外轴<=8192 会导致数据量增大减慢搬运, 192B为最大奇数内轴长度, 384B为最大偶数内轴长度
        bool willFitVnchwCond = outerSize > 8192 && (innerSize > 1) && (innerSize * dtypeSize <= 192 ||
                                (innerSize * dtypeSize <= 384 && innerSize % 2 == 0) ||
                                (innerSize * dtypeSize <= CACHELINE && innerSize % 4 == 0));
        bool willInnerSizeEqualC0 = (innerSize == (32 / dtypeSize));
        return (willFitVnchwCond && !innerAlign && !supportNd2NzOnTheWay && !willInnerSizeEqualC0);
    }
    return false;
}

ge::graphStatus MatmulV3BaseTiling::DoOpTiling()
{
    OP_TILING_CHECK(GetMoreArgs() != ge::GRAPH_SUCCESS, CUBE_INNER_ERR_REPORT(args_.opName, "invalid context"),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckDimsAligned310P() != ge::GRAPH_SUCCESS, CUBE_INNER_ERR_REPORT(args_.opName, "invalid context"),
        return ge::GRAPH_FAILED);

    if (InitTilingData() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    SetRunInfo();
    SetParamsV310();
    DoBasicTiling();
    SetNd2NzInfo();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulV3BaseTiling::CheckDimsAligned310P()
{
    if (compileInfo_.supportL12BtBf16 || compileInfo_.supportL0c2out) {
        return ge::GRAPH_SUCCESS;
    }
    if (args_.outFormat == ge::FORMAT_ND) {
        if (args_.nOriValue * bDtypeSize_ % BLOCK_BYTE_SIZE != 0) {
            OP_LOGE(args_.opName, "shape of n should be 32-byte aligned");
            return ge::GRAPH_FAILED;
        }
    }
    if (args_.aFormat == ge::FORMAT_ND) {
        if (args_.mOriValue * aDtypeSize_ % BLOCK_BYTE_SIZE != 0) {
            OP_LOGE(args_.opName, "shape of m should be 32-byte aligned");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulV3BaseTiling::InitTilingData()
{
    auto aFormat = args_.aFormat == ge::FORMAT_ND ? matmul_tiling::CubeFormat::ND : matmul_tiling::CubeFormat::NZ;
    auto bFormat = args_.bFormat == ge::FORMAT_ND ? matmul_tiling::CubeFormat::ND : matmul_tiling::CubeFormat::NZ;
    auto cFormat = args_.outFormat == ge::FORMAT_ND ? matmul_tiling::CubeFormat::ND : matmul_tiling::CubeFormat::NZ;
    mm_.SetAType(matmul_tiling::TPosition::GM, aFormat, DTYPE_MAP.at(args_.aType), args_.isATrans);
    mm_.SetBType(matmul_tiling::TPosition::GM, bFormat, DTYPE_MAP.at(args_.bType), args_.isBTrans);
    mm_.SetCType(matmul_tiling::TPosition::GM, cFormat, DTYPE_MAP.at(args_.cType));
    mm_.SetDim(compileInfo_.aicNum);
    mm_.SetShape(args_.mValue, args_.nValue, args_.kValue);
    mm_.SetOrgShape(args_.mValue, args_.nValue, args_.kValue);
    if (args_.hasBias) {
        mm_.SetBias(true);
        mm_.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, DTYPE_MAP.at(args_.biasType));
    }
    mm_.SetBufferSpace(compileInfo_.l1Size, compileInfo_.l0CSize, compileInfo_.ubSize);
    if (mm_.GetTiling(tilingData_.matmulTiling) == -1) {
        OP_LOGE(args_.opName, "MatmulV3 Get Tiling Failed!");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void MatmulV3BaseTiling::SetParamsV310()
{
    if (!compileInfo_.supportL12BtBf16) {
        return;
    }
    m256Align_ = true;
    kA256Align_ = true;
    kB256Align_ = true;
    n256Align_ = true;
    calcMBasic_ = CALC_M_BASIC_L0C_256;
    calcMNBasic_ = CALC_MN_BASIC_L0C_256;
    l2TileLength_ = L2_TILE_LENGTH_L2_128;
}

void MatmulV3BaseTiling::SetNd2NzInfo()
{
    tilingData_.set_baseAN(static_cast<uint32_t>(runInfo_.baseAN));
    tilingData_.set_baseAD(static_cast<uint32_t>(runInfo_.baseAD));
    tilingData_.set_baseBN(static_cast<uint32_t>(runInfo_.baseBN));
    tilingData_.set_baseBD(static_cast<uint32_t>(runInfo_.baseBD));
}

void MatmulV3BaseTiling::SetRunInfo()
{
    tilingData_.matmulRunInfo.set_transA(static_cast<uint32_t>(args_.isATrans));
    tilingData_.matmulRunInfo.set_transB(static_cast<uint32_t>(args_.isBTrans));
    tilingData_.matmulRunInfo.set_nd2nzA(static_cast<uint32_t>(args_.nd2nzA));
    tilingData_.matmulRunInfo.set_nd2nzB(static_cast<uint32_t>(args_.nd2nzB));
    tilingData_.matmulRunInfo.set_isHf32(static_cast<uint32_t>(args_.isHf32));
}

bool MatmulV3BaseTiling::GetTilingInputArgs(std::shared_ptr<void> &inputArgs, size_t &size)
{
    std::shared_ptr<tuningtiling::GemmInputArgs> matmulInputArgs = RuntimeKb::MakeShared<tuningtiling::GemmInputArgs>();
    if (matmulInputArgs == nullptr) {
        OP_LOGI(args_.opName, "get tiling from repo error: input args is nullptr.");
        return false;
    }
    uint64_t blockCube = 16;                                  // 16元素为32B对齐
    uint64_t reduce = (args_.aType == ge::DT_FLOAT) ? 8 : 16; // fp32元素数为8，fp16元素数16，用于32B对齐
    matmulInputArgs->trans_a_flag = args_.isATrans;
    matmulInputArgs->trans_b_flag = args_.isBTrans;
    matmulInputArgs->bias_flag = args_.hasBias;
    matmulInputArgs->a_format = args_.aFormat;
    matmulInputArgs->b_format = args_.bFormat;
    matmulInputArgs->out_format = args_.outFormat;
    matmulInputArgs->a_dtype = (args_.aType == ge::DT_BF16) ? ge::DT_FLOAT16 : args_.aType;
    matmulInputArgs->b_dtype = (args_.bType == ge::DT_BF16) ? ge::DT_FLOAT16 : args_.bType;
    matmulInputArgs->out_dtype = (args_.cType == ge::DT_BF16) ? ge::DT_FLOAT16 : args_.cType;
    matmulInputArgs->m = static_cast<int64_t>(MathUtil::Align(args_.mValue, blockCube));
    matmulInputArgs->k = static_cast<int64_t>(MathUtil::Align(args_.kValue, reduce));
    matmulInputArgs->n = static_cast<int64_t>(MathUtil::Align(args_.nValue, blockCube));
    matmulInputArgs->m_align_flag = static_cast<int64_t>(args_.mValue) == matmulInputArgs->m;
    matmulInputArgs->k_align_flag = static_cast<int64_t>(args_.kValue) == matmulInputArgs->k;
    matmulInputArgs->n_align_flag = static_cast<int64_t>(args_.nValue) == matmulInputArgs->n;
    matmulInputArgs->batch_a1 = 1;
    matmulInputArgs->batch_a2 = 1;
    matmulInputArgs->batch_a3 = 1;
    matmulInputArgs->batch_a4 = 1;
    matmulInputArgs->batch_b1 = 1;
    matmulInputArgs->batch_b2 = 1;
    matmulInputArgs->batch_b3 = 1;
    matmulInputArgs->batch_b4 = 1;
    matmulInputArgs->l1_fused_num = 0;
    matmulInputArgs->aub_double_num = 1;
    matmulInputArgs->bub_double_num = 1;
    matmulInputArgs->fused_double_operand_num = 0;
    matmulInputArgs->reserved_bool = false;
    matmulInputArgs->reserved_params1 = 0;
    matmulInputArgs->reserved_params2 = 0;
    matmulInputArgs->reserved_params3 = 0;
    matmulInputArgs->reserved_params4 = 0;
    matmulInputArgs->reserved_params5 = 0;
    matmulInputArgs->reserved_params6 = 0;

    inputArgs = matmulInputArgs;
    size = sizeof(tuningtiling::GemmInputArgs);
    DebugLog(matmulInputArgs);
    return true;
}

void MatmulV3BaseTiling::DebugLog(const std::shared_ptr<tuningtiling::GemmInputArgs> &inputArgs)
{
    OP_LOGD(args_.opName,
        "Binary tiling info dict, "
        "a_dtype: %d, a_format: %d, aub_double_num: %f, "
        "b_dtype: %d, b_format: %d, "
        "batch_a1: %ld, batch_a2: %ld, batch_a3: %ld, batch_a4: %ld, "
        "batch_b1: %ld, batch_b2: %ld, batch_b3: %ld, batch_b4: %ld, "
        "bias_flag: %d, bub_double_num: %f, fused_double_operand_num: %f, "
        "k: %ld, k_align_flag: %d, l1_fused_num: %f, m: %ld, m_align_flag: %d, "
        "n: %ld, n_align_flag: %d, out_dtype: %d, out_format: %d, "
        "reserved_bool: %d, reserved_params1: %lu, reserved_params2: %lu, reserved_params3: %lu, "
        "reserved_params4: %lu, reserved_params5: %lu, reserved_params6: %lu, "
        "trans_a_flag: %d, trans_b_flag: %d.",
        static_cast<int32_t>(inputArgs->a_dtype), static_cast<int32_t>(inputArgs->a_format), inputArgs->aub_double_num,
        static_cast<int32_t>(inputArgs->b_dtype), static_cast<int32_t>(inputArgs->b_format), inputArgs->batch_a1,
        inputArgs->batch_a2, inputArgs->batch_a3, inputArgs->batch_a4, inputArgs->batch_b1, inputArgs->batch_b2,
        inputArgs->batch_b3, inputArgs->batch_b4, static_cast<int32_t>(inputArgs->bias_flag), inputArgs->bub_double_num,
        inputArgs->fused_double_operand_num, inputArgs->k, static_cast<int32_t>(inputArgs->k_align_flag),
        inputArgs->l1_fused_num, inputArgs->m, static_cast<int32_t>(inputArgs->m_align_flag), inputArgs->n,
        static_cast<int32_t>(inputArgs->n_align_flag), static_cast<int32_t>(inputArgs->out_dtype),
        static_cast<int32_t>(inputArgs->out_format), static_cast<int32_t>(inputArgs->reserved_bool),
        inputArgs->reserved_params1, inputArgs->reserved_params2, inputArgs->reserved_params3,
        inputArgs->reserved_params4, inputArgs->reserved_params5, inputArgs->reserved_params6,
        static_cast<int32_t>(inputArgs->trans_a_flag), static_cast<int32_t>(inputArgs->trans_b_flag));
}

void MatmulV3BaseTiling::SetBaseBlockTiling()
{
    runInfo_.usedCoreNum = compileInfo_.aicNum;
    runInfo_.baseM = BASIC_BLOCK_SIZE_128;
    runInfo_.baseN = BASIC_BLOCK_SIZE_256;
    runInfo_.baseK = BASIC_BLOCK_K_128_BYTE / aDtypeSize_;
    if (compileInfo_.l0CSize == L0C_SIZE_256_KB) {
        runInfo_.baseM = BASIC_BLOCK_SIZE_256;
        return;
    }
    if (!args_.isATrans && !args_.isBTrans) {
        return;
    }
    if (args_.isATrans && args_.isBTrans) {
        runInfo_.baseM = BASIC_BLOCK_SIZE_256;
        runInfo_.baseN = BASIC_BLOCK_SIZE_128;
        return;
    }
    bool mEnableLarge = args_.mValue >= (compileInfo_.aicNum >> 1) * BASIC_BLOCK_SIZE_256;
    bool nEnableLarge = args_.nValue >= (compileInfo_.aicNum >> 1) * BASIC_BLOCK_SIZE_256;
    if (mEnableLarge && nEnableLarge) {
        uint64_t sizeMLarge = MathUtil::CeilDivision(args_.mValue, BASIC_BLOCK_SIZE_256) * args_.nValue +
            MathUtil::CeilDivision(args_.nValue, BASIC_BLOCK_SIZE_128) * args_.mValue;
        uint64_t sizeNLarge = MathUtil::CeilDivision(args_.mValue, BASIC_BLOCK_SIZE_128) * args_.nValue +
            MathUtil::CeilDivision(args_.nValue, BASIC_BLOCK_SIZE_256) * args_.mValue;
        if (sizeMLarge < sizeNLarge) {
            runInfo_.baseM = BASIC_BLOCK_SIZE_256;
            runInfo_.baseN = BASIC_BLOCK_SIZE_128;
        }
        return;
    }
    if (mEnableLarge && !nEnableLarge) {
        runInfo_.baseM = BASIC_BLOCK_SIZE_256;
        runInfo_.baseN = BASIC_BLOCK_SIZE_128;
        return;
    }
    if (!mEnableLarge && !nEnableLarge) {
        if (args_.mValue > args_.nValue) {
            runInfo_.baseM = BASIC_BLOCK_SIZE_256;
            runInfo_.baseN = BASIC_BLOCK_SIZE_128;
        }
    }
    return;
}

inline void Convert2AscendCTiling(Tiling &tbeTiling, const BatchmatmulRunParas &runParams, const MatmulV3Args &args,
    uint64_t aicNum, MatmulV3RunInfo &runInfo)
{
    if (tbeTiling.al1_full_load && !runParams.pattern_flag) {
        tbeTiling.kal1_16 = runParams.k / tbeTiling.k_dim;
        tbeTiling.m_al1 = MathUtil::CeilDivision(runParams.m, tbeTiling.m_dim * tbeTiling.m_l0);
    }

    if (tbeTiling.bl1_full_load && !runParams.pattern_flag) {
        tbeTiling.kbl1_16 = runParams.k / tbeTiling.k_dim;
        tbeTiling.n_bl1 = MathUtil::CeilDivision(runParams.n, tbeTiling.n_dim * tbeTiling.n_l0);
    }
    uint64_t blockCube = 16; // 16元素为32B对齐
    runInfo.usedCoreNum = static_cast<uint64_t>(tbeTiling.m_dim) * tbeTiling.n_dim;
    // tbe tiling超核时，ascend使能最大核数即可，kernel侧会按base块数量分配
    if (runInfo.usedCoreNum > aicNum) {
        runInfo.usedCoreNum = aicNum;
    }
    runInfo.baseM = static_cast<uint64_t>(tbeTiling.m_l0) * blockCube;
    runInfo.baseN = static_cast<uint64_t>(tbeTiling.n_l0) * blockCube;
    runInfo.baseK = static_cast<uint64_t>(tbeTiling.k_l0) * blockCube;
    runInfo.singleCoreM = runInfo.baseM;
    runInfo.singleCoreN = runInfo.baseN;
    runInfo.singleCoreK = args.kValue;
    runInfo.stepM = static_cast<uint64_t>(tbeTiling.m_al1);
    runInfo.stepN = static_cast<uint64_t>(tbeTiling.n_bl1);
    runInfo.iterateOrder = ITER_COL_FIRST;
    runInfo.stepKa = static_cast<uint64_t>(tbeTiling.kal1_16 / tbeTiling.k_l0);
    runInfo.stepKb = static_cast<uint64_t>(tbeTiling.kbl1_16 / tbeTiling.k_l0);
    runInfo.depthA1 = runInfo.stepM * runInfo.stepKa * static_cast<uint64_t>(tbeTiling.db_al1);
    runInfo.depthB1 = runInfo.stepN * runInfo.stepKb * static_cast<uint64_t>(tbeTiling.db_bl1);
    runInfo.dbL0c = static_cast<uint64_t>(tbeTiling.db_l0c);
    runInfo.l2Info.mTile = 1;
    runInfo.l2Info.nTile = 1;
    runInfo.l2Info.mTileBlock = 1;
    runInfo.l2Info.nTileBlock = 1;
}

inline void InitRunParams(const BatchmatmulCompileParas &compileParams, const MatmulV3Args &args,
                          BatchmatmulRunParas &runParams)
{
    runParams.trans_a_flag = args.isATrans;
    runParams.trans_b_flag = args.isBTrans;
    runParams.format_a_nd = true;
    runParams.format_b_nd = true;
    runParams.format_out_nd = true;
    runParams.format_a = ge::FORMAT_ND;
    runParams.format_b = ge::FORMAT_ND;
    runParams.format_out = ge::FORMAT_ND;
    runParams.reserved_bool = 0;
    runParams.nd_flag = true;
    runParams.use_pre_ub = false;
    runParams.weight_nz_flag = false;
    runParams.batch_a1 = 1;
    runParams.batch_a2 = 1;
    runParams.batch_a3 = 1;
    runParams.batch_a4 = 1;
    runParams.batch_b1 = 1;
    runParams.batch_b2 = 1;
    runParams.batch_b3 = 1;
    runParams.batch_b4 = 1;

    runParams.b_have_batch = false;
    runParams.is_batch_matmul_mode = false;
    runParams.is_batch_matmul_op = false;
    uint64_t blockCube = 16;                                  // 16元素为32B对齐
    uint64_t reduce = (args.aType == ge::DT_FLOAT) ? 8 : 16;  // fp32元素数为8，fp16元素数16，用于32B对齐
    bool alignedMKN =
        (args.mValue % blockCube == 0) && (args.kValue % reduce == 0) && (args.nValue % blockCube == 0);
    runParams.used_aligned_pattern = alignedMKN && runParams.nd_flag;
    runParams.bias_flag = args.hasBias;
    runParams.pattern_flag = compileParams.pattern_flag;
    runParams.unaligned_flag = !alignedMKN;
    runParams.zero_flag = compileParams.zero_flag;
    runParams.hf32_flag = 0;
    runParams.dtype_a = args.aType;
    runParams.dtype_b = args.bType;
    runParams.dtype_out = args.cType;
    runParams.dtype_bias = ge::GetSizeByDataType(args.biasType);
    runParams.m = static_cast<int64_t>(ops::CeilDiv(args.mValue, blockCube));
    runParams.k = static_cast<int64_t>(ops::CeilDiv(args.kValue, reduce));
    runParams.n = static_cast<int64_t>(ops::CeilDiv(args.nValue, blockCube));
    runParams.batch = 1;
    runParams.ori_shape_m = static_cast<int64_t>(args.mValue);
    runParams.ori_shape_k = static_cast<int64_t>(args.kValue);
    runParams.ori_shape_n = static_cast<int64_t>(args.nValue);
    runParams.bias_dtype = args.biasType;
}

void MatmulV3BaseTiling::GetV2Tiling()
{
    BatchmatmulCompileParas compileParams;
    compileParams.binary_mode_flag = true;
    compileParams.bias_flag = args_.hasBias;
    compileParams.pattern_flag = optiling::PlatformInfo::GetInstance().support_l0c2out();
    compileParams.zero_flag = false;
    compileParams.aub_double_num = 1;
    compileParams.bub_double_num = 1;

    BatchmatmulRunParas runParams;
    InitRunParams(compileParams, args_, runParams);

    Tiling tiling;
    tiling.tiling_id = std::numeric_limits<uint64_t>::max();
    GenTiling("MatMulV3", compileParams, runParams, tiling, context_);

    Convert2AscendCTiling(tiling, runParams, args_, compileInfo_.aicNum, runInfo_);
    return;
}

bool MatmulV3BaseTiling::CheckUbOverFlow(uint64_t nAligned16, uint64_t nValue, const uint64_t &baseN,
    const uint64_t &baseD, uint64_t dtypeSize)
{
    uint64_t nAlignedLoop = MathUtil::CeilDivision(nAligned16, baseN);
    uint64_t nValueLoop = MathUtil::CeilDivision(nValue, baseN);
    return ((nAlignedLoop != nValueLoop) &&
        ((nAligned16 - ((nValue / baseN) - 1) * baseN) * baseD > (compileInfo_.ubSize / NUMBER_TWO / dtypeSize)));
}

void MatmulV3BaseTiling::CalcNd2NzTiling(uint64_t dtypeSize, uint64_t nValue, uint64_t dValue, uint64_t &baseN,
    uint64_t &baseD)
{
    uint64_t vectorCoreNum = NUMBER_TWO * runInfo_.usedCoreNum;
    vectorCoreNum = std::max(vectorCoreNum, 1UL);
    uint64_t baseThres = VECTOR_D_BASE / dtypeSize;
    uint64_t c0 = BLOCK_BYTE_SIZE / dtypeSize;
    uint64_t nAligned16 = ops::CeilAlign(nValue, N_ALIGNED);
    uint64_t dAlignedC0 = ops::CeilAlign(dValue, c0);
    if (dValue <= baseThres) {
        baseD = std::max(ops::CeilAlign(dValue, c0), uint64_t(1));
        uint64_t nDim = vectorCoreNum;
        baseN = compileInfo_.ubSize / NUMBER_TWO / dtypeSize / baseD;
        uint64_t round = std::max(MathUtil::CeilDivision(MathUtil::CeilDivision(nAligned16, nDim), baseN), 1UL);
        baseN = std::max(MathUtil::CeilDivision(MathUtil::CeilDivision(nAligned16, nDim), round), NCALC_THRES);
        while (baseN > NCALC_THRES) {
            if (CheckUbOverFlow(nAligned16, nValue, baseN, baseD, dtypeSize)) {
                baseN--;
                continue;
            }
            break;
        }
        return;
    }
    uint64_t lastTail = 0;
    // bestBaseD = 4K / dtype, bestBaseN = UB / 2 / 4096
    uint64_t bestBaseN = NCALC_THRES;
    uint64_t bestBaseD = CALC_ND_BASIC.at(1) / dtypeSize;
    for (auto base : CALC_ND_BASIC) {
        baseD = std::max(std::min(dAlignedC0, base / dtypeSize), uint64_t(1));
        uint64_t dLoop = MathUtil::CeilDivision(dAlignedC0, baseD);
        uint64_t dTail = dAlignedC0 % baseD;
        if (dTail > 0 && dTail < (MIN_TAIL / dtypeSize)) {
            if (baseD * dtypeSize == CALC_ND_BASIC.at(0)) {  // if dtail < 0.5K && baseD == 3k, give up and return bestbase
                continue;
            }
            dLoop--;
            baseD = std::max(ops::CeilAlign(MathUtil::CeilDivision(dAlignedC0, dLoop),
                c0), uint64_t(1));;  // dtail< 0.5K, make baseD larger
        }
        baseN = std::max(compileInfo_.ubSize / NUMBER_TWO / dtypeSize / baseD, NUMBER_SIXTEEN);
        if (baseN * baseD * dtypeSize * NUMBER_TWO > compileInfo_.ubSize) { // check ub overflow
            continue;
        }
        if (CheckUbOverFlow(nAligned16, nValue, baseN, baseD, dtypeSize)) {
            continue;
        }
        uint64_t nLoop = MathUtil::CeilDivision(nAligned16, baseN);
        uint64_t tail = nLoop * dLoop % vectorCoreNum;
        while (baseN > NCALC_THRES) {
            if (CheckUbOverFlow(nAligned16, nValue, baseN, baseD, dtypeSize)) { // check ub overflow
                baseN--;
                nLoop = MathUtil::CeilDivision(nAligned16, baseN);
                tail = nLoop * dLoop % vectorCoreNum;
                continue;
            }
            if (tail == 0) {
                return;
            }
            if (tail > lastTail) {
                lastTail = tail;
                bestBaseD = baseD;
                bestBaseN = baseN;
            }
            baseN--;
            nLoop = MathUtil::CeilDivision(nAligned16, baseN);
            tail = nLoop * dLoop % vectorCoreNum;
        }
    }
    baseD = bestBaseD;
    baseN = bestBaseN;
    return;
}

void MatmulV3BaseTiling::DoNd2NzVectorTiling() {
    UpdateNd2nzFlag();
    OP_LOGI(args_.opName, "args_.nd2nzA = %d, args_.nd2nzB = %d.", static_cast<uint32_t>(args_.nd2nzA),
        static_cast<uint32_t>(args_.nd2nzB));
    if (args_.nd2nzA) {
        uint64_t nValue = args_.isATrans ? args_.kValue : args_.mValue;
        uint64_t dValue = args_.isATrans ? args_.mValue : args_.kValue;
        CalcNd2NzTiling(aDtypeSize_, nValue, dValue, runInfo_.baseAN, runInfo_.baseAD);
        OP_LOGI(args_.opName, "baseAN is %ld baseAD is %ld", runInfo_.baseAN, runInfo_.baseAD);
    }
    if (args_.nd2nzB) {
        uint64_t nValue = args_.isBTrans ? args_.nValue : args_.kValue;
        uint64_t dValue = args_.isBTrans ? args_.kValue : args_.nValue;
        CalcNd2NzTiling(bDtypeSize_, nValue, dValue, runInfo_.baseBN, runInfo_.baseBD);
        OP_LOGI(args_.opName, "baseBN is %ld baseBD is %ld", runInfo_.baseBN, runInfo_.baseBD);
    }
}

void MatmulV3BaseTiling::DoSmallShapeTiling()
{
    OP_LOGI(args_.opName, "meet small block num or meet small shape situation, enable small shape algorithm");
    FormulaicBaseBlockTiling();
    if (!(compileInfo_.l0CSize == L0C_SIZE_256_KB)) {
        if (args_.mValue < SMALL_SHAPE_THRES && args_.mValue > SMALL_SHAPE_LOWER_THRES &&
            runInfo_.baseN > SMALL_SHAPE_LOWER_THRES) {
            runInfo_.baseN = SMALL_SHAPE_LOWER_THRES;
        } else if (args_.nValue < SMALL_SHAPE_THRES && args_.nValue > SMALL_SHAPE_LOWER_THRES &&
            runInfo_.baseM > SMALL_SHAPE_LOWER_THRES) {
            runInfo_.baseM = SMALL_SHAPE_LOWER_THRES;
        } else if (runInfo_.baseM * runInfo_.baseN > L0C_THRES) {
            runInfo_.baseM = runInfo_.baseM > SMALL_SHAPE_LOWER_THRES && runInfo_.baseM < SMALL_SHAPE_THRES ?
                SMALL_SHAPE_LOWER_THRES :
                runInfo_.baseM;
            runInfo_.baseN = runInfo_.baseN > SMALL_SHAPE_LOWER_THRES && runInfo_.baseN < SMALL_SHAPE_THRES ?
                SMALL_SHAPE_LOWER_THRES :
                runInfo_.baseN;
        }
        if (runInfo_.baseM * runInfo_.baseN > L0C_THRES) {
            runInfo_.baseM = L0C_THRES / runInfo_.baseN;
        }
    }
}

void MatmulV3BaseTiling::DoSelectTiling()
{
    switch (tilingSelect_) {
        case TilingCalcSelect::ALL:
            DO_CACL_TILING_ENABLE(DoBL1FullloadWithFixpipeTiling())
            DO_CACL_TILING_ENABLE(DoAL1FullLoadTiling())
            DO_CACL_TILING_ENABLE(DoBL1FullLoadTiling())
            DO_CACL_TILING_ENABLE(DoL2CacheTiling())
            DO_CACL_TILING_ENABLE(DoSingleCoreSplitKTiling())
            DO_CACL_TILING_ENABLE(DoDeterministicMultiCoreSplitKTiling())
            DO_CACL_TILING_ENABLE(DoL2CacheTiling310P())
            break;
        case TilingCalcSelect::BASE:
            DO_CACL_TILING_ENABLE(DoL2CacheTiling())
            DO_CACL_TILING_ENABLE(DoL2CacheTiling310P())
            break;
        case TilingCalcSelect::SINGLE_CORE_SPLIT_K:
            DO_CACL_TILING_ENABLE(DoSingleCoreSplitKTiling())
            break;
        case TilingCalcSelect::DETERMINISTIC_SPLIT_K:
            DO_CACL_TILING_ENABLE(DoDeterministicMultiCoreSplitKTiling())
            break;
        default:
            break;
    }
}

void MatmulV3BaseTiling::DoBasicTiling()
{
    runInfo_.needUpdate = true;
    basicBlockBaseM_ = (compileInfo_.l0CSize == L0C_SIZE_256_KB) ? BASIC_BLOCK_SIZE_256 : BASIC_BLOCK_SIZE_128;
    ResetBase(runInfo_, compileInfo_.l0CSize, aDtypeSize_);

    runInfo_.singleCoreK = args_.kValue;
    uint64_t alignedMValue = ops::CeilAlign(args_.mValue, basicBlockBaseM_);
    uint64_t alignedNValue = ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_256);
    SetBaseBlockTiling();
    bool smallBlockNum = (alignedMValue / runInfo_.baseM) * (alignedNValue / runInfo_.baseN) < runInfo_.usedCoreNum;
    bool smallShape =
        (args_.mValue < SMALL_SHAPE_THRES) || (args_.nValue < SMALL_SHAPE_THRES); // 256为界限判断是否小shape
    if ((smallBlockNum || smallShape) && !compileInfo_.supportL12BtBf16) {
        DoSmallShapeTiling();
    }

    if (!compileInfo_.supportL0c2out) {
        runInfo_.baseM = std::min(runInfo_.baseM, args_.mValue);
        runInfo_.baseN = std::min(runInfo_.baseN, args_.nValue);
        runInfo_.baseK = std::min(runInfo_.baseK, args_.kValue);
    }

    if (compileInfo_.supportL12BtBf16) {
        const gert::Shape &shapeA = context_->GetInputShape(0)->GetStorageShape();
        const gert::Shape &shapeB = context_->GetInputShape(1)->GetStorageShape();
        const uint64_t dimsA = shapeA.GetDimNum();
        const uint64_t dimsB = shapeB.GetDimNum();
        // isolate bmm which input dims over 2
        if (smallBlockNum && (dimsA == 2 && dimsB == 2)) {
            FormulateBasicBlockDavid();
        }
        CalcTailBasicBlock();
    }

    CalL1Tiling();
    DoIncreTiling();

    DoSelectTiling();
    // add nd2nz tiling here
    DoNd2NzVectorTiling();
    if (args_.hasBias) {
      runInfo_.baseN = std::min(256UL, runInfo_.baseN);  // 有bias时 baseN 小于256
      if (tilingEnable_.tilingEnableSplitCore == TilingEnableSplitCore::BASE &&
          tilingEnable_.tilingEnableFullLoad != TilingEnableFullLoad::BL1_FULL_LOAD) {
            runInfo_.singleCoreN = runInfo_.baseN;
        }
    }
}

void MatmulV3BaseTiling::FormulateBasicBlockDavid()
{
    uint64_t mCore = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
    uint64_t nCore = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
    if (mCore <= nCore) {
        runInfo_.baseM = ops::CeilAlign(MathUtil::CeilDivision(args_.mValue, mCore), BASIC_BLOCK_SIZE_16);
        mCore = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
        if (mCore == 0) {
            OP_LOGE(args_.opName, "mCore is zero.");
        }
        nCore = runInfo_.usedCoreNum / mCore;
        runInfo_.baseN = ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, nCore), BASIC_BLOCK_SIZE_16);
    } else {
        runInfo_.baseN = ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, nCore), BASIC_BLOCK_SIZE_16);
        nCore = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
        mCore = runInfo_.usedCoreNum / nCore;
        runInfo_.baseM = ops::CeilAlign(MathUtil::CeilDivision(args_.mValue, mCore), BASIC_BLOCK_SIZE_16);
    }
    runInfo_.baseK = std::min(BASIC_BLOCK_SIZE_128 / aDtypeSize_, args_.kValue);
    runInfo_.baseK = ops::CeilAlign(runInfo_.baseK, BASIC_BLOCK_SIZE_16);
}

void MatmulV3BaseTiling::CalcTailBasicBlock()
{
    uint64_t mCnt = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
    uint64_t nCnt = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
    uint64_t tailCnt = mCnt * nCnt % compileInfo_.aicNum;
    tailCnt = mCnt * nCnt <= compileInfo_.aicNum ? 0 : tailCnt;
    runInfo_.l2Info.mTile = 1;
    runInfo_.l2Info.nTile = 1;
    if (tailCnt != 0) {
        while ((runInfo_.l2Info.mTile + 1) * runInfo_.l2Info.nTile * tailCnt <= compileInfo_.aicNum) {
            runInfo_.l2Info.mTile += 1;
            if (runInfo_.l2Info.mTile * (runInfo_.l2Info.nTile + 1) * tailCnt <= compileInfo_.aicNum) {
                runInfo_.l2Info.nTile += 1;
            }
        }
    }
}

void MatmulV3BaseTiling::FormulaicTilingNoTrans()
{
    uint64_t nCore = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN); // n方向上的base块个数
    CalcBase<CalcType::M_BY_BASE_NK>(calcMBasic_);
    // 非对齐场景
    if (!kA256Align_ && !n256Align_) {
        // A,B都会做ND2NZ，baseMNK 按照负载均衡分配
        BalanceBaseBlockTiling();
    } else if (!kA256Align_) {
        // A矩阵做ND2NZ，baseN*depth保证256Byte对齐，baseM任意切分
        runInfo_.baseN = BASIC_BLOCK_SIZE_256;
        runInfo_.baseK = BASIC_BLOCK_SIZE_128 / aDtypeSize_;
        runInfo_.baseM = CalBaseSize(nCore, compileInfo_.aicNum, args_.mValue, basicBlockBaseM_);
    } else if (!n256Align_) {
        // B矩阵做ND2NZ，baseK*depth保证256BYTE对齐，baseMN任意切分
        runInfo_.baseK = BASIC_BLOCK_SIZE_128 / aDtypeSize_;
        CalBaseMBaseN(runInfo_.baseM, runInfo_.baseN);
    }
}

void MatmulV3BaseTiling::FormulaicBaseBlockTiling()
{
    uint64_t mCore = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM); // m方向上的base块个数
    uint64_t nCore = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN); // n方向上的base块个数

    // ND场景，不需要ND2NZ操作, 说明数据内存方向的边是256B对齐的
    switch (trans_) {
        case MatmulV3Trans::NO_TRANS: {
            FormulaicTilingNoTrans();
            break;
        }
        case MatmulV3Trans::A_TRANS: {
            // [k, m], [k, n], m和n需要256B对齐，k可以任意切分
            // 如果对齐场景都为MTE2 bound，那非对齐场景的MTE2 bound会更严重，此时多核计算没有意义
            if (!m256Align_ && !n256Align_) {
                // A,B都会做ND2NZ，baseMNK 按照负载均衡分配
                BalanceBaseBlockTiling();
            } else if (!m256Align_) {
                // A矩阵做ND2NZ，baseN*depth保证256BYTE对齐，baseM任意切分
                runInfo_.baseM = CalBaseSize(nCore, compileInfo_.aicNum, args_.mValue, basicBlockBaseM_);
            } else if (!n256Align_) {
                runInfo_.baseN = CalBaseSize(mCore, compileInfo_.aicNum, args_.nValue, BASIC_BLOCK_SIZE_256);
            }
            break;
        }
        case MatmulV3Trans::B_TRANS: {
            // [m, k], [n, k], k需要256B对齐，m和n可以任意切分
            // 负载均匀切分
            CalcBase<CalcType::MN_BY_BASE_K>(calcMNBasic_);
            if (!kA256Align_) {
                // A,B都会做ND2NZ，baseMNK 按照负载均衡分配
                BalanceBaseBlockTiling();
            }
            break;
        }
        case MatmulV3Trans::AB_TRANS: {
            // [k, m], [n, k], m和k需要256B对齐，n可以任意切分
            runInfo_.baseN = CalBaseSize(mCore, compileInfo_.aicNum, args_.nValue, BASIC_BLOCK_SIZE_256);
            if (!m256Align_ && kB256Align_) {
                // A,B都会做ND2NZ，baseMNK 按照负载均衡分配
                BalanceBaseBlockTiling();
            } else if (!m256Align_) {
                CalBaseMBaseN(runInfo_.baseM, runInfo_.baseN);
            }
            break;
        }
        default:
            break;
    }
    // baseN>=512时, nd2nz vmuls指令dstRepeatStride会大于255, 只能设置repeat为1，指令效率不高
    if (args_.outFormat == ge::FORMAT_ND && !compileInfo_.supportL0c2out) {
        runInfo_.baseN = std::min(runInfo_.baseN, BASIC_BLOCK_SIZE_256);
    }
    return;
}

void MatmulV3BaseTiling::BalanceBaseBlockTiling()
{
    ResetBase(runInfo_, compileInfo_.l0CSize, aDtypeSize_);
    uint64_t nCore = 1;
    uint64_t mCore = 1;
    if (args_.mValue >= basicBlockBaseM_) {
        mCore = std::max(MathUtil::CeilDivision(args_.mValue, basicBlockBaseM_), 1UL);
        nCore = std::max(compileInfo_.aicNum / mCore, 1UL);
        runInfo_.baseN = ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, nCore), BASIC_ALIGN_16);
        runInfo_.baseN = std::min(runInfo_.baseN, BASIC_BLOCK_SIZE_256);
        uint64_t baseKa = compileInfo_.l0ASize / DB_SIZE / aDtypeSize_ / BASIC_BLOCK_SIZE_256;
        uint64_t baseKb = ops::FloorDiv(compileInfo_.l0BSize, DB_SIZE * bDtypeSize_ * runInfo_.baseN);
        runInfo_.baseK = ops::FloorAlign(std::min(baseKa, baseKb), BASIC_ALIGN_16);
    } else {
        uint64_t maxBaseN2 = args_.mValue >= 64 ? basicBlockBaseM_ : 1024; // 64 mlimit size; 1024 max size
        runInfo_.baseM = ops::CeilAlign(args_.mValue, BASIC_ALIGN_16);
        uint64_t nTile = std::max(args_.nValue / compileInfo_.aicNum, 1UL);
        runInfo_.baseN = ops::CeilAlign(nTile, BASIC_ALIGN_16);
        runInfo_.baseN = std::max(runInfo_.baseN, maxBaseN2);
        uint64_t maxBaseN = ops::FloorDiv(compileInfo_.l0CSize, LOC_DATA_SIZE * runInfo_.baseM);
        maxBaseN = ops::FloorAlign(maxBaseN, BASIC_ALIGN_16);
        if (args_.hasBias && compileInfo_.supportL0c2out) {
            maxBaseN = std::min(maxBaseN, compileInfo_.btSize / DATA_SIZE_FP32);
        }
        runInfo_.baseN = std::min(runInfo_.baseN, maxBaseN);
        nCore = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
        uint64_t tailCoreNum = 0;
        if (runInfo_.baseN > BASIC_BLOCK_SIZE_256) {
            tailCoreNum = MathUtil::CeilDivision(args_.nValue, BASIC_BLOCK_SIZE_256) % compileInfo_.aicNum;
        }
        if (nCore % compileInfo_.aicNum < tailCoreNum) {
            runInfo_.baseN = BASIC_BLOCK_SIZE_256;
        }
        uint64_t baseKa = ops::FloorDiv(compileInfo_.l0ASize, DB_SIZE * aDtypeSize_ * runInfo_.baseM);
        uint64_t baseKb = ops::FloorDiv(compileInfo_.l0BSize, DB_SIZE * bDtypeSize_ * runInfo_.baseN);
        runInfo_.baseK = ops::FloorAlign(std::min(baseKa, baseKb), BASIC_ALIGN_16);
    }

    if (runInfo_.baseM <= 0 || runInfo_.baseN <= 0 || runInfo_.baseK <= 0 || runInfo_.baseM % BASIC_ALIGN_16 != 0 ||
        runInfo_.baseN % BASIC_ALIGN_16 != 0 || runInfo_.baseK % BASIC_ALIGN_16 != 0) {
        ResetBase(runInfo_, compileInfo_.l0CSize, aDtypeSize_);
    }
    return;
}

bool MatmulV3BaseTiling::CheckBTSize(uint64_t baseN)
{
    if (!compileInfo_.supportL0c2out && baseN * DATA_SIZE_FP32 > compileInfo_.btSize) {
        return false;
    }
    return true;
}

template <CalcType T> void MatmulV3BaseTiling::CalcBase(const std::vector<std::vector<uint64_t>> &baseMNK)
{
    std::vector<std::vector<uint64_t>> calBaseMNK;
    size_t index = 0;
    uint64_t minLoadSize = -1;
    uint64_t nCnt = 1;
    uint64_t mCnt = 1;
    for (size_t i = 0; i < baseMNK.size(); ++i) {
        std::vector<uint64_t> tempBaseMNK(baseMNK[i].begin(), baseMNK[i].end());
        if (T == CalcType::M_BY_BASE_NK) {
            nCnt = MathUtil::CeilDivision(args_.nValue, baseMNK[i][1]);
            // baseM是除完n方向上的核数后，m方向上能分的最大base块
            tempBaseMNK[0] = CalBaseSize(nCnt, compileInfo_.aicNum, args_.mValue, baseMNK[i][0]);
            mCnt = MathUtil::CeilDivision(args_.mValue, tempBaseMNK[0]);
            calBaseMNK.push_back(tempBaseMNK);
        } else {
            CalBaseMBaseN(tempBaseMNK[0], tempBaseMNK[1], calcMNBasic_[i][0], calcMNBasic_[i][1]);
            calBaseMNK.push_back(tempBaseMNK);
            mCnt = MathUtil::CeilDivision(args_.mValue, tempBaseMNK[0]);
            nCnt = MathUtil::CeilDivision(args_.nValue, tempBaseMNK[1]);
        }
        if (args_.hasBias && !CheckBTSize(calBaseMNK[i][1])) {
            continue;
        }

        uint64_t tmpTailCore = (mCnt * nCnt) % compileInfo_.aicNum;
        uint64_t tmpLoadSize = (tempBaseMNK[0] + tempBaseMNK[1]) * (mCnt * nCnt / compileInfo_.aicNum);
        tmpLoadSize += (tmpTailCore > 0) ? (tempBaseMNK[0] + tempBaseMNK[1]) : 0;
        if (tmpLoadSize < minLoadSize) {
            index = i;
            minLoadSize = tmpLoadSize;
        }
    }
    runInfo_.baseM = calBaseMNK[index][0];               // 0: m index
    runInfo_.baseN = calBaseMNK[index][1];               // 1: n index
    runInfo_.baseK = calBaseMNK[index][2] / aDtypeSize_; // 2: k index
}

void MatmulV3BaseTiling::CalBaseMBaseN(uint64_t &baseM, uint64_t &baseN, uint64_t maxBaseM, uint64_t maxBaseN)
{
    // 根据 m方向和n方向上谁的基本块多，更新base块的大小
    uint64_t mCore = MathUtil::CeilDivision(args_.mValue, maxBaseM);
    uint64_t nCore = MathUtil::CeilDivision(args_.nValue, maxBaseN);
    if (mCore < nCore) {
        nCore = std::max(compileInfo_.aicNum / mCore, 1UL);
        baseN = ops::CeilAlign(std::max(args_.nValue / nCore, 1UL), BASIC_ALIGN_16);
        if (baseN > maxBaseN) {
            baseN = maxBaseN;
        }
    } else {
        mCore = std::max(compileInfo_.aicNum / nCore, 1UL);
        baseM = ops::CeilAlign(std::max(args_.mValue / mCore, 1UL), BASIC_ALIGN_16);
        if (baseM > maxBaseM) {
            baseM = maxBaseM;
        }
    }

    baseM = ops::CeilAlign(std::min(args_.mValue, baseM), BASIC_ALIGN_16);
    baseN = ops::CeilAlign(std::min(args_.nValue, baseN), BASIC_ALIGN_16);
    return;
}

void MatmulV3BaseTiling::CalL1TilingV200()
{
    runInfo_.singleCoreK = args_.kValue;
    runInfo_.stepM = 1;
    runInfo_.stepN = 1;
    runInfo_.depthA1 = 16; // 16 is full use l1 space
    runInfo_.depthB1 = 16; // 16 is full use l1 space
    runInfo_.singleCoreM = runInfo_.baseM;
    runInfo_.singleCoreN = runInfo_.baseN;

    if (args_.aFormat == ge::FORMAT_ND || args_.bFormat == ge::FORMAT_ND) {
        runInfo_.depthA1 = 6; // 6为经验值
        runInfo_.depthB1 = 6; // 6为经验值
    }
    runInfo_.stepKa = runInfo_.depthA1 / DB_SIZE;
    runInfo_.stepKb = runInfo_.depthB1 / DB_SIZE;
    if ((args_.aFormat == ge::FORMAT_FRACTAL_NZ) && (args_.bFormat == ge::FORMAT_FRACTAL_NZ)) {
        // ka全载
        if (args_.mValue <= runInfo_.baseM) {
            runInfo_.baseM = args_.mValue;
            if (runInfo_.baseM * args_.kValue * aDtypeSize_ < (compileInfo_.l1Size / NUM_HALF)) {
                runInfo_.depthA1 = MathUtil::CeilDivision(args_.kValue, runInfo_.baseK);
                runInfo_.stepKa = runInfo_.depthA1;
            }
        }
        // kb全载
        if (args_.nValue <= runInfo_.baseN) {
            runInfo_.baseN = args_.nValue;
            if (runInfo_.baseN * args_.kValue * bDtypeSize_ < (compileInfo_.l1Size / NUM_HALF)) {
                runInfo_.depthB1 = MathUtil::CeilDivision(args_.kValue, runInfo_.baseK);
                runInfo_.stepKb = runInfo_.depthB1;
            }
        }
    }
    return;
}

void MatmulV3BaseTiling::CalL1Tiling()
{
    if (!compileInfo_.supportL0c2out) {
        CalL1TilingV200();
        return;
    }
    uint64_t totalL1Size = compileInfo_.l1Size + 256; // 256B为预留给rpc使用，单算子不涉及
    uint64_t reserveBTSize = args_.hasBias ? BIAS_TABLE_NUM * DATA_SIZE_FP32 : 0;
    runInfo_.depthA1 = totalL1Size / NUM_HALF / runInfo_.baseM / runInfo_.baseK / aDtypeSize_; // 2: half of l1
    runInfo_.depthB1 = totalL1Size / NUM_HALF / runInfo_.baseN / runInfo_.baseK / bDtypeSize_; // 2: half of l1

    uint64_t depthASize = runInfo_.depthA1 * runInfo_.baseM * runInfo_.baseK * aDtypeSize_;
    uint64_t depthBSize = runInfo_.depthB1 * runInfo_.baseN * runInfo_.baseK * bDtypeSize_;
    if (depthASize + depthBSize > totalL1Size - reserveBTSize) {
        if (runInfo_.baseM <= runInfo_.baseN) {
            runInfo_.depthA1 = runInfo_.depthA1 / NUM_HALF; // 2: adjust deptch for l1 buffer
        } else {
            runInfo_.depthB1 = runInfo_.depthB1 / NUM_HALF; // 2: adjust deptch for l1 buffer
        }
    }
    runInfo_.stepKa = runInfo_.depthA1 / DB_SIZE;
    runInfo_.stepKb = runInfo_.depthB1 / DB_SIZE;

    UpdateL1TilingStepK(runInfo_.stepKa);
    UpdateL1TilingStepK(runInfo_.stepKb);

    if (runInfo_.stepKa >= runInfo_.stepKb) {
        runInfo_.stepKa = runInfo_.stepKa / runInfo_.stepKb * runInfo_.stepKb;
    } else {
        runInfo_.stepKb = runInfo_.stepKb / runInfo_.stepKa * runInfo_.stepKa;
    }
    runInfo_.depthA1 = runInfo_.stepKa * DB_SIZE; // depth % (stepKa * stepM) == 0
    runInfo_.depthB1 = runInfo_.stepKb * DB_SIZE; // depth % (stepKb * stepN) == 0
    runInfo_.singleCoreM = runInfo_.baseM;
    runInfo_.singleCoreN = runInfo_.baseN;
    return;
}

void MatmulV3BaseTiling::UpdateL1TilingStepK(uint64_t &stepK)
{
    if (stepK * runInfo_.baseK >= args_.kValue) {
        return;
    }

    if (stepK * runInfo_.baseK * aDtypeSize_ > BASIC_ALIGN_512) {
        if ((stepK * runInfo_.baseK * aDtypeSize_ % BASIC_ALIGN_512) != 0 &&
            (BASIC_ALIGN_512 % (runInfo_.baseK * aDtypeSize_)) == 0) {
            while (stepK * runInfo_.baseK * aDtypeSize_ % BASIC_ALIGN_512 != 0 && stepK > 1) {
                stepK--;
            }
        }
    } else if (stepK * runInfo_.baseK * aDtypeSize_ > BASIC_ALIGN_256) {
        if ((stepK * runInfo_.baseK * aDtypeSize_ % BASIC_ALIGN_256) != 0 &&
            (BASIC_ALIGN_256 % (runInfo_.baseK * aDtypeSize_)) == 0) {
            while (stepK * runInfo_.baseK * aDtypeSize_ % BASIC_ALIGN_256 != 0 && stepK > 1) {
                stepK--;
            }
        }
    }
}

void MatmulV3BaseTiling::InitL2SplitParams(MatmulV3L2SplitParams &l2SplitParams) const
{
    l2SplitParams.outBase = std::max(runInfo_.baseM, 1UL);
    l2SplitParams.innerBase = std::max(runInfo_.baseN, 1UL);
    l2SplitParams.outValue = args_.mValue;
    l2SplitParams.innerValue = args_.nValue;
    l2SplitParams.outDtypeSize = aDtypeSize_;
    l2SplitParams.innerDtypeSize = bDtypeSize_;
    if (runInfo_.baseN >= runInfo_.baseM) {
        l2SplitParams.outBase = runInfo_.baseN;
        l2SplitParams.innerBase = runInfo_.baseM;
        l2SplitParams.outValue = args_.nValue;
        l2SplitParams.innerValue = args_.mValue;
        l2SplitParams.outDtypeSize = bDtypeSize_;
        l2SplitParams.innerDtypeSize = aDtypeSize_;
    }

    l2SplitParams.maxConflictDim = 6;     // 24核最多冲突6核
    l2SplitParams.minConflictDim = 3;     // 24核内轴不亲和最多冲突3核
    if (compileInfo_.aicNum == 20) {      // 针对20核的场景
        l2SplitParams.maxConflictDim = 5; // 20核最多冲突5核
        l2SplitParams.minConflictDim = 4; // 20核内轴不亲和最多冲突4核
    }
}

bool MatmulV3BaseTiling::IsTailSmall(MatmulV3L2SplitParams &l2SplitParams, uint64_t outL2Split, uint64_t innerL2Split,
    uint64_t innerMaxConflict) const
{
    uint64_t outTailValue = ((l2SplitParams.outValue + outL2Split - 1) % outL2Split) + 1;
    uint64_t innerTailValue = ((l2SplitParams.innerValue + innerL2Split - 1) % innerL2Split) + 1;
    l2SplitParams.outTailCnt = MathUtil::CeilDivision(outTailValue, l2SplitParams.outBase);
    l2SplitParams.innerTailCnt = MathUtil::CeilDivision(innerTailValue, l2SplitParams.innerBase);
    bool isOutTailSmall = l2SplitParams.outTailCnt * l2SplitParams.maxConflictDim < compileInfo_.aicNum;
    bool isInnerTailSmall = l2SplitParams.innerTailCnt * innerMaxConflict < compileInfo_.aicNum;
    return (isOutTailSmall || isInnerTailSmall);
}

bool MatmulV3BaseTiling::CalcTile(uint64_t &outTile, uint64_t &innerTile, uint64_t &outL2Split, uint64_t &innerL2Split,
    const bool isInnerBad) const
{
    MatmulV3L2SplitParams l2SplitParams;
    InitL2SplitParams(l2SplitParams);
    uint64_t innerMaxConflict = isInnerBad ? l2SplitParams.minConflictDim : l2SplitParams.maxConflictDim;
    uint64_t outerMinUseDim = compileInfo_.aicNum / l2SplitParams.maxConflictDim;
    uint64_t innerMinUseDim = compileInfo_.aicNum / innerMaxConflict;
    uint64_t outOriShape = outL2Split;
    uint64_t innerOriShape = innerL2Split;
    uint64_t outConflict = 0;
    uint64_t innerConflict = 0;
    bool enableCache = false;

    for (uint64_t outerUseDim = compileInfo_.aicNum; outerUseDim >= outerMinUseDim; outerUseDim--) {
        for (uint64_t innerUseDim = compileInfo_.aicNum; innerUseDim >= innerMinUseDim; innerUseDim--) {
            uint64_t outTileTmp = std::max(outOriShape / (l2SplitParams.outBase * outerUseDim), 1UL);
            uint64_t innerTileTmp = std::max(innerOriShape / (l2SplitParams.innerBase * innerUseDim), 1UL);
            uint64_t outL2SplitTmp =
                MathUtil::Align(MathUtil::CeilDivision(l2SplitParams.outValue, outTileTmp), l2SplitParams.outBase);
            uint64_t innerL2SplitTmp = MathUtil::Align(MathUtil::CeilDivision(l2SplitParams.innerValue, innerTileTmp),
                l2SplitParams.innerBase);
            uint64_t totalSize = GetTotalSize(outL2SplitTmp, args_.kValue, innerL2SplitTmp, l2SplitParams.outDtypeSize,
                l2SplitParams.innerDtypeSize);
            if (totalSize <= args_.l2Ratio * 100 * MB_SIZE) { // 100M为实验数据，确保不会出现L2数据置换
                if (IsTailSmall(l2SplitParams, outL2SplitTmp, innerL2SplitTmp, innerMaxConflict)) {
                    continue;
                }
                uint64_t outConflictTmp = MathUtil::CeilDivision(compileInfo_.aicNum, l2SplitParams.outTailCnt);
                uint64_t innerConflictTmp = MathUtil::CeilDivision(compileInfo_.aicNum, l2SplitParams.innerTailCnt);
                bool isUpdate = !enableCache || (outConflict >= outConflictTmp && innerConflict >= innerConflictTmp);
                if (isUpdate) {
                    enableCache = true;
                    outTile = outTileTmp;
                    innerTile = innerTileTmp;
                    outL2Split = outL2SplitTmp;
                    innerL2Split = innerL2SplitTmp;
                    outConflict = outConflictTmp;
                    innerConflict = innerConflictTmp;
                    OP_LOGI(args_.opName,
                        "Update Params! OutTile: %lu, InnerTile: %lu, OutL2Split: %lu, InnerL2Split: %lu, "
                        "outConflict: %lu, innerConflict: %lu",
                        outTile, innerTile, outL2Split, innerL2Split, outConflict, innerConflict);
                }
            }
        }
    }
    return enableCache;
}

uint64_t MatmulV3BaseTiling::GetTotalSize(uint64_t m, uint64_t k, uint64_t n, uint64_t aDtype, uint64_t bDtype) const
{
    uint64_t sizeA = m * k * aDtype;
    uint64_t sizeB = k * n * bDtype;
    uint64_t sizeC = m * n * GetSizeByDataType(args_.cType);
    return sizeA + sizeB + sizeC;
}

bool MatmulV3BaseTiling::DoBL1FullloadWithFixpipeTiling() {
    if (!NeedSolveFixBound()) {
        return false;
    }
    runInfo_.baseN = ops::CeilAlign(args_.nValue, N_ALIGNED);
    // baseM / 2(aic: aiv = 1:2) * alignN * dtypesize < ubsize / 2(pingpong buffer)
    uint64_t baseMMax = compileInfo_.ubSize / BIAS_TABLE_NUM / cDtypeSize_;
    // baseM align to 128
    runInfo_.baseM = ops::FloorAlign(std::min(ops::FloorDiv(compileInfo_.l0CSize, runInfo_.baseN * LOC_DATA_SIZE),
                                              baseMMax), SMALL_SHAPE_LOWER_THRES);
    uint64_t baseKa = compileInfo_.l0ASize / DB_SIZE / aDtypeSize_ / runInfo_.baseM;
    uint64_t baseKb = compileInfo_.l0BSize / DB_SIZE / bDtypeSize_ / runInfo_.baseN;
    runInfo_.baseK = ops::FloorAlign(std::min(baseKa, baseKb), N_ALIGNED);
    // B full load && Bl1 stay in L1
    runInfo_.depthB1 = ops::CeilDiv(args_.kValue, runInfo_.baseK);
    runInfo_.stepKb = runInfo_.depthB1;
    runInfo_.depthA1 = ops::FloorDiv((compileInfo_.l1Size / bDtypeSize_ -
        runInfo_.depthB1 * runInfo_.baseN * runInfo_.baseK), (runInfo_.baseM * runInfo_.baseK));
    runInfo_.depthA1 = std::min(runInfo_.depthA1, runInfo_.depthB1);
    runInfo_.stepKa = runInfo_.depthA1;
    // K较大, 设置stepK为4, depthA1为8, 该tiling较优
    if (runInfo_.depthA1 >= 8) {
        runInfo_.stepKa = 4; // K较大, 设置stepK为4, 该tiling较优
        runInfo_.depthA1 = 8; // K较大, 设置depthA1为8, 该tiling较优
    }
    runInfo_.singleCoreM = runInfo_.baseM;
    runInfo_.singleCoreN = runInfo_.baseN;
    OP_TILING_CHECK(runInfo_.depthA1 == 0,
            CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "Tiling calculate failed"), return false);
    tilingEnable_.tilingEnableFixOpti = TilingEnableFixOpti::BASE_ENABLE_ALIGNOUT;
    tilingEnable_.tilingEnableSplitCore = TilingEnableSplitCore::BASE;
    tilingEnable_.tilingEnableFullLoad = TilingEnableFullLoad::BL1_FULL_LOAD;
    args_.nd2nzB = false;
    // VEC_NZ2ND当前仅支持float类型下非nd2nzA场景
    if (!args_.nd2nzA && args_.aType == ge::DT_FLOAT) {
        // k轴8对齐场景下收益更高 当前仅开放N轴小于192场景
        if (args_.kValue % 8 == 0 && args_.nValue <= 192) { 
            tilingEnable_.tilingEnableFixOpti = TilingEnableFixOpti::VEC_NZ2ND_UNALIGNOUT;
        }
    }
    OP_LOGI(args_.opName, "Hit NeedSolveFixBound.");
    return true;
}

bool MatmulV3BaseTiling::DoAL1FullLoadTiling()
{
    // only support fp32, and resreict transpose attrs as network cases
    if (!compileInfo_.supportL0c2out || args_.aType != ge::DT_FLOAT || args_.isATrans || !args_.isBTrans) {
        return false;
    }
    bool kAligned = args_.kValue % (BASIC_ALIGN_512 / aDtypeSize_) == 0;
    bool isValidMNK = (args_.mValue <= BASIC_ALIGN_16 && args_.nValue > BASIC_ALIGN_16 && args_.nValue <= BASIC_ALIGN_16
        * compileInfo_.aicNum && args_.kValue >= 4096 && kAligned); // 4096 is k threshold for al1_full_load
    uint64_t mAlignedValue = ops::CeilAlign(args_.mValue, BASIC_ALIGN_16);
    uint64_t kAlignedValue = ops::CeilAlign(args_.kValue, BLOCK_BYTE_SIZE / aDtypeSize_);
    uint64_t al1Size = mAlignedValue * kAlignedValue * aDtypeSize_;
    uint64_t bl1Size = BASIC_ALIGN_16 * BASIC_ALIGN_256 * bDtypeSize_ * DB_SIZE;
    uint64_t biasSize = args_.hasBias ? BASIC_ALIGN_16 * GetSizeByDataType(args_.biasType) : 0;
    if (!isValidMNK || al1Size > (compileInfo_.l1Size - bl1Size - biasSize)) {
      return false;
    }
    runInfo_.baseM = BASIC_ALIGN_16;
    runInfo_.baseN = BASIC_ALIGN_16;
    runInfo_.baseK = BASIC_ALIGN_256;
    runInfo_.stepN = 1;
    runInfo_.stepM = 1;
    runInfo_.stepKa = MathUtil::CeilDivision(args_.kValue, runInfo_.baseK);
    runInfo_.stepKb = 1;
    runInfo_.depthA1 = runInfo_.stepKa;
    runInfo_.depthB1 = DB_SIZE;
    runInfo_.singleCoreM = args_.mValue;
    runInfo_.singleCoreN = runInfo_.baseN;
    tilingEnable_.tilingEnableFullLoad = TilingEnableFullLoad::AL1_FULL_LOAD;
    tilingEnable_.tilingEnableSplitCore = TilingEnableSplitCore::BASE;
    runInfo_.dbL0c = runInfo_.baseM * runInfo_.baseN * GetSizeByDataType(ge::DT_FLOAT) * DB_SIZE <=
        compileInfo_.l0CSize ? DB_SIZE : 1;
    runInfo_.l2Info.mTileBlock = 1;
    runInfo_.l2Info.nTileBlock = MathUtil::CeilDivision(args_.nValue, runInfo_.singleCoreN);
    runInfo_.l2Info.calOrder = 1;
    return true;
}

bool MatmulV3BaseTiling::DoBL1FullLoadTiling()
{
    uint64_t c0 = BLOCK_BYTE_SIZE / aDtypeSize_;
    uint64_t innerSizeA = args_.isATrans ? args_.mValue : args_.kValue;
    uint64_t outerSizeA = args_.isATrans ? args_.kValue : args_.mValue;
    // Update cases which can used vnchw_conv. 72368 is an experiment threshold value
    bool nd2nzAUsingVnchwConv =
        (args_.aType == ge::DT_FLOAT && outerSizeA >= VNCHW_UP_THRES && innerSizeA <= c0 && innerSizeA > 1);
    bool supportNd2NzOnTheWayB = std::find(SUPPORT_ND2NZ_GM2L0.begin(), SUPPORT_ND2NZ_GM2L0.end(), args_.nValue) !=
                                           SUPPORT_ND2NZ_GM2L0.end() &&
                                 (!args_.isBTrans || std::find(SUPPORT_ND2NZ_GM2L0.begin(), SUPPORT_ND2NZ_GM2L0.end(),
                                    args_.kValue * aDtypeSize_) != SUPPORT_ND2NZ_GM2L0.end());
    // mValue should be 16 times more than max of k/nValue, and kValue should be no more than 256
    bool validMK = args_.mValue > 16 * std::max(args_.kValue, args_.nValue) && args_.kValue <= 256;
    uint64_t biasSize = args_.hasBias ? runInfo_.baseN * GetSizeByDataType(ge::DT_FLOAT) : 0; // 默认最高精度保证BF16
    bool bl1SizeValid = (compileInfo_.l1Size / NUM_HALF - biasSize) > args_.kValue * args_.nValue * bDtypeSize_;
    bool alignedBl1FullLoad = (supportNd2NzOnTheWayB && !args_.nd2nzB);
    if (!validMK || (!alignedBl1FullLoad && !(nd2nzAUsingVnchwConv && bl1SizeValid))) {
      return false;
    }
    tilingEnable_.tilingEnableFullLoad = TilingEnableFullLoad::BL1_FULL_LOAD;
    tilingEnable_.tilingEnableSplitCore = TilingEnableSplitCore::BASE;
    // fine tune tiling
    runInfo_.stepM = 1;
    runInfo_.baseN = std::min(args_.nValue, runInfo_.baseN);
    // BaseN need to do Block alignment
    runInfo_.baseN = args_.isBTrans ? ops::CeilAlign(runInfo_.baseN, N_ALIGNED) : ops::CeilAlign(runInfo_.baseN, c0);
    runInfo_.stepN = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
    runInfo_.stepKb = MathUtil::CeilDivision(args_.kValue, runInfo_.baseK);
    runInfo_.stepKa = runInfo_.stepKb;
    runInfo_.depthA1 = DB_SIZE * runInfo_.stepKa;
    runInfo_.depthB1 = runInfo_.stepN * runInfo_.stepKb;
    uint64_t dtypeSize = GetSizeByDataType(args_.aType);
    uint64_t loadSize = static_cast<uint64_t>(runInfo_.baseK) *
        (runInfo_.depthA1 * runInfo_.baseM + runInfo_.depthB1 * runInfo_.baseN) * dtypeSize;
    loadSize += args_.hasBias ? runInfo_.baseN * dtypeSize : 0;
    // Check L1 load size
    while (loadSize > compileInfo_.l1Size) {
        loadSize -= runInfo_.depthA1 * runInfo_.baseM * runInfo_.baseK * dtypeSize;
        runInfo_.baseM = runInfo_.baseM / NUM_HALF;
        loadSize += runInfo_.depthA1 * runInfo_.baseM * runInfo_.baseK * dtypeSize;
    }
    runInfo_.singleCoreM = DB_SIZE * runInfo_.baseM;
    runInfo_.singleCoreN = args_.nValue;
    dtypeSize = GetSizeByDataType(ge::DT_FLOAT);
    runInfo_.dbL0c = runInfo_.baseM * runInfo_.baseN * dtypeSize * DB_SIZE <= compileInfo_.l0CSize ? DB_SIZE : 1;
    runInfo_.l2Info.mTile = 1;
    runInfo_.l2Info.nTile = 1;
    runInfo_.l2Info.mTileBlock = MathUtil::CeilDivision(args_.mValue, runInfo_.singleCoreM);
    runInfo_.l2Info.nTileBlock = 1;
    runInfo_.l2Info.calOrder = 1;
    return true;
}

bool MatmulV3BaseTiling::NeedSolveFixBound()
{
    if (compileInfo_.supportL12BtBf16 || !compileInfo_.supportL0c2out) {
        return false;
    }
    if (args_.nValue >= BASIC_ALIGN_256) {
        OP_LOGD(args_.opName, "N is exceeded to 256 to deal with aligning.");
        return false;
    }
    // K越小，越容易fixpipe_bound， 限制K的大小为256
    bool fixpipeBound = (args_.kValue <= BASIC_ALIGN_256 && (args_.nValue % (BASIC_ALIGN_256 / cDtypeSize_) != 0) &&
                        ((BASIC_ALIGN_256 / cDtypeSize_) % args_.nValue != 0));
    if (!fixpipeBound) {
        OP_LOGD(args_.opName, "With calculating, this shape may not be fixpipe bound.");
        return false;
    }
    if (args_.mValue < compileInfo_.aicNum * MIN_TAIL) {
        OP_LOGD(args_.opName, "M value is too small since BL1 fullload temp needs long sequence to get better perf.");
        return false;
    }
    // avoid scalar bound when Bmatrix is too small
    uint64_t c0 = BLOCK_BYTE_SIZE / aDtypeSize_;
    bool scalarBound = (args_.nValue < c0) && (args_.kValue < c0);
    if (scalarBound) {
        OP_LOGD(args_.opName, "If B matrix is too small, it will be realized as scalar bound when N aligning to 512B.");
        return false;
    }
    bool notK256Align = args_.kValue < BASIC_ALIGN_256 / aDtypeSize_;
    bool notMte2Bound = (kA256Align_ || notK256Align) && !args_.isATrans;
    if (cDtypeSize_ == 2) { // 2 means fp16 or bf16
        if (!notMte2Bound) {
            return false;
        } else {
            args_.nd2nzA = false;
            args_.nd2nzB = false;
            return true;
        }
    }
    return true;
}

bool MatmulV3BaseTiling::DoL2CacheTiling()
{
    if (compileInfo_.supportL12BtBf16 || !compileInfo_.supportL0c2out) {
        tilingEnable_.tilingEnableSplitCore = TilingEnableSplitCore::BASE;
        OP_LOGD(args_.opName, "david not support l2cache kernel, enter base kernel.");
        return false;
    }
    uint64_t totalSize = GetTotalSize(args_.mValue, args_.kValue, args_.nValue, aDtypeSize_, bDtypeSize_);
    uint64_t mL2Split = args_.mValue;
    uint64_t nL2Split = args_.nValue;
    uint64_t mTile = 1;
    uint64_t nTile = 1;

    args_.l2Ratio = 1.0 * compileInfo_.l2Size / L2_SIZE_2;
    if (totalSize > args_.l2Ratio * 100 * MB_SIZE) { // 100M为实验数据，确保不会出现L2数据置换
        enableCache_ = (runInfo_.baseN >= runInfo_.baseM) ?
                      CalcTile(nTile, mTile, nL2Split, mL2Split, args_.isATrans) :
                      CalcTile(mTile, nTile, mL2Split, nL2Split, !args_.isBTrans);
    }
    uint64_t mTileBlock = MathUtil::CeilDivision(mL2Split, runInfo_.baseM);
    uint64_t nTileBlock = MathUtil::CeilDivision(nL2Split, runInfo_.baseN);
    mTile = MathUtil::CeilDivision(args_.mValue, mTileBlock * runInfo_.baseM);
    nTile = MathUtil::CeilDivision(args_.nValue, nTileBlock * runInfo_.baseN);

    if (tilingSelect_ == TilingCalcSelect::ALL && !enableCache_ && compileInfo_.supportL0c2out) {
        constexpr uint64_t mTileBlockBase = 4UL;
        mTileBlock = std::min(mTileBlockBase, MathUtil::CeilDivision(args_.mValue, runInfo_.baseM));
        nTileBlock = compileInfo_.aicNum / mTileBlockBase;
        nTileBlock = std::min(nTileBlock, MathUtil::CeilDivision(args_.nValue, runInfo_.baseN));
        mTile = MathUtil::CeilDivision(args_.mValue, mTileBlock * runInfo_.baseM);
        nTile = MathUtil::CeilDivision(args_.nValue, nTileBlock * runInfo_.baseN);
    }

    runInfo_.l2Info.mTile = mTile;
    runInfo_.l2Info.nTile = nTile;
    runInfo_.l2Info.mTileBlock = mTileBlock;
    runInfo_.l2Info.nTileBlock = nTileBlock;
    OP_LOGI(args_.opName, "Enter L2cache tile kernel.");
    return false; // 进去单核切K逻辑
}

bool MatmulV3BaseTiling::DoL2CacheTiling310P()
{
    if (compileInfo_.supportL12BtBf16 || compileInfo_.supportL0c2out) {
        return false;
    }
    // 310P use (4, 2) or (5, 2)
    uint64_t mTileBlock = compileInfo_.aicNum / 2;
    mTileBlock = std::min(mTileBlock, MathUtil::CeilDivision(args_.mValue, runInfo_.baseM));
    uint64_t nTileBlock = std::min(2UL, MathUtil::CeilDivision(args_.nValue, runInfo_.baseN));

    uint64_t mTile = MathUtil::CeilDivision(args_.mValue, mTileBlock * runInfo_.baseM);
    uint64_t nTile = MathUtil::CeilDivision(args_.nValue, nTileBlock * runInfo_.baseN);
    runInfo_.l2Info.mTile = mTile;
    runInfo_.l2Info.nTile = nTile;
    runInfo_.l2Info.mTileBlock = mTileBlock;
    runInfo_.l2Info.nTileBlock = nTileBlock;
    OP_LOGI(args_.opName, "Enter L2cache tile kernel.");
    return true;
}

void MatmulV3BaseTiling::CalTileFactor(uint64_t &nTile)
{
    // set the factor of 20 core
    vector<uint64_t> factor = { 1, 2, 4, 5, 10, 20 };
    // set the factor of 24 core
    if (compileInfo_.aicNum == 24) {
        factor = { 1, 2, 3, 4, 6, 8, 12, 24 };
    }
    // set the factor of 32 core
    if (compileInfo_.aicNum == 32) {
        factor = { 1, 2, 4, 8, 16, 32 };
    }
    for (uint64_t i = 0; i < factor.size(); ++i) {
        if (nTile <= factor[i]) {
            nTile = factor[i];
            break;
        }
    }
    if (nTile > compileInfo_.aicNum) {
        uint64_t tail = nTile % compileInfo_.aicNum;
        // 2 表明一半的核要计算尾块，拖尾比较严重
        if (tail > compileInfo_.aicNum / 2) {
            nTile = (nTile + compileInfo_.aicNum - 1) / compileInfo_.aicNum;
        } else {
            nTile = nTile / compileInfo_.aicNum;
        }
        nTile = nTile * compileInfo_.aicNum;
    }
    return;
}

void MatmulV3BaseTiling::SetBasicBlockOfNK33()
{
    // 128是3*3算法的基本块
    runInfo_.baseM = BASIC_BLOCK_SIZE_128;
    runInfo_.baseN = BASIC_BLOCK_SIZE_128;
    // 256 means base size is 256Byte
    runInfo_.baseK = BASIC_BLOCK_K_256_BYTE / aDtypeSize_;
    runInfo_.usedCoreNum = compileInfo_.aicNum;
    runInfo_.depthA1 = 6; // 6 = baseM * stepKa * DB_SIZE  1 * 3 * 2
    runInfo_.depthB1 = 9; // 3*3算法, 9 = baseN * stepKb 3 * 3
    runInfo_.stepM = 1; // 3*3算法, stepM需要设置为1
    runInfo_.stepN = 3; // 3*3算法, stepN需要设置为3

    runInfo_.stepKa = runInfo_.depthA1 / runInfo_.stepM / DB_SIZE;
    runInfo_.stepKb = runInfo_.depthB1 / runInfo_.stepN;
    runInfo_.iterateOrder = ITER_COL_FIRST;
}

void MatmulV3BaseTiling::SetBasicBlockOfMK33(MatmulV3RunInfo &runInfo)
{
    // 128是3*3算法的基本块
    runInfo.baseM = BASIC_BLOCK_SIZE_128;
    runInfo.baseN = BASIC_BLOCK_SIZE_128;
    // 256 means base size is 256Byte
    runInfo.baseK = BASIC_BLOCK_K_256_BYTE / aDtypeSize_;
    runInfo.usedCoreNum = compileInfo_.aicNum;
    runInfo.depthA1 = 9; // 3*3算法, 9 = baseM * stepKa
    runInfo.depthB1 = 6; // 6 = baseN * stepKb * DB_SIZE
    runInfo.stepM = 3;   // 3*3算法, stepM需要设置为3
    runInfo.stepN = 1;   // 3*3算法, stepN需要设置为1

    runInfo.stepKa = runInfo.depthA1 / runInfo.stepM;
    runInfo.stepKb = runInfo.depthB1 / runInfo.stepN / DB_SIZE;
    runInfo.iterateOrder = ITER_ROW_FIRST;
}

void MatmulV3BaseTiling::SetBasicBlockOf24(MatmulV3RunInfo &runInfo, uint64_t &mTile, uint64_t &nTile) const
{
    // 2*4算法
    if (mTile * nTile < compileInfo_.aicNum) {
        runInfo.depthA1 = 8; // 2*4算法, 8 = baseM * stepKa
        runInfo.depthB1 = 8; // 2*4算法, 8 = baseN * stepKb * DB_SIZE
        runInfo.stepM = 2;   // 2*4算法, stepM = 2
        runInfo.stepN = 1;
        runInfo.stepKa = runInfo.depthA1 / runInfo.stepM;
        runInfo.stepKb = runInfo.depthB1 / DB_SIZE;
    }
    // 384 = 3 * 128, M和N较小的场景
    if (args_.mValue < 384 || args_.nValue < 384) {
        runInfo.depthA1 = 8; // 2*4算法, 8 = baseM * stepKa * DB_SIZE
        runInfo.depthB1 = 8; // 2*4算法, 8 = baseN * stepKb
        runInfo.stepM = 1;
        runInfo.stepKa = 4; // M较小，放大k为4
        runInfo.stepN = 1;
        runInfo.stepKb = 4; // N较小，放大k为4
    }
}

void MatmulV3BaseTiling::DoIncreTiling()
{
    if (!compileInfo_.supportL0c2out) {
        OP_LOGI(args_.opName, "Ascend310P currently doesn't support IncreShape.");
        return;
    }
    if (args_.mValue > 128) { // 128: 增量场景要求M≤128
        OP_LOGI(args_.opName, "M > 128, doesn't belong to IncreShape.");
        return;
    }
    if (IsMixNd2nz()) { // 对应要求 N和K < 65536，K 256B对齐
        OP_LOGI(args_.opName, "IncreShape shouldn't be MixNd2nz, please check shape size.");
        return;
    }
    if (((args_.aType != ge::DT_FLOAT16) && (args_.aType != ge::DT_BF16)) ||
        ((args_.bType != ge::DT_FLOAT16) && (args_.bType != ge::DT_BF16)) ||
        ((args_.cType != ge::DT_FLOAT16) && (args_.cType != ge::DT_BF16))) {
        OP_LOGI(args_.opName, "Data type should be fp16 or bf16 in IncreShape, please check.");
        return;
    }
    if ((args_.aFormat != ge::FORMAT_ND) || (args_.bFormat != ge::FORMAT_ND) || (args_.outFormat != ge::FORMAT_ND)) {
        OP_LOGI(args_.opName, "Data format non-ND, not support in IncreShape.");
        return;
    }
    if (args_.isATrans || !args_.isBTrans) {
        OP_LOGI(args_.opName, "A is Trans or B not Trans, not support in IncreShape.");
        return;
    }
    if (args_.hasBias) {
        OP_LOGI(args_.opName, "Bias not support in IncreShape.");
        return;
    }
    tilingEnable_.tilingEnableSplitCore = TilingEnableSplitCore::BASE;
    tilingEnable_.tilingEnableFullLoad = TilingEnableFullLoad::BASE;
    tilingEnable_.tilingEnableFixOpti = TilingEnableFixOpti::BASE;
    GetV2Tiling();
    OP_LOGD(args_.opName,
        "Incre convert tbe-tiling: CoreNum(%lu) singleMNK(%lu %lu %lu) baseMNK(%lu %lu %lu) depthAB(%lu %lu) "
        "stepMNKaKb(%lu %lu %lu %lu) dbL0c(%lu)",
        runInfo_.usedCoreNum, runInfo_.singleCoreM, runInfo_.singleCoreN, runInfo_.singleCoreK, runInfo_.baseM,
        runInfo_.baseN, runInfo_.baseK, runInfo_.depthA1, runInfo_.depthB1, runInfo_.stepM, runInfo_.stepN,
        runInfo_.stepKa, runInfo_.stepKb, runInfo_.dbL0c);
}

bool MatmulV3BaseTiling::IsMixNd2nz() // check different platform
{
    bool nd2nz = false;
    if (compileInfo_.supportL12BtBf16) {
        return nd2nz; // current not support mix kernel
    }
    if (compileInfo_.supportL0c2out) {
        nd2nz = args_.nd2nzA || args_.nd2nzB;
    }
    return nd2nz;
}

bool MatmulV3BaseTiling::IsSupportSingleCoreSplitK() const
{
    // n非对齐为fixpipe bound, 走单核切K，由于存在串行的前vector处理，非对齐场景可能性能更差，维持原4M限制不变
    if (args_.isHf32 && !n256Align_ && args_.mValue * args_.nValue < DETER_THRES_OUT_SIZE * MB_SIZE) {
        return false;
    }
    if (args_.kValue >= SPLIT_K_THRES) {
        OP_LOGD(args_.opName, "K >= SPLIT_K_THRES, enable SingleCoreSplitK.");
        return true;
    }
    constexpr uint64_t splitKThres = 1024; // 1024: 切K阈值
    constexpr uint64_t splitBaseThres = BASIC_BLOCK_SIZE_128 * 3; // 3*128: 33算法 step=3
    constexpr uint64_t splitKCubeThres = 5; // 5: cube dound阈值
    bool isMKNLargeEnough = ((args_.mValue * args_.kValue >= splitKCubeThres * splitBaseThres * splitBaseThres) &&
        (args_.nValue >= splitKThres) &&
        (args_.mValue >= splitBaseThres) &&
        (args_.kValue >= splitBaseThres) &&
        (args_.mValue * args_.nValue >= splitKThres * splitBaseThres * compileInfo_.aicNum));
    if (!enableCache_ && isMKNLargeEnough) {
        OP_LOGD(args_.opName, "L2cache tile fail, m/k/n is enough to splitk kernel.");
        return true;
    }
    constexpr uint64_t mataThread = 8192UL;
    constexpr uint64_t tlbThread = 6144UL;
    // (k, m)(k, n) K在外轴内轴很大MTE2会有tlb问题, 内轴%8192==0,存在一定的mata问题
    bool tlbAndMataFlag = args_.isATrans && !args_.isBTrans && n256Align_ && args_.kValue >= 11000UL &&
                          ((args_.mValue % mataThread == 0 && args_.nValue >= tlbThread) ||
                           (args_.nValue % mataThread == 0 && args_.mValue >= tlbThread));
    if (tlbAndMataFlag) {
      return true;
    }

    // (m, k)(n, k),k为16384的倍数，则mata冲突特别严重，走单核切K，减少MTE2搬运量
    // 限制为fp16、bf16、B2场景
    constexpr uint64_t kThread = 16384;
    constexpr uint64_t mnMinThread = 1280;
    constexpr uint64_t mnMaxThread = 8192;
    bool mataFlag = !args_.isATrans && args_.isBTrans && args_.kValue % kThread == 0 &&
                    (args_.mValue >= mnMinThread && args_.mValue <= mnMaxThread) &&
                    (args_.nValue >= mnMinThread && args_.nValue <= mnMaxThread) &&
                    args_.aFormat == ge::FORMAT_ND && args_.bFormat == ge::FORMAT_ND &&
                    (args_.aType == ge::DT_FLOAT16 || args_.aType == ge::DT_BF16) &&
                    compileInfo_.aicNum == MIX_BLOCK_NUM;
    if (mataFlag) {
      OP_LOGD(args_.opName, "The shape is meeting the criteria for mata conflicts.");
      return true;
    }
    return false;
}

bool MatmulV3BaseTiling::CheckSingleTilingOk(MatmulV3RunInfo &tmpRunInfo)
{
    // singlen过小场景, A的搬运串行占比过高, N小于512， 一般mac利用率较低
    if (tmpRunInfo.singleCoreN < 512UL) {
        OP_LOGD(args_.opName, "singleN too small.");
        return false;
    }
    // 单核切K负载不均衡
    float32_t avgRatio = (static_cast<float32_t>(args_.mValue * args_.nValue) /
        static_cast<float32_t>(tmpRunInfo.singleCoreN * tmpRunInfo.singleCoreM * compileInfo_.aicNum));
    if (avgRatio < 0.7f) { // 0.7是经验值
        OP_LOGD(args_.opName, "singleK avg_ratio small than 0.7.");
        return false;
    }
    // singleCoreN 小于等于640 mac利用率小于0.6，多核切K场景下N在896到2048之间mac利用率一般大于0.6
    if (n256Align_ && args_.nValue >= 896UL && args_.nValue <= 2048UL &&
        (tmpRunInfo.singleCoreN <= 640UL || avgRatio < 0.85f)) {
            OP_LOGD(args_.opName, "multiK may better than singleK");
        return false;
    }
    constexpr uint64_t SINGLE_CORE_M_24 = 256UL;
    // singleM切分在256到384之间，使用33算法，但若时fp32的输出非对齐场景，singleK小容易fixpipe_bound
    if (tmpRunInfo.singleCoreM > SINGLE_CORE_M_24 && tmpRunInfo.singleCoreM <= MULTI_CORE_SINGLE_K &&
        !(args_.aType == ge::DT_FLOAT && !n256Align_)) {
        SetBasicBlockOfMK33(tmpRunInfo);
    }
    tmpRunInfo.singleCoreK = tmpRunInfo.stepKa * tmpRunInfo.baseK;
    tmpRunInfo.dbL0c = DB_SIZE;
    OP_TILING_CHECK(!compileInfo_.supportL0c2out || compileInfo_.supportL12BtBf16 || !IsSupportSingleCoreSplitK(),
        OP_LOGI(args_.opName, "MatMulV3 tiling not support SingleCoreSplitK."), return false);
    tilingEnable_.tilingEnableSplitCore = TilingEnableSplitCore::SINGLE_CORE_SPLIT_K;
    tilingEnable_.tilingEnableFullLoad = TilingEnableFullLoad::BASE;
    tilingEnable_.tilingEnableFixOpti = TilingEnableFixOpti::BASE;
    runInfo_ = tmpRunInfo;
    runInfo_.needUpdate = true;
    OP_LOGI(args_.opName, "MatMulV3 tiling enable state is SingleCoreSplitK");
    return true;
}

bool MatmulV3BaseTiling::DoSingleCoreSplitKTiling()
{
    if (!compileInfo_.supportL0c2out || compileInfo_.supportL12BtBf16 || !IsSupportSingleCoreSplitK()) {
        OP_LOGI(args_.opName, "MatMulV3 tiling not support SingleCoreSplitK.");
        return false;
    }
    MatmulV3RunInfo tmpRunInfo = runInfo_;
    uint64_t nL2TileLength = l2TileLength_; // L2缓存为192MB,切K时N为3072保证计算恰好被L2缓存容下
    SetBasicBlockOfMK33(tmpRunInfo);
    uint64_t mTile = MathUtil::CeilDivision(args_.mValue, tmpRunInfo.stepM * tmpRunInfo.baseM);
    uint64_t nTile = MathUtil::CeilDivision(args_.nValue, nL2TileLength);
    if (!args_.hasBias) {
        SetBasicBlockOf24(tmpRunInfo, mTile, nTile);
    }
    uint64_t mAlignLength = (args_.isATrans ? ALIGN_INNER : ALIGN_OUTER) / (aDtypeSize_);
    uint64_t nAlignLength = ALIGN_INNER / aDtypeSize_;
    mTile = MathUtil::CeilDivision(args_.mValue, tmpRunInfo.stepM * tmpRunInfo.baseM);
    CalTileFactor(nTile); // nTile向上靠近aic_num的因子， 如ntile=5，aic_num=24，调整ntile为6
    if (mTile * nTile >= compileInfo_.aicNum) {
        mTile = std::max(compileInfo_.aicNum / nTile, 1UL);
        tmpRunInfo.singleCoreM = ops::CeilAlign(MathUtil::CeilDivision(args_.mValue, mTile), mAlignLength);
        tmpRunInfo.singleCoreM = std::min(tmpRunInfo.singleCoreM, args_.mValue);
        tmpRunInfo.singleCoreN = ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, nTile), nAlignLength);
        tmpRunInfo.usedCoreNum = std::min(MathUtil::CeilDivision(args_.mValue, tmpRunInfo.singleCoreM) *
                                          MathUtil::CeilDivision(args_.nValue, tmpRunInfo.singleCoreN),
                                          compileInfo_.aicNum);
        if (tmpRunInfo.usedCoreNum == compileInfo_.aicNum) {
            return CheckSingleTilingOk(tmpRunInfo);
        }
    }
    nTile = std::max(compileInfo_.aicNum / mTile, 1UL);
    uint64_t totalCnt = 0UL;
    uint64_t singleCoreN = args_.nValue;
    constexpr uint64_t singleCoreNThreshold = 1024UL;
    // 由于singleM, singleN做了align, 实际核数可能不是24，ntile调整和核数一致, 满核, N已经很小，停止调整mtile, ntile
    while (nTile <= compileInfo_.aicNum && totalCnt < compileInfo_.aicNum && singleCoreN >= singleCoreNThreshold) {
        CalTileFactor(nTile);
        mTile = compileInfo_.aicNum / nTile;
        singleCoreN = ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, nTile), nAlignLength);
        uint64_t singleCoreM = ops::CeilAlign(MathUtil::CeilDivision(args_.mValue, mTile), mAlignLength);
        uint64_t mCnt = MathUtil::CeilDivision(args_.mValue, singleCoreM);
        uint64_t nCnt = MathUtil::CeilDivision(args_.nValue, singleCoreN);
        if (mCnt * nCnt > totalCnt) {
            totalCnt = mCnt * nCnt;
            tmpRunInfo.usedCoreNum = totalCnt;
            tmpRunInfo.singleCoreM = std::min(singleCoreM, args_.mValue);
            tmpRunInfo.singleCoreN = std::min(singleCoreN, args_.nValue);
        }
        nTile++;
    }
    return CheckSingleTilingOk(tmpRunInfo);
}

bool MatmulV3BaseTiling::SupportMultiSplitK() const
{
    // 判断是否为支持多核切k的芯片
    if (!compileInfo_.supportL12BtBf16 && !compileInfo_.supportL0c2out) {
        return false;
    }
    bool kIsEnoughMultiCore = args_.kValue >= compileInfo_.aicNum * MULTI_CORE_SINGLE_K;
    uint64_t mCnt = MathUtil::CeilDivision(args_.mValue, BASIC_BLOCK_SIZE_128);
    uint64_t nCnt = MathUtil::CeilDivision(args_.nValue, BASIC_BLOCK_SIZE_128);
    bool mNIsNotEnoughCore = (mCnt * nCnt < static_cast<int64_t>(compileInfo_.aicNum) / NUMBER_TWO);
    // 判断是否时M/N不能开满多核场景，M/N轴在外轴的场景切m/n不影响MTE2搬运效率，M/N可以切小保证多核能开启，属于cube_bound场景
    bool splitKScene = kIsEnoughMultiCore && mNIsNotEnoughCore && !(!args_.isATrans && args_.isBTrans);
    if (splitKScene) {
        return true;
    }
    if (args_.kValue >= MIN_SPLITK_K && args_.kValue <= MAX_SPLITK_K &&
        args_.mValue <= M_THRESHOLD && args_.nValue <= N_THRESHOLD && args_.isATrans && !args_.isBTrans) {
        OP_LOGD(args_.opName, "MultiCore splitK is supported");
        return true;
    }
    // 判断k轴是否大于27392，不大于跳出
    if (args_.kValue < SPLIT_K_THRES) {
        return false;
    }
    bool innerAlignA = args_.isATrans ? m256Align_ : kA256Align_;
    bool innerAlignB = args_.isBTrans ? kB256Align_ : n256Align_;
    bool mte2BoundBaseFp32 = (!innerAlignA || !innerAlignB) && args_.aType == ge::DT_FLOAT;
    // m, n, 均大于512，且N对齐，核数足够分
    if (mCnt >= 4UL && nCnt >= 4UL && !n256Align_ && (!mte2BoundBaseFp32) &&
        static_cast<float32_t>(mCnt * nCnt) / ops::CeilAlign(mCnt * nCnt, compileInfo_.aicNum) > 0.8f) {
        OP_LOGD(args_.opName, "MultiK will fixpipebound, use L2split");
        return false;
    }
    return true;
}

bool MatmulV3BaseTiling::IsNkOrder()
{
    // M > N && N方向需要对齐32B
    aDtypeSize_ = GetSizeByDataType(args_.aType);
    bDtypeSize_ = GetSizeByDataType(args_.bType);
    if (args_.mValue <= args_.nValue || !Is32BAlign(args_.nValue, bDtypeSize_)) {
        return false;
    }
    // M限制为>=128
    constexpr uint64_t mMinValue = 128;
    if (args_.mValue < mMinValue) {
        return false;
    }
    // 在N对齐256B并且M非对齐256B的场景下，由于M非对齐NK的性能优化有限，且N对齐256B时原方案搬运性能较好
    if (Is256BAlign(args_.nValue, bDtypeSize_) && !Is256BAlign(args_.mValue, aDtypeSize_)) {
        return false;
    }
    // M非对齐256B && N*dtype <= 256B && N*dtype %32B == 0时走回MK
    if (!Is256BAlign(args_.mValue, aDtypeSize_) && args_.nValue * bDtypeSize_ <= N_THRESHOLD && Is32BAlign(args_.nValue, bDtypeSize_)) {
        return false;
    }
    // N=16时，M超过2048则走回MK
    constexpr uint64_t nMinValue = 16;
    constexpr uint64_t mMaxValue = 2048;
    if (args_.mValue >= mMaxValue && args_.nValue == nMinValue) {
        return false;
    }
    return true;
}

bool MatmulV3BaseTiling::DoDeterministicMultiCoreSplitKTiling()
{
    if (compileInfo_.supportL12BtBf16 || !SupportMultiSplitK()) {
        return false;
    }
    tilingEnable_.tilingEnableSplitCore = TilingEnableSplitCore::DETERMINISTIC_SPLIT_K;
    tilingEnable_.tilingEnableFullLoad = TilingEnableFullLoad::BASE;
    tilingEnable_.tilingEnableFixOpti = TilingEnableFixOpti::BASE;
    OP_LOGI(args_.opName, "MatMulV3 tiling enable state is DeterministicMultiCoreSplitK.");

    //MK耗时：(MK + NK) / 1.6 + NK * (M / ML1 - 1) / 7.8
    //NK耗时：(MK + NK) / 1.6 + MK * (N / NL1 - 1) / 7.8
    // MK耗时和NK耗时的比较合并为M和N的比较
    uint64_t L2_SIZE_70_percent = compileInfo_.l2Size * 7 / 10;
    if (IsNkOrder()) {
        SetBasicBlockOfNK33();
        // 多核切K在k完成分核，m轴不分核, n轴做外层循环进行pingpong
        runInfo_.singleCoreM = args_.mValue;
        // singleCoreN * singleCoreK表示基本块大小，一次IterateALL完成singleCoreM * singleCoreN的计算
        runInfo_.singleCoreN = runInfo_.stepN * runInfo_.baseN;
        runInfo_.singleCoreK = runInfo_.stepKb * runInfo_.baseK;
        // update L2 cache split :(runInfo_.singleCoreM * runInfo_.singleCoreK * aDtypeSize_ + runInfo_.singleCoreK * runInfo_.singleCoreN * bDtypeSize_ + runInfo_.singleCoreM * runInfo_.singleCoreN * DATA_SIZE_FP32)
        // * compileInfo_.aicNum < compileInfo_.l2Size;
        uint64_t mL2Split = (L2_SIZE_70_percent / compileInfo_.aicNum - runInfo_.singleCoreK * std::min(runInfo_.singleCoreN, args_.nValue) * bDtypeSize_) /
                (runInfo_.singleCoreK * aDtypeSize_ + std::min(runInfo_.singleCoreN, args_.nValue) * DATA_SIZE_FP32);

        if (args_.mValue > mL2Split) {
            //需要切分L2
            mL2Split = ops::CeilAlign(mL2Split, BLOCK);
            uint64_t mCnt = ops::CeilDiv(args_.mValue, mL2Split);
            runInfo_.singleCoreM = ops::CeilAlign(ops::CeilDiv(args_.mValue, mCnt), BLOCK);
        }
    } else {
        SetBasicBlockOfMK33(runInfo_);
        // 多核切K在k完成分核，n轴不分核, m轴做外层循环进行pingpong
        runInfo_.singleCoreN = args_.nValue; //Is256BAlign(args_.nValue, DATA_SIZE_FP32) ? args_.nValue : ops::CeilAlign(args_.nValue * DATA_SIZE_FP32, ALIGN_BYTE) / DATA_SIZE_FP32;
        // singleCoreM * singleCoreK表示基本块大小，一次IterateALL完成singleCoreM * singleCoreN的计算
        runInfo_.singleCoreM = runInfo_.stepM * runInfo_.baseM;
        runInfo_.singleCoreK = runInfo_.stepKa * runInfo_.baseK;
        uint64_t nL2Split = (L2_SIZE_70_percent / compileInfo_.aicNum - runInfo_.singleCoreK * std::min(runInfo_.singleCoreM, args_.mValue) * aDtypeSize_) /
                            (runInfo_.singleCoreK * bDtypeSize_ + std::min(runInfo_.singleCoreM, args_.mValue) * DATA_SIZE_FP32);

        if (args_.nValue > nL2Split) {
            //需要切分L2
            nL2Split = ops::CeilAlign(nL2Split, BLOCK);
            uint64_t nCnt = ops::CeilDiv(args_.nValue, nL2Split);
            runInfo_.singleCoreN = ops::CeilAlign(ops::CeilDiv(args_.nValue, nCnt), BLOCK);
        }
    }
    uint64_t kCnt = MathUtil::CeilDivision(args_.kValue, runInfo_.singleCoreK);
    runInfo_.usedCoreNum = std::min(kCnt, compileInfo_.aicNum);
    runInfo_.dbL0c = DB_SIZE;

    // 内轴较大时，串行的mixnd2nz相较随路nd2nz场景存在劣化，内轴>512B暂不走mixnd2nz(仅多核切K模板)
    uint64_t innerSizeA = args_.isATrans ? args_.mValue : args_.kValue;
    uint64_t innerSizeB = args_.isBTrans ? args_.kValue : args_.nValue;

    // 内轴大于512B时随路方案获取更大的性能收益
    if ((innerSizeA >= ND2NZ_THRES / aDtypeSize_) && (innerSizeA <= ND2NZ_ON_THE_FLY_LIMIT)) {
        args_.nd2nzA = false;
    }
    if ((innerSizeB >= ND2NZ_THRES / bDtypeSize_) && (innerSizeB <= ND2NZ_ON_THE_FLY_LIMIT)) {
        args_.nd2nzB = false;
    }
    return true;
}

bool MatmulV3BaseTiling::CheckMMTilingDataIsVaild()
{
    return (CheckNumberIsVaild(runInfo_.usedCoreNum, args_.opName, "runInfo_.usedCoreNum") ||
        CheckNumberIsVaild2(runInfo_.singleCoreM, args_.opName, "runInfo_.singleCoreM") ||
        CheckNumberIsVaild2(runInfo_.singleCoreN, args_.opName, "runInfo_.singleCoreN") ||
        CheckNumberIsVaild2(runInfo_.singleCoreK, args_.opName, "runInfo_.singleCoreK") ||
        CheckNumberIsVaild2(runInfo_.baseM, args_.opName, "runInfo_.baseM") ||
        CheckNumberIsVaild2(runInfo_.baseN, args_.opName, "runInfo_.baseN") ||
        CheckNumberIsVaild2(runInfo_.baseK, args_.opName, "runInfo_.baseK") ||
        CheckNumberIsVaild(runInfo_.depthA1, args_.opName, "runInfo_.depthA1") ||
        CheckNumberIsVaild(runInfo_.depthB1, args_.opName, "runInfo_.depthB1") ||
        CheckNumberIsVaild(runInfo_.stepM, args_.opName, "runInfo_.baseK") ||
        CheckNumberIsVaild(runInfo_.stepN, args_.opName, "runInfo_.stepN") ||
        CheckNumberIsVaild(runInfo_.stepKa, args_.opName, "runInfo_.baseK") ||
        CheckNumberIsVaild(runInfo_.stepKb, args_.opName, "runInfo_.stepKb") ||
        CheckNumberIsVaild(runInfo_.iterateOrder, args_.opName, "runInfo_.iterateOrder") ||
        CheckNumberIsVaild(runInfo_.dbL0c, args_.opName, "runInfo_.dbL0c") ||
        CheckNumberIsVaild(runInfo_.l2Info.mTile, args_.opName, "runInfo_.l2Info.mTile") ||
        CheckNumberIsVaild(runInfo_.l2Info.nTile, args_.opName, "runInfo_.l2Info.nTile") ||
        CheckNumberIsVaild(runInfo_.l2Info.mTileBlock, args_.opName, "runInfo_.l2Info.mTileBlock") ||
        CheckNumberIsVaild(runInfo_.l2Info.nTileBlock, args_.opName, "runInfo_.l2Info.nTileBlock"));
}

ge::graphStatus MatmulV3BaseTiling::DoLibApiTiling()
{
    SetRunInfo();
    if (runInfo_.needUpdate) {
        if (CheckMMTilingDataIsVaild()) {
            return ge::FAILED;
        }
        tilingData_.matmulTiling.set_usedCoreNum(static_cast<uint32_t>(runInfo_.usedCoreNum));
        tilingData_.matmulTiling.set_singleCoreM(static_cast<uint32_t>(runInfo_.singleCoreM));
        tilingData_.matmulTiling.set_singleCoreN(static_cast<uint32_t>(runInfo_.singleCoreN));
        tilingData_.matmulTiling.set_singleCoreK(static_cast<uint32_t>(runInfo_.singleCoreK));
        tilingData_.matmulTiling.set_baseM(static_cast<uint32_t>(runInfo_.baseM));
        tilingData_.matmulTiling.set_baseN(static_cast<uint32_t>(runInfo_.baseN));
        tilingData_.matmulTiling.set_baseK(static_cast<uint32_t>(runInfo_.baseK));
        tilingData_.matmulTiling.set_depthA1(static_cast<uint32_t>(runInfo_.depthA1));
        tilingData_.matmulTiling.set_depthB1(static_cast<uint32_t>(runInfo_.depthB1));
        tilingData_.matmulTiling.set_stepM(static_cast<uint32_t>(runInfo_.stepM));
        tilingData_.matmulTiling.set_stepN(static_cast<uint32_t>(runInfo_.stepN));
        tilingData_.matmulTiling.set_stepKa(static_cast<uint32_t>(runInfo_.stepKa));
        tilingData_.matmulTiling.set_stepKb(static_cast<uint32_t>(runInfo_.stepKb));
        tilingData_.matmulTiling.set_iterateOrder(static_cast<uint32_t>(runInfo_.iterateOrder));
        tilingData_.matmulTiling.set_dbL0C(static_cast<uint32_t>(runInfo_.dbL0c));
        if (!compileInfo_.supportL0c2out) {
            tilingData_.matmulTiling.set_transLength(static_cast<uint32_t>(L0C_SIZE_256_KB / NUM_HALF));
            tilingData_.matmulTiling.set_shareUbSize(0);
        }
        tilingData_.tileL2cacheTiling.set_mTileCntL2(static_cast<uint32_t>(runInfo_.l2Info.mTile));
        tilingData_.tileL2cacheTiling.set_nTileCntL2(static_cast<uint32_t>(runInfo_.l2Info.nTile));
        tilingData_.tileL2cacheTiling.set_mTileBlock(static_cast<uint32_t>(runInfo_.l2Info.mTileBlock));
        tilingData_.tileL2cacheTiling.set_nTileBlock(static_cast<uint32_t>(runInfo_.l2Info.nTileBlock));
        tilingData_.tileL2cacheTiling.set_calOrder(static_cast<uint32_t>(runInfo_.l2Info.calOrder));
        tilingData_.l2cacheUseInfo.set_l2CacheFlag(l2CacheFlag_);
    }
    SetNd2NzInfo();
    DoTilingKey();
    L2Cache l2Cache(args_, tilingData_);
    l2Cache.SetL2CacheFlag(tilingEnable_, compileInfo_.l2Size, l2CacheFlag_);
    return ge::GRAPH_SUCCESS;
}

uint64_t MatmulV3BaseTiling::GetTilingKey() const
{
    return tilingKey_;
}

void MatmulV3BaseTiling::DoTilingKey()
{
    // 1: disable mix nd2nz 0: enable mix nd2nz
    uint64_t disableMixNd2nz = !IsMixNd2nz();
    tilingKey_ = GET_TILINGKEY(disableMixNd2nz, tilingEnable_.tilingEnableSplitCore,
                               tilingEnable_.tilingEnableFullLoad, 0, tilingEnable_.tilingEnableFixOpti); // tilingKey reverse:  01->10
    OP_LOGI(args_.opName, "DoTilingKey: %lu", tilingKey_);
}

ge::graphStatus MatmulV3BaseTiling::GetWorkspaceSize()
{
    uint64_t alignByte = 256 / aDtypeSize_;  // 256B 对齐shape
    workspaceSize_ = RPC_WORKSIZE * MB_SIZE; // 20MB reserve > 16MB for rpc
    if (tilingEnable_.tilingEnableSplitCore == TilingEnableSplitCore::SINGLE_CORE_SPLIT_K) {
        workspaceSize_ = args_.mValue * ops::CeilAlign(args_.nValue, alignByte) * DATA_SIZE_FP32 +
            RPC_WORKSIZE * MB_SIZE; // 20 means 20MB
    }
     OP_LOGI(args_.opName, "if tiling enable is deterministic splitk, workspace size is %lu", workspaceSize_);
    if (tilingEnable_.tilingEnableSplitCore == TilingEnableSplitCore::DETERMINISTIC_SPLIT_K) {
        uint64_t singleSize = static_cast<uint64_t>(tilingData_.matmulTiling.get_singleCoreN()) * static_cast<uint64_t>(tilingData_.matmulTiling.get_singleCoreM());
        workspaceSize_ =
            static_cast<uint64_t>(tilingData_.matmulTiling.get_usedCoreNum()) * singleSize * DB_SIZE * DATA_SIZE_FP32 +
            RPC_WORKSIZE * MB_SIZE;
    }
    if (tilingEnable_.tilingEnableFixOpti == TilingEnableFixOpti::BASE_ENABLE_ALIGNOUT) {
        workspaceSize_ += ops::CeilAlign(args_.nValue, CACHELINE / cDtypeSize_) * tilingData_.matmulTiling.get_baseM() *
           tilingData_.matmulTiling.get_usedCoreNum() * NUMBER_TWO * cDtypeSize_;
    }
    if (tilingEnable_.tilingEnableFixOpti == TilingEnableFixOpti::VEC_NZ2ND_UNALIGNOUT) {
        workspaceSize_ += ops::CeilAlign(args_.nValue, N_ALIGNED) * tilingData_.matmulTiling.get_baseM() *
           tilingData_.matmulTiling.get_usedCoreNum() * NUMBER_TWO * cDtypeSize_;
    }
    if (!compileInfo_.supportL0c2out) {
        return ge::GRAPH_SUCCESS;
    }
    uint64_t c0 = BLOCK_BYTE_SIZE / aDtypeSize_;
    uint64_t kALignC0 = ops::CeilAlign(args_.kValue, c0);
    uint64_t kALignN = ops::CeilAlign(args_.kValue, N_ALIGNED);
    // 非对齐场景需要预留一些workspace
    if (args_.nd2nzA) {
        if (args_.isATrans) {
            workspaceSize_ += ops::CeilAlign(args_.mValue, c0) * kALignN * aDtypeSize_;
        } else {
            workspaceSize_ += ops::CeilAlign(args_.mValue, N_ALIGNED) * kALignC0 * aDtypeSize_;
        }
    }
    if (args_.nd2nzB) {
        if (args_.isBTrans) {
            workspaceSize_ += ops::CeilAlign(args_.nValue, N_ALIGNED) * kALignC0 * bDtypeSize_;
        } else {
            workspaceSize_ += ops::CeilAlign(args_.nValue, c0) * kALignN * bDtypeSize_;
        }
    }
    OP_LOGI(args_.opName, "final workspace size is %lu", workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulV3BaseTiling::PostTiling()
{
    OP_TILING_CHECK(tilingData_.GetDataSize() % sizeof(uint64_t) != 0,
        OP_LOGE(args_.opName, "tiling data size[%zu] is not aligned to 8", tilingData_.GetDataSize()),
        return ge::GRAPH_FAILED);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetBlockDim(tilingData_.matmulTiling.get_usedCoreNum());
    context_->SetScheduleMode(1);
    size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
    OP_TILING_CHECK(workspaces == nullptr, CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "workspaces is null"),
        return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;

    return ge::GRAPH_SUCCESS;
}
}
}

