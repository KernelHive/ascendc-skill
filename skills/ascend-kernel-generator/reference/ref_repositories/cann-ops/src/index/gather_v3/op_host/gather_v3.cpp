/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include "gather_v3_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
namespace ops {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }
}  // namespace ops
namespace optiling {
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)
}  // namespace optiling

namespace optiling {

// input
constexpr int64_t X_INDEX = 0;
constexpr int64_t INDICES_INDEX = 1;
constexpr int64_t AXIS_INDEX = 2;

// output
constexpr int64_t Y_INDEX = 0;

// attr
constexpr int64_t BATCH_DIMS_ATTR_INDEX = 0;
constexpr int64_t NEG_IDX_SUPPORT_ATTR_INDEX = 1;
constexpr int64_t IMPL_MODE_ATTR_INDEX = 2;

constexpr int64_t FIX_PATTERN_MODEA = 2;
constexpr int64_t FIX_PATTERN_MODEB = 4;
constexpr int64_t VREDUCEV2_SIZE = 2;

constexpr int64_t ONE_IN_N_G_AXIS_LEN_THRE = 256;
constexpr int64_t ONE_IN_N_P_AXIS_LEN_THRE = 4;
constexpr int64_t ROBIN_PAGE_THRE = 5;
constexpr int64_t ROBIN_GYSIZE_THRE = 8;

constexpr int64_t UB_BUF_CNT = 2;
constexpr int64_t CORE_SPLIT_FACTOR = 2;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t FULL_BLOCK_SIZE = 256;
constexpr int64_t WORK_SPACE_SIZE = 1;
constexpr int64_t RSV_BLOCK = 8;
constexpr int64_t VGATHER_BUFF_EXPAND = 2;
constexpr int64_t SCALAR_MOVE_THRE = 2;

constexpr int64_t ALL_MOVE_TMPL_THRE = 512;
constexpr int64_t IO_RATIO_THRE = 20;
constexpr int64_t GYSIZE_CACHE_THRE = 4;

constexpr int64_t ALIGN_SIZE_FOR_2B = 256;
constexpr int64_t ALIGN_SIZE_FOR_1B = 512;

constexpr int64_t UB_OUT_RATIO = 8;
constexpr int64_t UB_IDX_RATIO = 1;

static const std::vector<ge::DataType> X_DTYPES = {ge::DT_INT8,    ge::DT_INT16,    ge::DT_INT32,  ge::DT_INT64,
                                                   ge::DT_UINT8,   ge::DT_UINT16,   ge::DT_UINT32, ge::DT_UINT64,
                                                   ge::DT_FLOAT16, ge::DT_FLOAT,    ge::DT_BF16,   ge::DT_BOOL};
static const std::string X_DTYPES_STR = "[int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32, bfloat16, bool]";
static const std::vector<ge::DataType> INT_DTYPES = {ge::DT_INT32, ge::DT_INT64};
static const std::string INT_DTYPES_STR = "[int32, int64]";

// 各轴序号
static constexpr int32_t AXIS_B = 0;
static constexpr int32_t AXIS_P = 1;
static constexpr int32_t AXIS_G = 2;
static constexpr int32_t AXIS_A = 3;
static constexpr int32_t AXIS_NUM = 4;

// 切分方式
static constexpr int32_t TILE_FULL = -2;
static constexpr int32_t TILE_NONE = -1;

// 各tensor序号
static constexpr int32_t TENSOR_X = 0;
static constexpr int32_t TENSOR_Y = 1;
static constexpr int32_t TENSOR_I = 2;
static constexpr int32_t TENSOR_NUM = 3;

static constexpr int64_t KEY_BASE_MOVE = 10000;
static constexpr int64_t KEY_BASE_CACHE = 20000;
static constexpr int64_t KEY_BASE_CACHE_UNALIGN = 30000;
static constexpr int64_t KEY_BASE_ROBIN = 40300;
static constexpr int64_t KEY_BASE_ROBIN_UNALIGN = 50300;
static constexpr int64_t KEY_BASE_ONE_IN_N = 60000;
static constexpr int64_t KEY_BASE_VGATHER = 70000;
static constexpr int64_t KEY_BASE_MOVE_UNALIGN = 80000;

static constexpr int64_t X_SPLIT_IN_B = 100;
static constexpr int64_t X_SPLIT_IN_P = 200;
static constexpr int64_t X_SPLIT_IN_G = 300;
static constexpr int64_t Y_SPLIT_IN_B = 10;
static constexpr int64_t Y_SPLIT_IN_P = 20;
static constexpr int64_t Y_SPLIT_IN_G = 30;

static constexpr int64_t KEY_MOVE_AG = 10430;
static constexpr int64_t KEY_MOVE_GG = 10330;
static constexpr int64_t KEY_MOVE_PB = 10200;
static constexpr int64_t KEY_MOVE_PG = 12330;
static constexpr int64_t KEY_MOVE_BB = 11100;

static constexpr int64_t KEY_CACHE_GG = KEY_BASE_CACHE + X_SPLIT_IN_G + Y_SPLIT_IN_G;
static constexpr int64_t KEY_CACHE_GP = KEY_BASE_CACHE + X_SPLIT_IN_G + Y_SPLIT_IN_P;
static constexpr int64_t KEY_CACHE_GB = KEY_BASE_CACHE + X_SPLIT_IN_G + Y_SPLIT_IN_B;
static constexpr int64_t KEY_CACHE_PG = KEY_BASE_CACHE + X_SPLIT_IN_P + Y_SPLIT_IN_G;
static constexpr int64_t KEY_CACHE_PP = KEY_BASE_CACHE + X_SPLIT_IN_P + Y_SPLIT_IN_P;
static constexpr int64_t KEY_CACHE_PB = KEY_BASE_CACHE + X_SPLIT_IN_P + Y_SPLIT_IN_B;
static constexpr int64_t KEY_CACHE_BG = KEY_BASE_CACHE + X_SPLIT_IN_B + Y_SPLIT_IN_G;
static constexpr int64_t KEY_CACHE_BG_SP1 = KEY_CACHE_BG + 1;
static constexpr int64_t KEY_CACHE_BG_SP2 = KEY_CACHE_BG + 2;
static constexpr int64_t KEY_CACHE_BP = KEY_BASE_CACHE + X_SPLIT_IN_B + Y_SPLIT_IN_P;
static constexpr int64_t KEY_CACHE_BB = KEY_BASE_CACHE + X_SPLIT_IN_B + Y_SPLIT_IN_B;

static constexpr int64_t KEY_CACHE_GG_U = KEY_BASE_CACHE_UNALIGN + X_SPLIT_IN_G + Y_SPLIT_IN_G;
static constexpr int64_t KEY_CACHE_GP_U = KEY_BASE_CACHE_UNALIGN + X_SPLIT_IN_G + Y_SPLIT_IN_P;
static constexpr int64_t KEY_CACHE_GB_U = KEY_BASE_CACHE_UNALIGN + X_SPLIT_IN_G + Y_SPLIT_IN_B;
static constexpr int64_t KEY_CACHE_PG_U = KEY_BASE_CACHE_UNALIGN + X_SPLIT_IN_P + Y_SPLIT_IN_G;
static constexpr int64_t KEY_CACHE_PP_U = KEY_BASE_CACHE_UNALIGN + X_SPLIT_IN_P + Y_SPLIT_IN_P;
static constexpr int64_t KEY_CACHE_PB_U = KEY_BASE_CACHE_UNALIGN + X_SPLIT_IN_P + Y_SPLIT_IN_B;
static constexpr int64_t KEY_CACHE_BG_U = KEY_BASE_CACHE_UNALIGN + X_SPLIT_IN_B + Y_SPLIT_IN_G;
static constexpr int64_t KEY_CACHE_BG_U_SP1 = KEY_CACHE_BG_U + 1;
static constexpr int64_t KEY_CACHE_BP_U = KEY_BASE_CACHE_UNALIGN + X_SPLIT_IN_B + Y_SPLIT_IN_P;
static constexpr int64_t KEY_CACHE_BB_U = KEY_BASE_CACHE_UNALIGN + X_SPLIT_IN_B + Y_SPLIT_IN_B;

static constexpr int64_t KEY_ROBIN_GG = KEY_BASE_ROBIN + Y_SPLIT_IN_G;
static constexpr int64_t KEY_ROBIN_GG_SP1 = KEY_ROBIN_GG + 1;
static constexpr int64_t KEY_ROBIN_GP = KEY_BASE_ROBIN + Y_SPLIT_IN_P;
static constexpr int64_t KEY_ROBIN_GB = KEY_BASE_ROBIN + Y_SPLIT_IN_B;
static constexpr int64_t KEY_ROBIN_GG_U = KEY_BASE_ROBIN_UNALIGN + Y_SPLIT_IN_G;
static constexpr int64_t KEY_ROBIN_GP_U = KEY_BASE_ROBIN_UNALIGN + Y_SPLIT_IN_P;
static constexpr int64_t KEY_ROBIN_GB_U = KEY_BASE_ROBIN_UNALIGN + Y_SPLIT_IN_B;

static constexpr int64_t KEY_ONE_IN_N_P = KEY_BASE_ONE_IN_N + Y_SPLIT_IN_P;
static constexpr int64_t KEY_ONE_IN_N_B = KEY_BASE_ONE_IN_N + Y_SPLIT_IN_B;

static constexpr int64_t KEY_VGATHER_GG = KEY_BASE_VGATHER + X_SPLIT_IN_G + Y_SPLIT_IN_G;
static constexpr int64_t KEY_VGATHER_GP = KEY_BASE_VGATHER + X_SPLIT_IN_G + Y_SPLIT_IN_P;
static constexpr int64_t KEY_VGATHER_GB = KEY_BASE_VGATHER + X_SPLIT_IN_G + Y_SPLIT_IN_B;
static constexpr int64_t KEY_VGATHER_PG = KEY_BASE_VGATHER + X_SPLIT_IN_P + Y_SPLIT_IN_G;
static constexpr int64_t KEY_VGATHER_PP = KEY_BASE_VGATHER + X_SPLIT_IN_P + Y_SPLIT_IN_P;
static constexpr int64_t KEY_VGATHER_PB = KEY_BASE_VGATHER + X_SPLIT_IN_P + Y_SPLIT_IN_B;
static constexpr int64_t KEY_VGATHER_BG = KEY_BASE_VGATHER + X_SPLIT_IN_B + Y_SPLIT_IN_G;
static constexpr int64_t KEY_VGATHER_BG_SP1 = KEY_VGATHER_BG + 1;
static constexpr int64_t KEY_VGATHER_BP = KEY_BASE_VGATHER + X_SPLIT_IN_B + Y_SPLIT_IN_P;
static constexpr int64_t KEY_VGATHER_BB = KEY_BASE_VGATHER + X_SPLIT_IN_B + Y_SPLIT_IN_B;
static constexpr int64_t KEY_VGATHER_SCALAR = KEY_BASE_VGATHER + 1;

static constexpr int64_t KEY_MOVE_PP_U = KEY_BASE_MOVE_UNALIGN + X_SPLIT_IN_P + Y_SPLIT_IN_P;
static constexpr int64_t KEY_MOVE_GG_U = KEY_BASE_MOVE_UNALIGN + X_SPLIT_IN_G + Y_SPLIT_IN_G;

struct GatherV3AxisTilingInfo {
    int64_t tileNum_;
    int64_t tileSize_;
    int64_t tileHead_;
};

class GatherV3Tiling {
public:
    explicit GatherV3Tiling(gert::TilingContext* context)
        : context_(context), nodeName_(context->GetNodeName()){};

    ge::graphStatus RunTiling4GatherV3();

private:
    ge::graphStatus CheckParams();
    ge::graphStatus CheckInputParams();
    ge::graphStatus CheckOutputParams();
    ge::graphStatus CheckAttrParams();
    ge::graphStatus FillCompileInfo();

    void AxisSplit(int64_t limit, int32_t tensorIdx, int32_t axisIdx);
    void AxisFullSplit(int32_t tensorIdx, int32_t axisIdx);
    bool IsApt2FullSplit(int64_t num);
    bool IsVReduceFixPattern();
    void AxisSlotSplit(int64_t slot, int32_t tensorIdx, int32_t axisIdx);
    void AxisLimitSplit(int64_t limit, int32_t tensorIdx, int32_t axisIdx);
    void AxisSquareSplit(int64_t limit, int32_t slot, int32_t tensorIdx, int32_t axisIdx);
    void AxisCoSplit(int64_t limit, int64_t coTileNum, int32_t tensorIdx, int32_t axisIdx);
    void AxisMultiSplit(int64_t limit, int32_t tensorIdx, int32_t axisIdx);
    void MergeAxes();
    void CalcAllMoveTmplOutputTilingA();
    bool CalcAllMoveTmplOutputTilingG(int64_t aLineNum);
    void CalcAllMoveTmplOutputTilingP(int64_t aLineNum);
    bool CalcAllMoveTmplOutputTilingB(int64_t bLineVolume);
    void AdjustAllMoveCoreTiling();
    bool ProcessMoveInReduceOut();
    void ProcessScalarX();
    void ProcessAllMoveTmpl();
    void SetCoreTilingData(int32_t bType, int32_t pType, int32_t gType, int32_t aType);
    void ProcessCoreTiling();
    void ProcessBufferSize();
    int64_t GetVolumeOfAxisA(int64_t remainBlock);
    void CalcAllCacheTmplOutputTiling(int64_t remainBlock);
    bool ProcessAllCacheShortTmpl();
    void ProcessAllCacheTmpl();
    void CalcAllCacheTmplUnalignOutputTiling(int64_t yLineVolume, int64_t idxBlockVolume);
    bool ProcessAllCacheUnalignTmpl();
    bool ProcessRobinTmpl();
    bool CalcRobinTmplUnalignInputTiling();
    bool CalcRobinTmplInputTiling(int64_t remainBlock);
    bool ProcessRobinTmplMultiCores();
    bool ProcessRobinUnalignTmpl();
    bool ProcessOneInTwoOrFour();
    void ProcessOneInNTmpl();
    void ProcessDirectGatherTmplOutputTiling(int64_t remainByte);
    void ProcessDirectGatherTmpl();
    void ProcessTiling();
    void FillTilingData();
    void PrintTilingData();
    void EnableDoubleBuffer();
    void DisableDoubleBuffer();

    gert::TilingContext* context_ = nullptr;
    std::string nodeName_ = "GatherV3";

    GatherV3TilingData tilingData_;
    int64_t tilingKey_;

    gert::Shape xShape_;
    gert::Shape indicesShape_;
    ge::DataType xDtype_ = ge::DT_UNDEFINED;

    int64_t realCoreNum_ = 0;

    int64_t xDimCount_ = 0;
    int64_t indicesDimCount_ = 0;

    int64_t xDtypeSize_ = 0;
    int64_t idxDtypeSize_ = 0;
    int64_t batchDimAttr_ = 0;
    int64_t axisAttr_ = 0;

    // 合轴之后各轴元素个数
    int64_t bSize_ = 1;
    int64_t pSize_ = 1;
    int64_t gxSize_ = 1;
    int64_t gySize_ = 1;     
    int64_t aSize_ = 1;
    int64_t axisSize_[TENSOR_NUM][AXIS_NUM];

    int64_t ubTotalBlockNum_ = 0;   // ub中可用的block总数
    int64_t aBlockNum_ = 0;     // 整个a轴占用的block数
    int64_t giBlockNum_ = 0;    // 索引的整个g轴占用的block数

    int64_t idxNumInBlock_ = 0;     // 一个block中可放下的idx数目

    bool doubleBuffer_ = false;

    int64_t xBufferSize_ = 0;
    int64_t yBufferSize_ = 0;
    int64_t idxBufferSize_ = 0;

    int64_t xAlignedUnitNum_ = 0;
    int64_t yAlignedUnitNum_ = 0;
    int64_t idxAlignedBlockVolume_ = 0;
    int64_t lineAlignedUnit_ = 0;

    // 各轴切分状态
    GatherV3AxisTilingInfo axisInfo_[TENSOR_NUM][AXIS_NUM];

    int64_t ubLineLimit_;

    int64_t tileTotalNum_;
};

template <typename T1, typename T2>
static inline T1 CeilDiv(T1 a, T2 b) {
    return (a + b - 1) / b;
}

template <typename T1, typename T2>
static inline T1 CeilAlign(T1 a, T2 b) {
    return CeilDiv(a, b) * b;
}

template <typename T1, typename T2>
static inline T1 min(T1 a, T2 b) {
    return a < b ? a : b;
}

template <typename T1, typename T2>
static inline T1 max(T1 a, T2 b) {
    return a > b ? a : b;
}

static inline void BalanceSplit(int64_t num, int64_t slot, int64_t &group, int64_t &tail) {
    group = CeilDiv(num, slot);
    tail = num % slot;
    tail = tail > 0 ? tail : slot;
}

static inline void BalanceSplitWithLimit(int64_t num, int64_t limit, 
                                  int64_t &group, int64_t &size, int64_t &head) {
    // 必须要分配的组个数
    group = CeilDiv(num, limit);

    // 以group作为slot再均衡切分
    BalanceSplit(num, group, size, head);
}

inline bool GatherV3Tiling::IsApt2FullSplit(int64_t num) {
    return num > realCoreNum_ / CORE_SPLIT_FACTOR;
}

inline bool GatherV3Tiling::IsVReduceFixPattern() {
    return (gxSize_ == FIX_PATTERN_MODEA || gxSize_ == FIX_PATTERN_MODEB);
}

inline void GatherV3Tiling::AxisSlotSplit(int64_t slot, int32_t tensorIdx, int32_t axisIdx) {
    int64_t num = axisSize_[tensorIdx][axisIdx];

    axisInfo_[tensorIdx][axisIdx].tileNum_ = min(num, slot);
    BalanceSplit(num, slot, 
                 axisInfo_[tensorIdx][axisIdx].tileSize_, 
                 axisInfo_[tensorIdx][axisIdx].tileHead_);
}

inline void GatherV3Tiling::AxisLimitSplit(int64_t limit, int32_t tensorIdx, int32_t axisIdx) {
    int64_t num = axisSize_[tensorIdx][axisIdx];
    
    BalanceSplitWithLimit(num, limit,
                          axisInfo_[tensorIdx][axisIdx].tileNum_,
                          axisInfo_[tensorIdx][axisIdx].tileSize_,
                          axisInfo_[tensorIdx][axisIdx].tileHead_);
}

inline void GatherV3Tiling::AxisSquareSplit(int64_t limit, int32_t slot, int32_t tensorIdx, int32_t axisIdx) {
    int64_t num = axisSize_[tensorIdx][axisIdx];
    if (num < limit * slot) {
        AxisSlotSplit(slot, tensorIdx, axisIdx);
        return;
    }

    AxisLimitSplit(limit, tensorIdx, axisIdx);
}

inline void GatherV3Tiling::AxisCoSplit(int64_t limit, int64_t coTileNum, int32_t tensorIdx, int32_t axisIdx) {
    int64_t slot = CeilDiv(realCoreNum_, coTileNum);
    AxisSquareSplit(limit, slot, tensorIdx, axisIdx);
}

inline void GatherV3Tiling::AxisMultiSplit(int64_t limit, int32_t tensorIdx, int32_t axisIdx) {
    int64_t totalTileNum = 1;

    for (int32_t axisLoop = 0; axisLoop < axisIdx; axisLoop++) {
        AxisFullSplit(tensorIdx, axisLoop);
        totalTileNum *= axisSize_[tensorIdx][axisIdx];
    }

    AxisCoSplit(limit, totalTileNum, tensorIdx, axisIdx);
}

inline void GatherV3Tiling::AxisSplit(int64_t limit, int32_t tensorIdx, int32_t axisIdx) {
    AxisLimitSplit(limit, tensorIdx, axisIdx);
}

inline void GatherV3Tiling::AxisFullSplit(int32_t tensorIdx, int32_t axisIdx) {
    axisInfo_[tensorIdx][axisIdx].tileNum_ = axisSize_[tensorIdx][axisIdx];
    axisInfo_[tensorIdx][axisIdx].tileSize_ = 1;
    axisInfo_[tensorIdx][axisIdx].tileHead_ = axisSize_[tensorIdx][axisIdx];
}

void GatherV3Tiling::EnableDoubleBuffer() {
    if (!doubleBuffer_) {
        doubleBuffer_ = true;
        ubTotalBlockNum_ /= UB_BUF_CNT;
    }
}

void GatherV3Tiling::DisableDoubleBuffer() {
    if (doubleBuffer_) {
        doubleBuffer_ = false;
        ubTotalBlockNum_ *= UB_BUF_CNT;
    }
}

ge::graphStatus GatherV3Tiling::RunTiling4GatherV3() {
    OP_TILING_CHECK(CheckParams() != ge::GRAPH_SUCCESS, OP_LOGE(nodeName_, "CheckParams failed."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(FillCompileInfo() != ge::GRAPH_SUCCESS, OP_LOGE(nodeName_, "FillCompileInfo failed."),
                    return ge::GRAPH_FAILED);
    OP_LOGD(nodeName_, "Platform info, ubSizePlatForm:%lu, totalCoreNum:%u.", ubTotalBlockNum_,
            realCoreNum_);
    OP_TILING_CHECK(realCoreNum_ == 0 || ubTotalBlockNum_ == 0,
                    OP_LOGE(nodeName_, "Invalid compile info."), return ge::GRAPH_FAILED);
                    
    MergeAxes();
    ProcessTiling();
    ProcessCoreTiling();
    ProcessBufferSize();

    if (tileTotalNum_ < realCoreNum_) {
        realCoreNum_ = tileTotalNum_;
    }

    context_->SetBlockDim(realCoreNum_);
    context_->SetTilingKey(static_cast<uint64_t>(tilingKey_));
    
    FillTilingData();
    PrintTilingData();
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = WORK_SPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus GatherV3Tiling::CheckParams() {
    OP_TILING_CHECK(CheckInputParams() != ge::GRAPH_SUCCESS, OP_LOGE(nodeName_, "CheckInputParams failed."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOutputParams() != ge::GRAPH_SUCCESS, OP_LOGE(nodeName_, "CheckOutputParams failed."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAttrParams() != ge::GRAPH_SUCCESS, OP_LOGE(nodeName_, "CheckAttrParams failed."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherV3Tiling::CheckInputParams() {
    auto xDescPtr = context_->GetInputDesc(X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, xDescPtr);
    xDtype_ = xDescPtr->GetDataType();
    OP_TILING_CHECK(std::find(X_DTYPES.begin(), X_DTYPES.end(), xDtype_) == X_DTYPES.end(),
                    OP_LOGE(nodeName_, "Input x dtype only supports %s, not %d.", X_DTYPES_STR.c_str(),
                            static_cast<int32_t>(xDtype_)),
                    return ge::GRAPH_FAILED);

    xDtypeSize_ = ge::GetSizeByDataType(xDtype_);
    OP_TILING_CHECK(xDtypeSize_ <= 0, OP_LOGE(nodeName_, "Get xDtypeSize[%ld] failed.", xDtypeSize_),
                    return ge::GRAPH_FAILED);

    auto xShapePtr = context_->GetInputShape(X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    xShape_ = xShapePtr->GetStorageShape();
    xDimCount_ = xShape_.GetDimNum();
    OP_TILING_CHECK(xDimCount_ < 0,
                    OP_LOGE(nodeName_, "Input x dim count[%ld] must >= 0.", xDimCount_),
                    return ge::GRAPH_FAILED);
    if (xDimCount_ == 0) {
        xDimCount_ = 1;
    }

    auto indicesDescPtr = context_->GetInputDesc(INDICES_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, indicesDescPtr);
    auto indicesDtype = indicesDescPtr->GetDataType();
    OP_TILING_CHECK(
        std::find(INT_DTYPES.begin(), INT_DTYPES.end(), indicesDtype) == INT_DTYPES.end(),
        OP_LOGE(nodeName_, "Input indices dtype not in %s.", INT_DTYPES_STR.c_str()),
        return ge::GRAPH_FAILED);

    idxDtypeSize_ = ge::GetSizeByDataType(indicesDtype);
    OP_TILING_CHECK(idxDtypeSize_ <= 0, OP_LOGE(nodeName_, "Get idxDtypeSize_[%ld] failed.", idxDtypeSize_),
                    return ge::GRAPH_FAILED);
    
    idxNumInBlock_ = BLOCK_SIZE / idxDtypeSize_;

    auto indicesShapePtr = context_->GetInputShape(INDICES_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, indicesShapePtr);
    indicesShape_ = indicesShapePtr->GetStorageShape();
    indicesDimCount_ = indicesShape_.GetDimNum();
    OP_TILING_CHECK(indicesDimCount_ <= 0,
                    OP_LOGE(nodeName_, "Input indices dim count[%ld] must >= 1.", indicesDimCount_),
                    return ge::GRAPH_FAILED);

    auto axisDescPtr = context_->GetInputDesc(AXIS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, axisDescPtr);
    auto axisDtype = axisDescPtr->GetDataType();
    OP_TILING_CHECK(
        std::find(INT_DTYPES.begin(), INT_DTYPES.end(), axisDtype) == INT_DTYPES.end(),
        OP_LOGE(nodeName_, "Input axis dtype not in %s.", INT_DTYPES_STR.c_str()),
        return ge::GRAPH_FAILED);

    auto axisTensor = context_->GetInputTensor(AXIS_INDEX);
    OP_TILING_CHECK(
        axisTensor->GetShapeSize() != 1,
        OP_LOGE(nodeName_, "Input axit size is not 1"),
        return ge::GRAPH_FAILED);

    if (axisDtype == ge::DT_INT32) {
        axisAttr_ = *axisTensor->GetData<int32_t>();
    } else {
        axisAttr_ = *axisTensor->GetData<int64_t>();
    }
    axisAttr_ = axisAttr_ < 0 ? xDimCount_ + axisAttr_ : axisAttr_;
    OP_TILING_CHECK(axisAttr_ < 0 || axisAttr_ >= xDimCount_ ,
                    OP_LOGE(nodeName_, "Invalid axis [%ld]", axisAttr_),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherV3Tiling::CheckOutputParams() {
    auto yDescPtr = context_->GetInputDesc(Y_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, yDescPtr);
    auto yDtype = yDescPtr->GetDataType();
    OP_TILING_CHECK(yDtype != xDtype_, OP_LOGE(nodeName_, "The dtype of y and x must be the same."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherV3Tiling::CheckAttrParams() {
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const int64_t* ptrBatchDim = attrs->GetAttrPointer<int64_t>(BATCH_DIMS_ATTR_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, ptrBatchDim);
    batchDimAttr_ = *ptrBatchDim;
    batchDimAttr_ = batchDimAttr_ < 0 ? indicesDimCount_ + batchDimAttr_ : batchDimAttr_;

    OP_TILING_CHECK(batchDimAttr_ < 0 || batchDimAttr_ >= indicesDimCount_,
                    OP_LOGE(nodeName_, "Invalid attr, batchDimAttr_: %ld.", batchDimAttr_), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherV3Tiling::FillCompileInfo() {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint32_t totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    if (totalCoreNum == 0) {
        OP_LOGE(tilingContext->GetNodeName(), "coreNum must greater than 0.");
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    realCoreNum_ = totalCoreNum;
    ubTotalBlockNum_ = ubSizePlatForm / BLOCK_SIZE - RSV_BLOCK;
    return ge::GRAPH_SUCCESS;
}

void GatherV3Tiling::MergeAxes() {
    if (xShape_.GetDimNum() == 0) {
        for (int64_t i = batchDimAttr_; i < indicesDimCount_; i++) {
            gySize_ *= indicesShape_.GetDim(i);
        }
    } else {
        for (int64_t i = 0; i < batchDimAttr_; i++) {
            bSize_ *= xShape_.GetDim(i);
        }

        for (int64_t i = batchDimAttr_; i < axisAttr_; i++) {
            pSize_ *= xShape_.GetDim(i);
        }

        gxSize_ = xShape_.GetDim(axisAttr_);
        
        for (int64_t i = batchDimAttr_; i < indicesDimCount_; i++) {
            gySize_ *= indicesShape_.GetDim(i);
        }

        for (int64_t i = axisAttr_ + 1; i < xDimCount_; i++) {
            aSize_ *= xShape_.GetDim(i);
        }
    }

    aBlockNum_ = CeilDiv(aSize_ * xDtypeSize_, BLOCK_SIZE);
    giBlockNum_ = CeilDiv(gySize_ * idxDtypeSize_, BLOCK_SIZE);

    // 初始化各轴长度
    axisSize_[TENSOR_X][AXIS_B] = bSize_;
    axisSize_[TENSOR_X][AXIS_P] = pSize_;
    axisSize_[TENSOR_X][AXIS_G] = gxSize_;
    axisSize_[TENSOR_X][AXIS_A] = aSize_;
    axisSize_[TENSOR_Y][AXIS_B] = bSize_;
    axisSize_[TENSOR_Y][AXIS_P] = pSize_;
    axisSize_[TENSOR_Y][AXIS_G] = gySize_;
    axisSize_[TENSOR_Y][AXIS_A] = aSize_;
    axisSize_[TENSOR_I][AXIS_B] = bSize_;
    axisSize_[TENSOR_I][AXIS_P] = 1;
    axisSize_[TENSOR_I][AXIS_G] = gySize_;
    axisSize_[TENSOR_I][AXIS_A] = 1;

    // 初始化默认切分结果 不切分
    axisInfo_[TENSOR_X][AXIS_B] = {1, bSize_, 1};
    axisInfo_[TENSOR_X][AXIS_P] = {1, pSize_, 1};
    axisInfo_[TENSOR_X][AXIS_G] = {1, gxSize_, 1};
    axisInfo_[TENSOR_X][AXIS_A] = {1, aSize_, 1};
    axisInfo_[TENSOR_Y][AXIS_B] = {1, bSize_, 1};
    axisInfo_[TENSOR_Y][AXIS_P] = {1, pSize_, 1};
    axisInfo_[TENSOR_Y][AXIS_G] = {1, gySize_, 1};
    axisInfo_[TENSOR_Y][AXIS_A] = {1, aSize_, 1};
    axisInfo_[TENSOR_I][AXIS_B] = {1, bSize_, 1};
    axisInfo_[TENSOR_I][AXIS_P] = {1, 1, 1};
    axisInfo_[TENSOR_I][AXIS_G] = {1, gySize_, 1};
    axisInfo_[TENSOR_I][AXIS_A] = {1, 1, 1};
}

// key      out     idx     core    
// 10430    a       g       [b, p, g_out, a_out]
// 10330    g       g       [b, p, g_idx]
// 10200    p       b       [b_out, p_out]
// 12330    [p, g]  g       [b, p_out, g_out]
// 11100    b       b       b_idx

void GatherV3Tiling::CalcAllMoveTmplOutputTilingA() {
    tilingKey_ = KEY_MOVE_AG;
    AxisFullSplit(TENSOR_Y, AXIS_B);
    AxisFullSplit(TENSOR_Y, AXIS_P);

    // 输出a轴切分
    int64_t yBlockVolume = (ubTotalBlockNum_ * UB_OUT_RATIO) / (UB_OUT_RATIO + UB_IDX_RATIO);

    AxisSplit(yBlockVolume * BLOCK_SIZE / xDtypeSize_, TENSOR_Y, AXIS_A);

    // 索引可以使用的ub block
    int64_t remainBlock = ubTotalBlockNum_ - CeilDiv(axisInfo_[TENSOR_Y][AXIS_A].tileSize_, BLOCK_SIZE);

    // 索引切分
    AxisFullSplit(TENSOR_I, AXIS_B);
    AxisCoSplit(remainBlock * idxNumInBlock_, axisInfo_[TENSOR_Y][AXIS_A].tileNum_, TENSOR_I, AXIS_G);
}

// 10330    g       g       [b, p, g_idx]
bool GatherV3Tiling::CalcAllMoveTmplOutputTilingG(int64_t aLineNum) {
    tilingKey_ = KEY_MOVE_GG;
    AxisFullSplit(TENSOR_Y, AXIS_B);
    AxisFullSplit(TENSOR_Y, AXIS_P);
    AxisFullSplit(TENSOR_I, AXIS_B);

    for (int64_t aLineLimit = aLineNum; aLineLimit > 0; aLineLimit--) {
        // 先切分输出g轴    
        AxisSplit(aLineLimit, TENSOR_Y, AXIS_G);

        int64_t remain_block = ubTotalBlockNum_ - axisInfo_[TENSOR_Y][AXIS_G].tileSize_* aBlockNum_;
        int64_t gElemLimit = min(remain_block * idxNumInBlock_ , gySize_);

        AxisSplit(gElemLimit, TENSOR_I, AXIS_G);

        if (axisInfo_[TENSOR_I][AXIS_G].tileSize_ >= axisInfo_[TENSOR_Y][AXIS_G].tileSize_) {
            return true;
        }
    }

    return false;
}

// 10200    p       b       [b_out, p_out]
// 12330    [p, g]  g       [b, p_out, g_out]
void GatherV3Tiling::CalcAllMoveTmplOutputTilingP(int64_t aLineNum) {
    int64_t gLineNum = aLineNum / gySize_;     // 能容纳g的行数    

    AxisFullSplit(TENSOR_Y, AXIS_B);
    AxisSplit(gLineNum, TENSOR_Y, AXIS_P);

    // 剩余UB可否放下索引的整个g轴
    int64_t remainBlock = ubTotalBlockNum_ - axisInfo_[TENSOR_Y][AXIS_P].tileSize_ * gySize_ * aBlockNum_;
    int64_t idxVolume = remainBlock * idxNumInBlock_;
    
    int64_t bLineNum = idxVolume / gySize_;
    if (bLineNum < 1) {
        tilingKey_ = KEY_MOVE_PG;
        AxisFullSplit(TENSOR_Y, AXIS_P);

        remainBlock = ubTotalBlockNum_ - axisInfo_[TENSOR_Y][AXIS_P].tileSize_ * gySize_ * aBlockNum_;
        idxVolume = remainBlock * idxNumInBlock_;

        AxisFullSplit(TENSOR_I, AXIS_B);
        AxisSplit(idxVolume, TENSOR_I, AXIS_G);
    } else {
        tilingKey_ = KEY_MOVE_PB;

        // 输出p切分，索引b切分，g轴不切
        AxisSplit(bLineNum, TENSOR_I, AXIS_B);
    }
}

bool GatherV3Tiling::CalcAllMoveTmplOutputTilingB(int64_t bLineVolume) {
    // b轴切分
    tilingKey_ = KEY_MOVE_BB;

    if (!IsApt2FullSplit(bSize_)) {
        return false;
    }

    if (bSize_ < realCoreNum_) {
        AxisFullSplit(TENSOR_Y, AXIS_B);
        AxisFullSplit(TENSOR_I, AXIS_B);
        return true;
    }

    AxisSquareSplit(bLineVolume, realCoreNum_, TENSOR_Y, AXIS_B);
    AxisSquareSplit(bLineVolume, realCoreNum_, TENSOR_I, AXIS_B);
    return true;
}

void GatherV3Tiling::AdjustAllMoveCoreTiling() {
    if (tilingKey_ != KEY_MOVE_GG) {
        return;
    }

    int64_t tileNum = bSize_ * pSize_ * axisInfo_[TENSOR_I][AXIS_G].tileNum_;
    if (tileNum >= realCoreNum_) {
        return;
    }

    int64_t tileMaxNum = bSize_ * pSize_ * gySize_;
    int64_t gTileSizeLimit = tileMaxNum / realCoreNum_;
    if (gTileSizeLimit < 1) {
        return;
    }

    AxisSplit(gTileSizeLimit, TENSOR_I, AXIS_G);
}

bool GatherV3Tiling::ProcessMoveInReduceOut() {
    if (aSize_ * xDtypeSize_ % BLOCK_SIZE == 0) {
        return false;
    }

    if (bSize_ > 1) {
        return false;
    }

    if (realCoreNum_ >= pSize_ * gySize_) {
        return false;
    }

    int64_t remainBlock = ubTotalBlockNum_;
    int64_t lineBlock = aBlockNum_;

    // 1Byte且尾轴长度为奇数，需要cast到2B后进行vreduce处理
    if (xDtypeSize_ == 1 && (aSize_ % VREDUCEV2_SIZE != 0)) {
        remainBlock -= RSV_BLOCK; // 1B <-> 2B cast，需额外保留8block，共16block。

        // 1B按cast到2B的长度计算
        if (aSize_ > FULL_BLOCK_SIZE / VREDUCEV2_SIZE) {
            // 尾轴长时，连带pad一同做cast
            lineBlock = CeilDiv(aSize_, BLOCK_SIZE) * VREDUCEV2_SIZE;
        } else {
            // 尾轴短时，尾轴短时，使用高维切分接口, 每个尾轴一个repeat
            lineBlock = CeilDiv(aSize_ * VREDUCEV2_SIZE, BLOCK_SIZE); 
        }
    }

    int64_t lineVolume = remainBlock / (lineBlock * UB_BUF_CNT); // 不考虑索引时的尾轴数量上限
    while (lineVolume > 0) {
        // 不考虑b轴的情况下，idx个数不会超过gySize_
        int64_t idxNum = min(lineVolume, gySize_);
        if (lineVolume * lineBlock * UB_BUF_CNT + CeilDiv(idxNum * idxDtypeSize_, BLOCK_SIZE) <= remainBlock) {
            break;
        }
        lineVolume--;
    }

    if (lineVolume < 1) {
        return false;
    }

    // 需要更新尾轴block大小，正确计算x,y buffer size.
    aBlockNum_ = lineBlock;
    if (lineVolume >= gySize_ && IsApt2FullSplit(pSize_)) {
        int64_t gLineNum = lineVolume / gySize_;
        AxisFullSplit(TENSOR_Y, AXIS_B);
        AxisSquareSplit(gLineNum, realCoreNum_, TENSOR_Y, AXIS_P);
        tilingKey_ = KEY_MOVE_PP_U;
        return true;
    }
    
    AxisFullSplit(TENSOR_Y, AXIS_B);
    AxisFullSplit(TENSOR_Y, AXIS_P);
    AxisCoSplit(lineVolume, pSize_, TENSOR_Y, AXIS_G);

    tilingKey_ = KEY_MOVE_GG_U;
    return true;
}

void GatherV3Tiling::ProcessScalarX() {
    int64_t gLineSize = gxSize_ * xDtypeSize_;
    int64_t remainBlock = ubTotalBlockNum_ - CeilDiv(gLineSize, BLOCK_SIZE);
    int64_t blockVolume = (remainBlock) / UB_BUF_CNT; // y和idx各占一半 y作为临时内存给idx校验和转换类型使用
    int64_t elemNum = blockVolume * BLOCK_SIZE / idxDtypeSize_;
    if (elemNum < gySize_) {
        EnableDoubleBuffer();
        blockVolume /= UB_BUF_CNT;
        elemNum = blockVolume * BLOCK_SIZE / idxDtypeSize_;
    }
    int64_t slot = realCoreNum_;
    AxisFullSplit(TENSOR_Y, AXIS_B);
    AxisFullSplit(TENSOR_Y, AXIS_P);
    AxisSlotSplit(slot, TENSOR_Y, AXIS_G);
    ubLineLimit_ = min(elemNum, axisInfo_[TENSOR_Y][AXIS_G].tileSize_);

    tilingKey_ = KEY_VGATHER_SCALAR;
    return;
}

void GatherV3Tiling::ProcessAllMoveTmpl() {
    if (ProcessMoveInReduceOut()) {
        return;
    }

    // 纯搬运模板充分使能doube buffer
    EnableDoubleBuffer();

    // 配比：输出8 索引1。计算可以放入多少个a轴
    int64_t aLineNum = (ubTotalBlockNum_ * UB_OUT_RATIO) / (aBlockNum_ * (UB_OUT_RATIO + UB_IDX_RATIO));
    if (aLineNum / gySize_ == 1 && IsApt2FullSplit(bSize_ * pSize_)) {
        // 1. 尝试只切输出p轴，索引只切b轴，即可放下索引整个g轴
        // 2. 若不可, 输出切p,g, 索引切g
        CalcAllMoveTmplOutputTilingP(aLineNum);
        return;
    }

    while (aLineNum >= 0 && 
           (aLineNum * aBlockNum_ + CeilDiv(aLineNum * idxDtypeSize_, BLOCK_SIZE)) > ubTotalBlockNum_) {
        aLineNum--;
    }

    // 快速路径 尽量用核 直接切g轴 10331 Y和I等切分
    int64_t outSize = bSize_ * pSize_;
    if (outSize < realCoreNum_ && aLineNum >= 1) {
        tilingKey_ = KEY_MOVE_GG;
        if (outSize * gySize_ < realCoreNum_) {
            AxisFullSplit(TENSOR_Y, AXIS_B);
            AxisFullSplit(TENSOR_Y, AXIS_P);
            AxisFullSplit(TENSOR_Y, AXIS_G);
            AxisFullSplit(TENSOR_I, AXIS_G);
            return;
        }

        if (realCoreNum_ * aLineNum >= outSize * gySize_) {
            AxisFullSplit(TENSOR_Y, AXIS_B);
            AxisFullSplit(TENSOR_Y, AXIS_P);
            AxisSlotSplit(realCoreNum_, TENSOR_Y, AXIS_G);
            AxisSlotSplit(realCoreNum_, TENSOR_I, AXIS_G);
            return;
        }
    }

    tilingKey_ = KEY_BASE_MOVE;

    int64_t bElemBlockNum = pSize_ * gySize_ * aBlockNum_ + giBlockNum_;
    if (bElemBlockNum < ubTotalBlockNum_) {
        // b轴切分
        if (CalcAllMoveTmplOutputTilingB(ubTotalBlockNum_ / bElemBlockNum)) {
            return;
        }
    }

    if (aLineNum >= gySize_ && IsApt2FullSplit(bSize_ * pSize_)) {
        // 1. 尝试只切输出p轴，索引只切b轴，即可放下索引整个g轴
        // 2. 若不可, 输出切p,g, 索引切g
        CalcAllMoveTmplOutputTilingP(aLineNum);
        return;
    }

    if (aLineNum >= 1) {
        // g轴切分
        if (CalcAllMoveTmplOutputTilingG(aLineNum)) {
            return;
        }
    }

    // a轴切分
    CalcAllMoveTmplOutputTilingA();
    return;
}

inline void GatherV3Tiling::SetCoreTilingData(int32_t bType, int32_t pType, int32_t gType, int32_t aType) {
    tilingData_.set_bTileNum(axisInfo_[bType][AXIS_B].tileNum_);
    tilingData_.set_bTileSize(axisInfo_[bType][AXIS_B].tileSize_);
    tilingData_.set_bTileHead(axisInfo_[bType][AXIS_B].tileHead_);

    tilingData_.set_pTileNum(axisInfo_[pType][AXIS_P].tileNum_);
    tilingData_.set_pTileSize(axisInfo_[pType][AXIS_P].tileSize_);
    tilingData_.set_pTileHead(axisInfo_[pType][AXIS_P].tileHead_);

    tilingData_.set_gTileNum(axisInfo_[gType][AXIS_G].tileNum_);
    tilingData_.set_gTileSize(axisInfo_[gType][AXIS_G].tileSize_);
    tilingData_.set_gTileHead(axisInfo_[gType][AXIS_G].tileHead_);

    tilingData_.set_aTileNum(axisInfo_[aType][AXIS_A].tileNum_);
    tilingData_.set_aTileSize(axisInfo_[aType][AXIS_A].tileSize_);
    tilingData_.set_aTileHead(axisInfo_[aType][AXIS_A].tileHead_);

    tileTotalNum_ = axisInfo_[bType][AXIS_B].tileNum_ * 
                    axisInfo_[pType][AXIS_P].tileNum_ * 
                    axisInfo_[gType][AXIS_G].tileNum_ *
                    axisInfo_[aType][AXIS_A].tileNum_;
}

inline void GatherV3Tiling::ProcessCoreTiling() {
    switch (tilingKey_) {
        case KEY_MOVE_AG:
        case KEY_MOVE_GG:
        case KEY_MOVE_PG:
            SetCoreTilingData(TENSOR_Y, TENSOR_Y, TENSOR_I, TENSOR_Y);
            break;
        case KEY_MOVE_PB:
            SetCoreTilingData(TENSOR_I, TENSOR_Y, TENSOR_Y, TENSOR_Y);
            break;
        case KEY_CACHE_PG:
        case KEY_CACHE_BG:
        case KEY_CACHE_BP:
        case KEY_CACHE_PG_U:
        case KEY_CACHE_BG_U:
        case KEY_CACHE_BP_U:
        case KEY_VGATHER_GG:
        case KEY_VGATHER_PG:
        case KEY_VGATHER_BG:
        case KEY_VGATHER_BP:
            SetCoreTilingData(TENSOR_X, TENSOR_X, TENSOR_X, TENSOR_X);
            break;
        default:
            SetCoreTilingData(TENSOR_Y, TENSOR_Y, TENSOR_Y, TENSOR_Y);
            break;
    }
}

inline void GatherV3Tiling::ProcessBufferSize() {
    switch(tilingKey_) {
        case KEY_MOVE_AG:
            xBufferSize_    = 0;
            yBufferSize_    = axisInfo_[TENSOR_Y][AXIS_A].tileSize_ * xDtypeSize_;
            idxBufferSize_  = axisInfo_[TENSOR_I][AXIS_G].tileSize_ * idxDtypeSize_;
            ubLineLimit_    = 1;
            break;
        case KEY_MOVE_GG:
            xBufferSize_    = 0;
            yBufferSize_    = axisInfo_[TENSOR_Y][AXIS_G].tileSize_ * aBlockNum_ * BLOCK_SIZE;
            idxBufferSize_  = axisInfo_[TENSOR_I][AXIS_G].tileSize_ * idxDtypeSize_;
            ubLineLimit_    = axisInfo_[TENSOR_Y][AXIS_G].tileSize_;
            break;
        case KEY_MOVE_PB:
            xBufferSize_    = 0;
            yBufferSize_    = axisInfo_[TENSOR_Y][AXIS_P].tileSize_ * gySize_ * aBlockNum_ * BLOCK_SIZE;
            idxBufferSize_  = axisInfo_[TENSOR_I][AXIS_B].tileSize_ * gySize_ * idxDtypeSize_;
            ubLineLimit_    = 1;
            break;
        case KEY_MOVE_PG:
            xBufferSize_    = 0;
            yBufferSize_    = 1 * axisInfo_[TENSOR_I][AXIS_G].tileSize_ * aBlockNum_ * BLOCK_SIZE;
            idxBufferSize_  = axisInfo_[TENSOR_I][AXIS_G].tileSize_ * idxDtypeSize_;
            ubLineLimit_    = 1;
            break;
        case KEY_MOVE_BB:
            xBufferSize_    = 0;
            yBufferSize_    = axisInfo_[TENSOR_Y][AXIS_B].tileSize_ * pSize_ * gySize_ * aBlockNum_ * BLOCK_SIZE;
            idxBufferSize_  = axisInfo_[TENSOR_I][AXIS_B].tileSize_ * gySize_ * idxDtypeSize_;
            ubLineLimit_    = axisInfo_[TENSOR_Y][AXIS_B].tileSize_;
            break;
        case KEY_CACHE_GG:
            xBufferSize_    = gxSize_ * aBlockNum_ * BLOCK_SIZE;
            yBufferSize_    = axisInfo_[TENSOR_Y][AXIS_G].tileSize_ * aBlockNum_ * BLOCK_SIZE;
            idxBufferSize_  = axisInfo_[TENSOR_Y][AXIS_G].tileSize_ * idxDtypeSize_;
            ubLineLimit_    = axisInfo_[TENSOR_Y][AXIS_G].tileSize_;
            break;
        case KEY_CACHE_GP:
            xBufferSize_    = gxSize_ * aBlockNum_ * BLOCK_SIZE;
            yBufferSize_    = axisInfo_[TENSOR_Y][AXIS_P].tileSize_ * gySize_ * aBlockNum_ * BLOCK_SIZE;
            idxBufferSize_  = gySize_ * idxDtypeSize_;
            ubLineLimit_    = 1;
            break;
        case KEY_CACHE_GB:
            xBufferSize_    = gxSize_ * aBlockNum_ * BLOCK_SIZE;
            yBufferSize_    = axisInfo_[TENSOR_Y][AXIS_B].tileSize_ * pSize_ * gySize_ * aBlockNum_ * BLOCK_SIZE;
            idxBufferSize_  = axisInfo_[TENSOR_Y][AXIS_B].tileSize_ * gySize_ * idxDtypeSize_;
            ubLineLimit_    = 1;
            break;
        case KEY_CACHE_PG:
            xBufferSize_    = axisInfo_[TENSOR_X][AXIS_P].tileSize_ * gxSize_ * aBlockNum_ * BLOCK_SIZE;
            yBufferSize_    = axisInfo_[TENSOR_Y][AXIS_G].tileSize_ * aBlockNum_ * BLOCK_SIZE;
            idxBufferSize_  = axisInfo_[TENSOR_Y][AXIS_G].tileSize_ * idxDtypeSize_;
            ubLineLimit_    = axisInfo_[TENSOR_Y][AXIS_G].tileSize_;
            break;
        case KEY_CACHE_PP:
            xBufferSize_    = axisInfo_[TENSOR_X][AXIS_P].tileSize_ * gxSize_ * aBlockNum_ * BLOCK_SIZE;
            yBufferSize_    = axisInfo_[TENSOR_Y][AXIS_P].tileSize_ * gySize_ * aBlockNum_ * BLOCK_SIZE;
            idxBufferSize_  = gySize_ * idxDtypeSize_;
            ubLineLimit_    = axisInfo_[TENSOR_X][AXIS_P].tileSize_;
            break;
        case KEY_CACHE_PB:
            xBufferSize_    = axisInfo_[TENSOR_X][AXIS_P].tileSize_ * gxSize_ * aBlockNum_ * BLOCK_SIZE;
            yBufferSize_    = axisInfo_[TENSOR_Y][AXIS_B].tileSize_ * pSize_ * gySize_ * aBlockNum_ * BLOCK_SIZE;
            idxBufferSize_  = axisInfo_[TENSOR_Y][AXIS_B].tileSize_ * gySize_ * idxDtypeSize_;
            ubLineLimit_    = axisInfo_[TENSOR_X][AXIS_P].tileSize_;
            break;
        case KEY_CACHE_BG:
        case KEY_CACHE_BG_SP1:
            xBufferSize_    = axisInfo_[TENSOR_X][AXIS_B].tileSize_ * pSize_ * gxSize_ * aBlockNum_ * BLOCK_SIZE;
            yBufferSize_    = axisInfo_[TENSOR_Y][AXIS_G].tileSize_ * aBlockNum_ * BLOCK_SIZE;
            idxBufferSize_  = axisInfo_[TENSOR_Y][AXIS_G].tileSize_ * idxDtypeSize_;
            ubLineLimit_    = axisInfo_[TENSOR_Y][AXIS_G].tileSize_;
            break;
        case KEY_CACHE_BG_SP2:
            xBufferSize_    = CeilAlign(gxSize_ * aSize_ * xDtypeSize_, BLOCK_SIZE);
            yBufferSize_    = CeilAlign(axisInfo_[TENSOR_Y][AXIS_G].tileSize_ * aSize_ * xDtypeSize_, BLOCK_SIZE);
            idxBufferSize_  = CeilAlign(axisInfo_[TENSOR_Y][AXIS_G].tileSize_ * idxDtypeSize_, BLOCK_SIZE);
            ubLineLimit_    = axisInfo_[TENSOR_Y][AXIS_G].tileSize_;
            break;
        case KEY_CACHE_BP:
            xBufferSize_    = axisInfo_[TENSOR_X][AXIS_B].tileSize_ * pSize_ * gxSize_ * aBlockNum_ * BLOCK_SIZE;
            yBufferSize_    = axisInfo_[TENSOR_Y][AXIS_P].tileSize_ * gySize_ * aBlockNum_ * BLOCK_SIZE;
            idxBufferSize_  = gySize_ * idxDtypeSize_;
            ubLineLimit_    = axisInfo_[TENSOR_Y][AXIS_P].tileSize_;
            break;
        case KEY_CACHE_BB:
            xBufferSize_    = axisInfo_[TENSOR_X][AXIS_B].tileSize_ * pSize_ * gxSize_ * aBlockNum_ * BLOCK_SIZE;
            yBufferSize_    = axisInfo_[TENSOR_Y][AXIS_B].tileSize_ * pSize_ * gySize_ * aBlockNum_ * BLOCK_SIZE;
            idxBufferSize_  = axisInfo_[TENSOR_Y][AXIS_B].tileSize_ * gySize_ * idxDtypeSize_;
            ubLineLimit_    = axisInfo_[TENSOR_X][AXIS_B].tileSize_;
            break;
        case KEY_CACHE_GG_U:
            xBufferSize_    = xAlignedUnitNum_ * lineAlignedUnit_ * aBlockNum_ * BLOCK_SIZE;
            yBufferSize_    = yAlignedUnitNum_ * lineAlignedUnit_ * aBlockNum_ * BLOCK_SIZE;
            idxBufferSize_  = idxAlignedBlockVolume_ * BLOCK_SIZE;
            ubLineLimit_    = axisInfo_[TENSOR_Y][AXIS_G].tileSize_;
            break;
        case KEY_CACHE_GP_U:
            xBufferSize_    = xAlignedUnitNum_ * lineAlignedUnit_ * aBlockNum_ * BLOCK_SIZE;
            yBufferSize_    = yAlignedUnitNum_ * lineAlignedUnit_ * aBlockNum_ * BLOCK_SIZE;
            idxBufferSize_  = idxAlignedBlockVolume_ * BLOCK_SIZE;
            ubLineLimit_    = 1;
            break;
        case KEY_CACHE_GB_U:
            xBufferSize_    = xAlignedUnitNum_ * lineAlignedUnit_ * aBlockNum_ * BLOCK_SIZE;
            yBufferSize_    = yAlignedUnitNum_ * lineAlignedUnit_ * aBlockNum_ * BLOCK_SIZE;
            idxBufferSize_  = idxAlignedBlockVolume_ * BLOCK_SIZE;
            ubLineLimit_    = 1;
            break;
        case KEY_CACHE_PG_U:
            xBufferSize_    = xAlignedUnitNum_ * lineAlignedUnit_ * aBlockNum_ * BLOCK_SIZE;
            yBufferSize_    = yAlignedUnitNum_ * lineAlignedUnit_ * aBlockNum_ * BLOCK_SIZE;
            idxBufferSize_  = idxAlignedBlockVolume_ * BLOCK_SIZE;
            ubLineLimit_    = axisInfo_[TENSOR_Y][AXIS_G].tileSize_;
            break;
        case KEY_CACHE_PP_U:
            xBufferSize_    = xAlignedUnitNum_ * lineAlignedUnit_ * aBlockNum_ * BLOCK_SIZE;
            yBufferSize_    = yAlignedUnitNum_ * lineAlignedUnit_ * aBlockNum_ * BLOCK_SIZE;
            idxBufferSize_  = idxAlignedBlockVolume_ * BLOCK_SIZE;
            ubLineLimit_    = axisInfo_[TENSOR_X][AXIS_P].tileSize_;
            break;
        case KEY_CACHE_PB_U:
            xBufferSize_    = xAlignedUnitNum_ * lineAlignedUnit_ * aBlockNum_ * BLOCK_SIZE;
            yBufferSize_    = yAlignedUnitNum_ * lineAlignedUnit_ * aBlockNum_ * BLOCK_SIZE;
            idxBufferSize_  = idxAlignedBlockVolume_ * BLOCK_SIZE;
            ubLineLimit_    = axisInfo_[TENSOR_X][AXIS_P].tileSize_;
            break;
        case KEY_CACHE_BG_U:
        case KEY_CACHE_BG_U_SP1:
            xBufferSize_    = xAlignedUnitNum_ * lineAlignedUnit_ * aBlockNum_ * BLOCK_SIZE;
            yBufferSize_    = yAlignedUnitNum_ * lineAlignedUnit_ * aBlockNum_ * BLOCK_SIZE;
            idxBufferSize_  = idxAlignedBlockVolume_ * BLOCK_SIZE;
            ubLineLimit_    = axisInfo_[TENSOR_Y][AXIS_G].tileSize_;
            break;
        case KEY_CACHE_BP_U:
            xBufferSize_    = xAlignedUnitNum_ * lineAlignedUnit_ * aBlockNum_ * BLOCK_SIZE;
            yBufferSize_    = yAlignedUnitNum_ * lineAlignedUnit_ * aBlockNum_ * BLOCK_SIZE;
            idxBufferSize_  = idxAlignedBlockVolume_ * BLOCK_SIZE;
            ubLineLimit_    = axisInfo_[TENSOR_Y][AXIS_P].tileSize_;
            break;
        case KEY_CACHE_BB_U:
            xBufferSize_    = xAlignedUnitNum_ * lineAlignedUnit_ * aBlockNum_ * BLOCK_SIZE;
            yBufferSize_    = yAlignedUnitNum_ * lineAlignedUnit_ * aBlockNum_ * BLOCK_SIZE;
            idxBufferSize_  = idxAlignedBlockVolume_ * BLOCK_SIZE;
            ubLineLimit_    = axisInfo_[TENSOR_X][AXIS_B].tileSize_;
            break;
        case KEY_ROBIN_GG:
            xBufferSize_    = axisInfo_[TENSOR_X][AXIS_G].tileSize_ * aBlockNum_ * BLOCK_SIZE;
            yBufferSize_    = gySize_ * aBlockNum_ * BLOCK_SIZE;
            idxBufferSize_  = giBlockNum_ * BLOCK_SIZE;
            ubLineLimit_    = axisInfo_[TENSOR_X][AXIS_G].tileSize_;
            break;
        case KEY_ROBIN_GG_SP1:
            xBufferSize_    = axisInfo_[TENSOR_X][AXIS_G].tileSize_ * aBlockNum_ * BLOCK_SIZE;
            yBufferSize_    = axisInfo_[TENSOR_Y][AXIS_G].tileSize_ * aBlockNum_ * BLOCK_SIZE;
            idxBufferSize_  = CeilAlign(axisInfo_[TENSOR_Y][AXIS_G].tileSize_ * idxDtypeSize_, BLOCK_SIZE);
            ubLineLimit_    = axisInfo_[TENSOR_X][AXIS_G].tileSize_;
            break;
        case KEY_ROBIN_GP:
            xBufferSize_    = axisInfo_[TENSOR_X][AXIS_G].tileSize_ * aBlockNum_ * BLOCK_SIZE;
            yBufferSize_    = axisInfo_[TENSOR_Y][AXIS_P].tileSize_ * gySize_ * aBlockNum_ * BLOCK_SIZE;
            idxBufferSize_  = giBlockNum_ * BLOCK_SIZE;
            ubLineLimit_    = axisInfo_[TENSOR_X][AXIS_G].tileSize_;
            break;
        case KEY_ROBIN_GB:
            xBufferSize_    = axisInfo_[TENSOR_X][AXIS_G].tileSize_ * aBlockNum_ * BLOCK_SIZE;
            yBufferSize_    = axisInfo_[TENSOR_Y][AXIS_B].tileSize_ * pSize_ * gySize_ * aBlockNum_ * BLOCK_SIZE;
            idxBufferSize_  = axisInfo_[TENSOR_Y][AXIS_B].tileSize_ * giBlockNum_ * BLOCK_SIZE;
            ubLineLimit_    = axisInfo_[TENSOR_X][AXIS_G].tileSize_;
            break;
        case KEY_ROBIN_GG_U:
            xBufferSize_    = xAlignedUnitNum_ * lineAlignedUnit_ * aBlockNum_ * BLOCK_SIZE;
            yBufferSize_    = yAlignedUnitNum_ * lineAlignedUnit_ * aBlockNum_ * BLOCK_SIZE;
            idxBufferSize_  = idxAlignedBlockVolume_ * BLOCK_SIZE;
            ubLineLimit_    = axisInfo_[TENSOR_X][AXIS_G].tileSize_;
            break;
        case KEY_ROBIN_GP_U:
            xBufferSize_    = xAlignedUnitNum_ * lineAlignedUnit_ * aBlockNum_ * BLOCK_SIZE;
            yBufferSize_    = yAlignedUnitNum_ * lineAlignedUnit_ * aBlockNum_ * BLOCK_SIZE;
            idxBufferSize_  = idxAlignedBlockVolume_ * BLOCK_SIZE;
            ubLineLimit_    = axisInfo_[TENSOR_X][AXIS_G].tileSize_;
            break;
        case KEY_ROBIN_GB_U:
            xBufferSize_    = xAlignedUnitNum_ * lineAlignedUnit_ * aBlockNum_ * BLOCK_SIZE;
            yBufferSize_    = yAlignedUnitNum_ * lineAlignedUnit_ * aBlockNum_ * BLOCK_SIZE;
            idxBufferSize_  = idxAlignedBlockVolume_ * BLOCK_SIZE;
            ubLineLimit_    = axisInfo_[TENSOR_X][AXIS_G].tileSize_;
            break;
        case KEY_ONE_IN_N_P:
        case KEY_ONE_IN_N_B:
            if (IsVReduceFixPattern()) {
                xBufferSize_    = axisInfo_[TENSOR_Y][AXIS_B].tileSize_ * 
                                  CeilAlign(axisInfo_[TENSOR_Y][AXIS_P].tileSize_ * gxSize_ * xDtypeSize_, BLOCK_SIZE);
            } else {
                xBufferSize_    = axisInfo_[TENSOR_Y][AXIS_B].tileSize_ * 
                                  axisInfo_[TENSOR_Y][AXIS_P].tileSize_ * CeilAlign(gxSize_ * xDtypeSize_, BLOCK_SIZE);
            }
            yBufferSize_    = axisInfo_[TENSOR_Y][AXIS_B].tileSize_ * CeilAlign(axisInfo_[TENSOR_Y][AXIS_P].tileSize_ * xDtypeSize_, BLOCK_SIZE);
            idxBufferSize_  = CeilAlign(axisInfo_[TENSOR_Y][AXIS_B].tileSize_ * idxDtypeSize_, BLOCK_SIZE);
            ubLineLimit_    = axisInfo_[TENSOR_Y][AXIS_B].tileSize_ * axisInfo_[TENSOR_Y][AXIS_P].tileSize_;
            break;
        case KEY_VGATHER_GG:
            xBufferSize_    = CeilAlign(gxSize_ * xDtypeSize_, BLOCK_SIZE);
            yBufferSize_    = CeilAlign(axisInfo_[TENSOR_Y][AXIS_G].tileSize_ * idxDtypeSize_, BLOCK_SIZE);
            idxBufferSize_  = yBufferSize_;
            ubLineLimit_    = axisInfo_[TENSOR_Y][AXIS_G].tileSize_;
            break;
        case KEY_VGATHER_GP:
            xBufferSize_    = CeilAlign(gxSize_ * xDtypeSize_, BLOCK_SIZE);
            yBufferSize_    = axisInfo_[TENSOR_Y][AXIS_P].tileSize_ * CeilAlign(gySize_ * idxDtypeSize_, BLOCK_SIZE) * VGATHER_BUFF_EXPAND;
            idxBufferSize_  = yBufferSize_;
            ubLineLimit_    = 1;
            break;
        case KEY_VGATHER_GB:
            xBufferSize_    = CeilAlign(gxSize_ * xDtypeSize_, BLOCK_SIZE);
            yBufferSize_    = axisInfo_[TENSOR_Y][AXIS_B].tileSize_ * pSize_ * CeilAlign(gySize_ * idxDtypeSize_, BLOCK_SIZE) * VGATHER_BUFF_EXPAND;
            idxBufferSize_  = yBufferSize_;
            ubLineLimit_    = 1;
            break;
        case KEY_VGATHER_PG:
            xBufferSize_    = CeilAlign(axisInfo_[TENSOR_X][AXIS_P].tileSize_ * gxSize_ * xDtypeSize_, BLOCK_SIZE);
            yBufferSize_    = CeilAlign(axisInfo_[TENSOR_Y][AXIS_G].tileSize_ * idxDtypeSize_, BLOCK_SIZE);
            idxBufferSize_  = yBufferSize_;
            ubLineLimit_    = axisInfo_[TENSOR_Y][AXIS_G].tileSize_;
            break;
        case KEY_VGATHER_PP:
            xBufferSize_    = CeilAlign(axisInfo_[TENSOR_X][AXIS_P].tileSize_ * gxSize_ * xDtypeSize_, BLOCK_SIZE);
            yBufferSize_    = axisInfo_[TENSOR_Y][AXIS_P].tileSize_ * CeilAlign(gySize_ * idxDtypeSize_, BLOCK_SIZE) * VGATHER_BUFF_EXPAND;
            idxBufferSize_  = yBufferSize_;
            ubLineLimit_    = axisInfo_[TENSOR_X][AXIS_P].tileSize_;
            break;
        case KEY_VGATHER_PB:
            xBufferSize_    = CeilAlign(axisInfo_[TENSOR_X][AXIS_P].tileSize_ * gxSize_ * xDtypeSize_, BLOCK_SIZE);
            yBufferSize_    = axisInfo_[TENSOR_Y][AXIS_B].tileSize_ * pSize_ * CeilAlign(gySize_ * idxDtypeSize_, BLOCK_SIZE) * VGATHER_BUFF_EXPAND;
            idxBufferSize_  = yBufferSize_;
            ubLineLimit_    = axisInfo_[TENSOR_X][AXIS_P].tileSize_;
            break;
        case KEY_VGATHER_BG:
            xBufferSize_    = CeilAlign(axisInfo_[TENSOR_X][AXIS_B].tileSize_ * pSize_ * gxSize_ * xDtypeSize_, BLOCK_SIZE);
            yBufferSize_    = CeilAlign(axisInfo_[TENSOR_Y][AXIS_G].tileSize_ * idxDtypeSize_, BLOCK_SIZE);
            idxBufferSize_  = yBufferSize_;
            ubLineLimit_    = axisInfo_[TENSOR_Y][AXIS_G].tileSize_;
            break;
        case KEY_VGATHER_SCALAR:
        case KEY_VGATHER_BG_SP1:
            xBufferSize_    = CeilAlign(gxSize_ * xDtypeSize_, BLOCK_SIZE);
            yBufferSize_    = CeilAlign(ubLineLimit_ * idxDtypeSize_, BLOCK_SIZE); // ubLineLimit_在tiling过程中设置好
            idxBufferSize_  = yBufferSize_;
            break;    
        case KEY_VGATHER_BP:
            xBufferSize_    = CeilAlign(axisInfo_[TENSOR_X][AXIS_B].tileSize_ * pSize_ * gxSize_ * xDtypeSize_, BLOCK_SIZE);
            yBufferSize_    = axisInfo_[TENSOR_Y][AXIS_P].tileSize_ * CeilAlign(gySize_ * idxDtypeSize_, BLOCK_SIZE);
            idxBufferSize_  = yBufferSize_;
            ubLineLimit_    = axisInfo_[TENSOR_Y][AXIS_P].tileSize_;
            break;
        case KEY_VGATHER_BB:
            xBufferSize_    = CeilAlign(axisInfo_[TENSOR_X][AXIS_B].tileSize_ * pSize_ * gxSize_ * xDtypeSize_, BLOCK_SIZE);
            yBufferSize_    = axisInfo_[TENSOR_Y][AXIS_B].tileSize_ * pSize_ * CeilAlign(gySize_ * idxDtypeSize_, BLOCK_SIZE) * VGATHER_BUFF_EXPAND;
            idxBufferSize_  = yBufferSize_;
            ubLineLimit_    = axisInfo_[TENSOR_X][AXIS_B].tileSize_ * pSize_;
            break;
        case KEY_MOVE_PP_U:
            xBufferSize_    = axisInfo_[TENSOR_Y][AXIS_P].tileSize_ * gySize_ * aBlockNum_ * BLOCK_SIZE;
            yBufferSize_    = xBufferSize_;
            idxBufferSize_  = CeilAlign(gySize_ * idxDtypeSize_, BLOCK_SIZE);
            ubLineLimit_    = axisInfo_[TENSOR_Y][AXIS_P].tileSize_;
            break;
        case KEY_MOVE_GG_U:
            xBufferSize_    = axisInfo_[TENSOR_Y][AXIS_G].tileSize_ * aBlockNum_ * BLOCK_SIZE;
            yBufferSize_    = axisInfo_[TENSOR_Y][AXIS_G].tileSize_ * aBlockNum_ * BLOCK_SIZE;
            idxBufferSize_  = CeilAlign(axisInfo_[TENSOR_Y][AXIS_G].tileSize_ * idxDtypeSize_, BLOCK_SIZE);
            ubLineLimit_    = axisInfo_[TENSOR_Y][AXIS_G].tileSize_;
            break;
        default:
            break;
    }
}

int64_t GatherV3Tiling::GetVolumeOfAxisA(int64_t remainBlock) {
    int64_t aLineVolume = remainBlock / aBlockNum_;
    while (aLineVolume > 0) {
        int64_t idxVolume = (remainBlock - aLineVolume * aBlockNum_) * idxNumInBlock_;
        if (idxVolume >= aLineVolume) {
            break;
        }
        aLineVolume--;
    }

    return aLineVolume;
}

void GatherV3Tiling::CalcAllCacheTmplOutputTiling(int64_t remainBlock) {
    int64_t bElemVolume = remainBlock / (pSize_ * gySize_ * aBlockNum_ + giBlockNum_);
    if (bElemVolume >= 1 && IsApt2FullSplit(bSize_)) {
        tilingKey_ += Y_SPLIT_IN_B;
        AxisSquareSplit(bElemVolume, realCoreNum_, TENSOR_Y, AXIS_B);
    } else if (gySize_ * aBlockNum_ + giBlockNum_ <= remainBlock && IsApt2FullSplit(bSize_ * pSize_)) {
        tilingKey_ += Y_SPLIT_IN_P;
        int64_t pElemVolume = (remainBlock - giBlockNum_) / (gySize_ * aBlockNum_);
        AxisFullSplit(TENSOR_Y, AXIS_B);
        AxisCoSplit(pElemVolume, bSize_, TENSOR_Y, AXIS_P);
    } else {
        tilingKey_ += Y_SPLIT_IN_G;
        int64_t aLineVolume = GetVolumeOfAxisA(remainBlock);
        AxisFullSplit(TENSOR_Y, AXIS_B);
        AxisFullSplit(TENSOR_Y, AXIS_P);
        AxisCoSplit(aLineVolume, bSize_ * pSize_, TENSOR_Y, AXIS_G);
    }

    if (tilingKey_ == KEY_CACHE_BG && bSize_ == 1 && pSize_ == 1) {
        tilingKey_ = KEY_CACHE_BG_SP1;
    }
}

bool GatherV3Tiling::ProcessAllCacheShortTmpl() {
    if (bSize_ * pSize_ > 1) {
        return false;
    }

    if (aBlockNum_ > SCALAR_MOVE_THRE) {
        return false;
    }

    if (aSize_ * xDtypeSize_ % BLOCK_SIZE == 0) {
        return false;
    }

    int xBlockNum = CeilDiv(gxSize_ * aSize_ * xDtypeSize_, BLOCK_SIZE);
    if (xBlockNum > ubTotalBlockNum_ / UB_BUF_CNT) {
        return false;
    }

    int64_t aLineNum = GetVolumeOfAxisA(ubTotalBlockNum_ - xBlockNum);
    tilingKey_ = KEY_CACHE_BG_SP2;
    
    AxisFullSplit(TENSOR_Y, AXIS_B);
    AxisFullSplit(TENSOR_Y, AXIS_P);
    if (gySize_ < realCoreNum_ && aLineNum >= 1) {
        AxisFullSplit(TENSOR_Y, AXIS_G);
        return true;
    }

    if (realCoreNum_ * aLineNum >= gySize_) {
        AxisSlotSplit(realCoreNum_, TENSOR_Y, AXIS_G);
        return true;
    }

    doubleBuffer_ = true;
    AxisSplit(aLineNum / UB_BUF_CNT, TENSOR_Y, AXIS_G);
    return true;
}

void GatherV3Tiling::ProcessAllCacheTmpl() {
    // 先确定input切分，再确定output切分，UB中比例各占一半
    int64_t ubBlockVolume = ubTotalBlockNum_ / UB_BUF_CNT;
    int64_t gaSize = gxSize_ * aBlockNum_;
    int64_t gaLineNum = ubBlockVolume / gaSize;

    // 如果输入ga至少能放下两个，则开启double buffer
    if (gaLineNum >= UB_BUF_CNT) {
        ubBlockVolume /= UB_BUF_CNT;
        gaLineNum /= UB_BUF_CNT;
        EnableDoubleBuffer();
    }

    // 快速路径
    if (bSize_ * pSize_ == 1) {
        tilingKey_ = KEY_CACHE_BG_SP1;

        int64_t aLineNum = GetVolumeOfAxisA(ubTotalBlockNum_ - gaSize);
        AxisFullSplit(TENSOR_Y, AXIS_B);
        AxisFullSplit(TENSOR_Y, AXIS_P);
        AxisSquareSplit(aLineNum, realCoreNum_, TENSOR_Y, AXIS_G);
        return;
    }

    tilingKey_ = KEY_BASE_CACHE;

    if (gaLineNum >= pSize_ && IsApt2FullSplit(bSize_)) {
        int64_t bElemSize = pSize_ * gaSize;
        int64_t bElemVolume = ubBlockVolume / bElemSize;

        AxisSquareSplit(bElemVolume, realCoreNum_, TENSOR_X, AXIS_B);

        // 输入在b轴做切分
        tilingKey_ += X_SPLIT_IN_B;

        CalcAllCacheTmplOutputTiling(ubTotalBlockNum_ - axisInfo_[TENSOR_X][AXIS_B].tileSize_ * bElemSize);
        return;
    }

    // 输入在p轴做切分
    if (gaLineNum > 1 && IsApt2FullSplit(bSize_ * pSize_)) {
        tilingKey_ += X_SPLIT_IN_P;
        AxisFullSplit(TENSOR_X, AXIS_B);
        AxisCoSplit(gaLineNum, bSize_, TENSOR_X, AXIS_P);
        CalcAllCacheTmplOutputTiling(ubTotalBlockNum_ - axisInfo_[TENSOR_X][AXIS_P].tileSize_ * gaSize);
        return;
    }

    // gx轴完整切分 走到这里一个gx轴肯定可以放下
    tilingKey_ += X_SPLIT_IN_G;
    AxisFullSplit(TENSOR_X, AXIS_B);
    AxisFullSplit(TENSOR_X, AXIS_P);
    CalcAllCacheTmplOutputTiling(ubTotalBlockNum_ - gaSize);
    return;
}

void GatherV3Tiling::CalcAllCacheTmplUnalignOutputTiling(int64_t yLineVolume, int64_t idxBlockVolume) {
    int64_t bElemVolume = yLineVolume / (pSize_ * gySize_);
    int64_t idxVolume = idxBlockVolume * idxNumInBlock_;

    if (bElemVolume >= 1 && IsApt2FullSplit(bSize_)) {
        for (int64_t bElemLimit = min(bSize_, bElemVolume); bElemLimit > 0; bElemLimit--) {
            AxisSplit(bElemLimit, TENSOR_Y, AXIS_B);
            if (idxVolume > (axisInfo_[TENSOR_Y][AXIS_B].tileSize_ * gySize_)) {
                AxisSquareSplit(bElemLimit, realCoreNum_, TENSOR_Y, AXIS_B);
                tilingKey_ += Y_SPLIT_IN_B;
                return;
            }
        }
    }
    
    if (idxVolume > gySize_) {
        int64_t pElemVolume = min((yLineVolume / gySize_), pSize_);
        if (pElemVolume >= 1) {
            tilingKey_ += Y_SPLIT_IN_P;
            AxisFullSplit(TENSOR_Y, AXIS_B);
            AxisCoSplit(pElemVolume, bSize_, TENSOR_Y, AXIS_P);
            return;
        }
    }

    int64_t gElemVolume = min(yLineVolume, idxVolume);

    tilingKey_ += Y_SPLIT_IN_G;
    AxisFullSplit(TENSOR_Y, AXIS_B);
    AxisFullSplit(TENSOR_Y, AXIS_P);
    AxisCoSplit(gElemVolume, bSize_ * pSize_, TENSOR_Y, AXIS_G);
}

bool GatherV3Tiling::ProcessAllCacheUnalignTmpl() {
    lineAlignedUnit_ = ALIGN_SIZE_FOR_2B;
    if (xDtypeSize_ == 1) {
        lineAlignedUnit_ = ALIGN_SIZE_FOR_1B;
    }

    // 双buffer流水
    EnableDoubleBuffer();

    // 对齐后计算x, y, idx的容量，如果unit奇数，优先x多用一块
    int64_t aLineVolume = ubTotalBlockNum_ / aBlockNum_;

    int64_t unitNum = aLineVolume / lineAlignedUnit_;
    if (unitNum < UB_BUF_CNT) {
        return false;
    }

    if (unitNum * lineAlignedUnit_ * aBlockNum_ == ubTotalBlockNum_) {
        if (unitNum > UB_BUF_CNT) {
            unitNum--;
        } else {
            return false;
        }
    }

    yAlignedUnitNum_ = unitNum / UB_BUF_CNT;
    xAlignedUnitNum_ = unitNum - yAlignedUnitNum_;
    idxAlignedBlockVolume_ = ubTotalBlockNum_ - unitNum * lineAlignedUnit_ * aBlockNum_;
    
    int64_t xLineVolume = xAlignedUnitNum_ * lineAlignedUnit_;
    int64_t yLineVolume = yAlignedUnitNum_ * lineAlignedUnit_;
    if (idxAlignedBlockVolume_ < 1) {
        return false;
    }

    int64_t gaLineNum = xLineVolume / gxSize_;
    if (gaLineNum < 1) {
        return false;
    }

    if (bSize_ * pSize_ == 1) {
        xAlignedUnitNum_ = CeilDiv(gxSize_, lineAlignedUnit_);
        yAlignedUnitNum_ = unitNum - xAlignedUnitNum_;
        yLineVolume = min(yAlignedUnitNum_ * lineAlignedUnit_, idxAlignedBlockVolume_ * idxNumInBlock_);

        tilingKey_ = KEY_CACHE_BG_U_SP1;
        AxisFullSplit(TENSOR_Y, AXIS_B);
        AxisFullSplit(TENSOR_Y, AXIS_P);
        if (gySize_ < realCoreNum_) {
            AxisFullSplit(TENSOR_Y, AXIS_G);
            return true;
        }

        if (gySize_ <= realCoreNum_ * yLineVolume) {
            AxisSlotSplit(realCoreNum_, TENSOR_Y, AXIS_G);
            return true;
        }

        AxisSplit(yLineVolume, TENSOR_Y, AXIS_G);
        return true;
    }

    tilingKey_ = KEY_BASE_CACHE_UNALIGN;

    if (gaLineNum >= pSize_ && IsApt2FullSplit(bSize_)) {
        // 输入在b轴做切分
        tilingKey_ += X_SPLIT_IN_B;
        int64_t bElemVolume = gaLineNum / pSize_;

        AxisSquareSplit(bElemVolume, realCoreNum_, TENSOR_X, AXIS_B);

        // 依据实际切分值再对齐计算
        int64_t bTileSize = axisInfo_[TENSOR_X][AXIS_B].tileSize_;
        xAlignedUnitNum_ = CeilDiv(bTileSize * pSize_ * gxSize_, lineAlignedUnit_);
        yAlignedUnitNum_ = unitNum - xAlignedUnitNum_;
        if (yAlignedUnitNum_ < 1) {
            return false;
        }

        CalcAllCacheTmplUnalignOutputTiling(yAlignedUnitNum_ * lineAlignedUnit_, idxAlignedBlockVolume_);
        return true;
    }

    // gx轴完整切分
    if (gaLineNum == 1) {
        tilingKey_ += X_SPLIT_IN_G;
        AxisFullSplit(TENSOR_X, AXIS_B);
        AxisFullSplit(TENSOR_X, AXIS_P);
        CalcAllCacheTmplUnalignOutputTiling(yLineVolume, idxAlignedBlockVolume_);
        return true;
    }

    // 输入在p轴做切分
    tilingKey_ += X_SPLIT_IN_P;
    AxisFullSplit(TENSOR_X, AXIS_B);
    AxisSplit(gaLineNum, TENSOR_X, AXIS_P);

    // 依据实际切分值再对齐计算
    int64_t pTileSize = axisInfo_[TENSOR_X][AXIS_P].tileSize_;
    xAlignedUnitNum_ = CeilDiv(pTileSize * gxSize_, lineAlignedUnit_);
    yAlignedUnitNum_ = unitNum - xAlignedUnitNum_;
    if (yAlignedUnitNum_ < 1) {
        return false;
    }   

    CalcAllCacheTmplUnalignOutputTiling(yAlignedUnitNum_ * lineAlignedUnit_, idxAlignedBlockVolume_);
    return true;
}

bool GatherV3Tiling::CalcRobinTmplInputTiling(int64_t remainBlock) {
    int64_t aLineNum = remainBlock / aBlockNum_;
    if (aLineNum < 1) {
        return false;
    }

    AxisFullSplit(TENSOR_X, AXIS_B);
    AxisFullSplit(TENSOR_X, AXIS_P);
    AxisSplit(aLineNum, TENSOR_X, AXIS_G);

    if (axisInfo_[TENSOR_X][AXIS_G].tileNum_ > ROBIN_PAGE_THRE) {
        return false;
    }
    return true;
}

bool GatherV3Tiling::ProcessRobinTmplMultiCores() {
    if (bSize_ * pSize_ == 1 && (aSize_ * xDtypeSize_) % BLOCK_SIZE == 0) {
        int64_t yPerCoreNum = CeilDiv(gySize_, realCoreNum_);
        if (yPerCoreNum < ROBIN_GYSIZE_THRE) {
            return false;
        }

        int64_t blockNum = yPerCoreNum * aBlockNum_ + 
                           CeilDiv(yPerCoreNum * idxDtypeSize_, BLOCK_SIZE) + 
                           CeilAlign(yPerCoreNum * sizeof(float), FULL_BLOCK_SIZE) / BLOCK_SIZE + 
                           CeilAlign(yPerCoreNum * sizeof(uint8_t), BLOCK_SIZE) / BLOCK_SIZE * 2 +
                           CeilDiv(yPerCoreNum * sizeof(int32_t), BLOCK_SIZE) * 3;

        int64_t remainBlockNum = (ubTotalBlockNum_ - blockNum) / UB_BUF_CNT; // x开双buffer
        if (remainBlockNum <= 0) {
            return false;
        }

        int64_t xInCoreNum = remainBlockNum / aBlockNum_;
        if (xInCoreNum <= 0) {
            return false;
        }

        int64_t loopCnt = CeilDiv(gxSize_, xInCoreNum);
        if (loopCnt <= ROBIN_PAGE_THRE) {
            tilingKey_ = KEY_ROBIN_GG_SP1;
            AxisFullSplit(TENSOR_X, AXIS_B);
            AxisFullSplit(TENSOR_X, AXIS_P);
            AxisSlotSplit(loopCnt, TENSOR_X, AXIS_G);

            AxisFullSplit(TENSOR_Y, AXIS_B);
            AxisFullSplit(TENSOR_Y, AXIS_P);
            AxisSlotSplit(realCoreNum_, TENSOR_Y, AXIS_G);

            return true;
        }
    }

    return false;
}

bool GatherV3Tiling::ProcessRobinTmpl() {
    if (ProcessRobinTmplMultiCores()) {
        return true;
    }

    if (aSize_ * xDtypeSize_ % BLOCK_SIZE != 0) {
        return false;
    }

    tilingKey_ = KEY_BASE_ROBIN;

    int64_t gLineBlockNum = gySize_ * aBlockNum_;
    int64_t ubBlockVolume = ubTotalBlockNum_ / UB_BUF_CNT;

    int64_t pLineNum = (ubBlockVolume - giBlockNum_) / gLineBlockNum;
    if (pLineNum < 1) {
        return false;
    }

    if (pLineNum > 1) {
        // 不止放下一个的时候，可以开启双buffer。
        EnableDoubleBuffer();
        ubBlockVolume /= UB_BUF_CNT;
        pLineNum /= UB_BUF_CNT;
    }

    if (pLineNum == 1) {
        tilingKey_ += Y_SPLIT_IN_G;
        AxisFullSplit(TENSOR_Y, AXIS_B);
        AxisFullSplit(TENSOR_Y, AXIS_P);

        int64_t remainBlock = ubTotalBlockNum_ - giBlockNum_ - gySize_ * aBlockNum_;
        return CalcRobinTmplInputTiling(remainBlock);
    }

    if (pLineNum < pSize_) {
        tilingKey_ += Y_SPLIT_IN_P;
        AxisFullSplit(TENSOR_Y, AXIS_B);
        AxisSplit(pLineNum, TENSOR_Y, AXIS_P);

        int64_t remainBlock = ubTotalBlockNum_ - giBlockNum_ - axisInfo_[TENSOR_Y][AXIS_P].tileSize_ * gySize_ * aBlockNum_;
        return CalcRobinTmplInputTiling(remainBlock);
    }

    tilingKey_ += Y_SPLIT_IN_B;
    int64_t pLineBlockNum = pSize_ * gySize_ * aBlockNum_ + giBlockNum_;
    int64_t bLineNum = ubBlockVolume / pLineBlockNum;
    if (bLineNum < 1) {
        return false;
    }
    AxisSplit(bLineNum, TENSOR_Y, AXIS_B);
    int64_t remainBlock = ubTotalBlockNum_ - axisInfo_[TENSOR_Y][AXIS_B].tileSize_ * (pSize_ * gySize_ * aBlockNum_ + giBlockNum_);
    return CalcRobinTmplInputTiling(remainBlock);
}

bool GatherV3Tiling::CalcRobinTmplUnalignInputTiling() {
    if (xAlignedUnitNum_ < 1) {
        return false;
    }

    int64_t xLineVolume = xAlignedUnitNum_ * lineAlignedUnit_;

    AxisFullSplit(TENSOR_X, AXIS_B);
    AxisFullSplit(TENSOR_X, AXIS_P);
    AxisSplit(xLineVolume, TENSOR_X, AXIS_G);

    return axisInfo_[TENSOR_X][AXIS_G].tileNum_ < ROBIN_PAGE_THRE;
}

bool GatherV3Tiling::ProcessRobinUnalignTmpl() {
    tilingKey_ = KEY_BASE_ROBIN_UNALIGN;

    lineAlignedUnit_ = ALIGN_SIZE_FOR_2B;
    if (xDtypeSize_ == 1) {
        lineAlignedUnit_ = ALIGN_SIZE_FOR_1B;
    }

    // 双buffer流水
    EnableDoubleBuffer();

    // 对齐后计算x, y, idx的容量，如果unit奇数，优先y多用一块
    int64_t aLineVolume = ubTotalBlockNum_ / aBlockNum_;

    int64_t unitNum = aLineVolume / lineAlignedUnit_;
    if (unitNum < UB_BUF_CNT) {
        return false;
    }

    // 给索引预留空间
    if (unitNum * lineAlignedUnit_ * aBlockNum_ == ubTotalBlockNum_) {
        if (unitNum > UB_BUF_CNT) {
            unitNum--;
        } else {
            return false;
        }
    }

    xAlignedUnitNum_ = unitNum / UB_BUF_CNT;
    yAlignedUnitNum_ = unitNum - yAlignedUnitNum_;
    idxAlignedBlockVolume_ = ubTotalBlockNum_ - unitNum * lineAlignedUnit_ * aBlockNum_;
    
    int64_t yLineVolume = yAlignedUnitNum_ * lineAlignedUnit_;
    if (idxAlignedBlockVolume_ < 1) {
        return false;
    }

    int64_t gaLineNum = yLineVolume / gySize_;
    if (gaLineNum < 1) {
        return false;
    }

    // gy轴完整切分
    if (gaLineNum == 1) {
        if (idxAlignedBlockVolume_ * BLOCK_SIZE < gySize_ * idxDtypeSize_) {
            return false;
        }

        tilingKey_ += Y_SPLIT_IN_G;
        AxisFullSplit(TENSOR_Y, AXIS_B);
        AxisFullSplit(TENSOR_Y, AXIS_P);
        return CalcRobinTmplUnalignInputTiling();
    }

    // 输入在p轴做切分
    if (gaLineNum < pSize_) {
        tilingKey_ += Y_SPLIT_IN_P;
        AxisFullSplit(TENSOR_Y, AXIS_B);
        AxisCoSplit(gaLineNum, bSize_, TENSOR_Y, AXIS_P);

        // 依据实际切分值再对齐计算
        int64_t pTileSize = axisInfo_[TENSOR_Y][AXIS_P].tileSize_;
        if (idxAlignedBlockVolume_ * BLOCK_SIZE < pTileSize * gySize_ * idxDtypeSize_) {
            return false;
        }

        yAlignedUnitNum_ = CeilDiv(pTileSize * gySize_, lineAlignedUnit_);
        xAlignedUnitNum_ = unitNum - yAlignedUnitNum_;
        return CalcRobinTmplUnalignInputTiling();
    }

    // 输入在b轴做切分
    tilingKey_ += Y_SPLIT_IN_B;
    int64_t bElemVolume = gaLineNum / pSize_;
    AxisSquareSplit(bElemVolume, realCoreNum_, TENSOR_Y, AXIS_B);

    // 依据实际切分值再对齐计算
    int64_t bTileSize = axisInfo_[TENSOR_Y][AXIS_B].tileSize_;
    if (idxAlignedBlockVolume_ * BLOCK_SIZE < bTileSize * gySize_ * idxDtypeSize_) {
        return false;
    }

    yAlignedUnitNum_ = CeilDiv(bTileSize * pSize_ * gySize_, lineAlignedUnit_);
    xAlignedUnitNum_ = unitNum - yAlignedUnitNum_;
    return CalcRobinTmplUnalignInputTiling();
}

bool GatherV3Tiling::ProcessOneInTwoOrFour() {
    tilingKey_ = KEY_BASE_ONE_IN_N;

    int64_t pLineBlockNum = CeilDiv(pSize_ * gxSize_ * xDtypeSize_, BLOCK_SIZE) + 
                            CeilDiv(pSize_ * xDtypeSize_, BLOCK_SIZE);

    int64_t pLineNum = ubTotalBlockNum_ / pLineBlockNum;
    if (pLineNum >= 1 && IsApt2FullSplit(bSize_)) {
        while (pLineNum > 0) {
            AxisSquareSplit(pLineNum, realCoreNum_, TENSOR_Y, AXIS_B);
            int64_t tileSize = axisInfo_[TENSOR_Y][AXIS_B].tileSize_;
            if (tileSize * pLineBlockNum + CeilDiv(tileSize * idxDtypeSize_, BLOCK_SIZE) <= ubTotalBlockNum_) {
                tilingKey_ += Y_SPLIT_IN_B;
                return true;
            }
            pLineNum--;
        }
    }

    AxisFullSplit(TENSOR_Y, AXIS_B);
    int64_t gLineNum = ubTotalBlockNum_ * BLOCK_SIZE / ((gxSize_ + 1) * xDtypeSize_);
    while (gLineNum > 0) {
        AxisCoSplit(gLineNum, bSize_, TENSOR_Y, AXIS_P);
        int64_t tileSize = axisInfo_[TENSOR_Y][AXIS_P].tileSize_;

        // x blocknum + y blocknum + 1block for idx
        if (CeilDiv(tileSize * gxSize_ * xDtypeSize_, BLOCK_SIZE) + 
            CeilDiv(tileSize * xDtypeSize_, BLOCK_SIZE) + 1 <= ubTotalBlockNum_) {
            tilingKey_ += Y_SPLIT_IN_P;
            return true;
        }

        gLineNum--;
    }

    return false;
}

void GatherV3Tiling::ProcessOneInNTmpl() {
    // 2选1或4选1场景，尝试更加紧凑的tiling
    if (IsVReduceFixPattern() && ProcessOneInTwoOrFour()) {
        if (axisInfo_[TENSOR_Y][AXIS_B].tileNum_ * axisInfo_[TENSOR_Y][AXIS_P].tileNum_ > realCoreNum_) {
            EnableDoubleBuffer();
            if (ProcessOneInTwoOrFour()) {
                return;
            }
            DisableDoubleBuffer();
        }
    }

    tilingKey_ = KEY_BASE_ONE_IN_N;

    EnableDoubleBuffer();

    constexpr int64_t MASK_BLOCK_NUM = 1;
    constexpr int64_t IDX_BLOCK_NUM = 1;

    // 输入 + 输出 + idx
    int64_t pLineBlockNum = pSize_ * CeilDiv(gxSize_ * xDtypeSize_, BLOCK_SIZE) + 
                            CeilDiv(pSize_ * xDtypeSize_, BLOCK_SIZE) + IDX_BLOCK_NUM;

    int64_t pLineNum = (ubTotalBlockNum_ - MASK_BLOCK_NUM) / pLineBlockNum;

    if (pLineNum >= 1 && IsApt2FullSplit(bSize_)) {
        tilingKey_ += Y_SPLIT_IN_B;
        AxisSquareSplit(pLineNum, realCoreNum_, TENSOR_Y, AXIS_B);
        return;
    }

    tilingKey_ += Y_SPLIT_IN_P;
    int64_t gLineBlockNum = CeilDiv(gxSize_ * xDtypeSize_, BLOCK_SIZE) + 1; // 输入 + 输出

    int64_t gLineNum = (ubTotalBlockNum_ - MASK_BLOCK_NUM - IDX_BLOCK_NUM) / gLineBlockNum; // 扣除一个mask 一个索引block
    
    AxisFullSplit(TENSOR_Y, AXIS_B);
    AxisCoSplit(gLineNum, bSize_, TENSOR_Y, AXIS_P);
}

void GatherV3Tiling::ProcessDirectGatherTmplOutputTiling(int64_t remainByte) {
    int64_t blockVolume = (remainByte / BLOCK_SIZE) / UB_BUF_CNT;
    int64_t gLineBlockNum = CeilDiv(gySize_ * idxDtypeSize_, BLOCK_SIZE) * UB_BUF_CNT;

    int64_t gLineNum = blockVolume / gLineBlockNum;

    if (gLineNum >= pSize_ && IsApt2FullSplit(bSize_)) {
        tilingKey_ += Y_SPLIT_IN_B;
        int64_t pLineNum = gLineNum / pSize_;
        AxisSquareSplit(pLineNum, realCoreNum_, TENSOR_Y, AXIS_B);
        return;
    }

    if (gLineNum >= 1) {
        tilingKey_ += Y_SPLIT_IN_P;
        AxisFullSplit(TENSOR_Y, AXIS_B);
        AxisCoSplit(gLineNum, bSize_, TENSOR_Y, AXIS_P);
        return;
    }

    tilingKey_ += Y_SPLIT_IN_G;
    AxisFullSplit(TENSOR_Y, AXIS_B);
    AxisFullSplit(TENSOR_Y, AXIS_P);

    int64_t elemNum = blockVolume * BLOCK_SIZE / idxDtypeSize_;
    AxisCoSplit(elemNum, bSize_ * pSize_, TENSOR_Y, AXIS_G);
    return;
}

void GatherV3Tiling::ProcessDirectGatherTmpl() {
    tilingKey_ = KEY_BASE_VGATHER;
    ubTotalBlockNum_ -= RSV_BLOCK;

    int64_t gLineSize = gxSize_ * xDtypeSize_;
    int64_t outSize = bSize_ * pSize_;

    if (outSize <= realCoreNum_) {
        int64_t remainBlock = ubTotalBlockNum_ - CeilDiv(gLineSize, BLOCK_SIZE);
        int64_t blockVolume = (remainBlock) / UB_BUF_CNT; // y和idx各占一半
        int64_t elemNum = blockVolume * BLOCK_SIZE / idxDtypeSize_;
        if (elemNum < gySize_) {
            EnableDoubleBuffer();
            blockVolume /= UB_BUF_CNT;
            elemNum = blockVolume * BLOCK_SIZE / idxDtypeSize_;
        }
        int64_t slot = realCoreNum_ / outSize;
        AxisFullSplit(TENSOR_Y, AXIS_B);
        AxisFullSplit(TENSOR_Y, AXIS_P);
        AxisSlotSplit(slot, TENSOR_Y, AXIS_G);
        ubLineLimit_ = min(elemNum, axisInfo_[TENSOR_Y][AXIS_G].tileSize_);

        tilingKey_ = KEY_VGATHER_BG_SP1;
        return;
    }

    if (gLineSize * UB_BUF_CNT <= ubTotalBlockNum_ * BLOCK_SIZE / UB_BUF_CNT) {
        EnableDoubleBuffer();
    }

    int64_t ubByteVolumeForInput = ubTotalBlockNum_ * BLOCK_SIZE / UB_BUF_CNT;

    int64_t gLineNum = ubByteVolumeForInput / gLineSize;

    if (gLineNum > pSize_ && IsApt2FullSplit(bSize_)) {
        tilingKey_ += X_SPLIT_IN_B;
        int64_t pLineNum = gLineNum / pSize_;
        AxisSplit(pLineNum, TENSOR_X, AXIS_B);

        int64_t remainByte = ubTotalBlockNum_ * BLOCK_SIZE - axisInfo_[TENSOR_X][AXIS_B].tileSize_ * pSize_ * gLineSize;
        ProcessDirectGatherTmplOutputTiling(remainByte);
        return;
    }

    if (gLineNum > 1) {
        tilingKey_ += X_SPLIT_IN_P;
        AxisFullSplit(TENSOR_X, AXIS_B);
        AxisCoSplit(gLineNum, bSize_, TENSOR_X, AXIS_P);

        int64_t remainByte = ubTotalBlockNum_ * BLOCK_SIZE - axisInfo_[TENSOR_X][AXIS_P].tileSize_ * gLineSize;
        ProcessDirectGatherTmplOutputTiling(remainByte);
        return;
    }

    tilingKey_ += X_SPLIT_IN_G;
    AxisFullSplit(TENSOR_X, AXIS_B);
    AxisFullSplit(TENSOR_X, AXIS_P);

    int64_t remainByte = ubTotalBlockNum_ * BLOCK_SIZE - gLineSize;
    ProcessDirectGatherTmplOutputTiling(remainByte);
    return;
}

void GatherV3Tiling::ProcessTiling() {
    if (xShape_.GetDimNum() == 0) {
        ProcessScalarX();
        return;
    }

    // 纯搬运模板 
    if (bSize_ * pSize_ * gySize_ <= realCoreNum_ * GYSIZE_CACHE_THRE) {
        ProcessAllMoveTmpl();
        AdjustAllMoveCoreTiling();
        return;
    }

    // 核内gather
    if (aSize_ == 1 && (xDtypeSize_ == sizeof(int16_t) || xDtypeSize_ == sizeof(int32_t))) {
        if (gySize_ == 1 && gxSize_ * xDtypeSize_ <= ONE_IN_N_G_AXIS_LEN_THRE && pSize_ > ONE_IN_N_P_AXIS_LEN_THRE) {
            ProcessOneInNTmpl();
            return;
        }
    
        if (gxSize_ / gySize_ <= IO_RATIO_THRE && gxSize_ * xDtypeSize_ <= ubTotalBlockNum_ * BLOCK_SIZE / UB_BUF_CNT) {
            ProcessDirectGatherTmpl();
            return;
        }
    }

    if (gxSize_ / gySize_ <= IO_RATIO_THRE) {
        if (ProcessAllCacheShortTmpl()) {
            return;
        }

        // 非对齐场景
        if ((aSize_ * xDtypeSize_) % BLOCK_SIZE != 0) {
            if (ProcessAllCacheUnalignTmpl()) {
                if (tilingKey_ == KEY_CACHE_BG_U && bSize_ == 1 && pSize_ == 1) {
                    tilingKey_ = KEY_CACHE_BG_U_SP1;
                }
                return;
            }

            DisableDoubleBuffer();

            if (ProcessRobinUnalignTmpl()) {
                return;
            }

            DisableDoubleBuffer();
        }

        // 输入的一个gx * a可全部放入UB 是否开启double buffer在内部判断
        if (gxSize_ * aBlockNum_ <= ubTotalBlockNum_ / UB_BUF_CNT) {
            ProcessAllCacheTmpl();
            return;
        }

        // 尝试对输入进行切分，输出g轴全载。
        if (ProcessRobinTmpl()) {
            return;
        }
    }

    // 纯搬运模板兜底
    ProcessAllMoveTmpl();
    AdjustAllMoveCoreTiling();
    return;
}

void GatherV3Tiling::FillTilingData() {
    OP_LOGD(nodeName_, "----->Tiling Key: %ld for (%ld, %ld, %ld, %ld, %ld) size (%ld %ld)", 
            tilingKey_, 
            bSize_, pSize_, gxSize_, gySize_, aSize_,
            xDtypeSize_, idxDtypeSize_);
    tilingData_.set_tilingKey(static_cast<int64_t>(tilingKey_));
    tilingData_.set_realCoreNum(realCoreNum_);
    tilingData_.set_bSize(bSize_);
    tilingData_.set_pSize(pSize_);
    tilingData_.set_gxSize(gxSize_);
    tilingData_.set_gySize(gySize_);
    tilingData_.set_aSize(aSize_);
    tilingData_.set_ubLineLimit(ubLineLimit_);
    tilingData_.set_xBufferSize(xBufferSize_);
    tilingData_.set_yBufferSize(yBufferSize_);
    tilingData_.set_idxBufferSize(idxBufferSize_);
    tilingData_.set_bufferNum(doubleBuffer_ ? UB_BUF_CNT : 1);

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), 
                             context_->GetRawTilingData()->GetCapacity());
                             context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
}

void GatherV3Tiling::PrintTilingData() {
    OP_LOGD(context_->GetNodeName(),  "start print tiling data             ");
    OP_LOGD(context_->GetNodeName(),  "tilingKey:            %ld", tilingData_.get_tilingKey());
    OP_LOGD(context_->GetNodeName(),  "realCoreNum:          %ld", tilingData_.get_realCoreNum());
    OP_LOGD(context_->GetNodeName(),  "ubLineLimit:          %ld", tilingData_.get_ubLineLimit());
    OP_LOGD(context_->GetNodeName(),  "xBufferSize:          %ld", tilingData_.get_xBufferSize());
    OP_LOGD(context_->GetNodeName(),  "yBufferSize:          %ld", tilingData_.get_yBufferSize());
    OP_LOGD(context_->GetNodeName(),  "idxBufferSize:        %ld", tilingData_.get_idxBufferSize());
    OP_LOGD(context_->GetNodeName(),  "bSize:                %ld", tilingData_.get_bSize());
    OP_LOGD(context_->GetNodeName(),  "pSize:                %ld", tilingData_.get_pSize());
    OP_LOGD(context_->GetNodeName(),  "gxSize:               %ld", tilingData_.get_gxSize());
    OP_LOGD(context_->GetNodeName(),  "gySize:               %ld", tilingData_.get_gySize()); 
    OP_LOGD(context_->GetNodeName(),  "aSize:                %ld", tilingData_.get_aSize());
    OP_LOGD(context_->GetNodeName(),  "bTileNum:             %ld", tilingData_.get_bTileNum());
    OP_LOGD(context_->GetNodeName(),  "pTileNum:             %ld", tilingData_.get_pTileNum());
    OP_LOGD(context_->GetNodeName(),  "gTileNum:             %ld", tilingData_.get_gTileNum());
    OP_LOGD(context_->GetNodeName(),  "aTileNum:             %ld", tilingData_.get_aTileNum());
    OP_LOGD(context_->GetNodeName(),  "bTileSize:            %ld", tilingData_.get_bTileSize());
    OP_LOGD(context_->GetNodeName(),  "pTileSize:            %ld", tilingData_.get_pTileSize());  
    OP_LOGD(context_->GetNodeName(),  "gTileSize:            %ld", tilingData_.get_gTileSize());  
    OP_LOGD(context_->GetNodeName(),  "aTileSize:            %ld", tilingData_.get_aTileSize());  
    OP_LOGD(context_->GetNodeName(),  "bTileHead:            %ld", tilingData_.get_bTileHead());  
    OP_LOGD(context_->GetNodeName(),  "pTileHead:            %ld", tilingData_.get_pTileHead());  
    OP_LOGD(context_->GetNodeName(),  "gTileHead:            %ld", tilingData_.get_gTileHead());  
    OP_LOGD(context_->GetNodeName(),  "aTileHead:            %ld", tilingData_.get_aTileHead());
    OP_LOGD(context_->GetNodeName(),  "bufferNum:            %ld", tilingData_.get_bufferNum());    
}

static ge::graphStatus Tiling4GatherV3(gert::TilingContext* context) {
    OP_LOGD(context->GetNodeName(), "GatherV3 tiling begin.");
    GatherV3Tiling tilingObject(context);
    OP_TILING_CHECK(tilingObject.RunTiling4GatherV3() != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "RunTiling4GatherV3 failed."),
                    return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "GatherV3 tiling end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GatherV3)
    .Tiling(Tiling4GatherV3)
    .TilingInputsDataDependency({AXIS_INDEX});
}  // namespace optiling
