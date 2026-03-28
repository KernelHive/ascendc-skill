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
 * \file swi_glu_grad_def.cpp
 * \brief tiling
 */
#include <chrono>
#include "register/op_impl_registry.h"
#include "swi_glu_grad_tiling.h"
#include "tiling/tiling_api.h"


namespace ops {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }
}  // namespace ops


namespace optiling {
#define unlikely(x) __builtin_expect((x), 0)
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
#define OP_LOGE_IF(condition, return_value, op_name, fmt, ...)                                                 \
  static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool"); \
  do {                                                                                                         \
    if (unlikely(condition)) {                                                                                 \
      OP_LOGE(op_name, fmt, ##__VA_ARGS__);                                                                    \
      return return_value;                                                                                     \
    }                                                                                                          \
  } while (0)

const uint32_t UB_RESERVED_BUFF = 0; // reserve 0k
const uint32_t L2_CACHE_LINE_SIZE = 512; // pack unit in cache 512B
const uint32_t UB_MIN_BLOCK_SIZE = 32; // align unit in cache 32B
const uint32_t BLOCK_SIZE_OF_64B = 64; // align unit in cache 64B
const uint32_t DEFAULT_BUFFER_NUM = 2;
const uint32_t MAX_BLOCK_COUNT = 4095; // datacopy指令包含的连续传输数据块的最大个数
const uint32_t MAX_BLOCK_LEN = 65535 * 32; // datacopy指令每个连续传输数据块的最长长度为65535，单位为32bytes
const uint32_t MAX_UINT32 = 4294967295;
const uint32_t MAX_CORE_NUMBER = 64;
const uint16_t DISCONTINE_COPY_MAX_BLOCKCNT = 4095; // 非连续拷贝，blockCount最大值,AscendC接口限制
const uint16_t DISCONTINE_COPY_MAX_BLOCKLEN = 65535; // 非连续拷贝，blockLen最大值,AscendC接口限制
const uint16_t DISCONTINE_COPY_MAX_STRIDE = 65535; // 非连续拷贝，srcStride/dstStride最大值,AscendC接口限制

const uint32_t XXGLU_TQUE_NUM = 3;
const uint32_t SWIGLU_TBUF_NUM_HALF = 2;
const uint32_t SWIGLU_TBUF_NUM_BF16 = 2;
const uint32_t SWIGLU_TBUF_NUM_FLOAT = 1;
const uint32_t XXGLU_BW_TQUE_NUM = 5;
const uint32_t SWIGLU_BW_TBUF_NUM_FLOAT = 2;
const uint32_t SWIGLU_BW_TBUF_NUM_BF16 = 5;
const uint32_t SWIGLU_BW_TBUF_NUM_HALF = 5;

template<typename T>
inline T AlignUp(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}
// align num to multiples of rnd, round down
template<typename T>
inline T AlignDown(T num, T rnd)
{
    return ((((rnd) == 0) || ((num) < (rnd))) ? 0 : ((num) / (rnd) * (rnd)));
}
template<typename T>
inline T DivCeil(T num, T div)
{
    return (((div) == 0) ? 0 : (((num) + (div) - 1) / (div)));
}

enum GLU_FLAG {
    SWIGLU_SINGLE,
    SWIGLU_GRAD_SINGLE
};

// Tiling优选参数
struct GluSingleTilingOptParam {
    // Maximum amount of data that can be transferred by an operator UB at a time. Unit:element
    uint32_t maxTileLen = 0;
    uint32_t optBaseRowLen = 0; // 最优的BaseRowLen
    uint32_t optBaseColLen = 0; // 最优的BaseColLen
    uint64_t optTotalTileNum = 0; // 最优的分割后的数据块数量
    uint64_t optBaseSize = 0; // 最优的分割后的base shape数据块的大小， optBaseRowLen*optBaseColLen, Unit:element
    uint64_t optBaseTileNum = 0; // 最优的分割后的base shape数据块数量，不包含尾块

    uint32_t totalUsedCoreNum = 0; // 最终实际使用的核数
    uint64_t tileNumPerCore = 0; // 每个核需要处理的TileNum，如果不均匀，按照多的计算
};

struct GluSingleTilingCalculator {
public:
    explicit GluSingleTilingCalculator(SwiGluTilingData *iTilingData) : tilingData(iTilingData) {}

    template<GLU_FLAG Glu_Flag>
    bool CalcTiling(uint32_t totalCore, uint64_t ubSize, int32_t dtype, platform_ascendc::SocVersion socVersion_);
    
    template <GLU_FLAG Glu_Flag>
    bool SetTotalShape(const gert::Shape &inShape, const int32_t inDim);

    void SaveTilingData(gert::TilingContext *context);
    inline bool isSupportSocV(uint32_t dtype, platform_ascendc::SocVersion socVersion_);
    SwiGluTilingData *tilingData;

    uint32_t totalUsedCoreNum = 0; // 最终实际使用的核数
private:
    template<GLU_FLAG Glu_Flag, uint16_t bufferNum>
    bool CalcOptTiling(uint64_t ubSize, int32_t dtype, GluSingleTilingOptParam &optTiling);

    template<GLU_FLAG Glu_Flag, uint16_t bufferNum>
    bool GetBufferNumAndDataLenPerUB(uint64_t ubSize, int32_t dtype, uint64_t &dataLenPerUB);

    template<GLU_FLAG Glu_Flag, uint16_t bufferNum>
    inline bool CalcUbMaxTileLen(uint64_t ubSize, int32_t dtype, GluSingleTilingOptParam &optTiling);

    inline void SaveOptBaseShape(uint32_t baseRowLen_, uint32_t baseColLen_, GluSingleTilingOptParam &optTiling);

    inline uint32_t getBaseColLenUpBound(GluSingleTilingOptParam &optTiling);

    inline uint32_t getBaseRowLenUpBound();

    inline bool MustBeSingleBaseRowLen(uint32_t baseColLen_);

    inline bool isInvalidBaseShape(uint32_t baseRowlen_, uint32_t baseColLen_);

    template<GLU_FLAG Glu_Flag>
    inline bool CalcOptBaseShape(GluSingleTilingOptParam &optTiling);

    uint32_t inputDTypeLen = 2;
    // Indicates the minimum processing data unit of the UB. Unit:element.
    // Formula: 32B/sizeof(DType). For example, if Dtype is BF16, ubMinBlockLen = 32/2 = 16
    uint32_t ubMinBlockLen = 0;
    // Length of the L2 cache line. Unit:element.
    // Formula: 512B/sizeof(DType). For example, if the Dtype is BF16, cacheLineLen = 512/2 = 256
    uint32_t cacheLineLen = 0;
    // baseColLen aligned package Len. elelment:Unit. 512-byte alignment or 32-byte alignment
    uint32_t alignPackLen = 0;
    uint32_t totalAvailableCore = 0; // total avaliable core in device
};

inline bool GetLengthByType(int32_t dtype, uint32_t& dsize) {
    switch (dtype) {
        case ge::DT_FLOAT16:
        case ge::DT_INT16:
        case ge::DT_UINT16:
        case ge::DT_BF16:
            dsize = sizeof(int16_t);
            return true;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
        case ge::DT_UINT32:
            dsize = sizeof(int32_t);
            return true;
        case ge::DT_DOUBLE:
        case ge::DT_INT64:
        case ge::DT_UINT64:
            dsize = sizeof(int64_t);
            return true;
        default:
            return false;
    }
}

    template <GLU_FLAG Glu_Flag>
    inline uint32_t GetSelfIdx()
    {
        return Glu_Flag == GLU_FLAG::SWIGLU_GRAD_SINGLE ? 1 : 0;
    }

    template <GLU_FLAG Glu_Flag>
    inline bool GluSingleTilingCalculator::SetTotalShape(const gert::Shape& inShape, const int32_t inDim)
    {
        int64_t shapeBefore = 1;
        int64_t shapeAfter = 1;
        int64_t dimNum = inShape.GetDimNum();
        if (inDim < -dimNum || inDim >= dimNum) {
            OP_LOGE((Glu_Flag == SWIGLU_SINGLE ? "SwiGlu" : "SwiGluGrad"), "SetTotalShape Unsupported inDim %d", inDim);
            return false;
        } 
        int64_t splitDim = inDim < 0 ? dimNum + inDim : inDim; // inDim default -1
        for (int64_t i = 0; i < splitDim; i++) {
            shapeBefore *= inShape.GetDim(i);
        }
        for (int64_t j = splitDim; j < dimNum; j++) {
            shapeAfter *= inShape.GetDim(j);
        }
        // 如果shape不是2的倍数,返回
        if (shapeAfter % 2 != 0) {
            OP_LOGE((Glu_Flag == SWIGLU_SINGLE ? "SwiGlu" : "SwiGluGrad"), "SetTotalShape Unsupported inDim %d, shapeAfter %ld %% 2 != 0", inDim, shapeAfter);
            return false;
        }

        tilingData->set_rowLen(shapeBefore);
        // colLen为原shape除以2
        tilingData->set_colLen(shapeAfter / 2);
        return true;
    }

    template <GLU_FLAG Glu_Flag, uint16_t bufferNum>
    bool GluSingleTilingCalculator::GetBufferNumAndDataLenPerUB(uint64_t ubSize, int32_t dtype, uint64_t& dataLenPerUB)
    {
        uint32_t singleDataSize = 0;
        switch (Glu_Flag) {
        case GLU_FLAG::SWIGLU_SINGLE:
            if (dtype == ge::DT_FLOAT16) {
                singleDataSize = bufferNum * XXGLU_TQUE_NUM * sizeof(int16_t) + SWIGLU_TBUF_NUM_HALF * sizeof(int32_t);
            } else if (dtype == ge::DT_BF16) {
                singleDataSize = bufferNum * XXGLU_TQUE_NUM * sizeof(int16_t) + SWIGLU_TBUF_NUM_BF16 * sizeof(int32_t);
            } else {
                singleDataSize = bufferNum * XXGLU_TQUE_NUM * sizeof(int32_t) + SWIGLU_TBUF_NUM_FLOAT * sizeof(int32_t);
            }
            dataLenPerUB = ubSize / singleDataSize;
            return true;
        case GLU_FLAG::SWIGLU_GRAD_SINGLE:
            if (dtype == ge::DT_FLOAT16) {
            singleDataSize = bufferNum * XXGLU_BW_TQUE_NUM * sizeof(int16_t) +
                             SWIGLU_BW_TBUF_NUM_HALF * sizeof(int32_t);
            } else if (dtype == ge::DT_BF16) {
            singleDataSize = bufferNum * XXGLU_BW_TQUE_NUM * sizeof(int16_t) +
                             SWIGLU_BW_TBUF_NUM_BF16 * sizeof(int32_t);
            } else {
            singleDataSize = bufferNum * XXGLU_BW_TQUE_NUM * sizeof(int32_t) +
                             SWIGLU_BW_TBUF_NUM_FLOAT * sizeof(int32_t);
            }
            dataLenPerUB = ubSize / singleDataSize;
            return true;
        default:
            return false;
    }
}

template <GLU_FLAG Glu_Flag, uint16_t bufferNum>
inline bool GluSingleTilingCalculator::CalcUbMaxTileLen(uint64_t ubSize, int32_t dtype, GluSingleTilingOptParam& optTiling)
{
    // get buffernum and maxTileLen
    uint64_t maxTileLenPerUB = 1;
    if (!GetBufferNumAndDataLenPerUB<Glu_Flag, bufferNum>(ubSize, dtype, maxTileLenPerUB)) {
        OP_LOGE((Glu_Flag == SWIGLU_SINGLE ? "SwiGlu" : "SwiGluGrad"), "CalcTiling Get bufferNum %d and maxTileLenPerUB %lu failed", bufferNum, maxTileLenPerUB);
        return false;
    }
    optTiling.maxTileLen = AlignDown<uint64_t>(maxTileLenPerUB, ubMinBlockLen);
    return true;
}

inline void GluSingleTilingCalculator::SaveOptBaseShape(uint32_t baseRowLen_, uint32_t baseColLen_, GluSingleTilingOptParam& optTiling)
{
    uint64_t totalTileNum = DivCeil<uint64_t>(tilingData->get_rowLen(), (baseRowLen_)) * DivCeil<uint64_t>(tilingData->get_colLen(), (baseColLen_));
    uint64_t baseSize = static_cast<uint64_t>(baseRowLen_) * baseColLen_;
    uint64_t baseTileNum = (tilingData->get_rowLen() / baseRowLen_) * (tilingData->get_colLen() / baseColLen_);
    uint32_t totalUsedCoreNum = std::min(totalTileNum, (uint64_t)totalAvailableCore);
    if ((optTiling.optTotalTileNum == 0) ||
        (totalUsedCoreNum > optTiling.totalUsedCoreNum) ||
        ((totalUsedCoreNum == optTiling.totalUsedCoreNum) && (totalTileNum < optTiling.optTotalTileNum)) ||
        ((totalUsedCoreNum == optTiling.totalUsedCoreNum) && (totalTileNum == optTiling.optTotalTileNum) && (baseSize > optTiling.optBaseSize)) ||
        ((totalUsedCoreNum == optTiling.totalUsedCoreNum) && (totalTileNum == optTiling.optTotalTileNum) && (baseSize = optTiling.optBaseSize) && (baseTileNum > optTiling.optBaseTileNum))) {
        optTiling.optBaseRowLen = baseRowLen_;
        optTiling.optBaseColLen = baseColLen_;
        optTiling.optTotalTileNum = totalTileNum;
        optTiling.optBaseSize = baseSize;
        optTiling.optBaseTileNum = baseTileNum;
        optTiling.totalUsedCoreNum = totalUsedCoreNum;
        optTiling.tileNumPerCore = DivCeil<uint64_t>(totalTileNum, totalUsedCoreNum);;
    }
}

inline uint32_t GluSingleTilingCalculator::getBaseColLenUpBound(GluSingleTilingOptParam& optTiling)
{
    uint32_t upBound = std::min(tilingData->get_colLen(), (uint64_t)optTiling.maxTileLen);
    if (tilingData->get_is32BAligned() == 1) {
        upBound = std::min(upBound, (uint32_t)DISCONTINE_COPY_MAX_BLOCKLEN);
    } else {
        upBound = std::min(upBound, (uint32_t)DISCONTINE_COPY_MAX_BLOCKLEN / inputDTypeLen);
    }

    if (upBound < tilingData->get_colLen() && upBound > cacheLineLen) {
        // 该种场景，每一个colLen至少被切割成2块，需要保证baseColLen为512B整数倍才高效
        return AlignDown<uint32_t>(upBound, cacheLineLen);
    } else {
        return upBound;
    }
}

inline uint32_t GluSingleTilingCalculator::getBaseRowLenUpBound()
{
    return std::min(tilingData->get_rowLen(), (uint64_t)DISCONTINE_COPY_MAX_BLOCKCNT);
}

/**
 * colLen 32B对齐时：若(colLen * 2 – baseCloLen) > 65535* ubMinBlockLen，则baseRowLen=1
 * colLen非32B对齐时：若(colLen * 2 – baseCloLen) * sizeof(Dtype) > 65535，则baseRowLen=1
 */
inline bool GluSingleTilingCalculator::MustBeSingleBaseRowLen(uint32_t baseColLen_)
{
    if (tilingData->get_is32BAligned() == 1) {
        // colLen 32B对齐时：若(colLen * 2 – baseCloLen) > 65535* ubMinBlockLen，则baseRowLen=1
        return ((tilingData->get_colLen() * 2 - baseColLen_) > (DISCONTINE_COPY_MAX_STRIDE * ubMinBlockLen));
    } else {
        // colLen非32B对齐时：若(colLen * 2 – baseCloLen) * sizeof(Dtype) > 65535，则baseRowLen=1
        return (((tilingData->get_colLen() * 2 - baseColLen_) * inputDTypeLen) > DISCONTINE_COPY_MAX_STRIDE);
    }
}

/**
 * 若则baseRowLen大于1，且约束判决baseRowLen必须等于1时，则是不合法的
 * colLen 32B对齐时：若(colLen * 2 – baseCloLen) > 65535* ubMinBlockLen，则baseRowLen=1
 * colLen非32B对齐时：若(colLen * 2 – baseCloLen) * sizeof(Dtype) > 65535，则baseRowLen=1
 */
inline bool GluSingleTilingCalculator::isInvalidBaseShape(uint32_t baseRowlen_, uint32_t baseColLen_)
{
    return ((baseRowlen_ < 1) || (baseRowlen_ > 1 && MustBeSingleBaseRowLen(baseColLen_)));
}
inline bool GluSingleTilingCalculator::isSupportSocV(uint32_t dtype, platform_ascendc::SocVersion socVersion_)
{
    if ((socVersion_ == platform_ascendc::SocVersion::ASCEND310P) && (dtype == ge::DT_BF16)) {
        return false; //310p dont support BF16
    } else {
        return true;
    }
}
template <GLU_FLAG Glu_Flag>
inline bool GluSingleTilingCalculator::CalcOptBaseShape(GluSingleTilingOptParam& optTiling)
{
    uint32_t baseColLen_ = getBaseColLenUpBound(optTiling);
    if (MustBeSingleBaseRowLen(baseColLen_)) {
        SaveOptBaseShape(1, baseColLen_, optTiling);
        return true;
    }

    while(true) {
        // colLen非32B对齐时，数据copy到ub时，每一行的尾部会补齐32B
        uint32_t baseRowlen_ = std::min(optTiling.maxTileLen / AlignUp<uint32_t>(baseColLen_, ubMinBlockLen), getBaseRowLenUpBound());
        if (isInvalidBaseShape(baseRowlen_, baseColLen_)) {
            return (optTiling.optTotalTileNum > 0);
        }
        // 保存较优的base shape
        SaveOptBaseShape(baseRowlen_, baseColLen_, optTiling);

        // baseColLen已经到达下限 或者 baseRowlen已经达到上限，无法继续调整，结束
        if (baseColLen_ <= alignPackLen || (baseRowlen_ >= getBaseRowLenUpBound())) {
            return true; // baseColLen无法继续调整了，结束
        }
        // 继续调整baseColLen
        // baseColLen为若alignPackLen的整数倍，则baseColLen减少1个alignPackLen的长度
        // 否则baseColLen减少到alignPackLen的整数倍（最接近的）
        if (baseColLen_ % alignPackLen == 0) {
            baseColLen_ -= alignPackLen;
        } else {
            baseColLen_ = AlignDown<uint32_t>(baseColLen_, alignPackLen);
        }
    }
}

// 如果开启double buffer，则bufferNum为2，否则为1
template <GLU_FLAG Glu_Flag, uint16_t bufferNum>
bool GluSingleTilingCalculator::CalcOptTiling(uint64_t ubSize, int32_t dtype, GluSingleTilingOptParam& optTiling)
{
    // 计算maxTilingLen
    if (!CalcUbMaxTileLen<Glu_Flag, bufferNum>(ubSize, dtype, optTiling)) {
        return false;
    }
    // 计算最优的base块形状
    if (!CalcOptBaseShape<Glu_Flag>(optTiling)) {
        return false;
    }
    return true;
}

template <GLU_FLAG Glu_Flag>
bool GluSingleTilingCalculator::CalcTiling(uint32_t totalCore, uint64_t ubSize, int32_t dtype,  platform_ascendc::SocVersion socVersion_)
{
    totalAvailableCore = totalCore;
    if (!GetLengthByType(dtype, inputDTypeLen)) {
        return false;
    }
    ubMinBlockLen = UB_MIN_BLOCK_SIZE / inputDTypeLen; // min block size
    cacheLineLen = L2_CACHE_LINE_SIZE / inputDTypeLen; // bandwidth max efficiency
    alignPackLen = cacheLineLen; // 默认512对齐，策略可调整
    // Is 32-byte aligned for split colLen?
    tilingData->set_is32BAligned(tilingData->get_colLen() % ubMinBlockLen == 0);
    // 310p not support Non-64B
    uint32_t blockSizeOf64B = BLOCK_SIZE_OF_64B / inputDTypeLen;
    if (((socVersion_ == platform_ascendc::SocVersion::ASCEND310P)) && (tilingData->get_colLen() % blockSizeOf64B != 0)) {
        OP_LOGE((Glu_Flag == SWIGLU_SINGLE ? "SwiGlu" : "SwiGluGrad"), "input shape is not support Non-64B aligned");
        return false;
    }
    // 先计算开启double buffer的tiling参数
    tilingData->set_isDoubleBuffer(1);
    GluSingleTilingOptParam optTilingDb;
    // 判断buffer = 2时是否计算成功
    if (!CalcOptTiling<Glu_Flag, 2>(ubSize, dtype, optTilingDb)) {
        return false;
    }
    GluSingleTilingOptParam *optTiling = &optTilingDb;
    // 如果double buffer开启的tiling参数中，每个核需要处理的tileNum等于2，尝试关闭double buffer;
    // 若关闭double buffer后只需要搬运1次数据，且使用的核没有减少, 则使用关闭double buffer的tiling
    // 判断tileNumPerCoer是否为2
    if (optTilingDb.tileNumPerCore == 2) {
        GluSingleTilingOptParam optTilingNoDb;
        if (CalcOptTiling<Glu_Flag, 1>(ubSize, dtype, optTilingNoDb) &&
            (optTilingNoDb.tileNumPerCore == 1) && (optTilingNoDb.totalUsedCoreNum >= optTilingDb.totalUsedCoreNum)) {
            optTiling = &optTilingNoDb;
            tilingData->set_isDoubleBuffer(0);
        }
    }
    // 记录最优的结果
    tilingData->set_baseRowLen(optTiling->optBaseRowLen);
    tilingData->set_baseColLen(optTiling->optBaseColLen);
    totalUsedCoreNum = optTiling->totalUsedCoreNum;
    return true;
}

inline static int64_t CeilDiv(int64_t value, int64_t factor) {
    int64_t valueNum = 0;
    if (factor == 0) {
        return value;
    }
    if (value % factor == 0) {
        valueNum = value / factor;
    } else {
        valueNum = value / factor + 1;
    }
    return valueNum;
}

static ge::graphStatus TilingPrepare4SwiGlu(gert::TilingParseContext* context) {
    return ge::GRAPH_SUCCESS;
}

template <GLU_FLAG Glu_Flag>
ge::graphStatus Tiling4SwiGlu(gert::TilingContext* context) {
    OP_LOGD(context->GetNodeName(), "Tiling4SwiGlu enter.");
    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t totalCore = platformInfo.GetCoreNumAiv();
    auto curShortSocName_ = platformInfo.GetSocVersion();
    uint64_t ubSize = 0;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (totalCore < 0 || totalCore >= MAX_CORE_NUMBER || ubSize <= UB_RESERVED_BUFF) {
        return ge::GRAPH_PARAM_INVALID;
    }
    ubSize -= UB_RESERVED_BUFF;

    SwiGluTilingData tilingData;
    auto inputDesc = context->GetInputDesc(0);
    ge::DataType dataType = inputDesc->GetDataType();

    auto inTensor = context->GetInputTensor(GetSelfIdx<Glu_Flag>());
    OPS_CHECK_NULL_WITH_CONTEXT(context, inTensor);
    auto inShape = inTensor->GetOriginShape();
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

    OP_LOGE_IF(attrs == nullptr, false, context->GetNodeName(), "Get op attrs failed.");

    uint32_t inDim = *(attrs->GetInt(0));

    GluSingleTilingCalculator tilingCalculator(&tilingData);
    if (!tilingCalculator.isSupportSocV(dataType, curShortSocName_)) {
        return ge::GRAPH_FAILED;
    }

    if (!tilingCalculator.SetTotalShape<Glu_Flag>(inShape, inDim) ||
        !tilingCalculator.CalcTiling<Glu_Flag>(totalCore, ubSize, dataType, curShortSocName_)) {
        return ge::GRAPH_FAILED;
    }

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    context->SetBlockDim(tilingCalculator.totalUsedCoreNum);
    context->SetTilingKey(dataType);
    return ge::GRAPH_SUCCESS;
}

struct SwiGluCompileInfo {};

IMPL_OP_OPTILING(SwiGluGrad).Tiling(Tiling4SwiGlu<SWIGLU_GRAD_SINGLE>).TilingParse<SwiGluCompileInfo>(TilingPrepare4SwiGlu);
}  // namespace optiling