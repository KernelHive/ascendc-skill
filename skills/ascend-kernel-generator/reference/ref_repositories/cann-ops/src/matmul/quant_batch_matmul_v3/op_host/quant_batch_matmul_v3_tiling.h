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
 * \file quant_batch_matmul_v3_tiling.h
 * \brief
 */
#ifndef QUANT_BATCH_MATMUL_V3_TILING_H
#define QUANT_BATCH_MATMUL_V3_TILING_H
#include <memory>
#include "tiling/tiling_base.h"
#include "cache_tiling.h"
#include "cube/algorithm/hash/tiling_cache.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_util.h"

#define OP_LOGI(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGD(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGW(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGE(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGE_IF(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)

namespace optiling {
#define CUBE_CALL_ERR_REPORT(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)
}  // namespace optiling

namespace optiling {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }

#define unlikely(x) __builtin_expect((x), 0)
#define OP_LOGE_IF(condition, return_value, op_name, fmt, ...)                                                 \
  static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool"); \
  do {                                                                                                         \
    if (unlikely(condition)) {                                                                                 \
      OP_LOGE(op_name, fmt, ##__VA_ARGS__);                                                                    \
      return return_value;                                                                                     \
    }                                                                                                          \
  } while (0)
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

namespace optiling {

// QuantBatchMatmulV3Tiling set QuantBatchMatmulV3Params tilingData, mc2 calls QuantBatchMatmulV3Tiling DoLibApiTiling
BEGIN_TILING_DATA_DEF(QuantBatchMatmulV3Params)
    TILING_DATA_FIELD_DEF(uint32_t, batchA);
    TILING_DATA_FIELD_DEF(uint32_t, batchB);
    TILING_DATA_FIELD_DEF(uint32_t, batchC);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreBatch);
    TILING_DATA_FIELD_DEF(uint32_t, isPerTensor);
    TILING_DATA_FIELD_DEF(uint32_t, isPertoken);
    TILING_DATA_FIELD_DEF(uint32_t, biasThreeDim);
    TILING_DATA_FIELD_DEF(uint32_t, ubCalcM);
    TILING_DATA_FIELD_DEF(uint32_t, ubCalcN);
    TILING_DATA_FIELD_DEF(uint32_t, needUbBuffer);
    TILING_DATA_FIELD_DEF(uint32_t, realSingleCoreM);
    TILING_DATA_FIELD_DEF(uint32_t, realSingleCoreN);
    TILING_DATA_FIELD_DEF(uint32_t, biasDtype); //代替原来的isBiasBf16
    TILING_DATA_FIELD_DEF(uint32_t, ubSize);
    TILING_DATA_FIELD_DEF(uint32_t, isMClash);
    TILING_DATA_FIELD_DEF(uint32_t, isNClash);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QuantBatchMatmulV3ParamsOp, QuantBatchMatmulV3Params)

BEGIN_TILING_DATA_DEF(L2cacheTileParam)
    TILING_DATA_FIELD_DEF(uint32_t, mTileCntL2);
    TILING_DATA_FIELD_DEF(uint32_t, nTileCntL2);
    TILING_DATA_FIELD_DEF(uint32_t, mTileBlock);
    TILING_DATA_FIELD_DEF(uint32_t, nTileBlock);
    TILING_DATA_FIELD_DEF(uint32_t, calOrder);
    TILING_DATA_FIELD_DEF(uint32_t, isBasicTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(L2cacheTileParamOp, L2cacheTileParam)

BEGIN_TILING_DATA_DEF(QuantBatchMatmulV3TilingData)
    TILING_DATA_FIELD_DEF_STRUCT(QuantBatchMatmulV3Params, params);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
    TILING_DATA_FIELD_DEF_STRUCT(L2cacheTileParam, tileL2cacheTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QuantBatchMatmulV3, QuantBatchMatmulV3TilingData)
REGISTER_TILING_DATA_CLASS(QuantBatchMatmulV3TilingDataOp, QuantBatchMatmulV3TilingData)

constexpr uint64_t BASIC_ALIGN_16 = 16;
constexpr uint64_t UINT32_MASK = 0x00000000ffffffff;

/**
 * QuantBatchMatmulInfo 增改参数规则：
 *   1. 新增参数需要初始化
 *   2. 检查mc2依赖文件,清单列表:
 *     ops/built-in/op_tiling/runtime/matmul_all_reduce/quant_matmul_all_reduce_tiling.h
 *     ops/built-in/op_tiling/runtime/matmul_all_reduce/quant_matmul_all_reduce_tiling.cc
 *     ops/built-in/op_tiling/runtime/matmul_all_reduce/quant_matmul_all_reduce_tiling_310_general.h
 *     ops/built-in/op_tiling/runtime/matmul_all_reduce/quant_matmul_all_reduce_tiling_310_general.cc
 *     ops/built-in/op_tiling/runtime/quant_batch_matmul_v3/quant_batch_matmul_v3_tiling.cc
 *   3. 增加新参数需要和MC2开发者说明
 */
struct QuantBatchMatmulInfo {
public:
    uint64_t GetMatmulApiMSize() const;  // mm api单次计算的M
    uint64_t GetTotalMatmulApiMSize(uint64_t baseM) const;
    uint64_t GetTotalBaseMCnt(uint64_t baseM) const;
    void Reset();  // 新增数据成员要修改Reset函数

    bool initFlag = false;  // 避免重复解析flag
    bool transA = false;
    bool transB = false;
    bool hasBias = false;
    uint64_t mSize = 0UL;
    uint64_t mSizePerNpu = 0UL;
    uint64_t kSize = 0UL;
    uint64_t nSize = 0UL;
    uint64_t batchA = 0UL;
    uint64_t batchA1 = 0UL;
    uint64_t batchA2 = 0UL;
    uint64_t batchA3 = 0UL;
    uint64_t batchA4 = 0UL;
    uint64_t batchB = 0UL;
    uint64_t batchB1 = 0UL;
    uint64_t batchB2 = 0UL;
    uint64_t batchB3 = 0UL;
    uint64_t batchB4 = 0UL;
    uint64_t batchC = 0UL;
    uint64_t batchBias = 0UL;
    ge::DataType aDtype = ge::DT_INT8;
    ge::DataType bDtype = ge::DT_INT8;
    ge::DataType cDtype = ge::DT_FLOAT16;
    ge::DataType biasDtype = ge::DT_INT32;
    ge::DataType scaleDtype = ge::DT_UINT64;
    bool isPerTensor = false;
    bool isPertoken = false;
    int64_t outDtype = 0L;
    uint32_t libApiWorkSpaceSize = 0U;
    uint64_t bf16ExtreWorkSpaceSize = 0UL;
    const char *opName = nullptr;
    ge::Format aFormat = ge::FORMAT_ND;
    ge::Format bFormat = ge::FORMAT_ND;
    ge::Format cFormat = ge::FORMAT_ND;  // 新增数据成员要修改Reset函数
};

enum class QuantBatchMatmulV3Trans {
    NO_TRANS = 0, // transA 0 transB 0
    A_TRANS = 1,  // transA 1 transB 0
    B_TRANS = 2,  // transA 0 transB 1
    AB_TRANS = 3  // transA 1 transB 1
};

class QuantBatchMatmulV3Tiling : public TilingBaseClass {
public:
    explicit QuantBatchMatmulV3Tiling(gert::TilingContext *context);
    QuantBatchMatmulV3Tiling(gert::TilingContext *context, QuantBatchMatmulV3TilingData *out);
    ~QuantBatchMatmulV3Tiling() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override;
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData，mc2使用的直接接口
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

    void InitCompileInfo();
    void Reset();
    ge::graphStatus CheckContext();
    bool CheckDtypeOnOnlyL0c2ub() const;
    bool CheckDtypeOnOnlyL0c2out() const;
    bool CheckDtypeOnOnlyL0c2outForSupportedList() const;
    bool CheckDtypeOnOnlyL0c2outForA4W4() const;
    bool CheckDtypeOnOnlyL0c2outForPertoken() const;
    bool CheckDtypeOnOnlyL0c2outForX1NZ() const;
    bool CheckDtypeOnOnlyL0c2outForUnclassified() const;
    bool AnalyzeDtype();
    bool AnalyzeAttrs();
    void SetFormat();

    // mc2使用的直接接口：begin
    virtual const gert::Shape GetX1Shape(const size_t index);
    virtual const gert::Shape GetX2Shape(const size_t index);
    virtual const gert::Shape &GetScaleShape(const size_t index);
    virtual const gert::StorageShape *GetPertokenShape(const size_t index);
    virtual const gert::StorageShape *GetBiasShape(const size_t index);
    bool AnalyzeInputs();
    // mc2使用的直接接口：end

    uint64_t GetTilingKey(bool isBasicTiling) const;
    virtual bool GetUbDequantExtreSpace();
    virtual ge::graphStatus CalcUbTiling();

    bool SetMatmulTilingFromTbeTiling();
    bool GetTbeTiling();
    void ProcessMSmall();
    int32_t GetIteratorOrder();
    bool SetPlatformInfoForTiling();
    uint64_t GetBatchSize(const gert::Shape &shape) const;
    bool InferOutBatchDim(const gert::Shape &x1Shape, const gert::Shape &x2Shape);
    void PrintTilingData();
    void PrintTbeTiling();
    void PrintTilingParams() const;

    ge::graphStatus CalcPertokenOptUbTiling();
    ge::graphStatus CalcUbTiling(uint32_t baseN, uint32_t baseM);
    void SpiltSingleCore(int32_t &singleCoreM, int32_t &singleCoreN);
    void SpiltForWorkSpaceLimit(int32_t singleCoreM, int32_t singleCoreN, int32_t blockDim);
    bool SetBlockDimsAndSingleCore(TCubeTiling &mt);
    bool CalcUsedL1AndUBSize(int32_t aL1Size, int32_t bL1Size, bool &fallback);
    bool CheckShapeInBoundary(const gert::Shape &shape, uint32_t shapeIdx) const;
    int8_t CheckFusionBatchA(const gert::Shape& x1Shape, const gert::Shape& x2Shape, const gert::Shape& biasShape,
                             uint64_t fusedDimValue) const;
    bool CheckOutputShapeAvailable() const;
    ge::graphStatus InitTilingData(matmul_tiling::MatmulApiTilingBase &mm, bool fallback = false);
    void AnalyzeBatchInfo(const gert::Shape &oriShapeA, const gert::Shape &oriShapeB);
    void Int4LowerAxisAlign(uint64_t &baseM, uint64_t &baseN) const;
    uint64_t CalcL1SizeForBiasAndScale();
    int32_t CalcND2NZSpace() const;
    void ConstructCacheParams(BatchmatmulCompileParas &compileParams, BatchmatmulRunParas &runParams) const;
    void ModifyCacheParams(BatchmatmulRunParas &runParams) const;
    bool NeedAtomiClean() const;
    void DoBatchFusion(uint64_t fusedDimValue);
    bool CheckDimValue(const gert::Shape &scaleShape, const gert::StorageShape *biasShape,
                       const gert::StorageShape *pertokenShape, const std::vector<int64_t> &dimValueOfMKN) const;
    bool CheckShapeInRangeForMandtoryInputs(size_t x1ShapeLen, size_t x2ShapeLen, size_t scaleShapeLen) const;
    bool CheckShapeInRangeForOptionalInputs(const gert::StorageShape* biasShape,
                                            const gert::StorageShape* pertokenShape) const;
    bool BiasShapeCheck(const gert::Shape &biasShape) const;
    bool CheckShape(const std::vector<gert::Shape *> &mandtoryShape, const gert::StorageShape* biasShape,
                    const gert::StorageShape* pertokenShape, const std::vector<int64_t> &dimValueOfMKN) const;
    uint64_t GetTotalSize(uint64_t m, uint64_t k, uint64_t n) const;
    void SetTransAttr(QuantBatchMatmulV3Trans &trans) const;
    uint32_t GetABankConflictSize();

    template<typename T>
    inline bool CheckNumberIsVaild(const T &num, const std::string &opName, const std::string &description) const {
        if (num > UINT32_MASK) {
            OP_LOGW(opName.c_str(), "%s size is greater than UINT32_MAX or less than 0, num:%s",
                        description.c_str(), std::to_string(num).c_str());
            return true;
        }
        return false;
    };

    // 需要对齐16的参数需要判断是否大于floorAlign(uint32_max, 16)
    template<typename T>
    inline bool CheckNumberIsVaild2(const T &num, const std::string &opName, const std::string &description) const {
        if (num > ops::FloorAlign(UINT32_MASK, BASIC_ALIGN_16)) {
            OP_LOGW(opName.c_str(), "%s size is greater than floorAlign(UINT32_MAX, 16) or less than 0, num:%s",
                        description.c_str(), std::to_string(num).c_str());
            return true;
        }
        return false;
    };

    template <typename T>
    T GetShapeWithDataType(T size, ge::DataType dtype) const
    {
        if (dtype == ge::DT_INT4) {
            return size + size;
        } else {
            return size / static_cast<T>(ge::GetSizeByDataType(dtype));
        }
    }

    template <typename T>
    T GetSizeWithDataType(T shape, ge::DataType dtype) const
    {
        if (dtype == ge::DT_INT4) {
            return (shape + 1) >> 1;
        } else {
            return shape * static_cast<T>(ge::GetSizeByDataType(dtype));
        }
    }

    static constexpr uint64_t MB_SIZE = 1024 * 1024;
    static constexpr uint64_t KB_SIZE = 1024;
    static constexpr uint32_t ROW_FIRST = 1;
    static constexpr uint32_t COL_FIRST = 2;
    static constexpr uint64_t NUM_DB = 2;

    // 新增数据成员请注意：如果是在GetShapeAttrsInfo函数过程中获取的，请放到QuantBatchMatmulInfo结构体中，或者保证在DoOpTiling赋值
    QuantBatchMatmulInfo &inputParams_;
    bool isBf16Opt_;
    bool isUbQuant_;
    QuantBatchMatmulV3TilingData tilingDataSelf_;
    QuantBatchMatmulV3TilingData &tilingData_;
    Tiling tbeTiling_;
    gert::GemmCompileInfo compileInfo_;
    bool compileInfoInit_ = false;
    bool isTilingOut_ = false;
};
}  // namespace optiling
#endif  // QUANT_BATCH_MATMUL_V3_TILING_H
