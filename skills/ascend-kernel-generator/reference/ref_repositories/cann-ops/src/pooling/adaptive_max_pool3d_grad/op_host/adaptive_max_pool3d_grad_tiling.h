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
 * \file adaptive_max_pool3d_grad_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ADAPTIVE_MAX_POOL3D_GRAD_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ADAPTIVE_MAX_POOL3D_GRAD_H

#include "tiling/tiling_base.h"
#include "tiling/tiling_type.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_templates_registry.h"
namespace ops {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
    if ((ptr) == nullptr) {                                                                        \
        std::printf("nullptr error!");                                                             \
        return ge::GRAPH_FAILED;                                                                     \
    }
const gert::Shape g_vec_1_shape = {1};
/**
 * Ensure that the returned shape is non-scalar.
 * When the dim num of shape is 0, this shape is considered to express a scalar.
 * This function returns the original shape when it receives a non-scalar shape, 
 * and returns the vector shape that returns a {1} when it receives a scalar shape
 * @param in_shape input shape
 * @return non-scalar shape
 */
inline const gert::Shape &EnsureNotScalar(const gert::Shape &in_shape) {
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}
}
namespace optiling {
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
    do {                                        \
    if (cond) {                               \
        log_func;                               \
        expr;                                   \
    }                                         \
    } while (0)
BEGIN_TILING_DATA_DEF(AdaptiveMaxPool3DGradTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, ncDim);
    TILING_DATA_FIELD_DEF(uint64_t, diDim);
    TILING_DATA_FIELD_DEF(uint64_t, hiDim);
    TILING_DATA_FIELD_DEF(uint64_t, wiDim);
    TILING_DATA_FIELD_DEF(uint64_t, doDim);
    TILING_DATA_FIELD_DEF(uint64_t, hoDim);
    TILING_DATA_FIELD_DEF(uint64_t, woDim);
    TILING_DATA_FIELD_DEF(uint64_t, kdMax);
    TILING_DATA_FIELD_DEF(uint64_t, khMax);
    TILING_DATA_FIELD_DEF(uint64_t, kwMax);
    TILING_DATA_FIELD_DEF(uint64_t, baseNc);
    TILING_DATA_FIELD_DEF(uint64_t, baseDo);
    TILING_DATA_FIELD_DEF(uint64_t, baseHo);
    TILING_DATA_FIELD_DEF(uint64_t, baseWo);
    TILING_DATA_FIELD_DEF(uint64_t, singleCoreNc);
    TILING_DATA_FIELD_DEF(uint64_t, singleCoreDo);
    TILING_DATA_FIELD_DEF(uint64_t, singleCoreHo);
    TILING_DATA_FIELD_DEF(uint64_t, singleCoreWo);
    TILING_DATA_FIELD_DEF(uint64_t, ncTail);
    TILING_DATA_FIELD_DEF(uint64_t, doTail);
    TILING_DATA_FIELD_DEF(uint64_t, hoTail);
    TILING_DATA_FIELD_DEF(uint64_t, woTail);
    TILING_DATA_FIELD_DEF(uint64_t, ncCnt);
    TILING_DATA_FIELD_DEF(uint64_t, doCnt);
    TILING_DATA_FIELD_DEF(uint64_t, hoCnt);
    TILING_DATA_FIELD_DEF(uint64_t, woCnt);
    TILING_DATA_FIELD_DEF(uint64_t, totalCnt);
    TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint64_t, totalUBSize);

    // scatter
    TILING_DATA_FIELD_DEF(uint64_t, ncRound);
    TILING_DATA_FIELD_DEF(uint64_t, ncRoundTail);
    TILING_DATA_FIELD_DEF(uint64_t, totalRound);
    TILING_DATA_FIELD_DEF(uint64_t, preCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AdaptiveMaxPool3DGrad, AdaptiveMaxPool3DGradTilingData)


// Index const
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t GRAD_INDEX = 1;
constexpr uint32_t ARGMAX_INDEX = 2;
constexpr size_t KSIZE_ATTR_INDEX = 0U;
constexpr size_t STRIDES_ATTR_INDEX = 1U;
constexpr size_t PADS_ATTR_INDEX = 2U;
constexpr size_t DILATION_ATTR_INDEX = 3U;
constexpr size_t CEIL_MODE_ATTR_INDEX = 4U;
// Params const
constexpr size_t NC_DIM_NUM = 2;
constexpr size_t NCDHW_DIM_NUM = 5;
constexpr uint32_t DTYPE_LEN_B8 = 1;
constexpr uint32_t DTYPE_LEN_B16 = 2;
constexpr uint32_t DTYPE_LEN_B32 = 4;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t MAX_BLOCK_COUNT = 65535;
constexpr uint32_t NUM_PER_REP_B16 = 128;
constexpr uint32_t NUM_PER_REP_B32 = 64;
constexpr uint32_t SELECT_RESERVED_UB_SIZE = 8192;
    constexpr uint64_t MAX_INT32 = 2147483647;
// Tiling const
constexpr uint32_t TILING_OVERLAP = 100;
constexpr uint32_t TILING_UB_NO_CUT = 0;
constexpr uint32_t TILING_UB_CUT_NC = 10;
constexpr uint32_t TILING_UB_CUT_DO = 20;
constexpr uint32_t TILING_UB_CUT_HO = 30;
constexpr uint32_t TILING_UB_CUT_WO = 40;
constexpr uint32_t TILING_UB_CUT_KD = 50;
constexpr uint32_t TILING_UB_CUT_KH = 60;
constexpr uint32_t TILING_UB_CUT_KW = 70;
constexpr uint32_t TILING_TYPE_NORMAL = 0;
constexpr uint32_t TILING_TYPE_CUTK = 1;
constexpr uint32_t TILING_TYPE_SCATTER = 2;

struct Tiling4AdaptiveMaxPool3DGradCompileInfo {
    platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
    uint64_t totalCoreNum = 0;
    uint64_t maxUbSize = 0;
};

struct AdaptiveMaxPool3DGradTilingParams {
    // Platform
    uint64_t maxUbSize{0};
    uint64_t totalCoreNum{0};
    // Input shape
    uint64_t xDtypeSize{0};
    uint64_t indexDtypeSize{0};
    uint64_t ncDim{0};
    uint64_t diDim{0};
    uint64_t hiDim{0};
    uint64_t wiDim{0};
    uint64_t doDim{0};
    uint64_t hoDim{0};
    uint64_t woDim{0};
    // Kernel size
    uint64_t kdMax{0};
    uint64_t khMax{0};
    uint64_t kwMax{0};
    // Cal params
    uint64_t dGcd{0};
    uint64_t vl{0};
    uint64_t baseNc{0};
    uint64_t baseDo{0};
    uint64_t baseHo{0};
    uint64_t baseWo{0};
    uint64_t ncTail{0};
    uint64_t doTail{0};
    uint64_t hoTail{0};
    uint64_t woTail{0};
    uint64_t ncCnt{0};
    uint64_t doCnt{0};
    uint64_t hoCnt{0};
    uint64_t woCnt{0};
    uint64_t totalCnt{0};
    uint64_t usedCoreNum{0};

    // Normal params
    uint64_t singleCoreNc{0};
    uint64_t singleCoreDo{0};
    uint64_t singleCoreHo{0};
    uint64_t singleCoreWo{0};
    // Scatter params
    uint64_t ncRound{0};
    uint64_t ncRoundTail{0};
    uint64_t totalRound{0};
    uint64_t preCoreNum{0};

    // Workspace
    uint64_t workspaceSize{0};
    // Tiling key parmas
    uint64_t tilingType{0};
    uint32_t ubCutAxis{0};
    bool isOverLap{false};
};

class AdaptiveMaxPool3DGradTilingBase : public TilingBaseClass {
public:
    explicit AdaptiveMaxPool3DGradTilingBase(gert::TilingContext *context) : TilingBaseClass(context) {}
    ~AdaptiveMaxPool3DGradTilingBase() override {}

    std::string nodeName = "AdaptiveMaxPool3DGrad";
    AdaptiveMaxPool3DGradTilingData tilingData;
    AdaptiveMaxPool3DGradTilingParams maxPoolGradParams;
    uint64_t CeilDiv(const uint64_t dividend, const uint64_t divisor);
    uint64_t Gcd(uint64_t a, uint64_t b);
    inline uint64_t CalKIndexStart(uint64_t &kIdx, uint64_t &innerDim, uint64_t &outerDim);
    inline uint64_t CalKIndexEnd(uint64_t &kIdx, uint64_t &innerDim, uint64_t &outerDim);
    inline uint64_t CalKIndexLen(uint64_t &kIdx, uint64_t &innerDim, uint64_t &outerDim);
    inline uint64_t GetMaxK(uint64_t &innerDim, uint64_t &outerDim);

    bool CheckInputShape();
    ge::graphStatus SetInputParams();
    ge::graphStatus CheckInputDtype();
    ge::graphStatus CheckInputValid();
    void SetCntTailTilingParams();
    void SetOtherInputParams();
    void SetBaseTilingData();
    void PrintTilingData();

protected:
    // Order: GetShapeAttrsInfo->GetPlatformInfo->
    //        IsCapable->DoOpTiling->DoLibApiTiling->
    //        GetWorkspaceSize->GetTilingKey
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
};

class AdaptiveMaxPool3DGradNormalTiling : public AdaptiveMaxPool3DGradTilingBase {
public:
    explicit AdaptiveMaxPool3DGradNormalTiling(gert::TilingContext *context)
        : AdaptiveMaxPool3DGradTilingBase(context)
    {}
    ~AdaptiveMaxPool3DGradNormalTiling() override {}

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;

private:
    uint64_t CalUBTotalSize(uint64_t baseDo, uint64_t baseHo, uint64_t baseWo);
    uint64_t CalBestBaseSize(uint64_t baseXoStart, uint64_t baseXoEnd);
    bool SetNormalParamsUB();
    bool SetNormalTilingParams();
    void SetOtherTilingParams();
    void SetNormalTilingData();
    void PrintNormalTilingData();
};

class AdaptiveMaxPool3DGradScatterTiling : public AdaptiveMaxPool3DGradTilingBase {
public:
    explicit AdaptiveMaxPool3DGradScatterTiling(gert::TilingContext *context)
        : AdaptiveMaxPool3DGradTilingBase(context)
    {}
    ~AdaptiveMaxPool3DGradScatterTiling() override {}

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    bool IsCapable() override;

private:
    bool SetScatterTilingParams();
    void SetOtherTilingParams();
    void SetScatterTilingData();
    void PrintScatterTilingData();
};
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_ADAPTIVE_MAX_POOL3D_GRAD_H
