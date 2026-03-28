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
 * \file group_norm_silu_tiling.cpp
 * \brief
 */
#include "group_norm_silu_tiling.h"

namespace optiling {
    static const uint64_t INPUT_IDX_X = 0;
    static const uint64_t INPUT_IDX_GAMMA = 1;
    static const uint64_t INPUT_IDX_BETA = 2;
    static const uint64_t X_SHAPE_MIN_LEN = 2;
    static const uint64_t INDEX_NUM_GROUPS = 0;
    static const uint64_t INDEX_EPSILON = 1;
    static const uint64_t INDEX_ACTIVATE_SILU = 2;
    static const uint64_t DIM_0 = 0;
    static const uint64_t DIM_1 = 1;
    static const uint64_t DEFAULT_PROCESSSIZE = 8192;
    static const uint64_t DEFAULT_NUMGROUPS = 32;
    static const uint64_t RESERVED_WORKSPACE_SIZE_910B = 16 * 1024 * 1024;
    static const uint64_t RESERVED_WORKSPACE_SIZE_310P = 2 * 1024 * 1024;
    static const uint64_t FLOAT32_BYTES = 4;
    static const uint64_t FLOAT16_BYTES = 2;
    static const uint64_t BLOCK_SIZE = 32;
    static const int64_t HW_CAP = 4096;
    static const int64_t UPPER_LIMIT_TWO = 4000;
    static const int64_t UPPER_LIMIT_ONE = 2700;
    static const uint64_t FLOAT_EIGHT = 8;
    static const uint64_t FLOAT_DOUBLE_EIGHT = 16;
    static const uint64_t GAMMA_BETA_UB_NUM = 6;
    static const uint64_t RESERVED_BLOCK_NUM = 2;
    static const uint64_t INPUT_OUTPUT_UB_NUM = 20;

    inline static int64_t CeilDiv(int64_t value, int64_t factor) {
        if (factor == 0) {
            return value;
        } else if (value % factor == 0) {
            return value / factor;
        } else {
            return value / factor + 1;
        }
    }

    inline static int64_t Gcd(int64_t a, int64_t b) {
        if (b == 0) {
            return a;
        }
        return Gcd(b, a % b);
    }

    inline static int64_t Lcm(int64_t a, int64_t b) {
        return a * b / Gcd(a, b);
    }

    static ge::graphStatus TilingPrepare4GroupNormSilu(gert::TilingParseContext* context) {
        auto compileInfo = context->GetCompiledInfo<GroupNormSiluCompileInfo>();
        auto platformInfo = context->GetPlatformInfo();
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
        if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P) {
            compileInfo->is310P = 1;
            compileInfo->totalCoreNum = compileInfo->totalCoreNum + ascendcPlatform.GetCoreNumVector();
        }

        uint64_t ubSizePlatForm = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
        return ge::GRAPH_SUCCESS;
    }

    static void SetAttrParams(const gert::TilingContext* context, GroupNormSiluTilingData& tilingData) {
        auto attrs = context->GetAttrs();
        const int64_t* numGroups = attrs->GetAttrPointer<int64_t>(INDEX_NUM_GROUPS);
        const float* epsilon = attrs->GetAttrPointer<float>(INDEX_EPSILON);
        const bool* activateSilu = attrs->GetAttrPointer<bool>(INDEX_ACTIVATE_SILU);
        tilingData.set_numGroups(*numGroups);
        tilingData.set_epsilon(*epsilon);
        tilingData.set_activateSilu(*activateSilu);
    }

    static void SetTilingParams(const gert::TilingContext* context, GroupNormSiluTilingData& tilingData) {
        auto xShape = context->GetInputShape(INPUT_IDX_X)->GetStorageShape();
        uint64_t hwNum = 1;
        uint64_t xDims = xShape.GetDimNum();
        for (uint64_t i = 2; i < xDims; i++) {
            hwNum = hwNum * xShape.GetDim(i);
        }
        tilingData.set_shapeC(xShape.GetDim(DIM_1));
        tilingData.set_shapeD(tilingData.get_shapeC() / tilingData.get_numGroups());
        tilingData.set_hwNum(hwNum);
        tilingData.set_elemNum(tilingData.get_shapeD() * hwNum);
        tilingData.set_processSize(DEFAULT_PROCESSSIZE);
    }

    static void SetBlockTiling(const gert::TilingContext* context, GroupNormSiluTilingData& tilingData) {
        auto xDtype = context->GetInputDesc(INPUT_IDX_X)->GetDataType();
        uint64_t xDtypeSize = ge::GetSizeByDataType(xDtype);
        auto compileInfo = context->GetCompileInfo<GroupNormSiluCompileInfo>();
        auto xShape = context->GetInputShape(INPUT_IDX_X)->GetStorageShape();
        uint64_t shapeN = xShape.GetDim(DIM_0);
        tilingData.set_numPerCore(CeilDiv(shapeN * tilingData.get_numGroups(), compileInfo->totalCoreNum));
        tilingData.set_realCoreNum(CeilDiv(shapeN * tilingData.get_numGroups(), tilingData.get_numPerCore()));
        if (tilingData.get_hwNum() < static_cast<int64_t>(BLOCK_SIZE / xDtypeSize) &&
           (tilingData.get_hwNum() != 1 || tilingData.get_shapeD() < static_cast<int64_t>(BLOCK_SIZE / xDtypeSize))) {
            tilingData.set_realCoreNum(1);
        }

        tilingData.set_numLastCore(shapeN * tilingData.get_numGroups() - \
                                   tilingData.get_numPerCore() * (tilingData.get_realCoreNum() - 1));
        // split coreNum according to N
        if (tilingData.get_hwNum() == 1 && tilingData.get_numGroups() == tilingData.get_shapeC()) {
            if (tilingData.get_shapeC() % (BLOCK_SIZE / xDtypeSize) != 0) {
                tilingData.set_realCoreNum(1);
                tilingData.set_numLastCore(shapeN);
            } else {
                tilingData.set_numPerCore(CeilDiv(shapeN, compileInfo->totalCoreNum));
                tilingData.set_realCoreNum(CeilDiv(shapeN, tilingData.get_numPerCore()));
            }
            tilingData.set_numLastCore(shapeN - tilingData.get_numPerCore() * (tilingData.get_realCoreNum() - 1));
        }

        uint64_t xShapeSize = xShape.GetShapeSize();
        if (xShapeSize == 0) {
            tilingData.set_realCoreNum(-1);
        }
    }

    static void SetTilingWithSmallHW(const gert::TilingContext* context, GroupNormSiluTilingData& tilingData) {
        auto xDtype = context->GetInputDesc(INPUT_IDX_X)->GetDataType();
        uint64_t xDtypeSize = ge::GetSizeByDataType(xDtype);
        auto gammaDtype = context->GetInputDesc(INPUT_IDX_GAMMA)->GetDataType();
        uint64_t gammaDtypeSize = ge::GetSizeByDataType(gammaDtype);
        if (tilingData.get_hwNum() == 1) {
            if (xDtypeSize == FLOAT32_BYTES) {
                tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_HW_ONE_B32));
            } else if (gammaDtypeSize == FLOAT32_BYTES) {
                tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_HW_ONE_MIXTYPE));
            } else {
                tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_HW_ONE_B16));
            }
        } else {
            if (xDtypeSize == FLOAT32_BYTES) {
                tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_BASIC_TEM_B32));
            } else if (gammaDtypeSize == FLOAT32_BYTES) {
                tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_BASIC_TEM_MIXTYPE));
            } else {
                tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_BASIC_TEM_B16));
            }
        }
    }

    static void SetUbTiling(GroupNormSiluTilingData& tilingData) {
        tilingData.set_loopNum(CeilDiv(tilingData.get_elemNum(), tilingData.get_processSize()));
        tilingData.set_loopTail(tilingData.get_elemNum() - \
                                tilingData.get_processSize() * (tilingData.get_loopNum() - 1));
        tilingData.set_innerLoopNum(CeilDiv(tilingData.get_hwNum(), tilingData.get_processSize()));
        tilingData.set_innerLoopTail(tilingData.get_hwNum() - \
                                     tilingData.get_processSize() * (tilingData.get_innerLoopNum() - 1));
    }

    static void SetTiling(const gert::TilingContext* context, GroupNormSiluTilingData& tilingData) {
        auto xDtype = context->GetInputDesc(INPUT_IDX_X)->GetDataType();
        uint64_t xDtypeSize = ge::GetSizeByDataType(xDtype);
        auto gammaDtype = context->GetInputDesc(INPUT_IDX_GAMMA)->GetDataType();
        uint64_t gammaDtypeSize = ge::GetSizeByDataType(gammaDtype);
        if (tilingData.get_hwNum() < static_cast<int64_t>(BLOCK_SIZE / xDtypeSize)) {
            SetTilingWithSmallHW(context, tilingData);
        } else if (xDtypeSize == FLOAT32_BYTES && tilingData.get_hwNum() % FLOAT_EIGHT == 0 &&
            tilingData.get_hwNum() <= HW_CAP &&
            tilingData.get_shapeC() + tilingData.get_numPerCore() <= UPPER_LIMIT_TWO) {
            tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_SMALL_SHAPE_B32));
        } else if (xDtypeSize == FLOAT32_BYTES &&
                   tilingData.get_shapeC() + tilingData.get_numPerCore() <= UPPER_LIMIT_TWO) {
            tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_HIGH_PERF_B32));
        } else if (xDtypeSize == FLOAT16_BYTES && tilingData.get_hwNum() % FLOAT_DOUBLE_EIGHT == 0 &&
                   tilingData.get_hwNum() <= HW_CAP &&
                   tilingData.get_shapeC() + tilingData.get_numPerCore() <= UPPER_LIMIT_ONE) {
            if (gammaDtypeSize == FLOAT32_BYTES) {
                tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_SMALL_SHAPE_MIXTYPE));
            } else {
                tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_SMALL_SHAPE_B16));
            }
        } else if (xDtypeSize == FLOAT16_BYTES &&
                   tilingData.get_shapeC() + tilingData.get_numPerCore() <= UPPER_LIMIT_ONE) {
            if (gammaDtypeSize == FLOAT32_BYTES) {
                tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_HIGH_PERF_MIXTYPE));
            } else {
                tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_HIGH_PERF_B16));
            }
        } else if (xDtypeSize == FLOAT32_BYTES) {
            tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_BASIC_TEM_B32));
        } else {
            if (gammaDtypeSize == FLOAT32_BYTES) {
                tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_BASIC_TEM_MIXTYPE));
            } else {
                tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_BASIC_TEM_B16));
            }
        }
    }

    static void SetProcessSize(const gert::TilingContext* context, GroupNormSiluTilingData& tilingData) {
        auto gammaDtype = context->GetInputDesc(INPUT_IDX_GAMMA)->GetDataType();
        uint64_t gammaDtypeSize = ge::GetSizeByDataType(gammaDtype);
        uint64_t gammaPerCore = (tilingData.get_numPerCore() + 1) * tilingData.get_shapeD();
        uint64_t gammaPerCoreRoundUp = CeilDiv(gammaPerCore, (BLOCK_SIZE / gammaDtypeSize)) * (BLOCK_SIZE / gammaDtypeSize);
        auto compileInfo = context->GetCompileInfo<GroupNormSiluCompileInfo>();
        uint64_t remainUbSize = compileInfo->ubSizePlatForm - gammaPerCoreRoundUp * gammaDtypeSize * GAMMA_BETA_UB_NUM - \
                                BLOCK_SIZE * RESERVED_BLOCK_NUM;
        int64_t maxProcessSize = remainUbSize / INPUT_OUTPUT_UB_NUM ;
        if (maxProcessSize < tilingData.get_hwNum()) {
            maxProcessSize = maxProcessSize / BLOCK_SIZE * BLOCK_SIZE;
        } else {
            int64_t lcmNum = Lcm(tilingData.get_hwNum(), (BLOCK_SIZE / gammaDtypeSize));
            maxProcessSize = maxProcessSize / lcmNum * lcmNum;
        }
        tilingData.set_processSize(maxProcessSize);
        tilingData.set_numGroups(gammaPerCoreRoundUp);
    }

    static void SetTilingSD(const gert::TilingContext* context, GroupNormSiluTilingData& tilingData) {
        auto xShape = context->GetInputShape(INPUT_IDX_X)->GetStorageShape();
        auto xDtypeSize = ge::GetSizeByDataType(context->GetInputDesc(INPUT_IDX_X)->GetDataType());
        uint64_t shapeN = xShape.GetDim(DIM_0);
        if (tilingData.get_realCoreNum() > 0 && tilingData.get_numGroups() == DEFAULT_NUMGROUPS && shapeN == 1 &&
            xDtypeSize == FLOAT16_BYTES && tilingData.get_hwNum() > 1) {
            auto compileInfo = context->GetCompileInfo<GroupNormSiluCompileInfo>();
            uint64_t numPerCoreRound = tilingData.get_numGroups() / compileInfo->totalCoreNum;
            tilingData.set_numPerCore(numPerCoreRound);
            tilingData.set_realCoreNum(compileInfo->totalCoreNum);
            tilingData.set_numLastCore(tilingData.get_numGroups() - \
                                       tilingData.get_numPerCore() * tilingData.get_realCoreNum());
            SetProcessSize(context, tilingData);
            tilingData.set_loopNum(tilingData.get_elemNum() / tilingData.get_processSize());
            tilingData.set_loopTail(tilingData.get_elemNum() - tilingData.get_processSize() * tilingData.get_loopNum());
            if (tilingData.get_processSize() > tilingData.get_hwNum()) {
                tilingData.set_innerLoopNum(tilingData.get_processSize() / tilingData.get_hwNum());
                tilingData.set_innerLoopTail(tilingData.get_processSize() - tilingData.get_hwNum() * tilingData.get_innerLoopNum());
            } else {
                tilingData.set_innerLoopNum(tilingData.get_hwNum() / tilingData.get_processSize());
                tilingData.set_innerLoopTail(tilingData.get_hwNum() - \
                                             tilingData.get_processSize() * tilingData.get_innerLoopNum());
            }
            tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_SPECIAL_SHAPE_SD));
        }
    }

    static ge::graphStatus Tiling4GroupNormSilu(gert::TilingContext* context) {
        // check input && attrs params
        GroupNormSiluTilingData tilingData;
        SetAttrParams(context, tilingData);
        SetTilingParams(context, tilingData);
        // block tiling
        SetBlockTiling(context, tilingData);
        // tiling key
        SetTiling(context, tilingData);
        // ub tiling
        SetUbTiling(tilingData);
        auto compileInfo = context->GetCompileInfo<GroupNormSiluCompileInfo>();
        size_t sysWorkspaceSize = RESERVED_WORKSPACE_SIZE_910B;
        if (compileInfo->is310P == 1) {
            sysWorkspaceSize = RESERVED_WORKSPACE_SIZE_310P;
            SetTilingSD(context, tilingData);
        }
        tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

        // block dim, tilingKey
        context->SetBlockDim(tilingData.get_realCoreNum());
        context->SetTilingKey(tilingData.get_tilingKey());
        size_t* workspaces = context->GetWorkspaceSizes(1);
        workspaces[0] = sysWorkspaceSize;

        return ge::GRAPH_SUCCESS;
    }

IMPL_OP_OPTILING(GroupNormSilu)
.Tiling(Tiling4GroupNormSilu)
.TilingParse<GroupNormSiluCompileInfo>(TilingPrepare4GroupNormSilu);
}  // namespace optiling
