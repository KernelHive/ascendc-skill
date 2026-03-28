/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TUNE_SPACE_FFN_TUNE_SPACE_H
#define TUNE_SPACE_FFN_TUNE_SPACE_H

#include "tune_space_register.h"
#include "ascendc_matmul_tune_space.h"

namespace OpTuneSpace {
const float CORE_USE_RATIO_THRESHOLD = 0;
const float L0C_USE_RATIO_THRESHOLD = 0.8f;
const float L0A_USE_RATIO_THRESHOLD = 0.8f;
const float L0B_USE_RATIO_THRESHOLD = 0.75f;
const float L1_USE_RATIO_THRESHOLD = 0.7f;

// For off-line generate Tiling
const float OFF_LINE_CL0_GENERATE_RATIO_MATMUL = 0.8f;
const int64_t OFF_LINE_CL0_GENERATE_NUM_MIN_MATMUL = 6;
const int64_t OFF_LINE_CL0_GENERATE_NUM_MAX_MATMUL = 12;
const float OFF_LINE_CUB_GENERATE_RATIO_MATMUL = 0.8f;
const int64_t OFF_LINE_CUB_GENERATE_NUM_MIN_MATMUL = 32;
const int64_t OFF_LINE_CUB_GENERATE_NUM_MAX_MATMUL = 256;
const float OFF_LINE_BLOCK_DIM_GENERATE_RATIO_MATMUL = 1.0;
const int64_t OFF_LINE_BLOCK_DIM_GENERATE_NUM_MIN_MATMUL = 4;
const int64_t OFF_LINE_BLOCK_DIM_GENERATE_NUM_MAX_MATMUL = 16;
const float OFF_LINE_ABL0_GENERATE_RATIO_MATMUL = 1;
const uint32_t OFF_LINE_MAX_TILING_NUM_MATMUL = 2000;

// moe 单个matmul的输入参数
struct FfnSingleInputArgs {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    bool biasFlag;
    ge::DataType xType;
    ge::DataType wType;
    ge::DataType bType;
};

// moe 两个matmul的输入参数
struct FfnInputArgs {
    int64_t tokens;
    FfnSingleInputArgs mm1FfnArgs;
    FfnSingleInputArgs mm2FfnArgs;
};

class FfnTuneSpace : public TuneSpace {
public:
    explicit FfnTuneSpace() = default;
    ~FfnTuneSpace() override = default;

    Status GetTuneSpace(gert::TilingContext* op, std::vector<nlohmann::json> &jsonTuneSpace) override;

private:
    Status GenMatmulTuneSpace(const FfnSingleInputArgs &ffnInputArgs,
        std::vector<tuningtiling::MatmulTunnerTiling> &mmTilingSpace);
    Status GenFfnTuneSpace(FfnInputArgs &ffnInputArgs,
        std::vector<tuningtiling::FfnTunnerTiling> &ffnTilingSpace);
    Status FfnTransInputArgs(const gert::TilingContext* op, FfnInputArgs &ffnInputArgs) const;
    void FfnTransBiasArgs(const gert::TilingContext* op, FfnInputArgs &ffnInputArgs) const;
    Status GetHardwareInfo(const gert::TilingContext* op);

    PlatformInfo hardwareInfo_;
    ResourceUseThreshold mmResourceThr_{
        .coreUseThr = CORE_USE_RATIO_THRESHOLD,
        .l0aUseThr = L0A_USE_RATIO_THRESHOLD,
        .l0bUseThr = L0B_USE_RATIO_THRESHOLD,
        .l0cUseThr = L0C_USE_RATIO_THRESHOLD,
        .l1UseThr = L1_USE_RATIO_THRESHOLD
    };
    OfflineAdjustArgs mmAdjustArgs_{
        .l0cArgs = {
            .ratio = OFF_LINE_CL0_GENERATE_RATIO_MATMUL,
            .minNum = OFF_LINE_CL0_GENERATE_NUM_MIN_MATMUL,
            .maxNum = OFF_LINE_CL0_GENERATE_NUM_MAX_MATMUL,
        },
        .cubeArgs = {
            .ratio = OFF_LINE_CUB_GENERATE_RATIO_MATMUL,
            .minNum = OFF_LINE_CUB_GENERATE_NUM_MIN_MATMUL,
            .maxNum = OFF_LINE_CUB_GENERATE_NUM_MAX_MATMUL,
        },
        .dimsArgs = {
            .ratio = OFF_LINE_BLOCK_DIM_GENERATE_RATIO_MATMUL,
            .minNum = OFF_LINE_BLOCK_DIM_GENERATE_NUM_MIN_MATMUL,
            .maxNum = OFF_LINE_BLOCK_DIM_GENERATE_NUM_MAX_MATMUL,
        },
        .l0abArgs = OFF_LINE_ABL0_GENERATE_RATIO_MATMUL,
        .maxTiling = OFF_LINE_MAX_TILING_NUM_MATMUL
    };
};
}
#endif // namespace OpTuneSpace