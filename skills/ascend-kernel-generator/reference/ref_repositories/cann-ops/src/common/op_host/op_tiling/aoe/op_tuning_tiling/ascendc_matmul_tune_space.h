/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TUNE_SPACE_ASCENDC_MATMUL_TUNE_SPACE_H
#define TUNE_SPACE_ASCENDC_MATMUL_TUNE_SPACE_H

#include "tune_space.h"
#include "ffn_tuning_tiling.h"

namespace OpTuneSpace {
using BlockDimTuple = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>;
using L0CParamTuple = std::tuple<BlockDimTuple, uint32_t, uint32_t, size_t>;
using L0ABParamTuple = std::tuple<L0CParamTuple, uint32_t, size_t, size_t>;
using CubeParamTuple = std::tuple<L0ABParamTuple, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, size_t>;
// BLOCKDIM_TUPLE_DEFINE
const size_t BATCH_DIM_VALUE_INDEX = 0;
const size_t M_DIM_VALUE_INDEX = 1;
const size_t N_DIM_VALUE_INDEX = 2;
const size_t K_DIM_VALUE_INDEX = 3;
// L0C_PARAM_TUPLE_DEFINE
const size_t BLOCKDIM_TUPLE_INDEX = 0;
const size_t N_VALUE_INDEX = 1;
const size_t M_VALUE_INDEX = 2;
const size_t L0C_SIZE_VALUE_INDEX = 3;
// L0AB_PARAM_TUPLE_DEFINE
const size_t L0C_PARAM_TUPLE_INDEX = 0;
const size_t K_VALUE_INDEX = 1;
const size_t L0A_SIZE_VALUE_INDEX = 2;
const size_t L0B_SIZE_VALUE_INDEX = 3;
// CUBE_PARAM_TUPLE_DEFINE
const size_t L0AB_PARAM_TUPLE_INDEX = 0;
const size_t DB_AL1_FLAG_INDEX = 1;
const size_t DB_BL1_FLAG_INDEX = 2;
const size_t STEP_M_INDEX = 3;
const size_t STEP_N_INDEX = 4;
const size_t STEP_KA_INDEX = 5;
const size_t STEP_KB_INDEX = 6;
const size_t L1_SIZE_VALUE_INDEX = 7;

const uint32_t CONFIG_SPACE_MAX_STEP = 10;
const uint32_t FIXPIPE_BIASTABLE_MAX = 256;
const uint32_t FP32_BITS = 32;
const uint32_t FP32_BYTES = 4;
const uint32_t PBUFFER_QUAD = 4;
const uint32_t PBUFFER_OFF = 1;
const uint32_t PBUFFER_DB = 2;
const uint32_t ITERATE_OFF = 0;
const uint32_t ITERATE_ON = 1;

// For CompareL0CTiling
const float MATMUL_L0C_BASE_N_RATIO = 0.3f;
const float MATMUL_L0C_BUFFER_UTILIZATION_RATIO = 0.7f;
// For CompareL0ABTiling
const float MATMUL_L0B_FULL_LOAD = 0.1f;
const float MATMUL_L0A_BUFFER_UTILIZATION_RATIO = 0.4f;
const float MATMUL_L0B_BUFFER_UTILIZATION_RATIO = 0.5f;
// For CompareL1ABTiling
const float MATMUL_L1_BUFFER_UTILIZATION_RATIO = 0.6f;
const float MATMUL_AL1_PBBUFFER_RATIO = 0.15f;
const float MATMUL_BL1_PBBUFFER_RATIO = 0.15f;
const float MATMUL_L1_STEPKA_RATIO = 0.05f;
const float MATMUL_L1_STEPKB_RATIO = 0.05f;
// For CompareBlockDimTiling
const uint32_t MATMUL_FULL_LOAD_SCORE = 100;

struct ResourceUseThreshold {
    float coreUseThr{0};
    float l0aUseThr{0};
    float l0bUseThr{0};
    float l0cUseThr{0};
    float l1UseThr{0};
};

struct MatmulOfflineArgs {
    float ratio{0};
    int64_t minNum{0};
    int64_t maxNum{0};
};

struct OfflineAdjustArgs {
    MatmulOfflineArgs l0cArgs;
    MatmulOfflineArgs cubeArgs;
    MatmulOfflineArgs dimsArgs;
    float l0abArgs{0};
    uint32_t maxTiling{0};
};

// the define of platform information
struct PlatformInfo {
    uint32_t aiCoreCnt{0};
    uint32_t l1Buffer{0};
    uint32_t l0aBuffer{0};
    uint32_t l0bBuffer{0};
    uint32_t l0cBuffer{0};
    uint32_t ubBuffer{0};
};

enum class TPosition {
    GM,
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
    MAX,
};

// matmul解空间生成需要的输入参数
struct MatmulInputArgs {
    int32_t batch = 0;
    int32_t m = 0;
    int32_t n = 0;
    int32_t k = 0;
    ge::DataType inputDtypeA = ge::DataType::DT_FLOAT16;
    ge::DataType inputDtypeB = ge::DataType::DT_FLOAT16;
    ge::DataType biasDtype = ge::DataType::DT_FLOAT16;
    ge::DataType outputDtypeC = ge::DataType::DT_FLOAT;
    ge::DataType madDtype = ge::DataType::DT_FLOAT;
    // 针对不同内存来源的输入增加表达以及计算逻辑
    TPosition inputPosA = TPosition::GM;
    TPosition inputPosB = TPosition::GM;
    TPosition outputPosC = TPosition::GM;
    TPosition inputPosBias = TPosition::GM;
    bool isBias = 0;
    int32_t fusedNumber = 1;
}; // single core mm

// matmul单维度全量解空间参数
struct MatmulTilingOutputSpace {
    std::vector<uint32_t> batchDim = {1};
    std::vector<uint32_t> mDim = {1};
    std::vector<uint32_t> nDim = {1};
    std::vector<uint32_t> kDim = {1};
    std::vector<uint32_t> baseM = {1};
    std::vector<uint32_t> baseN = {1};
    std::vector<uint32_t> baseK = {1};
    std::vector<uint32_t> stepM = {1};
    std::vector<uint32_t> stepN = {1};
    std::vector<uint32_t> stepKa = {1};
    std::vector<uint32_t> stepKb = {1};
    std::vector<uint32_t> dbAL1Flag = {1};
    std::vector<uint32_t> dbBL1Flag = {1};
    std::vector<uint32_t> iterateOrder = {1};
};

class AscendcMatmulTuneSpace {
public:
    explicit AscendcMatmulTuneSpace(const MatmulInputArgs &mmInputArgs, const PlatformInfo &hardwareInfo)
        : mmInputArgs_(mmInputArgs),
        hardwareInfo_(hardwareInfo) {
            // 根据不同类型计算设置K0/M0/N0
            reduceM0_ = FP32_BITS / ge::GetSizeByDataType(static_cast<ge::DataType>(mmInputArgs.inputDtypeA));
            reduceN0_ = FP32_BITS / ge::GetSizeByDataType(static_cast<ge::DataType>(mmInputArgs.inputDtypeB));
            reduceK0_ = FP32_BITS / ge::GetSizeByDataType(static_cast<ge::DataType>(mmInputArgs.inputDtypeB));
        }
    ~AscendcMatmulTuneSpace() = default;
    Status GetMatmulTilingSpace(std::vector<tuningtiling::MatmulTunnerTiling> &mmTilingSpace);

    void SetMatmulAdjustableParams(const ResourceUseThreshold &resourceThr, const OfflineAdjustArgs &adjustArgs)
    {
        mmResourceThr_ = resourceThr;
        mmAdjustArgs_ = adjustArgs;
    }
    void SetBatchDimConfigSpace(std::vector<uint32_t> batchDim)
    {
        batchDimConfigSpace_ = batchDim;
    }
    void SetNDimConfigSpace(std::vector<uint32_t> nDim)
    {
        nDimConfigSpace_ = nDim;
    }
    void SetMDimConfigSpace(std::vector<uint32_t> mDim)
    {
        mDimConfigSpace_ = mDim;
    }
    void SetKDimConfigSpace(std::vector<uint32_t> kDim)
    {
        kDimConfigSpace_ = kDim;
    }
    void SetBaseMConfigSpace(std::vector<uint32_t> baseM)
    {
        baseMConfigSpace_ = baseM;
    }
    void SetBaseNConfigSpace(std::vector<uint32_t> baseN)
    {
        baseNConfigSpace_ = baseN;
    }
    void SetBaseKConfigSpace(std::vector<uint32_t> baseK)
    {
        baseKConfigSpace_ = baseK;
    }
    void SetStepMConfigSpace(std::vector<uint32_t> stepM)
    {
        stepMConfigSpace_ = stepM;
    }
    void SetStepNConfigSpace(std::vector<uint32_t> stepN)
    {
        stepNConfigSpace_ = stepN;
    }
    void SetStepKaConfigSpace(std::vector<uint32_t> stepKa)
    {
        stepKaConfigSpace_ = stepKa;
    }
    void SetStepKbConfigSpace(std::vector<uint32_t> stepKb)
    {
        stepKbConfigSpace_ = stepKb;
    }
    void SetDoubleBufferAL1ConfigSpace(std::vector<uint32_t> dbAL1Flag)
    {
        dbAL1ConfigSpace_ = dbAL1Flag;
    }
    void SetDoubleBufferBL1ConfigSpace(std::vector<uint32_t> dbBL1Flag)
    {
        dbBL1ConfigSpace_ = dbBL1Flag;
    }
    void SetIterateOrderConfigSpace(std::vector<uint32_t> iterateOrder)
    {
        iterateOrderConfigSpace_ = iterateOrder;
    }

private:
    void GenerateMatmulConfigSpace();
    void GenerateDim(const uint32_t aiCoreCnt, const uint32_t oriShape, std::vector<uint32_t>& dimVec) const;
    void GenerateBlockDim();
    void GenerateDoubleBufferSpace();
    void CalculateAL0BufferSize(const std::vector<uint32_t>& aL0Param, const uint32_t pbAL0, uint32_t &result);
    void CalculateBL0BufferSize(const std::vector<uint32_t>& bL0Param, const uint32_t pbBL0, uint32_t &result);
    bool CheckExceedBufferSingleDim(const uint32_t value, const uint32_t anotherBlockSize,
        const ge::DataType dataType, const uint32_t maxSize, const uint32_t fusedNum) const;
    void CalculateMConfigSpace(const uint32_t oriShape, std::vector<uint32_t> &xVec);
    void CalculateNConfigSpace(const uint32_t oriShape, std::vector<uint32_t> &xVec);
    void GenerateBaseParam();
    void GenerateL1Param();
    void GetL1Steps(const uint32_t oriShape, std::vector<uint32_t> &step) const;
    float CalculateTilingBlockDimScore(const BlockDimTuple& blockTiling);
    void GenerateL0CTilingSpace(const BlockDimTuple& blockDimTuple, std::vector<uint32_t>& mCVec,
        std::vector<uint32_t>& nCVec);
    float CalculateTilingL0CScore(const L0CParamTuple& tiling);
    float CalculateTilingL0ABScore(const L0ABParamTuple& tiling);
    float CalculateTilingL1Score(const CubeParamTuple& tiling);
    Status GenerateCompleteTiling(const CubeParamTuple& tilingStruct);
    Status L0ABAnalyzer(const L0CParamTuple& l0cTiling);
    Status L0CAnalyzer(const BlockDimTuple& blockDim);
    Status L1ABAnalyzer(const L0ABParamTuple& l0abTiling);
    void L1BAnalyzer(const CubeParamTuple& tmpParamTuple, std::vector<CubeParamTuple>& tilingStruct);
    int64_t GenerateTuplesNum(const MatmulOfflineArgs &offlineArgs, const int64_t size) const;
    void BlockDimMKAnalyze(const uint32_t batchDimIndex, const uint32_t nDimIndex,
        std::vector<BlockDimTuple>& blockDim);
    Status BlockDimAnalyze();

    uint32_t reduceN0_{16};
    uint32_t reduceM0_{16};
    uint32_t reduceK0_{16};
    uint32_t tilingCount_{0U};
    uint32_t l1BuffDownThr_{0};
    uint32_t coreDownThr_{0};
    uint32_t biasTableMax_{0};
    bool configSpaceGenFlag_{false};
    ResourceUseThreshold mmResourceThr_;
    OfflineAdjustArgs mmAdjustArgs_;
    MatmulInputArgs mmInputArgs_;
    MatmulTilingOutputSpace tilingConfigSpace_;
    PlatformInfo hardwareInfo_;
    std::vector<tuningtiling::MatmulTunnerTiling> mmTilingSpace_;

    std::vector<uint32_t> batchDimConfigSpace_;
    std::vector<uint32_t> mDimConfigSpace_;
    std::vector<uint32_t> nDimConfigSpace_;
    std::vector<uint32_t> kDimConfigSpace_;
    std::vector<uint32_t> baseMConfigSpace_;
    std::vector<uint32_t> baseNConfigSpace_;
    std::vector<uint32_t> baseKConfigSpace_;
    std::vector<uint32_t> stepMConfigSpace_;
    std::vector<uint32_t> stepNConfigSpace_;
    std::vector<uint32_t> stepKaConfigSpace_;
    std::vector<uint32_t> stepKbConfigSpace_;
    std::vector<uint32_t> dbAL1ConfigSpace_;
    std::vector<uint32_t> dbBL1ConfigSpace_;
    std::vector<uint32_t> iterateOrderConfigSpace_;
};
}
#endif // namespace OpTuneSpace