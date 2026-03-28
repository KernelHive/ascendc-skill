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
 * \file matmul_v3_base_tiling.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_BASE_TILING_H__
#define __OP_HOST_MATMUL_V3_BASE_TILING_H__

#include "mat_mul_v3_tiling.h"
#include "mat_mul_v3_common.h"
#include "mat_mul_v3_compile_info.h"
#include "tiling/tiling_base.h"
#include "aoe/op_tuning_tiling/gemm_tuning_tiling.h"

namespace optiling {
struct Tiling;
namespace matmul_v3 {
class MatmulV3BaseTiling : public TilingBaseClass {
public:
 public:
    explicit MatmulV3BaseTiling(gert::TilingContext* context)
        : TilingBaseClass(context), tilingData_(tilingDataSelf_) {
    }
    MatmulV3BaseTiling(gert::TilingContext* context,
                       MatmulTilingData* tilingData,
                       TilingCalcSelect tilingSelect = TilingCalcSelect::BASE)
        : TilingBaseClass(context), tilingData_(*tilingData) {
        InitCompileInfo();
        tilingSelect_ = tilingSelect;
    }
    ~MatmulV3BaseTiling() override {
    }

protected:
    bool IsCapable() override { return true; };
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

    void InitCompileInfo();
protected:
    ge::graphStatus CheckArgs();
    ge::graphStatus GetArgs();
    virtual ge::graphStatus GetMoreArgs();
    ge::graphStatus CheckDimsAligned310P();
    ge::graphStatus InitTilingData();
    void SetRunInfo();
    void SetNd2NzInfo();
    void SetParamsV310();
    bool GetTilingFromRepo();
    bool GetTilingInputArgs(std::shared_ptr<void> &inputArgs, size_t &size);
    void DebugLog(const std::shared_ptr<tuningtiling::GemmInputArgs> &inputArgs);
    bool TranslateAoeTiling(tuningtiling::TuningTilingDefPtr &tuningTiling);
    void DoTilingKey();
    void DoBasicTiling();
    void FormulaicBaseBlockTiling();
    void FormulaicTilingNoTrans();
    void CalL1TilingV200();
    void CalL1Tiling();
    void UpdateL1TilingStepK(uint64_t& stepK);
    bool DoL2CacheTiling();
    bool DoL2CacheTiling310P();
    virtual void UpdateNd2nzFlag() {};
    void DoNd2NzVectorTiling();
    bool DoBL1FullloadWithFixpipeTiling();
    void CalcNd2NzTiling(uint64_t dtypeSize, uint64_t nValue, uint64_t dValue, uint64_t& baseN, uint64_t& baseD);
    bool CheckUbOverFlow(uint64_t nAligned16, uint64_t nValue, const uint64_t& baseN,
                         const uint64_t& baseD, uint64_t dtypeSize);
    bool CheckBTSize(uint64_t baseN);
    template<CalcType T>
    void CalcBase(const std::vector<std::vector<uint64_t>>& baseMNK);
    void BalanceBaseBlockTiling();
    void CalBaseMBaseN(uint64_t& baseM, uint64_t& baseN,
                       uint64_t maxBaseM = BASIC_BLOCK_SIZE_128, uint64_t maxBaseN = BASIC_BLOCK_SIZE_256);
    void SetBaseBlockTiling();
    void DoSmallShapeTiling();
    void DoSelectTiling();
    bool IsSupportSingleCoreSplitK() const;
    bool DoSingleCoreSplitKTiling();
    bool DoAL1FullLoadTiling();
    bool DoBL1FullLoadTiling();
    bool SupportMultiSplitK() const;
    bool DoDeterministicMultiCoreSplitKTiling();
    bool IsNkOrder();
    void CalTileFactor(uint64_t& nTile);
    void SetBasicBlockOfMK33(MatmulV3RunInfo &runInfo);
    void SetBasicBlockOfNK33();
    bool NeedSolveFixBound();
    bool CheckAoeTilingEnable(uint32_t aoeTilingEnable, const std::string &opName);
    void SetBasicBlockOf24(MatmulV3RunInfo &runInfo, uint64_t &mTile, uint64_t &nTile) const;
    bool IsMixNd2nz();
    void InitL2SplitParams(MatmulV3L2SplitParams &l2SplitParams) const;
    bool IsTailSmall(MatmulV3L2SplitParams &l2SplitParams, uint64_t outL2Split, uint64_t innerL2Split,
                     uint64_t innerMaxConflict) const;
    bool CalcTile(uint64_t &outTile, uint64_t &innerTile, uint64_t &outL2Split, uint64_t &innerL2Split,
                  const bool isInnerBad) const;
    uint64_t GetTotalSize(uint64_t m, uint64_t k, uint64_t n, uint64_t aDtype, uint64_t bDtype) const;
    void DoIncreTiling();
    void GetV2Tiling();
    bool CheckMMTilingDataIsVaild();
    void FormulateBasicBlockDavid();
    void CalcTailBasicBlock();
    bool CheckSingleTilingOk(MatmulV3RunInfo &tmpRunInfo);
    bool IsOnTheWay(ge::Format matFormat, uint64_t innerSize, uint64_t dtypeSize,
                    const std::vector<uint64_t> supportNd2nzList) const;
    bool NeedNd2NzVnchw(uint64_t outerSize, uint64_t innerSize, bool supportNd2NzOnTheWay,
                        uint64_t dtypeSize, ge::Format matFormat) const;

private:
    MatmulTilingData tilingDataSelf_;
protected:
    MatmulV3CompileInfo compileInfo_;
    MatmulV3Args args_;
    matmul_tiling::MultiCoreMatmulTiling mm_;
    MatmulTilingData &tilingData_;
    MatmulV3RunInfo runInfo_;
    uint64_t aDtypeSize_{1};
    uint64_t bDtypeSize_{1};
    uint64_t cDtypeSize_{1};
    MatmulV3Trans trans_ = MatmulV3Trans::NO_TRANS;
    TilingEnable tilingEnable_;
    bool m256Align_{false};
    bool kA256Align_{false};
    bool kB256Align_{false};
    bool n256Align_{false};
    bool enableCache_{true};
    std::vector<std::vector<uint64_t>> calcMBasic_;
    std::vector<std::vector<uint64_t>> calcMNBasic_;
    uint64_t l2TileLength_{0};
    uint64_t basicBlockBaseM_{BASIC_BLOCK_SIZE_256};
    uint32_t l2CacheFlag_{0};
    bool compileInfoInit_{false};
    TilingCalcSelect tilingSelect_ = TilingCalcSelect::ALL;
};
}
}
#endif // __OP_HOST_MATMUL_V3_BASE_TILING_H__

