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
 * \file conv3d_tiling_algorithm_pointwise.h
 * \brief
 */

#ifndef ASCENDC_TILING_CONV3D_TILING_ALGORITHM_POINTWISE_H
#define ASCENDC_TILING_CONV3D_TILING_ALGORITHM_POINTWISE_H

#include <cstdint>
#include "conv3d_tiling_base.h"
#include "conv3d_tiling_algorithm.h"

namespace conv3d_tiling {

constexpr uint32_t POINT_WISE_BIAS_CUBE_UNIT = 8;
constexpr uint64_t POINT_WISE_ALIGN_UNIT = 16;

class Conv3dTilingAlgorithmPointWise : public Conv3dTilingAlgorithm {
public:
    explicit Conv3dTilingAlgorithmPointWise(Conv3dTilingBase *tilingIns) : Conv3dTilingAlgorithm(tilingIns) {
        this->l1TilingFlag.isWeightBypass = false;
    }
    virtual ~Conv3dTilingAlgorithmPointWise() {}

private:
    bool CoreL1TilingMinWeightBypass() const override;
    bool NoneKABL1FullLoadWeightBypass() const override;
    uint64_t L1NoFullLoadFmapSize() const override;
    bool CheckL1Buffer() const override;
    uint64_t CalcBTSize(uint64_t currnL0) const override;
    void WeightBypassDecision() override;
    int64_t InitCalcL1Params() override;
    void InitABL1TilingMode() override;
    void GetL1TilingRange() override;
    void GetKL0TilingDecision() override;
    uint64_t CalcL1SizeForL0Tiling(uint64_t currmL0, uint64_t currnL0) const override;
    void GetKL0TilingRangeCommon(uint64_t k0);
    void SetKAL1KBL1TailRes() override;
};
} // namespace conv3d_tiling

#endif