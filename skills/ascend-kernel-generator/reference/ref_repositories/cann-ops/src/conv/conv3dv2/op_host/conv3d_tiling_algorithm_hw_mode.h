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
 * \file conv3d_tiling_algorithm_hw_mode.h
 * \brief
 */

#ifndef ASCENDC_TILING_CONV3D_TILING_ALGORITHM_HW_MODE_H
#define ASCENDC_TILING_CONV3D_TILING_ALGORITHM_HW_MODE_H

#include <cstdint>
#include "conv3d_tiling_base.h"
#include "conv3d_tiling_algorithm.h"

namespace conv3d_tiling {

class Conv3dTilingAlgorithmHwMode : public Conv3dTilingAlgorithm {
public:
    explicit Conv3dTilingAlgorithmHwMode(Conv3dTilingBase *tilingIns) : Conv3dTilingAlgorithm(tilingIns) {}
    virtual ~Conv3dTilingAlgorithmHwMode() {}

private:
    using Conv3dTilingAlgorithm::CalcL1SizeForL0Tiling;
    
    // L0 tiling
    void InitPingPong() override;
    void GetL0TilingRange() override;
    void L0TilingDecision() override;
    uint64_t CalcL1SizeForL0Tiling(uint64_t currhoL0, uint64_t currwoL0, uint64_t currnL0) const;

    // L1 tiling
    uint64_t CalcCurrFmapL1Size() const override;
    int64_t InitCalcL1ParamsForFmap() override;
    void GetL1TilingRangeForM() override;
    void InitL1TiLingMap() override;
    void InitABL1TilingMode() override;
    int64_t ProcessAllL1FullLoad() override;
    int64_t ProcessFmapL1FullLoad() override;
    void MAL1IdxIter() override;
    int64_t KABL1FullLoadIter() override;
    uint64_t KABL1FullLoadIterN() override;
    void SetMAL1NBL1ValueAndMode() override;
};
} // namespace conv3d_tiling

#endif