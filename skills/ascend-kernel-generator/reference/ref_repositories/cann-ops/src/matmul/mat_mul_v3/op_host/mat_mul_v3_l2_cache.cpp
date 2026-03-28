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
 * \file matmul_v3_l2_cache.cc
 * \brief
 */
#include "mat_mul_v3_l2_cache.h"
#include "tiling/tiling_type.h"
using namespace optiling::matmul_v3;

namespace optiling {
namespace matmul_v3 {

constexpr uint32_t ALL_L2_ENABLE_BIT = 0;
constexpr uint32_t A_L2_DISABLE_BIT = 1;
constexpr uint32_t B_L2_DISABLE_BIT = 2;
constexpr uint32_t BIAS_L2_DISABLE_BIT = 3;
constexpr uint32_t C_L2_DISABLE_BIT = 4;

void L2Cache::SetL2CacheFlagBase(bool &aEnableL2Cache, bool &bEnableL2Cache) const
{
    auto &tileL2cache = tilingData_.tileL2cacheTiling;
    if (tileL2cache.get_mTileCntL2() > 1 || tileL2cache.get_nTileCntL2() > 1) {
        bEnableL2Cache = tileL2cache.get_mTileCntL2() > 1;
        aEnableL2Cache = tileL2cache.get_nTileCntL2() > 1;
        return;
    }

    auto &matmulTiling = tilingData_.matmulTiling;
    // m切多核
    if (static_cast<uint64_t>(matmulTiling.get_singleCoreM()) < args_.mValue) {
        bEnableL2Cache = true;
    }
    // n切多核
    if (static_cast<uint64_t>(matmulTiling.get_singleCoreN()) < args_.nValue) {
        aEnableL2Cache = true;
    }
    // 判断单核是否可以全载,此场景K可以全载
    uint64_t mCnt = MathUtil::CeilDivision(static_cast<uint64_t>(args_.mValue), matmulTiling.get_singleCoreM());
    uint64_t nCnt = MathUtil::CeilDivision(static_cast<uint64_t>(args_.nValue), matmulTiling.get_singleCoreN());
    uint64_t totalCnt = mCnt * nCnt;
    uint64_t round = MathUtil::CeilDivision(totalCnt, matmulTiling.get_usedCoreNum());
    bool mFullLoad = (round <= 1UL);
    bool nFullLoad = (round <= 1UL);
    if (!mFullLoad && !nFullLoad) {
        bEnableL2Cache = true;
    }
}

void L2Cache::SetL2CacheFlagSingleCoreSplitK(bool &aEnableL2Cache, bool &bEnableL2Cache) const
{
    auto &matmulTiling = tilingData_.matmulTiling;
    // m切多核
    bEnableL2Cache = static_cast<uint64_t>(matmulTiling.get_singleCoreM()) < args_.mValue;
    // n切多核
    aEnableL2Cache = static_cast<uint64_t>(matmulTiling.get_singleCoreN()) < args_.nValue;
    // 判断单核是否可以全载
    bool mFullLoad = (matmulTiling.get_singleCoreM() <= matmulTiling.get_baseM() * matmulTiling.get_stepM());
    bool nFullLoad = (matmulTiling.get_singleCoreN() <= matmulTiling.get_baseN() * matmulTiling.get_stepN());
    bool kFullLoad = (static_cast<uint64_t>(matmulTiling.get_singleCoreK()) <= args_.kValue);
    // M是外循环，与算子计算逻辑强相关
    if (!mFullLoad && !nFullLoad) {
        bEnableL2Cache = true;
        aEnableL2Cache = !kFullLoad;
    }
}

void L2Cache::SetL2CacheFlagMultiCoreSplitK(bool &aEnableL2Cache, bool &bEnableL2Cache) const
{
    auto &matmulTiling = tilingData_.matmulTiling;
    // m切多核
    bEnableL2Cache = static_cast<uint64_t>(matmulTiling.get_singleCoreM()) < args_.mValue;
    // 判断单核是否可以全载
    uint64_t mCnt = MathUtil::CeilDivision(static_cast<uint64_t>(args_.mValue), matmulTiling.get_singleCoreM());
    uint64_t kCnt = MathUtil::CeilDivision(static_cast<uint64_t>(args_.kValue), matmulTiling.get_singleCoreK());
    uint64_t totalCnt = mCnt * kCnt;
    uint64_t round = MathUtil::CeilDivision(totalCnt, matmulTiling.get_usedCoreNum());
    bool mFullLoad = (round <= 1UL);
    bool nFullLoad = (static_cast<uint64_t>(matmulTiling.get_singleCoreN()) == args_.nValue);
    bool kFullLoad = (round <= 1UL);
    // N是外循环，与算子计算逻辑强相关
    if (!mFullLoad && !nFullLoad) {
        aEnableL2Cache = true;
        bEnableL2Cache = !kFullLoad;
    }
}

void L2Cache::SetL2CacheFlag(bool aEnableL2Cache, bool bEnableL2Cache, bool cEnableL2Cache,
                             bool biasEnableL2Cache, uint32_t &l2CacheFlag)
{
    if (aEnableL2Cache && bEnableL2Cache && cEnableL2Cache && biasEnableL2Cache) {
        l2CacheFlag |= (1 << ALL_L2_ENABLE_BIT);
        OP_LOGD(args_.opName, "l2CacheFlag: %u", l2CacheFlag);
        return;
    }

    if (!aEnableL2Cache) {
        l2CacheFlag |= (1 << A_L2_DISABLE_BIT);
    }

    if (!bEnableL2Cache) {
        l2CacheFlag |= (1 << B_L2_DISABLE_BIT);
    }

    if (!cEnableL2Cache) {
        l2CacheFlag |= (1 << C_L2_DISABLE_BIT);
    }

    if (!biasEnableL2Cache) {
        l2CacheFlag |= (1 << BIAS_L2_DISABLE_BIT);
    }

    OP_LOGI(args_.opName, "l2CacheFlag: %u", l2CacheFlag);
}

void L2Cache::SetL2CacheFlag(TilingEnable tilingEnable, uint64_t l2Size, uint32_t &l2CacheFlag)
{
    bool aEnableL2Cache = false;
    bool bEnableL2Cache = false;
    bool cEnableL2Cache = false;
    bool biasEnableL2Cache = true;
    auto &matmulTiling = tilingData_.matmulTiling;

    OP_LOGI(args_.opName, "mValue: %lu nValue: %lu kValue: %lu "
        "singleCoreM: %d singleCoreN: %d singleCoreK: %d "
        "baseM: %d baseN: %d baseK: %d "
        "stepM: %d stepN: %d stepKa: %d stepKb: %d "
        "depthA1: %d depthB1: %d tilingEnableSplitCore %d tilingEnableFullLoad %d tilingEnableFixOpti %d",
        args_.mValue, args_.nValue, args_.kValue,
        matmulTiling.get_singleCoreM(), matmulTiling.get_singleCoreN(), matmulTiling.get_singleCoreK(),
        matmulTiling.get_baseM(), matmulTiling.get_baseN(), matmulTiling.get_baseK(),
        matmulTiling.get_stepM(), matmulTiling.get_stepN(), matmulTiling.get_stepKa(), matmulTiling.get_stepKb(),
        matmulTiling.get_depthA1(), matmulTiling.get_depthB1(),
        static_cast<int32_t>(tilingEnable.tilingEnableSplitCore),
        static_cast<int32_t>(tilingEnable.tilingEnableFullLoad),
        static_cast<int32_t>(tilingEnable.tilingEnableFixOpti));

    uint64_t sizeC = args_.mValue * args_.nValue * GetSizeByDataType(args_.cType);
    cEnableL2Cache = sizeC <= l2Size;

    switch (tilingEnable.tilingEnableSplitCore) {
        case TilingEnableSplitCore::BASE:
            SetL2CacheFlagBase(aEnableL2Cache, bEnableL2Cache);
            break;
        case TilingEnableSplitCore::SINGLE_CORE_SPLIT_K:
            cEnableL2Cache = true;
            SetL2CacheFlagSingleCoreSplitK(aEnableL2Cache, bEnableL2Cache);
            break;
        case TilingEnableSplitCore::DETERMINISTIC_SPLIT_K:
            cEnableL2Cache = true;
            SetL2CacheFlagMultiCoreSplitK(aEnableL2Cache, bEnableL2Cache);
            break;
        default:
            break;
    }

    SetL2CacheFlag(aEnableL2Cache, bEnableL2Cache, cEnableL2Cache, biasEnableL2Cache, l2CacheFlag);
}

}
}
