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
 * \file conv3d_dw_v2_basic_block_tiling.cpp
 * \brief
 */

#include "conv3d_dw_v2_basic_block_tiling.h"
#include <map>
#include <numeric>
#include "cube_tiling_runtime.h"
#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "tiling/tiling_templates_registry.h"
#include "tiling/tiling_type.h"

using namespace optiling::cachetiling;

namespace {
constexpr uint32_t DB_ON = 2;
constexpr uint32_t DB_OFF = 1;
constexpr const uint32_t ROW_FIRST = 1;
constexpr const uint32_t COL_FIRST = 2;
constexpr uint32_t CORE_NUM_910B3 = 20;

constexpr int32_t KERNEL_HW_4 = 4;
constexpr int32_t KERNEL_HW_9 = 9;
constexpr int32_t KERNEL_HW_16 = 16;
constexpr int32_t MIN_BATCHDIM = 4;
constexpr uint32_t NUM_HALF = 2;
constexpr uint32_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint32_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint32_t BASIC_BLOCK_SIZE_64 = 64;
constexpr uint32_t SPLIT_M_K = 1;
constexpr uint32_t SPLIT_N_K = 2;
constexpr uint32_t SPLIT_M_N = 3;
constexpr uint32_t L1_DEPTH_16 = 16;
constexpr uint32_t L1_DEPTH_8 = 8;
constexpr uint32_t L1_DEPTH_4 = 4;
constexpr uint32_t L1_DEPTH_2 = 2;
constexpr uint32_t STEP_2 = 2;
constexpr int32_t BLOCK_CUBE = static_cast<int32_t>(AscendC::BLOCK_CUBE);
constexpr uint64_t L0A_SIZE = 65536;
constexpr uint64_t L0B_SIZE = 65536;
constexpr uint64_t L0C_SIZE = 131072;
}  // namespace

namespace optiling {

ge::graphStatus Conv3DDWV2BasicBlockTiling::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();
    if (!GetTbeTiling()) {
        OP_LOGE(context_->GetNodeName(), "GetTbeTiling failed");
        return ge::GRAPH_FAILED;
    }
    TConv3DDwTiling &dwt = tilingData_.dwTiling;
    SetShapeTiling(dwt);
    SetAttrTiling(dwt);
    SetBasicBlockAttrsTiling();

    TilingValueDw tilingParams;
    InitTilingValue(tilingParams);
    SetTilingValue(dwt, tilingParams);

    return ge::GRAPH_SUCCESS;
}

void Conv3DDWV2BasicBlockTiling::SetBasicBlockAttrsTiling()
{
    uint32_t fractalSize0 = tilingData_.dwTiling.get_channelSize();
    mmInfo_.mValue = static_cast<uint64_t>(runInfo_.cout1_g) * fractalSize0;
    mmInfo_.nValue = static_cast<uint64_t>(runInfo_.kd) * runInfo_.kh * runInfo_.kw *
        runInfo_.cin1_g * fractalSize0;
    mmInfo_.kValue = ops::CeilAlign(static_cast<uint64_t>(runInfo_.ho) * runInfo_.wo,
                                    static_cast<uint64_t>(fractalSize0));
    blockTiling_.usedCoreNum = coreNum_;
}

bool Conv3DDWV2BasicBlockTiling::IsCapable() {
    // 当前仅支持1*1和在一个基本块以上大小的2*2 3*3 4*4卷积
    uint64_t kernelHW = static_cast<uint64_t>(runInfo_.kh) * runInfo_.kw;
    bool isKernelSupport = (kernelHW == 1)
        || ((kernelHW == KERNEL_HW_4 || kernelHW == KERNEL_HW_9 || kernelHW == KERNEL_HW_16)
            && mmInfo_.mValue >= BASIC_BLOCK_SIZE_128 && mmInfo_.nValue >= BASIC_BLOCK_SIZE_128);

    bool isBasicBlockSupport = isKernelSupport && dtypeByte_ == ge::GetSizeByDataType(ge::DT_BF16)
        && runInfo_.real_g == 1 && runInfo_.dilation_d == 1 && context_->GetDeterministic() == false;
    return isBasicBlockSupport;
}

void Conv3DDWV2BasicBlockTiling::UpdateSingleCoreInfo()
{
    // 搬运对齐时默认向下取整，避免越过基本块运算导致重新触发L1载入
    blockTiling_.singleCoreM = blockTiling_.stepM * blockTiling_.blockBaseM;

    uint64_t maxStepKL1 = std::max(blockTiling_.stepKa, blockTiling_.stepKb) * blockTiling_.blockBaseK;
    blockTiling_.singleCoreK = std::max(maxStepKL1 / runInfo_.wo, static_cast<uint64_t>(1)) * runInfo_.wo;

    uint64_t l1Cin1 = std::max(blockTiling_.stepN * blockTiling_.blockBaseN /
        (runInfo_.kh * runInfo_.kw * tilingData_.dwTiling.get_channelSize()), static_cast<uint32_t>(1));
    blockTiling_.singleCoreN = l1Cin1 * runInfo_.kh * runInfo_.kw * tilingData_.dwTiling.get_channelSize();

    if (blockTiling_.coreBindDirection == SPLIT_M_K) {
        blockTiling_.singleCoreN = mmInfo_.nValue;
    } else if (blockTiling_.coreBindDirection == SPLIT_N_K) {
        blockTiling_.singleCoreM = mmInfo_.mValue;
    } else {
        blockTiling_.singleCoreK = mmInfo_.kValue;
    }

    uint64_t mCnt = ops::CeilDiv(mmInfo_.mValue, static_cast<uint64_t>(blockTiling_.singleCoreM));
    uint64_t kCnt = ops::CeilDiv(mmInfo_.kValue, static_cast<uint64_t>(blockTiling_.singleCoreK));
    uint64_t nCnt = ops::CeilDiv(mmInfo_.nValue, static_cast<uint64_t>(blockTiling_.singleCoreN));
    blockTiling_.totalCnt = static_cast<uint64_t>(runInfo_.batch) * runInfo_.dout * mCnt * kCnt * nCnt;
}

void Conv3DDWV2BasicBlockTiling::InitBaseMNK()
{
    if (blockTiling_.coreBindDirection == SPLIT_M_N) {
        // 不切K算法主要是MTE1 Bound场景，L0A搬运效率是L0B两倍以上，优先让L0A填满
        if (mmInfo_.mValue > BASIC_BLOCK_SIZE_128) {
            blockTiling_.blockBaseM = BASIC_BLOCK_SIZE_256;
            blockTiling_.blockBaseN = BASIC_BLOCK_SIZE_128;
        } else {
            blockTiling_.blockBaseM = BASIC_BLOCK_SIZE_128;
            blockTiling_.blockBaseN = BASIC_BLOCK_SIZE_256;
        }
        blockTiling_.blockBaseK = BASIC_BLOCK_SIZE_64;
        blockTiling_.dbL0C = DB_OFF;
    } else {
        // 切K算法, 默认128基本块保证L0C能开PingPong, 否则会断流
        blockTiling_.blockBaseM = BASIC_BLOCK_SIZE_128;
        blockTiling_.blockBaseN = BASIC_BLOCK_SIZE_128;
        blockTiling_.blockBaseK = BASIC_BLOCK_SIZE_128;
        blockTiling_.dbL0C = DB_ON;
    }

    uint64_t fractalSize0 = tilingData_.dwTiling.get_channelSize();
    uint64_t aL0Max = blockTiling_.blockBaseK * blockTiling_.blockBaseM;
    uint64_t bL0Max = blockTiling_.blockBaseK * blockTiling_.blockBaseN;

    // M或N方向不够一个基本块，适应性调小BaseM或者StepM
    blockTiling_.blockBaseM = std::min(static_cast<uint64_t>(blockTiling_.blockBaseM), mmInfo_.mValue);
    blockTiling_.blockBaseN = std::min(static_cast<uint64_t>(blockTiling_.blockBaseN), mmInfo_.nValue);

    // K方向不够一个基本块，适应性调小BaseK，否则根据BaseM和BaseN的情况调大BaseK并进行搬运对齐
    uint64_t alignedKValue = ops::CeilAlign(mmInfo_.kValue, fractalSize0);
    if (alignedKValue < blockTiling_.blockBaseK) {
        blockTiling_.blockBaseK = alignedKValue;
    } else {
        // 根据调小后的BaseM和BaseN调大BaseK
        uint64_t newBaseKa = std::max(aL0Max / blockTiling_.blockBaseM / fractalSize0,
            static_cast<uint64_t>(1)) * fractalSize0;
        uint64_t newBaseKb = std::max(bL0Max / blockTiling_.blockBaseN / fractalSize0,
            static_cast<uint64_t>(1)) * fractalSize0;
        uint64_t newBaseK = std::min(std::min(newBaseKa, newBaseKb), alignedKValue);
        blockTiling_.blockBaseK = newBaseK;

        // K在不超过L0约束情况下，优先满足搬运对齐
        if (runInfo_.wo < static_cast<int32_t>(newBaseK) && runInfo_.wo % fractalSize0 == 0) {
            blockTiling_.blockBaseK = newBaseK / runInfo_.wo * runInfo_.wo;
        }
    }
}

void Conv3DDWV2BasicBlockTiling::UpdateStepMNK()
{
    if (blockTiling_.depthA1 < L1_DEPTH_2) {
        blockTiling_.dbL1A = DB_OFF;
    }
    if (blockTiling_.depthB1 < L1_DEPTH_2) {
        blockTiling_.dbL1B = DB_OFF;
    }

    uint64_t baseMK = blockTiling_.blockBaseM * blockTiling_.blockBaseK;
    uint64_t baseNK = blockTiling_.blockBaseN * blockTiling_.blockBaseK;
    uint64_t aL1Max = baseMK * blockTiling_.depthA1 / blockTiling_.dbL1A;
    uint64_t bL1Max = baseNK * blockTiling_.depthB1 / blockTiling_.dbL1B;

    uint64_t maxMIter = ops::CeilDiv(mmInfo_.mValue, static_cast<uint64_t>(blockTiling_.blockBaseM));
    uint64_t maxNIter = ops::CeilDiv(mmInfo_.nValue, static_cast<uint64_t>(blockTiling_.blockBaseN));
    uint64_t maxKIter = ops::CeilDiv(mmInfo_.kValue, static_cast<uint64_t>(blockTiling_.blockBaseK));
    uint64_t minIter = 1;

    // 根据预置的StepM/StepN初始化StepKa和StepKb, 不超过K方向最大循环次数
    blockTiling_.stepKa = std::max(std::min(aL1Max / baseMK / blockTiling_.stepM, maxKIter), minIter);
    blockTiling_.stepKb = std::max(std::min(bL1Max / baseNK / blockTiling_.stepN, maxKIter), minIter);

    // 驻留的一侧允许适应性调整非K方向载入量，不超过最大循环次数
    if (blockTiling_.coreBindDirection == SPLIT_M_K) {
        blockTiling_.stepM = std::max(std::min(aL1Max / baseMK / blockTiling_.stepKa, maxMIter), minIter);
    } else if (blockTiling_.coreBindDirection == SPLIT_N_K) {
        blockTiling_.stepN = std::max(std::min(bL1Max / baseNK / blockTiling_.stepKb, maxNIter), minIter);
    }

    // 根据调整后的StepM和StepN调整StepKa和StepKb, 不超过K方向最大循环次数
    blockTiling_.stepKa = std::max(std::min(aL1Max / baseMK / blockTiling_.stepM, maxKIter), minIter);
    blockTiling_.stepKb = std::max(std::min(bL1Max / baseNK / blockTiling_.stepN, maxKIter), minIter);

    if (blockTiling_.coreBindDirection == SPLIT_M_K) {
        blockTiling_.stepKa = std::max(blockTiling_.stepKa / blockTiling_.stepKb, static_cast<uint32_t>(1)) * blockTiling_.stepKb;
    } else if (blockTiling_.coreBindDirection == SPLIT_N_K) {
        blockTiling_.stepKb = std::max(blockTiling_.stepKb / blockTiling_.stepKa, static_cast<uint32_t>(1)) * blockTiling_.stepKa;
    } else {
        if (blockTiling_.stepKa > blockTiling_.stepKb) {
            blockTiling_.stepKa = std::max(blockTiling_.stepKa / blockTiling_.stepKb, static_cast<uint32_t>(1)) * blockTiling_.stepKb;
        }
        if (blockTiling_.stepKb > blockTiling_.stepKa) {
            blockTiling_.stepKb = std::max(blockTiling_.stepKb / blockTiling_.stepKa, static_cast<uint32_t>(1)) * blockTiling_.stepKa;
        }
    }
}

void Conv3DDWV2BasicBlockTiling::MultiCoreSplitK()
{
    blockTiling_.iterateOrder = mmInfo_.mValue > mmInfo_.nValue ? 1 : 0;
    blockTiling_.coreBindDirection = mmInfo_.mValue > mmInfo_.nValue ? SPLIT_N_K : SPLIT_M_K;
    InitBaseMNK();

    // 默认24算法，depthA1/B1都为4*1*DB_ON, L1占用128*128*2*16=512KB
    // 默认策略，一侧pingpong全载后驻留，另一侧pingpong交替载入
    blockTiling_.coreBindOrder = ROW_FIRST;
    blockTiling_.dbL1A = DB_ON;
    blockTiling_.dbL1B = DB_ON;

    // L1配比算法，按照8+8,8+4,4+4,4+2,2+2,2+1,1+1进行阶梯匹配
    uint32_t depthA1 = L1_DEPTH_8;
    uint32_t depthB1 = L1_DEPTH_8;
    while (depthA1 >= 1 && depthB1 >= 1) {
        blockTiling_.depthA1 = depthA1;
        blockTiling_.depthB1 = depthB1;
        blockTiling_.stepM = 1;
        blockTiling_.stepN = 1;
        UpdateStepMNK();
        UpdateSingleCoreInfo();
        if (blockTiling_.totalCnt >= blockTiling_.usedCoreNum && !IsCurBlockL1Invalid()) {
            break;
        }
        // 大的一侧L1先减半，相等时先减少非驻留侧的L1占用
        if (depthA1 == 1 && depthB1 == 1) {
            break;
        } else if (depthA1 > depthB1) {
            depthA1 /= NUM_HALF;
        } else if (depthB1 > depthA1) {
            depthB1 /= NUM_HALF;
        } else if (blockTiling_.coreBindDirection == SPLIT_M_K) {
            depthB1 /= NUM_HALF;
        } else {
            depthA1 /= NUM_HALF;
        }
    }

    // 合法性兜底，防止w一次要搬运的过大，直接超L1
    if (IsCurBlockL1Invalid()) {
        ShrinkBaseBlock();
        UpdateStepMNK();
        UpdateSingleCoreInfo();
    }

    if (blockTiling_.totalCnt < blockTiling_.usedCoreNum) {
        blockTiling_.usedCoreNum = blockTiling_.totalCnt;
    }
}

void Conv3DDWV2BasicBlockTiling::MultiCoreSplitMN()
{
    blockTiling_.iterateOrder = mmInfo_.mValue > mmInfo_.nValue ? 1 : 0;
    blockTiling_.coreBindDirection = SPLIT_M_N;
    InitBaseMNK();

    // 默认策略，一侧pingpong全载后驻留，另一侧pingpong交替载入
    blockTiling_.coreBindOrder = ROW_FIRST;
    blockTiling_.dbL1A = DB_ON;
    blockTiling_.dbL1B = DB_ON;

    // L1配比算法，按照16个块往下进行对称阶梯衰减
    uint32_t depthA1 = L1_DEPTH_16;
    uint32_t depthB1 = L1_DEPTH_16;
    while (depthA1 >= 1 && depthB1 >= 1) {
        blockTiling_.depthA1 = depthA1;
        blockTiling_.depthB1 = depthB1;
        blockTiling_.stepM = 1;
        blockTiling_.stepN = 1;
        UpdateStepMNK();
        if (!IsCurBlockL1Invalid()) {
            break;
        }
        depthA1 = depthA1 > STEP_2 ? (depthA1 - STEP_2) : (depthA1 - 1);
        depthB1 = depthB1 > STEP_2 ? (depthB1 - STEP_2) : (depthB1 - 1);
    }

    // 合法性兜底，防止w一次要搬运的过大，直接超L1
    if (IsCurBlockL1Invalid()) {
        ShrinkBaseBlock();
        UpdateStepMNK();
    }

    UpdateSingleCoreInfo();
    if (blockTiling_.totalCnt < blockTiling_.usedCoreNum) {
        MultiCoreSplitK();
    }
}

uint64_t Conv3DDWV2BasicBlockTiling::CalculateL1SizeGap()
{
    uint32_t al1LoadSize = blockTiling_.blockBaseK * blockTiling_.blockBaseM * dtypeByte_;
    uint32_t bl1LoadSize = CalBL1Bound() * dtypeByte_;
    uint64_t deltaL1LoadSize = (al1LoadSize + bl1LoadSize > tilingData_.params.get_totalL1Size()) ?
                                al1LoadSize + bl1LoadSize - tilingData_.params.get_totalL1Size() : 0;
    return deltaL1LoadSize;
}

uint32_t Conv3DDWV2BasicBlockTiling::CalculateBl1Cin1CopyLen(uint32_t newBaseN)
{
    uint32_t kernelHW = static_cast<uint32_t>(runInfo_.kh * runInfo_.kw);
    uint32_t bL1N = ops::CeilDiv(newBaseN, tilingData_.dwTiling.get_channelSize());
    uint32_t bL1Cin1CopyLen = ops::CeilDiv(bL1N, kernelHW); // 向上取整，拖尾时默认多搬一行
    if (kernelHW > bL1N && kernelHW % bL1N != 0) {
        ++bL1Cin1CopyLen; // 此时bL1Cin1CopyLen为1, 每个基本块不足一行，考虑拖尾最多搬两行
    } else if (NUM_HALF * bL1N % kernelHW != 0) {
        ++bL1Cin1CopyLen; // 除了尾块是0.5，其他场景都要搬2行
    }
    return bL1Cin1CopyLen;
}

bool Conv3DDWV2BasicBlockTiling::ShrinkBlockBaseK()
{
    // k方向减小
    uint64_t fractalSize0 = tilingData_.dwTiling.get_channelSize();
    uint64_t deltaL1LoadSize = CalculateL1SizeGap();
    // 基本块K方向每减小C0, L1A装载大小减小deltaAl1PerC0
    uint64_t deltaAl1PerC0 = blockTiling_.blockBaseM * fractalSize0 * dtypeByte_;
    
    uint32_t bL1Cin1CopyLen = CalculateBl1Cin1CopyLen(blockTiling_.blockBaseN);
    // 基本块K方向每减小C0, L1B装载大小减小deltaAl1PerC0, 本身这个过程是阶跃的, 此处做线性处理
    uint64_t deltaBl1PerC0 = ops::CeilDiv(bL1Cin1CopyLen * fractalSize0 * runInfo_.wi * runInfo_.stride_h
                                            * fractalSize0 * dtypeByte_, static_cast<uint64_t>(runInfo_.wo));
    
    // 线性处理后, deltaBl1PerC0一定不小于实际每C0减小, 所以c0ShrinkCount不会大于实际需减小C0数量
    uint64_t c0ShrinkCount = ops::CeilDiv(deltaL1LoadSize, deltaAl1PerC0 + deltaBl1PerC0);
    uint64_t newBaseK = 0;
    if (blockTiling_.blockBaseK > c0ShrinkCount * fractalSize0) {
        newBaseK = blockTiling_.blockBaseK - c0ShrinkCount * fractalSize0;
    }
    if (newBaseK >= fractalSize0) {
        blockTiling_.blockBaseK = newBaseK;
        while(blockTiling_.blockBaseK > fractalSize0 && IsCurBlockL1Invalid()) {
            blockTiling_.blockBaseK -= fractalSize0;
            if (static_cast<int32_t>(blockTiling_.blockBaseK) <= runInfo_.wo
                && (runInfo_.wo % blockTiling_.blockBaseK == 0 || runInfo_.wo % fractalSize0 != 0)) {
                break;
            }
        }
        if (!IsCurBlockL1Invalid()) {
            return true;
        }
    } else {
        blockTiling_.blockBaseK = fractalSize0;
    }
    return false;
}

void Conv3DDWV2BasicBlockTiling::ShrinkBlockBaseMN()
{
    uint64_t kernelHW = static_cast<uint64_t>(runInfo_.kh * runInfo_.kw);
    uint64_t fractalSize0 = tilingData_.dwTiling.get_channelSize();
    // M和N方向减小, 首先让M和N大小平齐
    while (blockTiling_.blockBaseM > fractalSize0 && blockTiling_.blockBaseM > blockTiling_.blockBaseN 
            && IsCurBlockL1Invalid()) {
        blockTiling_.blockBaseM -= fractalSize0;
    }
    while (blockTiling_.blockBaseN > fractalSize0 && blockTiling_.blockBaseN > blockTiling_.blockBaseM
            && IsCurBlockL1Invalid()) {
        blockTiling_.blockBaseN -= fractalSize0;
    }
    if (!IsCurBlockL1Invalid()) {
        return;
    }
    uint64_t deltaAl1PerC0 = blockTiling_.blockBaseK * fractalSize0 * dtypeByte_;
    int32_t hoCal = 0;
    int32_t kBl1Size = blockTiling_.blockBaseK * blockTiling_.stepKb;
    if (kBl1Size % runInfo_.wo == 0 || runInfo_.wo % kBl1Size == 0) {
        hoCal = ops::CeilDiv(kBl1Size, runInfo_.wo);
    } else if (kBl1Size > runInfo_.wo) {
        hoCal = kBl1Size / runInfo_.wo + NUM_HALF;
    } else {
        hoCal = NUM_HALF;
    }
    int32_t hiCal = (hoCal - 1) * runInfo_.stride_h + (runInfo_.kh - 1) * runInfo_.dilation_h + 1;
    // 与K方向减小采用同样思路, 做线性化处理
    uint64_t deltaBl1PerC0 = ops::CeilDiv(static_cast<uint64_t>(hiCal) * runInfo_.wi * fractalSize0 * dtypeByte_, kernelHW);
    uint64_t deltaL1LoadSize = CalculateL1SizeGap();
    uint32_t c0ShrinkCount = ops::CeilDiv(deltaL1LoadSize, deltaAl1PerC0 + deltaBl1PerC0);
    if (static_cast<uint64_t>(blockTiling_.blockBaseM) < (c0ShrinkCount + 1) * fractalSize0) {
        blockTiling_.blockBaseM = fractalSize0;
        blockTiling_.blockBaseN = fractalSize0;
        return;
    }
    blockTiling_.blockBaseM -= (c0ShrinkCount * fractalSize0);
    blockTiling_.blockBaseN = blockTiling_.blockBaseM;
    uint32_t bL1Cin1CopyLen = CalculateBl1Cin1CopyLen(blockTiling_.blockBaseN);
    
    while (blockTiling_.blockBaseM > fractalSize0 && IsCurBlockL1Invalid()) {
        uint32_t newBl1Cin1CopyLen = CalculateBl1Cin1CopyLen(blockTiling_.blockBaseM);// 向上取整，拖尾时默认多搬一行
        if (newBl1Cin1CopyLen < bL1Cin1CopyLen) {
            blockTiling_.blockBaseN = blockTiling_.blockBaseM;
            bL1Cin1CopyLen = newBl1Cin1CopyLen;
        } else {
            blockTiling_.blockBaseM -= fractalSize0;
        }
    }
}

void Conv3DDWV2BasicBlockTiling::ShrinkBaseBlock()
{
    if (ShrinkBlockBaseK()) {
        return;
    }
    ShrinkBlockBaseMN();
    
    // M方向回调
    uint64_t fractalSize0 = tilingData_.dwTiling.get_channelSize();
    uint32_t al1LoadSize = blockTiling_.stepKa * blockTiling_.blockBaseK * blockTiling_.stepM *
                           blockTiling_.blockBaseM * dtypeByte_ * blockTiling_.dbL1A;
    uint32_t bl1LoadSize = CalBL1Bound() * dtypeByte_ * blockTiling_.dbL1B;
    uint64_t deltaL1LoadSize = tilingData_.params.get_totalL1Size() - al1LoadSize - bl1LoadSize;
    uint64_t deltaAl1PerC0M = blockTiling_.blockBaseK * fractalSize0 * dtypeByte_;
    uint64_t c0CompensateCountM = deltaL1LoadSize / deltaAl1PerC0M;
    uint64_t cL0Max = L0C_SIZE / dtypeByte_ / DB_ON;
    uint64_t newBaseMc = std::max(cL0Max / blockTiling_.blockBaseM / fractalSize0,
            static_cast<uint64_t>(1)) * fractalSize0;
    blockTiling_.blockBaseM = std::min(blockTiling_.blockBaseM + c0CompensateCountM * fractalSize0, mmInfo_.mValue);
    blockTiling_.blockBaseM = std::min(newBaseMc, static_cast<uint64_t>(blockTiling_.blockBaseM));
    // K方向回调
    uint32_t validBaseK = blockTiling_.blockBaseK;
    while (!IsCurBlockL1Invalid()) {
        validBaseK = blockTiling_.blockBaseK;
        blockTiling_.blockBaseK += fractalSize0;
    }
    blockTiling_.blockBaseK = validBaseK;

    uint64_t aL0Max = L0A_SIZE / dtypeByte_ / DB_ON;
    uint64_t bL0Max = L0B_SIZE / dtypeByte_ / DB_ON;

    uint64_t alignedKValue = ops::CeilAlign(mmInfo_.kValue, fractalSize0);
    if (alignedKValue < blockTiling_.blockBaseK) {
        blockTiling_.blockBaseK = alignedKValue;
    } else {
        // 根据调小后的BaseM和BaseN调大BaseK
        uint64_t newBaseKa = std::max(aL0Max / blockTiling_.blockBaseM / fractalSize0,
            static_cast<uint64_t>(1)) * fractalSize0;
        uint64_t newBaseKb = std::max(bL0Max / blockTiling_.blockBaseN / fractalSize0,
            static_cast<uint64_t>(1)) * fractalSize0;
        uint64_t newBaseK = std::min(std::min(newBaseKa, newBaseKb), alignedKValue);
        blockTiling_.blockBaseK = std::min(newBaseK, static_cast<uint64_t>(blockTiling_.blockBaseK));

        // K在不超过L0约束情况下，优先满足搬运对齐
        if (runInfo_.wo < static_cast<int32_t>(blockTiling_.blockBaseK) && runInfo_.wo % fractalSize0 == 0) {
            blockTiling_.blockBaseK = blockTiling_.blockBaseK / runInfo_.wo * runInfo_.wo;
        }
    }
}

uint64_t Conv3DDWV2BasicBlockTiling::IsCurBlockL1Invalid()
{
    uint32_t al1LoadSize = blockTiling_.stepKa * blockTiling_.blockBaseK * blockTiling_.stepM *
                           blockTiling_.blockBaseM * dtypeByte_ * blockTiling_.dbL1A;
    uint32_t bl1LoadSize = CalBL1Bound() * dtypeByte_ * blockTiling_.dbL1B;
    bool invalidL1LoadSize = al1LoadSize + bl1LoadSize > tilingData_.params.get_totalL1Size();
    return invalidL1LoadSize;
}

uint64_t Conv3DDWV2BasicBlockTiling::CalBL1Bound()
{
    int32_t hoCal = 0;
    int32_t kBl1Size = blockTiling_.blockBaseK * blockTiling_.stepKb;
    if (kBl1Size % runInfo_.wo == 0 || runInfo_.wo % kBl1Size == 0) {
        hoCal = ops::CeilDiv(kBl1Size, runInfo_.wo);
    } else if (kBl1Size > runInfo_.wo) {
        hoCal = kBl1Size / runInfo_.wo + NUM_HALF;
    } else {
        hoCal = NUM_HALF;
    }
    int32_t hiCal = (hoCal - 1) * runInfo_.stride_h + (runInfo_.kh - 1) * runInfo_.dilation_h + 1;
    uint32_t kernelHW = static_cast<uint32_t>(runInfo_.kh * runInfo_.kw);
    uint32_t bL1N = ops::CeilDiv(blockTiling_.stepN * blockTiling_.blockBaseN, tilingData_.dwTiling.get_channelSize());
    uint32_t bL1Cin1CopyLen = ops::CeilDiv(bL1N, kernelHW); // 向上取整，拖尾时默认多搬一行
    if (kernelHW > bL1N && kernelHW % bL1N != 0) {
        ++bL1Cin1CopyLen; // 此时bL1Cin1CopyLen为1, 每个基本块不足一行，考虑拖尾最多搬两行
    } else if (NUM_HALF * bL1N % kernelHW != 0) {
        ++bL1Cin1CopyLen; // 除了尾块是0.5，其他场景都要搬2行
    }
    uint64_t bL1Size = static_cast<uint64_t>(hiCal) * runInfo_.wi * bL1Cin1CopyLen *
                       tilingData_.dwTiling.get_channelSize();
    return bL1Size;
}

ge::graphStatus Conv3DDWV2BasicBlockTiling::DoOpTiling()
{
    uint64_t strideHW = static_cast<uint64_t>(runInfo_.stride_h) * runInfo_.stride_w;
    uint64_t kernelHW = static_cast<uint64_t>(runInfo_.kh) * runInfo_.kw;
    // MTE1压力较大且N方向满足搬运对齐，切MN
    if (strideHW > 1 && (kernelHW % BLOCK_CUBE == 0 || BLOCK_CUBE % kernelHW == 0)) {
        MultiCoreSplitMN();
    } else {
        MultiCoreSplitK();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3DDWV2BasicBlockTiling::DoLibApiTiling()
{
    TConv3DDwTiling &dwt = tilingData_.dwTiling;
    tilingData_.basicBlockTiling.set_usedCoreNum(blockTiling_.usedCoreNum);
    tilingData_.basicBlockTiling.set_singleCoreM(blockTiling_.singleCoreM);
    tilingData_.basicBlockTiling.set_singleCoreN(blockTiling_.singleCoreN);
    tilingData_.basicBlockTiling.set_singleCoreK(blockTiling_.singleCoreK);
    tilingData_.basicBlockTiling.set_coreBindOrder(blockTiling_.coreBindOrder);

    dwt.set_singleCoreHo(blockTiling_.singleCoreK / runInfo_.wo);
    dwt.set_baseM(blockTiling_.blockBaseM);
    dwt.set_baseN(blockTiling_.blockBaseN);
    dwt.set_baseK(blockTiling_.blockBaseK);
    dwt.set_stepM(blockTiling_.stepM);
    dwt.set_stepN(blockTiling_.stepN);
    dwt.set_stepKa(blockTiling_.stepKa);
    dwt.set_stepKb(blockTiling_.stepKb);
    dwt.set_iterateOrder(blockTiling_.iterateOrder);
    dwt.set_al1Pbuffer(blockTiling_.dbL1A);
    dwt.set_bl1Pbuffer(blockTiling_.dbL1B);
    dwt.set_cl0Pbuffer(blockTiling_.dbL0C);
    tilingData_.dwTiling.set_bl1Bound(CalBL1Bound());
    dwt.set_singleCoreCout(blockTiling_.singleCoreM);

    uint64_t l1Cin1 = std::max(blockTiling_.singleCoreN /
        (runInfo_.kh * runInfo_.kw * tilingData_.dwTiling.get_channelSize()), static_cast<uint64_t>(1));
    dwt.set_singleCoreCin(l1Cin1 * tilingData_.dwTiling.get_channelSize());

    PrintBasickBlockTilingData();
    return ge::GRAPH_SUCCESS;
}

uint64_t Conv3DDWV2BasicBlockTiling::GetTilingKey() const {
    return blockTiling_.coreBindDirection;
}

ge::graphStatus Conv3DDWV2BasicBlockTiling::PostTiling()
{
    OP_LOGD(opName_, "final tiling data size: %zu", tilingData_.GetDataSize());

    OP_TILING_CHECK(tilingData_.GetDataSize() % sizeof(uint64_t) != 0,
                    CUBE_INNER_ERR_REPORT(opName_, "tiling data size[%zu] not aligned to 8", tilingData_.GetDataSize()),
                    return ge::GRAPH_FAILED);
    context_->SetBlockDim(tilingData_.basicBlockTiling.get_usedCoreNum());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

void Conv3DDWV2BasicBlockTiling::PrintBasickBlockTilingData()
{
    Conv3DBackpropFilterV2Tiling::PrintTilingData();
    TConv3DDwBasicBlockTiling &tiling = tilingData_.basicBlockTiling;
    std::stringstream ss;
    ss << " singleCoreM: " << tiling.get_singleCoreM() << " singleCoreN: " << tiling.get_singleCoreN()
        << " singleCoreK: " << tiling.get_singleCoreK() << " coreBindOrder: " << tiling.get_coreBindOrder()
        << " usedCoreNum: " << tiling.get_usedCoreNum();
}

REGISTER_TILING_TEMPLATE("Conv3DBackpropFilterV2", Conv3DDWV2BasicBlockTiling, 0);

}  // namespace optiling
