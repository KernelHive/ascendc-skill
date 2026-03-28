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
 * \file weight_quant_batch_matmul_v2_adaptive_split_tiling.cpp
 * \brief
 */

#include "weight_quant_batch_matmul_v2_adaptive_split_tiling.h"
#include "weight_quant_batch_matmul_v2_tiling_key.h"

namespace {
struct CubeSplitResult {
    uint64_t mte2Cost;
    uint64_t cubeBlockDimM;
    uint64_t cubeBlockDimN;
};
}  // namespace

namespace optiling {
constexpr int32_t DOUBLE_BUFFER_NUM = 2;
constexpr int32_t QUARTER_BUFFER_NUM = 4;
constexpr int32_t SINGLE_BUFFER_NUM = 1;
constexpr uint64_t M_MAX_SIZE = 256UL;
constexpr uint64_t L1_N_MAX_SIZE = 256UL;
constexpr uint64_t VECTOR_REG_WIDTH = 256UL;
constexpr uint64_t M_MAX_SIZE_WITH_BIAS_QUANT = 240UL;
constexpr uint64_t INT4_CORRECTION_FACTOR = 2UL;
constexpr uint64_t KILOBYTE = 1024UL;
constexpr uint64_t NUM_TWO = 2UL;
constexpr uint64_t UB_SIZE_V100 = 248 * 1024; // 当前框架获取的UB空间为240KB，问题解决后删除
constexpr int32_t A_L1_MAX_SIZE = 256 * 1024;
constexpr int32_t A_L1_MAX_SIZE_WITH_BIAS_QUANT = 240 * 1024;

ge::graphStatus WeightQuantBatchMatmulV2TilingAS::PostTiling()
{
    OP_LOGD(opName_, "final tiling data size: %zu", tilingData_->GetDataSize());

    OP_TILING_CHECK(
        tilingData_->GetDataSize() % sizeof(uint64_t) != 0,
        VECTOR_INNER_ERR_REPORT_TILIING(opName_, "tiling data size[%zu] not aligned to 8", tilingData_->GetDataSize()),
        return ge::GRAPH_FAILED);

    context_->GetRawTilingData()->SetDataSize(tilingData_->GetDataSize());
    // 计算aic num n方向分核*m方向分核
    context_->SetBlockDim(tilingData_->get_cubeBlockDimM() * tilingData_->get_cubeBlockDimN());
    tilingData_->SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    return ge::GRAPH_SUCCESS;
}

bool WeightQuantBatchMatmulV2TilingAS::IsCapable()
{
    return true;
}

ge::graphStatus WeightQuantBatchMatmulV2TilingAS::InstantiateTilingData()
{
    if (tilingData_ == nullptr) {
        try {
            tilingData_ = std::make_unique<WeightQuantBatchMatmulV2ASTilingData>();
        } catch (std::bad_alloc &) {
            tilingData_ = nullptr;
            OP_LOGE(opName_, "tiling data memory allocation failed");
            return ge::GRAPH_FAILED;
        }
    }
    OP_TILING_CHECK(tilingData_ == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "failed to instantiate tilingData"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        context_->GetRawTilingData()->GetCapacity() < tilingData_->GetDataSize(),
        VECTOR_INNER_ERR_REPORT_TILIING(opName_, "tiling data capacity %zu < actual tiling data size %zu",
                                        context_->GetRawTilingData()->GetCapacity(), tilingData_->GetDataSize()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantBatchMatmulV2TilingAS::DoOpTiling()
{
    OP_TILING_CHECK(InstantiateTilingData() == ge::GRAPH_FAILED,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "unable to get pointer of tiling data"),
                    return ge::GRAPH_FAILED);
    tilingData_->set_mSize(matmulInfoPtr_->mSize);
    tilingData_->set_kSize(matmulInfoPtr_->kSize);
    tilingData_->set_nSize(matmulInfoPtr_->nSize);
    tilingData_->set_hasBias(matmulInfoPtr_->hasBias);

    bool highPerfFlag = CheckHighPerfScence();
    ComputeCubeTiling(highPerfFlag);
    if (highPerfFlag) {
        ComputeTailResplitTiling();
    } else {
        ComputeBasicTiling();
    }

    return ge::GRAPH_SUCCESS;
}

void WeightQuantBatchMatmulV2TilingAS::ComputeCubeTiling(bool highPerfFlag)
{
    SetDefaultMatmulTiling();
    ComputeCubeSplit(highPerfFlag);
    OptimizeMatmulTiling();
}

void WeightQuantBatchMatmulV2TilingAS::SetDefaultMatmulTiling()
{
    tilingData_->matmulTiling.set_M(matmulInfoPtr_->mSize);
    tilingData_->matmulTiling.set_Ka(matmulInfoPtr_->kSize);
    tilingData_->matmulTiling.set_N(matmulInfoPtr_->nSize);
    tilingData_->matmulTiling.set_Kb(matmulInfoPtr_->kSize);
    if (matmulInfoPtr_->hasBias) {
        tilingData_->matmulTiling.set_singleCoreM(M_MAX_SIZE_WITH_BIAS_QUANT);
    } else {
        tilingData_->matmulTiling.set_singleCoreM(M_MAX_SIZE);
    }
    tilingData_->matmulTiling.set_singleCoreK(matmulInfoPtr_->kSize);
    tilingData_->matmulTiling.set_singleCoreN(L1_N_MAX_SIZE);

    tilingData_->matmulTiling.set_baseM(tilingData_->matmulTiling.get_singleCoreM());
    tilingData_->matmulTiling.set_baseN(tilingData_->matmulTiling.get_singleCoreN());
    tilingData_->matmulTiling.set_baseK((compileInfoPtr_->l0aSize >> 1) / GetSizeByDataType(matmulInfoPtr_->aDtype) /
                                        tilingData_->matmulTiling.get_singleCoreN());
    tilingData_->matmulTiling.set_dbL0A(DOUBLE_BUFFER_NUM);
    tilingData_->matmulTiling.set_dbL0B(DOUBLE_BUFFER_NUM);
    tilingData_->matmulTiling.set_dbL0C(SINGLE_BUFFER_NUM);

    tilingData_->matmulTiling.set_stepM(1);
    tilingData_->matmulTiling.set_stepN(1);

    tilingData_->matmulTiling.set_stepKa(
        ops::CeilDiv(L1_N_MAX_SIZE, static_cast<uint64_t>(tilingData_->matmulTiling.get_baseK())));
    tilingData_->matmulTiling.set_depthA1(tilingData_->matmulTiling.get_stepKa() * DOUBLE_BUFFER_NUM);

    tilingData_->matmulTiling.set_stepKb(tilingData_->matmulTiling.get_stepKa());
    tilingData_->matmulTiling.set_depthB1(tilingData_->matmulTiling.get_depthA1());
    tilingData_->matmulTiling.set_iterateOrder(0);

    tilingData_->matmulTiling.set_isBias(static_cast<int32_t>(matmulInfoPtr_->hasBias));
    if (matmulInfoPtr_->cDtype == ge::DT_INT8) {
        tilingData_->matmulTiling.set_shareL1Size(DOUBLE_BUFFER_NUM * tilingData_->matmulTiling.get_baseN() *
                                                  sizeof(uint64_t));
    }
    tilingData_->matmulTiling.set_shareL0CSize(tilingData_->matmulTiling.get_baseM() *
                                               tilingData_->matmulTiling.get_baseN() * sizeof(float));
}

void WeightQuantBatchMatmulV2TilingAS::ComputeCubeSplit(bool highPerfFlag)
{
    tilingData_->set_aPreloadSize(0);
    if (highPerfFlag) {
        ComputeHighPerfSceneCubeSplit();
        // m方向有切分场景或非cacheline对齐，启用缓存
        int32_t minCacheLine = 128; // 缓存大小128B，对应int4为128个元素
        if (matmulInfoPtr_->bDtype == ge::DT_INT4) {
            minCacheLine = 256; // 缓存大小128B，对应int4为256个元素
        }
        tilingData_->set_weightL2Cacheable(tilingData_->get_cubeBlockDimM() > 1 ||
                                           tilingData_->get_kSize() % minCacheLine  != 0);
        return;
    }

    tilingData_->set_weightL2Cacheable(1);  // 默认开启l2缓存功能
    uint64_t mainBlockL1SizeDefault = L1_N_MAX_SIZE;
    uint32_t mainBlockCountDefault = ops::CeilDiv(matmulInfoPtr_->nSize, mainBlockL1SizeDefault);

    uint32_t cubeBlockDimMMax = ops::CeilDiv(matmulInfoPtr_->mSize, static_cast<uint64_t>(M_MAX_SIZE));
    if (matmulInfoPtr_->transB) {
        uint32_t halfAicNum = compileInfoPtr_->aicNum >> 1;
        if (mainBlockCountDefault <= halfAicNum) {
            // 分核只有一半，尝试切分粒度缩减后再重新切分
            mainBlockL1SizeDefault = mainBlockL1SizeDefault >> 1;
            mainBlockCountDefault = ops::CeilDiv(matmulInfoPtr_->nSize, mainBlockL1SizeDefault);
        }

        tilingData_->set_cubeBlockDimN(std::min(compileInfoPtr_->aicNum, mainBlockCountDefault));
        if (cubeBlockDimMMax <= (compileInfoPtr_->aicNum / tilingData_->get_cubeBlockDimN()) >> 1) {
            // 分核只够剩下可用核数量一半，尝试切分粒度缩减后再重新切分
            tilingData_->set_cubeBlockDimM(ops::CeilDiv(matmulInfoPtr_->mSize, static_cast<uint64_t>(M_MAX_SIZE) >> 1));
        } else {
            tilingData_->set_cubeBlockDimM(std::min(
                cubeBlockDimMMax, static_cast<uint32_t>(compileInfoPtr_->aicNum) / tilingData_->get_cubeBlockDimN()));
        }
    } else {
        tilingData_->set_cubeBlockDimN(std::min(compileInfoPtr_->aicNum, mainBlockCountDefault));
        tilingData_->set_cubeBlockDimM(
            std::min(cubeBlockDimMMax, compileInfoPtr_->aicNum / tilingData_->get_cubeBlockDimN()));
    }
}

void WeightQuantBatchMatmulV2TilingAS::ComputeHighPerfSceneCubeSplit()
{
    if ((matmulInfoPtr_->mSize <= M_MAX_SIZE && !matmulInfoPtr_->hasBias) ||
        (matmulInfoPtr_->mSize <= M_MAX_SIZE_WITH_BIAS_QUANT && matmulInfoPtr_->hasBias)) {
        /*
        * 增量场景下需要讨论cube以何种方式切分
        * 根据实验数据，weight切分后n=128的处理数据量在大部分场景下可以发挥带宽优势。
        * 因此从切分粒度的角度看，n切分后小于该阈值，或者m大于该阈值的场景，只在n方向分核可能导致整体搬运量过大(A矩阵的占比较高)。
        * 此时需要遍历可能存在的切分方式，找到理论最小搬运量的解。
        * 在其他场景下，默认只在n方向分核即可。
        */
        static const uint64_t tailL1SizeThreshold = 128UL;
        bool needPollingFlag = matmulInfoPtr_->nSize < tailL1SizeThreshold * compileInfoPtr_->aicNum ||
            (matmulInfoPtr_->nSize >= tailL1SizeThreshold * compileInfoPtr_->aicNum &&
             matmulInfoPtr_->mSize > tailL1SizeThreshold);
        if (!needPollingFlag) {
            tilingData_->set_cubeBlockDimM(1);
            tilingData_->set_cubeBlockDimN(compileInfoPtr_->aicNum);
            return;
        }
        // 开始轮询
        ::CubeSplitResult cubeSplitResult = {UINT64_MAX, 1, compileInfoPtr_->aicNum};
        for (uint64_t mBlkNum = 1UL;
                mBlkNum <= std::min(static_cast<uint64_t>(compileInfoPtr_->aicNum),
                                    ops::CeilDiv(matmulInfoPtr_->mSize, static_cast<uint64_t>(BLOCK_CUBE)));
                mBlkNum++) {
            uint64_t nBlkNumMax = compileInfoPtr_->aicNum / mBlkNum;
            uint64_t nL1SizeMin =
                ops::CeilAlign(ops::CeilDiv(matmulInfoPtr_->nSize, nBlkNumMax), static_cast<uint64_t>(BLOCK_CUBE));
            uint64_t nBlkNum = std::min(ops::CeilDiv(matmulInfoPtr_->nSize, nL1SizeMin), nBlkNumMax);

            // k轴的大小对mte2的搬运量无影响，此处无需计算
            uint64_t mte2Cost = GetSizeByDataType(matmulInfoPtr_->aDtype) * matmulInfoPtr_->mSize * nBlkNum +
                                GetSizeByDataType(matmulInfoPtr_->bDtype) * matmulInfoPtr_->nSize * mBlkNum;
            if (matmulInfoPtr_->bDtype == ge::DT_INT4) {
                mte2Cost = GetSizeByDataType(matmulInfoPtr_->aDtype) * matmulInfoPtr_->mSize * nBlkNum +
                           matmulInfoPtr_->nSize * mBlkNum / INT4_CORRECTION_FACTOR;
            }
            // n较小场景才会进入轮询，此时应该在保证mte2Cost最小的前提下，尽可能提高m的切分粒度，保证n在单核上不会太小
            if (mte2Cost <= cubeSplitResult.mte2Cost) {
                cubeSplitResult.mte2Cost = mte2Cost;
                cubeSplitResult.cubeBlockDimM = mBlkNum;
                cubeSplitResult.cubeBlockDimN = nBlkNum;
            }
        }
        // 根据轮询结果设置cube切分
        tilingData_->set_cubeBlockDimM(cubeSplitResult.cubeBlockDimM);
        tilingData_->set_cubeBlockDimN(cubeSplitResult.cubeBlockDimN);
    } else {
        // 全量尝试4 : 8的切分设置
        static const uint64_t blockDimMMax = 4UL;
        uint64_t realBlockDimM = matmulInfoPtr_->hasBias
                                     ? ops::CeilDiv(matmulInfoPtr_->mSize, M_MAX_SIZE_WITH_BIAS_QUANT)
                                     : ops::CeilDiv(matmulInfoPtr_->mSize, M_MAX_SIZE);
        tilingData_->set_cubeBlockDimM(std::min(blockDimMMax, realBlockDimM));
        uint64_t nBlkNumMax = compileInfoPtr_->aicNum / tilingData_->get_cubeBlockDimM();
        uint64_t nL1SizeMin =
            ops::CeilAlign(ops::CeilDiv(matmulInfoPtr_->nSize, nBlkNumMax), static_cast<uint64_t>(BLOCK_CUBE));
        tilingData_->set_cubeBlockDimN(std::min(ops::CeilDiv(matmulInfoPtr_->nSize, nL1SizeMin), nBlkNumMax));
    }
}

void WeightQuantBatchMatmulV2TilingAS::OptimizeMatmulTiling()
{
    // 修正m方向的实际大小
    tilingData_->matmulTiling.set_singleCoreM(
        ops::CeilDiv(matmulInfoPtr_->mSize, static_cast<uint64_t>(tilingData_->get_cubeBlockDimM())));

    tilingData_->matmulTiling.set_baseM(
        std::min(tilingData_->matmulTiling.get_baseM(),
                 ops::CeilAlign(tilingData_->matmulTiling.get_singleCoreM(), static_cast<int32_t>(BLOCK_CUBE))));

    // 根据M值计算L0A上K的最大值
    uint64_t l0aMaxBaseK = (compileInfoPtr_->l0aSize >> 1) / GetSizeByDataType(matmulInfoPtr_->aDtype) /
                           tilingData_->matmulTiling.get_singleCoreM() / BLOCK_CUBE * BLOCK_CUBE;
    // 根据l1的情况反推L0A上K的最大值
    if (matmulInfoPtr_->hasBias) {
        l0aMaxBaseK = std::min(l0aMaxBaseK, static_cast<uint64_t>((A_L1_MAX_SIZE_WITH_BIAS_QUANT >> 1)) /
                                                GetSizeByDataType(matmulInfoPtr_->aDtype) /
                                                tilingData_->matmulTiling.get_baseM() /
                                                tilingData_->matmulTiling.get_stepKa() / BLOCK_CUBE * BLOCK_CUBE);
    } else {
        l0aMaxBaseK = std::min(l0aMaxBaseK, static_cast<uint64_t>((A_L1_MAX_SIZE >> 1)) /
                                                GetSizeByDataType(matmulInfoPtr_->aDtype) /
                                                tilingData_->matmulTiling.get_baseM() /
                                                tilingData_->matmulTiling.get_stepKa() / BLOCK_CUBE * BLOCK_CUBE);
    }

    // 计算N分核后，单core上理论需要处理的最大N，并往64对齐
    uint64_t singleN = ops::CeilAlign(
        ops::CeilDiv(matmulInfoPtr_->nSize, static_cast<uint64_t>(tilingData_->get_cubeBlockDimN())), 64UL);
    tilingData_->matmulTiling.set_singleCoreN(std::min(L1_N_MAX_SIZE, singleN));
    tilingData_->matmulTiling.set_baseN(tilingData_->matmulTiling.get_singleCoreN());

    // 根据理论需要处理的N推算L0B上K的最大值
    uint64_t l0bMaxBaseK = (compileInfoPtr_->l0bSize >> 1) / GetSizeByDataType(matmulInfoPtr_->aDtype) /
                           tilingData_->matmulTiling.get_singleCoreN() / BLOCK_CUBE * BLOCK_CUBE;

    // l0b上K的取值可以分成三档，即64,128,256，默认设置为64。当前处理的singleCoreN不会超过256, 因此暂不考虑baseK=32。
    static const std::vector<uint64_t> L0_BASE_K_LIST = {256UL, 128UL, 64UL};
    tilingData_->matmulTiling.set_baseK(L0_BASE_K_LIST.back());// 初始化赋值
    if (matmulInfoPtr_->transB) { // 小N，非转置场景不做优化，非主要性能场景，避免tilingkey膨胀
        for(size_t listId = 0; listId < L0_BASE_K_LIST.size(); listId++) {
            if(std::min(l0bMaxBaseK, l0aMaxBaseK) >= L0_BASE_K_LIST[listId]) {
                tilingData_->matmulTiling.set_baseK(L0_BASE_K_LIST[listId]);
                break;
            }
        }
    }

    // 判断L1上的A矩阵是否可以全载
    uint64_t aL1MaxSize = (matmulInfoPtr_->hasBias || matmulInfoPtr_->cDtype == ge::DT_INT8)
                              ? A_L1_MAX_SIZE_WITH_BIAS_QUANT
                              : A_L1_MAX_SIZE;
    uint64_t kAlign = ops::CeilAlign(matmulInfoPtr_->kSize, static_cast<uint64_t>(BLOCK_CUBE));
    if (tilingData_->matmulTiling.get_baseM() * kAlign * GetSizeByDataType(matmulInfoPtr_->aDtype) <= aL1MaxSize) {
        tilingData_->matmulTiling.set_stepKa(
            ops::CeilDiv(kAlign, static_cast<uint64_t>(tilingData_->matmulTiling.get_baseK())));
        tilingData_->matmulTiling.set_depthA1(tilingData_->matmulTiling.get_stepKa());
    }
}

/*
 * 高性能场景准入条件：
 * 1. b转置 且a不转置
 * 2. per channel场景 非int8输出
 * 3. 有bias时，m<=M_MAX_SIZE, 无bias时，m<=M_MAX_SIZE_WITH_BIAS_QUANT
 */
bool WeightQuantBatchMatmulV2TilingAS::CheckHighPerfScence()
{
    if (matmulInfoPtr_->transB && !matmulInfoPtr_->transA && matmulInfoPtr_->antiQuantType == QuantType::PER_CHANNEL &&
        matmulInfoPtr_->cDtype != ge::DT_INT8) {
        OP_LOGD(opName_, "current shape match Adaptive tiling high perf scence.");
        return true;
    }
    return false;
}

/***
 * 在 8192<n<=8192*2的场景下，会存在尾块N方向较小的情况，此时需要重新计算基本块大小，尝试扩大baseN.
 * baseN扩大后，相应带来UB、L1、L0大小的变化需要重新计算。
 * 其中当把baseN扩大时，需要将baseK减半，而baseN最多扩张1倍，所以L0无需重新计算。
 */
void WeightQuantBatchMatmulV2TilingAS::RecalculateBaseBlockSize() {
    if (compileInfoPtr_->aicNum * L1_N_MAX_SIZE < matmulInfoPtr_->nSize &&
        compileInfoPtr_->aicNum * L1_N_MAX_SIZE * NUM_TWO > matmulInfoPtr_->nSize &&
        matmulInfoPtr_->antiQuantType == QuantType::PER_CHANNEL) {
        uint64_t tailSize =
            matmulInfoPtr_->nSize - tilingData_->get_mainBlockCount() * tilingData_->get_mainBlockL1Size();
        uint64_t tailBlockCount = tilingData_->get_cubeBlockDimN();
        if (tailBlockCount == 0) {
            return;
        }
        uint64_t firstTailBlockL1Size = tailSize / tailBlockCount;
        uint64_t firstTailBlockCount = tailBlockCount - tailSize % tailBlockCount;
        uint64_t secondTailBlockL1Size = 0;
        uint64_t secondTailBlockCount = 0;
        if (tailSize % tailBlockCount > 0) {
            secondTailBlockL1Size = firstTailBlockL1Size + 1;
            secondTailBlockCount = tailSize % tailBlockCount;
        }
        uint64_t extendedWeightMte2N =
            ops::CeilDiv(std::max(firstTailBlockL1Size, secondTailBlockL1Size), static_cast<uint64_t>(NUM_TWO));
        uint64_t extendedBaseN =
            ops::CeilAlign(std::max(firstTailBlockL1Size, secondTailBlockL1Size), static_cast<uint64_t>(BLOCK_CUBE));
        if (!CheckUbSizeAfterExtending(extendedWeightMte2N) || !CheckL1SizeAfterExtending(extendedBaseN)) {
            return;
        }
        // 满足空间要求后，修正tiling参数
        tilingData_->matmulTiling.set_baseN(extendedBaseN);
        tilingData_->matmulTiling.set_singleCoreN(extendedBaseN);
        tilingData_->matmulTiling.set_baseK(tilingData_->matmulTiling.get_baseK() / NUM_TWO);     // 将 baseK 减半
        tilingData_->matmulTiling.set_stepKa(tilingData_->matmulTiling.get_stepKa() * NUM_TWO);   // 将 stepKa 翻倍
        tilingData_->matmulTiling.set_stepKb(tilingData_->matmulTiling.get_stepKb() * NUM_TWO);   // 将 stepKb 翻倍
        tilingData_->matmulTiling.set_depthA1(tilingData_->matmulTiling.get_depthA1() * NUM_TWO); // 将 depthA1 翻倍
        tilingData_->matmulTiling.set_depthB1(tilingData_->matmulTiling.get_depthB1() * NUM_TWO); // 将 depthB1 翻倍
        tilingData_->set_firstTailBlockL1Size(firstTailBlockL1Size);
        tilingData_->set_firstTailBlockCount(firstTailBlockCount);
        tilingData_->set_secondTailBlockL1Size(secondTailBlockL1Size);
        tilingData_->set_secondTailBlockCount(secondTailBlockCount);
    }
}

bool WeightQuantBatchMatmulV2TilingAS::CheckL1SizeAfterExtending(uint64_t extendedBaseN) const {
    uint64_t al1Size = tilingData_->matmulTiling.get_baseM() * tilingData_->matmulTiling.get_depthA1() *
                       tilingData_->matmulTiling.get_baseK() * GetSizeByDataType(matmulInfoPtr_->aDtype);
    uint64_t extendedBl1Size = extendedBaseN * tilingData_->matmulTiling.get_depthB1() *
                               tilingData_->matmulTiling.get_baseK() * GetSizeByDataType(matmulInfoPtr_->aDtype);
    uint64_t aL1MaxSize = (matmulInfoPtr_->hasBias || matmulInfoPtr_->cDtype == ge::DT_INT8)
                              ? A_L1_MAX_SIZE_WITH_BIAS_QUANT
                              : A_L1_MAX_SIZE;
    if (aL1MaxSize - al1Size < extendedBl1Size - (compileInfoPtr_->l1Size >> 1)) {
        return false;
    }
    return true;
}

bool WeightQuantBatchMatmulV2TilingAS::CheckUbSizeAfterExtending(uint64_t extendedWeightMte2N) const {
    uint64_t weightLowBitBufferSize = extendedWeightMte2N * KILOBYTE;
    uint64_t weightF16BufferSize = 66 * KILOBYTE; // weightF16占用66KB
    if (matmulInfoPtr_->bDtype == ge::DT_INT4) {
        weightLowBitBufferSize = weightLowBitBufferSize / INT4_CORRECTION_FACTOR;
    }
    if (mte2Config_ == Mte2Configuration::MTE2_INNER_SIZE_512_BUF_NUM_4 ||
        mte2Config_ == Mte2Configuration::MTE2_INNER_SIZE_1024_BUF_NUM_2) {
        weightLowBitBufferSize = weightLowBitBufferSize * NUM_TWO;
    }
    uint64_t antiquantParamsBufferSize = 8 * KILOBYTE; // antiquant参数占用8KB
    if (weightLowBitBufferSize + weightF16BufferSize + antiquantParamsBufferSize > UB_SIZE_V100) {
        return false;
    }
    return true;
}

void WeightQuantBatchMatmulV2TilingAS::SetPreLoad() {
    uint64_t al1Size = tilingData_->matmulTiling.get_baseM() * tilingData_->matmulTiling.get_depthA1() *
                       tilingData_->matmulTiling.get_baseK() * GetSizeByDataType(matmulInfoPtr_->aDtype);
    if (tilingData_->get_cubeBlockDimM() == 1 &&
        matmulInfoPtr_->mSize >= 128 && // m大于等于128时有足够载入量才考虑perload
        (matmulInfoPtr_->mSize * matmulInfoPtr_->kSize * GetSizeByDataType(matmulInfoPtr_->aDtype) <= al1Size)) {
        tilingData_->set_aPreloadSize(
            ops::CeilAlign(ops::CeilDiv(static_cast<uint64_t>(matmulInfoPtr_->mSize * matmulInfoPtr_->kSize),
                                        static_cast<uint64_t>(tilingData_->get_cubeBlockDimN())),
                           static_cast<uint64_t>(64))); // 64 表示128B的cacheline对齐
    }
}

void WeightQuantBatchMatmulV2TilingAS::ComputeTailResplitTiling()
{
    algorithmSubCategory_ = OptimizationAlgorithmSubCategory::N_FIRST_TAIL_RESPLIT;

    // step 1. 尾块重切分
    ResplitTail();

    // step 2. 判定内轴(k)的核内切分规则
    SetInnerSize();

    // step 3. 重计算baseN baseK
    RecalculateBaseBlockSize();

    // step 4. 设置A的提前载入
    SetPreLoad();
}

void WeightQuantBatchMatmulV2TilingAS::ResplitTail()
{
    uint64_t mainL1SizeDefault = L1_N_MAX_SIZE;
    uint64_t mainBlockCountDefault = matmulInfoPtr_->nSize / mainL1SizeDefault;

    // 主块数量较多，则匀出一个主块给尾块
    if (mainBlockCountDefault >= tilingData_->get_cubeBlockDimN() ||
        matmulInfoPtr_->nSize % (mainL1SizeDefault * tilingData_->get_cubeBlockDimN()) == 0) {
        tilingData_->set_mainBlockL1Size(mainL1SizeDefault);
        if (matmulInfoPtr_->nSize % (mainL1SizeDefault * tilingData_->get_cubeBlockDimN()) == 0) {
            tilingData_->set_mainBlockCount(mainBlockCountDefault);
        } else {
            uint64_t blockCountFloorAlign =
                mainBlockCountDefault / tilingData_->get_cubeBlockDimN() * tilingData_->get_cubeBlockDimN();
            tilingData_->set_mainBlockCount(blockCountFloorAlign - tilingData_->get_cubeBlockDimN());
        }
    } else {
        tilingData_->set_mainBlockL1Size(0);
        tilingData_->set_mainBlockCount(0);
    }
    uint64_t tailSize = matmulInfoPtr_->nSize - tilingData_->get_mainBlockCount() * tilingData_->get_mainBlockL1Size();
    if (tailSize == 0) {
        tilingData_->set_firstTailBlockL1Size(0);
        tilingData_->set_firstTailBlockCount(0);
        tilingData_->set_secondTailBlockL1Size(0);
        tilingData_->set_secondTailBlockCount(0);
        return;
    }
    /* 一个主块+一个尾块，需要多核做2轮才能保证单次不超过基本块的最大规格。
     * 一个尾块，多核做1轮将可以保证单次不超过基本块的最大规格。
     */
    uint64_t tailBlockCount = mainBlockCountDefault >= tilingData_->get_cubeBlockDimN()
                                  ? 2 * tilingData_->get_cubeBlockDimN()
                                  : tilingData_->get_cubeBlockDimN();
    tilingData_->set_firstTailBlockL1Size(tailSize / tailBlockCount);
    tilingData_->set_firstTailBlockCount(tailBlockCount - tailSize % tailBlockCount);
    if (tailSize % tailBlockCount > 0) {
        tilingData_->set_secondTailBlockL1Size(tilingData_->get_firstTailBlockL1Size() + 1);
        tilingData_->set_secondTailBlockCount(tailSize % tailBlockCount);
    } else {
        tilingData_->set_secondTailBlockL1Size(0);
        tilingData_->set_secondTailBlockCount(0);
    }
}

void WeightQuantBatchMatmulV2TilingAS::SetInnerSize()
{
    static constexpr uint64_t MTE2_K_256 = 256UL;
    static constexpr uint64_t MTE2_K_512 = 512UL;
    static constexpr uint64_t MTE2_K_1024 = 1024UL;
    // 内轴切分仅支持512、1024切分,若weightL1K已超过这些配置，则直接调整内轴配置即可，无需进一步调整
    uint64_t weightL1K = tilingData_->matmulTiling.get_baseK() * tilingData_->matmulTiling.get_stepKb();
    if (weightL1K  >= MTE2_K_1024) {
        mte2Config_ = Mte2Configuration::MTE2_INNER_SIZE_1024_BUF_NUM_2;
        return;
    } else if (weightL1K >= MTE2_K_512) {
        mte2Config_ = Mte2Configuration::MTE2_INNER_SIZE_512_BUF_NUM_4;
        return;
    }

    uint64_t blockL1MaxSize =
        std::max(tilingData_->get_mainBlockL1Size(),
                 std::max(tilingData_->get_firstTailBlockL1Size(), tilingData_->get_secondTailBlockL1Size()));
    if (matmulInfoPtr_->kSize <= MTE2_K_256 * QUARTER_BUFFER_NUM) {
        // k轴较小场景，mte2内轴较大会造成较高的头开销，缩短内轴增加ub上的buffer数量
        mte2Config_ = Mte2Configuration::MTE2_INNER_SIZE_256_BUF_NUM_4;
    } else if (blockL1MaxSize > (L1_N_MAX_SIZE >> 1)) {
        // 实际处理的n轴大于ub上限的一半，ub的buffer数量无法进一步增加
        mte2Config_ = Mte2Configuration::MTE2_INNER_SIZE_512_BUF_NUM_2;
    } else if (matmulInfoPtr_->kSize <= MTE2_K_512 * QUARTER_BUFFER_NUM) {
        // 实际处理的n轴较小，可以适当增加ub载入量或载入频率。考虑到内轴较小，通过增加ub上的buffer数量提高载入频率
        mte2Config_ = Mte2Configuration::MTE2_INNER_SIZE_512_BUF_NUM_4;
    } else {
        // 实际处理的n轴较小，可以适当增加ub载入量或载入频率。考虑到内轴较大，通作增加内轴提高载入量
        mte2Config_ = Mte2Configuration::MTE2_INNER_SIZE_1024_BUF_NUM_2;
    }
}

void WeightQuantBatchMatmulV2TilingAS::ComputeBasicTiling()
{
    algorithmSubCategory_ = OptimizationAlgorithmSubCategory::N_FIRST_BASIC_BLOCK;
    if (matmulInfoPtr_->transB) {
        // step 1. n方向做尾块重切分
        ResplitTail();

        // step 2. 判定内轴(k)的核内切分规则
        mte2Config_ = Mte2Configuration::MTE2_INNER_SIZE_512_BUF_NUM_DEFAULT;
    } else {
        // step 1. n方向为内轴，无法做重切分
        tilingData_->set_mainBlockL1Size(tilingData_->matmulTiling.get_singleCoreN());
        tilingData_->set_mainBlockCount(matmulInfoPtr_->nSize / tilingData_->matmulTiling.get_singleCoreN() /
                                        tilingData_->get_cubeBlockDimN() * tilingData_->get_cubeBlockDimN());
        tilingData_->set_firstTailBlockL1Size(
            tilingData_->matmulTiling.get_singleCoreN());  // n为内轴，细粒度切分影响搬运效率，此处暂不考虑

        uint64_t tailSize =
            matmulInfoPtr_->nSize - tilingData_->get_mainBlockCount() * tilingData_->get_mainBlockL1Size();
        tilingData_->set_firstTailBlockCount(
            ops::CeilDiv(tailSize, static_cast<uint64_t>(tilingData_->matmulTiling.get_singleCoreN())));
        tilingData_->set_secondTailBlockL1Size(0);
        tilingData_->set_secondTailBlockCount(0);

        // step 2. 判定内轴(k)的核内切分规则
        mte2Config_ = Mte2Configuration::MTE2_INNER_SIZE_256_BUF_NUM_4;
    }
}

// 5、计算TilingKey
uint64_t WeightQuantBatchMatmulV2TilingAS::GetTilingKey() const
{
    TilingKeyConfigure tilingKeyConfigure;
    // 平台类型占2位(平台大类， 平台小类)，平台大类在高位，需要乘10
    tilingKeyConfigure.socVersionType = static_cast<uint8_t>(SocVersionType::SUPPORT_L1_TO_BT_BF16) * 10;
    tilingKeyConfigure.quantizationScenario = static_cast<uint8_t>(QuantizationScenario::DEFAULT);
    // 算法类型占2位(算法大类，算法小类)，算法大类在高位，需要乘10
    tilingKeyConfigure.algorithm = static_cast<uint8_t>(OptimizationAlgorithmCategory::VECTOR_ANTIQUANT) * 10 +
                                   static_cast<uint8_t>(algorithmSubCategory_);
    tilingKeyConfigure.transposeSituation =
        (static_cast<uint16_t>(matmulInfoPtr_->transA) << 1) | static_cast<uint16_t>(matmulInfoPtr_->transB);
    tilingKeyConfigure.antiquantType = static_cast<uint8_t>(matmulInfoPtr_->antiQuantType);
    tilingKeyConfigure.quantType = static_cast<uint8_t>(matmulInfoPtr_->quantType);

    if (matmulInfoPtr_->biasDtype == ge::DT_FLOAT && matmulInfoPtr_->hasBias) {
        // 按照tilingKey的定义，bias为float场景在原有tiling key的基础上加4做区分
        tilingKeyConfigure.optionInputSituation = (static_cast<uint16_t>(matmulInfoPtr_->hasAntiQuantOffset) << 1) + 4;
    } else {
        tilingKeyConfigure.optionInputSituation = static_cast<uint16_t>(matmulInfoPtr_->hasAntiQuantOffset) << 1;
    }

    tilingKeyConfigure.weightFormat = static_cast<uint8_t>(WeightFormat::ND);

    tilingKeyConfigure.templateCustom = static_cast<uint8_t>(mte2Config_);
    tilingKeyConfigure.apiConstexpr = 0;
    return tilingKeyConfigure.GenTilingKey();
}

// 6、计算Workspace 大小
ge::graphStatus WeightQuantBatchMatmulV2TilingAS::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opName_, "failed to get workspace size"),
                    return ge::GRAPH_FAILED);
    workspaces[0] = 16 * 1024 * 1024;  // asc要求workspace最低需要16 * 1024 * 1024 Byte
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling
