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
 * \file conv3d_backprop_filter_v2.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include "cube_tiling_runtime.h"
#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_templates_registry.h"
#include "tiling/tiling_type.h"
#include "cube/util/math_util.h"
#include "cube/util/cube_util.h"
#include "conv3d_backprop_filter_v2_tiling.h"

using namespace optiling::cachetiling;

namespace {

#define unlikely(x) __builtin_expect((x), 0)
#define OP_LOGE(op_name, ...) std::printf(op_name, ##__VA_ARGS__)
#define OP_LOGD(op_name, ...) std::printf(op_name, ##__VA_ARGS__)
#define OP_LOGE_IF(condition, return_value, op_name, fmt, ...)                                                 \
  static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool"); \
  do                                                                                                           \
  {                                                                                                            \
    if (unlikely(condition))                                                                                   \
    {                                                                                                          \
      OP_LOGE(op_name, fmt, ##__VA_ARGS__);                                                                    \
      return return_value;                                                                                     \
    }                                                                                                          \
  } while (0)
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)            \
    if ((ptr) == nullptr)                                    \
    {                                                        \
        std::printf("nullptr error!");                       \
        return ge::GRAPH_SUCCESS;                            \
    }  
const size_t X_INDEX = 0;
constexpr uint32_t DB_ON = 2;
constexpr uint32_t C04 = 4;
constexpr uint32_t C16 = 16;
constexpr uint32_t CORE_NUM_910B3 = 20;
constexpr uint32_t CORE_NUM_910B2 = 24;
constexpr float CORE_USED_THRESHOLD = 0.8f;
constexpr int32_t MIN_BATCHDIM = 4;
constexpr int32_t BLOCK_CUBE = static_cast<int32_t>(AscendC::BLOCK_CUBE);
constexpr uint64_t L1_SIZE = 512 * 1024 - 128;
constexpr int64_t L0_SIZE = 65536;
constexpr uint32_t BEST_BASE_M = 128;
constexpr uint32_t BEST_BASE_K = 128;
constexpr uint32_t BEST_BASE_N = 256;
constexpr uint32_t SECOND_BASE_M = 64;
constexpr uint32_t SECOND_BASE_K = 64;
constexpr uint32_t SECOND_BASE_N = 512;
constexpr uint32_t MIN_STEPK = 2;
constexpr uint32_t ROW_NUM = 2;
constexpr int32_t BUFFER_NUM_L1 = 4;
constexpr int64_t WORKSIZE = 16 * 1024 * 1024; // 16 * 1024 * 1024: 16M LibApiWorkSpaceSize
constexpr int32_t MAX_DILATION_D = 255;
// key:
// "N_Do_Co1_Ho_Wo_Ci1_Hi_Wi_Dk_Hk_Wk_strideD_strideH_strideW_
// _padFront_padBack_padUp_padDown_padLeft_padRight_dilationD_dilationH_dilationW"
// value:
// {batchDim, groupDim, dkDim, mDim, kDim, nDim,
// singleCoreBatch, singleCoreGroup, singleCoreCout, singleCoreCin, singleCoreHo,
// al0Pbuffer, bl0Pbuffer, cl0Pbuffer, al1Pbuffer, bl1Pbuffer, baseM, baseK, baseN,
// stepM, stepN, stepKa, stepKb, iterateOrder}
const std::map<std::string, optiling::TilingValueDw> TILINGDATA_MAP_B3 {
    {"1_9_32_64_64_32_64_64_3_3_3_1_1_1_0_0_1_1_1_1_1_1_1",
    {1, 1, 1, 2, 5, 2, 9, 1, 1, 256, 768, 13, 2, 2, 1, 1, 1, 128, 64, 256, 1, 1, 13, 13, 1, 61440}}, // x1_25
    {"1_17_8_256_256_8_256_256_3_3_3_1_1_1_0_0_1_1_1_1_1_1_1",
    {1, 1, 1, 1, 10, 2, 17, 1, 1, 128, 192, 26, 2, 2, 1, 2, 2, 128, 64, 256, 1, 1, 4, 4, 1, 131072}}, //x1_02
    {"1_17_16_128_128_16_128_128_3_3_3_1_1_1_0_0_1_1_1_1_1_1_1",
    {1, 1, 1, 1, 5, 4, 17, 1, 1, 256, 192, 26, 2, 2, 1, 2, 2, 128, 64, 256, 1, 1, 4, 4, 1, 131072}}, //x1_06
    {"1_17_16_128_128_32_128_128_3_3_3_1_1_1_0_0_1_1_1_1_1_1_1",
    {1, 1, 1, 1, 10, 2, 17, 1, 1, 256, 768, 13, 2, 2, 1, 2, 2, 128, 64, 256, 1, 1, 4, 4, 1, 131072}}, //x1_08
    {"1_9_16_128_128_8_128_128_3_3_3_1_1_1_0_0_1_1_1_1_1_1_1",
    {1, 1, 1, 1, 10, 2, 9, 1, 1, 256, 192, 13, 2, 2, 1, 2, 2, 128, 64, 256, 1, 1, 4, 4, 1, 131072}}, //x1_10
    {"1_9_16_128_128_16_128_128_3_3_3_1_1_1_0_0_1_1_1_1_1_1_1",
    {1, 1, 1, 1, 5, 4, 9, 1, 1, 256, 192, 26, 2, 2, 1, 2, 2, 128, 64, 256, 1, 1, 4, 4, 1, 131072}}, //x1_12
    {"16_20_4_128_128_4_130_130_3_3_3_1_1_1_0_0_0_0_0_0_1_1_1",
    {20, 1, 1, 1, 1, 1, 16, 1, 1, 64, 192, 128, 2, 2, 1, 2, 2, 64, 64, 256, 1, 1, 8, 8 ,1, 98304}}, //magvit_3
    {"16_20_8_64_64_8_66_66_3_3_3_1_1_1_0_0_0_0_0_0_1_1_1",
    {20, 1, 1, 1, 1, 1, 16, 1, 1, 128, 384, 64, 2, 2, 1, 2, 2, 128, 64, 256, 1, 1, 4, 4, 1, 131072}}, //magvit_5
};
const std::map<std::string, optiling::TilingValueDw> TILINGDATA_MAP_B2 {
    {"16_20_1_128_128_4_130_130_3_3_3_1_1_1_0_0_0_0_0_0_1_1_1",
    {2, 1, 1, 1, 6, 2, 160, 1, 1, 16, 96, 22, 2, 2, 2, 2, 1, 16, 16, 864, 1, 1, 176, 44, 1, 112320}}, // magvit_b16_1
    {"8_20_1_128_128_4_130_130_3_3_3_1_1_1_0_0_0_0_0_0_1_1_1",
    {2, 1, 1, 1, 6, 2, 80, 1, 1, 16, 96, 22, 2, 2, 2, 2, 1, 16, 16, 864, 1, 1, 176, 44, 1, 112320}}, // magvit_b8_1
    {"1_62_16_66_66_16_128_128_4_4_4_2_2_2_3_3_3_3_3_3_1_1_1",
    {1, 1, 1, 1, 22, 1, 62, 1, 1, 256, 1024, 3, 2, 2, 1, 1, 1, 128, 64, 256, 1, 1, 4, 4, 1, 27392}}, // videogpt_bf16_2
    {"1_9_16_128_128_16_128_128_3_3_3_1_1_1_0_0_1_1_1_1_1_1_1",
    {1, 1, 1, 2, 3, 4, 9, 1, 1, 128, 192, 43, 2, 2, 1, 2, 2, 128, 64, 256, 1, 1, 4, 4, 1, 131072}}, //x1_12
    {"16_20_8_64_64_8_66_66_3_3_3_1_1_1_0_0_0_0_0_0_1_1_1",
    {20, 1, 1, 1, 1, 1, 16, 1, 1, 128, 384, 64, 2, 2, 1, 2, 2, 128, 64, 256, 1, 1, 4, 4, 1, 131072}}, //magvit_b16_5
    {"16_20_16_32_32_16_34_34_3_3_3_1_1_1_0_0_0_0_0_0_1_1_1",
    {8, 1, 1, 1, 1, 3, 40, 1, 1, 256, 256, 32, 2, 2, 1, 2, 2, 128, 64, 256, 1, 1, 4, 4, 1, 131072}}, //magvit_b16_7
    {"1_60_16_64_64_16_130_130_4_4_4_2_2_2_0_0_0_0_0_0_1_1_1",
    {6, 1, 1, 1, 4, 1, 10, 1, 1, 256, 1024, 16, 2, 2, 1, 2, 2, 128, 64, 256, 1, 1, 4, 4, 1, 131072}}, //videoGPT_f240_h256_8

};

using Conv3DBackpropFilterV2CompileInfo = optiling::Conv3DCompileInfo;
}  // namespace

namespace optiling {

void Conv3DBackpropFilterV2Tiling::Reset()
{
    tilingData_.SetDataPtr(context_->GetRawTilingData()->GetData());
    libApiWorkSpaceSize_ = 0U;
    opName_ = nullptr;
}

ge::graphStatus Conv3DBackpropFilterV2Tiling::GetPlatformInfo() { return ge::GRAPH_SUCCESS; }

ge::graphStatus Conv3DBackpropFilterV2Tiling::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

bool Conv3DBackpropFilterV2Tiling::IsCapable() {
    return true;
}

ge::graphStatus Conv3DBackpropFilterV2Tiling::DoOpTiling()
{
    if (!GetTbeTiling()) {
        OP_LOGE(context_->GetNodeName(), "GetTbeTiling failed");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3DBackpropFilterV2Tiling::DoLibApiTiling()
{
    enableDeterministic_ = context_->GetDeterministic() ? 1 : 0;
    SetDwTilingFromTbeTiling();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

uint64_t Conv3DBackpropFilterV2Tiling::GetTilingKey() const
{
    return 0;
}

ge::graphStatus Conv3DBackpropFilterV2Tiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    size_t sysWorkspaceSize = WORKSIZE;
    size_t usrWorkspaceSize = 0;
    uint64_t dimNum = tilingData_.params.get_mDim() * tilingData_.params.get_nDim() *
                          tilingData_.params.get_batchDim() * tilingData_.params.get_kDim() *
                          tilingData_.params.get_groupDim() * tilingData_.params.get_dkDim();
    if (enableDeterministic_) { // enable deterministic calculation
        size_t singleSize = tilingData_.dwTiling.get_baseM() * tilingData_.dwTiling.get_baseN();
        usrWorkspaceSize = DB_ON * sizeof(int32_t) * dimNum * singleSize;
    } else if (context_->GetInputDesc(X_INDEX)->GetStorageFormat() == ge::FORMAT_NCDHW) { // Transdata merge and determinstic calculation are mutually exclusive.
        auto singleCoreHo = tilingData_.dwTiling.get_singleCoreHo();
        uint32_t singleCoreHi = (singleCoreHo - 1) * tilingData_.dwTiling.get_strideH()
            + (tilingData_.dwTiling.get_hk() - 1) * tilingData_.dwTiling.get_dilationH() + 1;
        singleCoreHi = (singleCoreHi < tilingData_.dwTiling.get_hi()) ? singleCoreHi : tilingData_.dwTiling.get_hi();
        auto singleCoreCin = tilingData_.dwTiling.get_singleCoreCin();
        uint64_t singleCoreTransdataSize = singleCoreCin * singleCoreHi * tilingData_.dwTiling.get_wi()
            * ge::GetSizeByDataType(ge::DT_BF16) * DB_ON;
        usrWorkspaceSize = coreNum_ * singleCoreTransdataSize;
    }
    workspaces[0] = sysWorkspaceSize + usrWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3DBackpropFilterV2Tiling::PostTiling()
{
    OP_LOGD(opName_, "final tiling data size: %zu", tilingData_.GetDataSize());

    OP_TILING_CHECK(tilingData_.GetDataSize() % sizeof(uint64_t) != 0,
                    CUBE_INNER_ERR_REPORT(opName_, "tiling data size[%zu] not aligned to 8", tilingData_.GetDataSize()),
                    return ge::GRAPH_FAILED);
    context_->SetBlockDim(tilingData_.params.get_mDim() * tilingData_.params.get_nDim() *
                          tilingData_.params.get_batchDim() * tilingData_.params.get_kDim() *
                          tilingData_.params.get_groupDim() * tilingData_.params.get_dkDim());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

bool Conv3DBackpropFilterV2Tiling::GetTbeTiling()
{
    const Conv3DBackpropFilterV2CompileInfo *compileInfoPtr =
        reinterpret_cast<const Conv3DBackpropFilterV2CompileInfo *>(context_->GetCompileInfo());
    OP_TILING_CHECK(compileInfoPtr == nullptr, CUBE_INNER_ERR_REPORT("Conv3DBackpropFilterV2", "compile_info is null"),
                    return false);
    cachetiling::Conv3DBpFilterTilingParam tilingParams(cachetiling::kConv3DBackpropFilterV2);
    tilingParams.binary_mode = compileInfoPtr->binary_mode;
    OP_LOGE_IF(!tilingParams.ParseOpInfo(context_, *compileInfoPtr), false, opName_,
               "Parse cache tiling params failed");
    dtypeByte_ = tilingParams.a_dtype_bytes;
    aDtype_ = tilingParams.a_dtype;
    OP_LOGE_IF(tilingParams.a_dtype != ge::DT_BF16 && tilingParams.a_dtype != ge::DT_FLOAT &&
                tilingParams.a_dtype != ge::DT_FLOAT16, false, opName_,
               "now only support dtype bf16/fp16/fp32, but get %s",
               ge::TypeUtils::DataTypeToAscendString(tilingParams.a_dtype).GetString());
    ReCalDilation(tilingParams);
    OP_LOGE_IF(tilingParams.dilation_d < 1 || tilingParams.dilation_d > MAX_DILATION_D, false, opName_,
                "dilation_d[%d] is invalid, it shoud be in range: [1, %d]", tilingParams.dilation_d, MAX_DILATION_D);
    OP_LOGE_IF(tilingParams.groups < 1 || tilingParams.groups > UINT16_MAX, false, opName_,
                "Groups[%d] is invalid, it shoud be in range: [1, %d]", tilingParams.groups, UINT16_MAX);

    int32_t kernelHDilation = (tilingParams.kernel_h - 1) * tilingParams.dilation_h + 1;
    OP_LOGE_IF(tilingParams.pad_u >= kernelHDilation, false, opName_,
        "pad_u[%d] is invalid, it should be less than (kernel_h - 1) * dilation_h + 1 = [%d]", tilingParams.pad_u, kernelHDilation);
    OP_LOGE_IF(tilingParams.pad_d >= kernelHDilation, false, opName_,
        "pad_d[%d] is invalid, it should be less than (kernel_h - 1) * dilation_h + 1 = [%d]", tilingParams.pad_d, kernelHDilation);

    tilingParams.conv1d_flag = false;
    tilingParams.load2d_flag = false;
    if (tilingParams.strideh_read_flag == 1) {
        OP_LOGD(opName_, "not support stride h read perf optimize template for now, close it");
        tilingParams.strideh_read_flag = 0;
        tilingParams.b_shape.h = tilingParams.sr_fmap_h;
        tilingParams.stride_h = tilingParams.sr_stride_h;
    }

    tilingData_.params.set_totalL1Size(tilingParams.platform_info.l1_size());
    tilingData_.dwTiling.set_channelSize(tilingParams.k0);
    tilingData_.dwTiling.set_m0(BLOCK_CUBE);
    tilingData_.dwTiling.set_n0(BLOCK_CUBE);
    tilingData_.dwTiling.set_k0(tilingParams.k0);
    tilingData_.dwTiling.set_hf32Flag(tilingParams.hf32_flag);

    coreNum_ = tilingParams.platform_info.core_num();
    if (!cachetiling::GetTiling<cachetiling::Conv3DBpFilterTilingParam, cachetiling::Conv3DBpFilterTiling,
                                Conv3dBpFilterRunInfo, cachetiling::Conv3DBpFilterHashParam,
                                cachetiling::Conv3DBpFilterHashItem>(tilingParams, tbeTiling_, runInfo_)) {
        OP_LOGE(opName_, "GetTiling interface failed");
        return false;
    }
    return true;
}

void Conv3DBackpropFilterV2Tiling::SetShapeTiling(TConv3DDwTiling &dwt)
{
    // shape
    dwt.set_batch(runInfo_.batch);
    dwt.set_cout(runInfo_.co);
    dwt.set_cin(runInfo_.ci);
    dwt.set_cout1G(runInfo_.cout1_g);
    dwt.set_cin1G(runInfo_.cin1_g);
    dwt.set_dout(runInfo_.dout);
    dwt.set_wo(runInfo_.wo);  // dedy o
    dwt.set_ho(runInfo_.ho);  // dedy h
    dwt.set_wi(runInfo_.wi);
    dwt.set_hi(runInfo_.hi);
    dwt.set_di(runInfo_.di);
    dwt.set_wk(runInfo_.kw);
    dwt.set_hk(runInfo_.kh);
    dwt.set_dk(runInfo_.kd);
}

void Conv3DBackpropFilterV2Tiling::SetAttrTiling(TConv3DDwTiling &dwt)
{
    // attr
    dwt.set_group(runInfo_.real_g);
    dwt.set_strideW(runInfo_.stride_w);
    dwt.set_strideH(runInfo_.stride_h);
    dwt.set_strideD(runInfo_.stride_d);
    dwt.set_padLeft(runInfo_.pad_l);
    dwt.set_padRight(runInfo_.pad_r);
    dwt.set_padUp(runInfo_.pad_u);
    dwt.set_padDown(runInfo_.pad_d);
    dwt.set_padFront(runInfo_.pad_f);
    dwt.set_padBack(runInfo_.pad_b);
    dwt.set_dilationW(runInfo_.dilation_w);
    dwt.set_dilationH(runInfo_.dilation_h);
    dwt.set_dilationD(runInfo_.dilation_d);
}

void Conv3DBackpropFilterV2Tiling::ReCalDilation(cachetiling::Conv3DBpFilterTilingParam &tilingParams)
{
    // if kernelD/H/W is equal to 1, dilationD/H/W should be set to 1
    if (tilingParams.kernel_d == 1) {
        OP_LOGD(opName_, "kernel_d is equal to 1, dilation_d will be set to 1");
        tilingParams.dilation_d = 1;
        runInfo_.dilation_d = 1;
    }

    if (tilingParams.kernel_h == 1) {
        OP_LOGD(opName_, "kernel_h is equal to 1, dilation_h will be set to 1");
        tilingParams.dilation_h = 1;
        runInfo_.dilation_h = 1;
    }

    if (tilingParams.kernel_w == 1) {
        OP_LOGD(opName_, "kernel_w is equal to 1, dilation_w will be set to 1");
        tilingParams.dilation_w = 1;
        runInfo_.dilation_w = 1;
    }
}

bool Conv3DBackpropFilterV2Tiling::CheckL0Size(uint32_t baseM, uint32_t baseN, uint32_t baseK, uint32_t byteSize)
{
    int64_t l0aSize = static_cast<int64_t>(baseM) * baseK * byteSize * DB_ON;
    int64_t l0bSize = static_cast<int64_t>(baseN) * baseK * byteSize * DB_ON;

    return l0aSize <= L0_SIZE && l0bSize <= L0_SIZE;
}

int32_t Conv3DBackpropFilterV2Tiling::GetDimFactor(const int64_t& value, const std::vector<int32_t>& factorLists)
{
    int32_t dimFactor = 1;
    for (uint32_t i = 0; i < factorLists.size(); i++) {
        if (value % factorLists[i] == 0) {
            dimFactor = factorLists[i];
            break;
        }
    }
    return dimFactor;
}

void Conv3DBackpropFilterV2Tiling::GetBatchDim(CoreDimDw& coreDim, int32_t dMaxFactor, int32_t batchDepthMaxFactor)
{
    coreDim.dDim = optiling::cachetiling::MathUtil::GetGcd(dMaxFactor, batchDepthMaxFactor);
    coreDim.batchDim = batchDepthMaxFactor / coreDim.dDim;
}

void Conv3DBackpropFilterV2Tiling::GetCoreDim(CoreDimDw& coreDim, uint32_t curCoreNum)
{
    // 获取当前核数的公因子作为可分给B*D的核数，并从大到小排序
    std::vector<int32_t> coreFactors = {};
    optiling::cachetiling::MathUtil::GetFactors(coreFactors, curCoreNum, curCoreNum);
    std::sort(coreFactors.rbegin(), coreFactors.rend());
    // 计算B,D能否被coreFactors中的核数整除
    int32_t dMaxFactor = GetDimFactor(static_cast<int64_t>(runInfo_.dout), coreFactors);
    int32_t bMaxFactor = GetDimFactor(static_cast<int64_t>(runInfo_.batch), coreFactors);
    int32_t batchDepthMaxFactor = GetDimFactor(static_cast<int64_t>(dMaxFactor * bMaxFactor), coreFactors);
    // 如果B*D可分核数少于4，直接返回，走tbe tiling
    if (batchDepthMaxFactor < MIN_BATCHDIM) {
        return;
    }
    // B*D最大公因子能整除总核数，直接全绑B*D，分核结束, batchDim * dDim = 20(B3);
    if ((dMaxFactor * bMaxFactor) % curCoreNum == 0) {
        coreDim.dDim = dMaxFactor;
        coreDim.batchDim = curCoreNum / dMaxFactor;
        return;
    }
    // B*D分不完，把剩下的核按情况全分给K或者全分给N,即满足B K或者B N均匀分核
    // 粗算K N方向的循环次数，如果K方向循环更大，先从K方向考虑，如果K分完后不小于基本块的情况下，直接全分给K，返回
    // 如果N方向循环更大，先从N方向考虑，如果N分完后不小于基本块的情况下，直接全分给N，返回
    // 如果上述都不满足也就是需要更靠更细粒度的分核以及对M分核，此时，直接return，走tbe tiling
    int32_t remainFactor = curCoreNum / batchDepthMaxFactor;
    int64_t maxK = static_cast<int64_t>(runInfo_.ho) * runInfo_.wo;
    int64_t maxN = static_cast<int64_t>(runInfo_.ci1) * BLOCK_CUBE * runInfo_.kd * runInfo_.kh * runInfo_.kw;
    int64_t iterK = ops::CeilDiv(maxK, static_cast<int64_t>(BEST_BASE_K / dtypeByte_));
    int64_t iterN = ops::CeilDiv(maxN, static_cast<int64_t>(BEST_BASE_N));
    int32_t singleCoreHo = ops::CeilDiv(static_cast<int32_t>(runInfo_.ho), remainFactor);
    int32_t kDim = ops::CeilDiv(runInfo_.ho, static_cast<int32_t>(singleCoreHo));
    int32_t singleCoreCin = ops::CeilDiv(runInfo_.cin1_g, remainFactor) * BLOCK_CUBE;
    int32_t nDim = ops::CeilDiv(runInfo_.cin1_g * BLOCK_CUBE, singleCoreCin);
    bool kSplit = maxK >= static_cast<int64_t>(remainFactor) * (BEST_BASE_K / dtypeByte_) &&
                    runInfo_.ho >= remainFactor && kDim == remainFactor;
    bool nSplit = maxN >= remainFactor * static_cast<int64_t>(BEST_BASE_N) && runInfo_.ci1 >= remainFactor &&
                    nDim == remainFactor;
    // 两个else if的意义：首先希望从iter大的轴去切，但是泛化场景中，有些case是由于wo或者kh kw很大使得该轴iter很大，
    // 可以切分的ho cin反而不满足切分条件
    if (iterK >= iterN) {
        if (kSplit) {
            GetBatchDim(coreDim, dMaxFactor, batchDepthMaxFactor);
            coreDim.kDim = remainFactor;
        } else if (nSplit) {
            GetBatchDim(coreDim, dMaxFactor, batchDepthMaxFactor);
            coreDim.nDim = remainFactor;
        }
    } else {
        if (nSplit) {
            GetBatchDim(coreDim, dMaxFactor, batchDepthMaxFactor);
            coreDim.nDim = remainFactor;
        } else if (kSplit) {
            GetBatchDim(coreDim, dMaxFactor, batchDepthMaxFactor);
            coreDim.kDim = remainFactor;
        }
    }
    return;
}

void Conv3DBackpropFilterV2Tiling::SetTilingParamByDimInfo(TilingValueDw& tilingParams, CoreDimDw& coreDim)
{
    // singleCore
    tilingParams.singleCoreBatch = static_cast<uint64_t>(ops::CeilDiv(runInfo_.batch, coreDim.batchDim)) *
                                    ops::CeilDiv(runInfo_.dout, coreDim.dDim);
    tilingParams.singleCoreGroup = 1;
    tilingParams.singleCoreDk = 1;
    tilingParams.singleCoreCout = ops::CeilDiv(static_cast<int32_t>(runInfo_.cout1_g), coreDim.mDim) * BLOCK_CUBE;
    tilingParams.singleCoreHo = ops::CeilDiv(runInfo_.ho, coreDim.kDim);
    int32_t singleCoreCin = ops::CeilDiv(runInfo_.cin1_g, coreDim.nDim) * BLOCK_CUBE;
    tilingParams.singleCoreCin = static_cast<int64_t>(singleCoreCin) * runInfo_.kd;
    tilingParams.batchDim = coreDim.batchDim * coreDim.dDim;
    tilingParams.groupDim = 1;
    tilingParams.dkDim = 1;
    tilingParams.mDim =
        ops::CeilDiv(runInfo_.cout1_g * BLOCK_CUBE, static_cast<int32_t>(tilingParams.singleCoreCout));
    tilingParams.kDim = ops::CeilDiv(runInfo_.ho, static_cast<int32_t>(tilingParams.singleCoreHo));
    tilingParams.nDim = ops::CeilDiv(runInfo_.cin1_g * BLOCK_CUBE, singleCoreCin);
}

void Conv3DBackpropFilterV2Tiling::InitCalTilingValue(TilingValueDw& tilingParams)
{
    // L0
    tilingParams.baseM = tbeTiling_.m_l0 * BLOCK_CUBE;
    tilingParams.baseK = tbeTiling_.k_l0 * tilingData_.dwTiling.get_k0();
    tilingParams.baseN = tbeTiling_.n_l0 * BLOCK_CUBE;
    // step
    tilingParams.stepM = tbeTiling_.m_al1;
    tilingParams.stepN = tbeTiling_.n_bl1;
    tilingParams.stepKa = tbeTiling_.k_al1 / tbeTiling_.k_l0;
    tilingParams.stepKb = tbeTiling_.k_bl1 / tbeTiling_.k_l0;
    // pingpong buffer
    tilingParams.al0Pbuffer = DB_ON;  // 默认开
    tilingParams.bl0Pbuffer = DB_ON;  // 默认开
    tilingParams.cl0Pbuffer = static_cast<uint32_t>(tbeTiling_.db_l0c);
    tilingParams.al1Pbuffer = static_cast<uint32_t>(tbeTiling_.db_al1);
    tilingParams.bl1Pbuffer = static_cast<uint32_t>(tbeTiling_.db_bl1);
    // fix dbl1 tiling
    uint64_t singleCoreHoWo = static_cast<uint64_t>(tilingParams.singleCoreHo) * runInfo_.wo;
    uint64_t kIter = ops::CeilDiv(singleCoreHoWo, static_cast<uint64_t>(tilingParams.baseK));
    if (tilingParams.al1Pbuffer > 1 && tilingParams.stepKa >= kIter) {
        tilingParams.al1Pbuffer = 1;
    }
    if (tilingParams.bl1Pbuffer > 1 && tilingParams.stepKb >= kIter) {
        tilingParams.bl1Pbuffer = 1;
    }

    // Temperary fix for current logic: When k cannot be fully loaded
    // into BL1, kIter / stepKb > bl1Pbuffer, the buffer in N direction
    // cannot be fully utilized. Thus stepN should be set to be 1. However,
    // this fix should be removed once different iteration direction is
    // allowed in kernel, e.g. Iterate K direction first then iterate M/N
    // direction. Also, the same problem may appear in AL1, but currently
    // it is too troublesome to find a testcase to hit the condition. As
    // this is a temperary solution, we keep the fix for Bl1 only.
    if (tilingParams.stepN > 1 && ops::CeilDiv(kIter, static_cast<uint64_t>(tilingParams.stepKb)) >
        tilingParams.bl1Pbuffer) {
        tilingParams.stepN = 1;
        OP_LOGD(opName_, "stepN is set to be 1 when Ceil(kIter:%lu, stepKb:%d) > bl1Pbuffer:%d",
            kIter, tilingParams.stepKb, tilingParams.bl1Pbuffer);
    }

    tilingParams.iterateOrder = 1;
    tilingParams.bl1Bound = runInfo_.bl1_bound;
}

void Conv3DBackpropFilterV2Tiling::CalCoreDimTiling(TilingValueDw& tilingParams, bool& enableTbeBlock)
{
    CoreDimDw coreDim;
    GetCoreDim(coreDim, coreNum_);
    SetTilingParamByDimInfo(tilingParams, coreDim);
    // 至少用满80%的核，否则走tbe
    uint64_t coreNumUsed = static_cast<uint64_t>(tilingParams.batchDim) * tilingParams.mDim *
                            tilingParams.kDim * tilingParams.nDim;
    enableTbeBlock = coreNumUsed < static_cast<uint64_t>(coreNum_ * CORE_USED_THRESHOLD) ||
                        coreNumUsed > coreNum_;
}

uint32_t Conv3DBackpropFilterV2Tiling::CalCin(const uint32_t& nL1Size)
{
    uint32_t kernelHW = runInfo_.kh * runInfo_.kw;
    uint32_t bL1N = ops::CeilDiv(nL1Size, static_cast<uint32_t>(BLOCK_CUBE));
    uint32_t ci = ops::CeilDiv(bL1N, kernelHW);
    uint32_t extendLine = 0;
    if (kernelHW > bL1N) {
        if (kernelHW % bL1N != 0) {
            extendLine = 1;
        }
    } else {
        if (2 * bL1N % kernelHW == 0) { // 2: 尾块0.5
            extendLine = 1;
        } else {
            extendLine = ROW_NUM;
        }
    }
    return (ci + extendLine) * BLOCK_CUBE;
}

int64_t Conv3DBackpropFilterV2Tiling::CalBL1Bound(TilingValueDw &tilingParams)
{
    int64_t kBL1Size = static_cast<int64_t>(tilingParams.baseK) * tilingParams.stepKb;
    int32_t hoCal = CubeUtil::CalcHo(kBL1Size, runInfo_.wo, opName_);
    int32_t kernelHDilation = (runInfo_.kh - 1) * runInfo_.dilation_h + 1;
    int32_t hiCal = CubeUtil::CalcHi(hoCal, runInfo_.stride_h, kernelHDilation, runInfo_.hi);
    int32_t ci = CalCin(tilingParams.baseN);
    return static_cast<int64_t>(hiCal) * runInfo_.wi * ci;
}

void Conv3DBackpropFilterV2Tiling::UpdateBaseStep(uint32_t &stepKa, uint32_t &stepKb, TilingValueDw &tilingParams)
{
    // 根据A B的size，粗略分配al1 bl1
    uint64_t amat = tilingParams.singleCoreHo * static_cast<uint64_t>(runInfo_.wo) *
                    tilingParams.baseM * tilingParams.stepM;
    uint32_t ci = CalCin(tilingParams.baseN * tilingParams.stepN);
    int32_t kernelHDilation = (runInfo_.kh - 1) * runInfo_.dilation_h + 1;
    uint64_t bmat = static_cast<uint64_t>(runInfo_.wi) * static_cast<uint64_t>(ci) *
                    CubeUtil::CalcHi(static_cast<int32_t>(tilingParams.singleCoreHo),
                                            runInfo_.stride_h, kernelHDilation, runInfo_.hi);
    float abRatio = static_cast<float>(amat) / static_cast<float>(amat + bmat);
    uint64_t al1 = static_cast<uint64_t>(L1_SIZE * abRatio) / tilingParams.al1Pbuffer;
    uint64_t bl1 = static_cast<uint64_t>(L1_SIZE * (1 - abRatio)) / tilingParams.bl1Pbuffer;
    // 根据al1反推hoA1,向下取整是为了不超过al1，但是要保证至少搬一行
    uint32_t hoA1 = std::max(static_cast<uint64_t>(1),
                             al1 / (static_cast<uint64_t>(runInfo_.wo) * tilingParams.baseM *
                                    tilingParams.stepM * dtypeByte_));
    stepKa = std::max(static_cast<uint64_t>(MIN_STEPK),
                        hoA1 * static_cast<uint64_t>(runInfo_.wo) / tilingParams.baseK);
    // 根据bl1反推hi,hi反推ho, 保证至少搬一行
    uint32_t hi = std::max(static_cast<uint64_t>(1),
                            bl1 / (static_cast<uint64_t>(runInfo_.wi) * ci * dtypeByte_));
    uint32_t hoB1 = std::max(static_cast<int64_t>(1),
                             static_cast<int64_t>((hi - 1 - (runInfo_.kh - 1) * runInfo_.dilation_h) /
                                                    runInfo_.stride_h + 1));
    stepKb = std::max(static_cast<uint64_t>(MIN_STEPK),
                        hoB1 * static_cast<uint64_t>(runInfo_.wo) / tilingParams.baseK);
    // 计算stepK的最大值
    uint64_t maxKIter = ops::CeilDiv(tilingParams.singleCoreHo * static_cast<uint64_t>(runInfo_.wo),
                                     static_cast<uint64_t>(tilingParams.baseK));
    stepKa = std::min(maxKIter, static_cast<uint64_t>(stepKa));
    stepKb = std::min(maxKIter, static_cast<uint64_t>(stepKb));
    // 调整Ka Kb为倍数关系
    if (stepKa > stepKb) {
        stepKa = stepKa / stepKb * stepKb;
    } else {
        stepKb = stepKb / stepKa * stepKa;
    }
}

void Conv3DBackpropFilterV2Tiling::UpdateBaseBlock(uint32_t& baseM, uint32_t& baseK, uint32_t& baseN,
    TilingValueDw& tilingParams)
{
    uint64_t maxSingleCoreK = tilingParams.singleCoreHo * static_cast<uint64_t>(runInfo_.wo);
    uint64_t maxSingleCoreN = tilingParams.singleCoreCin * static_cast<uint64_t>(runInfo_.kh) *
                                static_cast<uint64_t>(runInfo_.kw);
    // 如果baseM baseN很小，适当增大baseK，使得L0装尽可能多的数据，例如baseM&baseN<=128，baseK：64-->128
    baseM = (tilingParams.singleCoreCout > BEST_BASE_M) ? BEST_BASE_M : tilingParams.singleCoreCout;
    baseN = (maxSingleCoreN > static_cast<uint64_t>(BEST_BASE_N)) ? BEST_BASE_N : maxSingleCoreN;
    std::vector<uint32_t> baseFactor ={2, 4, 8};
    for (uint32_t num : baseFactor) {
        if (baseM * num <= BEST_BASE_N && baseN * num <= BEST_BASE_N) {
            baseK = (maxSingleCoreK > static_cast<uint64_t>(BEST_BASE_K / dtypeByte_ * num)) ?
                    (BEST_BASE_K / dtypeByte_ * num) :
                    ops::CeilAlign(maxSingleCoreK, static_cast<uint64_t>(BLOCK_CUBE));
        }
    }
    // 如果baseM <= 64 && maxSingleCoreN >= 512, 改用基本块组合SECOND_BASE_MKN:64 32 512
    if (baseM <= SECOND_BASE_M && maxSingleCoreK < maxSingleCoreN && SECOND_BASE_N <= maxSingleCoreN) {
            baseK = (maxSingleCoreK > static_cast<uint64_t>(SECOND_BASE_K / dtypeByte_)) ?
                    (SECOND_BASE_K / dtypeByte_) :
                    ops::CeilAlign(maxSingleCoreK, static_cast<uint64_t>(BLOCK_CUBE));
            baseN = (maxSingleCoreN > static_cast<uint64_t>(SECOND_BASE_N)) ? (SECOND_BASE_N) : maxSingleCoreN;
    }
}

bool Conv3DBackpropFilterV2Tiling::CalBaseBlockTiling(TilingValueDw& tilingParams)
{
    // 默认开启double buffer
    tilingParams.al0Pbuffer = DB_ON;
    tilingParams.bl0Pbuffer = DB_ON;
    tilingParams.cl0Pbuffer = 1;
    tilingParams.al1Pbuffer = DB_ON;
    tilingParams.bl1Pbuffer = DB_ON;
    tilingParams.iterateOrder = 1;
    // 默认采用最优基本块tiling
    uint32_t baseM = BEST_BASE_M;
    uint32_t baseN = BEST_BASE_N;
    uint32_t baseK = BEST_BASE_K / dtypeByte_;
    tilingParams.stepM = 1;
    tilingParams.stepN = 1;
    uint32_t stepKa = 1;
    uint32_t stepKb = 1;
    bool enableTBEtiling = false;
    // 调整baseM baseN baseK
    UpdateBaseBlock(baseM, baseK, baseN, tilingParams);
    tilingParams.baseM = baseM;
    tilingParams.baseN = baseN;
    tilingParams.baseK = baseK;
    // 调整stepK,尽量用满L1
    UpdateBaseStep(stepKa, stepKb, tilingParams);
    tilingParams.stepKa = stepKa;
    tilingParams.stepKb = stepKb;
    uint64_t bl1Bound = CalBL1Bound(tilingParams);
    tilingParams.bl1Bound = bl1Bound;
    uint64_t l1UsedSize = bl1Bound * tilingParams.bl1Pbuffer * dtypeByte_ +
                          tilingParams.stepM * tilingParams.baseM * tilingParams.baseK *
                          tilingParams.stepKa * tilingParams.al1Pbuffer * dtypeByte_;
    if (!enableTBEtiling && CheckL0Size(baseM, baseN, baseK, dtypeByte_) && l1UsedSize < L1_SIZE) {
        return true;
    }
    return false;
}

void Conv3DBackpropFilterV2Tiling::InitTilingValue(TilingValueDw& tilingParams)
{
    if (!enableDeterministic_) {
        bool enableTbeBlock = true;
        if (aDtype_ == ge::DT_BF16 && coreNum_ == CORE_NUM_910B3 && runInfo_.real_g == 1 && runInfo_.dilation_d == 1) {
            CalCoreDimTiling(tilingParams, enableTbeBlock);
        }
        if (!enableTbeBlock && CalBaseBlockTiling(tilingParams)) {
            return;
        }
    }
    // singleCore
    tilingParams.singleCoreBatch = static_cast<uint64_t>(ops::CeilDiv(runInfo_.batch, tbeTiling_.batch_dim)) *
                            ops::CeilDiv(runInfo_.dout, tbeTiling_.d_dim);
    tilingParams.singleCoreGroup = ops::CeilDiv(static_cast<int32_t>(runInfo_.real_g), tbeTiling_.group_dim);

    uint32_t c0 = tilingData_.dwTiling.get_channelSize();
    // group>1时cin要分group计算，无法和dk合并，单独分核处理
    if (runInfo_.real_g > 1 || runInfo_.dilation_d > 1 || enableDeterministic_) {
        tilingParams.dkDim = optiling::cachetiling::MathUtil::GetGcd(tbeTiling_.n_dim, runInfo_.kd);
        tilingParams.singleCoreDk = ops::CeilDiv(runInfo_.kd, static_cast<int32_t>(tilingParams.dkDim));
        tilingParams.nDim = tbeTiling_.n_dim / tilingParams.dkDim;
        tilingParams.singleCoreCin =
            ops::CeilDiv(static_cast<uint32_t>(runInfo_.cin1_g), tilingParams.nDim) * c0;
        tilingParams.nDim =
            ops::CeilDiv(static_cast<uint64_t>(runInfo_.cin1_g) * c0, tilingParams.singleCoreCin);
    } else {
        tilingParams.dkDim = 1;
        tilingParams.singleCoreDk = 1;
        tilingParams.singleCoreCin =
            ops::CeilDiv(static_cast<uint32_t>(runInfo_.cin1_g) * runInfo_.kd,
                static_cast<uint32_t>(tbeTiling_.n_dim)) * c0;
        tilingParams.nDim =
            ops::CeilDiv(static_cast<uint64_t>(runInfo_.cin1_g) * c0 * runInfo_.kd, tilingParams.singleCoreCin);
    }
    tilingParams.singleCoreCout =
            ops::CeilAlign(ops::CeilDiv(static_cast<int32_t>(runInfo_.cout1_g), tbeTiling_.m_dim) * c0,
                static_cast<uint32_t>(BLOCK_CUBE));
    tilingParams.mDim =
            ops::CeilDiv(runInfo_.cout1_g * c0, tilingParams.singleCoreCout);
    tilingParams.singleCoreHo = ops::CeilDiv(static_cast<int32_t>(runInfo_.ho), tbeTiling_.k_dim);
    // core alloc
    tilingParams.batchDim = ops::CeilDiv(static_cast<int64_t>(runInfo_.batch) * runInfo_.dout,
                                         static_cast<int64_t>(runInfo_.batch_dout_single_core));
    tilingParams.groupDim = tbeTiling_.group_dim;
    tilingParams.kDim = ops::CeilDiv(runInfo_.ho, static_cast<int32_t>(tilingParams.singleCoreHo));
    InitCalTilingValue(tilingParams);
}

void Conv3DBackpropFilterV2Tiling::SetTilingValue(TConv3DDwTiling &dwt, const TilingValueDw& tilingParams)
{
    tilingData_.params.set_groupDim(tilingParams.groupDim);
    tilingData_.params.set_batchDim(tilingParams.batchDim);
    tilingData_.params.set_dkDim(tilingParams.dkDim);
    tilingData_.params.set_mDim(tilingParams.mDim);
    tilingData_.params.set_nDim(tilingParams.nDim);
    tilingData_.params.set_kDim(tilingParams.kDim);
    // singleCore
    dwt.set_singleCoreGroup(tilingParams.singleCoreGroup);
    dwt.set_singleCoreBatch(tilingParams.singleCoreBatch);
    dwt.set_singleCoreDk(tilingParams.singleCoreDk);
    dwt.set_singleCoreCout(tilingParams.singleCoreCout);
    dwt.set_singleCoreCin(tilingParams.singleCoreCin);
    dwt.set_singleCoreHo(tilingParams.singleCoreHo);
    // L0
    dwt.set_baseM(tilingParams.baseM);
    dwt.set_baseN(tilingParams.baseN);
    dwt.set_baseK(tilingParams.baseK);
    // step
    dwt.set_stepM(tilingParams.stepM);
    dwt.set_stepN(tilingParams.stepN);
    dwt.set_stepKa(tilingParams.stepKa);
    dwt.set_stepKb(tilingParams.stepKb);
    // pingpong buffer
    dwt.set_al1Pbuffer(tilingParams.al1Pbuffer);
    dwt.set_bl1Pbuffer(tilingParams.bl1Pbuffer);
    dwt.set_al0Pbuffer(tilingParams.al0Pbuffer);
    dwt.set_bl0Pbuffer(tilingParams.bl0Pbuffer);
    dwt.set_cl0Pbuffer(tilingParams.cl0Pbuffer);
    // iterateOrder
    dwt.set_bl1Bound(tilingParams.bl1Bound);
    dwt.set_iterateOrder(tilingParams.iterateOrder);
}

void Conv3DBackpropFilterV2Tiling::SetDwTilingFromTbeTiling()
{
    TConv3DDwTiling &dwt = tilingData_.dwTiling;
    TilingValueDw tilingParams;
    SetShapeTiling(dwt);
    SetAttrTiling(dwt);
    if (dwt.get_k0() != BLOCK_CUBE) {
        InitTilingValue(tilingParams);
        SetTilingValue(dwt, tilingParams);
        return;
    }
    // key:
    // "N_Do_Co1_Ho_Wo_Ci1_Hi_Wi_Dk_Hk_Wk_strideD_strideH_strideW_
    // _padFront_padBack_padUp_padDown_padLeft_padRight_dilationD_dilationH_dilationW"
    std::string key = std::to_string(runInfo_.batch) + "_" + std::to_string(runInfo_.dout) + "_" +
                    std::to_string(runInfo_.cout1_g) + "_" + std::to_string(runInfo_.ho) + "_" +
                    std::to_string(runInfo_.wo) + "_" + std::to_string(runInfo_.cin1_g) + "_" +
                    std::to_string(runInfo_.hi) + "_" +
                    std::to_string(runInfo_.wi) + "_" + std::to_string(runInfo_.kd) + "_" +
                    std::to_string(runInfo_.kh) + "_" + std::to_string(runInfo_.kw) + "_" +
                    std::to_string(runInfo_.stride_d) + "_" + std::to_string(runInfo_.stride_h) + "_" +
                    std::to_string(runInfo_.stride_w) + "_" + std::to_string(runInfo_.pad_f) + "_" +
                    std::to_string(runInfo_.pad_b) + "_" + std::to_string(runInfo_.pad_u) + "_" +
                    std::to_string(runInfo_.pad_d) + "_" + std::to_string(runInfo_.pad_l) + "_" +
                    std::to_string(runInfo_.pad_r) + "_" + std::to_string(runInfo_.dilation_d) + "_" +
                    std::to_string(runInfo_.dilation_h) + "_" + std::to_string(runInfo_.dilation_w);
    if (coreNum_ == CORE_NUM_910B3 && TILINGDATA_MAP_B3.find(key) != TILINGDATA_MAP_B3.end() && !enableDeterministic_) {
        tilingParams = TILINGDATA_MAP_B3.at(key);
    } else if (coreNum_ == CORE_NUM_910B2 && TILINGDATA_MAP_B2.find(key) != TILINGDATA_MAP_B2.end() && !enableDeterministic_) {
        tilingParams = TILINGDATA_MAP_B2.at(key);
    } else {
        InitTilingValue(tilingParams);
    }
    SetTilingValue(dwt, tilingParams);
}

void Conv3DBackpropFilterV2Tiling::PrintTilingData()
{
    TConv3DDwTiling &tiling = tilingData_.dwTiling;
    std::stringstream ss;
    ss << "batch: " << tiling.get_batch() << " cin: " << tiling.get_cin() << " cout: " << tiling.get_cout()
       << " cin1G: " << tiling.get_cin1G() << " cout1G: " << tiling.get_cout1G() << " dout: " << tiling.get_dout()
       << " ho: " << tiling.get_ho() << " wo: " << tiling.get_wo() << " di: " << tiling.get_di()
       << " hi: " << tiling.get_hi() << " wi: " << tiling.get_wi() << " dk: " << tiling.get_dk()
       << " hk: " << tiling.get_hk() << " wk: " << tiling.get_wk() << " group: " << tiling.get_group()
       << " strideD: " << tiling.get_strideD() << " strideH: " << tiling.get_strideH()
       << " strideW: " << tiling.get_strideW() << " padFront: " << tiling.get_padFront()
       << " padBack: " << tiling.get_padBack() << " padUp: " << tiling.get_padUp()
       << " padDown: " << tiling.get_padDown() << " padLeft: " << tiling.get_padLeft()
       << " padRight: " << tiling.get_padRight() << " dilationD: " << tiling.get_dilationD()
       << " dilationH: " << tiling.get_dilationH() << " dilationW: " << tiling.get_dilationW()
       << " channelSize: " << tiling.get_channelSize() << " al0Pbuffer: " << tiling.get_al0Pbuffer()
       << " bl0Pbuffer: " << tiling.get_bl0Pbuffer() << " cl0Pbuffer: " << tiling.get_cl0Pbuffer()
       << " al1Pbuffer: " << tiling.get_al1Pbuffer() << " bl1Pbuffer: " << tiling.get_bl1Pbuffer()
       << " singleCoreBatch: " << tiling.get_singleCoreBatch() << " singleCoreGroup: " << tiling.get_singleCoreGroup()
       << " singleCoreCout: " << tiling.get_singleCoreCout() << " singleCoreCin: " << tiling.get_singleCoreCin()
       << " singleCoreHo: " << tiling.get_singleCoreHo() << " baseM: " << tiling.get_baseM()
       << " baseK: " << tiling.get_baseK() << " baseN: " << tiling.get_baseN() << " m0: " << tiling.get_m0()
       << " k0: " << tiling.get_k0() << " n0: " << tiling.get_n0() << " stepM: " << tiling.get_stepM()
       << " stepN: " << tiling.get_stepN() << " stepKa: " << tiling.get_stepKa() << " stepKb: " << tiling.get_stepKb()
       << " iterateOrder: " << tiling.get_iterateOrder() << " hf32Flag: " << tiling.get_hf32Flag()
       << " enableDeterministic_: " << (enableDeterministic_ ? "true" : "false");

    // OP_LOG_FULL(DLOG_DEBUG, opName_, "api tiling: %s", ss.str().c_str());
}

REGISTER_TILING_TEMPLATE("Conv3DBackpropFilterV2", Conv3DBackpropFilterV2Tiling, 1);

static ge::graphStatus Conv3DBackpropFilterV2TilingFunc(gert::TilingContext *context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingParseForConv3DBackpropFilterV2(gert::TilingParseContext *context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "platformInfoPtr is null");
    auto compileInfoPtr = context->GetCompiledInfo<Conv3DBackpropFilterV2CompileInfo>();
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "compileInfo is null");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->ParseRuntimePlatformInfo(context->GetNodeName(), *platformInfoPtr);
    compileInfoPtr->core_num = ascendcPlatform.GetCoreNumAic();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Conv3DBackpropFilterV2)
    .Tiling(Conv3DBackpropFilterV2TilingFunc)
    .TilingParse<Conv3DBackpropFilterV2CompileInfo>(TilingParseForConv3DBackpropFilterV2);
}  // namespace optiling








