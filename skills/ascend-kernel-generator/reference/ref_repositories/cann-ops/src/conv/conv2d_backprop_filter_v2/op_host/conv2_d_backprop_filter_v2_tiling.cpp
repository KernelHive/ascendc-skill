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
 * \file conv2_d_backprop_filter_v2_tiling.cpp
 * \brief
 */

#include "conv2d_backprop_filter_v2_tiling.h"

#include <map>
#include <numeric>

#include "tiling/tiling_templates_registry.h"
#include "tiling/tiling_type.h"
#include "cube_tiling_runtime.h"
#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "register/op_impl_registry.h"

namespace {
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
constexpr int32_t BYTE_BLOCK = 32;
constexpr uint32_t DB_ON = 2;
constexpr uint32_t C04 = 4;
constexpr uint32_t C16 = 16;
using Conv2DBackpropFilterV2CompileInfo = optiling::Conv2DBackPropCompileInfo;
constexpr uint64_t MAX_UINT16 = 65535;
constexpr uint64_t MAX_UINT32 = 4294967295;
}  // namespace

namespace optiling {

void Conv2DBackpropFilterV2Tiling::Reset()
{
    tilingData_.SetDataPtr(context_->GetRawTilingData()->GetData());
    opName_ = nullptr;
}

ge::graphStatus Conv2DBackpropFilterV2Tiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2DBackpropFilterV2Tiling::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

bool Conv2DBackpropFilterV2Tiling::IsCapable() { return true; }

ge::graphStatus Conv2DBackpropFilterV2Tiling::DoOpTiling()
{
    if (!GetTbeTiling()) {
        OP_LOGE(context_->GetNodeName(), "GetTbeTiling failed");
        return ge::GRAPH_FAILED;
    }
    tilingData_.params.set_batchDim(tbeTiling_.batch_dim);
    tilingData_.params.set_groupDim(tbeTiling_.group_dim);
    tilingData_.params.set_mDim(tbeTiling_.m_dim);
    tilingData_.params.set_kDim(tbeTiling_.k_dim);
    tilingData_.params.set_nDim(tbeTiling_.n_dim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2DBackpropFilterV2Tiling::DoLibApiTiling()
{
    SetDwTilingFromTbeTiling();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

uint64_t Conv2DBackpropFilterV2Tiling::GetTilingKey() const { return 0; }

ge::graphStatus Conv2DBackpropFilterV2Tiling::GetWorkspaceSize()
{
    const Conv2DBackPropCompileInfo *compileInfoPtr =
        reinterpret_cast<const Conv2DBackPropCompileInfo *>(context_->GetCompileInfo());
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = compileInfoPtr->lib_api_workspace_size;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2DBackpropFilterV2Tiling::PostTiling()
{
    OP_LOGD(opName_, "final tiling data size: %zu", tilingData_.GetDataSize());

    OP_TILING_CHECK(tilingData_.GetDataSize() % sizeof(uint64_t) != 0,
                    CUBE_INNER_ERR_REPORT(opName_, "tiling data size[%zu] not aligned to 8", tilingData_.GetDataSize()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTilingData(),
                    CUBE_INNER_ERR_REPORT(opName_, "check tiling data failed"),
                    return ge::GRAPH_FAILED);
    context_->SetBlockDim(tbeTiling_.m_dim * tbeTiling_.n_dim * tbeTiling_.batch_dim * tbeTiling_.k_dim);
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

bool Conv2DBackpropFilterV2Tiling::GetTbeTiling()
{
    const Conv2DBackPropCompileInfo *compileInfoPtr =
        reinterpret_cast<const Conv2DBackPropCompileInfo *>(context_->GetCompileInfo());
    cachetiling::Conv2DBpFilterTilingParam tilingParams(cachetiling::kConv2DBackpropFilterV2);
    tilingParams.aub_fused_num = compileInfoPtr->aub_num;
    tilingParams.bub_fused_num = compileInfoPtr->bub_num;
    tilingParams.binary_mode = compileInfoPtr->binary_mode;
    tilingParams.static_flag = compileInfoPtr->is_static_type;
    blockSize_ = BYTE_BLOCK / tilingParams.a_dtype_bytes;
    OP_LOGE_IF(!tilingParams.ParseOpInfo(context_, *compileInfoPtr), false, opName_,
               "Parse cache tiling params failed");
    GetChannelVal(tilingParams);
    if(!cachetiling::GetTiling<cachetiling::Conv2DBpFilterTilingParam, cachetiling::Conv2DBpFilterTiling,
                               Conv2dBpFilterRunInfo, cachetiling::Conv2DBpTilingHashParam,
                               cachetiling::Conv2DBpFilterHashItem>(tilingParams, tbeTiling_, runInfo_)) {
        OP_LOGE(opName_, "GetTiling interface failed");
        return false;
    }
    return true;
}

// C04: kernelH、kernelW不全为1 & fmap_c <=4
void Conv2DBackpropFilterV2Tiling::GetChannelVal(cachetiling::Conv2DBpFilterTilingParam &tilingParams) {
    if (tilingParams.b_shape.c <= C04 && (tilingParams.c_shape.h != 1 || tilingParams.c_shape.w != 1) &&
        tilingParams.b_shape.c > 0) {
        channelVal_ = C04;
    } else {
        channelVal_ = C16;
    }
}

void Conv2DBackpropFilterV2Tiling::SetDwTilingFromTbeTiling()
{
    TConvTiling &dwt = tilingData_.dwTiling;
    // shape
    dwt.set_N(runInfo_.batch);
    dwt.set_Cin(runInfo_.ci);
    dwt.set_Cout(runInfo_.co);
    dwt.set_Ho(runInfo_.ho); // dedy h
    dwt.set_Wo(runInfo_.wo); // dedy o
    dwt.set_Hi(runInfo_.hi);
    dwt.set_Wi(runInfo_.wi);
    dwt.set_Hk(runInfo_.kh);
    dwt.set_Wk(runInfo_.kw);
    // singleCore
    dwt.set_singleCoreBatch(ops::CeilDiv(static_cast<int32_t>(runInfo_.batch), tbeTiling_.batch_dim));
    dwt.set_singleCoreGroup(1);
    dwt.set_singleCoreCout(ops::CeilDiv(ops::CeilDiv(static_cast<int32_t>(runInfo_.co), tbeTiling_.m_dim), blockSize_) *
                           blockSize_);
    dwt.set_singleCoreCin(ops::CeilDiv(ops::CeilDiv(static_cast<int32_t>(runInfo_.ci), tbeTiling_.n_dim), blockSize_) *
                          blockSize_);
    dwt.set_singleCoreHo(ops::CeilDiv(static_cast<int32_t>(runInfo_.ho), tbeTiling_.k_dim)); // Ho / k_dim
    // blockSize_
    dwt.set_m0(blockSize_);
    dwt.set_k0(blockSize_);
    dwt.set_n0(blockSize_);
    // L0
    dwt.set_baseM(tbeTiling_.m_l0 * blockSize_);
    dwt.set_baseK(tbeTiling_.k_l0 * blockSize_);
    dwt.set_baseN(tbeTiling_.n_l0 * blockSize_);
    dwt.set_baseBatch(1);
    dwt.set_baseGroup(1);
    // step
    dwt.set_stepM(tbeTiling_.m_al1);
    dwt.set_stepN(tbeTiling_.n_bl1);
    dwt.set_stepKa(tbeTiling_.k_al1 / tbeTiling_.k_l0);
    dwt.set_stepKb(tbeTiling_.k_bl1 / tbeTiling_.k_l0);
    dwt.set_stepBatch(1);
    dwt.set_stepGroup(1);
    // pingpong buffer
    dwt.set_al0Pbuffer(DB_ON); // 默认开
    dwt.set_bl0Pbuffer(DB_ON); // 默认开
    dwt.set_cl0Pbuffer(static_cast<uint32_t>(tbeTiling_.db_l0c));
    dwt.set_al1Pbuffer(static_cast<uint32_t>(tbeTiling_.db_al1));
    dwt.set_bl1Pbuffer(static_cast<uint32_t>(tbeTiling_.db_bl1));
    // attr
    dwt.set_group(1);
    dwt.set_strideH(runInfo_.stride_h);
    dwt.set_strideW(runInfo_.stride_w);
    dwt.set_padL(runInfo_.pad_l); // left
    dwt.set_padR(runInfo_.pad_r); // right
    dwt.set_padT(runInfo_.pad_u); // top
    dwt.set_padB(runInfo_.pad_d); // below
    dwt.set_dilationH(runInfo_.dilation_h);
    dwt.set_dilationW(runInfo_.dilation_w);
    dwt.set_channelSize(channelVal_); // c16:16, c04:4
    dwt.set_iterateOrder(1);
}

bool Conv2DBackpropFilterV2Tiling::CheckTilingData() const {
    TConvTiling dwTiling = tilingData_.dwTiling;
    Conv2DBackpropFilterV2Params params = tilingData_.params;

    OP_TILING_CHECK(static_cast<uint64_t>(params.get_nDim()) * dwTiling.get_singleCoreCin() * dwTiling.get_Hk() * dwTiling.get_Wk() > MAX_UINT32,
                    CUBE_INNER_ERR_REPORT(opName_, "Cin tiling values is too large, please reset tiling values"),
                    return false);
    OP_TILING_CHECK(static_cast<uint64_t>(params.get_kDim()) * dwTiling.get_singleCoreHo() * dwTiling.get_Wo() > MAX_UINT32,
                    CUBE_INNER_ERR_REPORT(opName_, "Ho tiling values is too large, please reset tiling values"),
                    return false);
    uint32_t kRamin = dwTiling.get_Ho() - params.get_kDim() * dwTiling.get_singleCoreHo();
    uint32_t singleShapeK_ = kRamin < dwTiling.get_singleCoreHo() * dwTiling.get_Wo() ? kRamin : dwTiling.get_singleCoreHo() * dwTiling.get_Wo();
    uint32_t singleShapeHo_ = singleShapeK_ / dwTiling.get_Wo();
    uint64_t srcStride = static_cast<uint64_t>(dwTiling.get_Ho()) * dwTiling.get_Wo() -
                         static_cast<uint64_t>(singleShapeHo_) * dwTiling.get_Wo();
    OP_TILING_CHECK(srcStride > MAX_UINT16,
                    CUBE_INNER_ERR_REPORT(opName_, "dataCopyParam srcStride will be too large, please reset tiling value"), return false);
    uint64_t blockLenMax = static_cast<uint64_t>(singleShapeHo_) * dwTiling.get_Wo();
    OP_TILING_CHECK(blockLenMax > MAX_UINT16,
                    CUBE_INNER_ERR_REPORT(opName_, "dataCopyParam blockLen will be too large, please reset tiling value"), return false);
    uint64_t dstStrideLimit = blockLenMax * 32 - static_cast<uint64_t>(singleShapeHo_) * dwTiling.get_Wo();
    OP_TILING_CHECK(dstStrideLimit > MAX_UINT16,
                    CUBE_INNER_ERR_REPORT(opName_, "dataCopyParam dstStride will be too large, please reset tiling value"), return false);
    uint64_t blockLen = static_cast<uint64_t>(dwTiling.get_stepKa()) * dwTiling.get_baseK();
    OP_TILING_CHECK(blockLen > MAX_UINT16,
                    CUBE_INNER_ERR_REPORT(opName_, "dataCopyParam blockLen will be too large, please reset tiling value"), return false);
    uint64_t blockLen2B1Max = static_cast<uint64_t>(singleShapeHo_) * dwTiling.get_strideH() * dwTiling.get_Wi() * 4;
    OP_TILING_CHECK(blockLen2B1Max > MAX_UINT16,
                    CUBE_INNER_ERR_REPORT(opName_, "dataCopyParam blockLen will be too large, please reset tiling value"), return false);

    return true;
}

void Conv2DBackpropFilterV2Tiling::PrintTilingData() {
  TConvTiling &tiling = tilingData_.dwTiling;
  std::stringstream ss;
  ss << " N: " << tiling.get_N() << " Cin: " << tiling.get_Cin() << " Cout: " << tiling.get_Cout()
     << " Ho: " << tiling.get_Ho() << " Wo: " << tiling.get_Wo() << " Hi: " << tiling.get_Hi()
     << " Wi: " << tiling.get_Wi() << " Hk: " << tiling.get_Hk() << " Wk: " << tiling.get_Wk()
     << " singleCoreBatch: " << tiling.get_singleCoreBatch() << " singleCoreGroup: " << tiling.get_singleCoreGroup()
     << " singleCoreCout: " << tiling.get_singleCoreCout() << " singleCoreCin: " << tiling.get_singleCoreCin()
     << " singleCoreHo: " << tiling.get_singleCoreHo() << " m0: " << tiling.get_m0() << " k0: " << tiling.get_k0()
     << " n0: " << tiling.get_n0() << " baseM: " << tiling.get_baseM() << " baseK: " << tiling.get_baseK()
     << " baseN: " << tiling.get_baseN() << " baseBatch: " << tiling.get_baseBatch()
     << " baseGroup: " << tiling.get_baseGroup() << " stepM: " << tiling.get_stepM() << " stepN: " << tiling.get_stepN()
     << " stepKa: " << tiling.get_stepKa() << " stepKb: " << tiling.get_stepKb()
     << " stepBatch: " << tiling.get_stepBatch() << " stepGroup: " << tiling.get_stepGroup()
     << " al0Pbuffer: " << tiling.get_al0Pbuffer() << " bl0Pbuffer: " << tiling.get_bl0Pbuffer()
     << " cl0Pbuffer: " << tiling.get_cl0Pbuffer() << " al1Pbuffer: " << tiling.get_al1Pbuffer()
     << " bl1Pbuffer: " << tiling.get_bl1Pbuffer() << " group: " << tiling.get_group()
     << " strideH: " << tiling.get_strideH() << " strideW: " << tiling.get_strideW() << " padT: " << tiling.get_padT()
     << " padB: " << tiling.get_padB() << " padL: " << tiling.get_padL() << " padR: " << tiling.get_padR()
     << " dilationH: " << tiling.get_dilationH() << " dilationW: " << tiling.get_dilationW()
     << " channelSize: " << tiling.get_channelSize() << " iterateOrder: " << tiling.get_iterateOrder();
    OP_LOGD(opName_, "api tiling: %s", ss.str().c_str());
}

REGISTER_TILING_TEMPLATE("Conv2DBackpropFilterV2", Conv2DBackpropFilterV2Tiling, 0);

static ge::graphStatus Conv2DBackpropFilterV2TilingFunc(gert::TilingContext *context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingParseForConv2DBackpropFilterV2(gert::TilingParseContext *context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "platformInfoPtr is null");
    auto compileInfoPtr = context->GetCompiledInfo<Conv2DBackpropFilterV2CompileInfo>();
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "compileInfo is null");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->ParseRuntimePlatformInfo(context->GetNodeName(), *platformInfoPtr);
    compileInfoPtr->core_num = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->lib_api_workspace_size = ascendcPlatform.GetLibApiWorkSpaceSize();
    cachetiling::PlatformInfo platformInstance;
    platformInstance.SetRuntimePlatformInfo(*compileInfoPtr);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Conv2DBackpropFilterV2)
    .Tiling(Conv2DBackpropFilterV2TilingFunc)
    .TilingParse<Conv2DBackpropFilterV2CompileInfo>(TilingParseForConv2DBackpropFilterV2);
}  // namespace optiling
