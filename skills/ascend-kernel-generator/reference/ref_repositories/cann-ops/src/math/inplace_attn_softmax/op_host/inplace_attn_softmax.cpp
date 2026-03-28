/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file inplace_attn_softmax.cpp
 */

#include "inplace_attn_softmax_tiling_utils.h"
#include "inplace_attn_softmax_tiling.h"

#include <cstdio>

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

constexpr uint32_t WORK_SPACE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t SINGLE_UB_SIZE_FLOAT16 = 3;
constexpr uint32_t SINGLE_UB_SIZE_BFLOAT16 = 6;
constexpr uint32_t SoftMaxTilingSize = 4;

static std::map<const ge::DataType, const uint32_t> x_dTypeLen = {
    {ge::DT_FLOAT16, 2}, {ge::DT_BF16, 2}, {ge::DT_FLOAT, 4}};

inline static ge::graphStatus SetTilingDataForInplaceAttnSoftmax(
    gert::TilingContext *context, InplaceAttnSoftmaxTilingData &tilingData)
{
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetCompileInfo(gert::TilingContext *context, InplaceAttnSoftmaxCompileInfo &compileInfo)
{
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t totalCoreNum = ascendcPlatform.GetCoreNumAiv();  // 总核数
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    uint32_t ubSize = static_cast<uint32_t>(ubSizePlatForm);

    if (totalCoreNum <= 0 || ubSize <= 0) {
        printf("GetCompileInfo Failed, coreNum:%d, ubSize:%d. \n", totalCoreNum, ubSize);
        return ge::GRAPH_FAILED;
    }
    compileInfo.totalCore = totalCoreNum;
    compileInfo.ubSize = ubSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetTillingData(gert::TilingContext *context, InplaceAttnSoftmaxCompileInfo &compileInfo,
    InplaceAttnSoftmaxTilingParam &tilingParam, InplaceAttnSoftmaxTilingData &tilingData)
{
    // 1. 参数校验
    if (CheckOpParams(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto xDtype = context->GetInputDesc(0)->GetDataType();
    // 2. 计算rowlen和collen
    auto inputShape = context->GetInputTensor(INPUT_X_INDEX)->GetStorageShape();
    if (!SetTotalShape(inputShape, context, tilingData)) {
        printf("Get total shape failed!");
        return ge::GRAPH_FAILED;
    }
    //3、多维张量合成两轴
    uint32_t rowLen = tilingData.get_rowLen();
    //4、设置key、单个ub可容纳的数据量
    if (xDtype == ge::DT_FLOAT16) {
        tilingData.set_tilingKey(static_cast<uint32_t>(InplaceAttnSoftmaxTilingKey::TILINGKEY_FP16));
        compileInfo.inputDataByte = x_dTypeLen[xDtype];
        uint32_t used_ubSize = compileInfo.ubSize / SINGLE_UB_SIZE_BFLOAT16;
        compileInfo.dataNumSingleUb = used_ubSize / compileInfo.inputDataByte;
        compileInfo.block_num = BLOCK_SIZE / compileInfo.inputDataByte;
    } else if (xDtype == ge::DT_BF16) {
        tilingData.set_tilingKey(static_cast<uint32_t>(InplaceAttnSoftmaxTilingKey::TILINGKEY_BF16));
        compileInfo.inputDataByte = x_dTypeLen[xDtype];
        uint32_t used_ubSize = compileInfo.ubSize / SINGLE_UB_SIZE_BFLOAT16;
        compileInfo.dataNumSingleUb = used_ubSize / compileInfo.inputDataByte;
        compileInfo.block_num = BLOCK_SIZE / compileInfo.inputDataByte;
    } else {
        tilingData.set_tilingKey(static_cast<uint32_t>(InplaceAttnSoftmaxTilingKey::TILINGKEY_FP32));
        compileInfo.inputDataByte = x_dTypeLen[xDtype];
        uint32_t used_ubSize = compileInfo.ubSize / SINGLE_UB_SIZE_FLOAT16;
        compileInfo.dataNumSingleUb = used_ubSize / compileInfo.inputDataByte;
        compileInfo.block_num = BLOCK_SIZE / compileInfo.inputDataByte;
    }
    CalTilingData(context, compileInfo, tilingParam, tilingData);
    SetTilingData(compileInfo, tilingParam, tilingData);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    InplaceAttnSoftmaxCompileInfo compileInfo;
    InplaceAttnSoftmaxTilingParam tilingParam;
    InplaceAttnSoftmaxTilingData tilingData;

    if (GetCompileInfo(context, compileInfo) != ge::GRAPH_SUCCESS) {
        printf("GetCompileInfo failed.");
        return ge::GRAPH_FAILED;
    }

    // 读取属性和dtype来计算tilingKey
    GetTillingData(context, compileInfo, tilingParam, tilingData);
    SetTilingDataForInplaceAttnSoftmax(context, tilingData);

    //调用softmax接口
    auto xDtype = context->GetInputDesc(0)->GetDataType();
    ge::Shape srcShape;
    if(tilingData.get_basicRowLenHeadCore() > tilingData.get_basicRowLenTailCore()){
        std::vector<int64_t> shapeVec = {tilingData.get_basicRowLenHeadCore(),tilingData.get_basicColLen()};
        srcShape = ge::Shape(shapeVec);
    }else {
        std::vector<int64_t> shapeVec = {tilingData.get_basicRowLenTailCore(),tilingData.get_basicColLen()};
        srcShape = ge::Shape(shapeVec);
    }
    const uint32_t localWorkSpaceSize = AscendC::GetSoftMaxMinTmpSize(srcShape, SoftMaxTilingSize, false);
    AscendC::SoftMaxTilingFunc(srcShape, SoftMaxTilingSize, localWorkSpaceSize, tilingData.softmaxTilingData);
    context->SetBlockDim(tilingData.get_realCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(
        1);  // 通过框架获取workspace的指针，GetworkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    currentWorkspace[0] =
        tilingData.get_realCoreNum() * tilingData.get_colLen() + sysWorkspaceSize;  // 设置总的workspace的数值大小，总的workspace空间由框架来申请并管理。
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class InplaceAttnSoftmax : public OpDef {
public:
    explicit InplaceAttnSoftmax(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(InplaceAttnSoftmax);
}
