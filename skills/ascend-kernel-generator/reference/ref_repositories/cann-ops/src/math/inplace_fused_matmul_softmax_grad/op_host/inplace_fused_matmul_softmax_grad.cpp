/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "inplace_fused_matmul_softmax_grad_tiling.h"
#include "inplace_fused_matmul_softmax_grad_tiling_utils.h"

using namespace matmul_tiling;

namespace optiling {

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t L2_CACHE_LINE_SIZE = 512;  // pack unit in cache 512B

constexpr uint32_t SINGLE_UB_SIZE_BF16 = 20;
constexpr uint32_t SINGLE_UB_SIZE_FLOAT16 = 20;
constexpr uint32_t SINGLE_UB_SIZE_FLOAT32 = 20;
constexpr uint32_t ALIGN = 1;
constexpr uint32_t NO_ALIGN = 2;
constexpr uint64_t RSV_UB_SIZE = 10240;
constexpr uint32_t CORES_PER_BLOCK = 2;

std::map<const ge::DataType, const uint32_t> typeLen = {
    {ge::DT_FLOAT16, 2}, {ge::DT_BF16, 2}, {ge::DT_FLOAT, 4}};

ge::graphStatus TilingKeyChose(gert::TilingContext *context, InplaceFusedMatmulSoftmaxGradCompileInfo &compileInfo,
    InplaceFusedMatmulSoftmaxGradTilingParam &tilingParam, InplaceFusedMatmulSoftmaxGradTilingData &tilingData)
{
    uint32_t tilingKey = tilingData.baseTilingData.get_tilingKey();
    if (compileInfo.inputDataType == ge::DT_FLOAT16) {
        tilingKey += FLOAT16_BASE_TILING_KEY;
    } else if (compileInfo.inputDataType == ge::DT_BF16) {
        tilingKey += BFLOAT16_BASE_TILING_KEY;
    } else if (compileInfo.inputDataType == ge::DT_FLOAT) {
        tilingKey += FLOAT_BASE_TILING_KEY;
    }
    // 判断是否对齐
    tilingKey = tilingParam.alignColLen == tilingParam.colLen ? tilingKey + ALIGN : tilingKey + NO_ALIGN;
    tilingData.baseTilingData.set_tilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetBaseTilingData(gert::TilingContext *context, InplaceFusedMatmulSoftmaxGradCompileInfo &compileInfo,
    InplaceFusedMatmulSoftmaxGradTilingParam &tilingParam, InplaceFusedMatmulSoftmaxGradTilingData &tilingData)
{
    tilingData.baseTilingData.set_b(tilingParam.b);
    tilingData.baseTilingData.set_m(tilingParam.m);
    tilingData.baseTilingData.set_n(tilingParam.n);
    tilingData.baseTilingData.set_k(tilingParam.k);
    tilingData.baseTilingData.set_rowLen(tilingParam.b * tilingParam.m);
    tilingData.baseTilingData.set_colLen(tilingParam.n);
    tilingData.baseTilingData.set_alignColLen(tilingParam.alignColLen);
    tilingData.baseTilingData.set_rowLenPerHeadCore(tilingParam.rowLenPerHeadCore);
    tilingData.baseTilingData.set_rowLenPerTailCore(tilingParam.rowLenPerTailCore);

    tilingData.baseTilingData.set_basicRowLenHeadCore(tilingParam.optBaseRowLenHeadCore);
    tilingData.baseTilingData.set_basicRowLenTailCore(tilingParam.optBaseRowLenTailCore);
    
    tilingData.baseTilingData.set_realCoreNum(tilingParam.coreNumUsed);
    tilingData.baseTilingData.set_headCoreNum(tilingParam.headCoreNum);
    tilingData.baseTilingData.set_tailCoreNum(tilingParam.tailCoreNum);
    tilingData.baseTilingData.set_blockNum(tilingParam.blockNum);

    tilingData.baseTilingData.set_innerLoopTimes(tilingParam.innerLoopTimes);
    tilingData.baseTilingData.set_innerLoopHeadColLen(tilingParam.innerLoopHeadColLen);
    tilingData.baseTilingData.set_innerLoopTailColLen(tilingParam.innerLoopTailColLen);
    tilingData.baseTilingData.set_headLocalWorkSpaceSize(tilingParam.headLocalWorkSpaceSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetSoftMaxTilingData(gert::TilingContext *context, InplaceFusedMatmulSoftmaxGradCompileInfo &compileInfo,
    InplaceFusedMatmulSoftmaxGradTilingParam &tilingParam, InplaceFusedMatmulSoftmaxGradTilingData &tilingData)
{
    std::vector<int64_t> headShapeVec = {tilingParam.optBaseRowLenHeadCore, tilingParam.alignColLen};
    ge::Shape headSrcShape(headShapeVec);
    // 本样例中仅做为样例说明，通过GetSoftMaxMinTmpSize获取最小值并传入，来保证功能正确，开发者可以根据需要传入合适的空间大小
    const uint32_t headLocalWorkSpaceSize = AscendC::GetSoftMaxGradMinTmpSize(headSrcShape, sizeof(float), false, false);
    // 获取SoftMax Tiling参数
    AscendC::SoftMaxGradTilingFunc(
        headSrcShape, sizeof(float), headLocalWorkSpaceSize, tilingData.headSoftMaxGradTilingData);
    tilingParam.headLocalWorkSpaceSize = headLocalWorkSpaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetCubeTilingData(gert::TilingContext *context, InplaceFusedMatmulSoftmaxGradCompileInfo &compileInfo,
    InplaceFusedMatmulSoftmaxGradTilingParam &tilingParam, InplaceFusedMatmulSoftmaxGradTilingData &tilingData)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
    cubeTiling.SetDim(tilingParam.coreNumUsed);

    cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, static_cast<matmul_tiling::DataType>(compileInfo.inputDataType));
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, static_cast<matmul_tiling::DataType>(compileInfo.inputDataType));
    cubeTiling.SetCType(TPosition::GM, CubeFormat::ND, static_cast<matmul_tiling::DataType>(ge::DT_FLOAT));

    cubeTiling.SetOrgShape(tilingParam.m, tilingParam.n, tilingParam.k);
    cubeTiling.SetShape(tilingParam.m, tilingParam.n, tilingParam.k);  // single块
    cubeTiling.SetBias(false);
    cubeTiling.SetBufferSpace(compileInfo.l1Size, compileInfo.l0CSize, -1);
    if (cubeTiling.GetTiling(tilingData.cubeTilingData) == -1) {
        printf("GetTiling fail.\n");
        ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetTilingData(gert::TilingContext *context, InplaceFusedMatmulSoftmaxGradCompileInfo &compileInfo,
    InplaceFusedMatmulSoftmaxGradTilingParam &tilingParam, InplaceFusedMatmulSoftmaxGradTilingData &tilingData)
{
    SetSoftMaxTilingData(context, compileInfo, tilingParam, tilingData);
    SetBaseTilingData(context, compileInfo, tilingParam, tilingData);
    SetCubeTilingData(context, compileInfo, tilingParam, tilingData);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CalBaseTilingData(gert::TilingContext *context, InplaceFusedMatmulSoftmaxGradCompileInfo &compileInfo,
    InplaceFusedMatmulSoftmaxGradTilingParam &tilingParam, InplaceFusedMatmulSoftmaxGradTilingData &tilingData)
{
    // 1. 计算rowlen和collen
    uint32_t batchLen = tilingParam.b;
    uint32_t rowLenPerBatch = tilingParam.m;
    uint32_t rowLen = batchLen * rowLenPerBatch;
    tilingParam.colLen = tilingParam.n;

    // 8 15
    tilingParam.coreNumUsed = std::max(std::min(compileInfo.coreNum, rowLen), ONE);  // 8
    tilingParam.headCoreNum = rowLen % tilingParam.coreNumUsed;                        // 15 % 8 = 7
    tilingParam.headCoreNum = tilingParam.headCoreNum > 0 ? tilingParam.headCoreNum : tilingParam.coreNumUsed;
    tilingParam.tailCoreNum = tilingParam.coreNumUsed - tilingParam.headCoreNum;  // 8 - 7 = 1

    // rowLenPerHeadCore 指的是 一共有多少个row -> 每个核心处理的row数 上取整
    tilingParam.rowLenPerHeadCore = CeilDiv<uint32_t>(rowLen, tilingParam.coreNumUsed);  // 15 / 8 = 2
    if (tilingParam.tailCoreNum > 0) {
        tilingParam.rowLenPerTailCore =
            (rowLen - tilingParam.rowLenPerHeadCore * tilingParam.headCoreNum) / tilingParam.tailCoreNum;
    } else {
        tilingParam.rowLenPerTailCore = 0;
    }
    
    // Align ColLen
    tilingParam.alignColLen = AlignUp<uint32_t>(tilingParam.colLen, compileInfo.blockNum);
    tilingParam.blockNum = compileInfo.blockNum;
    if (tilingParam.alignColLen == 0) {
        printf("Unsupported alignColLen %d == 0 \n", tilingParam.alignColLen);
        return false;
    }

    // 小shape一次性算多行 大shape按列进行切分
    uint32_t ubAvailRowNum = compileInfo.dataNumSingleUb / tilingParam.alignColLen;
    if (ubAvailRowNum == 0) {
        // collen超过ub可用空间大小，需要循环处理colLen，大shape场景，当前不涉及，预留参数
        tilingParam.innerLoopHeadColLen = AlignDown<uint32_t>(compileInfo.dataNumSingleUb, compileInfo.blockNum);
        // LargeShape
        tilingParam.innerLoopTimes = tilingParam.colLen / tilingParam.innerLoopHeadColLen;
        tilingParam.innerLoopTailColLen = tilingParam.colLen % tilingParam.innerLoopHeadColLen;
        ubAvailRowNum = ONE;
        uint32_t new_tiling_key = tilingData.baseTilingData.get_tilingKey();
        new_tiling_key += BIG_SHAPE_BASE_TILING_KEY;
        tilingData.baseTilingData.set_tilingKey(new_tiling_key);
    } else {
        tilingParam.innerLoopHeadColLen = tilingParam.colLen;
        ubAvailRowNum = std::max(ubAvailRowNum, ONE);
    }
    
    tilingParam.optBaseRowLenHeadCore = std::min(std::min(ubAvailRowNum, tilingParam.rowLenPerHeadCore), COMPARE_INT);
    tilingParam.optBaseRowLenTailCore = std::min(std::min(ubAvailRowNum, tilingParam.rowLenPerTailCore), COMPARE_INT);
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus CalTilingData(gert::TilingContext *context, InplaceFusedMatmulSoftmaxGradCompileInfo &compileInfo,
    InplaceFusedMatmulSoftmaxGradTilingParam &tilingParam, InplaceFusedMatmulSoftmaxGradTilingData &tilingData)
{
    CalBaseTilingData(context, compileInfo, tilingParam, tilingData);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetBaseTilingData(gert::TilingContext *context, InplaceFusedMatmulSoftmaxGradCompileInfo &compileInfo,
    InplaceFusedMatmulSoftmaxGradTilingParam &tilingParam, InplaceFusedMatmulSoftmaxGradTilingData &tilingData)
{
    auto softmaxOutputDtype = context->GetInputDesc(SOFTMAX_OUTPUT_INDEX)->GetDataType();
    compileInfo.inputDataType = softmaxOutputDtype;
    compileInfo.inputDataByte = typeLen[softmaxOutputDtype];
    if (softmaxOutputDtype == ge::DT_FLOAT16) {
        compileInfo.dataNumSingleUb = (compileInfo.ubSize -RSV_UB_SIZE) / SINGLE_UB_SIZE_FLOAT16;
    }
    else if (softmaxOutputDtype == ge::DT_BF16) {
        compileInfo.dataNumSingleUb = (compileInfo.ubSize -RSV_UB_SIZE) / SINGLE_UB_SIZE_BF16;
    }
    else if (softmaxOutputDtype == ge::DT_FLOAT) {
        compileInfo.dataNumSingleUb = (compileInfo.ubSize -RSV_UB_SIZE) / SINGLE_UB_SIZE_FLOAT32;
    }
    // UB空间可处理的最大数据量，32-Byte对齐
    compileInfo.blockNum = BLOCK_SIZE / compileInfo.inputDataByte;
    compileInfo.cacheLineLen = L2_CACHE_LINE_SIZE / compileInfo.inputDataByte;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetCubeTilingData(gert::TilingContext *context, InplaceFusedMatmulSoftmaxGradCompileInfo &compileInfo,
    InplaceFusedMatmulSoftmaxGradTilingParam &tilingParam, InplaceFusedMatmulSoftmaxGradTilingData &tilingData)
{
    auto softmaxOutputShape = context->GetInputShape(SOFTMAX_OUTPUT_INDEX)->GetStorageShape();
    auto gradOutputShape = context->GetInputShape(GRAD_OUTPUT_INDEX)->GetStorageShape();
    int32_t dimNum = softmaxOutputShape.GetDimNum();
    if (dimNum < MIN_DIM_NUM) {
        return ge::GRAPH_FAILED;
    }

    int32_t batch = 1;
    int32_t no_bath_len = 2;
    for (int32_t dim = 0; dim < dimNum - no_bath_len; ++dim) {
        batch *= softmaxOutputShape.GetDim(dim);
    }
    tilingParam.b = static_cast<uint32_t>(batch);
    tilingParam.m = softmaxOutputShape.GetDim(static_cast<uint32_t>(dimNum - 2)); // 计算softmax输出形状的倒数第二个维度
    tilingParam.n = softmaxOutputShape.GetDim(static_cast<uint32_t>(dimNum - 1)); // 计算softmax输出形状的最后一个维度
    tilingParam.k = gradOutputShape.GetDim(static_cast<uint32_t>(gradOutputShape.GetDimNum() - 1)); // 计算梯度输出形状的最后一个维度

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetTilingData(gert::TilingContext *context, InplaceFusedMatmulSoftmaxGradCompileInfo &compileInfo,
    InplaceFusedMatmulSoftmaxGradTilingParam &tilingParam, InplaceFusedMatmulSoftmaxGradTilingData &tilingData)
{
    if (GetCubeTilingData(context, compileInfo, tilingParam, tilingData) != ge::GRAPH_SUCCESS) {
        printf("GetBaseTilingData failed.");
        return ge::GRAPH_FAILED;
    }
    if (GetBaseTilingData(context, compileInfo, tilingParam, tilingData) != ge::GRAPH_SUCCESS) {
        printf("GetBaseTilingData failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetCompileInfo(gert::TilingContext *context, InplaceFusedMatmulSoftmaxGradCompileInfo &compileInfo)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    compileInfo.coreNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo.aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo.aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfo.socVersion = ascendcPlatform.GetSocVersion();
    compileInfo.sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfo.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfo.l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfo.l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfo.l0CSize);
    if (compileInfo.coreNum <= 0 || compileInfo.ubSize <= 0 || compileInfo.l1Size <= 0) {
        printf("GetCompileInfo Failed, coreNum:%d, ubSize:%d, l1Size:%d. \n",
            compileInfo.coreNum,
            compileInfo.ubSize,
            compileInfo.l1Size);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    InplaceFusedMatmulSoftmaxGradCompileInfo compileInfo;
    InplaceFusedMatmulSoftmaxGradTilingParam tilingParam;
    InplaceFusedMatmulSoftmaxGradTilingData tilingData;

    if (CheckGradOpParams(context) != ge::GRAPH_SUCCESS) {
        printf("GetCompileInfo failed.");
        return ge::GRAPH_FAILED;
    }

    if (GetCompileInfo(context, compileInfo) != ge::GRAPH_SUCCESS) {
        printf("GetCompileInfo failed.");
        return ge::GRAPH_FAILED;
    }
    GetTilingData(context, compileInfo, tilingParam, tilingData);
    CalTilingData(context, compileInfo, tilingParam, tilingData);
    SetTilingData(context, compileInfo, tilingParam, tilingData);
    // 读取属性和dtype来计算tilingKey
    TilingKeyChose(context, compileInfo, tilingParam, tilingData);

    context->SetBlockDim(CeilDiv(tilingData.baseTilingData.get_realCoreNum(), CORES_PER_BLOCK));
    context->SetTilingKey(tilingData.baseTilingData.get_tilingKey());
    // large shape情况下需要提供workspace空间 大小为get_realCoreNum() * alingedColLen
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);  // 通过框架获取workspace的指针，GetworkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    currentWorkspace[0] = sizeof(float) * tilingData.baseTilingData.get_rowLen() * tilingData.baseTilingData.get_alignColLen() +
                          sysWorkspaceSize;  // 设置总的workspace的数值大小，总的workspace空间由框架来申请并管理。
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling


namespace ops {
class InplaceFusedMatmulSoftmaxGrad : public OpDef {
public:
    explicit InplaceFusedMatmulSoftmaxGrad(const char *name) : OpDef(name)
    {
        this->Input("softmaxOutput")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("gradOutput")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("values")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(InplaceFusedMatmulSoftmaxGrad);
}  // namespace ops
