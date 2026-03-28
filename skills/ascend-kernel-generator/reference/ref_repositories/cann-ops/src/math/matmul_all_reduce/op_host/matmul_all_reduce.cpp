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
 * @file matmul_all_reduce.cpp
 */

#include "../op_kernel/matmul_all_reduce_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "tiling/hccl/hccl_tiling.h"

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

// tiling
constexpr uint32_t RANK_NUM = 1;
constexpr uint32_t CUSTOM_TILING_KEY = 1000UL; // full mesh + no nd2nz + no cast bias
constexpr uint32_t TILE_M = 4096;
constexpr int32_t L1_BUFFER_SIZE = 512 * 1024;
constexpr uint64_t REQUIRED_DIM_NUM = 2;

namespace {
const std::set<ge::Format> SUPPORTED_FORMAT = {ge::FORMAT_NCL,  ge::FORMAT_NCDHW, ge::FORMAT_DHWCN,
                                         ge::FORMAT_NHWC, ge::FORMAT_NCHW,  ge::FORMAT_ND};
}


static ge::graphStatus ParamsCheck(gert::TilingContext* context)
{
    const gert::StorageShape* aShape = context->GetInputShape(0);
    const gert::StorageShape* bShape = context->GetInputShape(1);
    uint64_t aShapeDimNum = aShape->GetStorageShape().GetDimNum();
    uint64_t bShapeDimNum = bShape->GetStorageShape().GetDimNum();
    if (aShapeDimNum != REQUIRED_DIM_NUM || bShapeDimNum != REQUIRED_DIM_NUM) {
        ERROR_LOG("Input dim num must be 2.");
        return ge::GRAPH_FAILED;
    }
    auto aTensor = context->GetInputDesc(0);
    auto bTensor = context->GetInputDesc(1);
    auto output = context->GetOutputDesc(0);
    auto aShapeFormat = aTensor->GetStorageFormat();
    auto bShapeFormat = bTensor->GetStorageFormat();
    auto outputFormat = output->GetStorageFormat();
    if (aShapeFormat != outputFormat) {
        ERROR_LOG("a shape Format, output Format are not same, should be ND/ND");
        return ge::GRAPH_FAILED;
    }
    
    if (SUPPORTED_FORMAT.count(aShapeFormat) == 0 || SUPPORTED_FORMAT.count(bShapeFormat) == 0) {
        ERROR_LOG("a shape Format, b shape Format only support ND");
        return ge::GRAPH_FAILED;
    }

    auto isTransA = context->GetAttrs()->GetAttrPointer<bool>(2);
    if (*isTransA != false) {
        ERROR_LOG("Is trans A only support false");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MatmulAllReduceTilingFunc(gert::TilingContext *context) {
    // 对参数进行校验
    if (ParamsCheck(context) != ge::GRAPH_SUCCESS) {
        ERROR_LOG("Param check failed.");
        return ge::GRAPH_FAILED;
    }
    
    uint32_t index = 0U;
    auto group = context->GetAttrs()->GetAttrPointer<char>(index++);
    auto reduce_op = context->GetAttrs()->GetAttrPointer<char>(index++);
    auto isTransA = context->GetAttrs()->GetAttrPointer<bool>(index++);
    auto isTransB = context->GetAttrs()->GetAttrPointer<bool>(index++);
    auto comm_turn = context->GetAttrs()->GetAttrPointer<int>(index++);

    INFO_LOG("group is %s, reduce_op is %s, isTransA is %d, isTransB is %d, comm_turn is %d",
        group, reduce_op, *isTransA, *isTransB, comm_turn);

    uint64_t M = context->GetInputShape(0)->GetStorageShape().GetDim(0);
    uint64_t K = context->GetInputShape(0)->GetStorageShape().GetDim(1);
    uint64_t N = *isTransB ?
        context->GetInputShape(1)->GetStorageShape().GetDim(0) : context->GetInputShape(1)->GetStorageShape().GetDim(1);

    auto aTensorDesc = context->GetInputDesc(0);
    auto aType = aTensorDesc->GetDataType();

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aicCoreNum = ascendcPlatform.GetCoreNumAic();
    // set block dim & tiling key
    context->SetBlockDim(aicCoreNum);
    context->SetTilingKey(CUSTOM_TILING_KEY);

    // set work space size
    size_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize() + M * N * 2; // nd2nzLen + gmcFloat + gatherLen + biasLen + 16M
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize;

    uint64_t tileNum = M / TILE_M;
    uint64_t tailNum = (M % TILE_M == 0) ? 0 : 1;
    uint64_t tailM = M % TILE_M;
    INFO_LOG("tileNum %lu, tailNum %lu, tailM %lu", tileNum, tailNum, tailM);

    MatmulAllReduceTilingData *tiling = context->GetTilingData<MatmulAllReduceTilingData>();

    tiling->param.rankDim = RANK_NUM;
    tiling->param.rankM = M;
    tiling->param.rankN = N;
    tiling->param.rankK = K;
    tiling->param.isTransposeA = *isTransA ? 1 : 0;
    tiling->param.isTransposeB = *isTransB ? 1 : 0;
    tiling->param.tileCnt = tileNum;
    tiling->param.tailM = tailM;
    tiling->param.tailCnt = tailNum;
    tiling->param.determinism = 0;
    tiling->param.useBufferType = 1;
    // dataType 3 corresponds to DT_FLOAT16
    tiling->param.dataType = 3;
    // matmul tiling func
    auto matmulTilingFunc = [isTransA, isTransB, aicCoreNum, L1_BUFFER_SIZE] (int64_t m, int64_t n, int64_t k, TCubeTiling &cubeTiling) -> bool {
        matmul_tiling::MultiCoreMatmulTiling mmTiling;
        mmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, *isTransA);
        mmTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, *isTransB);
        mmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
        mmTiling.SetBias(false);
        mmTiling.SetDim(aicCoreNum);
        mmTiling.SetShape(m, n, k);
        mmTiling.SetOrgShape(m, n, k);
        mmTiling.SetBufferSpace(L1_BUFFER_SIZE, -1, -1);
        int32_t fixCoreM = -1;
        int32_t fixCoreK = -1;
        int32_t fixCoreN = -1;
        mmTiling.SetSingleShape(fixCoreM, fixCoreN, fixCoreK);
        if (mmTiling.GetTiling(cubeTiling) != 0) {
            return false;
        }
        return true;
    };
    // matmul tile tiling
    if (tileNum > 0) {
        if (!matmulTilingFunc(TILE_M, N, K, tiling->matmulTiling)) {
            ERROR_LOG("Get tile matmul tiling failed");
            return ge::GRAPH_FAILED;
        }
    }
    // matmul tail tiling
    if (tailNum > 0) {
        if (!matmulTilingFunc(tailM, N, K, tiling->tailTiling)) {
            ERROR_LOG("Get tail matmul tiling failed");
            return ge::GRAPH_FAILED;
        }
    }

    uint32_t opType = 2;
    std::string algConfig = "AllReduce=level0:fullmesh";
    uint32_t reduceType = 0;
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, algConfig, reduceType);
    mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling);
    
    return ge::GRAPH_SUCCESS;
}

namespace ops {
class MatmulAllReduce : public OpDef {
public:
    explicit MatmulAllReduce(const char *name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("group").String();
        this->Attr("reduce_op").AttrType(OPTIONAL).String("sum");
        this->Attr("is_trans_a").AttrType(OPTIONAL).Bool(false);
        this->Attr("is_trans_b").AttrType(OPTIONAL).Bool(false);
        this->Attr("comm_turn").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");
        this->AICore().SetTiling(MatmulAllReduceTilingFunc);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->MC2().HcclGroup("group");
    }
};

OP_ADD(MatmulAllReduce);
}
