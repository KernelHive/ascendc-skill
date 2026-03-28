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
 * @file complex_mat_mul.cpp
 */
#include "complex_mat_mul_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace matmul_tiling;

inline int AlignUP(int x, int p)
{
    return (x + (p - 1)) & ~(p - 1);
}

namespace optiling
{
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto shape_a = context->GetInputTensor(0)->GetOriginShape();
    auto shape_b = context->GetInputTensor(1)->GetOriginShape();
    if (shape_a.GetDimNum() < 2 || shape_a.GetDimNum() != shape_b.GetDimNum())
    {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    if (socVersion != platform_ascendc::SocVersion::ASCEND910B) {
        return ge::GRAPH_FAILED;
    }
    int dim = shape_a.GetDimNum();
    int32_t BatchSize = 1;
    for (int i = 0; i < dim - 2; i++)
    {
        if (shape_a.GetDim(i) != shape_b.GetDim(i))
        {
            return ge::GRAPH_FAILED;
        }

        BatchSize *= shape_a.GetDim(i);
    }

    int32_t M = shape_a.GetDim(dim - 2);
    int32_t K = shape_a.GetDim(dim - 1);
    int32_t N = shape_b.GetDim(dim - 1);

    MultiCoreMatmulTiling tiling(ascendcPlatform);
    MatMulTilingData tiling1;

    tiling.SetAType(TPosition::GM, CubeFormat::ND,
                    matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBType(TPosition::GM, CubeFormat::ND,
                    matmul_tiling::DataType::DT_FLOAT);
    tiling.SetCType(TPosition::GM, CubeFormat::ND,
                    matmul_tiling::DataType::DT_FLOAT);

    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;

    int32_t M0 = AlignUP(M, TILE_M);
    int32_t K0 = AlignUP(K, 64);
    int32_t N0 = AlignUP(N, TILE_N);

    tiling.SetShape(M0, N0, K0);
    tiling.SetOrgShape(M0, N0, K0);
    tiling.SetAlignSplit(TILE_M, TILE_N, -1);

    tiling.SetBias(false);

    tiling.SetBufferSpace(-1, -1, -1);
    uint32_t n_core = ascendcPlatform.GetCoreNumAiv();
    tiling.SetDim(n_core);

    while (n_core > 0 && tiling.GetTiling(tiling1.cubeTilingData) == -1)
    {
        n_core--;
        tiling.SetDim(n_core);
    }

    if (n_core == 0)
    {
        return ge::GRAPH_FAILED;
    }

    tiling1.set_M(M);
    tiling1.set_K(K);
    tiling1.set_N(N);
    tiling1.set_BatchSize(BatchSize);

    context->SetBlockDim((n_core + 1) / 2);
    tiling1.SaveToBuffer(context->GetRawTilingData()->GetData(),
                         context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling1.GetDataSize());

    size_t userWorkspaceSize = (3 * M0 * K0 + 3 * K0 * N0 + 3 * M0 * N0) * sizeof(float);
    size_t systemWorkspaceSize =
        static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge
{
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x1_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops
{
class ComplexMatMul : public OpDef
{
  public:
    explicit ComplexMatMul(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_COMPLEX64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_COMPLEX64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_COMPLEX64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_COMPLEX64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ComplexMatMul);
} // namespace ops