project_json_src='''
[
    {
        "op": "ThreeDimTensorMatrixMultiplicationCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "a",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            },
            {
                "name": "b",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            }
        ],
        "output_desc": [
            {
                "name": "c",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            }
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ThreeDimTensorMatrixMultiplicationCustomTilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ThreeDimTensorMatrixMultiplicationCustom, ThreeDimTensorMatrixMultiplicationCustomTilingData)
} // namespace optiling
"""

host_operator_src="""
#include "three_dim_tensor_matrix_multiplication_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace matmul_tiling;

namespace {
inline int64_t FlattenLeadingDims(const gert::Shape &shape)
{
    return shape.GetDim(0) * shape.GetDim(1);
}
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto shapeA = context->GetInputTensor(0)->GetOriginShape();
    auto shapeB = context->GetInputTensor(1)->GetOriginShape();
    if (shapeA.GetDimNum() != 3 || shapeB.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const int64_t batch = shapeA.GetDim(0);
    const int64_t mDimA = shapeA.GetDim(1);
    const int64_t kDim = shapeA.GetDim(2);
    const int64_t bKDim = shapeB.GetDim(0);
    const int64_t nDim = shapeB.GetDim(1);
    if (batch <= 0 || mDimA <= 0 || kDim <= 0 || bKDim <= 0 || nDim <= 0 || kDim != bKDim) {
        return ge::GRAPH_FAILED;
    }

    const int32_t mDim = static_cast<int32_t>(FlattenLeadingDims(shapeA));
    const int32_t nDim32 = static_cast<int32_t>(nDim);
    const int32_t kDim32 = static_cast<int32_t>(kDim);

    MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
    cubeTiling.SetDim(1);
    cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetShape(mDim, nDim32, kDim32);
    cubeTiling.SetOrgShape(mDim, nDim32, kDim32);
    cubeTiling.SetFixSplit(16, 16, -1);
    cubeTiling.SetBias(false);
    cubeTiling.SetBufferSpace(-1, -1, -1);

    ThreeDimTensorMatrixMultiplicationCustomTilingData tiling;
    if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(tiling.cubeTilingData.get_usedCoreNum());
    context->SetTilingKey(1);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t userWorkspaceSize = 0;
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    auto aShape = context->GetInputTensor(0)->GetOriginShape();
    auto bShape = context->GetInputTensor(1)->GetOriginShape();
    if (aShape.GetDimNum() != 3 || bShape.GetDimNum() != 2) {
        return GRAPH_FAILED;
    }
    if (aShape.GetDim(2) != bShape.GetDim(0)) {
        return GRAPH_FAILED;
    }

    gert::Shape *outShape = context->GetOutputShape(0);
    outShape->SetDimNum(3);
    outShape->SetDim(0, aShape.GetDim(0));
    outShape->SetDim(1, aShape.GetDim(1));
    outShape->SetDim(2, bShape.GetDim(1));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class ThreeDimTensorMatrixMultiplicationCustom : public OpDef {
public:
    explicit ThreeDimTensorMatrixMultiplicationCustom(const char *name) : OpDef(name)
    {
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(ThreeDimTensorMatrixMultiplicationCustom);
} // namespace ops
"""

kernel_src="""
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

template <typename aType, typename bType, typename cType> class ThreeDimTensorMatrixMultiplicationKernel {
public:
    __aicore__ inline ThreeDimTensorMatrixMultiplicationKernel() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const TCubeTiling &tiling);
    template <bool setTmpSpace = false> __aicore__ inline void Process(AscendC::TPipe *pipe);

    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, cType>>
        matmulObj;

    AscendC::GlobalTensor<aType> aGlobal;
    AscendC::GlobalTensor<bType> bGlobal;
    AscendC::GlobalTensor<cType> cGlobal;
    TCubeTiling tiling;
};

template <typename aType, typename bType, typename cType>
__aicore__ inline void ThreeDimTensorMatrixMultiplicationKernel<aType, bType, cType>::Init(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const TCubeTiling &tiling)
{
    this->tiling = tiling;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), tiling.M * tiling.N);
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
}

template <typename aType, typename bType, typename cType>
template <bool setTmpSpace>
__aicore__ inline void ThreeDimTensorMatrixMultiplicationKernel<aType, bType, cType>::Process(AscendC::TPipe *pipe)
{
    if constexpr (setTmpSpace) {
        AscendC::TBuf<> tmpMMFormatUb;
        AscendC::LocalTensor<uint8_t> mmformatUb;
        pipe->InitBuffer(tmpMMFormatUb, AscendC::TOTAL_VEC_LOCAL_SIZE);
        mmformatUb = tmpMMFormatUb.Get<uint8_t>(AscendC::TOTAL_VEC_LOCAL_SIZE);
        matmulObj.SetLocalWorkspace(mmformatUb);
    }

    matmulObj.SetTensorA(aGlobal);
    matmulObj.SetTensorB(bGlobal);
    matmulObj.IterateAll(cGlobal);
    matmulObj.End();
}

extern "C" __global__ __aicore__ void three_dim_tensor_matrix_multiplication_custom(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    ThreeDimTensorMatrixMultiplicationKernel<float, float, float> op;
    AscendC::TPipe pipe;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.matmulObj, &tilingData.cubeTilingData);
    op.Init(a, b, c, workspace, tilingData.cubeTilingData);
    if (TILING_KEY_IS(1)) {
        op.Process(&pipe);
    } else if (TILING_KEY_IS(2)) {
        op.Process<true>(&pipe);
    }
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor three_dim_tensor_matrix_multiplication_impl_npu(const at::Tensor &a, const at::Tensor &b)
{
    auto outputShape = std::vector<int64_t>{a.size(0), a.size(1), b.size(1)};
    at::Tensor result = at::empty(outputShape, a.options());
    EXEC_NPU_CMD(aclnnThreeDimTensorMatrixMultiplicationCustom, a, b, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m)
{
    m.impl("three_dim_tensor_matrix_multiplication_custom", &three_dim_tensor_matrix_multiplication_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("three_dim_tensor_matrix_multiplication_custom",
          &three_dim_tensor_matrix_multiplication_impl_npu,
          "three dimensional tensor matrix multiplication");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return custom_ops_lib.three_dim_tensor_matrix_multiplication_custom(a, b)
'''
