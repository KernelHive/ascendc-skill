project_json_src='''
[
    {
        "op": "StandardMatrixMultiplicationCustom",
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
BEGIN_TILING_DATA_DEF(StandardMatrixMultiplicationCustomTilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(StandardMatrixMultiplicationCustom, StandardMatrixMultiplicationCustomTilingData)
} // namespace optiling
"""

host_operator_src="""
#include "standard_matrix_multiplication_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace matmul_tiling;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto shapeA = context->GetInputTensor(0)->GetOriginShape();
    auto shapeB = context->GetInputTensor(1)->GetOriginShape();
    if (shapeA.GetDimNum() != 2 || shapeB.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const int32_t m = shapeA.GetDim(0);
    const int32_t k = shapeA.GetDim(1);
    const int32_t bk = shapeB.GetDim(0);
    const int32_t n = shapeB.GetDim(1);
    if (m <= 0 || k <= 0 || n <= 0 || bk != k) {
        return ge::GRAPH_FAILED;
    }

    MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
    cubeTiling.SetDim(2);
    cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetShape(m, n, k);
    cubeTiling.SetOrgShape(m, n, k);
    cubeTiling.SetFixSplit(128, 128, -1);
    cubeTiling.SetBias(false);
    cubeTiling.SetBufferSpace(-1, -1, -1);

    StandardMatrixMultiplicationCustomTilingData tiling;
    if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P) {
        context->SetBlockDim(2);
        context->SetTilingKey(2);
    } else {
        context->SetBlockDim(1);
        context->SetTilingKey(1);
    }

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = systemWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *aShape = context->GetInputShape(0);
    const gert::Shape *bShape = context->GetInputShape(1);
    if (aShape == nullptr || bShape == nullptr || aShape->GetDimNum() != 2 || bShape->GetDimNum() != 2) {
        return GRAPH_FAILED;
    }

    gert::Shape *cShape = context->GetOutputShape(0);
    cShape->SetDimNum(2);
    cShape->SetDim(0, aShape->GetDim(0));
    cShape->SetDim(1, bShape->GetDim(1));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class StandardMatrixMultiplicationCustom : public OpDef {
public:
    explicit StandardMatrixMultiplicationCustom(const char *name) : OpDef(name)
    {
        this->Input("a").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("b").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("c").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(StandardMatrixMultiplicationCustom);
} // namespace ops
"""

kernel_src="""
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

template <typename aType, typename bType, typename cType> class StandardMatrixMultiplicationKernel {
public:
    __aicore__ inline StandardMatrixMultiplicationKernel(){};
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const TCubeTiling &tiling);
    template <bool setTmpSpace = false> __aicore__ inline void Process(AscendC::TPipe *pipe);
    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetA, int32_t &offsetB,
                                      int32_t &offsetC);

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
__aicore__ inline void StandardMatrixMultiplicationKernel<aType, bType, cType>::Init(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const TCubeTiling &tiling)
{
    this->tiling = tiling;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), tiling.M * tiling.N);

    int32_t offsetA = 0;
    int32_t offsetB = 0;
    int32_t offsetC = 0;
    CalcOffset(GetBlockIdx(), tiling, offsetA, offsetB, offsetC);
    aGlobal = aGlobal[offsetA];
    bGlobal = bGlobal[offsetB];
    cGlobal = cGlobal[offsetC];
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
}

template <typename aType, typename bType, typename cType>
template <bool setTmpSpace>
__aicore__ inline void StandardMatrixMultiplicationKernel<aType, bType, cType>::Process(AscendC::TPipe *pipe)
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

template <typename aType, typename bType, typename cType>
__aicore__ inline void StandardMatrixMultiplicationKernel<aType, bType, cType>::CalcOffset(
    int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetA, int32_t &offsetB, int32_t &offsetC)
{
    auto mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
    auto mCoreIdx = blockIdx % mSingleBlocks;
    auto nCoreIdx = blockIdx / mSingleBlocks;

    offsetA = mCoreIdx * tiling.Ka * tiling.singleCoreM;
    offsetB = nCoreIdx * tiling.singleCoreN;
    offsetC = mCoreIdx * tiling.N * tiling.singleCoreM + nCoreIdx * tiling.singleCoreN;
}

extern "C" __global__ __aicore__ void standard_matrix_multiplication_custom(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    StandardMatrixMultiplicationKernel<float, float, float> matmulKernel;
    AscendC::TPipe pipe;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulKernel.matmulObj, &tilingData.cubeTilingData);
    matmulKernel.Init(a, b, c, workspace, tilingData.cubeTilingData);
    if (TILING_KEY_IS(1)) {
        matmulKernel.Process(&pipe);
    } else if (TILING_KEY_IS(2)) {
        matmulKernel.Process<true>(&pipe);
    }
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor standard_matrix_multiplication_impl_npu(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor result = at::empty({a.size(0), b.size(1)}, a.options());
    EXEC_NPU_CMD(aclnnStandardMatrixMultiplicationCustom, a, b, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("standard_matrix_multiplication_custom", &standard_matrix_multiplication_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("standard_matrix_multiplication_custom",
          &standard_matrix_multiplication_impl_npu,
          "standard matrix multiplication custom");
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
        return custom_ops_lib.standard_matrix_multiplication_custom(a, b)
'''
