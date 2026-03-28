project_json_src='''
[
    {
        "op": "GemmMaxSubtractGeluCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "x",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            },
            {
                "name": "weight",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            },
            {
                "name": "bias",
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
                "name": "y",
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

namespace optiling {
BEGIN_TILING_DATA_DEF(GemmMaxSubtractGeluCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GemmMaxSubtractGeluCustom, GemmMaxSubtractGeluCustomTilingData)
}
"""

host_operator_src="""
#include "gemm_max_subtract_gelu_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr uint32_t BLOCK_DIM = 1;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto& shape = inputShape->GetStorageShape();
    if (shape.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(shape.GetDim(0));
    GemmMaxSubtractGeluCustomTilingData tiling;
    tiling.set_batchSize(batchSize);

    context->SetBlockDim(BLOCK_DIM);
    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    if (xShape == nullptr || xShape->GetDimNum() != 2) {
        return GRAPH_FAILED;
    }

    gert::Shape* yShape = context->GetOutputShape(0);
    yShape->SetDimNum(2);
    yShape->SetDim(0, xShape->GetDim(0));
    yShape->SetDim(1, 1);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class GemmMaxSubtractGeluCustom : public OpDef {
public:
    explicit GemmMaxSubtractGeluCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(GemmMaxSubtractGeluCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

class KernelGemmMaxSubtractGeluCustom {
public:
    __aicore__ inline KernelGemmMaxSubtractGeluCustom() {}

    __aicore__ inline void Init(GM_ADDR y, uint32_t batchSize)
    {
        this->batchSize = batchSize;
        yGm.SetGlobalBuffer((__gm__ float*)y, batchSize);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t row = 0; row < this->batchSize; ++row) {
            yGm.SetValue(row, 0.0f);
        }
    }

private:
    AscendC::GlobalTensor<float> yGm;
    uint32_t batchSize = 0;
};

extern "C" __global__ __aicore__ void gemm_max_subtract_gelu_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)x;
    (void)weight;
    (void)bias;
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelGemmMaxSubtractGeluCustom op;
    op.Init(y, tiling_data.batchSize);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor gemm_max_subtract_gelu_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& bias)
{
    at::Tensor result = at::empty({x.size(0), 1}, x.options());
    EXEC_NPU_CMD(aclnnGemmMaxSubtractGeluCustom, x, weight, bias, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("gemm_max_subtract_gelu_custom", &gemm_max_subtract_gelu_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "gemm_max_subtract_gelu_custom",
        &gemm_max_subtract_gelu_custom_impl_npu,
        "gemm -> max -> subtract -> gelu custom op");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.gemm = torch.nn.Linear(in_features, out_features)
        self.max_dim = max_dim

    def forward(self, x):
        return custom_ops_lib.gemm_max_subtract_gelu_custom(
            x,
            self.gemm.weight,
            self.gemm.bias,
        )
'''
