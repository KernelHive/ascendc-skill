project_json_src='''
[
    {
        "op": "GemmDivideSumScalingCustom",
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
                "name": "scaling",
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
BEGIN_TILING_DATA_DEF(GemmDivideSumScalingCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, inputSize);
TILING_DATA_FIELD_DEF(uint32_t, hiddenSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GemmDivideSumScalingCustom, GemmDivideSumScalingCustomTilingData)
} // namespace optiling
"""

host_operator_src="""
#include "gemm_divide_sum_scaling_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr uint32_t BLOCK_DIM = 1;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *xShape = context->GetInputShape(0);
    const gert::StorageShape *weightShape = context->GetInputShape(1);
    const gert::StorageShape *scalingShape = context->GetInputShape(2);
    if (xShape == nullptr || weightShape == nullptr || scalingShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto &x = xShape->GetStorageShape();
    const auto &weight = weightShape->GetStorageShape();
    const auto &scaling = scalingShape->GetStorageShape();
    if (x.GetDimNum() != 2 || weight.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    if (x.GetDim(1) != weight.GetDim(1)) {
        return ge::GRAPH_FAILED;
    }
    if (scaling.GetShapeSize() != 1) {
        return ge::GRAPH_FAILED;
    }

    GemmDivideSumScalingCustomTilingData tiling;
    tiling.set_batchSize(static_cast<uint32_t>(x.GetDim(0)));
    tiling.set_inputSize(static_cast<uint32_t>(x.GetDim(1)));
    tiling.set_hiddenSize(static_cast<uint32_t>(weight.GetDim(0)));

    context->SetBlockDim(BLOCK_DIM);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    if (xShape == nullptr || xShape->GetDimNum() != 2) {
        return GRAPH_FAILED;
    }
    gert::Shape *yShape = context->GetOutputShape(0);
    yShape->SetDimNum(2);
    yShape->SetDim(0, xShape->GetDim(0));
    yShape->SetDim(1, 1);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class GemmDivideSumScalingCustom : public OpDef {
public:
    explicit GemmDivideSumScalingCustom(const char *name) : OpDef(name)
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
        this->Input("scaling")
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

OP_ADD(GemmDivideSumScalingCustom);
} // namespace ops
"""

kernel_src="""
#include "kernel_operator.h"

class KernelGemmDivideSumScaling {
public:
    __aicore__ inline KernelGemmDivideSumScaling() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR scaling,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t inputSize,
        uint32_t hiddenSize)
    {
        this->batchSize = batchSize;
        this->inputSize = inputSize;
        this->hiddenSize = hiddenSize;
        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * inputSize);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, hiddenSize * inputSize);
        scalingGm.SetGlobalBuffer((__gm__ float *)scaling, 1);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize);
    }

    __aicore__ inline void Process()
    {
        const float scalingValue = scalingGm.GetValue(0);
        for (uint32_t batchIdx = 0; batchIdx < this->batchSize; ++batchIdx) {
            const uint32_t xRowBase = batchIdx * this->inputSize;
            float reduced = 0.0f;
            for (uint32_t hiddenIdx = 0; hiddenIdx < this->hiddenSize; ++hiddenIdx) {
                const uint32_t weightRowBase = hiddenIdx * this->inputSize;
                float gemmValue = 0.0f;
                for (uint32_t inputIdx = 0; inputIdx < this->inputSize; ++inputIdx) {
                    gemmValue += xGm.GetValue(xRowBase + inputIdx) *
                        weightGm.GetValue(weightRowBase + inputIdx);
                }
                reduced += gemmValue * 0.5f;
            }
            yGm.SetValue(batchIdx, reduced * scalingValue);
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> weightGm;
    AscendC::GlobalTensor<float> scalingGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t inputSize;
    uint32_t hiddenSize;
};

extern "C" __global__ __aicore__ void gemm_divide_sum_scaling_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR scaling,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelGemmDivideSumScaling op;
    op.Init(
        x,
        weight,
        scaling,
        y,
        tiling_data.batchSize,
        tiling_data.inputSize,
        tiling_data.hiddenSize);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/extension.h>
#include <vector>
#include "pytorch_npu_helper.hpp"

at::Tensor gemm_divide_sum_scaling_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &scaling)
{
    auto outputShape = std::vector<int64_t>{x.size(0), 1};
    at::Tensor result = at::empty(outputShape, x.options());
    EXEC_NPU_CMD(aclnnGemmDivideSumScalingCustom, x, weight, scaling, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("gemm_divide_sum_scaling_custom", &gemm_divide_sum_scaling_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "gemm_divide_sum_scaling_custom",
        &gemm_divide_sum_scaling_custom_impl_npu,
        "fused gemm + divide + reduce_sum + scaling");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(hidden_size, input_size))
        self.register_buffer(
            "scaling_tensor",
            torch.tensor([scaling_factor], dtype=torch.float32),
        )

    def forward(self, x):
        return custom_ops_lib.gemm_divide_sum_scaling_custom(
            x,
            self.weight,
            self.scaling_tensor,
        )
'''
