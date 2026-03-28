project_json_src = '''
[
    {
        "op": "MatmulSigmoidSumCustom",
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

host_tiling_src = """
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulSigmoidSumCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, inputSize);
TILING_DATA_FIELD_DEF(uint32_t, hiddenSize);
TILING_DATA_FIELD_DEF(uint32_t, blockDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulSigmoidSumCustom, MatmulSigmoidSumCustomTilingData)
} // namespace optiling
"""

host_operator_src = """
#include "matmul_sigmoid_sum_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *xStorage = context->GetInputShape(0);
    const gert::StorageShape *weightStorage = context->GetInputShape(1);
    const gert::StorageShape *biasStorage = context->GetInputShape(2);
    if (xStorage == nullptr || weightStorage == nullptr || biasStorage == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &xShape = xStorage->GetStorageShape();
    const gert::Shape &weightShape = weightStorage->GetStorageShape();
    const gert::Shape &biasShape = biasStorage->GetStorageShape();
    if (xShape.GetDimNum() != 2 || weightShape.GetDimNum() != 2 || biasShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t inputSize = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t weightInputSize = static_cast<uint32_t>(weightShape.GetDim(0));
    const uint32_t hiddenSize = static_cast<uint32_t>(weightShape.GetDim(1));
    const uint32_t biasSize = static_cast<uint32_t>(biasShape.GetDim(0));
    if (batchSize == 0 || inputSize == 0 || hiddenSize == 0) {
        return ge::GRAPH_FAILED;
    }
    if (weightInputSize != inputSize || biasSize != hiddenSize) {
        return ge::GRAPH_FAILED;
    }

    MatmulSigmoidSumCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_inputSize(inputSize);
    tiling.set_hiddenSize(hiddenSize);
    tiling.set_blockDim(1U);

    context->SetBlockDim(1U);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *workspaceSize = context->GetWorkspaceSizes(1);
    workspaceSize[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    const gert::Shape *weightShape = context->GetInputShape(1);
    const gert::Shape *biasShape = context->GetInputShape(2);
    if (xShape == nullptr || weightShape == nullptr || biasShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDimNum() != 2 || weightShape->GetDimNum() != 2 || biasShape->GetDimNum() != 1) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDim(1) != weightShape->GetDim(0) || weightShape->GetDim(1) != biasShape->GetDim(0)) {
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
class MatmulSigmoidSumCustom : public OpDef {
public:
    explicit MatmulSigmoidSumCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(MatmulSigmoidSumCustom);
} // namespace ops
"""

kernel_src = """
#include "kernel_operator.h"

using namespace AscendC;

class MatmulSigmoidSumKernel {
public:
    __aicore__ inline MatmulSigmoidSumKernel() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t inputSize,
        uint32_t hiddenSize,
        uint32_t blockDim)
    {
        this->batchSize = batchSize;
        this->inputSize = inputSize;
        this->hiddenSize = hiddenSize;
        this->blockDim = blockDim;
        xGm.SetGlobalBuffer((__gm__ float *)x, static_cast<uint64_t>(batchSize) * inputSize);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, static_cast<uint64_t>(inputSize) * hiddenSize);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, hiddenSize);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize);
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreIdx = GetBlockIdx();
        for (uint32_t batchIdx = coreIdx; batchIdx < this->batchSize; batchIdx += this->blockDim) {
            const uint64_t xBase = static_cast<uint64_t>(batchIdx) * this->inputSize;
            float rowSum = 0.0f;
            for (uint32_t hiddenIdx = 0; hiddenIdx < this->hiddenSize; ++hiddenIdx) {
                float acc = biasGm.GetValue(hiddenIdx);
                for (uint32_t inputIdx = 0; inputIdx < this->inputSize; ++inputIdx) {
                    const float xValue = xGm.GetValue(xBase + inputIdx);
                    const float weightValue =
                        weightGm.GetValue(static_cast<uint64_t>(inputIdx) * this->hiddenSize + hiddenIdx);
                    acc += xValue * weightValue;
                }
                rowSum += Sigmoid(acc);
            }
            yGm.SetValue(batchIdx, rowSum);
        }
    }

private:
    __aicore__ inline float FastExp(float x) const
    {
        constexpr float ln2 = 0.69314718056f;
        if (x < -20.0f) {
            return 0.0f;
        }
        if (x > 20.0f) {
            x = 20.0f;
        }

        int32_t k = 0;
        while (x > 0.5f * ln2) {
            x -= ln2;
            ++k;
        }
        while (x < -0.5f * ln2) {
            x += ln2;
            --k;
        }

        const float x2 = x * x;
        const float x3 = x2 * x;
        const float x4 = x3 * x;
        const float x5 = x4 * x;
        float result = 1.0f + x + 0.5f * x2 + 0.16666667f * x3 + 0.04166667f * x4 + 0.0083333333f * x5;
        if (k > 0) {
            for (int32_t i = 0; i < k; ++i) {
                result *= 2.0f;
            }
        } else {
            for (int32_t i = 0; i < -k; ++i) {
                result *= 0.5f;
            }
        }
        return result;
    }

    __aicore__ inline float Sigmoid(float x) const
    {
        return 1.0f / (1.0f + FastExp(-x));
    }

    GlobalTensor<float> xGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> biasGm;
    GlobalTensor<float> yGm;
    uint32_t batchSize = 0;
    uint32_t inputSize = 0;
    uint32_t hiddenSize = 0;
    uint32_t blockDim = 1;
};

extern "C" __global__ __aicore__ void matmul_sigmoid_sum_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    MatmulSigmoidSumKernel op;
    op.Init(
        x,
        weight,
        bias,
        y,
        tilingData.batchSize,
        tilingData.inputSize,
        tilingData.hiddenSize,
        tilingData.blockDim);
    op.Process();
}
"""

python_bind_src = """
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor matmul_sigmoid_sum_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &bias)
{
    TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor");
    TORCH_CHECK(weight.dim() == 2, "weight must be a 2D tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
    TORCH_CHECK(x.size(1) == weight.size(0), "x.size(1) must match weight.size(0)");
    TORCH_CHECK(weight.size(1) == bias.size(0), "weight.size(1) must match bias.size(0)");

    auto outputShape = std::vector<int64_t>{x.size(0), 1};
    at::Tensor result = at::empty(outputShape, x.options());
    EXEC_NPU_CMD(aclnnMatmulSigmoidSumCustom, x, weight, bias, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m)
{
    m.impl("matmul_sigmoid_sum_custom", &matmul_sigmoid_sum_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("matmul_sigmoid_sum_custom", &matmul_sigmoid_sum_custom_impl_npu, "matmul + sigmoid + sum");
}
"""

model_src = '''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = torch.nn.Linear(input_size, hidden_size)

    def forward(self, x):
        weight = self.linear.weight.transpose(0, 1).contiguous()
        return custom_ops_lib.matmul_sigmoid_sum_custom(
            x,
            weight,
            self.linear.bias,
        )
'''
