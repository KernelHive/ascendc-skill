project_json_src = '''
[
    {
        "op": "VanillaRnnHiddenCustom",
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
                "name": "hidden",
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
BEGIN_TILING_DATA_DEF(VanillaRnnHiddenCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, inputSize);
TILING_DATA_FIELD_DEF(uint32_t, hiddenSize);
TILING_DATA_FIELD_DEF(uint32_t, blockDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(VanillaRnnHiddenCustom, VanillaRnnHiddenCustomTilingData)
} // namespace optiling
"""

host_operator_src = """
#include "vanilla_rnn_hidden_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *xStorage = context->GetInputShape(0);
    const gert::StorageShape *hiddenStorage = context->GetInputShape(1);
    const gert::StorageShape *weightStorage = context->GetInputShape(2);
    const gert::StorageShape *biasStorage = context->GetInputShape(3);
    if (xStorage == nullptr || hiddenStorage == nullptr || weightStorage == nullptr || biasStorage == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &xShape = xStorage->GetStorageShape();
    const gert::Shape &hiddenShape = hiddenStorage->GetStorageShape();
    const gert::Shape &weightShape = weightStorage->GetStorageShape();
    const gert::Shape &biasShape = biasStorage->GetStorageShape();
    if (xShape.GetDimNum() != 2 || hiddenShape.GetDimNum() != 2 || weightShape.GetDimNum() != 2 ||
        biasShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t inputSize = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t hiddenBatch = static_cast<uint32_t>(hiddenShape.GetDim(0));
    const uint32_t hiddenSize = static_cast<uint32_t>(hiddenShape.GetDim(1));
    const uint32_t weightRows = static_cast<uint32_t>(weightShape.GetDim(0));
    const uint32_t weightCols = static_cast<uint32_t>(weightShape.GetDim(1));
    const uint32_t biasSize = static_cast<uint32_t>(biasShape.GetDim(0));
    if (batchSize == 0 || inputSize == 0 || hiddenSize == 0) {
        return ge::GRAPH_FAILED;
    }
    if (hiddenBatch != batchSize || weightRows != hiddenSize || weightCols != inputSize + hiddenSize ||
        biasSize != hiddenSize) {
        return ge::GRAPH_FAILED;
    }

    VanillaRnnHiddenCustomTilingData tiling;
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
    const gert::Shape *hiddenShape = context->GetInputShape(1);
    const gert::Shape *weightShape = context->GetInputShape(2);
    const gert::Shape *biasShape = context->GetInputShape(3);
    if (hiddenShape == nullptr || weightShape == nullptr || biasShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (hiddenShape->GetDimNum() != 2 || weightShape->GetDimNum() != 2 || biasShape->GetDimNum() != 1) {
        return GRAPH_FAILED;
    }
    if (hiddenShape->GetDim(1) != weightShape->GetDim(0) ||
        weightShape->GetDim(1) != context->GetInputShape(0)->GetDim(1) + hiddenShape->GetDim(1) ||
        weightShape->GetDim(0) != biasShape->GetDim(0)) {
        return GRAPH_FAILED;
    }

    gert::Shape *yShape = context->GetOutputShape(0);
    *yShape = *hiddenShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class VanillaRnnHiddenCustom : public OpDef {
public:
    explicit VanillaRnnHiddenCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("hidden").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(VanillaRnnHiddenCustom);
} // namespace ops
"""

kernel_src = """
#include "kernel_operator.h"

using namespace AscendC;

class VanillaRnnHiddenKernel {
public:
    __aicore__ inline VanillaRnnHiddenKernel() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR hidden,
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
        hiddenGm.SetGlobalBuffer((__gm__ float *)hidden, static_cast<uint64_t>(batchSize) * hiddenSize);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, static_cast<uint64_t>(hiddenSize) * (inputSize + hiddenSize));
        biasGm.SetGlobalBuffer((__gm__ float *)bias, hiddenSize);
        yGm.SetGlobalBuffer((__gm__ float *)y, static_cast<uint64_t>(batchSize) * hiddenSize);
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreIdx = GetBlockIdx();
        for (uint32_t batchIdx = coreIdx; batchIdx < this->batchSize; batchIdx += this->blockDim) {
            const uint64_t xBase = static_cast<uint64_t>(batchIdx) * this->inputSize;
            const uint64_t hiddenBase = static_cast<uint64_t>(batchIdx) * this->hiddenSize;
            const uint64_t yBase = static_cast<uint64_t>(batchIdx) * this->hiddenSize;
            for (uint32_t hiddenIdx = 0; hiddenIdx < this->hiddenSize; ++hiddenIdx) {
                float acc = biasGm.GetValue(hiddenIdx);
                for (uint32_t inputIdx = 0; inputIdx < this->inputSize; ++inputIdx) {
                    const float xValue = xGm.GetValue(xBase + inputIdx);
                    const float wValue =
                        weightGm.GetValue(static_cast<uint64_t>(hiddenIdx) * (this->inputSize + this->hiddenSize) + inputIdx);
                    acc += xValue * wValue;
                }
                const uint64_t hiddenWeightBase =
                    static_cast<uint64_t>(hiddenIdx) * (this->inputSize + this->hiddenSize) + this->inputSize;
                for (uint32_t prevHiddenIdx = 0; prevHiddenIdx < this->hiddenSize; ++prevHiddenIdx) {
                    const float hiddenValue = hiddenGm.GetValue(hiddenBase + prevHiddenIdx);
                    const float wValue = weightGm.GetValue(hiddenWeightBase + prevHiddenIdx);
                    acc += hiddenValue * wValue;
                }
                yGm.SetValue(yBase + hiddenIdx, TanhApprox(acc));
            }
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

    __aicore__ inline float TanhApprox(float x) const
    {
        return 2.0f * Sigmoid(2.0f * x) - 1.0f;
    }

    GlobalTensor<float> xGm;
    GlobalTensor<float> hiddenGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> biasGm;
    GlobalTensor<float> yGm;
    uint32_t batchSize = 0;
    uint32_t inputSize = 0;
    uint32_t hiddenSize = 0;
    uint32_t blockDim = 1;
};

extern "C" __global__ __aicore__ void vanilla_rnn_hidden_custom(
    GM_ADDR x,
    GM_ADDR hidden,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    VanillaRnnHiddenKernel op;
    op.Init(
        x,
        hidden,
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

at::Tensor vanilla_rnn_hidden_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &hidden,
    const at::Tensor &weight,
    const at::Tensor &bias)
{
    TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor");
    TORCH_CHECK(hidden.dim() == 2, "hidden must be a 2D tensor");
    TORCH_CHECK(weight.dim() == 2, "weight must be a 2D tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
    TORCH_CHECK(x.size(0) == hidden.size(0), "x batch size must match hidden batch size");
    TORCH_CHECK(weight.size(0) == hidden.size(1), "weight rows must equal hidden size");
    TORCH_CHECK(weight.size(1) == x.size(1) + hidden.size(1), "weight cols must equal input + hidden");
    TORCH_CHECK(bias.size(0) == hidden.size(1), "bias size must equal hidden size");

    auto outputShape = std::vector<int64_t>{x.size(0), hidden.size(1)};
    at::Tensor result = at::empty(outputShape, x.options());
    EXEC_NPU_CMD(aclnnVanillaRnnHiddenCustom, x, hidden, weight, bias, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m)
{
    m.impl("vanilla_rnn_hidden_custom", &vanilla_rnn_hidden_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("vanilla_rnn_hidden_custom", &vanilla_rnn_hidden_custom_impl_npu, "vanilla rnn hidden update");
}
"""

model_src = '''
import torch
import torch_npu
import custom_ops_lib


batch_size = 8
input_size = 1024
hidden_size = 256
output_size = 128
sequence_length = 256


class ModelNew(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden = torch.randn((batch_size, hidden_size))
        self.i2h = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = torch.nn.Linear(hidden_size, output_size)
        self.tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.hidden = self.hidden.to(x.device)
        self.hidden = custom_ops_lib.vanilla_rnn_hidden_custom(
            x,
            self.hidden,
            self.i2h.weight,
            self.i2h.bias,
        )
        return self.hidden
'''
