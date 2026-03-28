project_json_src = '''
[
    {
        "op": "VanillaRnnCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "x",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "hidden",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "i2h_weight",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "i2h_bias",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "h2o_weight",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "h2o_bias",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            }
        ],
        "output_desc": [
            {
                "name": "output",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "new_hidden",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            }
        ]
    }
]
'''

host_tiling_src = """
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(VanillaRnnCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, inputSize);
TILING_DATA_FIELD_DEF(uint32_t, hiddenSize);
TILING_DATA_FIELD_DEF(uint32_t, outputSize);
TILING_DATA_FIELD_DEF(uint32_t, blockDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(VanillaRnnCustom, VanillaRnnCustomTilingData)
} // namespace optiling
"""

host_operator_src = """
#include "vanilla_rnn_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *xStorage = context->GetInputShape(0);
    const gert::StorageShape *hiddenStorage = context->GetInputShape(1);
    const gert::StorageShape *i2hWeightStorage = context->GetInputShape(2);
    const gert::StorageShape *i2hBiasStorage = context->GetInputShape(3);
    const gert::StorageShape *h2oWeightStorage = context->GetInputShape(4);
    const gert::StorageShape *h2oBiasStorage = context->GetInputShape(5);
    if (xStorage == nullptr || hiddenStorage == nullptr || i2hWeightStorage == nullptr ||
        i2hBiasStorage == nullptr || h2oWeightStorage == nullptr || h2oBiasStorage == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &xShape = xStorage->GetStorageShape();
    const gert::Shape &hiddenShape = hiddenStorage->GetStorageShape();
    const gert::Shape &i2hWeightShape = i2hWeightStorage->GetStorageShape();
    const gert::Shape &i2hBiasShape = i2hBiasStorage->GetStorageShape();
    const gert::Shape &h2oWeightShape = h2oWeightStorage->GetStorageShape();
    const gert::Shape &h2oBiasShape = h2oBiasStorage->GetStorageShape();
    if (xShape.GetDimNum() != 2 || hiddenShape.GetDimNum() != 2 || i2hWeightShape.GetDimNum() != 2 ||
        i2hBiasShape.GetDimNum() != 1 || h2oWeightShape.GetDimNum() != 2 || h2oBiasShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t inputSize = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t hiddenBatch = static_cast<uint32_t>(hiddenShape.GetDim(0));
    const uint32_t hiddenSize = static_cast<uint32_t>(hiddenShape.GetDim(1));
    const uint32_t i2hWeightRows = static_cast<uint32_t>(i2hWeightShape.GetDim(0));
    const uint32_t i2hWeightCols = static_cast<uint32_t>(i2hWeightShape.GetDim(1));
    const uint32_t i2hBiasSize = static_cast<uint32_t>(i2hBiasShape.GetDim(0));
    const uint32_t outputSize = static_cast<uint32_t>(h2oWeightShape.GetDim(0));
    const uint32_t h2oWeightCols = static_cast<uint32_t>(h2oWeightShape.GetDim(1));
    const uint32_t h2oBiasSize = static_cast<uint32_t>(h2oBiasShape.GetDim(0));

    if (batchSize == 0 || inputSize == 0 || hiddenSize == 0 || outputSize == 0) {
        return ge::GRAPH_FAILED;
    }
    if (hiddenBatch != batchSize || i2hWeightRows != hiddenSize || i2hWeightCols != inputSize + hiddenSize ||
        i2hBiasSize != hiddenSize || h2oWeightCols != hiddenSize || h2oBiasSize != outputSize) {
        return ge::GRAPH_FAILED;
    }

    VanillaRnnCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_inputSize(inputSize);
    tiling.set_hiddenSize(hiddenSize);
    tiling.set_outputSize(outputSize);
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
    const gert::Shape *hiddenShape = context->GetInputShape(1);
    const gert::Shape *h2oWeightShape = context->GetInputShape(4);
    if (xShape == nullptr || hiddenShape == nullptr || h2oWeightShape == nullptr) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(2);
    outputShape->SetDim(0, xShape->GetDim(0));
    outputShape->SetDim(1, h2oWeightShape->GetDim(0));

    gert::Shape *hiddenOutShape = context->GetOutputShape(1);
    *hiddenOutShape = *hiddenShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    context->SetOutputDataType(1, inputDataType);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class VanillaRnnCustom : public OpDef {
public:
    explicit VanillaRnnCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("hidden").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("i2h_weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("i2h_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("h2o_weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("h2o_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("output").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("new_hidden").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(VanillaRnnCustom);
} // namespace ops
"""

kernel_src = """
#include "kernel_operator.h"

using namespace AscendC;

class VanillaRnnKernel {
public:
    __aicore__ inline VanillaRnnKernel() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR hidden,
        GM_ADDR i2hWeight,
        GM_ADDR i2hBias,
        GM_ADDR h2oWeight,
        GM_ADDR h2oBias,
        GM_ADDR output,
        GM_ADDR newHidden,
        uint32_t batchSize,
        uint32_t inputSize,
        uint32_t hiddenSize,
        uint32_t outputSize,
        uint32_t blockDim)
    {
        this->batchSize = batchSize;
        this->inputSize = inputSize;
        this->hiddenSize = hiddenSize;
        this->outputSize = outputSize;
        this->blockDim = blockDim;

        xGm.SetGlobalBuffer((__gm__ float *)x, static_cast<uint64_t>(batchSize) * inputSize);
        hiddenGm.SetGlobalBuffer((__gm__ float *)hidden, static_cast<uint64_t>(batchSize) * hiddenSize);
        i2hWeightGm.SetGlobalBuffer((__gm__ float *)i2hWeight, static_cast<uint64_t>(hiddenSize) * (inputSize + hiddenSize));
        i2hBiasGm.SetGlobalBuffer((__gm__ float *)i2hBias, hiddenSize);
        h2oWeightGm.SetGlobalBuffer((__gm__ float *)h2oWeight, static_cast<uint64_t>(outputSize) * hiddenSize);
        h2oBiasGm.SetGlobalBuffer((__gm__ float *)h2oBias, outputSize);
        outputGm.SetGlobalBuffer((__gm__ float *)output, static_cast<uint64_t>(batchSize) * outputSize);
        newHiddenGm.SetGlobalBuffer((__gm__ float *)newHidden, static_cast<uint64_t>(batchSize) * hiddenSize);
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreIdx = GetBlockIdx();
        for (uint32_t batchIdx = coreIdx; batchIdx < this->batchSize; batchIdx += this->blockDim) {
            ComputeHidden(batchIdx);
            ComputeOutput(batchIdx);
        }
    }

private:
    __aicore__ inline void ComputeHidden(uint32_t batchIdx)
    {
        const uint64_t xBase = static_cast<uint64_t>(batchIdx) * this->inputSize;
        const uint64_t hiddenBase = static_cast<uint64_t>(batchIdx) * this->hiddenSize;
        const uint64_t newHiddenBase = static_cast<uint64_t>(batchIdx) * this->hiddenSize;
        for (uint32_t hiddenIdx = 0; hiddenIdx < this->hiddenSize; ++hiddenIdx) {
            float acc = i2hBiasGm.GetValue(hiddenIdx);
            const uint64_t i2hWeightBase = static_cast<uint64_t>(hiddenIdx) * (this->inputSize + this->hiddenSize);
            for (uint32_t inputIdx = 0; inputIdx < this->inputSize; ++inputIdx) {
                acc += xGm.GetValue(xBase + inputIdx) * i2hWeightGm.GetValue(i2hWeightBase + inputIdx);
            }
            const uint64_t hiddenWeightBase = i2hWeightBase + this->inputSize;
            for (uint32_t prevHiddenIdx = 0; prevHiddenIdx < this->hiddenSize; ++prevHiddenIdx) {
                acc += hiddenGm.GetValue(hiddenBase + prevHiddenIdx) * i2hWeightGm.GetValue(hiddenWeightBase + prevHiddenIdx);
            }
            newHiddenGm.SetValue(newHiddenBase + hiddenIdx, TanhApprox(acc));
        }
    }

    __aicore__ inline void ComputeOutput(uint32_t batchIdx)
    {
        const uint64_t newHiddenBase = static_cast<uint64_t>(batchIdx) * this->hiddenSize;
        const uint64_t outputBase = static_cast<uint64_t>(batchIdx) * this->outputSize;
        for (uint32_t outIdx = 0; outIdx < this->outputSize; ++outIdx) {
            float acc = h2oBiasGm.GetValue(outIdx);
            const uint64_t h2oWeightBase = static_cast<uint64_t>(outIdx) * this->hiddenSize;
            for (uint32_t hiddenIdx = 0; hiddenIdx < this->hiddenSize; ++hiddenIdx) {
                acc += newHiddenGm.GetValue(newHiddenBase + hiddenIdx) * h2oWeightGm.GetValue(h2oWeightBase + hiddenIdx);
            }
            outputGm.SetValue(outputBase + outIdx, acc);
        }
    }

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
    GlobalTensor<float> i2hWeightGm;
    GlobalTensor<float> i2hBiasGm;
    GlobalTensor<float> h2oWeightGm;
    GlobalTensor<float> h2oBiasGm;
    GlobalTensor<float> outputGm;
    GlobalTensor<float> newHiddenGm;
    uint32_t batchSize = 0;
    uint32_t inputSize = 0;
    uint32_t hiddenSize = 0;
    uint32_t outputSize = 0;
    uint32_t blockDim = 1;
};

extern "C" __global__ __aicore__ void vanilla_rnn_custom(
    GM_ADDR x,
    GM_ADDR hidden,
    GM_ADDR i2hWeight,
    GM_ADDR i2hBias,
    GM_ADDR h2oWeight,
    GM_ADDR h2oBias,
    GM_ADDR output,
    GM_ADDR newHidden,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    VanillaRnnKernel op;
    op.Init(
        x,
        hidden,
        i2hWeight,
        i2hBias,
        h2oWeight,
        h2oBias,
        output,
        newHidden,
        tilingData.batchSize,
        tilingData.inputSize,
        tilingData.hiddenSize,
        tilingData.outputSize,
        tilingData.blockDim);
    op.Process();
}
"""

python_bind_src = """
#include <torch/library.h>
#include <torch/extension.h>
#include <tuple>
#include <vector>
#include "pytorch_npu_helper.hpp"

std::tuple<at::Tensor, at::Tensor> vanilla_rnn_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &hidden,
    const at::Tensor &i2hWeight,
    const at::Tensor &i2hBias,
    const at::Tensor &h2oWeight,
    const at::Tensor &h2oBias)
{
    TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor");
    TORCH_CHECK(hidden.dim() == 2, "hidden must be a 2D tensor");
    TORCH_CHECK(i2hWeight.dim() == 2, "i2h_weight must be a 2D tensor");
    TORCH_CHECK(i2hBias.dim() == 1, "i2h_bias must be a 1D tensor");
    TORCH_CHECK(h2oWeight.dim() == 2, "h2o_weight must be a 2D tensor");
    TORCH_CHECK(h2oBias.dim() == 1, "h2o_bias must be a 1D tensor");
    TORCH_CHECK(x.size(0) == hidden.size(0), "x batch size must match hidden batch size");
    TORCH_CHECK(i2hWeight.size(0) == hidden.size(1), "i2h_weight rows must equal hidden size");
    TORCH_CHECK(i2hWeight.size(1) == x.size(1) + hidden.size(1), "i2h_weight cols must equal input + hidden");
    TORCH_CHECK(i2hBias.size(0) == hidden.size(1), "i2h_bias size must equal hidden size");
    TORCH_CHECK(h2oWeight.size(1) == hidden.size(1), "h2o_weight cols must equal hidden size");
    TORCH_CHECK(h2oBias.size(0) == h2oWeight.size(0), "h2o_bias size must equal output size");

    at::Tensor output = at::empty({x.size(0), h2oWeight.size(0)}, x.options());
    at::Tensor newHidden = at::empty({x.size(0), hidden.size(1)}, x.options());
    EXEC_NPU_CMD(aclnnVanillaRnnCustom, x, hidden, i2hWeight, i2hBias, h2oWeight, h2oBias, output, newHidden);
    return std::make_tuple(output, newHidden);
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m)
{
    m.impl("vanilla_rnn_custom", &vanilla_rnn_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("vanilla_rnn_custom", &vanilla_rnn_custom_impl_npu, "vanilla rnn");
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.hidden = self.hidden.to(x.device)
        output, new_hidden = custom_ops_lib.vanilla_rnn_custom(
            x,
            self.hidden,
            self.i2h.weight,
            self.i2h.bias,
            self.h2o.weight,
            self.h2o.bias,
        )
        self.hidden = new_hidden
        return output
'''
