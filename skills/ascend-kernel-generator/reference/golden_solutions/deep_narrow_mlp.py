import json


INPUT_DESC = [
    {"name": "x", "param_type": "required", "format": ["ND"], "type": ["float"]},
]
for layer_idx in range(9):
    INPUT_DESC.append(
        {"name": f"w{layer_idx}", "param_type": "required", "format": ["ND"], "type": ["float"]}
    )
    INPUT_DESC.append(
        {"name": f"b{layer_idx}", "param_type": "required", "format": ["ND"], "type": ["float"]}
    )

project_json_src = json.dumps(
    [
        {
            "op": "DeepNarrowMlpCustom",
            "language": "cpp",
            "input_desc": INPUT_DESC,
            "output_desc": [
                {
                    "name": "y",
                    "param_type": "required",
                    "format": ["ND"],
                    "type": ["float"],
                }
            ],
        }
    ],
    indent=4,
)

host_tiling_src = r"""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DeepNarrowMlpCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DeepNarrowMlpCustom, DeepNarrowMlpCustomTilingData)
} // namespace optiling
"""

host_operator_src = r"""
#include "deep_narrow_mlp_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
constexpr uint32_t kInputSize = 1000;
constexpr uint32_t kHiddenSize = 50;
constexpr uint32_t kOutputSize = 10;
constexpr uint32_t kLayerCount = 9;

bool CheckMatrixShape(const gert::StorageShape *shape, int64_t rows, int64_t cols)
{
    if (shape == nullptr) {
        return false;
    }
    const auto storageShape = shape->GetStorageShape();
    return storageShape.GetDimNum() == 2 && storageShape.GetDim(0) == rows && storageShape.GetDim(1) == cols;
}

bool CheckVectorShape(const gert::StorageShape *shape, int64_t length)
{
    if (shape == nullptr) {
        return false;
    }
    const auto storageShape = shape->GetStorageShape();
    return storageShape.GetDimNum() == 1 && storageShape.GetDim(0) == length;
}
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *xShape = context->GetInputShape(0);
    if (xShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto xStorageShape = xShape->GetStorageShape();
    if (xStorageShape.GetDimNum() != 2 || xStorageShape.GetDim(1) != kInputSize || xStorageShape.GetDim(0) <= 0) {
        return ge::GRAPH_FAILED;
    }

    if (!CheckMatrixShape(context->GetInputShape(1), kHiddenSize, kInputSize) ||
        !CheckVectorShape(context->GetInputShape(2), kHiddenSize) ||
        !CheckMatrixShape(context->GetInputShape(3), kHiddenSize, kHiddenSize) ||
        !CheckVectorShape(context->GetInputShape(4), kHiddenSize) ||
        !CheckMatrixShape(context->GetInputShape(5), kHiddenSize, kHiddenSize) ||
        !CheckVectorShape(context->GetInputShape(6), kHiddenSize) ||
        !CheckMatrixShape(context->GetInputShape(7), kHiddenSize, kHiddenSize) ||
        !CheckVectorShape(context->GetInputShape(8), kHiddenSize) ||
        !CheckMatrixShape(context->GetInputShape(9), kHiddenSize, kHiddenSize) ||
        !CheckVectorShape(context->GetInputShape(10), kHiddenSize) ||
        !CheckMatrixShape(context->GetInputShape(11), kHiddenSize, kHiddenSize) ||
        !CheckVectorShape(context->GetInputShape(12), kHiddenSize) ||
        !CheckMatrixShape(context->GetInputShape(13), kHiddenSize, kHiddenSize) ||
        !CheckVectorShape(context->GetInputShape(14), kHiddenSize) ||
        !CheckMatrixShape(context->GetInputShape(15), kHiddenSize, kHiddenSize) ||
        !CheckVectorShape(context->GetInputShape(16), kHiddenSize) ||
        !CheckMatrixShape(context->GetInputShape(17), kOutputSize, kHiddenSize) ||
        !CheckVectorShape(context->GetInputShape(18), kOutputSize)) {
        return ge::GRAPH_FAILED;
    }

    DeepNarrowMlpCustomTilingData tiling;
    tiling.set_batchSize(static_cast<uint32_t>(xStorageShape.GetDim(0)));
    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    workspaceSizes[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    if (xShape == nullptr || xShape->GetDimNum() != 2 || xShape->GetDim(1) != kInputSize || xShape->GetDim(0) <= 0) {
        return GRAPH_FAILED;
    }
    gert::Shape *yShape = context->GetOutputShape(0);
    yShape->SetDimNum(2);
    yShape->SetDim(0, xShape->GetDim(0));
    yShape->SetDim(1, kOutputSize);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class DeepNarrowMlpCustom : public OpDef {
public:
    explicit DeepNarrowMlpCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("w0").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("b0").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("w1").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("b1").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("w2").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("b2").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("w3").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("b3").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("w4").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("b4").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("w5").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("b5").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("w6").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("b6").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("w7").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("b7").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("w8").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("b8").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(DeepNarrowMlpCustom);
} // namespace ops
"""

kernel_src = r"""
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kInputSize = 1000;
constexpr uint32_t kHiddenSize = 50;
constexpr uint32_t kOutputSize = 10;

__aicore__ inline float Relu(float value)
{
    return value > 0.0f ? value : 0.0f;
}
} // namespace

class KernelDeepNarrowMlp {
public:
    __aicore__ inline KernelDeepNarrowMlp() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR w0, GM_ADDR b0,
        GM_ADDR w1, GM_ADDR b1,
        GM_ADDR w2, GM_ADDR b2,
        GM_ADDR w3, GM_ADDR b3,
        GM_ADDR w4, GM_ADDR b4,
        GM_ADDR w5, GM_ADDR b5,
        GM_ADDR w6, GM_ADDR b6,
        GM_ADDR w7, GM_ADDR b7,
        GM_ADDR w8, GM_ADDR b8,
        GM_ADDR y,
        uint32_t batchSize)
    {
        batchSize_ = batchSize;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X *>(x), static_cast<uint64_t>(batchSize) * kInputSize);
        w0Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_W0 *>(w0), kHiddenSize * kInputSize);
        b0Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_B0 *>(b0), kHiddenSize);
        w1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_W1 *>(w1), kHiddenSize * kHiddenSize);
        b1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_B1 *>(b1), kHiddenSize);
        w2Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_W2 *>(w2), kHiddenSize * kHiddenSize);
        b2Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_B2 *>(b2), kHiddenSize);
        w3Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_W3 *>(w3), kHiddenSize * kHiddenSize);
        b3Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_B3 *>(b3), kHiddenSize);
        w4Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_W4 *>(w4), kHiddenSize * kHiddenSize);
        b4Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_B4 *>(b4), kHiddenSize);
        w5Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_W5 *>(w5), kHiddenSize * kHiddenSize);
        b5Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_B5 *>(b5), kHiddenSize);
        w6Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_W6 *>(w6), kHiddenSize * kHiddenSize);
        b6Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_B6 *>(b6), kHiddenSize);
        w7Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_W7 *>(w7), kHiddenSize * kHiddenSize);
        b7Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_B7 *>(b7), kHiddenSize);
        w8Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_W8 *>(w8), kOutputSize * kHiddenSize);
        b8Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_B8 *>(b8), kOutputSize);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_Y *>(y), static_cast<uint64_t>(batchSize) * kOutputSize);
    }

    __aicore__ inline void Process()
    {
        float hiddenA[kHiddenSize];
        float hiddenB[kHiddenSize];

        for (uint32_t batch = 0; batch < batchSize_; ++batch) {
            ComputeFirstLayer(batch, hiddenA);
            ComputeHiddenLayer(hiddenA, hiddenB, w1Gm, b1Gm);
            ComputeHiddenLayer(hiddenB, hiddenA, w2Gm, b2Gm);
            ComputeHiddenLayer(hiddenA, hiddenB, w3Gm, b3Gm);
            ComputeHiddenLayer(hiddenB, hiddenA, w4Gm, b4Gm);
            ComputeHiddenLayer(hiddenA, hiddenB, w5Gm, b5Gm);
            ComputeHiddenLayer(hiddenB, hiddenA, w6Gm, b6Gm);
            ComputeHiddenLayer(hiddenA, hiddenB, w7Gm, b7Gm);
            ComputeOutputLayer(batch, hiddenB);
        }
    }

private:
    template <typename WeightTensor, typename BiasTensor>
    __aicore__ inline void ComputeHiddenLayer(const float *input, float *output, WeightTensor &weight, BiasTensor &bias)
    {
        for (uint32_t outIdx = 0; outIdx < kHiddenSize; ++outIdx) {
            float acc = static_cast<float>(bias.GetValue(outIdx));
            const uint32_t weightBase = outIdx * kHiddenSize;
            for (uint32_t inIdx = 0; inIdx < kHiddenSize; ++inIdx) {
                acc += input[inIdx] * static_cast<float>(weight.GetValue(weightBase + inIdx));
            }
            output[outIdx] = Relu(acc);
        }
    }

    __aicore__ inline void ComputeFirstLayer(uint32_t batch, float *output)
    {
        const uint32_t xBase = batch * kInputSize;
        for (uint32_t outIdx = 0; outIdx < kHiddenSize; ++outIdx) {
            float acc = static_cast<float>(b0Gm.GetValue(outIdx));
            const uint32_t weightBase = outIdx * kInputSize;
            for (uint32_t inIdx = 0; inIdx < kInputSize; ++inIdx) {
                acc += static_cast<float>(xGm.GetValue(xBase + inIdx)) * static_cast<float>(w0Gm.GetValue(weightBase + inIdx));
            }
            output[outIdx] = Relu(acc);
        }
    }

    __aicore__ inline void ComputeOutputLayer(uint32_t batch, const float *input)
    {
        const uint32_t yBase = batch * kOutputSize;
        for (uint32_t outIdx = 0; outIdx < kOutputSize; ++outIdx) {
            float acc = static_cast<float>(b8Gm.GetValue(outIdx));
            const uint32_t weightBase = outIdx * kHiddenSize;
            for (uint32_t inIdx = 0; inIdx < kHiddenSize; ++inIdx) {
                acc += input[inIdx] * static_cast<float>(w8Gm.GetValue(weightBase + inIdx));
            }
            yGm.SetValue(yBase + outIdx, static_cast<DTYPE_Y>(acc));
        }
    }

    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_W0> w0Gm;
    GlobalTensor<DTYPE_B0> b0Gm;
    GlobalTensor<DTYPE_W1> w1Gm;
    GlobalTensor<DTYPE_B1> b1Gm;
    GlobalTensor<DTYPE_W2> w2Gm;
    GlobalTensor<DTYPE_B2> b2Gm;
    GlobalTensor<DTYPE_W3> w3Gm;
    GlobalTensor<DTYPE_B3> b3Gm;
    GlobalTensor<DTYPE_W4> w4Gm;
    GlobalTensor<DTYPE_B4> b4Gm;
    GlobalTensor<DTYPE_W5> w5Gm;
    GlobalTensor<DTYPE_B5> b5Gm;
    GlobalTensor<DTYPE_W6> w6Gm;
    GlobalTensor<DTYPE_B6> b6Gm;
    GlobalTensor<DTYPE_W7> w7Gm;
    GlobalTensor<DTYPE_B7> b7Gm;
    GlobalTensor<DTYPE_W8> w8Gm;
    GlobalTensor<DTYPE_B8> b8Gm;
    GlobalTensor<DTYPE_Y> yGm;
    uint32_t batchSize_ = 0;
};

extern "C" __global__ __aicore__ void deep_narrow_mlp_custom(
    GM_ADDR x,
    GM_ADDR w0, GM_ADDR b0,
    GM_ADDR w1, GM_ADDR b1,
    GM_ADDR w2, GM_ADDR b2,
    GM_ADDR w3, GM_ADDR b3,
    GM_ADDR w4, GM_ADDR b4,
    GM_ADDR w5, GM_ADDR b5,
    GM_ADDR w6, GM_ADDR b6,
    GM_ADDR w7, GM_ADDR b7,
    GM_ADDR w8, GM_ADDR b8,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    KernelDeepNarrowMlp op;
    op.Init(
        x,
        w0, b0,
        w1, b1,
        w2, b2,
        w3, b3,
        w4, b4,
        w5, b5,
        w6, b6,
        w7, b7,
        w8, b8,
        y,
        tilingData.batchSize);
    op.Process();
}
"""

python_bind_src = r"""
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor deep_narrow_mlp_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &w0, const at::Tensor &b0,
    const at::Tensor &w1, const at::Tensor &b1,
    const at::Tensor &w2, const at::Tensor &b2,
    const at::Tensor &w3, const at::Tensor &b3,
    const at::Tensor &w4, const at::Tensor &b4,
    const at::Tensor &w5, const at::Tensor &b5,
    const at::Tensor &w6, const at::Tensor &b6,
    const at::Tensor &w7, const at::Tensor &b7,
    const at::Tensor &w8, const at::Tensor &b8)
{
    auto result = at::empty({x.size(0), 10}, x.options());
    EXEC_NPU_CMD(
        aclnnDeepNarrowMlpCustom,
        x,
        w0, b0,
        w1, b1,
        w2, b2,
        w3, b3,
        w4, b4,
        w5, b5,
        w6, b6,
        w7, b7,
        w8, b8,
        result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m)
{
    m.impl("deep_narrow_mlp_custom", &deep_narrow_mlp_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("deep_narrow_mlp_custom", &deep_narrow_mlp_custom_impl_npu, "deep narrow mlp custom");
}
"""

model_src = r'''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super(ModelNew, self).__init__()
        expected_hidden = [50, 50, 50, 50, 50, 50, 50, 50]
        if input_size != 1000 or output_size != 10 or list(hidden_layer_sizes) != expected_hidden:
            raise ValueError("deep_narrow_mlp_custom only supports input_size=1000, hidden_layer_sizes=[50]*8, output_size=10")

        layers = []
        current_input_size = input_size
        for hidden_size in hidden_layer_sizes:
            layers.append(torch.nn.Linear(current_input_size, hidden_size))
            current_input_size = hidden_size
        layers.append(torch.nn.Linear(current_input_size, output_size))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        return custom_ops_lib.deep_narrow_mlp_custom(
            x,
            self.layers[0].weight, self.layers[0].bias,
            self.layers[1].weight, self.layers[1].bias,
            self.layers[2].weight, self.layers[2].bias,
            self.layers[3].weight, self.layers[3].bias,
            self.layers[4].weight, self.layers[4].bias,
            self.layers[5].weight, self.layers[5].bias,
            self.layers[6].weight, self.layers[6].bias,
            self.layers[7].weight, self.layers[7].bias,
            self.layers[8].weight, self.layers[8].bias,
        )
'''
