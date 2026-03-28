import json


INPUT_DESC = [
    {"name": "x", "param_type": "required", "format": ["ND"], "type": ["float"]},
    {"name": "w0", "param_type": "required", "format": ["ND"], "type": ["float"]},
    {"name": "b0", "param_type": "required", "format": ["ND"], "type": ["float"]},
    {"name": "w1", "param_type": "required", "format": ["ND"], "type": ["float"]},
    {"name": "b1", "param_type": "required", "format": ["ND"], "type": ["float"]},
    {"name": "w2", "param_type": "required", "format": ["ND"], "type": ["float"]},
    {"name": "b2", "param_type": "required", "format": ["ND"], "type": ["float"]},
]


project_json_src = json.dumps(
    [
        {
            "op": "MlpCustom",
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
BEGIN_TILING_DATA_DEF(MlpCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MlpCustom, MlpCustomTilingData)
} // namespace optiling
"""


host_operator_src = r"""
#include "mlp_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
constexpr uint32_t kInputSize = 1000;
constexpr uint32_t kHiddenSize0 = 400;
constexpr uint32_t kHiddenSize1 = 800;
constexpr uint32_t kOutputSize = 500;

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
    if (xStorageShape.GetDimNum() != 2 || xStorageShape.GetDim(0) <= 0 || xStorageShape.GetDim(1) != kInputSize) {
        return ge::GRAPH_FAILED;
    }

    if (!CheckMatrixShape(context->GetInputShape(1), kHiddenSize0, kInputSize) ||
        !CheckVectorShape(context->GetInputShape(2), kHiddenSize0) ||
        !CheckMatrixShape(context->GetInputShape(3), kHiddenSize1, kHiddenSize0) ||
        !CheckVectorShape(context->GetInputShape(4), kHiddenSize1) ||
        !CheckMatrixShape(context->GetInputShape(5), kOutputSize, kHiddenSize1) ||
        !CheckVectorShape(context->GetInputShape(6), kOutputSize)) {
        return ge::GRAPH_FAILED;
    }

    MlpCustomTilingData tiling;
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
class MlpCustom : public OpDef {
public:
    explicit MlpCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("w0").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("b0").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("w1").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("b1").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("w2").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("b2").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(MlpCustom);
} // namespace ops
"""


kernel_src = r"""
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kInputSize = 1000;
constexpr uint32_t kHiddenSize0 = 400;
constexpr uint32_t kHiddenSize1 = 800;
constexpr uint32_t kOutputSize = 500;

__aicore__ inline float Relu(float value)
{
    return value > 0.0f ? value : 0.0f;
}
} // namespace

class KernelMlp {
public:
    __aicore__ inline KernelMlp() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR w0,
        GM_ADDR b0,
        GM_ADDR w1,
        GM_ADDR b1,
        GM_ADDR w2,
        GM_ADDR b2,
        GM_ADDR y,
        uint32_t batchSize)
    {
        batchSize_ = batchSize;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X *>(x), static_cast<uint64_t>(batchSize) * kInputSize);
        w0Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_W0 *>(w0), kHiddenSize0 * kInputSize);
        b0Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_B0 *>(b0), kHiddenSize0);
        w1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_W1 *>(w1), kHiddenSize1 * kHiddenSize0);
        b1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_B1 *>(b1), kHiddenSize1);
        w2Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_W2 *>(w2), kOutputSize * kHiddenSize1);
        b2Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_B2 *>(b2), kOutputSize);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_Y *>(y), static_cast<uint64_t>(batchSize) * kOutputSize);
    }

    __aicore__ inline void Process()
    {
        float hidden0[kHiddenSize0];
        float hidden1[kHiddenSize1];

        for (uint32_t batch = 0; batch < batchSize_; ++batch) {
            ComputeFirstLayer(batch, hidden0);
            ComputeSecondLayer(hidden0, hidden1);
            ComputeOutputLayer(batch, hidden1);
        }
    }

private:
    __aicore__ inline void ComputeFirstLayer(uint32_t batch, float *output)
    {
        const uint32_t xBase = batch * kInputSize;
        for (uint32_t outIdx = 0; outIdx < kHiddenSize0; ++outIdx) {
            float acc = static_cast<float>(b0Gm.GetValue(outIdx));
            const uint32_t weightBase = outIdx * kInputSize;
            for (uint32_t inIdx = 0; inIdx < kInputSize; ++inIdx) {
                acc += static_cast<float>(xGm.GetValue(xBase + inIdx)) *
                       static_cast<float>(w0Gm.GetValue(weightBase + inIdx));
            }
            output[outIdx] = Relu(acc);
        }
    }

    __aicore__ inline void ComputeSecondLayer(const float *input, float *output)
    {
        for (uint32_t outIdx = 0; outIdx < kHiddenSize1; ++outIdx) {
            float acc = static_cast<float>(b1Gm.GetValue(outIdx));
            const uint32_t weightBase = outIdx * kHiddenSize0;
            for (uint32_t inIdx = 0; inIdx < kHiddenSize0; ++inIdx) {
                acc += input[inIdx] * static_cast<float>(w1Gm.GetValue(weightBase + inIdx));
            }
            output[outIdx] = Relu(acc);
        }
    }

    __aicore__ inline void ComputeOutputLayer(uint32_t batch, const float *input)
    {
        const uint32_t yBase = batch * kOutputSize;
        for (uint32_t outIdx = 0; outIdx < kOutputSize; ++outIdx) {
            float acc = static_cast<float>(b2Gm.GetValue(outIdx));
            const uint32_t weightBase = outIdx * kHiddenSize1;
            for (uint32_t inIdx = 0; inIdx < kHiddenSize1; ++inIdx) {
                acc += input[inIdx] * static_cast<float>(w2Gm.GetValue(weightBase + inIdx));
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
    GlobalTensor<DTYPE_Y> yGm;
    uint32_t batchSize_ = 0;
};

extern "C" __global__ __aicore__ void mlp_custom(
    GM_ADDR x,
    GM_ADDR w0,
    GM_ADDR b0,
    GM_ADDR w1,
    GM_ADDR b1,
    GM_ADDR w2,
    GM_ADDR b2,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    KernelMlp op;
    op.Init(x, w0, b0, w1, b1, w2, b2, y, tilingData.batchSize);
    op.Process();
}
"""


python_bind_src = r"""
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor mlp_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &w0,
    const at::Tensor &b0,
    const at::Tensor &w1,
    const at::Tensor &b1,
    const at::Tensor &w2,
    const at::Tensor &b2)
{
    auto result = at::empty({x.size(0), 500}, x.options());
    EXEC_NPU_CMD(aclnnMlpCustom, x, w0, b0, w1, b1, w2, b2, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m)
{
    m.impl("mlp_custom", &mlp_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mlp_custom", &mlp_custom_impl_npu, "three-layer mlp custom");
}
"""


model_src = r'''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(ModelNew, self).__init__()
        if input_size != 1000 or list(layer_sizes) != [400, 800] or output_size != 500:
            raise ValueError("mlp_custom only supports input_size=1000, layer_sizes=[400, 800], output_size=500")

        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(input_size, layer_sizes[0]),
                torch.nn.Linear(layer_sizes[0], layer_sizes[1]),
                torch.nn.Linear(layer_sizes[1], output_size),
            ]
        )

    def forward(self, x):
        return custom_ops_lib.mlp_custom(
            x,
            self.layers[0].weight,
            self.layers[0].bias,
            self.layers[1].weight,
            self.layers[1].bias,
            self.layers[2].weight,
            self.layers[2].bias,
        )
'''
