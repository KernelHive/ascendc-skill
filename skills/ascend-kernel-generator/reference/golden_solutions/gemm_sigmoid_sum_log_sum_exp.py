project_json_src='''
[
    {
        "op": "GemmSigmoidSumLogSumExpCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "x",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "weight",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "bias",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            }
        ],
        "output_desc": [
            {
                "name": "y",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            }
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GemmSigmoidSumLogSumExpCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, inputSize);
TILING_DATA_FIELD_DEF(uint32_t, hiddenSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GemmSigmoidSumLogSumExpCustom, GemmSigmoidSumLogSumExpCustomTilingData)
} // namespace optiling
"""

host_operator_src="""
#include "gemm_sigmoid_sum_log_sum_exp_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *xShape = context->GetInputShape(0);
    const gert::StorageShape *weightShape = context->GetInputShape(1);
    const gert::StorageShape *biasShape = context->GetInputShape(2);
    if (xShape == nullptr || weightShape == nullptr || biasShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xStorageShape = xShape->GetStorageShape();
    const auto weightStorageShape = weightShape->GetStorageShape();
    const auto biasStorageShape = biasShape->GetStorageShape();
    if (xStorageShape.GetDimNum() != 2 || weightStorageShape.GetDimNum() != 2 || biasStorageShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const int64_t batchSize = xStorageShape.GetDim(0);
    const int64_t inputSize = xStorageShape.GetDim(1);
    const int64_t weightInput = weightStorageShape.GetDim(0);
    const int64_t hiddenSize = weightStorageShape.GetDim(1);
    if (batchSize <= 0 || inputSize <= 0 || hiddenSize <= 0) {
        return ge::GRAPH_FAILED;
    }
    if (inputSize != weightInput || biasStorageShape.GetDim(0) != hiddenSize) {
        return ge::GRAPH_FAILED;
    }

    GemmSigmoidSumLogSumExpCustomTilingData tiling;
    tiling.set_batchSize(static_cast<uint32_t>(batchSize));
    tiling.set_inputSize(static_cast<uint32_t>(inputSize));
    tiling.set_hiddenSize(static_cast<uint32_t>(hiddenSize));

    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
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
    yShape->SetDimNum(0);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class GemmSigmoidSumLogSumExpCustom : public OpDef {
public:
    explicit GemmSigmoidSumLogSumExpCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(GemmSigmoidSumLogSumExpCustom);
} // namespace ops
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class GemmSigmoidSumLogSumExpKernel {
public:
    static constexpr float LN2 = 0.69314718056f;

    __aicore__ inline GemmSigmoidSumLogSumExpKernel() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t inputSize,
        uint32_t hiddenSize)
    {
        this->batchSize = batchSize;
        this->inputSize = inputSize;
        this->hiddenSize = hiddenSize;

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x), static_cast<uint64_t>(batchSize) * inputSize);
        weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weight), static_cast<uint64_t>(inputSize) * hiddenSize);
        biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bias), hiddenSize);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y), 1);
    }

    __aicore__ inline void Process()
    {
        float runningMax = -3.40282347e+38f;
        float runningExpSum = 0.0f;

        for (uint32_t row = 0; row < this->batchSize; ++row) {
            const uint32_t xRowBase = row * this->inputSize;
            float rowSum = 0.0f;
            for (uint32_t col = 0; col < this->hiddenSize; ++col) {
                float acc = biasGm.GetValue(col);
                for (uint32_t kk = 0; kk < this->inputSize; ++kk) {
                    acc += xGm.GetValue(xRowBase + kk) * weightGm.GetValue(kk * this->hiddenSize + col);
                }
                rowSum += ApplySigmoid(acc);
            }

            if (row == 0 || rowSum > runningMax) {
                runningExpSum = (row == 0) ? 1.0f : runningExpSum * FastExp(runningMax - rowSum) + 1.0f;
                runningMax = rowSum;
            } else {
                runningExpSum += FastExp(rowSum - runningMax);
            }
        }

        yGm.SetValue(0, runningMax + FastLogPositive(runningExpSum));
    }

private:
    __aicore__ inline float ApplySigmoid(float x) const
    {
        if (x >= 8.0f) {
            return 0.99966466f;
        }
        if (x <= -8.0f) {
            return 0.00033535f;
        }
        return 1.0f / (1.0f + FastExp(-x));
    }

    __aicore__ inline float FastExp(float x) const
    {
        if (x < -20.0f) {
            return 0.0f;
        }
        int32_t k = 0;
        while (x > 0.5f * LN2) {
            x -= LN2;
            ++k;
        }
        while (x < -0.5f * LN2) {
            x += LN2;
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

    __aicore__ inline float FastLogPositive(float x) const
    {
        if (x <= 0.0f) {
            return -3.40282347e+38f;
        }
        int32_t k = 0;
        while (x > 2.0f) {
            x *= 0.5f;
            ++k;
        }
        while (x < 1.0f) {
            x *= 2.0f;
            --k;
        }
        const float y = (x - 1.0f) / (x + 1.0f);
        const float y2 = y * y;
        const float y3 = y2 * y;
        const float y5 = y3 * y2;
        const float y7 = y5 * y2;
        const float series = 2.0f * (y + y3 / 3.0f + y5 / 5.0f + y7 / 7.0f);
        return static_cast<float>(k) * LN2 + series;
    }

    GlobalTensor<float> xGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> biasGm;
    GlobalTensor<float> yGm;
    uint32_t batchSize = 0;
    uint32_t inputSize = 0;
    uint32_t hiddenSize = 0;
};

extern "C" __global__ __aicore__ void gemm_sigmoid_sum_log_sum_exp_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    GemmSigmoidSumLogSumExpKernel op;
    op.Init(x, weight, bias, y, tilingData.batchSize, tilingData.inputSize, tilingData.hiddenSize);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor gemm_sigmoid_sum_log_sum_exp_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &bias)
{
    at::Tensor result = at::empty({}, x.options());
    EXEC_NPU_CMD(aclnnGemmSigmoidSumLogSumExpCustom, x, weight, bias, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("gemm_sigmoid_sum_log_sum_exp_custom", &gemm_sigmoid_sum_log_sum_exp_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "gemm_sigmoid_sum_log_sum_exp_custom",
        &gemm_sigmoid_sum_log_sum_exp_custom_impl_npu,
        "gemm + sigmoid + sum + logsumexp custom");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        linear1 = torch.nn.Linear(input_size, hidden_size)
        self.weight = torch.nn.Parameter(linear1.weight.transpose(0, 1).contiguous())
        self.bias = torch.nn.Parameter(linear1.bias.detach().clone())
        self._unused_linear2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return custom_ops_lib.gemm_sigmoid_sum_log_sum_exp_custom(x, self.weight, self.bias)
'''
