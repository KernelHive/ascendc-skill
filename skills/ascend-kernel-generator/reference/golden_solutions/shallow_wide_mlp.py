project_json_src='''
[
    {
        "op": "ShallowWideMlpCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "x",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "w1",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "b1",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "w2",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "b2",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "w3",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "b3",
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
BEGIN_TILING_DATA_DEF(ShallowWideMlpCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, inputDim);
TILING_DATA_FIELD_DEF(uint32_t, hiddenDim1);
TILING_DATA_FIELD_DEF(uint32_t, hiddenDim2);
TILING_DATA_FIELD_DEF(uint32_t, outputDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ShallowWideMlpCustom, ShallowWideMlpCustomTilingData)
} // namespace optiling
"""

host_operator_src="""
#include "shallow_wide_mlp_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto shapeX = context->GetInputTensor(0)->GetOriginShape();
    auto shapeW1 = context->GetInputTensor(1)->GetOriginShape();
    auto shapeB1 = context->GetInputTensor(2)->GetOriginShape();
    auto shapeW2 = context->GetInputTensor(3)->GetOriginShape();
    auto shapeB2 = context->GetInputTensor(4)->GetOriginShape();
    auto shapeW3 = context->GetInputTensor(5)->GetOriginShape();
    auto shapeB3 = context->GetInputTensor(6)->GetOriginShape();

    if (shapeX.GetDimNum() != 2 || shapeW1.GetDimNum() != 2 || shapeB1.GetDimNum() != 1 ||
        shapeW2.GetDimNum() != 2 || shapeB2.GetDimNum() != 1 || shapeW3.GetDimNum() != 2 ||
        shapeB3.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const int32_t batchSize = shapeX.GetDim(0);
    const int32_t inputDim = shapeX.GetDim(1);
    const int32_t hiddenDim1 = shapeW1.GetDim(0);
    const int32_t hiddenDim2 = shapeW2.GetDim(0);
    const int32_t outputDim = shapeW3.GetDim(0);

    if (batchSize != 1 || inputDim != 1000 || hiddenDim1 != 2000 || hiddenDim2 != 2000 || outputDim != 10) {
        return ge::GRAPH_FAILED;
    }
    if (shapeW1.GetDim(1) != inputDim || shapeB1.GetDim(0) != hiddenDim1) {
        return ge::GRAPH_FAILED;
    }
    if (shapeW2.GetDim(1) != hiddenDim1 || shapeB2.GetDim(0) != hiddenDim2) {
        return ge::GRAPH_FAILED;
    }
    if (shapeW3.GetDim(1) != hiddenDim2 || shapeB3.GetDim(0) != outputDim) {
        return ge::GRAPH_FAILED;
    }

    ShallowWideMlpCustomTilingData tiling;
    tiling.set_batchSize(static_cast<uint32_t>(batchSize));
    tiling.set_inputDim(static_cast<uint32_t>(inputDim));
    tiling.set_hiddenDim1(static_cast<uint32_t>(hiddenDim1));
    tiling.set_hiddenDim2(static_cast<uint32_t>(hiddenDim2));
    tiling.set_outputDim(static_cast<uint32_t>(outputDim));

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
    const gert::Shape *w1Shape = context->GetInputShape(1);
    const gert::Shape *b1Shape = context->GetInputShape(2);
    const gert::Shape *w2Shape = context->GetInputShape(3);
    const gert::Shape *b2Shape = context->GetInputShape(4);
    const gert::Shape *w3Shape = context->GetInputShape(5);
    const gert::Shape *b3Shape = context->GetInputShape(6);
    if (xShape == nullptr || w1Shape == nullptr || b1Shape == nullptr || w2Shape == nullptr ||
        b2Shape == nullptr || w3Shape == nullptr || b3Shape == nullptr) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDimNum() != 2 || w1Shape->GetDimNum() != 2 || b1Shape->GetDimNum() != 1 ||
        w2Shape->GetDimNum() != 2 || b2Shape->GetDimNum() != 1 || w3Shape->GetDimNum() != 2 ||
        b3Shape->GetDimNum() != 1) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDim(1) != w1Shape->GetDim(1) || w1Shape->GetDim(0) != b1Shape->GetDim(0)) {
        return GRAPH_FAILED;
    }
    if (w2Shape->GetDim(1) != w1Shape->GetDim(0) || w2Shape->GetDim(0) != b2Shape->GetDim(0)) {
        return GRAPH_FAILED;
    }
    if (w3Shape->GetDim(1) != w2Shape->GetDim(0) || w3Shape->GetDim(0) != b3Shape->GetDim(0)) {
        return GRAPH_FAILED;
    }

    gert::Shape *outShape = context->GetOutputShape(0);
    outShape->SetDimNum(2);
    outShape->SetDim(0, xShape->GetDim(0));
    outShape->SetDim(1, w3Shape->GetDim(0));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class ShallowWideMlpCustom : public OpDef {
public:
    explicit ShallowWideMlpCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w1").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b1").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w2").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b2").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w3").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b3").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(ShallowWideMlpCustom);
} // namespace ops
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class ShallowWideMlpKernel {
public:
    __aicore__ inline ShallowWideMlpKernel() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR w1,
        GM_ADDR b1,
        GM_ADDR w2,
        GM_ADDR b2,
        GM_ADDR w3,
        GM_ADDR b3,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t inputDim,
        uint32_t hiddenDim1,
        uint32_t hiddenDim2,
        uint32_t outputDim)
    {
        this->batchSize = batchSize;
        this->inputDim = inputDim;
        this->hiddenDim1 = hiddenDim1;
        this->hiddenDim2 = hiddenDim2;
        this->outputDim = outputDim;

        xGm.SetGlobalBuffer((__gm__ float *)x, static_cast<uint64_t>(batchSize) * inputDim);
        w1Gm.SetGlobalBuffer((__gm__ float *)w1, static_cast<uint64_t>(hiddenDim1) * inputDim);
        b1Gm.SetGlobalBuffer((__gm__ float *)b1, hiddenDim1);
        w2Gm.SetGlobalBuffer((__gm__ float *)w2, static_cast<uint64_t>(hiddenDim2) * hiddenDim1);
        b2Gm.SetGlobalBuffer((__gm__ float *)b2, hiddenDim2);
        w3Gm.SetGlobalBuffer((__gm__ float *)w3, static_cast<uint64_t>(outputDim) * hiddenDim2);
        b3Gm.SetGlobalBuffer((__gm__ float *)b3, outputDim);
        yGm.SetGlobalBuffer((__gm__ float *)y, static_cast<uint64_t>(batchSize) * outputDim);
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() != 0) {
            return;
        }

        float hidden1[2000];
        float hidden2[2000];
        for (uint32_t batchIdx = 0; batchIdx < this->batchSize; ++batchIdx) {
            const uint64_t xBase = static_cast<uint64_t>(batchIdx) * this->inputDim;
            const uint64_t yBase = static_cast<uint64_t>(batchIdx) * this->outputDim;

            for (uint32_t out1 = 0; out1 < this->hiddenDim1; ++out1) {
                float acc1 = b1Gm.GetValue(out1);
                const uint64_t w1Base = static_cast<uint64_t>(out1) * this->inputDim;
                for (uint32_t inIdx = 0; inIdx < this->inputDim; ++inIdx) {
                    acc1 += xGm.GetValue(xBase + inIdx) * w1Gm.GetValue(w1Base + inIdx);
                }
                hidden1[out1] = acc1 > 0.0f ? acc1 : 0.0f;
            }

            for (uint32_t out2 = 0; out2 < this->hiddenDim2; ++out2) {
                float acc2 = b2Gm.GetValue(out2);
                const uint64_t w2Base = static_cast<uint64_t>(out2) * this->hiddenDim1;
                for (uint32_t out1 = 0; out1 < this->hiddenDim1; ++out1) {
                    acc2 += hidden1[out1] * w2Gm.GetValue(w2Base + out1);
                }
                hidden2[out2] = acc2 > 0.0f ? acc2 : 0.0f;
            }

            for (uint32_t out3 = 0; out3 < this->outputDim; ++out3) {
                float acc3 = b3Gm.GetValue(out3);
                const uint64_t w3Base = static_cast<uint64_t>(out3) * this->hiddenDim2;
                for (uint32_t out2 = 0; out2 < this->hiddenDim2; ++out2) {
                    acc3 += hidden2[out2] * w3Gm.GetValue(w3Base + out2);
                }
                yGm.SetValue(yBase + out3, acc3);
            }
        }
    }

private:
    GlobalTensor<float> xGm;
    GlobalTensor<float> w1Gm;
    GlobalTensor<float> b1Gm;
    GlobalTensor<float> w2Gm;
    GlobalTensor<float> b2Gm;
    GlobalTensor<float> w3Gm;
    GlobalTensor<float> b3Gm;
    GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t inputDim;
    uint32_t hiddenDim1;
    uint32_t hiddenDim2;
    uint32_t outputDim;
};

extern "C" __global__ __aicore__ void shallow_wide_mlp_custom(
    GM_ADDR x,
    GM_ADDR w1,
    GM_ADDR b1,
    GM_ADDR w2,
    GM_ADDR b2,
    GM_ADDR w3,
    GM_ADDR b3,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    ShallowWideMlpKernel op;
    op.Init(
        x,
        w1,
        b1,
        w2,
        b2,
        w3,
        b3,
        y,
        tilingData.batchSize,
        tilingData.inputDim,
        tilingData.hiddenDim1,
        tilingData.hiddenDim2,
        tilingData.outputDim);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor shallow_wide_mlp_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &w1,
    const at::Tensor &b1,
    const at::Tensor &w2,
    const at::Tensor &b2,
    const at::Tensor &w3,
    const at::Tensor &b3)
{
    at::Tensor result = at::empty({x.size(0), w3.size(0)}, x.options());
    EXEC_NPU_CMD(aclnnShallowWideMlpCustom, x, w1, b1, w2, b2, w3, b3, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("shallow_wide_mlp_custom", &shallow_wide_mlp_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("shallow_wide_mlp_custom", &shallow_wide_mlp_custom_impl_npu, "shallow wide mlp");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super().__init__()
        if len(hidden_layer_sizes) != 2:
            raise ValueError("This custom kernel expects exactly two hidden layers.")
        if input_size != 1000 or hidden_layer_sizes[0] != 2000 or hidden_layer_sizes[1] != 2000 or output_size != 10:
            raise ValueError("This custom kernel is specialized for 1000->2000->2000->10.")

        self.linear1 = torch.nn.Linear(input_size, hidden_layer_sizes[0])
        self.linear2 = torch.nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1])
        self.linear3 = torch.nn.Linear(hidden_layer_sizes[1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.shallow_wide_mlp_custom(
            x,
            self.linear1.weight,
            self.linear1.bias,
            self.linear2.weight,
            self.linear2.bias,
            self.linear3.weight,
            self.linear3.bias,
        )
'''
