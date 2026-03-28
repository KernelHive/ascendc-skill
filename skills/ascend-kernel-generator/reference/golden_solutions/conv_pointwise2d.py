project_json_src='''
[
    {
        "op": "ConvPointwise2dCustom",
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
BEGIN_TILING_DATA_DEF(ConvPointwise2dCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inChannels);
    TILING_DATA_FIELD_DEF(uint32_t, outChannels);
    TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvPointwise2dCustom, ConvPointwise2dCustomTilingData)
}
"""

host_operator_src="""
#include "conv_pointwise2d_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *inputShape = context->GetInputShape(0);
    const gert::StorageShape *weightShape = context->GetInputShape(1);
    if (inputShape == nullptr || weightShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xShape = inputShape->GetStorageShape();
    const auto wShape = weightShape->GetStorageShape();
    if (xShape.GetDimNum() != 4 || wShape.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }
    if (wShape.GetDim(2) != 1 || wShape.GetDim(3) != 1) {
        return ge::GRAPH_FAILED;
    }
    if (xShape.GetDim(1) != wShape.GetDim(1)) {
        return ge::GRAPH_FAILED;
    }

    ConvPointwise2dCustomTilingData tiling;
    tiling.set_batchSize(static_cast<uint32_t>(xShape.GetDim(0)));
    tiling.set_inChannels(static_cast<uint32_t>(xShape.GetDim(1)));
    tiling.set_outChannels(static_cast<uint32_t>(wShape.GetDim(0)));
    tiling.set_inputHeight(static_cast<uint32_t>(xShape.GetDim(2)));
    tiling.set_inputWidth(static_cast<uint32_t>(xShape.GetDim(3)));

    const uint32_t batchSize = tiling.get_batchSize();
    context->SetBlockDim(batchSize == 0 ? 1 : batchSize);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *workspaceSize = context->GetWorkspaceSizes(1);
    workspaceSize[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *inputShape = context->GetInputShape(0);
    const gert::Shape *weightShape = context->GetInputShape(1);
    if (inputShape == nullptr || weightShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDimNum() != 4 || weightShape->GetDimNum() != 4) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(4);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, weightShape->GetDim(0));
    outputShape->SetDim(2, inputShape->GetDim(2));
    outputShape->SetDim(3, inputShape->GetDim(3));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ConvPointwise2dCustom : public OpDef {
public:
    explicit ConvPointwise2dCustom(const char *name) : OpDef(name)
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
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(ConvPointwise2dCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConvPointwise2d {
public:
    __aicore__ inline KernelConvPointwise2d() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t inChannels,
        uint32_t outChannels,
        uint32_t inputHeight,
        uint32_t inputWidth)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->inputHeight = inputHeight;
        this->inputWidth = inputWidth;
        this->blockIdx = GetBlockIdx();
        this->inputPlaneSize = inputHeight * inputWidth;
        this->inputBatchStride = inChannels * this->inputPlaneSize;
        this->outputBatchStride = outChannels * this->inputPlaneSize;
        this->weightStride = inChannels;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * this->inputBatchStride);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, outChannels * inChannels);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * this->outputBatchStride);
    }

    __aicore__ inline void Process()
    {
        if (this->blockIdx >= this->batchSize) {
            return;
        }

        const uint32_t inputBatchBase = this->blockIdx * this->inputBatchStride;
        const uint32_t outputBatchBase = this->blockIdx * this->outputBatchStride;
        for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
            const uint32_t weightBase = outChannel * this->weightStride;
            const uint32_t outputChannelBase = outputBatchBase + outChannel * this->inputPlaneSize;
            for (uint32_t h = 0; h < this->inputHeight; ++h) {
                for (uint32_t w = 0; w < this->inputWidth; ++w) {
                    const uint32_t planeOffset = h * this->inputWidth + w;
                    float sum = 0.0f;
                    for (uint32_t inChannel = 0; inChannel < this->inChannels; ++inChannel) {
                        const uint32_t inputOffset =
                            inputBatchBase + inChannel * this->inputPlaneSize + planeOffset;
                        sum += xGm.GetValue(inputOffset) * weightGm.GetValue(weightBase + inChannel);
                    }
                    yGm.SetValue(outputChannelBase + planeOffset, sum);
                }
            }
        }
    }

private:
    GlobalTensor<float> xGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t blockIdx;
    uint32_t inputPlaneSize;
    uint32_t inputBatchStride;
    uint32_t outputBatchStride;
    uint32_t weightStride;
};

extern "C" __global__ __aicore__ void conv_pointwise2d_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvPointwise2d op;
    op.Init(
        x,
        weight,
        y,
        tiling_data.batchSize,
        tiling_data.inChannels,
        tiling_data.outChannels,
        tiling_data.inputHeight,
        tiling_data.inputWidth);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor conv_pointwise2d_custom_impl_npu(const at::Tensor &x, const at::Tensor &weight)
{
    TORCH_CHECK(x.dim() == 4, "conv_pointwise2d_custom expects x to be 4D NCHW");
    TORCH_CHECK(weight.dim() == 4, "conv_pointwise2d_custom expects weight to be 4D");
    TORCH_CHECK(weight.size(2) == 1 && weight.size(3) == 1, "weight must be 1x1");
    TORCH_CHECK(x.size(1) == weight.size(1), "input channels must match weight channels");

    auto outputShape = std::vector<int64_t>{x.size(0), weight.size(0), x.size(2), x.size(3)};
    at::Tensor result = at::empty(outputShape, x.options());
    EXEC_NPU_CMD(aclnnConvPointwise2dCustom, x, weight, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_pointwise2d_custom", &conv_pointwise2d_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_pointwise2d_custom", &conv_pointwise2d_custom_impl_npu, "conv_pointwise2d_custom");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False) -> None:
        super().__init__()
        if bias:
            raise ValueError("This AscendC implementation currently supports bias=False only.")
        self.conv2d = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv_pointwise2d_custom(x, self.conv2d.weight)
'''
