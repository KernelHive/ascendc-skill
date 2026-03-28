project_json_src='''
[
    {
        "op": "ConvStandard1dCustom",
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
BEGIN_TILING_DATA_DEF(ConvStandard1dCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inChannels);
    TILING_DATA_FIELD_DEF(uint32_t, outChannels);
    TILING_DATA_FIELD_DEF(uint32_t, inputLength);
    TILING_DATA_FIELD_DEF(uint32_t, kernelSize);
    TILING_DATA_FIELD_DEF(uint32_t, outputLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvStandard1dCustom, ConvStandard1dCustomTilingData)
}
"""

host_operator_src="""
#include "conv_standard1d_custom_tiling.h"
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
    if (xShape.GetDimNum() != 3 || wShape.GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t inChannels = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t inputLength = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t outChannels = static_cast<uint32_t>(wShape.GetDim(0));
    const uint32_t weightInChannels = static_cast<uint32_t>(wShape.GetDim(1));
    const uint32_t kernelSize = static_cast<uint32_t>(wShape.GetDim(2));
    if (inChannels != weightInChannels || inputLength < kernelSize) {
        return ge::GRAPH_FAILED;
    }

    ConvStandard1dCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_inputLength(inputLength);
    tiling.set_kernelSize(kernelSize);
    tiling.set_outputLength(inputLength - kernelSize + 1);

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
    if (inputShape->GetDimNum() != 3 || weightShape->GetDimNum() != 3) {
        return GRAPH_FAILED;
    }

    const int64_t inputLength = inputShape->GetDim(2);
    const int64_t kernelSize = weightShape->GetDim(2);
    if (inputLength < kernelSize) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(3);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, weightShape->GetDim(0));
    outputShape->SetDim(2, inputLength - kernelSize + 1);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ConvStandard1dCustom : public OpDef {
public:
    explicit ConvStandard1dCustom(const char *name) : OpDef(name)
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

OP_ADD(ConvStandard1dCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConvStandard1d {
public:
    __aicore__ inline KernelConvStandard1d() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t inChannels,
        uint32_t outChannels,
        uint32_t inputLength,
        uint32_t kernelSize,
        uint32_t outputLength)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->inputLength = inputLength;
        this->kernelSize = kernelSize;
        this->outputLength = outputLength;
        this->blockIdx = GetBlockIdx();
        this->inputBatchStride = inChannels * inputLength;
        this->outputBatchStride = outChannels * outputLength;
        this->weightStride = inChannels * kernelSize;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * this->inputBatchStride);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, outChannels * this->weightStride);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * this->outputBatchStride);
    }

    __aicore__ inline void Process()
    {
        if (this->blockIdx >= this->batchSize) {
            return;
        }

        const uint32_t xBatchBase = this->blockIdx * this->inputBatchStride;
        const uint32_t yBatchBase = this->blockIdx * this->outputBatchStride;
        for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
            const uint32_t weightBase = outChannel * this->weightStride;
            const uint32_t yChannelBase = yBatchBase + outChannel * this->outputLength;
            for (uint32_t outPos = 0; outPos < this->outputLength; ++outPos) {
                float sum = 0.0f;
                for (uint32_t inChannel = 0; inChannel < this->inChannels; ++inChannel) {
                    const uint32_t xChannelBase = xBatchBase + inChannel * this->inputLength;
                    const uint32_t wChannelBase = weightBase + inChannel * this->kernelSize;
                    for (uint32_t kernelPos = 0; kernelPos < this->kernelSize; ++kernelPos) {
                        const float xValue = xGm.GetValue(xChannelBase + outPos + kernelPos);
                        const float wValue = weightGm.GetValue(wChannelBase + kernelPos);
                        sum += xValue * wValue;
                    }
                }
                yGm.SetValue(yChannelBase + outPos, sum);
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
    uint32_t inputLength;
    uint32_t kernelSize;
    uint32_t outputLength;
    uint32_t blockIdx;
    uint32_t inputBatchStride;
    uint32_t outputBatchStride;
    uint32_t weightStride;
};

extern "C" __global__ __aicore__ void conv_standard1d_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvStandard1d op;
    op.Init(
        x,
        weight,
        y,
        tiling_data.batchSize,
        tiling_data.inChannels,
        tiling_data.outChannels,
        tiling_data.inputLength,
        tiling_data.kernelSize,
        tiling_data.outputLength);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor conv_standard1d_custom_impl_npu(const at::Tensor &x, const at::Tensor &weight)
{
    TORCH_CHECK(x.dim() == 3, "conv_standard1d_custom expects x to be 3D");
    TORCH_CHECK(weight.dim() == 3, "conv_standard1d_custom expects weight to be 3D");
    TORCH_CHECK(x.size(1) == weight.size(1), "input channels must match weight channels");
    TORCH_CHECK(x.size(2) >= weight.size(2), "input length must be >= kernel size");

    const int64_t outputLength = x.size(2) - weight.size(2) + 1;
    auto outputShape = std::vector<int64_t>{x.size(0), weight.size(0), outputLength};
    at::Tensor result = at::empty(outputShape, x.options());
    EXEC_NPU_CMD(aclnnConvStandard1dCustom, x, weight, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_standard1d_custom", &conv_standard1d_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_standard1d_custom", &conv_standard1d_custom_impl_npu, "conv_standard1d_custom");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if stride != 1 or padding != 0 or dilation != 1 or groups != 1 or bias:
            raise ValueError(
                "This AscendC implementation currently supports "
                "stride=1, padding=0, dilation=1, groups=1, bias=False only."
            )
        self.conv1d = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv_standard1d_custom(x, self.conv1d.weight)
'''
