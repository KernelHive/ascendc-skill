project_json_src='''
[
    {
        "op": "Conv2dReluHardSwishCustom",
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
        ],
        "attr": [
            {
                "name": "stride",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "padding",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "dilation",
                "param_type": "required",
                "type": "int"
            }
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Conv2dReluHardSwishCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inChannels);
    TILING_DATA_FIELD_DEF(uint32_t, outChannels);
    TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, kernelHeight);
    TILING_DATA_FIELD_DEF(uint32_t, kernelWidth);
    TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, stride);
    TILING_DATA_FIELD_DEF(uint32_t, padding);
    TILING_DATA_FIELD_DEF(uint32_t, dilation);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv2dReluHardSwishCustom, Conv2dReluHardSwishCustomTilingData)
}
"""

host_operator_src="""
#include "conv2d_relu_hard_swish_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
uint32_t ComputeOutputDim(int64_t input, int64_t kernelSize, int64_t stride, int64_t padding, int64_t dilation)
{
    if (stride <= 0 || dilation <= 0 || kernelSize <= 0) {
        return 0;
    }
    const int64_t effectiveKernel = dilation * (kernelSize - 1) + 1;
    const int64_t numerator = input + padding * 2 - effectiveKernel;
    if (numerator < 0) {
        return 0;
    }
    return static_cast<uint32_t>(numerator / stride + 1);
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *inputShape = context->GetInputShape(0);
    const gert::StorageShape *weightShape = context->GetInputShape(1);
    const gert::StorageShape *biasShape = context->GetInputShape(2);
    if (inputShape == nullptr || weightShape == nullptr || biasShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xShape = inputShape->GetStorageShape();
    const auto wShape = weightShape->GetStorageShape();
    const auto bShape = biasShape->GetStorageShape();
    if (xShape.GetDimNum() != 4 || wShape.GetDimNum() != 4 || bShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t *stridePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *paddingPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *dilationPtr = attrs->GetAttrPointer<int64_t>(2);
    if (stridePtr == nullptr || paddingPtr == nullptr || dilationPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t inChannels = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t inputHeight = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t inputWidth = static_cast<uint32_t>(xShape.GetDim(3));
    const uint32_t outChannels = static_cast<uint32_t>(wShape.GetDim(0));
    const uint32_t weightInChannels = static_cast<uint32_t>(wShape.GetDim(1));
    const uint32_t kernelHeight = static_cast<uint32_t>(wShape.GetDim(2));
    const uint32_t kernelWidth = static_cast<uint32_t>(wShape.GetDim(3));
    const uint32_t biasLength = static_cast<uint32_t>(bShape.GetDim(0));

    if (inChannels != weightInChannels || biasLength != outChannels) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t stride = static_cast<uint32_t>(*stridePtr);
    const uint32_t padding = static_cast<uint32_t>(*paddingPtr);
    const uint32_t dilation = static_cast<uint32_t>(*dilationPtr);
    const uint32_t outputHeight = ComputeOutputDim(inputHeight, kernelHeight, stride, padding, dilation);
    const uint32_t outputWidth = ComputeOutputDim(inputWidth, kernelWidth, stride, padding, dilation);

    Conv2dReluHardSwishCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_kernelHeight(kernelHeight);
    tiling.set_kernelWidth(kernelWidth);
    tiling.set_outputHeight(outputHeight);
    tiling.set_outputWidth(outputWidth);
    tiling.set_stride(stride);
    tiling.set_padding(padding);
    tiling.set_dilation(dilation);

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
    const gert::Shape *biasShape = context->GetInputShape(2);
    if (inputShape == nullptr || weightShape == nullptr || biasShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDimNum() != 4 || weightShape->GetDimNum() != 4 || biasShape->GetDimNum() != 1) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDim(1) != weightShape->GetDim(1) || biasShape->GetDim(0) != weightShape->GetDim(0)) {
        return GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return GRAPH_FAILED;
    }
    const int64_t *stridePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *paddingPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *dilationPtr = attrs->GetAttrPointer<int64_t>(2);
    if (stridePtr == nullptr || paddingPtr == nullptr || dilationPtr == nullptr) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(4);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, weightShape->GetDim(0));
    outputShape->SetDim(
        2,
        ComputeOutputDim(inputShape->GetDim(2), weightShape->GetDim(2), *stridePtr, *paddingPtr, *dilationPtr));
    outputShape->SetDim(
        3,
        ComputeOutputDim(inputShape->GetDim(3), weightShape->GetDim(3), *stridePtr, *paddingPtr, *dilationPtr));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Conv2dReluHardSwishCustom : public OpDef {
public:
    explicit Conv2dReluHardSwishCustom(const char *name) : OpDef(name)
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
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("stride").AttrType(REQUIRED).Int();
        this->Attr("padding").AttrType(REQUIRED).Int();
        this->Attr("dilation").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv2dReluHardSwishCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConv2dReluHardSwish {
public:
    __aicore__ inline KernelConv2dReluHardSwish() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t inChannels,
        uint32_t outChannels,
        uint32_t inputHeight,
        uint32_t inputWidth,
        uint32_t kernelHeight,
        uint32_t kernelWidth,
        uint32_t outputHeight,
        uint32_t outputWidth,
        uint32_t stride,
        uint32_t padding,
        uint32_t dilation)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->inputHeight = inputHeight;
        this->inputWidth = inputWidth;
        this->kernelHeight = kernelHeight;
        this->kernelWidth = kernelWidth;
        this->outputHeight = outputHeight;
        this->outputWidth = outputWidth;
        this->stride = stride;
        this->padding = padding;
        this->dilation = dilation;
        this->blockIdx = GetBlockIdx();
        this->inputChannelStride = inputHeight * inputWidth;
        this->outputChannelStride = outputHeight * outputWidth;
        this->inputBatchStride = inChannels * this->inputChannelStride;
        this->outputBatchStride = outChannels * this->outputChannelStride;
        this->weightOutStride = inChannels * kernelHeight * kernelWidth;
        this->weightInStride = kernelHeight * kernelWidth;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * this->inputBatchStride);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, outChannels * this->weightOutStride);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, outChannels);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * this->outputBatchStride);
    }

    __aicore__ inline float HardSwish(float value) const
    {
        float shifted = value + 3.0f;
        if (shifted < 0.0f) {
            shifted = 0.0f;
        }
        if (shifted > 6.0f) {
            shifted = 6.0f;
        }
        return value * shifted / 6.0f;
    }

    __aicore__ inline void Process()
    {
        if (this->blockIdx >= this->batchSize) {
            return;
        }

        const uint32_t xBatchBase = this->blockIdx * this->inputBatchStride;
        const uint32_t yBatchBase = this->blockIdx * this->outputBatchStride;
        const int32_t inputHeight = static_cast<int32_t>(this->inputHeight);
        const int32_t inputWidth = static_cast<int32_t>(this->inputWidth);

        for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
            const uint32_t weightBase = outChannel * this->weightOutStride;
            const uint32_t yChannelBase = yBatchBase + outChannel * this->outputChannelStride;
            const float biasValue = biasGm.GetValue(outChannel);
            for (uint32_t outH = 0; outH < this->outputHeight; ++outH) {
                const int32_t startH =
                    static_cast<int32_t>(outH) * static_cast<int32_t>(this->stride) -
                    static_cast<int32_t>(this->padding);
                for (uint32_t outW = 0; outW < this->outputWidth; ++outW) {
                    const int32_t startW =
                        static_cast<int32_t>(outW) * static_cast<int32_t>(this->stride) -
                        static_cast<int32_t>(this->padding);
                    float sum = biasValue;
                    for (uint32_t inChannel = 0; inChannel < this->inChannels; ++inChannel) {
                        const uint32_t xChannelBase = xBatchBase + inChannel * this->inputChannelStride;
                        const uint32_t wChannelBase = weightBase + inChannel * this->weightInStride;
                        for (uint32_t kernelH = 0; kernelH < this->kernelHeight; ++kernelH) {
                            const int32_t inH =
                                startH + static_cast<int32_t>(kernelH) * static_cast<int32_t>(this->dilation);
                            if (inH < 0 || inH >= inputHeight) {
                                continue;
                            }
                            for (uint32_t kernelW = 0; kernelW < this->kernelWidth; ++kernelW) {
                                const int32_t inW =
                                    startW + static_cast<int32_t>(kernelW) * static_cast<int32_t>(this->dilation);
                                if (inW < 0 || inW >= inputWidth) {
                                    continue;
                                }

                                const uint32_t xOffset =
                                    xChannelBase +
                                    static_cast<uint32_t>(inH) * this->inputWidth +
                                    static_cast<uint32_t>(inW);
                                const uint32_t wOffset =
                                    wChannelBase + kernelH * this->kernelWidth + kernelW;
                                sum += xGm.GetValue(xOffset) * weightGm.GetValue(wOffset);
                            }
                        }
                    }
                    if (sum < 0.0f) {
                        sum = 0.0f;
                    }
                    yGm.SetValue(yChannelBase + outH * this->outputWidth + outW, HardSwish(sum));
                }
            }
        }
    }

private:
    GlobalTensor<float> xGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> biasGm;
    GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t kernelHeight;
    uint32_t kernelWidth;
    uint32_t outputHeight;
    uint32_t outputWidth;
    uint32_t stride;
    uint32_t padding;
    uint32_t dilation;
    uint32_t blockIdx;
    uint32_t inputChannelStride;
    uint32_t outputChannelStride;
    uint32_t inputBatchStride;
    uint32_t outputBatchStride;
    uint32_t weightOutStride;
    uint32_t weightInStride;
};

extern "C" __global__ __aicore__ void conv2d_relu_hard_swish_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv2dReluHardSwish op;
    op.Init(
        x,
        weight,
        bias,
        y,
        tiling_data.batchSize,
        tiling_data.inChannels,
        tiling_data.outChannels,
        tiling_data.inputHeight,
        tiling_data.inputWidth,
        tiling_data.kernelHeight,
        tiling_data.kernelWidth,
        tiling_data.outputHeight,
        tiling_data.outputWidth,
        tiling_data.stride,
        tiling_data.padding,
        tiling_data.dilation);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor conv2d_relu_hard_swish_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation)
{
    TORCH_CHECK(x.dim() == 4, "x must be a 4D NCHW tensor");
    TORCH_CHECK(weight.dim() == 4, "weight must be a 4D tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
    TORCH_CHECK(x.size(1) == weight.size(1), "input channels must match weight.size(1)");
    TORCH_CHECK(weight.size(0) == bias.size(0), "bias length must match output channels");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(dilation > 0, "dilation must be positive");

    const int64_t effectiveKernelH = dilation * (weight.size(2) - 1) + 1;
    const int64_t effectiveKernelW = dilation * (weight.size(3) - 1) + 1;
    const int64_t outputHeight = (x.size(2) + padding * 2 - effectiveKernelH) / stride + 1;
    const int64_t outputWidth = (x.size(3) + padding * 2 - effectiveKernelW) / stride + 1;
    TORCH_CHECK(outputHeight >= 0 && outputWidth >= 0, "invalid output shape");

    at::Tensor result = at::empty({x.size(0), weight.size(0), outputHeight, outputWidth}, x.options());
    EXEC_NPU_CMD(aclnnConv2dReluHardSwishCustom, x, weight, bias, stride, padding, dilation, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv2d_relu_hard_swish_custom", &conv2d_relu_hard_swish_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv2d_relu_hard_swish_custom",
        &conv2d_relu_hard_swish_impl_npu,
        "conv2d_relu_hard_swish_custom");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.stride = 1
        self.padding = 0
        self.dilation = 1
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv2d_relu_hard_swish_custom(
            x,
            self.conv.weight,
            self.conv.bias,
            self.stride,
            self.padding,
            self.dilation,
        )
'''
