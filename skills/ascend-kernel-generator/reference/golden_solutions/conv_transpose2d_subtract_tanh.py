project_json_src='''
[
    {
        "op": "ConvTranspose2dSubtractTanhCustom",
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
                "name": "conv_bias",
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
                "name": "output_padding",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "groups",
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
BEGIN_TILING_DATA_DEF(ConvTranspose2dSubtractTanhCustomTilingData)
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
    TILING_DATA_FIELD_DEF(uint32_t, outputPadding);
    TILING_DATA_FIELD_DEF(uint32_t, groups);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    ConvTranspose2dSubtractTanhCustom,
    ConvTranspose2dSubtractTanhCustomTilingData)
}
"""

host_operator_src="""
#include "conv_transpose2d_subtract_tanh_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
uint32_t ComputeTransposedOutputDim(
    int64_t input,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    int64_t outputPadding)
{
    if (stride <= 0) {
        return 0;
    }
    const int64_t output = (input - 1) * stride - 2 * padding + kernel + outputPadding;
    return output > 0 ? static_cast<uint32_t>(output) : 0;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *inputShape = context->GetInputShape(0);
    const gert::StorageShape *weightShape = context->GetInputShape(1);
    const gert::StorageShape *convBiasShape = context->GetInputShape(2);
    const gert::StorageShape *biasShape = context->GetInputShape(3);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (inputShape == nullptr || weightShape == nullptr || convBiasShape == nullptr ||
        biasShape == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xShape = inputShape->GetStorageShape();
    const auto wShape = weightShape->GetStorageShape();
    const auto convBShape = convBiasShape->GetStorageShape();
    const auto bShape = biasShape->GetStorageShape();
    if (xShape.GetDimNum() != 4 || wShape.GetDimNum() != 4 || convBShape.GetDimNum() < 1 || bShape.GetDimNum() < 1) {
        return ge::GRAPH_FAILED;
    }

    const int64_t *stridePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *paddingPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *outputPaddingPtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t *groupsPtr = attrs->GetAttrPointer<int64_t>(3);
    if (stridePtr == nullptr || paddingPtr == nullptr || outputPaddingPtr == nullptr || groupsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t groups = static_cast<uint32_t>(*groupsPtr);
    if (groups == 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t inChannels = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t weightInputChannels = static_cast<uint32_t>(wShape.GetDim(0));
    const uint32_t outChannels = static_cast<uint32_t>(wShape.GetDim(1)) * groups;
    if (inChannels != weightInputChannels || inChannels % groups != 0 || outChannels == 0) {
        return ge::GRAPH_FAILED;
    }
    if (static_cast<uint32_t>(convBShape.GetDim(0)) != outChannels ||
        static_cast<uint32_t>(bShape.GetDim(0)) != outChannels) {
        return ge::GRAPH_FAILED;
    }

    ConvTranspose2dSubtractTanhCustomTilingData tiling;
    tiling.set_batchSize(static_cast<uint32_t>(xShape.GetDim(0)));
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_inputHeight(static_cast<uint32_t>(xShape.GetDim(2)));
    tiling.set_inputWidth(static_cast<uint32_t>(xShape.GetDim(3)));
    tiling.set_kernelHeight(static_cast<uint32_t>(wShape.GetDim(2)));
    tiling.set_kernelWidth(static_cast<uint32_t>(wShape.GetDim(3)));
    tiling.set_outputHeight(
        ComputeTransposedOutputDim(xShape.GetDim(2), wShape.GetDim(2), *stridePtr, *paddingPtr, *outputPaddingPtr));
    tiling.set_outputWidth(
        ComputeTransposedOutputDim(xShape.GetDim(3), wShape.GetDim(3), *stridePtr, *paddingPtr, *outputPaddingPtr));
    tiling.set_stride(static_cast<uint32_t>(*stridePtr));
    tiling.set_padding(static_cast<uint32_t>(*paddingPtr));
    tiling.set_outputPadding(static_cast<uint32_t>(*outputPaddingPtr));
    tiling.set_groups(groups);

    context->SetBlockDim(tiling.get_batchSize() == 0 ? 1 : tiling.get_batchSize());
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
    const gert::Shape *convBiasShape = context->GetInputShape(2);
    const gert::Shape *biasShape = context->GetInputShape(3);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (inputShape == nullptr || weightShape == nullptr || convBiasShape == nullptr ||
        biasShape == nullptr || attrs == nullptr) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDimNum() != 4 || weightShape->GetDimNum() != 4 ||
        convBiasShape->GetDimNum() < 1 || biasShape->GetDimNum() < 1) {
        return GRAPH_FAILED;
    }

    const int64_t *stridePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *paddingPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *outputPaddingPtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t *groupsPtr = attrs->GetAttrPointer<int64_t>(3);
    if (stridePtr == nullptr || paddingPtr == nullptr || outputPaddingPtr == nullptr || groupsPtr == nullptr) {
        return GRAPH_FAILED;
    }
    if (*groupsPtr <= 0) {
        return GRAPH_FAILED;
    }

    const int64_t outChannels = weightShape->GetDim(1) * (*groupsPtr);
    if (convBiasShape->GetDim(0) != outChannels || biasShape->GetDim(0) != outChannels) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(4);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, outChannels);
    outputShape->SetDim(
        2,
        ComputeTransposedOutputDim(
            inputShape->GetDim(2),
            weightShape->GetDim(2),
            *stridePtr,
            *paddingPtr,
            *outputPaddingPtr));
    outputShape->SetDim(
        3,
        ComputeTransposedOutputDim(
            inputShape->GetDim(3),
            weightShape->GetDim(3),
            *stridePtr,
            *paddingPtr,
            *outputPaddingPtr));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ConvTranspose2dSubtractTanhCustom : public OpDef {
public:
    explicit ConvTranspose2dSubtractTanhCustom(const char *name) : OpDef(name)
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
        this->Input("conv_bias")
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
        this->Attr("output_padding").AttrType(REQUIRED).Int();
        this->Attr("groups").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvTranspose2dSubtractTanhCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConvTranspose2dSubtractTanh {
public:
    __aicore__ inline KernelConvTranspose2dSubtractTanh() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR convBias,
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
        uint32_t groups)
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
        this->groups = groups;
        this->blockIdx = GetBlockIdx();
        this->inputChannelStride = inputHeight * inputWidth;
        this->outputChannelStride = outputHeight * outputWidth;
        this->inputBatchStride = inChannels * this->inputChannelStride;
        this->outputBatchStride = outChannels * this->outputChannelStride;
        this->weightOutputChannelsPerGroup = outChannels / groups;
        this->inputChannelsPerGroup = inChannels / groups;
        this->weightInputStride = this->weightOutputChannelsPerGroup * kernelHeight * kernelWidth;
        this->weightOutputStride = kernelHeight * kernelWidth;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * this->inputBatchStride);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, inChannels * this->weightInputStride);
        convBiasGm.SetGlobalBuffer((__gm__ float *)convBias, outChannels);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, outChannels);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * this->outputBatchStride);
        pipe.InitBuffer(tanhInputBuffer, sizeof(float));
        pipe.InitBuffer(tanhOutputBuffer, sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->blockIdx >= this->batchSize) {
            return;
        }

        const uint32_t xBatchBase = this->blockIdx * this->inputBatchStride;
        const uint32_t yBatchBase = this->blockIdx * this->outputBatchStride;

        for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
            const uint32_t groupIdx = outChannel / this->weightOutputChannelsPerGroup;
            const uint32_t groupOutChannel = outChannel % this->weightOutputChannelsPerGroup;
            const uint32_t inChannelStart = groupIdx * this->inputChannelsPerGroup;
            const uint32_t inChannelEnd = inChannelStart + this->inputChannelsPerGroup;
            const uint32_t yChannelBase = yBatchBase + outChannel * this->outputChannelStride;
            const float convBias = convBiasGm.GetValue(outChannel);
            const float channelBias = biasGm.GetValue(outChannel);

            for (uint32_t outH = 0; outH < this->outputHeight; ++outH) {
                for (uint32_t outW = 0; outW < this->outputWidth; ++outW) {
                    float sum = convBias;
                    for (uint32_t inChannel = inChannelStart; inChannel < inChannelEnd; ++inChannel) {
                        const uint32_t xChannelBase = xBatchBase + inChannel * this->inputChannelStride;
                        const uint32_t wChannelBase =
                            inChannel * this->weightInputStride + groupOutChannel * this->weightOutputStride;
                        for (uint32_t kernelH = 0; kernelH < this->kernelHeight; ++kernelH) {
                            const int32_t numerH =
                                static_cast<int32_t>(outH) + static_cast<int32_t>(this->padding) -
                                static_cast<int32_t>(kernelH);
                            if (numerH < 0 || numerH % static_cast<int32_t>(this->stride) != 0) {
                                continue;
                            }
                            const int32_t inH = numerH / static_cast<int32_t>(this->stride);
                            if (inH < 0 || inH >= static_cast<int32_t>(this->inputHeight)) {
                                continue;
                            }
                            for (uint32_t kernelW = 0; kernelW < this->kernelWidth; ++kernelW) {
                                const int32_t numerW =
                                    static_cast<int32_t>(outW) + static_cast<int32_t>(this->padding) -
                                    static_cast<int32_t>(kernelW);
                                if (numerW < 0 || numerW % static_cast<int32_t>(this->stride) != 0) {
                                    continue;
                                }
                                const int32_t inW = numerW / static_cast<int32_t>(this->stride);
                                if (inW < 0 || inW >= static_cast<int32_t>(this->inputWidth)) {
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
                    const float shifted = sum - channelBias;
                    AscendC::LocalTensor<float> tanhInput = tanhInputBuffer.Get<float>();
                    AscendC::LocalTensor<float> tanhOutput = tanhOutputBuffer.Get<float>();
                    tanhInput.SetValue(0, shifted);
                    AscendC::Tanh(tanhOutput, tanhInput, 1);
                    yGm.SetValue(yChannelBase + outH * this->outputWidth + outW, tanhOutput.GetValue(0));
                }
            }
        }
    }

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> tanhInputBuffer;
    TBuf<QuePosition::VECCALC> tanhOutputBuffer;
    GlobalTensor<float> xGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> convBiasGm;
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
    uint32_t groups;
    uint32_t blockIdx;
    uint32_t inputChannelStride;
    uint32_t outputChannelStride;
    uint32_t inputBatchStride;
    uint32_t outputBatchStride;
    uint32_t weightInputStride;
    uint32_t weightOutputStride;
    uint32_t weightOutputChannelsPerGroup;
    uint32_t inputChannelsPerGroup;
};

extern "C" __global__ __aicore__ void conv_transpose2d_subtract_tanh_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR conv_bias,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTranspose2dSubtractTanh op;
    op.Init(
        x,
        weight,
        conv_bias,
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
        tiling_data.groups);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include "pytorch_npu_helper.hpp"

at::Tensor conv_transpose2d_subtract_tanh_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &conv_bias,
    const at::Tensor &bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups)
{
    TORCH_CHECK(x.dim() == 4, "x must be a 4D NCHW tensor");
    TORCH_CHECK(weight.dim() == 4, "weight must be a 4D tensor");
    TORCH_CHECK(conv_bias.dim() == 1, "conv_bias must be a 1D tensor");
    TORCH_CHECK(bias.dim() >= 1, "bias must have at least one dimension");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(output_padding >= 0, "output_padding must be non-negative");
    TORCH_CHECK(groups > 0, "groups must be positive");
    TORCH_CHECK(output_padding < stride, "output_padding must be smaller than stride");
    TORCH_CHECK(x.size(1) == weight.size(0), "input channels must equal weight.size(0)");
    TORCH_CHECK(x.size(1) % groups == 0, "input channels must be divisible by groups");

    const int64_t outChannels = weight.size(1) * groups;
    TORCH_CHECK(conv_bias.size(0) == outChannels, "conv_bias.size(0) must equal output channels");
    TORCH_CHECK(bias.size(0) == outChannels, "bias.size(0) must equal output channels");

    const int64_t outH = (x.size(2) - 1) * stride - 2 * padding + weight.size(2) + output_padding;
    const int64_t outW = (x.size(3) - 1) * stride - 2 * padding + weight.size(3) + output_padding;
    TORCH_CHECK(outH > 0 && outW > 0, "invalid output shape");

    at::Tensor result = at::empty({x.size(0), outChannels, outH, outW}, x.options());
    EXEC_NPU_CMD(
        aclnnConvTranspose2dSubtractTanhCustom,
        x,
        weight,
        conv_bias,
        bias,
        stride,
        padding,
        output_padding,
        groups,
        result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "conv_transpose2d_subtract_tanh_custom",
        &conv_transpose2d_subtract_tanh_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv_transpose2d_subtract_tanh_custom",
        &conv_transpose2d_subtract_tanh_custom_impl_npu,
        "conv_transpose2d_subtract_tanh_custom");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias_shape,
        stride=2,
        padding=1,
        output_padding=1,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.bias = torch.nn.Parameter(torch.randn(bias_shape))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        return custom_ops_lib.conv_transpose2d_subtract_tanh_custom(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.conv_transpose.groups,
        )
'''
