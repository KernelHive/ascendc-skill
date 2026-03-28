project_json_src='''
[
    {
        "op": "ConvStandard2dSquareInputAsymmetricKernelDilatedPaddedCustom",
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
        ],
        "attr": [
            {
                "name": "stride",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "padding_h",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "padding_w",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "dilation_h",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "dilation_w",
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
BEGIN_TILING_DATA_DEF(ConvStandard2dSquareInputAsymmetricKernelDilatedPaddedCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inChannels);
    TILING_DATA_FIELD_DEF(uint32_t, outChannels);
    TILING_DATA_FIELD_DEF(uint32_t, inputSize);
    TILING_DATA_FIELD_DEF(uint32_t, kernelHeight);
    TILING_DATA_FIELD_DEF(uint32_t, kernelWidth);
    TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, stride);
    TILING_DATA_FIELD_DEF(uint32_t, paddingH);
    TILING_DATA_FIELD_DEF(uint32_t, paddingW);
    TILING_DATA_FIELD_DEF(uint32_t, dilationH);
    TILING_DATA_FIELD_DEF(uint32_t, dilationW);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    ConvStandard2dSquareInputAsymmetricKernelDilatedPaddedCustom,
    ConvStandard2dSquareInputAsymmetricKernelDilatedPaddedCustomTilingData)
}
"""

host_operator_src="""
#include "conv_standard2d_square_input_asymmetric_kernel_dilated_padded_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
uint32_t ComputeOutputDim(int64_t input, int64_t kernel, int64_t stride, int64_t padding, int64_t dilation)
{
    if (stride <= 0 || dilation <= 0) {
        return 0;
    }
    const int64_t effectiveKernel = dilation * (kernel - 1) + 1;
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
    if (inputShape == nullptr || weightShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xShape = inputShape->GetStorageShape();
    const auto wShape = weightShape->GetStorageShape();
    if (xShape.GetDimNum() != 4 || wShape.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int64_t *stridePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *paddingHPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *paddingWPtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t *dilationHPtr = attrs->GetAttrPointer<int64_t>(3);
    const int64_t *dilationWPtr = attrs->GetAttrPointer<int64_t>(4);
    if (stridePtr == nullptr || paddingHPtr == nullptr || paddingWPtr == nullptr ||
        dilationHPtr == nullptr || dilationWPtr == nullptr) {
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

    if (inputHeight != inputWidth || inChannels != weightInChannels) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t stride = static_cast<uint32_t>(*stridePtr);
    const uint32_t paddingH = static_cast<uint32_t>(*paddingHPtr);
    const uint32_t paddingW = static_cast<uint32_t>(*paddingWPtr);
    const uint32_t dilationH = static_cast<uint32_t>(*dilationHPtr);
    const uint32_t dilationW = static_cast<uint32_t>(*dilationWPtr);

    ConvStandard2dSquareInputAsymmetricKernelDilatedPaddedCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_inputSize(inputHeight);
    tiling.set_kernelHeight(kernelHeight);
    tiling.set_kernelWidth(kernelWidth);
    tiling.set_outputHeight(ComputeOutputDim(inputHeight, kernelHeight, stride, paddingH, dilationH));
    tiling.set_outputWidth(ComputeOutputDim(inputWidth, kernelWidth, stride, paddingW, dilationW));
    tiling.set_stride(stride);
    tiling.set_paddingH(paddingH);
    tiling.set_paddingW(paddingW);
    tiling.set_dilationH(dilationH);
    tiling.set_dilationW(dilationW);

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
    if (inputShape->GetDim(2) != inputShape->GetDim(3)) {
        return GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int64_t *stridePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *paddingHPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *paddingWPtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t *dilationHPtr = attrs->GetAttrPointer<int64_t>(3);
    const int64_t *dilationWPtr = attrs->GetAttrPointer<int64_t>(4);
    if (stridePtr == nullptr || paddingHPtr == nullptr || paddingWPtr == nullptr ||
        dilationHPtr == nullptr || dilationWPtr == nullptr) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(4);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, weightShape->GetDim(0));
    outputShape->SetDim(
        2,
        ComputeOutputDim(
            inputShape->GetDim(2),
            weightShape->GetDim(2),
            *stridePtr,
            *paddingHPtr,
            *dilationHPtr));
    outputShape->SetDim(
        3,
        ComputeOutputDim(
            inputShape->GetDim(3),
            weightShape->GetDim(3),
            *stridePtr,
            *paddingWPtr,
            *dilationWPtr));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ConvStandard2dSquareInputAsymmetricKernelDilatedPaddedCustom : public OpDef {
public:
    explicit ConvStandard2dSquareInputAsymmetricKernelDilatedPaddedCustom(const char *name) : OpDef(name)
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
        this->Attr("stride").AttrType(REQUIRED).Int();
        this->Attr("padding_h").AttrType(REQUIRED).Int();
        this->Attr("padding_w").AttrType(REQUIRED).Int();
        this->Attr("dilation_h").AttrType(REQUIRED).Int();
        this->Attr("dilation_w").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvStandard2dSquareInputAsymmetricKernelDilatedPaddedCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConvStandard2dSquareInputAsymmetricKernelDilatedPadded {
public:
    __aicore__ inline KernelConvStandard2dSquareInputAsymmetricKernelDilatedPadded() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t inChannels,
        uint32_t outChannels,
        uint32_t inputSize,
        uint32_t kernelHeight,
        uint32_t kernelWidth,
        uint32_t outputHeight,
        uint32_t outputWidth,
        uint32_t stride,
        uint32_t paddingH,
        uint32_t paddingW,
        uint32_t dilationH,
        uint32_t dilationW)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->inputSize = inputSize;
        this->kernelHeight = kernelHeight;
        this->kernelWidth = kernelWidth;
        this->outputHeight = outputHeight;
        this->outputWidth = outputWidth;
        this->stride = stride;
        this->paddingH = paddingH;
        this->paddingW = paddingW;
        this->dilationH = dilationH;
        this->dilationW = dilationW;
        this->blockIdx = GetBlockIdx();
        this->inputChannelStride = inputSize * inputSize;
        this->outputChannelStride = outputHeight * outputWidth;
        this->inputBatchStride = inChannels * this->inputChannelStride;
        this->outputBatchStride = outChannels * this->outputChannelStride;
        this->weightOutStride = inChannels * kernelHeight * kernelWidth;
        this->weightInStride = kernelHeight * kernelWidth;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * this->inputBatchStride);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, outChannels * this->weightOutStride);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * this->outputBatchStride);
    }

    __aicore__ inline void Process()
    {
        if (this->blockIdx >= this->batchSize) {
            return;
        }

        const uint32_t xBatchBase = this->blockIdx * this->inputBatchStride;
        const uint32_t yBatchBase = this->blockIdx * this->outputBatchStride;
        const int32_t inputSize = static_cast<int32_t>(this->inputSize);

        for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
            const uint32_t weightBase = outChannel * this->weightOutStride;
            const uint32_t yChannelBase = yBatchBase + outChannel * this->outputChannelStride;
            for (uint32_t outH = 0; outH < this->outputHeight; ++outH) {
                const int32_t startH =
                    static_cast<int32_t>(outH) * static_cast<int32_t>(this->stride) -
                    static_cast<int32_t>(this->paddingH);
                for (uint32_t outW = 0; outW < this->outputWidth; ++outW) {
                    const int32_t startW =
                        static_cast<int32_t>(outW) * static_cast<int32_t>(this->stride) -
                        static_cast<int32_t>(this->paddingW);
                    float sum = 0.0f;
                    for (uint32_t inChannel = 0; inChannel < this->inChannels; ++inChannel) {
                        const uint32_t xChannelBase = xBatchBase + inChannel * this->inputChannelStride;
                        const uint32_t wChannelBase = weightBase + inChannel * this->weightInStride;
                        for (uint32_t kernelH = 0; kernelH < this->kernelHeight; ++kernelH) {
                            const int32_t inH =
                                startH + static_cast<int32_t>(kernelH) * static_cast<int32_t>(this->dilationH);
                            if (inH < 0 || inH >= inputSize) {
                                continue;
                            }
                            for (uint32_t kernelW = 0; kernelW < this->kernelWidth; ++kernelW) {
                                const int32_t inW =
                                    startW + static_cast<int32_t>(kernelW) * static_cast<int32_t>(this->dilationW);
                                if (inW < 0 || inW >= inputSize) {
                                    continue;
                                }

                                const uint32_t xOffset =
                                    xChannelBase +
                                    static_cast<uint32_t>(inH) * this->inputSize +
                                    static_cast<uint32_t>(inW);
                                const uint32_t wOffset =
                                    wChannelBase + kernelH * this->kernelWidth + kernelW;
                                sum += xGm.GetValue(xOffset) * weightGm.GetValue(wOffset);
                            }
                        }
                    }
                    yGm.SetValue(yChannelBase + outH * this->outputWidth + outW, sum);
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
    uint32_t inputSize;
    uint32_t kernelHeight;
    uint32_t kernelWidth;
    uint32_t outputHeight;
    uint32_t outputWidth;
    uint32_t stride;
    uint32_t paddingH;
    uint32_t paddingW;
    uint32_t dilationH;
    uint32_t dilationW;
    uint32_t blockIdx;
    uint32_t inputChannelStride;
    uint32_t outputChannelStride;
    uint32_t inputBatchStride;
    uint32_t outputBatchStride;
    uint32_t weightOutStride;
    uint32_t weightInStride;
};

extern "C" __global__ __aicore__ void conv_standard2d_square_input_asymmetric_kernel_dilated_padded_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvStandard2dSquareInputAsymmetricKernelDilatedPadded op;
    op.Init(
        x,
        weight,
        y,
        tiling_data.batchSize,
        tiling_data.inChannels,
        tiling_data.outChannels,
        tiling_data.inputSize,
        tiling_data.kernelHeight,
        tiling_data.kernelWidth,
        tiling_data.outputHeight,
        tiling_data.outputWidth,
        tiling_data.stride,
        tiling_data.paddingH,
        tiling_data.paddingW,
        tiling_data.dilationH,
        tiling_data.dilationW);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor conv_standard2d_square_input_asymmetric_kernel_dilated_padded_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    int64_t stride,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w)
{
    TORCH_CHECK(x.dim() == 4, "x must be a 4D NCHW tensor");
    TORCH_CHECK(weight.dim() == 4, "weight must be a 4D tensor");
    TORCH_CHECK(x.size(2) == x.size(3), "x must have square spatial dimensions");
    TORCH_CHECK(x.size(1) == weight.size(1), "input channels must match weight.size(1)");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding_h >= 0 && padding_w >= 0, "padding must be non-negative");
    TORCH_CHECK(dilation_h > 0 && dilation_w > 0, "dilation must be positive");

    const int64_t kernelH = weight.size(2);
    const int64_t kernelW = weight.size(3);
    const int64_t effectiveKernelH = dilation_h * (kernelH - 1) + 1;
    const int64_t effectiveKernelW = dilation_w * (kernelW - 1) + 1;
    const int64_t outH = (x.size(2) + padding_h * 2 - effectiveKernelH) / stride + 1;
    const int64_t outW = (x.size(3) + padding_w * 2 - effectiveKernelW) / stride + 1;
    TORCH_CHECK(outH >= 0 && outW >= 0, "invalid output shape");

    at::Tensor result = at::empty({x.size(0), weight.size(0), outH, outW}, x.options());
    c10::optional<at::Tensor> bias = c10::nullopt;
    const int64_t strideData[2] = {stride, stride};
    const int64_t paddingData[2] = {padding_h, padding_w};
    const int64_t dilationData[2] = {dilation_h, dilation_w};
    const int64_t outputPaddingData[2] = {0, 0};
    const at::IntArrayRef strideArray(strideData, 2);
    const at::IntArrayRef paddingArray(paddingData, 2);
    const at::IntArrayRef dilationArray(dilationData, 2);
    const at::IntArrayRef outputPaddingArray(outputPaddingData, 2);
    bool transposed = false;
    const int64_t groups = 1;
    const int8_t cubeMathType = 0;

    EXEC_NPU_CMD(
        aclnnConvolution,
        x,
        weight,
        bias,
        strideArray,
        paddingArray,
        dilationArray,
        transposed,
        outputPaddingArray,
        groups,
        result,
        cubeMathType);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "conv_standard2d_square_input_asymmetric_kernel_dilated_padded_custom",
        &conv_standard2d_square_input_asymmetric_kernel_dilated_padded_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv_standard2d_square_input_asymmetric_kernel_dilated_padded_custom",
        &conv_standard2d_square_input_asymmetric_kernel_dilated_padded_custom_impl_npu,
        "conv_standard2d_square_input_asymmetric_kernel_dilated_padded_custom");
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
        kernel_size: tuple,
        stride: int = 1,
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        bias: bool = False,
    ) -> None:
        super().__init__()
        if bias:
            raise ValueError("This AscendC implementation currently supports bias=False only.")

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.conv2d = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv_standard2d_square_input_asymmetric_kernel_dilated_padded_custom(
            x,
            self.conv2d.weight,
            self.stride,
            self.padding[0],
            self.padding[1],
            self.dilation[0],
            self.dilation[1],
        )
'''
