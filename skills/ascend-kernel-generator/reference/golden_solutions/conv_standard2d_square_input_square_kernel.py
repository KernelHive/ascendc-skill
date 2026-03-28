project_json_src='''
[
    {
        "op": "ConvStandard2dSquareInputSquareKernelCustom",
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
BEGIN_TILING_DATA_DEF(ConvStandard2dSquareInputSquareKernelCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inChannels);
    TILING_DATA_FIELD_DEF(uint32_t, outChannels);
    TILING_DATA_FIELD_DEF(uint32_t, inputSize);
    TILING_DATA_FIELD_DEF(uint32_t, kernelSize);
    TILING_DATA_FIELD_DEF(uint32_t, outputSize);
    TILING_DATA_FIELD_DEF(uint32_t, stride);
    TILING_DATA_FIELD_DEF(uint32_t, padding);
    TILING_DATA_FIELD_DEF(uint32_t, dilation);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    ConvStandard2dSquareInputSquareKernelCustom,
    ConvStandard2dSquareInputSquareKernelCustomTilingData)
}
"""

host_operator_src="""
#include "conv_standard2d_square_input_square_kernel_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
uint32_t ComputeOutputDim(int64_t input, int64_t kernelSize, int64_t stride, int64_t padding, int64_t dilation)
{
    if (stride <= 0 || dilation <= 0) {
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

    if (inputHeight != inputWidth || kernelHeight != kernelWidth || inChannels != weightInChannels) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t stride = static_cast<uint32_t>(*stridePtr);
    const uint32_t padding = static_cast<uint32_t>(*paddingPtr);
    const uint32_t dilation = static_cast<uint32_t>(*dilationPtr);
    const uint32_t outputSize = ComputeOutputDim(inputHeight, kernelHeight, stride, padding, dilation);

    ConvStandard2dSquareInputSquareKernelCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_inputSize(inputHeight);
    tiling.set_kernelSize(kernelHeight);
    tiling.set_outputSize(outputSize);
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
    if (inputShape == nullptr || weightShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDimNum() != 4 || weightShape->GetDimNum() != 4) {
        return GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int64_t *stridePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *paddingPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *dilationPtr = attrs->GetAttrPointer<int64_t>(2);
    if (stridePtr == nullptr || paddingPtr == nullptr || dilationPtr == nullptr) {
        return GRAPH_FAILED;
    }

    const int64_t inputHeight = inputShape->GetDim(2);
    const int64_t inputWidth = inputShape->GetDim(3);
    const int64_t kernelHeight = weightShape->GetDim(2);
    const int64_t kernelWidth = weightShape->GetDim(3);
    if (inputHeight != inputWidth || kernelHeight != kernelWidth) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(4);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, weightShape->GetDim(0));
    const int64_t outputSize = ComputeOutputDim(inputHeight, kernelHeight, *stridePtr, *paddingPtr, *dilationPtr);
    outputShape->SetDim(2, outputSize);
    outputShape->SetDim(3, outputSize);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ConvStandard2dSquareInputSquareKernelCustom : public OpDef {
public:
    explicit ConvStandard2dSquareInputSquareKernelCustom(const char *name) : OpDef(name)
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
        this->Attr("padding").AttrType(REQUIRED).Int();
        this->Attr("dilation").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvStandard2dSquareInputSquareKernelCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConvStandard2dSquareInputSquareKernel {
public:
    __aicore__ inline KernelConvStandard2dSquareInputSquareKernel() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t inChannels,
        uint32_t outChannels,
        uint32_t inputSize,
        uint32_t kernelSize,
        uint32_t outputSize,
        uint32_t stride,
        uint32_t padding,
        uint32_t dilation)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->inputSize = inputSize;
        this->kernelSize = kernelSize;
        this->outputSize = outputSize;
        this->stride = stride;
        this->padding = padding;
        this->dilation = dilation;
        this->blockIdx = GetBlockIdx();
        this->inputChannelStride = inputSize * inputSize;
        this->outputChannelStride = outputSize * outputSize;
        this->inputBatchStride = inChannels * this->inputChannelStride;
        this->outputBatchStride = outChannels * this->outputChannelStride;
        this->weightOutStride = inChannels * kernelSize * kernelSize;
        this->weightInStride = kernelSize * kernelSize;

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
            for (uint32_t outH = 0; outH < this->outputSize; ++outH) {
                const int32_t startH =
                    static_cast<int32_t>(outH) * static_cast<int32_t>(this->stride) -
                    static_cast<int32_t>(this->padding);
                for (uint32_t outW = 0; outW < this->outputSize; ++outW) {
                    const int32_t startW =
                        static_cast<int32_t>(outW) * static_cast<int32_t>(this->stride) -
                        static_cast<int32_t>(this->padding);
                    float sum = 0.0f;
                    for (uint32_t inChannel = 0; inChannel < this->inChannels; ++inChannel) {
                        const uint32_t xChannelBase = xBatchBase + inChannel * this->inputChannelStride;
                        const uint32_t wChannelBase = weightBase + inChannel * this->weightInStride;
                        for (uint32_t kernelH = 0; kernelH < this->kernelSize; ++kernelH) {
                            const int32_t inH =
                                startH + static_cast<int32_t>(kernelH) * static_cast<int32_t>(this->dilation);
                            if (inH < 0 || inH >= inputSize) {
                                continue;
                            }
                            for (uint32_t kernelW = 0; kernelW < this->kernelSize; ++kernelW) {
                                const int32_t inW =
                                    startW + static_cast<int32_t>(kernelW) * static_cast<int32_t>(this->dilation);
                                if (inW < 0 || inW >= inputSize) {
                                    continue;
                                }

                                const uint32_t xOffset =
                                    xChannelBase +
                                    static_cast<uint32_t>(inH) * this->inputSize +
                                    static_cast<uint32_t>(inW);
                                const uint32_t wOffset =
                                    wChannelBase + kernelH * this->kernelSize + kernelW;
                                sum += xGm.GetValue(xOffset) * weightGm.GetValue(wOffset);
                            }
                        }
                    }
                    yGm.SetValue(yChannelBase + outH * this->outputSize + outW, sum);
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
    uint32_t kernelSize;
    uint32_t outputSize;
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

extern "C" __global__ __aicore__ void conv_standard2d_square_input_square_kernel_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvStandard2dSquareInputSquareKernel op;
    op.Init(
        x,
        weight,
        y,
        tiling_data.batchSize,
        tiling_data.inChannels,
        tiling_data.outChannels,
        tiling_data.inputSize,
        tiling_data.kernelSize,
        tiling_data.outputSize,
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

at::Tensor conv_standard2d_square_input_square_kernel_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    int64_t stride,
    int64_t padding,
    int64_t dilation)
{
    TORCH_CHECK(x.dim() == 4, "x must be a 4D NCHW tensor");
    TORCH_CHECK(weight.dim() == 4, "weight must be a 4D tensor");
    TORCH_CHECK(x.size(2) == x.size(3), "x must have square spatial dimensions");
    TORCH_CHECK(weight.size(2) == weight.size(3), "weight must have a square kernel");
    TORCH_CHECK(x.size(1) == weight.size(1), "input channels must match weight.size(1)");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(dilation > 0, "dilation must be positive");

    const int64_t kernelSize = weight.size(2);
    const int64_t effectiveKernel = dilation * (kernelSize - 1) + 1;
    const int64_t outputSize = (x.size(2) + padding * 2 - effectiveKernel) / stride + 1;
    TORCH_CHECK(outputSize >= 0, "invalid output shape");

    at::Tensor result = at::empty({x.size(0), weight.size(0), outputSize, outputSize}, x.options());
    c10::optional<at::Tensor> bias = c10::nullopt;
    const int64_t strideData[2] = {stride, stride};
    const int64_t paddingData[2] = {padding, padding};
    const int64_t dilationData[2] = {dilation, dilation};
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
        "conv_standard2d_square_input_square_kernel_custom",
        &conv_standard2d_square_input_square_kernel_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv_standard2d_square_input_square_kernel_custom",
        &conv_standard2d_square_input_square_kernel_custom_impl_npu,
        "conv_standard2d_square_input_square_kernel_custom");
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
        if groups != 1 or bias:
            raise ValueError(
                "This AscendC implementation currently supports groups=1 and bias=False only."
            )

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
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv_standard2d_square_input_square_kernel_custom(
            x,
            self.conv2d.weight,
            self.stride,
            self.padding,
            self.dilation,
        )
'''
