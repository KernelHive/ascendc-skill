project_json_src='''
[
    {
        "op": "ConvTransposed2dSquareInputSquareKernelCustom",
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
BEGIN_TILING_DATA_DEF(ConvTransposed2dSquareInputSquareKernelCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inChannels);
    TILING_DATA_FIELD_DEF(uint32_t, outChannels);
    TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, kernelSize);
    TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, stride);
    TILING_DATA_FIELD_DEF(uint32_t, padding);
    TILING_DATA_FIELD_DEF(uint32_t, outputPadding);
    TILING_DATA_FIELD_DEF(uint32_t, groups);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    ConvTransposed2dSquareInputSquareKernelCustom,
    ConvTransposed2dSquareInputSquareKernelCustomTilingData)
}
"""

host_operator_src="""
#include "conv_transposed2d_square_input_square_kernel_custom_tiling.h"
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
    const int64_t *outputPaddingPtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t *groupsPtr = attrs->GetAttrPointer<int64_t>(3);
    if (stridePtr == nullptr || paddingPtr == nullptr || outputPaddingPtr == nullptr || groupsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t inChannels = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t inputHeight = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t inputWidth = static_cast<uint32_t>(xShape.GetDim(3));
    const uint32_t weightInputChannels = static_cast<uint32_t>(wShape.GetDim(0));
    const uint32_t groupOutChannels = static_cast<uint32_t>(wShape.GetDim(1));
    const uint32_t kernelHeight = static_cast<uint32_t>(wShape.GetDim(2));
    const uint32_t kernelWidth = static_cast<uint32_t>(wShape.GetDim(3));

    if (inputHeight != inputWidth || kernelHeight != kernelWidth) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t stride = static_cast<uint32_t>(*stridePtr);
    const uint32_t padding = static_cast<uint32_t>(*paddingPtr);
    const uint32_t outputPadding = static_cast<uint32_t>(*outputPaddingPtr);
    const uint32_t groups = static_cast<uint32_t>(*groupsPtr);
    if (groups == 0 || stride == 0 || outputPadding >= stride) {
        return ge::GRAPH_FAILED;
    }
    if (inChannels != weightInputChannels || inChannels % groups != 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t outChannels = groupOutChannels * groups;

    ConvTransposed2dSquareInputSquareKernelCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_kernelSize(kernelHeight);
    tiling.set_outputHeight(
        ComputeTransposedOutputDim(inputHeight, kernelHeight, stride, padding, outputPadding));
    tiling.set_outputWidth(
        ComputeTransposedOutputDim(inputWidth, kernelWidth, stride, padding, outputPadding));
    tiling.set_stride(stride);
    tiling.set_padding(padding);
    tiling.set_outputPadding(outputPadding);
    tiling.set_groups(groups);

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
    const int64_t *outputPaddingPtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t *groupsPtr = attrs->GetAttrPointer<int64_t>(3);
    if (stridePtr == nullptr || paddingPtr == nullptr || outputPaddingPtr == nullptr || groupsPtr == nullptr) {
        return GRAPH_FAILED;
    }
    if (*stridePtr <= 0 || *paddingPtr < 0 || *outputPaddingPtr < 0 || *groupsPtr <= 0) {
        return GRAPH_FAILED;
    }
    if (*outputPaddingPtr >= *stridePtr) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDim(2) != inputShape->GetDim(3) || weightShape->GetDim(2) != weightShape->GetDim(3)) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(4);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, weightShape->GetDim(1) * (*groupsPtr));
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
class ConvTransposed2dSquareInputSquareKernelCustom : public OpDef {
public:
    explicit ConvTransposed2dSquareInputSquareKernelCustom(const char *name)
        : OpDef(name)
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
        this->Attr("output_padding").AttrType(REQUIRED).Int();
        this->Attr("groups").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvTransposed2dSquareInputSquareKernelCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConvTransposed2dSquareInputSquareKernel {
public:
    __aicore__ inline KernelConvTransposed2dSquareInputSquareKernel() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t inChannels,
        uint32_t outChannels,
        uint32_t inputHeight,
        uint32_t inputWidth,
        uint32_t kernelSize,
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
        this->kernelSize = kernelSize;
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
        this->weightInputStride = this->weightOutputChannelsPerGroup * kernelSize * kernelSize;
        this->weightOutputStride = kernelSize * kernelSize;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * this->inputBatchStride);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, inChannels * this->weightInputStride);
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
            const uint32_t groupIdx = outChannel / this->weightOutputChannelsPerGroup;
            const uint32_t groupOutChannel = outChannel % this->weightOutputChannelsPerGroup;
            const uint32_t inChannelStart = groupIdx * this->inputChannelsPerGroup;
            const uint32_t inChannelEnd = inChannelStart + this->inputChannelsPerGroup;
            const uint32_t yChannelBase = yBatchBase + outChannel * this->outputChannelStride;

            for (uint32_t outH = 0; outH < this->outputHeight; ++outH) {
                for (uint32_t outW = 0; outW < this->outputWidth; ++outW) {
                    float sum = 0.0f;
                    for (uint32_t inChannel = inChannelStart; inChannel < inChannelEnd; ++inChannel) {
                        const uint32_t xChannelBase = xBatchBase + inChannel * this->inputChannelStride;
                        const uint32_t wChannelBase =
                            inChannel * this->weightInputStride + groupOutChannel * this->weightOutputStride;
                        for (uint32_t kernelH = 0; kernelH < this->kernelSize; ++kernelH) {
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

                            for (uint32_t kernelW = 0; kernelW < this->kernelSize; ++kernelW) {
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
                                    wChannelBase + kernelH * this->kernelSize + kernelW;
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
    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t kernelSize;
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

extern "C" __global__ __aicore__ void conv_transposed2d_square_input_square_kernel_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTransposed2dSquareInputSquareKernel op;
    op.Init(
        x,
        weight,
        y,
        tiling_data.batchSize,
        tiling_data.inChannels,
        tiling_data.outChannels,
        tiling_data.inputHeight,
        tiling_data.inputWidth,
        tiling_data.kernelSize,
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
#include <vector>
#include "pytorch_npu_helper.hpp"

at::Tensor conv_transposed2d_square_input_square_kernel_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups)
{
    TORCH_CHECK(x.dim() == 4, "x must be a 4D NCHW tensor");
    TORCH_CHECK(weight.dim() == 4, "weight must be a 4D tensor");
    TORCH_CHECK(x.size(2) == x.size(3), "input must be square");
    TORCH_CHECK(weight.size(2) == weight.size(3), "kernel must be square");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(output_padding >= 0, "output padding must be non-negative");
    TORCH_CHECK(output_padding < stride, "output padding must be smaller than stride");
    TORCH_CHECK(groups > 0, "groups must be positive");
    TORCH_CHECK(x.size(1) == weight.size(0), "input channels must match weight.size(0)");
    TORCH_CHECK(x.size(1) % groups == 0, "input channels must be divisible by groups");

    const int64_t outputChannels = weight.size(1) * groups;
    const int64_t kernelSize = weight.size(2);
    const int64_t outH = (x.size(2) - 1) * stride - 2 * padding + kernelSize + output_padding;
    const int64_t outW = (x.size(3) - 1) * stride - 2 * padding + kernelSize + output_padding;
    TORCH_CHECK(outH > 0 && outW > 0, "invalid output shape");

    at::Tensor result = at::empty({x.size(0), outputChannels, outH, outW}, x.options());
    at::Tensor bias;
    std::vector<int64_t> strideVec = {stride, stride};
    std::vector<int64_t> paddingVec = {padding, padding};
    std::vector<int64_t> dilationVec = {1, 1};
    std::vector<int64_t> outputPaddingVec = {output_padding, output_padding};
    at::IntArrayRef strideArray(strideVec);
    at::IntArrayRef paddingArray(paddingVec);
    at::IntArrayRef dilationArray(dilationVec);
    at::IntArrayRef outputPaddingArray(outputPaddingVec);
    const bool transposed = true;
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
        "conv_transposed2d_square_input_square_kernel_custom",
        &conv_transposed2d_square_input_square_kernel_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv_transposed2d_square_input_square_kernel_custom",
        &conv_transposed2d_square_input_square_kernel_custom_impl_npu,
        "conv_transposed2d_square_input_square_kernel_custom");
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
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if bias:
            raise ValueError("This AscendC implementation currently supports bias=False only.")

        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.conv_transpose2d = torch.nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv_transposed2d_square_input_square_kernel_custom(
            x,
            self.conv_transpose2d.weight,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
        )
'''
