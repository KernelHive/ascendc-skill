project_json_src='''
[
    {
        "op": "ConvTransposed2dAsymmetricInputAsymmetricKernelCustom",
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
                "name": "stride_h",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "stride_w",
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
                "name": "output_padding_h",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "output_padding_w",
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
BEGIN_TILING_DATA_DEF(ConvTransposed2dAsymmetricInputAsymmetricKernelCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inChannels);
    TILING_DATA_FIELD_DEF(uint32_t, outChannels);
    TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, kernelHeight);
    TILING_DATA_FIELD_DEF(uint32_t, kernelWidth);
    TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, strideH);
    TILING_DATA_FIELD_DEF(uint32_t, strideW);
    TILING_DATA_FIELD_DEF(uint32_t, paddingH);
    TILING_DATA_FIELD_DEF(uint32_t, paddingW);
    TILING_DATA_FIELD_DEF(uint32_t, dilationH);
    TILING_DATA_FIELD_DEF(uint32_t, dilationW);
    TILING_DATA_FIELD_DEF(uint32_t, groups);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    ConvTransposed2dAsymmetricInputAsymmetricKernelCustom,
    ConvTransposed2dAsymmetricInputAsymmetricKernelCustomTilingData)
}
"""

host_operator_src="""
#include "conv_transposed2d_asymmetric_input_asymmetric_kernel_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
int64_t ComputeOutputDim(
    int64_t inputSize,
    int64_t kernelSize,
    int64_t stride,
    int64_t padding,
    int64_t outputPadding,
    int64_t dilation)
{
    if (inputSize <= 0 || kernelSize <= 0 || stride <= 0 || padding < 0 || outputPadding < 0 || dilation <= 0) {
        return -1;
    }
    if (outputPadding >= stride) {
        return -1;
    }
    return (inputSize - 1) * stride - 2 * padding + dilation * (kernelSize - 1) + outputPadding + 1;
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
    const int64_t *strideHPtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *strideWPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *paddingHPtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t *paddingWPtr = attrs->GetAttrPointer<int64_t>(3);
    const int64_t *outputPaddingHPtr = attrs->GetAttrPointer<int64_t>(4);
    const int64_t *outputPaddingWPtr = attrs->GetAttrPointer<int64_t>(5);
    const int64_t *dilationHPtr = attrs->GetAttrPointer<int64_t>(6);
    const int64_t *dilationWPtr = attrs->GetAttrPointer<int64_t>(7);
    const int64_t *groupsPtr = attrs->GetAttrPointer<int64_t>(8);
    if (strideHPtr == nullptr || strideWPtr == nullptr || paddingHPtr == nullptr || paddingWPtr == nullptr ||
        outputPaddingHPtr == nullptr || outputPaddingWPtr == nullptr || dilationHPtr == nullptr ||
        dilationWPtr == nullptr || groupsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t inChannels = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t inputHeight = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t inputWidth = static_cast<uint32_t>(xShape.GetDim(3));
    const uint32_t weightInChannels = static_cast<uint32_t>(wShape.GetDim(0));
    const uint32_t outChannelsPerGroup = static_cast<uint32_t>(wShape.GetDim(1));
    const uint32_t kernelHeight = static_cast<uint32_t>(wShape.GetDim(2));
    const uint32_t kernelWidth = static_cast<uint32_t>(wShape.GetDim(3));
    const uint32_t groups = static_cast<uint32_t>(*groupsPtr);

    if (groups == 0 || weightInChannels != inChannels || (inChannels % groups) != 0) {
        return ge::GRAPH_FAILED;
    }

    const int64_t outputHeight = ComputeOutputDim(
        inputHeight, kernelHeight, *strideHPtr, *paddingHPtr, *outputPaddingHPtr, *dilationHPtr);
    const int64_t outputWidth = ComputeOutputDim(
        inputWidth, kernelWidth, *strideWPtr, *paddingWPtr, *outputPaddingWPtr, *dilationWPtr);
    if (outputHeight <= 0 || outputWidth <= 0) {
        return ge::GRAPH_FAILED;
    }

    ConvTransposed2dAsymmetricInputAsymmetricKernelCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannelsPerGroup * groups);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_kernelHeight(kernelHeight);
    tiling.set_kernelWidth(kernelWidth);
    tiling.set_outputHeight(static_cast<uint32_t>(outputHeight));
    tiling.set_outputWidth(static_cast<uint32_t>(outputWidth));
    tiling.set_strideH(static_cast<uint32_t>(*strideHPtr));
    tiling.set_strideW(static_cast<uint32_t>(*strideWPtr));
    tiling.set_paddingH(static_cast<uint32_t>(*paddingHPtr));
    tiling.set_paddingW(static_cast<uint32_t>(*paddingWPtr));
    tiling.set_dilationH(static_cast<uint32_t>(*dilationHPtr));
    tiling.set_dilationW(static_cast<uint32_t>(*dilationWPtr));
    tiling.set_groups(groups);

    const uint32_t blockDim = batchSize == 0 ? 1 : batchSize * (outChannelsPerGroup * groups);
    context->SetBlockDim(blockDim);
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
    const int64_t *strideHPtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *strideWPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *paddingHPtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t *paddingWPtr = attrs->GetAttrPointer<int64_t>(3);
    const int64_t *outputPaddingHPtr = attrs->GetAttrPointer<int64_t>(4);
    const int64_t *outputPaddingWPtr = attrs->GetAttrPointer<int64_t>(5);
    const int64_t *dilationHPtr = attrs->GetAttrPointer<int64_t>(6);
    const int64_t *dilationWPtr = attrs->GetAttrPointer<int64_t>(7);
    const int64_t *groupsPtr = attrs->GetAttrPointer<int64_t>(8);
    if (strideHPtr == nullptr || strideWPtr == nullptr || paddingHPtr == nullptr || paddingWPtr == nullptr ||
        outputPaddingHPtr == nullptr || outputPaddingWPtr == nullptr || dilationHPtr == nullptr ||
        dilationWPtr == nullptr || groupsPtr == nullptr) {
        return GRAPH_FAILED;
    }

    const int64_t inChannels = inputShape->GetDim(1);
    const int64_t weightInChannels = weightShape->GetDim(0);
    const int64_t outChannelsPerGroup = weightShape->GetDim(1);
    const int64_t groups = *groupsPtr;
    if (groups <= 0 || inChannels != weightInChannels || (inChannels % groups) != 0) {
        return GRAPH_FAILED;
    }

    const int64_t outputHeight = ComputeOutputDim(
        inputShape->GetDim(2), weightShape->GetDim(2), *strideHPtr, *paddingHPtr, *outputPaddingHPtr, *dilationHPtr);
    const int64_t outputWidth = ComputeOutputDim(
        inputShape->GetDim(3), weightShape->GetDim(3), *strideWPtr, *paddingWPtr, *outputPaddingWPtr, *dilationWPtr);
    if (outputHeight <= 0 || outputWidth <= 0) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(4);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, outChannelsPerGroup * groups);
    outputShape->SetDim(2, outputHeight);
    outputShape->SetDim(3, outputWidth);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ConvTransposed2dAsymmetricInputAsymmetricKernelCustom : public OpDef {
public:
    explicit ConvTransposed2dAsymmetricInputAsymmetricKernelCustom(const char *name) : OpDef(name)
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

        this->Attr("stride_h").Int();
        this->Attr("stride_w").Int();
        this->Attr("padding_h").Int();
        this->Attr("padding_w").Int();
        this->Attr("output_padding_h").Int();
        this->Attr("output_padding_w").Int();
        this->Attr("dilation_h").Int();
        this->Attr("dilation_w").Int();
        this->Attr("groups").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(ConvTransposed2dAsymmetricInputAsymmetricKernelCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConvTransposed2dAsymmetricInputAsymmetricKernel {
public:
    __aicore__ inline KernelConvTransposed2dAsymmetricInputAsymmetricKernel() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
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
        uint32_t strideH,
        uint32_t strideW,
        uint32_t paddingH,
        uint32_t paddingW,
        uint32_t dilationH,
        uint32_t dilationW,
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
        this->strideH = strideH;
        this->strideW = strideW;
        this->paddingH = paddingH;
        this->paddingW = paddingW;
        this->dilationH = dilationH;
        this->dilationW = dilationW;
        this->groups = groups;
        this->outChannelsPerGroup = outChannels / groups;
        this->inChannelsPerGroup = inChannels / groups;
        this->blockIdx = GetBlockIdx();
        this->inputBatchStride = inChannels * inputHeight * inputWidth;
        this->outputBatchStride = outChannels * outputHeight * outputWidth;
        this->weightStride = outChannelsPerGroup * kernelHeight * kernelWidth;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * this->inputBatchStride);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, inChannels * this->weightStride);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * this->outputBatchStride);
    }

    __aicore__ inline void Process()
    {
        if (this->batchSize == 0 || this->outChannels == 0) {
            return;
        }

        const uint32_t totalBlocks = this->batchSize * this->outChannels;
        if (this->blockIdx >= totalBlocks) {
            return;
        }

        const uint32_t batchIdx = this->blockIdx / this->outChannels;
        const uint32_t outChannel = this->blockIdx % this->outChannels;
        const uint32_t groupIdx = outChannel / this->outChannelsPerGroup;
        const uint32_t outChannelInGroup = outChannel % this->outChannelsPerGroup;
        const uint32_t inChannelBegin = groupIdx * this->inChannelsPerGroup;
        const uint32_t inChannelEnd = inChannelBegin + this->inChannelsPerGroup;
        const uint32_t yChannelBase =
            batchIdx * this->outputBatchStride + outChannel * this->outputHeight * this->outputWidth;

        for (uint32_t outH = 0; outH < this->outputHeight; ++outH) {
            for (uint32_t outW = 0; outW < this->outputWidth; ++outW) {
                float sum = 0.0f;
                for (uint32_t inChannel = inChannelBegin; inChannel < inChannelEnd; ++inChannel) {
                    const uint32_t xChannelBase =
                        batchIdx * this->inputBatchStride + inChannel * this->inputHeight * this->inputWidth;
                    const uint32_t weightBase =
                        inChannel * this->weightStride +
                        outChannelInGroup * this->kernelHeight * this->kernelWidth;
                    for (uint32_t kernelH = 0; kernelH < this->kernelHeight; ++kernelH) {
                        const int32_t numeratorH =
                            static_cast<int32_t>(outH) + static_cast<int32_t>(this->paddingH) -
                            static_cast<int32_t>(kernelH * this->dilationH);
                        if (numeratorH < 0 || (numeratorH % static_cast<int32_t>(this->strideH)) != 0) {
                            continue;
                        }
                        const int32_t inH = numeratorH / static_cast<int32_t>(this->strideH);
                        if (inH < 0 || inH >= static_cast<int32_t>(this->inputHeight)) {
                            continue;
                        }
                        for (uint32_t kernelW = 0; kernelW < this->kernelWidth; ++kernelW) {
                            const int32_t numeratorW =
                                static_cast<int32_t>(outW) + static_cast<int32_t>(this->paddingW) -
                                static_cast<int32_t>(kernelW * this->dilationW);
                            if (numeratorW < 0 || (numeratorW % static_cast<int32_t>(this->strideW)) != 0) {
                                continue;
                            }
                            const int32_t inW = numeratorW / static_cast<int32_t>(this->strideW);
                            if (inW < 0 || inW >= static_cast<int32_t>(this->inputWidth)) {
                                continue;
                            }

                            const uint32_t xOffset =
                                xChannelBase + static_cast<uint32_t>(inH) * this->inputWidth +
                                static_cast<uint32_t>(inW);
                            const uint32_t weightOffset =
                                weightBase + kernelH * this->kernelWidth + kernelW;
                            sum += xGm.GetValue(xOffset) * weightGm.GetValue(weightOffset);
                        }
                    }
                }
                const uint32_t yOffset = yChannelBase + outH * this->outputWidth + outW;
                yGm.SetValue(yOffset, sum);
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
    uint32_t kernelHeight;
    uint32_t kernelWidth;
    uint32_t outputHeight;
    uint32_t outputWidth;
    uint32_t strideH;
    uint32_t strideW;
    uint32_t paddingH;
    uint32_t paddingW;
    uint32_t dilationH;
    uint32_t dilationW;
    uint32_t groups;
    uint32_t outChannelsPerGroup;
    uint32_t inChannelsPerGroup;
    uint32_t blockIdx;
    uint32_t inputBatchStride;
    uint32_t outputBatchStride;
    uint32_t weightStride;
};

extern "C" __global__ __aicore__ void conv_transposed2d_asymmetric_input_asymmetric_kernel_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTransposed2dAsymmetricInputAsymmetricKernel op;
    op.Init(
        x,
        weight,
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
        tiling_data.strideH,
        tiling_data.strideW,
        tiling_data.paddingH,
        tiling_data.paddingW,
        tiling_data.dilationH,
        tiling_data.dilationW,
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

at::Tensor conv_transposed2d_asymmetric_input_asymmetric_kernel_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    int64_t strideH,
    int64_t strideW,
    int64_t paddingH,
    int64_t paddingW,
    int64_t outputPaddingH,
    int64_t outputPaddingW,
    int64_t dilationH,
    int64_t dilationW,
    int64_t groups)
{
    TORCH_CHECK(x.dim() == 4, "conv_transposed2d custom expects x to be 4D");
    TORCH_CHECK(weight.dim() == 4, "conv_transposed2d custom expects weight to be 4D");
    TORCH_CHECK(strideH > 0 && strideW > 0, "stride must be positive");
    TORCH_CHECK(paddingH >= 0 && paddingW >= 0, "padding must be non-negative");
    TORCH_CHECK(outputPaddingH >= 0 && outputPaddingW >= 0, "output_padding must be non-negative");
    TORCH_CHECK(dilationH > 0 && dilationW > 0, "dilation must be positive");
    TORCH_CHECK(outputPaddingH < strideH, "output_padding_h must be smaller than stride_h");
    TORCH_CHECK(outputPaddingW < strideW, "output_padding_w must be smaller than stride_w");
    TORCH_CHECK(groups > 0, "groups must be positive");
    TORCH_CHECK(
        x.size(1) == weight.size(0),
        "input channels must equal weight.size(0) for ConvTranspose2d");
    TORCH_CHECK(
        x.size(1) % groups == 0,
        "input channels must be divisible by groups");

    const int64_t outputChannels = weight.size(1) * groups;
    const int64_t outputHeight =
        (x.size(2) - 1) * strideH - 2 * paddingH + dilationH * (weight.size(2) - 1) + outputPaddingH + 1;
    const int64_t outputWidth =
        (x.size(3) - 1) * strideW - 2 * paddingW + dilationW * (weight.size(3) - 1) + outputPaddingW + 1;
    TORCH_CHECK(outputHeight > 0 && outputWidth > 0, "computed output shape must be positive");

    at::Tensor result = at::empty({x.size(0), outputChannels, outputHeight, outputWidth}, x.options());
    at::Tensor bias;
    std::vector<int64_t> strideVec = {strideH, strideW};
    std::vector<int64_t> paddingVec = {paddingH, paddingW};
    std::vector<int64_t> dilationVec = {dilationH, dilationW};
    std::vector<int64_t> outputPaddingVec = {outputPaddingH, outputPaddingW};
    at::IntArrayRef strides(strideVec);
    at::IntArrayRef paddings(paddingVec);
    at::IntArrayRef dilations(dilationVec);
    at::IntArrayRef outputPaddingRef(outputPaddingVec);
    const bool transposed = true;
    const int8_t cubeMathType = 0;

    EXEC_NPU_CMD(
        aclnnConvolution,
        x,
        weight,
        bias,
        strides,
        paddings,
        dilations,
        transposed,
        outputPaddingRef,
        groups,
        result,
        cubeMathType);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "conv_transposed2d_asymmetric_input_asymmetric_kernel_custom",
        &conv_transposed2d_asymmetric_input_asymmetric_kernel_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv_transposed2d_asymmetric_input_asymmetric_kernel_custom",
        &conv_transposed2d_asymmetric_input_asymmetric_kernel_custom_impl_npu,
        "conv_transposed2d_asymmetric_input_asymmetric_kernel_custom");
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
        kernel_size: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        output_padding: tuple[int, int] = (0, 0),
        dilation: tuple[int, int] = (1, 1),
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if bias:
            raise ValueError("This AscendC implementation currently supports bias=False only.")

        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.conv_transpose2d = torch.nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv_transposed2d_asymmetric_input_asymmetric_kernel_custom(
            x,
            self.conv_transpose2d.weight,
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.output_padding[0],
            self.output_padding[1],
            self.dilation[0],
            self.dilation[1],
            self.groups,
        )
'''
