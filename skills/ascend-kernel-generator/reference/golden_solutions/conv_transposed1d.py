project_json_src='''
[
    {
        "op": "ConvTransposed1dCustom",
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
BEGIN_TILING_DATA_DEF(ConvTransposed1dCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inChannels);
    TILING_DATA_FIELD_DEF(uint32_t, outChannels);
    TILING_DATA_FIELD_DEF(uint32_t, inputLength);
    TILING_DATA_FIELD_DEF(uint32_t, kernelSize);
    TILING_DATA_FIELD_DEF(uint32_t, outputLength);
    TILING_DATA_FIELD_DEF(uint32_t, stride);
    TILING_DATA_FIELD_DEF(uint32_t, padding);
    TILING_DATA_FIELD_DEF(uint32_t, outputPadding);
    TILING_DATA_FIELD_DEF(uint32_t, groups);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTransposed1dCustom, ConvTransposed1dCustomTilingData)
}
"""

host_operator_src="""
#include "conv_transposed1d_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
int64_t ComputeOutputLength(
    int64_t inputLength,
    int64_t kernelSize,
    int64_t stride,
    int64_t padding,
    int64_t outputPadding)
{
    if (inputLength <= 0 || kernelSize <= 0 || stride <= 0 || padding < 0 || outputPadding < 0) {
        return -1;
    }
    if (outputPadding >= stride) {
        return -1;
    }
    return (inputLength - 1) * stride - 2 * padding + kernelSize + outputPadding;
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
    if (xShape.GetDimNum() != 3 || wShape.GetDimNum() != 3) {
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
    const uint32_t inputLength = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t weightInChannels = static_cast<uint32_t>(wShape.GetDim(0));
    const uint32_t outChannelsPerGroup = static_cast<uint32_t>(wShape.GetDim(1));
    const uint32_t kernelSize = static_cast<uint32_t>(wShape.GetDim(2));
    const uint32_t groups = static_cast<uint32_t>(*groupsPtr);

    if (groups == 0 || weightInChannels != inChannels || (inChannels % groups) != 0) {
        return ge::GRAPH_FAILED;
    }

    const int64_t outputLength = ComputeOutputLength(
        inputLength, kernelSize, *stridePtr, *paddingPtr, *outputPaddingPtr);
    if (outputLength <= 0) {
        return ge::GRAPH_FAILED;
    }

    ConvTransposed1dCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannelsPerGroup * groups);
    tiling.set_inputLength(inputLength);
    tiling.set_kernelSize(kernelSize);
    tiling.set_outputLength(static_cast<uint32_t>(outputLength));
    tiling.set_stride(static_cast<uint32_t>(*stridePtr));
    tiling.set_padding(static_cast<uint32_t>(*paddingPtr));
    tiling.set_outputPadding(static_cast<uint32_t>(*outputPaddingPtr));
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
    if (inputShape->GetDimNum() != 3 || weightShape->GetDimNum() != 3) {
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

    const int64_t inChannels = inputShape->GetDim(1);
    const int64_t weightInChannels = weightShape->GetDim(0);
    const int64_t outChannelsPerGroup = weightShape->GetDim(1);
    const int64_t groups = *groupsPtr;
    if (groups <= 0 || inChannels != weightInChannels || (inChannels % groups) != 0) {
        return GRAPH_FAILED;
    }

    const int64_t outputLength = ComputeOutputLength(
        inputShape->GetDim(2), weightShape->GetDim(2), *stridePtr, *paddingPtr, *outputPaddingPtr);
    if (outputLength <= 0) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(3);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, outChannelsPerGroup * groups);
    outputShape->SetDim(2, outputLength);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ConvTransposed1dCustom : public OpDef {
public:
    explicit ConvTransposed1dCustom(const char *name) : OpDef(name)
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

        this->Attr("stride").Int();
        this->Attr("padding").Int();
        this->Attr("output_padding").Int();
        this->Attr("groups").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(ConvTransposed1dCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConvTransposed1d {
public:
    __aicore__ inline KernelConvTransposed1d() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t inChannels,
        uint32_t outChannels,
        uint32_t inputLength,
        uint32_t kernelSize,
        uint32_t outputLength,
        uint32_t stride,
        uint32_t padding,
        uint32_t groups)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->inputLength = inputLength;
        this->kernelSize = kernelSize;
        this->outputLength = outputLength;
        this->stride = stride;
        this->padding = padding;
        this->groups = groups;
        this->outChannelsPerGroup = outChannels / groups;
        this->inChannelsPerGroup = inChannels / groups;
        this->blockIdx = GetBlockIdx();
        this->inputBatchStride = inChannels * inputLength;
        this->outputBatchStride = outChannels * outputLength;
        this->weightStride = outChannelsPerGroup * kernelSize;

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
            batchIdx * this->outputBatchStride + outChannel * this->outputLength;

        for (uint32_t outPos = 0; outPos < this->outputLength; ++outPos) {
            float sum = 0.0f;
            for (uint32_t inChannel = inChannelBegin; inChannel < inChannelEnd; ++inChannel) {
                const uint32_t xChannelBase =
                    batchIdx * this->inputBatchStride + inChannel * this->inputLength;
                const uint32_t weightBase =
                    inChannel * this->weightStride + outChannelInGroup * this->kernelSize;
                for (uint32_t kernelPos = 0; kernelPos < this->kernelSize; ++kernelPos) {
                    const int32_t numerator =
                        static_cast<int32_t>(outPos) + static_cast<int32_t>(this->padding) -
                        static_cast<int32_t>(kernelPos);
                    if (numerator < 0 || (numerator % static_cast<int32_t>(this->stride)) != 0) {
                        continue;
                    }
                    const int32_t inputPos = numerator / static_cast<int32_t>(this->stride);
                    if (inputPos < 0 || inputPos >= static_cast<int32_t>(this->inputLength)) {
                        continue;
                    }

                    const float xValue = xGm.GetValue(xChannelBase + static_cast<uint32_t>(inputPos));
                    const float wValue = weightGm.GetValue(weightBase + kernelPos);
                    sum += xValue * wValue;
                }
            }
            yGm.SetValue(yChannelBase + outPos, sum);
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
    uint32_t stride;
    uint32_t padding;
    uint32_t groups;
    uint32_t outChannelsPerGroup;
    uint32_t inChannelsPerGroup;
    uint32_t blockIdx;
    uint32_t inputBatchStride;
    uint32_t outputBatchStride;
    uint32_t weightStride;
};

extern "C" __global__ __aicore__ void conv_transposed1d_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTransposed1d op;
    op.Init(
        x,
        weight,
        y,
        tiling_data.batchSize,
        tiling_data.inChannels,
        tiling_data.outChannels,
        tiling_data.inputLength,
        tiling_data.kernelSize,
        tiling_data.outputLength,
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

at::Tensor conv_transposed1d_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    int64_t stride,
    int64_t padding,
    int64_t outputPadding,
    int64_t groups)
{
    TORCH_CHECK(x.dim() == 3, "conv_transposed1d_custom expects x to be 3D");
    TORCH_CHECK(weight.dim() == 3, "conv_transposed1d_custom expects weight to be 3D");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(outputPadding >= 0, "output_padding must be non-negative");
    TORCH_CHECK(outputPadding < stride, "output_padding must be smaller than stride");
    TORCH_CHECK(groups > 0, "groups must be positive");
    TORCH_CHECK(
        x.size(1) == weight.size(0),
        "input channels must equal weight.size(0) for ConvTranspose1d");
    TORCH_CHECK(
        x.size(1) % groups == 0,
        "input channels must be divisible by groups");

    const int64_t outputChannels = weight.size(1) * groups;
    const int64_t outputLength =
        (x.size(2) - 1) * stride - 2 * padding + weight.size(2) + outputPadding;
    TORCH_CHECK(outputLength > 0, "computed output length must be positive");

    // Route 1D transpose convolution through the stable 2D ACLNN path with H=1.
    at::Tensor x2d = x.unsqueeze(2);
    at::Tensor weight2d = weight.unsqueeze(2);
    at::Tensor result2d = at::empty({x.size(0), outputChannels, 1, outputLength}, x.options());

    at::Tensor bias;
    std::vector<int64_t> strideVec = {1, stride};
    std::vector<int64_t> paddingVec = {0, padding};
    std::vector<int64_t> dilationVec = {1, 1};
    const bool transposed = true;
    std::vector<int64_t> outputPaddingVec = {0, outputPadding};
    at::IntArrayRef strides(strideVec);
    at::IntArrayRef paddings(paddingVec);
    at::IntArrayRef dilations(dilationVec);
    at::IntArrayRef outputPaddingRef(outputPaddingVec);
    const int8_t cubeMathType = 0;

    EXEC_NPU_CMD(
        aclnnConvolution,
        x2d,
        weight2d,
        bias,
        strides,
        paddings,
        dilations,
        transposed,
        outputPaddingRef,
        groups,
        result2d,
        cubeMathType);
    return result2d.squeeze(2);
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_transposed1d_custom", &conv_transposed1d_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv_transposed1d_custom",
        &conv_transposed1d_custom_impl_npu,
        "conv_transposed1d_custom");
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
        self.conv1d_transpose = torch.nn.ConvTranspose1d(
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
        return custom_ops_lib.conv_transposed1d_custom(
            x,
            self.conv1d_transpose.weight,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
        )
'''
