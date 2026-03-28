project_json_src='''
[
    {
        "op": "ConvTransposed1dAsymmetricInputSquareKernelPaddedStridedDilatedCustom",
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
BEGIN_TILING_DATA_DEF(ConvTransposed1dAsymmetricInputSquareKernelPaddedStridedDilatedCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inChannels);
    TILING_DATA_FIELD_DEF(uint32_t, outChannels);
    TILING_DATA_FIELD_DEF(uint32_t, inputLength);
    TILING_DATA_FIELD_DEF(uint32_t, kernelSize);
    TILING_DATA_FIELD_DEF(uint32_t, outputLength);
    TILING_DATA_FIELD_DEF(uint32_t, stride);
    TILING_DATA_FIELD_DEF(uint32_t, padding);
    TILING_DATA_FIELD_DEF(uint32_t, dilation);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    ConvTransposed1dAsymmetricInputSquareKernelPaddedStridedDilatedCustom,
    ConvTransposed1dAsymmetricInputSquareKernelPaddedStridedDilatedCustomTilingData)
}
"""

host_operator_src="""
#include "conv_transposed1d_asymmetric_input_square_kernel_padded_strided_dilated_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
constexpr uint32_t kStride = 2;
constexpr uint32_t kPadding = 1;
constexpr uint32_t kDilation = 2;
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

    const uint32_t batchSize = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t inChannels = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t inputLength = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t weightInChannels = static_cast<uint32_t>(wShape.GetDim(0));
    const uint32_t outChannels = static_cast<uint32_t>(wShape.GetDim(1));
    const uint32_t kernelSize = static_cast<uint32_t>(wShape.GetDim(2));

    if (kernelSize == 0 || inChannels != weightInChannels) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t outputLength =
        (inputLength - 1) * kStride - 2 * kPadding + (kernelSize - 1) * kDilation + 1;

    ConvTransposed1dAsymmetricInputSquareKernelPaddedStridedDilatedCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_inputLength(inputLength);
    tiling.set_kernelSize(kernelSize);
    tiling.set_outputLength(outputLength);
    tiling.set_stride(kStride);
    tiling.set_padding(kPadding);
    tiling.set_dilation(kDilation);

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
    if (kernelSize <= 0) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(3);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, weightShape->GetDim(1));
    outputShape->SetDim(
        2,
        (inputLength - 1) * static_cast<int64_t>(kStride)
            - 2 * static_cast<int64_t>(kPadding)
            + (kernelSize - 1) * static_cast<int64_t>(kDilation) + 1);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ConvTransposed1dAsymmetricInputSquareKernelPaddedStridedDilatedCustom : public OpDef {
public:
    explicit ConvTransposed1dAsymmetricInputSquareKernelPaddedStridedDilatedCustom(const char *name)
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

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(ConvTransposed1dAsymmetricInputSquareKernelPaddedStridedDilatedCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConvTransposed1dAsymmetricInputSquareKernelPaddedStridedDilated {
public:
    __aicore__ inline KernelConvTransposed1dAsymmetricInputSquareKernelPaddedStridedDilated() {}

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
        uint32_t dilation)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->inputLength = inputLength;
        this->kernelSize = kernelSize;
        this->outputLength = outputLength;
        this->stride = stride;
        this->padding = padding;
        this->dilation = dilation;
        this->blockIdx = GetBlockIdx();
        this->inputBatchStride = inChannels * inputLength;
        this->outputBatchStride = outChannels * outputLength;
        this->weightStride = outChannels * kernelSize;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * this->inputBatchStride);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, inChannels * this->weightStride);
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
            const uint32_t yChannelBase = yBatchBase + outChannel * this->outputLength;
            for (uint32_t outPos = 0; outPos < this->outputLength; ++outPos) {
                float sum = 0.0f;
                for (uint32_t inChannel = 0; inChannel < this->inChannels; ++inChannel) {
                    const uint32_t xChannelBase = xBatchBase + inChannel * this->inputLength;
                    const uint32_t wChannelBase = inChannel * this->weightStride + outChannel * this->kernelSize;
                    for (uint32_t kernelPos = 0; kernelPos < this->kernelSize; ++kernelPos) {
                        const int32_t numerator = static_cast<int32_t>(outPos) +
                            static_cast<int32_t>(this->padding) -
                            static_cast<int32_t>(kernelPos * this->dilation);
                        if (numerator < 0) {
                            continue;
                        }
                        if (numerator % static_cast<int32_t>(this->stride) != 0) {
                            continue;
                        }
                        const int32_t inputPos = numerator / static_cast<int32_t>(this->stride);
                        if (inputPos < 0 || inputPos >= static_cast<int32_t>(this->inputLength)) {
                            continue;
                        }
                        const float xValue = xGm.GetValue(xChannelBase + static_cast<uint32_t>(inputPos));
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
    uint32_t stride;
    uint32_t padding;
    uint32_t dilation;
    uint32_t blockIdx;
    uint32_t inputBatchStride;
    uint32_t outputBatchStride;
    uint32_t weightStride;
};

extern "C" __global__ __aicore__ void conv_transposed1d_asymmetric_input_square_kernel_padded_strided_dilated_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTransposed1dAsymmetricInputSquareKernelPaddedStridedDilated op;
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
        tiling_data.dilation);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

namespace {
constexpr int64_t kStride = 2;
constexpr int64_t kPadding = 1;
constexpr int64_t kDilation = 2;
}

at::Tensor conv_transposed1d_asymmetric_input_square_kernel_padded_strided_dilated_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight)
{
    TORCH_CHECK(x.dim() == 3, "conv_transposed1d custom expects x to be 3D");
    TORCH_CHECK(weight.dim() == 3, "conv_transposed1d custom expects weight to be 3D");
    TORCH_CHECK(x.size(1) == weight.size(0), "input channels must match weight.size(0)");

    const int64_t outputLength =
        (x.size(2) - 1) * kStride - 2 * kPadding + (weight.size(2) - 1) * kDilation + 1;
    TORCH_CHECK(outputLength > 0, "output length must be positive");

    auto outputShape = std::vector<int64_t>{x.size(0), weight.size(1), outputLength};
    at::Tensor result = at::empty(outputShape, x.options());
    EXEC_NPU_CMD(aclnnConvTransposed1dAsymmetricInputSquareKernelPaddedStridedDilatedCustom, x, weight, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "conv_transposed1d_asymmetric_input_square_kernel_padded_strided_dilated_custom",
        &conv_transposed1d_asymmetric_input_square_kernel_padded_strided_dilated_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv_transposed1d_asymmetric_input_square_kernel_padded_strided_dilated_custom",
        &conv_transposed1d_asymmetric_input_square_kernel_padded_strided_dilated_custom_impl_npu,
        "conv_transposed1d_asymmetric_input_square_kernel_padded_strided_dilated_custom");
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
        bias: bool = False,
    ) -> None:
        super().__init__()
        if stride != 2 or padding != 1 or dilation != 2 or bias:
            raise ValueError(
                "This AscendC implementation currently supports "
                "stride=2, padding=1, dilation=2, bias=False only."
            )
        self.conv1d_transpose = torch.nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv_transposed1d_asymmetric_input_square_kernel_padded_strided_dilated_custom(
            x, self.conv1d_transpose.weight)
'''
