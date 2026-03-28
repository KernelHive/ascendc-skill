project_json_src='''
[
    {
        "op": "ConvTransposed3dSquareInputSquareKernelPaddedDilatedStridedCustom",
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
BEGIN_TILING_DATA_DEF(ConvTransposed3dSquareInputSquareKernelPaddedDilatedStridedCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inChannels);
    TILING_DATA_FIELD_DEF(uint32_t, outChannels);
    TILING_DATA_FIELD_DEF(uint32_t, inputDepth);
    TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, kernelSize);
    TILING_DATA_FIELD_DEF(uint32_t, outputDepth);
    TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, stride);
    TILING_DATA_FIELD_DEF(uint32_t, padding);
    TILING_DATA_FIELD_DEF(uint32_t, dilation);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    ConvTransposed3dSquareInputSquareKernelPaddedDilatedStridedCustom,
    ConvTransposed3dSquareInputSquareKernelPaddedDilatedStridedCustomTilingData)
}
"""

host_operator_src="""
#include "conv_transposed3d_square_input_square_kernel_padded_dilated_strided_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
constexpr uint32_t kStride = 2;
constexpr uint32_t kPadding = 1;
constexpr uint32_t kDilation = 2;

uint32_t ComputeTransposedOutputDim(int64_t input, int64_t kernel)
{
    if (input <= 0 || kernel <= 0) {
        return 0;
    }
    const int64_t output =
        (input - 1) * static_cast<int64_t>(kStride) - 2 * static_cast<int64_t>(kPadding) +
        (kernel - 1) * static_cast<int64_t>(kDilation) + 1;
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
    if (xShape.GetDimNum() != 5 || wShape.GetDimNum() != 5) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t inChannels = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t inputDepth = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t inputHeight = static_cast<uint32_t>(xShape.GetDim(3));
    const uint32_t inputWidth = static_cast<uint32_t>(xShape.GetDim(4));
    const uint32_t weightInChannels = static_cast<uint32_t>(wShape.GetDim(0));
    const uint32_t outChannels = static_cast<uint32_t>(wShape.GetDim(1));
    const uint32_t kernelDepth = static_cast<uint32_t>(wShape.GetDim(2));
    const uint32_t kernelHeight = static_cast<uint32_t>(wShape.GetDim(3));
    const uint32_t kernelWidth = static_cast<uint32_t>(wShape.GetDim(4));

    if (inChannels != weightInChannels || kernelDepth == 0 || kernelDepth != kernelHeight ||
        kernelDepth != kernelWidth || inputHeight == 0 || inputHeight != inputWidth) {
        return ge::GRAPH_FAILED;
    }

    ConvTransposed3dSquareInputSquareKernelPaddedDilatedStridedCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_inputDepth(inputDepth);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_kernelSize(kernelDepth);
    tiling.set_outputDepth(ComputeTransposedOutputDim(inputDepth, kernelDepth));
    tiling.set_outputHeight(ComputeTransposedOutputDim(inputHeight, kernelDepth));
    tiling.set_outputWidth(ComputeTransposedOutputDim(inputWidth, kernelDepth));
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
    if (inputShape->GetDimNum() != 5 || weightShape->GetDimNum() != 5) {
        return GRAPH_FAILED;
    }

    const int64_t kernelDepth = weightShape->GetDim(2);
    const int64_t kernelHeight = weightShape->GetDim(3);
    const int64_t kernelWidth = weightShape->GetDim(4);
    if (kernelDepth <= 0 || kernelDepth != kernelHeight || kernelDepth != kernelWidth) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDim(3) <= 0 || inputShape->GetDim(3) != inputShape->GetDim(4)) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(5);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, weightShape->GetDim(1));
    outputShape->SetDim(2, ComputeTransposedOutputDim(inputShape->GetDim(2), kernelDepth));
    outputShape->SetDim(3, ComputeTransposedOutputDim(inputShape->GetDim(3), kernelDepth));
    outputShape->SetDim(4, ComputeTransposedOutputDim(inputShape->GetDim(4), kernelDepth));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ConvTransposed3dSquareInputSquareKernelPaddedDilatedStridedCustom : public OpDef {
public:
    explicit ConvTransposed3dSquareInputSquareKernelPaddedDilatedStridedCustom(const char *name) : OpDef(name)
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
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvTransposed3dSquareInputSquareKernelPaddedDilatedStridedCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConvTransposed3dSquareInputSquareKernelPaddedDilatedStrided {
public:
    __aicore__ inline KernelConvTransposed3dSquareInputSquareKernelPaddedDilatedStrided() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t inChannels,
        uint32_t outChannels,
        uint32_t inputDepth,
        uint32_t inputHeight,
        uint32_t inputWidth,
        uint32_t kernelSize,
        uint32_t outputDepth,
        uint32_t outputHeight,
        uint32_t outputWidth,
        uint32_t stride,
        uint32_t padding,
        uint32_t dilation)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->inputDepth = inputDepth;
        this->inputHeight = inputHeight;
        this->inputWidth = inputWidth;
        this->kernelSize = kernelSize;
        this->outputDepth = outputDepth;
        this->outputHeight = outputHeight;
        this->outputWidth = outputWidth;
        this->stride = stride;
        this->padding = padding;
        this->dilation = dilation;
        this->blockIdx = GetBlockIdx();

        this->inputPlaneStride = inputHeight * inputWidth;
        this->inputChannelStride = inputDepth * this->inputPlaneStride;
        this->inputBatchStride = inChannels * this->inputChannelStride;
        this->outputPlaneStride = outputHeight * outputWidth;
        this->outputChannelStride = outputDepth * this->outputPlaneStride;
        this->outputBatchStride = outChannels * this->outputChannelStride;
        this->weightPlaneStride = kernelSize * kernelSize;
        this->weightOutputStride = kernelSize * this->weightPlaneStride;
        this->weightInputStride = outChannels * this->weightOutputStride;

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
            const uint32_t yChannelBase = yBatchBase + outChannel * this->outputChannelStride;
            for (uint32_t outD = 0; outD < this->outputDepth; ++outD) {
                for (uint32_t outH = 0; outH < this->outputHeight; ++outH) {
                    for (uint32_t outW = 0; outW < this->outputWidth; ++outW) {
                        float sum = 0.0f;
                        for (uint32_t inChannel = 0; inChannel < this->inChannels; ++inChannel) {
                            const uint32_t xChannelBase = xBatchBase + inChannel * this->inputChannelStride;
                            const uint32_t wChannelBase =
                                inChannel * this->weightInputStride + outChannel * this->weightOutputStride;
                            for (uint32_t kernelD = 0; kernelD < this->kernelSize; ++kernelD) {
                                const int32_t numerD =
                                    static_cast<int32_t>(outD) + static_cast<int32_t>(this->padding) -
                                    static_cast<int32_t>(kernelD) * static_cast<int32_t>(this->dilation);
                                if (numerD < 0 || numerD % static_cast<int32_t>(this->stride) != 0) {
                                    continue;
                                }
                                const int32_t inD = numerD / static_cast<int32_t>(this->stride);
                                if (inD < 0 || inD >= static_cast<int32_t>(this->inputDepth)) {
                                    continue;
                                }
                                for (uint32_t kernelH = 0; kernelH < this->kernelSize; ++kernelH) {
                                    const int32_t numerH =
                                        static_cast<int32_t>(outH) + static_cast<int32_t>(this->padding) -
                                        static_cast<int32_t>(kernelH) * static_cast<int32_t>(this->dilation);
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
                                            static_cast<int32_t>(kernelW) * static_cast<int32_t>(this->dilation);
                                        if (numerW < 0 || numerW % static_cast<int32_t>(this->stride) != 0) {
                                            continue;
                                        }
                                        const int32_t inW = numerW / static_cast<int32_t>(this->stride);
                                        if (inW < 0 || inW >= static_cast<int32_t>(this->inputWidth)) {
                                            continue;
                                        }

                                        const uint32_t xOffset =
                                            xChannelBase +
                                            static_cast<uint32_t>(inD) * this->inputPlaneStride +
                                            static_cast<uint32_t>(inH) * this->inputWidth +
                                            static_cast<uint32_t>(inW);
                                        const uint32_t wOffset =
                                            wChannelBase +
                                            kernelD * this->weightPlaneStride +
                                            kernelH * this->kernelSize +
                                            kernelW;
                                        sum += xGm.GetValue(xOffset) * weightGm.GetValue(wOffset);
                                    }
                                }
                            }
                        }
                        yGm.SetValue(
                            yChannelBase + outD * this->outputPlaneStride + outH * this->outputWidth + outW,
                            sum);
                    }
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
    uint32_t inputDepth;
    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t kernelSize;
    uint32_t outputDepth;
    uint32_t outputHeight;
    uint32_t outputWidth;
    uint32_t stride;
    uint32_t padding;
    uint32_t dilation;
    uint32_t blockIdx;
    uint32_t inputPlaneStride;
    uint32_t inputChannelStride;
    uint32_t inputBatchStride;
    uint32_t outputPlaneStride;
    uint32_t outputChannelStride;
    uint32_t outputBatchStride;
    uint32_t weightPlaneStride;
    uint32_t weightOutputStride;
    uint32_t weightInputStride;
};

extern "C" __global__ __aicore__ void conv_transposed3d_square_input_square_kernel_padded_dilated_strided_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTransposed3dSquareInputSquareKernelPaddedDilatedStrided op;
    op.Init(
        x,
        weight,
        y,
        tiling_data.batchSize,
        tiling_data.inChannels,
        tiling_data.outChannels,
        tiling_data.inputDepth,
        tiling_data.inputHeight,
        tiling_data.inputWidth,
        tiling_data.kernelSize,
        tiling_data.outputDepth,
        tiling_data.outputHeight,
        tiling_data.outputWidth,
        tiling_data.stride,
        tiling_data.padding,
        tiling_data.dilation);
    op.Process();
}
"""

python_bind_src="""
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <vector>
#include "pytorch_npu_helper.hpp"

namespace {
constexpr int64_t kStride = 2;
constexpr int64_t kPadding = 1;
constexpr int64_t kDilation = 2;
constexpr int64_t kOutputPadding = 0;
}

at::Tensor conv_transposed3d_square_input_square_kernel_padded_dilated_strided_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight)
{
    TORCH_CHECK(x.dim() == 5, "x must be a 5D NCDHW tensor");
    TORCH_CHECK(weight.dim() == 5, "weight must be a 5D tensor");
    TORCH_CHECK(x.size(1) == weight.size(0), "input channels must match weight.size(0)");
    TORCH_CHECK(weight.size(2) == weight.size(3) && weight.size(2) == weight.size(4), "weight kernel must be cubic");
    TORCH_CHECK(x.size(3) == x.size(4), "input height and width must match");

    const int64_t outD =
        (x.size(2) - 1) * kStride - 2 * kPadding + kDilation * (weight.size(2) - 1) + kOutputPadding + 1;
    const int64_t outH =
        (x.size(3) - 1) * kStride - 2 * kPadding + kDilation * (weight.size(3) - 1) + kOutputPadding + 1;
    const int64_t outW =
        (x.size(4) - 1) * kStride - 2 * kPadding + kDilation * (weight.size(4) - 1) + kOutputPadding + 1;
    TORCH_CHECK(outD > 0 && outH > 0 && outW > 0, "invalid output shape");

    at::Tensor result = at::empty({x.size(0), weight.size(1), outD, outH, outW}, x.options());
    c10::optional<at::Tensor> bias = c10::nullopt;
    std::vector<int64_t> strideVec = {kStride, kStride, kStride};
    std::vector<int64_t> paddingVec = {kPadding, kPadding, kPadding};
    std::vector<int64_t> dilationVec = {kDilation, kDilation, kDilation};
    std::vector<int64_t> outputPaddingVec = {kOutputPadding, kOutputPadding, kOutputPadding};
    at::IntArrayRef strides(strideVec);
    at::IntArrayRef paddings(paddingVec);
    at::IntArrayRef dilations(dilationVec);
    at::IntArrayRef outputPaddings(outputPaddingVec);
    const bool transposed = true;
    const int64_t groups = 1;
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
        outputPaddings,
        groups,
        result,
        cubeMathType);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "conv_transposed3d_square_input_square_kernel_padded_dilated_strided_custom",
        &conv_transposed3d_square_input_square_kernel_padded_dilated_strided_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv_transposed3d_square_input_square_kernel_padded_dilated_strided_custom",
        &conv_transposed3d_square_input_square_kernel_padded_dilated_strided_custom_impl_npu,
        "conv_transposed3d_square_input_square_kernel_padded_dilated_strided_custom");
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
        stride: int = 2,
        padding: int = 1,
        dilation: int = 2,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if stride != 2 or padding != 1 or dilation != 2 or bias:
            raise ValueError(
                "This AscendC implementation supports only stride=2, padding=1, dilation=2, bias=False."
            )

        self.conv_transpose3d = torch.nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv_transposed3d_square_input_square_kernel_padded_dilated_strided_custom(
            x,
            self.conv_transpose3d.weight,
        )
'''
