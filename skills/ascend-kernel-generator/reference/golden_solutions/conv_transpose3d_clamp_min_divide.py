project_json_src='''
[
    {
        "op": "ConvTranspose3dClampMinDivideCustom",
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
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTranspose3dClampMinDivideCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inChannels);
    TILING_DATA_FIELD_DEF(uint32_t, outChannels);
    TILING_DATA_FIELD_DEF(uint32_t, inputDepth);
    TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, kernelDepth);
    TILING_DATA_FIELD_DEF(uint32_t, kernelHeight);
    TILING_DATA_FIELD_DEF(uint32_t, kernelWidth);
    TILING_DATA_FIELD_DEF(uint32_t, outputDepth);
    TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, strideD);
    TILING_DATA_FIELD_DEF(uint32_t, strideH);
    TILING_DATA_FIELD_DEF(uint32_t, strideW);
    TILING_DATA_FIELD_DEF(uint32_t, paddingD);
    TILING_DATA_FIELD_DEF(uint32_t, paddingH);
    TILING_DATA_FIELD_DEF(uint32_t, paddingW);
    TILING_DATA_FIELD_DEF(uint32_t, dilationD);
    TILING_DATA_FIELD_DEF(uint32_t, dilationH);
    TILING_DATA_FIELD_DEF(uint32_t, dilationW);
    TILING_DATA_FIELD_DEF(uint32_t, outputPaddingD);
    TILING_DATA_FIELD_DEF(uint32_t, outputPaddingH);
    TILING_DATA_FIELD_DEF(uint32_t, outputPaddingW);
    TILING_DATA_FIELD_DEF(uint32_t, groups);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    ConvTranspose3dClampMinDivideCustom,
    ConvTranspose3dClampMinDivideCustomTilingData)
}
"""

host_operator_src="""
#include "conv_transpose3d_clamp_min_divide_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
constexpr uint32_t kStride = 2;
constexpr uint32_t kPadding = 1;
constexpr uint32_t kDilation = 1;
constexpr uint32_t kOutputPadding = 0;
constexpr uint32_t kGroups = 1;

uint32_t ComputeTransposedOutputDim(
    int64_t input,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t outputPadding)
{
    if (stride <= 0 || dilation <= 0) {
        return 0;
    }
    const int64_t effectiveKernel = dilation * (kernel - 1) + 1;
    const int64_t output = (input - 1) * stride - 2 * padding + effectiveKernel + outputPadding;
    return output > 0 ? static_cast<uint32_t>(output) : 0;
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
    if (xShape.GetDimNum() != 5 || wShape.GetDimNum() != 5 || bShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t inChannels = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t inputDepth = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t inputHeight = static_cast<uint32_t>(xShape.GetDim(3));
    const uint32_t inputWidth = static_cast<uint32_t>(xShape.GetDim(4));
    const uint32_t weightInputChannels = static_cast<uint32_t>(wShape.GetDim(0));
    const uint32_t outChannelsPerGroup = static_cast<uint32_t>(wShape.GetDim(1));
    const uint32_t kernelDepth = static_cast<uint32_t>(wShape.GetDim(2));
    const uint32_t kernelHeight = static_cast<uint32_t>(wShape.GetDim(3));
    const uint32_t kernelWidth = static_cast<uint32_t>(wShape.GetDim(4));
    const uint32_t outChannels = outChannelsPerGroup * kGroups;

    if (inChannels != weightInputChannels || bShape.GetDim(0) != outChannels) {
        return ge::GRAPH_FAILED;
    }

    ConvTranspose3dClampMinDivideCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_inputDepth(inputDepth);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_kernelDepth(kernelDepth);
    tiling.set_kernelHeight(kernelHeight);
    tiling.set_kernelWidth(kernelWidth);
    tiling.set_outputDepth(
        ComputeTransposedOutputDim(inputDepth, kernelDepth, kStride, kPadding, kDilation, kOutputPadding));
    tiling.set_outputHeight(
        ComputeTransposedOutputDim(inputHeight, kernelHeight, kStride, kPadding, kDilation, kOutputPadding));
    tiling.set_outputWidth(
        ComputeTransposedOutputDim(inputWidth, kernelWidth, kStride, kPadding, kDilation, kOutputPadding));
    tiling.set_strideD(kStride);
    tiling.set_strideH(kStride);
    tiling.set_strideW(kStride);
    tiling.set_paddingD(kPadding);
    tiling.set_paddingH(kPadding);
    tiling.set_paddingW(kPadding);
    tiling.set_dilationD(kDilation);
    tiling.set_dilationH(kDilation);
    tiling.set_dilationW(kDilation);
    tiling.set_outputPaddingD(kOutputPadding);
    tiling.set_outputPaddingH(kOutputPadding);
    tiling.set_outputPaddingW(kOutputPadding);
    tiling.set_groups(kGroups);

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
    if (inputShape->GetDimNum() != 5 || weightShape->GetDimNum() != 5 || biasShape->GetDimNum() != 1) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(5);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, weightShape->GetDim(1));
    outputShape->SetDim(
        2,
        ComputeTransposedOutputDim(inputShape->GetDim(2), weightShape->GetDim(2), kStride, kPadding, kDilation, kOutputPadding));
    outputShape->SetDim(
        3,
        ComputeTransposedOutputDim(inputShape->GetDim(3), weightShape->GetDim(3), kStride, kPadding, kDilation, kOutputPadding));
    outputShape->SetDim(
        4,
        ComputeTransposedOutputDim(inputShape->GetDim(4), weightShape->GetDim(4), kStride, kPadding, kDilation, kOutputPadding));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ConvTranspose3dClampMinDivideCustom : public OpDef {
public:
    explicit ConvTranspose3dClampMinDivideCustom(const char *name)
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

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvTranspose3dClampMinDivideCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr float kClampMin = -1.0f;
constexpr float kDivisor = 2.0f;
}

class KernelConvTranspose3dClampMinDivide {
public:
    __aicore__ inline KernelConvTranspose3dClampMinDivide() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t inChannels,
        uint32_t outChannels,
        uint32_t inputDepth,
        uint32_t inputHeight,
        uint32_t inputWidth,
        uint32_t kernelDepth,
        uint32_t kernelHeight,
        uint32_t kernelWidth,
        uint32_t outputDepth,
        uint32_t outputHeight,
        uint32_t outputWidth,
        uint32_t strideD,
        uint32_t strideH,
        uint32_t strideW,
        uint32_t paddingD,
        uint32_t paddingH,
        uint32_t paddingW,
        uint32_t dilationD,
        uint32_t dilationH,
        uint32_t dilationW,
        uint32_t groups)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->inputDepth = inputDepth;
        this->inputHeight = inputHeight;
        this->inputWidth = inputWidth;
        this->kernelDepth = kernelDepth;
        this->kernelHeight = kernelHeight;
        this->kernelWidth = kernelWidth;
        this->outputDepth = outputDepth;
        this->outputHeight = outputHeight;
        this->outputWidth = outputWidth;
        this->strideD = strideD;
        this->strideH = strideH;
        this->strideW = strideW;
        this->paddingD = paddingD;
        this->paddingH = paddingH;
        this->paddingW = paddingW;
        this->dilationD = dilationD;
        this->dilationH = dilationH;
        this->dilationW = dilationW;
        this->groups = groups;
        this->blockIdx = GetBlockIdx();
        this->inputPlaneStride = inputHeight * inputWidth;
        this->inputChannelStride = inputDepth * this->inputPlaneStride;
        this->inputBatchStride = inChannels * this->inputChannelStride;
        this->outputPlaneStride = outputHeight * outputWidth;
        this->outputChannelStride = outputDepth * this->outputPlaneStride;
        this->outputBatchStride = outChannels * this->outputChannelStride;
        this->weightOutputChannelsPerGroup = outChannels / groups;
        this->inputChannelsPerGroup = inChannels / groups;
        this->weightKernelStride = kernelDepth * kernelHeight * kernelWidth;
        this->weightInputStride = this->weightOutputChannelsPerGroup * this->weightKernelStride;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * this->inputBatchStride);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, inChannels * this->weightInputStride);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, outChannels);
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

            for (uint32_t outD = 0; outD < this->outputDepth; ++outD) {
                for (uint32_t outH = 0; outH < this->outputHeight; ++outH) {
                    for (uint32_t outW = 0; outW < this->outputWidth; ++outW) {
                        float sum = biasGm.GetValue(outChannel);
                        for (uint32_t inChannel = inChannelStart; inChannel < inChannelEnd; ++inChannel) {
                            const uint32_t xChannelBase = xBatchBase + inChannel * this->inputChannelStride;
                            const uint32_t wChannelBase =
                                inChannel * this->weightInputStride + groupOutChannel * this->weightKernelStride;
                            for (uint32_t kernelD = 0; kernelD < this->kernelDepth; ++kernelD) {
                                const int32_t numerD =
                                    static_cast<int32_t>(outD) + static_cast<int32_t>(this->paddingD) -
                                    static_cast<int32_t>(kernelD) * static_cast<int32_t>(this->dilationD);
                                if (numerD < 0 || numerD % static_cast<int32_t>(this->strideD) != 0) {
                                    continue;
                                }
                                const int32_t inD = numerD / static_cast<int32_t>(this->strideD);
                                if (inD < 0 || inD >= static_cast<int32_t>(this->inputDepth)) {
                                    continue;
                                }
                                for (uint32_t kernelH = 0; kernelH < this->kernelHeight; ++kernelH) {
                                    const int32_t numerH =
                                        static_cast<int32_t>(outH) + static_cast<int32_t>(this->paddingH) -
                                        static_cast<int32_t>(kernelH) * static_cast<int32_t>(this->dilationH);
                                    if (numerH < 0 || numerH % static_cast<int32_t>(this->strideH) != 0) {
                                        continue;
                                    }
                                    const int32_t inH = numerH / static_cast<int32_t>(this->strideH);
                                    if (inH < 0 || inH >= static_cast<int32_t>(this->inputHeight)) {
                                        continue;
                                    }
                                    for (uint32_t kernelW = 0; kernelW < this->kernelWidth; ++kernelW) {
                                        const int32_t numerW =
                                            static_cast<int32_t>(outW) + static_cast<int32_t>(this->paddingW) -
                                            static_cast<int32_t>(kernelW) * static_cast<int32_t>(this->dilationW);
                                        if (numerW < 0 || numerW % static_cast<int32_t>(this->strideW) != 0) {
                                            continue;
                                        }
                                        const int32_t inW = numerW / static_cast<int32_t>(this->strideW);
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
                                            kernelD * this->kernelHeight * this->kernelWidth +
                                            kernelH * this->kernelWidth +
                                            kernelW;
                                        sum += xGm.GetValue(xOffset) * weightGm.GetValue(wOffset);
                                    }
                                }
                            }
                        }

                        if (sum < kClampMin) {
                            sum = kClampMin;
                        }
                        sum /= kDivisor;
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
    GlobalTensor<float> biasGm;
    GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t inputDepth;
    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t kernelDepth;
    uint32_t kernelHeight;
    uint32_t kernelWidth;
    uint32_t outputDepth;
    uint32_t outputHeight;
    uint32_t outputWidth;
    uint32_t strideD;
    uint32_t strideH;
    uint32_t strideW;
    uint32_t paddingD;
    uint32_t paddingH;
    uint32_t paddingW;
    uint32_t dilationD;
    uint32_t dilationH;
    uint32_t dilationW;
    uint32_t groups;
    uint32_t blockIdx;
    uint32_t inputPlaneStride;
    uint32_t inputChannelStride;
    uint32_t inputBatchStride;
    uint32_t outputPlaneStride;
    uint32_t outputChannelStride;
    uint32_t outputBatchStride;
    uint32_t weightInputStride;
    uint32_t weightKernelStride;
    uint32_t weightOutputChannelsPerGroup;
    uint32_t inputChannelsPerGroup;
};

extern "C" __global__ __aicore__ void conv_transpose3d_clamp_min_divide_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTranspose3dClampMinDivide op;
    op.Init(
        x,
        weight,
        bias,
        y,
        tiling_data.batchSize,
        tiling_data.inChannels,
        tiling_data.outChannels,
        tiling_data.inputDepth,
        tiling_data.inputHeight,
        tiling_data.inputWidth,
        tiling_data.kernelDepth,
        tiling_data.kernelHeight,
        tiling_data.kernelWidth,
        tiling_data.outputDepth,
        tiling_data.outputHeight,
        tiling_data.outputWidth,
        tiling_data.strideD,
        tiling_data.strideH,
        tiling_data.strideW,
        tiling_data.paddingD,
        tiling_data.paddingH,
        tiling_data.paddingW,
        tiling_data.dilationD,
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

namespace {
constexpr int64_t kStride = 2;
constexpr int64_t kPadding = 1;
constexpr int64_t kDilation = 1;
constexpr int64_t kOutputPadding = 0;
constexpr int64_t kGroups = 1;
constexpr double kClampMin = -1.0;
constexpr double kDivisor = 2.0;
}

at::Tensor conv_transpose3d_clamp_min_divide_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &bias)
{
    TORCH_CHECK(x.dim() == 5, "x must be a 5D NCDHW tensor");
    TORCH_CHECK(weight.dim() == 5, "weight must be a 5D tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
    TORCH_CHECK(x.size(1) == weight.size(0), "input channels must match weight.size(0)");
    TORCH_CHECK(weight.size(1) == bias.size(0), "bias size must match output channels");

    const int64_t outChannels = weight.size(1) * kGroups;
    const int64_t outD =
        (x.size(2) - 1) * kStride - 2 * kPadding + kDilation * (weight.size(2) - 1) + kOutputPadding + 1;
    const int64_t outH =
        (x.size(3) - 1) * kStride - 2 * kPadding + kDilation * (weight.size(3) - 1) + kOutputPadding + 1;
    const int64_t outW =
        (x.size(4) - 1) * kStride - 2 * kPadding + kDilation * (weight.size(4) - 1) + kOutputPadding + 1;
    TORCH_CHECK(outD > 0 && outH > 0 && outW > 0, "invalid output shape");

    at::Tensor result = at::empty({x.size(0), outChannels, outD, outH, outW}, x.options());
    std::vector<int64_t> strideVec = {kStride, kStride, kStride};
    std::vector<int64_t> paddingVec = {kPadding, kPadding, kPadding};
    std::vector<int64_t> dilationVec = {kDilation, kDilation, kDilation};
    std::vector<int64_t> outputPaddingVec = {kOutputPadding, kOutputPadding, kOutputPadding};
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
        kGroups,
        result,
        cubeMathType);

    result = at::clamp_min(result, kClampMin);
    result = result / kDivisor;
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_transpose3d_clamp_min_divide", &conv_transpose3d_clamp_min_divide_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv_transpose3d_clamp_min_divide",
        &conv_transpose3d_clamp_min_divide_impl_npu,
        "conv_transpose3d_clamp_min_divide");
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
        stride: int,
        padding: int,
        min_value: float,
        divisor: float,
    ) -> None:
        super().__init__()
        if stride != 2 or padding != 1:
            raise ValueError("This implementation supports only stride=2 and padding=1.")
        if float(min_value) != -1.0:
            raise ValueError("This implementation supports only min_value=-1.0.")
        if float(divisor) != 2.0:
            raise ValueError("This implementation supports only divisor=2.0.")

        self.conv_transpose3d = torch.nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv_transpose3d_clamp_min_divide(
            x,
            self.conv_transpose3d.weight,
            self.conv_transpose3d.bias,
        )
'''
