project_json_src='''
[
    {
        "op": "Conv3dReluLeakyReluGeluSigmoidBiasAddCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "x",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "weight",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "conv_bias",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "bias_add",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            }
        ],
        "output_desc": [
            {
                "name": "y",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            }
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Conv3dReluLeakyReluGeluSigmoidBiasAddCustomTilingData)
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
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    Conv3dReluLeakyReluGeluSigmoidBiasAddCustom,
    Conv3dReluLeakyReluGeluSigmoidBiasAddCustomTilingData)
}
"""

host_operator_src="""
#include "conv3d_relu_leaky_relu_gelu_sigmoid_bias_add_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
uint32_t ComputeOutputDim(int64_t input, int64_t kernelSize)
{
    if (input < kernelSize) {
        return 0;
    }
    return static_cast<uint32_t>(input - kernelSize + 1);
}

uint32_t MinUint32(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *inputShape = context->GetInputShape(0);
    const gert::StorageShape *weightShape = context->GetInputShape(1);
    const gert::StorageShape *convBiasShape = context->GetInputShape(2);
    const gert::StorageShape *biasAddShape = context->GetInputShape(3);
    if (inputShape == nullptr || weightShape == nullptr || convBiasShape == nullptr || biasAddShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xShape = inputShape->GetStorageShape();
    const auto wShape = weightShape->GetStorageShape();
    const auto cbShape = convBiasShape->GetStorageShape();
    const auto baShape = biasAddShape->GetStorageShape();
    if (xShape.GetDimNum() != 5 || wShape.GetDimNum() != 5 || cbShape.GetDimNum() != 1 || baShape.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t inChannels = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t inputDepth = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t inputHeight = static_cast<uint32_t>(xShape.GetDim(3));
    const uint32_t inputWidth = static_cast<uint32_t>(xShape.GetDim(4));

    const uint32_t outChannels = static_cast<uint32_t>(wShape.GetDim(0));
    const uint32_t weightInChannels = static_cast<uint32_t>(wShape.GetDim(1));
    const uint32_t kernelDepth = static_cast<uint32_t>(wShape.GetDim(2));
    const uint32_t kernelHeight = static_cast<uint32_t>(wShape.GetDim(3));
    const uint32_t kernelWidth = static_cast<uint32_t>(wShape.GetDim(4));

    if (inChannels != weightInChannels) {
        return ge::GRAPH_FAILED;
    }
    if (static_cast<uint32_t>(cbShape.GetDim(0)) != outChannels) {
        return ge::GRAPH_FAILED;
    }
    if (static_cast<uint32_t>(baShape.GetDim(0)) != outChannels ||
        baShape.GetDim(1) != 1 || baShape.GetDim(2) != 1 || baShape.GetDim(3) != 1) {
        return ge::GRAPH_FAILED;
    }

    Conv3dReluLeakyReluGeluSigmoidBiasAddCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_inputDepth(inputDepth);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_kernelDepth(kernelDepth);
    tiling.set_kernelHeight(kernelHeight);
    tiling.set_kernelWidth(kernelWidth);
    tiling.set_outputDepth(ComputeOutputDim(inputDepth, kernelDepth));
    tiling.set_outputHeight(ComputeOutputDim(inputHeight, kernelHeight));
    tiling.set_outputWidth(ComputeOutputDim(inputWidth, kernelWidth));

    context->SetBlockDim(MinUint32(batchSize, 32));
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
    const gert::Shape *biasAddShape = context->GetInputShape(3);
    if (inputShape == nullptr || weightShape == nullptr || convBiasShape == nullptr || biasAddShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDimNum() != 5 || weightShape->GetDimNum() != 5 || convBiasShape->GetDimNum() != 1 || biasAddShape->GetDimNum() != 4) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDim(1) != weightShape->GetDim(1)) {
        return GRAPH_FAILED;
    }
    if (convBiasShape->GetDim(0) != weightShape->GetDim(0) || biasAddShape->GetDim(0) != weightShape->GetDim(0)) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(5);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, weightShape->GetDim(0));
    outputShape->SetDim(2, inputShape->GetDim(2) - weightShape->GetDim(2) + 1);
    outputShape->SetDim(3, inputShape->GetDim(3) - weightShape->GetDim(3) + 1);
    outputShape->SetDim(4, inputShape->GetDim(4) - weightShape->GetDim(4) + 1);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Conv3dReluLeakyReluGeluSigmoidBiasAddCustom : public OpDef {
public:
    explicit Conv3dReluLeakyReluGeluSigmoidBiasAddCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("conv_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias_add").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv3dReluLeakyReluGeluSigmoidBiasAddCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConv3dReluLeakyReluGeluSigmoidBiasAdd {
public:
    static constexpr float LN2 = 0.69314718056f;

    __aicore__ inline KernelConv3dReluLeakyReluGeluSigmoidBiasAdd() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR convBias,
        GM_ADDR biasAdd,
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
        uint32_t outputWidth)
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
        this->blockIdx = GetBlockIdx();
        this->blockNum = GetBlockNum();

        this->inputWidthStride = 1;
        this->inputHeightStride = inputWidth;
        this->inputDepthStride = inputHeight * inputWidth;
        this->inputChannelStride = inputDepth * this->inputDepthStride;
        this->inputBatchStride = inChannels * this->inputChannelStride;

        this->outputWidthStride = 1;
        this->outputHeightStride = outputWidth;
        this->outputDepthStride = outputHeight * outputWidth;
        this->outputChannelStride = outputDepth * this->outputDepthStride;
        this->outputBatchStride = outChannels * this->outputChannelStride;

        this->weightWidthStride = 1;
        this->weightHeightStride = kernelWidth;
        this->weightDepthStride = kernelHeight * kernelWidth;
        this->weightInChannelStride = kernelDepth * this->weightDepthStride;
        this->weightOutChannelStride = inChannels * this->weightInChannelStride;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * this->inputBatchStride);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, outChannels * this->weightOutChannelStride);
        convBiasGm.SetGlobalBuffer((__gm__ float *)convBias, outChannels);
        biasAddGm.SetGlobalBuffer((__gm__ float *)biasAdd, outChannels);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * this->outputBatchStride);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t batchIdx = this->blockIdx; batchIdx < this->batchSize; batchIdx += this->blockNum) {
            ComputeBatch(batchIdx);
        }
    }

private:
    __aicore__ inline void ComputeBatch(uint32_t batchIdx)
    {
        const uint32_t xBatchBase = batchIdx * this->inputBatchStride;
        const uint32_t yBatchBase = batchIdx * this->outputBatchStride;

        for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
            const uint32_t weightBase = outChannel * this->weightOutChannelStride;
            const uint32_t yChannelBase = yBatchBase + outChannel * this->outputChannelStride;
            const float convBiasValue = convBiasGm.GetValue(outChannel);
            const float biasAddValue = biasAddGm.GetValue(outChannel);

            for (uint32_t outD = 0; outD < this->outputDepth; ++outD) {
                for (uint32_t outH = 0; outH < this->outputHeight; ++outH) {
                    for (uint32_t outW = 0; outW < this->outputWidth; ++outW) {
                        float value = convBiasValue;
                        for (uint32_t inChannel = 0; inChannel < this->inChannels; ++inChannel) {
                            const uint32_t xChannelBase = xBatchBase + inChannel * this->inputChannelStride;
                            const uint32_t wChannelBase = weightBase + inChannel * this->weightInChannelStride;
                            for (uint32_t kernelD = 0; kernelD < this->kernelDepth; ++kernelD) {
                                const uint32_t inD = outD + kernelD;
                                for (uint32_t kernelH = 0; kernelH < this->kernelHeight; ++kernelH) {
                                    const uint32_t inH = outH + kernelH;
                                    for (uint32_t kernelW = 0; kernelW < this->kernelWidth; ++kernelW) {
                                        const uint32_t inW = outW + kernelW;
                                        const uint32_t xOffset =
                                            xChannelBase +
                                            inD * this->inputDepthStride +
                                            inH * this->inputHeightStride +
                                            inW * this->inputWidthStride;
                                        const uint32_t wOffset =
                                            wChannelBase +
                                            kernelD * this->weightDepthStride +
                                            kernelH * this->weightHeightStride +
                                            kernelW * this->weightWidthStride;
                                        value += xGm.GetValue(xOffset) * weightGm.GetValue(wOffset);
                                    }
                                }
                            }
                        }

                        value = ApplyRelu(value);
                        value = ApplyLeakyRelu(value);
                        value = ApplyGelu(value);
                        value = ApplySigmoid(value);
                        value += biasAddValue;

                        const uint32_t yOffset =
                            yChannelBase +
                            outD * this->outputDepthStride +
                            outH * this->outputHeightStride +
                            outW * this->outputWidthStride;
                        yGm.SetValue(yOffset, value);
                    }
                }
            }
        }
    }

    __aicore__ inline float ApplyRelu(float x) const
    {
        return x > 0.0f ? x : 0.0f;
    }

    __aicore__ inline float ApplyLeakyRelu(float x) const
    {
        const float negativeSlope = 0.01f;
        return x > 0.0f ? x : x * negativeSlope;
    }

    __aicore__ inline float ApplyGelu(float x) const
    {
        const float coeff = 0.79788458f;
        const float inner = coeff * (x + 0.044715f * x * x * x);
        return 0.5f * x * (1.0f + FastTanh(inner));
    }

    __aicore__ inline float ApplySigmoid(float x) const
    {
        if (x >= 8.0f) {
            return 0.99966466f;
        }
        if (x <= -8.0f) {
            return 0.00033535f;
        }
        return 1.0f / (1.0f + FastExp(-x));
    }

    __aicore__ inline float FastTanh(float x) const
    {
        if (x > 5.0f) {
            return 1.0f;
        }
        if (x < -5.0f) {
            return -1.0f;
        }
        const float expPos = FastExp(x);
        const float expNeg = FastExp(-x);
        return (expPos - expNeg) / (expPos + expNeg);
    }

    __aicore__ inline float FastExp(float x) const
    {
        if (x < -20.0f) {
            return 0.0f;
        }
        int32_t k = 0;
        while (x > 0.5f * LN2) {
            x -= LN2;
            ++k;
        }
        while (x < -0.5f * LN2) {
            x += LN2;
            --k;
        }

        const float x2 = x * x;
        const float x3 = x2 * x;
        const float x4 = x3 * x;
        const float x5 = x4 * x;
        float result = 1.0f + x + 0.5f * x2 + 0.16666667f * x3 + 0.04166667f * x4 + 0.0083333333f * x5;
        if (k > 0) {
            for (int32_t i = 0; i < k; ++i) {
                result *= 2.0f;
            }
        } else {
            for (int32_t i = 0; i < -k; ++i) {
                result *= 0.5f;
            }
        }
        return result;
    }

private:
    GlobalTensor<float> xGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> convBiasGm;
    GlobalTensor<float> biasAddGm;
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
    uint32_t inputWidthStride;
    uint32_t inputHeightStride;
    uint32_t inputDepthStride;
    uint32_t inputChannelStride;
    uint32_t inputBatchStride;
    uint32_t outputWidthStride;
    uint32_t outputHeightStride;
    uint32_t outputDepthStride;
    uint32_t outputChannelStride;
    uint32_t outputBatchStride;
    uint32_t weightWidthStride;
    uint32_t weightHeightStride;
    uint32_t weightDepthStride;
    uint32_t weightInChannelStride;
    uint32_t weightOutChannelStride;
    uint32_t blockIdx;
    uint32_t blockNum;
};

extern "C" __global__ __aicore__ void conv3d_relu_leaky_relu_gelu_sigmoid_bias_add_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR conv_bias,
    GM_ADDR bias_add,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv3dReluLeakyReluGeluSigmoidBiasAdd op;
    op.Init(
        x,
        weight,
        conv_bias,
        bias_add,
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
        tiling_data.outputWidth);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor conv3d_relu_leaky_relu_gelu_sigmoid_bias_add_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &conv_bias,
    const at::Tensor &bias_add)
{
    TORCH_CHECK(x.dim() == 5, "x must be a 5D NCDHW tensor");
    TORCH_CHECK(weight.dim() == 5, "weight must be a 5D OIDHW tensor");
    TORCH_CHECK(conv_bias.dim() == 1, "conv_bias must be a 1D tensor");
    TORCH_CHECK(bias_add.dim() == 4, "bias_add must be a 4D tensor with shape [C, 1, 1, 1]");
    TORCH_CHECK(x.size(1) == weight.size(1), "input channels must match weight.size(1)");
    TORCH_CHECK(weight.size(0) == conv_bias.size(0), "conv bias length must match output channels");
    TORCH_CHECK(weight.size(0) == bias_add.size(0), "bias_add size(0) must match output channels");
    TORCH_CHECK(bias_add.size(1) == 1 && bias_add.size(2) == 1 && bias_add.size(3) == 1, "bias_add must have shape [C,1,1,1]");
    TORCH_CHECK(x.size(2) >= weight.size(2) && x.size(3) >= weight.size(3) && x.size(4) >= weight.size(4), "kernel must fit inside input");

    const int64_t outputDepth = x.size(2) - weight.size(2) + 1;
    const int64_t outputHeight = x.size(3) - weight.size(3) + 1;
    const int64_t outputWidth = x.size(4) - weight.size(4) + 1;

    at::Tensor result = at::empty({x.size(0), weight.size(0), outputDepth, outputHeight, outputWidth}, x.options());
    EXEC_NPU_CMD(aclnnConv3dReluLeakyReluGeluSigmoidBiasAddCustom, x, weight, conv_bias, bias_add, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv3d_relu_leaky_relu_gelu_sigmoid_bias_add_custom", &conv3d_relu_leaky_relu_gelu_sigmoid_bias_add_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv3d_relu_leaky_relu_gelu_sigmoid_bias_add_custom",
        &conv3d_relu_leaky_relu_gelu_sigmoid_bias_add_impl_npu,
        "conv3d_relu_leaky_relu_gelu_sigmoid_bias_add_custom");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias_shape) -> None:
        super().__init__()
        self.conv = torch.nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = torch.nn.Parameter(torch.randn(bias_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv3d_relu_leaky_relu_gelu_sigmoid_bias_add_custom(
            x,
            self.conv.weight,
            self.conv.bias,
            self.bias,
        )
'''
