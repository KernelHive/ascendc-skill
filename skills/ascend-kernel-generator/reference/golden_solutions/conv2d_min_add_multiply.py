project_json_src='''
[
    {
        "op": "Conv2dMinAddMultiplyCustom",
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
                "name": "add_bias",
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
                "name": "min_value",
                "param_type": "required",
                "type": "float"
            },
            {
                "name": "scaling_factor",
                "param_type": "required",
                "type": "float"
            }
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Conv2dMinAddMultiplyCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inChannels);
    TILING_DATA_FIELD_DEF(uint32_t, outChannels);
    TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, kernelHeight);
    TILING_DATA_FIELD_DEF(uint32_t, kernelWidth);
    TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
    TILING_DATA_FIELD_DEF(float, minValue);
    TILING_DATA_FIELD_DEF(float, scalingFactor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv2dMinAddMultiplyCustom, Conv2dMinAddMultiplyCustomTilingData)
}
"""

host_operator_src="""
#include "conv2d_min_add_multiply_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
uint32_t ComputeConvOutputDim(int64_t input, int64_t kernel)
{
    if (input < kernel || kernel <= 0) {
        return 0;
    }
    return static_cast<uint32_t>(input - kernel + 1);
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *inputShape = context->GetInputShape(0);
    const gert::StorageShape *weightShape = context->GetInputShape(1);
    const gert::StorageShape *convBiasShape = context->GetInputShape(2);
    const gert::StorageShape *addBiasShape = context->GetInputShape(3);
    if (inputShape == nullptr || weightShape == nullptr || convBiasShape == nullptr || addBiasShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xShape = inputShape->GetStorageShape();
    const auto wShape = weightShape->GetStorageShape();
    const auto convBShape = convBiasShape->GetStorageShape();
    const auto addBShape = addBiasShape->GetStorageShape();
    if (xShape.GetDimNum() != 4 || wShape.GetDimNum() != 4 || convBShape.GetDimNum() != 1 || addBShape.GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const float *minValuePtr = attrs->GetAttrPointer<float>(0);
    const float *scalingFactorPtr = attrs->GetAttrPointer<float>(1);
    if (minValuePtr == nullptr || scalingFactorPtr == nullptr) {
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

    if (inChannels != weightInChannels || convBShape.GetDim(0) != static_cast<int64_t>(outChannels)) {
        return ge::GRAPH_FAILED;
    }
    if (addBShape.GetDim(0) != static_cast<int64_t>(outChannels) || addBShape.GetDim(1) != 1 || addBShape.GetDim(2) != 1) {
        return ge::GRAPH_FAILED;
    }

    Conv2dMinAddMultiplyCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_kernelHeight(kernelHeight);
    tiling.set_kernelWidth(kernelWidth);
    tiling.set_outputHeight(ComputeConvOutputDim(inputHeight, kernelHeight));
    tiling.set_outputWidth(ComputeConvOutputDim(inputWidth, kernelWidth));
    tiling.set_minValue(*minValuePtr);
    tiling.set_scalingFactor(*scalingFactorPtr);

    const uint32_t blockDim = batchSize == 0 ? 1 : (batchSize > 8 ? 8 : batchSize);
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
    const gert::Shape *convBiasShape = context->GetInputShape(2);
    const gert::Shape *addBiasShape = context->GetInputShape(3);
    if (inputShape == nullptr || weightShape == nullptr || convBiasShape == nullptr || addBiasShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDimNum() != 4 || weightShape->GetDimNum() != 4 || convBiasShape->GetDimNum() != 1 || addBiasShape->GetDimNum() != 3) {
        return GRAPH_FAILED;
    }

    if (inputShape->GetDim(1) != weightShape->GetDim(1) || weightShape->GetDim(0) != convBiasShape->GetDim(0)) {
        return GRAPH_FAILED;
    }
    if (addBiasShape->GetDim(0) != weightShape->GetDim(0) || addBiasShape->GetDim(1) != 1 || addBiasShape->GetDim(2) != 1) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(4);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, weightShape->GetDim(0));
    outputShape->SetDim(2, ComputeConvOutputDim(inputShape->GetDim(2), weightShape->GetDim(2)));
    outputShape->SetDim(3, ComputeConvOutputDim(inputShape->GetDim(3), weightShape->GetDim(3)));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Conv2dMinAddMultiplyCustom : public OpDef {
public:
    explicit Conv2dMinAddMultiplyCustom(const char *name) : OpDef(name)
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
        this->Input("add_bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("min_value").AttrType(REQUIRED).Float();
        this->Attr("scaling_factor").AttrType(REQUIRED).Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv2dMinAddMultiplyCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConv2dMinAddMultiply {
public:
    __aicore__ inline KernelConv2dMinAddMultiply() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR convBias,
        GM_ADDR addBias,
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
        float minValue,
        float scalingFactor)
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
        this->minValue = minValue;
        this->scalingFactor = scalingFactor;
        this->blockIdx = GetBlockIdx();
        this->blockNum = GetBlockNum();
        this->inputChannelStride = inputHeight * inputWidth;
        this->inputBatchStride = inChannels * this->inputChannelStride;
        this->weightOutStride = inChannels * kernelHeight * kernelWidth;
        this->weightInStride = kernelHeight * kernelWidth;
        this->outputChannelStride = outputHeight * outputWidth;
        this->outputBatchStride = outChannels * this->outputChannelStride;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * this->inputBatchStride);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, outChannels * this->weightOutStride);
        convBiasGm.SetGlobalBuffer((__gm__ float *)convBias, outChannels);
        addBiasGm.SetGlobalBuffer((__gm__ float *)addBias, outChannels);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * this->outputBatchStride);
    }

    __aicore__ inline void Process()
    {
        if (this->outputHeight == 0 || this->outputWidth == 0) {
            return;
        }

        for (uint32_t batchIdx = this->blockIdx; batchIdx < this->batchSize; batchIdx += this->blockNum) {
            const uint32_t inputBatchBase = batchIdx * this->inputBatchStride;
            const uint32_t outputBatchBase = batchIdx * this->outputBatchStride;
            for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
                const uint32_t weightBase = outChannel * this->weightOutStride;
                const float convBiasValue = convBiasGm.GetValue(outChannel);
                const float addBiasValue = addBiasGm.GetValue(outChannel);
                const uint32_t outputChannelBase = outputBatchBase + outChannel * this->outputChannelStride;

                for (uint32_t outH = 0; outH < this->outputHeight; ++outH) {
                    for (uint32_t outW = 0; outW < this->outputWidth; ++outW) {
                        float value = convBiasValue;
                        for (uint32_t inChannel = 0; inChannel < this->inChannels; ++inChannel) {
                            const uint32_t inputChannelBase = inputBatchBase + inChannel * this->inputChannelStride;
                            const uint32_t weightChannelBase = weightBase + inChannel * this->weightInStride;
                            for (uint32_t kernelH = 0; kernelH < this->kernelHeight; ++kernelH) {
                                const uint32_t inputRowBase = inputChannelBase + (outH + kernelH) * this->inputWidth;
                                const uint32_t weightRowBase = weightChannelBase + kernelH * this->kernelWidth;
                                for (uint32_t kernelW = 0; kernelW < this->kernelWidth; ++kernelW) {
                                    value += xGm.GetValue(inputRowBase + outW + kernelW) *
                                        weightGm.GetValue(weightRowBase + kernelW);
                                }
                            }
                        }

                        if (value > this->minValue) {
                            value = this->minValue;
                        }
                        value = (value + addBiasValue) * this->scalingFactor;
                        yGm.SetValue(outputChannelBase + outH * this->outputWidth + outW, value);
                    }
                }
            }
        }
    }

private:
    GlobalTensor<float> xGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> convBiasGm;
    GlobalTensor<float> addBiasGm;
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
    uint32_t blockIdx;
    uint32_t blockNum;
    uint32_t inputChannelStride;
    uint32_t inputBatchStride;
    uint32_t weightOutStride;
    uint32_t weightInStride;
    uint32_t outputChannelStride;
    uint32_t outputBatchStride;
    float minValue;
    float scalingFactor;
};

extern "C" __global__ __aicore__ void conv2d_min_add_multiply_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR convBias,
    GM_ADDR addBias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv2dMinAddMultiply op;
    op.Init(
        x,
        weight,
        convBias,
        addBias,
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
        tiling_data.minValue,
        tiling_data.scalingFactor);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor conv2d_min_add_multiply_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &conv_bias,
    const at::Tensor &add_bias,
    double min_value,
    double scaling_factor)
{
    TORCH_CHECK(x.dim() == 4, "x must be a 4D NCHW tensor");
    TORCH_CHECK(weight.dim() == 4, "weight must be a 4D OIHW tensor");
    TORCH_CHECK(conv_bias.dim() == 1, "conv_bias must be a 1D tensor");
    TORCH_CHECK(add_bias.dim() == 3, "add_bias must be a 3D tensor shaped [C,1,1]");
    TORCH_CHECK(x.size(1) == weight.size(1), "input channels must match weight.size(1)");
    TORCH_CHECK(weight.size(0) == conv_bias.size(0), "conv_bias length must match output channels");
    TORCH_CHECK(add_bias.size(0) == weight.size(0), "add_bias channels must match output channels");
    TORCH_CHECK(add_bias.size(1) == 1 && add_bias.size(2) == 1, "add_bias must have shape [C,1,1]");

    const int64_t outputHeight = x.size(2) - weight.size(2) + 1;
    const int64_t outputWidth = x.size(3) - weight.size(3) + 1;
    TORCH_CHECK(outputHeight >= 0 && outputWidth >= 0, "invalid convolution output shape");

    at::Tensor result = at::empty({x.size(0), weight.size(0), outputHeight, outputWidth}, x.options());
    EXEC_NPU_CMD(
        aclnnConv2dMinAddMultiplyCustom,
        x,
        weight,
        conv_bias,
        add_bias,
        min_value,
        scaling_factor,
        result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv2d_min_add_multiply_custom", &conv2d_min_add_multiply_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv2d_min_add_multiply_custom",
        &conv2d_min_add_multiply_custom_impl_npu,
        "conv2d_min_add_multiply_custom");
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
        constant_value: float,
        bias_shape,
        scaling_factor: float,
    ) -> None:
        super().__init__()
        self.constant_value = float(constant_value)
        self.scaling_factor = float(scaling_factor)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = torch.nn.Parameter(torch.randn(bias_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv2d_min_add_multiply_custom(
            x,
            self.conv.weight,
            self.conv.bias,
            self.bias,
            self.constant_value,
            self.scaling_factor,
        )
'''
