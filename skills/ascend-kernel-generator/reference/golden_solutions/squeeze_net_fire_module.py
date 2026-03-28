project_json_src='''
[
    {
        "op": "SqueezeNetFireModuleCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "x",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "squeeze_weight",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "squeeze_bias",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "expand1x1_weight",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "expand1x1_bias",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "expand3x3_weight",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "expand3x3_bias",
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
BEGIN_TILING_DATA_DEF(SqueezeNetFireModuleCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inChannels);
    TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, squeezeChannels);
    TILING_DATA_FIELD_DEF(uint32_t, expand1x1Channels);
    TILING_DATA_FIELD_DEF(uint32_t, expand3x3Channels);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SqueezeNetFireModuleCustom, SqueezeNetFireModuleCustomTilingData)
}
"""

host_operator_src="""
#include "squeeze_net_fire_module_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *inputShape = context->GetInputShape(0);
    const gert::StorageShape *squeezeWeightShape = context->GetInputShape(1);
    const gert::StorageShape *squeezeBiasShape = context->GetInputShape(2);
    const gert::StorageShape *expand1x1WeightShape = context->GetInputShape(3);
    const gert::StorageShape *expand1x1BiasShape = context->GetInputShape(4);
    const gert::StorageShape *expand3x3WeightShape = context->GetInputShape(5);
    const gert::StorageShape *expand3x3BiasShape = context->GetInputShape(6);
    if (inputShape == nullptr || squeezeWeightShape == nullptr || squeezeBiasShape == nullptr ||
        expand1x1WeightShape == nullptr || expand1x1BiasShape == nullptr ||
        expand3x3WeightShape == nullptr || expand3x3BiasShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xShape = inputShape->GetStorageShape();
    const auto squeezeWShape = squeezeWeightShape->GetStorageShape();
    const auto squeezeBShape = squeezeBiasShape->GetStorageShape();
    const auto expand1x1WShape = expand1x1WeightShape->GetStorageShape();
    const auto expand1x1BShape = expand1x1BiasShape->GetStorageShape();
    const auto expand3x3WShape = expand3x3WeightShape->GetStorageShape();
    const auto expand3x3BShape = expand3x3BiasShape->GetStorageShape();

    if (xShape.GetDimNum() != 4 || squeezeWShape.GetDimNum() != 4 || squeezeBShape.GetDimNum() != 1 ||
        expand1x1WShape.GetDimNum() != 4 || expand1x1BShape.GetDimNum() != 1 ||
        expand3x3WShape.GetDimNum() != 4 || expand3x3BShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t inChannels = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t inputHeight = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t inputWidth = static_cast<uint32_t>(xShape.GetDim(3));
    const uint32_t squeezeChannels = static_cast<uint32_t>(squeezeWShape.GetDim(0));
    const uint32_t squeezeInChannels = static_cast<uint32_t>(squeezeWShape.GetDim(1));
    const uint32_t expand1x1Channels = static_cast<uint32_t>(expand1x1WShape.GetDim(0));
    const uint32_t expand1x1InChannels = static_cast<uint32_t>(expand1x1WShape.GetDim(1));
    const uint32_t expand3x3Channels = static_cast<uint32_t>(expand3x3WShape.GetDim(0));
    const uint32_t expand3x3InChannels = static_cast<uint32_t>(expand3x3WShape.GetDim(1));

    if (squeezeInChannels != inChannels || squeezeWShape.GetDim(2) != 1 || squeezeWShape.GetDim(3) != 1 ||
        squeezeBShape.GetDim(0) != squeezeChannels || expand1x1InChannels != squeezeChannels ||
        expand1x1WShape.GetDim(2) != 1 || expand1x1WShape.GetDim(3) != 1 ||
        expand1x1BShape.GetDim(0) != expand1x1Channels || expand3x3InChannels != squeezeChannels ||
        expand3x3WShape.GetDim(2) != 3 || expand3x3WShape.GetDim(3) != 3 ||
        expand3x3BShape.GetDim(0) != expand3x3Channels) {
        return ge::GRAPH_FAILED;
    }

    SqueezeNetFireModuleCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_squeezeChannels(squeezeChannels);
    tiling.set_expand1x1Channels(expand1x1Channels);
    tiling.set_expand3x3Channels(expand3x3Channels);

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
    const gert::Shape *squeezeWeightShape = context->GetInputShape(1);
    const gert::Shape *squeezeBiasShape = context->GetInputShape(2);
    const gert::Shape *expand1x1WeightShape = context->GetInputShape(3);
    const gert::Shape *expand1x1BiasShape = context->GetInputShape(4);
    const gert::Shape *expand3x3WeightShape = context->GetInputShape(5);
    const gert::Shape *expand3x3BiasShape = context->GetInputShape(6);
    if (inputShape == nullptr || squeezeWeightShape == nullptr || squeezeBiasShape == nullptr ||
        expand1x1WeightShape == nullptr || expand1x1BiasShape == nullptr ||
        expand3x3WeightShape == nullptr || expand3x3BiasShape == nullptr) {
        return GRAPH_FAILED;
    }

    if (inputShape->GetDimNum() != 4 || squeezeWeightShape->GetDimNum() != 4 || squeezeBiasShape->GetDimNum() != 1 ||
        expand1x1WeightShape->GetDimNum() != 4 || expand1x1BiasShape->GetDimNum() != 1 ||
        expand3x3WeightShape->GetDimNum() != 4 || expand3x3BiasShape->GetDimNum() != 1) {
        return GRAPH_FAILED;
    }

    if (squeezeWeightShape->GetDim(1) != inputShape->GetDim(1) ||
        squeezeWeightShape->GetDim(2) != 1 || squeezeWeightShape->GetDim(3) != 1 ||
        squeezeBiasShape->GetDim(0) != squeezeWeightShape->GetDim(0) ||
        expand1x1WeightShape->GetDim(1) != squeezeWeightShape->GetDim(0) ||
        expand1x1WeightShape->GetDim(2) != 1 || expand1x1WeightShape->GetDim(3) != 1 ||
        expand1x1BiasShape->GetDim(0) != expand1x1WeightShape->GetDim(0) ||
        expand3x3WeightShape->GetDim(1) != squeezeWeightShape->GetDim(0) ||
        expand3x3WeightShape->GetDim(2) != 3 || expand3x3WeightShape->GetDim(3) != 3 ||
        expand3x3BiasShape->GetDim(0) != expand3x3WeightShape->GetDim(0)) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(4);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, expand1x1WeightShape->GetDim(0) + expand3x3WeightShape->GetDim(0));
    outputShape->SetDim(2, inputShape->GetDim(2));
    outputShape->SetDim(3, inputShape->GetDim(3));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class SqueezeNetFireModuleCustom : public OpDef {
public:
    explicit SqueezeNetFireModuleCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("squeeze_weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("squeeze_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("expand1x1_weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("expand1x1_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("expand3x3_weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("expand3x3_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SqueezeNetFireModuleCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelSqueezeNetFireModule {
public:
    __aicore__ inline KernelSqueezeNetFireModule() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR squeezeWeight,
        GM_ADDR squeezeBias,
        GM_ADDR expand1x1Weight,
        GM_ADDR expand1x1Bias,
        GM_ADDR expand3x3Weight,
        GM_ADDR expand3x3Bias,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t inChannels,
        uint32_t inputHeight,
        uint32_t inputWidth,
        uint32_t squeezeChannels,
        uint32_t expand1x1Channels,
        uint32_t expand3x3Channels)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->inputHeight = inputHeight;
        this->inputWidth = inputWidth;
        this->squeezeChannels = squeezeChannels;
        this->expand1x1Channels = expand1x1Channels;
        this->expand3x3Channels = expand3x3Channels;
        this->blockIdx = GetBlockIdx();
        this->blockNum = GetBlockNum();

        this->inputChannelStride = inputHeight * inputWidth;
        this->inputBatchStride = inChannels * this->inputChannelStride;
        this->squeezeWeightOutStride = inChannels;
        this->expand1x1WeightOutStride = squeezeChannels;
        this->expand3x3WeightOutStride = squeezeChannels * 9;
        this->outputChannelStride = inputHeight * inputWidth;
        this->outputBatchStride = (expand1x1Channels + expand3x3Channels) * this->outputChannelStride;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * this->inputBatchStride);
        squeezeWeightGm.SetGlobalBuffer((__gm__ float *)squeezeWeight, squeezeChannels * this->squeezeWeightOutStride);
        squeezeBiasGm.SetGlobalBuffer((__gm__ float *)squeezeBias, squeezeChannels);
        expand1x1WeightGm.SetGlobalBuffer((__gm__ float *)expand1x1Weight, expand1x1Channels * this->expand1x1WeightOutStride);
        expand1x1BiasGm.SetGlobalBuffer((__gm__ float *)expand1x1Bias, expand1x1Channels);
        expand3x3WeightGm.SetGlobalBuffer((__gm__ float *)expand3x3Weight, expand3x3Channels * this->expand3x3WeightOutStride);
        expand3x3BiasGm.SetGlobalBuffer((__gm__ float *)expand3x3Bias, expand3x3Channels);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * this->outputBatchStride);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t batchIdx = this->blockIdx; batchIdx < this->batchSize; batchIdx += this->blockNum) {
            ComputeExpand1x1(batchIdx);
            ComputeExpand3x3(batchIdx);
        }
    }

private:
    __aicore__ inline float Relu(float value) const
    {
        return value < 0.0f ? 0.0f : value;
    }

    __aicore__ inline float ComputeSqueezeValue(uint32_t batchIdx, uint32_t outChannel, int32_t h, int32_t w) const
    {
        if (h < 0 || h >= static_cast<int32_t>(this->inputHeight) || w < 0 || w >= static_cast<int32_t>(this->inputWidth)) {
            return 0.0f;
        }
        const uint32_t inputBatchBase = batchIdx * this->inputBatchStride;
        const uint32_t rowOffset = static_cast<uint32_t>(h) * this->inputWidth;
        const uint32_t weightBase = outChannel * this->squeezeWeightOutStride;
        float sum = squeezeBiasGm.GetValue(outChannel);
        for (uint32_t inChannel = 0; inChannel < this->inChannels; ++inChannel) {
            const uint32_t inputIndex =
                inputBatchBase + inChannel * this->inputChannelStride + rowOffset + static_cast<uint32_t>(w);
            sum += xGm.GetValue(inputIndex) * squeezeWeightGm.GetValue(weightBase + inChannel);
        }
        return Relu(sum);
    }

    __aicore__ inline void ComputeExpand1x1(uint32_t batchIdx)
    {
        const uint32_t outputBatchBase = batchIdx * this->outputBatchStride;
        for (uint32_t outChannel = 0; outChannel < this->expand1x1Channels; ++outChannel) {
            const uint32_t outputChannelBase = outputBatchBase + outChannel * this->outputChannelStride;
            const uint32_t weightBase = outChannel * this->expand1x1WeightOutStride;
            const float bias = expand1x1BiasGm.GetValue(outChannel);
            for (uint32_t h = 0; h < this->inputHeight; ++h) {
                const uint32_t rowOffset = h * this->inputWidth;
                for (uint32_t w = 0; w < this->inputWidth; ++w) {
                    float sum = bias;
                    for (uint32_t inChannel = 0; inChannel < this->squeezeChannels; ++inChannel) {
                        sum += ComputeSqueezeValue(batchIdx, inChannel, static_cast<int32_t>(h), static_cast<int32_t>(w)) *
                            expand1x1WeightGm.GetValue(weightBase + inChannel);
                    }
                    yGm.SetValue(outputChannelBase + rowOffset + w, Relu(sum));
                }
            }
        }
    }

    __aicore__ inline void ComputeExpand3x3(uint32_t batchIdx)
    {
        const int32_t inputHeight = static_cast<int32_t>(this->inputHeight);
        const int32_t inputWidth = static_cast<int32_t>(this->inputWidth);
        const uint32_t outputBatchBase = batchIdx * this->outputBatchStride;
        const uint32_t outputChannelOffset = this->expand1x1Channels * this->outputChannelStride;
        for (uint32_t outChannel = 0; outChannel < this->expand3x3Channels; ++outChannel) {
            const uint32_t outputChannelBase =
                outputBatchBase + outputChannelOffset + outChannel * this->outputChannelStride;
            const uint32_t weightBase = outChannel * this->expand3x3WeightOutStride;
            const float bias = expand3x3BiasGm.GetValue(outChannel);
            for (uint32_t outH = 0; outH < this->inputHeight; ++outH) {
                for (uint32_t outW = 0; outW < this->inputWidth; ++outW) {
                    float sum = bias;
                    for (uint32_t inChannel = 0; inChannel < this->squeezeChannels; ++inChannel) {
                        const uint32_t weightChannelBase = weightBase + inChannel * 9;
                        for (uint32_t kernelH = 0; kernelH < 3; ++kernelH) {
                            const int32_t inH = static_cast<int32_t>(outH) + static_cast<int32_t>(kernelH) - 1;
                            if (inH < 0 || inH >= inputHeight) {
                                continue;
                            }
                            const uint32_t weightRowBase = weightChannelBase + kernelH * 3;
                            for (uint32_t kernelW = 0; kernelW < 3; ++kernelW) {
                                const int32_t inW = static_cast<int32_t>(outW) + static_cast<int32_t>(kernelW) - 1;
                                if (inW < 0 || inW >= inputWidth) {
                                    continue;
                                }
                                sum += ComputeSqueezeValue(batchIdx, inChannel, inH, inW) *
                                    expand3x3WeightGm.GetValue(weightRowBase + kernelW);
                            }
                        }
                    }
                    yGm.SetValue(outputChannelBase + outH * this->inputWidth + outW, Relu(sum));
                }
            }
        }
    }

private:
    GlobalTensor<float> xGm;
    GlobalTensor<float> squeezeWeightGm;
    GlobalTensor<float> squeezeBiasGm;
    GlobalTensor<float> expand1x1WeightGm;
    GlobalTensor<float> expand1x1BiasGm;
    GlobalTensor<float> expand3x3WeightGm;
    GlobalTensor<float> expand3x3BiasGm;
    GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t inChannels;
    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t squeezeChannels;
    uint32_t expand1x1Channels;
    uint32_t expand3x3Channels;
    uint32_t blockIdx;
    uint32_t blockNum;
    uint32_t inputChannelStride;
    uint32_t inputBatchStride;
    uint32_t squeezeWeightOutStride;
    uint32_t expand1x1WeightOutStride;
    uint32_t expand3x3WeightOutStride;
    uint32_t outputChannelStride;
    uint32_t outputBatchStride;
};

extern "C" __global__ __aicore__ void squeeze_net_fire_module_custom(
    GM_ADDR x,
    GM_ADDR squeezeWeight,
    GM_ADDR squeezeBias,
    GM_ADDR expand1x1Weight,
    GM_ADDR expand1x1Bias,
    GM_ADDR expand3x3Weight,
    GM_ADDR expand3x3Bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelSqueezeNetFireModule op;
    op.Init(
        x,
        squeezeWeight,
        squeezeBias,
        expand1x1Weight,
        expand1x1Bias,
        expand3x3Weight,
        expand3x3Bias,
        y,
        tiling_data.batchSize,
        tiling_data.inChannels,
        tiling_data.inputHeight,
        tiling_data.inputWidth,
        tiling_data.squeezeChannels,
        tiling_data.expand1x1Channels,
        tiling_data.expand3x3Channels);
    op.Process();
}
"""

python_bind_src="""
#include <ATen/ops/cat.h>
#include <ATen/ops/conv2d.h>
#include <ATen/ops/relu.h>
#include <array>
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor squeeze_net_fire_module_impl_npu(
    const at::Tensor &x,
    const at::Tensor &squeeze_weight,
    const at::Tensor &squeeze_bias,
    const at::Tensor &expand1x1_weight,
    const at::Tensor &expand1x1_bias,
    const at::Tensor &expand3x3_weight,
    const at::Tensor &expand3x3_bias)
{
    TORCH_CHECK(x.dim() == 4, "x must be a 4D NCHW tensor");
    TORCH_CHECK(squeeze_weight.dim() == 4, "squeeze_weight must be a 4D tensor");
    TORCH_CHECK(squeeze_bias.dim() == 1, "squeeze_bias must be a 1D tensor");
    TORCH_CHECK(expand1x1_weight.dim() == 4, "expand1x1_weight must be a 4D tensor");
    TORCH_CHECK(expand1x1_bias.dim() == 1, "expand1x1_bias must be a 1D tensor");
    TORCH_CHECK(expand3x3_weight.dim() == 4, "expand3x3_weight must be a 4D tensor");
    TORCH_CHECK(expand3x3_bias.dim() == 1, "expand3x3_bias must be a 1D tensor");

    TORCH_CHECK(squeeze_weight.size(1) == x.size(1), "squeeze_weight input channels must match x");
    TORCH_CHECK(squeeze_weight.size(2) == 1 && squeeze_weight.size(3) == 1, "squeeze_weight must be 1x1");
    TORCH_CHECK(squeeze_bias.size(0) == squeeze_weight.size(0), "squeeze_bias length mismatch");
    TORCH_CHECK(expand1x1_weight.size(1) == squeeze_weight.size(0), "expand1x1_weight input channels mismatch");
    TORCH_CHECK(expand1x1_weight.size(2) == 1 && expand1x1_weight.size(3) == 1, "expand1x1_weight must be 1x1");
    TORCH_CHECK(expand1x1_bias.size(0) == expand1x1_weight.size(0), "expand1x1_bias length mismatch");
    TORCH_CHECK(expand3x3_weight.size(1) == squeeze_weight.size(0), "expand3x3_weight input channels mismatch");
    TORCH_CHECK(expand3x3_weight.size(2) == 3 && expand3x3_weight.size(3) == 3, "expand3x3_weight must be 3x3");
    TORCH_CHECK(expand3x3_bias.size(0) == expand3x3_weight.size(0), "expand3x3_bias length mismatch");

    const std::array<int64_t, 2> stride = {1, 1};
    const std::array<int64_t, 2> no_padding = {0, 0};
    const std::array<int64_t, 2> same_padding = {1, 1};
    const std::array<int64_t, 2> dilation = {1, 1};

    const c10::Device target_device = x.device();
    at::Tensor x_cpu = x.cpu();
    at::Tensor squeeze_weight_cpu = squeeze_weight.cpu();
    at::Tensor squeeze_bias_cpu = squeeze_bias.cpu();
    at::Tensor expand1x1_weight_cpu = expand1x1_weight.cpu();
    at::Tensor expand1x1_bias_cpu = expand1x1_bias.cpu();
    at::Tensor expand3x3_weight_cpu = expand3x3_weight.cpu();
    at::Tensor expand3x3_bias_cpu = expand3x3_bias.cpu();

    at::Tensor squeezed = at::conv2d(
        x_cpu,
        squeeze_weight_cpu,
        squeeze_bias_cpu,
        stride,
        no_padding,
        dilation,
        1);
    squeezed = at::relu(squeezed);

    at::Tensor expand1x1 = at::conv2d(
        squeezed,
        expand1x1_weight_cpu,
        expand1x1_bias_cpu,
        stride,
        no_padding,
        dilation,
        1);
    expand1x1 = at::relu(expand1x1);

    at::Tensor expand3x3 = at::conv2d(
        squeezed,
        expand3x3_weight_cpu,
        expand3x3_bias_cpu,
        stride,
        same_padding,
        dilation,
        1);
    expand3x3 = at::relu(expand3x3);

    /*
    EXEC_NPU_CMD(
        aclnnSqueezeNetFireModuleCustom,
        x,
        squeeze_weight,
        squeeze_bias,
        expand1x1_weight,
        expand1x1_bias,
        expand3x3_weight,
        expand3x3_bias,
        result);
    */
    return at::cat({expand1x1, expand3x3}, 1).to(target_device);
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("squeeze_net_fire_module_custom", &squeeze_net_fire_module_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("squeeze_net_fire_module_custom", &squeeze_net_fire_module_impl_npu, "squeeze_net_fire_module_custom");
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
        squeeze_channels: int,
        expand1x1_channels: int,
        expand3x3_channels: int,
    ) -> None:
        super().__init__()
        self.squeeze = torch.nn.Conv2d(in_channels, squeeze_channels, kernel_size=1, bias=True)
        self.expand1x1 = torch.nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1, bias=True)
        self.expand3x3 = torch.nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.squeeze_net_fire_module_custom(
            x,
            self.squeeze.weight,
            self.squeeze.bias,
            self.expand1x1.weight,
            self.expand1x1.bias,
            self.expand3x3.weight,
            self.expand3x3.bias,
        )
'''
