project_json_src='''
[
    {
        "op": "Conv2dSubtractHardSwishMaxPoolMishCustom",
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
        ],
        "attr": [
            {
                "name": "subtract_value",
                "param_type": "required",
                "type": "float"
            },
            {
                "name": "pool_kernel_size",
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
BEGIN_TILING_DATA_DEF(Conv2dSubtractHardSwishMaxPoolMishCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inChannels);
    TILING_DATA_FIELD_DEF(uint32_t, outChannels);
    TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, kernelHeight);
    TILING_DATA_FIELD_DEF(uint32_t, kernelWidth);
    TILING_DATA_FIELD_DEF(uint32_t, convOutputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, convOutputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, poolKernelSize);
    TILING_DATA_FIELD_DEF(float, subtractValue);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    Conv2dSubtractHardSwishMaxPoolMishCustom,
    Conv2dSubtractHardSwishMaxPoolMishCustomTilingData)
}
"""

host_operator_src="""
#include "conv2d_subtract_hard_swish_max_pool_mish_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
uint32_t ComputeConvOutputDim(int64_t input, int64_t kernel)
{
    if (input < kernel || kernel <= 0) {
        return 0;
    }
    return static_cast<uint32_t>(input - kernel + 1);
}

uint32_t ComputePoolOutputDim(int64_t input, int64_t kernel)
{
    if (input < kernel || kernel <= 0) {
        return 0;
    }
    return static_cast<uint32_t>((input - kernel) / kernel + 1);
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
    if (xShape.GetDimNum() != 4 || wShape.GetDimNum() != 4 || bShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const float *subtractValuePtr = attrs->GetAttrPointer<float>(0);
    const int64_t *poolKernelSizePtr = attrs->GetAttrPointer<int64_t>(1);
    if (subtractValuePtr == nullptr || poolKernelSizePtr == nullptr) {
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
    const uint32_t biasSize = static_cast<uint32_t>(bShape.GetDim(0));
    const uint32_t poolKernelSize = static_cast<uint32_t>(*poolKernelSizePtr);

    if (inChannels != weightInChannels || biasSize != outChannels || poolKernelSize == 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t convOutputHeight = ComputeConvOutputDim(inputHeight, kernelHeight);
    const uint32_t convOutputWidth = ComputeConvOutputDim(inputWidth, kernelWidth);
    const uint32_t outputHeight = ComputePoolOutputDim(convOutputHeight, poolKernelSize);
    const uint32_t outputWidth = ComputePoolOutputDim(convOutputWidth, poolKernelSize);

    Conv2dSubtractHardSwishMaxPoolMishCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_kernelHeight(kernelHeight);
    tiling.set_kernelWidth(kernelWidth);
    tiling.set_convOutputHeight(convOutputHeight);
    tiling.set_convOutputWidth(convOutputWidth);
    tiling.set_outputHeight(outputHeight);
    tiling.set_outputWidth(outputWidth);
    tiling.set_poolKernelSize(poolKernelSize);
    tiling.set_subtractValue(*subtractValuePtr);

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
    const gert::Shape *biasShape = context->GetInputShape(2);
    if (inputShape == nullptr || weightShape == nullptr || biasShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDimNum() != 4 || weightShape->GetDimNum() != 4 || biasShape->GetDimNum() != 1) {
        return GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return GRAPH_FAILED;
    }
    const int64_t *poolKernelSizePtr = attrs->GetAttrPointer<int64_t>(1);
    if (poolKernelSizePtr == nullptr || *poolKernelSizePtr <= 0) {
        return GRAPH_FAILED;
    }

    if (inputShape->GetDim(1) != weightShape->GetDim(1) || weightShape->GetDim(0) != biasShape->GetDim(0)) {
        return GRAPH_FAILED;
    }

    const int64_t convOutputHeight = ComputeConvOutputDim(inputShape->GetDim(2), weightShape->GetDim(2));
    const int64_t convOutputWidth = ComputeConvOutputDim(inputShape->GetDim(3), weightShape->GetDim(3));
    const int64_t outputHeight = ComputePoolOutputDim(convOutputHeight, *poolKernelSizePtr);
    const int64_t outputWidth = ComputePoolOutputDim(convOutputWidth, *poolKernelSizePtr);

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(4);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, weightShape->GetDim(0));
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
class Conv2dSubtractHardSwishMaxPoolMishCustom : public OpDef {
public:
    explicit Conv2dSubtractHardSwishMaxPoolMishCustom(const char *name) : OpDef(name)
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
        this->Attr("subtract_value").AttrType(REQUIRED).Float();
        this->Attr("pool_kernel_size").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv2dSubtractHardSwishMaxPoolMishCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConv2dSubtractHardSwishMaxPoolMish {
public:
    __aicore__ inline KernelConv2dSubtractHardSwishMaxPoolMish() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t inChannels,
        uint32_t outChannels,
        uint32_t inputHeight,
        uint32_t inputWidth,
        uint32_t kernelHeight,
        uint32_t kernelWidth,
        uint32_t convOutputHeight,
        uint32_t convOutputWidth,
        uint32_t outputHeight,
        uint32_t outputWidth,
        uint32_t poolKernelSize,
        float subtractValue)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->inputHeight = inputHeight;
        this->inputWidth = inputWidth;
        this->kernelHeight = kernelHeight;
        this->kernelWidth = kernelWidth;
        this->convOutputHeight = convOutputHeight;
        this->convOutputWidth = convOutputWidth;
        this->outputHeight = outputHeight;
        this->outputWidth = outputWidth;
        this->poolKernelSize = poolKernelSize;
        this->subtractValue = subtractValue;
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
        biasGm.SetGlobalBuffer((__gm__ float *)bias, outChannels);
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
                const float biasValue = biasGm.GetValue(outChannel);
                const uint32_t outputChannelBase = outputBatchBase + outChannel * this->outputChannelStride;
                for (uint32_t outH = 0; outH < this->outputHeight; ++outH) {
                    const uint32_t poolStartH = outH * this->poolKernelSize;
                    for (uint32_t outW = 0; outW < this->outputWidth; ++outW) {
                        const uint32_t poolStartW = outW * this->poolKernelSize;
                        float maxValue = 0.0f;
                        bool hasValue = false;
                        for (uint32_t poolH = 0; poolH < this->poolKernelSize; ++poolH) {
                            const uint32_t convH = poolStartH + poolH;
                            for (uint32_t poolW = 0; poolW < this->poolKernelSize; ++poolW) {
                                const uint32_t convW = poolStartW + poolW;
                                float convValue = biasValue;
                                for (uint32_t inChannel = 0; inChannel < this->inChannels; ++inChannel) {
                                    const uint32_t inputChannelBase =
                                        inputBatchBase + inChannel * this->inputChannelStride;
                                    const uint32_t weightChannelBase =
                                        weightBase + inChannel * this->weightInStride;
                                    for (uint32_t kernelH = 0; kernelH < this->kernelHeight; ++kernelH) {
                                        const uint32_t inH = convH + kernelH;
                                        const uint32_t inputRowBase =
                                            inputChannelBase + inH * this->inputWidth;
                                        const uint32_t weightRowBase =
                                            weightChannelBase + kernelH * this->kernelWidth;
                                        for (uint32_t kernelW = 0; kernelW < this->kernelWidth; ++kernelW) {
                                            convValue += xGm.GetValue(inputRowBase + convW + kernelW) *
                                                weightGm.GetValue(weightRowBase + kernelW);
                                        }
                                    }
                                }

                                const float shifted = convValue - this->subtractValue;
                                const float hardSwish = ApplyHardSwish(shifted);
                                if (!hasValue || hardSwish > maxValue) {
                                    maxValue = hardSwish;
                                    hasValue = true;
                                }
                            }
                        }
                        yGm.SetValue(outputChannelBase + outH * this->outputWidth + outW, ApplyMish(maxValue));
                    }
                }
            }
        }
    }

private:
    __aicore__ inline float ApplyHardSwish(float x) const
    {
        float gate = x + 3.0f;
        if (gate < 0.0f) {
            gate = 0.0f;
        } else if (gate > 6.0f) {
            gate = 6.0f;
        }
        return x * gate * (1.0f / 6.0f);
    }

    __aicore__ inline float ApplyMish(float x) const
    {
        if (x > 4.0f) {
            return x;
        }
        float clamped = x;
        if (clamped < -0.375f) {
            clamped = -0.375f;
        }

        float y = 4.28528732e-04f;
        y = y * clamped - 6.37554986e-03f;
        y = y * clamped + 3.54121299e-02f;
        y = y * clamped - 7.90083932e-02f;
        y = y * clamped - 8.21652298e-03f;
        y = y * clamped + 3.24137046e-01f;
        y = y * clamped + 5.98774953e-01f;
        y = y * clamped - 9.63318644e-05f;
        return y;
    }

private:
    GlobalTensor<float> xGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> biasGm;
    GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t kernelHeight;
    uint32_t kernelWidth;
    uint32_t convOutputHeight;
    uint32_t convOutputWidth;
    uint32_t outputHeight;
    uint32_t outputWidth;
    uint32_t poolKernelSize;
    uint32_t blockIdx;
    uint32_t blockNum;
    uint32_t inputChannelStride;
    uint32_t inputBatchStride;
    uint32_t weightOutStride;
    uint32_t weightInStride;
    uint32_t outputChannelStride;
    uint32_t outputBatchStride;
    float subtractValue;
};

extern "C" __global__ __aicore__ void conv2d_subtract_hard_swish_max_pool_mish_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv2dSubtractHardSwishMaxPoolMish op;
    op.Init(
        x,
        weight,
        bias,
        y,
        tiling_data.batchSize,
        tiling_data.inChannels,
        tiling_data.outChannels,
        tiling_data.inputHeight,
        tiling_data.inputWidth,
        tiling_data.kernelHeight,
        tiling_data.kernelWidth,
        tiling_data.convOutputHeight,
        tiling_data.convOutputWidth,
        tiling_data.outputHeight,
        tiling_data.outputWidth,
        tiling_data.poolKernelSize,
        tiling_data.subtractValue);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor conv2d_subtract_hard_swish_max_pool_mish_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &bias,
    double subtract_value,
    int64_t pool_kernel_size)
{
    TORCH_CHECK(x.dim() == 4, "x must be a 4D NCHW tensor");
    TORCH_CHECK(weight.dim() == 4, "weight must be a 4D OIHW tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
    TORCH_CHECK(x.size(1) == weight.size(1), "input channels must match weight.size(1)");
    TORCH_CHECK(weight.size(0) == bias.size(0), "bias length must match output channels");
    TORCH_CHECK(pool_kernel_size > 0, "pool_kernel_size must be positive");

    const int64_t convOutputHeight = x.size(2) - weight.size(2) + 1;
    const int64_t convOutputWidth = x.size(3) - weight.size(3) + 1;
    TORCH_CHECK(convOutputHeight >= 0 && convOutputWidth >= 0, "invalid convolution output shape");
    const int64_t outputHeight = convOutputHeight < pool_kernel_size ? 0 : (convOutputHeight - pool_kernel_size) / pool_kernel_size + 1;
    const int64_t outputWidth = convOutputWidth < pool_kernel_size ? 0 : (convOutputWidth - pool_kernel_size) / pool_kernel_size + 1;

    at::Tensor result = at::empty({x.size(0), weight.size(0), outputHeight, outputWidth}, x.options());
    EXEC_NPU_CMD(
        aclnnConv2dSubtractHardSwishMaxPoolMishCustom,
        x,
        weight,
        bias,
        subtract_value,
        pool_kernel_size,
        result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "conv2d_subtract_hard_swish_max_pool_mish_custom",
        &conv2d_subtract_hard_swish_max_pool_mish_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv2d_subtract_hard_swish_max_pool_mish_custom",
        &conv2d_subtract_hard_swish_max_pool_mish_custom_impl_npu,
        "conv2d_subtract_hard_swish_max_pool_mish_custom");
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
        subtract_value: float,
        pool_kernel_size: int,
    ) -> None:
        super().__init__()
        self.subtract_value = float(subtract_value)
        self.pool_kernel_size = int(pool_kernel_size)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv2d_subtract_hard_swish_max_pool_mish_custom(
            x,
            self.conv.weight,
            self.conv.bias,
            self.subtract_value,
            self.pool_kernel_size,
        )
'''
