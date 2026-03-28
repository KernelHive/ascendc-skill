project_json_src='''
[
    {
        "op": "ConvTranspose3dMaxMaxSumCoreCustom",
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
BEGIN_TILING_DATA_DEF(ConvTranspose3dMaxMaxSumCoreCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, channelCount);
    TILING_DATA_FIELD_DEF(uint32_t, inputDepth);
    TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, pool1Depth);
    TILING_DATA_FIELD_DEF(uint32_t, pool1Height);
    TILING_DATA_FIELD_DEF(uint32_t, pool1Width);
    TILING_DATA_FIELD_DEF(uint32_t, outputDepth);
    TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    ConvTranspose3dMaxMaxSumCoreCustom,
    ConvTranspose3dMaxMaxSumCoreCustomTilingData)
}
"""

host_operator_src="""
#include "conv_transpose3d_max_max_sum_core_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
uint32_t ComputePoolOutDim(int64_t input, int64_t kernel, int64_t stride)
{
    if (input < kernel) {
        return 0;
    }
    return static_cast<uint32_t>((input - kernel) / stride + 1);
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto shape = inputShape->GetStorageShape();
    if (shape.GetDimNum() != 5) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(shape.GetDim(0));
    const uint32_t channelCount = static_cast<uint32_t>(shape.GetDim(1));
    const uint32_t inputDepth = static_cast<uint32_t>(shape.GetDim(2));
    const uint32_t inputHeight = static_cast<uint32_t>(shape.GetDim(3));
    const uint32_t inputWidth = static_cast<uint32_t>(shape.GetDim(4));

    const uint32_t pool1Depth = ComputePoolOutDim(inputDepth, 2, 2);
    const uint32_t pool1Height = ComputePoolOutDim(inputHeight, 2, 2);
    const uint32_t pool1Width = ComputePoolOutDim(inputWidth, 2, 2);
    const uint32_t outputDepth = ComputePoolOutDim(pool1Depth, 3, 3);
    const uint32_t outputHeight = ComputePoolOutDim(pool1Height, 3, 3);
    const uint32_t outputWidth = ComputePoolOutDim(pool1Width, 3, 3);

    ConvTranspose3dMaxMaxSumCoreCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_channelCount(channelCount);
    tiling.set_inputDepth(inputDepth);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_pool1Depth(pool1Depth);
    tiling.set_pool1Height(pool1Height);
    tiling.set_pool1Width(pool1Width);
    tiling.set_outputDepth(outputDepth);
    tiling.set_outputHeight(outputHeight);
    tiling.set_outputWidth(outputWidth);

    context->SetBlockDim(batchSize > 0 ? batchSize : 1);
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
    if (inputShape == nullptr || inputShape->GetDimNum() != 5) {
        return GRAPH_FAILED;
    }

    const uint32_t pool1Depth = ComputePoolOutDim(inputShape->GetDim(2), 2, 2);
    const uint32_t pool1Height = ComputePoolOutDim(inputShape->GetDim(3), 2, 2);
    const uint32_t pool1Width = ComputePoolOutDim(inputShape->GetDim(4), 2, 2);
    const uint32_t outputDepth = ComputePoolOutDim(pool1Depth, 3, 3);
    const uint32_t outputHeight = ComputePoolOutDim(pool1Height, 3, 3);
    const uint32_t outputWidth = ComputePoolOutDim(pool1Width, 3, 3);

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(5);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, 1);
    outputShape->SetDim(2, outputDepth);
    outputShape->SetDim(3, outputHeight);
    outputShape->SetDim(4, outputWidth);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ConvTranspose3dMaxMaxSumCoreCustom : public OpDef {
public:
    explicit ConvTranspose3dMaxMaxSumCoreCustom(const char *name) : OpDef(name)
    {
        this->Input("x")
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

OP_ADD(ConvTranspose3dMaxMaxSumCoreCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kFirstPoolKernel = 2;
constexpr uint32_t kFirstPoolStride = 2;
constexpr uint32_t kSecondPoolKernel = 3;
constexpr uint32_t kSecondPoolStride = 3;
constexpr float kNegInf = -3.40282347e+38f;
}

class KernelConvTranspose3dMaxMaxSumCore {
public:
    __aicore__ inline KernelConvTranspose3dMaxMaxSumCore() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t channelCount,
        uint32_t inputDepth,
        uint32_t inputHeight,
        uint32_t inputWidth,
        uint32_t pool1Depth,
        uint32_t pool1Height,
        uint32_t pool1Width,
        uint32_t outputDepth,
        uint32_t outputHeight,
        uint32_t outputWidth)
    {
        this->batchSize = batchSize;
        this->channelCount = channelCount;
        this->inputDepth = inputDepth;
        this->inputHeight = inputHeight;
        this->inputWidth = inputWidth;
        this->pool1Depth = pool1Depth;
        this->pool1Height = pool1Height;
        this->pool1Width = pool1Width;
        this->outputDepth = outputDepth;
        this->outputHeight = outputHeight;
        this->outputWidth = outputWidth;
        this->blockIdx = GetBlockIdx();

        this->inputPlaneStride = inputHeight * inputWidth;
        this->inputChannelStride = inputDepth * this->inputPlaneStride;
        this->inputBatchStride = channelCount * this->inputChannelStride;
        this->outputPlaneStride = outputHeight * outputWidth;
        this->outputBatchStride = outputDepth * this->outputPlaneStride;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * this->inputBatchStride);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * this->outputBatchStride);
    }

    __aicore__ inline void Process()
    {
        if (this->blockIdx >= this->batchSize) {
            return;
        }

        const uint32_t batchInputBase = this->blockIdx * this->inputBatchStride;
        const uint32_t batchOutputBase = this->blockIdx * this->outputBatchStride;

        for (uint32_t outD = 0; outD < this->outputDepth; ++outD) {
            for (uint32_t outH = 0; outH < this->outputHeight; ++outH) {
                for (uint32_t outW = 0; outW < this->outputWidth; ++outW) {
                    float reducedSum = 0.0f;
                    for (uint32_t channel = 0; channel < this->channelCount; ++channel) {
                        const uint32_t channelBase = batchInputBase + channel * this->inputChannelStride;
                        float secondPoolMax = kNegInf;
                        bool secondPoolValid = false;

                        for (uint32_t pool1DOffset = 0; pool1DOffset < kSecondPoolKernel; ++pool1DOffset) {
                            const uint32_t pool1D = outD * kSecondPoolStride + pool1DOffset;
                            if (pool1D >= this->pool1Depth) {
                                continue;
                            }
                            for (uint32_t pool1HOffset = 0; pool1HOffset < kSecondPoolKernel; ++pool1HOffset) {
                                const uint32_t pool1H = outH * kSecondPoolStride + pool1HOffset;
                                if (pool1H >= this->pool1Height) {
                                    continue;
                                }
                                for (uint32_t pool1WOffset = 0; pool1WOffset < kSecondPoolKernel; ++pool1WOffset) {
                                    const uint32_t pool1W = outW * kSecondPoolStride + pool1WOffset;
                                    if (pool1W >= this->pool1Width) {
                                        continue;
                                    }

                                    float firstPoolMax = kNegInf;
                                    bool firstPoolValid = false;
                                    const uint32_t inputDBase = pool1D * kFirstPoolStride;
                                    const uint32_t inputHBase = pool1H * kFirstPoolStride;
                                    const uint32_t inputWBase = pool1W * kFirstPoolStride;

                                    for (uint32_t kd = 0; kd < kFirstPoolKernel; ++kd) {
                                        const uint32_t inD = inputDBase + kd;
                                        if (inD >= this->inputDepth) {
                                            continue;
                                        }
                                        for (uint32_t kh = 0; kh < kFirstPoolKernel; ++kh) {
                                            const uint32_t inH = inputHBase + kh;
                                            if (inH >= this->inputHeight) {
                                                continue;
                                            }
                                            for (uint32_t kw = 0; kw < kFirstPoolKernel; ++kw) {
                                                const uint32_t inW = inputWBase + kw;
                                                if (inW >= this->inputWidth) {
                                                    continue;
                                                }

                                                const uint32_t inputOffset =
                                                    channelBase +
                                                    inD * this->inputPlaneStride +
                                                    inH * this->inputWidth +
                                                    inW;
                                                const float value = xGm.GetValue(inputOffset);
                                                if (!firstPoolValid || value > firstPoolMax) {
                                                    firstPoolMax = value;
                                                    firstPoolValid = true;
                                                }
                                            }
                                        }
                                    }

                                    if (firstPoolValid && (!secondPoolValid || firstPoolMax > secondPoolMax)) {
                                        secondPoolMax = firstPoolMax;
                                        secondPoolValid = true;
                                    }
                                }
                            }
                        }

                        reducedSum += secondPoolValid ? secondPoolMax : 0.0f;
                    }

                    const uint32_t outputOffset =
                        batchOutputBase +
                        outD * this->outputPlaneStride +
                        outH * this->outputWidth +
                        outW;
                    yGm.SetValue(outputOffset, reducedSum);
                }
            }
        }
    }

private:
    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t channelCount;
    uint32_t inputDepth;
    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t pool1Depth;
    uint32_t pool1Height;
    uint32_t pool1Width;
    uint32_t outputDepth;
    uint32_t outputHeight;
    uint32_t outputWidth;
    uint32_t blockIdx;
    uint32_t inputPlaneStride;
    uint32_t inputChannelStride;
    uint32_t inputBatchStride;
    uint32_t outputPlaneStride;
    uint32_t outputBatchStride;
};

extern "C" __global__ __aicore__ void conv_transpose3d_max_max_sum_core_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTranspose3dMaxMaxSumCore op;
    op.Init(
        x,
        y,
        tiling_data.batchSize,
        tiling_data.channelCount,
        tiling_data.inputDepth,
        tiling_data.inputHeight,
        tiling_data.inputWidth,
        tiling_data.pool1Depth,
        tiling_data.pool1Height,
        tiling_data.pool1Width,
        tiling_data.outputDepth,
        tiling_data.outputHeight,
        tiling_data.outputWidth);
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
int64_t ComputeConvTransposeOutDim(
    int64_t input,
    int64_t kernel,
    int64_t stride,
    int64_t padding)
{
    return (input - 1) * stride - 2 * padding + kernel;
}

int64_t ComputePoolOutDim(int64_t input, int64_t kernel, int64_t stride)
{
    if (input < kernel) {
        return 0;
    }
    return (input - kernel) / stride + 1;
}
}

at::Tensor conv_transpose3d_max_max_sum_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &bias,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_d,
    int64_t padding_h,
    int64_t padding_w)
{
    TORCH_CHECK(x.dim() == 5, "x must be a 5D NCDHW tensor");
    TORCH_CHECK(weight.dim() == 5, "weight must be a 5D tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
    TORCH_CHECK(stride_d > 0 && stride_h > 0 && stride_w > 0, "stride must be positive");
    TORCH_CHECK(padding_d >= 0 && padding_h >= 0 && padding_w >= 0, "padding must be non-negative");
    TORCH_CHECK(x.size(1) == weight.size(0), "input channels must match weight.size(0)");

    const int64_t outChannels = weight.size(1);
    TORCH_CHECK(bias.size(0) == outChannels, "bias length must match output channels");

    const int64_t convDepth = ComputeConvTransposeOutDim(x.size(2), weight.size(2), stride_d, padding_d);
    const int64_t convHeight = ComputeConvTransposeOutDim(x.size(3), weight.size(3), stride_h, padding_h);
    const int64_t convWidth = ComputeConvTransposeOutDim(x.size(4), weight.size(4), stride_w, padding_w);
    TORCH_CHECK(convDepth > 0 && convHeight > 0 && convWidth > 0, "invalid conv_transpose3d output shape");

    at::Tensor convResult = at::empty({x.size(0), outChannels, convDepth, convHeight, convWidth}, x.options());
    std::vector<int64_t> strideVec = {stride_d, stride_h, stride_w};
    std::vector<int64_t> paddingVec = {padding_d, padding_h, padding_w};
    std::vector<int64_t> dilationVec = {1, 1, 1};
    std::vector<int64_t> outputPaddingVec = {0, 0, 0};
    at::IntArrayRef strideArray(strideVec);
    at::IntArrayRef paddingArray(paddingVec);
    at::IntArrayRef dilationArray(dilationVec);
    at::IntArrayRef outputPaddingArray(outputPaddingVec);
    const bool transposed = true;
    const int64_t groups = 1;
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
        convResult,
        cubeMathType);

    const int64_t pool1Depth = ComputePoolOutDim(convDepth, 2, 2);
    const int64_t pool1Height = ComputePoolOutDim(convHeight, 2, 2);
    const int64_t pool1Width = ComputePoolOutDim(convWidth, 2, 2);
    const int64_t outputDepth = ComputePoolOutDim(pool1Depth, 3, 3);
    const int64_t outputHeight = ComputePoolOutDim(pool1Height, 3, 3);
    const int64_t outputWidth = ComputePoolOutDim(pool1Width, 3, 3);
    TORCH_CHECK(outputDepth >= 0 && outputHeight >= 0 && outputWidth >= 0, "invalid pooled output shape");

    at::Tensor result = at::empty({x.size(0), 1, outputDepth, outputHeight, outputWidth}, x.options());
    EXEC_NPU_CMD(aclnnConvTranspose3dMaxMaxSumCoreCustom, convResult, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_transpose3d_max_max_sum_custom", &conv_transpose3d_max_max_sum_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv_transpose3d_max_max_sum_custom",
        &conv_transpose3d_max_max_sum_custom_impl_npu,
        "conv_transpose3d + max_pool3d + max_pool3d + channel_sum");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.stride = (stride, stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = (padding, padding, padding) if isinstance(padding, int) else tuple(padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv_transpose3d_max_max_sum_custom(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.stride[0],
            self.stride[1],
            self.stride[2],
            self.padding[0],
            self.padding[1],
            self.padding[2],
        )
'''
