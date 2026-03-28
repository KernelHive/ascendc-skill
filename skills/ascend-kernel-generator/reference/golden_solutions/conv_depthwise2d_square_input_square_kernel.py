project_json_src='''
[
    {
        "op": "ConvDepthwise2dSquareInputSquareKernelCustom",
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
            }
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvDepthwise2dSquareInputSquareKernelCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, channelCount);
    TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, kernelSize);
    TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, stride);
    TILING_DATA_FIELD_DEF(uint32_t, padding);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    ConvDepthwise2dSquareInputSquareKernelCustom,
    ConvDepthwise2dSquareInputSquareKernelCustomTilingData)
}
"""

host_operator_src="""
#include "conv_depthwise2d_square_input_square_kernel_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
uint32_t ComputeOutputDim(int64_t input, int64_t kernelSize, int64_t stride, int64_t padding)
{
    const int64_t numerator = input + padding * 2 - kernelSize;
    if (stride <= 0 || numerator < 0) {
        return 0;
    }
    return static_cast<uint32_t>(numerator / stride + 1);
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    ConvDepthwise2dSquareInputSquareKernelCustomTilingData tiling;
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
    const int64_t *stridePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *paddingPtr = attrs->GetAttrPointer<int64_t>(1);
    if (stridePtr == nullptr || paddingPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t channelCount = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t inputHeight = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t inputWidth = static_cast<uint32_t>(xShape.GetDim(3));
    const uint32_t kernelSize = static_cast<uint32_t>(wShape.GetDim(2));
    const uint32_t stride = static_cast<uint32_t>(*stridePtr);
    const uint32_t padding = static_cast<uint32_t>(*paddingPtr);
    const uint32_t outputHeight = ComputeOutputDim(inputHeight, kernelSize, stride, padding);
    const uint32_t outputWidth = ComputeOutputDim(inputWidth, kernelSize, stride, padding);

    context->SetBlockDim(batchSize == 0 ? 1 : batchSize);
    tiling.set_batchSize(batchSize);
    tiling.set_channelCount(channelCount);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_kernelSize(kernelSize);
    tiling.set_outputHeight(outputHeight);
    tiling.set_outputWidth(outputWidth);
    tiling.set_stride(stride);
    tiling.set_padding(padding);
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
    const gert::Shape *xShape = context->GetInputShape(0);
    const gert::Shape *wShape = context->GetInputShape(1);
    if (xShape == nullptr || wShape == nullptr || xShape->GetDimNum() != 4 || wShape->GetDimNum() != 4) {
        return GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int64_t *stridePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *paddingPtr = attrs->GetAttrPointer<int64_t>(1);
    if (stridePtr == nullptr || paddingPtr == nullptr) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(4);
    outputShape->SetDim(0, xShape->GetDim(0));
    outputShape->SetDim(1, xShape->GetDim(1));
    outputShape->SetDim(2, ComputeOutputDim(xShape->GetDim(2), wShape->GetDim(2), *stridePtr, *paddingPtr));
    outputShape->SetDim(3, ComputeOutputDim(xShape->GetDim(3), wShape->GetDim(3), *stridePtr, *paddingPtr));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ConvDepthwise2dSquareInputSquareKernelCustom : public OpDef {
public:
    explicit ConvDepthwise2dSquareInputSquareKernelCustom(const char *name) : OpDef(name)
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
        this->Attr("stride").AttrType(REQUIRED).Int();
        this->Attr("padding").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvDepthwise2dSquareInputSquareKernelCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConvDepthwise2dSquareInputSquareKernel {
public:
    __aicore__ inline KernelConvDepthwise2dSquareInputSquareKernel() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t channelCount,
        uint32_t inputHeight,
        uint32_t inputWidth,
        uint32_t kernelSize,
        uint32_t outputHeight,
        uint32_t outputWidth,
        uint32_t stride,
        uint32_t padding)
    {
        this->batchSize = batchSize;
        this->channelCount = channelCount;
        this->inputHeight = inputHeight;
        this->inputWidth = inputWidth;
        this->kernelSize = kernelSize;
        this->outputHeight = outputHeight;
        this->outputWidth = outputWidth;
        this->stride = stride;
        this->padding = padding;
        this->blockIdx = GetBlockIdx();
        this->inputPlaneSize = inputHeight * inputWidth;
        this->outputPlaneSize = outputHeight * outputWidth;
        this->elementsPerBatch = channelCount * this->inputPlaneSize;
        this->outputElementsPerBatch = channelCount * this->outputPlaneSize;
        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * this->elementsPerBatch);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, channelCount * kernelSize * kernelSize);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * this->outputElementsPerBatch);
    }

    __aicore__ inline void Process()
    {
        if (this->blockIdx >= this->batchSize) {
            return;
        }

        const uint32_t batchBase = this->blockIdx * this->elementsPerBatch;
        const uint32_t outputBatchBase = this->blockIdx * this->outputElementsPerBatch;
        const int32_t inputHeight = static_cast<int32_t>(this->inputHeight);
        const int32_t inputWidth = static_cast<int32_t>(this->inputWidth);
        const int32_t kernelSize = static_cast<int32_t>(this->kernelSize);
        const int32_t stride = static_cast<int32_t>(this->stride);
        const int32_t padding = static_cast<int32_t>(this->padding);

        for (uint32_t channelIdx = 0; channelIdx < this->channelCount; ++channelIdx) {
            const uint32_t inputBase = batchBase + channelIdx * this->inputPlaneSize;
            const uint32_t outputBase = outputBatchBase + channelIdx * this->outputPlaneSize;
            const uint32_t weightBase = channelIdx * this->kernelSize * this->kernelSize;
            for (uint32_t outH = 0; outH < this->outputHeight; ++outH) {
                const int32_t startH = static_cast<int32_t>(outH) * stride - padding;
                for (uint32_t outW = 0; outW < this->outputWidth; ++outW) {
                    const int32_t startW = static_cast<int32_t>(outW) * stride - padding;
                    float sum = 0.0f;
                    for (int32_t kernelH = 0; kernelH < kernelSize; ++kernelH) {
                        const int32_t inH = startH + kernelH;
                        if (inH < 0 || inH >= inputHeight) {
                            continue;
                        }
                        for (int32_t kernelW = 0; kernelW < kernelSize; ++kernelW) {
                            const int32_t inW = startW + kernelW;
                            if (inW < 0 || inW >= inputWidth) {
                                continue;
                            }
                            const uint32_t inputOffset =
                                inputBase + static_cast<uint32_t>(inH) * this->inputWidth + static_cast<uint32_t>(inW);
                            const uint32_t weightOffset =
                                weightBase + static_cast<uint32_t>(kernelH) * this->kernelSize + static_cast<uint32_t>(kernelW);
                            sum += xGm.GetValue(inputOffset) * weightGm.GetValue(weightOffset);
                        }
                    }
                    yGm.SetValue(outputBase + outH * this->outputWidth + outW, sum);
                }
            }
        }
    }

private:
    GlobalTensor<float> xGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t channelCount;
    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t kernelSize;
    uint32_t outputHeight;
    uint32_t outputWidth;
    uint32_t stride;
    uint32_t padding;
    uint32_t blockIdx;
    uint32_t inputPlaneSize;
    uint32_t outputPlaneSize;
    uint32_t elementsPerBatch;
    uint32_t outputElementsPerBatch;
};

extern "C" __global__ __aicore__ void conv_depthwise2d_square_input_square_kernel_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvDepthwise2dSquareInputSquareKernel op;
    op.Init(
        x,
        weight,
        y,
        tiling_data.batchSize,
        tiling_data.channelCount,
        tiling_data.inputHeight,
        tiling_data.inputWidth,
        tiling_data.kernelSize,
        tiling_data.outputHeight,
        tiling_data.outputWidth,
        tiling_data.stride,
        tiling_data.padding);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor conv_depthwise2d_square_input_square_kernel_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    int64_t stride,
    int64_t padding)
{
    TORCH_CHECK(x.dim() == 4, "x must be a 4D NCHW tensor");
    TORCH_CHECK(weight.dim() == 4, "weight must be a 4D tensor");
    TORCH_CHECK(weight.size(1) == 1, "depthwise weight second dimension must be 1");
    TORCH_CHECK(x.size(1) == weight.size(0), "input channels must equal weight.size(0) for depthwise convolution");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");

    const int64_t kernelH = weight.size(2);
    const int64_t kernelW = weight.size(3);
    const int64_t outH = (x.size(2) + padding * 2 - kernelH) / stride + 1;
    const int64_t outW = (x.size(3) + padding * 2 - kernelW) / stride + 1;
    TORCH_CHECK(outH >= 0 && outW >= 0, "invalid output shape");

    at::Tensor result = at::empty({x.size(0), x.size(1), outH, outW}, x.options());
    c10::optional<at::Tensor> bias = c10::nullopt;
    const int64_t kernelSizeData[2] = {kernelH, kernelW};
    const int64_t strideData[2] = {stride, stride};
    const int64_t paddingData[2] = {padding, padding};
    const int64_t dilationData[2] = {1, 1};
    const at::IntArrayRef kernelSize(kernelSizeData, 2);
    const at::IntArrayRef strideArray(strideData, 2);
    const at::IntArrayRef paddingArray(paddingData, 2);
    const at::IntArrayRef dilationArray(dilationData, 2);
    const int8_t cubeMathType = 0;

    EXEC_NPU_CMD(
        aclnnConvDepthwise2d,
        x,
        weight,
        kernelSize,
        bias,
        strideArray,
        paddingArray,
        dilationArray,
        result,
        cubeMathType);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "conv_depthwise2d_square_input_square_kernel_custom",
        &conv_depthwise2d_square_input_square_kernel_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv_depthwise2d_square_input_square_kernel_custom",
        &conv_depthwise2d_square_input_square_kernel_custom_impl_npu,
        "Depthwise Conv2d custom op");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0) -> None:
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.conv2d = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv_depthwise2d_square_input_square_kernel_custom(
            x,
            self.conv2d.weight,
            self.stride,
            self.padding,
        )
'''
