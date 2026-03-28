project_json_src='''
[
    {
        "op": "ConvDepthwiseSeparable2dCustom",
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
                "name": "depthwise_weight",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            },
            {
                "name": "pointwise_weight",
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
                "name": "dilation",
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
BEGIN_TILING_DATA_DEF(ConvDepthwiseSeparable2dCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inChannels);
    TILING_DATA_FIELD_DEF(uint32_t, outChannels);
    TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, kernelHeight);
    TILING_DATA_FIELD_DEF(uint32_t, kernelWidth);
    TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, stride);
    TILING_DATA_FIELD_DEF(uint32_t, padding);
    TILING_DATA_FIELD_DEF(uint32_t, dilation);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    ConvDepthwiseSeparable2dCustom,
    ConvDepthwiseSeparable2dCustomTilingData)
}
"""

host_operator_src="""
#include "conv_depthwise_separable2d_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
uint32_t ComputeOutputDim(int64_t input, int64_t kernelSize, int64_t stride, int64_t padding, int64_t dilation)
{
    const int64_t effectiveKernel = dilation * (kernelSize - 1) + 1;
    const int64_t numerator = input + padding * 2 - effectiveKernel;
    if (stride <= 0 || dilation <= 0 || numerator < 0) {
        return 0;
    }
    return static_cast<uint32_t>(numerator / stride + 1);
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    ConvDepthwiseSeparable2dCustomTilingData tiling;
    const gert::StorageShape *inputShape = context->GetInputShape(0);
    const gert::StorageShape *depthwiseWeightShape = context->GetInputShape(1);
    const gert::StorageShape *pointwiseWeightShape = context->GetInputShape(2);
    if (inputShape == nullptr || depthwiseWeightShape == nullptr || pointwiseWeightShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xShape = inputShape->GetStorageShape();
    const auto dwShape = depthwiseWeightShape->GetStorageShape();
    const auto pwShape = pointwiseWeightShape->GetStorageShape();
    if (xShape.GetDimNum() != 4 || dwShape.GetDimNum() != 4 || pwShape.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int64_t *stridePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *paddingPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *dilationPtr = attrs->GetAttrPointer<int64_t>(2);
    if (stridePtr == nullptr || paddingPtr == nullptr || dilationPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t inChannels = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t outChannels = static_cast<uint32_t>(pwShape.GetDim(0));
    const uint32_t inputHeight = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t inputWidth = static_cast<uint32_t>(xShape.GetDim(3));
    const uint32_t kernelHeight = static_cast<uint32_t>(dwShape.GetDim(2));
    const uint32_t kernelWidth = static_cast<uint32_t>(dwShape.GetDim(3));
    const uint32_t stride = static_cast<uint32_t>(*stridePtr);
    const uint32_t padding = static_cast<uint32_t>(*paddingPtr);
    const uint32_t dilation = static_cast<uint32_t>(*dilationPtr);
    const uint32_t outputHeight = ComputeOutputDim(inputHeight, kernelHeight, stride, padding, dilation);
    const uint32_t outputWidth = ComputeOutputDim(inputWidth, kernelWidth, stride, padding, dilation);

    context->SetBlockDim(batchSize == 0 ? 1 : batchSize);
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_kernelHeight(kernelHeight);
    tiling.set_kernelWidth(kernelWidth);
    tiling.set_outputHeight(outputHeight);
    tiling.set_outputWidth(outputWidth);
    tiling.set_stride(stride);
    tiling.set_padding(padding);
    tiling.set_dilation(dilation);
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
    const gert::Shape *dwShape = context->GetInputShape(1);
    const gert::Shape *pwShape = context->GetInputShape(2);
    if (xShape == nullptr || dwShape == nullptr || pwShape == nullptr ||
        xShape->GetDimNum() != 4 || dwShape->GetDimNum() != 4 || pwShape->GetDimNum() != 4) {
        return GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int64_t *stridePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *paddingPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *dilationPtr = attrs->GetAttrPointer<int64_t>(2);
    if (stridePtr == nullptr || paddingPtr == nullptr || dilationPtr == nullptr) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(4);
    outputShape->SetDim(0, xShape->GetDim(0));
    outputShape->SetDim(1, pwShape->GetDim(0));
    outputShape->SetDim(2, ComputeOutputDim(xShape->GetDim(2), dwShape->GetDim(2), *stridePtr, *paddingPtr, *dilationPtr));
    outputShape->SetDim(3, ComputeOutputDim(xShape->GetDim(3), dwShape->GetDim(3), *stridePtr, *paddingPtr, *dilationPtr));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ConvDepthwiseSeparable2dCustom : public OpDef {
public:
    explicit ConvDepthwiseSeparable2dCustom(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("depthwise_weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("pointwise_weight")
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
        this->Attr("dilation").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvDepthwiseSeparable2dCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConvDepthwiseSeparable2d {
public:
    __aicore__ inline KernelConvDepthwiseSeparable2d() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR depthwiseWeight,
        GM_ADDR pointwiseWeight,
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
        uint32_t stride,
        uint32_t padding,
        uint32_t dilation)
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
        this->stride = stride;
        this->padding = padding;
        this->dilation = dilation;
        this->blockIdx = GetBlockIdx();
        this->inputPlaneSize = inputHeight * inputWidth;
        this->outputPlaneSize = outputHeight * outputWidth;
        this->inputBatchSize = inChannels * this->inputPlaneSize;
        this->outputBatchSize = outChannels * this->outputPlaneSize;
        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * this->inputBatchSize);
        depthwiseWeightGm.SetGlobalBuffer((__gm__ float *)depthwiseWeight, inChannels * kernelHeight * kernelWidth);
        pointwiseWeightGm.SetGlobalBuffer((__gm__ float *)pointwiseWeight, outChannels * inChannels);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * this->outputBatchSize);
    }

    __aicore__ inline void Process()
    {
        if (this->blockIdx >= this->batchSize) {
            return;
        }

        const uint32_t inputBatchBase = this->blockIdx * this->inputBatchSize;
        const uint32_t outputBatchBase = this->blockIdx * this->outputBatchSize;

        for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
            const uint32_t pointwiseOutBase = outChannel * this->inChannels;
            const uint32_t outputChannelBase = outputBatchBase + outChannel * this->outputPlaneSize;
            for (uint32_t outH = 0; outH < this->outputHeight; ++outH) {
                const int32_t startH = static_cast<int32_t>(outH) * static_cast<int32_t>(this->stride) -
                                       static_cast<int32_t>(this->padding);
                for (uint32_t outW = 0; outW < this->outputWidth; ++outW) {
                    const int32_t startW = static_cast<int32_t>(outW) * static_cast<int32_t>(this->stride) -
                                           static_cast<int32_t>(this->padding);
                    float pointwiseSum = 0.0f;
                    for (uint32_t inChannel = 0; inChannel < this->inChannels; ++inChannel) {
                        const uint32_t inputChannelBase = inputBatchBase + inChannel * this->inputPlaneSize;
                        const uint32_t depthwiseBase = inChannel * this->kernelHeight * this->kernelWidth;
                        float depthwiseSum = 0.0f;
                        for (uint32_t kernelH = 0; kernelH < this->kernelHeight; ++kernelH) {
                            const int32_t inH = startH + static_cast<int32_t>(kernelH * this->dilation);
                            if (inH < 0 || inH >= static_cast<int32_t>(this->inputHeight)) {
                                continue;
                            }
                            for (uint32_t kernelW = 0; kernelW < this->kernelWidth; ++kernelW) {
                                const int32_t inW = startW + static_cast<int32_t>(kernelW * this->dilation);
                                if (inW < 0 || inW >= static_cast<int32_t>(this->inputWidth)) {
                                    continue;
                                }
                                const uint32_t inputOffset =
                                    inputChannelBase + static_cast<uint32_t>(inH) * this->inputWidth +
                                    static_cast<uint32_t>(inW);
                                const uint32_t depthwiseOffset =
                                    depthwiseBase + kernelH * this->kernelWidth + kernelW;
                                depthwiseSum +=
                                    xGm.GetValue(inputOffset) * depthwiseWeightGm.GetValue(depthwiseOffset);
                            }
                        }
                        pointwiseSum += depthwiseSum * pointwiseWeightGm.GetValue(pointwiseOutBase + inChannel);
                    }
                    yGm.SetValue(outputChannelBase + outH * this->outputWidth + outW, pointwiseSum);
                }
            }
        }
    }

private:
    GlobalTensor<float> xGm;
    GlobalTensor<float> depthwiseWeightGm;
    GlobalTensor<float> pointwiseWeightGm;
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
    uint32_t stride;
    uint32_t padding;
    uint32_t dilation;
    uint32_t blockIdx;
    uint32_t inputPlaneSize;
    uint32_t outputPlaneSize;
    uint32_t inputBatchSize;
    uint32_t outputBatchSize;
};

extern "C" __global__ __aicore__ void conv_depthwise_separable2d_custom(
    GM_ADDR x,
    GM_ADDR depthwise_weight,
    GM_ADDR pointwise_weight,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvDepthwiseSeparable2d op;
    op.Init(
        x,
        depthwise_weight,
        pointwise_weight,
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
        tiling_data.stride,
        tiling_data.padding,
        tiling_data.dilation);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor conv_depthwise_separable2d_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &depthwise_weight,
    const at::Tensor &pointwise_weight,
    int64_t stride,
    int64_t padding,
    int64_t dilation)
{
    TORCH_CHECK(x.dim() == 4, "x must be a 4D NCHW tensor");
    TORCH_CHECK(depthwise_weight.dim() == 4, "depthwise_weight must be a 4D tensor");
    TORCH_CHECK(pointwise_weight.dim() == 4, "pointwise_weight must be a 4D tensor");
    TORCH_CHECK(depthwise_weight.size(1) == 1, "depthwise_weight second dimension must be 1");
    TORCH_CHECK(x.size(1) == depthwise_weight.size(0), "input channels must equal depthwise_weight.size(0)");
    TORCH_CHECK(pointwise_weight.size(1) == x.size(1), "pointwise_weight input channels must match x.size(1)");
    TORCH_CHECK(pointwise_weight.size(2) == 1 && pointwise_weight.size(3) == 1, "pointwise_weight must be 1x1");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(dilation > 0, "dilation must be positive");

    const int64_t kernelH = depthwise_weight.size(2);
    const int64_t kernelW = depthwise_weight.size(3);
    const int64_t effectiveKernelH = dilation * (kernelH - 1) + 1;
    const int64_t effectiveKernelW = dilation * (kernelW - 1) + 1;
    const int64_t outH = (x.size(2) + padding * 2 - effectiveKernelH) / stride + 1;
    const int64_t outW = (x.size(3) + padding * 2 - effectiveKernelW) / stride + 1;
    TORCH_CHECK(outH >= 0 && outW >= 0, "invalid output shape");

    c10::optional<at::Tensor> bias = c10::nullopt;
    const int64_t depthwiseKernelData[2] = {kernelH, kernelW};
    const int64_t strideData[2] = {stride, stride};
    const int64_t paddingData[2] = {padding, padding};
    const int64_t dilationData[2] = {dilation, dilation};
    const at::IntArrayRef depthwiseKernelSize(depthwiseKernelData, 2);
    const at::IntArrayRef strideArray(strideData, 2);
    const at::IntArrayRef paddingArray(paddingData, 2);
    const at::IntArrayRef dilationArray(dilationData, 2);
    const int8_t cubeMathType = 0;

    at::Tensor depthwiseResult = at::empty({x.size(0), x.size(1), outH, outW}, x.options());
    EXEC_NPU_CMD(
        aclnnConvDepthwise2d,
        x,
        depthwise_weight,
        depthwiseKernelSize,
        bias,
        strideArray,
        paddingArray,
        dilationArray,
        depthwiseResult,
        cubeMathType);

    const int64_t pointwiseStrideData[2] = {1, 1};
    const int64_t pointwisePaddingData[2] = {0, 0};
    const int64_t pointwiseDilationData[2] = {1, 1};
    const int64_t pointwiseOutputPaddingData[2] = {0, 0};
    const at::IntArrayRef pointwiseStride(pointwiseStrideData, 2);
    const at::IntArrayRef pointwisePadding(pointwisePaddingData, 2);
    const at::IntArrayRef pointwiseDilation(pointwiseDilationData, 2);
    const at::IntArrayRef pointwiseOutputPadding(pointwiseOutputPaddingData, 2);
    const bool transposed = false;
    const int64_t groups = 1;
    at::Tensor result = at::empty({x.size(0), pointwise_weight.size(0), outH, outW}, x.options());
    EXEC_NPU_CMD(
        aclnnConvolution,
        depthwiseResult,
        pointwise_weight,
        bias,
        pointwiseStride,
        pointwisePadding,
        pointwiseDilation,
        transposed,
        pointwiseOutputPadding,
        groups,
        result,
        cubeMathType);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "conv_depthwise_separable2d_custom",
        &conv_depthwise_separable2d_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv_depthwise_separable2d_custom",
        &conv_depthwise_separable2d_custom_impl_npu,
        "Depthwise separable Conv2d custom op");
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
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.depthwise = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv_depthwise_separable2d_custom(
            x,
            self.depthwise.weight,
            self.pointwise.weight,
            self.stride,
            self.padding,
            self.dilation,
        )
'''
