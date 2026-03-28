project_json_src='''
[
    {
        "op": "ConvDepthwise2dAsymmetricInputSquareKernelCustom",
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
#include <cstdint>
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvDepthwise2dAsymmetricInputSquareKernelCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, channels);
    TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, kernelSize);
    TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, stride);
    TILING_DATA_FIELD_DEF(uint32_t, padding);
    TILING_DATA_FIELD_DEF(uint32_t, totalRows);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    ConvDepthwise2dAsymmetricInputSquareKernelCustom,
    ConvDepthwise2dAsymmetricInputSquareKernelCustomTilingData)
}
"""

host_operator_src="""
#include <cstdint>
#include "conv_depthwise2d_asymmetric_input_square_kernel_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
inline bool IsValidConvShape(
    const gert::Shape* inputShape,
    const gert::Shape* weightShape,
    int64_t stride,
    int64_t padding,
    uint32_t& batch,
    uint32_t& channels,
    uint32_t& inputHeight,
    uint32_t& inputWidth,
    uint32_t& kernelSize,
    uint32_t& outputHeight,
    uint32_t& outputWidth)
{
    if (inputShape == nullptr || weightShape == nullptr) {
        return false;
    }
    if (inputShape->GetDimNum() != 4 || weightShape->GetDimNum() != 4) {
        return false;
    }
    if (stride <= 0 || padding < 0) {
        return false;
    }

    batch = static_cast<uint32_t>(inputShape->GetDim(0));
    channels = static_cast<uint32_t>(inputShape->GetDim(1));
    inputHeight = static_cast<uint32_t>(inputShape->GetDim(2));
    inputWidth = static_cast<uint32_t>(inputShape->GetDim(3));
    const uint32_t kernelHeight = static_cast<uint32_t>(weightShape->GetDim(2));
    const uint32_t kernelWidth = static_cast<uint32_t>(weightShape->GetDim(3));

    if (batch == 0 || channels == 0 || inputHeight == 0 || inputWidth == 0 ||
        kernelHeight == 0 || kernelWidth == 0) {
        return false;
    }
    if (static_cast<uint32_t>(weightShape->GetDim(0)) != channels || weightShape->GetDim(1) != 1) {
        return false;
    }
    if (kernelHeight != kernelWidth) {
        return false;
    }

    kernelSize = kernelHeight;
    const int64_t outH = (static_cast<int64_t>(inputHeight) + 2 * padding - static_cast<int64_t>(kernelSize)) / stride + 1;
    const int64_t outW = (static_cast<int64_t>(inputWidth) + 2 * padding - static_cast<int64_t>(kernelSize)) / stride + 1;
    if (outH <= 0 || outW <= 0) {
        return false;
    }

    outputHeight = static_cast<uint32_t>(outH);
    outputWidth = static_cast<uint32_t>(outW);
    return true;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* inputStorageShape = context->GetInputShape(0);
    const gert::StorageShape* weightStorageShape = context->GetInputShape(1);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (inputStorageShape == nullptr || weightStorageShape == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const int64_t* stridePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t* paddingPtr = attrs->GetAttrPointer<int64_t>(1);
    if (stridePtr == nullptr || paddingPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& inputShape = inputStorageShape->GetStorageShape();
    const gert::Shape& weightShape = weightStorageShape->GetStorageShape();

    uint32_t batch = 0;
    uint32_t channels = 0;
    uint32_t inputHeight = 0;
    uint32_t inputWidth = 0;
    uint32_t kernelSize = 0;
    uint32_t outputHeight = 0;
    uint32_t outputWidth = 0;
    if (!IsValidConvShape(
            &inputShape,
            &weightShape,
            *stridePtr,
            *paddingPtr,
            batch,
            channels,
            inputHeight,
            inputWidth,
            kernelSize,
            outputHeight,
            outputWidth)) {
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t blockDim = ascendcPlatform.GetCoreNumAiv();
    if (blockDim == 0) {
        blockDim = 1;
    }
    const uint32_t totalRows = batch * channels * outputHeight;
    if (totalRows < blockDim) {
        blockDim = totalRows;
    }
    if (blockDim == 0) {
        blockDim = 1;
    }

    ConvDepthwise2dAsymmetricInputSquareKernelCustomTilingData tiling;
    tiling.set_blockDim(blockDim);
    tiling.set_batch(batch);
    tiling.set_channels(channels);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_kernelSize(kernelSize);
    tiling.set_outputHeight(outputHeight);
    tiling.set_outputWidth(outputWidth);
    tiling.set_stride(static_cast<uint32_t>(*stridePtr));
    tiling.set_padding(static_cast<uint32_t>(*paddingPtr));
    tiling.set_totalRows(totalRows);

    context->SetBlockDim(blockDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    const gert::Shape* weightShape = context->GetInputShape(1);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return GRAPH_FAILED;
    }

    const int64_t* stridePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t* paddingPtr = attrs->GetAttrPointer<int64_t>(1);
    if (stridePtr == nullptr || paddingPtr == nullptr) {
        return GRAPH_FAILED;
    }

    uint32_t batch = 0;
    uint32_t channels = 0;
    uint32_t inputHeight = 0;
    uint32_t inputWidth = 0;
    uint32_t kernelSize = 0;
    uint32_t outputHeight = 0;
    uint32_t outputWidth = 0;
    if (!IsValidConvShape(
            inputShape,
            weightShape,
            *stridePtr,
            *paddingPtr,
            batch,
            channels,
            inputHeight,
            inputWidth,
            kernelSize,
            outputHeight,
            outputWidth)) {
        return GRAPH_FAILED;
    }

    gert::Shape* outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(4);
    outputShape->SetDim(0, batch);
    outputShape->SetDim(1, channels);
    outputShape->SetDim(2, outputHeight);
    outputShape->SetDim(3, outputWidth);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ConvDepthwise2dAsymmetricInputSquareKernelCustom : public OpDef {
public:
    explicit ConvDepthwise2dAsymmetricInputSquareKernelCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvDepthwise2dAsymmetricInputSquareKernelCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

class KernelConvDepthwise2dAsymmetricInputSquareKernelCustom {
public:
    __aicore__ inline KernelConvDepthwise2dAsymmetricInputSquareKernelCustom() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR y,
        uint32_t batch,
        uint32_t channels,
        uint32_t inputHeight,
        uint32_t inputWidth,
        uint32_t kernelSize,
        uint32_t outputHeight,
        uint32_t outputWidth,
        uint32_t stride,
        uint32_t padding,
        uint32_t blockDim,
        uint32_t totalRows)
    {
        this->batch = batch;
        this->channels = channels;
        this->inputHeight = inputHeight;
        this->inputWidth = inputWidth;
        this->kernelSize = kernelSize;
        this->outputHeight = outputHeight;
        this->outputWidth = outputWidth;
        this->stride = stride;
        this->padding = padding;
        this->blockDim = blockDim;
        this->totalRows = totalRows;
        this->inputPlane = inputHeight * inputWidth;
        this->outputPlane = outputHeight * outputWidth;
        this->kernelPlane = kernelSize * kernelSize;
        xGm.SetGlobalBuffer((__gm__ float*)x, batch * channels * inputPlane);
        weightGm.SetGlobalBuffer((__gm__ float*)weight, channels * kernelPlane);
        yGm.SetGlobalBuffer((__gm__ float*)y, batch * channels * outputPlane);
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        for (uint32_t rowIdx = blockIdx; rowIdx < totalRows; rowIdx += blockDim) {
            ProcessRow(rowIdx);
        }
    }

private:
    __aicore__ inline void ProcessRow(uint32_t rowIdx)
    {
        const uint32_t ncIdx = rowIdx / outputHeight;
        const uint32_t outY = rowIdx % outputHeight;
        const uint32_t batchIdx = ncIdx / channels;
        const uint32_t channelIdx = ncIdx % channels;
        const uint32_t inputBase = (batchIdx * channels + channelIdx) * inputPlane;
        const uint32_t outputBase = (batchIdx * channels + channelIdx) * outputPlane + outY * outputWidth;
        const uint32_t weightBase = channelIdx * kernelPlane;

        for (uint32_t outX = 0; outX < outputWidth; ++outX) {
            float sum = 0.0f;
            const int32_t startY = static_cast<int32_t>(outY * stride) - static_cast<int32_t>(padding);
            const int32_t startX = static_cast<int32_t>(outX * stride) - static_cast<int32_t>(padding);
            for (uint32_t ky = 0; ky < kernelSize; ++ky) {
                const int32_t inY = startY + static_cast<int32_t>(ky);
                if (inY < 0 || inY >= static_cast<int32_t>(inputHeight)) {
                    continue;
                }
                for (uint32_t kx = 0; kx < kernelSize; ++kx) {
                    const int32_t inX = startX + static_cast<int32_t>(kx);
                    if (inX < 0 || inX >= static_cast<int32_t>(inputWidth)) {
                        continue;
                    }
                    const uint32_t inputIdx = inputBase + static_cast<uint32_t>(inY) * inputWidth + static_cast<uint32_t>(inX);
                    const uint32_t weightIdx = weightBase + ky * kernelSize + kx;
                    sum += xGm.GetValue(inputIdx) * weightGm.GetValue(weightIdx);
                }
            }
            yGm.SetValue(outputBase + outX, sum);
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> weightGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batch;
    uint32_t channels;
    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t kernelSize;
    uint32_t outputHeight;
    uint32_t outputWidth;
    uint32_t stride;
    uint32_t padding;
    uint32_t blockDim;
    uint32_t totalRows;
    uint32_t inputPlane;
    uint32_t outputPlane;
    uint32_t kernelPlane;
};

extern "C" __global__ __aicore__ void conv_depthwise2d_asymmetric_input_square_kernel_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    KernelConvDepthwise2dAsymmetricInputSquareKernelCustom op;
    op.Init(
        x,
        weight,
        y,
        tilingData.batch,
        tilingData.channels,
        tilingData.inputHeight,
        tilingData.inputWidth,
        tilingData.kernelSize,
        tilingData.outputHeight,
        tilingData.outputWidth,
        tilingData.stride,
        tilingData.padding,
        tilingData.blockDim,
        tilingData.totalRows);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include "pytorch_npu_helper.hpp"

at::Tensor conv_depthwise2d_asymmetric_input_square_kernel_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight,
    int64_t stride,
    int64_t padding)
{
    TORCH_CHECK(x.dim() == 4, "x must be a 4D NCHW tensor");
    TORCH_CHECK(weight.dim() == 4, "weight must be a 4D tensor");
    TORCH_CHECK(weight.size(1) == 1, "depthwise weight second dimension must be 1");
    TORCH_CHECK(weight.size(2) == weight.size(3), "weight kernel must be square");
    TORCH_CHECK(x.size(1) == weight.size(0), "input channels must equal weight.size(0)");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");

    std::vector<int64_t> strideVec = {stride, stride};
    std::vector<int64_t> paddingVec = {padding, padding};
    std::vector<int64_t> dilationVec = {1, 1};
    std::vector<int64_t> outputPaddingVec = {0, 0};
    at::IntArrayRef strides(strideVec);
    at::IntArrayRef paddings(paddingVec);
    at::IntArrayRef dilations(dilationVec);
    at::IntArrayRef outputPadding(outputPaddingVec);
    at::Tensor bias;
    constexpr bool transposed = false;
    constexpr int8_t cubeMathType = 0;
    const int64_t groups = x.size(1);
    const int64_t outputHeight = (x.size(2) + padding * 2 - weight.size(2)) / stride + 1;
    const int64_t outputWidth = (x.size(3) + padding * 2 - weight.size(3)) / stride + 1;
    TORCH_CHECK(outputHeight > 0 && outputWidth > 0, "invalid output shape");

    at::Tensor result = at::empty({x.size(0), x.size(1), outputHeight, outputWidth}, x.options());
    EXEC_NPU_CMD(
        aclnnConvolution,
        x,
        weight,
        bias,
        strides,
        paddings,
        dilations,
        transposed,
        outputPadding,
        groups,
        result,
        cubeMathType);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "conv_depthwise2d_asymmetric_input_square_kernel_custom",
        &conv_depthwise2d_asymmetric_input_square_kernel_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv_depthwise2d_asymmetric_input_square_kernel_custom",
        &conv_depthwise2d_asymmetric_input_square_kernel_custom_impl_npu,
        "conv_depthwise2d_asymmetric_input_square_kernel_custom");
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
    ) -> None:
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.conv2d = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv_depthwise2d_asymmetric_input_square_kernel_custom(
            x,
            self.conv2d.weight,
            self.stride,
            self.padding,
        )
'''
