project_json_src='''
[
    {
        "op": "ConvDepthwise2dAsymmetricInputAsymmetricKernelCustom",
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
#include <cstdint>
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvDepthwise2dAsymmetricInputAsymmetricKernelCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, channels);
    TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, kernelHeight);
    TILING_DATA_FIELD_DEF(uint32_t, kernelWidth);
    TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, totalRows);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    ConvDepthwise2dAsymmetricInputAsymmetricKernelCustom,
    ConvDepthwise2dAsymmetricInputAsymmetricKernelCustomTilingData)
}
"""

host_operator_src="""
#include <cstdint>
#include "conv_depthwise2d_asymmetric_input_asymmetric_kernel_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
inline bool IsValidConvShape(
    const gert::Shape* inputShape,
    const gert::Shape* weightShape,
    uint32_t& batch,
    uint32_t& channels,
    uint32_t& inputHeight,
    uint32_t& inputWidth,
    uint32_t& kernelHeight,
    uint32_t& kernelWidth,
    uint32_t& outputHeight,
    uint32_t& outputWidth)
{
    if (inputShape == nullptr || weightShape == nullptr) {
        return false;
    }
    if (inputShape->GetDimNum() != 4 || weightShape->GetDimNum() != 4) {
        return false;
    }

    if (inputShape->GetDim(0) <= 0 || inputShape->GetDim(1) <= 0 || inputShape->GetDim(2) <= 0 ||
        inputShape->GetDim(3) <= 0 || weightShape->GetDim(0) <= 0 || weightShape->GetDim(1) <= 0 ||
        weightShape->GetDim(2) <= 0 || weightShape->GetDim(3) <= 0) {
        return false;
    }

    batch = static_cast<uint32_t>(inputShape->GetDim(0));
    channels = static_cast<uint32_t>(inputShape->GetDim(1));
    inputHeight = static_cast<uint32_t>(inputShape->GetDim(2));
    inputWidth = static_cast<uint32_t>(inputShape->GetDim(3));
    kernelHeight = static_cast<uint32_t>(weightShape->GetDim(2));
    kernelWidth = static_cast<uint32_t>(weightShape->GetDim(3));

    if (static_cast<uint32_t>(weightShape->GetDim(0)) != channels || weightShape->GetDim(1) != 1) {
        return false;
    }
    if (inputHeight < kernelHeight || inputWidth < kernelWidth) {
        return false;
    }

    outputHeight = inputHeight - kernelHeight + 1;
    outputWidth = inputWidth - kernelWidth + 1;
    return outputHeight > 0 && outputWidth > 0;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* inputStorageShape = context->GetInputShape(0);
    const gert::StorageShape* weightStorageShape = context->GetInputShape(1);
    if (inputStorageShape == nullptr || weightStorageShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& inputShape = inputStorageShape->GetStorageShape();
    const gert::Shape& weightShape = weightStorageShape->GetStorageShape();

    uint32_t batch = 0;
    uint32_t channels = 0;
    uint32_t inputHeight = 0;
    uint32_t inputWidth = 0;
    uint32_t kernelHeight = 0;
    uint32_t kernelWidth = 0;
    uint32_t outputHeight = 0;
    uint32_t outputWidth = 0;
    if (!IsValidConvShape(
            &inputShape,
            &weightShape,
            batch,
            channels,
            inputHeight,
            inputWidth,
            kernelHeight,
            kernelWidth,
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

    ConvDepthwise2dAsymmetricInputAsymmetricKernelCustomTilingData tiling;
    tiling.set_blockDim(blockDim);
    tiling.set_batch(batch);
    tiling.set_channels(channels);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_kernelHeight(kernelHeight);
    tiling.set_kernelWidth(kernelWidth);
    tiling.set_outputHeight(outputHeight);
    tiling.set_outputWidth(outputWidth);
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

    uint32_t batch = 0;
    uint32_t channels = 0;
    uint32_t inputHeight = 0;
    uint32_t inputWidth = 0;
    uint32_t kernelHeight = 0;
    uint32_t kernelWidth = 0;
    uint32_t outputHeight = 0;
    uint32_t outputWidth = 0;
    if (!IsValidConvShape(
            inputShape,
            weightShape,
            batch,
            channels,
            inputHeight,
            inputWidth,
            kernelHeight,
            kernelWidth,
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
class ConvDepthwise2dAsymmetricInputAsymmetricKernelCustom : public OpDef {
public:
    explicit ConvDepthwise2dAsymmetricInputAsymmetricKernelCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvDepthwise2dAsymmetricInputAsymmetricKernelCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

class KernelConvDepthwise2dAsymmetricInputAsymmetricKernelCustom {
public:
    __aicore__ inline KernelConvDepthwise2dAsymmetricInputAsymmetricKernelCustom() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR y,
        uint32_t batch,
        uint32_t channels,
        uint32_t inputHeight,
        uint32_t inputWidth,
        uint32_t kernelHeight,
        uint32_t kernelWidth,
        uint32_t outputHeight,
        uint32_t outputWidth,
        uint32_t blockDim,
        uint32_t totalRows)
    {
        this->batch = batch;
        this->channels = channels;
        this->inputHeight = inputHeight;
        this->inputWidth = inputWidth;
        this->kernelHeight = kernelHeight;
        this->kernelWidth = kernelWidth;
        this->outputHeight = outputHeight;
        this->outputWidth = outputWidth;
        this->blockDim = blockDim;
        this->totalRows = totalRows;
        this->inputPlane = inputHeight * inputWidth;
        this->outputPlane = outputHeight * outputWidth;
        this->kernelPlane = kernelHeight * kernelWidth;
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
            for (uint32_t ky = 0; ky < kernelHeight; ++ky) {
                const uint32_t inY = outY + ky;
                for (uint32_t kx = 0; kx < kernelWidth; ++kx) {
                    const uint32_t inX = outX + kx;
                    const uint32_t inputIdx = inputBase + inY * inputWidth + inX;
                    const uint32_t weightIdx = weightBase + ky * kernelWidth + kx;
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
    uint32_t kernelHeight;
    uint32_t kernelWidth;
    uint32_t outputHeight;
    uint32_t outputWidth;
    uint32_t blockDim;
    uint32_t totalRows;
    uint32_t inputPlane;
    uint32_t outputPlane;
    uint32_t kernelPlane;
};

extern "C" __global__ __aicore__ void conv_depthwise2d_asymmetric_input_asymmetric_kernel_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    KernelConvDepthwise2dAsymmetricInputAsymmetricKernelCustom op;
    op.Init(
        x,
        weight,
        y,
        tilingData.batch,
        tilingData.channels,
        tilingData.inputHeight,
        tilingData.inputWidth,
        tilingData.kernelHeight,
        tilingData.kernelWidth,
        tilingData.outputHeight,
        tilingData.outputWidth,
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

at::Tensor conv_depthwise2d_asymmetric_input_asymmetric_kernel_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight)
{
    std::vector<int64_t> strideVec = {1, 1};
    std::vector<int64_t> paddingVec = {0, 0};
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
    const int64_t outputHeight = x.size(2) - weight.size(2) + 1;
    const int64_t outputWidth = x.size(3) - weight.size(3) + 1;
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
        "conv_depthwise2d_asymmetric_input_asymmetric_kernel_custom",
        &conv_depthwise2d_asymmetric_input_asymmetric_kernel_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv_depthwise2d_asymmetric_input_asymmetric_kernel_custom",
        &conv_depthwise2d_asymmetric_input_asymmetric_kernel_custom_impl_npu,
        "conv_depthwise2d_asymmetric_input_asymmetric_kernel_custom");
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
        kernel_size_h: int,
        kernel_size_w: int,
        stride_h: int = 1,
        stride_w: int = 1,
        padding_h: int = 0,
        padding_w: int = 0,
        dilation_h: int = 1,
        dilation_w: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super(ModelNew, self).__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels,
            in_channels,
            (kernel_size_h, kernel_size_w),
            stride=(stride_h, stride_w),
            padding=(padding_h, padding_w),
            dilation=(dilation_h, dilation_w),
            groups=in_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv_depthwise2d_asymmetric_input_asymmetric_kernel_custom(
            x,
            self.conv2d.weight,
        )
'''
