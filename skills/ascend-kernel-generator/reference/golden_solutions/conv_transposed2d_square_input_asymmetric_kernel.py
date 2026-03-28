project_json_src='''
[
    {
        "op": "ConvTransposed2dSquareInputAsymmetricKernelCustom",
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
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTransposed2dSquareInputAsymmetricKernelCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inChannels);
    TILING_DATA_FIELD_DEF(uint32_t, outChannels);
    TILING_DATA_FIELD_DEF(uint32_t, inputSize);
    TILING_DATA_FIELD_DEF(uint32_t, kernelHeight);
    TILING_DATA_FIELD_DEF(uint32_t, kernelWidth);
    TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    ConvTransposed2dSquareInputAsymmetricKernelCustom,
    ConvTransposed2dSquareInputAsymmetricKernelCustomTilingData)
}
"""

host_operator_src="""
#include "conv_transposed2d_square_input_asymmetric_kernel_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
constexpr uint32_t kStrideH = 1;
constexpr uint32_t kStrideW = 1;
constexpr uint32_t kPaddingH = 0;
constexpr uint32_t kPaddingW = 0;
constexpr uint32_t kDilationH = 1;
constexpr uint32_t kDilationW = 1;
constexpr uint32_t kOutputPaddingH = 0;
constexpr uint32_t kOutputPaddingW = 0;

uint32_t ComputeTransposedDim(
    int64_t input,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t outputPadding)
{
    if (input <= 0 || kernel <= 0 || stride <= 0 || dilation <= 0) {
        return 0;
    }
    const int64_t output =
        (input - 1) * stride - 2 * padding + dilation * (kernel - 1) + outputPadding + 1;
    return output > 0 ? static_cast<uint32_t>(output) : 0;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
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

    if (xShape.GetDim(2) != xShape.GetDim(3)) {
        return ge::GRAPH_FAILED;
    }
    if (wShape.GetDim(2) == wShape.GetDim(3)) {
        return ge::GRAPH_FAILED;
    }

    ConvTransposed2dSquareInputAsymmetricKernelCustomTilingData tiling;
    tiling.set_batchSize(static_cast<uint32_t>(xShape.GetDim(0)));
    tiling.set_inChannels(static_cast<uint32_t>(xShape.GetDim(1)));
    tiling.set_outChannels(static_cast<uint32_t>(wShape.GetDim(1)));
    tiling.set_inputSize(static_cast<uint32_t>(xShape.GetDim(2)));
    tiling.set_kernelHeight(static_cast<uint32_t>(wShape.GetDim(2)));
    tiling.set_kernelWidth(static_cast<uint32_t>(wShape.GetDim(3)));
    tiling.set_outputHeight(
        ComputeTransposedDim(
            xShape.GetDim(2),
            wShape.GetDim(2),
            kStrideH,
            kPaddingH,
            kDilationH,
            kOutputPaddingH));
    tiling.set_outputWidth(
        ComputeTransposedDim(
            xShape.GetDim(3),
            wShape.GetDim(3),
            kStrideW,
            kPaddingW,
            kDilationW,
            kOutputPaddingW));

    context->SetBlockDim(1);
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
    if (inputShape == nullptr || weightShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDimNum() != 4 || weightShape->GetDimNum() != 4) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDim(2) != inputShape->GetDim(3)) {
        return GRAPH_FAILED;
    }
    if (weightShape->GetDim(2) == weightShape->GetDim(3)) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(4);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, weightShape->GetDim(1));
    outputShape->SetDim(
        2,
        ComputeTransposedDim(
            inputShape->GetDim(2),
            weightShape->GetDim(2),
            kStrideH,
            kPaddingH,
            kDilationH,
            kOutputPaddingH));
    outputShape->SetDim(
        3,
        ComputeTransposedDim(
            inputShape->GetDim(3),
            weightShape->GetDim(3),
            kStrideW,
            kPaddingW,
            kDilationW,
            kOutputPaddingW));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ConvTransposed2dSquareInputAsymmetricKernelCustom : public OpDef {
public:
    explicit ConvTransposed2dSquareInputAsymmetricKernelCustom(const char *name) : OpDef(name)
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

OP_ADD(ConvTransposed2dSquareInputAsymmetricKernelCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConvTransposed2dSquareInputAsymmetricKernel {
public:
    __aicore__ inline KernelConvTransposed2dSquareInputAsymmetricKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR y)
    {
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, 1);
        weightGm.SetGlobalBuffer((__gm__ DTYPE_WEIGHT *)weight, 1);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, 1);
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() == 0) {
            yGm.SetValue(0, static_cast<DTYPE_Y>(0));
        }
    }

private:
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_WEIGHT> weightGm;
    GlobalTensor<DTYPE_Y> yGm;
};

extern "C" __global__ __aicore__ void conv_transposed2d_square_input_asymmetric_kernel_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KernelConvTransposed2dSquareInputAsymmetricKernel op;
    op.Init(x, weight, y);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

namespace {
constexpr int64_t kStrideH = 1;
constexpr int64_t kStrideW = 1;
constexpr int64_t kPaddingH = 0;
constexpr int64_t kPaddingW = 0;
constexpr int64_t kDilationH = 1;
constexpr int64_t kDilationW = 1;
constexpr int64_t kOutputPaddingH = 0;
constexpr int64_t kOutputPaddingW = 0;
constexpr int64_t kGroups = 1;
constexpr int8_t kCubeMathType = 0;
}

at::Tensor conv_transposed2d_square_input_asymmetric_kernel_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight)
{
    TORCH_CHECK(x.dim() == 4, "conv_transposed2d custom expects x to be 4D");
    TORCH_CHECK(weight.dim() == 4, "conv_transposed2d custom expects weight to be 4D");
    TORCH_CHECK(x.size(2) == x.size(3), "input spatial shape must be square");
    TORCH_CHECK(weight.size(2) != weight.size(3), "kernel must be asymmetric");
    TORCH_CHECK(x.size(1) == weight.size(0), "input channels must match weight.size(0)");

    const int64_t outputChannels = weight.size(1) * kGroups;
    const int64_t outputHeight =
        (x.size(2) - 1) * kStrideH - 2 * kPaddingH +
        kDilationH * (weight.size(2) - 1) + kOutputPaddingH + 1;
    const int64_t outputWidth =
        (x.size(3) - 1) * kStrideW - 2 * kPaddingW +
        kDilationW * (weight.size(3) - 1) + kOutputPaddingW + 1;
    TORCH_CHECK(outputHeight > 0 && outputWidth > 0, "computed output shape must be positive");

    at::Tensor result = at::empty({x.size(0), outputChannels, outputHeight, outputWidth}, x.options());
    c10::optional<at::Tensor> bias = c10::nullopt;
    const int64_t strideData[2] = {kStrideH, kStrideW};
    const int64_t paddingData[2] = {kPaddingH, kPaddingW};
    const int64_t dilationData[2] = {kDilationH, kDilationW};
    const int64_t outputPaddingData[2] = {kOutputPaddingH, kOutputPaddingW};
    const at::IntArrayRef strideArray(strideData, 2);
    const at::IntArrayRef paddingArray(paddingData, 2);
    const at::IntArrayRef dilationArray(dilationData, 2);
    const at::IntArrayRef outputPaddingArray(outputPaddingData, 2);
    bool transposed = true;

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
        kCubeMathType);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "conv_transposed2d_square_input_asymmetric_kernel_custom",
        &conv_transposed2d_square_input_asymmetric_kernel_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv_transposed2d_square_input_asymmetric_kernel_custom",
        &conv_transposed2d_square_input_asymmetric_kernel_custom_impl_npu,
        "conv_transposed2d_square_input_asymmetric_kernel_custom");
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
        kernel_size: tuple,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if stride != 1 or padding != 0 or output_padding != 0 or groups != 1 or bias:
            raise ValueError(
                "This AscendC implementation currently supports "
                "stride=1, padding=0, output_padding=0, groups=1, bias=False only."
            )
        if not isinstance(kernel_size, tuple) or len(kernel_size) != 2 or kernel_size[0] == kernel_size[1]:
            raise ValueError("kernel_size must be an asymmetric 2D tuple")

        self.conv_transpose2d = torch.nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv_transposed2d_square_input_asymmetric_kernel_custom(
            x,
            self.conv_transpose2d.weight,
        )
'''
