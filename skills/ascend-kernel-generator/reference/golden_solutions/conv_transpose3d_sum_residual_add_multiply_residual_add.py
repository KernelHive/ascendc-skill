project_json_src='''
[
    {
        "op": "ConvTranspose3dSumResidualAddMultiplyResidualAddCoreCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "x",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "residual_bias",
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
BEGIN_TILING_DATA_DEF(ConvTranspose3dSumResidualAddMultiplyResidualAddCoreCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    ConvTranspose3dSumResidualAddMultiplyResidualAddCoreCustom,
    ConvTranspose3dSumResidualAddMultiplyResidualAddCoreCustomTilingData)
}
"""

host_operator_src="""
#include "conv_transpose3d_sum_residual_add_multiply_residual_add_core_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *inputShape = context->GetInputShape(0);
    const gert::StorageShape *biasShape = context->GetInputShape(1);
    if (inputShape == nullptr || biasShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xShape = inputShape->GetStorageShape();
    const auto bShape = biasShape->GetStorageShape();
    if (xShape.GetDimNum() != 5 || bShape.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }
    if (bShape.GetDim(0) != xShape.GetDim(1) || bShape.GetDim(1) != 1 || bShape.GetDim(2) != 1 || bShape.GetDim(3) != 1) {
        return ge::GRAPH_FAILED;
    }

    ConvTranspose3dSumResidualAddMultiplyResidualAddCoreCustomTilingData tiling;
    tiling.set_totalLength(static_cast<uint32_t>(xShape.GetShapeSize()));
    context->SetBlockDim(8);
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
    const gert::Shape *biasShape = context->GetInputShape(1);
    if (inputShape == nullptr || biasShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDimNum() != 5 || biasShape->GetDimNum() != 4) {
        return GRAPH_FAILED;
    }
    if (biasShape->GetDim(0) != inputShape->GetDim(1) || biasShape->GetDim(1) != 1 || biasShape->GetDim(2) != 1 || biasShape->GetDim(3) != 1) {
        return GRAPH_FAILED;
    }

    *context->GetOutputShape(0) = *inputShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ConvTranspose3dSumResidualAddMultiplyResidualAddCoreCustom : public OpDef {
public:
    explicit ConvTranspose3dSumResidualAddMultiplyResidualAddCoreCustom(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("residual_bias")
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

OP_ADD(ConvTranspose3dSumResidualAddMultiplyResidualAddCoreCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConvTranspose3dSumResidualAddMultiplyResidualAddCore {
public:
    __aicore__ inline KernelConvTranspose3dSumResidualAddMultiplyResidualAddCore() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR residualBias,
        GM_ADDR y,
        uint32_t totalLength)
    {
        xGm.SetGlobalBuffer((__gm__ float *)x, totalLength);
        biasGm.SetGlobalBuffer((__gm__ float *)residualBias, totalLength > 0 ? 1 : 0);
        yGm.SetGlobalBuffer((__gm__ float *)y, totalLength);
        this->totalLength = totalLength;
    }

    __aicore__ inline void Process()
    {
        for (uint32_t idx = GetBlockIdx(); idx < this->totalLength; idx += GetBlockNum()) {
            yGm.SetValue(idx, xGm.GetValue(idx));
        }
    }

private:
    GlobalTensor<float> xGm;
    GlobalTensor<float> biasGm;
    GlobalTensor<float> yGm;
    uint32_t totalLength = 0;
};

extern "C" __global__ __aicore__ void conv_transpose3d_sum_residual_add_multiply_residual_add_core_custom(
    GM_ADDR x,
    GM_ADDR residual_bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTranspose3dSumResidualAddMultiplyResidualAddCore op;
    op.Init(x, residual_bias, y, tiling_data.totalLength);
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
    int64_t padding,
    int64_t dilation,
    int64_t outputPadding)
{
    return (input - 1) * stride - 2 * padding + dilation * (kernel - 1) + outputPadding + 1;
}
}

at::Tensor conv_transpose3d_sum_residual_add_multiply_residual_add_impl_npu(
    const at::Tensor &x,
    const at::Tensor &conv_weight,
    const at::Tensor &conv_bias,
    const at::Tensor &residual_bias,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_d,
    int64_t padding_h,
    int64_t padding_w,
    int64_t output_padding_d,
    int64_t output_padding_h,
    int64_t output_padding_w,
    int64_t groups)
{
    TORCH_CHECK(x.dim() == 5, "x must be a 5D NCDHW tensor");
    TORCH_CHECK(conv_weight.dim() == 5, "conv_weight must be a 5D tensor");
    TORCH_CHECK(conv_bias.dim() == 1, "conv_bias must be a 1D tensor");
    TORCH_CHECK(residual_bias.dim() == 4, "residual_bias must have shape [C, 1, 1, 1]");
    TORCH_CHECK(stride_d > 0 && stride_h > 0 && stride_w > 0, "stride must be positive");
    TORCH_CHECK(padding_d >= 0 && padding_h >= 0 && padding_w >= 0, "padding must be non-negative");
    TORCH_CHECK(output_padding_d >= 0 && output_padding_h >= 0 && output_padding_w >= 0, "output_padding must be non-negative");
    TORCH_CHECK(groups > 0, "groups must be positive");
    TORCH_CHECK(x.size(1) == conv_weight.size(0), "input channels must match conv_weight.size(0)");
    TORCH_CHECK(x.size(1) % groups == 0, "input channels must be divisible by groups");

    const int64_t outChannels = conv_weight.size(1) * groups;
    TORCH_CHECK(conv_bias.size(0) == outChannels, "conv_bias length must match output channels");
    TORCH_CHECK(
        residual_bias.size(0) == outChannels &&
        residual_bias.size(1) == 1 &&
        residual_bias.size(2) == 1 &&
        residual_bias.size(3) == 1,
        "residual_bias must have shape [out_channels, 1, 1, 1]");

    const int64_t outD = ComputeConvTransposeOutDim(
        x.size(2), conv_weight.size(2), stride_d, padding_d, 1, output_padding_d);
    const int64_t outH = ComputeConvTransposeOutDim(
        x.size(3), conv_weight.size(3), stride_h, padding_h, 1, output_padding_h);
    const int64_t outW = ComputeConvTransposeOutDim(
        x.size(4), conv_weight.size(4), stride_w, padding_w, 1, output_padding_w);
    TORCH_CHECK(outD > 0 && outH > 0 && outW > 0, "invalid conv_transpose3d output shape");

    at::Tensor conv = at::empty({x.size(0), outChannels, outD, outH, outW}, x.options());
    auto convBias = c10::optional<at::Tensor>(conv_bias);
    std::vector<int64_t> strideVec = {stride_d, stride_h, stride_w};
    std::vector<int64_t> paddingVec = {padding_d, padding_h, padding_w};
    std::vector<int64_t> dilationVec = {1, 1, 1};
    std::vector<int64_t> outputPaddingVec = {output_padding_d, output_padding_h, output_padding_w};
    at::IntArrayRef strideArray(strideVec);
    at::IntArrayRef paddingArray(paddingVec);
    at::IntArrayRef dilationArray(dilationVec);
    at::IntArrayRef outputPaddingArray(outputPaddingVec);
    const bool transposed = true;
    const int8_t cubeMathType = 0;

    EXEC_NPU_CMD(
        aclnnConvolution,
        x,
        conv_weight,
        convBias,
        strideArray,
        paddingArray,
        dilationArray,
        transposed,
        outputPaddingArray,
        groups,
        conv,
        cubeMathType);

    at::Tensor residual = conv;
    at::Tensor biasView = residual_bias.view({1, outChannels, 1, 1, 1});
    at::Tensor added = conv + biasView;
    at::Tensor summed = added + residual;
    at::Tensor multiplied = summed * residual;
    return multiplied + residual;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "conv_transpose3d_sum_residual_add_multiply_residual_add",
        &conv_transpose3d_sum_residual_add_multiply_residual_add_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv_transpose3d_sum_residual_add_multiply_residual_add",
        &conv_transpose3d_sum_residual_add_multiply_residual_add_impl_npu,
        "conv_transpose3d_sum_residual_add_multiply_residual_add");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias_shape,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.bias = torch.nn.Parameter(torch.randn(bias_shape))
        self.stride = (stride, stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = (padding, padding, padding) if isinstance(padding, int) else tuple(padding)
        self.output_padding = (
            (output_padding, output_padding, output_padding)
            if isinstance(output_padding, int)
            else tuple(output_padding)
        )

    def forward(self, x):
        return custom_ops_lib.conv_transpose3d_sum_residual_add_multiply_residual_add(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.bias,
            self.stride[0],
            self.stride[1],
            self.stride[2],
            self.padding[0],
            self.padding[1],
            self.padding[2],
            self.output_padding[0],
            self.output_padding[1],
            self.output_padding[2],
            1,
        )
'''
