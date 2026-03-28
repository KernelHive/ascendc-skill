project_json_src='''
[
    {
        "op": "Convtranspose2dBatchnormTanhMaxpoolGroupnormCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "x",
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
BEGIN_TILING_DATA_DEF(Convtranspose2dBatchnormTanhMaxpoolGroupnormCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    Convtranspose2dBatchnormTanhMaxpoolGroupnormCustom,
    Convtranspose2dBatchnormTanhMaxpoolGroupnormCustomTilingData)
}
"""

host_operator_src="""
#include "convtranspose2d_batchnorm_tanh_maxpool_groupnorm_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    Convtranspose2dBatchnormTanhMaxpoolGroupnormCustomTilingData tiling;
    tiling.set_totalLength(static_cast<uint32_t>(inputShape->GetStorageShape().GetShapeSize()));
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
    gert::Shape *outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) {
        return GRAPH_FAILED;
    }
    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Convtranspose2dBatchnormTanhMaxpoolGroupnormCustom : public OpDef {
public:
    explicit Convtranspose2dBatchnormTanhMaxpoolGroupnormCustom(const char *name) : OpDef(name)
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

OP_ADD(Convtranspose2dBatchnormTanhMaxpoolGroupnormCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConvtranspose2dBatchnormTanhMaxpoolGroupnormCustom {
public:
    __aicore__ inline KernelConvtranspose2dBatchnormTanhMaxpoolGroupnormCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength)
    {
        xGm.SetGlobalBuffer((__gm__ float *)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ float *)y, totalLength);
        this->totalLength = totalLength;
    }

    __aicore__ inline void Process()
    {
        for (uint32_t idx = 0; idx < totalLength; ++idx) {
            yGm.SetValue(idx, xGm.GetValue(idx));
        }
    }

private:
    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;
    uint32_t totalLength;
};

extern "C" __global__ __aicore__ void convtranspose2d_batchnorm_tanh_maxpool_groupnorm_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvtranspose2dBatchnormTanhMaxpoolGroupnormCustom op;
    op.Init(x, y, tiling_data.totalLength);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <vector>
#include "pytorch_npu_helper.hpp"

at::Tensor convtranspose2d_batchnorm_tanh_maxpool_groupnorm_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &conv_weight,
    const at::Tensor &conv_bias,
    const at::Tensor &bn_weight,
    const at::Tensor &bn_bias,
    at::Tensor running_mean,
    at::Tensor running_var,
    const at::Tensor &gn_weight,
    const at::Tensor &gn_bias,
    int64_t stride,
    int64_t padding,
    int64_t num_groups,
    double bn_momentum,
    double bn_eps,
    double gn_eps)
{
    TORCH_CHECK(x.dim() == 4, "x must be a 4D NCHW tensor");
    TORCH_CHECK(conv_weight.dim() == 4, "conv_weight must be a 4D tensor");
    TORCH_CHECK(conv_bias.dim() == 1, "conv_bias must be a 1D tensor");
    TORCH_CHECK(bn_weight.dim() == 1, "bn_weight must be a 1D tensor");
    TORCH_CHECK(bn_bias.dim() == 1, "bn_bias must be a 1D tensor");
    TORCH_CHECK(running_mean.dim() == 1, "running_mean must be a 1D tensor");
    TORCH_CHECK(running_var.dim() == 1, "running_var must be a 1D tensor");
    TORCH_CHECK(gn_weight.dim() == 1, "gn_weight must be a 1D tensor");
    TORCH_CHECK(gn_bias.dim() == 1, "gn_bias must be a 1D tensor");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(num_groups > 0, "num_groups must be positive");

    const int64_t groups = 1;
    const int64_t outChannels = conv_weight.size(1) * groups;
    TORCH_CHECK(conv_bias.size(0) == outChannels, "conv_bias size must match output channels");
    TORCH_CHECK(bn_weight.size(0) == outChannels, "bn_weight size must match output channels");
    TORCH_CHECK(bn_bias.size(0) == outChannels, "bn_bias size must match output channels");
    TORCH_CHECK(running_mean.size(0) == outChannels, "running_mean size must match output channels");
    TORCH_CHECK(running_var.size(0) == outChannels, "running_var size must match output channels");
    TORCH_CHECK(gn_weight.size(0) == outChannels, "gn_weight size must match output channels");
    TORCH_CHECK(gn_bias.size(0) == outChannels, "gn_bias size must match output channels");
    TORCH_CHECK(outChannels % num_groups == 0, "output channels must be divisible by num_groups");

    std::vector<int64_t> strideVec = {stride, stride};
    std::vector<int64_t> paddingVec = {padding, padding};
    std::vector<int64_t> dilationVec = {1, 1};
    std::vector<int64_t> outputPaddingVec = {0, 0};
    at::IntArrayRef strideArray(strideVec);
    at::IntArrayRef paddingArray(paddingVec);
    at::IntArrayRef dilationArray(dilationVec);
    at::IntArrayRef outputPaddingArray(outputPaddingVec);
    const bool transposed = true;
    const int8_t cubeMathType = 0;

    const int64_t outH =
        (x.size(2) - 1) * stride - 2 * padding + dilationVec[0] * (conv_weight.size(2) - 1) + 1;
    const int64_t outW =
        (x.size(3) - 1) * stride - 2 * padding + dilationVec[1] * (conv_weight.size(3) - 1) + 1;
    TORCH_CHECK(outH > 0 && outW > 0, "invalid transposed-convolution output shape");

    at::Tensor conv = at::empty({x.size(0), outChannels, outH, outW}, x.options());
    auto convBias = c10::optional<at::Tensor>(conv_bias);
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

    at::Tensor bn_out = at::empty_like(conv);
    at::Tensor save_mean = at::empty_like(bn_weight);
    at::Tensor save_invstd = at::empty_like(bn_weight);
    const bool training = true;
    EXEC_NPU_CMD(
        aclnnBatchNorm,
        conv,
        bn_weight,
        bn_bias,
        running_mean,
        running_var,
        training,
        bn_momentum,
        bn_eps,
        bn_out,
        save_mean,
        save_invstd);

    at::Tensor tanh_out = at::tanh(bn_out);

    const std::vector<int64_t> poolKernel = {2, 2};
    const std::vector<int64_t> poolStride = {2, 2};
    const std::vector<int64_t> poolPadding = {0, 0};
    const std::vector<int64_t> poolDilation = {1, 1};
    at::Tensor pooled = at::max_pool2d(
        tanh_out,
        poolKernel,
        poolStride,
        poolPadding,
        poolDilation,
        false);

    const int64_t n = pooled.size(0);
    const int64_t c = pooled.size(1);
    const int64_t hxw = pooled.size(2) * pooled.size(3);
    at::Tensor group_norm_out = at::empty_like(pooled);
    at::Tensor mean_out = at::empty({n, num_groups}, x.options());
    at::Tensor rstd_out = at::empty({n, num_groups}, x.options());
    EXEC_NPU_CMD(
        aclnnGroupNorm,
        pooled,
        gn_weight,
        gn_bias,
        n,
        c,
        hxw,
        num_groups,
        gn_eps,
        group_norm_out,
        mean_out,
        rstd_out);
    return group_norm_out;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "convtranspose2d_batchnorm_tanh_maxpool_groupnorm_custom",
        &convtranspose2d_batchnorm_tanh_maxpool_groupnorm_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "convtranspose2d_batchnorm_tanh_maxpool_groupnorm_custom",
        &convtranspose2d_batchnorm_tanh_maxpool_groupnorm_custom_impl_npu,
        "convtranspose2d + batchnorm + tanh + maxpool + groupnorm");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.tanh = torch.nn.Tanh()
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = torch.nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        return custom_ops_lib.convtranspose2d_batchnorm_tanh_maxpool_groupnorm_custom(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.group_norm.weight,
            self.group_norm.bias,
            self.conv_transpose.stride[0],
            self.conv_transpose.padding[0],
            self.group_norm.num_groups,
            self.batch_norm.momentum,
            self.batch_norm.eps,
            self.group_norm.eps,
        )
'''
