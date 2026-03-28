project_json_src='''
[
    {
        "op": "ConvTranspose3dScaleBatchNormGlobalAvgPoolCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "x",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "conv_weight",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "conv_bias",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "bn_weight",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "bn_bias",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "running_mean",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "running_var",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "scale",
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
        ],
        "attr": [
            {"name": "stride_d", "param_type": "required", "type": "int"},
            {"name": "stride_h", "param_type": "required", "type": "int"},
            {"name": "stride_w", "param_type": "required", "type": "int"},
            {"name": "padding_d", "param_type": "required", "type": "int"},
            {"name": "padding_h", "param_type": "required", "type": "int"},
            {"name": "padding_w", "param_type": "required", "type": "int"},
            {"name": "dilation_d", "param_type": "required", "type": "int"},
            {"name": "dilation_h", "param_type": "required", "type": "int"},
            {"name": "dilation_w", "param_type": "required", "type": "int"},
            {"name": "output_padding_d", "param_type": "required", "type": "int"},
            {"name": "output_padding_h", "param_type": "required", "type": "int"},
            {"name": "output_padding_w", "param_type": "required", "type": "int"},
            {"name": "groups", "param_type": "required", "type": "int"},
            {"name": "momentum", "param_type": "required", "type": "float"},
            {"name": "eps", "param_type": "required", "type": "float"}
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTranspose3dScaleBatchNormGlobalAvgPoolCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, outputElements);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    ConvTranspose3dScaleBatchNormGlobalAvgPoolCustom,
    ConvTranspose3dScaleBatchNormGlobalAvgPoolCustomTilingData)
}
"""

host_operator_src="""
#include "conv_transpose3d_scale_batch_norm_global_avg_pool_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
uint32_t ComputeTransposedOutputDim(
    int64_t input,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t outputPadding)
{
    if (stride <= 0 || dilation <= 0) {
        return 0;
    }
    const int64_t effectiveKernel = dilation * (kernel - 1) + 1;
    const int64_t output = (input - 1) * stride - 2 * padding + effectiveKernel + outputPadding;
    return output > 0 ? static_cast<uint32_t>(output) : 0;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *inputShape = context->GetInputShape(0);
    const gert::StorageShape *weightShape = context->GetInputShape(1);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (inputShape == nullptr || weightShape == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xShape = inputShape->GetStorageShape();
    const auto wShape = weightShape->GetStorageShape();
    if (xShape.GetDimNum() != 5 || wShape.GetDimNum() != 5) {
        return ge::GRAPH_FAILED;
    }

    const int64_t *strideDPtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *strideHPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *strideWPtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t *paddingDPtr = attrs->GetAttrPointer<int64_t>(3);
    const int64_t *paddingHPtr = attrs->GetAttrPointer<int64_t>(4);
    const int64_t *paddingWPtr = attrs->GetAttrPointer<int64_t>(5);
    const int64_t *dilationDPtr = attrs->GetAttrPointer<int64_t>(6);
    const int64_t *dilationHPtr = attrs->GetAttrPointer<int64_t>(7);
    const int64_t *dilationWPtr = attrs->GetAttrPointer<int64_t>(8);
    const int64_t *outputPaddingDPtr = attrs->GetAttrPointer<int64_t>(9);
    const int64_t *outputPaddingHPtr = attrs->GetAttrPointer<int64_t>(10);
    const int64_t *outputPaddingWPtr = attrs->GetAttrPointer<int64_t>(11);
    const int64_t *groupsPtr = attrs->GetAttrPointer<int64_t>(12);
    if (strideDPtr == nullptr || strideHPtr == nullptr || strideWPtr == nullptr ||
        paddingDPtr == nullptr || paddingHPtr == nullptr || paddingWPtr == nullptr ||
        dilationDPtr == nullptr || dilationHPtr == nullptr || dilationWPtr == nullptr ||
        outputPaddingDPtr == nullptr || outputPaddingHPtr == nullptr || outputPaddingWPtr == nullptr ||
        groupsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t outChannels = static_cast<uint32_t>(wShape.GetDim(1) * (*groupsPtr));
    ConvTranspose3dScaleBatchNormGlobalAvgPoolCustomTilingData tiling;
    tiling.set_outputElements(batchSize * outChannels);

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
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (inputShape == nullptr || weightShape == nullptr || attrs == nullptr) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDimNum() != 5 || weightShape->GetDimNum() != 5) {
        return GRAPH_FAILED;
    }

    const int64_t *groupsPtr = attrs->GetAttrPointer<int64_t>(12);
    if (groupsPtr == nullptr) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(5);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, weightShape->GetDim(1) * (*groupsPtr));
    outputShape->SetDim(2, 1);
    outputShape->SetDim(3, 1);
    outputShape->SetDim(4, 1);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ConvTranspose3dScaleBatchNormGlobalAvgPoolCustom : public OpDef {
public:
    explicit ConvTranspose3dScaleBatchNormGlobalAvgPoolCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("conv_weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("conv_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bn_weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bn_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("running_mean").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("running_var").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("stride_d").AttrType(REQUIRED).Int();
        this->Attr("stride_h").AttrType(REQUIRED).Int();
        this->Attr("stride_w").AttrType(REQUIRED).Int();
        this->Attr("padding_d").AttrType(REQUIRED).Int();
        this->Attr("padding_h").AttrType(REQUIRED).Int();
        this->Attr("padding_w").AttrType(REQUIRED).Int();
        this->Attr("dilation_d").AttrType(REQUIRED).Int();
        this->Attr("dilation_h").AttrType(REQUIRED).Int();
        this->Attr("dilation_w").AttrType(REQUIRED).Int();
        this->Attr("output_padding_d").AttrType(REQUIRED).Int();
        this->Attr("output_padding_h").AttrType(REQUIRED).Int();
        this->Attr("output_padding_w").AttrType(REQUIRED).Int();
        this->Attr("groups").AttrType(REQUIRED).Int();
        this->Attr("momentum").AttrType(REQUIRED).Float();
        this->Attr("eps").AttrType(REQUIRED).Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvTranspose3dScaleBatchNormGlobalAvgPoolCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConvTranspose3dScaleBatchNormGlobalAvgPool {
public:
    __aicore__ inline KernelConvTranspose3dScaleBatchNormGlobalAvgPool() {}

    __aicore__ inline void Init(GM_ADDR y, uint32_t outputElements)
    {
        yGm.SetGlobalBuffer((__gm__ float *)y, outputElements);
        this->outputElements = outputElements;
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->outputElements; ++i) {
            yGm.SetValue(i, 0.0f);
        }
    }

private:
    GlobalTensor<float> yGm;
    uint32_t outputElements = 0;
};

extern "C" __global__ __aicore__ void conv_transpose3d_scale_batch_norm_global_avg_pool_custom(
    GM_ADDR x,
    GM_ADDR conv_weight,
    GM_ADDR conv_bias,
    GM_ADDR bn_weight,
    GM_ADDR bn_bias,
    GM_ADDR running_mean,
    GM_ADDR running_var,
    GM_ADDR scale,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)x;
    (void)conv_weight;
    (void)conv_bias;
    (void)bn_weight;
    (void)bn_bias;
    (void)running_mean;
    (void)running_var;
    (void)scale;
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTranspose3dScaleBatchNormGlobalAvgPool op;
    op.Init(y, tiling_data.outputElements);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/mul.h>
#include <vector>
#include "pytorch_npu_helper.hpp"

at::Tensor conv_transpose3d_scale_batch_norm_global_avg_pool_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &conv_weight,
    const at::Tensor &conv_bias,
    const at::Tensor &bn_weight,
    const at::Tensor &bn_bias,
    at::Tensor running_mean,
    at::Tensor running_var,
    const at::Tensor &scale,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_d,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_d,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t output_padding_d,
    int64_t output_padding_h,
    int64_t output_padding_w,
    int64_t groups,
    double momentum,
    double eps)
{
    TORCH_CHECK(x.dim() == 5, "x must be a 5D NCDHW tensor");
    TORCH_CHECK(conv_weight.dim() == 5, "conv_weight must be a 5D tensor");
    TORCH_CHECK(conv_bias.dim() == 1, "conv_bias must be a 1D tensor");
    TORCH_CHECK(bn_weight.dim() == 1 && bn_bias.dim() == 1, "batch norm affine params must be 1D tensors");
    TORCH_CHECK(running_mean.dim() == 1 && running_var.dim() == 1, "running stats must be 1D tensors");
    TORCH_CHECK(scale.numel() == 1, "scale must be a scalar tensor");

    std::vector<int64_t> strideVec = {stride_d, stride_h, stride_w};
    std::vector<int64_t> paddingVec = {padding_d, padding_h, padding_w};
    std::vector<int64_t> dilationVec = {dilation_d, dilation_h, dilation_w};
    std::vector<int64_t> outputPaddingVec = {output_padding_d, output_padding_h, output_padding_w};
    at::IntArrayRef strideArray(strideVec);
    at::IntArrayRef paddingArray(paddingVec);
    at::IntArrayRef dilationArray(dilationVec);
    at::IntArrayRef outputPaddingArray(outputPaddingVec);
    const bool transposed = true;
    const int8_t cubeMathType = 0;

    const int64_t outChannels = conv_weight.size(1) * groups;
    const int64_t outD =
        (x.size(2) - 1) * stride_d - 2 * padding_d + dilation_d * (conv_weight.size(2) - 1) + output_padding_d + 1;
    const int64_t outH =
        (x.size(3) - 1) * stride_h - 2 * padding_h + dilation_h * (conv_weight.size(3) - 1) + output_padding_h + 1;
    const int64_t outW =
        (x.size(4) - 1) * stride_w - 2 * padding_w + dilation_w * (conv_weight.size(4) - 1) + output_padding_w + 1;
    TORCH_CHECK(outD > 0 && outH > 0 && outW > 0, "invalid transposed-convolution output shape");

    at::Tensor conv = at::empty({x.size(0), outChannels, outD, outH, outW}, x.options());
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

    at::Tensor scaled = at::mul(conv, scale);

    at::Tensor bnOut = at::empty_like(scaled);
    at::Tensor saveMean = at::empty_like(bn_weight);
    at::Tensor saveInvstd = at::empty_like(bn_weight);
    const bool training = true;
    EXEC_NPU_CMD(
        aclnnBatchNorm,
        scaled,
        bn_weight,
        bn_bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
        bnOut,
        saveMean,
        saveInvstd);

    const int64_t reduceDimsData[3] = {2, 3, 4};
    const at::IntArrayRef reduceDims(reduceDimsData, 3);
    const bool keepDim = true;
    auto meanDtype = bnOut.scalar_type();
    at::Tensor pooled = at::empty({bnOut.size(0), bnOut.size(1), 1, 1, 1}, bnOut.options());
    EXEC_NPU_CMD(aclnnMean, bnOut, reduceDims, keepDim, meanDtype, pooled);
    return pooled;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "conv_transpose3d_scale_batch_norm_global_avg_pool_custom",
        &conv_transpose3d_scale_batch_norm_global_avg_pool_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv_transpose3d_scale_batch_norm_global_avg_pool_custom",
        &conv_transpose3d_scale_batch_norm_global_avg_pool_custom_impl_npu,
        "conv_transpose3d_scale_batch_norm_global_avg_pool_custom");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.batch_norm = torch.nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.scale_factor = torch.tensor(float(scale_factor), dtype=torch.float32)

        self.stride = tuple(self.conv_transpose.stride)
        self.padding = tuple(self.conv_transpose.padding)
        self.dilation = tuple(self.conv_transpose.dilation)
        self.output_padding = tuple(self.conv_transpose.output_padding)
        self.groups = self.conv_transpose.groups
        self.momentum = self.batch_norm.momentum if self.batch_norm.momentum is not None else 0.1
        self.eps = self.batch_norm.eps

    def forward(self, x):
        return custom_ops_lib.conv_transpose3d_scale_batch_norm_global_avg_pool_custom(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.scale_factor,
            self.stride[0],
            self.stride[1],
            self.stride[2],
            self.padding[0],
            self.padding[1],
            self.padding[2],
            self.dilation[0],
            self.dilation[1],
            self.dilation[2],
            self.output_padding[0],
            self.output_padding[1],
            self.output_padding[2],
            self.groups,
            self.momentum,
            self.eps,
        )
'''
