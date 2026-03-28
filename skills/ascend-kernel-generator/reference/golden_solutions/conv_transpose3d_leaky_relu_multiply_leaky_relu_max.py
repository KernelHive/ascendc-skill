project_json_src='''
[
    {
        "op": "ConvTranspose3dLeakyReluMultiplyLeakyReluMaxCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "x",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "weight",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "bias",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "multiplier",
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
            {"name": "negative_slope", "param_type": "required", "type": "float"},
            {"name": "pool_kernel_size", "param_type": "required", "type": "int"},
            {"name": "pool_stride", "param_type": "required", "type": "int"},
            {"name": "pool_padding", "param_type": "required", "type": "int"},
            {"name": "pool_dilation", "param_type": "required", "type": "int"}
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTranspose3dLeakyReluMultiplyLeakyReluMaxCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, outputElements);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    ConvTranspose3dLeakyReluMultiplyLeakyReluMaxCustom,
    ConvTranspose3dLeakyReluMultiplyLeakyReluMaxCustomTilingData)
}
"""

host_operator_src="""
#include "conv_transpose3d_leaky_relu_multiply_leaky_relu_max_custom_tiling.h"
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

uint32_t ComputePoolOutputDim(
    int64_t input,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    int64_t dilation)
{
    if (stride <= 0 || kernel <= 0 || dilation <= 0) {
        return 0;
    }
    const int64_t effectiveKernel = dilation * (kernel - 1) + 1;
    const int64_t numerator = input + 2 * padding - effectiveKernel;
    if (numerator < 0) {
        return 0;
    }
    return static_cast<uint32_t>(numerator / stride + 1);
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
    const int64_t *poolKernelSizePtr = attrs->GetAttrPointer<int64_t>(14);
    const int64_t *poolStridePtr = attrs->GetAttrPointer<int64_t>(15);
    const int64_t *poolPaddingPtr = attrs->GetAttrPointer<int64_t>(16);
    const int64_t *poolDilationPtr = attrs->GetAttrPointer<int64_t>(17);
    if (strideDPtr == nullptr || strideHPtr == nullptr || strideWPtr == nullptr ||
        paddingDPtr == nullptr || paddingHPtr == nullptr || paddingWPtr == nullptr ||
        dilationDPtr == nullptr || dilationHPtr == nullptr || dilationWPtr == nullptr ||
        outputPaddingDPtr == nullptr || outputPaddingHPtr == nullptr || outputPaddingWPtr == nullptr ||
        groupsPtr == nullptr || poolKernelSizePtr == nullptr || poolStridePtr == nullptr ||
        poolPaddingPtr == nullptr || poolDilationPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t outChannels = static_cast<uint32_t>(wShape.GetDim(1) * (*groupsPtr));
    const uint32_t convOutD = ComputeTransposedOutputDim(
        xShape.GetDim(2), wShape.GetDim(2), *strideDPtr, *paddingDPtr, *dilationDPtr, *outputPaddingDPtr);
    const uint32_t convOutH = ComputeTransposedOutputDim(
        xShape.GetDim(3), wShape.GetDim(3), *strideHPtr, *paddingHPtr, *dilationHPtr, *outputPaddingHPtr);
    const uint32_t convOutW = ComputeTransposedOutputDim(
        xShape.GetDim(4), wShape.GetDim(4), *strideWPtr, *paddingWPtr, *dilationWPtr, *outputPaddingWPtr);
    const uint32_t poolOutD = ComputePoolOutputDim(
        convOutD, *poolKernelSizePtr, *poolStridePtr, *poolPaddingPtr, *poolDilationPtr);
    const uint32_t poolOutH = ComputePoolOutputDim(
        convOutH, *poolKernelSizePtr, *poolStridePtr, *poolPaddingPtr, *poolDilationPtr);
    const uint32_t poolOutW = ComputePoolOutputDim(
        convOutW, *poolKernelSizePtr, *poolStridePtr, *poolPaddingPtr, *poolDilationPtr);

    ConvTranspose3dLeakyReluMultiplyLeakyReluMaxCustomTilingData tiling;
    tiling.set_outputElements(batchSize * outChannels * poolOutD * poolOutH * poolOutW);

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
    const int64_t *poolKernelSizePtr = attrs->GetAttrPointer<int64_t>(14);
    const int64_t *poolStridePtr = attrs->GetAttrPointer<int64_t>(15);
    const int64_t *poolPaddingPtr = attrs->GetAttrPointer<int64_t>(16);
    const int64_t *poolDilationPtr = attrs->GetAttrPointer<int64_t>(17);
    if (strideDPtr == nullptr || strideHPtr == nullptr || strideWPtr == nullptr ||
        paddingDPtr == nullptr || paddingHPtr == nullptr || paddingWPtr == nullptr ||
        dilationDPtr == nullptr || dilationHPtr == nullptr || dilationWPtr == nullptr ||
        outputPaddingDPtr == nullptr || outputPaddingHPtr == nullptr || outputPaddingWPtr == nullptr ||
        groupsPtr == nullptr || poolKernelSizePtr == nullptr || poolStridePtr == nullptr ||
        poolPaddingPtr == nullptr || poolDilationPtr == nullptr) {
        return GRAPH_FAILED;
    }

    const uint32_t convOutD = ComputeTransposedOutputDim(
        inputShape->GetDim(2), weightShape->GetDim(2), *strideDPtr, *paddingDPtr, *dilationDPtr, *outputPaddingDPtr);
    const uint32_t convOutH = ComputeTransposedOutputDim(
        inputShape->GetDim(3), weightShape->GetDim(3), *strideHPtr, *paddingHPtr, *dilationHPtr, *outputPaddingHPtr);
    const uint32_t convOutW = ComputeTransposedOutputDim(
        inputShape->GetDim(4), weightShape->GetDim(4), *strideWPtr, *paddingWPtr, *dilationWPtr, *outputPaddingWPtr);

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(5);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, weightShape->GetDim(1) * (*groupsPtr));
    outputShape->SetDim(2, ComputePoolOutputDim(convOutD, *poolKernelSizePtr, *poolStridePtr, *poolPaddingPtr, *poolDilationPtr));
    outputShape->SetDim(3, ComputePoolOutputDim(convOutH, *poolKernelSizePtr, *poolStridePtr, *poolPaddingPtr, *poolDilationPtr));
    outputShape->SetDim(4, ComputePoolOutputDim(convOutW, *poolKernelSizePtr, *poolStridePtr, *poolPaddingPtr, *poolDilationPtr));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ConvTranspose3dLeakyReluMultiplyLeakyReluMaxCustom : public OpDef {
public:
    explicit ConvTranspose3dLeakyReluMultiplyLeakyReluMaxCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("multiplier").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
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
        this->Attr("negative_slope").AttrType(REQUIRED).Float();
        this->Attr("pool_kernel_size").AttrType(REQUIRED).Int();
        this->Attr("pool_stride").AttrType(REQUIRED).Int();
        this->Attr("pool_padding").AttrType(REQUIRED).Int();
        this->Attr("pool_dilation").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvTranspose3dLeakyReluMultiplyLeakyReluMaxCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConvTranspose3dLeakyReluMultiplyLeakyReluMax {
public:
    __aicore__ inline KernelConvTranspose3dLeakyReluMultiplyLeakyReluMax() {}

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

extern "C" __global__ __aicore__ void conv_transpose3d_leaky_relu_multiply_leaky_relu_max_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR multiplier,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)x;
    (void)weight;
    (void)bias;
    (void)multiplier;
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTranspose3dLeakyReluMultiplyLeakyReluMax op;
    op.Init(y, tiling_data.outputElements);
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

namespace {
int64_t ComputeTransposedOutputDim(
    int64_t input,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t outputPadding)
{
    const int64_t effectiveKernel = dilation * (kernel - 1) + 1;
    return (input - 1) * stride - 2 * padding + effectiveKernel + outputPadding;
}
}

at::Tensor conv_transpose3d_leaky_relu_multiply_leaky_relu_max_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &bias,
    const at::Tensor &multiplier,
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
    double negative_slope,
    int64_t pool_kernel_size,
    int64_t pool_stride,
    int64_t pool_padding,
    int64_t pool_dilation)
{
    TORCH_CHECK(x.dim() == 5, "x must be a 5D NCDHW tensor");
    TORCH_CHECK(weight.dim() == 5, "weight must be a 5D tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
    TORCH_CHECK(multiplier.numel() == weight.size(1) * groups, "multiplier must have one value per output channel");
    TORCH_CHECK(groups > 0, "groups must be positive");

    const int64_t outChannels = weight.size(1) * groups;
    const int64_t outD =
        ComputeTransposedOutputDim(x.size(2), weight.size(2), stride_d, padding_d, dilation_d, output_padding_d);
    const int64_t outH =
        ComputeTransposedOutputDim(x.size(3), weight.size(3), stride_h, padding_h, dilation_h, output_padding_h);
    const int64_t outW =
        ComputeTransposedOutputDim(x.size(4), weight.size(4), stride_w, padding_w, dilation_w, output_padding_w);
    TORCH_CHECK(outD > 0 && outH > 0 && outW > 0, "invalid transposed convolution output shape");

    at::Tensor conv = at::empty({x.size(0), outChannels, outD, outH, outW}, x.options());
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
    auto biasOptional = c10::optional<at::Tensor>(bias);

    EXEC_NPU_CMD(
        aclnnConvolution,
        x,
        weight,
        biasOptional,
        strideArray,
        paddingArray,
        dilationArray,
        transposed,
        outputPaddingArray,
        groups,
        conv,
        cubeMathType);

    at::Tensor activated1 = at::leaky_relu(conv, negative_slope);
    at::Tensor reshapedMultiplier = multiplier.reshape({1, outChannels, 1, 1, 1});
    at::Tensor scaled = at::mul(activated1, reshapedMultiplier);
    at::Tensor activated2 = at::leaky_relu(scaled, negative_slope);

    std::vector<int64_t> poolKernel = {pool_kernel_size, pool_kernel_size, pool_kernel_size};
    std::vector<int64_t> poolStride = {pool_stride, pool_stride, pool_stride};
    std::vector<int64_t> poolPadding = {pool_padding, pool_padding, pool_padding};
    std::vector<int64_t> poolDilation = {pool_dilation, pool_dilation, pool_dilation};
    return at::max_pool3d(
        activated2,
        at::IntArrayRef(poolKernel),
        at::IntArrayRef(poolStride),
        at::IntArrayRef(poolPadding),
        at::IntArrayRef(poolDilation),
        false);
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "conv_transpose3d_leaky_relu_multiply_leaky_relu_max_custom",
        &conv_transpose3d_leaky_relu_multiply_leaky_relu_max_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv_transpose3d_leaky_relu_multiply_leaky_relu_max_custom",
        &conv_transpose3d_leaky_relu_multiply_leaky_relu_max_custom_impl_npu,
        "conv_transpose3d + leaky_relu + multiply + leaky_relu + max");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.multiplier = torch.nn.Parameter(torch.randn(multiplier_shape, dtype=self.conv_transpose.weight.dtype))
        self.negative_slope = 0.2
        self.pool_kernel_size = 2
        self.pool_stride = 2
        self.pool_padding = 0
        self.pool_dilation = 1

        if isinstance(self.conv_transpose.stride, int):
            self.stride = (self.conv_transpose.stride,) * 3
        else:
            self.stride = tuple(self.conv_transpose.stride)
        if isinstance(self.conv_transpose.padding, int):
            self.padding = (self.conv_transpose.padding,) * 3
        else:
            self.padding = tuple(self.conv_transpose.padding)
        if isinstance(self.conv_transpose.output_padding, int):
            self.output_padding = (self.conv_transpose.output_padding,) * 3
        else:
            self.output_padding = tuple(self.conv_transpose.output_padding)
        if isinstance(self.conv_transpose.dilation, int):
            self.dilation = (self.conv_transpose.dilation,) * 3
        else:
            self.dilation = tuple(self.conv_transpose.dilation)
        self.groups = self.conv_transpose.groups

    def forward(self, x):
        return custom_ops_lib.conv_transpose3d_leaky_relu_multiply_leaky_relu_max_custom(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.multiplier,
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
            self.negative_slope,
            self.pool_kernel_size,
            self.pool_stride,
            self.pool_padding,
            self.pool_dilation,
        )
'''
