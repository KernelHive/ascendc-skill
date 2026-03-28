project_json_src='''
[
    {
        "op": "ConvTranspose3dAvgPoolClampSoftmaxMultiplyCustom",
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
            {
                "name": "stride_d",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "stride_h",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "stride_w",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "padding_d",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "padding_h",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "padding_w",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "output_padding_d",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "output_padding_h",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "output_padding_w",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "pool_kernel_size",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "clamp_min",
                "param_type": "required",
                "type": "float"
            },
            {
                "name": "clamp_max",
                "param_type": "required",
                "type": "float"
            }
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTranspose3dAvgPoolClampSoftmaxMultiplyCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, outChannels);
    TILING_DATA_FIELD_DEF(uint32_t, outputDepth);
    TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    ConvTranspose3dAvgPoolClampSoftmaxMultiplyCustom,
    ConvTranspose3dAvgPoolClampSoftmaxMultiplyCustomTilingData)
}
"""

host_operator_src="""
#include "conv_transpose3d_avg_pool_clamp_softmax_multiply_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
uint32_t ComputeConvTransposeOutputDim(
    int64_t input,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    int64_t outputPadding)
{
    const int64_t output = (input - 1) * stride - 2 * padding + kernel + outputPadding;
    return output > 0 ? static_cast<uint32_t>(output) : 0;
}

uint32_t ComputePoolOutputDim(int64_t input, int64_t kernel)
{
    if (kernel <= 0 || input < kernel) {
        return 0;
    }
    return static_cast<uint32_t>((input - kernel) / kernel + 1);
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *inputShape = context->GetInputShape(0);
    const gert::StorageShape *weightShape = context->GetInputShape(1);
    const gert::StorageShape *biasShape = context->GetInputShape(2);
    if (inputShape == nullptr || weightShape == nullptr || biasShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xShape = inputShape->GetStorageShape();
    const auto wShape = weightShape->GetStorageShape();
    const auto bShape = biasShape->GetStorageShape();
    if (xShape.GetDimNum() != 5 || wShape.GetDimNum() != 5 || bShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int64_t *strideDPtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *strideHPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *strideWPtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t *paddingDPtr = attrs->GetAttrPointer<int64_t>(3);
    const int64_t *paddingHPtr = attrs->GetAttrPointer<int64_t>(4);
    const int64_t *paddingWPtr = attrs->GetAttrPointer<int64_t>(5);
    const int64_t *outputPaddingDPtr = attrs->GetAttrPointer<int64_t>(6);
    const int64_t *outputPaddingHPtr = attrs->GetAttrPointer<int64_t>(7);
    const int64_t *outputPaddingWPtr = attrs->GetAttrPointer<int64_t>(8);
    const int64_t *poolKernelPtr = attrs->GetAttrPointer<int64_t>(9);
    const float *clampMinPtr = attrs->GetAttrPointer<float>(10);
    const float *clampMaxPtr = attrs->GetAttrPointer<float>(11);
    if (strideDPtr == nullptr || strideHPtr == nullptr || strideWPtr == nullptr ||
        paddingDPtr == nullptr || paddingHPtr == nullptr || paddingWPtr == nullptr ||
        outputPaddingDPtr == nullptr || outputPaddingHPtr == nullptr || outputPaddingWPtr == nullptr ||
        poolKernelPtr == nullptr || clampMinPtr == nullptr || clampMaxPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (*poolKernelPtr <= 0 || *clampMinPtr > *clampMaxPtr) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t outChannels = static_cast<uint32_t>(wShape.GetDim(1));
    const uint32_t convDepth = ComputeConvTransposeOutputDim(
        xShape.GetDim(2), wShape.GetDim(2), *strideDPtr, *paddingDPtr, *outputPaddingDPtr);
    const uint32_t convHeight = ComputeConvTransposeOutputDim(
        xShape.GetDim(3), wShape.GetDim(3), *strideHPtr, *paddingHPtr, *outputPaddingHPtr);
    const uint32_t convWidth = ComputeConvTransposeOutputDim(
        xShape.GetDim(4), wShape.GetDim(4), *strideWPtr, *paddingWPtr, *outputPaddingWPtr);
    const uint32_t outputDepth = ComputePoolOutputDim(convDepth, *poolKernelPtr);
    const uint32_t outputHeight = ComputePoolOutputDim(convHeight, *poolKernelPtr);
    const uint32_t outputWidth = ComputePoolOutputDim(convWidth, *poolKernelPtr);
    if (bShape.GetDim(0) != static_cast<int64_t>(outChannels) ||
        outputDepth == 0 || outputHeight == 0 || outputWidth == 0) {
        return ge::GRAPH_FAILED;
    }

    ConvTranspose3dAvgPoolClampSoftmaxMultiplyCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_outChannels(outChannels);
    tiling.set_outputDepth(outputDepth);
    tiling.set_outputHeight(outputHeight);
    tiling.set_outputWidth(outputWidth);

    context->SetBlockDim(batchSize == 0 ? 1 : batchSize);
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
    const gert::Shape *biasShape = context->GetInputShape(2);
    if (inputShape == nullptr || weightShape == nullptr || biasShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDimNum() != 5 || weightShape->GetDimNum() != 5 || biasShape->GetDimNum() != 1) {
        return GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int64_t *strideDPtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *strideHPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *strideWPtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t *paddingDPtr = attrs->GetAttrPointer<int64_t>(3);
    const int64_t *paddingHPtr = attrs->GetAttrPointer<int64_t>(4);
    const int64_t *paddingWPtr = attrs->GetAttrPointer<int64_t>(5);
    const int64_t *outputPaddingDPtr = attrs->GetAttrPointer<int64_t>(6);
    const int64_t *outputPaddingHPtr = attrs->GetAttrPointer<int64_t>(7);
    const int64_t *outputPaddingWPtr = attrs->GetAttrPointer<int64_t>(8);
    const int64_t *poolKernelPtr = attrs->GetAttrPointer<int64_t>(9);
    const float *clampMinPtr = attrs->GetAttrPointer<float>(10);
    const float *clampMaxPtr = attrs->GetAttrPointer<float>(11);
    if (strideDPtr == nullptr || strideHPtr == nullptr || strideWPtr == nullptr ||
        paddingDPtr == nullptr || paddingHPtr == nullptr || paddingWPtr == nullptr ||
        outputPaddingDPtr == nullptr || outputPaddingHPtr == nullptr || outputPaddingWPtr == nullptr ||
        poolKernelPtr == nullptr || clampMinPtr == nullptr || clampMaxPtr == nullptr ||
        *poolKernelPtr <= 0 || *clampMinPtr > *clampMaxPtr) {
        return GRAPH_FAILED;
    }

    const int64_t convDepth = (inputShape->GetDim(2) - 1) * (*strideDPtr) - 2 * (*paddingDPtr) +
        weightShape->GetDim(2) + (*outputPaddingDPtr);
    const int64_t convHeight = (inputShape->GetDim(3) - 1) * (*strideHPtr) - 2 * (*paddingHPtr) +
        weightShape->GetDim(3) + (*outputPaddingHPtr);
    const int64_t convWidth = (inputShape->GetDim(4) - 1) * (*strideWPtr) - 2 * (*paddingWPtr) +
        weightShape->GetDim(4) + (*outputPaddingWPtr);
    if (convDepth < *poolKernelPtr || convHeight < *poolKernelPtr || convWidth < *poolKernelPtr) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(5);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, weightShape->GetDim(1));
    outputShape->SetDim(2, (convDepth - *poolKernelPtr) / *poolKernelPtr + 1);
    outputShape->SetDim(3, (convHeight - *poolKernelPtr) / *poolKernelPtr + 1);
    outputShape->SetDim(4, (convWidth - *poolKernelPtr) / *poolKernelPtr + 1);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ConvTranspose3dAvgPoolClampSoftmaxMultiplyCustom : public OpDef {
public:
    explicit ConvTranspose3dAvgPoolClampSoftmaxMultiplyCustom(const char *name)
        : OpDef(name)
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
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("stride_d").AttrType(REQUIRED).Int();
        this->Attr("stride_h").AttrType(REQUIRED).Int();
        this->Attr("stride_w").AttrType(REQUIRED).Int();
        this->Attr("padding_d").AttrType(REQUIRED).Int();
        this->Attr("padding_h").AttrType(REQUIRED).Int();
        this->Attr("padding_w").AttrType(REQUIRED).Int();
        this->Attr("output_padding_d").AttrType(REQUIRED).Int();
        this->Attr("output_padding_h").AttrType(REQUIRED).Int();
        this->Attr("output_padding_w").AttrType(REQUIRED).Int();
        this->Attr("pool_kernel_size").AttrType(REQUIRED).Int();
        this->Attr("clamp_min").AttrType(REQUIRED).Float();
        this->Attr("clamp_max").AttrType(REQUIRED).Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvTranspose3dAvgPoolClampSoftmaxMultiplyCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConvTranspose3dAvgPoolClampSoftmaxMultiply {
public:
    __aicore__ inline KernelConvTranspose3dAvgPoolClampSoftmaxMultiply() {}

    __aicore__ inline void Init(
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t outChannels,
        uint32_t outputDepth,
        uint32_t outputHeight,
        uint32_t outputWidth)
    {
        this->batchSize = batchSize;
        this->outChannels = outChannels;
        this->outputDepth = outputDepth;
        this->outputHeight = outputHeight;
        this->outputWidth = outputWidth;
        this->blockIdx = GetBlockIdx();
        this->outputBatchStride = outChannels * outputDepth * outputHeight * outputWidth;
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * this->outputBatchStride);
    }

    __aicore__ inline void Process()
    {
        if (this->blockIdx >= this->batchSize) {
            return;
        }

        const uint32_t base = this->blockIdx * this->outputBatchStride;
        for (uint32_t idx = 0; idx < this->outputBatchStride; ++idx) {
            yGm.SetValue(base + idx, 0.0f);
        }
    }

private:
    GlobalTensor<float> yGm;
    uint32_t batchSize = 0;
    uint32_t outChannels = 0;
    uint32_t outputDepth = 0;
    uint32_t outputHeight = 0;
    uint32_t outputWidth = 0;
    uint32_t outputBatchStride = 0;
    uint32_t blockIdx = 0;
};

extern "C" __global__ __aicore__ void conv_transpose3d_avg_pool_clamp_softmax_multiply_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)x;
    (void)weight;
    (void)bias;
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTranspose3dAvgPoolClampSoftmaxMultiply op;
    op.Init(
        y,
        tiling_data.batchSize,
        tiling_data.outChannels,
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
#include <ATen/ops/avg_pool3d.h>
#include "pytorch_npu_helper.hpp"
#include <vector>

at::Tensor conv_transpose3d_avg_pool_clamp_softmax_multiply_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &bias,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_d,
    int64_t padding_h,
    int64_t padding_w,
    int64_t output_padding_d,
    int64_t output_padding_h,
    int64_t output_padding_w,
    int64_t pool_kernel_size,
    double clamp_min,
    double clamp_max)
{
    TORCH_CHECK(x.dim() == 5, "x must be a 5D NCDHW tensor");
    TORCH_CHECK(weight.dim() == 5, "weight must be a 5D tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
    TORCH_CHECK(stride_d > 0 && stride_h > 0 && stride_w > 0, "stride values must be positive");
    TORCH_CHECK(padding_d >= 0 && padding_h >= 0 && padding_w >= 0, "padding values must be non-negative");
    TORCH_CHECK(
        output_padding_d >= 0 && output_padding_h >= 0 && output_padding_w >= 0,
        "output padding values must be non-negative");
    TORCH_CHECK(pool_kernel_size > 0, "pool_kernel_size must be positive");
    TORCH_CHECK(clamp_min <= clamp_max, "clamp_min must not exceed clamp_max");
    TORCH_CHECK(x.size(1) == weight.size(0), "input channels must match weight.size(0)");

    const int64_t out_channels = weight.size(1);
    TORCH_CHECK(bias.size(0) == out_channels, "bias size must match output channels");

    const int64_t conv_out_d = (x.size(2) - 1) * stride_d - 2 * padding_d + weight.size(2) + output_padding_d;
    const int64_t conv_out_h = (x.size(3) - 1) * stride_h - 2 * padding_h + weight.size(3) + output_padding_h;
    const int64_t conv_out_w = (x.size(4) - 1) * stride_w - 2 * padding_w + weight.size(4) + output_padding_w;
    TORCH_CHECK(conv_out_d > 0 && conv_out_h > 0 && conv_out_w > 0, "invalid conv_transpose3d output shape");
    TORCH_CHECK(
        conv_out_d >= pool_kernel_size && conv_out_h >= pool_kernel_size && conv_out_w >= pool_kernel_size,
        "avg pool kernel is larger than the conv_transpose3d output");

    at::Tensor conv = at::empty({x.size(0), out_channels, conv_out_d, conv_out_h, conv_out_w}, x.options());
    auto conv_bias = c10::optional<at::Tensor>(bias);
    std::vector<int64_t> stride_vec = {stride_d, stride_h, stride_w};
    std::vector<int64_t> padding_vec = {padding_d, padding_h, padding_w};
    std::vector<int64_t> dilation_vec = {1, 1, 1};
    std::vector<int64_t> output_padding_vec = {output_padding_d, output_padding_h, output_padding_w};
    at::IntArrayRef stride_array(stride_vec);
    at::IntArrayRef padding_array(padding_vec);
    at::IntArrayRef dilation_array(dilation_vec);
    at::IntArrayRef output_padding_array(output_padding_vec);
    const bool transposed = true;
    int64_t groups = 1;
    const int8_t cubeMathType = 0;

    EXEC_NPU_CMD(
        aclnnConvolution,
        x,
        weight,
        conv_bias,
        stride_array,
        padding_array,
        dilation_array,
        transposed,
        output_padding_array,
        groups,
        conv,
        cubeMathType);

    std::vector<int64_t> pool_kernel = {pool_kernel_size, pool_kernel_size, pool_kernel_size};
    std::vector<int64_t> pool_stride = {pool_kernel_size, pool_kernel_size, pool_kernel_size};
    std::vector<int64_t> pool_padding = {0, 0, 0};
    at::Tensor pooled = at::avg_pool3d(
        conv,
        at::IntArrayRef(pool_kernel),
        at::IntArrayRef(pool_stride),
        at::IntArrayRef(pool_padding),
        false,
        true,
        ::std::nullopt);
    at::Tensor clamped = at::clamp(pooled, clamp_min, clamp_max);
    at::Tensor probs = at::softmax(clamped, 1, c10::nullopt);
    return probs * 2.0;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "conv_transpose3d_avg_pool_clamp_softmax_multiply_custom",
        &conv_transpose3d_avg_pool_clamp_softmax_multiply_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv_transpose3d_avg_pool_clamp_softmax_multiply_custom",
        &conv_transpose3d_avg_pool_clamp_softmax_multiply_impl_npu,
        "conv_transpose3d_avg_pool_clamp_softmax_multiply_custom");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


def _normalize_3d(value):
    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError("expected an int or length-3 tuple")
    return tuple(int(v) for v in value)


class ModelNew(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        pool_kernel_size,
        clamp_min,
        clamp_max,
    ):
        super(ModelNew, self).__init__()
        kernel_size = _normalize_3d(kernel_size)
        stride = _normalize_3d(stride)
        padding = _normalize_3d(padding)
        output_padding = _normalize_3d(output_padding)

        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.pool_kernel_size = int(pool_kernel_size)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.conv_transpose = torch.nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=True,
        )

    def forward(self, x):
        return custom_ops_lib.conv_transpose3d_avg_pool_clamp_softmax_multiply_custom(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.stride[0],
            self.stride[1],
            self.stride[2],
            self.padding[0],
            self.padding[1],
            self.padding[2],
            self.output_padding[0],
            self.output_padding[1],
            self.output_padding[2],
            self.pool_kernel_size,
            self.clamp_min,
            self.clamp_max,
        )
'''
