project_json_src='''
[
    {
        "op": "Ct3dMulMaxGapClampCoreCustom",
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
        ],
        "attr": [
            {
                "name": "scale",
                "param_type": "required",
                "type": "float"
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
BEGIN_TILING_DATA_DEF(Ct3dMulMaxGapClampCoreCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, channelCount);
    TILING_DATA_FIELD_DEF(uint32_t, inputDepth);
    TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, poolOutDepth);
    TILING_DATA_FIELD_DEF(uint32_t, poolOutHeight);
    TILING_DATA_FIELD_DEF(uint32_t, poolOutWidth);
    TILING_DATA_FIELD_DEF(uint32_t, poolKernelSize);
    TILING_DATA_FIELD_DEF(float, poolElementReciprocal);
    TILING_DATA_FIELD_DEF(float, scale);
    TILING_DATA_FIELD_DEF(float, clampMin);
    TILING_DATA_FIELD_DEF(float, clampMax);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    Ct3dMulMaxGapClampCoreCustom,
    Ct3dMulMaxGapClampCoreCustomTilingData)
}
"""

host_operator_src="""
#include "ct3d_mul_max_gap_clamp_core_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
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
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (inputShape == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto shape = inputShape->GetStorageShape();
    if (shape.GetDimNum() != 5) {
        return ge::GRAPH_FAILED;
    }

    const float *scalePtr = attrs->GetAttrPointer<float>(0);
    const int64_t *poolKernelPtr = attrs->GetAttrPointer<int64_t>(1);
    const float *clampMinPtr = attrs->GetAttrPointer<float>(2);
    const float *clampMaxPtr = attrs->GetAttrPointer<float>(3);
    if (scalePtr == nullptr || poolKernelPtr == nullptr || clampMinPtr == nullptr || clampMaxPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(shape.GetDim(0));
    const uint32_t channelCount = static_cast<uint32_t>(shape.GetDim(1));
    const uint32_t inputDepth = static_cast<uint32_t>(shape.GetDim(2));
    const uint32_t inputHeight = static_cast<uint32_t>(shape.GetDim(3));
    const uint32_t inputWidth = static_cast<uint32_t>(shape.GetDim(4));
    const uint32_t poolOutDepth = ComputePoolOutputDim(inputDepth, *poolKernelPtr);
    const uint32_t poolOutHeight = ComputePoolOutputDim(inputHeight, *poolKernelPtr);
    const uint32_t poolOutWidth = ComputePoolOutputDim(inputWidth, *poolKernelPtr);

    if (*poolKernelPtr <= 0 || *clampMinPtr > *clampMaxPtr ||
        poolOutDepth == 0 || poolOutHeight == 0 || poolOutWidth == 0) {
        return ge::GRAPH_FAILED;
    }

    Ct3dMulMaxGapClampCoreCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_channelCount(channelCount);
    tiling.set_inputDepth(inputDepth);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_poolOutDepth(poolOutDepth);
    tiling.set_poolOutHeight(poolOutHeight);
    tiling.set_poolOutWidth(poolOutWidth);
    tiling.set_poolKernelSize(static_cast<uint32_t>(*poolKernelPtr));
    tiling.set_poolElementReciprocal(1.0f / static_cast<float>(poolOutDepth * poolOutHeight * poolOutWidth));
    tiling.set_scale(*scalePtr);
    tiling.set_clampMin(*clampMinPtr);
    tiling.set_clampMax(*clampMaxPtr);

    context->SetBlockDim(batchSize > 0 ? batchSize : 1);
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
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (inputShape == nullptr || attrs == nullptr || inputShape->GetDimNum() != 5) {
        return GRAPH_FAILED;
    }

    const int64_t *poolKernelPtr = attrs->GetAttrPointer<int64_t>(1);
    const float *clampMinPtr = attrs->GetAttrPointer<float>(2);
    const float *clampMaxPtr = attrs->GetAttrPointer<float>(3);
    if (poolKernelPtr == nullptr || clampMinPtr == nullptr || clampMaxPtr == nullptr ||
        *poolKernelPtr <= 0 || *clampMinPtr > *clampMaxPtr ||
        inputShape->GetDim(2) < *poolKernelPtr || inputShape->GetDim(3) < *poolKernelPtr ||
        inputShape->GetDim(4) < *poolKernelPtr) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(5);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, inputShape->GetDim(1));
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
class Ct3dMulMaxGapClampCoreCustom : public OpDef {
public:
    explicit Ct3dMulMaxGapClampCoreCustom(const char *name)
        : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("scale").AttrType(REQUIRED).Float();
        this->Attr("pool_kernel_size").AttrType(REQUIRED).Int();
        this->Attr("clamp_min").AttrType(REQUIRED).Float();
        this->Attr("clamp_max").AttrType(REQUIRED).Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Ct3dMulMaxGapClampCoreCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr float kNegInf = -3.40282347e+38f;
}

class KernelConvTranspose3dMultiplyMaxGlobalAvgPoolClampCore {
public:
    __aicore__ inline KernelConvTranspose3dMultiplyMaxGlobalAvgPoolClampCore() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t channelCount,
        uint32_t inputDepth,
        uint32_t inputHeight,
        uint32_t inputWidth,
        uint32_t poolOutDepth,
        uint32_t poolOutHeight,
        uint32_t poolOutWidth,
        uint32_t poolKernelSize,
        float poolElementReciprocal,
        float scale,
        float clampMin,
        float clampMax)
    {
        this->batchSize = batchSize;
        this->channelCount = channelCount;
        this->inputDepth = inputDepth;
        this->inputHeight = inputHeight;
        this->inputWidth = inputWidth;
        this->poolOutDepth = poolOutDepth;
        this->poolOutHeight = poolOutHeight;
        this->poolOutWidth = poolOutWidth;
        this->poolKernelSize = poolKernelSize;
        this->poolElementReciprocal = poolElementReciprocal;
        this->scale = scale;
        this->clampMin = clampMin;
        this->clampMax = clampMax;
        this->blockIdx = GetBlockIdx();

        this->inputPlaneStride = inputHeight * inputWidth;
        this->inputChannelStride = inputDepth * this->inputPlaneStride;
        this->inputBatchStride = channelCount * this->inputChannelStride;
        this->outputBatchStride = channelCount;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * this->inputBatchStride);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * this->outputBatchStride);
    }

    __aicore__ inline void Process()
    {
        if (this->blockIdx >= this->batchSize) {
            return;
        }

        const uint32_t xBatchBase = this->blockIdx * this->inputBatchStride;
        const uint32_t yBatchBase = this->blockIdx * this->outputBatchStride;

        for (uint32_t channel = 0; channel < this->channelCount; ++channel) {
            const uint32_t xChannelBase = xBatchBase + channel * this->inputChannelStride;
            float pooledSum = 0.0f;

            for (uint32_t outD = 0; outD < this->poolOutDepth; ++outD) {
                const uint32_t inBaseD = outD * this->poolKernelSize;
                for (uint32_t outH = 0; outH < this->poolOutHeight; ++outH) {
                    const uint32_t inBaseH = outH * this->poolKernelSize;
                    for (uint32_t outW = 0; outW < this->poolOutWidth; ++outW) {
                        const uint32_t inBaseW = outW * this->poolKernelSize;
                        float maxValue = kNegInf;
                        for (uint32_t kd = 0; kd < this->poolKernelSize; ++kd) {
                            const uint32_t inD = inBaseD + kd;
                            for (uint32_t kh = 0; kh < this->poolKernelSize; ++kh) {
                                const uint32_t inH = inBaseH + kh;
                                for (uint32_t kw = 0; kw < this->poolKernelSize; ++kw) {
                                    const uint32_t inW = inBaseW + kw;
                                    const uint32_t offset =
                                        xChannelBase +
                                        inD * this->inputPlaneStride +
                                        inH * this->inputWidth +
                                        inW;
                                    const float scaledValue = xGm.GetValue(offset) * this->scale;
                                    if (scaledValue > maxValue) {
                                        maxValue = scaledValue;
                                    }
                                }
                            }
                        }
                        pooledSum += maxValue;
                    }
                }
            }

            float avgValue = pooledSum * this->poolElementReciprocal;
            if (avgValue < this->clampMin) {
                avgValue = this->clampMin;
            }
            if (avgValue > this->clampMax) {
                avgValue = this->clampMax;
            }
            yGm.SetValue(yBatchBase + channel, avgValue);
        }
    }

private:
    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;
    uint32_t batchSize = 0;
    uint32_t channelCount = 0;
    uint32_t inputDepth = 0;
    uint32_t inputHeight = 0;
    uint32_t inputWidth = 0;
    uint32_t poolOutDepth = 0;
    uint32_t poolOutHeight = 0;
    uint32_t poolOutWidth = 0;
    uint32_t poolKernelSize = 0;
    float poolElementReciprocal = 1.0f;
    float scale = 1.0f;
    float clampMin = 0.0f;
    float clampMax = 1.0f;
    uint32_t blockIdx = 0;
    uint32_t inputPlaneStride = 0;
    uint32_t inputChannelStride = 0;
    uint32_t inputBatchStride = 0;
    uint32_t outputBatchStride = 0;
};

extern "C" __global__ __aicore__ void ct3d_mul_max_gap_clamp_core_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTranspose3dMultiplyMaxGlobalAvgPoolClampCore op;
    op.Init(
        x,
        y,
        tiling_data.batchSize,
        tiling_data.channelCount,
        tiling_data.inputDepth,
        tiling_data.inputHeight,
        tiling_data.inputWidth,
        tiling_data.poolOutDepth,
        tiling_data.poolOutHeight,
        tiling_data.poolOutWidth,
        tiling_data.poolKernelSize,
        tiling_data.poolElementReciprocal,
        tiling_data.scale,
        tiling_data.clampMin,
        tiling_data.clampMax);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/ops/adaptive_avg_pool3d.h>
#include <ATen/ops/max_pool3d.h>
#include "pytorch_npu_helper.hpp"
#include <vector>

at::Tensor conv_transpose3d_multiply_max_global_avg_pool_clamp_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &bias,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_d,
    int64_t padding_h,
    int64_t padding_w,
    double scale,
    int64_t pool_kernel_size,
    double clamp_min,
    double clamp_max)
{
    TORCH_CHECK(x.dim() == 5, "x must be a 5D NCDHW tensor");
    TORCH_CHECK(weight.dim() == 5, "weight must be a 5D tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
    TORCH_CHECK(x.size(1) == weight.size(0), "input channels must match weight.size(0)");
    TORCH_CHECK(bias.size(0) == weight.size(1), "bias size must match output channels");
    TORCH_CHECK(stride_d > 0 && stride_h > 0 && stride_w > 0, "stride values must be positive");
    TORCH_CHECK(padding_d >= 0 && padding_h >= 0 && padding_w >= 0, "padding values must be non-negative");
    TORCH_CHECK(pool_kernel_size > 0, "pool_kernel_size must be positive");
    TORCH_CHECK(clamp_min <= clamp_max, "clamp_min must not exceed clamp_max");

    const int64_t conv_out_d = (x.size(2) - 1) * stride_d - 2 * padding_d + weight.size(2);
    const int64_t conv_out_h = (x.size(3) - 1) * stride_h - 2 * padding_h + weight.size(3);
    const int64_t conv_out_w = (x.size(4) - 1) * stride_w - 2 * padding_w + weight.size(4);
    TORCH_CHECK(conv_out_d > 0 && conv_out_h > 0 && conv_out_w > 0, "invalid conv_transpose3d output shape");
    TORCH_CHECK(
        conv_out_d >= pool_kernel_size && conv_out_h >= pool_kernel_size && conv_out_w >= pool_kernel_size,
        "pool kernel is larger than the transposed convolution output");

    at::Tensor conv_result = at::empty(
        {x.size(0), weight.size(1), conv_out_d, conv_out_h, conv_out_w},
        x.options());
    auto conv_bias = c10::optional<at::Tensor>(bias);
    std::vector<int64_t> stride_vec = {stride_d, stride_h, stride_w};
    std::vector<int64_t> padding_vec = {padding_d, padding_h, padding_w};
    std::vector<int64_t> dilation_vec = {1, 1, 1};
    std::vector<int64_t> output_padding_vec = {0, 0, 0};
    at::IntArrayRef stride_array(stride_vec);
    at::IntArrayRef padding_array(padding_vec);
    at::IntArrayRef dilation_array(dilation_vec);
    at::IntArrayRef output_padding_array(output_padding_vec);
    const bool transposed = true;
    const int64_t groups = 1;
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
        conv_result,
        cubeMathType);

    float scale_value = static_cast<float>(scale);
    float clamp_min_value = static_cast<float>(clamp_min);
    float clamp_max_value = static_cast<float>(clamp_max);
    std::vector<int64_t> pool_kernel_vec = {pool_kernel_size, pool_kernel_size, pool_kernel_size};
    std::vector<int64_t> pool_stride_vec = {pool_kernel_size, pool_kernel_size, pool_kernel_size};
    std::vector<int64_t> pool_padding_vec = {0, 0, 0};
    std::vector<int64_t> pool_dilation_vec = {1, 1, 1};
    at::Tensor scaled = conv_result * scale_value;
    at::Tensor pooled = at::max_pool3d(
        scaled,
        at::IntArrayRef(pool_kernel_vec),
        at::IntArrayRef(pool_stride_vec),
        at::IntArrayRef(pool_padding_vec),
        at::IntArrayRef(pool_dilation_vec),
        false);
    at::Tensor reduced = at::adaptive_avg_pool3d(pooled, {1, 1, 1});
    return at::clamp(reduced, clamp_min_value, clamp_max_value);
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "conv_transpose3d_multiply_max_global_avg_pool_clamp_custom",
        &conv_transpose3d_multiply_max_global_avg_pool_clamp_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv_transpose3d_multiply_max_global_avg_pool_clamp_custom",
        &conv_transpose3d_multiply_max_global_avg_pool_clamp_impl_npu,
        "conv_transpose3d_multiply_max_global_avg_pool_clamp_custom");
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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        kernel_size = _normalize_3d(kernel_size)
        stride = _normalize_3d(stride)
        padding = _normalize_3d(padding)

        self.stride = stride
        self.padding = padding
        self.scale = float(scale)
        self.maxpool_kernel_size = int(maxpool_kernel_size)
        self.clamp_min = 0.0
        self.clamp_max = 1.0
        self.conv_transpose = torch.nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )

    def forward(self, x):
        return custom_ops_lib.conv_transpose3d_multiply_max_global_avg_pool_clamp_custom(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.stride[0],
            self.stride[1],
            self.stride[2],
            self.padding[0],
            self.padding[1],
            self.padding[2],
            self.scale,
            self.maxpool_kernel_size,
            self.clamp_min,
            self.clamp_max,
        )
'''
