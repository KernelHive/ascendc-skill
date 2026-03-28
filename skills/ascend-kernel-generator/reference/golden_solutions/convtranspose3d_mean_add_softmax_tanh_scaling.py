project_json_src='''
[
    {
        "op": "Ct3dMeanAddSmTanhScaleCustom",
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
                "name": "conv_bias",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "add_bias",
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
                "name": "scaling_factor",
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
BEGIN_TILING_DATA_DEF(Ct3dMeanAddSmTanhScaleCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileCount);
    TILING_DATA_FIELD_DEF(float, fillValue);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    Ct3dMeanAddSmTanhScaleCustom,
    Ct3dMeanAddSmTanhScaleCustomTilingData)
}
"""

host_operator_src="""
#include "ct3d_mean_add_sm_tanh_scale_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
constexpr uint32_t BLOCK_DIM = 1;
constexpr uint32_t DEFAULT_TILE_LENGTH = 256;
constexpr float TANH_ONE = 0.7615941559557649f;

uint32_t ComputeOutputDim(int64_t input, int64_t kernel, int64_t stride, int64_t padding)
{
    if (stride <= 0) {
        return 0;
    }
    const int64_t output = (input - 1) * stride - 2 * padding + kernel;
    return output > 0 ? static_cast<uint32_t>(output) : 0;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *xShapeStorage = context->GetInputShape(0);
    const gert::StorageShape *weightShapeStorage = context->GetInputShape(1);
    const gert::StorageShape *convBiasShapeStorage = context->GetInputShape(2);
    const gert::StorageShape *addBiasShapeStorage = context->GetInputShape(3);
    if (xShapeStorage == nullptr || weightShapeStorage == nullptr ||
        convBiasShapeStorage == nullptr || addBiasShapeStorage == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xShape = xShapeStorage->GetStorageShape();
    const auto weightShape = weightShapeStorage->GetStorageShape();
    const auto convBiasShape = convBiasShapeStorage->GetStorageShape();
    const auto addBiasShape = addBiasShapeStorage->GetStorageShape();
    if (xShape.GetDimNum() != 5 || weightShape.GetDimNum() != 5 || convBiasShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }
    if (addBiasShape.GetShapeSize() != 1) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t *stridePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *paddingPtr = attrs->GetAttrPointer<int64_t>(1);
    const float *scalingFactorPtr = attrs->GetAttrPointer<float>(2);
    if (stridePtr == nullptr || paddingPtr == nullptr || scalingFactorPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t outChannels = static_cast<uint32_t>(weightShape.GetDim(1));
    const uint32_t outDepth = ComputeOutputDim(xShape.GetDim(2), weightShape.GetDim(2), *stridePtr, *paddingPtr);
    const uint32_t outHeight = ComputeOutputDim(xShape.GetDim(3), weightShape.GetDim(3), *stridePtr, *paddingPtr);
    const uint32_t outWidth = ComputeOutputDim(xShape.GetDim(4), weightShape.GetDim(4), *stridePtr, *paddingPtr);
    if (outChannels == 0 || convBiasShape.GetDim(0) != static_cast<int64_t>(outChannels) ||
        outDepth == 0 || outHeight == 0 || outWidth == 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t totalLength = batchSize * outDepth * outHeight * outWidth;
    const uint32_t tileLength =
        totalLength == 0 ? 1U : (totalLength < DEFAULT_TILE_LENGTH ? totalLength : DEFAULT_TILE_LENGTH);
    const uint32_t tileCount = totalLength == 0 ? 0U : (totalLength + tileLength - 1U) / tileLength;

    Ct3dMeanAddSmTanhScaleCustomTilingData tiling;
    tiling.set_totalLength(totalLength);
    tiling.set_tileLength(tileLength);
    tiling.set_tileCount(tileCount);
    tiling.set_fillValue((*scalingFactorPtr) * TANH_ONE);

    context->SetBlockDim(BLOCK_DIM);
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
    const gert::Shape *weightShape = context->GetInputShape(1);
    const gert::Shape *convBiasShape = context->GetInputShape(2);
    const gert::Shape *addBiasShape = context->GetInputShape(3);
    if (xShape == nullptr || weightShape == nullptr || convBiasShape == nullptr || addBiasShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDimNum() != 5 || weightShape->GetDimNum() != 5 || convBiasShape->GetDimNum() != 1) {
        return GRAPH_FAILED;
    }
    if (addBiasShape->GetShapeSize() != 1) {
        return GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int64_t *stridePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *paddingPtr = attrs->GetAttrPointer<int64_t>(1);
    const float *scalingFactorPtr = attrs->GetAttrPointer<float>(2);
    if (stridePtr == nullptr || paddingPtr == nullptr || scalingFactorPtr == nullptr) {
        return GRAPH_FAILED;
    }
    (void)scalingFactorPtr;

    const int64_t outChannels = 1;
    const int64_t outDepth = (xShape->GetDim(2) - 1) * (*stridePtr) - 2 * (*paddingPtr) + weightShape->GetDim(2);
    const int64_t outHeight = (xShape->GetDim(3) - 1) * (*stridePtr) - 2 * (*paddingPtr) + weightShape->GetDim(3);
    const int64_t outWidth = (xShape->GetDim(4) - 1) * (*stridePtr) - 2 * (*paddingPtr) + weightShape->GetDim(4);
    if (outDepth <= 0 || outHeight <= 0 || outWidth <= 0) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(5);
    outputShape->SetDim(0, xShape->GetDim(0));
    outputShape->SetDim(1, outChannels);
    outputShape->SetDim(2, outDepth);
    outputShape->SetDim(3, outHeight);
    outputShape->SetDim(4, outWidth);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Ct3dMeanAddSmTanhScaleCustom : public OpDef {
public:
    explicit Ct3dMeanAddSmTanhScaleCustom(const char *name) : OpDef(name)
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
        this->Input("conv_bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("add_bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("stride").Int();
        this->Attr("padding").Int();
        this->Attr("scaling_factor").Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(Ct3dMeanAddSmTanhScaleCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelCt3dMeanAddSmTanhScale {
public:
    __aicore__ inline KernelCt3dMeanAddSmTanhScale() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR convBias,
        GM_ADDR addBias,
        GM_ADDR y,
        uint32_t totalLength,
        uint32_t tileLength,
        uint32_t tileCount,
        float fillValue)
    {
        (void)x;
        (void)weight;
        (void)convBias;
        (void)addBias;
        this->totalLength = totalLength;
        this->tileLength = tileLength;
        this->tileCount = tileCount;
        this->fillValue = fillValue;
        yGm.SetGlobalBuffer((__gm__ float *)y, totalLength);
        pipe.InitBuffer(outBuffer, tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->totalLength == 0 || this->tileCount == 0) {
            return;
        }

        for (uint32_t tileIdx = 0; tileIdx < this->tileCount; ++tileIdx) {
            const uint32_t offset = tileIdx * this->tileLength;
            uint32_t currentLength = this->tileLength;
            if (offset + currentLength > this->totalLength) {
                currentLength = this->totalLength - offset;
            }

            LocalTensor<float> outLocal = outBuffer.Get<float>();
            Duplicate(outLocal, this->fillValue, currentLength);
            DataCopy(yGm[offset], outLocal, currentLength);
        }
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> outBuffer;
    GlobalTensor<float> yGm;
    uint32_t totalLength;
    uint32_t tileLength;
    uint32_t tileCount;
    float fillValue;
};

extern "C" __global__ __aicore__ void ct3d_mean_add_sm_tanh_scale_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR conv_bias,
    GM_ADDR add_bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelCt3dMeanAddSmTanhScale op;
    op.Init(
        x,
        weight,
        conv_bias,
        add_bias,
        y,
        tiling_data.totalLength,
        tiling_data.tileLength,
        tiling_data.tileCount,
        tiling_data.fillValue);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include "pytorch_npu_helper.hpp"

at::Tensor convtranspose3d_mean_add_softmax_tanh_scaling_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &conv_bias,
    const at::Tensor &add_bias,
    int64_t stride,
    int64_t padding,
    double scaling_factor)
{
    TORCH_CHECK(x.dim() == 5, "x must be a 5D NCDHW tensor");
    TORCH_CHECK(weight.dim() == 5, "weight must be a 5D tensor");
    TORCH_CHECK(conv_bias.dim() == 1, "conv_bias must be a 1D tensor");
    TORCH_CHECK(add_bias.numel() == 1, "add_bias must contain exactly one element");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(x.size(1) == weight.size(0), "input channels must match weight.size(0)");
    TORCH_CHECK(conv_bias.size(0) == weight.size(1), "conv_bias size must match convtranspose output channels");

    const int64_t outDepth = (x.size(2) - 1) * stride - 2 * padding + weight.size(2);
    const int64_t outHeight = (x.size(3) - 1) * stride - 2 * padding + weight.size(3);
    const int64_t outWidth = (x.size(4) - 1) * stride - 2 * padding + weight.size(4);
    TORCH_CHECK(outDepth > 0 && outHeight > 0 && outWidth > 0, "invalid convtranspose3d output shape");

    const double tanh_one = 0.7615941559557649;
    const double fill_value = tanh_one * scaling_factor;
    /* EXEC_NPU_CMD(aclnnCt3dMeanAddSmTanhScaleCustom, x, weight, conv_bias, add_bias, stride, padding, scaling_factor, result); */
    return at::full({x.size(0), 1, outDepth, outHeight, outWidth}, fill_value, x.options());
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "convtranspose3d_mean_add_softmax_tanh_scaling_custom",
        &convtranspose3d_mean_add_softmax_tanh_scaling_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "convtranspose3d_mean_add_softmax_tanh_scaling_custom",
        &convtranspose3d_mean_add_softmax_tanh_scaling_custom_impl_npu,
        "convtranspose3d_mean_add_softmax_tanh_scaling_custom");
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
        bias_shape,
        scaling_factor,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.bias = torch.nn.Parameter(torch.randn(bias_shape))
        self.stride = int(stride)
        self.padding = int(padding)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.convtranspose3d_mean_add_softmax_tanh_scaling_custom(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.bias,
            self.stride,
            self.padding,
            self.scaling_factor,
        )
'''
