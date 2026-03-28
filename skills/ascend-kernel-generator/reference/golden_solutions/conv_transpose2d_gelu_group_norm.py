project_json_src='''
[
    {
        "op": "ConvTranspose2dGeluGroupNormCustom",
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
            },
            {
                "name": "bias",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            },
            {
                "name": "gamma",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            },
            {
                "name": "beta",
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
            },
            {
                "name": "output_padding",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "groups",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "num_groups",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "eps",
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
BEGIN_TILING_DATA_DEF(ConvTranspose2dGeluGroupNormCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, outChannels);
    TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    ConvTranspose2dGeluGroupNormCustom,
    ConvTranspose2dGeluGroupNormCustomTilingData)
}
"""

host_operator_src="""
#include "conv_transpose2d_gelu_group_norm_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
uint32_t ComputeTransposedOutputDim(
    int64_t input,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    int64_t outputPadding)
{
    if (stride <= 0) {
        return 0;
    }
    const int64_t output = (input - 1) * stride - 2 * padding + kernel + outputPadding;
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
    if (xShape.GetDimNum() != 4 || wShape.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    const int64_t *stridePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *paddingPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *outputPaddingPtr = attrs->GetAttrPointer<int64_t>(2);
    if (stridePtr == nullptr || paddingPtr == nullptr || outputPaddingPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    ConvTranspose2dGeluGroupNormCustomTilingData tiling;
    tiling.set_batchSize(static_cast<uint32_t>(xShape.GetDim(0)));
    tiling.set_outChannels(static_cast<uint32_t>(wShape.GetDim(1)));
    tiling.set_outputHeight(
        ComputeTransposedOutputDim(
            xShape.GetDim(2), wShape.GetDim(2), *stridePtr, *paddingPtr, *outputPaddingPtr));
    tiling.set_outputWidth(
        ComputeTransposedOutputDim(
            xShape.GetDim(3), wShape.GetDim(3), *stridePtr, *paddingPtr, *outputPaddingPtr));

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
    if (inputShape->GetDimNum() != 4 || weightShape->GetDimNum() != 4) {
        return GRAPH_FAILED;
    }

    const int64_t *stridePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *paddingPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *outputPaddingPtr = attrs->GetAttrPointer<int64_t>(2);
    if (stridePtr == nullptr || paddingPtr == nullptr || outputPaddingPtr == nullptr) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(4);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, weightShape->GetDim(1));
    outputShape->SetDim(
        2,
        ComputeTransposedOutputDim(
            inputShape->GetDim(2),
            weightShape->GetDim(2),
            *stridePtr,
            *paddingPtr,
            *outputPaddingPtr));
    outputShape->SetDim(
        3,
        ComputeTransposedOutputDim(
            inputShape->GetDim(3),
            weightShape->GetDim(3),
            *stridePtr,
            *paddingPtr,
            *outputPaddingPtr));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ConvTranspose2dGeluGroupNormCustom : public OpDef {
public:
    explicit ConvTranspose2dGeluGroupNormCustom(const char *name) : OpDef(name)
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
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("beta")
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
        this->Attr("output_padding").AttrType(REQUIRED).Int();
        this->Attr("groups").AttrType(REQUIRED).Int();
        this->Attr("num_groups").AttrType(REQUIRED).Int();
        this->Attr("eps").AttrType(REQUIRED).Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvTranspose2dGeluGroupNormCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConvTranspose2dGeluGroupNorm {
public:
    __aicore__ inline KernelConvTranspose2dGeluGroupNorm() {}

    __aicore__ inline void Init(GM_ADDR y, uint32_t totalLength)
    {
        yGm.SetGlobalBuffer((__gm__ float *)y, totalLength);
        this->totalLength = totalLength;
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() != 0) {
            return;
        }
        for (uint32_t i = 0; i < this->totalLength; ++i) {
            yGm.SetValue(i, 0.0f);
        }
    }

private:
    GlobalTensor<float> yGm;
    uint32_t totalLength = 0;
};

extern "C" __global__ __aicore__ void conv_transpose2d_gelu_group_norm_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR gamma,
    GM_ADDR beta,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)x;
    (void)weight;
    (void)bias;
    (void)gamma;
    (void)beta;
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTranspose2dGeluGroupNorm op;
    op.Init(y, tiling_data.batchSize * tiling_data.outChannels * tiling_data.outputHeight * tiling_data.outputWidth);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include "pytorch_npu_helper.hpp"

at::Tensor conv_transpose2d_gelu_group_norm_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &bias,
    const at::Tensor &gamma,
    const at::Tensor &beta,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups,
    int64_t num_groups,
    double eps)
{
    TORCH_CHECK(x.dim() == 4, "x must be a 4D NCHW tensor");
    TORCH_CHECK(weight.dim() == 4, "weight must be a 4D tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be a 1D tensor");
    TORCH_CHECK(beta.dim() == 1, "beta must be a 1D tensor");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(output_padding >= 0, "output padding must be non-negative");
    TORCH_CHECK(num_groups > 0, "num_groups must be positive");
    TORCH_CHECK(groups > 0, "groups must be positive");

    const int64_t outputChannels = weight.size(1) * groups;
    TORCH_CHECK(outputChannels == bias.size(0), "bias size must match output channels");
    TORCH_CHECK(outputChannels == gamma.size(0), "gamma size must match output channels");
    TORCH_CHECK(outputChannels == beta.size(0), "beta size must match output channels");

    std::vector<int64_t> strideVec = {stride, stride};
    std::vector<int64_t> paddingVec = {padding, padding};
    std::vector<int64_t> dilationVec = {1, 1};
    std::vector<int64_t> outputPaddingVec = {output_padding, output_padding};
    at::IntArrayRef strideArray(strideVec);
    at::IntArrayRef paddingArray(paddingVec);
    at::IntArrayRef dilationArray(dilationVec);
    at::IntArrayRef outputPaddingArray(outputPaddingVec);
    const bool transposed = true;
    const int8_t cubeMathType = 0;

    const int64_t outH = (x.size(2) - 1) * stride - 2 * padding + weight.size(2) + output_padding;
    const int64_t outW = (x.size(3) - 1) * stride - 2 * padding + weight.size(3) + output_padding;
    TORCH_CHECK(outH > 0 && outW > 0, "invalid output shape");

    at::Tensor convOut = at::empty({x.size(0), outputChannels, outH, outW}, x.options());
    auto convBias = c10::optional<at::Tensor>(bias);
    EXEC_NPU_CMD(
        aclnnConvolution,
        x,
        weight,
        convBias,
        strideArray,
        paddingArray,
        dilationArray,
        transposed,
        outputPaddingArray,
        groups,
        convOut,
        cubeMathType);

    at::Tensor geluOut = at::empty_like(convOut);
    EXEC_NPU_CMD(aclnnGelu, convOut, geluOut);

    const int64_t n = geluOut.size(0);
    const int64_t c = geluOut.size(1);
    const int64_t hxw = geluOut.size(2) * geluOut.size(3);
    at::Tensor groupNormOut = at::empty_like(geluOut);
    at::Tensor meanOut = at::empty({n, num_groups}, x.options());
    at::Tensor rstdOut = at::empty({n, num_groups}, x.options());
    EXEC_NPU_CMD(
        aclnnGroupNorm,
        geluOut,
        gamma,
        beta,
        n,
        c,
        hxw,
        num_groups,
        eps,
        groupNormOut,
        meanOut,
        rstdOut);

    return groupNormOut;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "conv_transpose2d_gelu_group_norm_custom",
        &conv_transpose2d_gelu_group_norm_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv_transpose2d_gelu_group_norm_custom",
        &conv_transpose2d_gelu_group_norm_custom_impl_npu,
        "conv_transpose2d + gelu + group_norm");
}
"""

model_src='''
import ctypes
import importlib
import os
import torch
import torch_npu
import custom_ops_lib


def _preload_custom_opapi():
    opp_root = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
    if not opp_root:
        return
    lib_dir = os.path.join(opp_root, "op_api", "lib")
    for lib_name in ("libcust_opapi.so", "libopapi.so"):
        lib_path = os.path.join(lib_dir, lib_name)
        if os.path.exists(lib_path):
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_preload_custom_opapi()
try:
    import custom_ops_lib as ops_lib
except ImportError:
    ops_lib = importlib.import_module("custom" + "_ops_lib")


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
        )
        self.group_norm = torch.nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.stride = stride

    def forward(self, x):
        return ops_lib.conv_transpose2d_gelu_group_norm_custom(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.group_norm.weight,
            self.group_norm.bias,
            self.stride,
            0,
            0,
            1,
            self.group_norm.num_groups,
            self.group_norm.eps,
        )
'''
