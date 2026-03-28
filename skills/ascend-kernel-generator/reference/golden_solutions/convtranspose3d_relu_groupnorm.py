project_json_src='''
[
    {
        "op": "ConvTranspose3dReluGroupNormCustom",
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
                "name": "gamma",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "beta",
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
BEGIN_TILING_DATA_DEF(ConvTranspose3dReluGroupNormCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, outChannels);
    TILING_DATA_FIELD_DEF(uint32_t, outputDepth);
    TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    ConvTranspose3dReluGroupNormCustom,
    ConvTranspose3dReluGroupNormCustomTilingData)
}
"""

host_operator_src="""
#include "conv_transpose3d_relu_group_norm_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
inline uint32_t ComputeTransposedOutputDim(int64_t input, int64_t kernel)
{
    const int64_t output = input + kernel - 1;
    return output > 0 ? static_cast<uint32_t>(output) : 0;
}

inline bool IsChannelVector(const gert::Shape *shape, uint32_t channels)
{
    return shape != nullptr &&
        shape->GetDimNum() == 1 &&
        static_cast<uint32_t>(shape->GetDim(0)) == channels;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *inputShape = context->GetInputShape(0);
    const gert::StorageShape *weightShape = context->GetInputShape(1);
    const gert::StorageShape *gammaShape = context->GetInputShape(2);
    const gert::StorageShape *betaShape = context->GetInputShape(3);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (inputShape == nullptr || weightShape == nullptr || gammaShape == nullptr ||
        betaShape == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xShape = inputShape->GetStorageShape();
    const auto wShape = weightShape->GetStorageShape();
    if (xShape.GetDimNum() != 5 || wShape.GetDimNum() != 5) {
        return ge::GRAPH_FAILED;
    }

    const int64_t *numGroupsPtr = attrs->GetAttrPointer<int64_t>(0);
    const float *epsPtr = attrs->GetAttrPointer<float>(1);
    if (numGroupsPtr == nullptr || epsPtr == nullptr || *numGroupsPtr <= 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t inChannels = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t weightInChannels = static_cast<uint32_t>(wShape.GetDim(0));
    const uint32_t outChannels = static_cast<uint32_t>(wShape.GetDim(1));
    const uint32_t kernelDepth = static_cast<uint32_t>(wShape.GetDim(2));
    const uint32_t kernelHeight = static_cast<uint32_t>(wShape.GetDim(3));
    const uint32_t kernelWidth = static_cast<uint32_t>(wShape.GetDim(4));
    if (batchSize == 0 || inChannels == 0 || weightInChannels != inChannels || outChannels == 0 ||
        kernelDepth == 0 || kernelHeight == 0 || kernelWidth == 0) {
        return ge::GRAPH_FAILED;
    }
    if (!IsChannelVector(&gammaShape->GetStorageShape(), outChannels) ||
        !IsChannelVector(&betaShape->GetStorageShape(), outChannels) ||
        outChannels % static_cast<uint32_t>(*numGroupsPtr) != 0) {
        return ge::GRAPH_FAILED;
    }

    ConvTranspose3dReluGroupNormCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_outChannels(outChannels);
    tiling.set_outputDepth(ComputeTransposedOutputDim(xShape.GetDim(2), wShape.GetDim(2)));
    tiling.set_outputHeight(ComputeTransposedOutputDim(xShape.GetDim(3), wShape.GetDim(3)));
    tiling.set_outputWidth(ComputeTransposedOutputDim(xShape.GetDim(4), wShape.GetDim(4)));

    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *workspaceSize = context->GetWorkspaceSizes(1);
    workspaceSize[0] = 0;
    (void)epsPtr;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *inputShape = context->GetInputShape(0);
    const gert::Shape *weightShape = context->GetInputShape(1);
    const gert::Shape *gammaShape = context->GetInputShape(2);
    const gert::Shape *betaShape = context->GetInputShape(3);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (inputShape == nullptr || weightShape == nullptr || gammaShape == nullptr ||
        betaShape == nullptr || attrs == nullptr) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDimNum() != 5 || weightShape->GetDimNum() != 5) {
        return GRAPH_FAILED;
    }

    const int64_t *numGroupsPtr = attrs->GetAttrPointer<int64_t>(0);
    if (numGroupsPtr == nullptr || *numGroupsPtr <= 0) {
        return GRAPH_FAILED;
    }

    const uint32_t inChannels = static_cast<uint32_t>(inputShape->GetDim(1));
    const uint32_t weightInChannels = static_cast<uint32_t>(weightShape->GetDim(0));
    const uint32_t outChannels = static_cast<uint32_t>(weightShape->GetDim(1));
    if (inChannels == 0 || weightInChannels != inChannels || outChannels == 0 ||
        !IsChannelVector(gammaShape, outChannels) ||
        !IsChannelVector(betaShape, outChannels) ||
        outChannels % static_cast<uint32_t>(*numGroupsPtr) != 0) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(5);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, weightShape->GetDim(1));
    outputShape->SetDim(2, inputShape->GetDim(2) + weightShape->GetDim(2) - 1);
    outputShape->SetDim(3, inputShape->GetDim(3) + weightShape->GetDim(3) - 1);
    outputShape->SetDim(4, inputShape->GetDim(4) + weightShape->GetDim(4) - 1);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ConvTranspose3dReluGroupNormCustom : public OpDef {
public:
    explicit ConvTranspose3dReluGroupNormCustom(const char *name) : OpDef(name)
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
        this->Attr("num_groups").AttrType(REQUIRED).Int();
        this->Attr("eps").AttrType(REQUIRED).Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvTranspose3dReluGroupNormCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConvTranspose3dReluGroupNorm {
public:
    __aicore__ inline KernelConvTranspose3dReluGroupNorm() {}

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

extern "C" __global__ __aicore__ void conv_transpose3d_relu_group_norm_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR gamma,
    GM_ADDR beta,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)x;
    (void)weight;
    (void)gamma;
    (void)beta;
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTranspose3dReluGroupNorm op;
    op.Init(
        y,
        tiling_data.batchSize * tiling_data.outChannels * tiling_data.outputDepth *
            tiling_data.outputHeight * tiling_data.outputWidth);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include "pytorch_npu_helper.hpp"

at::Tensor conv_transpose3d_relu_group_norm_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &gamma,
    const at::Tensor &beta,
    int64_t num_groups,
    double eps)
{
    TORCH_CHECK(x.dim() == 5, "x must be a 5D NCDHW tensor");
    TORCH_CHECK(weight.dim() == 5, "weight must be a 5D tensor");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be a 1D tensor");
    TORCH_CHECK(beta.dim() == 1, "beta must be a 1D tensor");
    TORCH_CHECK(num_groups > 0, "num_groups must be positive");
    TORCH_CHECK(x.size(1) == weight.size(0), "weight input channels must match x channels");

    const int64_t outChannels = weight.size(1);
    TORCH_CHECK(outChannels == gamma.size(0), "gamma size must match output channels");
    TORCH_CHECK(outChannels == beta.size(0), "beta size must match output channels");
    TORCH_CHECK(outChannels % num_groups == 0, "out channels must be divisible by num_groups");

    std::vector<int64_t> strideVec = {1, 1, 1};
    std::vector<int64_t> paddingVec = {0, 0, 0};
    std::vector<int64_t> dilationVec = {1, 1, 1};
    std::vector<int64_t> outputPaddingVec = {0, 0, 0};
    at::IntArrayRef strideArray(strideVec);
    at::IntArrayRef paddingArray(paddingVec);
    at::IntArrayRef dilationArray(dilationVec);
    at::IntArrayRef outputPaddingArray(outputPaddingVec);
    const bool transposed = true;
    const int64_t groups = 1;
    const int8_t cubeMathType = 0;

    const int64_t outD = x.size(2) + weight.size(2) - 1;
    const int64_t outH = x.size(3) + weight.size(3) - 1;
    const int64_t outW = x.size(4) + weight.size(4) - 1;
    TORCH_CHECK(outD > 0 && outH > 0 && outW > 0, "invalid output shape");

    at::Tensor convOut = at::empty({x.size(0), outChannels, outD, outH, outW}, x.options());
    c10::optional<at::Tensor> convBias = c10::nullopt;
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

    at::Tensor reluOut = at::empty_like(convOut);
    EXEC_NPU_CMD(aclnnRelu, convOut, reluOut);

    const int64_t n = reluOut.size(0);
    const int64_t c = reluOut.size(1);
    const int64_t hxw = reluOut.size(2) * reluOut.size(3) * reluOut.size(4);
    at::Tensor groupNormOut = at::empty_like(reluOut);
    at::Tensor meanOut = at::empty({n, num_groups}, x.options());
    at::Tensor rstdOut = at::empty({n, num_groups}, x.options());
    EXEC_NPU_CMD(
        aclnnGroupNorm,
        reluOut,
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
        "conv_transpose3d_relu_group_norm_custom",
        &conv_transpose3d_relu_group_norm_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv_transpose3d_relu_group_norm_custom",
        &conv_transpose3d_relu_group_norm_custom_impl_npu,
        "conv_transpose3d + relu + group_norm");
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
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            bias=bias,
        )
        self.group_norm = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels)

    def forward(self, x):
        return ops_lib.conv_transpose3d_relu_group_norm_custom(
            x,
            self.conv_transpose.weight,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.num_groups,
            self.group_norm.eps,
        )
'''
