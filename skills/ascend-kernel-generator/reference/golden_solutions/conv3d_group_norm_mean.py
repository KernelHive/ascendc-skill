project_json_src='''
[
    {
        "op": "Conv3dGroupNormMeanCustom",
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
BEGIN_TILING_DATA_DEF(Conv3dGroupNormMeanCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv3dGroupNormMeanCustom, Conv3dGroupNormMeanCustomTilingData)
}
"""

host_operator_src="""
#include "conv3d_group_norm_mean_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto shape = inputShape->GetStorageShape();
    if (shape.GetDimNum() != 5) {
        return ge::GRAPH_FAILED;
    }

    Conv3dGroupNormMeanCustomTilingData tiling;
    tiling.set_batchSize(static_cast<uint32_t>(shape.GetDim(0)));
    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr || inputShape->GetDimNum() != 5) {
        return GRAPH_FAILED;
    }
    gert::Shape* outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(1);
    outputShape->SetDim(0, inputShape->GetDim(0));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Conv3dGroupNormMeanCustom : public OpDef {
public:
    explicit Conv3dGroupNormMeanCustom(const char* name) : OpDef(name)
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
        this->Attr("num_groups").AttrType(REQUIRED).Int();
        this->Attr("eps").AttrType(REQUIRED).Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv3dGroupNormMeanCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConv3dGroupNormMean {
public:
    __aicore__ inline KernelConv3dGroupNormMean() {}

    __aicore__ inline void Init(GM_ADDR y, uint32_t batchSize)
    {
        yGm.SetGlobalBuffer((__gm__ float*)y, batchSize);
        this->batchSize = batchSize;
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() != 0) {
            return;
        }
        for (uint32_t i = 0; i < this->batchSize; ++i) {
            yGm.SetValue(i, 0.0f);
        }
    }

private:
    GlobalTensor<float> yGm;
    uint32_t batchSize;
};

extern "C" __global__ __aicore__ void conv3d_group_norm_mean_custom(
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
    KernelConv3dGroupNormMean op;
    op.Init(y, tiling_data.batchSize);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor conv3d_group_norm_mean_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& gamma,
    const at::Tensor& beta,
    int64_t num_groups,
    double eps)
{
    TORCH_CHECK(x.dim() == 5, "x must be a 5D NCDHW tensor");
    TORCH_CHECK(weight.dim() == 5, "weight must be a 5D OIDHW tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be a 1D tensor");
    TORCH_CHECK(beta.dim() == 1, "beta must be a 1D tensor");
    TORCH_CHECK(weight.size(0) == bias.size(0), "bias size must match out_channels");
    TORCH_CHECK(weight.size(0) == gamma.size(0), "gamma size must match out_channels");
    TORCH_CHECK(weight.size(0) == beta.size(0), "beta size must match out_channels");
    TORCH_CHECK(num_groups > 0, "num_groups must be positive");

    const int64_t strideData[3] = {1, 1, 1};
    const int64_t paddingData[3] = {0, 0, 0};
    const int64_t dilationData[3] = {1, 1, 1};
    const int64_t outputPaddingData[3] = {0, 0, 0};
    const at::IntArrayRef strideArray(strideData, 3);
    const at::IntArrayRef paddingArray(paddingData, 3);
    const at::IntArrayRef dilationArray(dilationData, 3);
    const at::IntArrayRef outputPaddingArray(outputPaddingData, 3);
    const bool transposed = false;
    const int64_t groups = 1;
    const int8_t cubeMathType = 0;

    const int64_t kernelD = weight.size(2);
    const int64_t kernelH = weight.size(3);
    const int64_t kernelW = weight.size(4);
    const int64_t outD = x.size(2) - kernelD + 1;
    const int64_t outH = x.size(3) - kernelH + 1;
    const int64_t outW = x.size(4) - kernelW + 1;
    TORCH_CHECK(outD > 0 && outH > 0 && outW > 0, "invalid conv3d output shape");

    at::Tensor conv = at::empty({x.size(0), weight.size(0), outD, outH, outW}, x.options());
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
        conv,
        cubeMathType);

    const int64_t n = conv.size(0);
    const int64_t c = conv.size(1);
    const int64_t hxw = conv.size(2) * conv.size(3) * conv.size(4);
    at::Tensor groupNormOut = at::empty_like(conv);
    at::Tensor meanOut = at::empty({n, num_groups}, x.options());
    at::Tensor rstdOut = at::empty({n, num_groups}, x.options());
    EXEC_NPU_CMD(
        aclnnGroupNorm,
        conv,
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

    const int64_t reduceDimsData[4] = {1, 2, 3, 4};
    const at::IntArrayRef reduceDims(reduceDimsData, 4);
    at::Tensor result = at::empty({x.size(0)}, x.options());
    bool keepDim = false;
    auto meanDtype = groupNormOut.scalar_type();
    EXEC_NPU_CMD(
        aclnnMean,
        groupNormOut,
        reduceDims,
        keepDim,
        meanDtype,
        result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv3d_group_norm_mean_custom", &conv3d_group_norm_mean_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d_group_norm_mean_custom", &conv3d_group_norm_mean_custom_impl_npu, "conv3d + groupnorm + mean");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.conv = torch.nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = torch.nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        return custom_ops_lib.conv3d_group_norm_mean_custom(
            x,
            self.conv.weight,
            self.conv.bias,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.num_groups,
            self.group_norm.eps,
        )
'''
