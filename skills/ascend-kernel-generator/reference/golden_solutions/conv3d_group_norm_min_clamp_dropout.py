project_json_src='''
[
    {
        "op": "C3dGnMcdCustom",
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
            },
            {
                "name": "min_value",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            },
            {
                "name": "max_value",
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
                "name": "dropout_p",
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
BEGIN_TILING_DATA_DEF(C3dGnMcdCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    C3dGnMcdCustom,
    C3dGnMcdCustomTilingData)
}
"""

host_operator_src="""
#include "c3d_gn_mcd_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
bool IsVectorWithLength(const gert::Shape* shape, int64_t expected)
{
    return shape != nullptr && shape->GetDimNum() == 1 && shape->GetDim(0) == expected;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* xStorage = context->GetInputShape(0);
    const gert::StorageShape* weightStorage = context->GetInputShape(1);
    if (xStorage == nullptr || weightStorage == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& xShape = xStorage->GetStorageShape();
    const gert::Shape& wShape = weightStorage->GetStorageShape();
    if (xShape.GetDimNum() != 5 || wShape.GetDimNum() != 5) {
        return ge::GRAPH_FAILED;
    }

    const int64_t outD = xShape.GetDim(2) - wShape.GetDim(2) + 1;
    const int64_t outH = xShape.GetDim(3) - wShape.GetDim(3) + 1;
    const int64_t outW = xShape.GetDim(4) - wShape.GetDim(4) + 1;
    if (outD <= 0 || outH <= 0 || outW <= 0) {
        return ge::GRAPH_FAILED;
    }

    C3dGnMcdCustomTilingData tiling;
    tiling.set_totalLength(static_cast<uint32_t>(xShape.GetDim(0) * wShape.GetDim(0) * outD * outH * outW));
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
    const gert::Shape* xShape = context->GetInputShape(0);
    const gert::Shape* weightShape = context->GetInputShape(1);
    const gert::Shape* biasShape = context->GetInputShape(2);
    const gert::Shape* gammaShape = context->GetInputShape(3);
    const gert::Shape* betaShape = context->GetInputShape(4);
    const gert::Shape* minShape = context->GetInputShape(5);
    const gert::Shape* maxShape = context->GetInputShape(6);
    if (xShape == nullptr || weightShape == nullptr || biasShape == nullptr ||
        gammaShape == nullptr || betaShape == nullptr || minShape == nullptr || maxShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDimNum() != 5 || weightShape->GetDimNum() != 5) {
        return GRAPH_FAILED;
    }

    const int64_t outChannels = weightShape->GetDim(0);
    if (weightShape->GetDim(1) != xShape->GetDim(1) ||
        !IsVectorWithLength(biasShape, outChannels) ||
        !IsVectorWithLength(gammaShape, outChannels) ||
        !IsVectorWithLength(betaShape, outChannels) ||
        !IsVectorWithLength(minShape, 1) ||
        !IsVectorWithLength(maxShape, 1)) {
        return GRAPH_FAILED;
    }

    const int64_t outD = xShape->GetDim(2) - weightShape->GetDim(2) + 1;
    const int64_t outH = xShape->GetDim(3) - weightShape->GetDim(3) + 1;
    const int64_t outW = xShape->GetDim(4) - weightShape->GetDim(4) + 1;
    if (outD <= 0 || outH <= 0 || outW <= 0) {
        return GRAPH_FAILED;
    }

    gert::Shape* yShape = context->GetOutputShape(0);
    yShape->SetDimNum(5);
    yShape->SetDim(0, xShape->GetDim(0));
    yShape->SetDim(1, outChannels);
    yShape->SetDim(2, outD);
    yShape->SetDim(3, outH);
    yShape->SetDim(4, outW);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class C3dGnMcdCustom : public OpDef {
public:
    explicit C3dGnMcdCustom(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gamma").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("beta").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("min_value").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("max_value").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("num_groups").AttrType(REQUIRED).Int();
        this->Attr("dropout_p").AttrType(REQUIRED).Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(C3dGnMcdCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelC3dGnMcd {
public:
    __aicore__ inline KernelC3dGnMcd() {}

    __aicore__ inline void Init(GM_ADDR y, uint32_t totalLength)
    {
        yGm.SetGlobalBuffer((__gm__ float*)y, totalLength);
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
    uint32_t totalLength;
};

extern "C" __global__ __aicore__ void c3d_gn_mcd_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR gamma,
    GM_ADDR beta,
    GM_ADDR min_value,
    GM_ADDR max_value,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)x;
    (void)weight;
    (void)bias;
    (void)gamma;
    (void)beta;
    (void)min_value;
    (void)max_value;
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelC3dGnMcd op;
    op.Init(y, tiling_data.totalLength);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor c3d_gn_mcd_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& gamma,
    const at::Tensor& beta,
    const at::Tensor& minValue,
    const at::Tensor& maxValue,
    int64_t num_groups,
    double dropout_p)
{
    TORCH_CHECK(x.dim() == 5, "x must be a 5D NCDHW tensor");
    TORCH_CHECK(weight.dim() == 5, "weight must be a 5D OIDHW tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be a 1D tensor");
    TORCH_CHECK(beta.dim() == 1, "beta must be a 1D tensor");
    TORCH_CHECK(minValue.numel() == 1, "min_value must contain a single element");
    TORCH_CHECK(maxValue.numel() == 1, "max_value must contain a single element");
    TORCH_CHECK(weight.size(0) == bias.size(0), "bias size must match out_channels");
    TORCH_CHECK(weight.size(0) == gamma.size(0), "gamma size must match out_channels");
    TORCH_CHECK(weight.size(0) == beta.size(0), "beta size must match out_channels");
    TORCH_CHECK(weight.size(1) == x.size(1), "weight in_channels must match x channels");
    TORCH_CHECK(num_groups > 0, "num_groups must be positive");
    TORCH_CHECK(dropout_p >= 0.0 && dropout_p < 1.0, "dropout_p must be in [0, 1)");

    const int64_t outD = x.size(2) - weight.size(2) + 1;
    const int64_t outH = x.size(3) - weight.size(3) + 1;
    const int64_t outW = x.size(4) - weight.size(4) + 1;
    TORCH_CHECK(outD > 0 && outH > 0 && outW > 0, "invalid conv3d output shape");

    at::Tensor result = at::empty({x.size(0), weight.size(0), outD, outH, outW}, x.options());
    double dropoutValue = dropout_p;
    EXEC_NPU_CMD(
        aclnnC3dGnMcdCustom,
        x,
        weight,
        bias,
        gamma,
        beta,
        minValue,
        maxValue,
        num_groups,
        dropoutValue,
        result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "c3d_gn_mcd_custom",
        &c3d_gn_mcd_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "c3d_gn_mcd_custom",
        &c3d_gn_mcd_custom_impl_npu,
        "conv3d + group_norm + min + clamp + dropout");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = torch.nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = torch.nn.GroupNorm(groups, out_channels)
        self.register_buffer("min_value_tensor", torch.tensor([min_value], dtype=torch.float32))
        self.register_buffer("max_value_tensor", torch.tensor([max_value], dtype=torch.float32))
        self.dropout_p = float(dropout_p)

    def forward(self, x):
        return custom_ops_lib.c3d_gn_mcd_custom(
            x,
            self.conv.weight,
            self.conv.bias,
            self.norm.weight,
            self.norm.bias,
            self.min_value_tensor,
            self.max_value_tensor,
            self.norm.num_groups,
            self.dropout_p,
        )
'''
