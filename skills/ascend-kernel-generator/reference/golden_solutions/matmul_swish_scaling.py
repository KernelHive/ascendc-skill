project_json_src = '''
[
    {
        "op": "MatmulSwishScalingCustom",
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
                "name": "scaling_factor",
                "param_type": "optional",
                "type": "float",
                "default_value": "1.0"
            }
        ]
    }
]
'''

host_tiling_src = """
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulSwishScalingCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, mDim);
TILING_DATA_FIELD_DEF(uint32_t, nDim);
TILING_DATA_FIELD_DEF(uint32_t, kDim);
TILING_DATA_FIELD_DEF(uint32_t, blockDim);
TILING_DATA_FIELD_DEF(float, scalingFactor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    MatmulSwishScalingCustom,
    MatmulSwishScalingCustomTilingData)
} // namespace optiling
"""

host_operator_src = """
#include "matmul_swish_scaling_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto shapeX = context->GetInputTensor(0)->GetOriginShape();
    auto shapeWeight = context->GetInputTensor(1)->GetOriginShape();
    auto shapeBias = context->GetInputTensor(2)->GetOriginShape();

    if (shapeX.GetDimNum() != 2 || shapeWeight.GetDimNum() != 2 || shapeBias.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const int32_t mDim = shapeX.GetDim(0);
    const int32_t kDim = shapeX.GetDim(1);
    const int32_t weightK = shapeWeight.GetDim(0);
    const int32_t nDim = shapeWeight.GetDim(1);
    if (mDim <= 0 || nDim <= 0 || kDim <= 0 || weightK != kDim || shapeBias.GetDim(0) != nDim) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const float *scalingPtr = attrs == nullptr ? nullptr : attrs->GetAttrPointer<float>(0);

    MatmulSwishScalingCustomTilingData tiling;
    tiling.set_mDim(static_cast<uint32_t>(mDim));
    tiling.set_nDim(static_cast<uint32_t>(nDim));
    tiling.set_kDim(static_cast<uint32_t>(kDim));
    tiling.set_blockDim(static_cast<uint32_t>(mDim >= 8 ? 8 : mDim));
    tiling.set_scalingFactor(scalingPtr == nullptr ? 1.0f : *scalingPtr);

    context->SetBlockDim(mDim >= 8 ? 8 : mDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    const gert::Shape *weightShape = context->GetInputShape(1);
    const gert::Shape *biasShape = context->GetInputShape(2);
    if (xShape == nullptr || weightShape == nullptr || biasShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDimNum() != 2 || weightShape->GetDimNum() != 2 || biasShape->GetDimNum() != 1) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDim(1) != weightShape->GetDim(0) || weightShape->GetDim(1) != biasShape->GetDim(0)) {
        return GRAPH_FAILED;
    }

    gert::Shape *outShape = context->GetOutputShape(0);
    outShape->SetDimNum(2);
    outShape->SetDim(0, xShape->GetDim(0));
    outShape->SetDim(1, weightShape->GetDim(1));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class MatmulSwishScalingCustom : public OpDef {
public:
    explicit MatmulSwishScalingCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("scaling_factor").AttrType(OPTIONAL).Float(1.0);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(MatmulSwishScalingCustom);
} // namespace ops
"""

kernel_src = """
#include "kernel_operator.h"

using namespace AscendC;

class MatmulSwishScalingKernel {
public:
    __aicore__ inline MatmulSwishScalingKernel() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR y,
        uint32_t mDim,
        uint32_t nDim,
        uint32_t kDim,
        uint32_t blockDim,
        float scalingFactor)
    {
        this->mDim = mDim;
        this->nDim = nDim;
        this->kDim = kDim;
        this->blockDim = blockDim;
        this->scalingFactor = scalingFactor;

        xGm.SetGlobalBuffer((__gm__ float *)x, static_cast<uint64_t>(mDim) * kDim);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, static_cast<uint64_t>(kDim) * nDim);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, nDim);
        yGm.SetGlobalBuffer((__gm__ float *)y, static_cast<uint64_t>(mDim) * nDim);
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreIdx = GetBlockIdx();
        for (uint32_t row = coreIdx; row < this->mDim; row += this->blockDim) {
            const uint64_t xOffset = static_cast<uint64_t>(row) * this->kDim;
            const uint64_t yOffset = static_cast<uint64_t>(row) * this->nDim;
            for (uint32_t col = 0; col < this->nDim; ++col) {
                float acc = biasGm.GetValue(col);
                for (uint32_t kk = 0; kk < this->kDim; ++kk) {
                    const float xValue = xGm.GetValue(xOffset + kk);
                    const float wValue = weightGm.GetValue(static_cast<uint64_t>(kk) * this->nDim + col);
                    acc += xValue * wValue;
                }
                const float swishValue = acc * Sigmoid(acc);
                yGm.SetValue(yOffset + col, swishValue * this->scalingFactor);
            }
        }
    }

private:
    __aicore__ inline float FastExp(float x) const
    {
        constexpr float ln2 = 0.69314718056f;
        if (x < -20.0f) {
            return 0.0f;
        }
        if (x > 20.0f) {
            x = 20.0f;
        }

        int32_t k = 0;
        while (x > 0.5f * ln2) {
            x -= ln2;
            ++k;
        }
        while (x < -0.5f * ln2) {
            x += ln2;
            --k;
        }

        const float x2 = x * x;
        const float x3 = x2 * x;
        const float x4 = x3 * x;
        const float x5 = x4 * x;
        float result = 1.0f + x + 0.5f * x2 + 0.16666667f * x3 + 0.04166667f * x4 + 0.0083333333f * x5;
        if (k > 0) {
            for (int32_t i = 0; i < k; ++i) {
                result *= 2.0f;
            }
        } else {
            for (int32_t i = 0; i < -k; ++i) {
                result *= 0.5f;
            }
        }
        return result;
    }

    __aicore__ inline float Sigmoid(float x) const
    {
        return 1.0f / (1.0f + FastExp(-x));
    }

    GlobalTensor<float> xGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> biasGm;
    GlobalTensor<float> yGm;
    uint32_t mDim = 0;
    uint32_t nDim = 0;
    uint32_t kDim = 0;
    uint32_t blockDim = 1;
    float scalingFactor = 1.0f;
};

extern "C" __global__ __aicore__ void matmul_swish_scaling_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    MatmulSwishScalingKernel op;
    op.Init(
        x,
        weight,
        bias,
        y,
        tiling_data.mDim,
        tiling_data.nDim,
        tiling_data.kDim,
        tiling_data.blockDim,
        tiling_data.scalingFactor);
    op.Process();
}
"""

python_bind_src = """
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor matmul_swish_scaling_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &bias,
    double scaling_factor)
{
    TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor");
    TORCH_CHECK(weight.dim() == 2, "weight must be a 2D tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
    TORCH_CHECK(x.size(1) == weight.size(0), "x.size(1) must match weight.size(0)");
    TORCH_CHECK(weight.size(1) == bias.size(0), "bias size must match weight.size(1)");

    at::Tensor result = at::empty({x.size(0), weight.size(1)}, x.options());
    EXEC_NPU_CMD(aclnnMatmulSwishScalingCustom, x, weight, bias, scaling_factor, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("matmul_swish_scaling_custom", &matmul_swish_scaling_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "matmul_swish_scaling_custom",
        &matmul_swish_scaling_custom_impl_npu,
        "matmul + swish + scaling");
}
"""

model_src = '''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = torch.nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        weight = self.matmul.weight.transpose(0, 1).contiguous()
        return custom_ops_lib.matmul_swish_scaling_custom(
            x,
            weight,
            self.matmul.bias,
            self.scaling_factor,
        )
'''
