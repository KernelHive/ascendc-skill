project_json_src='''
[
    {
        "op": "MatmulScalingResidualAddCustom",
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
                "param_type": "required",
                "type": "float",
                "default_value": "0.5"
            }
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulScalingResidualAddCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, mDim);
TILING_DATA_FIELD_DEF(uint32_t, nDim);
TILING_DATA_FIELD_DEF(uint32_t, kDim);
TILING_DATA_FIELD_DEF(uint32_t, tileNum);
TILING_DATA_FIELD_DEF(uint32_t, blockDim);
TILING_DATA_FIELD_DEF(float, scalingFactor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    MatmulScalingResidualAddCustom,
    MatmulScalingResidualAddCustomTilingData)
} // namespace optiling
"""

host_operator_src="""
#include "matmul_scaling_residual_add_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto shapeX = context->GetInputTensor(0)->GetOriginShape();
    auto shapeWeight = context->GetInputTensor(1)->GetOriginShape();
    auto shapeBias = context->GetInputTensor(2)->GetOriginShape();

    if (shapeX.GetDimNum() != 2 || shapeWeight.GetDimNum() != 2 || shapeBias.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const int32_t m = shapeX.GetDim(0);
    const int32_t k = shapeX.GetDim(1);
    const int32_t weightK = shapeWeight.GetDim(0);
    const int32_t n = shapeWeight.GetDim(1);
    if (m <= 0 || k <= 0 || n <= 0 || k != weightK || shapeBias.GetDim(0) != n) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const float *scalingFactor = attrs == nullptr ? nullptr : attrs->GetAttrPointer<float>(0);
    if (scalingFactor == nullptr) {
        return ge::GRAPH_FAILED;
    }

    MatmulScalingResidualAddCustomTilingData tiling;
    const uint32_t blockDim = static_cast<uint32_t>(m >= 8 ? 8 : m);
    tiling.set_mDim(static_cast<uint32_t>(m));
    tiling.set_nDim(static_cast<uint32_t>(n));
    tiling.set_kDim(static_cast<uint32_t>(k));
    tiling.set_tileNum(TILE_NUM);
    tiling.set_blockDim(blockDim);
    tiling.set_scalingFactor(*scalingFactor);
    context->SetBlockDim(blockDim);

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
class MatmulScalingResidualAddCustom : public OpDef {
public:
    explicit MatmulScalingResidualAddCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("scaling_factor").AttrType(REQUIRED).Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(MatmulScalingResidualAddCustom);
} // namespace ops
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class MatmulScalingResidualAddKernel {
public:
    __aicore__ inline MatmulScalingResidualAddKernel() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR y,
        uint32_t mDim,
        uint32_t nDim,
        uint32_t kDim,
        uint32_t tileNum,
        uint32_t blockDim,
        float scalingFactor,
        GM_ADDR workspace)
    {
        this->mDim = mDim;
        this->nDim = nDim;
        this->kDim = kDim;
        this->tileNum = tileNum;
        this->blockDim = blockDim;
        this->scalingFactor = scalingFactor;
        (void)workspace;

        xGm.SetGlobalBuffer((__gm__ float *)x, static_cast<uint64_t>(mDim) * kDim);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, static_cast<uint64_t>(kDim) * nDim);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, nDim);
        yGm.SetGlobalBuffer((__gm__ float *)y, static_cast<uint64_t>(mDim) * nDim);
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreIdx = GetBlockIdx();
        const uint32_t tileWidth = this->nDim < this->tileNum ? this->nDim : CeilDiv(this->nDim, this->tileNum);

        for (uint32_t row = coreIdx; row < this->mDim; row += this->blockDim) {
            const uint64_t xOffset = static_cast<uint64_t>(row) * this->kDim;
            const uint64_t yOffset = static_cast<uint64_t>(row) * this->nDim;
            for (uint32_t tile = 0; tile < this->tileNum; ++tile) {
                const uint32_t colBegin = tile * tileWidth;
                if (colBegin >= this->nDim) {
                    break;
                }
                const uint32_t colEnd = MinU32(this->nDim, colBegin + tileWidth);
                for (uint32_t col = colBegin; col < colEnd; ++col) {
                    float acc = biasGm.GetValue(col);
                    for (uint32_t kk = 0; kk < this->kDim; ++kk) {
                        const float xValue = xGm.GetValue(xOffset + kk);
                        const float wValue = weightGm.GetValue(static_cast<uint64_t>(kk) * this->nDim + col);
                        acc += xValue * wValue;
                    }
                    const float output = acc * this->scalingFactor + acc;
                    yGm.SetValue(yOffset + col, output);
                }
            }
        }
    }

private:
    __aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs)
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline uint32_t CeilDiv(uint32_t lhs, uint32_t rhs)
    {
        return (lhs + rhs - 1) / rhs;
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> weightGm;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t mDim;
    uint32_t nDim;
    uint32_t kDim;
    uint32_t tileNum;
    uint32_t blockDim;
    float scalingFactor;
};

extern "C" __global__ __aicore__ void matmul_scaling_residual_add_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    MatmulScalingResidualAddKernel op;
    op.Init(
        x,
        weight,
        bias,
        y,
        tilingData.mDim,
        tilingData.nDim,
        tilingData.kDim,
        tilingData.tileNum,
        tilingData.blockDim,
        tilingData.scalingFactor,
        workspace);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor matmul_scaling_residual_add_custom_impl_npu(
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
    EXEC_NPU_CMD(aclnnMatmulScalingResidualAddCustom, x, weight, bias, scaling_factor, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "matmul_scaling_residual_add_custom",
        &matmul_scaling_residual_add_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "matmul_scaling_residual_add_custom",
        &matmul_scaling_residual_add_custom_impl_npu,
        "matmul + scaling + residual add");
}
"""

model_src='''
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
        return custom_ops_lib.matmul_scaling_residual_add_custom(
            x,
            weight,
            self.matmul.bias,
            self.scaling_factor,
        )
'''
