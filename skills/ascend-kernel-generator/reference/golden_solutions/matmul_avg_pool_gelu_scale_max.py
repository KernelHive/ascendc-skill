project_json_src='''
[
    {
        "op": "MatmulAvgPoolGeluScaleMaxCustom",
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
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulAvgPoolGeluScaleMaxCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inFeatures);
    TILING_DATA_FIELD_DEF(uint32_t, outFeatures);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    MatmulAvgPoolGeluScaleMaxCustom,
    MatmulAvgPoolGeluScaleMaxCustomTilingData)
}
"""

host_operator_src="""
#include "matmul_avg_pool_gelu_scale_max_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *xStorage = context->GetInputShape(0);
    const gert::StorageShape *weightStorage = context->GetInputShape(1);
    const gert::StorageShape *biasStorage = context->GetInputShape(2);
    if (xStorage == nullptr || weightStorage == nullptr || biasStorage == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xShape = xStorage->GetStorageShape();
    const auto weightShape = weightStorage->GetStorageShape();
    const auto biasShape = biasStorage->GetStorageShape();
    if (xShape.GetDimNum() != 2 || weightShape.GetDimNum() != 2 || biasShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t inFeatures = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t weightInFeatures = static_cast<uint32_t>(weightShape.GetDim(0));
    const uint32_t outFeatures = static_cast<uint32_t>(weightShape.GetDim(1));
    if (batchSize == 0 || inFeatures == 0 || outFeatures == 0) {
        return ge::GRAPH_FAILED;
    }
    if (weightInFeatures != inFeatures || static_cast<uint32_t>(biasShape.GetDim(0)) != outFeatures) {
        return ge::GRAPH_FAILED;
    }

    MatmulAvgPoolGeluScaleMaxCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_inFeatures(inFeatures);
    tiling.set_outFeatures(outFeatures);

    context->SetBlockDim(batchSize);
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

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(2);
    outputShape->SetDim(0, xShape->GetDim(0));
    outputShape->SetDim(1, weightShape->GetDim(1));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class MatmulAvgPoolGeluScaleMaxCustom : public OpDef {
public:
    explicit MatmulAvgPoolGeluScaleMaxCustom(const char *name) : OpDef(name)
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

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(MatmulAvgPoolGeluScaleMaxCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelMatmulAvgPoolGeluScaleMax {
public:
    __aicore__ inline KernelMatmulAvgPoolGeluScaleMax() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t inFeatures,
        uint32_t outFeatures)
    {
        this->batchSize = batchSize;
        this->inFeatures = inFeatures;
        this->outFeatures = outFeatures;

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x), static_cast<uint64_t>(batchSize) * inFeatures);
        weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weight), static_cast<uint64_t>(inFeatures) * outFeatures);
        biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bias), outFeatures);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y), static_cast<uint64_t>(batchSize) * outFeatures);
    }

    __aicore__ inline void Process()
    {
        const uint32_t row = GetBlockIdx();
        if (row >= this->batchSize) {
            return;
        }

        const uint32_t xRowBase = row * this->inFeatures;
        const uint32_t yRowBase = row * this->outFeatures;
        for (uint32_t outCol = 0; outCol < this->outFeatures; ++outCol) {
            float acc = biasGm.GetValue(outCol);
            for (uint32_t kk = 0; kk < this->inFeatures; ++kk) {
                const float xValue = xGm.GetValue(xRowBase + kk);
                const float weightValue = weightGm.GetValue(kk * this->outFeatures + outCol);
                acc += xValue * weightValue;
            }
            yGm.SetValue(yRowBase + outCol, acc);
        }
    }

private:
    GlobalTensor<float> xGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> biasGm;
    GlobalTensor<float> yGm;
    uint32_t batchSize = 0;
    uint32_t inFeatures = 0;
    uint32_t outFeatures = 0;
};

extern "C" __global__ __aicore__ void matmul_avg_pool_gelu_scale_max_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    KernelMatmulAvgPoolGeluScaleMax op;
    op.Init(
        x,
        weight,
        bias,
        y,
        tilingData.batchSize,
        tilingData.inFeatures,
        tilingData.outFeatures);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor matmul_avg_pool_gelu_scale_max_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &bias)
{
    TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor");
    TORCH_CHECK(weight.dim() == 2, "weight must be a 2D tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
    TORCH_CHECK(x.size(1) == weight.size(0), "x.size(1) must match weight.size(0)");
    TORCH_CHECK(weight.size(1) == bias.size(0), "weight.size(1) must match bias.size(0)");

    at::Tensor result = at::empty({x.size(0), weight.size(1)}, x.options());
    EXEC_NPU_CMD(aclnnMatmulAvgPoolGeluScaleMaxCustom, x, weight, bias, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("matmul_avg_pool_gelu_scale_max_custom", &matmul_avg_pool_gelu_scale_max_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "matmul_avg_pool_gelu_scale_max_custom",
        &matmul_avg_pool_gelu_scale_max_custom_impl_npu,
        "matmul stage for matmul_avg_pool_gelu_scale_max");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = torch.nn.Linear(in_features, out_features)
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor

    def _post_process(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, out_features = x.shape
        pooled = x.view(batch_size, out_features // self.pool_kernel_size, self.pool_kernel_size).mean(dim=2)
        gelu = 0.5 * pooled * (1.0 + torch.erf(pooled / 1.4142135623730951))
        return (gelu * self.scale_factor).amax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.matmul.weight.transpose(0, 1).contiguous()
        bias = self.matmul.bias.contiguous()
        matmul_out = custom_ops_lib.matmul_avg_pool_gelu_scale_max_custom(
            x.contiguous(),
            weight,
            bias,
        )
        return self._post_process(matmul_out)
'''
