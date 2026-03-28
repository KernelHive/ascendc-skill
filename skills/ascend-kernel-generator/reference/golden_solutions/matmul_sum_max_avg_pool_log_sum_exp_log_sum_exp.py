project_json_src='''
[
    {
        "op": "MatmulSumMaxAvgPoolLogSumExpLogSumExpCustom",
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
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulSumMaxAvgPoolLogSumExpLogSumExpCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inFeatures);
    TILING_DATA_FIELD_DEF(uint32_t, outFeatures);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    MatmulSumMaxAvgPoolLogSumExpLogSumExpCustom,
    MatmulSumMaxAvgPoolLogSumExpLogSumExpCustomTilingData)
}
"""

host_operator_src="""
#include "matmul_sum_max_avg_pool_log_sum_exp_log_sum_exp_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
inline bool IsMatrix(const gert::Shape *shape)
{
    return shape != nullptr && shape->GetDimNum() == 2;
}

inline bool IsVectorWithLength(const gert::Shape *shape, int64_t expected)
{
    return shape != nullptr && shape->GetDimNum() == 1 && shape->GetDim(0) == expected;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *xShape = context->GetInputShape(0);
    const gert::StorageShape *weightShape = context->GetInputShape(1);
    const gert::StorageShape *biasShape = context->GetInputShape(2);
    if (xShape == nullptr || weightShape == nullptr || biasShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xStorageShape = xShape->GetStorageShape();
    const auto weightStorageShape = weightShape->GetStorageShape();
    const auto biasStorageShape = biasShape->GetStorageShape();
    if (xStorageShape.GetDimNum() != 2 || weightStorageShape.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const int64_t batchSize = xStorageShape.GetDim(0);
    const int64_t inFeatures = xStorageShape.GetDim(1);
    const int64_t outFeatures = weightStorageShape.GetDim(0);
    if (batchSize <= 0 || inFeatures <= 0 || outFeatures <= 0) {
        return ge::GRAPH_FAILED;
    }
    if (weightStorageShape.GetDim(1) != inFeatures ||
        !IsVectorWithLength(&biasStorageShape, outFeatures)) {
        return ge::GRAPH_FAILED;
    }

    MatmulSumMaxAvgPoolLogSumExpLogSumExpCustomTilingData tiling;
    tiling.set_batchSize(static_cast<uint32_t>(batchSize));
    tiling.set_inFeatures(static_cast<uint32_t>(inFeatures));
    tiling.set_outFeatures(static_cast<uint32_t>(outFeatures));

    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    const gert::Shape *weightShape = context->GetInputShape(1);
    const gert::Shape *biasShape = context->GetInputShape(2);
    if (!IsMatrix(xShape) || !IsMatrix(weightShape)) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDim(1) != weightShape->GetDim(1) ||
        !IsVectorWithLength(biasShape, weightShape->GetDim(0))) {
        return GRAPH_FAILED;
    }

    gert::Shape *yShape = context->GetOutputShape(0);
    yShape->SetDimNum(2);
    yShape->SetDim(0, xShape->GetDim(0));
    yShape->SetDim(1, 1);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class MatmulSumMaxAvgPoolLogSumExpLogSumExpCustom : public OpDef {
public:
    explicit MatmulSumMaxAvgPoolLogSumExpLogSumExpCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(MatmulSumMaxAvgPoolLogSumExpLogSumExpCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelMatmulSumMaxAvgPoolLogSumExpLogSumExp {
public:
    __aicore__ inline KernelMatmulSumMaxAvgPoolLogSumExpLogSumExp() {}

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
        weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weight), static_cast<uint64_t>(outFeatures) * inFeatures);
        biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bias), outFeatures);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y), batchSize);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t batchIdx = 0; batchIdx < this->batchSize; ++batchIdx) {
            const uint64_t xBase = static_cast<uint64_t>(batchIdx) * this->inFeatures;
            const uint64_t yBase = static_cast<uint64_t>(batchIdx);
            float finalValue = 0.0f;
            for (uint32_t outIdx = 0; outIdx < this->outFeatures; ++outIdx) {
                const uint64_t weightBase = static_cast<uint64_t>(outIdx) * this->inFeatures;
                float acc = biasGm.GetValue(outIdx);
                for (uint32_t inIdx = 0; inIdx < this->inFeatures; ++inIdx) {
                    acc += xGm.GetValue(xBase + inIdx) * weightGm.GetValue(weightBase + inIdx);
                }
                finalValue += acc;
            }
            yGm.SetValue(yBase, finalValue);
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

extern "C" __global__ __aicore__ void matmul_sum_max_avg_pool_log_sum_exp_log_sum_exp_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    KernelMatmulSumMaxAvgPoolLogSumExpLogSumExp op;
    op.Init(x, weight, bias, y, tilingData.batchSize, tilingData.inFeatures, tilingData.outFeatures);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor matmul_sum_max_avg_pool_log_sum_exp_log_sum_exp_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &bias)
{
    TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor");
    TORCH_CHECK(weight.dim() == 2, "weight must be a 2D tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
    TORCH_CHECK(x.size(1) == weight.size(1), "x.size(1) must match weight.size(1)");
    TORCH_CHECK(weight.size(0) == bias.size(0), "weight.size(0) must match bias.size(0)");

    at::Tensor result = at::empty({x.size(0), 1}, x.options());
    EXEC_NPU_CMD(aclnnMatmulSumMaxAvgPoolLogSumExpLogSumExpCustom, x, weight, bias, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "matmul_sum_max_avg_pool_log_sum_exp_log_sum_exp_custom",
        &matmul_sum_max_avg_pool_log_sum_exp_log_sum_exp_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "matmul_sum_max_avg_pool_log_sum_exp_log_sum_exp_custom",
        &matmul_sum_max_avg_pool_log_sum_exp_log_sum_exp_custom_impl_npu,
        "matmul + sum + max + avg_pool + log_sum_exp + log_sum_exp custom");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.matmul_sum_max_avg_pool_log_sum_exp_log_sum_exp_custom(
            x.contiguous(),
            self.linear.weight.contiguous(),
            self.linear.bias.contiguous(),
        )
'''
