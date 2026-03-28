project_json_src='''
[
    {
        "op": "MatmulMinSubtractCustom",
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
            },
            {
                "name": "constant",
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
BEGIN_TILING_DATA_DEF(MatmulMinSubtractCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, inFeatures);
TILING_DATA_FIELD_DEF(uint32_t, outFeatures);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulMinSubtractCustom, MatmulMinSubtractCustomTilingData)
} // namespace optiling
"""

host_operator_src="""
#include "matmul_min_subtract_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
inline bool IsVectorWithLength(const gert::Shape* shape, int64_t expected)
{
    return shape != nullptr && shape->GetDimNum() == 1 && shape->GetDim(0) == expected;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* xShape = context->GetInputShape(0);
    const gert::StorageShape* weightShape = context->GetInputShape(1);
    const gert::StorageShape* biasShape = context->GetInputShape(2);
    const gert::StorageShape* constantShape = context->GetInputShape(3);
    if (xShape == nullptr || weightShape == nullptr || biasShape == nullptr || constantShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xStorageShape = xShape->GetStorageShape();
    const auto weightStorageShape = weightShape->GetStorageShape();
    const auto biasStorageShape = biasShape->GetStorageShape();
    const auto constantStorageShape = constantShape->GetStorageShape();
    if (xStorageShape.GetDimNum() != 2 || weightStorageShape.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const int32_t batchSize = static_cast<int32_t>(xStorageShape.GetDim(0));
    const int32_t inFeatures = static_cast<int32_t>(xStorageShape.GetDim(1));
    const int32_t outFeatures = static_cast<int32_t>(weightStorageShape.GetDim(0));
    const int32_t weightInFeatures = static_cast<int32_t>(weightStorageShape.GetDim(1));
    if (batchSize <= 0 || inFeatures <= 0 || outFeatures <= 0 || weightInFeatures != inFeatures) {
        return ge::GRAPH_FAILED;
    }

    if (!IsVectorWithLength(&biasStorageShape, outFeatures) ||
        !IsVectorWithLength(&constantStorageShape, 1)) {
        return ge::GRAPH_FAILED;
    }

    MatmulMinSubtractCustomTilingData tiling;
    tiling.set_batchSize(static_cast<uint32_t>(batchSize));
    tiling.set_inFeatures(static_cast<uint32_t>(inFeatures));
    tiling.set_outFeatures(static_cast<uint32_t>(outFeatures));

    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    const gert::Shape* weightShape = context->GetInputShape(1);
    if (xShape == nullptr || weightShape == nullptr || xShape->GetDimNum() != 2 || weightShape->GetDimNum() != 2) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDim(1) != weightShape->GetDim(1)) {
        return GRAPH_FAILED;
    }

    gert::Shape* yShape = context->GetOutputShape(0);
    yShape->SetDimNum(2);
    yShape->SetDim(0, xShape->GetDim(0));
    yShape->SetDim(1, weightShape->GetDim(0));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class MatmulMinSubtractCustom : public OpDef {
public:
    explicit MatmulMinSubtractCustom(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("constant").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(MatmulMinSubtractCustom);
} // namespace ops
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class MatmulMinSubtractKernel {
public:
    __aicore__ inline MatmulMinSubtractKernel() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR constant,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t inFeatures,
        uint32_t outFeatures)
    {
        this->batchSize = batchSize;
        this->inFeatures = inFeatures;
        this->outFeatures = outFeatures;

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), static_cast<uint64_t>(batchSize) * inFeatures);
        weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weight), static_cast<uint64_t>(outFeatures) * inFeatures);
        biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias), outFeatures);
        constantGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(constant), 1);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), static_cast<uint64_t>(batchSize) * outFeatures);
    }

    __aicore__ inline void Process()
    {
        const float constantValue = constantGm.GetValue(0);
        for (uint32_t batchIdx = 0; batchIdx < this->batchSize; ++batchIdx) {
            const uint64_t xBase = static_cast<uint64_t>(batchIdx) * this->inFeatures;
            const uint64_t yBase = static_cast<uint64_t>(batchIdx) * this->outFeatures;
            for (uint32_t outIdx = 0; outIdx < this->outFeatures; ++outIdx) {
                float acc = biasGm.GetValue(outIdx);
                const uint64_t weightBase = static_cast<uint64_t>(outIdx) * this->inFeatures;
                for (uint32_t inIdx = 0; inIdx < this->inFeatures; ++inIdx) {
                    acc += xGm.GetValue(xBase + inIdx) * weightGm.GetValue(weightBase + inIdx);
                }
                const float minValue = acc < constantValue ? acc : constantValue;
                yGm.SetValue(yBase + outIdx, minValue - constantValue);
            }
        }
    }

private:
    GlobalTensor<float> xGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> biasGm;
    GlobalTensor<float> constantGm;
    GlobalTensor<float> yGm;
    uint32_t batchSize = 0;
    uint32_t inFeatures = 0;
    uint32_t outFeatures = 0;
};

extern "C" __global__ __aicore__ void matmul_min_subtract_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR constant,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    MatmulMinSubtractKernel op;
    op.Init(x, weight, bias, constant, y, tilingData.batchSize, tilingData.inFeatures, tilingData.outFeatures);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor matmul_min_subtract_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& constant)
{
    at::Tensor result = at::empty({x.size(0), weight.size(0)}, x.options());
    EXEC_NPU_CMD(aclnnMatmulMinSubtractCustom, x, weight, bias, constant, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("matmul_min_subtract_custom", &matmul_min_subtract_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_min_subtract_custom", &matmul_min_subtract_custom_impl_npu, "matmul + min + subtract custom");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.constant = torch.nn.Parameter(torch.tensor([constant], dtype=torch.float32))

    def forward(self, x):
        return custom_ops_lib.matmul_min_subtract_custom(
            x,
            self.linear.weight,
            self.linear.bias,
            self.constant,
        )
'''
