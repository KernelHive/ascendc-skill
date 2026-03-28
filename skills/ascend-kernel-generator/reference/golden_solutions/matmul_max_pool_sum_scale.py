project_json_src = '''
[
    {
        "op": "MatmulMaxPoolSumScaleCustom",
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
                "name": "scale",
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

host_tiling_src = """
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulMaxPoolSumScaleCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, inFeatures);
TILING_DATA_FIELD_DEF(uint32_t, outFeatures);
TILING_DATA_FIELD_DEF(uint32_t, pooledLength);
TILING_DATA_FIELD_DEF(uint32_t, kernelSize);
TILING_DATA_FIELD_DEF(uint32_t, stride);
TILING_DATA_FIELD_DEF(uint32_t, blockDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    MatmulMaxPoolSumScaleCustom,
    MatmulMaxPoolSumScaleCustomTilingData)
} // namespace optiling
"""

host_operator_src = """
#include "matmul_max_pool_sum_scale_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
constexpr uint32_t kKernelSize = 2;
constexpr uint32_t kStride = 2;
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *xStorage = context->GetInputShape(0);
    const gert::StorageShape *weightStorage = context->GetInputShape(1);
    const gert::StorageShape *biasStorage = context->GetInputShape(2);
    const gert::StorageShape *scaleStorage = context->GetInputShape(3);
    if (xStorage == nullptr || weightStorage == nullptr || biasStorage == nullptr || scaleStorage == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &xShape = xStorage->GetStorageShape();
    const gert::Shape &weightShape = weightStorage->GetStorageShape();
    const gert::Shape &biasShape = biasStorage->GetStorageShape();
    const gert::Shape &scaleShape = scaleStorage->GetStorageShape();

    if (xShape.GetDimNum() != 2 || weightShape.GetDimNum() != 2 || biasShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }
    if (!(scaleShape.GetDimNum() == 0 || (scaleShape.GetDimNum() == 1 && scaleShape.GetDim(0) == 1))) {
        return ge::GRAPH_FAILED;
    }

    const int64_t batchSize = xShape.GetDim(0);
    const int64_t inFeatures = xShape.GetDim(1);
    const int64_t weightInFeatures = weightShape.GetDim(0);
    const int64_t outFeatures = weightShape.GetDim(1);
    if (batchSize <= 0 || inFeatures <= 0 || outFeatures <= 0) {
        return ge::GRAPH_FAILED;
    }
    if (weightInFeatures != inFeatures || biasShape.GetDim(0) != outFeatures || outFeatures < static_cast<int64_t>(kKernelSize)) {
        return ge::GRAPH_FAILED;
    }

    const int64_t pooledLength = (outFeatures - static_cast<int64_t>(kKernelSize)) / static_cast<int64_t>(kStride) + 1;
    if (pooledLength <= 0) {
        return ge::GRAPH_FAILED;
    }

    MatmulMaxPoolSumScaleCustomTilingData tiling;
    tiling.set_batchSize(static_cast<uint32_t>(batchSize));
    tiling.set_inFeatures(static_cast<uint32_t>(inFeatures));
    tiling.set_outFeatures(static_cast<uint32_t>(outFeatures));
    tiling.set_pooledLength(static_cast<uint32_t>(pooledLength));
    tiling.set_kernelSize(kKernelSize);
    tiling.set_stride(kStride);

    tiling.set_blockDim(1U);
    context->SetBlockDim(1U);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    const gert::Shape *weightShape = context->GetInputShape(1);
    const gert::Shape *biasShape = context->GetInputShape(2);
    const gert::Shape *scaleShape = context->GetInputShape(3);
    if (xShape == nullptr || weightShape == nullptr || biasShape == nullptr || scaleShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDimNum() != 2 || weightShape->GetDimNum() != 2 || biasShape->GetDimNum() != 1) {
        return GRAPH_FAILED;
    }
    if (!(scaleShape->GetDimNum() == 0 || (scaleShape->GetDimNum() == 1 && scaleShape->GetDim(0) == 1))) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDim(1) != weightShape->GetDim(0) || biasShape->GetDim(0) != weightShape->GetDim(1)) {
        return GRAPH_FAILED;
    }
    if (weightShape->GetDim(1) < 2) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(1);
    outputShape->SetDim(0, xShape->GetDim(0));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class MatmulMaxPoolSumScaleCustom : public OpDef {
public:
    explicit MatmulMaxPoolSumScaleCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(MatmulMaxPoolSumScaleCustom);
} // namespace ops
"""

kernel_src = """
#include "kernel_operator.h"

using namespace AscendC;

class MatmulMaxPoolSumScaleKernel {
public:
    __aicore__ inline MatmulMaxPoolSumScaleKernel() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR scale,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t inFeatures,
        uint32_t outFeatures,
        uint32_t pooledLength,
        uint32_t kernelSize,
        uint32_t stride,
        uint32_t blockDim)
    {
        this->batchSize = batchSize;
        this->inFeatures = inFeatures;
        this->outFeatures = outFeatures;
        this->pooledLength = pooledLength;
        this->kernelSize = kernelSize;
        this->stride = stride;
        this->blockDim = blockDim;

        xGm.SetGlobalBuffer((__gm__ float *)x, static_cast<uint64_t>(batchSize) * inFeatures);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, static_cast<uint64_t>(inFeatures) * outFeatures);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, outFeatures);
        scaleGm.SetGlobalBuffer((__gm__ float *)scale, 1);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize);
    }

    __aicore__ inline void Process()
    {
        const float scaleValue = scaleGm.GetValue(0);
        for (uint32_t row = 0; row < this->batchSize; ++row) {
            const uint64_t xBase = static_cast<uint64_t>(row) * this->inFeatures;
            float reduced = 0.0f;
            for (uint32_t poolIdx = 0; poolIdx < this->pooledLength; ++poolIdx) {
                const uint32_t startCol = poolIdx * this->stride;
                float maxValue = ComputeMatmulValue(xBase, startCol);
                for (uint32_t k = 1; k < this->kernelSize; ++k) {
                    const float candidate = ComputeMatmulValue(xBase, startCol + k);
                    if (candidate > maxValue) {
                        maxValue = candidate;
                    }
                }
                reduced += maxValue;
            }
            yGm.SetValue(row, reduced * scaleValue);
        }
    }

private:
    __aicore__ inline float ComputeMatmulValue(uint64_t xBase, uint32_t col) const
    {
        float acc = biasGm.GetValue(col);
        for (uint32_t kk = 0; kk < this->inFeatures; ++kk) {
            const float xValue = xGm.GetValue(xBase + kk);
            const float wValue = weightGm.GetValue(static_cast<uint64_t>(kk) * this->outFeatures + col);
            acc += xValue * wValue;
        }
        return acc;
    }

    GlobalTensor<float> xGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> biasGm;
    GlobalTensor<float> scaleGm;
    GlobalTensor<float> yGm;
    uint32_t batchSize = 0;
    uint32_t inFeatures = 0;
    uint32_t outFeatures = 0;
    uint32_t pooledLength = 0;
    uint32_t kernelSize = 0;
    uint32_t stride = 0;
    uint32_t blockDim = 1;
};

extern "C" __global__ __aicore__ void matmul_max_pool_sum_scale_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR scale,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    MatmulMaxPoolSumScaleKernel op;
    op.Init(
        x,
        weight,
        bias,
        scale,
        y,
        tiling_data.batchSize,
        tiling_data.inFeatures,
        tiling_data.outFeatures,
        tiling_data.pooledLength,
        tiling_data.kernelSize,
        tiling_data.stride,
        tiling_data.blockDim);
    op.Process();
}
"""

python_bind_src = """
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor matmul_max_pool_sum_scale_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &bias,
    const at::Tensor &scale)
{
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(scale.numel() == 1, "scale must contain exactly one value");
    TORCH_CHECK(x.size(1) == weight.size(0), "x.size(1) must match weight.size(0)");
    TORCH_CHECK(weight.size(1) == bias.size(0), "weight.size(1) must match bias.size(0)");
    TORCH_CHECK(weight.size(1) >= 2, "out_features must be at least 2");

    at::Tensor result = at::empty({x.size(0)}, x.options());
    EXEC_NPU_CMD(aclnnMatmulMaxPoolSumScaleCustom, x, weight, bias, scale, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("matmul_max_pool_sum_scale_custom", &matmul_max_pool_sum_scale_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "matmul_max_pool_sum_scale_custom",
        &matmul_max_pool_sum_scale_custom_impl_npu,
        "matmul + max_pool1d + sum + scale");
}
"""

model_src = '''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        if kernel_size != 2:
            raise ValueError("This AscendC implementation currently supports kernel_size=2 only.")
        self.matmul = torch.nn.Linear(in_features, out_features)
        self.register_buffer("scale_tensor", torch.tensor([scale_factor], dtype=torch.float32))

    def forward(self, x):
        weight = self.matmul.weight.transpose(0, 1).contiguous()
        return custom_ops_lib.matmul_max_pool_sum_scale_custom(
            x,
            weight,
            self.matmul.bias,
            self.scale_tensor,
        )
'''
