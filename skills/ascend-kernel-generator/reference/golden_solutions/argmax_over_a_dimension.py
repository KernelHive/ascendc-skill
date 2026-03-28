project_json_src='''
[
    {
        "op": "ArgmaxOverADimensionCustom",
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
                    "int64"
                ]
            }
        ],
        "attr": [
            {
                "name": "dim",
                "param_type": "required",
                "type": "int"
            }
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ArgmaxOverADimensionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, reduceSize);
    TILING_DATA_FIELD_DEF(uint32_t, colSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ArgmaxOverADimensionCustom, ArgmaxOverADimensionCustomTilingData)
}
"""

host_operator_src="""
#include "argmax_over_a_dimension_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    ArgmaxOverADimensionCustomTilingData tiling;
    const gert::StorageShape *inputShape = context->GetInputShape(0);
    const auto shape = inputShape->GetStorageShape();
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int64_t *dimPtr = attrs->GetAttrPointer<int64_t>(0);

    if (shape.GetDimNum() != 3 || dimPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t dim = *dimPtr;
    if (dim < 0) {
        dim += shape.GetDimNum();
    }
    if (dim != 1) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(shape.GetDim(0));
    const uint32_t reduceSize = static_cast<uint32_t>(shape.GetDim(1));
    const uint32_t colSize = static_cast<uint32_t>(shape.GetDim(2));
    if (batchSize == 0 || reduceSize == 0 || colSize == 0) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(batchSize);
    tiling.set_batchSize(batchSize);
    tiling.set_reduceSize(reduceSize);
    tiling.set_colSize(colSize);
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
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int64_t *dimPtr = attrs->GetAttrPointer<int64_t>(0);
    if (inputShape->GetDimNum() != 3 || dimPtr == nullptr) {
        return GRAPH_FAILED;
    }

    int64_t dim = *dimPtr;
    if (dim < 0) {
        dim += inputShape->GetDimNum();
    }
    if (dim != 1) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(2);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, inputShape->GetDim(2));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_INT64);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ArgmaxOverADimensionCustom : public OpDef {
public:
    explicit ArgmaxOverADimensionCustom(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("dim").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ArgmaxOverADimensionCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelArgmaxOverADimension {
public:
    __aicore__ inline KernelArgmaxOverADimension() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t reduceSize,
        uint32_t colSize)
    {
        this->batchSize = batchSize;
        this->reduceSize = reduceSize;
        this->colSize = colSize;
        this->blockIdx = GetBlockIdx();

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * reduceSize * colSize);
        yGm.SetGlobalBuffer((__gm__ int64_t *)y, batchSize * colSize);
    }

    __aicore__ inline void Process()
    {
        if (this->blockIdx >= this->batchSize) {
            return;
        }

        const uint32_t batchBase = this->blockIdx * this->reduceSize * this->colSize;
        const uint32_t outBase = this->blockIdx * this->colSize;

        for (uint32_t colIdx = 0; colIdx < this->colSize; ++colIdx) {
            float maxValue = xGm.GetValue(batchBase + colIdx);
            int64_t maxIndex = 0;
            for (uint32_t reduceIdx = 1; reduceIdx < this->reduceSize; ++reduceIdx) {
                const uint32_t offset = batchBase + reduceIdx * this->colSize + colIdx;
                const float value = xGm.GetValue(offset);
                if (value > maxValue) {
                    maxValue = value;
                    maxIndex = static_cast<int64_t>(reduceIdx);
                }
            }
            yGm.SetValue(outBase + colIdx, maxIndex);
        }
    }

private:
    GlobalTensor<float> xGm;
    GlobalTensor<int64_t> yGm;
    uint32_t batchSize;
    uint32_t reduceSize;
    uint32_t colSize;
    uint32_t blockIdx;
};

extern "C" __global__ __aicore__ void argmax_over_a_dimension_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelArgmaxOverADimension op;
    op.Init(
        x,
        y,
        tiling_data.batchSize,
        tiling_data.reduceSize,
        tiling_data.colSize);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor argmax_over_a_dimension_custom_impl_npu(
    const at::Tensor &x,
    int64_t dim)
{
    TORCH_CHECK(x.dim() == 3, "argmax_over_a_dimension_custom expects a 3D tensor");
    if (dim < 0) {
        dim += x.dim();
    }
    TORCH_CHECK(dim == 1, "argmax_over_a_dimension_custom currently only supports dim=1");

    at::Tensor result = at::empty({x.size(0), x.size(2)}, x.options().dtype(at::kLong));
    EXEC_NPU_CMD(aclnnArgmaxOverADimensionCustom, x, dim, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("argmax_over_a_dimension_custom", &argmax_over_a_dimension_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "argmax_over_a_dimension_custom",
        &argmax_over_a_dimension_custom_impl_npu,
        "Argmax over dimension custom op");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.argmax_over_a_dimension_custom(x, self.dim)
'''
