project_json_src='''
[
    {
        "op": "ArgminOverADimensionCustom",
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
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ArgminOverADimensionCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, reduceSize);
TILING_DATA_FIELD_DEF(uint32_t, innerSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ArgminOverADimensionCustom, ArgminOverADimensionCustomTilingData)
}
"""

host_operator_src="""
#include "argmin_over_a_dimension_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    ArgminOverADimensionCustomTilingData tiling;
    const gert::StorageShape *inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto &shape = inputShape->GetStorageShape();
    if (shape.GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(shape.GetDim(0));
    const uint32_t reduceSize = static_cast<uint32_t>(shape.GetDim(1));
    const uint32_t innerSize = static_cast<uint32_t>(shape.GetDim(2));
    if (batchSize == 0 || reduceSize == 0 || innerSize == 0) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(1);
    tiling.set_batchSize(batchSize);
    tiling.set_reduceSize(reduceSize);
    tiling.set_innerSize(innerSize);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *workspaceSize = context->GetWorkspaceSizes(1);
    workspaceSize[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    if (xShape == nullptr || xShape->GetDimNum() != 3) {
        return GRAPH_FAILED;
    }

    gert::Shape *yShape = context->GetOutputShape(0);
    *yShape = {xShape->GetDim(0), xShape->GetDim(2)};
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_INT64);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class ArgminOverADimensionCustom : public OpDef {
public:
    explicit ArgminOverADimensionCustom(const char *name) : OpDef(name)
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

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(ArgminOverADimensionCustom);
} // namespace ops
"""

kernel_src="""
#include "kernel_operator.h"
#include <cstdint>

using namespace AscendC;

class KernelArgminOverADimension {
public:
    __aicore__ inline KernelArgminOverADimension() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t reduceSize,
        uint32_t innerSize)
    {
        this->batchSize = batchSize;
        this->reduceSize = reduceSize;
        this->innerSize = innerSize;
        this->blockIdx = GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ float *)x, this->batchSize * this->reduceSize * this->innerSize);
        yGm.SetGlobalBuffer((__gm__ int64_t *)y, this->batchSize * this->innerSize);
    }

    __aicore__ inline void Process()
    {
        if (this->blockIdx != 0) {
            return;
        }

        for (uint32_t batchIdx = 0; batchIdx < this->batchSize; ++batchIdx) {
            const uint32_t batchInputOffset = batchIdx * this->reduceSize * this->innerSize;
            const uint32_t batchOutputOffset = batchIdx * this->innerSize;
            for (uint32_t innerIdx = 0; innerIdx < this->innerSize; ++innerIdx) {
                float minValue = xGm.GetValue(batchInputOffset + innerIdx);
                int64_t minIndex = 0;
                for (uint32_t reduceIdx = 1; reduceIdx < this->reduceSize; ++reduceIdx) {
                    const uint32_t offset = batchInputOffset + reduceIdx * this->innerSize + innerIdx;
                    const float value = xGm.GetValue(offset);
                    if (value < minValue) {
                        minValue = value;
                        minIndex = static_cast<int64_t>(reduceIdx);
                    }
                }
                yGm.SetValue(batchOutputOffset + innerIdx, minIndex);
            }
        }
    }

private:
    GlobalTensor<float> xGm;
    GlobalTensor<int64_t> yGm;
    uint32_t batchSize;
    uint32_t reduceSize;
    uint32_t innerSize;
    uint32_t blockIdx;
};

extern "C" __global__ __aicore__ void argmin_over_a_dimension_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelArgminOverADimension op;
    op.Init(
        x,
        y,
        tiling_data.batchSize,
        tiling_data.reduceSize,
        tiling_data.innerSize);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>
#include <vector>

at::Tensor argmin_over_a_dimension_impl_npu(const at::Tensor &x)
{
    TORCH_CHECK(x.dim() == 3, "argmin_over_a_dimension_custom expects a 3D tensor");
    std::vector<int64_t> outputShape = {x.size(0), x.size(2)};
    at::Tensor result = at::empty(outputShape, x.options().dtype(at::kLong));
    EXEC_NPU_CMD(aclnnArgminOverADimensionCustom, x, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("argmin_over_a_dimension_custom", &argmin_over_a_dimension_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("argmin_over_a_dimension_custom", &argmin_over_a_dimension_impl_npu, "argmin over dim=1");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim != 1:
            raise NotImplementedError("Only dim=1 is supported by this AscendC kernel.")
        return custom_ops_lib.argmin_over_a_dimension_custom(x)
'''
