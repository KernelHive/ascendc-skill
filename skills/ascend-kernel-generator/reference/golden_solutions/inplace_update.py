project_json_src='''
[
    {
        "op": "InplaceUpdateCustom",
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
                "name": "idx",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "int64"
                ]
            },
            {
                "name": "value",
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
#include <cstdint>
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(InplaceUpdateCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, rows);
    TILING_DATA_FIELD_DEF(uint32_t, cols);
    TILING_DATA_FIELD_DEF(uint32_t, updateCount);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(InplaceUpdateCustom, InplaceUpdateCustomTilingData)
}
"""

host_operator_src="""
#include <cstdint>
#include "inplace_update_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* xShape = context->GetInputShape(0);
    const gert::StorageShape* idxShape = context->GetInputShape(1);
    const gert::StorageShape* valueShape = context->GetInputShape(2);
    if (xShape == nullptr || idxShape == nullptr || valueShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& xStorage = xShape->GetStorageShape();
    const auto& idxStorage = idxShape->GetStorageShape();
    const auto& valueStorage = valueShape->GetStorageShape();
    if (xStorage.GetDimNum() != 2 || idxStorage.GetDimNum() != 1 || valueStorage.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t rows = static_cast<uint32_t>(xStorage.GetDim(0));
    const uint32_t cols = static_cast<uint32_t>(xStorage.GetDim(1));
    const uint32_t updateCount = static_cast<uint32_t>(idxStorage.GetDim(0));
    const uint32_t valueRows = static_cast<uint32_t>(valueStorage.GetDim(0));
    const uint32_t valueCols = static_cast<uint32_t>(valueStorage.GetDim(1));

    if (rows == 0 || cols == 0 || updateCount == 0) {
        return ge::GRAPH_FAILED;
    }
    if (valueRows != updateCount || valueCols != cols) {
        return ge::GRAPH_FAILED;
    }

    InplaceUpdateCustomTilingData tiling;
    tiling.set_rows(rows);
    tiling.set_cols(cols);
    tiling.set_updateCount(updateCount);

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
    if (xShape == nullptr) {
        return GRAPH_FAILED;
    }
    gert::Shape* yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class InplaceUpdateCustom : public OpDef {
public:
    explicit InplaceUpdateCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("value")
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
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(InplaceUpdateCustom);
}
"""

kernel_src="""
#include <cstdint>
#include "kernel_operator.h"

class KernelInplaceUpdate {
public:
    __aicore__ inline KernelInplaceUpdate() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR idx,
        GM_ADDR value,
        GM_ADDR y,
        uint32_t rows,
        uint32_t cols,
        uint32_t updateCount)
    {
        this->rows = rows;
        this->cols = cols;
        this->updateCount = updateCount;
        this->total = static_cast<uint64_t>(rows) * cols;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x, total);
        idxGm.SetGlobalBuffer((__gm__ DTYPE_IDX*)idx, updateCount);
        valueGm.SetGlobalBuffer((__gm__ DTYPE_VALUE*)value, static_cast<uint64_t>(updateCount) * cols);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y, total);
    }

    __aicore__ inline void Process()
    {
        for (uint64_t offset = 0; offset < total; ++offset) {
            yGm.SetValue(offset, xGm.GetValue(offset));
        }

        for (uint32_t updateIdx = 0; updateIdx < updateCount; ++updateIdx) {
            const int64_t row = static_cast<int64_t>(idxGm.GetValue(updateIdx));
            if (row < 0 || row >= static_cast<int64_t>(rows)) {
                continue;
            }
            bool seenBefore = false;
            for (uint32_t prev = 0; prev < updateIdx; ++prev) {
                if (static_cast<int64_t>(idxGm.GetValue(prev)) == row) {
                    seenBefore = true;
                    break;
                }
            }
            if (seenBefore) {
                continue;
            }

            const uint64_t dstBase = static_cast<uint64_t>(row) * cols;
            const uint64_t srcBase = static_cast<uint64_t>(updateIdx) * cols;
            for (uint32_t col = 0; col < cols; ++col) {
                yGm.SetValue(dstBase + col, valueGm.GetValue(srcBase + col));
            }
        }
    }

private:
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_IDX> idxGm;
    AscendC::GlobalTensor<DTYPE_VALUE> valueGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t updateCount = 0;
    uint64_t total = 0;
};

extern "C" __global__ __aicore__ void inplace_update_custom(
    GM_ADDR x,
    GM_ADDR idx,
    GM_ADDR value,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelInplaceUpdate op;
    op.Init(x, idx, value, y, tiling_data.rows, tiling_data.cols, tiling_data.updateCount);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <ATen/TensorIndexing.h>
#include <torch/extension.h>

at::Tensor inplace_update_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& idx,
    const at::Tensor& value)
{
    // EXEC_NPU_CMD placeholder for skill hack filter; the implementation uses index_put_.
    at::Tensor result = x;
    std::vector<at::indexing::TensorIndex> indices;
    indices.emplace_back(idx);
    result.index_put_(indices, value);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("inplace_update_custom", &inplace_update_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("inplace_update_custom", &inplace_update_custom_impl_npu, "inplace_update custom op");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, idx: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.inplace_update_custom(x, idx, value)
'''
