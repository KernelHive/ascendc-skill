project_json_src='''
[
    {
        "op": "ScatterAddCustom",
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
                "name": "index",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "int64"
                ]
            },
            {
                "name": "src",
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
BEGIN_TILING_DATA_DEF(ScatterAddCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, rows);
    TILING_DATA_FIELD_DEF(uint32_t, cols);
    TILING_DATA_FIELD_DEF(uint32_t, indexCols);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterAddCustom, ScatterAddCustomTilingData)
}
"""

host_operator_src="""
#include <cstdint>
#include "scatter_add_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* xShape = context->GetInputShape(0);
    const gert::StorageShape* indexShape = context->GetInputShape(1);
    const gert::StorageShape* srcShape = context->GetInputShape(2);
    if (xShape == nullptr || indexShape == nullptr || srcShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& xStorage = xShape->GetStorageShape();
    const auto& indexStorage = indexShape->GetStorageShape();
    const auto& srcStorage = srcShape->GetStorageShape();
    if (xStorage.GetDimNum() != 2 || indexStorage.GetDimNum() != 2 || srcStorage.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t rows = static_cast<uint32_t>(xStorage.GetDim(0));
    const uint32_t cols = static_cast<uint32_t>(xStorage.GetDim(1));
    const uint32_t indexRows = static_cast<uint32_t>(indexStorage.GetDim(0));
    const uint32_t indexCols = static_cast<uint32_t>(indexStorage.GetDim(1));
    const uint32_t srcRows = static_cast<uint32_t>(srcStorage.GetDim(0));
    const uint32_t srcCols = static_cast<uint32_t>(srcStorage.GetDim(1));
    if (rows == 0 || cols == 0 || indexCols == 0) {
        return ge::GRAPH_FAILED;
    }
    if (indexRows != rows || srcRows != rows || srcCols != indexCols) {
        return ge::GRAPH_FAILED;
    }

    ScatterAddCustomTilingData tiling;
    tiling.set_rows(rows);
    tiling.set_cols(cols);
    tiling.set_indexCols(indexCols);

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
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ScatterAddCustom : public OpDef {
public:
    explicit ScatterAddCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("src")
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

OP_ADD(ScatterAddCustom);
}
"""

kernel_src="""
#include <cstdint>
#include "kernel_operator.h"

class KernelScatterAdd {
public:
    __aicore__ inline KernelScatterAdd() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR index,
        GM_ADDR src,
        GM_ADDR y,
        uint32_t rows,
        uint32_t cols,
        uint32_t indexCols)
    {
        this->rows = rows;
        this->cols = cols;
        this->indexCols = indexCols;
        this->total = static_cast<uint64_t>(rows) * cols;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x, total);
        indexGm.SetGlobalBuffer((__gm__ DTYPE_INDEX*)index, static_cast<uint64_t>(rows) * indexCols);
        srcGm.SetGlobalBuffer((__gm__ DTYPE_SRC*)src, static_cast<uint64_t>(rows) * indexCols);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y, total);
    }

    __aicore__ inline void Process()
    {
        for (uint64_t offset = 0; offset < total; ++offset) {
            yGm.SetValue(offset, xGm.GetValue(offset));
        }

        for (uint32_t row = 0; row < rows; ++row) {
            const uint64_t rowBase = static_cast<uint64_t>(row) * cols;
            const uint64_t updateBase = static_cast<uint64_t>(row) * indexCols;
            for (uint32_t col = 0; col < indexCols; ++col) {
                const int64_t dstCol = static_cast<int64_t>(indexGm.GetValue(updateBase + col));
                if (dstCol < 0 || dstCol >= static_cast<int64_t>(cols)) {
                    continue;
                }
                const uint64_t dstOffset = rowBase + static_cast<uint64_t>(dstCol);
                const float acc = static_cast<float>(yGm.GetValue(dstOffset));
                const float update = static_cast<float>(srcGm.GetValue(updateBase + col));
                yGm.SetValue(dstOffset, static_cast<DTYPE_Y>(acc + update));
            }
        }
    }

private:
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_INDEX> indexGm;
    AscendC::GlobalTensor<DTYPE_SRC> srcGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t indexCols = 0;
    uint64_t total = 0;
};

extern "C" __global__ __aicore__ void scatter_add_custom(
    GM_ADDR x,
    GM_ADDR index,
    GM_ADDR src,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelScatterAdd op;
    op.Init(x, index, src, y, tiling_data.rows, tiling_data.cols, tiling_data.indexCols);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor scatter_add_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& index,
    const at::Tensor& src)
{
    at::Tensor result = at::empty_like(x);
    EXEC_NPU_CMD(aclnnScatterAddCustom, x, index, src, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("scatter_add_custom", &scatter_add_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scatter_add_custom", &scatter_add_custom_impl_npu, "scatter_add custom op");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.scatter_add_custom(x, index, src)
'''
