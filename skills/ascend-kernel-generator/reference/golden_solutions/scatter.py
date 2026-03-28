project_json_src='''
[
    {
        "op": "ScatterCustom",
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
                "name": "updates",
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
BEGIN_TILING_DATA_DEF(ScatterCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, rows);
    TILING_DATA_FIELD_DEF(uint32_t, cols);
    TILING_DATA_FIELD_DEF(uint32_t, indexCols);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterCustom, ScatterCustomTilingData)
}
"""

host_operator_src="""
#include <cstdint>
#include "scatter_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* xShape = context->GetInputShape(0);
    const gert::StorageShape* indexShape = context->GetInputShape(1);
    const gert::StorageShape* updatesShape = context->GetInputShape(2);
    if (xShape == nullptr || indexShape == nullptr || updatesShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& xStorage = xShape->GetStorageShape();
    const auto& indexStorage = indexShape->GetStorageShape();
    const auto& updatesStorage = updatesShape->GetStorageShape();
    if (xStorage.GetDimNum() != 2 || indexStorage.GetDimNum() != 2 || updatesStorage.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t rows = static_cast<uint32_t>(xStorage.GetDim(0));
    const uint32_t cols = static_cast<uint32_t>(xStorage.GetDim(1));
    const uint32_t indexRows = static_cast<uint32_t>(indexStorage.GetDim(0));
    const uint32_t indexCols = static_cast<uint32_t>(indexStorage.GetDim(1));
    const uint32_t updatesRows = static_cast<uint32_t>(updatesStorage.GetDim(0));
    const uint32_t updatesCols = static_cast<uint32_t>(updatesStorage.GetDim(1));
    if (rows == 0 || cols == 0 || indexRows != rows || updatesRows != rows || updatesCols != indexCols) {
        return ge::GRAPH_FAILED;
    }

    ScatterCustomTilingData tiling;
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
    gert::Shape* outputShape = context->GetOutputShape(0);
    *outputShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ScatterCustom : public OpDef {
public:
    explicit ScatterCustom(const char* name) : OpDef(name)
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
        this->Input("updates")
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

OP_ADD(ScatterCustom);
}
"""

kernel_src="""
#include <cstdint>
#include "kernel_operator.h"

class KernelScatter {
public:
    __aicore__ inline KernelScatter() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR index,
        GM_ADDR updates,
        GM_ADDR y,
        uint32_t rows,
        uint32_t cols,
        uint32_t indexCols)
    {
        this->rows = rows;
        this->cols = cols;
        this->indexCols = indexCols;
        this->xElemCount = static_cast<uint64_t>(rows) * cols;
        this->updatesElemCount = static_cast<uint64_t>(rows) * indexCols;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x, xElemCount);
        indexGm.SetGlobalBuffer((__gm__ DTYPE_INDEX*)index, updatesElemCount);
        updatesGm.SetGlobalBuffer((__gm__ DTYPE_UPDATES*)updates, updatesElemCount);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y, xElemCount);
        pipe.InitBuffer(rowBuf, cols * sizeof(DTYPE_Y));
        pipe.InitBuffer(indexBuf, indexCols * sizeof(DTYPE_INDEX));
        pipe.InitBuffer(updatesBuf, indexCols * sizeof(DTYPE_UPDATES));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t row = 0; row < rows; ++row) {
            if (row > 0) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            const uint64_t rowBase = static_cast<uint64_t>(row) * cols;
            const uint64_t updateBase = static_cast<uint64_t>(row) * indexCols;
            AscendC::LocalTensor<DTYPE_Y> rowLocal = rowBuf.Get<DTYPE_Y>();
            AscendC::LocalTensor<DTYPE_INDEX> indexLocal = indexBuf.Get<DTYPE_INDEX>();
            AscendC::LocalTensor<DTYPE_UPDATES> updatesLocal = updatesBuf.Get<DTYPE_UPDATES>();
            AscendC::DataCopy(rowLocal, xGm[rowBase], cols);
            AscendC::DataCopy(indexLocal, indexGm[updateBase], indexCols);
            AscendC::DataCopy(updatesLocal, updatesGm[updateBase], indexCols);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            for (uint32_t col = 0; col < indexCols; ++col) {
                const int64_t dstCol = static_cast<int64_t>(indexLocal.GetValue(col));
                if (dstCol < 0 || dstCol >= static_cast<int64_t>(cols)) {
                    continue;
                }
                rowLocal.SetValue(static_cast<uint32_t>(dstCol), static_cast<DTYPE_Y>(updatesLocal.GetValue(col)));
            }
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::DataCopy(yGm[rowBase], rowLocal, cols);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> updatesBuf;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_INDEX> indexGm;
    AscendC::GlobalTensor<DTYPE_UPDATES> updatesGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t indexCols = 0;
    uint64_t xElemCount = 0;
    uint64_t updatesElemCount = 0;
};

extern "C" __global__ __aicore__ void scatter_custom(
    GM_ADDR x,
    GM_ADDR index,
    GM_ADDR updates,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelScatter op;
    op.Init(x, index, updates, y, tiling_data.rows, tiling_data.cols, tiling_data.indexCols);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor scatter_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& index,
    const at::Tensor& updates)
{
    at::Tensor result = at::empty_like(x);
    EXEC_NPU_CMD(aclnnScatterCustom, x, index, updates, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("scatter_custom", &scatter_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scatter_custom", &scatter_custom_impl_npu, "scatter custom op");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, index, updates):
        return custom_ops_lib.scatter_custom(x, index, updates)
'''
