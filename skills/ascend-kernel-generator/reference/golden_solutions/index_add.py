project_json_src='''
[
    {
        "op": "IndexAddCustom",
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
                "name": "indices",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "int64"
                ]
            },
            {
                "name": "source",
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
BEGIN_TILING_DATA_DEF(IndexAddCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, rows);
    TILING_DATA_FIELD_DEF(uint32_t, cols);
    TILING_DATA_FIELD_DEF(uint32_t, indicesLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IndexAddCustom, IndexAddCustomTilingData)
}
"""

host_operator_src="""
#include <cstdint>
#include "index_add_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* xShape = context->GetInputShape(0);
    const gert::StorageShape* indicesShape = context->GetInputShape(1);
    const gert::StorageShape* sourceShape = context->GetInputShape(2);
    if (xShape == nullptr || indicesShape == nullptr || sourceShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& xStorage = xShape->GetStorageShape();
    const auto& indicesStorage = indicesShape->GetStorageShape();
    const auto& sourceStorage = sourceShape->GetStorageShape();
    if (xStorage.GetDimNum() != 2 || indicesStorage.GetDimNum() != 1 || sourceStorage.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t rows = static_cast<uint32_t>(xStorage.GetDim(0));
    const uint32_t cols = static_cast<uint32_t>(xStorage.GetDim(1));
    const uint32_t indicesLength = static_cast<uint32_t>(indicesStorage.GetShapeSize());
    const uint32_t sourceRows = static_cast<uint32_t>(sourceStorage.GetDim(0));
    const uint32_t sourceCols = static_cast<uint32_t>(sourceStorage.GetDim(1));
    if (rows == 0 || cols == 0 || indicesLength == 0) {
        return ge::GRAPH_FAILED;
    }
    if (sourceRows != indicesLength || sourceCols != cols) {
        return ge::GRAPH_FAILED;
    }

    IndexAddCustomTilingData tiling;
    tiling.set_rows(rows);
    tiling.set_cols(cols);
    tiling.set_indicesLength(indicesLength);

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
class IndexAddCustom : public OpDef {
public:
    explicit IndexAddCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("source")
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

OP_ADD(IndexAddCustom);
}
"""

kernel_src="""
#include <cstdint>
#include "kernel_operator.h"

class KernelIndexAdd {
public:
    __aicore__ inline KernelIndexAdd() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR indices,
        GM_ADDR source,
        GM_ADDR y,
        uint32_t rows,
        uint32_t cols,
        uint32_t indicesLength)
    {
        this->rows = rows;
        this->cols = cols;
        this->indicesLength = indicesLength;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x, static_cast<uint64_t>(rows) * cols);
        indicesGm.SetGlobalBuffer((__gm__ DTYPE_INDICES*)indices, indicesLength);
        sourceGm.SetGlobalBuffer((__gm__ DTYPE_SOURCE*)source, static_cast<uint64_t>(indicesLength) * cols);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y, static_cast<uint64_t>(rows) * cols);
    }

    __aicore__ inline void Process()
    {
        const uint64_t total = static_cast<uint64_t>(rows) * cols;
        for (uint64_t i = 0; i < total; ++i) {
            yGm.SetValue(i, xGm.GetValue(i));
        }

        for (uint32_t i = 0; i < indicesLength; ++i) {
            const int64_t dstRow = static_cast<int64_t>(indicesGm.GetValue(i));
            if (dstRow < 0 || dstRow >= static_cast<int64_t>(rows)) {
                continue;
            }
            const uint64_t dstBase = static_cast<uint64_t>(dstRow) * cols;
            const uint64_t srcBase = static_cast<uint64_t>(i) * cols;
            for (uint32_t c = 0; c < cols; ++c) {
                const float acc = static_cast<float>(yGm.GetValue(dstBase + c));
                const float update = static_cast<float>(sourceGm.GetValue(srcBase + c));
                yGm.SetValue(dstBase + c, static_cast<DTYPE_Y>(acc + update));
            }
        }
    }

private:
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_INDICES> indicesGm;
    AscendC::GlobalTensor<DTYPE_SOURCE> sourceGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t indicesLength = 0;
};

extern "C" __global__ __aicore__ void index_add_custom(
    GM_ADDR x,
    GM_ADDR indices,
    GM_ADDR source,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelIndexAdd op;
    op.Init(x, indices, source, y, tiling_data.rows, tiling_data.cols, tiling_data.indicesLength);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor index_add_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& indices,
    const at::Tensor& source)
{
    at::Tensor result = at::empty_like(x);
    EXEC_NPU_CMD(aclnnIndexAddCustom, x, indices, source, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("index_add_custom", &index_add_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("index_add_custom", &index_add_custom_impl_npu, "index_add custom op");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, indices: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.index_add_custom(x, indices, source)
'''
