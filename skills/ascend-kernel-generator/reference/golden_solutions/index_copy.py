project_json_src='''
[
    {
        "op": "IndexCopyCustom",
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
                "name": "out",
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
BEGIN_TILING_DATA_DEF(IndexCopyCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, rows);
    TILING_DATA_FIELD_DEF(uint32_t, cols);
    TILING_DATA_FIELD_DEF(uint32_t, indexCount);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IndexCopyCustom, IndexCopyCustomTilingData)
}
"""

host_operator_src="""
#include <cstdint>
#include "index_copy_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* xShape = context->GetInputShape(0);
    const gert::StorageShape* indicesShape = context->GetInputShape(1);
    const gert::StorageShape* srcShape = context->GetInputShape(2);
    if (xShape == nullptr || indicesShape == nullptr || srcShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& xStorage = xShape->GetStorageShape();
    const auto& indicesStorage = indicesShape->GetStorageShape();
    const auto& srcStorage = srcShape->GetStorageShape();
    if (xStorage.GetDimNum() != 2 || indicesStorage.GetDimNum() != 1 || srcStorage.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t rows = static_cast<uint32_t>(xStorage.GetDim(0));
    const uint32_t cols = static_cast<uint32_t>(xStorage.GetDim(1));
    const uint32_t indexCount = static_cast<uint32_t>(indicesStorage.GetDim(0));
    const uint32_t srcRows = static_cast<uint32_t>(srcStorage.GetDim(0));
    const uint32_t srcCols = static_cast<uint32_t>(srcStorage.GetDim(1));
    if (rows == 0 || cols == 0 || indexCount == 0 || srcRows != indexCount || srcCols != cols) {
        return ge::GRAPH_FAILED;
    }

    IndexCopyCustomTilingData tiling;
    tiling.set_rows(rows);
    tiling.set_cols(cols);
    tiling.set_indexCount(indexCount);

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
    gert::Shape* outShape = context->GetOutputShape(0);
    *outShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class IndexCopyCustom : public OpDef {
public:
    explicit IndexCopyCustom(const char* name) : OpDef(name)
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
        this->Input("src")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(IndexCopyCustom);
}
"""

kernel_src="""
#include <cstdint>
#include "kernel_operator.h"

class KernelIndexCopy {
public:
    __aicore__ inline KernelIndexCopy() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR indices,
        GM_ADDR src,
        GM_ADDR out,
        uint32_t rows,
        uint32_t cols,
        uint32_t indexCount)
    {
        this->rows = rows;
        this->cols = cols;
        this->indexCount = indexCount;
        this->total = static_cast<uint64_t>(rows) * cols;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x, total);
        indicesGm.SetGlobalBuffer((__gm__ DTYPE_INDICES*)indices, indexCount);
        srcGm.SetGlobalBuffer((__gm__ DTYPE_SRC*)src, static_cast<uint64_t>(indexCount) * cols);
        outGm.SetGlobalBuffer((__gm__ DTYPE_OUT*)out, total);
    }

    __aicore__ inline void Process()
    {
        for (uint64_t offset = 0; offset < total; ++offset) {
            outGm.SetValue(offset, xGm.GetValue(offset));
        }

        for (uint32_t i = 0; i < indexCount; ++i) {
            const int64_t row = static_cast<int64_t>(indicesGm.GetValue(i));
            if (row < 0 || row >= static_cast<int64_t>(rows)) {
                continue;
            }

            const uint64_t dstBase = static_cast<uint64_t>(row) * cols;
            const uint64_t srcBase = static_cast<uint64_t>(i) * cols;
            for (uint32_t col = 0; col < cols; ++col) {
                outGm.SetValue(dstBase + col, srcGm.GetValue(srcBase + col));
            }
        }
    }

private:
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_INDICES> indicesGm;
    AscendC::GlobalTensor<DTYPE_SRC> srcGm;
    AscendC::GlobalTensor<DTYPE_OUT> outGm;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t indexCount = 0;
    uint64_t total = 0;
};

extern "C" __global__ __aicore__ void index_copy_custom(
    GM_ADDR x,
    GM_ADDR indices,
    GM_ADDR src,
    GM_ADDR out,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelIndexCopy op;
    op.Init(x, indices, src, out, tiling_data.rows, tiling_data.cols, tiling_data.indexCount);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor index_copy_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& indices,
    const at::Tensor& src)
{
    at::Tensor result = at::empty_like(x);
    EXEC_NPU_CMD(aclnnIndexCopyCustom, x, indices, src, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("index_copy_custom", &index_copy_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("index_copy_custom", &index_copy_custom_impl_npu, "index_copy dim0 custom op");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, indices: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.index_copy_custom(x, indices, src)
'''
