project_json_src='''
[
    {
        "op": "TakeAlongDimCustom",
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
                    "int32"
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
BEGIN_TILING_DATA_DEF(TakeAlongDimCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, rows);
    TILING_DATA_FIELD_DEF(uint32_t, inputCols);
    TILING_DATA_FIELD_DEF(uint32_t, outputCols);
    TILING_DATA_FIELD_DEF(uint32_t, total);
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TakeAlongDimCustom, TakeAlongDimCustomTilingData)
}
"""

host_operator_src="""
#include <cstdint>
#include "take_along_dim_custom_tiling.h"
#include "register/op_def_registry.h"
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* xShape = context->GetInputShape(0);
    const gert::StorageShape* indicesShape = context->GetInputShape(1);
    if (xShape == nullptr || indicesShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& xStorage = xShape->GetStorageShape();
    const auto& indicesStorage = indicesShape->GetStorageShape();
    if (xStorage.GetDimNum() != 2 || indicesStorage.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t rows = static_cast<uint32_t>(xStorage.GetDim(0));
    const uint32_t inputCols = static_cast<uint32_t>(xStorage.GetDim(1));
    const uint32_t indexRows = static_cast<uint32_t>(indicesStorage.GetDim(0));
    const uint32_t outputCols = static_cast<uint32_t>(indicesStorage.GetDim(1));
    if (rows == 0 || inputCols == 0 || outputCols == 0 || indexRows != rows) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t total64 = static_cast<uint64_t>(rows) * outputCols;
    if (total64 == 0 || total64 > static_cast<uint64_t>(UINT32_MAX)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t blockDim = 1;

    TakeAlongDimCustomTilingData tiling;
    tiling.set_rows(rows);
    tiling.set_inputCols(inputCols);
    tiling.set_outputCols(outputCols);
    tiling.set_total(static_cast<uint32_t>(total64));
    tiling.set_blockDim(blockDim);

    context->SetBlockDim(blockDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    workspaceSize[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* indicesShape = context->GetInputShape(1);
    if (indicesShape == nullptr || indicesShape->GetDimNum() != 2) {
        return GRAPH_FAILED;
    }

    gert::Shape* yShape = context->GetOutputShape(0);
    *yShape = *indicesShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class TakeAlongDimCustom : public OpDef {
public:
    explicit TakeAlongDimCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
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

OP_ADD(TakeAlongDimCustom);
}
"""

kernel_src="""
#include <cstdint>
#include "kernel_operator.h"

class KernelTakeAlongDim {
public:
    __aicore__ inline KernelTakeAlongDim() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR indices,
        GM_ADDR y,
        uint32_t rows,
        uint32_t inputCols,
        uint32_t outputCols,
        uint32_t total,
        uint32_t blockDim)
    {
        this->rows = rows;
        this->inputCols = inputCols;
        this->outputCols = outputCols;
        this->total = total;
        this->blockDim = blockDim;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x, static_cast<uint64_t>(rows) * inputCols);
        indicesGm.SetGlobalBuffer((__gm__ DTYPE_INDICES*)indices, total);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y, total);
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        for (uint32_t linear = blockIdx; linear < total; linear += blockDim) {
            const uint32_t row = linear / outputCols;
            const int32_t gatherIndex = static_cast<int32_t>(indicesGm.GetValue(linear));
            if (gatherIndex < 0 || gatherIndex >= static_cast<int32_t>(inputCols)) {
                yGm.SetValue(linear, static_cast<DTYPE_Y>(0));
                continue;
            }

            const uint64_t srcOffset = static_cast<uint64_t>(row) * inputCols + static_cast<uint64_t>(gatherIndex);
            yGm.SetValue(linear, xGm.GetValue(srcOffset));
        }
    }

private:
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_INDICES> indicesGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t rows = 0;
    uint32_t inputCols = 0;
    uint32_t outputCols = 0;
    uint32_t total = 0;
    uint32_t blockDim = 1;
};

extern "C" __global__ __aicore__ void take_along_dim_custom(
    GM_ADDR x,
    GM_ADDR indices,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelTakeAlongDim op;
    op.Init(
        x,
        indices,
        y,
        tiling_data.rows,
        tiling_data.inputCols,
        tiling_data.outputCols,
        tiling_data.total,
        tiling_data.blockDim);
    op.Process();
}
"""

python_bind_src="""
#include <vector>
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor take_along_dim_custom_impl_npu(const at::Tensor& x, const at::Tensor& indices)
{
    TORCH_CHECK(x.dim() == 2, "take_along_dim_custom expects x to be 2D");
    TORCH_CHECK(indices.dim() == 2, "take_along_dim_custom expects indices to be 2D");
    TORCH_CHECK(x.size(0) == indices.size(0), "take_along_dim_custom expects matching row count");
    TORCH_CHECK(indices.scalar_type() == at::kInt, "take_along_dim_custom expects int32 indices");

    std::vector<int64_t> outputShape(indices.sizes().begin(), indices.sizes().end());
    at::Tensor result = at::empty(outputShape, x.options());
    EXEC_NPU_CMD(aclnnTakeAlongDimCustom, x, indices, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("take_along_dim_custom", &take_along_dim_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("take_along_dim_custom", &take_along_dim_custom_impl_npu, "take_along_dim custom op");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.take_along_dim_custom(x, idx.to(torch.int32))
'''
