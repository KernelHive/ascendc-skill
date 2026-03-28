project_json_src='''
[
    {
        "op": "SumReductionOverADimensionCustom",
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
                    "float"
                ]
            }
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SumReductionOverADimensionCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, reduceSize);
TILING_DATA_FIELD_DEF(uint32_t, colCount);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SumReductionOverADimensionCustom, SumReductionOverADimensionCustomTilingData)
}
"""

host_operator_src="""
#include "sum_reduction_over_a_dimension_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr uint32_t BLOCK_DIM = 1;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    SumReductionOverADimensionCustomTilingData tiling;
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
    const uint32_t colCount = static_cast<uint32_t>(shape.GetDim(2));
    if (batchSize == 0 || reduceSize == 0 || colCount == 0) {
        return ge::GRAPH_FAILED;
    }

    tiling.set_batchSize(batchSize);
    tiling.set_reduceSize(reduceSize);
    tiling.set_colCount(colCount);

    context->SetBlockDim(BLOCK_DIM);
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
    const gert::Shape *inputShape = context->GetInputShape(0);
    if (inputShape->GetDimNum() != 3) {
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
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class SumReductionOverADimensionCustom : public OpDef {
public:
    explicit SumReductionOverADimensionCustom(const char *name) : OpDef(name)
    {
        this->Input("x")
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

OP_ADD(SumReductionOverADimensionCustom);
} // namespace ops
"""

kernel_src="""
#include "kernel_operator.h"

class KernelSumReductionOverADimension {
public:
    __aicore__ inline KernelSumReductionOverADimension() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t reduceSize,
        uint32_t colCount)
    {
        this->batchSize = batchSize;
        this->reduceSize = reduceSize;
        this->colCount = colCount;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * reduceSize * colCount);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * colCount);
        pipe.InitBuffer(rowBuf, colCount * sizeof(float));
        pipe.InitBuffer(accBuf, colCount * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
            const uint32_t batchBase = batchIdx * reduceSize * colCount;
            const uint32_t outBase = batchIdx * colCount;
            AscendC::LocalTensor<float> accLocal = accBuf.Get<float>();
            for (uint32_t col = 0; col < colCount; ++col) {
                accLocal.SetValue(col, 0.0f);
            }

            for (uint32_t reduceIdx = 0; reduceIdx < reduceSize; ++reduceIdx) {
                AscendC::LocalTensor<float> rowLocal = rowBuf.Get<float>();
                AscendC::DataCopy(rowLocal, xGm[batchBase + reduceIdx * colCount], colCount);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
                for (uint32_t col = 0; col < colCount; ++col) {
                    const float sum = accLocal.GetValue(col) + rowLocal.GetValue(col);
                    accLocal.SetValue(col, sum);
                }
            }

            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::DataCopy(yGm[outBase], accLocal, colCount);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t reduceSize;
    uint32_t colCount;
};

extern "C" __global__ __aicore__ void sum_reduction_over_a_dimension_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelSumReductionOverADimension op;
    op.Init(
        x,
        y,
        tiling_data.batchSize,
        tiling_data.reduceSize,
        tiling_data.colCount);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor sum_reduction_over_a_dimension_impl_npu(const at::Tensor &x)
{
    auto outputShape = std::vector<int64_t>{x.size(0), x.size(2)};
    at::Tensor result = at::empty(outputShape, x.options());
    EXEC_NPU_CMD(aclnnSumReductionOverADimensionCustom, x, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("sum_reduction_over_a_dimension_custom", &sum_reduction_over_a_dimension_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_reduction_over_a_dimension_custom", &sum_reduction_over_a_dimension_impl_npu, "sum reduce over dim=1 with keepdim");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim != 1:
            raise ValueError("This AscendC implementation currently supports dim=1 only.")
        return custom_ops_lib.sum_reduction_over_a_dimension_custom(x).unsqueeze(1)
'''
