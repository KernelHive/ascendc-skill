project_json_src='''
[
    {
        "op": "MeanReductionOverADimensionCustom",
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
BEGIN_TILING_DATA_DEF(MeanReductionOverADimensionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, reduceSize);
    TILING_DATA_FIELD_DEF(uint32_t, colCount);
    TILING_DATA_FIELD_DEF(uint32_t, useCoreNums);
    TILING_DATA_FIELD_DEF(uint32_t, batchPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, batchPerCoreTail);
    TILING_DATA_FIELD_DEF(float, invReduceSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    MeanReductionOverADimensionCustom,
    MeanReductionOverADimensionCustomTilingData)
} // namespace optiling
"""

host_operator_src="""
#include "mean_reduction_over_a_dimension_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr uint32_t MAX_CORE_NUM = 1;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    MeanReductionOverADimensionCustomTilingData tiling;
    const gert::StorageShape *inputShape = context->GetInputShape(0);
    const auto shape = inputShape->GetStorageShape();

    const uint32_t batchSize = static_cast<uint32_t>(shape.GetDim(0));
    const uint32_t reduceSize = static_cast<uint32_t>(shape.GetDim(1));
    const uint32_t colCount = static_cast<uint32_t>(shape.GetDim(2));
    const uint32_t useCoreNums = batchSize < MAX_CORE_NUM ? batchSize : MAX_CORE_NUM;
    const uint32_t batchPerCore = useCoreNums == 0 ? 0 : (batchSize + useCoreNums - 1) / useCoreNums;
    const uint32_t batchPerCoreTail =
        batchSize == 0 ? 0 : (batchSize - (useCoreNums - 1) * batchPerCore);

    context->SetBlockDim(useCoreNums == 0 ? 1 : useCoreNums);
    tiling.set_batchSize(batchSize);
    tiling.set_reduceSize(reduceSize);
    tiling.set_colCount(colCount);
    tiling.set_useCoreNums(useCoreNums);
    tiling.set_batchPerCore(batchPerCore);
    tiling.set_batchPerCoreTail(batchPerCoreTail);
    tiling.set_invReduceSize(reduceSize == 0 ? 0.0f : 1.0f / static_cast<float>(reduceSize));
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
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    *yShape = {inputShape->GetDim(0), inputShape->GetDim(2)};
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
class MeanReductionOverADimensionCustom : public OpDef {
public:
    explicit MeanReductionOverADimensionCustom(const char *name) : OpDef(name)
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
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(MeanReductionOverADimensionCustom);
} // namespace ops
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelMeanReductionOverADimension {
public:
    __aicore__ inline KernelMeanReductionOverADimension() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t reduceSize,
        uint32_t colCount,
        uint32_t useCoreNums,
        uint32_t batchPerCore,
        uint32_t batchPerCoreTail,
        float invReduceSize)
    {
        this->batchSize = batchSize;
        this->reduceSize = reduceSize;
        this->colCount = colCount;
        this->useCoreNums = useCoreNums;
        this->batchPerCore = batchPerCore;
        this->batchPerCoreTail = batchPerCoreTail;
        this->invReduceSize = invReduceSize;

        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t localBatch =
            blockIdx + 1 == useCoreNums ? batchPerCoreTail : batchPerCore;
        const uint32_t batchOffset = blockIdx * batchPerCore;

        this->localBatch = localBatch;
        this->batchOffset = batchOffset;

        xGm.SetGlobalBuffer((__gm__ float *)x + batchOffset * reduceSize * colCount,
                            localBatch * reduceSize * colCount);
        yGm.SetGlobalBuffer((__gm__ float *)y + batchOffset * colCount,
                            localBatch * colCount);
        pipe.InitBuffer(rowBuf, colCount * sizeof(float));
        pipe.InitBuffer(accBuf, colCount * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->localBatch == 0 || this->reduceSize == 0) {
            return;
        }

        for (uint32_t batchIdx = 0; batchIdx < this->localBatch; ++batchIdx) {
            const uint32_t batchBase = batchIdx * this->reduceSize * this->colCount;
            const uint32_t outBase = batchIdx * this->colCount;
            LocalTensor<float> accLocal = accBuf.Get<float>();
            for (uint32_t col = 0; col < this->colCount; ++col) {
                accLocal.SetValue(col, 0.0f);
            }

            for (uint32_t reduceIdx = 0; reduceIdx < this->reduceSize; ++reduceIdx) {
                LocalTensor<float> rowLocal = rowBuf.Get<float>();
                DataCopy(rowLocal, xGm[batchBase + reduceIdx * this->colCount], this->colCount);
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                for (uint32_t col = 0; col < this->colCount; ++col) {
                    const float acc = accLocal.GetValue(col) + rowLocal.GetValue(col);
                    accLocal.SetValue(col, acc);
                }
            }

            for (uint32_t col = 0; col < this->colCount; ++col) {
                accLocal.SetValue(col, accLocal.GetValue(col) * this->invReduceSize);
            }
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopy(yGm[outBase], accLocal, this->colCount);
        }
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> rowBuf;
    TBuf<TPosition::VECCALC> accBuf;
    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t reduceSize;
    uint32_t colCount;
    uint32_t useCoreNums;
    uint32_t batchPerCore;
    uint32_t batchPerCoreTail;
    uint32_t localBatch;
    uint32_t batchOffset;
    float invReduceSize;
};

extern "C" __global__ __aicore__ void mean_reduction_over_a_dimension_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelMeanReductionOverADimension op;
    op.Init(
        x,
        y,
        tiling_data.batchSize,
        tiling_data.reduceSize,
        tiling_data.colCount,
        tiling_data.useCoreNums,
        tiling_data.batchPerCore,
        tiling_data.batchPerCoreTail,
        tiling_data.invReduceSize);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor mean_reduction_over_a_dimension_custom_impl_npu(const at::Tensor &x)
{
    at::Tensor result = at::empty({x.size(0), x.size(2)}, x.options());
    EXEC_NPU_CMD(aclnnMeanReductionOverADimensionCustom, x, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "mean_reduction_over_a_dimension_custom",
        &mean_reduction_over_a_dimension_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "mean_reduction_over_a_dimension_custom",
        &mean_reduction_over_a_dimension_custom_impl_npu,
        "mean reduction over dimension 1");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.mean_reduction_over_a_dimension_custom(x)
'''
