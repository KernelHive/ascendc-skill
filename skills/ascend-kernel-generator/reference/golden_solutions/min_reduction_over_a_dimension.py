project_json_src='''
[
    {
        "op": "MinReductionOverADimensionCustom",
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
BEGIN_TILING_DATA_DEF(MinReductionOverADimensionCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, reduceSize);
TILING_DATA_FIELD_DEF(uint32_t, innerSize);
TILING_DATA_FIELD_DEF(uint32_t, tileInner);
TILING_DATA_FIELD_DEF(uint32_t, tileCount);
TILING_DATA_FIELD_DEF(uint32_t, lastTileInner);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MinReductionOverADimensionCustom, MinReductionOverADimensionCustomTilingData)
}
"""

host_operator_src="""
#include "min_reduction_over_a_dimension_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr uint32_t TILE_INNER = 16;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    MinReductionOverADimensionCustomTilingData tiling;
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
    const uint32_t tileCount = (innerSize + TILE_INNER - 1U) / TILE_INNER;
    const uint32_t lastTileInner = innerSize == 0 ? 0 : innerSize - (tileCount - 1U) * TILE_INNER;

    context->SetBlockDim(batchSize);
    tiling.set_batchSize(batchSize);
    tiling.set_reduceSize(reduceSize);
    tiling.set_innerSize(innerSize);
    tiling.set_tileInner(TILE_INNER);
    tiling.set_tileCount(tileCount);
    tiling.set_lastTileInner(lastTileInner);

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
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class MinReductionOverADimensionCustom : public OpDef {
public:
    explicit MinReductionOverADimensionCustom(const char *name) : OpDef(name)
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

OP_ADD(MinReductionOverADimensionCustom);
} // namespace ops
"""

kernel_src="""
#include "kernel_operator.h"
#include <cfloat>

class KernelMinReductionOverADimension {
public:
    __aicore__ inline KernelMinReductionOverADimension() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t reduceSize,
        uint32_t innerSize,
        uint32_t tileInner,
        uint32_t tileCount,
        uint32_t lastTileInner)
    {
        this->batchSize = batchSize;
        this->reduceSize = reduceSize;
        this->innerSize = innerSize;
        this->tileInner = tileInner;
        this->tileCount = tileCount;
        this->lastTileInner = lastTileInner;
        this->batchIdx = AscendC::GetBlockIdx();
        this->isValid = this->batchIdx < this->batchSize;
        if (!this->isValid) {
            return;
        }

        const uint32_t batchInputOffset = this->batchIdx * this->reduceSize * this->innerSize;
        const uint32_t batchOutputOffset = this->batchIdx * this->innerSize;
        xGm.SetGlobalBuffer((__gm__ float *)x + batchInputOffset, this->reduceSize * this->innerSize);
        yGm.SetGlobalBuffer((__gm__ float *)y + batchOutputOffset, this->innerSize);

        pipe.InitBuffer(inQueueX, 1, this->reduceSize * this->tileInner * sizeof(float));
        pipe.InitBuffer(outQueueY, 1, this->tileInner * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (!this->isValid) {
            return;
        }

        for (uint32_t tileIdx = 0; tileIdx < this->tileCount; ++tileIdx) {
            const uint32_t currentInner = tileIdx + 1U == this->tileCount ? this->lastTileInner : this->tileInner;
            CopyIn(tileIdx, currentInner);
            Compute();
            CopyOut(tileIdx, currentInner);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t tileIdx, uint32_t currentInner)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::Duplicate<float>(xLocal, FLT_MAX, this->reduceSize * this->tileInner);
        for (uint32_t row = 0; row < this->reduceSize; ++row) {
            const uint32_t rowOffset = row * this->tileInner;
            const uint32_t gmOffset = row * this->innerSize + tileIdx * this->tileInner;
            AscendC::DataCopy(xLocal[rowOffset], xGm[gmOffset], currentInner);
        }
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        uint32_t shape[2] = {this->reduceSize, this->tileInner};
        AscendC::ReduceMin<float, AscendC::Pattern::Reduce::RA, true>(yLocal, xLocal, shape, true);
        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx, uint32_t currentInner)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[tileIdx * this->tileInner], yLocal, currentInner);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t reduceSize;
    uint32_t innerSize;
    uint32_t tileInner;
    uint32_t tileCount;
    uint32_t lastTileInner;
    uint32_t batchIdx;
    bool isValid;
};

extern "C" __global__ __aicore__ void min_reduction_over_a_dimension_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelMinReductionOverADimension op;
    op.Init(
        x,
        y,
        tiling_data.batchSize,
        tiling_data.reduceSize,
        tiling_data.innerSize,
        tiling_data.tileInner,
        tiling_data.tileCount,
        tiling_data.lastTileInner);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>
#include <vector>

at::Tensor min_reduction_over_a_dimension_impl_npu(const at::Tensor &x)
{
    TORCH_CHECK(x.dim() == 3, "min_reduction_over_a_dimension_custom expects a 3D tensor");
    std::vector<int64_t> outputShape = {x.size(0), x.size(2)};
    at::Tensor result = at::empty(outputShape, x.options());
    EXEC_NPU_CMD(aclnnMinReductionOverADimensionCustom, x, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("min_reduction_over_a_dimension_custom", &min_reduction_over_a_dimension_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("min_reduction_over_a_dimension_custom", &min_reduction_over_a_dimension_impl_npu, "reduce min over dim=1");
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
        return custom_ops_lib.min_reduction_over_a_dimension_custom(x)
'''
