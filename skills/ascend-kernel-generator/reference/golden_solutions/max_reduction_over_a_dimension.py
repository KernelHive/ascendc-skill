project_json_src='''
[
    {
        "op": "MaxReductionOverADimensionCustom",
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
BEGIN_TILING_DATA_DEF(MaxReductionOverADimensionCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, reduceSize);
TILING_DATA_FIELD_DEF(uint32_t, colSize);
TILING_DATA_FIELD_DEF(uint32_t, colTileSize);
TILING_DATA_FIELD_DEF(uint32_t, colTileCount);
TILING_DATA_FIELD_DEF(uint32_t, reduceTmpSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaxReductionOverADimensionCustom, MaxReductionOverADimensionCustomTilingData)
}
"""

host_operator_src="""
#include "max_reduction_over_a_dimension_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace optiling {
constexpr uint32_t BLOCK_DIM = 1;
constexpr uint32_t COL_TILE_SIZE = 64;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    MaxReductionOverADimensionCustomTilingData tiling;
    const auto inputShape = context->GetInputShape(0)->GetOriginShape();
    if (inputShape.GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(inputShape.GetDim(0));
    const uint32_t reduceSize = static_cast<uint32_t>(inputShape.GetDim(1));
    const uint32_t colSize = static_cast<uint32_t>(inputShape.GetDim(2));
    if (batchSize == 0 || reduceSize == 0 || colSize == 0) {
        return ge::GRAPH_FAILED;
    }
    if ((colSize % COL_TILE_SIZE) != 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t maxTmpSize = 0;
    uint32_t minTmpSize = 0;
    ge::Shape reduceShape({static_cast<int64_t>(reduceSize), static_cast<int64_t>(COL_TILE_SIZE)});
    AscendC::GetReduceMaxMaxMinTmpSize(
        reduceShape,
        ge::DT_FLOAT,
        AscendC::ReducePattern::RA,
        true,
        false,
        maxTmpSize,
        minTmpSize);

    tiling.set_batchSize(batchSize);
    tiling.set_reduceSize(reduceSize);
    tiling.set_colSize(colSize);
    tiling.set_colTileSize(COL_TILE_SIZE);
    tiling.set_colTileCount(colSize / COL_TILE_SIZE);
    tiling.set_reduceTmpSize(minTmpSize);

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
class MaxReductionOverADimensionCustom : public OpDef {
public:
    explicit MaxReductionOverADimensionCustom(const char *name) : OpDef(name)
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

OP_ADD(MaxReductionOverADimensionCustom);
} // namespace ops
"""

kernel_src="""
#include "kernel_operator.h"

class KernelMaxReductionOverADimension {
public:
    __aicore__ inline KernelMaxReductionOverADimension() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t reduceSize,
        uint32_t colSize,
        uint32_t colTileSize,
        uint32_t colTileCount,
        uint32_t reduceTmpSize)
    {
        this->batchSize = batchSize;
        this->reduceSize = reduceSize;
        this->colSize = colSize;
        this->colTileSize = colTileSize;
        this->colTileCount = colTileCount;
        this->reduceTmpSize = reduceTmpSize;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * reduceSize * colSize);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * colSize);
        pipe.InitBuffer(inQueueX, 1, reduceSize * colTileSize * sizeof(float));
        pipe.InitBuffer(outQueueY, 1, colTileSize * sizeof(float));
        pipe.InitBuffer(tmpBuffer, reduceTmpSize);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
            for (uint32_t tileIdx = 0; tileIdx < colTileCount; ++tileIdx) {
                CopyIn(batchIdx, tileIdx);
                Compute();
                CopyOut(batchIdx, tileIdx);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t batchIdx, uint32_t tileIdx)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        const uint32_t colOffset = tileIdx * colTileSize;
        const uint32_t batchOffset = batchIdx * reduceSize * colSize;
        for (uint32_t rowIdx = 0; rowIdx < reduceSize; ++rowIdx) {
            AscendC::DataCopy(
                xLocal[rowIdx * colTileSize],
                xGm[batchOffset + rowIdx * colSize + colOffset],
                colTileSize);
        }
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::LocalTensor<uint8_t> tmpLocal = tmpBuffer.Get<uint8_t>();
        uint32_t srcShape[2] = {reduceSize, colTileSize};
        AscendC::ReduceMax<float, AscendC::Pattern::Reduce::RA>(yLocal, xLocal, tmpLocal, srcShape, true);
        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t batchIdx, uint32_t tileIdx)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        const uint32_t outOffset = batchIdx * colSize + tileIdx * colTileSize;
        AscendC::DataCopy(yGm[outOffset], yLocal, colTileSize);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuffer;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t reduceSize;
    uint32_t colSize;
    uint32_t colTileSize;
    uint32_t colTileCount;
    uint32_t reduceTmpSize;
};

extern "C" __global__ __aicore__ void max_reduction_over_a_dimension_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelMaxReductionOverADimension op;
    op.Init(
        x,
        y,
        tiling_data.batchSize,
        tiling_data.reduceSize,
        tiling_data.colSize,
        tiling_data.colTileSize,
        tiling_data.colTileCount,
        tiling_data.reduceTmpSize);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor max_reduction_over_a_dimension_impl_npu(const at::Tensor &x)
{
    auto outputShape = std::vector<int64_t>{x.size(0), x.size(2)};
    at::Tensor result = at::empty(outputShape, x.options());
    EXEC_NPU_CMD(aclnnMaxReductionOverADimensionCustom, x, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("max_reduction_over_a_dimension_custom", &max_reduction_over_a_dimension_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_reduction_over_a_dimension_custom", &max_reduction_over_a_dimension_impl_npu, "max reduce over dim=1");
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
        return custom_ops_lib.max_reduction_over_a_dimension_custom(x)
'''
