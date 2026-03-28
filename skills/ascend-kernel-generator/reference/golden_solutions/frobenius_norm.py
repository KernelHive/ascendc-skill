project_json_src='''
[
    {
        "op": "FrobeniusNormCustom",
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
BEGIN_TILING_DATA_DEF(FrobeniusNormCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalLength);
TILING_DATA_FIELD_DEF(uint32_t, tileNum);
TILING_DATA_FIELD_DEF(uint32_t, tileLength);
TILING_DATA_FIELD_DEF(uint32_t, lastTileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FrobeniusNormCustom, FrobeniusNormCustomTilingData)
} // namespace optiling
"""

host_operator_src="""
#include "frobenius_norm_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr uint32_t BLOCK_DIM = 1;
constexpr uint32_t TILE_LENGTH = 2048;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    FrobeniusNormCustomTilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    uint32_t tileNum = (totalLength + TILE_LENGTH - 1) / TILE_LENGTH;
    uint32_t lastTileLength = totalLength == 0 ? 0 : totalLength - (tileNum - 1) * TILE_LENGTH;

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(tileNum);
    tiling.set_tileLength(TILE_LENGTH);
    tiling.set_lastTileLength(lastTileLength);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
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
} // namespace ge

namespace ops {
class FrobeniusNormCustom : public OpDef {
public:
    explicit FrobeniusNormCustom(const char* name) : OpDef(name)
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

OP_ADD(FrobeniusNormCustom);
} // namespace ops
"""

kernel_src="""
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelFrobeniusNorm {
public:
    __aicore__ inline KernelFrobeniusNorm() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t totalLength,
        uint32_t tileNum,
        uint32_t tileLength,
        uint32_t lastTileLength)
    {
        this->totalLength = totalLength;
        this->tileNum = tileNum;
        this->tileLength = tileLength;
        this->lastTileLength = lastTileLength;

        xGm.SetGlobalBuffer((__gm__ float*)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(calcBuf, this->tileLength * 3 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->totalLength == 0) {
            return;
        }

        float sumSquares = 0.0f;
        for (uint32_t tileIdx = 0; tileIdx < this->tileNum; ++tileIdx) {
            uint32_t currentLength = GetCurrentLength(tileIdx);
            CopyIn(tileIdx, currentLength);
            ReduceTile(currentLength, sumSquares);
        }

        AscendC::LocalTensor<float> scalarLocal = calcBuf.Get<float>();
        scalarLocal.SetValue(0, sumSquares);
        AscendC::Sqrt(scalarLocal, scalarLocal, 1);
        float normValue = scalarLocal.GetValue(0);

        for (uint32_t tileIdx = 0; tileIdx < this->tileNum; ++tileIdx) {
            uint32_t currentLength = GetCurrentLength(tileIdx);
            CopyIn(tileIdx, currentLength);
            NormalizeTile(tileIdx, currentLength, normValue);
        }
    }

private:
    __aicore__ inline uint32_t GetCurrentLength(uint32_t tileIdx) const
    {
        return tileIdx + 1 == this->tileNum ? this->lastTileLength : this->tileLength;
    }

    __aicore__ inline void CopyIn(uint32_t tileIdx, uint32_t currentLength)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[tileIdx * this->tileLength], currentLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void ReduceTile(uint32_t currentLength, float& sumSquares)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> squareLocal = calcBuf.Get<float>();
        AscendC::LocalTensor<float> reduceTmp = calcBuf.Get<float>()[this->tileLength];

        AscendC::Mul(squareLocal, xLocal, xLocal, currentLength);
        AscendC::ReduceSum<float>(squareLocal, squareLocal, reduceTmp, currentLength);
        sumSquares += squareLocal.GetValue(0);

        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void NormalizeTile(uint32_t tileIdx, uint32_t currentLength, float normValue)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::LocalTensor<float> normLocal = calcBuf.Get<float>();

        AscendC::Duplicate(normLocal, normValue, currentLength);
        AscendC::Div(yLocal, xLocal, normLocal, currentLength);
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
        CopyOut(tileIdx, currentLength);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx, uint32_t currentLength)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[tileIdx * this->tileLength], yLocal, currentLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t totalLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lastTileLength;
};

extern "C" __global__ __aicore__ void frobenius_norm_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelFrobeniusNorm op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum, tiling_data.tileLength, tiling_data.lastTileLength);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor frobenius_norm_impl_npu(const at::Tensor& x) {
    at::Tensor result = at::empty_like(x);
    EXEC_NPU_CMD(aclnnFrobeniusNormCustom, x, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("frobenius_norm_custom", &frobenius_norm_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("frobenius_norm_custom", &frobenius_norm_impl_npu, "frobenius norm normalization");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.frobenius_norm_custom(x)
'''
