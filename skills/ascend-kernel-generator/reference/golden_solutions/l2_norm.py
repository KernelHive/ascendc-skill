project_json_src='''
[
    {
        "op": "L2NormCustom",
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
BEGIN_TILING_DATA_DEF(L2NormCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, rowCount);
  TILING_DATA_FIELD_DEF(uint32_t, colCount);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(L2NormCustom, L2NormCustomTilingData)
}
"""

host_operator_src="""
#include "l2_norm_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 1;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    L2NormCustomTilingData tiling;
    const auto inputShape = context->GetInputTensor(0)->GetStorageShape();
    if (inputShape.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const int64_t rowCount = inputShape.GetDim(0);
    const int64_t colCount = inputShape.GetDim(1);
    if (rowCount <= 0 || colCount <= 0) {
        return ge::GRAPH_FAILED;
    }
    if ((colCount * static_cast<int64_t>(sizeof(float))) % 32 != 0) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_rowCount(static_cast<uint32_t>(rowCount));
    tiling.set_colCount(static_cast<uint32_t>(colCount));
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class L2NormCustom : public OpDef {
public:
    explicit L2NormCustom(const char* name) : OpDef(name)
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

OP_ADD(L2NormCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelL2Norm {
public:
    __aicore__ inline KernelL2Norm() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t rowCount, uint32_t colCount)
    {
        this->rowCount = rowCount;
        this->colCount = colCount;
        xGm.SetGlobalBuffer((__gm__ float *)x, rowCount * colCount);
        yGm.SetGlobalBuffer((__gm__ float *)y, rowCount * colCount);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, colCount * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, colCount * sizeof(float));
        pipe.InitBuffer(calcBuf, colCount * sizeof(float) * 2);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t rowIdx = 0; rowIdx < rowCount; ++rowIdx) {
            CopyIn(rowIdx);
            Compute();
            CopyOut(rowIdx);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t rowIdx)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[rowIdx * colCount], colCount);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::LocalTensor<float> squareLocal = calcBuf.Get<float>();
        AscendC::LocalTensor<float> reduceTmp = calcBuf.Get<float>()[colCount];

        AscendC::Mul(squareLocal, xLocal, xLocal, colCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum<float>(squareLocal, squareLocal, reduceTmp, colCount);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        float denom = squareLocal.GetValue(0);
        squareLocal.SetValue(0, denom);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::Sqrt(squareLocal, squareLocal, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        denom = squareLocal.GetValue(0);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::Duplicate(squareLocal, denom, colCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(yLocal, xLocal, squareLocal, colCount);

        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t rowIdx)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[rowIdx * colCount], yLocal, colCount);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t rowCount;
    uint32_t colCount;
};

extern "C" __global__ __aicore__ void l2_norm_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelL2Norm op;
    op.Init(x, y, tiling_data.rowCount, tiling_data.colCount);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor l2_norm_custom_impl_npu(const at::Tensor& self) {
    at::Tensor result = at::empty_like(self);
    EXEC_NPU_CMD(aclnnL2NormCustom, self, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("l2_norm_custom", &l2_norm_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("l2_norm_custom", &l2_norm_custom_impl_npu, "L2 normalization along dim=1");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.l2_norm_custom(x)
'''
