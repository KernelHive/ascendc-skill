project_json_src='''
[
    {
        "op": "RmsNormLowerDimCustom",
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
BEGIN_TILING_DATA_DEF(RmsNormLowerDimCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, rowCount);
  TILING_DATA_FIELD_DEF(uint32_t, colCount);
  TILING_DATA_FIELD_DEF(uint32_t, useCoreNums);
  TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
  TILING_DATA_FIELD_DEF(uint32_t, rowsPerCoreTail);
  TILING_DATA_FIELD_DEF(float, epsilon);
  TILING_DATA_FIELD_DEF(float, invColCount);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RmsNormLowerDimCustom, RmsNormLowerDimCustomTilingData)
}
"""

host_operator_src="""
#include "rms_norm_lower_dim_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr uint32_t MAX_BLOCK_DIM = 32;

template <typename T>
inline T CeilDiv(T a, T b)
{
    return b == 0 ? a : (a + b - 1) / b;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& shape = inputShape->GetStorageShape();
    if (shape.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t rowCount = static_cast<uint32_t>(shape.GetDim(0));
    const uint32_t colCount = static_cast<uint32_t>(shape.GetDim(1));
    if (rowCount == 0 || colCount == 0) {
        return ge::GRAPH_FAILED;
    }
    if ((colCount * static_cast<uint32_t>(sizeof(float))) % 32 != 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t useCoreNums = rowCount < MAX_BLOCK_DIM ? rowCount : MAX_BLOCK_DIM;
    const uint32_t rowsPerCore = CeilDiv(rowCount, useCoreNums == 0 ? 1U : useCoreNums);
    const uint32_t rowsPerCoreTail = rowCount - rowsPerCore * (useCoreNums - 1);

    const float* epsilonAttr = context->GetAttrs()->GetAttrPointer<float>(0);
    const float epsilon = epsilonAttr == nullptr ? 1e-5f : *epsilonAttr;

    RmsNormLowerDimCustomTilingData tiling;
    context->SetBlockDim(useCoreNums == 0 ? 1U : useCoreNums);
    tiling.set_rowCount(rowCount);
    tiling.set_colCount(colCount);
    tiling.set_useCoreNums(useCoreNums == 0 ? 1U : useCoreNums);
    tiling.set_rowsPerCore(rowsPerCore);
    tiling.set_rowsPerCoreTail(rowsPerCoreTail);
    tiling.set_epsilon(epsilon);
    tiling.set_invColCount(1.0f / static_cast<float>(colCount));
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
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

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class RmsNormLowerDimCustom : public OpDef {
public:
    explicit RmsNormLowerDimCustom(const char* name) : OpDef(name)
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
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-5f);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(RmsNormLowerDimCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelRmsNormLowerDim {
public:
    __aicore__ inline KernelRmsNormLowerDim() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t colCount,
        uint32_t useCoreNums,
        uint32_t rowsPerCore,
        uint32_t rowsPerCoreTail,
        float epsilon,
        float invColCount)
    {
        this->colCount = colCount;
        this->epsilon = epsilon;
        this->invColCount = invColCount;

        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t localRowCount = blockIdx + 1 == useCoreNums ? rowsPerCoreTail : rowsPerCore;
        const uint32_t startRow = blockIdx * rowsPerCore;
        this->localRowCount = localRowCount;

        xGm.SetGlobalBuffer((__gm__ float*)x + startRow * colCount, localRowCount * colCount);
        yGm.SetGlobalBuffer((__gm__ float*)y + startRow * colCount, localRowCount * colCount);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, colCount * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, colCount * sizeof(float));
        pipe.InitBuffer(calcBuf, colCount * sizeof(float) * 2);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t rowIdx = 0; rowIdx < localRowCount; ++rowIdx) {
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
        float meanSquare = squareLocal.GetValue(0) * invColCount + epsilon;
        squareLocal.SetValue(0, meanSquare);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);

        AscendC::Sqrt(squareLocal, squareLocal, 1);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        float invRms = 1.0f / squareLocal.GetValue(0);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);

        AscendC::Duplicate(squareLocal, invRms, colCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(yLocal, xLocal, squareLocal, colCount);

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
    uint32_t localRowCount;
    uint32_t colCount;
    float epsilon;
    float invColCount;
};

extern "C" __global__ __aicore__ void rms_norm_lower_dim_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelRmsNormLowerDim op;
    op.Init(
        x,
        y,
        tiling_data.colCount,
        tiling_data.useCoreNums,
        tiling_data.rowsPerCore,
        tiling_data.rowsPerCoreTail,
        tiling_data.epsilon,
        tiling_data.invColCount);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor rms_norm_lower_dim_custom_impl_npu(const at::Tensor& self, double epsilon)
{
    at::Tensor result = at::empty_like(self);
    float epsilonValue = static_cast<float>(epsilon);
    EXEC_NPU_CMD(aclnnRmsNormLowerDimCustom, self, epsilonValue, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("rms_norm_lower_dim_custom", &rms_norm_lower_dim_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm_lower_dim_custom", &rms_norm_lower_dim_custom_impl_npu, "RMS normalization along dim=1");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.rms_norm_lower_dim_custom(x, self.eps)
'''
