project_json_src='''
[
    {
        "op": "RmspropCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "param",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            },
            {
                "name": "grad",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            },
            {
                "name": "v",
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
                "name": "updated_param",
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
BEGIN_TILING_DATA_DEF(RmspropCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, rowCount);
  TILING_DATA_FIELD_DEF(uint32_t, colCount);
  TILING_DATA_FIELD_DEF(uint32_t, useCoreNums);
  TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
  TILING_DATA_FIELD_DEF(uint32_t, rowsPerCoreTail);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RmspropCustom, RmspropCustomTilingData)
}
"""

host_operator_src="""
#include "rmsprop_custom_tiling.h"
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

    RmspropCustomTilingData tiling;
    context->SetBlockDim(useCoreNums == 0 ? 1U : useCoreNums);
    tiling.set_rowCount(rowCount);
    tiling.set_colCount(colCount);
    tiling.set_useCoreNums(useCoreNums == 0 ? 1U : useCoreNums);
    tiling.set_rowsPerCore(rowsPerCore);
    tiling.set_rowsPerCoreTail(rowsPerCoreTail);
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
    const gert::Shape* paramShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    *outputShape = *paramShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class RmspropCustom : public OpDef {
public:
    explicit RmspropCustom(const char* name) : OpDef(name)
    {
        this->Input("param")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("updated_param")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(RmspropCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;
constexpr float RMSPROP_LR = 1.0e-3f;
constexpr float RMSPROP_ALPHA = 0.99f;
constexpr float RMSPROP_ONE_MINUS_ALPHA = 0.01f;
constexpr float RMSPROP_EPS = 1.0e-8f;

class KernelRmsprop {
public:
    __aicore__ inline KernelRmsprop() {}

    __aicore__ inline void Init(
        GM_ADDR param,
        GM_ADDR grad,
        GM_ADDR v,
        GM_ADDR updatedParam,
        uint32_t colCount,
        uint32_t useCoreNums,
        uint32_t rowsPerCore,
        uint32_t rowsPerCoreTail)
    {
        this->colCount = colCount;
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t localRowCount = blockIdx + 1 == useCoreNums ? rowsPerCoreTail : rowsPerCore;
        const uint32_t startRow = blockIdx * rowsPerCore;
        this->localRowCount = localRowCount;

        paramGm.SetGlobalBuffer((__gm__ float*)param + startRow * colCount, localRowCount * colCount);
        gradGm.SetGlobalBuffer((__gm__ float*)grad + startRow * colCount, localRowCount * colCount);
        vGm.SetGlobalBuffer((__gm__ float*)v + startRow * colCount, localRowCount * colCount);
        updatedParamGm.SetGlobalBuffer((__gm__ float*)updatedParam + startRow * colCount, localRowCount * colCount);

        pipe.InitBuffer(inQueueParam, BUFFER_NUM, colCount * sizeof(float));
        pipe.InitBuffer(inQueueGrad, BUFFER_NUM, colCount * sizeof(float));
        pipe.InitBuffer(inQueueV, BUFFER_NUM, colCount * sizeof(float));
        pipe.InitBuffer(outQueueParam, BUFFER_NUM, colCount * sizeof(float));
        pipe.InitBuffer(tmpBuffer, colCount * sizeof(float) * 3);
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
        AscendC::LocalTensor<float> paramLocal = inQueueParam.AllocTensor<float>();
        AscendC::LocalTensor<float> gradLocal = inQueueGrad.AllocTensor<float>();
        AscendC::LocalTensor<float> vLocal = inQueueV.AllocTensor<float>();
        AscendC::DataCopy(paramLocal, paramGm[rowIdx * colCount], colCount);
        AscendC::DataCopy(gradLocal, gradGm[rowIdx * colCount], colCount);
        AscendC::DataCopy(vLocal, vGm[rowIdx * colCount], colCount);
        inQueueParam.EnQue(paramLocal);
        inQueueGrad.EnQue(gradLocal);
        inQueueV.EnQue(vLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> paramLocal = inQueueParam.DeQue<float>();
        AscendC::LocalTensor<float> gradLocal = inQueueGrad.DeQue<float>();
        AscendC::LocalTensor<float> vLocal = inQueueV.DeQue<float>();
        AscendC::LocalTensor<float> outLocal = outQueueParam.AllocTensor<float>();

        AscendC::LocalTensor<float> updatedV = tmpBuffer.Get<float>();
        AscendC::LocalTensor<float> denom = tmpBuffer.Get<float>()[colCount];
        AscendC::LocalTensor<float> step = tmpBuffer.Get<float>()[colCount * 2];

        AscendC::Mul(updatedV, gradLocal, gradLocal, colCount);
        AscendC::Muls(updatedV, updatedV, RMSPROP_ONE_MINUS_ALPHA, colCount);
        AscendC::Muls(vLocal, vLocal, RMSPROP_ALPHA, colCount);
        AscendC::Add(updatedV, vLocal, updatedV, colCount);
        AscendC::Sqrt(denom, updatedV, colCount);
        AscendC::Adds(denom, denom, RMSPROP_EPS, colCount);
        AscendC::Muls(step, gradLocal, RMSPROP_LR, colCount);
        AscendC::Div(step, step, denom, colCount);
        AscendC::Sub(outLocal, paramLocal, step, colCount);

        outQueueParam.EnQue<float>(outLocal);
        inQueueParam.FreeTensor(paramLocal);
        inQueueGrad.FreeTensor(gradLocal);
        inQueueV.FreeTensor(vLocal);
    }

    __aicore__ inline void CopyOut(uint32_t rowIdx)
    {
        AscendC::LocalTensor<float> outLocal = outQueueParam.DeQue<float>();
        AscendC::DataCopy(updatedParamGm[rowIdx * colCount], outLocal, colCount);
        outQueueParam.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuffer;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueParam;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueGrad;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueV;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueParam;
    AscendC::GlobalTensor<float> paramGm;
    AscendC::GlobalTensor<float> gradGm;
    AscendC::GlobalTensor<float> vGm;
    AscendC::GlobalTensor<float> updatedParamGm;
    uint32_t localRowCount;
    uint32_t colCount;
};

extern "C" __global__ __aicore__ void rmsprop_custom(
    GM_ADDR param,
    GM_ADDR grad,
    GM_ADDR v,
    GM_ADDR updatedParam,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelRmsprop op;
    op.Init(
        param,
        grad,
        v,
        updatedParam,
        tiling_data.colCount,
        tiling_data.useCoreNums,
        tiling_data.rowsPerCore,
        tiling_data.rowsPerCoreTail);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor rmsprop_custom_impl_npu(
    const at::Tensor& param,
    const at::Tensor& grad,
    const at::Tensor& v)
{
    at::Tensor paramContiguous = param.contiguous();
    at::Tensor gradContiguous = grad.contiguous();
    at::Tensor vContiguous = v.contiguous();
    at::Tensor result = at::empty_like(paramContiguous);
    EXEC_NPU_CMD(
        aclnnRmspropCustom,
        paramContiguous,
        gradContiguous,
        vContiguous,
        result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("rmsprop_custom", &rmsprop_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsprop_custom", &rmsprop_custom_impl_npu, "RMSProp update");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, lr: float = 1e-3, alpha: float = 0.99, eps: float = 1e-8) -> None:
        super().__init__()
        self.lr = lr
        self.alpha = alpha
        self.eps = eps

    def forward(self, param: torch.Tensor, grad: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.rmsprop_custom(param, grad, v)
'''
