project_json_src='''
[
    {
        "op": "AdagradCustom",
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
                "name": "accum",
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
BEGIN_TILING_DATA_DEF(AdagradCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AdagradCustom, AdagradCustomTilingData)
}
"""

host_operator_src="""
#include "adagrad_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 2048;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AdagradCustomTilingData tiling;
    const uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);
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
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class AdagradCustom : public OpDef {
public:
    explicit AdagradCustom(const char* name) : OpDef(name)
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
        this->Input("accum")
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

OP_ADD(AdagradCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelAdagrad {
public:
    __aicore__ inline KernelAdagrad() {}

    __aicore__ inline void Init(
        GM_ADDR param,
        GM_ADDR grad,
        GM_ADDR accum,
        GM_ADDR updatedParam,
        uint32_t totalLength,
        uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        paramGm.SetGlobalBuffer((__gm__ float*)param + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        gradGm.SetGlobalBuffer((__gm__ float*)grad + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        accumGm.SetGlobalBuffer((__gm__ float*)accum + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        updatedParamGm.SetGlobalBuffer((__gm__ float*)updatedParam + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueParam, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueGrad, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueAccum, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueParam, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuffer, this->tileLength * sizeof(float) * 3);
    }

    __aicore__ inline void Process()
    {
        const int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; ++i) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<float> paramLocal = inQueueParam.AllocTensor<float>();
        AscendC::LocalTensor<float> gradLocal = inQueueGrad.AllocTensor<float>();
        AscendC::LocalTensor<float> accumLocal = inQueueAccum.AllocTensor<float>();
        AscendC::DataCopy(paramLocal, paramGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(gradLocal, gradGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(accumLocal, accumGm[progress * this->tileLength], this->tileLength);
        inQueueParam.EnQue(paramLocal);
        inQueueGrad.EnQue(gradLocal);
        inQueueAccum.EnQue(accumLocal);
    }

    __aicore__ inline void Compute()
    {
        constexpr float kLearningRate = 1.0e-2f;
        constexpr float kEpsilon = 1.0e-10f;

        AscendC::LocalTensor<float> paramLocal = inQueueParam.DeQue<float>();
        AscendC::LocalTensor<float> gradLocal = inQueueGrad.DeQue<float>();
        AscendC::LocalTensor<float> accumLocal = inQueueAccum.DeQue<float>();
        AscendC::LocalTensor<float> outLocal = outQueueParam.AllocTensor<float>();

        AscendC::LocalTensor<float> updatedAccum = tmpBuffer.Get<float>();
        AscendC::LocalTensor<float> denom = tmpBuffer.Get<float>()[this->tileLength];
        AscendC::LocalTensor<float> step = tmpBuffer.Get<float>()[this->tileLength * 2];

        AscendC::Mul(updatedAccum, gradLocal, gradLocal, this->tileLength);
        AscendC::Add(updatedAccum, updatedAccum, accumLocal, this->tileLength);
        AscendC::Sqrt(denom, updatedAccum, this->tileLength);
        AscendC::Adds(denom, denom, kEpsilon, this->tileLength);
        AscendC::Muls(step, gradLocal, kLearningRate, this->tileLength);
        AscendC::Div(step, step, denom, this->tileLength);
        AscendC::Sub(outLocal, paramLocal, step, this->tileLength);

        outQueueParam.EnQue<float>(outLocal);
        inQueueParam.FreeTensor(paramLocal);
        inQueueGrad.FreeTensor(gradLocal);
        inQueueAccum.FreeTensor(accumLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> outLocal = outQueueParam.DeQue<float>();
        AscendC::DataCopy(updatedParamGm[progress * this->tileLength], outLocal, this->tileLength);
        outQueueParam.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffer;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueParam;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueGrad;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueAccum;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueParam;
    AscendC::GlobalTensor<float> paramGm;
    AscendC::GlobalTensor<float> gradGm;
    AscendC::GlobalTensor<float> accumGm;
    AscendC::GlobalTensor<float> updatedParamGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void adagrad_custom(
    GM_ADDR param,
    GM_ADDR grad,
    GM_ADDR accum,
    GM_ADDR updatedParam,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelAdagrad op;
    op.Init(param, grad, accum, updatedParam, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor adagrad_custom_impl_npu(
    const at::Tensor& param,
    const at::Tensor& grad,
    const at::Tensor& accum)
{
    at::Tensor paramContiguous = param.contiguous();
    at::Tensor gradContiguous = grad.contiguous();
    at::Tensor accumContiguous = accum.contiguous();
    at::Tensor result = at::empty_like(paramContiguous);
    EXEC_NPU_CMD(aclnnAdagradCustom, paramContiguous, gradContiguous, accumContiguous, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("adagrad_custom", &adagrad_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adagrad_custom", &adagrad_custom_impl_npu, "Adagrad update");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, lr: float = 1e-2, eps: float = 1e-10) -> None:
        super().__init__()
        self.lr = lr
        self.eps = eps

    def forward(self, param: torch.Tensor, grad: torch.Tensor, accum: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.adagrad_custom(param, grad, accum)
'''
