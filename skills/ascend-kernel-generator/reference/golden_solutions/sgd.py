project_json_src='''
[
    {
        "op": "SgdCustom",
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
                "name": "velocity",
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
BEGIN_TILING_DATA_DEF(SgdCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, tailLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SgdCustom, SgdCustomTilingData)
}
"""

host_operator_src="""
#include "sgd_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr uint32_t BLOCK_DIM = 1;
constexpr uint32_t TILE_LENGTH = 1024;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t totalLength = inputShape->GetOriginShape().GetShapeSize();
    if (totalLength == 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t tileNum = totalLength / TILE_LENGTH;
    const uint32_t tailLength = totalLength % TILE_LENGTH;

    SgdCustomTilingData tiling;
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileLength(TILE_LENGTH);
    tiling.set_tileNum(tileNum);
    tiling.set_tailLength(tailLength);
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
class SgdCustom : public OpDef {
public:
    explicit SgdCustom(const char* name) : OpDef(name)
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
        this->Input("velocity")
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

OP_ADD(SgdCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;
constexpr float SGD_MOMENTUM = 0.9f;
constexpr float SGD_LR = 1.0e-2f;

class KernelSgd {
public:
    __aicore__ inline KernelSgd() {}

    __aicore__ inline void Init(
        GM_ADDR param,
        GM_ADDR grad,
        GM_ADDR velocity,
        GM_ADDR updatedParam,
        uint32_t totalLength,
        uint32_t tileLength,
        uint32_t tileNum,
        uint32_t tailLength)
    {
        this->totalLength = totalLength;
        this->tileLength = tileLength;
        this->tileNum = tileNum;
        this->tailLength = tailLength;

        paramGm.SetGlobalBuffer((__gm__ float*)param, totalLength);
        gradGm.SetGlobalBuffer((__gm__ float*)grad, totalLength);
        velocityGm.SetGlobalBuffer((__gm__ float*)velocity, totalLength);
        updatedParamGm.SetGlobalBuffer((__gm__ float*)updatedParam, totalLength);

        pipe.InitBuffer(inQueueParam, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(inQueueGrad, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(inQueueVelocity, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(outQueueParam, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuffer, tileLength * sizeof(float) * 2);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t tileIdx = 0; tileIdx < tileNum; ++tileIdx) {
            CopyIn(tileIdx, tileLength);
            Compute(tileLength);
            CopyOut(tileIdx, tileLength);
        }
        if (tailLength > 0) {
            CopyIn(tileNum, tailLength);
            Compute(tailLength);
            CopyOut(tileNum, tailLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t tileIdx, uint32_t currentLength)
    {
        AscendC::LocalTensor<float> paramLocal = inQueueParam.AllocTensor<float>();
        AscendC::LocalTensor<float> gradLocal = inQueueGrad.AllocTensor<float>();
        AscendC::LocalTensor<float> velocityLocal = inQueueVelocity.AllocTensor<float>();
        const uint32_t offset = tileIdx * tileLength;
        AscendC::DataCopy(paramLocal, paramGm[offset], currentLength);
        AscendC::DataCopy(gradLocal, gradGm[offset], currentLength);
        AscendC::DataCopy(velocityLocal, velocityGm[offset], currentLength);
        inQueueParam.EnQue(paramLocal);
        inQueueGrad.EnQue(gradLocal);
        inQueueVelocity.EnQue(velocityLocal);
    }

    __aicore__ inline void Compute(uint32_t currentLength)
    {
        AscendC::LocalTensor<float> paramLocal = inQueueParam.DeQue<float>();
        AscendC::LocalTensor<float> gradLocal = inQueueGrad.DeQue<float>();
        AscendC::LocalTensor<float> velocityLocal = inQueueVelocity.DeQue<float>();
        AscendC::LocalTensor<float> outLocal = outQueueParam.AllocTensor<float>();

        AscendC::LocalTensor<float> tmpVelocity = tmpBuffer.Get<float>();
        AscendC::LocalTensor<float> tmpStep = tmpBuffer.Get<float>()[tileLength];

        AscendC::Muls(tmpVelocity, velocityLocal, SGD_MOMENTUM, currentLength);
        AscendC::Add(tmpStep, tmpVelocity, gradLocal, currentLength);
        AscendC::Muls(tmpStep, tmpStep, SGD_LR, currentLength);
        AscendC::Sub(outLocal, paramLocal, tmpStep, currentLength);

        outQueueParam.EnQue<float>(outLocal);
        inQueueParam.FreeTensor(paramLocal);
        inQueueGrad.FreeTensor(gradLocal);
        inQueueVelocity.FreeTensor(velocityLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx, uint32_t currentLength)
    {
        AscendC::LocalTensor<float> outLocal = outQueueParam.DeQue<float>();
        const uint32_t offset = tileIdx * tileLength;
        AscendC::DataCopy(updatedParamGm[offset], outLocal, currentLength);
        outQueueParam.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuffer;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueParam;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueGrad;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueVelocity;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueParam;
    AscendC::GlobalTensor<float> paramGm;
    AscendC::GlobalTensor<float> gradGm;
    AscendC::GlobalTensor<float> velocityGm;
    AscendC::GlobalTensor<float> updatedParamGm;
    uint32_t totalLength;
    uint32_t tileLength;
    uint32_t tileNum;
    uint32_t tailLength;
};

extern "C" __global__ __aicore__ void sgd_custom(
    GM_ADDR param,
    GM_ADDR grad,
    GM_ADDR velocity,
    GM_ADDR updatedParam,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelSgd op;
    op.Init(
        param,
        grad,
        velocity,
        updatedParam,
        tiling_data.totalLength,
        tiling_data.tileLength,
        tiling_data.tileNum,
        tiling_data.tailLength);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor sgd_custom_impl_npu(
    const at::Tensor& param,
    const at::Tensor& grad,
    const at::Tensor& velocity)
{
    at::Tensor result = at::empty_like(param);
    EXEC_NPU_CMD(aclnnSgdCustom, param, grad, velocity, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("sgd_custom", &sgd_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sgd_custom", &sgd_custom_impl_npu, "sgd update");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib

class ModelNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, param, grad, velocity):
        return custom_ops_lib.sgd_custom(param, grad, velocity)
'''
