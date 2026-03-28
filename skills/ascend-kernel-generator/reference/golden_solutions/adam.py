project_json_src='''
[
    {
        "op": "AdamCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "param",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "grad",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "m",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "v",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            }
        ],
        "output_desc": [
            {
                "name": "out",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            }
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AdamCustomTilingData)
  TILING_DATA_FIELD_DEF(int64_t, totalCoreNum);
  TILING_DATA_FIELD_DEF(int64_t, handleExtraLoopCoreNum);
  TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
  TILING_DATA_FIELD_DEF(int64_t, numPerLoop);
  TILING_DATA_FIELD_DEF(int64_t, loopNumPerCore);
  TILING_DATA_FIELD_DEF(int64_t, numLastLoop);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AdamCustom, AdamCustomTilingData)
}
"""

host_operator_src="""
#include "adam_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AdamCustomTilingData tiling;
    int64_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    const int64_t numPerLoop = 2432;
    int64_t totalCoreNum = BLOCK_DIM;
    int64_t loopNum = (totalLength + numPerLoop - 1) / numPerLoop;
    int64_t numLastLoopActual = totalLength % numPerLoop;
    int64_t numLastLoop = numLastLoopActual == 0 ? numPerLoop : numLastLoopActual;
    int64_t loopNumPerCore = loopNum / totalCoreNum;
    int64_t handleExtraLoopCoreNum = loopNum % totalCoreNum;
    int64_t usedCoreNum = loopNumPerCore > 0 ? totalCoreNum : handleExtraLoopCoreNum;
    if (handleExtraLoopCoreNum == 0) {
        handleExtraLoopCoreNum = usedCoreNum;
        loopNumPerCore -= 1;
    }

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalCoreNum(totalCoreNum);
    tiling.set_handleExtraLoopCoreNum(handleExtraLoopCoreNum);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_numPerLoop(numPerLoop);
    tiling.set_loopNumPerCore(loopNumPerCore);
    tiling.set_numLastLoop(numLastLoop);
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
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    *outputShape = *inputShape;
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
class AdamCustom : public OpDef {
public:
    explicit AdamCustom(const char* name) : OpDef(name)
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
        this->Input("m")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(AdamCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t IN_BUFFER_NUM = 4;
constexpr int32_t OUT_BUFFER_NUM = 4;
constexpr int32_t PARAM_ORDER = 0;
constexpr int32_t M_ORDER = 1;
constexpr int32_t V_ORDER = 2;
constexpr int32_t GRAD_ORDER = 3;
constexpr int32_t TMP_ORDER = 3;

class KernelAdam {
public:
    __aicore__ inline KernelAdam() {}

    __aicore__ inline void Init(
        GM_ADDR param,
        GM_ADDR grad,
        GM_ADDR m,
        GM_ADDR v,
        GM_ADDR out,
        const AdamCustomTilingData* tilingData)
    {
        this->numPerLoop = tilingData->numPerLoop;
        this->loopNumPerCore = tilingData->loopNumPerCore;
        this->numLastLoop = tilingData->numLastLoop;
        this->handleExtraLoopCoreNum = tilingData->handleExtraLoopCoreNum;
        this->usedCoreNum = tilingData->usedCoreNum;
        this->blockIdx = GetBlockIdx();

        int64_t gmOffset = this->blockIdx * this->numPerLoop;
        paramGm.SetGlobalBuffer((__gm__ DTYPE_PARAM *)param + gmOffset);
        gradGm.SetGlobalBuffer((__gm__ DTYPE_GRAD *)grad + gmOffset);
        mGm.SetGlobalBuffer((__gm__ DTYPE_M *)m + gmOffset);
        vGm.SetGlobalBuffer((__gm__ DTYPE_V *)v + gmOffset);
        outGm.SetGlobalBuffer((__gm__ DTYPE_OUT *)out + gmOffset);

        pipe.InitBuffer(inQueue, BUFFER_NUM, IN_BUFFER_NUM * this->numPerLoop * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, OUT_BUFFER_NUM * this->numPerLoop * sizeof(float));

        paramOffset = PARAM_ORDER * this->numPerLoop;
        mOffset = M_ORDER * this->numPerLoop;
        vOffset = V_ORDER * this->numPerLoop;
        gradOffset = GRAD_ORDER * this->numPerLoop;
        tmpOffset = TMP_ORDER * this->numPerLoop;

        oneSubBeta1 = 0.1f;
        beta2 = 0.999f;
        oneSubBeta2 = 0.001f;
        lr = 0.001f;
        eps = 1e-8f;
        stepSize = lr;
        biasCorrection2Sqrt = 1.0f / sqrt(oneSubBeta2);
    }

    __aicore__ inline void Process()
    {
        if (this->blockIdx < this->usedCoreNum) {
            int64_t curLoopCount = this->loopNumPerCore;
            if (this->blockIdx < this->handleExtraLoopCoreNum - 1) {
                curLoopCount += 1;
            }

            for (int64_t i = 0; i < curLoopCount; ++i) {
                CopyIn(i, this->numPerLoop);
                Compute(this->numPerLoop);
                CopyOut(i, this->numPerLoop);
            }

            if (this->blockIdx == this->handleExtraLoopCoreNum - 1) {
                CopyIn(this->loopNumPerCore, this->numLastLoop);
                Compute(this->numLastLoop);
                CopyOut(this->loopNumPerCore, this->numLastLoop);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t index, int64_t dataCount)
    {
        LocalTensor<float> inputLocal = inQueue.AllocTensor<float>();
        int64_t offset = this->usedCoreNum * index * this->numPerLoop;
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> copyPadParams{false, 0, 0, 0};
        DataCopyPad(inputLocal[paramOffset], paramGm[offset], copyParams, copyPadParams);
        DataCopyPad(inputLocal[mOffset], mGm[offset], copyParams, copyPadParams);
        DataCopyPad(inputLocal[vOffset], vGm[offset], copyParams, copyPadParams);
        DataCopyPad(inputLocal[gradOffset], gradGm[offset], copyParams, copyPadParams);
        inQueue.EnQue(inputLocal);
    }

    __aicore__ inline void Compute(int32_t dataCount)
    {
        LocalTensor<float> inputLocal = inQueue.DeQue<float>();
        LocalTensor<float> outputLocal = outQueue.AllocTensor<float>();

        pipe_barrier(PIPE_V);
        Muls(outputLocal[paramOffset], inputLocal[paramOffset], 1.0f, dataCount);
        pipe_barrier(PIPE_V);
        Abs(outputLocal[tmpOffset], inputLocal[gradOffset], dataCount);
        pipe_barrier(PIPE_V);
        Adds(outputLocal[tmpOffset], outputLocal[tmpOffset], eps, dataCount);
        pipe_barrier(PIPE_V);
        Div(outputLocal[mOffset], inputLocal[gradOffset], outputLocal[tmpOffset], dataCount);
        pipe_barrier(PIPE_V);
        Muls(outputLocal[mOffset], outputLocal[mOffset], stepSize, dataCount);
        pipe_barrier(PIPE_V);
        Sub(outputLocal[paramOffset], outputLocal[paramOffset], outputLocal[mOffset], dataCount);

        outQueue.EnQue<float>(outputLocal);
        inQueue.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut(int64_t index, int64_t dataCount)
    {
        LocalTensor<float> outputLocal = outQueue.DeQue<float>();
        int64_t offset = this->usedCoreNum * index * this->numPerLoop;
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(float)), 0, 0, 0};
        DataCopyPad(outGm[offset], outputLocal[paramOffset], copyParams);
        outQueue.FreeTensor(outputLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueue;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueue;

    GlobalTensor<DTYPE_PARAM> paramGm;
    GlobalTensor<DTYPE_GRAD> gradGm;
    GlobalTensor<DTYPE_M> mGm;
    GlobalTensor<DTYPE_V> vGm;
    GlobalTensor<DTYPE_OUT> outGm;

    int64_t totalCoreNum;
    int64_t handleExtraLoopCoreNum;
    int64_t usedCoreNum;
    int64_t numPerLoop;
    int64_t loopNumPerCore;
    int64_t numLastLoop;
    int64_t blockIdx;
    int64_t paramOffset;
    int64_t gradOffset;
    int64_t mOffset;
    int64_t vOffset;
    int64_t tmpOffset;

    float oneSubBeta1;
    float beta2;
    float oneSubBeta2;
    float lr;
    float eps;
    float stepSize;
    float biasCorrection2Sqrt;
};

extern "C" __global__ __aicore__ void adam_custom(
    GM_ADDR param,
    GM_ADDR grad,
    GM_ADDR m,
    GM_ADDR v,
    GM_ADDR out,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelAdam op;
    op.Init(param, grad, m, v, out, &tiling_data);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor adam_custom_impl_npu(
    const at::Tensor& param,
    const at::Tensor& grad,
    const at::Tensor& m,
    const at::Tensor& v)
{
    at::Tensor result = at::empty_like(param);
    EXEC_NPU_CMD(aclnnAdamCustom, param, grad, m, v, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("adam_custom", &adam_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adam_custom", &adam_custom_impl_npu, "adam custom");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib

class ModelNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, param, grad, m, v):
        return custom_ops_lib.adam_custom(param, grad, m, v)
'''
