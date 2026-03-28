project_json_src='''
[
    {
        "op": "ReduceSumCustom",
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
                "name": "z",
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
BEGIN_TILING_DATA_DEF(TilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalLength);
TILING_DATA_FIELD_DEF(uint32_t, outLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ReduceSumCustom, TilingData)
} // namespace optiling

"""

host_operator_src="""
#include "reduce_sum_custom_tiling.h"
#include "register/op_def_registry.h"
#define REDUCE_TILING_0 1
#define REDUCE_TILING_1 2
#define REDUCE_TILING_2 3

namespace optiling {
constexpr uint32_t BLOCK_DIM = 1;
constexpr uint32_t ONE_REPEAT_LEN = 256;
constexpr uint32_t ONE_BLOCK_LEN = 32;
constexpr uint32_t OUT_SHAPE = 32;
constexpr uint32_t FLOAT_THRESHOLD0 = ONE_REPEAT_LEN / sizeof(float);
constexpr uint32_t FLOAT_THRESHOLD1 = ONE_REPEAT_LEN / sizeof(float) * ONE_BLOCK_LEN / sizeof(float);
constexpr uint32_t FLOAT_THRESHOLD2 = ONE_REPEAT_LEN / sizeof(float) * ONE_REPEAT_LEN / sizeof(float);
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    auto inputDtype = context->GetInputTensor(0)->GetDataType();
    if (totalLength <= FLOAT_THRESHOLD0 && inputDtype == ge::DT_FLOAT) {
        context->SetTilingKey(REDUCE_TILING_0);
    } else if (totalLength <= FLOAT_THRESHOLD1 && inputDtype == ge::DT_FLOAT) {
        context->SetTilingKey(REDUCE_TILING_1);
    } else if (totalLength <= FLOAT_THRESHOLD2 && inputDtype == ge::DT_FLOAT) {
        context->SetTilingKey(REDUCE_TILING_2);
    }
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_outLength(OUT_SHAPE);
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
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = {optiling::OUT_SHAPE};
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class ReduceSumCustom : public OpDef {
public:
    explicit ReduceSumCustom(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(ReduceSumCustom);
} // namespace ops
"""

kernel_src="""
#include "kernel_operator.h"
#define REDUCE_TILING_0 1
#define REDUCE_TILING_1 2
#define REDUCE_TILING_2 3

class KernelReduce {
static constexpr uint32_t DEFAULT_BLK_STRIDE = 1;
static constexpr uint32_t DEFAULT_REP_STRIDE = 8;
static constexpr uint32_t REP_LEN = 256;
static constexpr uint32_t BLK_LEN = 32;
public:
    __aicore__ inline KernelReduce() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t totalLength, uint32_t outLength)
    {
        this->totalLength = totalLength;
        this->outLength = outLength;

        xGm.SetGlobalBuffer((__gm__ float *)x, totalLength);
        zGm.SetGlobalBuffer((__gm__ float *)z, outLength);
        pipe.InitBuffer(inQueueX, 1, totalLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, 1, outLength * sizeof(float));
    }
    __aicore__ inline void Process1()
    {
        CopyIn();
        Compute1();
        CopyOut();
    }
    __aicore__ inline void Process2()
    {
        CopyIn();
        Compute2();
        CopyOut();
    }
    __aicore__ inline void Process3()
    {
        CopyIn();
        Compute3();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm, totalLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute1()
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        constexpr int64_t maskLen = REP_LEN / sizeof(float);
        AscendC::WholeReduceSum<float>(zLocal, xLocal, maskLen, 1,
            DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void Compute2()
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        pipe.InitBuffer(calcBuf, totalLength * sizeof(float));
        AscendC::LocalTensor<float> tempTensor1 = calcBuf.Get<float>();
        constexpr uint32_t c0Count = BLK_LEN / sizeof(float);
        const uint32_t blockNum0 = (totalLength + c0Count - 1) / c0Count;
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<float>(0, totalLength);
        AscendC::BlockReduceSum<float, false>(tempTensor1, xLocal, AscendC::MASK_PLACEHOLDER, 1,
            DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<float>(0, blockNum0);
        AscendC::WholeReduceSum<float, false>(zLocal, tempTensor1, AscendC::MASK_PLACEHOLDER, 1,
            DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();
        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void Compute3()
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        pipe.InitBuffer(calcBuf, totalLength * sizeof(float));
        AscendC::LocalTensor<float> tempTensor1 = calcBuf.Get<float>();
        const uint32_t repeatNum = (totalLength * sizeof(float) + REP_LEN - 1) / REP_LEN;
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<float>(0, totalLength);
        AscendC::WholeReduceSum<float, false>(tempTensor1, xLocal, AscendC::MASK_PLACEHOLDER, 1,
            DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<float>(0, repeatNum);
        AscendC::WholeReduceSum<float, false>(zLocal, tempTensor1, AscendC::MASK_PLACEHOLDER, 1,
            DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();
        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        AscendC::DataCopy(zGm, zLocal, this->outLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t totalLength;
    uint32_t outLength;
};

extern "C" __global__ __aicore__ void reduce_sum_custom(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelReduce op;
    op.Init(x, z, tiling_data.totalLength, tiling_data.outLength);
    if (TILING_KEY_IS(REDUCE_TILING_0)) {
        op.Process1();
    } else if (TILING_KEY_IS(REDUCE_TILING_1)) {
        op.Process2();
    } else if (TILING_KEY_IS(REDUCE_TILING_2)) {
        op.Process3();
    }
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor reduce_sum_impl_npu(const at::Tensor& x) {
    at::Tensor result = at::empty({}, x.options());
    EXEC_NPU_CMD(aclnnReduceSumCustom, x, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("reduce_sum_custom", &reduce_sum_impl_npu);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduce_sum_custom", &reduce_sum_impl_npu, "reduce sum");
}
"""
model_src='''
import torch
import torch_npu
import custom_ops_lib
class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.reduce_sum_custom(x)
'''
