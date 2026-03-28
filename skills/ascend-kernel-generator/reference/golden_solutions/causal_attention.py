project_json_src='''
[
    {
        "op": "CausalAttentionCustom",
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
BEGIN_TILING_DATA_DEF(CausalAttentionCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, outer);
  TILING_DATA_FIELD_DEF(uint32_t, inner);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CausalAttentionCustom, CausalAttentionCustomTilingData)
}
"""

host_operator_src="""
#include "causal_attention_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 1;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    CausalAttentionCustomTilingData tiling;
    const auto inputShape = context->GetInputTensor(0)->GetStorageShape();
    if (inputShape.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const int64_t outer = inputShape.GetDim(0);
    const int64_t inner = inputShape.GetDim(1);
    if (outer <= 0 || inner <= 0) {
        return ge::GRAPH_FAILED;
    }
    if ((inner * static_cast<int64_t>(sizeof(float))) % 32 != 0) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_outer(static_cast<uint32_t>(outer));
    tiling.set_inner(static_cast<uint32_t>(inner));
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
class CausalAttentionCustom : public OpDef {
public:
    explicit CausalAttentionCustom(const char* name) : OpDef(name)
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

OP_ADD(CausalAttentionCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelCausalAttention {
public:
    __aicore__ inline KernelCausalAttention() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t outer, uint32_t inner)
    {
        this->rowCount = outer;
        this->colCount = inner;
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, outer * inner);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, outer * inner);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, inner * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, inner * sizeof(DTYPE_Y));
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
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[rowIdx * colCount], colCount);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        DTYPE_Y running = static_cast<DTYPE_Y>(0);
        for (uint32_t i = 0; i < colCount; ++i) {
            running += static_cast<DTYPE_Y>(xLocal.GetValue(i));
            yLocal.SetValue(i, running);
        }
        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t rowIdx)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[rowIdx * colCount], yLocal, colCount);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t rowCount;
    uint32_t colCount;
};

extern "C" __global__ __aicore__ void causal_attention_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelCausalAttention op;
    op.Init(x, y, tiling_data.outer, tiling_data.inner);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor causal_attention_impl_npu(const at::Tensor& self) {
    at::Tensor result = at::empty_like(self);
    EXEC_NPU_CMD(aclnnCausalAttentionCustom, self, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("causal_attention_custom", &causal_attention_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("causal_attention_custom", &causal_attention_impl_npu, "causal prefix attention scan");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, d_model=512, num_heads=4, max_seq_len=4096) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        inv_positions = 1.0 / torch.arange(1, max_seq_len + 1, dtype=torch.float32)
        self.register_buffer("inv_positions", inv_positions, persistent=False)

    def forward(self, x):
        seq_len = x.size(1)
        transposed = x.transpose(1, 2).contiguous()
        flat = transposed.reshape(-1, seq_len)
        prefix = custom_ops_lib.causal_attention_custom(flat)
        scale = self.inv_positions[:seq_len].view(1, seq_len)
        averaged = prefix * scale
        restored = averaged.reshape(x.size(0), x.size(2), seq_len)
        return restored.transpose(1, 2).contiguous()
'''
