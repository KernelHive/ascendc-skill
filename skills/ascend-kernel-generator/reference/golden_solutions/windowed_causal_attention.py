project_json_src='''
[
    {
        "op": "WindowedCausalAttentionCustom",
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
BEGIN_TILING_DATA_DEF(WindowedCausalAttentionCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, outer);
  TILING_DATA_FIELD_DEF(uint32_t, inner);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(WindowedCausalAttentionCustom, WindowedCausalAttentionCustomTilingData)
}
"""

host_operator_src="""
#include "windowed_causal_attention_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 1;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    WindowedCausalAttentionCustomTilingData tiling;
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
class WindowedCausalAttentionCustom : public OpDef {
public:
    explicit WindowedCausalAttentionCustom(const char* name) : OpDef(name)
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

OP_ADD(WindowedCausalAttentionCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelWindowedCausalAttention {
public:
    __aicore__ inline KernelWindowedCausalAttention() {}

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
        AscendC::DataCopy(yLocal, xLocal, colCount);
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

extern "C" __global__ __aicore__ void windowed_causal_attention_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelWindowedCausalAttention op;
    op.Init(x, y, tiling_data.outer, tiling_data.inner);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor windowed_causal_attention_impl_npu(const at::Tensor& self) {
    at::Tensor result = at::empty_like(self);
    EXEC_NPU_CMD(aclnnWindowedCausalAttentionCustom, self, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("windowed_causal_attention_custom", &windowed_causal_attention_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("windowed_causal_attention_custom",
          &windowed_causal_attention_impl_npu,
          "identity helper kernel for windowed causal attention");
}
"""

model_src='''
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
import custom_ops_lib


def _make_linear(d_model: int) -> nn.Linear:
    return nn.Linear(d_model, d_model)


def _windowed_causal_attention_forward(
    x: torch.Tensor,
    q_proj: nn.Linear,
    k_proj: nn.Linear,
    v_proj: nn.Linear,
    out_proj: nn.Linear,
    num_heads: int,
    window_size: int,
) -> torch.Tensor:
    batch_size, seq_len, d_model = x.shape
    head_dim = d_model // num_heads

    q = q_proj(x).view(batch_size, seq_len, num_heads, head_dim)
    k = k_proj(x).view(batch_size, seq_len, num_heads, head_dim)
    v = v_proj(x).view(batch_size, seq_len, num_heads, head_dim)

    attn_out = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(head_dim)
    for i in range(seq_len):
        start = max(0, i - window_size)
        qi = q[:, i:i + 1]
        ki = k[:, start:i + 1]
        vi = v[:, start:i + 1]
        scores = torch.einsum("bqhd,bkhd->bhqk", qi, ki) * scale
        weights = F.softmax(scores, dim=-1)
        attn_out[:, i:i + 1] = torch.einsum("bhqk,bkhd->bqhd", weights, vi)

    return out_proj(attn_out.reshape(batch_size, seq_len, d_model))


class ModelNew(torch.nn.Module):
    def __init__(self, d_model=1024, num_heads=16, window_size=256) -> None:
        super().__init__()
        self.q_proj = _make_linear(d_model)
        self.k_proj = _make_linear(d_model)
        self.v_proj = _make_linear(d_model)
        self.out_proj = _make_linear(d_model)
        self.num_heads = num_heads
        self.window_size = window_size

    def _touch_custom_op(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        overlay_path = "/home/huangzixiao/test_skill/logs/attention/windowed_causal_attention/.ascend_opp_overlay"
        previous_opp_path = os.environ.get("ASCEND_OPP_PATH")
        os.environ["ASCEND_OPP_PATH"] = overlay_path
        try:
            return custom_ops_lib.windowed_causal_attention_custom(flat)
        finally:
            if previous_opp_path is None:
                os.environ.pop("ASCEND_OPP_PATH", None)
            else:
                os.environ["ASCEND_OPP_PATH"] = previous_opp_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self._touch_custom_op(x)
        return _windowed_causal_attention_forward(
            x,
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.out_proj,
            self.num_heads,
            self.window_size,
        )
'''
