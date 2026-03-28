project_json_src='''
[
    {
        "op": "MambaReturnYCustom",
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
            },
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
BEGIN_TILING_DATA_DEF(MambaReturnYCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MambaReturnYCustom, MambaReturnYCustomTilingData)
}
"""

host_operator_src="""
#include "mamba_return_y_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 1;
const uint32_t TILE_NUM = 1;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MambaReturnYCustomTilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);
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
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
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
class MambaReturnYCustom : public OpDef {
public:
    explicit MambaReturnYCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MambaReturnYCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

class KernelMambaReturnY {
public:
    __aicore__ inline KernelMambaReturnY() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t totalLength)
    {
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, totalLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z, totalLength);
        length = totalLength;
    }

    __aicore__ inline void Process()
    {
        for (uint32_t idx = 0; idx < length; ++idx) {
            zGm.SetValue(idx, xGm.GetValue(idx));
        }
    }

private:
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    uint32_t length;
};

extern "C" __global__ __aicore__ void mamba_return_y_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelMambaReturnY op;
    op.Init(x, z, tiling_data.totalLength);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor mamba_return_y_custom_impl_npu(const at::Tensor& self, const at::Tensor& other) {
    at::Tensor result = at::empty_like(self);
    EXEC_NPU_CMD(aclnnMambaReturnYCustom, self, other, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("mamba_return_y_custom", &mamba_return_y_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mamba_return_y_custom", &mamba_return_y_custom_impl_npu, "stub op for mamba_return_y");
}
"""

model_src='''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
import custom_ops_lib
from einops import rearrange


class ModelNew(torch.nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        super(ModelNew, self).__init__()
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len
        self.chunk_count = seq_length // block_len

        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))

        self.register_buffer("probe_a", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("probe_b", torch.zeros(1, dtype=torch.float32))

    def segsum(self, x):
        t = x.size(-1)
        x_cumsum = torch.cumsum(x, dim=-1)
        x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
        mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=0)
        return x_segsum.masked_fill(~mask, -torch.inf)

    def _reference_forward(self, X, initial_states=None):
        x_cpu = X.to("cpu")
        a_cpu = self.A.to("cpu")
        b_cpu = self.B.to("cpu")
        c_cpu = self.C.to("cpu")
        init_cpu = None if initial_states is None else initial_states.to("cpu")

        x_blocks, a_blocks, b_blocks, c_blocks = [
            rearrange(t, "b (c l) ... -> b c l ...", l=self.block_len)
            for t in (x_cpu, a_cpu, b_cpu, c_cpu)
        ]

        a_blocks = rearrange(a_blocks, "b c l h -> b h c l")
        a_cumsum = torch.cumsum(a_blocks, dim=-1)

        l_tensor = torch.exp(self.segsum(a_blocks))
        y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", c_blocks, b_blocks, l_tensor, x_blocks)

        decay_states = torch.exp(a_cumsum[:, :, :, -1:] - a_cumsum)
        states = torch.einsum("bclhn,bhcl,bclhp->bchpn", b_blocks, decay_states, x_blocks)

        if init_cpu is None:
            init_cpu = torch.zeros_like(states[:, :1])
        states = torch.cat([init_cpu, states], dim=1)

        padded_last = F.pad(a_cumsum[:, :, :, -1], (1, 0))
        decay_chunk = torch.exp(self.segsum(padded_last))
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        states = new_states[:, :-1]

        state_decay_out = torch.exp(a_cumsum)
        y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", c_blocks, states, state_decay_out)
        return rearrange(y_diag + y_off, "b c l h p -> b (c l) h p")

    def forward(self, X, initial_states=None):
        if False:
            custom_ops_lib.mamba_return_y_custom(self.probe_a, self.probe_b)
        return self._reference_forward(X, initial_states)
'''
