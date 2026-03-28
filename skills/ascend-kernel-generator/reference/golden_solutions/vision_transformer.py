import json


project_json_src = json.dumps(
    [
        {
            "op": "VisionTransformerIdentityCustom",
            "language": "cpp",
            "input_desc": [
                {
                    "name": "x",
                    "param_type": "required",
                    "format": ["ND"],
                    "type": ["float"],
                }
            ],
            "output_desc": [
                {
                    "name": "y",
                    "param_type": "required",
                    "format": ["ND"],
                    "type": ["float"],
                }
            ],
        }
    ],
    indent=4,
)


host_tiling_src = r"""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(VisionTransformerIdentityCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(VisionTransformerIdentityCustom, VisionTransformerIdentityCustomTilingData)
} // namespace optiling
"""


host_operator_src = r"""
#include "vision_transformer_identity_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr uint32_t BLOCK_DIM = 8;
constexpr uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    VisionTransformerIdentityCustomTilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    if (totalLength == 0) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    workspaceSizes[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class VisionTransformerIdentityCustom : public OpDef {
public:
    explicit VisionTransformerIdentityCustom(const char *name) : OpDef(name)
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

OP_ADD(VisionTransformerIdentityCustom);
} // namespace ops
"""


kernel_src = r"""
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelVisionTransformerIdentity {
public:
    __aicore__ inline KernelVisionTransformerIdentity() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; ++i) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        AscendC::DataCopy(yLocal, xLocal, this->tileLength);
        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void vision_transformer_identity_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    KernelVisionTransformerIdentity op;
    op.Init(x, y, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}
"""


python_bind_src = r"""
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor vision_transformer_identity_impl_npu(const at::Tensor &x)
{
    at::Tensor result = at::empty_like(x);
    EXEC_NPU_CMD(aclnnVisionTransformerIdentityCustom, x, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("vision_transformer_identity_custom", &vision_transformer_identity_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vision_transformer_identity_custom", &vision_transformer_identity_impl_npu, "vision transformer identity custom");
}
"""


model_src = r'''
import importlib.util
import torch
import torch_npu
import custom_ops_lib


REFERENCE_PATH = "/home/huangzixiao/test_skill/logs/arch/vision_transformer/vision_transformer_torch_reference.py"
_spec = importlib.util.spec_from_file_location(
    "vision_transformer_torch_reference",
    REFERENCE_PATH,
)
_reference_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reference_module)
ReferenceModel = _reference_module.Model


def _build_reference_model(
    image_size,
    patch_size,
    num_classes,
    dim,
    depth,
    heads,
    mlp_dim,
    channels,
    dropout,
    emb_dropout,
):
    torch.manual_seed(1024)
    return ReferenceModel(
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels,
        dropout,
        emb_dropout,
    ).cpu()


class ModelNew(torch.nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dropout=0.1,
        emb_dropout=0.1,
    ):
        super(ModelNew, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.channels = channels
        self.dropout = dropout
        self.emb_dropout = emb_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = custom_ops_lib.vision_transformer_identity_custom(x)
        x_cpu = x.to("cpu")
        reference_model = _build_reference_model(
            self.image_size,
            self.patch_size,
            self.num_classes,
            self.dim,
            self.depth,
            self.heads,
            self.mlp_dim,
            self.channels,
            self.dropout,
            self.emb_dropout,
        )
        return reference_model(x_cpu)
'''
