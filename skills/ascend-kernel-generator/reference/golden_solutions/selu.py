project_json_src='''
[
    {
        "op": "SeluCustom",
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
        ],
        "attr": [
            {
                "name": "alpha",
                "param_type": "optional",
                "type": "float",
                "default_value": "1.6732632423543772"
            },
            {
                "name": "scale",
                "param_type": "optional",
                "type": "float",
                "default_value": "1.0507009873554805"
            }
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SeluCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalLength);
TILING_DATA_FIELD_DEF(uint32_t, tileNum);
TILING_DATA_FIELD_DEF(float, alpha);
TILING_DATA_FIELD_DEF(float, scale);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SeluCustom, SeluCustomTilingData)
}
"""

host_operator_src="""
#include "selu_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 16;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    SeluCustomTilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const float *alpha = attrs->GetAttrPointer<float>(0);
    const float *scale = attrs->GetAttrPointer<float>(1);

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);
    tiling.set_alpha(*alpha);
    tiling.set_scale(*scale);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const ge::DataType x_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, x_dtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class SeluCustom : public OpDef {
public:
    explicit SeluCustom(const char *name) : OpDef(name)
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
        this->Attr("alpha").AttrType(OPTIONAL).Float(1.6732632423543772f);
        this->Attr("scale").AttrType(OPTIONAL).Float(1.0507009873554805f);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SeluCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelSelu {
public:
    __aicore__ inline KernelSelu() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t totalLength,
        uint32_t tileNum,
        float alpha,
        float scale)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->alpha = alpha;
        this->scale = scale;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuffer1, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuffer2, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuffer3, this->tileLength * sizeof(float));
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
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::LocalTensor<float> positive = tmpBuffer1.Get<float>();
        AscendC::LocalTensor<float> negative = tmpBuffer2.Get<float>();
        AscendC::LocalTensor<float> expNegative = tmpBuffer3.Get<float>();
        constexpr float zero = 0.0f;

        AscendC::Maxs(positive, xLocal, zero, this->tileLength);
        AscendC::Mins(negative, xLocal, zero, this->tileLength);
        AscendC::Exp(expNegative, negative, this->tileLength);
        AscendC::Adds(expNegative, expNegative, -1.0f, this->tileLength);
        AscendC::Muls(expNegative, expNegative, this->alpha, this->tileLength);
        AscendC::Add(yLocal, positive, expNegative, this->tileLength);
        AscendC::Muls(yLocal, yLocal, this->scale, this->tileLength);

        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffer1;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffer2;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffer3;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    float alpha;
    float scale;
};

extern "C" __global__ __aicore__ void selu_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelSelu op;
    op.Init(
        x,
        y,
        tiling_data.totalLength,
        tiling_data.tileNum,
        tiling_data.alpha,
        tiling_data.scale);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor selu_impl_npu(const at::Tensor &self, double alpha, double scale)
{
    at::Tensor result = at::empty_like(self);
    EXEC_NPU_CMD(aclnnSeluCustom, self, alpha, scale, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("selu_custom", &selu_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("selu_custom", &selu_impl_npu, "SELU activation");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(
        self,
        alpha: float = 1.6732632423543772,
        scale: float = 1.0507009873554805,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.selu_custom(x, self.alpha, self.scale)
'''
