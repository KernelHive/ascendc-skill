project_json_src='''
[
    {
        "op": "MseLossCustom",
        "input_desc": [
            {
                "name": "predict",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            },
            {
                "name": "label",
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
BEGIN_TILING_DATA_DEF(MseLossCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, blockLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);
    TILING_DATA_FIELD_DEF(uint32_t, tileLength);
    TILING_DATA_FIELD_DEF(uint32_t, lasttileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MseLossCustom, MseLossCustomTilingData)
}

"""

host_operator_src="""

#include "mse_loss_custom_tiling.h"
#include "register/op_def_registry.h"
#include <string.h>

namespace optiling {
const uint32_t BLOCK_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MseLossCustomTilingData tiling;
    uint32_t sizeofdatatype = 4;
    uint32_t totalLengthAligned;
    uint32_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();

    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;
    uint32_t ub_block_num = 1024;
    uint32_t tile_num;

    if (ub_block_num % 2 != 0) {
      ub_block_num = ub_block_num - 1;
    }

    if (totalLength % ALIGN_NUM != 0) {
        totalLengthAligned =
            ((totalLength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
    } else {
        totalLengthAligned = totalLength;
    }

    tiling.set_totalLength(totalLength);

    auto block_dim = 1;
    context->SetBlockDim(block_dim);

    uint32_t blockLength = 0;
    uint32_t tileLength = 0;
    uint32_t lasttileLength = 0;

    blockLength = totalLengthAligned / block_dim;
    tile_num = blockLength / ALIGN_NUM / ub_block_num;

    if ((totalLengthAligned / block_dim / ALIGN_NUM) % ub_block_num == 0 || tile_num == 0) {
        if (tile_num == 0) {
            tile_num = 1;
        }
        if (blockLength < ub_block_num * ALIGN_NUM) {
            tileLength = ((blockLength / ALIGN_NUM) + 1) / 2 * 2 * ALIGN_NUM;
            lasttileLength = tileLength;
        }
        else {
            tileLength = ub_block_num * ALIGN_NUM;
            lasttileLength = tileLength;
        }
    }
    else {
        tile_num = tile_num + 1;
        tileLength = ub_block_num * ALIGN_NUM;
        lasttileLength = blockLength - (tile_num - 1) * tileLength;
    }

    tiling.set_blockLength(blockLength);
    tiling.set_tileNum(tile_num);
    tiling.set_tileLength(tileLength);
    tiling.set_lasttileLength(lasttileLength);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class MseLossCustom : public OpDef {
public:
    explicit MseLossCustom(const char* name) : OpDef(name)
    {
        this->Input("predict")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("label")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MseLossCustom);
}
"""

kernel_src="""

#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 1;

class KernelMseLoss {
public:
    __aicore__ inline KernelMseLoss() {}
    __aicore__ inline void Init(GM_ADDR predict, GM_ADDR label, GM_ADDR y,
                                uint32_t totalLength, uint32_t blockLength,
                                uint32_t tileNum, uint32_t tileLength,
                                uint32_t lasttileLength) {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");

        this->totalLength = static_cast<int32_t>(totalLength);
        this->totalLength_f32 = static_cast<float>(this->totalLength);

        this->blockLength = blockLength;
        this->tileNum = tileNum;
        this->tileLength = tileLength / BUFFER_NUM;
        this->lasttileLength = lasttileLength;

        xGm.SetGlobalBuffer((__gm__ float*)predict + this->blockLength * AscendC::GetBlockIdx(),
                            this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float*)label + this->blockLength * AscendC::GetBlockIdx(),
                            this->blockLength);
        outGm.SetGlobalBuffer(
            (__gm__ float*)y + this->blockLength * AscendC::GetBlockIdx(), 32);

        this->reduce_num = this->tileNum * BUFFER_NUM;
        uint32_t reduce_align = (this->reduce_num + 31) / 32 * 32;

        pipe.InitBuffer(inQueueIN, BUFFER_NUM, this->tileLength * 2 * sizeof(float));
        pipe.InitBuffer(tempBuf, reduce_align * sizeof(float));
    }

    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
        }
        AscendC::LocalTensor<float> temp1 = tempBuf.Get<float>();
        AscendC::LocalTensor<float> temp2 = inQueueIN.AllocTensor<float>();

        AscendC::Duplicate(temp2, (float)0, this->tileLength);
        AscendC::ReduceSum<float>(temp1, temp1, temp2, this->reduce_num);

        float len = static_cast<float>(this->totalLength_f32);
        temp2.SetValue(0, len);
        AscendC::Div(temp1, temp1, temp2, 1);

        outGm.SetValue(0, temp1.GetValue(0));
        inQueueIN.FreeTensor(temp2);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        AscendC::LocalTensor<float> inLocal = inQueueIN.AllocTensor<float>();

        if (progress == this->tileNum - 1) {
            if (progress == 0) {
                AscendC::DataCopy(inLocal[0], xGm[0], this->tileLength);
                AscendC::DataCopy(inLocal[this->tileLength], yGm[0], this->tileLength);
            }
            else {
                AscendC::DataCopy(
                    inLocal[0],
                    xGm[(progress - 1) * this->tileLength + this->lasttileLength],
                    this->tileLength);
                AscendC::DataCopy(
                    inLocal[this->tileLength],
                    yGm[(progress - 1) * this->tileLength + this->lasttileLength],
                    this->tileLength);
            }
        }
        else {
            AscendC::DataCopy(inLocal[0], xGm[progress * this->tileLength],
                    this->tileLength);
            AscendC::DataCopy(inLocal[this->tileLength], yGm[progress * this->tileLength],
                    this->tileLength);
        }
        inQueueIN.EnQue(inLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        AscendC::LocalTensor<float> inLocal = inQueueIN.DeQue<float>();
        AscendC::LocalTensor<float> xLocal = inLocal;
        AscendC::LocalTensor<float> yLocal = inLocal[this->tileLength];
        AscendC::LocalTensor<float> temp1 = tempBuf.Get<float>();

        AscendC::Sub(yLocal, xLocal, yLocal, this->tileLength);
        AscendC::Mul(yLocal, yLocal, yLocal, this->tileLength);
        AscendC::ReduceSum<float>(yLocal, yLocal, xLocal, this->tileLength);
        temp1.SetValue(progress, yLocal.GetValue(0));

        inQueueIN.FreeTensor(inLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueIN;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueOUT;
    AscendC::TBuf<> tempBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> outGm;
    float totalLength_f32;
    int32_t totalLength;
    uint32_t reduce_num;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lasttileLength;
};

extern "C" __global__ __aicore__ void mse_loss_custom(GM_ADDR predict, GM_ADDR label, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMseLoss op;

    op.Init(predict, label, y, tiling_data.totalLength, tiling_data.blockLength,
            tiling_data.tileNum, tiling_data.tileLength, tiling_data.lasttileLength);
    op.Process();
}

"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor mse_loss_impl_npu(const at::Tensor& predict, const at::Tensor& label) {
    at::Tensor result = at::empty({}, predict.options());
    EXEC_NPU_CMD(aclnnMseLossCustom, predict, label, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("mse_loss_custom", &mse_loss_impl_npu);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mse_loss_custom", &mse_loss_impl_npu, "mse loss");
}

"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.mse_loss_custom(predictions, targets)
'''
