project_json_src='''
[
    {
        "op": "HingeLossCustom",
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
BEGIN_TILING_DATA_DEF(HingeLossCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, blockLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);
    TILING_DATA_FIELD_DEF(uint32_t, tileLength);
    TILING_DATA_FIELD_DEF(uint32_t, lasttileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(HingeLossCustom, HingeLossCustomTilingData)
}
"""

host_operator_src="""
#include "hinge_loss_custom_tiling.h"
#include "register/op_def_registry.h"
#include <string.h>

namespace optiling {
const uint32_t BLOCK_SIZE = 32;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    HingeLossCustomTilingData tiling;
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
        totalLengthAligned = ((totalLength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
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
        } else {
            tileLength = ub_block_num * ALIGN_NUM;
            lasttileLength = tileLength;
        }
    } else {
        tile_num = tile_num + 1;
        tileLength = ub_block_num * ALIGN_NUM;
        lasttileLength = blockLength - (tile_num - 1) * tileLength;
    }

    tiling.set_blockLength(blockLength);
    tiling.set_tileNum(tile_num);
    tiling.set_tileLength(tileLength);
    tiling.set_lasttileLength(lasttileLength);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class HingeLossCustom : public OpDef {
public:
    explicit HingeLossCustom(const char* name) : OpDef(name)
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
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(HingeLossCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 1;

class KernelHingeLoss {
public:
    __aicore__ inline KernelHingeLoss() {}

    __aicore__ inline void Init(GM_ADDR predict, GM_ADDR label, GM_ADDR y,
                                uint32_t totalLength, uint32_t blockLength,
                                uint32_t tileNum, uint32_t tileLength,
                                uint32_t lasttileLength)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");

        this->totalLength = static_cast<int32_t>(totalLength);
        this->totalLengthF32 = static_cast<float>(this->totalLength);
        this->blockLength = blockLength;
        this->tileNum = tileNum;
        this->tileLength = tileLength / BUFFER_NUM;
        this->lasttileLength = lasttileLength;

        predictGm.SetGlobalBuffer((__gm__ float*)predict + this->blockLength * AscendC::GetBlockIdx(),
                                  this->blockLength);
        labelGm.SetGlobalBuffer((__gm__ float*)label + this->blockLength * AscendC::GetBlockIdx(),
                                this->blockLength);
        outGm.SetGlobalBuffer((__gm__ float*)y + this->blockLength * AscendC::GetBlockIdx(), 32);

        this->reduceNum = this->tileNum * BUFFER_NUM;
        uint32_t reduceAlign = (this->reduceNum + 31) / 32 * 32;

        pipe.InitBuffer(inQueue, BUFFER_NUM, this->tileLength * 2 * sizeof(float));
        pipe.InitBuffer(tempBuf, reduceAlign * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
        }

        AscendC::LocalTensor<float> temp = tempBuf.Get<float>();
        AscendC::LocalTensor<float> scalarBuf = inQueue.AllocTensor<float>();

        AscendC::Duplicate(scalarBuf, 0.0f, this->tileLength);
        AscendC::ReduceSum<float>(temp, temp, scalarBuf, this->reduceNum);

        scalarBuf.SetValue(0, this->totalLengthF32);
        AscendC::Div(temp, temp, scalarBuf, 1);

        outGm.SetValue(0, temp.GetValue(0));
        inQueue.FreeTensor(scalarBuf);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<float> inLocal = inQueue.AllocTensor<float>();

        if (progress == this->tileNum - 1) {
            if (progress == 0) {
                AscendC::DataCopy(inLocal[0], predictGm[0], this->tileLength);
                AscendC::DataCopy(inLocal[this->tileLength], labelGm[0], this->tileLength);
            } else {
                AscendC::DataCopy(
                    inLocal[0],
                    predictGm[(progress - 1) * this->tileLength + this->lasttileLength],
                    this->tileLength);
                AscendC::DataCopy(
                    inLocal[this->tileLength],
                    labelGm[(progress - 1) * this->tileLength + this->lasttileLength],
                    this->tileLength);
            }
        } else {
            AscendC::DataCopy(inLocal[0], predictGm[progress * this->tileLength], this->tileLength);
            AscendC::DataCopy(inLocal[this->tileLength], labelGm[progress * this->tileLength], this->tileLength);
        }

        inQueue.EnQue(inLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> inLocal = inQueue.DeQue<float>();
        AscendC::LocalTensor<float> predictLocal = inLocal;
        AscendC::LocalTensor<float> labelLocal = inLocal[this->tileLength];
        AscendC::LocalTensor<float> temp = tempBuf.Get<float>();
        constexpr float kNegOne = -1.0f;
        constexpr float kOne = 1.0f;
        constexpr float kZero = 0.0f;

        AscendC::Mul(predictLocal, predictLocal, labelLocal, this->tileLength);
        AscendC::Muls(predictLocal, predictLocal, kNegOne, this->tileLength);
        AscendC::Adds(predictLocal, predictLocal, kOne, this->tileLength);
        AscendC::Maxs(predictLocal, predictLocal, kZero, this->tileLength);
        AscendC::ReduceSum<float>(predictLocal, predictLocal, labelLocal, this->tileLength);
        temp.SetValue(progress, predictLocal.GetValue(0));

        inQueue.FreeTensor(inLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TBuf<> tempBuf;
    AscendC::GlobalTensor<float> predictGm;
    AscendC::GlobalTensor<float> labelGm;
    AscendC::GlobalTensor<float> outGm;
    float totalLengthF32;
    int32_t totalLength;
    uint32_t reduceNum;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lasttileLength;
};

extern "C" __global__ __aicore__ void hinge_loss_custom(
    GM_ADDR predict,
    GM_ADDR label,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelHingeLoss op;
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

at::Tensor hinge_loss_impl_npu(const at::Tensor& predict, const at::Tensor& label)
{
    at::Tensor result = at::empty({}, predict.options());
    EXEC_NPU_CMD(aclnnHingeLossCustom, predict, label, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("hinge_loss_custom", &hinge_loss_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hinge_loss_custom", &hinge_loss_impl_npu, "hinge loss");
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
        return custom_ops_lib.hinge_loss_custom(predictions, targets)
'''
