project_json_src='''
[
    {
        "op": "TripletMarginLossCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "anchor",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            },
            {
                "name": "positive",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            },
            {
                "name": "negative",
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
                "name": "loss",
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
                "name": "margin",
                "param_type": "optional",
                "type": "float",
                "default_value": "1.0"
            }
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TripletMarginLossCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, featureLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);
    TILING_DATA_FIELD_DEF(uint32_t, tileLength);
    TILING_DATA_FIELD_DEF(uint32_t, lastTileLength);
    TILING_DATA_FIELD_DEF(float, batchReciprocal);
    TILING_DATA_FIELD_DEF(float, margin);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TripletMarginLossCustom, TripletMarginLossCustomTilingData)
}
"""

host_operator_src="""
#include "triplet_margin_loss_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TripletMarginLossCustomTilingData tiling;
    auto inputShape = context->GetInputShape(0);
    uint32_t dimNum = inputShape->GetStorageShape().GetDimNum();
    uint32_t batchSize = 1;
    if (dimNum > 0) {
        batchSize = static_cast<uint32_t>(inputShape->GetStorageShape().GetDim(0));
    }

    uint32_t totalLength = static_cast<uint32_t>(inputShape->GetStorageShape().GetShapeSize());
    uint32_t featureLength = batchSize == 0 ? totalLength : totalLength / batchSize;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const float* margin = attrs->GetAttrPointer<float>(0);

    constexpr uint32_t ALIGN_NUM = 8;
    constexpr uint32_t DEFAULT_TILE = 256;
    uint32_t tileLength = featureLength < DEFAULT_TILE ? featureLength : DEFAULT_TILE;
    if (tileLength == 0) {
        tileLength = ALIGN_NUM;
    }
    if (tileLength % ALIGN_NUM != 0) {
        tileLength = ((tileLength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
    }
    if (tileLength > featureLength && featureLength != 0) {
        tileLength = featureLength;
    }

    uint32_t tileNum = featureLength == 0 ? 1 : (featureLength + tileLength - 1) / tileLength;
    uint32_t lastTileLength = featureLength == 0 ? tileLength : featureLength - (tileNum - 1) * tileLength;
    if (lastTileLength == 0) {
        lastTileLength = tileLength;
    }

    context->SetBlockDim(1);
    tiling.set_batchSize(batchSize);
    tiling.set_featureLength(featureLength);
    tiling.set_tileNum(tileNum);
    tiling.set_tileLength(tileLength);
    tiling.set_lastTileLength(lastTileLength);
    tiling.set_batchReciprocal(batchSize == 0 ? 1.0f : 1.0f / static_cast<float>(batchSize));
    tiling.set_margin(*margin);

    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity()
    );
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    gert::Shape* outputShape = context->GetOutputShape(0);
    *outputShape = gert::Shape();
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class TripletMarginLossCustom : public OpDef {
public:
    explicit TripletMarginLossCustom(const char* name) : OpDef(name)
    {
        this->Input("anchor")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("positive")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("negative")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("loss")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("margin").AttrType(OPTIONAL).Float(1.0f);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(TripletMarginLossCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelTripletMarginLoss {
public:
    __aicore__ inline KernelTripletMarginLoss() {}

    __aicore__ inline void Init(
        GM_ADDR anchor,
        GM_ADDR positive,
        GM_ADDR negative,
        GM_ADDR loss,
        uint32_t batchSize,
        uint32_t featureLength,
        uint32_t tileNum,
        uint32_t tileLength,
        uint32_t lastTileLength,
        float batchReciprocal,
        float margin
    ) {
        this->batchSize = batchSize;
        this->featureLength = featureLength;
        this->tileNum = tileNum;
        this->tileLength = tileLength;
        this->lastTileLength = lastTileLength;
        this->batchReciprocal = batchReciprocal;
        this->margin = margin;
        this->epsilon = 1e-6f;

        anchorGm.SetGlobalBuffer((__gm__ float*)anchor, batchSize * featureLength);
        positiveGm.SetGlobalBuffer((__gm__ float*)positive, batchSize * featureLength);
        negativeGm.SetGlobalBuffer((__gm__ float*)negative, batchSize * featureLength);
        lossGm.SetGlobalBuffer((__gm__ float*)loss, 1);

        pipe.InitBuffer(inQueue, BUFFER_NUM, this->tileLength * 3 * sizeof(float));
        pipe.InitBuffer(calcBuf, this->tileLength * 2 * sizeof(float));
    }

    __aicore__ inline void Process() {
        float lossAcc = 0.0f;
        AscendC::LocalTensor<float> scratch0 = calcBuf.Get<float>();
        for (uint32_t row = 0; row < this->batchSize; ++row) {
            float apAcc = 0.0f;
            float anAcc = 0.0f;

            for (uint32_t tileIdx = 0; tileIdx < this->tileNum; ++tileIdx) {
                uint32_t currentLength = tileIdx + 1 == this->tileNum ? this->lastTileLength : this->tileLength;
                CopyIn(row, tileIdx, currentLength);
                Compute(currentLength, apAcc, anAcc);
            }

            scratch0.SetValue(0, apAcc);
            AscendC::Sqrt(scratch0, scratch0, 1);
            float distAp = scratch0.GetValue(0);

            scratch0.SetValue(0, anAcc);
            AscendC::Sqrt(scratch0, scratch0, 1);
            float distAn = scratch0.GetValue(0);

            float rowLoss = distAp - distAn + this->margin;
            if (rowLoss < 0.0f) {
                rowLoss = 0.0f;
            }
            lossAcc += rowLoss;
        }
        lossGm.SetValue(0, lossAcc * this->batchReciprocal);
    }

private:
    __aicore__ inline void CopyIn(uint32_t row, uint32_t tileIdx, uint32_t currentLength) {
        AscendC::LocalTensor<float> local = inQueue.AllocTensor<float>();
        uint32_t offset = row * this->featureLength + tileIdx * this->tileLength;
        AscendC::DataCopy(local[0], anchorGm[offset], currentLength);
        AscendC::DataCopy(local[this->tileLength], positiveGm[offset], currentLength);
        AscendC::DataCopy(local[this->tileLength * 2], negativeGm[offset], currentLength);
        inQueue.EnQue(local);
    }

    __aicore__ inline void Compute(uint32_t currentLength, float& apAcc, float& anAcc) {
        AscendC::LocalTensor<float> local = inQueue.DeQue<float>();
        AscendC::LocalTensor<float> anchorLocal = local;
        AscendC::LocalTensor<float> positiveLocal = local[this->tileLength];
        AscendC::LocalTensor<float> negativeLocal = local[this->tileLength * 2];
        AscendC::LocalTensor<float> scratch0 = calcBuf.Get<float>();
        AscendC::LocalTensor<float> scratch1 = calcBuf.Get<float>()[this->tileLength];

        AscendC::Sub(scratch0, anchorLocal, positiveLocal, currentLength);
        AscendC::Abs(scratch0, scratch0, currentLength);
        AscendC::Adds(scratch0, scratch0, this->epsilon, currentLength);
        AscendC::Mul(scratch0, scratch0, scratch0, currentLength);
        AscendC::ReduceSum<float>(scratch0, scratch0, scratch1, currentLength);
        apAcc += scratch0.GetValue(0);

        AscendC::Sub(scratch0, anchorLocal, negativeLocal, currentLength);
        AscendC::Abs(scratch0, scratch0, currentLength);
        AscendC::Adds(scratch0, scratch0, this->epsilon, currentLength);
        AscendC::Mul(scratch0, scratch0, scratch0, currentLength);
        AscendC::ReduceSum<float>(scratch0, scratch0, scratch1, currentLength);
        anAcc += scratch0.GetValue(0);

        inQueue.FreeTensor(local);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TBuf<> calcBuf;
    AscendC::GlobalTensor<float> anchorGm;
    AscendC::GlobalTensor<float> positiveGm;
    AscendC::GlobalTensor<float> negativeGm;
    AscendC::GlobalTensor<float> lossGm;
    uint32_t batchSize;
    uint32_t featureLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lastTileLength;
    float batchReciprocal;
    float margin;
    float epsilon;
};

extern "C" __global__ __aicore__ void triplet_margin_loss_custom(
    GM_ADDR anchor,
    GM_ADDR positive,
    GM_ADDR negative,
    GM_ADDR loss,
    GM_ADDR workspace,
    GM_ADDR tiling
) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelTripletMarginLoss op;
    op.Init(
        anchor,
        positive,
        negative,
        loss,
        tiling_data.batchSize,
        tiling_data.featureLength,
        tiling_data.tileNum,
        tiling_data.tileLength,
        tiling_data.lastTileLength,
        tiling_data.batchReciprocal,
        tiling_data.margin
    );
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor triplet_margin_loss_impl_npu(
    const at::Tensor& anchor,
    const at::Tensor& positive,
    const at::Tensor& negative,
    double margin
) {
    at::Tensor result = at::empty({}, anchor.options());
    EXEC_NPU_CMD(aclnnTripletMarginLossCustom, anchor, positive, negative, margin, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("triplet_margin_loss_custom", &triplet_margin_loss_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("triplet_margin_loss_custom", &triplet_margin_loss_impl_npu, "triplet margin loss");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        return custom_ops_lib.triplet_margin_loss_custom(anchor, positive, negative, self.margin)
'''
