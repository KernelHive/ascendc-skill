project_json_src='''
[
    {
        "op": "CosineSimilarityLossCustom",
        "input_desc": [
            {
                "name": "predictions",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            },
            {
                "name": "targets",
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
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CosineSimilarityLossCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, featureLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);
    TILING_DATA_FIELD_DEF(uint32_t, tileLength);
    TILING_DATA_FIELD_DEF(uint32_t, lastTileLength);
    TILING_DATA_FIELD_DEF(float, batchReciprocal);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CosineSimilarityLossCustom, CosineSimilarityLossCustomTilingData)
}
"""

host_operator_src="""
#include "cosine_similarity_loss_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    CosineSimilarityLossCustomTilingData tiling;
    auto inputShape = context->GetInputShape(0);
    uint32_t dimNum = inputShape->GetStorageShape().GetDimNum();
    uint32_t batchSize = 1;
    if (dimNum > 0) {
        batchSize = static_cast<uint32_t>(inputShape->GetStorageShape().GetDim(0));
    }

    uint32_t totalLength = static_cast<uint32_t>(inputShape->GetStorageShape().GetShapeSize());
    uint32_t featureLength = batchSize == 0 ? totalLength : totalLength / batchSize;

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
class CosineSimilarityLossCustom : public OpDef {
public:
    explicit CosineSimilarityLossCustom(const char* name) : OpDef(name)
    {
        this->Input("predictions")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("targets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("loss")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(CosineSimilarityLossCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"
#include <math.h>

constexpr int32_t BUFFER_NUM = 1;

class KernelCosineSimilarityLoss {
public:
    __aicore__ inline KernelCosineSimilarityLoss() {}

    __aicore__ inline void Init(
        GM_ADDR predictions,
        GM_ADDR targets,
        GM_ADDR loss,
        uint32_t batchSize,
        uint32_t featureLength,
        uint32_t tileNum,
        uint32_t tileLength,
        uint32_t lastTileLength,
        float batchReciprocal
    ) {
        this->batchSize = batchSize;
        this->featureLength = featureLength;
        this->tileNum = tileNum;
        this->tileLength = tileLength;
        this->lastTileLength = lastTileLength;
        this->batchReciprocal = batchReciprocal;
        this->epsilon = 1e-12f;

        predictionsGm.SetGlobalBuffer((__gm__ float*)predictions, batchSize * featureLength);
        targetsGm.SetGlobalBuffer((__gm__ float*)targets, batchSize * featureLength);
        lossGm.SetGlobalBuffer((__gm__ float*)loss, 1);

        pipe.InitBuffer(inQueue, BUFFER_NUM, this->tileLength * 2 * sizeof(float));
        pipe.InitBuffer(calcBuf, this->tileLength * 2 * sizeof(float));
    }

    __aicore__ inline void Process() {
        float lossAcc = 0.0f;
        AscendC::LocalTensor<float> scratch0 = calcBuf.Get<float>();
        for (uint32_t row = 0; row < this->batchSize; ++row) {
            float dotAcc = 0.0f;
            float predNormAcc = 0.0f;
            float targetNormAcc = 0.0f;

            for (uint32_t tileIdx = 0; tileIdx < this->tileNum; ++tileIdx) {
                uint32_t currentLength = tileIdx + 1 == this->tileNum ? this->lastTileLength : this->tileLength;
                CopyIn(row, tileIdx, currentLength);
                Compute(currentLength, dotAcc, predNormAcc, targetNormAcc);
            }

            float denom = predNormAcc * targetNormAcc;
            if (denom < this->epsilon) {
                denom = this->epsilon;
            }
            scratch0.SetValue(0, denom);
            AscendC::Sqrt(scratch0, scratch0, 1);
            float cosine = dotAcc / scratch0.GetValue(0);
            lossAcc += (1.0f - cosine);
        }
        lossGm.SetValue(0, lossAcc * this->batchReciprocal);
    }

private:
    __aicore__ inline void CopyIn(uint32_t row, uint32_t tileIdx, uint32_t currentLength) {
        AscendC::LocalTensor<float> local = inQueue.AllocTensor<float>();
        uint32_t offset = row * this->featureLength + tileIdx * this->tileLength;
        AscendC::DataCopy(local[0], predictionsGm[offset], currentLength);
        AscendC::DataCopy(local[this->tileLength], targetsGm[offset], currentLength);
        inQueue.EnQue(local);
    }

    __aicore__ inline void Compute(
        uint32_t currentLength,
        float& dotAcc,
        float& predNormAcc,
        float& targetNormAcc
    ) {
        AscendC::LocalTensor<float> local = inQueue.DeQue<float>();
        AscendC::LocalTensor<float> predLocal = local;
        AscendC::LocalTensor<float> targetLocal = local[this->tileLength];
        AscendC::LocalTensor<float> scratch0 = calcBuf.Get<float>();
        AscendC::LocalTensor<float> scratch1 = calcBuf.Get<float>()[this->tileLength];

        AscendC::Mul(scratch0, predLocal, targetLocal, currentLength);
        AscendC::ReduceSum<float>(scratch0, scratch0, scratch1, currentLength);
        dotAcc += scratch0.GetValue(0);

        AscendC::Mul(scratch0, predLocal, predLocal, currentLength);
        AscendC::ReduceSum<float>(scratch0, scratch0, scratch1, currentLength);
        predNormAcc += scratch0.GetValue(0);

        AscendC::Mul(scratch0, targetLocal, targetLocal, currentLength);
        AscendC::ReduceSum<float>(scratch0, scratch0, scratch1, currentLength);
        targetNormAcc += scratch0.GetValue(0);

        inQueue.FreeTensor(local);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TBuf<> calcBuf;
    AscendC::GlobalTensor<float> predictionsGm;
    AscendC::GlobalTensor<float> targetsGm;
    AscendC::GlobalTensor<float> lossGm;
    uint32_t batchSize;
    uint32_t featureLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lastTileLength;
    float batchReciprocal;
    float epsilon;
};

extern "C" __global__ __aicore__ void cosine_similarity_loss_custom(
    GM_ADDR predictions,
    GM_ADDR targets,
    GM_ADDR loss,
    GM_ADDR workspace,
    GM_ADDR tiling
) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelCosineSimilarityLoss op;
    op.Init(
        predictions,
        targets,
        loss,
        tiling_data.batchSize,
        tiling_data.featureLength,
        tiling_data.tileNum,
        tiling_data.tileLength,
        tiling_data.lastTileLength,
        tiling_data.batchReciprocal
    );
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor cosine_similarity_loss_impl_npu(const at::Tensor& predictions, const at::Tensor& targets) {
    at::Tensor result = at::empty({}, predictions.options());
    EXEC_NPU_CMD(aclnnCosineSimilarityLossCustom, predictions, targets, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("cosine_similarity_loss_custom", &cosine_similarity_loss_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cosine_similarity_loss_custom", &cosine_similarity_loss_impl_npu, "cosine similarity loss");
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
        return custom_ops_lib.cosine_similarity_loss_custom(predictions, targets)
'''
