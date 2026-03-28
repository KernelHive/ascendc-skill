project_json_src='''
[
    {
        "op": "EmbeddingCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "weight",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            },
            {
                "name": "indices",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "int64"
                ]
            }
        ],
        "output_desc": [
            {
                "name": "output",
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
#include <cstdint>
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(EmbeddingCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, vocabSize);
    TILING_DATA_FIELD_DEF(uint32_t, embeddingDim);
    TILING_DATA_FIELD_DEF(uint32_t, totalTokens);
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(EmbeddingCustom, EmbeddingCustomTilingData)
}
"""

host_operator_src="""
#include <cstdint>
#include "embedding_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* weightShape = context->GetInputShape(0);
    const gert::StorageShape* indicesShape = context->GetInputShape(1);
    if (weightShape == nullptr || indicesShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& weightStorage = weightShape->GetStorageShape();
    const auto& indicesStorage = indicesShape->GetStorageShape();
    if (weightStorage.GetDimNum() != 2 || indicesStorage.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t vocabSize = static_cast<uint32_t>(weightStorage.GetDim(0));
    const uint32_t embeddingDim = static_cast<uint32_t>(weightStorage.GetDim(1));
    const uint32_t totalTokens = static_cast<uint32_t>(indicesStorage.GetShapeSize());
    if (vocabSize == 0 || embeddingDim == 0 || totalTokens == 0) {
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t blockDim = ascendcPlatform.GetCoreNumAiv();
    if (blockDim == 0) {
        blockDim = 1;
    }
    if (totalTokens < blockDim) {
        blockDim = totalTokens;
    }
    if (blockDim == 0) {
        blockDim = 1;
    }

    EmbeddingCustomTilingData tiling;
    tiling.set_vocabSize(vocabSize);
    tiling.set_embeddingDim(embeddingDim);
    tiling.set_totalTokens(totalTokens);
    tiling.set_blockDim(blockDim);

    context->SetBlockDim(blockDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* weightShape = context->GetInputShape(0);
    const gert::Shape* indicesShape = context->GetInputShape(1);
    if (weightShape == nullptr || indicesShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (weightShape->GetDimNum() != 2 || indicesShape->GetDimNum() != 2) {
        return GRAPH_FAILED;
    }

    gert::Shape* outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(3);
    outputShape->SetDim(0, indicesShape->GetDim(0));
    outputShape->SetDim(1, indicesShape->GetDim(1));
    outputShape->SetDim(2, weightShape->GetDim(1));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class EmbeddingCustom : public OpDef {
public:
    explicit EmbeddingCustom(const char* name) : OpDef(name)
    {
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(EmbeddingCustom);
}
"""

kernel_src="""
#include <cstdint>
#include "kernel_operator.h"

class KernelEmbedding {
public:
    __aicore__ inline KernelEmbedding() {}

    __aicore__ inline void Init(
        GM_ADDR weight,
        GM_ADDR indices,
        GM_ADDR output,
        uint32_t vocabSize,
        uint32_t embeddingDim,
        uint32_t totalTokens,
        uint32_t blockDim)
    {
        this->vocabSize = vocabSize;
        this->embeddingDim = embeddingDim;
        this->totalTokens = totalTokens;
        this->blockDim = blockDim;

        weightGm.SetGlobalBuffer((__gm__ DTYPE_WEIGHT*)weight, static_cast<uint64_t>(vocabSize) * embeddingDim);
        indicesGm.SetGlobalBuffer((__gm__ DTYPE_INDICES*)indices, totalTokens);
        outputGm.SetGlobalBuffer((__gm__ DTYPE_OUTPUT*)output, static_cast<uint64_t>(totalTokens) * embeddingDim);
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        for (uint32_t tokenIdx = blockIdx; tokenIdx < totalTokens; tokenIdx += blockDim) {
            const int64_t rawIndex = static_cast<int64_t>(indicesGm.GetValue(tokenIdx));
            if (rawIndex < 0 || rawIndex >= static_cast<int64_t>(vocabSize)) {
                continue;
            }

            const uint64_t weightBase = static_cast<uint64_t>(rawIndex) * embeddingDim;
            const uint64_t outputBase = static_cast<uint64_t>(tokenIdx) * embeddingDim;
            for (uint32_t d = 0; d < embeddingDim; ++d) {
                outputGm.SetValue(outputBase + d, weightGm.GetValue(weightBase + d));
            }
        }
    }

private:
    AscendC::GlobalTensor<DTYPE_WEIGHT> weightGm;
    AscendC::GlobalTensor<DTYPE_INDICES> indicesGm;
    AscendC::GlobalTensor<DTYPE_OUTPUT> outputGm;
    uint32_t vocabSize = 0;
    uint32_t embeddingDim = 0;
    uint32_t totalTokens = 0;
    uint32_t blockDim = 1;
};

extern "C" __global__ __aicore__ void embedding_custom(
    GM_ADDR weight,
    GM_ADDR indices,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelEmbedding op;
    op.Init(
        weight,
        indices,
        output,
        tiling_data.vocabSize,
        tiling_data.embeddingDim,
        tiling_data.totalTokens,
        tiling_data.blockDim);
    op.Process();
}
"""

python_bind_src="""
#include <vector>
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor embedding_custom_impl_npu(const at::Tensor& weight, const at::Tensor& indices)
{
    std::vector<int64_t> outputShape(indices.sizes().begin(), indices.sizes().end());
    outputShape.push_back(weight.size(1));
    at::Tensor result = at::empty(outputShape, weight.options());
    EXEC_NPU_CMD(aclnnEmbeddingCustom, weight, indices, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("embedding_custom", &embedding_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("embedding_custom", &embedding_custom_impl_npu, "Embedding gather custom op");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(100000, 768)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.embedding_custom(self.embedding.weight, indices)
'''
