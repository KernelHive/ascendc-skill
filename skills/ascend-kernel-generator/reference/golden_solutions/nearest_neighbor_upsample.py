project_json_src='''
[
    {
        "op": "NearestNeighborUpsampleCustom",
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
BEGIN_TILING_DATA_DEF(NearestNeighborUpsampleCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, numNc);
    TILING_DATA_FIELD_DEF(uint32_t, inHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inWidth);
    TILING_DATA_FIELD_DEF(uint32_t, outHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outWidth);
    TILING_DATA_FIELD_DEF(uint32_t, useCoreNums);
    TILING_DATA_FIELD_DEF(uint32_t, workPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, workPerCoreTail);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(NearestNeighborUpsampleCustom, NearestNeighborUpsampleCustomTilingData)
}
"""

host_operator_src="""
#include <cstdint>
#include "nearest_neighbor_upsample_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& shape = inputShape->GetStorageShape();
    if (shape.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batch = static_cast<uint32_t>(shape.GetDim(0));
    const uint32_t channels = static_cast<uint32_t>(shape.GetDim(1));
    const uint32_t inHeight = static_cast<uint32_t>(shape.GetDim(2));
    const uint32_t inWidth = static_cast<uint32_t>(shape.GetDim(3));
    if (inHeight == 0 || inWidth == 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t numNc = batch * channels;
    const uint32_t outHeight = inHeight * 4;
    const uint32_t outWidth = inWidth * 4;
    const uint32_t useCoreNums = 1;
    const uint32_t workPerCore = numNc;
    const uint32_t workPerCoreTail = numNc;

    NearestNeighborUpsampleCustomTilingData tiling;
    tiling.set_numNc(numNc);
    tiling.set_inHeight(inHeight);
    tiling.set_inWidth(inWidth);
    tiling.set_outHeight(outHeight);
    tiling.set_outWidth(outWidth);
    tiling.set_useCoreNums(useCoreNums);
    tiling.set_workPerCore(workPerCore);
    tiling.set_workPerCoreTail(workPerCoreTail);

    context->SetBlockDim(1);
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
    const gert::Shape* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr || inputShape->GetDimNum() != 4) {
        return GRAPH_FAILED;
    }

    gert::Shape* outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(4);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, inputShape->GetDim(1));
    outputShape->SetDim(2, inputShape->GetDim(2) * 4);
    outputShape->SetDim(3, inputShape->GetDim(3) * 4);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class NearestNeighborUpsampleCustom : public OpDef {
public:
    explicit NearestNeighborUpsampleCustom(const char* name) : OpDef(name)
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

OP_ADD(NearestNeighborUpsampleCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

class KernelNearestNeighborUpsample {
public:
    __aicore__ inline KernelNearestNeighborUpsample() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t numNc,
        uint32_t inHeight,
        uint32_t inWidth,
        uint32_t outHeight,
        uint32_t outWidth,
        uint32_t useCoreNums,
        uint32_t workPerCore,
        uint32_t workPerCoreTail)
    {
        this->inHeight = inHeight;
        this->inWidth = inWidth;
        this->outHeight = outHeight;
        this->outWidth = outWidth;
        this->useCoreNums = useCoreNums;
        this->inputPlane = inHeight * inWidth;
        this->outputPlane = outHeight * outWidth;

        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t localWork = blockIdx + 1 == useCoreNums ? workPerCoreTail : workPerCore;
        const uint32_t workOffset = blockIdx * workPerCore;

        this->localWork = localWork;
        this->workOffset = workOffset;

        xGm.SetGlobalBuffer((__gm__ float*)x + workOffset * inputPlane, localWork * inputPlane);
        yGm.SetGlobalBuffer((__gm__ float*)y + workOffset * outputPlane, localWork * outputPlane);
    }

    __aicore__ inline void Process()
    {
        if (localWork == 0) {
            return;
        }

        for (uint32_t ncIdx = 0; ncIdx < localWork; ++ncIdx) {
            ProcessPlane(ncIdx);
        }
    }

private:
    __aicore__ inline void ProcessPlane(uint32_t ncIdx)
    {
        const uint32_t inputBase = ncIdx * inputPlane;
        const uint32_t outputBase = ncIdx * outputPlane;

        for (uint32_t outY = 0; outY < outHeight; ++outY) {
            const uint32_t inY = outY >> 2;
            const uint32_t inputRowBase = inputBase + inY * inWidth;
            const uint32_t outputRowBase = outputBase + outY * outWidth;
            for (uint32_t outX = 0; outX < outWidth; ++outX) {
                const uint32_t inX = outX >> 2;
                yGm.SetValue(outputRowBase + outX, xGm.GetValue(inputRowBase + inX));
            }
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t inHeight;
    uint32_t inWidth;
    uint32_t outHeight;
    uint32_t outWidth;
    uint32_t useCoreNums;
    uint32_t localWork;
    uint32_t workOffset;
    uint32_t inputPlane;
    uint32_t outputPlane;
};

extern "C" __global__ __aicore__ void nearest_neighbor_upsample_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    KernelNearestNeighborUpsample op;
    op.Init(
        x,
        y,
        tilingData.numNc,
        tilingData.inHeight,
        tilingData.inWidth,
        tilingData.outHeight,
        tilingData.outWidth,
        tilingData.useCoreNums,
        tilingData.workPerCore,
        tilingData.workPerCoreTail);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor nearest_neighbor_upsample_impl_npu(const at::Tensor& self) {
    const int64_t outHeight = self.size(2) * 4;
    const int64_t outWidth = self.size(3) * 4;
    at::Tensor result = at::empty({self.size(0), self.size(1), outHeight, outWidth}, self.options());
    EXEC_NPU_CMD(aclnnNearestNeighborUpsampleCustom, self, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("nearest_neighbor_upsample_custom", &nearest_neighbor_upsample_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nearest_neighbor_upsample_custom", &nearest_neighbor_upsample_impl_npu, "nearest neighbor upsample x4");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return custom_ops_lib.nearest_neighbor_upsample_custom(x)
'''
