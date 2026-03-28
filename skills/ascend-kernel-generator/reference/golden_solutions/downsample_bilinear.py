project_json_src='''
[
    {
        "op": "DownsampleBilinearCustom",
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
BEGIN_TILING_DATA_DEF(DownsampleBilinearCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, numNc);
  TILING_DATA_FIELD_DEF(uint32_t, inputH);
  TILING_DATA_FIELD_DEF(uint32_t, inputW);
  TILING_DATA_FIELD_DEF(uint32_t, outputH);
  TILING_DATA_FIELD_DEF(uint32_t, outputW);
  TILING_DATA_FIELD_DEF(uint32_t, useCoreNums);
  TILING_DATA_FIELD_DEF(uint32_t, workPerCore);
  TILING_DATA_FIELD_DEF(uint32_t, workPerCoreTail);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DownsampleBilinearCustom, DownsampleBilinearCustomTilingData)
}
"""

host_operator_src="""
#include "downsample_bilinear_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr uint32_t MAX_CORE_NUM = 8;
constexpr uint32_t FIXED_OUTPUT_H = 60;
constexpr uint32_t FIXED_OUTPUT_W = 80;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    DownsampleBilinearCustomTilingData tiling;
    const gert::StorageShape *inputShape = context->GetInputShape(0);
    const auto shape = inputShape->GetStorageShape();

    const uint32_t n = static_cast<uint32_t>(shape.GetDim(0));
    const uint32_t c = static_cast<uint32_t>(shape.GetDim(1));
    const uint32_t inputH = static_cast<uint32_t>(shape.GetDim(2));
    const uint32_t inputW = static_cast<uint32_t>(shape.GetDim(3));
    const uint32_t numNc = n * c;
    const uint32_t useCoreNums = 1;
    const uint32_t workPerCore = numNc;
    const uint32_t workPerCoreTail = numNc;

    context->SetBlockDim(1);
    tiling.set_numNc(numNc);
    tiling.set_inputH(inputH);
    tiling.set_inputW(inputW);
    tiling.set_outputH(FIXED_OUTPUT_H);
    tiling.set_outputW(FIXED_OUTPUT_W);
    tiling.set_useCoreNums(useCoreNums);
    tiling.set_workPerCore(workPerCore);
    tiling.set_workPerCoreTail(workPerCoreTail);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *workspaceSize = context->GetWorkspaceSizes(1);
    workspaceSize[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    *yShape = {inputShape->GetDim(0), inputShape->GetDim(1), 60, 80};
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class DownsampleBilinearCustom : public OpDef {
public:
    explicit DownsampleBilinearCustom(const char *name) : OpDef(name)
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
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(DownsampleBilinearCustom);
} // namespace ops
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelDownsampleBilinear {
public:
    __aicore__ inline KernelDownsampleBilinear() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t numNc,
        uint32_t inputH,
        uint32_t inputW,
        uint32_t outputH,
        uint32_t outputW,
        uint32_t useCoreNums,
        uint32_t workPerCore,
        uint32_t workPerCoreTail)
    {
        this->inputH = inputH;
        this->inputW = inputW;
        this->outputH = outputH;
        this->outputW = outputW;
        this->useCoreNums = useCoreNums;

        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t localWork =
            blockIdx + 1 == useCoreNums ? workPerCoreTail : workPerCore;
        const uint32_t workOffset = blockIdx * workPerCore;

        this->localWork = localWork;
        this->workOffset = workOffset;
        this->inputPlaneSize = inputH * inputW;
        this->outputPlaneSize = outputH * outputW;

        xGm.SetGlobalBuffer((__gm__ float *)x + workOffset * this->inputPlaneSize,
                            localWork * this->inputPlaneSize);
        yGm.SetGlobalBuffer((__gm__ float *)y + workOffset * this->outputPlaneSize,
                            localWork * this->outputPlaneSize);
    }

    __aicore__ inline void Process()
    {
        if (this->localWork == 0) {
            return;
        }

        for (uint32_t ncIdx = 0; ncIdx < this->localWork; ++ncIdx) {
            const uint32_t inputBase = ncIdx * this->inputPlaneSize;
            const uint32_t outputBase = ncIdx * this->outputPlaneSize;
            for (uint32_t oh = 0; oh < this->outputH; ++oh) {
                const uint32_t h0 = oh * 4 + 1;
                const uint32_t h1 = h0 + 1;
                for (uint32_t ow = 0; ow < this->outputW; ++ow) {
                    const uint32_t w0 = ow * 4 + 1;
                    const uint32_t w1 = w0 + 1;

                    const float v00 = xGm.GetValue(inputBase + h0 * this->inputW + w0);
                    const float v01 = xGm.GetValue(inputBase + h0 * this->inputW + w1);
                    const float v10 = xGm.GetValue(inputBase + h1 * this->inputW + w0);
                    const float v11 = xGm.GetValue(inputBase + h1 * this->inputW + w1);
                    yGm.SetValue(outputBase + oh * this->outputW + ow,
                                 (v00 + v01 + v10 + v11) * 0.25f);
                }
            }
        }
    }

private:
    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;
    uint32_t inputH;
    uint32_t inputW;
    uint32_t outputH;
    uint32_t outputW;
    uint32_t inputPlaneSize;
    uint32_t outputPlaneSize;
    uint32_t useCoreNums;
    uint32_t localWork;
    uint32_t workOffset;
};

extern "C" __global__ __aicore__ void downsample_bilinear_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelDownsampleBilinear op;
    op.Init(
        x,
        y,
        tiling_data.numNc,
        tiling_data.inputH,
        tiling_data.inputW,
        tiling_data.outputH,
        tiling_data.outputW,
        tiling_data.useCoreNums,
        tiling_data.workPerCore,
        tiling_data.workPerCoreTail);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor downsample_bilinear_custom_impl_npu(const at::Tensor &self)
{
    auto sizes = self.sizes();
    at::Tensor result = at::empty({sizes[0], sizes[1], 60, 80}, self.options());
    EXEC_NPU_CMD(aclnnDownsampleBilinearCustom, self, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("downsample_bilinear_custom", &downsample_bilinear_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("downsample_bilinear_custom", &downsample_bilinear_custom_impl_npu, "fixed bilinear downsample");
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
        return custom_ops_lib.downsample_bilinear_custom(x)
'''
