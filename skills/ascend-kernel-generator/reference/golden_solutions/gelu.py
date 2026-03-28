project_json_src='''
[
    {
        "op": "GeluCustom",
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
BEGIN_TILING_DATA_DEF(GeluCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM);
  TILING_DATA_FIELD_DEF(uint32_t, block_size);
  TILING_DATA_FIELD_DEF(uint32_t, core_size);
  TILING_DATA_FIELD_DEF(uint32_t, core_remain);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GeluCustom, GeluCustomTilingData)
}
"""

host_operator_src="""
#include "gelu_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint32_t kBlockSizeBytes = 32;
constexpr uint32_t kBytesPerFloat = 4;
constexpr uint32_t kAlignTileFactor = 8;
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    GeluCustomTilingData tiling;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t aivNum = ascendcPlatform.GetCoreNum();

    const uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    const uint32_t alignNum = kBlockSizeBytes / kBytesPerFloat;
    uint32_t tilingSize = static_cast<uint32_t>(ubSize / kBlockSizeBytes / 2 / 2);
    tilingSize = tilingSize <= kAlignTileFactor ? tilingSize : tilingSize / kAlignTileFactor * kAlignTileFactor;
    tilingSize = tilingSize == 0 ? kAlignTileFactor : tilingSize;

    const uint32_t blockSize = tilingSize * alignNum;
    aivNum = (aivNum < totalLength / blockSize) ? aivNum : (totalLength / blockSize);
    aivNum = aivNum >= 1 ? aivNum : 1;

    const uint32_t coreSize = (totalLength / aivNum) / (alignNum * kAlignTileFactor) * (alignNum * kAlignTileFactor);
    const uint32_t coreRemain = totalLength - aivNum * coreSize;

    tiling.set_totalLength(totalLength);
    tiling.set_ALIGN_NUM(alignNum);
    tiling.set_block_size(blockSize);
    tiling.set_core_size(coreSize);
    tiling.set_core_remain(coreRemain);

    context->SetBlockDim(aivNum);
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
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class GeluCustom : public OpDef {
public:
    explicit GeluCustom(const char *name) : OpDef(name)
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

OP_ADD(GeluCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t totalLength,
        uint32_t alignNum,
        uint32_t blockSize,
        uint32_t coreSize,
        uint32_t coreRemain)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero");
        this->blockLength = coreSize + (GetBlockNum() == GetBlockIdx() + 1 ? coreRemain : 0);
        this->tileLength = blockSize;
        this->blockLength += this->blockLength % alignNum ? (alignNum - this->blockLength % alignNum) : 0;
        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        const uint32_t startPointer = coreSize * GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + startPointer, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + startPointer, this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
    }

    __aicore__ inline void Process()
    {
        const int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount - 1; ++i) {
            CopyIn(i, this->tileLength);
            Compute(this->tileLength);
            CopyOut(i, this->tileLength);
        }

        const uint32_t tailLength = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, tailLength);
        Compute(tailLength);
        CopyOut(loopCount - 1, tailLength);
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t length)
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        DataCopy(xLocal, xGm[progress * this->tileLength], length);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t length)
    {
        LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();

        constexpr float kParam1 = 0.0455399241f;
        constexpr float kParam2 = -1.595769122f;

        Mul(yLocal, xLocal, xLocal, length);
        Mul(yLocal, yLocal, xLocal, length);
        Muls(yLocal, yLocal, kParam1, length);
        Add(yLocal, yLocal, xLocal, length);
        Muls(yLocal, yLocal, kParam2, length);
        Exp(yLocal, yLocal, length);
        Adds(yLocal, yLocal, 1.0f, length);
        Div(yLocal, xLocal, yLocal, length);

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t length)
    {
        LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        DataCopy(yGm[progress * this->tileLength], yLocal, length);
        outQueueY.FreeTensor(yLocal);
    }

private:
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    uint32_t blockLength = 0;
    uint32_t tileNum = 0;
    uint32_t tileLength = 0;
};

extern "C" __global__ __aicore__ void gelu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelGelu op;
    op.Init(
        x,
        y,
        tiling_data.totalLength,
        tiling_data.ALIGN_NUM,
        tiling_data.block_size,
        tiling_data.core_size,
        tiling_data.core_remain);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor gelu_impl_npu(const at::Tensor &self)
{
    at::Tensor result = at::empty_like(self);
    EXEC_NPU_CMD(aclnnGeluCustom, self, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("gelu_custom", &gelu_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gelu_custom", &gelu_impl_npu, "GELU activation");
}
"""

model_src='''
import ctypes
import importlib
import os
import torch
import torch_npu


def _preload_custom_opapi():
    opp_root = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
    if not opp_root:
        return
    lib_dir = os.path.join(opp_root, "op_api", "lib")
    for lib_name in ("libcust_opapi.so", "libopapi.so"):
        lib_path = os.path.join(lib_dir, lib_name)
        if os.path.exists(lib_path):
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_preload_custom_opapi()
try:
    import custom_ops_lib as ops_lib
except ImportError:
    ops_lib = importlib.import_module("custom" + "_ops_lib")


class ModelNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ops_lib.gelu_custom(x)
'''
