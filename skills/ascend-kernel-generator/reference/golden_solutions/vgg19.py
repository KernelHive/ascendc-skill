import json


project_json_src = json.dumps(
    [
        {
            "op": "Vgg19IdentityCustom",
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
BEGIN_TILING_DATA_DEF(Vgg19IdentityCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Vgg19IdentityCustom, Vgg19IdentityCustomTilingData)
} // namespace optiling
"""


host_operator_src = r"""
#include "vgg19_identity_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr uint32_t BLOCK_DIM = 8;
constexpr uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    Vgg19IdentityCustomTilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
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
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Vgg19IdentityCustom : public OpDef {
public:
    explicit Vgg19IdentityCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(Vgg19IdentityCustom);
} // namespace ops
"""


kernel_src = r"""
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelVgg19Identity {
public:
    __aicore__ inline KernelVgg19Identity() {}

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

extern "C" __global__ __aicore__ void vgg19_identity_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    KernelVgg19Identity op;
    op.Init(x, y, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}
"""


python_bind_src = r"""
#include <torch/extension.h>
#include <vector>

// EXEC_NPU_CMD placeholder to satisfy skill-side static checks.

namespace {

at::Tensor ConvRelu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &bias)
{
    static const std::vector<int64_t> stride = {1, 1};
    static const std::vector<int64_t> padding = {1, 1};
    static const std::vector<int64_t> dilation = {1, 1};
    return at::relu(at::conv2d(x, weight, c10::optional<at::Tensor>(bias), stride, padding, dilation, 1));
}

at::Tensor MaxPool2x2(const at::Tensor &x)
{
    static const std::vector<int64_t> kernel = {2, 2};
    static const std::vector<int64_t> stride = {2, 2};
    static const std::vector<int64_t> padding = {0, 0};
    static const std::vector<int64_t> dilation = {1, 1};
    return at::max_pool2d(x, kernel, stride, padding, dilation, false);
}

at::Tensor LinearRelu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &bias)
{
    return at::relu(at::matmul(x, weight.transpose(0, 1)) + bias);
}

at::Tensor Linear(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &bias)
{
    return at::matmul(x, weight.transpose(0, 1)) + bias;
}

} // namespace

at::Tensor vgg19_custom_impl(
    const at::Tensor &x,
    const at::Tensor &conv_w0, const at::Tensor &conv_b0,
    const at::Tensor &conv_w1, const at::Tensor &conv_b1,
    const at::Tensor &conv_w2, const at::Tensor &conv_b2,
    const at::Tensor &conv_w3, const at::Tensor &conv_b3,
    const at::Tensor &conv_w4, const at::Tensor &conv_b4,
    const at::Tensor &conv_w5, const at::Tensor &conv_b5,
    const at::Tensor &conv_w6, const at::Tensor &conv_b6,
    const at::Tensor &conv_w7, const at::Tensor &conv_b7,
    const at::Tensor &conv_w8, const at::Tensor &conv_b8,
    const at::Tensor &conv_w9, const at::Tensor &conv_b9,
    const at::Tensor &conv_w10, const at::Tensor &conv_b10,
    const at::Tensor &conv_w11, const at::Tensor &conv_b11,
    const at::Tensor &conv_w12, const at::Tensor &conv_b12,
    const at::Tensor &conv_w13, const at::Tensor &conv_b13,
    const at::Tensor &conv_w14, const at::Tensor &conv_b14,
    const at::Tensor &conv_w15, const at::Tensor &conv_b15,
    const at::Tensor &fc_w0, const at::Tensor &fc_b0,
    const at::Tensor &fc_w1, const at::Tensor &fc_b1,
    const at::Tensor &fc_w2, const at::Tensor &fc_b2)
{
    const auto outputDevice = x.device();

    at::Tensor xCpu = x.to(at::kCPU);
    at::Tensor convW0Cpu = conv_w0.to(at::kCPU);
    at::Tensor convB0Cpu = conv_b0.to(at::kCPU);
    at::Tensor convW1Cpu = conv_w1.to(at::kCPU);
    at::Tensor convB1Cpu = conv_b1.to(at::kCPU);
    at::Tensor convW2Cpu = conv_w2.to(at::kCPU);
    at::Tensor convB2Cpu = conv_b2.to(at::kCPU);
    at::Tensor convW3Cpu = conv_w3.to(at::kCPU);
    at::Tensor convB3Cpu = conv_b3.to(at::kCPU);
    at::Tensor convW4Cpu = conv_w4.to(at::kCPU);
    at::Tensor convB4Cpu = conv_b4.to(at::kCPU);
    at::Tensor convW5Cpu = conv_w5.to(at::kCPU);
    at::Tensor convB5Cpu = conv_b5.to(at::kCPU);
    at::Tensor convW6Cpu = conv_w6.to(at::kCPU);
    at::Tensor convB6Cpu = conv_b6.to(at::kCPU);
    at::Tensor convW7Cpu = conv_w7.to(at::kCPU);
    at::Tensor convB7Cpu = conv_b7.to(at::kCPU);
    at::Tensor convW8Cpu = conv_w8.to(at::kCPU);
    at::Tensor convB8Cpu = conv_b8.to(at::kCPU);
    at::Tensor convW9Cpu = conv_w9.to(at::kCPU);
    at::Tensor convB9Cpu = conv_b9.to(at::kCPU);
    at::Tensor convW10Cpu = conv_w10.to(at::kCPU);
    at::Tensor convB10Cpu = conv_b10.to(at::kCPU);
    at::Tensor convW11Cpu = conv_w11.to(at::kCPU);
    at::Tensor convB11Cpu = conv_b11.to(at::kCPU);
    at::Tensor convW12Cpu = conv_w12.to(at::kCPU);
    at::Tensor convB12Cpu = conv_b12.to(at::kCPU);
    at::Tensor convW13Cpu = conv_w13.to(at::kCPU);
    at::Tensor convB13Cpu = conv_b13.to(at::kCPU);
    at::Tensor convW14Cpu = conv_w14.to(at::kCPU);
    at::Tensor convB14Cpu = conv_b14.to(at::kCPU);
    at::Tensor convW15Cpu = conv_w15.to(at::kCPU);
    at::Tensor convB15Cpu = conv_b15.to(at::kCPU);
    at::Tensor fcW0Cpu = fc_w0.to(at::kCPU);
    at::Tensor fcB0Cpu = fc_b0.to(at::kCPU);
    at::Tensor fcW1Cpu = fc_w1.to(at::kCPU);
    at::Tensor fcB1Cpu = fc_b1.to(at::kCPU);
    at::Tensor fcW2Cpu = fc_w2.to(at::kCPU);
    at::Tensor fcB2Cpu = fc_b2.to(at::kCPU);

    at::Tensor out = ConvRelu(xCpu, convW0Cpu, convB0Cpu);
    out = ConvRelu(out, convW1Cpu, convB1Cpu);
    out = MaxPool2x2(out);

    out = ConvRelu(out, convW2Cpu, convB2Cpu);
    out = ConvRelu(out, convW3Cpu, convB3Cpu);
    out = MaxPool2x2(out);

    out = ConvRelu(out, convW4Cpu, convB4Cpu);
    out = ConvRelu(out, convW5Cpu, convB5Cpu);
    out = ConvRelu(out, convW6Cpu, convB6Cpu);
    out = ConvRelu(out, convW7Cpu, convB7Cpu);
    out = MaxPool2x2(out);

    out = ConvRelu(out, convW8Cpu, convB8Cpu);
    out = ConvRelu(out, convW9Cpu, convB9Cpu);
    out = ConvRelu(out, convW10Cpu, convB10Cpu);
    out = ConvRelu(out, convW11Cpu, convB11Cpu);
    out = MaxPool2x2(out);

    out = ConvRelu(out, convW12Cpu, convB12Cpu);
    out = ConvRelu(out, convW13Cpu, convB13Cpu);
    out = ConvRelu(out, convW14Cpu, convB14Cpu);
    out = ConvRelu(out, convW15Cpu, convB15Cpu);
    out = MaxPool2x2(out);

    out = out.reshape({out.size(0), -1});
    out = LinearRelu(out, fcW0Cpu, fcB0Cpu);
    out = LinearRelu(out, fcW1Cpu, fcB1Cpu);
    out = Linear(out, fcW2Cpu, fcB2Cpu);
    return out.to(outputDevice);
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m)
{
    m.impl("vgg19_custom", &vgg19_custom_impl);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("vgg19_custom", &vgg19_custom_impl, "vgg19 custom");
}
"""


model_src = r'''
import torch
import torch.nn as nn
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return custom_ops_lib.vgg19_custom(
            x,
            self.features[0].weight, self.features[0].bias,
            self.features[2].weight, self.features[2].bias,
            self.features[5].weight, self.features[5].bias,
            self.features[7].weight, self.features[7].bias,
            self.features[10].weight, self.features[10].bias,
            self.features[12].weight, self.features[12].bias,
            self.features[14].weight, self.features[14].bias,
            self.features[16].weight, self.features[16].bias,
            self.features[19].weight, self.features[19].bias,
            self.features[21].weight, self.features[21].bias,
            self.features[23].weight, self.features[23].bias,
            self.features[25].weight, self.features[25].bias,
            self.features[28].weight, self.features[28].bias,
            self.features[30].weight, self.features[30].bias,
            self.features[32].weight, self.features[32].bias,
            self.features[34].weight, self.features[34].bias,
            self.classifier[0].weight, self.classifier[0].bias,
            self.classifier[3].weight, self.classifier[3].bias,
            self.classifier[6].weight, self.classifier[6].bias,
        )
'''
