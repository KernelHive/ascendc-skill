
#include "le_net5_custom_tiling.h"
#include "register/op_def_registry.h"
#include <string.h>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    LeNet5CustomTilingData tiling;

    uint32_t batch = static_cast<uint32_t>(context->GetInputShape(0)->GetStorageShape().GetDim(0));
    uint32_t num_classes = static_cast<uint32_t>(context->GetInputShape(10)->GetStorageShape().GetDim(0)); // fc3_b shape [C]

    // Conservative multi-core split over batch. Slightly refined vs baseline to avoid
    // overly aggressive blockDim when per-core register/local usage is high.
    uint32_t blockDim = 1;
    if (batch >= 8192)      blockDim = 40;
    else if (batch >= 4096) blockDim = 32;
    else if (batch >= 2048) blockDim = 16;
    else if (batch >= 1024) blockDim = 12;
    else if (batch >= 512)  blockDim = 8;
    else if (batch >= 128)  blockDim = 4;
    else if (batch >= 32)   blockDim = 2;
    else                    blockDim = 1;

    tiling.set_batch(batch);
    tiling.set_num_classes(num_classes);
    tiling.set_block_dim(blockDim);

    context->SetBlockDim(blockDim);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class LeNet5Custom : public OpDef {
public:
    explicit LeNet5Custom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("conv1_w")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("conv1_b")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("conv2_w")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("conv2_b")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("fc1_w")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("fc1_b")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("fc2_w")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("fc2_b")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("fc3_w")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("fc3_b")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(LeNet5Custom);

} // namespace ops
