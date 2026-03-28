
#include "conv_pointwise2d_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

static inline uint32_t CeilDivU32(uint32_t a, uint32_t b) {
    return (a + b - 1U) / b;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvPointwise2dCustomTilingData tiling;

    // Fixed specialization contract.
    const uint32_t N = 16;
    const uint32_t CIN = 64;
    const uint32_t COUT = 128;
    const uint32_t H = 1024;
    const uint32_t W = 1024;

    // Stripe parameters:
    // We compute two adjacent 16-wide stripes per task (total 32 W positions).
    // W=1024 divisible by 32 => no tail on W.
    const uint32_t WTILE = 16;
    const uint32_t WGROUP = 2;
    const uint32_t WGROUP_LEN = WTILE * WGROUP;

    const uint32_t W_GROUPS = CeilDivU32(W, WGROUP_LEN);
    const uint32_t TASKS = N * H * W_GROUPS;

    tiling.set_n(N);
    tiling.set_cin(CIN);
    tiling.set_cout(COUT);
    tiling.set_h(H);
    tiling.set_w(W);

    tiling.set_w_tile(WTILE);
    tiling.set_w_group(WGROUP);
    tiling.set_w_group_len(WGROUP_LEN);
    tiling.set_w_groups(W_GROUPS);
    tiling.set_tasks(TASKS);

    // Raise occupancy vs baseline; keep conservative.
    context->SetBlockDim(128);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class ConvPointwise2dCustom : public OpDef {
public:
    explicit ConvPointwise2dCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("weight")
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

OP_ADD(ConvPointwise2dCustom);

} // namespace ops
