###### v2 版本 TilingData（废弃）

## 说明

该结构体废弃，并将在后续版本移除，请不要使用该结构体。无需直接对该结构体中的成员进行设置，统一使用 Hccl Tiling 提供的接口设置即可。

## 功能说明

AI CPU 启动下发通信任务前，需获取固定的通信配置，如下表所示。在算子实现中，由 Tiling 组装通信配置项，通过配置固定参数和固定参数顺序的 Tiling Data，将通信配置信息在调用 AI CPU 通信接口时传递给 AI CPU。

## 参数说明

### 表 15-915 v2 版本 Hccl TilingData 参数说明

| 参数名 | 描述 |
|--------|------|
| version | uint32_t 类型。用于区分 TilingData 版本。<br>v2 版本的 TilingData 结构体中，version 字段仅支持取值为 2。<br>**注意**：该字段在 v2 版本 TilingData 中的位置，同 v1 版本的 preparePosition 字段。<br>当该字段取值为 2 时，为 v2 版本的结构体，当取值为 1 时，为 v1 版本的结构体，请使用 Mc2Msg 结构体。 |
| mc2HcommCnt | uint32_t 类型。表示各通信域中通信任务总个数。当前该参数支持的最大取值为 3。 |
| serverCfg | Mc2ServerCfg 类型。集合通信 server 端通用参数配置。 |
| hcom | Mc2HcommCfg 类型。各通信域中每个通信任务的参数配置。<br>在通信算子 TilingData 的定义中，根据各通信域中通信任务总个数，共需要定义 mc2HcommCnt 个 Mc2HcommCfg 结构体。<br>例如：mc2HcommCnt=2，则需要依次定义 2 个 Mc2HcommCfg 类型的参数，自定义参数名，比如 hcom1、hcom2。 |

### 表 15-916 Mc2ServerCfg 结构体说明

| 参数名 | 描述 |
|--------|------|
| version | 预留字段，不需要配置。 |
| debugMode | 预留字段，不需要配置。 |
| sendArgIndex | 预留字段，不需要配置。 |
| recvArgIndex | 预留字段，不需要配置。 |
| commOutArgIndex | 预留字段，不需要配置。 |
| reserved | 预留字段，不需要配置。 |

### 表 15-917 Mc2HcommCfg 结构体说明

| 参数名 | 描述 |
|--------|------|
| skipLocalRankCopy | 预留字段，不需要配置。 |
| skipBufferWindowCopy | 预留字段，不需要配置。 |
| stepSize | 预留字段，不需要配置。 |
| reserved | 预留字段，不需要配置。 |
| groupName | 当前通信任务所在的通信域。char * 类型，支持最大长度 128。 |
| algConfig | 通信算法配置。char * 类型，支持最大长度 128。<br>当前支持的取值为：<br>• `"AllGather=level0:doublering"`：AllGather 通信任务。<br>• `"ReduceScatter=level0:doublering"`：ReduceScatter 通信任务。<br>• `"AlltoAll=level0:fullmesh;level1:pairwise"`：AlltoAllV 通信任务。 |
| opType | 表示通信任务类型。uint32_t 类型，取值详见 HcclCMDType 参数说明。 |
| reduceType | 归约操作类型，仅对有归约操作的通信任务生效。uint32_t 类型，取值详见 HcclReduceOp 参数说明。 |

## 约束说明

- 如果需要使用 v2 版本的 Tiling 结构体，必须设置 Tiling 结构体的第一个参数 `version=2`。
- 算子的 Tiling Data 结构需要完整包含 v2 版本 Hccl TilingData 参数，其中各参数需要严格按照对应参数的结构来定义。

## 调用示例

如下为自定义算子 `AlltoallvDoubleCommCustom` 的算子原型。该算子有两对输入输出，其中 `x1`、`y1` 是 ep 通信域的 AlltoAllV 任务的输入输出，`x2`、`y2` 是 tp 通信域的 AlltoAllV 任务的输入输出。

```cpp
namespace ops {
class AlltoallvDoubleCommCustom : public OpDef {
public:
    explicit AlltoallvDoubleCommCustom(const char *name) : OpDef(name) {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Output("y1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("group_ep").AttrType(REQUIRED).String();
        this->Attr("group_tp").AttrType(REQUIRED).String();
        this->Attr("ep_world_size").AttrType(REQUIRED).Int();
        this->Attr("tp_world_size").AttrType(REQUIRED).Int();
        this->AICore().SetTiling(optiling::AlltoAllVDoubleCommCustomTilingFunc);
        this->AICore().AddConfig("ascendxxx"); // ascendxxx 请修改为对应的昇腾 AI 处理器型号。
        this->MC2().HcclGroup({"group_ep", "group_tp"});
    }
};
OP_ADD(AlltoallvDoubleCommCustom);
}
```

如下为该自定义算子 Tiling Data 声明和实现。

该自定义算子 Tiling Data 的声明中：首先定义 `version` 字段，设置为 2，表明为 v2 版本的通信算子 Tiling 结构体。其次，定义 `mc2HcommCnt` 字段，本例 `AlltoallvDoubleCommCustom` 算子的 kernel 实现中，共 2 个 AlltoAllV 通信任务，该参数取值为 2。然后定义 server 通用参数配置，`Mc2ServerCfg`。最后，定义 2 个 `Mc2HcommCfg` 结构体，表示各通信域中的每个通信任务参数配置。

```cpp
// Hccl TilingData 声明
BEGIN_TILING_DATA_DEF(AlltoallvDoubleCommCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, version); // hccl tiling 结构体的版本，设为 2
    TILING_DATA_FIELD_DEF(uint32_t, mc2HcommCnt); // 各通信域中的通信算子总个数，当前最多支持 3 个。AlltoallvDoubleCommCustom 算子 kernel 实现中每个通信域中均用了 1 个 AlltoAllV，因此设为 2
    TILING_DATA_FIELD_DEF_STRUCT(Mc2ServerCfg, serverCfg); // server 通用参数配置，融合算子级
    TILING_DATA_FIELD_DEF_STRUCT(Mc2HcommCfg, hcom1); // 各通信域中的每个通信任务参数配置，算子级，共有 mc2HcommCnt 个 Mc2HcommCfg
    TILING_DATA_FIELD_DEF_STRUCT(Mc2HcommCfg, hcom2);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AlltoallvDoubleCommCustom, AlltoallvDoubleCommCustomTilingData);

// Hccl TilingData 配置片段
static ge::graphStatus AlltoAllVDoubleCommCustomTilingFunc(gert::TilingContext *context) {
    char *group1 = const_cast<char *>(context->GetAttrs()->GetAttrPointer<char>(0));
    char *group2 = const_cast<char *>(context->GetAttrs()->GetAttrPointer<char>(1));

    AlltoallvDoubleCommCustomTilingData tiling;
    tiling.set_version(2);
    tiling.set_mc2HcommCnt(2);
    tiling.serverCfg.set_debugMode(0);

    tiling.hcom1.set_opType(8);
    tiling.hcom1.set_reduceType(4);
    tiling.hcom1.set_groupName(group1);
    tiling.hcom1.set_algConfig("AlltoAll=level0:fullmesh;level1:pairwise");

    tiling.hcom2.set_opType(8);
    tiling.hcom2.set_reduceType(4);
    tiling.hcom2.set_groupName(group2);
    tiling.hcom2.set_algConfig("AlltoAll=level0:fullmesh;level1:pairwise");

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
```
