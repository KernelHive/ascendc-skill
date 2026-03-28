###### SetStepSize

## 功能说明

设置细粒度通信时，通信算法的步长，即设置细粒度通信时，一次子通信任务执行或准备执行的通信算法的步骤数。

例如，图1 使用pairwise算法的AlltoAllV通信步骤示意图中，该细粒度通信场景下，AlltoAllV通信任务的通信步长为1。

## 函数原型

```cpp
uint32_t SetStepSize(uint8_t stepSize)
```

## 参数说明

| 参数名   | 输入/输出 | 描述 |
|----------|-----------|------|
| stepSize | 输入      | 设置细粒度通信时，每次通信的步长。0表示当前非细粒度通信。 |

## 返回值说明

- 0表示设置成功。
- 非0表示设置失败。

## 约束说明

Atlas A2 训练系列产品/Atlas A2 推理系列产品暂不支持该接口。

## 调用示例

```cpp
static ge::graphStatus AllToAllVCustomTilingFunc(gert::TilingContext *context)
{
    AllReduceCustomV3TilingData *tiling = context->GetTilingData<AllToAllVCustomV3TilingData>();
    const std::string groupName = "testGroup";
    const std::string algConfig = "AlltoAll=level0:fullmesh;level1:pairwise";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupName, HCCL_CMD_ALLTOALLV, algConfig, 0);
    mc2CcTilingConfig.SetStepSize(1U);
    mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling);
    return ge::GRAPH_SUCCESS;
}
```
