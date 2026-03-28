###### GetTiling

## 功能说明
获取 `Mc2InitTiling` 参数和 `Mc2CcTiling` 参数。

## 函数原型
```cpp
uint32_t GetTiling(::Mc2InitTiling &tiling)
uint32_t GetTiling(::Mc2CcTiling &tiling)
```

## 参数说明

**表 15-903 参数说明**

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| tiling | 输出 | Tiling 结构体存储的 Tiling 信息。 |

## 返回值说明
- 返回值为 0，则 tiling 计算成功，该 Tiling 结构体的值可以用于后续计算。
- 返回值非 0，则 tiling 计算失败，该 Tiling 结果无法使用。

## 约束说明
无

## 调用示例
```cpp
const char *groupName = "testGroup";
uint32_t opType = HCCL_CMD_REDUCE_SCATTER;
std::string algConfig = "ReduceScatter=level0:fullmesh";
uint32_t reduceType = HCCL_REDUCE_SUM;
AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupName, opType, algConfig, reduceType);
mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling); // tiling 为算子组装的 TilingData 结构体，获取 Mc2InitTiling
mc2CcTilingConfig.GetTiling(tiling->reduceScatterTiling); // tiling 为算子组装的 TilingData 结构体，获取 Mc2CcTiling
```
