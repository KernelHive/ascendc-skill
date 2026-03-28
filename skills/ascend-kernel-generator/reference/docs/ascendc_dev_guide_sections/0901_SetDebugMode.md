###### SetDebugMode

## 功能说明
设置调测模式。

## 函数原型
```c
uint32_t SetDebugMode(uint8_t debugMode)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| debugMode | 输入 | 表示选择的调测模式，uint8_t类型。参数支持的取值如下：<br>● 1：关闭Hccl高阶API通信功能。<br>● 2：打印消息队列和Prepare消息执行次数等信息。<br>● 3：打印Prepare消息源数据buffer和目的数据buffer中的数据。<br>● 4：打印AI CPU服务端执行通信任务的各阶段时间戳和耗时，每执行30个算子打印一次。 |

## 返回值说明
- 0表示设置成功。
- 非0表示设置失败。

## 约束说明
无

## 调用示例
```cpp
const char *groupName = "testGroup";
uint32_t opType = HCCL_CMD_REDUCE_SCATTER;
std::string algConfig = "ReduceScatter=level0:fullmesh";
uint32_t reduceType = HCCL_REDUCE_SUM;
AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupName, opType, algConfig, reduceType);
mc2CcTilingConfig.SetDebugMode(3); // 设置调测模式
mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
mc2CcTilingConfig.GetTiling(tiling->reduceScatterTiling);
```
