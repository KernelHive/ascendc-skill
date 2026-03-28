##### InputsDataDependency

## 函数功能

设置算子计算依赖第几个输入 tensor 的值。

所谓的数据依赖，是指算子的计算不仅依赖于输入 tensor 的 shape，还依赖输入 tensor 的具体值。

## 函数原型

```cpp
OpImplRegisterV2 &InputsDataDependency(std::initializer_list<int32_t> inputs)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| inputs | 输入 | 指定算子计算需要依赖的输入 index 列表。举例来说，`inputs={0, 3}`，说明该算子的计算需要依赖第 0、3 个输入的 tensor 值。 |

## 返回值说明

返回算子的 `OpImplRegisterV2` 对象，该对象新增设置算子计算依赖第几个输入 tensor 的值。

## 约束说明

无。
