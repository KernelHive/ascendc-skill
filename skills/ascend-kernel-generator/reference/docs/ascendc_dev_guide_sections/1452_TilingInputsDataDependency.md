##### TilingInputsDataDependency

## 函数功能

标记 Tiling 计算时需要依赖算子第几个输入 tensor 的值，同时标记 tiling 计算支持执行的位置。

## 函数原型

```cpp
OpImplRegisterV2 &TilingInputsDataDependency(std::initializer_list<int32_t> inputs)
```

```cpp
OpImplRegisterV2 &TilingInputsDataDependency(std::initializer_list<int32_t> inputs,
                                             std::initializer_list<TilingPlacement> placements)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| inputs | 输入 | 指定算子 tiling 计算需要依赖的输入 index 列表。举例来说，`inputs={0, 3}`，说明该算子的 tiling 计算需要依赖第 0、3 个输入的 tensor 值。 |
| placements | 输入 | 指定算子 tiling 计算可以执行的位置，0 代表支持在 host 侧执行，1 代表支持在 AI CPU 上执行。如果不包含本参数，代表只支持在 host 执行。 |

## 返回值说明

返回算子的 `OpImplRegisterV2` 对象，该对象新增注册了算子 tiling 值依赖输入的第 index 个 tensor 值以及可执行的位置。

## 约束说明

无。
