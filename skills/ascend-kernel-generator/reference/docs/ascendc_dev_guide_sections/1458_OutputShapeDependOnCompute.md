##### OutputShapeDependOnCompute

## 函数功能

注册 shape 依赖于计算得到的输出列表。某些算子（例如 NonZero，用于统计 tensor 中非零值的个数）在计算完成前无法得知算子输出的 shape 信息，只有在算子计算完成后才能获取。

该类算子在原型定义时，需要使用 `OutputShapeDependOnCompute` 接口进行标识，同时在算子核函数中将实际输出 shape 写入到出参中，便于框架侧基于该信息进行输出内存的管理。

## 函数原型

```cpp
OpImplRegisterV2 &OutputShapeDependOnCompute(std::initializer_list<int32_t> outputs)
```

## 参数说明

| 参数     | 输入/输出 | 说明                 |
|----------|-----------|----------------------|
| outputs  | 输入      | 指定输出 index 列表。 |

## 返回值说明

返回算子的 `OpImplRegisterV2` 对象，该对象新增注册了 shape 依赖于计算得到的输出列表。

## 约束说明

- 只能用于标识算子输出。
