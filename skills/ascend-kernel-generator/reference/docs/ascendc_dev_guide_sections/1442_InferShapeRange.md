##### InferShapeRange

## 函数功能

注册算子的 InferShapeRange 函数。

用户需要为算子编写一个 `InferShapeRangeKernelFunc` 类型的函数，并使用该接口进行注册。

`InferShapeRangeKernelFunc` 类型定义如下：

```cpp
using InferShapeRangeKernelFunc = UINT32 (*)(InferShapeRangeContext *);
```

InferShapeRange 函数的原型是确定的，其接受一个 `InferShapeRangeContext` 类型作为输入。在此 context 上，可以获取到输入、输出的 shape 指针等内容（算子原型定义上的输入、输出、属性信息）。InferShapeRange 成功后，返回 `ge::GRAPH_SUCCESS`，其他返回值被认为推导失败。推导失败后，执行过程结束退出。

## 函数原型

```cpp
OpImplRegisterV2 &InferShapeRange(InferShapeRangeKernelFunc infer_shape_range_func)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| `infer_shape_range_func` | 输入 | 要注册的自定义 `infer_shape_range_func` 函数，类型为 `InferShapeRangeKernelFunc`。 |

## 返回值说明

返回算子的 `OpImplRegisterV2` 对象，该对象新增注册了 InferShapeRange 函数 `infer_shape_range_func`。

## 约束说明

无。
