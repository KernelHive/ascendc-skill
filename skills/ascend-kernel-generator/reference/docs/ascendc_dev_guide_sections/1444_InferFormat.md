##### InferFormat

## 函数功能

注册算子的 InferFormat 函数。

对于部分格式敏感的 Cube 算子，使用更适应底层硬件的内部格式，可以带来较大的性能收益，因此开发者需要实现 InferFormat 函数并注册。

## 函数原型

```cpp
OpImplRegisterV2 &InferFormat(InferFormatFunc infer_format_func)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| `infer_format_func` | 输入 | 要注册的自定义 InferFormat 函数，类型为 `InferFormatFunc`。 |

`InferFormatFunc` 类型定义如下：

```cpp
using InferFormatFunc = UINT32 (*)(InferFormatContext *context);
```

## 返回值说明

返回算子的 `OpImplRegisterV2` 对象，该对象新增注册了 InferFormat 函数 `infer_format_func`。

## 约束说明

无。
