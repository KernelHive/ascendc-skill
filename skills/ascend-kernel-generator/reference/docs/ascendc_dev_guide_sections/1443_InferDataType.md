##### InferDataType

## 函数功能
注册算子的 InferDataType 函数。

用户需要为算子编写一个 `InferDataTypeKernelFunc` 类型的函数，并使用该接口进行注册。

`InferDataTypeKernelFunc` 类型定义如下：

```cpp
using InferDataTypeKernelFunc = UINT32 (*)(InferDataTypeContext *);
```

## 函数原型
```cpp
OpImplRegisterV2 &InferDataType(InferDataTypeKernelFunc infer_datatype_func)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| `infer_datatype_func` | 输入 | 要注册的自定义 InferDataType 函数，类型为 `InferDataTypeKernelFunc`。 |

## 返回值说明
返回算子的 `OpImplRegisterV2` 对象，该对象新增注册了 InferDataType 函数 `infer_datatype_func`。

## 约束说明
无。
