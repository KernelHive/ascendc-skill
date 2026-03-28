###### SetInferDataType

## 功能说明

使用图模式时，需要调用该接口注册 DataType 推导函数。

## 函数原型

```cpp
OpDef &SetInferDataType(gert::OpImplRegisterV2::InferDataTypeKernelFunc func)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| func | 输入 | DataType 推导函数。InferDataTypeKernelFunc 类型定义如下，入参类型参考 15.2.2.12 InferDataTypeContext：<br>`using InferDataTypeKernelFunc = UINT32 (*)(InferDataTypeContext *);` |

## 返回值说明

OpDef 算子定义，OpDef 请参考 15.1.6.1.3 OpDef。

## 约束说明

无
