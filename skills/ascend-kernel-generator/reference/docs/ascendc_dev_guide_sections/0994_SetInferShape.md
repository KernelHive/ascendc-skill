###### SetInferShape

## 功能说明

使用图模式时，需要调用该接口注册 Shape 推导函数。

## 函数原型

```cpp
OpDef &SetInferShape(gert::OpImplRegisterV2::InferShapeKernelFunc func)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| func | 输入 | Shape 推导函数。InferShapeKernelFunc 类型定义如下，入参类型参考 InferShapeContext：<br>`using InferShapeKernelFunc = UINT32 (*)(InferShapeContext *);` |

## 返回值说明

OpDef 算子定义，OpDef 请参考 15.1.6.1.3 OpDef。

## 约束说明

无
