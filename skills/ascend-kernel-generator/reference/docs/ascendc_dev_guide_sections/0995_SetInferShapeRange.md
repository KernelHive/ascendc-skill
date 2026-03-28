###### SetInferShapeRange

## 功能说明

使用图模式时，需要调用该接口注册 ShapeRange 推导函数。

## 函数原型

```cpp
OpDef &SetInferShapeRange(gert::OpImplRegisterV2::InferShapeRangeKernelFunc func)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| func | 输入 | ShapeRange 推导函数。InferShapeRangeKernelFunc 类型定义如下，入参类型参考 InferShapeRangeContext：<br>`using InferShapeRangeKernelFunc = UINT32 (*)(InferShapeRangeContext *);` |

## 返回值说明

OpDef 算子定义，OpDef 请参考 15.1.6.1.3 OpDef。

## 约束说明

无
