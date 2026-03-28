##### InferOutDataTypeSameWithFirstInput

## 函数功能

注册一种 datatype 推导规则，该规则将算子第一个输入的 datatype 作为所有输出的 datatype。

## 函数原型

```cpp
OpImplRegisterV2 &InferOutDataTypeSameWithFirstInput()
```

## 参数说明

无

## 返回值说明

返回算子的 `OpImplRegisterV2` 对象，该对象新增注册了算子 datatype 推导规则。

## 约束说明

- 注册此规则，可以不用再注册自定义推导规则。若同时注册了 `InferDataType` 和 `InferOutDataTypeByFirstInput`，将使能最后注册的规则。
- 若算子无输入或第一个输入 datatype 为未定义（`DT_UNDEFINED`），推导将报错。
