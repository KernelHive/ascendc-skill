###### ScalarList

## 功能说明

配置该参数后，自动生成的单算子 API（aclnnxxx）接口中，输入类型为 `aclScalarList` 类型。

## 函数原型

```cpp
OpParamDef &ScalarList()
```

## 参数说明

无

## 返回值说明

`OpParamDef` 算子定义，`OpParamDef` 请参考 15.1.6.1.4 OpParamDef。

## 约束说明

- 仅支持对算子输入做该参数配置，如果对算子输出配置该参数，则配置无效。
- 该接口仅在如下场景支持：
  - 通过单算子 API 执行的方式开发单算子调用应用。
  - 间接调用单算子 API（aclnnxxx）接口：Pytorch 框架单算子直调的场景。

## 调用示例

```cpp
this->Input("x")
.ScalarList()
```
