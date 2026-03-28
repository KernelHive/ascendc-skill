###### OutputShapeDependOnCompute

## 功能说明

标识算子输出的 shape 是否依赖于计算得到。某些算子（如 NonZero，用于统计 tensor 中非零值的个数）在计算完成前无法得知算子输出的 shape 信息，只有在算子计算完成后才能获取。

该类算子在原型定义时，需要使用 `OutputShapeDependOnCompute` 接口进行标识，同时在算子核函数中将实际输出 shape 写入到出参中，便于框架侧基于该信息进行输出内存的管理。对应的 kernel 侧实现请参考输出 shape 依赖计算的算子 kernel 实现。

## 函数原型

```cpp
OpParamDef &OutputShapeDependOnCompute()
```

## 参数说明

无

## 返回值说明

返回 `OpParamDef` 算子定义。`OpParamDef` 请参考 15.1.6.1.4 OpParamDef。

## 约束说明

- 只能用于标识算子输出。
- 基于旧版本 CANN 包（不支持 `OutputShapeDependOnCompute` 特性）生成的自定义算子工程，无法兼容 `OutputShapeDependOnCompute` 接口。在使用非当前版本 CANN 包生成的自定义算子工程时，需特别注意兼容性问题。
- 您可以通过查看自定义算子工程下 `cmake/util/ascendc_impl_build.py` 中有无 `output_shape_depend_on_compute` 字段来确认当前工程是否支持该特性。如果未找到该字段，则需要重新生成自定义算子工程以启用 `OutputShapeDependOnCompute` 特性。

## 调用示例

```cpp
this->Input("x1")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND});

this->Input("x2")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND});

this->Output("y1")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND})
    .OutputShapeDependOnCompute();
```
