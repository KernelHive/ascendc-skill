##### GetDynamicInputDesc

## 函数功能

根据算子原型定义中的输入索引获取对应动态输入的 tensor 描述信息。

## 函数原型

```cpp
const CompileTimeTensorDesc *GetDynamicInputDesc(const size_t ir_index, const size_t relative_index) const
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| ir_index | 输入 | 算子 IR 原型定义中的输入索引，从 0 开始计数。 |
| relative_index | 输入 | 该输入实例化后的相对 index，例如某个 DYNAMIC_INPUT 实例化了 3 个输入，那么 relative_index 的有效范围是 [0,2]。 |

## 返回值说明

返回 `CompileTimeTensorDesc` 指针。当 `index` 或 `relative_index` 非法时，返回空指针。

关于 `CompileTimeTensorDesc` 的定义，请参见 [15.2.2.2 CompileTimeTensorDesc](#15222-compiletimetensordesc)。

## 约束说明

无。
