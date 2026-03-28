##### GetRequiredInputDesc

## 函数功能

根据算子原型定义中的输入索引获取对应必选输入的 tensor 描述信息。

## 函数原型

```cpp
const CompileTimeTensorDesc *GetRequiredInputDesc(const size_t ir_index) const
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| ir_index | 输入 | 算子 IR 原型定义中的输入索引，从 0 开始计数 |

## 返回值说明

`CompileTimeTensorDesc` 指针，index 非法时，返回空指针。

> 关于 `CompileTimeTensorDesc` 的定义，请参见 15.2.2.2 CompileTimeTensorDesc。

## 约束说明

无

## 调用示例

```cpp
// 假设已存在 KernelContext *context
auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
// 假设某个算子的 IR 原型的第 0 个输入是普通输入，且实际有 1 个输入
auto optional_input_td = extend_context->GetRequiredInputDesc(0); // 拿到第 0 个输入的 tensor 描述
```
