##### GetInputTdInfo

## 函数功能

根据输入索引信息，获取算子的对应输入 Tensor 描述。注意，编译时无法确定的 shape 信息不在 Tensor 描述中（由于编译时无法确定 shape，因此该 Tensor 描述里不包含 shape 信息）。

## 函数原型

```cpp
const CompileTimeTensorDesc *GetInputTdInfo(const size_t index) const
```

## 参数说明

| 参数   | 输入/输出 | 说明                         |
|--------|-----------|------------------------------|
| index  | 输入      | 算子的输入索引，从 0 开始计数 |

## 返回值说明

返回 const 类型的 Tensor 描述信息。

## 约束说明

无。

## 调用示例

```cpp
auto compute_node_info = extend_kernel_context->GetComputeNodeInfo();
auto input_td = compute_node_info->GetInputTdInfo(input_index);
```
