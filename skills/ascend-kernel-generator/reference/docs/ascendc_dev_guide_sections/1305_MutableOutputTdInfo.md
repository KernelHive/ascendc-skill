##### MutableOutputTdInfo

## 函数功能

获取算子的输出 Tensor 描述。注意，编译时无法确定的 shape 信息不在 Tensor 描述中（由于编译时无法确定 shape，因此该 Tensor 描述里不包含 shape 信息）。

## 函数原型

```cpp
CompileTimeTensorDesc *MutableOutputTdInfo(const size_t index) const
```

## 参数说明

| 参数   | 输入/输出 | 说明                               |
|--------|-----------|------------------------------------|
| index  | 输入      | 算子输出对应的索引，从 0 开始计数。 |

## 返回值说明

Tensor 描述，返回对象为非 const。

## 约束说明

无。

## 调用示例

```cpp
for (size_t i = 0; i < node->GetAllOutDataAnchorsSize(); ++i) {
  auto td = compute_node_info.MutableOutputTdInfo(i);
  // ...
}
```
