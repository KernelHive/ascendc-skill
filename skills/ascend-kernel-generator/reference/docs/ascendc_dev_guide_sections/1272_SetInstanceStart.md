##### SetInstanceStart

## 函数功能

设置算子某个 IR 输入在实际输入中的起始序号（index）。

## 函数原型

```cpp
void SetInstanceStart(const uint32_t instance_start)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| instance_start | 输入 | 首个实例化 Anchor 的 index |

## 返回值说明

无。

## 约束说明

无。

## 调用示例

```cpp
const auto &ir_inputs = node->GetOpDesc()->GetIrInputs(); // 算子 IR 原型定义的所有输入
for (size_t i = 0; i < ir_inputs.size(); ++i) {
    auto ins_info = compute_node_info.MutableInputInstanceInfo(i); // 获取第 i 个 IR 输入对应的 AnchorInstanceInfo 对象
    GE_ASSERT_NOTNULL(ins_info);

    size_t input_index = ir_index_to_instance_index_pair_map[i].first; // 获取统计后的算子 IR 输入对应的实际输入 index
    ins_info->SetInstanceStart(input_index); // 将该信息保存到 IR 输入对应的 AnchorInstanceInfo 对象中
}
```
