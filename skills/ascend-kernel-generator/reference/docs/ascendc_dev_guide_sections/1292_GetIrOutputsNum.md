##### GetIrOutputsNum

## 函数功能

获取算子 IR 原型定义中的输出个数。

## 函数原型

```cpp
size_t GetIrOutputsNum() const
```

## 参数说明

无。

## 返回值说明

IR 原型中定义的输出个数，`size_t` 类型。

## 约束说明

无。

## 调用示例

```cpp
size_t index = compute_node_info->GetIrOutputsNum();
```
