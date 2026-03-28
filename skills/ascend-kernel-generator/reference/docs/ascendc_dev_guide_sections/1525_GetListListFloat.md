##### GetListListFloat

## 函数功能

获取 `ContinuousVectorVector *` 类型的属性值，即二维数组且每个元素类型为 `float`。

## 函数原型

```cpp
const ContinuousVectorVector *GetListListFloat(const size_t index) const
```

## 参数说明

| 参数   | 输入/输出 | 说明                           |
|--------|-----------|--------------------------------|
| index  | 输入      | 属性在 IR 原型定义中的索引。   |

## 返回值说明

指向属性值的指针。

## 约束说明

无。

## 调用示例

```cpp
// 假设某算子的 IR 原型定义中，第一个属性的类型是二维数组 float 类型
const RuntimeAttrs * runtime_attrs = kernel_context->GetAttrs();
const ContinuousVectorVector *attr0 = runtime_attrs->GetListListFloat(0);
```
