##### SetExpandIndex

## 函数功能
将第 index 轴设置为补维轴。

## 函数原型
```cpp
void SetExpandIndex(const AxisIndex index)
```

## 参数说明

| 参数  | 输入/输出 | 说明                     |
|-------|-----------|--------------------------|
| index | 输入      | 第 index 根轴为补维轴。 |

```cpp
using AxisIndex = uint64_t;
```

## 返回值说明
无。

## 约束说明
无。

## 调用示例
```cpp
ExpandDimsType type1("1001");
type1.SetExpandIndex(1); // 补维规则 mask_=1101
```
