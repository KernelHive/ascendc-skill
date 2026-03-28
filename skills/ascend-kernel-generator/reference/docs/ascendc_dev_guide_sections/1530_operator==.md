##### operator==

## 函数功能

判断与另外一个 Shape 对象是否相等。如果两个 Shape 的维度数量相等，并且每个维度的值都相等，则认为两个 Shape 相等。

## 函数原型

```cpp
bool operator==(const Shape &rht) const
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| rht  | 输入      | 另一个 Shape 对象 |

## 返回值说明

- `true`：相等
- `false`：不相等

## 约束说明

无。

## 调用示例

```cpp
Shape shape0({3, 256, 256});
Shape shape1({1, 3, 256, 256});
auto is_same_shape = shape0 == shape1; // 返回值为 false，不相等
```
