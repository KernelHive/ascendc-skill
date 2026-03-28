##### operator!=

## 函数功能

判断与另一个 Shape 对象是否不等。

## 函数原型

```cpp
bool operator!=(const Shape &rht) const
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| rht  | 输入       | 另一个 Shape 对象。 |

## 返回值说明

- `true`：不相等
- `false`：相等

## 约束说明

无。

## 调用示例

```cpp
Shape shape0({3, 256, 256});
Shape shape1({1, 3, 256, 256});
auto is_diff_shape = shape0 != shape1; // 返回值为 true，不相等
```
