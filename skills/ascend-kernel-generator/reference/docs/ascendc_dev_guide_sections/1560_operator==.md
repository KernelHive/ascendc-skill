##### operator==

## 函数功能

判断 shape 是否相等。

## 函数原型

```cpp
bool operator==(const StorageShape &other) const
```

## 参数说明

| 参数  | 输入/输出 | 说明         |
|-------|-----------|--------------|
| other | 输入      | 另一个 shape |

## 返回值说明

- `true`：相等
- `false`：不相等

## 约束说明

无。

## 调用示例

```cpp
StorageShape shape0({3, 256, 256}, {256, 256, 3});
StorageShape shape1({3, 256, 256}, {3, 256, 256});
bool is_same_shape = shape0 == shape1; // false
```
