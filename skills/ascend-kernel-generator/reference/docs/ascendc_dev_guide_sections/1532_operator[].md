##### operator[]

## 函数功能
获取指定 index 轴的 dim 值。

## 函数原型
```cpp
const int64_t &operator[](const size_t idx) const
int64_t &operator[](const size_t idx)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| idx  | 输入      | dim 的 index，调用者需要保证 index 合法。 |

## 返回值说明
- `const int64_t &operator[](const size_t idx) const`：dim 值，在 `idx >= kMaxDimNum` 时，行为未定义。
- `int64_t &operator[](const size_t idx)`：dim 值，在 `idx >= kMaxDimNum` 时，行为未定义。

## 约束说明
调用者需要保证 index 合法，即 `idx < kMaxDimNum`。

## 调用示例
```cpp
Shape shape0({3, 256, 256});
auto dim0 = shape0[0]; // 3
auto dim5 = shape0[5]; // 0
auto invalid_dim = shape0[kMaxDimNum]; // 行为未定义
```
