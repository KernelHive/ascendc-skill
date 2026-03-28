##### GetDim

## 函数功能
获取对应 idx 轴的 dim 值。

## 函数原型
```cpp
int64_t GetDim(const size_t idx) const
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| idx  | 输入      | dim 的 index，调用者需要保证 index 合法。 |

## 返回值说明
dim 值，在 idx >= kMaxDimNum 时，返回 `kInvalidDimValue`。

## 约束说明
调用者需要保证 index 合法，即 idx < kMaxDimNum。

## 调用示例
```cpp
Shape shape0({3, 256, 256});
auto dim0 = shape0.GetDim(0); // 3
auto invalid_dim = shape0.GetDim(kMaxDimNum); // kInvalidDimValue
```
