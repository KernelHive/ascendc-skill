##### SetSize

## 函数功能
设置 Tensor 的内存大小。

## 函数原型
```cpp
void SetSize(const size_t size)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| size | 输入 | Tensor 的内存大小，单位是字节。 |

## 返回值说明
无。

## 约束说明
无。

## 调用示例
```cpp
StorageShape sh({1, 2, 3}, {1, 2, 3});
Tensor t = {sh, {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, kOnHost, ge::DT_FLOAT, nullptr};
t.SetSize(0U);
```
