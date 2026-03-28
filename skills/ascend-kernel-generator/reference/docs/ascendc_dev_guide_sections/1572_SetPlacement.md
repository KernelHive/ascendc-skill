##### SetPlacement

## 函数功能
设置张量的放置位置。

## 函数原型
```cpp
void SetPlacement(const TensorPlacement placement)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| placement | 输入 | 张量的放置位置。 |

> 关于 TensorPlacement 类型的定义，请参见 15.2.2.38 TensorPlacement。

## 返回值说明
无。

## 约束说明
无。

## 调用示例
```cpp
auto addr = reinterpret_cast<void *>(0x10);
TensorData td(addr, nullptr);
auto td_place = td.SetPlacement(kOnHost);
```
