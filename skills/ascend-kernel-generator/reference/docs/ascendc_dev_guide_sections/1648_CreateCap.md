##### CreateCap

## 函数功能

根据指定的最大容量创建一个 TilingData 类实例。

## 函数原型

```cpp
static std::unique_ptr<uint8_t[]> CreateCap(const size_t cap_size)
```

## 参数说明

| 参数     | 输入/输出 | 说明                     |
|----------|-----------|--------------------------|
| cap_size | 输入      | 最大容量，单位为字节。   |

## 返回值说明

TilingData 的实例指针。

## 约束说明

无。

## 调用示例

```cpp
auto td_buf = TilingData::CreateCap(100U);
auto td = reinterpret_cast<TilingData *>(td_buf.get());
```
