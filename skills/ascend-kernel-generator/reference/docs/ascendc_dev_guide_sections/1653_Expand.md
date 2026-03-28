##### Expand

## 函数功能

该函数用于将 TilingData 扩展指定的大小。

## 函数原型

```cpp
void* Expand(size_t size)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| size | 输入 | 需要扩展的大小，单位为字节。 |

## 返回值说明

返回扩展对应 size 后 TilingData 的内存地址。

## 约束说明

扩展后的总大小不能超过 TilingData 的最大容量。

## 调用示例

```cpp
auto td_buf = TilingData::CreateCap(100U);
auto td = reinterpret_cast<TilingData*>(td_buf.get());

auto ptr = td->Expand(64U);
size_t cap = td->GetCapacity(); // 64U
```
