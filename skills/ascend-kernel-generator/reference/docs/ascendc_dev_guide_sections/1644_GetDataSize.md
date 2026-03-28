##### GetDataSize

## 函数功能
获取 tiling data 长度。

## 函数原型
```cpp
size_t GetDataSize() const
```

## 参数说明
无。

## 返回值说明
tiling data 长度。

## 约束说明
无。

## 调用示例
```cpp
auto td_buf = TilingData::CreateCap(100U);
auto td = reinterpret_cast<TilingData *>(td_buf.get());

size_t data_size = td->GetDataSize(); // 0

td->SetDataSize(100U);
data_size = td->GetDataSize(); // 100
```
