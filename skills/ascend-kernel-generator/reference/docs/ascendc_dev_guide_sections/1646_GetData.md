##### GetData

## 函数功能
获取 TilingData 的数据指针。

## 函数原型
```cpp
void *GetData()
```
```cpp
const void *GetData() const
```

## 参数说明
无。

## 返回值说明
数据指针。

## 约束说明
无。

## 调用示例
```cpp
auto td_buf = TilingData::CreateCap(100U);
auto td = reinterpret_cast<TilingData *>(td_buf.get());
auto tiling_data_ptr = td->GetData(); // td_buf.get() + sizeof(TilingData)
```
