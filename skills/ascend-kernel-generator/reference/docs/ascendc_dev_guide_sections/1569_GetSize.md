##### GetSize

## 函数功能
获取 tensor 数据的内存大小。

## 函数原型
```cpp
size_t GetSize() const
```

## 参数说明
无。

## 返回值说明
tensor 所占内存大小，单位为字节。

## 约束说明
无。

## 调用示例
```cpp
auto addr = reinterpret_cast<void *>(0x10);
TensorData td(addr, HostAddrManager, 100U, kOnHost);
auto td_size = td.GetSize(); // 100U
```
