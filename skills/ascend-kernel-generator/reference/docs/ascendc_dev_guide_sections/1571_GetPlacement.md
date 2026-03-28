##### GetPlacement

## 函数功能
获取 tensor 的 placement，即 tensor 数据所在的设备位置。

## 函数原型
```cpp
TensorPlacement GetPlacement() const
```

## 参数说明
无。

## 返回值说明
返回 tensor 的 placement。关于 TensorPlacement 类型的定义，请参见 15.2.2.38 TensorPlacement。

## 约束说明
无。

## 调用示例
```cpp
auto addr = reinterpret_cast<void *>(0x10);
TensorData td(addr, HostAddrManager, 100U, kOnHost);
auto td_place = td.GetPlacement(); // kOnHost
```
