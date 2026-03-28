##### GetAddr

## 函数功能
获取 tensor 数据地址。若存在 manager 函数，则由 manager 函数给出地址。

## 函数原型
```cpp
TensorAddress GetAddr() const
```

## 参数说明
无。

## 返回值说明
tensor 地址。

## 约束说明
无。

## 调用示例
```cpp
auto addr = reinterpret_cast<void *>(0x10);
TensorData td(addr, nullptr);
auto addr = td.GetAddr(); // 0x10
```
