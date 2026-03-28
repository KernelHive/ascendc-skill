##### SetSize

## 函数功能
设置 tensor 数据的内存大小。

## 函数原型
```cpp
void SetSize(const size_t size)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| size | 输入 | tensor 的内存大小，单位为字节。 |

## 返回值说明
无。

## 约束说明
无。

## 调用示例
```cpp
auto addr = reinterpret_cast<void *>(0x10);
TensorData td(addr, HostAddrManager, 100U, kOnHost);
td.SetSize(10U);
```
