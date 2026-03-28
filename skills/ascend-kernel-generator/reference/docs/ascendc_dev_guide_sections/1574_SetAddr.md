##### SetAddr

## 函数功能
设置 tensor 地址。

## 函数原型
```cpp
ge::graphStatus SetAddr(const ConstTensorAddressPtr addr, TensorAddrManager manager)
```

## 参数说明

| 参数     | 输入/输出 | 说明 |
|----------|-----------|------|
| `addr`   | 输入      | tensor 地址。<br>`using ConstTensorAddressPtr = const void *;` |
| `manager`| 输入      | tensor 的管理函数。<br>`using TensorAddrManager = ge::graphStatus (*)(TensorAddress addr, TensorOperateType operate_type, void **out);` |

## 返回值说明
- 成功时返回 `ge::GRAPH_SUCCESS`
- 失败时返回 `manager` 管理函数中定义的错误码

## 约束说明
无。

## 调用示例
```cpp
auto addr = reinterpret_cast<void *>(0x10);
TensorData td(addr, nullptr);
td.SetAddr(addr, HostAddrManager);
```
