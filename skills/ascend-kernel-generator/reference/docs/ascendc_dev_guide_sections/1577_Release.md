##### Release

## 函数功能
释放对 TensorAddress 的所有权。本接口调用后，本对象不再管理 TensorAddress，而且 TensorAddress 并没有被释放，因此调用者负责通过 manager 释放 TensorAddress。

## 函数原型
```cpp
TensorAddress Release(TensorAddrManager &manager)
```

## 参数说明

| 参数     | 输入/输出 | 说明                                                                 |
|----------|------------|----------------------------------------------------------------------|
| manager  | 输出       | Tensor 的管理函数，用于管理返回的 TensorAddress。若本对象无所有权，则 manager 被设置为 `nullptr`。 |

## 返回值说明
返回本对象持有的 TensorAddress 指针。若本对象不持有任何指针，则返回 `nullptr`。

## 约束说明
无

## 调用示例
```cpp
auto addr = reinterpret_cast<void *>(0x10);
TensorData td(addr, HostAddrManager, 100U, kOnHost);
TensorAddrManager NewHostAddrManager = nullptr;
TensorAddress ta = td.Release(NewHostAddrManager);
```
