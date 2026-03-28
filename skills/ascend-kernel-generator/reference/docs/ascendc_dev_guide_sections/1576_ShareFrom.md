##### ShareFrom

## 函数功能

使当前的 TensorData 对象共享另一个对象的内存以及内存管理函数。

## 函数原型

```cpp
ge::graphStatus ShareFrom(const TensorData &other)
```

## 参数说明

| 参数   | 输入/输出 | 说明                   |
| ------ | --------- | ---------------------- |
| other  | 输入      | 另一个 TensorData 对象 |

## 返回值说明

成功时返回 `ge::GRAPH_SUCCESS`。

## 约束说明

无。

## 调用示例

```cpp
auto addr = reinterpret_cast<void *>(0x10);
TensorData td1(addr, HostAddrManager, 100U, kOnHost);
TensorData td2(addr, nullptr);
td2.ShareFrom(td1);
```
