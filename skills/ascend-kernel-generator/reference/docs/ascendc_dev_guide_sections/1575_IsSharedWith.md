##### IsSharedWith

## 函数功能

判断当前 TensorData 对象与另一个对象是否共享一块内存以及使用同一个内存管理函数。

## 函数原型

```cpp
bool IsSharedWith(const TensorData &other) const
```

## 参数说明

| 参数   | 输入/输出 | 说明                   |
|--------|-----------|------------------------|
| other  | 输入      | 另一个 TensorData 对象 |

## 返回值说明

- `true`：两个对象共享一块内存以及使用同一个内存管理函数
- `false`：两个对象不共享内存或不使用同一个内存管理函数

## 约束说明

无。

## 调用示例

```cpp
auto addr = reinterpret_cast<void *>(0x10);
TensorData td1(addr, HostAddrManager, 100U, kOnHost);
TensorData td2(addr, HostAddrManager, 100U, kOnHost);
bool is_shared_td = td1.IsSharedWith(td2); // true
```
