##### MallocAdvise

## 函数功能

在用户内存池中根据指定 size 大小申请 device 内存，建议申请的内存地址为 addr。

## 函数原型

```cpp
virtual MemBlock *MallocAdvise(size_t size, void *addr)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| size   | 输入      | 指定需要申请内存大小。 |
| addr   | 输入      | 建议申请的内存地址为 addr。 |

## 返回值

返回 15.2.3.19 MemBlock 指针。

## 异常处理

无。

## 约束说明

虚函数需要用户实现，如若未实现，默认同 15.2.3.1.2 Malloc 功能相同。
