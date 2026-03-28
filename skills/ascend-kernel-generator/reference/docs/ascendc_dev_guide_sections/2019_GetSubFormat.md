#### GetSubFormat

## 函数功能

从实际 format 中解析出子 format 信息。

## 函数原型

```cpp
inline int32_t GetSubFormat(int32_t format)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| format | 输入 | 实际 format（4 字节大小，第 1 个字节的高四位为预留字段，低四位为 C0 format 段，第 2-3 字节为子 format 信息，第 4 字节为主 format 信息）。 |

## 返回值

实际 format 中包含的子 format。

## 异常处理

无。

## 约束说明

无。
