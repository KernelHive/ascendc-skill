#### GetFormatFromSub

## 函数功能

根据传入的主 format 和子 format 信息得到实际的 format。

实际 format 为 4 字节大小，第 1 个字节的高四位为预留字段，低四位为 C0 format，第 2-3 字节为子 format 信息，第 4 字节为主 format 信息，如下：

```
/*
 * ---------------------------------------------------
 * | 4 bits | 4bits | 2 bytes | 1 byte |
 * |------------|-------------|----------------|--------|
 * | reserved | C0 format | Sub format | format |
 * ---------------------------------------------------
 */
```

## 函数原型

```c
inline int32_t GetFormatFromSub(int32_t primary_format, int32_t sub_format)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| `primary_format` | 输入 | 主 format 信息，值不超过 `0xffU` |
| `sub_format` | 输入 | 子 format 信息，值不超过 `0xffffU` |

## 返回值

指定的主 format 和子 format 对应的实际 format。

## 异常处理

无。

## 约束说明

无。
