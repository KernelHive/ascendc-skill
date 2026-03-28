#### GetFormatFromC

## 函数功能

根据传入的 format 和 C0 format 信息得到实际的 format。

实际 format 为 4 字节大小，结构如下：

```
/*
* ---------------------------------------------------
* | 4 bits | 4bits | 2 bytes | 1 byte |
* |------------|-------------|----------------|--------|
* | reserved | C0 format | Sub format | format |
* ---------------------------------------------------
*/
```

- 第 1 个字节的高四位为预留字段，低四位为 C0 format
- 第 2-3 字节为子 format 信息
- 第 4 字节为主 format 信息

## 函数原型

```c
inline int32_t GetFormatFromC0(int32_t format, int32_t c0_format)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| format | 输入 | 子 format 与主 format 信息，值不超过 0xffffffU |
| c0_format | 输入 | C0 format 信息，值不超过 0xfU |

## 返回值

指定的 format 和 c0_format 对应的实际 format。

## 异常处理

无。

## 约束说明

无。
