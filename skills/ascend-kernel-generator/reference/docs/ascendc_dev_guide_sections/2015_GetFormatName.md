#### GetFormatName

## 函数功能

根据传入的 Format 类型，获取 Format 的字符串描述。

## 函数原型

```c
const char_t *GetFormatName(Format format)
```

## 参数说明

| 参数   | 输入/输出 | 说明             |
|--------|-----------|------------------|
| format | 输入      | Format 枚举值。 |

## 返回值

该 Format 所对应的字符串描述，若 Format 不合法或不被识别，则返回 `nullptr`。

## 异常处理

无。

## 约束说明

返回的字符串不可被修改。
