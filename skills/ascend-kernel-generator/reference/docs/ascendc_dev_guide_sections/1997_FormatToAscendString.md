##### FormatToAscendString

## 函数功能

将 Format 类型值转化为字符串表达。

使用该接口需要包含 `type_utils.h` 头文件。

```cpp
#include "graph/utils/type_utils.h"
```

## 函数原型

```cpp
static AscendString FormatToAscendString(const Format &format)
```

## 参数说明

| 参数   | 输入/输出 | 说明                                                         |
| ------ | --------- | ------------------------------------------------------------ |
| format | 输入      | 待转换的 Format，支持的 Format 请参考 15.2.3.59 Format。     |

## 返回值说明

转换后的 Format 字符串，AscendString 类型。

## 约束说明

无。

## 调用示例

```cpp
ge::Format format = ge::Format::FORMAT_NHWC;
auto format_str = ge::TypeUtils::FormatToAscendString(format);
const char *ptr = format_str.GetString();
```
