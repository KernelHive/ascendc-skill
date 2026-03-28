#### ConvertToAscendString

## 函数功能

模板函数，接受一个模板参数 T，并将其转换为 AscendString 类型。这个函数的主要功能是将不同类型的字符串转换为 AscendString 类型。

## 函数原型

```cpp
template<typename T> ge::AscendString ConvertToAscendString(T str)
```

## 支持类型

支持以下几种拓展：

- `template<> inline ge::AscendString ConvertToAscendString<const char*>(const char *str)`  
  对于 `const char*` 类型的字符串，直接使用 AscendString 的构造函数进行转换。

- `template<> inline ge::AscendString ConvertToAscendString<std::string>(std::string str)`  
  对于 `std::string` 类型的字符串，先将其转换为 `const char*` 类型，然后再进行转换。

- `template<> inline ge::AscendString ConvertToAscendString<ge::AscendString>(ge::AscendString str)`  
  对于 `AscendString` 类型的字符串，直接返回 AscendString 类型字符串。

## 参数说明

| 参数 | 输入/输出 | 说明           |
|------|------------|----------------|
| str  | 输入       | 待转换的字符串 |

## 返回值

转换后的 AscendString 类型字符串。

## 异常处理

无。

## 约束说明

无。
