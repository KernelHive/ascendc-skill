#### ConvertToListAscendString

## 函数功能

定义了一个模板函数 `ConvertToListAscendString`，用于将不同类型的字符串列表转换为 `AscendString` 类型的列表。

## 函数原型

```cpp
template<typename T>
std::vector<ge::AscendString> ConvertToListAscendString(T strs)
```

## 支持的特化

- **`std::vector<std::string>` 类型**  
  对于 `std::vector<std::string>` 类型的字符串列表，先将其转换为 `std::vector<const char *>` 类型，然后再进行转换。

  ```cpp
  template<>
  inline std::vector<ge::AscendString>
  ConvertToListAscendString(std::vector<std::string> strs)
  ```

- **`std::vector<ge::AscendString>` 类型**  
  对于 `std::vector<ge::AscendString>` 类型的字符串列表，直接返回原列表。

  ```cpp
  template<>
  inline std::vector<ge::AscendString>
  ConvertToListAscendString(std::vector<ge::AscendString> strs)
  ```

## 参数说明

| 参数 | 输入/输出 | 说明               |
|------|------------|--------------------|
| strs | 输入       | 待转换的字符串列表 |

## 返回值

转换后的 `AscendString` 类型字符串列表。

## 异常处理

无。

## 约束说明

无。
