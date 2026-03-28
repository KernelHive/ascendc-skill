##### DynamicInputRegister

## 函数功能

算子动态输入注册。

## 函数原型

```cpp
void DynamicInputRegister(const char_t *name, const uint32_t num, bool is_push_back = true)
void DynamicInputRegister(const char_t *name, const uint32_t num, const char_t *datatype_symbol, bool is_push_back = true)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| name | 输入 | 算子的动态输入名。 |
| num | 输入 | 添加的动态输入个数。 |
| datatype_symbol | 输入 | 动态输入的数据类型。 |
| is_push_back | 输入 | - true表示在尾部追加动态输入。<br>- false表示在头部追加动态输入。 |

## 返回值

无。

## 异常处理

无。

## 约束说明

无。
