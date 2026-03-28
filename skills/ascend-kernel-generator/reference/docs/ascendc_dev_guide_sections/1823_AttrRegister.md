##### AttrRegister

## 功能说明

泛型属性注册接口。

## 函数原型

```cpp
void AttrRegister(const char_t *name, const AttrValue &attr_value);
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| name | 输入 | 属性名 |
| attr_value | 输入 | 属性取值 |

## 返回值说明

graphStatus 类型：成功，返回 `GRAPH_SUCCESS`，否则，返回 `GRAPH_FAILED`。

## 异常处理

无

## 约束说明

无
