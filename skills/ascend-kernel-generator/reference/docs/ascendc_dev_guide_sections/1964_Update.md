##### Update

## 函数功能
更新 TensorDesc 的 Shape、Format、DataType 属性。

## 函数原型
```cpp
void Update(const Shape &shape, Format format = FORMAT_ND, DataType dt = DT_FLOAT)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| shape  | 输入      | 需刷新的 Shape 对象。 |
| format | 输入      | 需刷新的 Format 对象，默认取值 `FORMAT_ND`。 |
| dt     | 输入      | 需刷新的 DataType 对象，默认取值 `DT_FLOAT`。 |

## 返回值
无。

## 异常处理
无。

## 约束说明
无。
