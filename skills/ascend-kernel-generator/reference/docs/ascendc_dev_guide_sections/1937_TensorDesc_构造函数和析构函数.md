##### TensorDesc 构造函数和析构函数

## 函数功能

TensorDesc 构造函数和析构函数。

## 函数原型

```cpp
TensorDesc()
~TensorDesc() = default
explicit TensorDesc(Shape shape, Format format = FORMAT_ND, DataType dt = DT_FLOAT)
TensorDesc(const TensorDesc &desc)
TensorDesc(TensorDesc &&desc)
TensorDesc &operator=(const TensorDesc &desc)
TensorDesc &operator=(TensorDesc &&desc)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| shape | 输入 | Shape 对象 |
| format | 输入 | Format 对象，默认取值 FORMAT_ND。<br>关于 Format 数据类型的定义，请参见 15.2.3.59 Format |
| dt | 输入 | DataType 对象，默认取值 DT_FLOAT。<br>关于 DataType 数据类型的定义，请参见 15.2.3.58 DataType |
| desc | 输入 | 待拷贝或者移动的 TensorDesc 对象 |

## 返回值

TensorDesc 构造函数返回 TensorDesc 类型的对象。

## 异常处理

无。

## 约束说明

无。
