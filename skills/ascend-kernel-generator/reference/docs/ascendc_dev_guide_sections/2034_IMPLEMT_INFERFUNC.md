#### IMPLEMT_INFERFUNC

## 函数功能

封装算子的 InferShape 函数。

该函数传入的 OpType 为基于 Operator 类派生出来的子类，会自动生成一个类型为此子类的对象 op，可以使用子类的成员函数获取输入输出描述的方法，从而进行 InferShape 的实现。

基于 OpType 派生出来的子类 op 的成员函数如下：

- `op.set_input_x(Operator &v, const string &srcName)`：将网络中算子 v 的输出 srcName 设置为当前算子的输入 x。
- `op.get_input_desc_x()`：获取该算子的输入 x 的描述信息，返回对象为 TensorDesc 类型。
- `op.update_input_desc_x(const TensorDesc& tensorDesc)`：更新输入 x 的描述信息，包括 shape、datatype 与 format。
- `op.get_output_desc_y()`：获取该算子的输出 y 的描述信息，返回对象 TensorDesc 类型。
- `op.update_output_desc_y(const TensorDesc& tensorDesc)`：更新输出 y 的描述信息，包括 shape、datatype 与 format。
- `op.get_attr_attr1(AscendString &val)`：获取算子属性 attr1 的值 val。

## 函数原型

```cpp
IMPLEMT_INFERFUNC(op_name, func_name)
```

## 约束说明

无。

## 参数说明

| 参数名    | 输入/输出 | 描述                         |
|-----------|-----------|------------------------------|
| op_name   | 输入      | 算子类型                     |
| func_name | 输入      | InferShape 函数名，用户自定义 |

## 返回值

无。
