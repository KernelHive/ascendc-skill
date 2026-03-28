#### 原型定义接口（REG_OP）

## 函数原型

函数原型定义示例如下：

```cpp
REG_OP(xxx)
.INPUT(x1, type)
.OPTIONAL_INPUT(x2, type)
.DYNAMIC_INPUT(x3, type)
.OUTPUT(y1, type)
.DYNAMIC_OUTPUT(y3, type)
.REQUIRED_ATTR(a, type)
.ATTR(b, type, default_value)
.GRAPH(z1)
.DYNAMIC_GRAPH(z2)
.OP_END_FACTORY_REG(xxx)
```

## 功能说明

定义算子的原型，包括算子的输入、输出、属性以及对应的数据类型。

进行如上算子原型定义后，即相当于向GE注册了该算子的原型，告知GE对应类型的算子应该具备哪些输入、输出与属性；同时相当于定义了一个op::xxx的Class，开发者可以include该原型头文件，然后实例化该Class进行IR模型构建，如下所示：

```cpp
conv = op::Conv2D()
conv.set_input_x(feature_map_data)
conv.set_input_filter(weight_data)
```

具体的模型构建可以参考《图模式开发指南》。

## 接口说明

| 接口名称 | 接口说明 | 衍生接口（可用于IR模型构建） |
|---------|---------|----------------------------|
| REG_OP(x) | 定义一个算子原型，算子类型为x | REG_OP |
| .INPUT(x, type) | 定义输入名称（x）和类型(type)。类型为TensorType类型，例如：<br>● TensorType{DT_FLOAT}<br>● TensorType({DT_FLOAT, DT_INT8})<br>● TensorType::ALL()<br>关于TensorType类，请参见TensorType类说明 | INPUT |
| .OPTIONAL_INPUT(x, type) | 定义可选输入的名称（x）和类型（type）。类型为TensorType类型，例如：<br>● TensorType{DT_FLOAT}<br>● TensorType({DT_FLOAT, DT_INT8})<br>● TensorType::ALL()<br>关于TensorType类，请参见TensorType类说明 | OPTIONAL_INPUT |
| .DYNAMIC_INPUT(x, type) | 定义动态输入的名称（x）和类型（type）。类型为TensorType类型，例如：<br>● TensorType{DT_FLOAT}<br>● TensorType({DT_FLOAT, DT_INT8})<br>● TensorType::ALL()<br>关于TensorType类，请参见TensorType类说明 | DYNAMIC_INPUT |
| .OUTPUT(x, type) | 定义输出的名称（x）和类型（type）。类型为TensorType类型，例如：<br>● TensorType{DT_FLOAT}<br>● TensorType({DT_FLOAT, DT_INT8})<br>● TensorType::ALL()<br>关于TensorType类，请参见TensorType类说明 | OUTPUT |
| .DYNAMIC_OUTPUT(x, type) | 定义动态输出的名称（x）和类型（type）。类型为TensorType类型，例如：<br>● TensorType{DT_FLOAT}<br>● TensorType({DT_FLOAT, DT_INT8})<br>● TensorType::ALL()<br>关于TensorType类，请参见TensorType类说明 | DYNAMIC_OUTPUT |
| .REQUIRED_ATTR(x, type) | 定义必备属性的名称（x）和类型（type）。type的可选值包括：<br>● Int，属性类型为int64_t<br>● Float，属性类型为float<br>● String，属性类型为string<br>● Bool，属性类型为bool<br>● Tensor，属性类型为Tensor<br>● Type，属性为Type枚举定义<br>● NamedAttrs，属性类型为NamedAttrs<br>● AscendString，属性类型为AscendString<br>● ListInt，属性类型为vector<int64_t>，int64_t列表<br>● ListFloat，属性类型为vector<float>，float列表<br>● ListString，属性类型为vector<string>，string列表<br>● ListBool，属性类型为vector<bool>，bool列表<br>● ListTensor，属性类型为vector<Tensor>，Tensor列表<br>● Bytes，属性类型为Buffer<br>● ListType，属性类型为vector<Type>，Type列表<br>● ListListInt，属性类型为vector<vector<int64_t>>，2维列表<br>● ListAscendString，属性类型为vector<AscendString>，AscendString列表<br>● ListNamedAttrs，属性类型为vector<NamedAttrs>，NamedAttrs列表 | REQUIRED_ATTR |
| .ATTR(x, type, default_value) | 定义可选属性的名称、类型以及默认值。当用户不设置算子对象的属性时，会使用此处设置的默认值。type的可选值包括：<br>● Int，属性类型为int64_t<br>● Float，属性类型为float<br>● String，属性类型为string<br>● Bool，属性类型为bool<br>● Tensor，属性类型为Tensor<br>● Type，属性为Type枚举定义<br>● NamedAttrs，属性类型为NamedAttrs<br>● AscendString，属性类型为AscendString<br>● ListInt，属性类型为vector<int64_t>，int64_t列表<br>● ListFloat，属性类型为vector<float>，float列表<br>● ListString，属性类型为vector<string>，string列表<br>● ListBool，属性类型为vector<bool>，bool列表<br>● ListTensor，属性类型为vector<Tensor>，Tensor列表<br>● Bytes，属性类型为Buffer<br>● ListType，属性类型为vector<Type>，Type列表<br>● ListListInt，属性类型为vector<vector<int64_t>>，2维列表<br>● ListAscendString，属性类型为vector<AscendString>，AscendString列表<br>● ListNamedAttrs，属性类型为vector<NamedAttrs>，NamedAttrs列表<br>定义示例：<br>● .ATTR(mode, Int, 1)<br>● .ATTR(pad, ListInt, {0, 0, 0, 0}) | ATTR |
| .GRAPH(z1) | 注册算子中包含的子图信息，输入z1为子图名称。例如If算子注册的子图为：.GRAPH(then_branch) .GRAPH(else_branch)。对于同一个算子，注册的算子子图名称需要保持唯一 | GRAPH |
| .DYNAMIC_GRAPH(z2) | 注册动态算子子图信息，输入z2为子图名称。例如Case算子注册的子图为：.DYNAMIC_GRAPH(branches)。对于同一个算子，注册的算子子图名称需要保持唯一 | DYNAMIC_GRAPH |
| .INFER_SHAPE_AND_TYPE() | 该接口为历史遗留兼容性接口，当前版本用户无需使用 | - |
| .OP_END_FACTORY_REG(x) | 与REG_OP配对，结束算子原型定义。算子类型（x）与REG_OP(x)中的类型相同 | - |

## 说明

OpReg类中的OpReg &N()接口的功能是为了用户进行算子注册的时候，使用.**的方式调用OpReg类的接口，例如.INPUT(x, type)、.OUTPUT(x, type)，无其他含义。

## 返回值

无

## 约束和限制说明

- REG_OP的算子类型必须全局唯一
- 同一个算子的输入名称之间不能重复
- 同一个算子的输出名称之间不能重复
- 同一个算子的属性名称之间不能重复

## 调用示例和相关 API

### 动态输入的算子原型定义示例

```cpp
REG_OP(AddN)
.DYNAMIC_INPUT(x, TensorType({NumberType(), DT_VARIANT}))
.OUTPUT(y, TensorType({NumberType(), DT_VARIANT}))
.REQUIRED_ATTR(N, Int)
.OP_END_FACTORY_REG(AddN)
```

### 多输入的算子原型定义示例

```cpp
REG_OP(GreaterEqual)
.INPUT(x1, TensorType::RealNumberType())
.INPUT(x2, TensorType::RealNumberType())
.OUTPUT(y, TensorType({DT_BOOL}))
.OP_END_FACTORY_REG(GreaterEqual)
```

### 注册子图的算子原型定义示例

```cpp
REG_OP(If)
.INPUT(cond, TensorType::ALL())
.DYNAMIC_INPUT(input, TensorType::ALL())
.DYNAMIC_OUTPUT(output, TensorType::ALL())
.GRAPH(then_branch)
.GRAPH(else_branch)
.OP_END_FACTORY_REG(If)
```
