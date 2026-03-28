### Einsum

## 功能

Einsum算子使用爱因斯坦求和约定评估Tensor序列上的代数Tensor运算，形式为 `term1, term2 -> y_term`。公式字符串包含逗号分隔的小写字母序列。每个项对应于操作数Tensor，项中的字符对应于操作数维度。

## 输入

**x**：输入Tensor
- 数据类型：float16、float
- 支持非连续的tensor
- 不支持空tensor
- 数据格式支持ND
- 输入Tensor个数：1~2147483647个

## 属性

**equation**：爱因斯坦求和的下标，一个和表达式字符串。

## 输出

**y**：输出Tensor
- 数据类型：float16、float、int32
- 支持非连续的tensor
- 不支持空tensor
- 数据格式支持ND

## 约束

目前ONNX的Einsum只支持双输入的20种情况：

例如：`"abc,cde->abde"`
- `abc`表示第一个张量的维度，其中a、b和c是该Tensor的三个维度
- `cde`表示第二个张量的维度，其中c、d和e是该Tensor的三个维度
- `abde`表示输出Tensor的维度

在计算过程中，Einsum会沿着共享的维度c进行求和操作。也就是说，c维度在两个Tensor中都会出现，Einsum会对这个维度进行求和操作，而其余维度（a、b、d、e）会保留下来。

支持的表达式：
```
"abc,cde->abde"
"abc,cde->abde"
"abcd,aecd->aceb"
"abcd,adbe->acbe"
"abcd,cde->abe"
"abc,cd->abd"
"abc,dc->abd"
"abc,abd->dc"
"abc,dec->abde"
"abc,abde->dec"
"abcd,aecd->acbe"
"abcd,acbe->aecd"
"abcd,ecd->abe"
"abcd,abe->ecd"
"abcd,acbe->adbe"
"abcd,abde->abce"
"abcd,abce->abde"
"abcd,aebd->aebc"
"abcd,abce->acde"
"abc,abd->acd"
"ab,cb->ac"
```

## 支持的ONNX版本

支持的ONNX版本Opset：v12、v13、v14、v15、v16、v17、v18
