##### GetInstanceStart

## 函数功能

获取算子某个 IR 输入在实际输入中的起始序号（index）。

## 函数原型

```cpp
size_t GetInstanceStart() const
```

## 参数说明

无。

## 返回值说明

算子某个 IR 输入在实际输入中的起始序号（index）。

## 约束说明

无。

## 调用示例

```cpp
AnchorInstanceInfo anchor_0(0, 10); // IR原型定义的第1个输入是动态输入，动态输入的实际输入个数为10
AnchorInstanceInfo anchor_1(10, 1); // IR原型定义的第2个输入是必选输入，必选输入的实际输入个数必需为1，该输入排在实际输入的第10个
auto start_of_anchor_1 = anchor_1.GetInstanceStart(); // start_of_anchor_1 = 10
```
