##### GetInstanceNum

## 函数功能
获取 IR 原型定义某个输入对应的实际输入个数。

## 函数原型
```cpp
size_t GetInstanceNum() const
```

## 参数说明
无。

## 返回值说明
IR 原型定义某个输入对应的实际输入个数。

## 约束说明
无。

## 调用示例
```cpp
AnchorInstanceInfo anchor_0(0, 10); // IR 原型定义的第一个输入是动态输入，且有 10 个实际输入
auto input_num_0 = anchor_0.GetInstanceNum(); // input_num_0 = 10
```
