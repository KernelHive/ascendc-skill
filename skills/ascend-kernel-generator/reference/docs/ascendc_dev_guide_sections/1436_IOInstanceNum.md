##### IOInstanceNum

## 函数功能

当输入 IR 原型实例个数不为 1 时（一般是可选输入或动态输入场景），需要设置算子每个输入 IR 原型的实例个数，用于构造各子类 Context 的基类 ExtendedKernelContext 中 ComputeNodeInfo 信息。

## 函数原型

```cpp
T &IOInstanceNum(
    const std::vector<uint32_t> &input_instance_num,
    const std::vector<uint32_t> &output_instance_num
)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| `input_instance_num` | 输入 | 是一个 vector 数组输入，vector 的 size 代表算子原型输入个数，vector 数组上每个位置的数字代表每个 IR 原型输入的实例个数。 |
| `output_instance_num` | 输入 | 是一个 vector 数组输入，vector 的 size 代表算子原型输出个数，vector 数组上每个位置的数字代表每个 IR 原型输出的实例个数。 |

## 返回值说明

返回子类对象 T 类型的引用，用于支持子类链式调用。

## 约束说明

此接口与 `IONum` 接口互斥。仅需调用 2 种接口的一种即可，以先调用的接口为准。
