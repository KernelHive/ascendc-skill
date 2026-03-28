##### OpName

## 函数功能

设置算子名称，用于构造各子类 Context 的基类 ExtendedKernelContext 中 ComputeNodeInfo 信息。

## 函数原型

```cpp
T &OpName(const ge::AscendString &op_name)
```

## 参数说明

| 参数    | 输入/输出 | 说明       |
|---------|-----------|------------|
| op_name | 输入      | 算子名称。 |

## 返回值说明

返回子类对象 T 类型的引用，用于支持子类链式调用。

## 约束说明

无
