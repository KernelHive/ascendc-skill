##### Inputs

## 函数功能

设置 Context 的 values 的输入指针，values 承载的类型为 `void*` 的指针数组。

## 函数原型

```cpp
OpKernelContextBuilder &Inputs(std::vector<void *> inputs)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| inputs | 输入 | 输入指针数组，所有权归调用者管理，调用者需要保证输入指针生命周期长于 Build 产生的 ContextHolder 对象。 |

## 返回值说明

返回 OpKernelContextBuilder 对象本身，用于链式调用。

## 约束说明

在调用 Build 方法之前，必须调用该接口，否则构造出的 KernelContext 将包含未定义数据。
