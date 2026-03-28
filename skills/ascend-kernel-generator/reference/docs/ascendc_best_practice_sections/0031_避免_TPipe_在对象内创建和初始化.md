### 避免 TPipe 在对象内创建和初始化

## 优先级

中

## 编译器背景知识

创建类对象时，会分配内存空间，用于存储类中的相关成员变量或函数。当类中变量需要参与计算时，变量值从内存被加载到寄存器，计算完成后，变量从寄存器存储回内存。

Scalar 常量折叠和常量传播是编译器编译时的优化方式。优化前编译器会判断变量是否只初始化过一次或只赋值过一次，若满足此编译优化的前提条件，变量值将会尽量驻留在寄存器中，从而在后续使用变量时，将减少读取内存的操作，提升运行性能。

## 描述

TPipe 是用来管理全局内存和同步的框架，用户可以调用 TPipe 的接口，为 TQue/TBuf 进行内存分配。

在编写 Ascend C 算子过程中，经常用一个类存放计算所需的相关变量，这里称类名为 `KernelExample`。当 TPipe 对象在 `KernelExample` 类的实现中定义并初始化时，TPipe 对象的内存空间在整个 `KernelExample` 对象的内存空间之中。

需要注意的是，创建 TPipe 对象时，对象初始化会设置全局变量的 TPipe 指针，这导致 `KernelExample` 对象的内存有被外部污染的风险，此时编译器的编译优化将采取保守策略，不会对 `KernelExample` 对象中的 Scalar 变量进行常量折叠和常量传播。

因此，在任何场景下，我们都建议将 TPipe 对象创建于 `KernelExample` 类外部，使得 TPipe 对象的内存空间独立于 `KernelExample` 类对象的内存空间，触发编译器对 `KernelExample` 类内 Scalar 的编译优化，减少算子 Scalar 指令耗时。

## 反例

代码中 TPipe 对象由 `KernelExample` 类内部创建并初始化，影响编译器 Scalar 折叠优化，在 NPU 侧导致 Scalar 无谓增加。

```cpp
template <typename ComputeT> class KernelExample {
public:
    __aicore__ inline KernelExample() {}

    __aicore__ inline void Init(...)
    {
        ...
        pipe.InitBuffer(xxxBuf, BUFFER_NUM, xxxSize);
        ...
    }

private:
    ...
    TPipe pipe;
    ...
};

extern "C" __global__ __aicore__ void example_kernel(...)
{
    ...
    KernelExample<float> op;
    op.Init(...);
    ...
}
```

## 正例

改为由 Kernel 入口函数创建 TPipe 对象，在 `KernelExample` 类中保存 TPipe 指针使用。

```cpp
template <typename ComputeT> class KernelExample {
public:
    __aicore__ inline KernelExample() {}

    __aicore__ inline void Init(..., TPipe* pipeIn)
    {
        ...
        pipe = pipeIn;
        pipe->InitBuffer(xxxBuf, BUFFER_NUM, xxxSize);
        ...
    }

private:
    ...
    TPipe* pipe;
    ...
};

extern "C" __global__ __aicore__ void example_kernel(...)
{
    ...
    TPipe pipe;
    KernelExample<float> op;
    op.Init(..., &pipe);
    ...
}
```

## 性能对比

- 图 5-18 aiv_scalar_time 优化前后对比
- 图 5-19 aiv_scalar_ratio 优化前后对比

通过性能数据对比可以看出，Scalar 优化明显，平均时间从 281us 减少到 236us，下降 17%；平均 scalar_time 时延占比从 21% 下降到 17%。因此在 Scalar bound（达到上限）的场景下可以使用此优化措施。
