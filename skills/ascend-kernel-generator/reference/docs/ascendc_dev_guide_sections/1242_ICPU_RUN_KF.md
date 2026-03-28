#### ICPU_RUN_KF

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 AI Core | √ |
| Atlas 推理系列产品 Vector Core | x |
| Atlas 训练系列产品 | √ |

## 功能说明

进行核函数的 CPU 侧运行验证时，CPU 调测总入口，完成 CPU 侧的算子程序调用。

## 函数原型

```c
#define ICPU_RUN_KF(func, blkdim, ...)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| func | 输入 | 算子的 kernel 函数指针 |
| blkdim | 输入 | 算子的核心数，corenum |
| ... | 输入 | 所有的入参和出参，依次填入，当前参数个数限制为32个，超出32时会出现编译错误 |

## 返回值说明

无

## 约束说明

- 除了 func、blkdim 以外，其他的变量都必须是通过 GmAlloc 分配的共享内存的指针
- 传入的参数的数量和顺序都必须和 kernel 保持一致

## 调用示例

```c
ICPU_RUN_KF(sort_kernel0, coreNum, (uint8_t*)x, (uint8_t*)y);
```
