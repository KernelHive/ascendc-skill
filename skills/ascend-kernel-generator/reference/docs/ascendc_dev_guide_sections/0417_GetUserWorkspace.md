###### GetUserWorkspace

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | √ |

## 功能说明

获取用户使用的workspace指针。workspace的具体介绍请参考[12.6 如何使用workspace](../12.6-how-to-use-workspace.md)。

Kernel直调开发方式下，如果未开启HAVE_WORKSPACE编译选项，框架不会自动设置系统workspace。如果使用了[15.1.5.2.1 Matmul Kernel侧接口](../15.1.5.2.1-matmul-kernel-interface.md)等需要系统workspace的高阶API，kernel侧需要通过`SetSysWorkSpace`设置系统workspace，此时用户workspace需要通过该接口获取。

## 函数原型

```cpp
__aicore__ inline GM_ADDR GetUserWorkspace(GM_ADDR workspace)
```

## 参数说明

**表 15-386 接口参数说明**

| 参数名称 | 输入/输出 | 描述 |
|----------|-----------|------|
| workspace | 输入 | 传入workspace的指针，包括系统workspace和用户使用的workspace |

## 约束说明

无

## 返回值说明

用户使用workspace指针。
