## acldvppFinalize

## 支持的产品型号

- Atlas 推理系列产品
- Atlas 200I/500 A2 推理产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
- Atlas A3 训练系列产品/Atlas A3 推理系列产品

## 功能说明

DVPP去初始化函数，在算子功能接口之后、退出进程前调用本接口。重复调用本接口不会报错，建议 `acldvppInit` 接口与本接口配套使用，分别完成DVPP的初始化、去初始化。

调用本接口或 `aclFinalize` 接口，均可实现DVPP去初始化，两者区别在于：

- 调用本接口仅完成DVPP去初始化
- 调用 `aclFinalize` 接口可完成ACL接口中各子功能（包含DVPP）的去初始化

若两个接口都调用，也不返回失败。

## 函数原型

```c
acldvppStatus acldvppFinalize()
```

## 参数说明

无

## 返回值说明

返回 `acldvppStatus` 状态码，具体请参见 [6.2 acldvpp返回码]。
