###### SetCcTiling（废弃）

## 说明

该接口废弃，并将在后续版本移除，请不要使用该接口。请使用 SetCcTilingV2 接口设置通信算法的 Tiling 地址。

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品 AI Core | x |
| Atlas 推理系列产品 Vector Core | x |
| Atlas 训练系列产品 | x |

## 功能说明

用于设置 Hccl 客户端通信算法的 Tiling 地址。

## 函数原型

```cpp
__aicore__ inline int32_t SetCcTiling(__gm__ void *ccOpTilingData)
```

## 参数说明

**表 15-897 接口参数说明**

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| ccOpTilingData | 输入 | 通信算法的 Mc2CcTiling 参数的地址。Mc2CcTiling Data 在 Host 侧计算得出，具体请参考表 2 Mc2CcTiling 参数说明，由框架传递到 Kernel 函数中使用，完整示例请参考 8.13.1.2-调用示例。 |

## 返回值说明

- `HCCL_SUCCESS`，表示成功。
- `HCCL_FAILED`，表示失败。

## 约束说明

- 参数相同的同一种通信算法在调用 Prepare 接口前只需要调用一次本接口，否则需要多次调用本接口。请参考调用示例：
  - 仅有一个通信算法
  - 一个通信域有多个通信算法

- 同一种通信算法只支持设置一个 ccOpTilingData 地址；对于同一种通信算法，重复调用本接口会覆盖该通信算法的 ccOpTilingData 地址。请参考调用示例：
  - 一个通信域两个相同的算法

- 若调用本接口，必须与传 initTiling 地址的 Init 接口配合使用，且 Init 接口在本接口前被调用。

- 若调用本接口，必须使用标准 C++ 语法定义 TilingData 结构体的开发方式，具体请参考《Ascend C 算子开发指南》中的“算子实现 > 工程化算子开发 > Host 侧 tiling 实现 > 使用标准 C++ 语法定义 Tiling 结构体”。
