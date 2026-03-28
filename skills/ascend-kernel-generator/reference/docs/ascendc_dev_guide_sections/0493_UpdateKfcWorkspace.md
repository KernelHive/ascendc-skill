###### UpdateKfcWorkspace

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | × |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 AI Core | × |
| Atlas 推理系列产品 Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

更新用于 `CubeResGroupHandle` 消息通信区的内存地址。用户使用 `CubeResGroupHandle` 接口时，需要用此接口自主管理空间地址。

## 函数原型

```cpp
__aicore__ inline void UpdateKfcWorkspace(uint32_t offset)
```

## 参数说明

**表 接口参数说明**

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| offset | 输入 | 更新 workspace 地址为原地址偏移 offset 后的地址。 |

## 返回值说明

无。

## 约束说明

本接口不能和 `CreateCubeResGroup` 接口同时使用。

## 调用示例

```cpp
AscendC::KfcWorkspace desc(workspaceGM);
desc.UpdateKfcWorkspace(1024);
```
