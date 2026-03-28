##### SetMMLayoutTransform

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 AI Core | × |
| Atlas 推理系列产品 Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

设置 Mmad 计算时优先通过 M/N 中的哪个方向。

## 函数原型

```cpp
__aicore__ inline void SetMMLayoutTransform(bool mmLayoutMode)
```

## 参数说明

**表 15-431 参数说明**

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| `mmLayoutMode` | 输入 | 控制 Mmad 优先通过 M/N 的哪个方向，bool 型，支持如下两种取值：<br>● `true`：代表 CUBE 将首先通过 N 方向，然后通过 M 方向产生结果。<br>● `false`：代表 CUBE 将首先通过 M 方向，然后通过 N 方向生成结果。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```cpp
bool mmLayoutMode = true;
AscendC::SetMMLayoutTransform(mmLayoutMode);
```
