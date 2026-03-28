##### SetHF32TransMode

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

设置 HF32 模式取整的具体方式，需要先使用 `SetHF32Mode` 开启 HF32 取整模式。

## 函数原型

```cpp
__aicore__ inline void SetHF32TransMode(bool hf32TransMode)
```

## 参数说明

**表 15-433 参数说明**

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| `hf32TransMode` | 输入 | Mmad HF32 取整模式控制入参，bool 类型。支持如下两种取值：<br>● `true`：则 FP32 将以向零靠近的方式四舍五入为 HF32。<br>● `false`：则 FP32 将以最接近偶数的方式四舍五入为 HF32。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```cpp
bool hf32TransMode = true;
AscendC::SetHF32TransMode(hf32TransMode);
```
