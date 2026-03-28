###### GetBaseAddr

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | √ |

## 功能说明

根据传入的 `logicPos`（逻辑抽象位置），获取该位置的基础地址，只在 CPU 调试场景下此接口生效。通常用于计算 Tensor 在 `logicPos` 的偏移地址，即 Tensor 地址减去 `GetBaseAddr` 返回值。

## 函数原型

```cpp
inline uint8_t* GetBaseAddr(int8_t logicPos)
```

## 参数说明

| 参数名 | 输入/输出 | 含义 |
|--------|-----------|------|
| `logicPos` | 输入 | 逻辑位置类型。该类型具体说明请参考 15.1.4.4.12 `TPosition`。 |

## 约束说明

NA

## 返回值说明

逻辑位置对应的基地址。

## 调用示例

```cpp
auto absAddr = GetTPipePtr()->GetBaseAddr(static_cast<int8_t>(pos));
```
