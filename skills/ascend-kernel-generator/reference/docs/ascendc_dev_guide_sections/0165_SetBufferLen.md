##### SetBufferLen

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

设置Buffer长度。当用户调用`operator[]`函数创建新LocalTensor时，建议调用该接口设置新LocalTensor长度，便于编译器对内存及同步进行自动优化。

## 函数原型

```cpp
__aicore__ inline void SetBufferLen(uint32_t dataLen)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| dataLen | 输入 | Buffer长度，单位为字节 |

## 返回值说明

无

## 约束说明

无

## 调用示例

参考调用示例。
