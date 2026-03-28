##### SetUserTag

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

为Tensor添加用户自定义信息，用户可以根据需要设置对应的Tag。后续可通过GetUserTag获取指定Tensor的Tag信息，并根据Tag信息对Tensor进行相应操作。

## 函数原型

```cpp
__aicore__ inline void SetUserTag(const TTagType tag)
```

## 参数说明

**表 15-38 参数说明**

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| tag | 输入 | 设置的Tag信息，类型TTagType对应为int32_t |

## 返回值说明

无

## 约束说明

无

## 调用示例

参考调用示例。
