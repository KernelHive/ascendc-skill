###### GetCoreNumAiv

## 功能说明

获取当前硬件平台 AI Core 中 Vector 核数。若 AI Core 的架构为 Cube、Vector 分离模式，返回 Vector Core 的核数；耦合模式返回 AI Core 的核数。

## 函数原型

```cpp
uint32_t GetCoreNumAiv(void) const
```

## 参数说明

无

## 返回值说明

- 针对 Atlas 训练系列产品，耦合模式，返回 AI Core 的核数
- 针对 Atlas 推理系列产品，耦合模式，返回 AI Core 的核数
- Atlas A2 训练系列产品/Atlas A2 推理系列产品，分离模式，返回 Vector Core 的核数
- Atlas A3 训练系列产品/Atlas A3 推理系列产品，分离模式，返回 Vector Core 的核数

## 约束说明

无

## 调用示例

```cpp
ge::graphStatus TilingXXX(gert::TilingContext* context) {
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  auto aicNum = ascendcPlatform.GetCoreNumAic();
  auto aivNum = ascendcPlatform.GetCoreNumAiv();
  // ...按照 aivNum 切分
  context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum));
  return ret;
}
```
