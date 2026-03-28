###### GetCoreNumVector

## 功能说明

用于获取硬件平台独立的 Vector Core 的核数。

该接口仅在 Atlas 推理系列产品有效，其他硬件平台型号均返回 0。

## 函数原型

```cpp
uint32_t GetCoreNumVector(void) const
```

## 参数说明

无

## 返回值说明

返回硬件平台 Vector Core 的核数。

## 约束说明

- Atlas 训练系列产品，不支持该接口，返回 0
- Atlas 推理系列产品，支持该接口，返回硬件平台 Vector Core 的核数
- Atlas A2 训练系列产品/Atlas A2 推理系列产品不支持该接口，返回 0
- Atlas A3 训练系列产品/Atlas A3 推理系列产品不支持该接口，返回 0
- Atlas 200I/500 A2 推理产品不支持该接口，返回 0

## 调用示例

```cpp
ge::graphStatus TilingXXX(gert::TilingContext* context) {
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  auto aivCoreNum = ascendcPlatform.GetCoreNumAiv();
  auto vectorCoreNum = ascendcPlatform.GetCoreNumVector();
  auto allVecCoreNums = aivCoreNum + vectorCoreNum;
  // ...按照 allVecCoreNums 切分
  return ret;
}
```
