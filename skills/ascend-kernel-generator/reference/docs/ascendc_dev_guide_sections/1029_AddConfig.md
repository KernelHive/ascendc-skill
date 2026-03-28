###### AddConfig

## 功能说明

注册算子支持的AI处理器型号以及 OpAICoreConfig 信息。

## 函数原型

```cpp
void AddConfig(const char *soc)
void AddConfig(const char *soc, OpAICoreConfig &aicore_config)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| soc | 输入 | 支持的AI处理器型号。填写规则请参考算子工程目录下编译配置项文件 CMakePresets.json 中的 ASCEND_COMPUTE_UNIT 字段，该字段取值在使用 msOpGen 创建工程时自动生成。 |
| aicore_config | 输入 | AI Core配置信息请参考 OpAICoreConfig 定义。 |

## 返回值说明

无

## 约束说明

不传入 aicore_config 参数时，对 OpAICoreConfig 结构中的部分参数会配置成默认值，具体的参数和默认值如下表所示：

**表 不传入 aicore_config 参数时，OpAICoreConfig 默认配置**

| 配置参数 | 说明 | 默认值 |
|----------|------|--------|
| DynamicCompileStaticFlag | 用于标识该算子实现是否支持入图时的静态Shape编译。 | true |
| DynamicFormatFlag | 标识是否根据 SetOpSelectFormat 设置的函数自动推导算子输入输出支持的 dtype 和 format。 | true |
| DynamicRankSupportFlag | 标识算子是否支持 dynamicRank（动态维度）。 | true |
| DynamicShapeSupportFlag | 用于标识该算子是否支持入图时的动态Shape场景。 | true |
| NeedCheckSupportFlag | 标识是否在算子融合阶段调用算子参数校验函数进行 data type 与 Shape 的校验。 | false |
| PrecisionReduceFlag | 此字段用于进行 ATC 模型转换或者进行网络调测时，控制算子的精度模式。 | true |
