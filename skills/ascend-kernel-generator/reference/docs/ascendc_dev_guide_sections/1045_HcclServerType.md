###### HcclServerType

## 功能说明

配置 Hccl 的服务端类型。

## 函数原型

```cpp
void HcclServerType(enum HcclServerType type, const char *soc = nullptr)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| type | 输入 | Hccl 的服务端类型，类型为 `HcclServerType` 枚举类，定义如下：<br><br>```cpp<br>namespace ops {<br>enum HcclServerType : uint32_t {<br>    AICPU,  // AI CPU 服务端<br>    AICORE, // AI Core 服务端<br>    MAX     // 预留参数，不支持使用<br>};<br>}<br>``` |
| soc | 输入 | 昇腾 AI 处理器型号。为该型号配置服务端类型。<br><br>- 可选参数，`nullptr` 或者 `""` 表示为算子支持的所有型号配置服务端类型。<br>- soc 取值需确保在算子支持的昇腾 AI 处理器型号范围内，即已经调用 `AddConfig` 接口注册。<br>- 填写规则请参考算子工程目录下编译配置项文件 `CMakePresets.json` 中的 `ASCEND_COMPUTE_UNIT` 字段，该字段取值在使用 `msOpGen` 创建工程时自动生成。 |

## 约束说明

- 使用该接口前，算子需要先通过 MC2 接口注册该算子是通算融合算子，注册后即返回一个 `OpMC2Def` 结构。
- 同时为特定昇腾 AI 处理器型号和所有昇腾 AI 处理器型号配置服务端类型时，特定昇腾 AI 处理器型号配置的优先级更高。

## 调用示例

```cpp
class MC2Custom : public OpDef {
public:
    MC2Custom(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("z").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("group").AttrType(REQUIRED).String();
        this->AICore().AddConfig("ascendxxx1");
        this->AICore().AddConfig("ascendxxx2");
        this->MC2().HcclGroup("group"); // 配置通信域名称为 group
        this->MC2().HcclServerType(HcclServerType::AICPU, "ascendxxx1"); // 配置 ascendxxx1 型号的通信模式为 AI CPU
        this->MC2().HcclServerType(HcclServerType::AICORE); // 配置其他型号即 ascendxxx2 的通信模式为 AI Core
    }
};
OP_ADD(MC2Custom);
```
