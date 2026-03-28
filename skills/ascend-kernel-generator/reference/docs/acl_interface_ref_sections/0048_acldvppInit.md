## acldvppInit

## 支持的产品型号
- Atlas 推理系列产品
- Atlas 200I/500 A2 推理产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
- Atlas A3 训练系列产品/Atlas A3 推理系列产品

## 功能说明
DVPP初始化函数，需在调用算子功能接口前调用。重复调用本接口不会报错，建议本接口与`acldvppFinalize`接口配套使用，分别完成DVPP的初始化、去初始化。

调用本接口或`aclInit`接口，均可实现DVPP初始化，两者区别在于：
- 调用本接口仅完成DVPP初始化
- 调用`aclInit`接口可完成acl接口中各子功能（包含DVPP）的初始化

若两个接口都调用，也不返回失败。

## 函数原型
```c
acldvppStatus acldvppInit(const char *configPath)
```

## 参数说明
| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| configPath | 输入 | 预留参数，配置文件所在路径的指针，包含文件名，当前需要配置为空指针 |

## 返回值说明
返回`acldvppStatus`状态码，具体请参见6.2 acldvpp返回码。
