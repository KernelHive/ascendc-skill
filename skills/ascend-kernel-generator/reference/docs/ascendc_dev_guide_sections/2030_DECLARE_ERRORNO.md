#### DECLARE_ERRORNO

该宏对外提供如下四个错误码供用户使用：

- **SUCCESS**：成功
- **FAILED**：失败
- **PARAM_INVALID**：参数不合法
- **SCOPE_NOT_CHANGED**：Scope融合规则未匹配到，忽略当前pass

声明如下所示：

```c
DECLARE_ERRORNO(0, 0, SUCCESS, 0);
DECLARE_ERRORNO(0xFF, 0xFF, FAILED, 0xFFFFFFFF);
DECLARE_ERRORNO_COMMON(PARAM_INVALID, 1); // 50331649
DECLARE_ERRORNO(SYSID_FWK, 1, SCOPE_NOT_CHANGED, 201);
```

您可以在 `${INSTALL_DIR}/compiler/include/register/register_error_codes.h` 下查看错误码定义。

其中，`${INSTALL_DIR}` 请替换为 CANN 软件安装后文件存储路径。若安装的是 Ascend-cann-toolkit 软件包，以 root 安装为例，则安装后文件存储路径为：`/usr/local/Ascend/ascend-toolkit/latest`。
