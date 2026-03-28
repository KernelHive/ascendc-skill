#### REGISTER_CUSTOM_PASS

开发人员可以选择将改图函数注册到框架中，由框架在编译最开始调用自定义改图 Pass。调用 `REGISTER_CUSTOM_PASS` 进行自定义 Pass 注册。

调用时以 `REGISTER_CUSTOM_PASS` 开始，以“.”连接 `CustomPassFn` 等接口。例如：

```cpp
#include "register/register_custom_pass.h"
REGISTER_CUSTOM_PASS("pass_name").CustomPassFn(CustomPassFunc);
```

您可以在 `${INSTALL_DIR}/include/register/register_custom_pass.h` 下查看接口定义。`${INSTALL_DIR}` 请替换为 CANN 软件安装后文件存储路径。若安装的是 Ascend-cann-toolkit 软件包，以 root 安装举例，则安装后文件存储路径为：`/usr/local/Ascend/ascend-toolkit/latest`。
