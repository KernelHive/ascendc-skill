#### ArgDescType

```cpp
头文件位于 CANN 软件安装后文件存储路径下的 `include/graph/arg_desc_info.h`。

```cpp
enum class ArgDescType {
  kIrInput = 0,       // 输入
  kIrOutput,          // 输出
  kWorkspace,         // Workspace地址
  kTiling,            // Tiling地址
  kHiddenInput,       // IR上不表达的额外输入
  kCustomValue,       // 自定义内容
  kIrInputDesc,       // 具有描述信息的输入地址
  kIrOutputDesc,      // 具有描述信息的输出地址
  kInputInstance,     // 实例化的输入
  kOutputInstance,    // 实例化的输出
  kEnd
};
```
