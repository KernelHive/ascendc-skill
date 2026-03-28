###### LoadTilingLibrary

## 功能说明

根据输入的路径，加载对应的 Tiling 动态库。开发者基于工程化算子开发方式完成算子实现后，可通过算子包编译或算子动态库编译获取对应的 Tiling 动态库文件。

- **算子包编译**：Tiling 实现对应的动态库为算子包部署目录下的 `liboptiling.so`。具体路径可参考 6.7.6.2 算子包部署。
- **动态库编译**：Tiling 实现集成在算子动态库 `libcust_opapi.so` 中。具体路径可参考 6.7.7 算子动态库和静态库编译。

## 函数原型

```cpp
bool LoadTilingLibrary(const char *tilingSoPath) const
```

## 参数说明

| 参数名       | 输入/输出 | 描述                                       |
|--------------|-----------|--------------------------------------------|
| tilingSoPath | 输入      | Tiling 动态库的路径，支持相对路径与绝对路径 |

## 返回值说明

- `true`：Tiling 动态库加载成功
- `false`：Tiling 动态库加载失败。具体错误可参考 Log 信息。

关于日志配置和查看，请参考《环境变量参考》中“辅助功能 > 日志”章节。

## 约束说明

无

## 调用示例

```cpp
context_ascendc::OpTilingRegistry tmpIns;
bool flag = tmpIns.LoadTilingLibrary("/your/path/to/so_path/liboptiling.so");
if (flag == false) {
    std::cout << "Load tiling so failed" << std::endl;
    // ...
}
// ...
```
