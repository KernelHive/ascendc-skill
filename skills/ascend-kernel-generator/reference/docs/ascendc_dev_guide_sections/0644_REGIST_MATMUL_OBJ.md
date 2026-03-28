###### REGIST_MATMUL_OBJ

## 功能说明

初始化 Matmul 对象。

## 函数原型

```c
REGIST_MATMUL_OBJ(tpipe, workspace, ...)
```

## 参数说明

| 参数名     | 输入/输出 | 描述                                                                 |
|------------|-----------|----------------------------------------------------------------------|
| tpipe      | 输入      | Tpipe 对象                                                           |
| workspace  | 输入      | 系统 workspace 指针                                                  |
| ...        | 输入      | 可变参数，传入 Matmul 对象和与之对应的 Tiling 结构，要求 Tiling 结构的数据类型为 TCubeTiling 结构。Tiling 参数可以通过 Host 侧 GetTiling 接口获取，并传递到 kernel 侧使用 |

## 返回值说明

无

## 约束说明

- 在分离模式中，本接口必须在 InitBuffer 接口前调用。
- 在程序中，最多支持定义 4 个 Matmul 对象。
- 当代码中只有一个 Matmul 对象时，本接口可以不传入 tiling 参数，通过 Init 接口单独传入 tiling 参数。
- 当代码中有多个 Matmul 对象时，必须满足 Matmul 对象与其 tiling 参数一一对应，依次传入，具体方式请参考调用示例。

## 调用示例

```c
Tpipe pipe;

// 推荐：初始化单个 matmul 对象，传入 tiling 参数
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);

// 推荐：初始化多个 matmul 对象，传入对应的 tiling 参数
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm1, mm1tiling, mm2, mm2tiling, mm3, mm3tiling, mm4, mm4tiling);

// 初始化单个 matmul 对象，未传入 tiling 参数。注意，该场景下需要使用 Init 接口单独传入 tiling 参数。这种方式将 matmul 对象的初始化和 tiling 的设置分离，比如，Tiling 可变的场景，可通过这种方式多次对 Tiling 进行重新设置
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm);
mm.Init(&tiling);
```
