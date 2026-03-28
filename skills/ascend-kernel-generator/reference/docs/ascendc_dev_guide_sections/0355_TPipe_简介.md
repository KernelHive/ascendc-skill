###### TPipe 简介

TPipe 用于统一管理 Device 端内存等资源，一个 Kernel 函数必须且只能初始化一个 TPipe 对象。其主要功能包括：

## 内存资源管理

通过 TPipe 的 `InitBuffer` 接口，可以为 TQue 和 TBuf 分配内存，分别用于队列的内存初始化和临时变量内存的初始化。

## 同步事件管理

通过 TPipe 的 `AllocEventID`、`ReleaseEventID` 等接口，可以申请和释放事件 ID，用于同步控制。
