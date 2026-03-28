# AscendC Kernel Generator 工作流程图

本文档提供了 AscendC Kernel Generator 的详细工作流程图，包括主流程和错误处理流程。

## 主流程图

```mermaid
flowchart TD
    Start([用户输入算子需求]) --> Parse[解析需求<br/>提取算子名、类别、I/O规格、数学语义]
    Parse --> CheckTorch{用户是否提供<br/>Torch实现?}
    
    CheckTorch -->|是| ValidateTorch[验证Torch实现格式<br/>包含Model、get_inputs等]
    CheckTorch -->|否| GenTorch[生成Torch参考实现<br/>add_torch_reference.py]
    
    ValidateTorch -->|格式正确| UseTorch[直接使用用户实现]
    ValidateTorch -->|格式不符| ModifyTorch[修改为用户实现格式]
    
    GenTorch --> Design[设计算子原型与AscendC方案]
    UseTorch --> Design
    ModifyTorch --> Design
    
    Design --> Search[检索相似算子实现<br/>golden_solutions/ref_repositories/docs]
    Search --> GenKernel[生成AscendC Kernel描述文件<br/>project_json_src/host_tiling_src<br/>kernel_src/python_bind_src/model_src]
    
    GenKernel --> Verify[调用验证服务<br/>scripts/verify.py]
    Verify --> Compile[编译验证]
    Compile -->|失败| CompileError[编译错误]
    Compile -->|成功| Correctness[正确性验证]
    
    Correctness -->|失败| CorrectnessError[正确性错误]
    Correctness -->|成功| Performance[性能测试]
    
    Performance -->|失败| PerfError[性能异常]
    Performance -->|成功| Success[验证成功]
    
    CompileError --> Analyze[错误分析<br/>检索参考实现和文档]
    CorrectnessError --> Analyze
    PerfError --> Analyze
    
    Analyze --> Fix[修复代码<br/>最小必要修改]
    Fix --> Verify
    
    Success --> SaveGolden[保存Golden Solution<br/>scripts/save_golden_solution.py]
    SaveGolden --> UpdateInfo[更新info.json<br/>记录性能指标和硬件信息]
    UpdateInfo --> Cleanup[清理临时文件<br/>scripts/cleanup_tmp.py]
    Cleanup --> End([完成])
    
    style Start fill:#e1f5ff
    style End fill:#c8e6c9
    style Success fill:#c8e6c9
    style CompileError fill:#ffcdd2
    style CorrectnessError fill:#ffcdd2
    style PerfError fill:#fff9c4
    style SaveGolden fill:#b3e5fc
    style Search fill:#f3e5f5
```

## 错误修复经验提取流程

```mermaid
flowchart TD
    Start([验证失败<br/>生成traj.json]) --> CheckTraj{traj.json<br/>是否存在?}
    CheckTraj -->|否| End1([跳过经验提取])
    CheckTraj -->|是| ErrorStats[统计错误代码数量<br/>get_error_code_num.py]
    
    ErrorStats --> CheckCount{错误代码数量<br/>和正确代码数量<br/>是否都>0?}
    CheckCount -->|否| End1
    CheckCount -->|是| GenPairs[生成错误-修复代码对<br/>get_code_diff.py<br/>输出: error_fix_pairs.json]
    
    GenPairs --> LoopStart{遍历所有<br/>错误-修复对}
    LoopStart --> ViewContent[查看错误修复内容<br/>get_content.py<br/>显示: error_code/code_diff/error]
    
    ViewContent --> Analyze[分析错误原因<br/>和修复方案]
    Analyze --> WriteSummary[撰写修复总结<br/>extract_error_fix_into_experience.py<br/>写入summary字段]
    
    WriteSummary --> NextPair{还有更多<br/>错误-修复对?}
    NextPair -->|是| LoopStart
    NextPair -->|否| SaveExp[保存到经验库<br/>save_fix_traj.py<br/>写入: reference/issues/op.json]
    
    SaveExp --> End2([完成经验提取])
    
    style Start fill:#e1f5ff
    style End1 fill:#ffcdd2
    style End2 fill:#c8e6c9
    style Analyze fill:#fff9c4
    style SaveExp fill:#b3e5fc
```

## 验证服务调用流程

```mermaid
sequenceDiagram
    participant User as 用户/Agent
    participant Client as verify.py客户端
    participant Server as verify_server.py
    participant Env as multi-kernel-bench Env
    participant NPU as NPU设备
    
    User->>Client: 调用 verify.py<br/>--op --reference_path --kernel_code_path
    Client->>Server: HTTP POST /verify<br/>{op, reference_path, kernel_code_path}
    
    Server->>Env: 创建Env实例
    Env->>Env: 加载Torch参考实现
    Env->>Env: 加载AscendC Kernel代码
    
    Env->>NPU: 编译算子
    NPU-->>Env: 编译结果
    
    alt 编译成功
        Env->>NPU: 运行正确性测试
        NPU-->>Env: 正确性结果
        
        alt 正确性通过
            Env->>NPU: 运行性能测试
            NPU-->>Env: 性能指标
            Env-->>Server: {exit_code: 0, correctness: true, performance: {...}}
        else 正确性失败
            Env-->>Server: {exit_code: 1, correctness: false, error: {...}}
        end
    else 编译失败
        Env-->>Server: {exit_code: 1, compile_error: {...}}
    end
    
    Server-->>Client: JSON响应
    Client->>Client: 保存结果到result_json_path
    Client-->>User: 返回验证结果
```

## 知识库检索流程

```mermaid
flowchart LR
    Start([需要检索参考实现]) --> CheckType{检索类型}
    
    CheckType -->|Golden Solutions| SearchGS[检索<br/>reference/golden_solutions/*.py]
    CheckType -->|开源仓库| SearchRepo[检索<br/>reference/ref_repositories/]
    CheckType -->|官方文档| SearchDocs[检索<br/>reference/docs/]
    
    SearchGS --> FilterGS[过滤相似算子<br/>按类别/名称匹配]
    SearchRepo --> FilterRepo[过滤相似算子<br/>按功能/API匹配]
    SearchDocs --> FilterDocs[过滤相关章节<br/>按关键词匹配]
    
    FilterGS --> Merge[合并检索结果]
    FilterRepo --> Merge
    FilterDocs --> Merge
    
    Merge --> Rank[按相关性排序]
    Rank --> Return[返回Top-K结果]
    Return --> Use[用于代码生成/修复]
    
    style Start fill:#e1f5ff
    style Use fill:#c8e6c9
    style Merge fill:#f3e5f5
```

## 数据流图

```mermaid
flowchart TB
    subgraph Input["输入数据"]
        UserReq[用户需求]
        TorchRef[Torch参考实现]
    end
    
    subgraph Process["处理流程"]
        Parse[需求解析]
        Gen[代码生成]
        Verify[验证]
    end
    
    subgraph Knowledge["知识库"]
        Golden[Golden Solutions<br/>reference/golden_solutions/]
        Issues[错误修复经验<br/>reference/issues/]
        Docs[官方文档<br/>reference/docs/]
        Repos[开源仓库<br/>reference/ref_repositories/]
    end
    
    subgraph Output["输出数据"]
        KernelCode[AscendC Kernel代码]
        VerifyResult[验证结果JSON]
        SavedGolden[保存的Golden Solution]
        SavedExp[保存的修复经验]
    end
    
    UserReq --> Parse
    TorchRef --> Parse
    Parse --> Gen
    Golden --> Gen
    Docs --> Gen
    Repos --> Gen
    Gen --> KernelCode
    KernelCode --> Verify
    Verify --> VerifyResult
    VerifyResult -->|成功| SavedGolden
    VerifyResult -->|失败| Issues
    Issues --> SavedExp
    SavedGolden --> Golden
    SavedExp --> Issues
    
    style Input fill:#e3f2fd
    style Process fill:#f3e5f5
    style Knowledge fill:#fff9c4
    style Output fill:#c8e6c9
```

## 阶段说明

### 阶段 1: 需求输入与解析
- 用户提供算子需求（语义描述、Torch 伪代码或输入输出规格）
- 系统解析并提取：
  - 算子名称和类别
  - 输入/输出张量规格
  - 数学语义
  - 约束条件

### 阶段 2: Torch 参考实现
- 检查用户是否提供 Torch 实现
- 如果提供，验证格式是否符合要求
- 如果未提供或格式不符，生成标准格式的 Torch 参考实现

### 阶段 3: 算子设计
- 设计算子原型（JSON 描述）
- 设计 Host 侧实现（tiling、operator）
- 设计 Kernel 侧实现
- 检索相似算子作为参考

### 阶段 4: 代码生成
- 生成完整的 AscendC Kernel 描述文件
- 包含所有必需的代码段（project_json_src、host_tiling_src 等）

### 阶段 5: 验证测试
- 编译验证
- 正确性验证
- 性能测试

### 阶段 6: 结果处理
- **成功**：保存为 Golden Solution
- **失败**：进入错误分析和修复循环

### 阶段 7: 后处理
- 清理临时文件
- 更新知识库索引
