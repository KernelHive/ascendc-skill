import ast
import torch
import types

class CodeFilter:
    def __init__(self):
        self._build_forbidden_ops()
        
    def _build_forbidden_ops(self):
        all_ops_atens = {
            f"torch.ops.aten.{name}"
            for name in torch.ops.aten.__dict__.keys()
            if not name.startswith("__")
        }
        all_nns = {
            f"torch.nn.{name}"
            for name in torch.nn.__dict__.keys()
            if not name.startswith("__")
        }
        all_torchs = {
            f"torch.{name}"
            for name, obj in torch.__dict__.items()
            if not (name.startswith("__") and name.endswith("__"))
            and not isinstance(obj, types.ModuleType)
        }
        self.allowed_ops = {
            'torch.tensor','torch.ops.aten.eye', 'torch.rand', 'torch.empty_strided',
            'torch.nn.parameter', 'torch.ops.aten.mode', 'torch.ParameterDict',
            'torch.mode', 'torch.randn', 'torch.ops.aten.zeros_like', 'torch.zeros',
            'torch.nn.ParameterList', 'torch.ones_like', 'torch.nn.modules',
            'torch.nn.factory_kwargs', 'torch.ops.aten.new_zeros', 'torch.eye',
            'torch.ops.aten.new_empty', 'torch.empty_like', 'torch.nn.Parameter',
            'torch.FloatTensor', 'torch.ops.aten.view', 'torch.ops.aten.copy',
            'torch.ops.aten.new_ones', 'torch.ops.aten.ones_like', 'torch.zeros_like',
            'torch.ops.aten.new_empty_strided', 'torch.ops.aten.randn',
            'torch.ops.aten.empty_strided', 'torch.numel', 'torch.ops.aten.full_like',
            'torch.empty', 'torch.Tensor', 'torch.BoolTensor', 'torch.full',
            'torch.ops.aten.new_full', 'torch.ops.aten.full', 'torch.ops.aten.zeros',
            'torch.ops.aten.rand', 'torch.IntTensor', 'torch.full_like',
            'torch.nn.common_types', 'torch.nn.ModuleList', 'torch.ops.aten.ones',
            'torch.ones', 'torch.ops.aten.empty_like', 'torch.LongTensor', "torch.is_tensor"
        }
        
        self.forbidden_ops = (all_nns | all_ops_atens | all_torchs) - self.allowed_ops
        
    def filter_code(self, code):
        """
        规则：
        1. 只检查类 ModelNew，并且只关心它的 __init__ 和 forward。
        2. forward 中不能直接调用 forbidden 的 torch / torch.nn / torch.ops.aten。
        3. forward 中不能直接 self.xxx()，如果 xxx 在 __init__ 里是 forbidden_value。
        4. 整个文件必须 import custom_ops_lib（包括 from / as 等等写法）。
        5. 在 ModelNew 里（任意方法）必须至少调用一次 custom_ops_lib 的函数/算子。
        """
        try:
            tree = ast.parse(code)
            analyzer = ClassAnalyzer(self.forbidden_ops, target_class_name="ModelNew")
            analyzer.visit(tree)

            if not analyzer.has_model_class:
                analyzer.violations.append(
                    "You must define a class `ModelNew` that inherits from torch.nn.Module and the kernel function method you implemented in the custom_ops_lib will be called within ModelNew.."
                )

            # 全局规则：必须 import custom_ops_lib
            if not analyzer.has_custom_lib_import:
                analyzer.violations.append(
                    "You must import custom_ops_lib (import custom_ops_lib as xxx / from custom_ops_lib import xxx)"
                )

            # 全局规则：必须在 ModelNew 中调用 custom_ops_lib
            if not analyzer.has_custom_lib_call:
                analyzer.violations.append("The kernel function method you implemented in the custom_ops_lib must be called within ModelNew.")

            if analyzer.violations:
                # 只取第一条错误
                first = analyzer.violations[0]
                print("Code does not meet requirements, found the following violation:")
                print(f"- {first}")
                return False, first
            else:
                print("Code meets requirements!")
                return True, "success"

        except SyntaxError as e:
            print(f"Code syntax error: {e}")
            return False, f"Code syntax error: {e}"

class ClassAnalyzer(ast.NodeVisitor):
    def __init__(self, forbidden_ops, target_class_name="ModelNew"):
        self.forbidden_ops = forbidden_ops
        self.target_class_name = target_class_name

        self.init_attributes = {} 
        self.violations = []

        self.in_init = False
        self.in_forward = False
        self.current_class = None
        self.import_aliases = {}

        # custom_ops_lib 状态
        self.has_custom_lib_import = False   # 是否有 import custom_ops_lib 系列
        self.has_custom_lib_call = False     # 是否在 ModelNew 中调用过

        self.has_model_class = False

    # ---------- import 检查 ----------

    def visit_Import(self, node):
        # 不要因为 violations 直接 return，会影响 custom_ops_lib 检测
        for alias in node.names:
            self.import_aliases[alias.asname or alias.name] = alias.name
            # import custom_ops_lib / import custom_ops_lib as col
            if alias.name == "custom_ops_lib":
                self.has_custom_lib_import = True
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module_name = node.module
        for alias in node.names:
            full_name = f"{module_name}.{alias.name}"
            self.import_aliases[alias.asname or alias.name] = full_name
            # from custom_ops_lib import xxx
            if module_name == "custom_ops_lib":
                self.has_custom_lib_import = True
        self.generic_visit(node)

    # ---------- 类处理，只精确处理 ModelNew ----------

    def visit_ClassDef(self, node):
        # 我们会遍历所有类，但只有 target_class_name 会被当成“当前类”分析
        prev_class = self.current_class

        if node.name == self.target_class_name:
            self.current_class = node.name

            # 只对继承 torch.nn.Module 的 ModelNew 做模型分析
            is_module = False
            for base in node.bases:
                full_attr_str = self.get_full_attr(base)
                root = full_attr_str.split(".")[0]
                if root in self.import_aliases:
                    full_attr_str = full_attr_str.replace(root, self.import_aliases[root], 1)
                if full_attr_str == "torch.nn.Module":
                    is_module = True
                    break

            if is_module:
                self.has_model_class = True
                self.analyze_model_class(node)
            # 继续遍历类体（里面的函数）
            self.generic_visit(node)

            self.current_class = prev_class
        else:
            # 非 ModelNew 的类：不改变 current_class，直接遍历（但不会触发 forward 检查逻辑）
            self.generic_visit(node)

    def analyze_model_class(self, node):
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if item.name == '__init__':
                    self.in_init = True
                    self.visit(item)
                    self.in_init = False
                elif item.name == 'forward':
                    self.in_forward = True
                    self.visit(item)
                    self.in_forward = False
                    # 不要因为 violations 提前退出，这会影响 custom_ops_lib 的检测

    # ---------- 赋值（记录 self.xxx 类别） ----------

    def visit_Assign(self, node):
        if self.in_init:
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == 'self'
                ):
                    attr_name = target.attr
                    value = node.value

                    if isinstance(value, ast.Constant):
                        self.init_attributes[attr_name] = "simple_value"
                    elif isinstance(value, ast.Name):
                        self.init_attributes[attr_name] = "simple_name"
                    elif isinstance(value, ast.Call):
                        # self.xxx = self.some_method(...)
                        if (
                            isinstance(value.func, ast.Attribute)
                            and isinstance(value.func.value, ast.Name)
                            and value.func.value.id == 'self'
                        ):
                            self.init_attributes[attr_name] = "simple_func"
                        else:
                            is_forbidden, _ = self._check_forbidden_ops(value)
                            if is_forbidden:
                                self.init_attributes[attr_name] = "forbidden_value"
                            else:
                                self.init_attributes[attr_name] = "simple_tensor"
                    else:
                        self.init_attributes[attr_name] = "unknown"

        self.generic_visit(node)

    # ---------- 函数调用 ----------

    def visit_Call(self, node):
        # 1. 不管前面有没有 violation，优先检查 custom_ops_lib 调用
        if (
            self.current_class == self.target_class_name
            and not self.has_custom_lib_call
        ):
            if self._is_custom_lib_call(node):
                self.has_custom_lib_call = True

        # 2. 下面才是原来的 forward 规则检查
        if self.in_forward:
            is_forbidden, full_str = self._check_forbidden_ops(node)
            if is_forbidden:
                self.violations.append(
                    f"In the forward method, a prohibited method is directly called: {full_str}(). You should implement the operations from forward() in class `Model` in the custom_ops_lib and call it from there."
                )
                # 这里 return 只结束当前 Call 的深入遍历，不影响后续语句
                return

            # 检查 self.xxx()，且 xxx 在 __init__ 里是 forbidden_value
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'self'
            ):
                full_attr = self.get_full_attr(node.func)
                parts = full_attr.split(".")
                if len(parts) >= 2:
                    attr_name = parts[1]
                    if (
                        attr_name in self.init_attributes
                        and self.init_attributes[attr_name] == "forbidden_value"
                        and len(parts) == 2
                    ):
                        self.violations.append(
                            f"In the forward method, the model layer is directly called: self.{attr_name}(). You should implement the operations from forward() in class `Model` in the custom_ops_lib and call it from there."
                        )
                        return

        self.generic_visit(node)

    # ---------- 工具函数 ----------
    def get_full_attr(self, node, init=None):
        attrs = []
        while isinstance(node, ast.Attribute):
            attrs.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            attrs.append(node.id)
        if init:
            attrs[-1] = init
        return ".".join(reversed(attrs))

    def _check_forbidden_ops(self, node):
        # node 是 ast.Call
        if isinstance(node.func, ast.Attribute):
            full_attr_str = self.get_full_attr(node.func)  # 例如 "torch.matmul"
            root = full_attr_str.split(".")[0]
            if root in self.import_aliases:
                full_attr_str = full_attr_str.replace(
                    root, self.import_aliases[root], 1
                )
            if full_attr_str in self.forbidden_ops:
                return True, full_attr_str

        elif isinstance(node.func, ast.Name):
            func_name = node.func.id
            if (
                func_name in self.import_aliases
                and self.import_aliases[func_name].startswith('torch')
            ):
                full_attr_str = self.import_aliases[func_name]
                if full_attr_str in self.forbidden_ops:
                    return True, full_attr_str

        return False, ""

    def _is_custom_lib_call(self, node):
        """
        判断某个调用是否来自 custom_ops_lib：
        - import custom_ops_lib; custom_ops_lib.xxx(...)
        - import custom_ops_lib as col; col.xxx(...)
        - from custom_ops_lib import my_op; my_op(...)
        """
        # custom_ops_lib.xxx(...)
        if isinstance(node.func, ast.Attribute):
            full_attr_str = self.get_full_attr(node.func)  # "custom_ops_lib.xxx" 或 "col.xxx"
            parts = full_attr_str.split(".")
            if not parts:
                return False
            root = parts[0]
            # root 可能是别名
            if root in self.import_aliases:
                root_full = self.import_aliases[root]
            else:
                root_full = root
            if root_full == "custom_ops_lib":
                return True

        # from custom_ops_lib import xxx; xxx(...)
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.import_aliases:
                full_name = self.import_aliases[func_name]  # "custom_ops_lib.xxx"
                if full_name.startswith("custom_ops_lib."):
                    return True

        return False

def filter_code_result(code: str):

    f = CodeFilter()
    return f.filter_code(code)

def find_EXEC_NPU_CMD(generated_code: str):
    """
    检查 generated_code 中是否存在非注释的 EXEC_NPU_CMD。
    同时处理 C++ 单行注释(//)和块注释(/* ... */)。
    返回 (True, "pass 1/2") 当且仅当存在至少一处非注释的 EXEC_NPU_CMD。
    """
    in_block_comment = False
    found_only_in_comments = False

    for line in generated_code.splitlines():
        # 处理块注释状态
        if in_block_comment:
            if "*/" in line:
                in_block_comment = False
            continue

        # 处理本行内块注释（/* ... */ 同行开闭）
        effective_line = line
        if "/*" in effective_line:
            open_pos = effective_line.index("/*")
            if "*/" in effective_line[open_pos:]:
                close_pos = effective_line.index("*/", open_pos) + 2
                effective_line = effective_line[:open_pos] + effective_line[close_pos:]
            else:
                in_block_comment = True
                effective_line = effective_line[:open_pos]

        if "EXEC_NPU_CMD" not in effective_line:
            continue

        # 检查是否在单行注释后面
        comment_pos = effective_line.find("//")
        cmd_pos = effective_line.find("EXEC_NPU_CMD")
        if comment_pos != -1 and comment_pos < cmd_pos:
            # EXEC_NPU_CMD 在 // 注释之后
            found_only_in_comments = True
            continue

        # 找到有效的非注释 EXEC_NPU_CMD
        return True, "pass 1/2"

    if found_only_in_comments:
        return False, "EXEC_NPU_CMD only found in comments"
    return False, "EXEC_NPU_CMD not defined"

def filter_code_result_all(generated_code: str):
    context = {}
    try:
        compile(generated_code, "<string>", "exec")
        exec(generated_code, context)  # For Python, use exec() (be careful with untrusted code)
    except Exception as e:
        raise Exception(f'Error in generated code {e}')
    python_bind_src = context['python_bind_src']
    
    valid1 = find_EXEC_NPU_CMD(python_bind_src)
    if valid1[0]:
        valid2 = filter_code_result(context['model_src'])
        return valid2
    else:
        return valid1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Filter hacked code")
    parser.add_argument("--code_path", type=str, required=True)
    args = parser.parse_args()
    with open(args.code_path, "r", encoding="utf-8") as f:
        code = f.read()
    print(filter_code_result_all(code))