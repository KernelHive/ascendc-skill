import os
import subprocess
import shutil
import random
import string
import re
import json
from config import op_engineer_dir, ascendc_device, project_root_path, system_name
from utils.utils import underscore_to_pascalcase
from datetime import datetime


def _prepare_readable_opp_overlay(base_dir):
    root_opp = "/usr/local/Ascend/ascend-toolkit/latest/opp"
    vendors_config = os.path.join(root_opp, "vendors", "config.ini")
    if os.access(vendors_config, os.R_OK):
        return None

    overlay_dir = os.path.join(base_dir, "opp_overlay")
    vendors_dir = os.path.join(overlay_dir, "vendors")
    os.makedirs(vendors_dir, exist_ok=True)

    with open(os.path.join(vendors_dir, "config.ini"), "w", encoding="utf-8") as f:
        f.write("load_priority=customize\n")

    link_names = [
        "built-in",
        "op_impl",
        "op_proto",
        "framework",
        "scene.info",
        "version.info",
    ]
    for name in link_names:
        link_path = os.path.join(overlay_dir, name)
        if os.path.lexists(link_path):
            os.unlink(link_path)
        os.symlink(os.path.join(root_opp, name), link_path)

    customize_link = os.path.join(vendors_dir, "customize")
    if os.path.lexists(customize_link):
        os.unlink(customize_link)
    os.symlink(os.path.join(root_opp, "vendors", "customize"), customize_link)
    return overlay_dir


def _pascal_to_snake(name):
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def ascend_compile(generated_code, op, context):
    op = op + '_custom'
    op_capital=underscore_to_pascalcase(op)
    
    # Generate unique operator identifier with random 16-character string
    random_suffix = ''.join(random.choices(string.ascii_letters.lower(), k=16))
    date_suffix = datetime.now().strftime("%Y%m%d%H%M%S")
    op_identifier = f"{op}_{random_suffix}{date_suffix}"
    op_capital_identifier = underscore_to_pascalcase(op_identifier)
    generated_code = re.sub(op_capital, op_capital_identifier, generated_code)
    generated_code = re.sub(op, op_identifier, generated_code)
    
    # Create unique directory in tmp
    tmp_dir = os.path.join(project_root_path, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    target_directory = os.path.join(tmp_dir, op_identifier)
    deploy_path = os.path.join(target_directory, 'opp')
    os.makedirs(target_directory, exist_ok=True)
    # Store the operator identifier in context for later use
    context['op_identifier'] = op_identifier
    context['op_capital_identifier'] = op_capital_identifier

    try:
        compile(generated_code, "<string>", "exec")
        exec(generated_code, context)  # For Python, use exec() (be careful with untrusted code)
    except Exception as e:
        raise Exception(f'Error in generated code: {e}')
    
    # create ascendc project
    if os.path.exists(target_directory):
        print("[INFO] Operator project already exists, deleted")
        shutil.rmtree(target_directory)
    os.makedirs(target_directory, exist_ok=True)
    with open(os.path.join(target_directory, f'{op_identifier}.json'), 'w') as f:
        f.write(context.get('project_json_src'))
    try:
        print("[INFO] Begin create operator project")
        os.chdir(target_directory)
        result = subprocess.run(["msopgen", 'gen', '-i', f'{op_identifier}.json', '-c', ascendc_device, '-lan', 'cpp', '-out', op_capital_identifier], check=True, capture_output=True, text=True)
        print("[INFO] Create operator project succeeded")
    except subprocess.CalledProcessError as e:
        print("[INFO] Create operator project failed!")
        # print("Exit Code:", e.returncode)
        # print("Error Output:\n", e.stdout)
        # print("Error Output:\n", e.stderr)
        feedback = f'Exit Code: {e.returncode}\nError Output:\n{e.stdout}\n\n{e.stderr}'
        raise Exception(feedback) 

    # write code to specific location
    project_desc = json.loads(context.get('project_json_src'))
    project_op = project_desc[0]['op']
    project_op_snake = _pascal_to_snake(project_op)

    with open(os.path.join(target_directory, op_capital_identifier, 'op_host', f'{project_op_snake}_tiling.h'), 'w') as f:
        f.write(context.get('host_tiling_src'))

    with open(os.path.join(target_directory, op_capital_identifier, 'op_host', f'{project_op_snake}.cpp'), 'w') as f:
        f.write(context.get('host_operator_src'))

    with open(os.path.join(target_directory, op_capital_identifier, 'op_kernel', f'{project_op_snake}.cpp'), 'w') as f:
        f.write(context.get('kernel_src'))

    # Copy CppExtension to target directory
    cpp_extension_src = os.path.join(op_engineer_dir, 'CppExtension')
    cpp_extension_dst = os.path.join(target_directory, 'CppExtension')
    if os.path.exists(cpp_extension_dst):
        shutil.rmtree(cpp_extension_dst)
    shutil.copytree(cpp_extension_src, cpp_extension_dst)
    
    with open(os.path.join(cpp_extension_dst, 'csrc', f'op.cpp'), 'w') as f:
        f.write(context.get('python_bind_src'))

    try:
        environ_varible = 'ASCEND_CUSTOM_OPP_PATH' # this varible will purturb build if set
        os.environ.pop(environ_varible, None)
        print("[INFO] Begin build")
        os.chdir(os.path.join(target_directory, op_capital_identifier))
        build_env = os.environ.copy()
        overlay_dir = _prepare_readable_opp_overlay(target_directory)
        if overlay_dir:
            build_env["ASCEND_OPP_PATH"] = overlay_dir
        result = subprocess.run(["./build.sh"], check=True, capture_output=True, text=True, env=build_env)
        print("[INFO] Build succeeded")
    except subprocess.CalledProcessError as e:
        print("[INFO] Build failed!")
        error_output = ''
        for line in e.stdout.split('\n'):
            if '[ERROR]' in line or 'error:' in line:
                # print(line)
                error_output += line
                error_output += '\n'
        for line in e.stderr.split('\n'):
            if '[ERROR]' in line or 'error:' in line:
                # print(line)
                error_output += line
                error_output += '\n'
        feedback = f'Exit Code: {e.returncode}\nError Output:\n{error_output}'
        raise Exception(feedback)

    try:
        print("[INFO] Begin deploy")
        os.chdir(os.path.join(target_directory, op_capital_identifier, 'build_out'))
        # result = subprocess.run([f"./custom_opp_{system_name}_aarch64.run"], check=True, capture_output=True, text=True)
        result = subprocess.run([f"./custom_opp_{system_name}_aarch64.run", f'--install-path={deploy_path}'], check=True, capture_output=True, text=True)
        print("[INFO] Deploy succeeded")
    except subprocess.CalledProcessError as e:
        print("[INFO] Deploy failed!")
        feedback = f'Exit Code: {e.returncode}\nError Output:\n{e.stdout}'
        raise Exception(feedback)

    try:
        print("[INFO] Begin pybind")
        os.chdir(cpp_extension_dst)
        with open(os.path.join(cpp_extension_dst, 'setup.py'), 'r') as f:
            setup_content = f.read()
        setup_content = re.sub(r'custom_ops', op_identifier, setup_content)
        with open(os.path.join(cpp_extension_dst, 'setup.py'), 'w') as f:
            f.write(setup_content)
        result = subprocess.run(['bash', "build_and_run.sh"], check=True, capture_output=True, text=True)
        print("[INFO] Pybind succeeded\n")
    except subprocess.CalledProcessError as e:
        # Print error if build.sh fails
        print("[INFO] Pybind failed!")
        feedback = f'Exit Code: {e.returncode}\nError Output:\n{e.stdout}'
        raise Exception(feedback)

    runtime_opp_path = overlay_dir if overlay_dir else "/usr/local/Ascend/ascend-toolkit/latest/opp"

    # Update ASCEND_CUSTOM_OPP_PATH
    custom_opp_path = f"{target_directory}/opp/vendors/customize"
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = custom_opp_path
    os.environ["ASCEND_OPP_PATH"] = runtime_opp_path

    # Update LD_LIBRARY_PATH
    custom_lib_path = f"{target_directory}/opp/vendors/customize/op_api/lib/"
    existing_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    # 保存原始 LD_LIBRARY_PATH（如果还没有保存）
    if "LD_LIBRARY_PATH_ORIGINAL" not in os.environ:
        os.environ["LD_LIBRARY_PATH_ORIGINAL"] = existing_ld_path
    if custom_lib_path not in existing_ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{custom_lib_path}:{existing_ld_path}"
    
    # 使用re库，修改context['model_src']的所有custom_ops为op_identifier
    if 'model_src' in context:
        context['model_src'] = re.sub(r'custom_ops', op_identifier, context['model_src'])
    try:
        compile(context['model_src'], "<string>", "exec")
        exec(context['model_src'], context)  # For Python, use exec() (be careful with untrusted code)
    except Exception as e:
        raise Exception(f'Error in generated code: {e}')

    os.chdir(project_root_path)
    return {
        'op': op,
        'op_capital': op_capital,
        'op_identifier': op_identifier,
        'op_capital_identifier': op_capital_identifier,
        'model_src': context.get('model_src'),
        'custom_opp_path': custom_opp_path,
        'opp_path': runtime_opp_path,
        'lib_path': f"{custom_lib_path}:{existing_ld_path}",
    }



if __name__ == '__main__':
    import torch
    import torch_npu
    import custom_ops_lib
    op = 'relu'
    generated_method = getattr(custom_ops_lib, op)
