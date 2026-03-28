import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

import fitz  # PyMuPDF
import requests
from playwright.sync_api import sync_playwright
from slugify import slugify

DOWNLOAD_PAGE = "https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/index/download"
TARGET_TITLE = "Ascend C算子开发"
STATE_PATH = Path("state.json")
DOWNLOAD_DIR = Path("downloads")
OUT_DIR = Path("ascend_dev_guide_sections")

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEC_NO_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.*)$")
STEP_RE = re.compile(r"^步骤\s*([0-9]+)\s*(.*)$")
ORDERED_RE = re.compile(r"^([0-9]+)[\.\)、)]\s*(.+)$")
LETTER_RE = re.compile(r"^([a-zA-Z])[\.\)]\s*(.+)$")
BULLET_RE = re.compile(r"^[\-\*\+•●▪◦–—]+\s*(.+)$")
CODE_HEAD_RE = re.compile(
    r"^(#\s*include|#\s*define|#\s*if|#\s*endif|extern\b|template\b|typedef\b|using\b|class\b|struct\b|enum\b|namespace\b|if\s*\(|for\s*\(|while\s*\(|switch\s*\(|return\b|auto\b|const\b|static\b|void\b|inline\b|constexpr\b|public\s*:|private\s*:|protected\s*:|__global__\b|__aicore__\b|__aicpu__\b)"
)
SHELL_RE = re.compile(r"^(?:\$|cmake\b|python3?\b|bash\b|sh\b|make\b|pip\b|export\b|source\b|msopgen\b|bisheng(?:\s|-))")
FIGURE_RE = re.compile(r"^(图|表)\s*\d")
CJK_SPACE_RE = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])")
CPP_QUALIFIER_RE = re.compile(r"^(?:extern|inline|static|constexpr|virtual|friend|__global__|__aicore__|__aicpu__)$")
CPP_TYPE_HINT_RE = re.compile(
    r"^(?:void|bool|char|short|int|long|float|double|size_t|int\d+_t|uint\d+_t|acl\w*|AscendC::\w*|Kernel\w*)$"
)
CODE_FILE_RE = re.compile(r"(?:[A-Za-z0-9_./${}-]+)?\.(?:c|cc|cpp|cxx|h|hpp|asc|aicpu|o|so|a)(?![A-Za-z0-9_])")
CLI_OPTION_RE = re.compile(r"(?:^|\s)--?[A-Za-z][A-Za-z0-9_-]*")
CLI_FLAG_LINE_RE = re.compile(r"^--?[A-Za-z][A-Za-z0-9_]*(?:[=:\-][A-Za-z0-9_./${}-]+)?$")
CODE_WORD_SEQ_RE = re.compile(r"^[A-Za-z0-9_./${}<>\-+=:]+(?:\s+[A-Za-z0-9_./${}<>\-+=:]+)*$")
CODE_STRONG_TOKEN_RE = re.compile(
    r"\b(?:#include|#define|class|struct|template|constexpr|return|if|for|while|switch|"
    r"int\d*_t|uint\d*_t|int|float|double|void|char|bool|size_t)\b"
)
OPTION_TABLE_ROW_RE = re.compile(r"^(?P<option>.+?)\s+(?P<required>是|否)\s+(?P<desc>.+)$")
TABLE_TITLE_START_RE = re.compile(r"^表\d+-\d+\s+\S")
TABLE_TITLE_INLINE_RE = re.compile(r"表\d+-\d+\s+\S")
TABLE_ROW_TRAIL_NUM_RE = re.compile(r"^(?P<left>.+?)\s+(?P<num>\d{3,5})(?P<tail>[A-Za-z\u4e00-\u9fff]*)$")
CMAKE_VAR_HEAD_RE = re.compile(r"^(CMAKE_[A-Z0-9_]+)\s+(.*)$")
LIB_HEAD_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_.+-]*)\s+(.*)$")
PARAM_DESC_LABEL_RE = re.compile(r"(描述|含义|功能|说明)")
PARAM_TABLE_HEADER_KEY_RE = re.compile(r"^(参数名|参数名称)\b")
PARAM_ROW_IO_RE = re.compile(r"^(输入/输出|输入输出|输入|输出|入参|出参|输|入|出|入/输出|输/入|无)$")
API_TABLE_ROW_START_RE = re.compile(r"^(?:基础API|Utils API|高阶API)\s*>")
API_TABLE_HEADER_NOISE_RE = re.compile(r"接口分类接口名称(?:备注)?")
API_NAME_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:/[A-Za-z_][A-Za-z0-9_]*)?")
API_CLASS_TAIL_WORDS = [
    "基础算术",
    "逻辑计算",
    "复合计算",
    "比较与选择",
    "类型转换",
    "归约计算",
    "数据转换",
    "数据填充",
    "排序组合",
    "离散与聚合",
    "掩码操作",
    "量化设置",
    "基础数据搬运",
    "增强数据搬运",
    "切片数据搬运",
    "核内同步",
    "核间同步",
    "缓存控制",
    "系统变量访问",
    "系统变量访 问",
    "算法",
    "容器函数",
    "类型特性",
    "type_traits",
]
CMAKE_CONTINUATION_WORDS = {
    "PROPERTIES",
    "LANGUAGE",
    "PRIVATE",
    "PUBLIC",
    "STATIC",
    "SHARED",
    "ASC",
    "AICPU",
    "REQUIRED",
}


def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text("utf-8"))
    return {}


def save_state(state: dict):
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), "utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def get_target_pdf_url_via_playwright(target_title: str = TARGET_TITLE) -> dict:
    """
    直接抓页面里所有 .pdf 链接，并筛选出 href 中包含 target_title 的那个。
    兼容 URL 编码、空格差异。
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"]
        )
        context = browser.new_context(
            locale="zh-CN",
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()

        page.goto(DOWNLOAD_PAGE, wait_until="domcontentloaded", timeout=120000)

        # 关键：不等 ul.list，直接等 pdf 链接出现（你截图里 href 直接是 .pdf）
        page.wait_for_selector("a[href$='.pdf']", timeout=120000)

        # 拿到所有 pdf href
        hrefs = page.locator("a[href$='.pdf']").evaluate_all(
            "els => els.map(e => e.getAttribute('href')).filter(Boolean)"
        )

        # 做一个归一化：解码 + 去掉多余空格，方便匹配
        def norm(s: str) -> str:
            return unquote(s).replace("\u00a0", " ").replace(" ", "").strip()

        target_key = norm(target_title)

        # 先用 href 内容匹配（最稳）
        chosen = None
        for h in hrefs:
            if target_key in norm(h):
                chosen = h
                break

        # 兜底：如果 target_title 没出现在 href（少数情况），就按“opdevg + ascendc”特征筛
        if not chosen:
            for h in hrefs:
                nh = norm(h)
                if "opdevg" in nh.lower() and ("ascendc" in nh.lower() or "ascendc" in nh.lower()):
                    chosen = h
                    break

        browser.close()

        if not chosen:
            raise RuntimeError(f"未找到目标 PDF 链接。候选 pdf 链接数量={len(hrefs)}，示例={hrefs[:5]}")

        return {"pdf_url": chosen, "label": target_title}

def download_pdf(pdf_url: str, out_path: Path):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Referer": DOWNLOAD_PAGE,
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    r = requests.get(pdf_url, headers=headers, stream=True, timeout=60, allow_redirects=True)
    r.raise_for_status()
    with out_path.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def squeeze_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def sanitize_title(title: str) -> str:
    title = squeeze_ws(title)
    # 文件名不宜太长
    if len(title) > 80:
        title = title[:80].rstrip()
    return title


def safe_filename_cn(title: str, max_len: int = 80) -> str:
    """
    保留中文/英文/数字，清理 macOS/Linux/Windows 都不允许的文件名字符。
    """
    t = sanitize_title(title)

    # 替换 Windows/macOS/Linux 不适合的字符
    t = re.sub(r'[\\/:*?"<>|]', "_", t)

    # 去掉控制字符
    t = re.sub(r"[\x00-\x1f\x7f]", "", t)

    # 末尾的点和空格在 Windows 下有坑
    t = t.strip(" .")

    # 太长就截断
    if len(t) > max_len:
        t = t[:max_len].rstrip()

    # 兜底
    return t or "untitled"


def split_sec_no(title: str) -> tuple[str, str]:
    m = SEC_NO_RE.match(sanitize_title(title))
    if not m:
        t = sanitize_title(title)
        return "", t
    sec_no = m.group(1)
    name = sanitize_title(m.group(2)) or sanitize_title(title)
    return sec_no, name


def strip_sec_no(title: str) -> str:
    return split_sec_no(title)[1]


def norm_for_match(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", "", s)
    s = s.replace(".", "")
    return s.lower()


def find_title_line_index(lines: list[str], title: str) -> int | None:
    """
    在 lines 里找最像标题的行索引。优先精确包含（归一化后）。
    """
    key_full = norm_for_match(sanitize_title(title))
    key_plain = norm_for_match(strip_sec_no(title))

    for i, ln in enumerate(lines):
        nln = norm_for_match(ln)
        if key_full and key_full in nln:
            return i
    for i, ln in enumerate(lines):
        nln = norm_for_match(ln)
        if key_plain and key_plain in nln:
            return i
    return None


def normalize_prose_line(line: str) -> str:
    s = squeeze_ws(line)
    s = CJK_SPACE_RE.sub("", s)
    s = re.sub(r"(\d)\.\s+(\d)", r"\1.\2", s)
    s = re.sub(r"\s+([，。；：！？、）】》])", r"\1", s)
    s = re.sub(r"([（【《])\s+", r"\1", s)
    # OCR 噪声：范围文本中的 "Num AI处理器最大核数" -> "AI处理器最大核数"
    s = re.sub(
        r"\[\s*1\s*,\s*[Nn]um\s*AI处理器最大核数\s*\]",
        "[1, AI处理器最大核数]",
        s,
    )
    s = re.sub(r"\b[Nn]um\s*AI处理器最大核数\b", "AI处理器最大核数", s)
    # 避免 __inline__/__gm__ 这类标识被 markdown 解析为强调
    s = re.sub(r"(?<!`)__(?P<id>[A-Za-z][A-Za-z0-9_]*)__", r"`__\g<id>__`", s)
    # 避免正文里的 "#" 被 markdown 解释成标题
    if s.startswith("#"):
        s = "\\" + s
    return s


def normalize_option_cell(text: str) -> str:
    t = squeeze_ws(text)
    if not t:
        return t
    if t.startswith("-"):
        return t
    if t.startswith("cce-aicpu-"):
        return f"--{t}"
    if "，--" in t:
        return f"-{t}"
    if t in {"help", "x", "o", "c", "shared", "lib", "g", "fPIC", "O"}:
        return f"-{t}"
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*(?:\s+<[^>]+>)?", t):
        return f"-{t}"
    return t


def escape_table_cell(text: str) -> str:
    return text.replace("|", r"\|")


def is_block_boundary(line: str) -> bool:
    t = line.strip()
    return t.startswith("## ") or t.startswith("### ") or bool(TABLE_TITLE_START_RE.match(t))


def split_embedded_table_titles(lines: list[str]) -> list[str]:
    out: list[str] = []
    in_code = False
    for line in lines:
        if line.startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue
        m = TABLE_TITLE_INLINE_RE.search(line)
        if not m or m.start() == 0:
            out.append(line)
            continue
        prev = line[m.start() - 1]
        if prev not in "。；:：）】》":
            out.append(line)
            continue
        head = line[:m.start()].rstrip()
        tail = line[m.start():].strip()
        if head:
            out.append(head)
        if tail:
            out.append(tail)
    return out


def repair_cmake_var_and_desc(var_name: str, desc: str) -> tuple[str, str]:
    name = var_name
    d = squeeze_ws(desc)
    if name.startswith("CMAKE_BUILD_T"):
        name = "CMAKE_BUILD_TYPE"
        d = d.replace("：YPE", "：")
    elif name.startswith("CMAKE_INSTALL_"):
        name = "CMAKE_INSTALL_PREFIX"
    elif name.startswith("CMAKE_CXX_CO"):
        name = "CMAKE_CXX_COMPILER_LAUNCHER"
        d = d.replace("程序MPILER_LAUNCH 为", "程序为")
        d = d.replace("并ER 提高", "并提高")
    d = d.replace("●", "<br>- ")
    d = d.replace("：<br>-", "：<br>-")
    return name, d


def repair_lib_name(name: str, desc: str) -> tuple[str, str]:
    n = name.strip()
    d = squeeze_ws(desc)
    special = {
        "libascendc_runti": "libascendc_runtime.a",
        "libascend_dump.s": "libascend_dump.so",
        "liberror_manager.": "liberror_manager.so",
    }
    if n in special:
        fixed = special[n]
        m = re.search(r"。([A-Za-z0-9_.+-]+)$", d)
        if m:
            d = d[: m.start() + 1].strip()
        return fixed, d

    m = re.search(r"。([A-Za-z0-9_.+-]+)$", d)
    if m and not re.search(r"\.(?:so|a)$", n):
        suffix = m.group(1)
        cand = n + suffix
        if re.search(r"\.(?:so|a)$", cand):
            n = cand
            d = d[: m.start() + 1].strip()
    return n, d


def append_desc_cell(base: str, extra: str) -> str:
    e = squeeze_ws(extra)
    if not e:
        return base
    return f"{base}<br>{e}" if base else e


def join_prose(prev: str, cur: str) -> str:
    if not prev:
        return cur
    if re.search(r"[A-Za-z0-9_]$", prev) and re.match(r"^[A-Za-z0-9_]", cur):
        return f"{prev} {cur}"
    return f"{prev}{cur}"


def should_start_new_paragraph(prev_text: str, cur_text: str, prev_indent: int, cur_indent: int) -> bool:
    if prev_text.endswith(("。", "！", "？", "；")) and cur_indent >= prev_indent + 4:
        return True
    if prev_text.endswith(("。", "！", "？")) and re.match(r"^(说明|注意|提示|例如|如下|常见|执行配置|步骤)", cur_text):
        return True
    return False


def normalize_list_item(line: str) -> str | None:
    t = squeeze_ws(line)
    if looks_like_cli_option_line(t):
        return None
    m = STEP_RE.match(t)
    if m:
        num, rest = m.group(1), squeeze_ws(m.group(2))
        rest = re.sub(r"(?<!`)__(?P<id>[A-Za-z][A-Za-z0-9_]*)__", r"`__\g<id>__`", rest)
        return f"{num}. 步骤{num} {rest}".rstrip()
    m = ORDERED_RE.match(t)
    if m:
        num, rest = m.group(1), squeeze_ws(m.group(2))
        rest = re.sub(r"(?<!`)__(?P<id>[A-Za-z][A-Za-z0-9_]*)__", r"`__\g<id>__`", rest)
        if re.match(r"^[，。；：:!?！？]", rest):
            return None
        if re.match(r"^\d+(?:\.\d+)+", rest):
            return None
        return f"{num}. {rest}".rstrip()
    m = BULLET_RE.match(t)
    if m:
        body = squeeze_ws(m.group(1))
        body = re.sub(r"(?<!`)__(?P<id>[A-Za-z][A-Za-z0-9_]*)__", r"`__\g<id>__`", body)
        return f"- {body}"
    m = LETTER_RE.match(t)
    if m:
        body = squeeze_ws(m.group(2))
        body = re.sub(r"(?<!`)__(?P<id>[A-Za-z][A-Za-z0-9_]*)__", r"`__\g<id>__`", body)
        return f"- {m.group(1)}. {body}"
    return None


def looks_like_cli_option_line(line: str) -> bool:
    t = squeeze_ws(line)
    if not t:
        return False
    compact = t
    if re.match(r"^--?\s+\S+$", t):
        compact = re.sub(r"^(-{1,2})\s+", r"\1", t)
    if " " in compact:
        return False
    return bool(CLI_FLAG_LINE_RE.match(compact))


def looks_like_code_continuation_line(line: str) -> bool:
    t = squeeze_ws(line)
    if not t:
        return False
    if re.search(r"[\u4e00-\u9fff]", t):
        return False
    if len(t) > 180:
        return False
    if not CODE_WORD_SEQ_RE.fullmatch(t):
        return False
    if any(sym in t for sym in ("_", ".", "/", "$", "=", "<", ">")):
        return True
    tokens = t.split()
    if any(tok in CMAKE_CONTINUATION_WORDS for tok in tokens):
        return True
    return t.isupper()


def is_code_like(line: str, prev_is_code: bool) -> bool:
    t = line.strip()
    if not t:
        return False
    # OCR 常见注释残片：位于代码上下文时必须归入代码块
    if prev_is_code and t in {"/", "*", "/*", "*/", "//"}:
        return True
    if normalize_list_item(t) is not None:
        return False
    if FIGURE_RE.match(t):
        return False
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", t))
    if t.startswith("${") and has_cjk:
        return False
    if prev_is_code and looks_like_cli_option_line(t):
        return True
    if is_hash_comment_code_line(t, prev_is_code):
        return True
    if CODE_HEAD_RE.search(t):
        return True
    if SHELL_RE.search(t):
        # 形如“cmake，如果版本不符合要求...”属于正文说明，不应误判为代码
        if has_cjk and not re.search(r"(?:\$\s*|--?[A-Za-z]|[|`]|/[A-Za-z0-9_.-])", t):
            return False
        return True
    if t.startswith("//") or t.startswith("/*") or t.endswith("*/"):
        return True
    if re.fullmatch(r"(public|private|protected)\s*:", t):
        return True
    if looks_like_cpp_signature_line(t):
        return True
    if re.match(r"^[A-Za-z_][A-Za-z0-9_:<>]*\s*\(.*\)\s*;?\s*(//.*)?$", t):
        return True
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*\(.*\)\s*(//.*)?$", t):
        return True
    if re.match(r"^[A-Za-z_][A-Za-z0-9_:<>\s*&]+\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*[^;]+;\s*(//.*)?$", t):
        return True
    if not has_cjk and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*\(.*$", t):
        return True
    if has_cjk and ";" in t and re.search(r"[A-Za-z_][A-Za-z0-9_:.>]*\s*\(", t):
        return True
    if any(sym in t for sym in ("├──", "└──", "│")):
        return True
    if any(sym in t for sym in ("{", "}", ";", "::", "->", "<<<", ">>>", "#include", "#define")) and not has_cjk:
        return True
    if re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)", t) and re.search(r"[=<>]", t) and not has_cjk:
        return True
    if prev_is_code and looks_like_code_continuation_line(t):
        return True
    if prev_is_code and looks_like_mixed_token_cjk_continuation(t):
        return True
    if prev_is_code and CODE_FILE_RE.search(t):
        return True
    if prev_is_code and not has_cjk and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", t):
        return True
    if prev_is_code and re.fullmatch(r"[.·…]{3,}", t):
        return True
    if prev_is_code and (not has_cjk) and re.search(r"[(){}\[\];,<>#=]", t):
        return True
    return False


def looks_like_code_bridge_line(line: str, prev_code_line: str, next_nonempty_line: str | None) -> bool:
    """
    代码块桥接行判定：
    若当前行位于两段代码之间，且与代码风格连续（注释、缩进、符号、关键字等），
    优先保留在代码块内，避免代码块被意外断开。
    """
    t = squeeze_ws(line)
    if not t:
        return False
    if normalize_list_item(t) is not None:
        return False
    if looks_like_heading(t) or TABLE_TITLE_START_RE.match(t):
        return False

    prev = squeeze_ws(prev_code_line)

    # 明确代码符号/关键字/类型
    if re.search(r"[;{}()\[\]]", t):
        return True
    if CODE_STRONG_TOKEN_RE.search(t):
        return True
    if t.startswith("//") or t.startswith("/*") or t.endswith("*/"):
        return True

    # 上一行是代码注释或代码行，当前行像注释续行
    if prev:
        if should_attach_code_comment_continuation(t, prev):
            return True
        if should_attach_hash_comment_continuation(t, prev):
            return True
        if should_attach_inline_comment_continuation(t, prev):
            return True
        if prev.endswith(("-", ",", "(", "{", "->", "::")):
            return True

    # “夹在前后代码之间”时，允许弱代码行继续归入代码块
    if next_nonempty_line and is_code_like(next_nonempty_line, prev_is_code=True):
        if re.search(r"[A-Za-z_][A-Za-z0-9_:.<>-]*", t):
            return True
        # 常见自然语言短句，但处在代码注释上下文中
        if prev.startswith("//") and len(t) <= 80:
            return True
    return False


def should_break_code_block_for_prose(line: str) -> bool:
    t = squeeze_ws(line)
    if not t:
        return False
    if t.startswith(("//", "/*", "*", "#")):
        return False
    if re.search(r"[{}\[\];=:#`]", t):
        return False
    if re.search(r"[A-Za-z_][A-Za-z0-9_:.<>-]*\s*\(", t):
        return False
    if not re.search(r"[\u4e00-\u9fff]", t):
        return False
    if t in {"说明", "示例", "例如"}:
        return True
    if looks_like_plain_paragraph(t):
        return True
    if len(t) >= 8 and t.startswith(("以下", "其中", "说明", "注意", "开发者", "该", "本节", "多数情况下", "为了")):
        return True
    return False


def is_hash_comment_code_line(line: str, prev_is_code: bool) -> bool:
    t = line.strip()
    if not t.startswith("#"):
        return False
    body = squeeze_ws(t.lstrip("#"))
    if not body:
        return False
    if re.search(r"\b(?:bisheng|cmake|make|g\+\+|gcc|python3?|bash|sh)\b", body):
        return True
    if body.startswith("-"):
        return True
    if CODE_FILE_RE.search(body):
        return True
    if re.search(r"\.(?:c|cc|cpp|cxx|h|hpp|asc|aicpu|so|a)(?![A-Za-z0-9_])", body):
        return True
    if CLI_OPTION_RE.search(body):
        return True
    if prev_is_code:
        return True
    return False


def looks_like_cpp_signature_line(line: str) -> bool:
    t = squeeze_ws(line)
    if not t:
        return False
    if normalize_list_item(t) is not None:
        return False
    if re.search(r"[\u4e00-\u9fff]", t):
        return False
    if "(" not in t or ")" not in t:
        return False

    prefix = t.split("(", 1)[0].strip()
    if not prefix:
        return False

    raw_tokens = [tok for tok in re.split(r"\s+", prefix) if tok]
    tokens = [tok.strip("*&") for tok in raw_tokens if tok.strip("*&")]
    if len(tokens) < 2:
        return False
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tokens[-1]):
        return False

    qualifier_or_type = False
    for tok in tokens[:-1]:
        if CPP_QUALIFIER_RE.match(tok) or CPP_TYPE_HINT_RE.match(tok):
            qualifier_or_type = True
            break

    return qualifier_or_type or len(tokens) >= 3


def looks_like_mixed_token_cjk_continuation(line: str) -> bool:
    t = squeeze_ws(line)
    if not t:
        return False
    if normalize_list_item(t) is not None:
        return False
    if t.startswith(("●", "-", "步骤", "表", "图")):
        return False
    if not re.match(r"^[A-Za-z0-9_]+[\u4e00-\u9fff]", t):
        return False
    if re.search(r"[{};#]", t):
        return False
    if re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", t):
        return False
    if len(t) > 100:
        return False
    return True


def normalize_heading_text(line: str) -> str:
    t = squeeze_ws(line)
    m = SEC_NO_RE.match(t)
    if m:
        t = squeeze_ws(m.group(2))
    t = t.rstrip("：:").strip()
    return t


def looks_like_heading(line: str) -> bool:
    t = squeeze_ws(line)
    if not t:
        return False
    if t.startswith("#"):
        return False
    if re.fullmatch(r"[.·…]{3,}", t):
        return False
    if normalize_list_item(t) is not None:
        return False
    if is_code_like(t, prev_is_code=False):
        return False
    if FIGURE_RE.match(t):
        return False
    if re.search(r"[，。；：！？?,!]", t):
        return False
    if SEC_NO_RE.match(t):
        return True
    if t in {"说明", "实现流程", "约束条件", "完整样例", "常见问题"}:
        return True
    if len(t) <= 14:
        return True
    if len(t) <= 28 and re.search(r"(实现|流程|定义|调用|参数|设置|概述|介绍|场景|规则|模板|说明|步骤|格式|获取|输出|选项|变量|链接库)", t):
        return True
    return False


def is_strong_heading(line: str) -> bool:
    t = squeeze_ws(line)
    if not t or re.search(r"[，。；：！？?,!]", t):
        return False
    if t.startswith("#"):
        return False
    return len(t) <= 36 and bool(re.search(r"(实现|流程|定义|调用|参数|设置|概述|介绍|场景|规则|模板|说明|格式|获取|输出|选项|变量|链接库)", t))


def should_attach_code_comment_continuation(line: str, last_code_line: str) -> bool:
    curr = squeeze_ws(line)
    prev = last_code_line.strip()
    if not prev.startswith("//"):
        return False
    if not curr or normalize_list_item(curr) is not None:
        return False
    if re.search(r"[{};<>#]", curr):
        return False
    if re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", curr):
        return False
    if len(curr) > 60:
        return False
    return bool(re.search(r"[\u4e00-\u9fff]", curr))


def should_attach_hash_comment_continuation(line: str, last_code_line: str) -> bool:
    curr = squeeze_ws(line)
    prev = last_code_line.strip()
    if not prev.startswith("#"):
        return False
    if not curr or normalize_list_item(curr) is not None:
        return False
    if curr.startswith(("●", "-", "步骤")):
        return False
    if curr.startswith("#"):
        return False
    if len(curr) > 120:
        return False
    return bool(re.search(r"[\u4e00-\u9fff]", curr))


def should_attach_inline_comment_continuation(line: str, last_code_line: str) -> bool:
    curr = squeeze_ws(line)
    prev = last_code_line.strip()
    if "//" not in prev:
        return False
    if prev.startswith("//"):
        return False
    comment_part = prev.split("//", 1)[1].strip()
    if comment_part.endswith(("。", ".", "！", "!", "？", "?", "；", ";")):
        return False
    if not curr or normalize_list_item(curr) is not None:
        return False
    if curr.startswith(("●", "-", "步骤", "表", "图")):
        return False
    if not (
        re.match(r"^[\u4e00-\u9fff]", curr)
        or looks_like_mixed_token_cjk_continuation(curr)
    ):
        return False
    if re.search(r"[{};#]", curr):
        return False
    if re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", curr):
        return False
    if len(curr) > 80:
        return False
    return True


def normalize_code_lines(lines: list[str]) -> list[str]:
    cleaned = [ln.replace("\u00a0", " ").replace("\t", "    ").rstrip() for ln in lines]
    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    if not cleaned:
        return []

    indents = [len(ln) - len(ln.lstrip(" ")) for ln in cleaned if ln.strip()]
    cut = min(indents) if indents else 0
    return [ln[cut:] if len(ln) >= cut else ln for ln in cleaned]


def parse_param_table_header(line: str) -> tuple[int, str, str] | None:
    if not line.startswith("### "):
        return None
    head = squeeze_ws(line[4:])
    if not PARAM_TABLE_HEADER_KEY_RE.search(head):
        return None
    desc_matches = list(PARAM_DESC_LABEL_RE.finditer(head))
    if not desc_matches:
        return None
    desc_label = desc_matches[-1].group(1)
    first_col = "参数名称" if "参数名称" in head else "参数名"
    has_io = bool(re.search(r"(输入/输出|输入/输|输入/|入/|输出|输\b|输入\b|出\b)", head))
    return (3 if has_io else 2), first_col, desc_label


def is_param_table_header_fragment(line: str) -> bool:
    t = squeeze_ws(line.lstrip("#").strip())
    if t in {"入/", "出", "入", "输出", "输入", "输入/", "输入/输出", "入/输出", "输", "输/", "输/入", "出/", "说明"}:
        return True
    if line.startswith("### "):
        h = squeeze_ws(line[4:])
        if h in {"入/", "出", "入", "输出", "输入", "输入/", "输入/输出", "入/输出", "输", "输/", "输/入", "出/", "说明"}:
            return True
    return False


def looks_like_param_row_name(token: str) -> bool:
    t = squeeze_ws(token)
    if not t:
        return False
    if t.startswith(("-", "●", "•", "#")):
        return False
    if t.startswith(("针对", "类型为", "支持", "该参数", "此参数", "如果", "注意")):
        return False
    if t.startswith("Atlas"):
        return False
    if len(t) > 48:
        return False
    if t in {"输", "入", "出", "说明"}:
        return False
    if re.search(r"[，。；：！？,;:]", t):
        return False
    return True


def normalize_param_io(io_text: str) -> str:
    t = squeeze_ws(io_text)
    if t == "入参":
        return "输入"
    if t == "出参":
        return "输出"
    if t == "输入输出":
        return "输入/输出"
    if t in {"输", "入"}:
        return "输入"
    if t == "出":
        return "输出"
    if t in {"入/输出", "输/入"}:
        return "输入/输出"
    return t


def parse_param_row_2col(line: str) -> list[str] | None:
    t = squeeze_ws(line)
    m = re.match(r"^(?P<name>\S+)\s+(?P<desc>.+)$", t)
    if not m:
        return None
    name = m.group("name")
    if not looks_like_param_row_name(name):
        return None
    return [name, squeeze_ws(m.group("desc"))]


def parse_param_row_3col(line: str) -> list[str] | None:
    t = squeeze_ws(line)
    m = re.match(
        r"^(?P<name>.+?)\s+(?P<io>输入/输出|输入输出|输入|输出|入参|出参|输|入|出|入/输出|输/入|无)\s*(?P<desc>.*)$",
        t,
    )
    if not m:
        return None
    name = m.group("name")
    if not looks_like_param_row_name(name):
        return None
    io_val = normalize_param_io(m.group("io"))
    if not PARAM_ROW_IO_RE.match(io_val):
        return None
    return [name, io_val, squeeze_ws(m.group("desc"))]


def parse_param_row_3col_relaxed(line: str) -> list[str] | None:
    """
    宽松解析参数行，处理 OCR 造成的“输入/输出”列黏连与尾部残片。
    例如：srcShape 输输入的shape信息。入
    """
    t = squeeze_ws(line)
    m = re.match(r"^(?P<name>\S+)\s+(?P<body>.+)$", t)
    if not m:
        return None
    name = squeeze_ws(m.group("name"))
    body = squeeze_ws(m.group("body"))
    if not looks_like_param_row_name(name):
        return None

    io_tokens = ["输入/输出", "输入输出", "输入", "输出", "入参", "出参", "入/输出", "输/入", "输", "入", "出"]
    io_raw = None
    desc = body

    for tok in io_tokens:
        if body.startswith(tok):
            io_raw = tok
            desc = squeeze_ws(body[len(tok):])
            break

    if io_raw is None:
        m_tail = re.match(r"^(?P<desc>.+?)\s*(?P<io>输入/输出|输入输出|输入|输出|入参|出参|入/输出|输/入|输|入|出)$", body)
        if m_tail:
            io_raw = m_tail.group("io")
            desc = squeeze_ws(m_tail.group("desc"))

    if io_raw is None:
        return None

    io_val = normalize_param_io(io_raw)
    if not PARAM_ROW_IO_RE.match(io_val):
        return None

    desc = re.sub(r"(输入/输出|输入输出|输入|输出|入参|出参|入/输出|输/入|输|入|出)$", "", desc).strip()
    return [name, io_val, desc]


def looks_like_param_continuation_line(line: str, rows: list[list[str]]) -> bool:
    """
    表格续行判定：该行更像当前单元格补充说明时，禁止划到新列/新行。
    """
    t = squeeze_ws(line)
    if not t or not rows:
        return False
    # 明确像“新参数行”时，不能误判为续行（常见于分页后紧跟的新行）
    if parse_param_row_3col(t) is not None or parse_param_row_3col_relaxed(t) is not None:
        return False
    if parse_param_row_2col(t) is not None:
        return False
    if t in {"输", "入", "出", "说明", "入/输出", "输/入"}:
        return True
    if t.startswith(("输", "入", "出")) and len(t) <= 16:
        return True
    # OCR 拆词 + 输入输出碎片，例如: "rce 入"
    if re.fullmatch(r"[A-Za-z]{1,8}\s+(?:输|入|出|入/输出|输/入)", t):
        return True
    # 常见 OCR 拆词碎片：rce、ype 等
    if re.fullmatch(r"[A-Za-z]{1,4}", t):
        return True
    last_desc = squeeze_ws(rows[-1][-1]) if rows[-1] else ""
    if last_desc and not re.search(r"[。.!?；;]$", last_desc):
        # 上一个单元格语义未结束，本行默认并回上一单元格
        return True
    return False


def repair_param_row(row: list[str], ncol: int) -> list[str]:
    if ncol == 2:
        name, desc = row
    else:
        name, io_val, desc = row
    name = squeeze_ws(name)
    desc = squeeze_ws(desc)

    if name == "sharedTmpB":
        name = "sharedTmpBuffer"
    if name == "maxLiveNo" and "deCount" in desc:
        name = "maxLiveNodeCount"
        desc = desc.replace("deCount ", "")
    if name == "repeatTim" and "迭代次数" in desc:
        name = "repeatTimes"
    desc = desc.replace("该方式uffer", "该方式")
    if name == "enSequentia" and desc.endswith("lWrite"):
        name = "enSequentialWrite"
        desc = squeeze_ws(desc[: -len("lWrite")])

    if ncol == 2:
        return [name, desc]
    return [name, io_val, desc]


def parse_param_table_block(
    lines: list[str], start: int, header_info: tuple[int, str, str]
) -> tuple[list[str] | None, int]:
    ncol, first_col, desc_col = header_info
    rows: list[list[str]] = []
    k = start + 1

    while k < len(lines):
        cur = lines[k]
        t = cur.strip()

        if cur.startswith("## "):
            break
        if TABLE_TITLE_START_RE.match(t):
            break
        if cur.startswith("### "):
            if parse_param_table_header(cur) is not None:
                k += 1
                continue
            if is_param_table_header_fragment(cur):
                k += 1
                continue
            # 某些参数行会被 OCR 误识别成三级标题，需按行内容继续解析
            heading_payload = squeeze_ws(cur[4:])
            # 参数表中的“说明/备注/注意”常为单元格内部小标题，不应提前终止表格
            if rows and heading_payload in {"说明", "备注", "注意"}:
                rows[-1][-1] = append_desc_cell(rows[-1][-1], heading_payload)
                k += 1
                continue
            if ncol == 3:
                row = parse_param_row_3col(heading_payload) or parse_param_row_3col_relaxed(heading_payload)
            else:
                row = parse_param_row_2col(heading_payload)
            if row is not None and not looks_like_param_continuation_line(heading_payload, rows):
                rows.append(repair_param_row(row, ncol))
                k += 1
                continue
            if rows and looks_like_param_continuation_line(heading_payload, rows):
                rows[-1][-1] = append_desc_cell(rows[-1][-1], heading_payload)
                k += 1
                continue
            break
        if not t:
            k += 1
            continue
        if is_param_table_header_fragment(t):
            k += 1
            continue

        if cur.startswith("```"):
            code_lines: list[str] = []
            k += 1
            while k < len(lines) and not lines[k].startswith("```"):
                code_line = squeeze_ws(lines[k])
                if code_line:
                    code_lines.append(f"`{escape_table_cell(code_line)}`")
                k += 1
            if k < len(lines) and lines[k].startswith("```"):
                k += 1
            if rows and code_lines:
                rows[-1][-1] = append_desc_cell(rows[-1][-1], "<br>".join(code_lines))
            continue

        if ncol == 3:
            row = parse_param_row_3col(t) or parse_param_row_3col_relaxed(t)
        else:
            row = parse_param_row_2col(t)
        if row is not None and not looks_like_param_continuation_line(t, rows):
            rows.append(repair_param_row(row, ncol))
            k += 1
            continue

        if rows:
            rows[-1][-1] = append_desc_cell(rows[-1][-1], t)
        k += 1

    if not rows:
        return None, start + 1

    if ncol == 2:
        table_lines = [
            f"### {first_col} {desc_col}",
            "",
            f"| {first_col} | {desc_col} |",
            "| --- | --- |",
        ]
        for name, desc in rows:
            table_lines.append(
                f"| {escape_table_cell(name)} | {escape_table_cell(desc)} |"
            )
    else:
        table_lines = [
            f"### {first_col} 输入/输出 {desc_col}",
            "",
            f"| {first_col} | 输入/输出 | {desc_col} |",
            "| --- | --- | --- |",
        ]
        for name, io_val, desc in rows:
            table_lines.append(
                f"| {escape_table_cell(name)} | {escape_table_cell(io_val)} | {escape_table_cell(desc)} |"
            )
    table_lines.append("")
    return table_lines, k


def postprocess_param_table_blocks(lines: list[str]) -> list[str]:
    out: list[str] = []
    in_code = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            in_code = not in_code
            out.append(line)
            i += 1
            continue
        if in_code:
            out.append(line)
            i += 1
            continue

        header = parse_param_table_header(line)
        if header is not None:
            block, nxt = parse_param_table_block(lines, i, header)
            if block:
                out.extend(block)
                i = nxt
                continue

        out.append(line)
        i += 1
    return out


def parse_markdown_param_row_3col(line: str) -> list[str] | None:
    t = line.strip()
    if not (t.startswith("|") and t.endswith("|")):
        return None
    cells = [squeeze_ws(x) for x in t.strip("|").split("|")]
    if len(cells) < 3:
        return None
    if all(re.fullmatch(r":?-{3,}:?", c) for c in cells[:3]):
        return None
    name, io_val, desc = cells[0], normalize_param_io(cells[1]), cells[2]
    if not looks_like_param_row_name(name):
        return None
    if not PARAM_ROW_IO_RE.match(io_val):
        return None
    return [name, io_val, desc]


def postprocess_broken_param_markdown_tables(lines: list[str]) -> list[str]:
    """
    修复已被错误转换成 markdown 的参数表：
    - | 参数名 | 输入/ | 描述 |
    - 表格后“### 说明”与续行未并入单元格
    - 行如：minValue 输出 ...
    """
    out: list[str] = []
    i = 0
    in_code = False
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            in_code = not in_code
            out.append(line)
            i += 1
            continue
        if in_code:
            out.append(line)
            i += 1
            continue

        t = line.strip()
        is_bad_header = t in {"| 参数名 | 输入/ | 描述 |", "| 参数名 | 输入/输出 | 描述 |"}
        if not is_bad_header and t.startswith("| 参数") and ("输入/" in t) and ("描述" in t):
            is_bad_header = True
        if not is_bad_header:
            out.append(line)
            i += 1
            continue

        # 跳过分隔线
        j = i + 1
        if j < len(lines) and lines[j].strip().startswith("| ---"):
            j += 1

        rows: list[list[str]] = []
        # 先收现有 markdown 行
        while j < len(lines):
            row = parse_markdown_param_row_3col(lines[j])
            if row is None:
                break
            rows.append(repair_param_row(row, 3))
            j += 1

        # 再收表后残留的说明/参数行，直到明确边界
        k = j
        while k < len(lines):
            cur = lines[k]
            s = cur.strip()
            if not s:
                k += 1
                continue
            if cur.startswith("## "):
                break
            # 遇到下一张“表x-y ...”标题，必须断开，避免两张表被吞并
            if TABLE_TITLE_START_RE.match(s):
                break
            # 遇到新的参数表头，说明已进入下一张表
            if parse_param_table_header(cur) is not None or is_param_table_header_fragment(cur):
                break
            # 遇到 markdown 表头（参数表/API表/通用表）也应断开
            if re.fullmatch(r"\|\s*参数名\s*\|\s*输入/输出\s*\|\s*描述\s*\|", s):
                break
            if re.fullmatch(r"\|\s*参数名\s*\|\s*描述\s*\|", s):
                break
            if re.fullmatch(r"\|\s*[^|]+\s*\|\s*[^|]+\s*\|", s) and k + 1 < len(lines):
                nxt = lines[k + 1].strip()
                if nxt.startswith("| ---"):
                    break
            if cur.startswith("### ") and squeeze_ws(cur[4:]) in {"返回值说明", "约束说明", "调用示例", "函数原型", "功能说明", "参数说明"}:
                break
            if cur.startswith("### ") and squeeze_ws(cur[4:]) in {"说明", "备注", "注意"}:
                if rows:
                    rows[-1][-1] = append_desc_cell(rows[-1][-1], squeeze_ws(cur[4:]))
                k += 1
                continue

            cand = parse_param_row_3col(s) or parse_param_row_3col_relaxed(s)
            if cand is not None:
                rows.append(repair_param_row(cand, 3))
                k += 1
                continue

            if rows:
                rows[-1][-1] = append_desc_cell(rows[-1][-1], s)
            k += 1

        # 输出修复后的表
        out.append("| 参数名 | 输入/输出 | 描述 |")
        out.append("| --- | --- | --- |")
        for name, io_val, desc in rows:
            out.append(f"| {escape_table_cell(name)} | {escape_table_cell(io_val)} | {escape_table_cell(desc)} |")
        out.append("")
        i = k

    return out


def postprocess_split_embedded_table_titles(lines: list[str]) -> list[str]:
    """
    兜底修复：当下一张“表x-y ...参数说明”标题被 OCR 粘进上一张表的某个单元格时，
    将其从单元格拆出，恢复为独立标题行，避免两张表被合并。
    """
    out: list[str] = []
    in_code = False
    title_re = re.compile(
        r"(?P<before>.*?)(?P<title>表\d+-\d+\s*.*?(?:接口参数说明|结构体参数说明|模板参数说明|参数说明))\s*$"
    )

    for line in lines:
        if line.startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue

        s = line.strip()
        if not (s.startswith("|") and s.endswith("|")):
            out.append(line)
            continue

        cells = [x.strip() for x in s.strip("|").split("|")]
        split_idx = -1
        split_before = ""
        split_title = ""
        for idx, cell in enumerate(cells):
            m = title_re.search(cell)
            if m and m.group("before").strip():
                split_idx = idx
                split_before = squeeze_ws(m.group("before"))
                split_title = squeeze_ws(m.group("title"))
                break
        if split_idx < 0:
            out.append(line)
            continue

        cells[split_idx] = split_before
        rebuilt = "| " + " | ".join(escape_table_cell(c) for c in cells) + " |"
        out.append(rebuilt)
        out.append("")
        out.append(split_title)
        out.append("")
    return out


def postprocess_broken_api_markdown_tables(lines: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    in_code = False
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            in_code = not in_code
            out.append(line)
            i += 1
            continue
        if in_code:
            out.append(line)
            i += 1
            continue

        head = line.strip()
        if re.fullmatch(r"\|\s*接口分类\s*\|\s*接口名称\s*\|", head):
            block, nxt = parse_api_scope_markdown_table_block(lines, i, with_remark=False)
            if block:
                out.extend(block)
                i = nxt
                continue
        if re.fullmatch(r"\|\s*接口分类\s*\|\s*接口名称\s*\|\s*备注\s*\|", head):
            block, nxt = parse_api_scope_markdown_table_block(lines, i, with_remark=True)
            if block:
                out.extend(block)
                i = nxt
                continue

        out.append(line)
        i += 1
    return out


def postprocess_promote_paramname_desc_tables(lines: list[str]) -> list[str]:
    """
    修复分页导致的三列表降级：
    | 参数名称 | 说明 |  ->  | 参数名称 | 数据类型 | 说明 |
    当“说明”列稳定以数据类型开头时，自动提升为三列。
    """
    out: list[str] = []
    i = 0
    in_code = False
    dtype_re = re.compile(
        r"^(?:u?int(?:8|16|32|64)?_t|int|float|double|bool|half|bfloat16_t|size_t|void|char|string)\b",
        re.IGNORECASE,
    )

    def clean_pipe_artifact(s: str) -> str:
        t = squeeze_ws(s.replace(r"\|", " "))
        t = t.strip("|").strip()
        return squeeze_ws(t)

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            in_code = not in_code
            out.append(line)
            i += 1
            continue
        if in_code:
            out.append(line)
            i += 1
            continue

        head = line.strip()
        if head not in {"| 参数名称 | 说明 |", "| 参数名 | 说明 |"}:
            out.append(line)
            i += 1
            continue

        j = i + 1
        if j < len(lines) and lines[j].strip().startswith("| ---"):
            j += 1

        rows: list[list[str]] = []
        hit = 0
        while j < len(lines):
            t = lines[j].strip()
            if not (t.startswith("|") and t.endswith("|")):
                break
            cells = [clean_pipe_artifact(x) for x in t.strip("|").split("|")]
            if cells and all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                j += 1
                continue
            if len(cells) < 2:
                break
            name = clean_pipe_artifact(cells[0])
            desc_or_all = clean_pipe_artifact(" ".join(cells[1:]))
            dtype = ""
            desc = desc_or_all
            m = re.match(r"^(?P<dtype>\S+)\s+(?P<desc>.+)$", desc_or_all)
            if m and dtype_re.match(m.group("dtype")):
                dtype = m.group("dtype")
                desc = squeeze_ws(m.group("desc"))
                hit += 1
            rows.append([name, dtype, desc])
            j += 1

        if rows and hit >= 2:
            left = "参数名称" if "参数名称" in head else "参数名"
            out.append(f"| {left} | 数据类型 | 说明 |")
            out.append("| --- | --- | --- |")
            for n0, d0, s0 in rows:
                out.append(
                    f"| {escape_table_cell(n0)} | {escape_table_cell(d0)} | {escape_table_cell(s0)} |"
                )
            out.append("")
            i = j
            continue

        out.append(line)
        i += 1
    return out


def postprocess_split_embedded_param_rows(lines: list[str]) -> list[str]:
    """
    修复三列表中“说明”单元格误吞后续参数行的问题（常见于分页续表）：
    | 参数名称 | 数据类型 | 说明 |
    """
    out: list[str] = []
    i = 0
    in_code = False
    dtype_re = r"(?:u?int(?:8|16|32|64)?_t|int|float|double|bool|half|bfloat16_t|size_t|void|char|string)"
    emb_re = re.compile(
        rf"(?:^|<br>)(?P<name>[A-Za-z][A-Za-z0-9_ ,，]{{0,80}}?)\s*(?:,|，)?\s*(?P<dtype>{dtype_re})\s+"
    )

    def clean_cell(s: str) -> str:
        return squeeze_ws(s.replace(r"\|", " ").strip("|").strip())

    def repair_name_dtype_desc(name: str, dtype: str, desc: str) -> tuple[str, str, str]:
        n = squeeze_ws(name)
        d = squeeze_ws(dtype)
        s = squeeze_ws(desc)
        s = re.sub(
            r"\[\s*1\s*,\s*[Nn]um\s*AI处理器最大核数\s*\]",
            "[1, AI处理器最大核数]",
            s,
        )
        s = re.sub(r"\b[Nn]um\s*AI处理器最大核数\b", "AI处理器最大核数", s)

        # 分页断裂导致首列参数列表丢项
        if n in {"M, N, Ka", "M,N,Ka"} and re.search(r"\bKb\b", s):
            n = "M, N, Ka, Kb"
        if n == "baseM" and ("baseN" in s) and ("baseK" in s):
            n = "baseM, baseN, baseK"
        if n == "depthA1" and ("depthB1" in s):
            n = "depthA1, depthB1"
        if n == "stepM" and ("stepN" in s) and ("stepKa" in s) and ("stepKb" in s):
            n = "stepM, stepN, stepKa, stepKb"
        # 参数名尾段误入说明列（如 usedCore + Num）
        if n == "usedCore" and re.search(r"\busedCoreNum\b", s):
            n = "usedCoreNum"
        if n.startswith("singleCor") and ("singleCore" in s):
            n = "singleCoreM, singleCoreN, singleCoreK"
            s = s.replace("该参数取值必eM, 须大于0。", "该参数取值必须大于0。")
            s = s.replace("singleCor singleCoreK", "singleCoreK")
            s = s.replace("singleCoreM <=eN", "singleCoreM <= N")
            s = s.replace("singleCoreN <= N。 singleCor注意", "singleCoreN <= N。注意")
            s = s.replace("singleCor", "singleCore")
            s = s.replace("singleCoreeM", "singleCoreM")
            s = s.replace("singleCoreeN", "singleCoreN")
            s = s.replace("singleCoreeK", "singleCoreK")
            s = s.replace("singleCoreM <= N, M", "singleCoreM <= M")
            s = s.replace("singleCore注意", "注意")
            s = squeeze_ws(s)
        return n, d, s

    def split_embedded(desc: str) -> tuple[str, list[list[str]]]:
        d = squeeze_ws(desc).replace("<br>类型<br>", "<br>").replace("<br>类型", "<br>")
        ms = list(emb_re.finditer(d))
        if not ms:
            return desc, []
        # 必须是“中途出现”内嵌参数，避免误切正常开头
        if ms[0].start() == 0:
            return desc, []
        head = d[: ms[0].start()].strip()
        if not head:
            return desc, []
        extra_rows: list[list[str]] = []
        for idx, m in enumerate(ms):
            seg_start = m.end()
            seg_end = ms[idx + 1].start() if idx + 1 < len(ms) else len(d)
            seg_desc = squeeze_ws(d[seg_start:seg_end]).strip("<br>").strip()
            name = squeeze_ws(m.group("name")).strip(",，")
            dtype = m.group("dtype")
            if not name or not seg_desc:
                continue
            extra_rows.append([name, dtype, seg_desc])
        if not extra_rows:
            return desc, []
        return head, extra_rows

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            in_code = not in_code
            out.append(line)
            i += 1
            continue
        if in_code:
            out.append(line)
            i += 1
            continue

        head = line.strip()
        if head not in {"| 参数名称 | 数据类型 | 说明 |", "| 参数名 | 数据类型 | 说明 |"}:
            out.append(line)
            i += 1
            continue

        out.append(line)
        i += 1
        if i < len(lines) and lines[i].strip().startswith("| ---"):
            out.append(lines[i])
            i += 1

        while i < len(lines):
            t = lines[i].strip()
            if not (t.startswith("|") and t.endswith("|")):
                break
            cells = [clean_cell(x) for x in t.strip("|").split("|")]
            if cells and all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                out.append(lines[i])
                i += 1
                continue
            if len(cells) < 3:
                out.append(lines[i])
                i += 1
                continue
            name = cells[0]
            dtype = cells[1]
            desc = clean_cell(" ".join(cells[2:]))
            head_desc, extras = split_embedded(desc)
            name, dtype, head_desc = repair_name_dtype_desc(name, dtype, head_desc)
            out.append(
                f"| {escape_table_cell(name)} | {escape_table_cell(dtype)} | {escape_table_cell(head_desc)} |"
            )
            for n0, d0, s0 in extras:
                n0, d0, s0 = repair_name_dtype_desc(n0, d0, s0)
                out.append(
                    f"| {escape_table_cell(n0)} | {escape_table_cell(d0)} | {escape_table_cell(s0)} |"
                )
            i += 1

    return out


def postprocess_detach_prose_from_param_tables(lines: list[str]) -> list[str]:
    """
    修复参数表中误吸收正文的问题：
    - 把“多数情况下，用户通过调用...”等正文从单元格拆出为表后正文
    - 修复分页断裂参数名 singleBatc + hM/hN
    """
    out: list[str] = []
    i = 0
    in_code = False
    param_headers = {
        "| 参数名称 | 数据类型 | 说明 |",
        "| 参数名 | 数据类型 | 说明 |",
        "| 参数名 | 输入/输出 | 描述 |",
    }
    prose_markers = [
        "多数情况下，用户通过调用",
        "如果用户自定义",
        "如果用户通过调用",
        "请参考如下",
        "一组合法的",
    ]

    def clean_cell(s: str) -> str:
        return squeeze_ws(s.replace(r"\|", " ").strip("|").strip())

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            in_code = not in_code
            out.append(line)
            i += 1
            continue
        if in_code:
            out.append(line)
            i += 1
            continue

        head = line.strip()
        if head not in param_headers:
            out.append(line)
            i += 1
            continue

        out.append(line)
        i += 1
        if i < len(lines) and lines[i].strip().startswith("| ---"):
            out.append(lines[i])
            i += 1

        detached: list[str] = []
        while i < len(lines):
            t = lines[i].strip()
            if not (t.startswith("|") and t.endswith("|")):
                break
            cells = [clean_cell(x) for x in t.strip("|").split("|")]
            if cells and all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                out.append(lines[i])
                i += 1
                continue
            if len(cells) < 3:
                out.append(lines[i])
                i += 1
                continue

            name = cells[0]
            io_or_dtype = cells[1]
            desc = clean_cell(" ".join(cells[2:]))

            # 修复 singleBatc + hM/hN 断裂
            if name == "singleBatc":
                if re.search(r"hM(?:<br>)?$", desc):
                    name = "singleBatchM"
                    desc = re.sub(r"hM(?:<br>)?$", "", desc)
                    desc = squeeze_ws(desc).rstrip("，,")
                elif re.search(r"hN(?:<br>)?$", desc):
                    name = "singleBatchN"
                    desc = re.sub(r"hN(?:<br>)?$", "", desc)
                    desc = squeeze_ws(desc).rstrip("，,")
            # 修复“首列尾缀误并到第三列末尾”的分页错位：
            # shareMod + "...关注。 e" -> shareMode + "...关注。"
            compact_desc = squeeze_ws(desc.replace("<br>", ""))
            m_tail = re.match(r"^(?P<body>该参数预留，开发者无需关注。?)(?P<frag>[A-Za-z]{1,4})$", compact_desc)
            if m_tail and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{1,40}", name):
                name = name + m_tail.group("frag")
                desc = m_tail.group("body")
            if name == "batchN" and desc.endswith("<br>类型"):
                desc = desc[: -len("<br>类型")].strip()

            # 正文吸收剥离
            split_at = -1
            for mk in prose_markers:
                pos = desc.find(mk)
                if pos > 0:
                    split_at = pos
                    break
            if split_at > 0:
                keep = squeeze_ws(desc[:split_at]).rstrip("，,")
                spill = squeeze_ws(desc[split_at:])
                if keep:
                    desc = keep
                if spill:
                    detached.extend([x.strip() for x in spill.split("<br>") if x.strip()])

            out.append(
                f"| {escape_table_cell(name)} | {escape_table_cell(io_or_dtype)} | {escape_table_cell(desc)} |"
            )
            i += 1

        if detached:
            out.append("")
            out.extend(detached)
            out.append("")

    return out


def postprocess_rebalance_two_col_constraint_rows(lines: list[str]) -> list[str]:
    """
    修复两列表中同一行的跨列串位：
    典型模式：第二列里混入“db这里表示为...”，实际应归第一列。
    """
    out: list[str] = []
    i = 0
    in_code = False
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            in_code = not in_code
            out.append(line)
            i += 1
            continue
        if in_code:
            out.append(line)
            i += 1
            continue

        t = line.strip()
        if not (t.startswith("|") and t.endswith("|")):
            out.append(line)
            i += 1
            continue

        # 识别两列表头（通过分隔线）
        if i + 1 < len(lines):
            sep = lines[i + 1].strip()
            if sep.startswith("|") and sep.endswith("|"):
                sep_cells = [squeeze_ws(x) for x in sep.strip("|").split("|")]
                sep_cells = [x for x in sep_cells if x]
                if len(sep_cells) == 2 and all(re.fullmatch(r":?-{3,}:?", c) for c in sep_cells):
                    out.append(line)
                    out.append(lines[i + 1])
                    i += 2
                    while i < len(lines):
                        row = lines[i].strip()
                        if not (row.startswith("|") and row.endswith("|")):
                            break
                        cells = [squeeze_ws(x.replace(r"\|", " ")) for x in row.strip("|").split("|")]
                        if len(cells) < 2:
                            out.append(lines[i])
                            i += 1
                            continue
                        left = squeeze_ws(cells[0])
                        right = squeeze_ws(" ".join(cells[1:]))

                        # 去掉分页导致的重复表头行（避免变成数据行）
                        if left == "约束条件" and right == "说明":
                            i += 1
                            continue

                        if " db这里表示为" in right:
                            pre, post = right.split(" db这里表示为", 1)
                            pre = squeeze_ws(pre)
                            moved = "db这里表示为" + squeeze_ws(post)
                            # 如果右列前缀是“depthA1的取值与...”这类句子，
                            # 且左列末尾出现重复 token（如 depthA1 depthA1），去掉重复。
                            m = re.match(r"^(?P<tok>[A-Za-z_][A-Za-z0-9_]*)的取值与", pre)
                            if m:
                                tok = m.group("tok")
                                left = re.sub(rf"\b{re.escape(tok)}\s+{re.escape(tok)}$", tok, left)
                            else:
                                # 右列被截断成“的取值与...”时，仍尝试消除左列尾部重复 token
                                left = re.sub(
                                    r"\b(?P<tok>[A-Za-z_][A-Za-z0-9_]*)\s+(?P=tok)$",
                                    r"\g<tok>",
                                    left,
                                )
                            left = append_desc_cell(left, moved)
                            right = pre

                        # 修复首尾 token 互换：左列尾部误带 A/B/C/Bias，右列前缀缺失
                        m_swap = re.match(r"^(?P<prefix>.*\s)(?P<tag>A|B|C|Bias)$", left)
                        if m_swap and (
                            right.startswith(("矩阵", "的", "base块"))
                            or right.startswith(("矩阵base块", "的base块"))
                        ):
                            tag = m_swap.group("tag")
                            left = squeeze_ws(m_swap.group("prefix"))
                            # 防止重复添加
                            if not right.startswith(tag):
                                right = f"{tag}{right}"

                        # 修复“右列前缀变量丢失”：`的取值与...` -> `depthA1的取值与...`
                        if right.startswith("的") and ("取值与" in right):
                            m_var = re.search(r"=\s*(?P<v>[A-Za-z_][A-Za-z0-9_]*)", left)
                            if m_var:
                                v = m_var.group("v")
                                if not right.startswith(v):
                                    right = f"{v}{right}"

                        # 修复 l0*_size 与 “其中l0c_type...” 误入第二列：
                        # 左列应为：... < l0x_size，其中...
                        # 右列应为：X矩阵base块不超过l0x buffer大小
                        m_l0 = re.match(
                            r"^(?P<tag>[ABC])矩阵base块不超过(?P<l0>l0[abc])\s+(?P<mid>.+?)\s+buffer大小(?P<tail>.*)$",
                            right,
                        )
                        if m_l0 and "<" in left:
                            moved = squeeze_ws(m_l0.group("mid"))
                            if moved:
                                # 去掉左列末尾误残留的单字母标签（如 "< C"）
                                left = re.sub(r"\s<[ ]*[ABC]$", " <", left)
                                left = re.sub(r"\s+[ABC]$", "", left)
                                if left.rstrip().endswith("<"):
                                    left = left.rstrip() + " " + moved
                                else:
                                    left = append_desc_cell(left, moved)
                            right = f"{m_l0.group('tag')}矩阵base块不超过{m_l0.group('l0')} buffer大小"
                            tail = squeeze_ws(m_l0.group("tail"))
                            if tail:
                                if tail in {"类型。", "类型"}:
                                    left = append_desc_cell(left, tail)
                                else:
                                    right = append_desc_cell(right, tail)

                        # 修复“约束条件行内混杂”：第一列后续内容被第二列吸收
                        if "AL1Size" in left and ("BL1Size" in right or "L1 buffer大小限制" in right):
                            # 1) 把 "阵在L1上的缓存块大小BL1Size必须满足：" 回填左列
                            m_bl = re.search(r"阵在L1上的缓存块大小BL1Size必须满足：", right)
                            if m_bl:
                                moved = m_bl.group(0)
                                left = append_desc_cell(left, moved)
                                right = squeeze_ws(right[m_bl.end() :])
                            # 2) 把无/有bias公式段与“其中...计算方式如下”回填左列
                            split_pos = -1
                            for mk in ("●无bias场景", "- 无bias场景", "其中，AL1Size、BL1Size的计算方式如下："):
                                pos = right.find(mk)
                                if pos >= 0 and (split_pos < 0 or pos < split_pos):
                                    split_pos = pos
                            if split_pos >= 0:
                                moved = squeeze_ws(right[split_pos:])
                                right = squeeze_ws(right[:split_pos]).rstrip("；;,")
                                if moved:
                                    left = append_desc_cell(left, moved)

                        # 修复 baseN 对齐描述误入第二列
                        if left.rstrip().endswith("baseM *") and "baseN按照NZ格式的分形对齐" in right:
                            left = squeeze_ws(left + " baseN按照NZ格式的分形对齐")
                            right = right.replace("baseN按照NZ格式的分形对齐", "", 1)
                            right = right.replace("的base 块", "的base块")
                            right = squeeze_ws(right)

                        # 修复 MDL 补充约束行左右列互吸收（kaStepIter_/kbStepIter_）
                        if (
                            "kaStepIter_" in left
                            and "% kbStepIter_" in left
                            and "MDL模板K方向循环搬运要求" in right
                        ):
                            right_clean = squeeze_ws(right)
                            # 取出可能落入右列的两个公式定义
                            m_ka = re.search(r"kaStepIter_\s*=\s*CeilDiv\([^)]*\)", right_clean)
                            m_kb = re.search(r"kbStepIter_\s*=\s*CeilDiv\([^)]*\)", right_clean)

                            left_parts = [left]
                            if "% kaStepIter_ = 0" in right_clean:
                                left_parts.append("或者kbStepIter_ % kaStepIter_ = 0")
                            if m_ka:
                                left_parts.append(squeeze_ws(m_ka.group(0)))
                            if m_kb:
                                left_parts.append(squeeze_ws(m_kb.group(0)))
                            left = "<br>".join(dict.fromkeys([x for x in left_parts if x]))

                            right_parts = [
                                "MDL模板K方向循环搬运要求Ka和Kb方向迭代次数为倍数关系",
                                "kaStepIter_：Ka方向循环搬运迭代次数",
                                "kbStepIter_：Kb方向循环搬运迭代次数",
                            ]
                            right = "<br>".join(right_parts)

                        # 修复首列吸收次列：`K方向非全载时，M/N方向只能=1` 应在右列
                        # 典型：左列 "... stepM K方向非全载时，M方向只能= 1" 右列仅"逐块搬运"
                        m_absorb = re.search(r"(K方向非全载时，[MN]方向只能=\s*1)\s*$", left)
                        if m_absorb and right.replace("。", "") in {"逐块搬运"}:
                            moved = squeeze_ws(m_absorb.group(1))
                            left = squeeze_ws(left[: m_absorb.start()]).rstrip("，,")
                            # 回补左列被吞掉的 "= 1"
                            if re.search(r"stepM\s*$", left):
                                left = left + " = 1"
                            elif re.search(r"stepN\s*$", left):
                                left = left + " = 1"
                            moved = re.sub(r"=\s*1", "", moved)
                            right = squeeze_ws(f"{moved}{right}")

                        # 修复“本应两行两列却被并成一行”的约束条件场景
                        # 典型：左列同时出现“AL1Size/BL1Size约束”和
                        # “baseM * baseK, baseK * baseN...”两段内容。
                        if (
                            "baseM * baseK, baseK * baseN" in left
                            and "按照NZ格式的分形对齐" in left
                            and "AL1Size" in left
                        ):
                            p2 = left.find("baseM * baseK, baseK * baseN")
                            left1 = squeeze_ws(left[:p2]).rstrip("，,")
                            left2 = squeeze_ws(left[p2:])
                            right1 = right
                            right2 = ""

                            m_right1 = re.search(r"(A矩阵、B矩阵和Bias.+)$", left1)
                            if m_right1:
                                right1 = squeeze_ws(m_right1.group(1))
                                left1 = squeeze_ws(left1[: m_right1.start()]).rstrip("，,")

                            m_right2 = re.search(r"(A矩阵、B矩阵、C矩阵.+)$", left2)
                            if m_right2:
                                right2 = squeeze_ws(m_right2.group(1))
                                left2 = squeeze_ws(left2[: m_right2.start()]).rstrip("，,")

                            if left1 and right1:
                                out.append(f"| {escape_table_cell(left1)} | {escape_table_cell(right1)} |")
                            if left2 and right2:
                                out.append(f"| {escape_table_cell(left2)} | {escape_table_cell(right2)} |")
                            elif left2:
                                out.append(f"| {escape_table_cell(left2)} | {escape_table_cell(right)} |")
                            i += 1
                            continue

                        out.append(f"| {escape_table_cell(left)} | {escape_table_cell(right)} |")
                        i += 1
                    continue

        out.append(line)
        i += 1

    return out


def postprocess_recover_constraint_table_continuations(lines: list[str]) -> list[str]:
    """
    修复“约束条件”两列表分页续行掉出表格的问题：
    - 将 `| - | 转置场景： |` 以及其后续公式/列表并回前一行左单元格
    - 将“左右列粘在同一行”的文本尽量拆回两列新行
    """
    out: list[str] = []
    i = 0
    in_code = False

    def is_boundary(s: str) -> bool:
        t = s.strip()
        return (
            (not t)
            or t.startswith("## ")
            or t.startswith("### ")
            or bool(TABLE_TITLE_START_RE.match(t))
        )

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            in_code = not in_code
            out.append(line)
            i += 1
            continue
        if in_code:
            out.append(line)
            i += 1
            continue

        t = line.strip()
        if not TABLE_TITLE_START_RE.match(t) or ("约束条件" not in t):
            out.append(line)
            i += 1
            continue

        # 保留标题
        out.append(line)
        i += 1
        # 保留空行
        while i < len(lines) and not lines[i].strip():
            out.append(lines[i])
            i += 1
        # 需要紧随两列表头
        if i + 1 >= len(lines):
            continue
        h = lines[i].strip()
        sep = lines[i + 1].strip()
        if not (h.startswith("|") and h.endswith("|") and sep.startswith("|") and sep.endswith("|")):
            continue
        sep_cells = [x.strip() for x in sep.strip("|").split("|") if x.strip()]
        if not (len(sep_cells) == 2 and all(re.fullmatch(r":?-{3,}:?", c) for c in sep_cells)):
            continue

        out.append(lines[i])
        out.append(lines[i + 1])
        i += 2

        rows: list[list[str]] = []
        while i < len(lines):
            r = lines[i].strip()
            if not (r.startswith("|") and r.endswith("|")):
                break
            cells = [squeeze_ws(x.replace(r"\|", " ")) for x in r.strip("|").split("|")]
            if len(cells) >= 2 and not all(re.fullmatch(r":?-{3,}:?", c) for c in cells[:2]):
                left = squeeze_ws(cells[0])
                right = squeeze_ws(" ".join(cells[1:]))

                # 去掉分页导致的重复表头行（避免变成数据行）
                if left == "约束条件" and right == "说明":
                    i += 1
                    continue

                # 与前序重平衡一致：把误吸收到右列的左列后续内容回填
                if "AL1Size" in left and ("BL1Size" in right or "L1 buffer大小限制" in right):
                    m_bl = re.search(r"阵在L1上的缓存块大小BL1Size必须满足：", right)
                    if m_bl:
                        left = append_desc_cell(left, m_bl.group(0))
                        right = squeeze_ws(right[m_bl.end() :])
                    split_pos = -1
                    for mk in ("●无bias场景", "- 无bias场景", "其中，AL1Size、BL1Size的计算方式如下："):
                        pos = right.find(mk)
                        if pos >= 0 and (split_pos < 0 or pos < split_pos):
                            split_pos = pos
                    if split_pos >= 0:
                        moved = squeeze_ws(right[split_pos:])
                        right = squeeze_ws(right[:split_pos]).rstrip("；;,")
                        if moved:
                            left = append_desc_cell(left, moved)

                if left.rstrip().endswith("baseM *") and "baseN按照NZ格式的分形对齐" in right:
                    left = squeeze_ws(left + " baseN按照NZ格式的分形对齐")
                    right = right.replace("baseN按照NZ格式的分形对齐", "", 1)
                    right = right.replace("的base ", "的base")
                    right = squeeze_ws(right)

                # 修复 MDL 补充约束行左右列互吸收（kaStepIter_/kbStepIter_）
                if (
                    "kaStepIter_" in left
                    and "% kbStepIter_" in left
                    and "MDL模板K方向循环搬运要求" in right
                ):
                    right_clean = squeeze_ws(right)
                    m_ka = re.search(r"kaStepIter_\s*=\s*CeilDiv\([^)]*\)", right_clean)
                    m_kb = re.search(r"kbStepIter_\s*=\s*CeilDiv\([^)]*\)", right_clean)

                    left_parts = [left]
                    if "% kaStepIter_ = 0" in right_clean:
                        left_parts.append("或者kbStepIter_ % kaStepIter_ = 0")
                    if m_ka:
                        left_parts.append(squeeze_ws(m_ka.group(0)))
                    if m_kb:
                        left_parts.append(squeeze_ws(m_kb.group(0)))
                    left = "<br>".join(dict.fromkeys([x for x in left_parts if x]))

                    right = "<br>".join(
                        [
                            "MDL模板K方向循环搬运要求Ka和Kb方向迭代次数为倍数关系",
                            "kaStepIter_：Ka方向循环搬运迭代次数",
                            "kbStepIter_：Kb方向循环搬运迭代次数",
                        ]
                    )

                # 修复首列吸收次列：`K方向非全载时，M/N方向只能=1` 应在右列
                m_absorb = re.search(r"(K方向非全载时，[MN]方向只能=\s*1)\s*$", left)
                if m_absorb and right.replace("。", "") in {"逐块搬运"}:
                    moved = squeeze_ws(m_absorb.group(1))
                    left = squeeze_ws(left[: m_absorb.start()]).rstrip("，,")
                    if re.search(r"stepM\s*$", left):
                        left = left + " = 1"
                    elif re.search(r"stepN\s*$", left):
                        left = left + " = 1"
                    moved = re.sub(r"=\s*1", "", moved)
                    right = squeeze_ws(f"{moved}{right}")

                rows.append([left, right])
            i += 1

        # 处理 `| - | ... |` 续行标记
        k = i
        if rows and rows[-1][0] == "-" and len(rows) >= 2:
            marker = rows.pop()
            rows[-1][0] = append_desc_cell(rows[-1][0], marker[1])

            extra_lines: list[str] = []
            while k < len(lines):
                cur = lines[k]
                s = cur.strip()
                if is_boundary(s):
                    break
                if s.startswith("|") and s.endswith("|"):
                    break
                if cur.startswith("```"):
                    code_lines: list[str] = []
                    k += 1
                    while k < len(lines) and not lines[k].startswith("```"):
                        cs = squeeze_ws(lines[k])
                        if cs:
                            code_lines.append(cs)
                        k += 1
                    if k < len(lines) and lines[k].startswith("```"):
                        k += 1
                    if code_lines:
                        extra_lines.append("<br>".join(code_lines))
                    continue
                if s:
                    extra_lines.append(s)
                k += 1

            if extra_lines:
                merged_text = "<br>".join(extra_lines)
                # 尝试把“左右列粘连的一行”拆回两列新行
                m_joined = re.match(
                    r"^(?P<left>.+?)\s+(?P<right>A矩阵、B矩阵、C矩阵.+?需要满足对齐约束：)$",
                    merged_text,
                )
                if m_joined:
                    rows.append([squeeze_ws(m_joined.group("left")), squeeze_ws(m_joined.group("right"))])
                else:
                    rows[-1][0] = append_desc_cell(rows[-1][0], merged_text)
            i = k

        # 处理“转置场景/计算方式如下”后续公式掉到表外
        if rows and any(x in rows[-1][0] for x in {"计算方式如下", "转置场景：", "非转置场景："}):
            k2 = i
            extra2: list[str] = []
            while k2 < len(lines):
                cur = lines[k2]
                s = cur.strip()
                if not s:
                    k2 += 1
                    continue
                if is_boundary(s):
                    break
                if s.startswith("|") and s.endswith("|"):
                    break
                if cur.startswith("```"):
                    code_lines: list[str] = []
                    k2 += 1
                    while k2 < len(lines) and not lines[k2].startswith("```"):
                        cs = squeeze_ws(lines[k2])
                        if cs:
                            code_lines.append(cs)
                        k2 += 1
                    if k2 < len(lines) and lines[k2].startswith("```"):
                        k2 += 1
                    if code_lines:
                        extra2.append("<br>".join(code_lines))
                    continue
                if s:
                    extra2.append(s)
                k2 += 1
            if extra2:
                merged2 = "<br>".join(extra2)
                m_joined2 = re.match(
                    r"^(?P<left>.+?)\s+(?P<right>A矩阵、B矩阵、C矩阵.+?需要满足对齐约束：)$",
                    merged2,
                )
                if m_joined2:
                    rows.append([squeeze_ws(m_joined2.group("left")), squeeze_ws(m_joined2.group("right"))])
                else:
                    rows[-1][0] = append_desc_cell(rows[-1][0], merged2)
                i = k2

        for left, right in rows:
            out.append(f"| {escape_table_cell(left)} | {escape_table_cell(right)} |")
        out.append("")

    return out


def wrap_code_runs_in_segment(lines: list[str]) -> list[str]:
    out: list[str] = []
    run: list[str] = []

    def flush_run():
        nonlocal run
        if not run:
            return
        if len([x for x in run if x.strip()]) >= 2:
            out.append("```cpp")
            out.extend(run)
            out.append("```")
        else:
            out.extend(run)
        run = []

    for idx, ln in enumerate(lines):
        t = ln.strip()
        prev_is_code = bool(run)
        is_code = False
        if t and not t.startswith("|"):
            is_code = is_code_like(ln, prev_is_code=prev_is_code)
        if is_code:
            run.append(ln)
            continue
        flush_run()
        out.append(ln)

    flush_run()
    return out


def postprocess_call_example_code_blocks(lines: list[str]) -> list[str]:
    """
    在“调用示例”段落中补齐缺失的代码围栏，防止程序段被当正文。
    """
    out: list[str] = []
    i = 0
    in_code = False
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            in_code = not in_code
            out.append(line)
            i += 1
            continue
        if in_code:
            out.append(line)
            i += 1
            continue

        if line.strip() == "### 调用示例":
            out.append(line)
            i += 1
            seg: list[str] = []
            while i < len(lines):
                cur = lines[i]
                if cur.startswith("## ") or (cur.startswith("### ") and cur.strip() != "### 调用示例"):
                    break
                seg.append(cur)
                i += 1
            if any(x.startswith("```") for x in seg):
                out.extend(seg)
            else:
                out.extend(wrap_code_runs_in_segment(seg))
            continue

        out.append(line)
        i += 1
    return out


def split_code_block_by_prose_leaks(fence: str, body: list[str]) -> list[str]:
    out: list[str] = []
    buf: list[str] = []

    def flush_buf():
        nonlocal buf
        if not buf:
            return
        out.append(fence)
        out.extend(buf)
        out.append("```")
        buf = []

    for ln in body:
        cur = squeeze_ws(ln)
        prev = squeeze_ws(buf[-1]) if buf else ""
        # 代码注释续行修复：OCR 将注释第二行丢掉注释符时，补回注释前缀
        if (
            cur
            and re.search(r"[\u4e00-\u9fff]", cur)
            and prev.startswith("#")
            and not cur.startswith(("#", "//", "/*", "*"))
            and not re.search(r"[{}();=]", cur)
        ):
            buf.append(f"# {cur}")
            continue
        if (
            cur
            and re.search(r"[\u4e00-\u9fff]", cur)
            and (prev.startswith("//") or prev.startswith("/*") or prev.startswith("*"))
            and not cur.startswith(("#", "//", "/*", "*"))
            and not re.search(r"[{}();=]", cur)
        ):
            buf.append(f"// {cur}")
            continue

        if should_break_code_block_for_prose(ln):
            flush_buf()
            out.append(ln)
        else:
            buf.append(ln)
    flush_buf()
    return out


def postprocess_code_block_prose_leaks(lines: list[str]) -> list[str]:
    """
    兜底修复：若 fenced code block 内混入明显中文正文句，将其移出代码块。
    """
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith("```"):
            out.append(line)
            i += 1
            continue

        fence = line if line.strip() else "```cpp"
        i += 1
        body: list[str] = []
        while i < len(lines) and not lines[i].startswith("```"):
            body.append(lines[i])
            i += 1
        if i < len(lines) and lines[i].startswith("```"):
            i += 1

        repaired = split_code_block_by_prose_leaks(fence, body)
        if repaired:
            out.extend(repaired)
        else:
            out.append(fence)
            out.append("```")
    return out


def postprocess_merge_comment_prose_between_code_blocks(lines: list[str]) -> list[str]:
    """
    修复模式：
    ```cpp
    // 注释前半句...
    ```
    注释后半句（被误识别成正文）
    ```cpp
    ...
    ```
    将中间正文并回注释，避免代码被切成两段。
    """
    out: list[str] = []
    i = 0
    while i < len(lines):
        if not lines[i].startswith("```"):
            out.append(lines[i])
            i += 1
            continue

        # 读取第一个代码块
        block1: list[str] = [lines[i]]
        i += 1
        while i < len(lines) and not lines[i].startswith("```"):
            block1.append(lines[i])
            i += 1
        if i < len(lines) and lines[i].startswith("```"):
            block1.append(lines[i])
            i += 1

        # 提取 block1 最后一条非空代码行
        last_code = ""
        for ln in reversed(block1[1:-1]):
            if squeeze_ws(ln):
                last_code = squeeze_ws(ln)
                break

        # 读取中间段（到下一个代码块开始前）
        mid_start = i
        mids: list[str] = []
        while i < len(lines) and not lines[i].startswith("```"):
            mids.append(lines[i])
            i += 1
            if len(mids) > 6:  # 安全阈值，避免吞掉大段正文
                break

        can_merge = False
        if i < len(lines) and lines[i].startswith("```") and mids:
            # 中间段需全是短正文，不含标题/表格/列表
            nonempty = [squeeze_ws(x) for x in mids if squeeze_ws(x)]
            if nonempty and len(nonempty) <= 2:
                bad_struct = any(
                    x.startswith(("##", "###", "|", "- ", "* "))
                    for x in nonempty
                )
                prose_like = all(
                    re.search(r"[\u4e00-\u9fff]", x)
                    and not re.search(r"[{}[\];=]", x)
                    for x in nonempty
                )
                prev_is_comment = last_code.startswith("//") or last_code.startswith("#") or last_code.startswith("/*") or last_code.startswith("*")
                can_merge = (not bad_struct) and prose_like and prev_is_comment

        if not can_merge:
            out.extend(block1)
            out.extend(mids)
            continue

        # 合并到 block1 尾部（closing fence 之前）
        prefix = "// "
        if last_code.startswith("#"):
            prefix = "# "
        for m in [squeeze_ws(x) for x in mids if squeeze_ws(x)]:
            block1.insert(-1, f"{prefix}{m}")
        out.extend(block1)
        # 不消费后续第二个代码块起始 fence，让循环下一轮处理
    return out


def postprocess_merge_adjacent_code_blocks(lines: list[str]) -> list[str]:
    """
    合并仅由空行分隔的相邻代码块，减少被错误切碎的 fenced code。
    """
    out: list[str] = []
    i = 0
    while i < len(lines):
        if not lines[i].startswith("```"):
            out.append(lines[i])
            i += 1
            continue

        fence = lines[i]
        i += 1
        body: list[str] = []
        while i < len(lines) and not lines[i].startswith("```"):
            body.append(lines[i])
            i += 1
        if i < len(lines) and lines[i].startswith("```"):
            i += 1

        # 尝试吸收后续仅空行分隔的代码块
        while True:
            j = i
            blanks: list[str] = []
            while j < len(lines) and lines[j].strip() == "":
                blanks.append(lines[j])
                j += 1
            if j >= len(lines) or not lines[j].startswith("```"):
                # 没有紧邻代码块，保留空行
                out.append(fence)
                out.extend(body)
                out.append("```")
                out.extend(blanks)
                i = j
                break

            # 读取下一个代码块
            next_fence = lines[j]
            j += 1
            next_body: list[str] = []
            while j < len(lines) and not lines[j].startswith("```"):
                next_body.append(lines[j])
                j += 1
            if j >= len(lines):
                out.append(fence)
                out.extend(body)
                out.append("```")
                i = j
                break
            j += 1  # 跳过 next closing fence

            # 合并（沿用第一个 fence 语言）
            if body and body[-1].strip() != "":
                body.append("")
            body.extend(next_body)
            i = j
            # 继续尝试吸收更多代码块
    return out


def postprocess_option_table_blocks(lines: list[str]) -> list[str]:
    out: list[str] = []
    in_code = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            in_code = not in_code
            out.append(line)
            i += 1
            continue
        if in_code or line.strip() != "### 选项 是否 说明":
            out.append(line)
            i += 1
            continue

        j = i + 1
        while j < len(lines) and lines[j].strip() == "":
            j += 1

        if j < len(lines) and lines[j].strip() == "必需":
            j += 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1

        body: list[str] = []
        k = j
        while k < len(lines):
            cur = lines[k]
            if cur.startswith("## ") or cur.startswith("### ") or cur.startswith("```"):
                break
            body.append(cur)
            k += 1

        rows: list[list[str]] = []
        for raw in body:
            t = raw.strip()
            if not t:
                continue
            if t.startswith("- "):
                content = squeeze_ws(t[2:])
                m = OPTION_TABLE_ROW_RE.match(content)
                if not m:
                    continue
                opt = normalize_option_cell(m.group("option"))
                req = m.group("required")
                desc = m.group("desc")
                rows.append([opt, req, desc])
            elif rows:
                rows[-1][2] = f"{rows[-1][2]}<br>{squeeze_ws(t)}"

        if len(rows) < 1:
            out.append(line)
            i += 1
            continue

        out.append("### 选项 是否必需 说明")
        out.append("")
        out.append("| 选项 | 是否必需 | 说明 |")
        out.append("| --- | --- | --- |")
        for opt, req, desc in rows:
            out.append(
                f"| {escape_table_cell(opt)} | {escape_table_cell(req)} | {escape_table_cell(desc)} |"
            )
        out.append("")
        i = k

    return out


def postprocess_common_table_blocks(lines: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    in_code = False
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            in_code = not in_code
            out.append(line)
            i += 1
            continue
        if in_code:
            out.append(line)
            i += 1
            continue

        head = line.strip()
        if head == "### 产品支持情况":
            block, nxt = parse_product_support_overview_block(lines, i)
            if block:
                out.extend(block)
                i = nxt
                continue
        if head == "### 产品 是否支持":
            block, nxt = parse_product_support_table_block(lines, i)
            if block:
                out.extend(block)
                i = nxt
                continue
        if head.startswith("### 产品 "):
            block, nxt = parse_product_support_heading_block(lines, i)
            if block:
                out.extend(block)
                i = nxt
                continue
        if re.fullmatch(r"\|\s*产品\s*\|\s*是否支持\s*\|", head):
            block, nxt = parse_product_support_markdown_table_block(lines, i)
            if block:
                out.extend(block)
                i = nxt
                continue
        if TABLE_TITLE_START_RE.match(head):
            block, nxt = parse_operator_spec_table_block(lines, i)
            if block:
                out.extend(block)
                i = nxt
                continue
            block, nxt = parse_address_mapping_table_block(lines, i)
            if block:
                out.extend(block)
                i = nxt
                continue
            block, nxt = parse_operator_shape_spec_table_block(lines, i)
            if block:
                out.extend(block)
                i = nxt
                continue
            block, nxt = parse_generic_titled_table_block(lines, i)
            if block:
                out.extend(block)
                i = nxt
                continue
        if head == "### 接口分类 接口名称":
            block, nxt = parse_api_scope_table_block(lines, i, with_remark=False)
            if block:
                out.extend(block)
                i = nxt
                continue
        if head == "### 接口分类 接口名称 备注":
            block, nxt = parse_api_scope_table_block(lines, i, with_remark=True)
            if block:
                out.extend(block)
                i = nxt
                continue
        if re.fullmatch(r"\|\s*接口分类\s*\|\s*接口名称\s*\|", head):
            block, nxt = parse_api_scope_markdown_table_block(lines, i, with_remark=False)
            if block:
                out.extend(block)
                i = nxt
                continue
        if re.fullmatch(r"\|\s*接口分类\s*\|\s*接口名称\s*\|\s*备注\s*\|", head):
            block, nxt = parse_api_scope_markdown_table_block(lines, i, with_remark=True)
            if block:
                out.extend(block)
                i = nxt
                continue
        if head == "### 变量 配置说明":
            block, nxt = parse_var_table_block(lines, i)
            if block:
                out.extend(block)
                i = nxt
                continue
        if head == "### 名称 作用描述 使用场景":
            block, nxt = parse_lib_table_3col_block(lines, i)
            if block:
                out.extend(block)
                i = nxt
                continue
        if head == "### 名称 作用描述":
            block, nxt = parse_lib_table_2col_block(lines, i)
            if block:
                out.extend(block)
                i = nxt
                continue

        out.append(line)
        i += 1

    return out


def parse_var_table_block(lines: list[str], start: int) -> tuple[list[str] | None, int]:
    j = start + 1
    while j < len(lines) and lines[j].strip() == "":
        j += 1

    rows: list[list[str]] = []
    k = j
    while k < len(lines):
        cur = lines[k]
        t = cur.strip()
        if is_block_boundary(cur):
            break

        if cur.startswith("```"):
            code_lines: list[str] = []
            k += 1
            while k < len(lines) and not lines[k].startswith("```"):
                code_line = squeeze_ws(lines[k])
                if code_line:
                    code_lines.append(f"`{escape_table_cell(code_line)}`")
                k += 1
            if k < len(lines) and lines[k].startswith("```"):
                k += 1
            if rows and code_lines:
                rows[-1][1] = append_desc_cell(rows[-1][1], "<br>".join(code_lines))
            continue

        if not t:
            k += 1
            continue

        m = CMAKE_VAR_HEAD_RE.match(t)
        if m:
            var_name, desc = repair_cmake_var_and_desc(m.group(1), m.group(2))
            rows.append([var_name, desc])
            k += 1
            continue

        if rows:
            rows[-1][1] = append_desc_cell(rows[-1][1], t)
        k += 1

    if not rows:
        return None, start + 1

    table_lines = [
        "### 变量 配置说明",
        "",
        "| 变量 | 配置说明 |",
        "| --- | --- |",
    ]
    for var_name, desc in rows:
        table_lines.append(
            f"| {escape_table_cell(var_name)} | {escape_table_cell(desc)} |"
        )
    table_lines.append("")
    return table_lines, k


def parse_lib_table_3col_block(lines: list[str], start: int) -> tuple[list[str] | None, int]:
    j = start + 1
    while j < len(lines) and lines[j].strip() == "":
        j += 1

    rows: list[list[str]] = []
    k = j
    while k < len(lines):
        cur = lines[k]
        t = cur.strip()
        if is_block_boundary(cur):
            break
        if not t:
            k += 1
            continue

        m = LIB_HEAD_RE.match(t)
        if m and "." in m.group(1):
            name, rest = repair_lib_name(m.group(1), m.group(2))
            desc, scene = squeeze_ws(rest), ""
            if "。" in desc:
                pos = desc.find("。")
                scene = squeeze_ws(desc[pos + 1 :])
                desc = desc[: pos + 1]
            rows.append([name, desc, scene])
            k += 1
            continue

        if rows:
            if rows[-1][2]:
                rows[-1][2] = append_desc_cell(rows[-1][2], t)
            else:
                rows[-1][2] = squeeze_ws(t)
        k += 1

    if not rows:
        return None, start + 1

    table_lines = [
        "### 名称 作用描述 使用场景",
        "",
        "| 名称 | 作用描述 | 使用场景 |",
        "| --- | --- | --- |",
    ]
    for name, desc, scene in rows:
        table_lines.append(
            f"| {escape_table_cell(name)} | {escape_table_cell(desc)} | {escape_table_cell(scene)} |"
        )
    table_lines.append("")
    return table_lines, k


def parse_lib_table_2col_block(lines: list[str], start: int) -> tuple[list[str] | None, int]:
    j = start + 1
    while j < len(lines) and lines[j].strip() == "":
        j += 1

    rows: list[list[str]] = []
    k = j
    while k < len(lines):
        cur = lines[k]
        t = cur.strip()
        if is_block_boundary(cur):
            break
        if not t:
            k += 1
            continue

        m = LIB_HEAD_RE.match(t)
        if m and m.group(1).startswith("lib"):
            name, desc = repair_lib_name(m.group(1), m.group(2))
            rows.append([name, squeeze_ws(desc)])
            k += 1
            continue

        if rows:
            rows[-1][1] = append_desc_cell(rows[-1][1], t)
        k += 1

    if not rows:
        return None, start + 1

    table_lines = [
        "### 名称 作用描述",
        "",
        "| 名称 | 作用描述 |",
        "| --- | --- |",
    ]
    for name, desc in rows:
        table_lines.append(
            f"| {escape_table_cell(name)} | {escape_table_cell(desc)} |"
        )
    table_lines.append("")
    return table_lines, k


def split_header_cells(line: str) -> list[str]:
    t = squeeze_ws(line)
    if not t:
        return []
    cells = [squeeze_ws(x) for x in re.split(r"\s{2,}", t) if squeeze_ws(x)]
    if len(cells) >= 2:
        return cells
    parts = [x for x in t.split(" ") if x]
    if "data" in parts and "type" in parts:
        merged: list[str] = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and parts[i] == "data" and parts[i + 1] == "type":
                merged.append("data type")
                i += 2
                continue
            merged.append(parts[i])
            i += 1
        parts = merged
    if 2 <= len(parts) <= 5:
        # 例如：组件分类 组件名称 组件功能 / 接口名 功能描述
        return parts
    if len(parts) == 2:
        return parts
    if t == "name shape data type format":
        return ["name", "shape", "data type", "format"]
    return []


def split_row_cells_by_ncol(line: str, ncol: int) -> list[str]:
    t = squeeze_ws(line)
    if not t:
        return []
    cells = [squeeze_ws(x) for x in re.split(r"\s{2,}", t) if squeeze_ws(x)]
    if len(cells) >= ncol:
        if len(cells) > ncol:
            cells = cells[: ncol - 1] + [" ".join(cells[ncol - 1 :])]
        return cells

    if ncol == 2:
        # 公式/英文条件 + 中文说明的两列表，优先在首个中文字符处切分，避免只取首词
        m_cjk = re.search(r"[\u4e00-\u9fff]", t)
        if m_cjk and m_cjk.start() >= 4:
            left = squeeze_ws(t[: m_cjk.start()])
            right = squeeze_ws(t[m_cjk.start() :])
            if left and right:
                return [left, right]
        m_num = TABLE_ROW_TRAIL_NUM_RE.match(t)
        if m_num:
            left = squeeze_ws(m_num.group("left"))
            num = m_num.group("num")
            tail = squeeze_ws(m_num.group("tail"))
            if tail in {"系列产品"}:
                left = join_prose(left, tail)
                return [left, num]
            return [left, f"{num}{tail}".strip()]
        parts = [x for x in t.split(" ") if x]
        if len(parts) >= 2 and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_:-]*", parts[0]):
            return [parts[0], " ".join(parts[1:]).strip()]
        parts = [x for x in t.split(" ") if x]
        if len(parts) >= 2 and len(parts[-1]) <= 16:
            return [" ".join(parts[:-1]).strip(), parts[-1]]
    else:
        if ncol == 3:
            parts = [x for x in t.split(" ") if x]
            if len(parts) >= 3:
                first = parts[0]
                rem = t[len(first):].strip()
                m_cjk = re.search(r"[\u4e00-\u9fff]", rem)
                if m_cjk and m_cjk.start() > 0:
                    second = squeeze_ws(rem[: m_cjk.start()])
                    third = squeeze_ws(rem[m_cjk.start() :])
                    if second and third:
                        return [first, second, third]
        parts = [x for x in t.split(" ") if x]
        if len(parts) >= ncol:
            return parts[: ncol - 1] + [" ".join(parts[ncol - 1 :]).strip()]
    return []


def looks_like_plain_paragraph(line: str) -> bool:
    t = squeeze_ws(line)
    if not t:
        return False
    if t.startswith(("图", "表", "步骤")):
        return False
    if re.search(r"[。！？]", t):
        return True
    if len(t) > 60 and re.search(r"[\u4e00-\u9fff]", t):
        return True
    return False


def parse_generic_titled_table_block(lines: list[str], start: int) -> tuple[list[str] | None, int]:
    title = lines[start].strip()
    j = start + 1
    while j < len(lines) and not lines[j].strip():
        j += 1
    if j >= len(lines):
        return None, start + 1
    # 已是 markdown 表格
    if lines[j].strip().startswith("|"):
        return None, start + 1

    head = lines[j].strip()
    if head.startswith("### "):
        head = squeeze_ws(head[4:])
    header_cells = split_header_cells(head)
    # 分页续表常见：表头被拆成“参数名称 说明” + 下一行“数据类型”
    if len(header_cells) == 2:
        c0 = header_cells[0].replace(" ", "")
        c1 = header_cells[1].replace(" ", "")
        if c0 in {"参数名", "参数名称"} and c1 in {"说明", "描述"}:
            k_frag = j + 1
            while k_frag < len(lines) and not lines[k_frag].strip():
                k_frag += 1
            if k_frag < len(lines):
                frag = squeeze_ws(lines[k_frag].strip())
                if frag in {"数据类型", "类型", "输入/输出"}:
                    header_cells = [header_cells[0], frag, header_cells[1]]
                    j = k_frag

    # 列头都识别不到时，放弃处理，避免吞掉表后内容
    if len(header_cells) < 2:
        return None, start + 1

    ncol = len(header_cells)
    rows: list[list[str]] = []
    k = j + 1

    while k < len(lines):
        cur = lines[k]
        t = cur.strip()
        if not t:
            k += 1
            continue
        if cur.startswith("## ") or cur.startswith("```"):
            break
        if TABLE_TITLE_START_RE.match(t):
            break
        if cur.startswith("### "):
            payload = squeeze_ws(cur[4:])
            if payload in {"功能说明", "返回值说明", "约束说明", "调用示例", "说明", "注意", "参数说明"}:
                break
            t = payload

        row = split_row_cells_by_ncol(t, ncol)
        if (not row) and ncol >= 3:
            parts = [x for x in t.split(" ") if x]
            if rows and len(parts) == ncol - 1:
                row = [rows[-1][0]] + parts
        if row:
            rows.append(row)
            k += 1
            continue

        # 无法识别为表行时，优先判为表后正文/代码起点并停止，防止吞内容
        if rows and (looks_like_plain_paragraph(t) or is_code_like(t, prev_is_code=False)):
            break
        if rows and (("。" in t) or ("：" in t) or t.startswith(("以下", "例如", "其中"))):
            break
        if rows:
            rows[-1][-1] = append_desc_cell(rows[-1][-1], t)
            k += 1
            continue
        break

    if not rows:
        return None, start + 1

    # 兜底：分页导致“数据类型”列表头丢失时，从内容恢复三列结构
    if (
        ncol == 2
        and len(header_cells) == 2
        and header_cells[0].replace(" ", "") in {"参数名", "参数名称"}
        and header_cells[1].replace(" ", "") in {"说明", "描述"}
    ):
        dtype_re = re.compile(
            r"^(?:u?int(?:8|16|32|64)?_t|int|float|double|bool|half|bfloat16_t|size_t|void|char|string)\b",
            re.IGNORECASE,
        )
        promoted: list[list[str]] = []
        hit = 0
        for r in rows:
            if len(r) < 2:
                promoted.append((r + ["", ""])[:3])
                continue
            name = squeeze_ws(r[0])
            cell = squeeze_ws(r[1])
            m = re.match(r"^(?P<dtype>\S+)\s+(?P<desc>.+)$", cell)
            if m and dtype_re.match(m.group("dtype")):
                promoted.append([name, m.group("dtype"), squeeze_ws(m.group("desc"))])
                hit += 1
            else:
                promoted.append([name, "", cell])
        if hit >= 1:
            header_cells = [header_cells[0], "数据类型", header_cells[1]]
            ncol = 3
            rows = promoted

    out = [title, ""]
    out.append("| " + " | ".join(escape_table_cell(x) for x in header_cells) + " |")
    out.append("| " + " | ".join(["---"] * ncol) + " |")
    for row in rows:
        fixed = (row + [""] * ncol)[:ncol]
        out.append("| " + " | ".join(escape_table_cell(x) for x in fixed) + " |")
    out.append("")
    return out, k


def parse_address_mapping_table_block(lines: list[str], start: int) -> tuple[list[str] | None, int]:
    title = lines[start].strip()
    if "地址空间映射关系" not in title:
        return None, start + 1

    k = start + 1
    while k < len(lines) and not lines[k].strip():
        k += 1

    rows: list[list[str]] = []
    while k < len(lines):
        cur = lines[k]
        t = cur.strip()
        if not t:
            k += 1
            continue
        if cur.startswith("## ") or cur.startswith("```"):
            break
        if TABLE_TITLE_START_RE.match(t):
            break
        if cur.startswith("### "):
            payload = squeeze_ws(cur[4:])
            if payload in {"说明", "注意", "功能说明", "参数说明", "函数原型", "返回值说明", "约束说明", "调用示例"}:
                break
            t = payload

        if ("地址空间限定符" in t and "物理存储空间" in t):
            k += 1
            continue

        if t.startswith("|") and t.endswith("|"):
            cells = [squeeze_ws(x) for x in t.strip("|").split("|")]
            if cells and all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                k += 1
                continue
            if len(cells) >= 2:
                name = cells[0].strip("`")
                desc = " ".join(cells[1:]).strip()
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                    name = f"__{name}__"
                elif re.fullmatch(r"__?[A-Za-z][A-Za-z0-9_]*__?", name):
                    if not name.startswith("__"):
                        name = f"__{name.strip('_')}__"
                rows.append([name, desc])
                k += 1
                continue

        m = re.match(r"^`?(?P<name>__?[A-Za-z][A-Za-z0-9_]*__?)`?\s+(?P<desc>.+)$", t)
        if m:
            name = m.group("name")
            if not name.startswith("__"):
                name = f"__{name.strip('_')}__"
            rows.append([name, squeeze_ws(m.group("desc"))])
            k += 1
            continue

        if rows and (looks_like_plain_paragraph(t) or normalize_list_item(t) is not None):
            break
        if rows:
            rows[-1][1] = append_desc_cell(rows[-1][1], t)
            k += 1
            continue
        k += 1

    if not rows:
        return None, start + 1

    out = [title, "", "| 地址空间限定符 | AI Core物理存储空间 |", "| --- | --- |"]
    for name, desc in rows:
        out.append(f"| {escape_table_cell(f'`{name}`')} | {escape_table_cell(desc)} |")
    out.append("")
    return out, k


def normalize_support_flag(text: str) -> str | None:
    t = squeeze_ws(text).strip("()[]{}<>.,;:，。；：")
    if not t:
        return None
    if t in {"√", "✓", "✔", "厂", "丁", "丅", "V", "v", "Y", "y", "支持"}:
        return "√"
    if t in {"x", "X", "×", "✗", "✘", "不支持"}:
        return "x"
    return None


def parse_product_support_row(text: str) -> tuple[str, str] | None:
    t = squeeze_ws(text).replace("产品是否支持", "").strip()
    if not t:
        return None
    m = re.match(r"^(?P<prod>.+?)\s*(?P<flag>\S+)$", t)
    if not m:
        return None
    prod = squeeze_ws(m.group("prod"))
    flag = normalize_support_flag(m.group("flag"))
    if not flag:
        return None
    return prod, flag


def parse_product_support_table_block(lines: list[str], start: int) -> tuple[list[str] | None, int]:
    if lines[start].strip() != "### 产品 是否支持":
        return None, start + 1

    rows: list[tuple[str, str]] = []
    k = start + 1
    while k < len(lines):
        cur = lines[k]
        t = cur.strip()
        if not t:
            k += 1
            continue
        if cur.startswith("## ") or cur.startswith("```"):
            break
        if cur.startswith("### "):
            payload = squeeze_ws(cur[4:])
            if payload in {"产品 是否支持", "产品是否支持"}:
                k += 1
                continue
            parsed = parse_product_support_row(payload)
            if parsed:
                rows.append(parsed)
                k += 1
                continue
            break
        if t in {"产品 是否支持", "产品是否支持"}:
            k += 1
            continue
        parsed = parse_product_support_row(t)
        if parsed:
            rows.append(parsed)
            k += 1
            continue
        if rows and (looks_like_plain_paragraph(t) or is_code_like(t, prev_is_code=False)):
            break
        if rows:
            last_prod, last_flag = rows[-1]
            rows[-1] = (append_desc_cell(last_prod, t), last_flag)
            k += 1
            continue
        k += 1

    if len(rows) < 1:
        return None, start + 1

    out = [
        "### 产品 是否支持",
        "",
        "| 产品 | 是否支持 |",
        "| --- | --- |",
    ]
    for prod, flag in rows:
        out.append(f"| {escape_table_cell(prod)} | {escape_table_cell(flag)} |")
    out.append("")
    return out, k


def parse_product_support_markdown_table_block(lines: list[str], start: int) -> tuple[list[str] | None, int]:
    if not re.fullmatch(r"\|\s*产品\s*\|\s*是否支持\s*\|", lines[start].strip()):
        return None, start + 1
    k = start + 1
    if k < len(lines) and lines[k].strip().startswith("| ---"):
        k += 1
    rows: list[tuple[str, str]] = []
    while k < len(lines):
        t = lines[k].strip()
        if not t:
            k += 1
            continue
        if not (t.startswith("|") and t.endswith("|")):
            break
        cells = [squeeze_ws(x) for x in t.strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
            k += 1
            continue
        if len(cells) < 2:
            k += 1
            continue
        prod = cells[0]
        flag = normalize_support_flag(cells[1])
        if not flag:
            parsed = parse_product_support_row(" ".join(cells[:2]))
            if parsed:
                prod, flag = parsed
            elif rows:
                prev_prod, prev_flag = rows[-1]
                rows[-1] = (append_desc_cell(prev_prod, " ".join(cells)), prev_flag)
                k += 1
                continue
            else:
                k += 1
                continue
        rows.append((prod, flag))
        k += 1
    if not rows:
        return None, start + 1
    out = ["| 产品 | 是否支持 |", "| --- | --- |"]
    for prod, flag in rows:
        out.append(f"| {escape_table_cell(prod)} | {escape_table_cell(flag)} |")
    return out, k


def parse_product_support_overview_block(lines: list[str], start: int) -> tuple[list[str] | None, int]:
    if lines[start].strip() != "### 产品支持情况":
        return None, start + 1

    k = start + 1
    while k < len(lines) and not lines[k].strip():
        k += 1
    if k < len(lines):
        lead = lines[k].strip()
        # 已是规范结构，交给其他解析器
        if lead == "### 产品 是否支持" or re.fullmatch(r"\|\s*产品\s*\|\s*是否支持\s*\|", lead):
            return None, start + 1
    if k < len(lines) and lines[k].startswith("```"):
        return None, start + 1

    support_char_re = re.compile(r"(?<![A-Za-z0-9_])(√|✓|✔|厂|丁|丅|V|v|Y|y|x|X|×|✗|✘)(?![A-Za-z0-9_])")
    raw_rows: list[tuple[str, list[str]]] = []
    header_hints: list[str] = []
    cur = k
    while cur < len(lines):
        s = lines[cur].strip()
        if not s:
            cur += 1
            continue
        if lines[cur].startswith("## "):
            break
        if lines[cur].startswith("### "):
            # 允许“### 产品支持情况”自身，其他三级标题视作结束
            if cur == start:
                cur += 1
                continue
            break
        if TABLE_TITLE_START_RE.match(s):
            break
        if s.startswith("|") and s.endswith("|"):
            break

        if ("产品" in s and "支持" in s) and ("Atlas" not in s):
            header_hints.append(s)
            cur += 1
            continue

        flags: list[str] = []
        spans: list[tuple[int, int]] = []
        for m in support_char_re.finditer(s):
            f = normalize_support_flag(m.group(1))
            if not f:
                continue
            flags.append(f)
            spans.append((m.start(), m.end()))
        if not flags:
            cur += 1
            continue

        product = s
        for a, b in reversed(spans):
            product = product[:a] + " " + product[b:]
        product = squeeze_ws(product).strip("，。；：")
        if product:
            raw_rows.append((product, flags))
        cur += 1

    if not raw_rows:
        return None, start + 1

    nflag = max((len(fs) for _, fs in raw_rows), default=0)
    if nflag <= 0:
        return None, start + 1

    hint = " ".join(header_hints)
    headers = ["产品"]
    if nflag == 1:
        headers.append("是否支持")
    elif nflag >= 2 and ("软" in hint and "硬" in hint):
        headers.extend(["是否支持（软同步原型）", "是否支持（硬同步原型）"])
        for idx in range(3, nflag + 1):
            headers.append(f"是否支持（列{idx}）")
    else:
        for idx in range(1, nflag + 1):
            headers.append(f"是否支持（列{idx}）")

    out = ["### 产品支持情况", ""]
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for prod, flags in raw_rows:
        cells = [prod] + flags + [""] * max(0, nflag - len(flags))
        out.append("| " + " | ".join(escape_table_cell(c) for c in cells[: len(headers)]) + " |")
    out.append("")
    return out, cur


def parse_product_support_heading_block(lines: list[str], start: int) -> tuple[list[str] | None, int]:
    head = lines[start].strip()
    if not head.startswith("### 产品 "):
        return None, start + 1

    support_char_re = re.compile(r"(?<![A-Za-z0-9_])(√|✓|✔|厂|丁|丅|V|v|Y|y|x|X|×|✗|✘)(?![A-Za-z0-9_])")

    def parse_support_row(text: str) -> tuple[str, list[str], str] | None:
        s = squeeze_ws(text)
        if not s:
            return None
        spans: list[tuple[int, int]] = []
        flags: list[str] = []
        for m in support_char_re.finditer(s):
            f = normalize_support_flag(m.group(1))
            if not f:
                continue
            spans.append((m.start(), m.end()))
            flags.append(f)
        if not spans:
            return None
        first_a, _ = spans[0]
        _, last_b = spans[-1]
        left = squeeze_ws(s[:first_a])
        tail = squeeze_ws(s[last_b:])
        suffix_as_prod = bool(re.fullmatch(r"(列?产品|系列产品)?", tail))
        prod = squeeze_ws((left + " " + tail).strip()) if suffix_as_prod else left
        prod = squeeze_ws(prod).replace("产品是否支持", "").strip()
        remark = "" if suffix_as_prod else tail
        if not prod:
            return None
        return prod, flags, remark

    k = start + 1
    header_hints: list[str] = [head.replace("###", "").strip()]
    rows: list[tuple[str, list[str], str]] = []
    in_code = False
    while k < len(lines):
        raw = lines[k]
        s = raw.strip()
        if raw.startswith("```"):
            in_code = not in_code
            k += 1
            continue
        if in_code:
            k += 1
            continue
        if not s:
            k += 1
            continue
        if raw.startswith("## "):
            break
        if raw.startswith("### "):
            break
        if TABLE_TITLE_START_RE.match(s):
            break
        if s.startswith("|") and s.endswith("|"):
            break

        parsed = parse_support_row(s)
        if parsed:
            rows.append(parsed)
        elif ("支持" in s) or ("原型" in s) or ("接口" in s):
            header_hints.append(s)
        k += 1

    if not rows:
        return None, start + 1

    nflag = max((len(fs) for _, fs, _ in rows), default=0)
    if nflag <= 0:
        return None, start + 1
    hint = " ".join(header_hints)
    has_remark = ("备注" in hint) or any(bool(rm) for _, _, rm in rows)

    cols = ["产品"]
    if nflag == 1:
        cols.append("是否支持")
    elif nflag >= 2 and ("软" in hint and "硬" in hint):
        cols.extend(["是否支持（软同步原型）", "是否支持（硬同步原型）"])
        for idx in range(3, nflag + 1):
            cols.append(f"是否支持（列{idx}）")
    elif nflag >= 2 and ("栈" in hint and "GM" in hint):
        cols.extend(["是否支持（栈地址）", "是否支持（GM地址）"])
        for idx in range(3, nflag + 1):
            cols.append(f"是否支持（列{idx}）")
    else:
        for idx in range(1, nflag + 1):
            cols.append(f"是否支持（列{idx}）")
    if has_remark:
        cols.append("备注")

    out = [head, ""]
    out.append("| " + " | ".join(cols) + " |")
    out.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for prod, flags, remark in rows:
        cells = [prod] + flags + [""] * max(0, nflag - len(flags))
        if has_remark:
            cells.append(remark)
        out.append("| " + " | ".join(escape_table_cell(c) for c in cells[: len(cols)]) + " |")
    out.append("")
    return out, k


def parse_operator_spec_table_block(lines: list[str], start: int) -> tuple[list[str] | None, int]:
    """
    解析“算子设计规格/算子规格”这类复杂跨行表，统一输出为两列 Markdown 表：
    | 项目 | 值 |
    """
    title = lines[start].strip()
    if ("算子设计规格" not in title) and ("算子规格" not in title):
        return None, start + 1

    items: list[tuple[str, str]] = []
    k = start + 1
    pending_key = ""
    input_lines: list[str] = []
    seen_content = 0

    def normalize_operator_input_lines(raw_lines: list[str]) -> list[str]:
        out_lines: list[str] = []
        for ln in raw_lines:
            t = squeeze_ws(ln.strip())
            if not t:
                continue
            t = t.strip("|").strip()
            t = t.replace(r"\|", " ")
            t_low = t.lower()
            # 去掉嵌套表头
            if re.fullmatch(r"name\s+shape\s+data\s*type\s+format", t_low):
                continue
            multi_rows = list(
                re.finditer(
                    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?:[（(](?P<io>输入|输出)[)）])?\s*[（(](?P<shape>[^)）]+)[)）]\s*(?P<dtype>[A-Za-z0-9_]+)\s+(?P<fmt>[A-Za-z0-9_]+)",
                    t,
                )
            )
            if multi_rows:
                for m0 in multi_rows:
                    io_tag = m0.group("io") or ""
                    io_seg = f"（{io_tag}）" if io_tag else ""
                    shape = squeeze_ws(m0.group("shape"))
                    out_lines.append(
                        f"{m0.group('name')}{io_seg}: ({shape}), {m0.group('dtype')}, {m0.group('fmt')}"
                    )
                continue
            if "|" in t:
                cells = [squeeze_ws(x) for x in t.split("|") if squeeze_ws(x)]
                if len(cells) >= 4 and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", cells[0]):
                    out_lines.append(f"{cells[0]}: {cells[1]}, {cells[2]}, {cells[3]}")
                    continue
            m = re.match(
                r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\((?P<shape>[^)]*)\)\s+(?P<dtype>[A-Za-z0-9_]+)\s+(?P<fmt>[A-Za-z0-9_]+)$",
                t,
            )
            if m:
                shape = squeeze_ws(m.group("shape"))
                out_lines.append(
                    f"{m.group('name')}: ({shape}), {m.group('dtype')}, {m.group('fmt')}"
                )
                continue
            out_lines.append(t)
        # 去重保序
        dedup: list[str] = []
        seen: set[str] = set()
        for x in out_lines:
            if x in seen:
                continue
            seen.add(x)
            dedup.append(x)
        return dedup

    def parse_name_shape_dtype_fmt(text: str) -> tuple[str, str, str, str, str] | None:
        t = squeeze_ws(text)
        m = re.match(
            r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?:[（(](?P<io>输入|输出)[)）])?\s*:\s*[（(](?P<shape>[^)）]*)[)）]\s*,\s*(?P<dtype>[A-Za-z0-9_]+)\s*,\s*(?P<fmt>[A-Za-z0-9_]+)$",
            t,
        )
        if m:
            return (
                m.group("io") or "",
                m.group("name"),
                f"({squeeze_ws(m.group('shape'))})",
                m.group("dtype"),
                m.group("fmt"),
            )
        m2 = re.match(
            r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?:[（(](?P<io>输入|输出)[)）])?\s*[（(](?P<shape>[^)）]*)[)）]\s+(?P<dtype>[A-Za-z0-9_]+)\s+(?P<fmt>[A-Za-z0-9_]+)$",
            t,
        )
        if m2:
            return (
                m2.group("io") or "",
                m2.group("name"),
                f"({squeeze_ws(m2.group('shape'))})",
                m2.group("dtype"),
                m2.group("fmt"),
            )
        return None

    def upsert_item(key: str, val: str):
        key = squeeze_ws(key)
        val = squeeze_ws(val)
        if not key or not val:
            return
        if key == "使用的主要接口":
            # OCR 常把多行接口压成一行，按“接口Xxx”边界补回换行
            val = re.sub(r"(接口)([A-Za-z])", r"\1<br>\2", val)
        for i, (k0, v0) in enumerate(items):
            if k0 == key:
                items[i] = (k0, append_desc_cell(v0, val))
                return
        items.append((key, val))

    while k < len(lines) and (k - start) <= 120:
        cur = lines[k]
        t = cur.strip()

        if not t:
            k += 1
            continue
        if cur.startswith("## "):
            break
        if TABLE_TITLE_START_RE.match(t) and k != start + 1:
            break
        if cur.startswith("### "):
            h = squeeze_ws(cur[4:])
            if h.startswith("算子类型 "):
                upsert_item("算子类型", h[len("算子类型 ") :])
                seen_content += 1
                k += 1
                continue
            if h.startswith("算子实现文件 "):
                upsert_item("算子实现文件名称", h[len("算子实现文件 ") :])
                seen_content += 1
                k += 1
                continue
            if h in {"核函数开发", "核函数定义", "算子类实现", "调用示例", "函数实现"}:
                break
            # 其他三级标题视作边界
            if seen_content > 0:
                break
            k += 1
            continue

        if t.startswith("算子类型"):
            upsert_item("算子类型", t[len("算子类型") :].strip())
            seen_content += 1
            k += 1
            continue
        if t.startswith("|"):
            cells = [squeeze_ws(x) for x in t.strip("|").split("|")]
            cells = [x for x in cells if x]
            if cells and all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                k += 1
                continue
            if len(cells) >= 2:
                key = cells[0]
                val = " ".join(cells[1:]).strip()
                if key.startswith("算子类型"):
                    upsert_item("算子类型", val)
                elif key.startswith(("算子输入", "算子输入输出")):
                    pending_key = "算子输入输出" if "输出" in key else "算子输入"
                    hdr = key.replace("算子输入输出", "").replace("算子输入", "").strip()
                    line0 = " ".join([x for x in [hdr, val] if x]).strip()
                    if line0:
                        input_lines.append(line0)
                elif key.startswith("算子输出"):
                    upsert_item("算子输出", val)
                elif key.startswith("核函数名称"):
                    upsert_item("核函数名称", val)
                else:
                    upsert_item(key, val)
                seen_content += 1
                k += 1
                continue
        if t in {"（OpType）", "(OpType)"}:
            k += 1
            continue
        if t.startswith("算子输入输出"):
            pending_key = "算子输入输出"
            rest = t[len("算子输入输出") :].strip()
            if rest:
                input_lines.append(rest)
            seen_content += 1
            k += 1
            continue
        if t.startswith("算子输入"):
            pending_key = "算子输入"
            rest = t[len("算子输入") :].strip()
            if rest:
                input_lines.append(rest)
            seen_content += 1
            k += 1
            continue
        if t.startswith("算子输出"):
            upsert_item("算子输出", t[len("算子输出") :].strip())
            seen_content += 1
            k += 1
            continue
        if t.startswith("核函数名称"):
            upsert_item("核函数名称", t[len("核函数名称") :].strip())
            seen_content += 1
            k += 1
            continue
        if t.startswith("使用的主要接") or t.startswith("使用的主要接口"):
            txt = t.replace("使用的主要接", "").replace("接口口", "接口")
            upsert_item("使用的主要接口", txt.strip())
            pending_key = "使用的主要接口"
            seen_content += 1
            k += 1
            continue
        if t.startswith("算子实现文件"):
            upsert_item("算子实现文件名称", t[len("算子实现文件") :].strip())
            seen_content += 1
            k += 1
            continue
        if t == "名称":
            k += 1
            continue

        if cur.startswith("```"):
            code_lines: list[str] = []
            k += 1
            while k < len(lines) and not lines[k].startswith("```"):
                c = squeeze_ws(lines[k])
                if c:
                    code_lines.append(c)
                k += 1
            if k < len(lines) and lines[k].startswith("```"):
                k += 1
            if pending_key in {"算子输入", "算子输入输出"} and code_lines:
                input_lines.extend(code_lines)
            elif pending_key and code_lines:
                upsert_item(pending_key, "<br>".join(code_lines))
            continue

        if pending_key in {"算子输入", "算子输入输出"}:
            input_lines.append(t)
            k += 1
            continue
        if pending_key == "使用的主要接口":
            upsert_item("使用的主要接口", t)
            k += 1
            continue

        if seen_content > 0:
            # 内容区结束
            break
        k += 1

    if input_lines:
        key = "算子输入输出" if any("输出" in x for x in input_lines[:1]) else "算子输入"
        normalized_input = normalize_operator_input_lines(input_lines)
        upsert_item(key, "<br>".join(normalized_input if normalized_input else input_lines))

    if len(items) < 2:
        return None, start + 1

    # 固定核心字段顺序，避免“算子输入”被拖到末尾
    preferred_order = [
        "算子类型",
        "算子输入",
        "算子输入输出",
        "算子输出",
        "核函数名称",
        "使用的主要接口",
        "算子实现文件名称",
    ]
    rank = {k: i for i, k in enumerate(preferred_order)}
    ordered_items = sorted(
        enumerate(items),
        key=lambda iv: (rank.get(iv[1][0], 1000 + iv[0])),
    )

    # 若包含“算子输入”结构化内容，输出宽表，避免把小表压进单元格
    input_val = ""
    for _, (kk, vv) in ordered_items:
        if kk in {"算子输入", "算子输入输出"}:
            input_val = vv
            break
    parsed_input_rows: list[tuple[str, str, str, str, str]] = []
    if input_val:
        for seg in [squeeze_ws(x) for x in input_val.split("<br>") if squeeze_ws(x)]:
            p = parse_name_shape_dtype_fmt(seg)
            if p:
                parsed_input_rows.append(p)

    input_struct_rows = [x for x in parsed_input_rows if x[0] != "输出"]
    leaked_output_rows = [x for x in parsed_input_rows if x[0] == "输出"]
    has_output_key = any(kk == "算子输出" for _, (kk, _) in ordered_items)

    if parsed_input_rows:
        def html_escape(text: str) -> str:
            return (
                text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

        out = [title, "", "<table>", "  <thead>", "    <tr><th>项目</th><th>值</th></tr>", "  </thead>", "  <tbody>"]
        for _, (kk, vv) in ordered_items:
            if kk in {"算子输入", "算子输入输出"}:
                emit_rows = input_struct_rows if input_struct_rows else parsed_input_rows
                out.append("    <tr>")
                out.append(f"      <td>{html_escape(kk)}</td>")
                out.append("      <td>")
                out.append("        <table>")
                out.append("          <thead><tr><th>name</th><th>shape</th><th>data type</th><th>format</th></tr></thead>")
                out.append("          <tbody>")
                for _, n0, s0, d0, f0 in emit_rows:
                    out.append(
                        "            <tr>"
                        f"<td>{html_escape(n0)}</td>"
                        f"<td>{html_escape(s0)}</td>"
                        f"<td>{html_escape(d0)}</td>"
                        f"<td>{html_escape(f0)}</td>"
                        "</tr>"
                    )
                out.append("          </tbody>")
                out.append("        </table>")
                out.append("      </td>")
                out.append("    </tr>")
                if leaked_output_rows and not has_output_key:
                    leaked_vals = [
                        f"{n0} {s0} {d0} {f0}" for _, n0, s0, d0, f0 in leaked_output_rows
                    ]
                    out.append(
                        "    <tr>"
                        f"<td>{html_escape('算子输出')}</td>"
                        f"<td>{html_escape('<br>'.join(leaked_vals)).replace('&lt;br&gt;', '<br>')}</td>"
                        "</tr>"
                    )
                continue

            if kk == "使用的主要接口":
                segs = [squeeze_ws(x) for x in vv.split("<br>") if squeeze_ws(x)]
                val = "<br>".join(html_escape(x) for x in segs) if segs else ""
                out.append(f"    <tr><td>{html_escape(kk)}</td><td>{val}</td></tr>")
                continue

            p_out = parse_name_shape_dtype_fmt(vv)
            if p_out:
                _, n0, s0, d0, f0 = p_out
                val = f"{n0} {s0} {d0} {f0}"
                out.append(f"    <tr><td>{html_escape(kk)}</td><td>{html_escape(val)}</td></tr>")
            else:
                val = html_escape(vv).replace("&lt;br&gt;", "<br>")
                out.append(f"    <tr><td>{html_escape(kk)}</td><td>{val}</td></tr>")
        out.extend(["  </tbody>", "</table>", ""])
        return out, k

    out = [title, "", "| 项目 | 值 |", "| --- | --- |"]
    for _, (kk, vv) in ordered_items:
        out.append(f"| {escape_table_cell(kk)} | {escape_table_cell(vv)} |")
    out.append("")
    return out, k


def parse_operator_shape_spec_table_block(lines: list[str], start: int) -> tuple[list[str] | None, int]:
    """
    解析“表x-y 算子规格”这类表（非“算子设计规格”）：
    常见列为：输入 | Shape | Data type | Format
    """
    title = lines[start].strip()
    if ("算子规格" not in title) or ("设计规格" in title):
        return None, start + 1

    header = ["输入", "Shape", "Data type", "Format"]
    rows: list[list[str]] = []

    k = start + 1
    while k < len(lines):
        cur = lines[k]
        t = cur.strip()
        if not t:
            k += 1
            continue
        if cur.startswith("## "):
            break
        if TABLE_TITLE_START_RE.match(t):
            break
        if cur.startswith("### "):
            break

        if cur.startswith("```"):
            k += 1
            while k < len(lines) and not lines[k].startswith("```"):
                inner = squeeze_ws(lines[k])
                if inner:
                    cells = [squeeze_ws(x) for x in inner.split(" ") if squeeze_ws(x)]
                    if len(cells) >= 4:
                        line_for_parse = "| " + " | ".join(cells[:3] + [" ".join(cells[3:])]) + " |"
                        cells2 = [squeeze_ws(x) for x in line_for_parse.strip("|").split("|")]
                        cells2 = [c for c in cells2 if c != ""]
                        if cells2:
                            ncol = len(header)
                            if len(cells2) > ncol:
                                cells2 = cells2[: ncol - 1] + [" ".join(cells2[ncol - 1 :])]
                            elif len(cells2) < ncol:
                                cells2 = cells2 + [""] * (ncol - len(cells2))
                            if ncol == 4:
                                m_dt = re.match(r"^(?P<dtype>[A-Za-z0-9_]+)\s+(?P<fmt>[A-Za-z0-9_]+)$", cells2[3])
                                if m_dt and re.search(r"\d", cells2[2]):
                                    cells2 = [
                                        cells2[0],
                                        squeeze_ws(f"{cells2[1]} {cells2[2]}"),
                                        m_dt.group("dtype"),
                                        m_dt.group("fmt"),
                                    ]
                            rows.append(cells2)
                k += 1
            if k < len(lines) and lines[k].startswith("```"):
                k += 1
            continue

        line_for_parse = t
        if t.startswith("|") and t.endswith("|"):
            cells = [squeeze_ws(x) for x in t.strip("|").split("|")]
            if cells and all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                k += 1
                continue
        else:
            if rows and normalize_list_item(t) is not None:
                break
            if rows and (looks_like_plain_paragraph(t) or is_code_like(t, prev_is_code=False)):
                break
            # OCR 行内表：按空格尝试拆分为 4 列
            cells = [squeeze_ws(x) for x in t.split(" ") if squeeze_ws(x)]
            if len(cells) >= 4:
                line_for_parse = "| " + " | ".join(cells[:3] + [" ".join(cells[3:])]) + " |"
            else:
                k += 1
                continue

        cells = [squeeze_ws(x) for x in line_for_parse.strip("|").split("|")]
        cells = [c for c in cells if c != ""]
        if not cells:
            k += 1
            continue

        lower = " ".join(cells).lower()
        if ("shape" in lower and "data" in lower and "format" in lower):
            header = []
            i = 0
            while i < len(cells):
                if i + 1 < len(cells) and cells[i].lower() == "data" and cells[i + 1].lower() == "type":
                    header.append("Data type")
                    i += 2
                else:
                    header.append(cells[i])
                    i += 1
            # 形如：Data | type Format
            if len(header) == 4 and header[2].lower() == "data" and header[3].lower().startswith("type"):
                header[2] = "Data type"
                tail = squeeze_ws(header[3][len("type") :])
                header[3] = tail if tail else "Format"
            if len(header) >= 1 and header[0].lower() == "input":
                header[0] = "输入"
            if len(header) >= 2 and header[1].lower() == "shape":
                header[1] = "Shape"
            if len(header) >= 3 and header[2].lower() in {"data type", "datatype"}:
                header[2] = "Data type"
            if len(header) >= 4 and header[3].lower() == "format":
                header[3] = "Format"
            k += 1
            continue

        ncol = len(header)
        if len(cells) > ncol:
            cells = cells[: ncol - 1] + [" ".join(cells[ncol - 1 :])]
        elif len(cells) < ncol:
            cells = cells + [""] * (ncol - len(cells))
        if ncol == 4:
            # 形如：a | 128, | 1024 | float16 ND
            m_dt = re.match(r"^(?P<dtype>[A-Za-z0-9_]+)\s+(?P<fmt>[A-Za-z0-9_]+)$", cells[3])
            if m_dt and re.search(r"\d", cells[2]):
                cells = [
                    cells[0],
                    squeeze_ws(f"{cells[1]} {cells[2]}"),
                    m_dt.group("dtype"),
                    m_dt.group("fmt"),
                ]
        rows.append(cells)
        k += 1

    if len(rows) < 1:
        return None, start + 1

    out = [title, "", "| " + " | ".join(escape_table_cell(x) for x in header) + " |"]
    out.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in rows:
        out.append("| " + " | ".join(escape_table_cell(x) for x in r) + " |")
    out.append("")
    return out, k


def clean_api_table_text(text: str) -> str:
    t = squeeze_ws(text)
    if not t:
        return t
    t = API_TABLE_HEADER_NOISE_RE.sub("", t)
    t = t.replace("接口分类 接口名称 备注", "")
    t = t.replace("接口分类 接口名称", "")
    t = t.replace("系统变量访 问", "系统变量访问")
    return squeeze_ws(t)


def looks_like_api_table_row_start(text: str) -> bool:
    return bool(API_TABLE_ROW_START_RE.match(squeeze_ws(text)))


def add_api_class_tail(api_class: str, tail: str) -> str:
    c = squeeze_ws(api_class)
    t = squeeze_ws(tail)
    if not t or t in c:
        return c
    if t in {"算法", "容器函数", "类型特性", "type_traits"} and not c.endswith(">"):
        if c.endswith(("C++标准库", "模板库函数")):
            return f"{c} > {t}".strip()
    return f"{c} {t}".strip()


def join_api_name_parts(left: str, right: str) -> str:
    l = squeeze_ws(left)
    r = squeeze_ws(right)
    if not l:
        return r
    if not r:
        return l
    if l.endswith(("、", "，", ",", "/", "(", "（")):
        return f"{l}{r}"
    return join_prose(l, r)


def split_api_class_and_name(text: str) -> tuple[str | None, str | None]:
    t = clean_api_table_text(text)
    if not t:
        return None, None
    for m in API_NAME_TOKEN_RE.finditer(t):
        token = m.group(0)
        prefix = t[: m.start()].strip()
        if prefix.count(">") < 1:
            continue
        if not re.search(r"[\u4e00-\u9fff]", prefix):
            continue
        if token in {"API", "Utils", "Atlas", "Core"}:
            continue
        return squeeze_ws(prefix), squeeze_ws(t[m.start() :])
    return None, None


def extract_api_class_tail_prefix(text: str) -> tuple[str, str]:
    t = squeeze_ws(text)
    for w in sorted(API_CLASS_TAIL_WORDS, key=len, reverse=True):
        if t.startswith(w):
            return w, squeeze_ws(t[len(w) :])
    return "", t


def looks_like_api_remark_text(text: str) -> bool:
    t = squeeze_ws(text)
    if not t:
        return False
    if t.startswith(">") or "不支持" in t or "TSCM" in t:
        return True
    if re.search(r"[A-Za-z_]", t):
        return False
    return bool(re.search(r"[\u4e00-\u9fff]", t))


def should_arrow_be_class_tail(api_class: str, right: str) -> bool:
    cls = squeeze_ws(api_class)
    r = squeeze_ws(right)
    if not r:
        return False
    tail, _ = extract_api_class_tail_prefix(r)
    if not tail:
        return False
    if "不支持" in r or "TSCM" in r:
        return False
    if cls.startswith(("Utils API >", "高阶API >")):
        return True
    return bool(API_NAME_TOKEN_RE.search(r))


def repair_api_class_and_name(api_class: str, api_name: str) -> tuple[str, str]:
    cls = clean_api_table_text(api_class)
    name = clean_api_table_text(api_name).lstrip("-").strip()
    cls = cls.replace("基础API > 数据搬运增强数据搬运", "基础API > 数据搬运 > 增强数据搬运")

    if cls.endswith("访") and name.startswith("问"):
        cls = f"{cls}问"
        name = name[1:].lstrip()
    cls = cls.replace("系统变量访", "系统变量访问")
    if "系统变量访问" in cls:
        name = re.sub(r"^问\s+", "", name)
        name = re.sub(r"([、,，])\s*问\s+", r"\1", name)

    for w in sorted(API_CLASS_TAIL_WORDS, key=len, reverse=True):
        p = re.compile(rf"([、,，])\s*{re.escape(w)}\s+")
        if p.search(name):
            name = p.sub(r"\1", name, count=1)
            cls = add_api_class_tail(cls, w)
        if name.startswith(f"{w} "):
            name = squeeze_ws(name[len(w) :])
            cls = add_api_class_tail(cls, w)
        if name.endswith(w) and re.search(r"[A-Za-z0-9_)\]）]$", name[: -len(w)]):
            name = name[: -len(w)].rstrip(" 、,，")
            cls = add_api_class_tail(cls, w)
        m = re.match(rf"^([A-Za-z_][A-Za-z0-9_:/()<>+-]*){re.escape(w)}$", name)
        if m:
            name = m.group(1)
            cls = add_api_class_tail(cls, w)

    name = squeeze_ws(name)
    name = re.sub(r"\s+([，。；：！？、）】》])", r"\1", name)
    name = re.sub(r"([（【《])\s+", r"\1", name)
    if (cls == "高阶API >" or cls == "高阶API") and name.startswith("C++标准库 "):
        cls = "高阶API > C++标准库"
        name = squeeze_ws(name[len("C++标准库 ") :])
    if cls == "高阶API > 类型特性" and name.startswith("C++标准库 "):
        cls = "高阶API > C++标准库 > 类型特性"
        name = squeeze_ws(name[len("C++标准库 ") :])
    if cls == "高阶API > 模板库函数 type_traits":
        cls = "高阶API > 模板库函数 > type_traits"
    if cls == "基础API >" and (
        name.startswith("Copy、DataCopyPad")
        or name.startswith("DataCopy")
        or name.startswith("VECIN/")
    ):
        cls = "基础API > 数据搬运"
    if cls.endswith(">") and name == "SetDeqScale":
        cls = add_api_class_tail(cls, "量化设置")
    elif cls.endswith(">"):
        cls = cls.rstrip(">").rstrip()
    cls = cls.replace("系统变量访问问", "系统变量访问")
    return cls, name


def parse_api_scope_row(line: str, with_remark: bool) -> list[str] | None:
    t = clean_api_table_text(line)
    api_class, rest = split_api_class_and_name(t)
    if not api_class or not rest:
        return None

    api_name = rest
    remark = "-" if with_remark else ""

    if with_remark:
        if "不支持" in api_class:
            before, rem = api_class.split("不支持", 1)
            api_class = squeeze_ws(before)
            remark = f"不支持{squeeze_ws(rem)}" if squeeze_ws(rem) else "不支持"

        arrow = re.search(r"\s*->\s*", rest)
        if arrow:
            api_name = squeeze_ws(rest[: arrow.start()])
            right = clean_api_table_text(rest[arrow.end() :])
            if right:
                if should_arrow_be_class_tail(api_class, right):
                    tail, tail_rest = extract_api_class_tail_prefix(right)
                    api_class = add_api_class_tail(api_class, tail)
                    if tail_rest:
                        api_name = join_api_name_parts(api_name, tail_rest)
                    remark = "-"
                else:
                    remark = right
        else:
            m = re.search(r"\s+-(?!>)\s*", rest)
            if m:
                left = squeeze_ws(rest[: m.start()])
                right = clean_api_table_text(rest[m.end() :])
                api_name = left
                if not right:
                    remark = "-"
                else:
                    tail, tail_rest = extract_api_class_tail_prefix(right)
                    if tail:
                        api_class = add_api_class_tail(api_class, tail)
                        if tail_rest:
                            api_name = join_api_name_parts(api_name, tail_rest)
                        remark = "-"
                    elif looks_like_api_remark_text(right):
                        remark = right
                    else:
                        api_name = join_api_name_parts(api_name, right)
                        remark = "-"

    api_class, api_name = repair_api_class_and_name(api_class, api_name)
    if not api_class or not api_name:
        return None
    if with_remark:
        return [api_class, api_name, clean_api_table_text(remark) or "-"]
    return [api_class, api_name]


def parse_api_scope_table_block(
    lines: list[str], start: int, with_remark: bool
) -> tuple[list[str] | None, int]:
    j = start + 1
    while j < len(lines) and lines[j].strip() == "":
        j += 1

    rows: list[list[str]] = []
    pending_tail = ""
    trailing_table_title = ""
    k = j
    while k < len(lines):
        stop_after_current = False
        cur = lines[k]
        t = cur.strip()
        if cur.startswith("```"):
            break
        if cur.startswith("## "):
            break
        if TABLE_TITLE_START_RE.match(t):
            break
        if cur.startswith("### "):
            heading_text = clean_api_table_text(cur[4:].strip().rstrip("-").strip())
            if not heading_text:
                k += 1
                continue
            if heading_text in API_CLASS_TAIL_WORDS:
                pending_tail = heading_text
                k += 1
                continue
            if heading_text.startswith("接口分类"):
                k += 1
                continue
            break
        if not t:
            k += 1
            continue

        t = clean_api_table_text(t)
        if not t:
            k += 1
            continue
        m_title = TABLE_TITLE_INLINE_RE.search(t)
        if m_title and m_title.start() > 0:
            trailing_table_title = squeeze_ws(t[m_title.start() :])
            t = squeeze_ws(t[: m_title.start()])
            stop_after_current = True
            if not t:
                break

        if looks_like_api_table_row_start(t):
            row = parse_api_scope_row(t, with_remark)
            if row:
                if pending_tail:
                    if row[0].endswith(">"):
                        row[0] = add_api_class_tail(row[0], pending_tail)
                    pending_tail = ""
                rows.append(row)
            k += 1
            if stop_after_current:
                break
            continue

        if with_remark and rows and re.match(
            r"^(基础数据搬运|增强数据搬运|随路转换ND2NZ搬运|随路转换NZ2ND搬运|随路量化激活搬运|Copy、DataCopyPad、)",
            t,
        ):
            base = rows[-1][0]
            m = re.match(r"^(.*>)\s*", base)
            base_prefix = m.group(1) if m else base
            synthetic = f"{base_prefix} {t}"
            row = parse_api_scope_row(synthetic, with_remark)
            if row:
                if pending_tail:
                    if row[0].endswith(">"):
                        row[0] = add_api_class_tail(row[0], pending_tail)
                    pending_tail = ""
                rows.append(row)
            k += 1
            if stop_after_current:
                break
            continue

        if not rows:
            k += 1
            if stop_after_current:
                break
            continue

        if with_remark and (t.startswith("- >") or t.startswith("-")):
            extra = squeeze_ws(t.lstrip("-").lstrip(">"))
            if extra:
                rows[-1][2] = append_desc_cell(rows[-1][2], extra)
            k += 1
            if stop_after_current:
                break
            continue

        if with_remark and looks_like_api_remark_text(t):
            rows[-1][2] = append_desc_cell(rows[-1][2], t.lstrip("> ").strip())
        else:
            rows[-1][1] = append_desc_cell(rows[-1][1], t)
        k += 1
        if stop_after_current:
            break

    if len(rows) < 2:
        return None, start + 1

    if with_remark:
        rows = normalize_api_scope_special_rows(rows)

    if with_remark:
        table_lines = [
            "### 接口分类 接口名称 备注",
            "",
            "| 接口分类 | 接口名称 | 备注 |",
            "| --- | --- | --- |",
        ]
        for api_class, api_name, remark in rows:
            table_lines.append(
                f"| {escape_table_cell(api_class)} | {escape_table_cell(api_name)} | {escape_table_cell(remark)} |"
            )
    else:
        table_lines = [
            "### 接口分类 接口名称",
            "",
            "| 接口分类 | 接口名称 |",
            "| --- | --- |",
        ]
        for api_class, api_name in rows:
            table_lines.append(
                f"| {escape_table_cell(api_class)} | {escape_table_cell(api_name)} |"
            )
    table_lines.append("")
    if trailing_table_title:
        table_lines.append(trailing_table_title)
        table_lines.append("")
    return table_lines, k


def parse_api_scope_markdown_table_block(
    lines: list[str], start: int, with_remark: bool
) -> tuple[list[str] | None, int]:
    """
    修复已被错误转成 markdown 的 API 范围表：
    - 行被拆成 “基础API | > | ...”
    - 表内续行被错误落成普通文本
    """
    k = start + 1
    emit_heading = True
    p = start - 1
    while p >= 0 and not lines[p].strip():
        p -= 1
    if p >= 0 and lines[p].strip() in {"### 接口分类 接口名称", "### 接口分类 接口名称 备注"}:
        emit_heading = False
    if k < len(lines) and re.fullmatch(r"\|\s*:?-{3,}:?\s*\|\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)?\|", lines[k].strip()):
        k += 1

    rows: list[list[str]] = []
    trailing_table_title = ""
    while k < len(lines):
        cur = lines[k]
        t = cur.strip()
        if not t:
            k += 1
            continue
        if cur.startswith("## ") or cur.startswith("```"):
            break
        if TABLE_TITLE_START_RE.match(t):
            break
        if cur.startswith("### "):
            payload = clean_api_table_text(cur[4:].strip().rstrip("-").strip())
            if payload.startswith("接口分类"):
                k += 1
                continue
            if with_remark and payload.startswith("切片数据搬运"):
                base = rows[-1][0] if rows else "基础API > 数据搬运 > DataCopy"
                if "DataCopy" not in base:
                    base = "基础API > 数据搬运 > DataCopy"
                rows.append([base, "切片数据搬运", "-"])
                k += 1
                continue
            break

        m_title = TABLE_TITLE_INLINE_RE.search(t)
        if m_title and m_title.start() > 0:
            trailing_table_title = squeeze_ws(t[m_title.start() :])
            t = squeeze_ws(t[: m_title.start()])
            if not t:
                break

        if t.startswith("|") and t.endswith("|"):
            cells = [squeeze_ws(x) for x in t.strip("|").split("|")]
            if cells and all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                k += 1
                continue

            if with_remark:
                if len(cells) >= 3:
                    c0, c1 = cells[0], cells[1]
                    c2 = " ".join(cells[2:]).strip()
                elif len(cells) == 2:
                    c0, c1, c2 = cells[0], cells[1], "-"
                else:
                    k += 1
                    continue

                if c1 in {">", "->"}:
                    row = parse_api_scope_row(f"{c0} > {c2}", with_remark=True)
                else:
                    cls, name = repair_api_class_and_name(c0, c1)
                    row = [cls or c0, name or c1, clean_api_table_text(c2) or "-"]

                if row:
                    rows.append(row)
                elif rows:
                    rows[-1][2] = append_desc_cell(rows[-1][2], clean_api_table_text(" ".join(cells)))
                k += 1
                continue

            if len(cells) >= 2:
                c0, c1 = cells[0], " ".join(cells[1:]).strip()
                row = parse_api_scope_row(f"{c0} {c1}", with_remark=False)
                if row:
                    rows.append(row)
                else:
                    cls, name = repair_api_class_and_name(c0, c1)
                    if cls and name:
                        rows.append([cls, name])
            k += 1
            continue

        parsed = parse_api_scope_row(t, with_remark=with_remark)
        if parsed:
            rows.append(parsed)
            k += 1
            continue

        if not rows:
            break

        if with_remark:
            if t.startswith("-"):
                extra = squeeze_ws(t.lstrip("-").lstrip(">"))
                if extra:
                    rows[-1][2] = append_desc_cell(rows[-1][2], extra)
            elif t.startswith("随路转换ND2NZ搬运"):
                base = rows[-1][0] if rows else "基础API > 数据搬运 > DataCopy"
                if "DataCopy" not in base:
                    base = "基础API > 数据搬运 > DataCopy"
                rows.append([base, "随路转换ND2NZ搬运", t])
            elif looks_like_api_remark_text(t):
                rows[-1][2] = append_desc_cell(rows[-1][2], t.lstrip("> ").strip())
            else:
                if looks_like_plain_paragraph(t):
                    break
                rows[-1][1] = append_desc_cell(rows[-1][1], t)
        else:
            if looks_like_plain_paragraph(t):
                break
            rows[-1][1] = append_desc_cell(rows[-1][1], t)
        k += 1

    if not rows:
        return None, start + 1

    if with_remark:
        rows = normalize_api_scope_special_rows(rows)
        out = []
        if emit_heading:
            out.extend(["### 接口分类 接口名称 备注", ""])
        out.extend(["| 接口分类 | 接口名称 | 备注 |", "| --- | --- | --- |"])
        for api_class, api_name, remark in rows:
            out.append(
                f"| {escape_table_cell(api_class)} | {escape_table_cell(api_name)} | {escape_table_cell(remark)} |"
            )
        out.append("")
        if trailing_table_title:
            out.append(trailing_table_title)
            out.append("")
        return out, k

    out = []
    if emit_heading:
        out.extend(["### 接口分类 接口名称", ""])
    out.extend(["| 接口分类 | 接口名称 |", "| --- | --- |"])
    for api_class, api_name in rows:
        out.append(f"| {escape_table_cell(api_class)} | {escape_table_cell(api_name)} |")
    out.append("")
    if trailing_table_title:
        out.append(trailing_table_title)
        out.append("")
    return out, k


def normalize_api_scope_special_rows(rows: list[list[str]]) -> list[list[str]]:
    def normalize_move_remark(text: str) -> str:
        t = squeeze_ws(text)
        t = re.sub(r"(?<!-)>\s*TSCM", "-> TSCM", t)
        t = t.replace("通路的数<br>据搬运。", "通路的数据搬运。")
        t = t.replace("通路的数 据搬运。", "通路的数据搬运。")
        t = t.replace("通路的数数据搬运。", "通路的数据搬运。")
        t = t.replace("通路的数据数据搬运。", "通路的数据搬运。")
        # 去除重复的 TSCM 备注行
        t = t.replace("-> TSCM<br>通路的数据搬运。", "-> TSCM通路的数据搬运。")
        t = t.replace("-> TSCM通路的数据搬运。<br>-> TSCM通路的数据搬运。", "-> TSCM通路的数据搬运。")
        return t

    def split_move_variant_remark(text: str) -> tuple[str, str]:
        t = normalize_move_remark(text)
        marker = "增强数据搬运不支持"
        if marker not in t:
            return t, ""
        left, right = t.split(marker, 1)
        left = left.rstrip("<br>")
        right_full = f"不支持{right}".strip()
        return normalize_move_remark(left), normalize_move_remark(right_full)

    out: list[list[str]] = []
    inserted_slice_row = False

    for cls, name, remark in rows:
        cls = squeeze_ws(cls)
        name = squeeze_ws(name)
        remark = normalize_move_remark(remark)

        if cls == "-" and name.startswith("> TSCM"):
            if out:
                extra = re.sub(r"^>\s*TSCM", "-> TSCM", name)
                merged = extra
                if remark and remark != "-":
                    merged = append_desc_cell(merged, remark)
                out[-1][2] = append_desc_cell(out[-1][2], merged)
            continue

        m_move = re.fullmatch(r"基础API > 数据搬运 > (基础数据搬运|增强数据搬运)", cls)
        if m_move and "VECIN/" in name:
            out.append(
                [
                    "基础API > 数据搬运 > DataCopy",
                    m_move.group(1),
                    "不支持VECIN/VECCALC/VECOUT<br>-> TSCM通路的数据搬运。",
                ]
            )
            continue

        # OCR 常把“增强数据搬运”并入“基础数据搬运”的备注里，这里强制拆成两行
        if cls == "基础API > 数据搬运 > DataCopy" and name == "基础数据搬运":
            base_remark, enhanced_remark = split_move_variant_remark(remark)
            out.append([cls, name, base_remark or "-"])
            if enhanced_remark:
                out.append([cls, "增强数据搬运", enhanced_remark])
            continue

        if cls == "基础API > 数据搬运 > DataCopy" and name.startswith("随路转换ND2NZ搬运"):
            if ("随路转换NZ2ND搬运" in remark) or ("随路量化激活搬运" in remark):
                out.append(
                    [
                        cls,
                        "随路转换ND2NZ搬运<br>随路转换NZ2ND搬运<br>随路量化激活搬运",
                        "不支持VECIN/VECCALC/VECOUT<br>-> TSCM通路的数据搬运。",
                    ]
                )
                continue

        if cls == "基础API > 数据搬运 > 随路转换":
            if not inserted_slice_row:
                out.append(["基础API > 数据搬运 > DataCopy", "切片数据搬运", "-"])
                inserted_slice_row = True
            out.append(
                [
                    "基础API > 数据搬运 > DataCopy",
                    "随路转换ND2NZ搬运<br>随路转换NZ2ND搬运<br>随路量化激活搬运",
                    "不支持VECIN/VECCALC/VECOUT<br>-> TSCM通路的数据搬运。",
                ]
            )
            continue

        if cls == "基础API > 数据搬运 >" and (
            name.startswith("Copy、DataCopyPad")
            or name.startswith("DataCopy")
            or name.startswith("VECIN/")
        ):
            cls = "基础API > 数据搬运"

        out.append([cls, name, remark])
        if cls.startswith("基础API > 数据搬运") and name == "切片数据搬运":
            inserted_slice_row = True

    return out


def render_section_markdown(cur_title: str, raw_lines: list[str]) -> str:
    _, heading_title = split_sec_no(cur_title)
    title_norms = {
        norm_for_match(cur_title),
        norm_for_match(strip_sec_no(cur_title)),
        norm_for_match(heading_title),
    }

    blocks: list[str] = [f"## {heading_title}"]
    para_buf: list[tuple[str, int]] = []
    code_buf: list[str] = []
    prev_raw_blank = True
    pending_blank_after_code = False

    def append_blank():
        if blocks and blocks[-1] != "":
            blocks.append("")

    def flush_para():
        nonlocal para_buf
        if not para_buf:
            return

        merged: list[str] = []
        curr = ""
        curr_indent = 0
        for line, indent in para_buf:
            text = normalize_prose_line(line)
            if not text:
                continue
            if not curr:
                curr = text
                curr_indent = indent
                continue
            if should_start_new_paragraph(curr, text, curr_indent, indent):
                merged.append(curr.strip())
                curr = text
            else:
                curr = join_prose(curr, text)
            curr_indent = indent
        if curr:
            merged.append(curr.strip())

        for p in merged:
            if p:
                blocks.append(p)
                append_blank()

        para_buf = []

    def flush_code():
        nonlocal code_buf
        if not code_buf:
            return
        lines = normalize_code_lines(code_buf)
        if lines:
            blocks.append("```cpp")
            blocks.extend(lines)
            blocks.append("```")
            append_blank()
        code_buf = []

    for idx, raw in enumerate(raw_lines):
        raw = raw.replace("\u00a0", " ").rstrip("\n")
        stripped = raw.strip()
        indent = len(raw) - len(raw.lstrip(" "))
        next_nonempty: str | None = None
        for j in range(idx + 1, len(raw_lines)):
            cand = raw_lines[j].replace("\u00a0", " ").strip()
            if cand:
                next_nonempty = cand
                break

        if not stripped:
            if code_buf:
                pending_blank_after_code = True
                prev_raw_blank = True
                continue
            flush_para()
            append_blank()
            prev_raw_blank = True
            continue

        if pending_blank_after_code:
            if is_code_like(raw, prev_is_code=True):
                code_buf.append("")
            else:
                flush_code()
            pending_blank_after_code = False

        if norm_for_match(stripped) in title_norms:
            prev_raw_blank = False
            continue

        # 代码块中若出现明显正文句子（非注释/非语法行），先结束代码块，避免吞正文
        if code_buf and should_break_code_block_for_prose(raw):
            flush_code()

        # 代码块一旦开始，优先用“代码连续性”判定，避免被列表/标题规则打断
        if code_buf and (
            looks_like_code_bridge_line(raw, code_buf[-1], next_nonempty)
            or is_code_like(raw, prev_is_code=True)
            or should_attach_code_comment_continuation(stripped, code_buf[-1])
            or should_attach_hash_comment_continuation(stripped, code_buf[-1])
            or should_attach_inline_comment_continuation(stripped, code_buf[-1])
        ):
            flush_para()
            code_buf.append(raw)
            prev_raw_blank = False
            continue

        list_item = normalize_list_item(stripped)
        if (prev_raw_blank or SEC_NO_RE.match(stripped) or is_strong_heading(stripped)) and looks_like_heading(stripped):
            flush_para()
            flush_code()
            blocks.append(f"### {normalize_heading_text(stripped)}")
            append_blank()
            prev_raw_blank = False
            continue
        if list_item is not None:
            flush_para()
            flush_code()
            blocks.append(list_item)
            prev_raw_blank = False
            continue
        if is_code_like(raw, prev_is_code=bool(code_buf)):
            flush_para()
            code_buf.append(raw)
            prev_raw_blank = False
            continue
        if code_buf and looks_like_code_bridge_line(raw, code_buf[-1], next_nonempty):
            flush_para()
            code_buf.append(raw)
            prev_raw_blank = False
            continue
        if code_buf:
            flush_code()

        para_buf.append((raw, indent))
        prev_raw_blank = False

    flush_para()
    flush_code()

    while blocks and not blocks[-1]:
        blocks.pop()

    compact: list[str] = []
    for line in blocks:
        if line == "" and compact and compact[-1] == "":
            continue
        compact.append(line)

    compact = split_embedded_table_titles(compact)
    compact = postprocess_option_table_blocks(compact)
    compact = postprocess_common_table_blocks(compact)
    compact = postprocess_split_embedded_table_titles(compact)
    compact = postprocess_broken_api_markdown_tables(compact)
    compact = postprocess_param_table_blocks(compact)
    compact = postprocess_broken_param_markdown_tables(compact)
    compact = postprocess_promote_paramname_desc_tables(compact)
    compact = postprocess_split_embedded_param_rows(compact)
    compact = postprocess_detach_prose_from_param_tables(compact)
    compact = postprocess_rebalance_two_col_constraint_rows(compact)
    compact = postprocess_recover_constraint_table_continuations(compact)
    compact = postprocess_call_example_code_blocks(compact)
    compact = postprocess_code_block_prose_leaks(compact)
    compact = postprocess_merge_comment_prose_between_code_blocks(compact)
    compact = postprocess_merge_adjacent_code_blocks(compact)

    final_compact: list[str] = []
    for line in compact:
        if line == "" and final_compact and final_compact[-1] == "":
            continue
        final_compact.append(line)

    return "\n".join(final_compact) + "\n"

def export_sections_to_markdown(pdf_path: Path, out_dir: Path):
    """
    按 PDF 目录书签切分为 md，并在同一页内按“小节标题行”进一步切分。
    - 默认：导出 level>=2 的叶子节点（更细）
    - 切分逻辑：从本节标题行开始收集，到下节标题行之前停止（可跨页）
    """
    doc = fitz.open(pdf_path)
    toc = doc.get_toc(simple=True)
    if not toc:
        doc.close()
        raise RuntimeError("PDF 没有目录/书签（toc 为空），无法按章节拆分。")

    # ---------- 1) 清洗 toc ----------
    cleaned = []
    for lvl, title, page1 in toc:
        t = sanitize_title(title)
        if not t or t in {"目录", "Contents", "Table of Contents"}:
            continue
        cleaned.append((lvl, t, page1))

    if not cleaned:
        doc.close()
        raise RuntimeError("清洗后 toc 为空，无法拆分。")

    # ---------- 2) 标记是否有子节点 ----------
    has_child = [False] * len(cleaned)
    for i in range(len(cleaned) - 1):
        if cleaned[i + 1][0] > cleaned[i][0]:
            has_child[i] = True

    # ---------- 3) 选择要导出的条目：叶子 + level>=2 ----------
    MIN_LEVEL = 2
    LEAF_ONLY = True

    selected = []
    for i, (lvl, title, page1) in enumerate(cleaned):
        if lvl < MIN_LEVEL:
            continue
        if LEAF_ONLY and has_child[i]:
            continue
        selected.append((i, lvl, title, page1))

    if not selected:
        # 兜底：导出 level>=2 全部
        for i, (lvl, title, page1) in enumerate(cleaned):
            if lvl >= MIN_LEVEL:
                selected.append((i, lvl, title, page1))

    # ---------- 4) 对每节做“页内切分” ----------
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, (toc_idx, lvl, cur_title, cur_page1) in enumerate(selected):
        cur_start_page = max(0, cur_page1 - 1)

        # 边界按“清洗后的 toc 中，下一个同级/上级标题”确定，避免把下一大节带进当前文件。
        next_title = None
        next_start_page = doc.page_count  # default end
        for j in range(toc_idx + 1, len(cleaned)):
            n_lvl, n_title, n_page1 = cleaned[j]
            if n_lvl <= lvl:
                next_title = n_title
                next_start_page = max(0, n_page1 - 1)
                break

        # 文件名：0000_1.2 环境准备.md（保留中文）
        sec_no, name_title = split_sec_no(cur_title)

        safe_name = safe_filename_cn(name_title)
        prefix = f"{sec_no} " if sec_no else ""
        filename = f"{idx:04d}_{prefix}{safe_name}.md"
        out_path = out_dir / filename

        started = False
        section_lines: list[str] = []

        # 计算页范围：如果下一节起始页在后面，就只读到 next_start_page-1；
        # 只有当 next_start_page == cur_start_page 才做同页切分。
        end_page = doc.page_count - 1 if next_title is None else max(cur_start_page, next_start_page - 1)
        boundary_page = next_start_page if next_title is not None else None


        def page_lines(pno: int) -> list[str]:
            page = doc.load_page(pno)
    # 你如果有 clip/过滤页眉页脚，就放在这里
            return page.get_text("text", sort=True).splitlines()

        boundary_has_next = False
        if next_title is not None:
            # 先在 next_start_page 探测是否存在 next_title
            blines = page_lines(boundary_page)
            boundary_stop_pos = find_title_line_index(blines, next_title)
            if boundary_stop_pos is not None and boundary_stop_pos > 0:
                boundary_has_next = True
                # ✅ 只有确认边界页存在 next_title，才把边界页包含进来
                end_page = max(end_page, boundary_page)

        for pno in range(cur_start_page, end_page + 1):
    # 用 sort=True 更稳，正文不容易丢
            #lines = doc.load_page(pno).get_text("text", sort=True).splitlines()
            page = doc.load_page(pno)
            rect = page.rect  # 页面尺寸
            # 保留中间 8%~92% 的高度（你可以调整比例）
            clip = fitz.Rect(rect.x0, rect.y0 + rect.height * 0.08,
                             rect.x1, rect.y0 + rect.height * 0.92)

            text = page.get_text("text", sort=True, clip=clip)
            lines = text.splitlines()

            if not started:
                pos = find_title_line_index(lines, cur_title)
                if pos is None:
                    continue
                lines = lines[pos:]
                started = True

    # ✅ 只有“下一节也在同一页”才在本页里找 stop
            
            if boundary_has_next and next_title and (pno == boundary_page):
                # lines 是当前页（且从 cur_title 起可能已裁过）
                # 注意：如果这一页不是起始页，lines 没裁掉前面；这里用整页 lines 来截断
                lines_full = lines
                stop_pos = find_title_line_index(lines_full, next_title)
                if stop_pos is not None and stop_pos > 0:
                    lines = lines_full[:stop_pos]
                    if lines:
                        section_lines.extend(lines)
                    break


            if lines:
                section_lines.extend(lines)

        # 如果整个循环都没开始，给个提示（避免空文件悄悄生成）
        if not started:
            content = f"## {name_title}\n\n> ⚠️ 未能在正文中定位到该小节标题（PDF 文本抽取可能丢失标题行）。\n"
            out_path.write_text(content, "utf-8")
            continue

        rendered = render_section_markdown(cur_title, section_lines)
        # 再做一次兜底修复，避免个别章节在前序流程后仍残留坏表格
        rendered_lines = rendered.splitlines()
        rendered_lines = postprocess_broken_api_markdown_tables(rendered_lines)
        rendered_lines = postprocess_promote_paramname_desc_tables(rendered_lines)
        rendered_lines = postprocess_split_embedded_param_rows(rendered_lines)
        rendered_lines = postprocess_detach_prose_from_param_tables(rendered_lines)
        rendered_lines = postprocess_rebalance_two_col_constraint_rows(rendered_lines)
        rendered_lines = postprocess_recover_constraint_table_continuations(rendered_lines)
        out_path.write_text("\n".join(rendered_lines) + "\n", "utf-8")

    doc.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-page", required=True, help="CANN 下载页 URL")
    parser.add_argument("--target-title", default="Ascend C算子开发")
    parser.add_argument("--force", action="store_true", help="force re-export even if unchanged")
    args = parser.parse_args()

    global DOWNLOAD_PAGE, TARGET_TITLE
    DOWNLOAD_PAGE = args.download_page
    TARGET_TITLE = args.target_title

    state = load_state()

    info = get_target_pdf_url_via_playwright(TARGET_TITLE)
    pdf_url = info["pdf_url"]
    label = info.get("label", "")

    # force 则不提前退出
    if (not args.force) and state.get("pdf_url") == pdf_url:
        print("No update: pdf_url unchanged.")
        return



    # 下载到临时文件，再计算 hash
    pdf_name = slugify(label or "ascendc_doc", separator="_") or "ascendc_doc"
    pdf_path = DOWNLOAD_DIR / f"{pdf_name}.pdf"

    print(f"Downloading: {pdf_url}")
    download_pdf(pdf_url, pdf_path)

    file_hash = sha256_file(pdf_path)
    if (not args.force) and state.get("sha256") == file_hash:
        print("No update: sha256 unchanged.")
        state["pdf_url"] = pdf_url  # URL可能变但内容未变，也记录一下
        save_state(state)
        return

    # 清空旧输出（可选：也可以按版本输出到子目录）
    for f in OUT_DIR.glob("*.md"):
        f.unlink()

    print("Exporting sections to markdown...")
    export_sections_to_markdown(pdf_path, OUT_DIR)

    state.update({
        "pdf_url": pdf_url,
        "label": label,
        "sha256": file_hash,
    })
    save_state(state)

    print("Done.")


if __name__ == "__main__":
    main()
