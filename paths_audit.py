#!/usr/bin/env python3
import os, re, csv, ast, sys, json
from pathlib import Path
from collections import defaultdict, deque

REPO_ROOT = Path.cwd()

# what to scan for usages
INCLUDE_EXT = {
    ".py",".ipynb",".sh",".bash",".zsh",".env",".json",".yml",".yaml",".toml",
    ".ini",".cfg",".txt",".md",".csv",".tsv"
}
EXCLUDE_DIRS = {".git",".hg",".svn",".mypy_cache","__pycache__",".vscode",".idea",
                ".venv","venv","env","node_modules",".vscode-server",".cache",
                "data","results","checkpoints",".ruff_cache",".pytest_cache"}

# ---------- helpers ----------
def iter_files(root: Path):
    for dp, dns, fns in os.walk(root):
        # prune
        dns[:] = [d for d in dns if d not in EXCLUDE_DIRS]
        for fn in fns:
            p = Path(dp) / fn
            if p.suffix.lower() in INCLUDE_EXT or p.name in (".env", "paths.py"):
                yield p

def read_text_safely(p: Path) -> str:
    try:
        return p.read_text(errors="ignore")
    except Exception:
        return ""

# ---------- .env parsing & cycle detection ----------
ENV_LINE = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)=(.*)$')

def parse_env_file(env_path: Path):
    out = {}
    for raw in read_text_safely(env_path).splitlines():
        s = raw.strip()
        if not s or s.startswith("#"): continue
        m = ENV_LINE.match(s)
        if not m: continue
        k, v = m.group(1), m.group(2).strip()
        if len(v)>=2 and v[0]==v[-1] and v[0] in ('"',"'"):
            v = v[1:-1]
        out[k] = v
    return out

VAR_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")

def env_deps(value: str):
    deps = set()
    for a,b in VAR_REF.findall(value or ""):
        deps.add(a or b)
    return deps

def detect_cycles(graph: dict):
    # graph: node -> set(deps)
    color = {}  # white=None, gray=1, black=2
    stack, cycles = [], []

    def dfs(u):
        color[u] = 1
        stack.append(u)
        for v in graph.get(u, ()):
            if color.get(v) == 0 or color.get(v) is None:
                dfs(v)
            elif color.get(v) == 1:
                # cycle
                if v in stack:
                    i = stack.index(v)
                    cycles.append(stack[i:] + [v])
        color[u] = 2
        stack.pop()

    for n in graph:
        if color.get(n) in (None,0):
            dfs(n)
    return cycles

# ---------- paths.py parsing & dependency graph ----------
class PathExprEval(ast.NodeVisitor):
    # allow common constructs: Path, os.path.join, simple +, f-strings, division for Path, os.getenv
    ALLOWED_FUNCS = {
        ("Path",): Path,
        ("os","path","join"): os.path.join,
        ("os","getenv"): lambda k, default=None: os.getenv(k, default),
    }
    def __init__(self, names, env_map):
        self.names = names
        self.env_map = env_map

    def visit_Constant(self, node): return node.value
    def visit_Str(self, node): return node.s
    def visit_Name(self, node):
        if node.id in self.names:
            return self.names[node.id]
        # allow REPO_ROOT symbol if defined upstream by caller
        if node.id == "REPO_ROOT":
            return str(REPO_ROOT)
        raise ValueError(f"Name {node.id} not available")
    def visit_JoinedStr(self, node):
        parts = []
        for v in node.values:
            if isinstance(v, (ast.Str, ast.Constant)):
                parts.append(str(getattr(v, "s", getattr(v, "value", ""))))
            else:
                parts.append(str(self.visit(v)))
        return "".join(parts)
    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Div):
            return str(Path(left) / right)
        raise ValueError("Unsupported binop")
    def visit_Call(self, node):
        # resolve dotted function
        func_key = []
        f = node.func
        while isinstance(f, ast.Attribute):
            func_key.insert(0, f.attr)
            f = f.value
        if isinstance(f, ast.Name):
            func_key.insert(0, f.id)
        tkey = tuple(func_key)
        fn = self.ALLOWED_FUNCS.get(tkey)
        if fn is None:
            raise ValueError(f"Call not allowed: {'.'.join(tkey)}")
        args = [self.visit(a) for a in node.args]
        kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords}
        # special case: os.getenv -> consult env_map first
        if tkey == ("os","getenv") and args:
            return self.env_map.get(args[0], os.getenv(args[0], kwargs.get("default") if "default" in kwargs else (args[1] if len(args)>1 else "")))
        out = fn(*args, **kwargs)
        if isinstance(out, Path): out = str(out)
        return out
    def generic_visit(self, node):
        raise ValueError(f"Unsupported node: {type(node).__name__}")

def paths_py_extract(p: Path, env_map: dict):
    src = read_text_safely(p)
    out, deps = {}, defaultdict(set)
    try:
        tree = ast.parse(src, filename=str(p))
    except Exception:
        return out, deps
    names = {}
    evaluator = PathExprEval(names, env_map)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0], ast.Name):
            var = node.targets[0].id
            # collect dependencies by scanning Name nodes inside value
            dep_names = set()
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Name) and sub.id != var:
                    dep_names.add(sub.id)
            deps[var].update(dep_names)
            # evaluate if possible
            try:
                val = evaluator.visit(node.value)
                if isinstance(val, Path): val = str(val)
                if isinstance(val, str):
                    out[var] = val
                    names[var] = val
            except Exception:
                # leave unevaluated; still record for graph
                pass
    return out, deps

# ---------- usage search ----------
def search_tokens_usages(tokens, files):
    patterns = {t: re.compile(rf"(\b{re.escape(t)}\b|\$\{{{re.escape(t)}\}})") for t in tokens}
    hits = defaultdict(list)
    for p in files:
        if p.suffix.lower() == ".ipynb":  # skip notebooks here too
            continue
        text = read_text_safely(p)
        if not text:
            continue
        for t, pat in patterns.items():
            for i, line in enumerate(text.splitlines(), 1):
                if pat.search(line):
                    hits[t].append((str(p.relative_to(REPO_ROOT)), i, line.strip()))
        for t in tokens:
            if f"os.getenv('{t}'" in text or f'os.getenv("{t}"' in text:
                for i, line in enumerate(text.splitlines(), 1):
                    if f"os.getenv('{t}'" in line or f'os.getenv("{t}"' in line:
                        hits[t].append((str(p.relative_to(REPO_ROOT)), i, line.strip()))
    return hits


LITERAL_PATH_PAT = re.compile(
    r"""(?x)
    (?P<q>["'])       # opening quote
    (?P<p>
       (\.{1,2}/|/)?  # relative ./ or ../ or absolute /
       [^"' \t\n\r]{1,} # path body
    )
    (?P=q)            # closing same quote
    """
)

import base64
LITERAL_PATH_PAT = re.compile(
    r"""(?x)
    (?P<q>["'])               # opening quote
    (?P<p>
       (?:\.{1,2}/|/)[^"' \t\n\r]{1,}  # must start with ./ ../ or / and have at least one char
    )
    (?P=q)                    # closing same quote
    """
)

def looks_like_base64(s: str) -> bool:
    # crude but fast: long, only base64 chars, or common PNG/JPEG magics
    if len(s) < 80:
        return False
    if s.startswith(("iVBORw0KGgo", "/9j/")):  # png/jpeg base64 magics
        return True
    return bool(re.fullmatch(r'[A-Za-z0-9+/=]{80,}', s))

def find_literal_paths(files):
    rows = []
    seen = set()
    MAX_LEN = 300  # avoid absurd strings
    for p in files:
        # Skip notebooks to avoid embedded base64; (optional) you can parse JSON later
        if p.suffix.lower() == ".ipynb":
            continue
        text = read_text_safely(p)
        if not text:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for m in LITERAL_PATH_PAT.finditer(line):
                s = m.group("p")

                # quick filters
                if len(s) > MAX_LEN:
                    continue
                if s.startswith(("http://", "https://", "data:")):
                    continue
                if looks_like_base64(s):
                    continue
                # must contain at least one additional slash beyond the prefix
                if s.count("/") < 1:
                    continue

                key = (str(p), i, s)
                if key in seen:
                    continue
                seen.add(key)

                # Resolve relative to the file's directory
                base = p.parent
                target = Path(s)
                abs_p = target if target.is_absolute() else (base / target)

                # Existence check with guard
                try:
                    exists = abs_p.exists()
                    abs_s = str(abs_p.resolve()) if exists else str(abs_p)
                except OSError:
                    exists = False
                    abs_s = str(abs_p)

                rows.append((str(p.relative_to(REPO_ROOT)), i, s, abs_s, "OK" if exists else "MISSING"))
    return rows


# ---------- main inventory ----------
def main():
    # collect all .env and paths.py files
    env_files = []
    paths_files = []
    all_files = list(iter_files(REPO_ROOT))
    for f in all_files:
        if f.name == ".env":
            env_files.append(f)
        elif f.name == "paths.py":
            paths_files.append(f)

    # index .env: per-file map + per-file dependency graph
    env_records = []  # (file, key, raw, expanded, abs_path, exists)
    env_graphs = {}   # per file: key -> deps
    # resolution: expand ${VAR} using only the same-file env map (no inheritance) and os.environ as fallback
    for ef in env_files:
        env_map = parse_env_file(ef)
        # deps for cycle check
        g = {k: env_deps(v) for k,v in env_map.items()}
        env_graphs[str(ef)] = g
        # expand
        def repl(m):
            key = m.group(1) or m.group(2)
            return env_map.get(key, os.environ.get(key, m.group(0)))
        for k, raw in env_map.items():
            expanded = VAR_REF.sub(repl, raw or "")
            expanded = os.path.expanduser(expanded)
            # compute absolute relative to the env file's directory if not absolute
            abs_path = str((ef.parent / expanded).resolve()) if expanded and not os.path.isabs(expanded) else expanded
            exists = os.path.exists(abs_path) if expanded else False
            env_records.append((str(ef.relative_to(REPO_ROOT)), k, raw, expanded, abs_path, "OK" if exists else "MISSING"))

    # index paths.py: evaluate simple expressions and collect deps
    paths_records = []  # (file, var, raw_or_eval, abs_path, exists, note)
    paths_dep_graphs = {}  # per file: var -> deps
    for pf in paths_files:
        # use that directory's .env for os.getenv
        local_env = parse_env_file(pf.parent/".env") if (pf.parent/".env").exists() else {}
        vals, deps = paths_py_extract(pf, local_env)
        paths_dep_graphs[str(pf)] = deps
        # get raw source to fallback raw strings for unevaluated
        src = read_text_safely(pf)
        try:
            tree = ast.parse(src, filename=str(pf))
        except Exception:
            tree = None
        raw_by_name = {}
        if tree:
            for node in tree.body:
                if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0], ast.Name):
                    var = node.targets[0].id
                    raw_by_name[var] = ast.get_source_segment(src, node.value) or ""
        for var in sorted(set(list(vals.keys()) + list(raw_by_name.keys()))):
            v = vals.get(var)
            note = ""
            if v is None:
                v = raw_by_name.get(var,"")
                note = "unevaluated"
            # expand ~; if relative, resolve against file dir
            v_expanded = os.path.expanduser(v) if isinstance(v, str) else v
            if isinstance(v_expanded, str) and v_expanded and not os.path.isabs(v_expanded):
                abs_p = str((pf.parent / v_expanded).resolve())
            else:
                abs_p = v_expanded if isinstance(v_expanded,str) else ""
            exists = os.path.exists(abs_p) if abs_p else False
            paths_records.append((str(pf.relative_to(REPO_ROOT)), var, v, abs_p, "OK" if exists else "MISSING", note))

    # collisions: same name defined with different values across files
    def to_key_map(records, name_index, value_index):
        d = defaultdict(list)
        for r in records:
            d[r[name_index]].append(r)
        return d
    env_by_name = to_key_map(env_records, 1, 3)
    paths_by_name = to_key_map(paths_records, 1, 2)

    env_collisions = {k:{r[2] for r in v} for k,v in env_by_name.items() if len({r[2] for r in v})>1}
    path_collisions = {k:{str(r[2]) for r in v} for k,v in paths_by_name.items() if len({str(r[2]) for r in v})>1}

    # usages: search tokens across repo (names only)
    tokens = set(env_by_name.keys()) | set(paths_by_name.keys())
    usage_hits = search_tokens_usages(tokens, all_files)

    # literal hard-coded paths
    try:
        literal_rows = find_literal_paths(all_files)
    except Exception as e:
        print("WARN: literal path scan failed:", e, file=sys.stderr)
        literal_rows = []


    # cycles
    cycle_lines = []
    for f,g in env_graphs.items():
        cyc = detect_cycles(g) if g else []
        for c in cyc:
            cycle_lines.append(f"[.env] {Path(f).relative_to(REPO_ROOT)}: " + " -> ".join(c))
    for f,g in paths_dep_graphs.items():
        cyc = detect_cycles(g) if g else []
        for c in cyc:
            cycle_lines.append(f"[paths.py] {Path(f).relative_to(REPO_ROOT)}: " + " -> ".join(c))

    # write reports
    with open("all_paths_catalog.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["SourceType",".File","VarName","RawValue","ExpandedOrEval","AbsolutePath","Exists","Collides","Notes"])
        for (file,k,raw,expanded,abs_p,exists) in env_records:
            coll = "YES" if k in env_collisions else ""
            w.writerow([".env", file, k, raw, expanded, abs_p, exists, coll, ""])
        for (file,var,val,abs_p,exists,note) in paths_records:
            coll = "YES" if var in path_collisions else ""
            w.writerow(["paths.py", file, var, val, val, abs_p, exists, coll, note])

    with open("token_usages.tsv","w") as f:
        f.write("Token\tFile\tLine\tSnippet\n")
        for t in sorted(tokens):
            for file,line,snip in usage_hits.get(t,[]):
                f.write(f"{t}\t{file}\t{line}\t{snip}\n")

    with open("literal_paths.tsv","w") as f:
        f.write("File\tLine\tLiteral\tResolvedAbsolute\tExists\n")
        for row in literal_rows:
            f.write("\t".join(map(str,row))+"\n")

    with open("dependency_cycles.txt","w") as f:
        if not cycle_lines:
            f.write("No cycles detected.\n")
        else:
            f.write("\n".join(cycle_lines)+"\n")

    print("Wrote:")
    print("  all_paths_catalog.csv")
    print("  token_usages.tsv")
    print("  literal_paths.tsv")
    print("  dependency_cycles.txt")
    print("\nTip: open the TSVs in VS Code and filter; fix collisions first, then cycles, then missing paths.")

if __name__ == "__main__":
    main()
