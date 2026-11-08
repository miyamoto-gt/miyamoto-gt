"""自分をモデルに見立てた評価グラフ(学習曲線 / 言語×分野 / Attention weights)を描く。

使い方:
  python profile_charts.py            # GitHub API から実データを取得して描画
  python profile_charts.py --sample   # API を使わずサンプルデータで描画(見た目の確認用)

環境変数 GITHUB_TOKEN があれば API の利用制限が緩くなる(Actions では自動で入る)。
出力: assets/profile-charts-dark.svg と assets/profile-charts-light.svg
"""

from __future__ import annotations

import datetime as dt
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import yaml

API = "https://api.github.com"
OUT = Path("assets")

THEMES = {
    "dark": dict(bg="#0d1117", fg="#e6edf3", sub="#8b949e", grid="#30363d",
                 train="#58a6ff", val="#f78166", cmap="magma", bars=("#30363d", "#d2a8ff")),
    "light": dict(bg="#ffffff", fg="#1f2328", sub="#59636e", grid="#d1d9e0",
                  train="#0969da", val="#cf222e", cmap="magma_r", bars=("#d1d9e0", "#8250df")),
}


# ---------------------------------------------------------------- データ取得

# コードの import や依存ファイルから見つけるライブラリ・ツール(左: 探す文字列 / 右: 図に出す名前)
TOOLS = {
    "torch": "PyTorch", "transformers": "Transformers", "peft": "PEFT", "trl": "TRL",
    "unsloth": "Unsloth", "bitsandbytes": "bitsandbytes", "datasets": "HF Datasets",
    "sentence_transformers": "SBERT", "sentence-transformers": "SBERT",
    "mlx_lm": "MLX", "mlx-lm": "MLX", "ollama": "Ollama", "llama_cpp": "llama.cpp",
    "langchain": "LangChain", "faiss": "FAISS", "chromadb": "Chroma",
    "sklearn": "scikit-learn", "scikit-learn": "scikit-learn", "statsmodels": "statsmodels",
    "pandas": "pandas", "numpy": "NumPy", "matplotlib": "matplotlib",
    "gymnasium": "Gymnasium", "gym": "Gymnasium", "stable_baselines3": "SB3",
    "fastapi": "FastAPI", "flask": "Flask", "streamlit": "Streamlit", "gradio": "Gradio",
    "openai": "OpenAI API", "google.generativeai": "Gemini API", "@google/generative-ai": "Gemini API",
    "boto3": "AWS SDK", "express": "Express", "react": "React",
}
# ツールを大まかな種類にまとめる(heatmap_rows: groups のとき使う)
TOOL_GROUPS = {
    "Deep Learning": ["PyTorch", "Transformers", "PEFT", "TRL", "Unsloth", "bitsandbytes", "HF Datasets", "MLX"],
    "Classic ML & Data": ["pandas", "scikit-learn", "statsmodels", "NumPy", "matplotlib"],
    "Retrieval & LLM": ["SBERT", "LangChain", "FAISS", "Chroma", "Ollama", "llama.cpp",
                        "OpenAI API", "Gemini API", "AWS SDK"],
    "Serving & Web": ["FastAPI", "Flask", "Streamlit", "Gradio", "Express", "React"],
    "RL": ["Gymnasium", "SB3"],
}
CODE_EXT = {".py": "Python", ".ipynb": "Jupyter Notebook", ".js": "JavaScript", ".ts": "TypeScript",
            ".jsx": "JavaScript", ".tsx": "TypeScript"}
DEP_FILES = ("requirements.txt", "requirements-dev.txt", "pyproject.toml", "package.json")


def find_tools(text: str) -> set[str]:
    """ソースや依存ファイルの中身から、使っているツールを見つける。"""
    found = set()
    for line in text.splitlines():
        s = line.strip().strip('",')
        for key, name in TOOLS.items():
            k = re.escape(key)
            if re.search(rf"^(?:from|import)\s+{k}\b", s) or re.search(rf"require\(['\"]{k}['\"]\)", s) \
               or re.search(rf"from\s+['\"]{k}['\"]", s) or re.match(rf"^{k}\s*(?:[=<>~!\[;:\"]|$)", s):
                found.add(name)
    return found


def fetch_tools(gh, name: str, langs_allowed: set | None) -> set[str]:
    """リポジトリのファイル一覧から依存ファイルとコードを読み、使っているツールを集める。"""
    info = gh.get(f"/repos/{name}")
    tree = gh.get(f"/repos/{name}/git/trees/{info['default_branch']}", recursive=1).get("tree", [])
    picks = []
    for f in tree:
        path = f.get("path", "")
        base, ext = os.path.basename(path), os.path.splitext(path)[1].lower()
        if f.get("type") != "blob" or f.get("size", 0) > 400_000 or "node_modules" in path:
            continue
        if base in DEP_FILES:
            dep_lang = {"JavaScript", "TypeScript"} if base == "package.json" else {"Python", "Jupyter Notebook"}
            if langs_allowed is None or dep_lang & langs_allowed:
                picks.append(path)
        elif ext in CODE_EXT and (langs_allowed is None or CODE_EXT[ext] in langs_allowed):
            picks.append(path)
    tools = set()
    for path in picks[:60]:
        r = gh.s.get(f"{API}/repos/{name}/contents/{path}", timeout=30,
                     headers={"Accept": "application/vnd.github.raw"})
        if r.ok:
            text = r.text
            if path.endswith(".ipynb"):  # ノートブックはコードセルの中身だけ見る
                try:
                    text = "\n".join("".join(c.get("source", [])) for c in json.loads(text).get("cells", [])
                                      if c.get("cell_type") == "code")
                except ValueError:
                    pass
            tools |= find_tools(text)
    return tools

class GitHub:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers["Accept"] = "application/vnd.github+json"
        if tok := os.environ.get("GITHUB_TOKEN"):
            self.s.headers["Authorization"] = f"Bearer {tok}"

    def get(self, path, **params):
        r = self.s.get(f"{API}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def pages(self, path, **params):
        page = 1
        while True:
            items = self.get(path, per_page=100, page=page, **params)
            if not items:
                return
            yield from items
            if len(items) < 100:
                return
            page += 1


def is_mine(c: dict, user: str, my_emails: set[str]) -> bool:
    """GitHubアカウントに紐づいたコミットか、my_emails に書いたメールのコミットなら自分のもの。"""
    login = (c.get("author") or {}).get("login", "")
    email = (c["commit"]["author"].get("email") or "").lower()
    return login.lower() == user.lower() or email in my_emails


def fetch(cfg: dict) -> dict:
    gh, user = GitHub(), cfg["user"]
    # メールは公開ファイルに書かず、Actions の Secret「MY_EMAILS」(カンマ区切り)から読む
    secret = os.environ.get("MY_EMAILS", "")
    my_emails = {e.strip().lower() for e in re.split(r"[,、，\s]+", secret) if e.strip()}
    my_emails |= {e.lower() for e in (cfg.get("my_emails") or [])}
    print(f"my_emails: {len(my_emails)} 件読み込み", file=sys.stderr)  # アドレス自体はログに出さない
    since = f"{cfg['start']}T00:00:00Z"
    names = {r["full_name"] for r in gh.pages(f"/users/{user}/repos", type="owner") if not r["fork"]}
    names |= set(cfg.get("extra_repos") or [])
    names -= set(cfg.get("exclude_repos") or [])

    repos = []
    for name in sorted(names):
        try:
            info = gh.get(f"/repos/{name}")
            langs = gh.get(f"/repos/{name}/languages")
            commits = [c["commit"]["author"]["date"][:10]
                       for c in gh.pages(f"/repos/{name}/commits", since=since)
                       if is_mine(c, user, my_emails)]
        except requests.HTTPError as e:
            print(f"skip {name}: {e}", file=sys.stderr)
            continue
        tools = (cfg.get("tool_overrides") or {}).get(name) or []
        if not tools and cfg.get("heatmap_rows", "languages") in ("tools", "groups"):
            mine = (cfg.get("my_languages") or {}).get(name)
            try:
                tools = sorted(fetch_tools(gh, name, set(mine) if mine else None))
            except requests.HTTPError as e:
                print(f"tools skip {name}: {e}", file=sys.stderr)
        repos.append(dict(name=name, topics=info.get("topics", []), languages=langs, commits=commits, tools=tools))
        print(f"{name}: {len(commits)} commits, topics={info.get('topics', [])}", file=sys.stderr)
    return {"repos": repos}


def sample(cfg: dict) -> dict:
    """見た目確認用のダミーデータ。"""
    rnd = random.Random(0)
    start = dt.date.fromisoformat(str(cfg["start"]))
    span = (dt.date.today() - start).days

    def dates(n, lo=0.0, hi=1.0):
        return [str(start + dt.timedelta(days=int(span * rnd.uniform(lo, hi)))) for _ in range(n)]

    return {"repos": [
        dict(name="miyamoto-gt/chatbot", topics=["web-app"], languages={"JavaScript": 9000, "HTML": 4000, "CSS": 3000}, commits=dates(25, .15, .35)),
        dict(name="miyamoto-gt/weather_model", topics=["data-analysis"], languages={"Python": 12000, "Jupyter Notebook": 30000}, commits=dates(20, .25, .45)),
        dict(name="miyamoto-gt/weather_predict_site", topics=["web-app", "data-analysis"], languages={"Python": 5000, "JavaScript": 3000, "HTML": 2000, "CSS": 2000}, commits=dates(15, .35, .5)),
        dict(name="miyamoto-gt/Transformer0-1", topics=["model-impl"], languages={"Python": 20000, "Jupyter Notebook": 15000}, commits=dates(40, .7, 1)),
        dict(name="Hackit-2026/Team18", topics=[], languages={"JavaScript": 25000, "Python": 8000, "HTML": 5000, "CSS": 6000}, commits=dates(30, .75, .85)),
        dict(name="jphacks/kz_2607", topics=[], languages={"Python": 18000, "TypeScript": 22000}, commits=dates(35, .9, 1)),
    ]}


# ---------------------------------------------------------------- 集計

def as_list(v) -> list:
    return list(v) if isinstance(v, (list, tuple)) else [v]


def norm(s: str) -> str:
    """表記ゆれをそろえる: 大文字小文字・ハイフン・スペース・先頭の web を無視する。
    web-backend / Web backend / backend → "backend" """
    s = re.sub(r"[^a-z0-9]", "", str(s).lower())
    return s[3:] if s.startswith("web") and len(s) > 3 else s


def topic_label(topic: str, cfg: dict) -> list[str]:
    """Topic を図の列名(1つ以上)に変換する。対象外の Topic なら空のリスト。
    - domains に書いた Topic は、そこに書いた名前を使う(リストで書けば複数の列に入る)
    - auto_prefixes(例: ml-, role-)で始まる Topic は、自動で列にする
      ml-pretraining → "Pretraining" / role-backend → "Backend"
    """
    if topic in cfg["domains"]:
        return as_list(cfg["domains"][topic])
    # 完全一致しなくても、表記ゆれだけの違いなら同じタグとみなす(web-backend と backend など)
    for key, label in cfg["domains"].items():
        if norm(key) == norm(topic):
            return as_list(label)
    for prefix in cfg.get("auto_prefixes") or []:
        if topic.startswith(prefix) and len(topic) > len(prefix):
            word = topic[len(prefix):].replace("-", " ")
            return [word[:1].upper() + word[1:]]
    return []


def repo_domains(repo: dict, cfg: dict) -> list[str]:
    over = (cfg.get("domain_overrides") or {}).get(repo["name"])
    tags = over if over else repo["topics"]
    return list(dict.fromkeys(l for t in tags for l in topic_label(t, cfg)))


def domain_labels(data: dict, cfg: dict) -> list[str]:
    """図に出す列。どれかのリポジトリで使われているものだけ(空の列は出さない)。
    並びは domains に書いた順 → 自動で増えた列の順。"""
    used = {d for r in data["repos"] for d in repo_domains(r, cfg)}
    order = list(dict.fromkeys(l for v in cfg["domains"].values() for l in as_list(v)))
    order += sorted(used - set(order))
    return [d for d in order if d in used]


def lang_targets(lang: str, doms: list[str], cfg: dict) -> list[str]:
    """その言語を、リポジトリのどの列に配るか。
    domain_languages で「この列はこの言語」と決めてあれば、その列にだけ入れる
    (例: HTML/CSS は Web frontend、Python は Web backend)。"""
    # 列名は表記ゆれを無視して照合する("Web backend" と "backend" を同じ列とみなす)
    rules = {norm(k): set(v or []) for k, v in (cfg.get("domain_languages") or {}).items()}
    own = [d for d in doms if lang in rules.get(norm(d), set())]
    return own or doms


def repo_languages(repo: dict, cfg: dict) -> dict:
    """チーム開発のリポジトリでは、自分が書いた言語だけを数える(my_languages で指定)。"""
    mine = (cfg.get("my_languages") or {}).get(repo["name"])
    skip = set(cfg.get("exclude_languages") or [])
    return {k: v for k, v in repo["languages"].items()
            if (not mine or k in mine) and k not in skip}


def months_between(start: dt.date, end: dt.date) -> list[dt.date]:
    out, d = [], start.replace(day=1)
    while d <= end:
        out.append(d)
        d = (d.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    return out


def learning_curve(data: dict, cfg: dict):
    """train loss = 累計コミット数から、val loss = 触った(言語, 分野)の組の数から計算する。
    どちらも「経験が増えるほど下がる」比喩としての loss。"""
    start = dt.date.fromisoformat(str(cfg["start"]))
    months = months_between(start, dt.date.today())
    first_seen: dict[tuple, str] = {}
    all_commits = []
    for r in data["repos"]:
        if not r["commits"]:
            continue
        all_commits += r["commits"]
        first = min(r["commits"])
        for lang in repo_languages(r, cfg):
            for dom in repo_domains(r, cfg) or ["other"]:
                key = (lang, dom)
                first_seen[key] = min(first_seen.get(key, first), first)

    # 各月の末日(今月は今日)時点の値を、x = 月の位置 + 経過割合 に置く
    today = dt.date.today()
    xs, train, val = [0.0], [2.5], [2.5]
    for i, m in enumerate(months):
        next_m = (m.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        end = min(next_m, today + dt.timedelta(days=1))
        n_commits = sum(c < str(end) for c in all_commits)
        n_pairs = sum(d < str(end) for d in first_seen.values())
        xs.append(i + (end - m).days / (next_m - m).days)
        train.append(2.5 / (1 + n_commits / 100))
        val.append(2.5 / (1 + n_pairs / 4))
    return months, xs, train, val, len(all_commits)


def tool_domain(data: dict, cfg: dict):
    """各ツール(または種類)を、どの分野で使っているか(行ごとに合計1)。リポジトリ1つを1票として数える。
    names には、各マスで実際に使ったツール名を入れる(マスの中に文字で出す用)。"""
    mat: dict[str, Counter] = defaultdict(Counter)
    names: dict[tuple, Counter] = defaultdict(Counter)
    # どこでも使う定番ライブラリは、指定がなければ外す
    skip = set(cfg.get("exclude_tools", ["NumPy", "matplotlib"]) or [])
    groups = cfg.get("tool_groups") or TOOL_GROUPS
    grouped = cfg.get("heatmap_rows") == "groups"
    for r in data["repos"]:
        doms = repo_domains(r, cfg)
        tools = set(r.get("tools", [])) - skip
        rows = defaultdict(set)
        for tool in tools:
            if grouped:
                for g, members in groups.items():
                    if tool in members:
                        rows[g].add(tool)
            else:
                rows[tool].add(tool)
        for row, used in rows.items():
            for d in doms:
                mat[row][d] += 1.0 / len(doms)
                names[(row, d)].update(used)
    doms = domain_labels(data, cfg)
    rows = sorted(mat, key=lambda k: -sum(mat[k].values()))[: cfg.get("top_tools", 8)]
    v = np.array([[mat[k][d] for d in doms] for k in rows]) if rows else np.zeros((0, len(doms)))
    v = v / np.maximum(v.sum(axis=1, keepdims=True), 1e-9)
    cell = [[[n for n, _ in names[(k, d)].most_common(cfg.get("names_per_cell", 2))] for d in doms] for k in rows]
    return rows, doms, v, (cell if grouped else None)


def language_domain(data: dict, cfg: dict):
    """各言語が、どの分野にどれくらい使われているか(言語ごとに合計1へ正規化)。"""
    mat: dict[str, Counter] = defaultdict(Counter)
    for r in data["repos"]:
        doms = repo_domains(r, cfg)
        langs = repo_languages(r, cfg)
        total = sum(langs.values()) or 1
        for lang, b in langs.items():
            targets = lang_targets(lang, doms, cfg) if doms else []
            for d in targets:
                mat[lang][d] += b / total / len(targets)
    doms = domain_labels(data, cfg)
    langs = sorted(mat, key=lambda l: -sum(mat[l].values()))[: cfg.get("top_languages", 6)]
    v = np.array([[mat[l][d] for d in doms] for l in langs]) if langs else np.zeros((0, len(doms)))
    v = v / np.maximum(v.sum(axis=1, keepdims=True), 1e-9)
    return langs, doms, v


def attention(data: dict, cfg: dict):
    """直近 N 日のコミットを分野ごとに数え、合計1の重みにする。
    attention_days: all なら、開始月からの全期間で数える。"""
    days = cfg.get("attention_days", 30)
    cutoff = str(cfg["start"]) if str(days).lower() == "all" else str(dt.date.today() - dt.timedelta(days=int(days)))
    cnt = Counter()
    for r in data["repos"]:
        doms = repo_domains(r, cfg)
        recent = sum(c >= cutoff for c in r["commits"])
        for d in doms:
            cnt[d] += recent / len(doms)
    total = sum(cnt.values())
    items = sorted(((d, cnt[d] / total if total else 0) for d in domain_labels(data, cfg)), key=lambda x: -x[1])
    return items


# ---------------------------------------------------------------- 描画

def setup_font():
    from matplotlib import font_manager
    for name in ["Noto Sans CJK JP", "Noto Sans JP", "Hiragino Sans", "IPAexGothic"]:
        if any(name in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = [name, "DejaVu Sans"]
            return


def draw(data: dict, cfg: dict, theme: str, path: Path, is_sample: bool):
    t = THEMES[theme]
    plt.rcParams.update({
        "figure.facecolor": t["bg"], "axes.facecolor": t["bg"], "savefig.facecolor": t["bg"],
        "text.color": t["fg"], "axes.labelcolor": t["fg"], "xtick.color": t["sub"], "ytick.color": t["sub"],
        "axes.edgecolor": t["grid"], "font.size": 13, "svg.fonttype": "path",
        "axes.titlesize": 16, "axes.titleweight": "bold",
    })
    # 上段に学習曲線を横いっぱいに、下段にヒートマップと Attention を並べる
    fig, ax_map = plt.subplot_mosaic([["curve", "curve"], ["heat", "attn"]], figsize=(12, 10),
                                     gridspec_kw={"height_ratios": [1, 1.05], "width_ratios": [1.15, 1]})
    axs = [ax_map["curve"], ax_map["heat"], ax_map["attn"]]

    # ① 学習曲線
    months, x, train, val, n = learning_curve(data, cfg)
    ax = axs[0]
    ax.plot(x, train, "o-", c=t["train"], ms=6, lw=2.2, label="train loss (commits)")
    ax.plot(x, val, "o--", c=t["val"], ms=6, lw=2.2, label="val loss (new language × domain)")
    last_xi, flip = -9.0, False
    for ms in sorted(cfg.get("milestones") or [], key=lambda m: str(m["date"])):
        d = dt.date.fromisoformat(str(ms["date"]))
        i = (d.year - months[0].year) * 12 + d.month - months[0].month
        if d > dt.date.today() or i < 0:
            continue
        xi = i + (d.day - 1) / 31
        yi = float(np.interp(xi, x, train))
        flip = (not flip) if xi - last_xi < 0.9 else False  # 近い節目は上下交互に置く
        last_xi = xi
        ax.plot([xi], [yi], marker="*", ms=15, c=t["fg"], zorder=3)
        ax.annotate(ms["label"], (xi, yi), xytext=(0, 20 if flip else -38), textcoords="offset points",
                    ha="center", fontsize=11, color=t["sub"])
    # 目盛りは各月の1日の位置に置く(線の右端 = 今日)
    ax.set_xticks(list(range(len(months))), [m.strftime("%b") for m in months])
    ax.set_xlim(-.1, len(months))
    ax.set_ylim(0, 2.7)
    ax.set_title(f"Learning Curve — {cfg['user']}  ({n} commits)", loc="left")
    ax.legend(frameon=False, fontsize=11, loc="upper right")

    # ② 言語 × 分野
    by_tools = cfg.get("heatmap_rows", "languages") in ("tools", "groups")
    if by_tools:
        langs, doms, v, cell_names = tool_domain(data, cfg)
    else:
        (langs, doms, v), cell_names = language_domain(data, cfg), None
    ax = axs[1]
    if len(langs):
        cmap = plt.get_cmap(t["cmap"])
        vmax = 1.15  # 1.00 のマスが真っ黒にならないよう、色の上限に少し余裕を持たせる
        ax.imshow(v, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        if len(doms) <= 4 or max(len(d) for d in doms) <= 9:  # 短い列名なら横向きのまま
            ax.set_xticks(range(len(doms)), [d.replace(" ", "\n") for d in doms])
        else:  # 列が多いときは斜めにして重なりを防ぐ
            ax.set_xticks(range(len(doms)), doms, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(langs)), langs)
        for i in range(len(langs)):
            for j in range(len(doms)):
                r, g, b, _ = cmap(v[i, j] / vmax)
                light_cell = 0.299 * r + 0.587 * g + 0.114 * b > 0.5  # セルの明るさで文字色を決める
                color = "#1f2328" if light_cell else "#ffffff"
                small = len(doms) > 5
                names = cell_names[i][j] if cell_names and cfg.get("show_tool_names") else []
                # 混同行列らしく数値を大きく。種類でまとめたときは、下に実際に使ったツール名を小さく添える
                ax.text(j, i - (0.17 if names else 0), f"{v[i, j]:.2f}", ha="center", va="center",
                        fontsize=10 if small else 12, fontweight="bold", color=color)
                if names:
                    ax.text(j, i + 0.2, "\n".join(names), ha="center", va="center", linespacing=1.15,
                            fontsize=7 if small else 8, color=color, alpha=0.85)
    ax.set_title(cfg.get("heatmap_title", ("Tools × Domain" if cfg.get("heatmap_rows") == "tools" else "What I use in each field") if by_tools else "Language × Domain"), loc="left")

    # ③ Attention weights
    items = attention(data, cfg)
    ax = axs[2]
    labels = [k for k, _ in items][::-1]
    weights = [w for _, w in items][::-1]
    colors = [t["bars"][1] if w == max(weights) and w > 0 else t["bars"][0] for w in weights]
    ax.barh(labels, weights, color=colors)
    for i, w in enumerate(weights):
        ax.text(w + .01, i, f"{w:.2f}", va="center", fontsize=11)
    ax.set_xlim(0, max(max(weights, default=0) * 1.3, .1))
    days = cfg.get("attention_days", 30)
    period = "all time" if str(days).lower() == "all" else f"last {days} days"
    ax.set_title(f"Attention weights — {period}", loc="left")

    stamp = f"updated {dt.date.today()}" + ("  ·  SAMPLE DATA" if is_sample else "")
    fig.text(.995, .01, stamp, ha="right", fontsize=10, color=t["sub"])
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    text = Path("profile_config.yml").read_text(encoding="utf-8")
    text = re.sub(r"\}\s*\.+\s*$", "}", text, flags=re.M)  # 「}.」のような行末の余計なピリオドを消す
    try:
        cfg = yaml.safe_load(text)
    except yaml.YAMLError as e:
        sys.exit(f"profile_config.yml の書き方に誤りがあります:\n{e}")
    is_sample = "--sample" in sys.argv
    data = sample(cfg) if is_sample else fetch(cfg)
    setup_font()
    OUT.mkdir(exist_ok=True)
    for theme in THEMES:
        p = OUT / f"profile-charts-{theme}.svg"
        draw(data, cfg, theme, p, is_sample)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
