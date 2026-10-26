"""自分をモデルに見立てた評価グラフ(学習曲線 / 言語×分野 / Attention weights)を描く。

使い方:
  python profile_charts.py            # GitHub API から実データを取得して描画
  python profile_charts.py --sample   # API を使わずサンプルデータで描画(見た目の確認用)

環境変数 GITHUB_TOKEN があれば API の利用制限が緩くなる(Actions では自動で入る)。
出力: assets/profile-charts-dark.svg と assets/profile-charts-light.svg
"""

from __future__ import annotations

import datetime as dt
import os
import random
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


def fetch(cfg: dict) -> dict:
    gh, user = GitHub(), cfg["user"]
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
                       for c in gh.pages(f"/repos/{name}/commits", author=user, since=since)]
        except requests.HTTPError as e:
            print(f"skip {name}: {e}", file=sys.stderr)
            continue
        repos.append(dict(name=name, topics=info.get("topics", []), languages=langs, commits=commits))
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

def repo_domains(repo: dict, cfg: dict) -> list[str]:
    over = (cfg.get("domain_overrides") or {}).get(repo["name"])
    tags = over if over else repo["topics"]
    return [t for t in tags if t in cfg["domains"]]


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
        for lang in r["languages"]:
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
        train.append(2.5 / (1 + n_commits / 40))
        val.append(2.5 / (1 + n_pairs / 4))
    return months, xs, train, val, len(all_commits)


def language_domain(data: dict, cfg: dict):
    """各言語が、どの分野にどれくらい使われているか(言語ごとに合計1へ正規化)。"""
    mat: dict[str, Counter] = defaultdict(Counter)
    for r in data["repos"]:
        doms = repo_domains(r, cfg)
        total = sum(r["languages"].values()) or 1
        for lang, b in r["languages"].items():
            for d in doms:
                mat[lang][d] += b / total / len(doms)
    doms = list(cfg["domains"])
    langs = sorted(mat, key=lambda l: -sum(mat[l].values()))[: cfg.get("top_languages", 6)]
    v = np.array([[mat[l][d] for d in doms] for l in langs]) if langs else np.zeros((0, len(doms)))
    v = v / np.maximum(v.sum(axis=1, keepdims=True), 1e-9)
    return langs, [cfg["domains"][d] for d in doms], v


def attention(data: dict, cfg: dict):
    """直近 N 日のコミットを分野ごとに数え、合計1の重みにする。"""
    cutoff = str(dt.date.today() - dt.timedelta(days=cfg.get("attention_days", 30)))
    cnt = Counter()
    for r in data["repos"]:
        doms = repo_domains(r, cfg)
        recent = sum(c >= cutoff for c in r["commits"])
        for d in doms:
            cnt[d] += recent / len(doms)
    total = sum(cnt.values())
    items = sorted(((cfg["domains"][d], cnt[d] / total if total else 0) for d in cfg["domains"]), key=lambda x: -x[1])
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
        "axes.edgecolor": t["grid"], "font.size": 10, "svg.fonttype": "path",
    })
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.3, 1.1, 1]})

    # ① 学習曲線
    months, x, train, val, n = learning_curve(data, cfg)
    ax = axs[0]
    ax.plot(x, train, "o-", c=t["train"], ms=4, label="train loss (commits)")
    ax.plot(x, val, "o--", c=t["val"], ms=4, label="val loss (new language × domain)")
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
        ax.plot([xi], [yi], marker="*", ms=11, c=t["fg"], zorder=3)
        ax.annotate(ms["label"], (xi, yi), xytext=(0, 18 if flip else -32), textcoords="offset points",
                    ha="center", fontsize=8, color=t["sub"])
    ax.set_xticks([i + .5 for i in range(len(months))], [m.strftime("%b") for m in months])
    ax.set_xlim(-.1, len(months))
    ax.set_ylim(0, 2.7)
    ax.set_title(f"Learning Curve — {cfg['user']}  ({n} commits)", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # ② 言語 × 分野
    langs, doms, v = language_domain(data, cfg)
    ax = axs[1]
    if len(langs):
        cmap = plt.get_cmap(t["cmap"])
        ax.imshow(v, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(doms)), [d.replace(" ", "\n") for d in doms])
        ax.set_yticks(range(len(langs)), langs)
        for i in range(len(langs)):
            for j in range(len(doms)):
                r, g, b, _ = cmap(v[i, j])
                light_cell = 0.299 * r + 0.587 * g + 0.114 * b > 0.5  # セルの明るさで文字色を決める
                ax.text(j, i, f"{v[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="#1f2328" if light_cell else "#ffffff")
    ax.set_title("Language × Domain", loc="left")

    # ③ Attention weights
    items = attention(data, cfg)
    ax = axs[2]
    labels = [k for k, _ in items][::-1]
    weights = [w for _, w in items][::-1]
    colors = [t["bars"][1] if w == max(weights) and w > 0 else t["bars"][0] for w in weights]
    ax.barh(labels, weights, color=colors)
    for i, w in enumerate(weights):
        ax.text(w + .01, i, f"{w:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, max(max(weights, default=0) * 1.3, .1))
    ax.set_title(f"Attention weights — last {cfg.get('attention_days', 30)} days", loc="left")

    stamp = f"updated {dt.date.today()}" + ("  ·  SAMPLE DATA" if is_sample else "")
    fig.text(.995, .01, stamp, ha="right", fontsize=8, color=t["sub"])
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    cfg = yaml.safe_load(Path("profile_config.yml").read_text(encoding="utf-8"))
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
