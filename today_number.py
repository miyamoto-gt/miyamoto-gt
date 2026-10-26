"""今日の数: 元日から数えた日数を素数判定し、今日の日付(MMDD)が円周率の何桁目に現れるかを調べて README を更新する。"""

import datetime
import re
import sys
from zoneinfo import ZoneInfo

import mpmath
import sympy

PI_DIGITS = 100_000  # 4桁の並び(MMDD)は、すべてこの範囲に現れる
START, END = "<!-- TODAY:START -->", "<!-- TODAY:END -->"


def pi_digits(n: int) -> str:
    mpmath.mp.dps = n + 10
    return mpmath.nstr(mpmath.pi, n + 5)[2 : n + 2]  # 小数点以下のみ


def factor_str(n: int) -> str:
    parts = []
    for p, e in sympy.factorint(n).items():
        parts.append(f"{p}<sup>{e}</sup>" if e > 1 else str(p))
    return " × ".join(parts)


def divisor_kind(n: int) -> str:
    s = sympy.divisor_sigma(n) - n  # 自分以外の約数の和
    if s == n:
        return "完全数 ✨"
    return "過剰数" if s > n else "不足数"


def build(today: datetime.date) -> str:
    n = today.timetuple().tm_yday
    mmdd = today.strftime("%m%d")
    pos = pi_digits(PI_DIGITS).find(mmdd) + 1

    if sympy.isprime(n):
        prime_line = f"**{n}** は **素数** です 🎉"
    else:
        prime_line = f"**{n}** は素数ではありません({n} = {factor_str(n)})"

    pi_line = (
        f"今日の日付 **{mmdd}** は、円周率の小数点以下 **{pos:,}桁目** から現れます"
        if pos > 0
        else f"今日の日付 **{mmdd}** は、円周率の最初の{PI_DIGITS:,}桁には現れません"
    )

    return "\n".join(
        [
            START,
            f"### 🔢 今日の数 ({today:%Y-%m-%d})",
            "",
            f"- 今日は元日から **{n}日目**。{prime_line}",
            f"- 約数の和で分類すると **{divisor_kind(n)}**",
            f"- π: {pi_line}",
            END,
        ]
    )


def main(readme: str = "README.md") -> None:
    today = datetime.datetime.now(ZoneInfo("Asia/Tokyo")).date()
    block = build(today)
    text = open(readme, encoding="utf-8").read()
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    if not pattern.search(text):
        sys.exit(f"{readme} に {START} と {END} が見つかりません")
    open(readme, "w", encoding="utf-8").write(pattern.sub(lambda _: block, text))
    print(block)


if __name__ == "__main__":
    main(*sys.argv[1:])
