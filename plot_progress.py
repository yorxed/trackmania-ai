"""
plot_progress.py - график обучения агента по training_log.csv (его пишет tm_dqn.py, по строке на заезд).

Панели (ось X - номер заезда, если не сказано иное):
  1. Прогресс, м: серое - каждый заезд, цветная линия - среднее по окну, чёрная ступенька - лучший результат.
  2. Награда за заезд (и среднее по окну).
  3. Время финиша, с (только финишировавшие заезды) и доля финишей в окне. Пунктир - время эталона (--ref-time).
  4. Чем кончались заезды: доли причин по группам заездов.
  5. ГДЕ заканчиваются заезды (ось X - метры трассы): сколько заездов закончилось на каждом участке, по причинам.
     Считается по последним --last заездам. Высокие столбики - места, которые агент не может пройти.
  6. Эпсилон (доля случайных действий) по заездам: видно, когда исследование почти выключилось.
Красные пунктиры - смена этапа обучения.

Кроме картинки в консоль печатается список самых "смертельных" участков.

Запуск (из папки проекта):
    python plot_progress.py                      # нарисовать, сохранить training_progress.png и показать окно
    python plot_progress.py --no-show            # только сохранить картинку
    python plot_progress.py --watch 120          # каждые 120 с перерисовывать картинку (запусти в 2-м терминале)
    python plot_progress.py --ref-time 25.35     # пунктир со временем эталона
    python plot_progress.py --last 300           # панель 5 по последним 300 заездам (по умолчанию 500)
    python plot_progress.py --window 100         # окно скользящего среднего, заездов
    python plot_progress.py --log eval_log.csv --out eval.png   # график заездов оценки (tm_dqn.py --play)

Можно открывать во время обучения: файл лога только читается. Нужен matplotlib.
"""

import argparse
import csv
import os
import time
from collections import Counter

import numpy as np

REASON_COLORS = {"финиш": "tab:green", "падение": "tab:red", "падение с трассы": "tab:red",
                 "застряла": "tab:orange", "нет прогресса": "tab:brown", "время": "tab:gray"}
HOTSPOT_BIN_M = 25


def read_log(path):
    """Читает лог; битую (недописанную) последнюю строку пропускает."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append(dict(
                    stage=int(r["stage"]), reward=float(r["reward"]), progress=float(r["progress_m"]),
                    reason=r["reason"], finish_ms=float(r["finish_ms"]) if r["finish_ms"] else np.nan,
                    eps=float(r["eps"]) if r.get("eps") not in (None, "") else np.nan,
                    t=int(r["unix_time"])))
            except (ValueError, KeyError, TypeError):
                continue
    return rows


def moving_avg(y, w):
    y = np.asarray(y, float)
    w = max(1, min(w, len(y)))
    if w == 1:
        return y
    out = np.convolve(y, np.ones(w) / w, mode="valid")
    return np.concatenate([np.full(w - 1, np.nan), out])      # слева, пока окно не набралось, линии нет


def hotspots(rows, last, top=6):
    """Участки трассы (по HOTSPOT_BIN_M метров), на которых чаще всего заканчиваются НЕфинишировавшие заезды."""
    sel = [r for r in rows[-last:] if r["reason"] != "финиш"]
    bins = {}
    for r in sel:
        b = int(r["progress"] // HOTSPOT_BIN_M) * HOTSPOT_BIN_M
        bins.setdefault(b, Counter())[r["reason"]] += 1
    ranked = sorted(bins.items(), key=lambda kv: -sum(kv[1].values()))[:top]
    return ranked, len(sel), len(rows[-last:])


def print_hotspots(rows, last):
    ranked, n_fail, n_all = hotspots(rows, last)
    if not ranked:
        return
    print(f"\nГде заканчиваются заезды (последние {n_all} заездов, без финишей: {n_fail}):")
    for b, c in ranked:
        total = sum(c.values())
        detail = ", ".join(f"{k} {v}" for k, v in c.most_common())
        print(f"  {b:5d}-{b + HOTSPOT_BIN_M:5d} м: {total:4d} заездов ({100.0 * total / n_fail:.0f}% всех неудач)  [{detail}]")


def draw(rows, window, ref_time, last, plt):
    n = len(rows)
    x = np.arange(1, n + 1)
    prog = np.array([r["progress"] for r in rows])
    rew = np.array([r["reward"] for r in rows])
    fin = np.array([r["finish_ms"] for r in rows]) / 1000.0
    finished = ~np.isnan(fin)
    stages = np.array([r["stage"] for r in rows])
    eps = np.array([r["eps"] for r in rows])

    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(3, 6)
    a1, a2 = fig.add_subplot(gs[0, 0:3]), fig.add_subplot(gs[0, 3:6])
    a3, a4 = fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6])
    a5, a6 = fig.add_subplot(gs[2, 0:4]), fig.add_subplot(gs[2, 4:6])
    elapsed = (rows[-1]["t"] - rows[0]["t"]) / 3600.0 if n > 1 else 0.0
    fig.suptitle(f"Обучение агента: {n} заездов за {elapsed:.1f} ч | лучший прогресс {prog.max():.0f} м | "
                 f"финишей {int(finished.sum())}" + (f" | лучшее время {np.nanmin(fin):.2f} с" if finished.any() else ""),
                 fontsize=13)

    # 1. прогресс
    a1.plot(x, prog, ".", color="lightgray", markersize=3, label="заезд")
    a1.plot(x, moving_avg(prog, window), color="tab:blue", linewidth=2, label=f"среднее ({window})")
    a1.step(x, np.maximum.accumulate(prog), color="black", linewidth=1.2, where="post", label="лучший")
    a1.set_title("Прогресс за заезд, м")
    a1.set_xlabel("заезд")
    a1.legend(loc="upper left")

    # 2. награда
    a2.plot(x, rew, ".", color="lightgray", markersize=3)
    a2.plot(x, moving_avg(rew, window), color="tab:purple", linewidth=2, label=f"среднее ({window})")
    a2.set_title("Награда за заезд")
    a2.set_xlabel("заезд")
    a2.legend(loc="upper left")

    # 3. время финиша и доля финишей
    if finished.any():
        a3.plot(x[finished], fin[finished], "o", color="tab:green", markersize=4, label="время финиша, с")
        best_t = np.fmin.accumulate(np.where(finished, fin, np.inf))
        best_t = np.where(np.isinf(best_t), np.nan, best_t)
        a3.step(x, best_t, color="black", linewidth=1.2, where="post", label="лучшее время")
    else:
        a3.text(0.5, 0.5, "финишей пока не было", ha="center", va="center", transform=a3.transAxes)
    if ref_time:
        a3.axhline(ref_time, color="tab:orange", linestyle="--", label=f"эталон {ref_time:g} с")
    a3.set_title("Время финиша, с  (и доля финишей справа)")
    a3.set_xlabel("заезд")
    a3.set_ylabel("с")
    b = a3.twinx()
    b.plot(x, moving_avg(finished.astype(float) * 100, window), color="tab:red", alpha=0.7, label="доля финишей, %")
    b.set_ylim(0, 105)
    b.set_ylabel("% финишей в окне")
    h1, l1 = a3.get_legend_handles_labels()
    h2, l2 = b.get_legend_handles_labels()
    a3.legend(h1 + h2, l1 + l2, loc="upper right")

    # 4. причины конца заездов по группам
    group = max(10, n // 20)
    edges = list(range(0, n, group))
    reasons = [r for r, _ in Counter(r["reason"] for r in rows).most_common()]
    bottoms = np.zeros(len(edges))
    for reason in reasons:
        share = np.array([100.0 * sum(1 for r in rows[e:e + group] if r["reason"] == reason) / len(rows[e:e + group])
                          for e in edges])
        a4.bar([e + group / 2 for e in edges], share, width=group * 0.95, bottom=bottoms, label=reason,
               color=REASON_COLORS.get(reason))
        bottoms += share
    a4.set_title(f"Чем кончались заезды, % (группы по {group})")
    a4.set_xlabel("заезд")
    a4.set_ylim(0, 100)
    a4.legend(loc="upper left", fontsize=8, ncol=2)

    # 5. где заканчиваются заезды (ось X - метры)
    sel = rows[-last:]
    pmax = max(r["progress"] for r in sel) + HOTSPOT_BIN_M
    bins = np.arange(0, pmax + HOTSPOT_BIN_M, HOTSPOT_BIN_M)
    bottom = np.zeros(len(bins) - 1)
    for reason in [r for r, _ in Counter(r["reason"] for r in sel).most_common()]:
        cnt, _ = np.histogram([r["progress"] for r in sel if r["reason"] == reason], bins=bins)
        a5.bar(bins[:-1], cnt, width=HOTSPOT_BIN_M * 0.95, align="edge", bottom=bottom, label=reason,
               color=REASON_COLORS.get(reason))
        bottom += cnt
    a5.set_title(f"ГДЕ заканчиваются заезды (последние {len(sel)}): высокий столбик - непройденный участок")
    a5.set_xlabel(f"прогресс вдоль эталона, м (столбики по {HOTSPOT_BIN_M} м)")
    a5.set_ylabel("заездов")
    a5.legend(loc="upper left", fontsize=8, ncol=2)

    # 6. эпсилон
    if np.isnan(eps).all():
        a6.text(0.5, 0.5, "в логе нет колонки eps", ha="center", va="center", transform=a6.transAxes)
    else:
        a6.plot(x, eps, color="tab:cyan")
        a6.set_ylim(0, 1.05)
    a6.set_title("ε: доля случайных действий")
    a6.set_xlabel("заезд")

    # границы этапов: только на панелях, где ось X - номер заезда
    for i in range(1, n):
        if stages[i] != stages[i - 1]:
            for a in (a1, a2, a3, a4, a6):
                a.axvline(i + 0.5, color="tab:red", linestyle=":", alpha=0.8)
            a1.text(i + 0.5, a1.get_ylim()[1], f" этап {stages[i]}", color="tab:red", va="top", fontsize=9)
    for a in (a1, a2, a3, a4, a5, a6):
        a.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def main():
    p = argparse.ArgumentParser(description="График обучения по training_log.csv")
    p.add_argument("--log", default="training_log.csv")
    p.add_argument("--out", default="training_progress.png")
    p.add_argument("--window", type=int, default=50, help="окно скользящего среднего (заездов)")
    p.add_argument("--last", type=int, default=500, help="панель 5 и список участков: по последним N заездам")
    p.add_argument("--ref-time", type=float, default=None, help="время эталона, с: рисует пунктир")
    p.add_argument("--no-show", action="store_true", help="не открывать окно, только сохранить картинку")
    p.add_argument("--watch", type=float, default=None, metavar="СЕК", help="перерисовывать картинку каждые СЕК секунд")
    args = p.parse_args()

    import matplotlib
    if args.no_show or args.watch:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def once():
        if not os.path.exists(args.log):
            print(f"Файла {args.log} ещё нет - агент пока не закончил ни одного заезда (или запущен без лога).")
            return None
        rows = read_log(args.log)
        if len(rows) < 2:
            print("В логе меньше двух заездов - рисовать нечего.")
            return None
        fig = draw(rows, args.window, args.ref_time, args.last, plt)
        fig.savefig(args.out, dpi=105)
        print(f"[{time.strftime('%H:%M:%S')}] заездов {len(rows)} -> {args.out}")
        print_hotspots(rows, args.last)
        return fig

    if args.watch:
        print(f"Обновляю {args.out} каждые {args.watch:g} с. Остановка: Ctrl+C.")
        try:
            while True:
                fig = once()
                if fig is not None:
                    plt.close(fig)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            return
    fig = once()
    if fig is not None and not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()