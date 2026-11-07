"""
plot_progress.py - график обучения агента по training_log.csv (его пишет tm_dqn.py, по строке на заезд).

Что на графике (4 панели, по оси X - номер заезда):
  1. Прогресс, м: как далеко агент уезжает за заезд. Серое - каждый заезд, цветная линия - среднее
     по окну, чёрная ступенька - лучший результат за всё время.
  2. Награда за заезд (и среднее по окну).
  3. Время финиша, с (только заезды, дошедшие до финиша), и доля финишей в окне.
     Пунктир - время твоего эталона (если указать --ref-time).
  4. Чем кончались заезды: доли причин (финиш, падение, застряла...) по группам заездов.

Запуск (из папки проекта):
    python plot_progress.py                      # нарисовать, сохранить training_progress.png и показать окно
    python plot_progress.py --no-show            # только сохранить картинку
    python plot_progress.py --watch 120          # каждые 120 с перерисовывать картинку (запусти в 2-м терминале)
    python plot_progress.py --ref-time 30        # добавить линию времени эталона (30 с)
    python plot_progress.py --window 100         # окно скользящего среднего, заездов
    python plot_progress.py --log other.csv --out other.png

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


def read_log(path):
    """Читает лог; битую (недописанную) последнюю строку пропускает."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append(dict(
                    stage=int(r["stage"]), reward=float(r["reward"]), progress=float(r["progress_m"]),
                    reason=r["reason"], finish_ms=float(r["finish_ms"]) if r["finish_ms"] else np.nan,
                    t=int(r["unix_time"])))
            except (ValueError, KeyError, TypeError):
                continue
    return rows


def moving_avg(y, w):
    y = np.asarray(y, float)
    w = max(1, min(w, len(y)))
    if w == 1:
        return y
    k = np.ones(w) / w
    out = np.convolve(y, k, mode="valid")
    return np.concatenate([np.full(w - 1, np.nan), out])      # слева, пока окно не набралось, линии нет


def draw(rows, window, ref_time, plt):
    n = len(rows)
    x = np.arange(1, n + 1)
    prog = np.array([r["progress"] for r in rows])
    rew = np.array([r["reward"] for r in rows])
    fin = np.array([r["finish_ms"] for r in rows]) / 1000.0
    finished = ~np.isnan(fin)
    stages = np.array([r["stage"] for r in rows])

    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    elapsed = (rows[-1]["t"] - rows[0]["t"]) / 3600.0 if n > 1 else 0.0
    best_p = prog.max()
    fig.suptitle(f"Обучение агента: {n} заездов за {elapsed:.1f} ч | лучший прогресс {best_p:.0f} м | "
                 f"финишей {int(finished.sum())}" + (f" | лучшее время {np.nanmin(fin):.2f} с" if finished.any() else ""),
                 fontsize=13)

    # 1. прогресс
    a = ax[0, 0]
    a.plot(x, prog, ".", color="lightgray", markersize=3, label="заезд")
    a.plot(x, moving_avg(prog, window), color="tab:blue", linewidth=2, label=f"среднее ({window})")
    a.step(x, np.maximum.accumulate(prog), color="black", linewidth=1.2, where="post", label="лучший")
    a.set_title("Прогресс за заезд, м")
    a.set_xlabel("заезд")
    a.legend(loc="upper left")

    # 2. награда
    a = ax[0, 1]
    a.plot(x, rew, ".", color="lightgray", markersize=3)
    a.plot(x, moving_avg(rew, window), color="tab:purple", linewidth=2, label=f"среднее ({window})")
    a.set_title("Награда за заезд")
    a.set_xlabel("заезд")
    a.legend(loc="upper left")

    # 3. время финиша и доля финишей
    a = ax[1, 0]
    if finished.any():
        a.plot(x[finished], fin[finished], "o", color="tab:green", markersize=4, label="время финиша, с")
        best_t = np.fmin.accumulate(np.where(finished, fin, np.inf))
        best_t = np.where(np.isinf(best_t), np.nan, best_t)
        a.step(x, best_t, color="black", linewidth=1.2, where="post", label="лучшее время")
    else:
        a.text(0.5, 0.5, "финишей пока не было", ha="center", va="center", transform=a.transAxes)
    if ref_time:
        a.axhline(ref_time, color="tab:orange", linestyle="--", label=f"эталон {ref_time:g} с")
    a.set_title("Время финиша, с  (и доля финишей справа)")
    a.set_xlabel("заезд")
    a.set_ylabel("с")
    b = a.twinx()
    b.plot(x, moving_avg(finished.astype(float) * 100, window), color="tab:red", alpha=0.7, label="доля финишей, %")
    b.set_ylim(0, 105)
    b.set_ylabel("% финишей в окне")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = b.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, loc="upper right")

    # 4. причины конца заездов по группам
    a = ax[1, 1]
    group = max(10, n // 20)
    edges = list(range(0, n, group))
    reasons = [r for r, _ in Counter(r["reason"] for r in rows).most_common()]
    bottoms = np.zeros(len(edges))
    for reason in reasons:
        share = np.array([100.0 * sum(1 for r in rows[e:e + group] if r["reason"] == reason) / len(rows[e:e + group])
                          for e in edges])
        a.bar([e + group / 2 for e in edges], share, width=group * 0.95, bottom=bottoms, label=reason,
              color=REASON_COLORS.get(reason))
        bottoms += share
    a.set_title(f"Чем кончались заезды, % (группы по {group})")
    a.set_xlabel("заезд")
    a.set_ylim(0, 100)
    a.legend(loc="upper left", fontsize=8, ncol=2)

    # границы этапов: вертикальные линии на всех панелях
    for i in range(1, n):
        if stages[i] != stages[i - 1]:
            for a in ax.flat:
                a.axvline(i + 0.5, color="tab:red", linestyle=":", alpha=0.8)
            ax[0, 0].text(i + 0.5, ax[0, 0].get_ylim()[1], f" этап {stages[i]}", color="tab:red", va="top", fontsize=9)
    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def main():
    p = argparse.ArgumentParser(description="График обучения по training_log.csv")
    p.add_argument("--log", default="training_log.csv")
    p.add_argument("--out", default="training_progress.png")
    p.add_argument("--window", type=int, default=50, help="окно скользящего среднего (заездов)")
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
        fig = draw(rows, args.window, args.ref_time, plt)
        fig.savefig(args.out, dpi=110)
        print(f"[{time.strftime('%H:%M:%S')}] заездов {len(rows)} -> {args.out}")
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
