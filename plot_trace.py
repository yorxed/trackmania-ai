"""
plot_trace.py - разбор одного заезда агента: что он делал и почему остановился.

Как получить данные: запусти оценку с записью трассы (один заезд, игра детерминирована):
    python tm_dqn.py --live --play --play-episodes 1 --trace trace.csv --reference reference.csv
Потом:
    python plot_trace.py                                  # trace.csv + reference.csv -> trace.png (и окно)
    python plot_trace.py --zoom 120 --last-s 8            # увеличить 120 м вокруг конца, показать последние 8 с
    python plot_trace.py --episode 1 --no-show

Панели:
  1. Вид сверху: эталон (оранжевый) и путь агента, цвет точек = скорость. Красный крест - где заезд закончился.
  2. То же, увеличено вокруг конца заезда (--zoom метров): видно, упёрся ли агент в стену.
  3. Скорость по трассе: у агента и у тебя (эталон) на тех же метрах. Видно, где агент едет медленнее тебя.
  4. Последние --last-s секунд: руль, газ, тормоз и скорость. Видно, что агент ДЕЛАЛ перед остановкой.
В консоль печатается краткая сводка последних секунд (какие действия выбирал, какая была скорость).
"""

import argparse
import csv
import math
from collections import Counter

import numpy as np

ACTION_NAMES_V2 = ["газ прямо", "газ влево 1/2", "газ вправо 1/2", "газ влево", "газ вправо",
                   "накат прямо", "тормоз", "накат влево", "накат вправо"]          # старый набор из 9 действий
ACTION_NAMES = ["газ прямо", "газ влево 0.15", "газ вправо 0.15", "газ влево 0.3", "газ вправо 0.3",
                "газ влево 0.6", "газ вправо 0.6", "газ влево до упора", "газ вправо до упора",
                "накат прямо", "тормоз", "накат влево", "накат вправо"]            # текущий набор из 13


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description="Разбор заезда агента по trace.csv")
    p.add_argument("--trace", default="trace.csv")
    p.add_argument("--reference", default="reference.csv")
    p.add_argument("--episode", type=int, default=None, help="номер заезда в trace.csv (по умолчанию последний)")
    p.add_argument("--zoom", type=float, default=100.0, help="радиус увеличенной панели вокруг конца, м")
    p.add_argument("--last-s", type=float, default=6.0, help="сколько последних секунд показать на панели 4")
    p.add_argument("--out", default="trace.png")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    rows = read_csv(args.trace)
    if not rows:
        raise SystemExit(f"{args.trace}: пустой файл")
    ep = max(int(r["episode"]) for r in rows) if args.episode is None else args.episode
    rows = [r for r in rows if int(r["episode"]) == ep]
    if len(rows) < 3:
        raise SystemExit(f"В {args.trace} нет заезда {ep} (или в нём меньше трёх решений)")
    t = np.array([float(r["t_ms"]) for r in rows]) / 1000.0
    x, z = np.array([float(r["x"]) for r in rows]), np.array([float(r["z"]) for r in rows])
    spd = np.array([float(r["speed_kmh"]) for r in rows])
    prog = np.array([float(r["progress_m"]) for r in rows])
    act = np.array([int(r["action"]) for r in rows])
    steer = np.array([float(r["steer"]) for r in rows]) / 65536.0
    gas, brk = np.array([int(r["gas"]) for r in rows]), np.array([int(r["brake"]) for r in rows])

    ref = None
    try:
        rr = read_csv(args.reference)
        rx, rz = np.array([float(r["pos_x"]) for r in rr]), np.array([float(r["pos_z"]) for r in rr])
        rs = np.array([float(r["speed"]) for r in rr])
        rdist = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(rx), np.diff(rz)))])
        ref = (rx, rz, rs, rdist)
    except (OSError, KeyError, ValueError):
        print(f"Эталон {args.reference} не прочитан - рисую только заезд агента.")

    dist = None
    if ref:
        grid = np.arange(0.0, ref[3][-1], 1.0)                      # эталон, пересчитанный через каждый метр
        gx, gz = np.interp(grid, ref[3], ref[0]), np.interp(grid, ref[3], ref[1])
        dist = np.array([np.sqrt(((gx - px) ** 2 + (gz - pz) ** 2).min()) for px, pz in zip(x, z)])

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    end_x, end_z = x[-1], z[-1]
    fig.suptitle(f"Заезд агента: {len(rows)} решений, {t[-1]:.1f} с, прогресс {prog.max():.0f} м, "
                 f"конец на {prog[-1]:.0f} м, скорость в конце {spd[-1]:.0f} км/ч", fontsize=13)

    for a, zoom in ((ax[0, 0], False), (ax[0, 1], True)):
        if ref:
            a.plot(ref[0], ref[1], color="tab:orange", linewidth=1.5, label="эталон")
        sc = a.scatter(x, z, c=spd, s=10, cmap="viridis", zorder=3)
        a.scatter([end_x], [end_z], marker="x", color="red", s=140, linewidths=3, zorder=5, label="конец заезда")
        a.set_aspect("equal", adjustable="box" if zoom else "datalim")
        a.grid(alpha=0.3)
        if zoom:
            a.set_xlim(end_x - args.zoom, end_x + args.zoom)
            a.set_ylim(end_z - args.zoom, end_z + args.zoom)
            a.set_title(f"Вокруг конца заезда (±{args.zoom:g} м)")
        else:
            a.set_title("Вид сверху (цвет = скорость, км/ч)")
            a.legend(loc="best")
        a.set_xlabel("X, м")
        a.set_ylabel("Z, м")
    fig.colorbar(sc, ax=ax[0, 1], label="скорость, км/ч", shrink=0.8)

    a = ax[1, 0]
    a.plot(prog, spd, color="tab:blue", label="агент")
    if ref:
        a.plot(ref[3], ref[2], color="tab:orange", label="эталон (ты)")
    a.axvline(prog[-1], color="red", linestyle=":", label="конец заезда")
    a.set_title("Скорость вдоль трассы (красная линия справа - расстояние до эталона)")
    a.set_xlabel("прогресс, м")
    a.set_ylabel("км/ч")
    a.grid(alpha=0.3)
    h1, l1 = a.get_legend_handles_labels()
    if dist is not None:
        d2 = a.twinx()
        d2.plot(prog, dist, color="tab:red", alpha=0.7, label="расстояние до эталона, м")
        d2.axhline(16, color="tab:red", linestyle="--", alpha=0.4)     # примерная полуширина трассы
        d2.set_ylabel("м от эталона")
        h2, l2 = d2.get_legend_handles_labels()
        h1, l1 = h1 + h2, l1 + l2
    a.legend(h1, l1, loc="upper left")

    a = ax[1, 1]
    m = t >= t[-1] - args.last_s
    tt = t[m] - t[-1]
    a.plot(tt, steer[m], color="tab:purple", label="руль (-1..1)")
    a.fill_between(tt, 0, gas[m] * 0.5, step="mid", color="tab:green", alpha=0.4, label="газ")
    a.fill_between(tt, 0, -brk[m] * 0.5, step="mid", color="tab:red", alpha=0.4, label="тормоз")
    a.set_ylim(-1.1, 1.1)
    a.set_xlabel("секунды до конца заезда")
    a.set_ylabel("ввод")
    b = a.twinx()
    b.plot(tt, spd[m], color="gray", linewidth=2, label="скорость")
    b.set_ylabel("км/ч")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = b.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    a.set_title(f"Последние {args.last_s:g} с: что делал агент")
    a.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3.0)
    fig.savefig(args.out, dpi=105)
    print("Сохранено:", args.out)

    # сводка последних секунд
    k = t >= t[-1] - 3.0
    names = ACTION_NAMES_V2 if "q12" not in rows[0] else ACTION_NAMES
    cnt = Counter(int(v) for v in act[k])
    if dist is not None:
        far = np.where(dist > 20.0)[0]
        print(f"\nРасстояние до эталона: максимум {dist.max():.0f} м, в конце {dist[-1]:.0f} м.")
        if len(far):
            i = int(far[0])
            print(f"Дальше 20 м от эталона агент впервые оказался на {prog[i]:.0f} м прогресса (t = {t[i]:.1f} с, "
                  f"скорость {spd[i]:.0f} км/ч).")
        else:
            print("Дальше 20 м от эталона агент не уходил.")
    print(f"\nПоследние 3 с (решений {int(k.sum())}): скорость {spd[k].min():.0f}-{spd[k].max():.0f} км/ч, "
          f"сместился на {math.hypot(x[k][-1] - x[k][0], z[k][-1] - z[k][0]):.1f} м")
    print("Выбранные действия: " + ", ".join(f"{names[a]} {100 * c / k.sum():.0f}%" for a, c in cnt.most_common()))
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()