"""
plot_ghost.py - показывает на графике лучшую траекторию (призрак) агента и твой эталон.

Запуск (из папки проекта, рядом с tm_common.py):
    python plot_ghost.py                              # призрак из dqn_state.npz + эталон reference.csv (что найдётся)
    python plot_ghost.py --state dqn_state.npz        # только призрак агента
    python plot_ghost.py --reference reference.csv    # только эталон
    python plot_ghost.py --reference reference.csv --ref-episode 1
    python plot_ghost.py --save ghost.png             # сохранить картинку вместо окна

На левом графике вид сверху (x, z): старт - зелёная точка, конец - красная.
На правом - высота (y) вдоль пути: по нему видно трамплины и обрывы.
Если на графике обе линии, а время агента меньше времени эталона - агент уже побил твой эталон.
Нужен matplotlib:  python -m pip install matplotlib
"""

import argparse
import os

from tm_common import Ghost


def describe(name, g):
    t = "не финишировал" if not g.finished else f"финиш {g.finish_ms / 1000:.2f} с"
    return f"{name}: {g.length:.0f} м, {t}"


def main():
    p = argparse.ArgumentParser(description="График лучшей траектории")
    p.add_argument("--state", default="dqn_state.npz", help="файл сохранения агента")
    p.add_argument("--reference", default="reference.csv", help="CSV эталона от TMlogger.py")
    p.add_argument("--ref-episode", type=int, default=None, help="номер заезда в CSV (по умолчанию последний)")
    p.add_argument("--save", metavar="PNG", help="сохранить в файл вместо показа окна")
    args = p.parse_args()

    items = []   # (подпись, Ghost, цвет)
    if os.path.exists(args.state):
        g = Ghost.from_state_file(args.state)
        if g is None:
            print(f"В {args.state} призрака нет.")
        else:
            items.append((describe("Призрак агента", g), g, "tab:blue"))
    else:
        print(f"Файла {args.state} нет - пропускаю призрак агента.")
    if os.path.exists(args.reference):
        g = Ghost.from_csv(args.reference, args.ref_episode)
        items.append((describe("Эталон", g), g, "tab:orange"))
    else:
        print(f"Файла {args.reference} нет - пропускаю эталон.")
    if not items:
        raise SystemExit("Нечего рисовать: нет ни призрака агента, ни эталона.")

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ay) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [3, 2]})
    for label, g, color in items:
        ax.plot(g.raw_x, g.raw_z, color=color, label=label, linewidth=1.8)
        ax.scatter(g.raw_x[0], g.raw_z[0], color="green", s=60, zorder=5)
        ax.scatter(g.raw_x[-1], g.raw_z[-1], color="red", s=60, zorder=5)
        if g.y is not None:
            ay.plot(g.s, g.y, color=color, label=label)
        print(label)
    ax.set_xlabel("X, м")
    ax.set_ylabel("Z, м")
    ax.set_title("Лучшая траектория (вид сверху). Зелёная точка - старт, красная - конец")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend()
    ay.set_xlabel("пройдено вдоль пути, м")
    ay.set_ylabel("высота Y, м")
    ay.set_title("Высота вдоль пути")
    ay.grid(alpha=0.3)
    if ay.has_data():
        ay.legend()
    else:
        ay.text(0.5, 0.5, "в этом призраке нет высоты", ha="center", va="center", transform=ay.transAxes)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=130)
        print("Сохранено:", args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
