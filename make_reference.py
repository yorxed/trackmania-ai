"""
make_reference.py - вырезает один заезд из telemetry.csv и сохраняет его в ОТДЕЛЬНЫЙ файл эталона.

Зачем: telemetry.csv перезаписывается при каждом запуске TMlogger.py. Эталон должен лежать
в своём файле (reference.csv), который логгер не трогает.

Примеры (из папки проекта):
    python make_reference.py                          # последний заезд из telemetry.csv -> reference.csv
    python make_reference.py --episode 1              # заезд с номером 1
    python make_reference.py --finish-ms 30000        # обрезать всё после 30.000 с (хвост после финиша)
    python make_reference.py --src telemetry.csv --out reference.csv --episode 1 --finish-ms 30000
    python make_reference.py --list                   # показать, какие заезды есть в файле
    python make_reference.py --force                  # перезаписать уже существующий эталон

По умолчанию существующий reference.csv НЕ перезаписывается - чтобы случайно не потерять эталон.
Запуск агента потом: python tm_dqn.py --live --reference reference.csv
"""

import argparse
import csv
import math
import os


def main():
    p = argparse.ArgumentParser(description="Сохранить заезд из telemetry.csv как эталон")
    p.add_argument("--src", default="telemetry.csv", help="откуда брать (по умолчанию telemetry.csv)")
    p.add_argument("--out", default="reference.csv", help="куда сохранить (по умолчанию reference.csv)")
    p.add_argument("--episode", type=int, default=None, help="номер заезда (по умолчанию последний)")
    p.add_argument("--finish-ms", type=int, default=None, help="обрезать заезд по этому времени (мс): всё позже отбрасывается")
    p.add_argument("--list", action="store_true", help="только показать заезды в файле")
    p.add_argument("--force", action="store_true", help="перезаписать существующий эталон")
    args = p.parse_args()

    with open(args.src, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = {}
        for r in reader:
            rows.setdefault(int(r["episode"]), []).append(r)
    if not rows:
        raise SystemExit(f"{args.src}: файл пустой")

    if args.list:
        for ep, rs in sorted(rows.items()):
            print(f"заезд {ep}: строк {len(rs)}, последнее время {int(float(rs[-1]['time_ms'])) / 1000:.2f} с")
        return

    ep = max(rows) if args.episode is None else args.episode
    if ep not in rows:
        raise SystemExit(f"Нет заезда {ep}. Есть: {sorted(rows)}")
    rs = rows[ep]
    if args.finish_ms is not None:
        rs = [r for r in rs if int(float(r["time_ms"])) <= args.finish_ms]
    if len(rs) < 3:
        raise SystemExit("В заезде слишком мало строк - проверь номер заезда и --finish-ms")

    if os.path.exists(args.out) and not args.force:
        raise SystemExit(f"{args.out} уже существует - не перезаписываю. Добавь --force, если так и надо.")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rs:
            r = dict(r)
            r["episode"] = 0             # в эталоне ровно один заезд
            w.writerow(r)

    xs, zs = [float(r["pos_x"]) for r in rs], [float(r["pos_z"]) for r in rs]
    length = sum(math.hypot(xs[i] - xs[i - 1], zs[i] - zs[i - 1]) for i in range(1, len(xs)))
    print(f"Эталон сохранён: {args.out}")
    print(f"  заезд {ep} из {args.src}, строк {len(rs)}, длина ~{length:.0f} м, время {int(float(rs[-1]['time_ms'])) / 1000:.2f} с")


if __name__ == "__main__":
    main()
