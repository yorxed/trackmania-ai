"""
TMlogger.py - запись телеметрии из Trackmania Nations Forever (TMInterface 1.4.x).

Подключается к игре и раз в LOG_INTERVAL_MS пишет в CSV:
  * 14 исходных столбцов (как раньше): time_ms, speed, pos_x/y/z, vel_x/y/z, yaw, pitch, roll, gas, brake, steer
  * episode           - номер заезда (растёт, когда время гонки пошло назад: рестарт / give up)
  * 16 столбцов obs_* - ТО ЖЕ наблюдение, которое видит нейросеть (считается в tm_common.py)

Так по CSV видно, что именно получает агент, и можно проверить нормализацию и конвенцию yaw.

Как запускать:
  1. Запусти игру через TMInterface 1.4.3 и начни заезд.
  2. python TMlogger.py                                  - просто писать телеметрию и наблюдение
     python TMlogger.py --ghost dqn_state.npz            - считать признаки относительно призрака агента
     python TMlogger.py --append                         - дописывать в существующий файл, а не стирать
     python TMlogger.py --show-obs                       - печатать в консоль и вектор наблюдения
     python TMlogger.py --test-drive                     - сам жмёт газ (проверка, что управление работает)
  3. Поезжай вручную. Остановка: Ctrl+C.

ВАЖНО: один сервер TMInterface обслуживает одного клиента. Пока работает логгер, агент (tm_dqn.py)
подключиться не сможет, и наоборот. Запускай по очереди.

Нужен файл tm_common.py в той же папке.
"""

import argparse
import csv
import os

from tminterface.client import Client, run_client
from tminterface.interface import TMInterface

from tm_common import Ghost, ObservationBuilder, OBS_NAMES, RawState

RAW_COLUMNS = ["time_ms", "speed", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z",
               "yaw", "pitch", "roll", "gas", "brake", "steer"]
HEADER = RAW_COLUMNS + ["episode"] + [f"obs_{n}" for n in OBS_NAMES]


class LoggerClient(Client):
    def __init__(self, out_path, interval_ms, ghost, append, show_obs, test_drive):
        super().__init__()
        self.interval_ms, self.show_obs, self.test_drive = interval_ms, show_obs, test_drive
        self.builder = ObservationBuilder(ghost)
        self.last_t = None
        self.episode = 0
        new_file = not (append and os.path.exists(out_path) and os.path.getsize(out_path) > 0)
        self.csv_file = open(out_path, "w" if new_file else "a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.csv_file)
        if new_file:
            self.writer.writerow(HEADER)

    def on_registered(self, iface: TMInterface):
        print(f"Подключено к {iface.server_name}")
        iface.log("Python logger connected", "success")

    def on_run_step(self, iface: TMInterface, _time: int):
        # _time - игровое время гонки в мс. До старта оно отрицательное.
        if self.last_t is not None and _time < self.last_t:     # время пошло назад = новый заезд
            self.episode += 1
            self.builder.reset()
        self.last_t = _time
        if _time < 0:
            return

        if self.test_drive:
            iface.set_input_state(accelerate=True, steer=0)     # ввод применяется на следующем тике

        if _time % self.interval_ms != 0:
            return

        state = iface.get_simulation_state()
        raw = RawState.from_sim_state(state)
        raw.prev_steer = state.input_steer                       # руль, который сейчас нажат
        obs, _progress = self.builder.build(raw)

        pos, vel = state.position, state.velocity
        row = [
            _time, state.display_speed,
            round(pos[0], 3), round(pos[1], 3), round(pos[2], 3),
            round(vel[0], 3), round(vel[1], 3), round(vel[2], 3),
            round(raw.yaw, 4), round(raw.pitch, 4), round(raw.roll, 4),
            int(state.input_accelerate), int(state.input_brake), state.input_steer,
            self.episode,
            *[round(float(v), 4) for v in obs],
        ]
        self.writer.writerow(row)
        self.csv_file.flush()

        print(f"эп={self.episode} t={_time:6d} мс | скорость={state.display_speed:4d} | "
              f"поз=({row[2]:8.2f}, {row[3]:7.2f}, {row[4]:8.2f}) | yaw={row[8]:6.2f} | "
              f"газ={row[11]} тормоз={row[12]} руль={row[13]}")
        if self.show_obs:
            print("   obs:", " ".join(f"{n}={v:+.2f}" for n, v in zip(OBS_NAMES, obs)))

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        print(f"Чекпоинт: {current}/{target}")

    def on_deregistered(self, iface: TMInterface):
        print("Клиент отключён")
        self.close()

    def on_shutdown(self, iface: TMInterface):
        print("Игра закрыта")
        self.close()

    def close(self):
        if not self.csv_file.closed:
            self.csv_file.close()


def main():
    p = argparse.ArgumentParser(description="Запись телеметрии и наблюдения Trackmania")
    p.add_argument("--out", default="telemetry.csv", help="файл CSV (по умолчанию telemetry.csv)")
    p.add_argument("--interval", type=int, default=100, help="как часто писать, мс (кратно 10)")
    p.add_argument("--ghost", metavar="NPZ", help="файл сохранения агента (dqn_state.npz): взять оттуда призрак")
    p.add_argument("--append", action="store_true", help="дописывать в существующий файл, а не перезаписывать")
    p.add_argument("--show-obs", action="store_true", help="печатать вектор наблюдения в консоль")
    p.add_argument("--test-drive", action="store_true", help="сам жмёт газ")
    p.add_argument("--server", default="TMInterface0", help="имя сервера TMInterface")
    args = p.parse_args()

    ghost = None
    if args.ghost:
        ghost = Ghost.from_state_file(args.ghost)
        print("Призрак загружен." if ghost else "В файле нет призрака: признаки призрака будут нулевыми.")

    print(f"Подключаюсь к {args.server}...")
    client = LoggerClient(args.out, args.interval, ghost, args.append, args.show_obs, args.test_drive)
    try:
        run_client(client, args.server)
    finally:
        client.close()


if __name__ == "__main__":
    main()