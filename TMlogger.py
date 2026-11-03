"""
Диагностический скрипт: подключается к TMInterface 1.4.x и выводит состояние машины.

Как запускать:
  1. Запусти игру через TMInterface 1.4.3 и начни заезд на любой трассе.
  2. В отдельном окне: python tm_logger.py
  3. Поезжай вручную и смотри, как меняются числа. Ctrl+C - остановка.

Данные пишутся в консоль и в файл telemetry.csv (рядом со скриптом).
"""

import csv
import sys

from tminterface.client import Client, run_client
from tminterface.interface import TMInterface

# ---- Настройки ----
LOG_INTERVAL_MS = 100      # как часто писать (кратно 10 мс)
CSV_PATH = "telemetry.csv"
PRINT_TO_CONSOLE = True    # печатать строки в консоль Python
TEST_DRIVE = False         # True: скрипт сам нажмёт газ (проверка, что управление работает)
SERVER_NAME = "TMInterface0"


class LoggerClient(Client):
    def __init__(self):
        super().__init__()
        self.csv_file = open(CSV_PATH, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow([
            "time_ms", "speed",
            "pos_x", "pos_y", "pos_z",
            "vel_x", "vel_y", "vel_z",
            "yaw", "pitch", "roll",
            "gas", "brake", "steer",
        ])

    def on_registered(self, iface: TMInterface):
        print(f"Подключено к {iface.server_name}")
        iface.log("Python logger connected", "success")

    def on_run_step(self, iface: TMInterface, _time: int):
        # _time - игровое время гонки в мс. До старта оно может быть отрицательным.
        if _time < 0:
            return

        if TEST_DRIVE:
            # По документации ввод применяется на следующем физическом тике.
            iface.set_input_state(accelerate=True, steer=0)

        if _time % LOG_INTERVAL_MS != 0:
            return

        state = iface.get_simulation_state()

        speed = state.display_speed
        pos = state.position
        vel = state.velocity
        ypr = state.yaw_pitch_roll

        row = [
            _time, speed,
            round(pos[0], 3), round(pos[1], 3), round(pos[2], 3),
            round(vel[0], 3), round(vel[1], 3), round(vel[2], 3),
            round(float(ypr[0]), 4), round(float(ypr[1]), 4), round(float(ypr[2]), 4),
            int(state.input_accelerate), int(state.input_brake), state.input_steer,
        ]
        self.writer.writerow(row)
        self.csv_file.flush()

        if PRINT_TO_CONSOLE:
            print(
                f"t={_time:6d} мс | скорость={speed:4d} | "
                f"поз=({row[2]:8.2f}, {row[3]:7.2f}, {row[4]:8.2f}) | "
                f"yaw={row[8]:6.2f} | газ={row[11]} тормоз={row[12]} руль={row[13]}"
            )

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
    server_name = f"TMInterface{sys.argv[1]}" if len(sys.argv) > 1 else SERVER_NAME
    print(f"Подключаюсь к {server_name}...")
    client = LoggerClient()
    try:
        run_client(client, server_name)
    finally:
        client.close()


if __name__ == "__main__":
    main()