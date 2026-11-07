"""
train_forever.py - запускает tm_dqn.py и перезапускает его, если он упал. Для обучения "на всю ночь".

Что делает:
  * не даёт Windows уснуть и выключить экран, пока работает;
  * если tm_dqn.py завершился с ошибкой (потеря связи с игрой и т.п.) - ждёт и запускает снова;
    обучение продолжается с последнего сохранения;
  * если агент падает сразу при старте несколько раз подряд (игра не запущена, ошибка в коде) -
    останавливается, а не крутится вечно;
  * весь вывод дублирует в train_forever.log.

Перед запуском: игра запущена через TMInterface 1.4.3, трасса загружена, TMlogger.py закрыт.

Запуск (всё после своих параметров уходит в tm_dqn.py):
    python train_forever.py --reference reference.csv --git-push
Остановка: Ctrl+C (агент сохранится сам).

Игру этот скрипт НЕ перезапускает: если игра закрылась или зависла, обучение ждёт, пока ты её откроешь.
"""

import argparse
import os
import subprocess
import sys
import time

ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x00000001, 0x00000002


def keep_awake(on):
    """Windows: запретить сон и выключение экрана, пока скрипт работает."""
    try:
        import ctypes
        flags = ES_CONTINUOUS | ((ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED) if on else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass            # не Windows - ничего страшного


def main():
    p = argparse.ArgumentParser(description="Перезапускает tm_dqn.py при вылетах")
    p.add_argument("--agent", default="tm_dqn.py", help="какой скрипт запускать")
    p.add_argument("--pause", type=float, default=15.0, help="пауза перед перезапуском, с")
    p.add_argument("--quick-s", type=float, default=30.0, help="работа короче этого = быстрый вылет, с")
    p.add_argument("--max-quick-fails", type=int, default=5, help="столько быстрых вылетов подряд - остановиться")
    p.add_argument("--log", default="train_forever.log", help="файл журнала перезапусков")
    p.add_argument("--plot-every", type=float, default=300.0,
                   help="каждые столько секунд обновлять training_progress.png (0 - не рисовать)")
    args, agent_args = p.parse_known_args()
    cmd = [sys.executable, args.agent, "--live", *agent_args]

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(args.log, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    keep_awake(True)
    plotter = None
    if args.plot_every > 0 and os.path.exists("plot_progress.py"):
        plotter = subprocess.Popen([sys.executable, "plot_progress.py", "--watch", str(args.plot_every)])
        log(f"график обучения: training_progress.png (обновляется каждые {args.plot_every:g} с)")
    quick = 0
    try:
        while True:
            log("запуск: " + " ".join(cmd))
            t0 = time.time()
            proc = subprocess.Popen(cmd)
            try:
                code = proc.wait()
            except KeyboardInterrupt:
                log("Ctrl+C: останавливаю агента (он сохранится)")
                try:
                    proc.wait(timeout=200)
                except Exception:
                    proc.terminate()
                break
            ran = time.time() - t0
            log(f"агент завершился с кодом {code}, проработал {ran:.0f} с")
            if code == 0:
                log("код 0 - штатное завершение, перезапуск не нужен")
                break
            quick = quick + 1 if ran < args.quick_s else 0
            if quick >= args.max_quick_fails:
                log(f"{quick} быстрых вылетов подряд - останавливаюсь. Проверь: игра запущена? трасса загружена? "
                    "нет ошибки в коде (запусти tm_dqn.py вручную и посмотри текст)?")
                break
            log(f"перезапуск через {args.pause:.0f} с")
            try:
                time.sleep(args.pause)
            except KeyboardInterrupt:
                log("Ctrl+C во время паузы - выход")
                break
    finally:
        keep_awake(False)
        if plotter is not None:
            plotter.terminate()


if __name__ == "__main__":
    main()
