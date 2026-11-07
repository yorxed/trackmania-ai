"""
menu_skip.py - нажатие клавиш в окне игры, чтобы выйти из меню после финиша (только Windows).

Используется из tm_dqn.py (флаг --menu-keys), но можно запускать и отдельно, чтобы ПОДОБРАТЬ клавишу:
    1. Запусти игру, проедь трассу до финиша - появится меню.
    2. В терминале:   python menu_skip.py --keys enter --delay 5
    3. За 5 секунд кликни на окно игры (меню должно быть на экране). Скрипт нажмёт клавишу.
    4. Если заезд начался заново - эта клавиша подходит. Если нет - пробуй другую: esc, space, down,enter ...

Клавиши: enter, esc, space, delete, backspace, tab, up, down, left, right, a-z, 0-9. Через запятую - по очереди
(например "down,enter"): так можно выбрать пункт меню и подтвердить.

БЕЗОПАСНОСТЬ: клавиши отправляются, только если активное окно - игра (по заголовку окна). Иначе
Enter/Delete попали бы в VS Code или проводник. Если заголовок окна игры у тебя другой - укажи
--game-window "кусок заголовка".
"""

import argparse
import sys
import time

# (скан-код, расширенная клавиша). Игры на DirectInput читают скан-коды, поэтому шлём их.
SCANCODES = {
    "enter": (0x1C, False), "esc": (0x01, False), "space": (0x39, False), "backspace": (0x0E, False),
    "tab": (0x0F, False), "delete": (0x53, True), "up": (0x48, True), "down": (0x50, True),
    "left": (0x4B, True), "right": (0x4D, True),
}
LETTERS = {c: (sc, False) for c, sc in zip("qwertyuiop", range(0x10, 0x1A))}
LETTERS.update({c: (sc, False) for c, sc in zip("asdfghjkl", range(0x1E, 0x27))})
LETTERS.update({c: (sc, False) for c, sc in zip("zxcvbnm", range(0x2C, 0x33))})
LETTERS.update({c: (sc, False) for c, sc in zip("1234567890", range(0x02, 0x0C))})
SCANCODES.update(LETTERS)

DEFAULT_TITLES = ("trackmania", "tmforever", "tmnationsforever", "nations forever")
KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, KEYEVENTF_SCANCODE = 0x1, 0x2, 0x8


def parse_keys(text):
    """'down,enter' -> ['down', 'enter']; проверяет, что все клавиши известны."""
    keys = [k.strip().lower() for k in (text or "").split(",") if k.strip()]
    bad = [k for k in keys if k not in SCANCODES]
    if bad:
        raise ValueError(f"неизвестные клавиши: {bad}. Доступны: {sorted(SCANCODES)}")
    return keys


def foreground_title():
    """Заголовок активного окна ('' если не удалось или не Windows)."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        n = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        return buf.value
    except Exception:
        return ""


def game_is_foreground(extra_title=None):
    t = foreground_title().lower()
    titles = DEFAULT_TITLES + ((extra_title.lower(),) if extra_title else ())
    return any(s in t for s in titles)


def _tap(key, hold_s=0.08):
    import ctypes
    sc, ext = SCANCODES[key]
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_EXTENDEDKEY if ext else 0)
    ctypes.windll.user32.keybd_event(0, sc, flags, 0)
    time.sleep(hold_s)
    ctypes.windll.user32.keybd_event(0, sc, flags | KEYEVENTF_KEYUP, 0)


def press_sequence(keys, extra_title=None, gap_s=0.4):
    """Нажимает клавиши по очереди, но только если активно окно игры. Возвращает True, если нажал."""
    if not keys:
        return False
    if not sys.platform.startswith("win"):
        print("menu_skip: нажатие клавиш работает только на Windows")
        return False
    if not game_is_foreground(extra_title):
        print(f"menu_skip: активно не окно игры ('{foreground_title()}') - клавиши не отправляю")
        return False
    for k in keys:
        _tap(k)
        time.sleep(gap_s)
    return True


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Проверка клавиши для выхода из меню после финиша")
    p.add_argument("--keys", default="enter", help="клавиши через запятую, например enter или down,enter")
    p.add_argument("--delay", type=float, default=5.0, help="сколько секунд даётся, чтобы переключиться на игру")
    p.add_argument("--game-window", default=None, help="кусок заголовка окна игры, если он нестандартный")
    a = p.parse_args()
    ks = parse_keys(a.keys)
    print(f"Через {a.delay:g} с нажму: {ks}. Кликни на окно игры, меню должно быть открыто.")
    time.sleep(a.delay)
    print("нажато" if press_sequence(ks, a.game_window) else "не нажато (см. сообщение выше)")
