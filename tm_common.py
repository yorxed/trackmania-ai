"""
tm_common.py - общая часть для логгера (TMlogger.py) и агента (tm_dqn.py).

Здесь живёт всё, что превращает данные игры в то, что видит сеть:
  * RawState             - сырое состояние машины из игры (то, что пишет логгер в CSV)
  * build_observation    - из RawState делает вектор из 16 чисел для нейросети
  * Ghost                - траектория (путь лучшего заезда), от которой считаются прогресс и 6 признаков
  * ObservationBuilder   - собирает всё вместе для одного заезда

Зачем отдельный файл: одно и то же наблюдение нужно и агенту, и логгеру (чтобы записывать в CSV
ровно то, что видит сеть). Если бы код был в двух местах, они бы рано или поздно разошлись.

ВСЕ настройки нормализации и конвенции угла yaw - ТОЛЬКО здесь. Меняешь тут - меняется везде.
Положи этот файл в одну папку с TMlogger.py и tm_dqn.py.
"""

import csv
import math

import numpy as np

# =============================================================================
# НАСТРОЙКИ НАБЛЮДЕНИЯ
# =============================================================================

# --- Нормализация (подбери по своим данным из CSV) ---
POS_SCALE = 400.0      # м: смещение от СТАРТА делим на это число (≈ макс. расстояние от старта до любой точки трассы)
POS_CLIP = 3.0         # обрезка смещения: вход не уходит дальше ±POS_CLIP
V_MAX = 100.0          # м/с: скорость вперёд (~360 км/ч)
V_SIDE_MAX = 20.0      # м/с: боковой снос
V_UP_MAX = 30.0        # м/с: вертикальная скорость
STEER_MAX = 65536      # диапазон руля в TMInterface: [-65536, 65536]

# --- Соглашение об угле yaw (ПРОВЕРЬ на своих данных) ---
# Курс машины h = YAW_SIGN * yaw + YAW_OFFSET; "вперёд" в плоскости (x, z) = (sin h, cos h).
YAW_SIGN, YAW_OFFSET, SIDE_SIGN = 1.0, 0.0, 1.0

# --- Призрак ---
GHOST_SPACING = 1.0                      # м: шаг, с которым путь пересчитывается равномерно
GHOST_BACK, GHOST_FWD = 20, 80           # окно поиска ближайшей точки (в точках), защита от скачков
GHOST_ANGLE_SPAN = 3                     # точек в каждую сторону для оценки направления (сглаживание)
# Взгляд вперёд зависит от СКОРОСТИ: смотрим туда, куда машина приедет через 0.5, 1 и 2 секунды.
# На 100 м/с (360 км/ч) это 50, 100 и 200 м; на малой скорости не ближе GHOST_LOOKAHEAD_MIN_M.
GHOST_LOOKAHEAD_S = (0.5, 1.0, 2.0)
GHOST_LOOKAHEAD_MIN_M = (10.0, 20.0, 40.0)
GHOST_LOOKAHEAD_MAX_M = 300.0
GHOST_HALF_WIDTH = 16.0                  # м: масштаб бокового смещения от призрака


def wrap_angle(a):
    """Приводит угол в диапазон [-pi, pi]. Нужен из-за скачка 3.14 -> -3.14."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# =============================================================================
# СЫРОЕ СОСТОЯНИЕ
# =============================================================================

class RawState:
    """Состояние машины в один момент. Те же величины, что пишет логгер в CSV (кроме времени)."""

    def __init__(self, speed, pos, vel, yaw, pitch, roll, prev_steer=0.0):
        self.speed = speed        # display_speed (км/ч, как на спидометре)
        self.pos = pos            # (x, y, z)
        self.vel = vel            # (vx, vy, vz), м/с, мировые оси
        self.yaw, self.pitch, self.roll = yaw, pitch, roll
        self.prev_steer = prev_steer

    @staticmethod
    def from_sim_state(state, prev_steer=0.0):
        """Из объекта SimStateData, который отдаёт iface.get_simulation_state()."""
        ypr = state.yaw_pitch_roll
        return RawState(
            speed=float(state.display_speed),
            pos=tuple(float(v) for v in state.position),
            vel=tuple(float(v) for v in state.velocity),
            yaw=float(ypr[0]), pitch=float(ypr[1]), roll=float(ypr[2]),
            prev_steer=float(prev_steer),
        )


# =============================================================================
# НАБЛЮДЕНИЕ
# =============================================================================

OBS_NAMES = ["v_forward", "v_side", "v_up", "sin_heading", "cos_heading",
             "dx_from_start", "dz_from_start", "pitch", "roll", "prev_steer",
             "sin_ghost_err", "cos_ghost_err", "ghost_lateral",
             "ghost_turn_0.5s", "ghost_turn_1s", "ghost_turn_2s"]
OBS_SIZE = len(OBS_NAMES)
NO_GHOST_FEATURES = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)   # пока призрака нет: ошибки курса нет (cos = 1)


def build_observation(raw: RawState, origin=(0.0, 0.0), ghost_feats=NO_GHOST_FEATURES):
    """
    16 чисел примерно в [-1, 1]. Порядок фиксирован: сеть различает входы только по позиции в векторе.
      0-4   скорость вперёд, скорость вбок, вертикальная скорость, sin и cos курса
      5-6   смещение от точки старта заезда (dx, dz)
      7-9   наклон, крен, прошлый руль
      10-15 признаки относительно призрака (см. Ghost.features)
    """
    h = YAW_SIGN * raw.yaw + YAW_OFFSET
    sh, ch = math.sin(h), math.cos(h)
    vx, vy, vz = raw.vel
    v_forward = vx * sh + vz * ch
    v_side = SIDE_SIGN * (vx * ch - vz * sh)
    clip = lambda v: max(-1.0, min(1.0, v))
    pclip = lambda v: max(-POS_CLIP, min(POS_CLIP, v))
    return np.array([
        clip(v_forward / V_MAX), clip(v_side / V_SIDE_MAX), clip(vy / V_UP_MAX),
        sh, ch,
        pclip((raw.pos[0] - origin[0]) / POS_SCALE), pclip((raw.pos[2] - origin[1]) / POS_SCALE),
        raw.pitch / math.pi, raw.roll / math.pi, clip(raw.prev_steer / STEER_MAX),
        *ghost_feats,
    ], dtype=np.float32)


# =============================================================================
# ПРИЗРАК
# =============================================================================

class Ghost:
    """Путь лучшего заезда. Прогресс = расстояние вдоль него от старта."""

    def __init__(self, xs, zs, finished=False, finish_ms=None, ys=None):
        xs, zs = np.asarray(xs, float), np.asarray(zs, float)
        ys = None if ys is None or len(ys) != len(xs) else np.asarray(ys, float)   # ys - высота (pos_y), необязательна
        self.raw_x, self.raw_z, self.raw_y = xs, zs, ys
        self.finished, self.finish_ms = bool(finished), finish_ms
        keep = [0]
        for i in range(1, len(xs)):
            if math.hypot(xs[i] - xs[keep[-1]], zs[i] - zs[keep[-1]]) > 0.5:
                keep.append(i)
        xs, zs = xs[keep], zs[keep]
        ys = None if ys is None else ys[keep]
        if len(xs) < 3:                       # заезд почти не двигался
            xs = np.array([xs[0], xs[0] + 1.0, xs[0] + 2.0])
            zs = np.array([zs[0], zs[0], zs[0]])
            ys = None
        seg = np.hypot(np.diff(xs), np.diff(zs))
        s = np.concatenate([[0.0], np.cumsum(seg)])
        grid = np.arange(0.0, s[-1], GHOST_SPACING)
        self.x, self.z, self.s = np.interp(grid, s, xs), np.interp(grid, s, zs), grid
        self.y = None if ys is None else np.interp(grid, s, ys)
        self.n, self.length, self.last_idx = len(grid), float(s[-1]), 0
        i = np.arange(self.n)
        a, b = np.maximum(i - GHOST_ANGLE_SPAN, 0), np.minimum(i + GHOST_ANGLE_SPAN, self.n - 1)
        self.angle = np.arctan2(self.x[b] - self.x[a], self.z[b] - self.z[a])

    @staticmethod
    def from_state_file(path):
        """Загрузить призрак из файла сохранения агента (dqn_state.npz). None, если призрака там нет."""
        d = np.load(path, allow_pickle=False)
        if "ghost_x" not in d.files or d["ghost_x"].size < 3:
            return None
        ms = int(d["ghost_ms"])
        ys = d["ghost_y"] if "ghost_y" in d.files else None          # старые сохранения без высоты тоже читаются
        return Ghost(d["ghost_x"], d["ghost_z"], bool(int(d["ghost_fin"])), None if ms < 0 else ms, ys)

    @staticmethod
    def from_csv(path, episode=None):
        """
        Эталонный призрак из CSV логгера (TMlogger.py): твой собственный заезд.
        episode - номер заезда в файле (столбец episode); по умолчанию последний.
        Заезд считается финишировавшим, время финиша = время последней строки заезда.
        Берись за заезд, который ты довёл до финиша и в котором логгер писал до самого конца!
        """
        rows = {}
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.setdefault(int(r["episode"]), []).append(r)
        if not rows:
            raise ValueError(f"{path}: файл пустой")
        ep = max(rows) if episode is None else episode
        if ep not in rows:
            raise ValueError(f"{path}: нет заезда {ep}; в файле есть: {sorted(rows)}")
        rs = rows[ep]
        return Ghost([float(r["pos_x"]) for r in rs], [float(r["pos_z"]) for r in rs], True,
                     int(float(rs[-1]["time_ms"])), [float(r["pos_y"]) for r in rs])

    def key(self):
        """Больше = лучше: финишировавший > нефинишировавшего; среди финишировавших быстрее; иначе длиннее."""
        return (1, -self.finish_ms) if self.finished else (0, self.length)

    def reset(self):
        self.last_idx = 0

    def height(self):
        """Высота призрака (y) в ближайшей к машине точке (после progress_at). None, если высоты нет."""
        return None if self.y is None else float(self.y[self.last_idx])

    def progress_at(self, x, z):
        """Пройденное вдоль призрака расстояние (м) для точки (x, z). Ищет ближайшую точку в окне вокруг прошлой."""
        lo, hi = max(0, self.last_idx - GHOST_BACK), min(self.n, self.last_idx + GHOST_FWD)
        d2 = (self.x[lo:hi] - x) ** 2 + (self.z[lo:hi] - z) ** 2
        idx = lo + int(np.argmin(d2))
        self.last_idx = idx
        p = float(self.s[idx])
        if idx == self.n - 1:                 # уехали дальше конца призрака - засчитываем и это
            p += math.hypot(x - self.x[-1], z - self.z[-1])
        return p

    def lateral_m(self, x, z):
        """Боковое смещение от призрака, метры (со знаком). Вызывать после progress_at."""
        idx = self.last_idx
        th = self.angle[idx]
        return SIDE_SIGN * ((x - self.x[idx]) * math.cos(th) - (z - self.z[idx]) * math.sin(th))

    def features(self, x, z, h, speed=0.0):
        """6 признаков относительно призрака. Вызывать после progress_at (он выбирает ближайшую точку).
        speed - горизонтальная скорость машины, м/с: от неё зависит, как далеко вперёд смотрим на поворот."""
        idx = self.last_idx
        th = self.angle[idx]
        err = wrap_angle(th - h)
        lateral = self.lateral_m(x, z)
        dists = [min(GHOST_LOOKAHEAD_MAX_M, max(dmin, speed * t))
                 for t, dmin in zip(GHOST_LOOKAHEAD_S, GHOST_LOOKAHEAD_MIN_M)]
        ahead = [wrap_angle(self.angle[min(self.n - 1, idx + int(round(d / GHOST_SPACING)))] - h) for d in dists]
        return (math.sin(err), math.cos(err), max(-1.0, min(1.0, lateral / GHOST_HALF_WIDTH)),
                ahead[0] / math.pi, ahead[1] / math.pi, ahead[2] / math.pi)


class ObservationBuilder:
    """
    Собирает наблюдение для ОДНОГО заезда: помнит точку старта и (если есть) призрак.
    build(raw) возвращает (obs, progress):
        obs      - вектор из OBS_SIZE чисел
        progress - пройденное расстояние (м): вдоль призрака, а если призрака нет - от точки старта
    На каждом новом заезде вызывай reset().
    """

    def __init__(self, ghost=None):
        self.ghost = ghost
        self.start = None
        self.reset()

    def reset(self):
        self.start = None
        if self.ghost is not None:
            self.ghost.reset()

    def build(self, raw):
        x, z = raw.pos[0], raw.pos[2]
        if self.start is None:
            self.start = (x, z)
        h = YAW_SIGN * raw.yaw + YAW_OFFSET
        feats = NO_GHOST_FEATURES
        if self.ghost is not None:
            progress = self.ghost.progress_at(x, z)
            feats = self.ghost.features(x, z, h, math.hypot(raw.vel[0], raw.vel[2]))
        else:
            progress = math.hypot(x - self.start[0], z - self.start[1])
        return build_observation(raw, self.start, feats), progress