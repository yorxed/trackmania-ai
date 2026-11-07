"""
tm_dqn.py - DQN для Trackmania Nations Forever (TMInterface 1.4.x).

Это объединение двух предыдущих файлов:
  * из tm_agent.py:     DQN (сеть Q, буфер опыта, целевая сеть, ε-жадность), таблица действий;
  * из tm_evolution.py: траектория, которую агент создаёт САМ ("призрак" = его лучший заезд),
                        этапы обучения по условиям, сохранение между запусками.

Как это устроено
  Агент (одна сеть) ездит заезд за заездом. На каждом решении (раз в 50 мс):
     наблюдение s -> сеть -> действие a -> игра -> награда r и новое наблюдение s2.
  Переходы (s, a, r, s2, done) копятся в буфере, и по случайным пачкам из него сеть учится.

  Награда за шаг (считается из телеметрии и призрака, эталона из файла нет):
     + прирост пройденного ВДОЛЬ призрака          (главная часть)
     - маленький штраф за каждый шаг               (чтобы не стоять)
     + бонус за чекпоинт и за финиш
     - штраф, если застряла или перестала продвигаться
  Призрак - путь лучшего заезда на данный момент. Заезд лучше, если он финишировал, а среди
  финишировавших - быстрее; среди нефинишировавших - длиннее.

Этапы (переключаются по условию, а не по числу эпизодов):
  1. "Доехать":    награда за прогресс и финиш.
                   -> этап 2, когда финишировали >= 3 из последних 5 заездов.
  2. "Ускориться": добавляются награда за время финиша и штраф за время.
                   -> этап 3, когда рекорд не улучшался STAGE2_STALL_EPISODES заездов подряд.
  3. "Совершенствоваться": то же самое; работает бесконечно, между запусками.

Сохранение: веса, буфер опыта, призрак, этап и счётчики пишутся на диск; следующий запуск
продолжает с того же места.

ЧТО ОСТАВЛЕНО ТЕБЕ: ядро DQN - метод DQNBackend.learn (три TODO). Пока они не дописаны,
скрипт при старте сообщит об этом и остановится. Формулы - в моём объяснении DQN (пункты 4-5).

Файлы рядом: tm_common.py (наблюдение, призрак) и TMlogger.py (запись телеметрии).

Установка:  python -m pip install numpy torch tminterface
Запуск без игры (выдуманная трасса, после того как допишешь learn):
    python tm_dqn.py --selftest
Запуск с игрой (игра через TMInterface 1.4.3, трасса загружена, заезд начат):
    python tm_dqn.py --live            # обучение
    python tm_dqn.py --live --play     # только ехать (жадно), без обучения и сохранения
    python tm_dqn.py --live --fresh    # начать заново, игнорируя сохранение

ВАЖНО: режим --live я написал по документации клиента, с настоящей игрой не запускал. Конвенция
угла yaw (YAW_SIGN / YAW_OFFSET в tm_common.py) не проверена. PyTorch-часть (DQNBackend) мне запустить негде,
поэтому в ней возможны мелкие ошибки: присылай текст, если что-то упадёт.
"""

import argparse
import math
import os
import random
import sys
from collections import deque
from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

# Общая часть (наблюдение, призрак, настройки нормализации и конвенция yaw) - в tm_common.py.
# Файл tm_common.py должен лежать рядом.
from tm_common import (STEER_MAX, OBS_SIZE, RawState, Ghost, ObservationBuilder)

# =============================================================================
# 1. НАСТРОЙКИ  (нормализацию входов и конвенцию yaw ищи в tm_common.py)
# =============================================================================

# --- DQN ---
HIDDEN = 64
GAMMA = 0.99
LR = 3e-4
BATCH_SIZE = 64
BUFFER_SIZE = 100_000
WARMUP = 1000                 # сколько переходов накопить, прежде чем начинать учиться
TRAIN_EVERY = 4               # учимся раз в столько решений
TARGET_UPDATE_EVERY = 1000    # целевая сеть обновляется раз в столько решений
EPS_START, EPS_END, EPS_DECAY_STEPS = 1.0, 0.05, 200_000
# Исследование: случайное действие "залипает" на несколько решений подряд. Иначе случайные
# повороты руля каждые 50 мс гасят друг друга, и агент почти никогда не повернёт по-настоящему.
EXPLORE_HOLD = (5, 25)        # на сколько решений подряд (от, до) удерживается случайное действие
EPS_BOOST_AT_STAGE = 0.3      # при смене этапа ε поднимается до этого значения (немного исследуем заново)

# --- Заезд ---
DECISION_EVERY_MS = 50
MAX_EPISODE_MS = 90_000
STUCK_WINDOW_MS, STUCK_MIN_MOVE_M = 3000, 3.0
NO_PROGRESS_MS = 8000
GAME_SPEED = 5.0

# --- Награда (на одно решение) ---
PROGRESS_SCALE = 0.1                          # награда за метр продвижения вдоль призрака
PROGRESS_CLIP = (-2.0, 8.0)                   # прирост за шаг обрезаем: защита от скачков индекса на призраке
STEP_PENALTY = {1: 0.01, 2: 0.03, 3: 0.03}    # штраф за каждое решение
CP_REWARD = 20.0                               # за каждый чекпоинт (финиш тоже считается)
FINISH_REWARD = 30.0
TIME_REWARD_PER_SEC = {1: 0.0, 2: 0.3, 3: 0.3}  # за секунду, сэкономленную до MAX_EPISODE_MS
CRASH_PENALTY = -5.0                          # если застряла / нет прогресса

# --- Этапы ---
STAGE1_FINISHES, STAGE1_WINDOW = 3, 5         # этап 2, когда финишировали 3 из последних 5
STAGE2_STALL_EPISODES = 30                    # этап 3, когда столько заездов нет нового рекорда

# --- Сохранение ---
STATE_BASE = "dqn_state"       # будут файлы dqn_state.pt (веса) и dqn_state.npz (всё остальное)
SAVE_EVERY_EPISODES = 5
SAVE_BUFFER = True             # буфер на диске занимает порядка 15 МБ


# =============================================================================
# 2. ДЕЙСТВИЯ
# =============================================================================
# (accelerate, brake, steer). steer: отрицательный = влево, положительный = вправо.
ACTIONS = [
    (True,  False,       0),   # 0 газ, прямо
    (True,  False,  -32768),   # 1 газ, влево вполсилы
    (True,  False,   32768),   # 2 газ, вправо вполсилы
    (True,  False,  -65536),   # 3 газ, влево до упора
    (True,  False,   65536),   # 4 газ, вправо до упора
    (False, False,       0),   # 5 накат, прямо
    (False, True,        0),   # 6 тормоз, прямо
    (False, False,  -65536),   # 7 накат, влево
    (False, False,   65536),   # 8 накат, вправо
]
N_ACTIONS = len(ACTIONS)


# =============================================================================
# 3. ЗАЕЗД: считает прогресс, награду, конец заезда
# =============================================================================

class Tracker:
    """Следит за одним заездом. update() вызывается раз в DECISION_EVERY_MS."""

    def __init__(self, ghost, stage):
        self.ghost, self.stage = ghost, stage
        self.prev_steer = 0.0
        self.observer = ObservationBuilder(ghost)    # считает наблюдение и прогресс (сбрасывает призрак)
        self.last_progress = None
        self.best_progress = 0.0
        self.last_gain_t = 0
        self.cp, self.cp_paid = 0, 0
        self.finished, self.finish_ms = False, None
        self.done, self.reason = False, ""
        self.total_reward, self.steps = 0.0, 0
        self.samples = deque()
        self.path_x, self.path_z = [], []

    def on_checkpoint(self, current, target, t_ms):
        self.cp = current
        if current >= target and not self.finished:
            self.finished, self.finish_ms = True, int(t_ms)

    def update(self, raw, t_ms):
        """
        Возвращает (obs, reward, terminal, truncated):
          terminal  - заезд закончился "по-настоящему" (финиш, застряла): будущей награды нет
          truncated - оборвали по времени: будущая награда была бы, просто мы остановились
        """
        raw.prev_steer = self.prev_steer
        x, z = raw.pos[0], raw.pos[2]
        self.path_x.append(x)
        self.path_z.append(z)
        self.steps += 1
        obs, p = self.observer.build(raw)     # p - прогресс: вдоль призрака, а без призрака - от старта

        # --- награда ---
        reward = -STEP_PENALTY[self.stage]
        if self.last_progress is not None:
            reward += PROGRESS_SCALE * max(PROGRESS_CLIP[0], min(PROGRESS_CLIP[1], p - self.last_progress))
        self.last_progress = p
        if p > self.best_progress + 1.0:
            self.last_gain_t = t_ms
        self.best_progress = max(self.best_progress, p)
        new_cps, self.cp_paid = self.cp - self.cp_paid, self.cp
        reward += CP_REWARD * new_cps
        terminal = truncated = False
        if raw.pos[1] < 20:
            reward -= 50.0
            terminal = True
            self.reason = "падение с трассы"
        if self.finished:
            reward += FINISH_REWARD
            reward += TIME_REWARD_PER_SEC[self.stage] * max(0.0, (MAX_EPISODE_MS - self.finish_ms) / 1000.0)
            terminal, self.reason = True, "финиш"
        elif t_ms - self.last_gain_t > NO_PROGRESS_MS:
            reward += CRASH_PENALTY
            terminal, self.reason = True, "нет прогресса"
        else:
            self.samples.append((t_ms, x, z))
            while len(self.samples) > 1 and self.samples[1][0] <= t_ms - STUCK_WINDOW_MS:
                self.samples.popleft()
            t0, x0, z0 = self.samples[0]
            if t_ms >= STUCK_WINDOW_MS and t_ms - t0 >= 0.9 * STUCK_WINDOW_MS \
                    and math.hypot(x - x0, z - z0) < STUCK_MIN_MOVE_M:
                reward += CRASH_PENALTY
                terminal, self.reason = True, "застряла"
            elif t_ms >= MAX_EPISODE_MS:
                truncated, self.reason = True, "время"
        self.done = terminal or truncated
        self.total_reward += reward
        return obs, reward, terminal, truncated


# =============================================================================
# 4. СЕТЬ И ЯДРО DQN (PyTorch). Вся зависимость от torch - в этом блоке.
# =============================================================================

if HAVE_TORCH:

    class QNet(nn.Module):
        """Q(s, .): вход - 16 чисел наблюдения, выход - по числу на каждое из 9 действий."""

        def __init__(self, n_inputs=OBS_SIZE, n_actions=N_ACTIONS, hidden=HIDDEN):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_inputs, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, n_actions),      # без активации: Q может быть любым числом
            )

        def forward(self, x):
            return self.net(x)


class DQNBackend:
    """
    Всё, что связано с PyTorch: две сети (основная и целевая), оптимизатор, шаг обучения, сохранение.
    Остальная часть программы общается с ним через пять методов: q_values, learn, sync_target, save, load.
    """

    def __init__(self, n_inputs=OBS_SIZE, n_actions=N_ACTIONS):
        if not HAVE_TORCH:
            sys.exit("Нужен PyTorch: python -m pip install torch")
        self.q = QNet(n_inputs, n_actions)
        self.target = QNet(n_inputs, n_actions)
        self.target.load_state_dict(self.q.state_dict())     # целевая сеть стартует копией основной
        self.target.eval()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=LR)

    def q_values(self, obs):
        """Q-значения всех действий для одного наблюдения: np.ndarray длины N_ACTIONS."""
        with torch.no_grad():
            return self.q(torch.from_numpy(obs.astype(np.float32)).unsqueeze(0))[0].numpy()

    def learn(self, s, a, r, s2, d):
        """
        Один шаг обучения по пачке из буфера. Входы - numpy-массивы:
            s, s2: (B, 16) float32     a: (B,) int64     r, d: (B,) float32   (d = 1, если заезд закончился)
        Верни число (значение потерь).

        ТВОЯ ЧАСТЬ - допиши три шага (формулы: пункты 4-5 объяснения DQN):
        """
        s, s2 = torch.from_numpy(s), torch.from_numpy(s2)
        a, r, d = torch.from_numpy(a), torch.from_numpy(r), torch.from_numpy(d)

        # TODO 1: q = Q(s, a) только для сделанных действий.
        #         Подсказка: self.q(s) даёт (B, 9); выбрать по a поможет .gather(1, a.unsqueeze(1)).squeeze(1)
        q = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        # TODO 2: цель y = r + GAMMA * (1 - d) * max_a' Q_target(s2, a'), БЕЗ градиента (with torch.no_grad()).
        #         Подсказка: self.target(s2).max(dim=1).values
        with torch.no_grad():
            y = r + GAMMA * (1 - d) * self.target(s2).max(dim=1).values
        # TODO 3: loss между q и y (MSE или Huber: F.smooth_l1_loss), затем
        #         self.opt.zero_grad(); loss.backward(); self.opt.step(); return loss.item()
        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()
        raise NotImplementedError("Допиши DQNBackend.learn: TODO 1-3 (см. комментарии внутри метода)")

    def sync_target(self):
        self.target.load_state_dict(self.q.state_dict())

    def save(self, path):
        torch.save({"q": self.q.state_dict(), "target": self.target.state_dict(),
                    "opt": self.opt.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self.q.load_state_dict(ckpt["q"])
        self.target.load_state_dict(ckpt["target"])
        self.opt.load_state_dict(ckpt["opt"])


def check_learn_is_written(backend):
    """Пробный вызов learn на пустой пачке: если TODO не дописаны, объясняем сразу, а не через 1000 шагов."""
    z = lambda *sh, dt=np.float32: np.zeros(sh, dtype=dt)
    try:
        backend.learn(z(2, OBS_SIZE), np.zeros(2, dtype=np.int64), z(2), z(2, OBS_SIZE), z(2))
    except NotImplementedError as e:
        sys.exit(f"\n{e}\nОткрой tm_dqn.py, найди DQNBackend.learn и допиши три шага, потом запусти снова.")


# =============================================================================
# 5. БУФЕР ОПЫТА
# =============================================================================

class ReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE):
        self.cap = capacity
        self.s = np.zeros((capacity, OBS_SIZE), np.float32)
        self.s2 = np.zeros((capacity, OBS_SIZE), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.d = np.zeros(capacity, np.float32)
        self.n, self.pos = 0, 0

    def __len__(self):
        return self.n

    def add(self, s, a, r, s2, d):
        i = self.pos
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.pos = (self.pos + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.n, size=batch_size)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]

    def dump(self):
        k = self.n
        return dict(buf_s=self.s[:k], buf_s2=self.s2[:k], buf_a=self.a[:k], buf_r=self.r[:k],
                    buf_d=self.d[:k], buf_pos=self.pos)

    def restore(self, d):
        k = len(d["buf_a"])
        self.s[:k], self.s2[:k], self.a[:k], self.r[:k], self.d[:k] = \
            d["buf_s"], d["buf_s2"], d["buf_a"], d["buf_r"], d["buf_d"]
        self.n, self.pos = k, int(d["buf_pos"]) % self.cap


# =============================================================================
# 6. ОБУЧАЮЩИЙСЯ АГЕНТ: ε, обучение, призрак, этапы, сохранение
# =============================================================================

class Trainer:
    def __init__(self, backend, state_base=STATE_BASE, fresh=False, play=False, seed=None, reference=None):
        self.backend, self.state_base, self.play = backend, state_base, play
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.buffer = ReplayBuffer()
        self.step_count, self.episodes, self.stage = 0, 0, 1
        self.best_time_ms, self.eps_since_best = None, 0
        self.recent_finishes = deque(maxlen=STAGE1_WINDOW)
        self.ghost = None
        self.eps_offset = 0           # сдвиг счётчика для ε при смене этапа
        self.last_loss = None
        self._hold_left, self._hold_action = 0, 0
        have = (not fresh) and state_base and os.path.exists(state_base + ".npz") \
            and os.path.exists(state_base + ".pt")
        if have and self._load():
            print(f"Загружено сохранение: эпизодов {self.episodes}, шагов {self.step_count}, этап {self.stage}, "
                  f"рекорд {self._fmt(self.best_time_ms)}")
        else:
            print("Начинаю с нуля.")
        self._apply_reference(reference)

    def _apply_reference(self, ref):
        """Эталон (твой заезд из CSV) становится призраком, если он лучше текущего (финишировал и быстрее).
        Когда агент проедет быстрее эталона, призрак заменится его заездом."""
        if ref is None:
            return
        if self.ghost is None or ref.key() > self.ghost.key():
            self.ghost = ref
            print(f"Призрак = эталон: длина {ref.length:.0f} м, время {ref.finish_ms / 1000:.2f} с.")
        else:
            print("Текущий призрак лучше эталона - оставляю его.")

    @staticmethod
    def _fmt(ms):
        return "нет" if ms is None else f"{ms / 1000:.2f} с"

    def epsilon(self):
        if self.play:
            return 0.0
        k = max(0, self.step_count - self.eps_offset)
        return max(EPS_END, EPS_START - (EPS_START - EPS_END) * k / EPS_DECAY_STEPS)

    def act(self, obs):
        if self._hold_left > 0:                       # продолжаем удерживать случайное действие
            self._hold_left -= 1
            return self._hold_action
        if random.random() < self.epsilon() / (0.5 * (EXPLORE_HOLD[0] + EXPLORE_HOLD[1])):   # начинаем серию
            self._hold_action = random.randrange(N_ACTIONS)
            self._hold_left = random.randint(*EXPLORE_HOLD) - 1
            return self._hold_action
        return int(np.argmax(self.backend.q_values(obs)))

    def observe(self, s, a, r, s2, done):
        """Запомнить переход и (иногда) сделать шаг обучения."""
        if self.play:
            return
        self.buffer.add(s, a, r, s2, float(done))
        self.step_count += 1
        if len(self.buffer) >= max(WARMUP, BATCH_SIZE) and self.step_count % TRAIN_EVERY == 0:
            self.last_loss = self.backend.learn(*self.buffer.sample(BATCH_SIZE))
        if self.step_count % TARGET_UPDATE_EVERY == 0:
            self.backend.sync_target()

    def new_tracker(self):
        return Tracker(self.ghost, self.stage)

    def end_episode(self, tr):
        """Подвести итог заезда: призрак, рекорд, этапы, сохранение. Возвращает строку-отчёт."""
        self.episodes += 1
        note = ""
        if not self.play:
            cand = Ghost(tr.path_x, tr.path_z, tr.finished, tr.finish_ms)
            if self.ghost is None or cand.key() > self.ghost.key():
                self.ghost = cand
            improved = tr.finished and (self.best_time_ms is None or tr.finish_ms < self.best_time_ms)
            if improved:
                self.best_time_ms, self.eps_since_best = tr.finish_ms, 0
                note = "  * новый рекорд"
            elif self.stage == 2:
                self.eps_since_best += 1
            self.recent_finishes.append(bool(tr.finished))
            if self.stage == 1 and len(self.recent_finishes) == STAGE1_WINDOW \
                    and sum(self.recent_finishes) >= STAGE1_FINISHES:
                self._next_stage(2, "ЭТАП 2: теперь учимся ехать быстро")
                note += "  >>> ЭТАП 2: теперь учимся ехать быстро"
            elif self.stage == 2 and self.eps_since_best >= STAGE2_STALL_EPISODES:
                self._next_stage(3, "ЭТАП 3: тонкая доводка")
                note += "  >>> ЭТАП 3: тонкая доводка, рекорд без остановки"
            if self.state_base and self.episodes % SAVE_EVERY_EPISODES == 0:
                self._save()
        loss = "-" if self.last_loss is None else f"{self.last_loss:.3f}"
        return (f"эп {self.episodes:4d} | этап {self.stage} | eps {self.epsilon():.2f} | награда {tr.total_reward:7.1f} | "
                f"прогресс {tr.best_progress:6.0f} м | cp {tr.cp} | {tr.reason:13s} | решений {tr.steps:5d} | "
                f"потери {loss} | рекорд {self._fmt(self.best_time_ms)}{note}")

    def _next_stage(self, stage, _msg):
        self.stage = stage
        self.eps_since_best = 0
        # поднять ε до EPS_BOOST_AT_STAGE: на новом этапе награда другая, нужно немного поисследовать
        k_target = (EPS_START - EPS_BOOST_AT_STAGE) / (EPS_START - EPS_END) * EPS_DECAY_STEPS
        self.eps_offset = int(self.step_count - k_target)

    # ---------- сохранение ----------
    def _save(self):
        g = self.ghost
        data = dict(
            step_count=self.step_count, episodes=self.episodes, stage=self.stage,
            best_time_ms=-1 if self.best_time_ms is None else self.best_time_ms,
            eps_since_best=self.eps_since_best, eps_offset=self.eps_offset,
            recent=np.array(list(self.recent_finishes), dtype=np.int64),
            ghost_x=g.raw_x if g else np.zeros(0), ghost_z=g.raw_z if g else np.zeros(0),
            ghost_fin=int(g.finished) if g else 0,
            ghost_ms=(g.finish_ms if g and g.finish_ms is not None else -1),
        )
        if SAVE_BUFFER:
            data.update(self.buffer.dump())
        tmp = self.state_base + ".tmp.npz"
        np.savez(tmp, **data)
        os.replace(tmp, self.state_base + ".npz")
        self.backend.save(self.state_base + ".pt")

    def _load(self):
        try:
            d = np.load(self.state_base + ".npz", allow_pickle=False)
            self.backend.load(self.state_base + ".pt")
            self.step_count, self.episodes, self.stage = int(d["step_count"]), int(d["episodes"]), int(d["stage"])
            bt = int(d["best_time_ms"])
            self.best_time_ms = None if bt < 0 else bt
            self.eps_since_best, self.eps_offset = int(d["eps_since_best"]), int(d["eps_offset"])
            self.recent_finishes.extend(bool(v) for v in d["recent"])
            if d["ghost_x"].size >= 3:
                gm = int(d["ghost_ms"])
                self.ghost = Ghost(d["ghost_x"], d["ghost_z"], bool(int(d["ghost_fin"])), None if gm < 0 else gm)
            if "buf_a" in d.files:
                self.buffer.restore(d)
            return True
        except Exception as e:                       # повреждённое сохранение не должно ломать запуск
            print(f"Не удалось прочитать сохранение ({e}) - начинаю с нуля.")
            return False


# =============================================================================
# 7. ПОДКЛЮЧЕНИЕ К ИГРЕ
# =============================================================================

def make_client(trainer):
    from tminterface.client import Client

    class DQNClient(Client):
        def __init__(self):
            super().__init__()
            self.tr = None
            self.last_t = None
            self.last_decision = -10**9
            self.waiting_restart = False
            self.prev = None              # (obs, action) предыдущего решения

        def on_registered(self, iface):
            iface.set_timeout(10_000)
            iface.set_speed(GAME_SPEED)
            iface.log("Python DQN connected", "success")
            print(f"Подключено. Этап {trainer.stage}, эпизодов {trainer.episodes}. Ctrl+C - остановка.")

        def _restart(self, iface):
            self.tr, self.prev = None, None
            self.waiting_restart = True
            iface.give_up()

        def on_run_step(self, iface, _time):
            restarted = self.last_t is not None and _time < self.last_t   # время пошло назад = новый заезд
            self.last_t = _time
            if restarted:
                self.waiting_restart = False
                self.tr, self.prev = trainer.new_tracker(), None
                self.last_decision = -10**9
            elif self.tr is None and not self.waiting_restart:
                self._restart(iface)                   # скрипт запущен посреди заезда: начнём честно с нуля
                return
            if _time < 0 or self.tr is None:
                return
            if _time - self.last_decision < DECISION_EVERY_MS:
                return
            self.last_decision = _time

            raw = RawState.from_sim_state(iface.get_simulation_state())
            obs, reward, terminal, truncated = self.tr.update(raw, _time)
            if self.prev is not None:
                trainer.observe(self.prev[0], self.prev[1], reward, obs, terminal)
            if self.tr.done:
                print(trainer.end_episode(self.tr))
                self._restart(iface)
                return
            a = trainer.act(obs)
            acc, brk, steer = ACTIONS[a]
            iface.set_input_state(accelerate=acc, brake=brk, steer=steer)
            self.tr.prev_steer = float(steer)
            self.prev = (obs, a)

        def on_checkpoint_count_changed(self, iface, current, target):
            if self.tr is not None and self.last_t is not None:
                self.tr.on_checkpoint(current, target, self.last_t)

        def on_shutdown(self, iface):
            if not trainer.play and trainer.state_base:
                trainer._save()
            print("Игра закрыта, прогресс сохранён.")

    return DQNClient()


def load_reference(args):
    """Эталон из CSV (--reference), или None."""
    if not getattr(args, "reference", None):
        return None
    return Ghost.from_csv(args.reference, getattr(args, "ref_episode", None))


def mode_live(args):
    from tminterface.client import run_client
    backend = DQNBackend()
    check_learn_is_written(backend)
    # в режиме --play сохранение только читается: Trainer при play=True ничего не пишет на диск
    trainer = Trainer(backend, state_base=args.state, fresh=args.fresh, play=args.play, reference=load_reference(args))
    client = make_client(trainer)
    print("Подключаюсь к TMInterface0...")
    run_client(client, "TMInterface0")


# =============================================================================
# 8. САМОПРОВЕРКА БЕЗ ИГРЫ: выдуманная простая трасса
# =============================================================================

class FakeTrack:
    """Упрощённая 2D-"игра" для проверки всего конвейера. Физика ненастоящая."""
    WAYPOINTS = [(300, 200), (300, 400), (420, 520), (600, 520), (700, 600), (700, 800)]
    CHECKPOINTS = [2, 4]
    HALF_WIDTH = 12.0
    V_CAP = 40.0

    def reset(self):
        self.x, self.z = map(float, self.WAYPOINTS[0])
        self.h, self.v, self.cp, self.crashed = 0.0, 0.0, 0, False

    def _dist_to_path(self):
        best = 1e9
        for (ax, az), (bx, bz) in zip(self.WAYPOINTS, self.WAYPOINTS[1:]):
            dx, dz = bx - ax, bz - az
            t = max(0.0, min(1.0, ((self.x - ax) * dx + (self.z - az) * dz) / (dx * dx + dz * dz)))
            best = min(best, math.hypot(self.x - (ax + t * dx), self.z - (az + t * dz)))
        return best

    def raw(self, prev_steer=0.0):
        return RawState(speed=self.v * 3.6, pos=(self.x, 10.0, self.z),
                        vel=(self.v * math.sin(self.h), 0.0, self.v * math.cos(self.h)),
                        yaw=self.h, pitch=0.0, roll=0.0, prev_steer=prev_steer)

    def tick(self, action, dt=0.01):
        acc, brk, steer = action
        if not self.crashed:
            if acc:
                self.v += 25.0 * dt
            if brk:
                self.v = max(0.0, self.v - 50.0 * dt)
            self.v = min(self.V_CAP, max(0.0, self.v - 0.02 * self.v * dt * 10))
            self.h += (steer / STEER_MAX) * 1.5 * dt / (1.0 + self.v / 40.0)
            self.x += self.v * math.sin(self.h) * dt
            self.z += self.v * math.cos(self.h) * dt
            if self._dist_to_path() > self.HALF_WIDTH:
                self.crashed, self.v = True, 0.0
        target = len(self.CHECKPOINTS) + 1
        wx, wz = self.WAYPOINTS[self.CHECKPOINTS[self.cp] if self.cp < len(self.CHECKPOINTS) else -1]
        if not self.crashed and math.hypot(self.x - wx, self.z - wz) < 15.0:
            self.cp += 1
            return self.cp, target
        return None


def run_fake_episode(env, trainer):
    """Один заезд на выдуманной трассе тем же кодом, что и в игре. Возвращает Tracker."""
    env.reset()
    tr = trainer.new_tracker()
    t, prev, action = 0, None, ACTIONS[0]
    while True:
        if t % DECISION_EVERY_MS == 0:
            obs, reward, terminal, truncated = tr.update(env.raw(tr.prev_steer), t)
            if prev is not None:
                trainer.observe(prev[0], prev[1], reward, obs, terminal)
            if tr.done:
                return tr
            a = trainer.act(obs)
            action = ACTIONS[a]
            tr.prev_steer = float(action[2])
            prev = (obs, a)
        res = env.tick(action)
        t += 10
        if res:
            tr.on_checkpoint(res[0], res[1], t)


def mode_selftest(args, backend=None):
    backend = backend or DQNBackend()
    check_learn_is_written(backend)
    trainer = Trainer(backend, state_base=None, fresh=True, seed=args.seed, reference=load_reference(args))
    env = FakeTrack()
    print(f"Выдуманная трасса, эпизодов {args.episodes}. Параметров сети: "
          f"{OBS_SIZE * HIDDEN + HIDDEN + HIDDEN * HIDDEN + HIDDEN + HIDDEN * N_ACTIONS + N_ACTIONS}\n")
    for i in range(args.episodes):
        tr = run_fake_episode(env, trainer)
        line = trainer.end_episode(tr)
        if args.verbose or i % args.every == 0 or "ЭТАП" in line or "новый рекорд" in line:
            print(line)


# =============================================================================

def main():
    p = argparse.ArgumentParser(description="DQN для Trackmania: призрак, этапы, сохранение")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--selftest", action="store_true", help="обучение на выдуманной трассе, без игры")
    g.add_argument("--live", action="store_true", help="подключиться к игре")
    p.add_argument("--episodes", type=int, default=300, help="сколько заездов в --selftest")
    p.add_argument("--every", type=int, default=10, help="--selftest: печатать каждый N-й заезд")
    p.add_argument("--verbose", action="store_true", help="--selftest: печатать каждый заезд")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--state", default=STATE_BASE, help="база имени файлов сохранения")
    p.add_argument("--fresh", action="store_true", help="игнорировать сохранение")
    p.add_argument("--reference", metavar="CSV", help="эталон: CSV твоего заезда от TMlogger.py (станет призраком)")
    p.add_argument("--ref-episode", type=int, default=None, help="номер заезда в CSV (столбец episode); по умолчанию последний")
    p.add_argument("--play", action="store_true", help="--live: только ехать (ε=0), без обучения и сохранения")
    args = p.parse_args()
    mode_selftest(args) if args.selftest else mode_live(args)


if __name__ == "__main__":
    main()