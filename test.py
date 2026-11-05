import numpy as np

data = np.load("dqn_state.npz", allow_pickle=True)

print("Содержимое dqn_state.npz:")
for key in data.files:
    value = data[key]
    print(f"{key}: shape={value.shape}, dtype={value.dtype}")

    if value.ndim == 0:
        print("  Значение:", value.item())

        x = data["ghost_x"]
z = data["ghost_z"]

distances = np.sqrt(np.diff(x)**2 + np.diff(z)**2)
total_distance = np.sum(distances)

print("\nАнализ траектории:")
print("Первая точка:", x[0], z[0])
print("Последняя точка:", x[-1], z[-1])
print("Длина траектории:", round(total_distance, 2), "м")
print("Максимальное перемещение от старта:", round(np.max(np.sqrt((x-x[0])**2 + (z-z[0])**2)), 2), "м")
print("Среднее расстояние между точками:", round(np.mean(distances), 3), "м")
print("Максимальное расстояние между точками:", round(np.max(distances), 3), "м")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))

plt.plot(x, z, label="Траектория призрака")
plt.scatter(x[0], z[0], color="green", label="Старт")
plt.scatter(x[-1], z[-1], color="red", label="Последняя точка")

plt.xlabel("X")
plt.ylabel("Z")
plt.title("Траектория лучшего заезда агента")
plt.legend()
plt.axis("equal")
plt.grid()

plt.show()

print("\nПоследние 10 точек призрака:")

for i in range(-10, 0):
    print(i, "X:", x[i], "Z:", z[i])

print("\nПроверка последней точки:")
print("X:", x[-1], "Z:", z[-1])