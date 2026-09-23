import numpy as np
from fractions import Fraction as Frac


class Simplex:
    def __init__(self, func, limits, b_values, signs, find_max=True):
        self.func = [Frac(x) for x in func]
        self.limits = [[Frac(x) for x in row] for row in limits]
        self.b = [Frac(x) for x in b_values]
        self.signs = signs
        self.find_max = find_max

        self.rows_count = len(b_values)
        self.orig_vars_count = len(func)

        self.table = None
        self.basis = []
        self.isk_vars = []  # индексы искусственных переменных
        self.names = []

    def show_table(self, title):
        print(f"\n {title}")
        # Простой вывод заголовка
        head = "Базис | " + " | ".join(f"{n:>4}" for n in self.names) + " |    b"
        print("-" * len(head))
        print(head)
        print("-" * len(head))

        for i in range(self.rows_count):
            current_basis = self.names[self.basis[i]]
            row_data = " | ".join(f"{str(val):>4}" for val in self.table[i, :-1])
            print(f" {current_basis:>4} | {row_data} | {str(self.table[i, -1]):>4}")

        print("-" * len(head))
        delta_data = " | ".join(f"{str(val):>4}" for val in self.table[-1, :-1])
        print(f"    Δ | {delta_data} | {str(self.table[-1, -1]):>4}")
        print("-" * len(head))

    def make_canonical(self):
        # Если ищем максимум, меняем знаки у функции
        self.c = [-x for x in self.func] if self.find_max else list(self.func)
        self.names = [f"x{i + 1}" for i in range(self.orig_vars_count)]

        dop_vars = []
        isk_vars = []

        # Разбираемся со знаками и добавляем новые переменные
        for i in range(self.rows_count):
            if self.b[i] < 0:
                self.b[i] *= -1
                self.limits[i] = [-x for x in self.limits[i]]
                if self.signs[i] == '<=':
                    self.signs[i] = '>='
                elif self.signs[i] == '>=':
                    self.signs[i] = '<='

            if self.signs[i] == '<=':
                dop_vars.append((i, 1))
                self.basis.append(len(self.names))
                self.names.append(f"x{len(self.names) + 1}")

            elif self.signs[i] == '>=':
                dop_vars.append((i, -1))
                self.names.append(f"x{len(self.names) + 1}")

                isk_vars.append(i)
                self.basis.append(len(self.names))
                self.isk_vars.append(len(self.names))
                self.names.append(f"x{len(self.names) + 1}")

            elif self.signs[i] == '=':
                isk_vars.append(i)
                self.basis.append(len(self.names))
                self.isk_vars.append(len(self.names))
                self.names.append(f"x{len(self.names) + 1}")

        # Создаем пустую таблицу (матрицу)
        total_vars = len(self.names)
        self.table = np.zeros((self.rows_count + 1, total_vars + 1), dtype=object)

        # Заполняем цифрами
        for i in range(self.rows_count):
            self.table[i, :self.orig_vars_count] = self.limits[i]
            self.table[i, -1] = self.b[i]

        for idx, (row, val) in enumerate(dop_vars):
            self.table[row, self.orig_vars_count + idx] = Frac(val)

        for idx, row in enumerate(isk_vars):
            self.table[row, self.orig_vars_count + len(dop_vars) + idx] = Frac(1)

    def recalculate(self, row, col):
        # Это правило прямоугольника
        element = self.table[row, col]
        self.table[row, :] = self.table[row, :] / element

        for i in range(len(self.table)):
            if i != row:
                self.table[i, :] -= self.table[i, col] * self.table[row, :]

        self.basis[row] = col  # меняем базис

    def run_iterations(self, phase):
        step = 1
        while True:
            self.show_table(f"{phase}. Шаг {step}")
            delta = self.table[-1, :-1]

            # Ищем самый большой минус
            min_val = min(delta)
            if min_val >= 0:
                print("Минусов больше нет. Конец этапа.")
                break

            col = np.argmin(delta)

            # Ищем строку по минимальному делению
            ratios = []
            for i in range(self.rows_count):
                if self.table[i, col] > 0:
                    ratios.append(self.table[i, -1] / self.table[i, col])
                else:
                    ratios.append(float('inf'))

            if min(ratios) == float('inf'):
                print("Ошибка: функция бесконечна!")
                return

            row = ratios.index(min(ratios))

            print(f"Разрешающий элемент: {self.table[row, col]}")
            print(f"Вместо {self.names[self.basis[row]]} заходит {self.names[col]}")

            self.recalculate(row, col)
            step += 1

    def solve(self):
        self.make_canonical()

        # 1. Вспомогательная задача (если есть искусственные переменные)
        if len(self.isk_vars) > 0:
            print("\nРешаем Вспомогательную задачу:")
            for col in self.isk_vars:
                self.table[-1, col] = Frac(1)

            # Подставляем базис в строку дельта
            for i in range(self.rows_count):
                b_col = self.basis[i]
                if self.table[-1, b_col] != 0:
                    self.table[-1, :] -= self.table[i, :] * self.table[-1, b_col]

            self.run_iterations("Фаза 1")

            if self.table[-1, -1] < 0:
                print("Решений нет!")
                return

            # Удаляем колонки искусственных переменных
            keep = [i for i in range(len(self.names)) if i not in self.isk_vars] + [-1]
            self.table = self.table[:, keep]
            self.names = [n for i, n in enumerate(self.names) if i not in self.isk_vars]

            # Сдвигаем индексы базиса
            for i in range(len(self.basis)):
                shift = sum(1 for art in self.isk_vars if art < self.basis[i])
                self.basis[i] -= shift

        # 2. Основная задача
        print("\nРешаем Основную задачу:")
        self.table[-1, :] = 0

        for i in range(self.orig_vars_count):
            self.table[-1, i] = self.c[i]

        for i in range(self.rows_count):
            b_col = self.basis[i]
            if self.table[-1, b_col] != 0:
                self.table[-1, :] -= self.table[i, :] * self.table[-1, b_col]

        self.run_iterations("Фаза 2")

        # 3. Вывод ответа
        print("\nФинальный ответ:")
        result = [Frac(0)] * self.orig_vars_count
        for i in range(self.rows_count):
            if self.basis[i] < self.orig_vars_count:
                result[self.basis[i]] = self.table[i, -1]

        print(f"Точка X* = ({', '.join(str(x) for x in result)})")

        final_z = self.table[-1, -1] if not self.find_max else -self.table[-1, -1]
        print(f"Максимальное значение Z = {final_z}")


# Подставлены параметры для 9 варианта
if __name__ == "__main__":
    func = [2, 1, 3, 2]

    limits = [
        [1, 2, 1, 0],
        [1, 0, 1, 1],
        [0, 1, 0, 1]
    ]
    b_values = [11, 8, 3]
    signs = ['<=', '=', '>=']

    task = Simplex(func, limits, b_values, signs, find_max=True)
    task.solve()
