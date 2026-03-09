import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. СОЗДАНИЕ И ПРЕОБРАЗОВАНИЕ МАССИВОВ
# ============================================================

def create_vector(firast_nomber: [int, float], last_nomber: [int, float], step: [int, float] = 1) -> np.ndarray:
    """Создает вектор с использованием np.arange.

    Примечание для отчета: демонстрация особенности работы np.arange
    с нецелыми числами из-за погрешностей вычислений с плавающей точкой.

    Args:
        firast_nomber: Начальное значение
        last_nomber: Конечное значение (не включается)
        step: Шаг между элементами

    Returns:
        Вектор с элементами от firast_nomber до last_nomber с шагом step
    """
    vector = np.arange(firast_nomber, last_nomber, step)
    return vector


def create_matrix(m: [int], n: [int]) -> np.ndarray:
    """Создает матрицу случайных чисел заданного размера.

    Args:
        m: Количество строк
        n: Количество столбцов

    Returns:
        Матрица размером m×n со случайными значениями из [0, 1)
    """
    matrix = np.random.rand(m, n)
    return matrix


def reshape_arr(arr: [list, np.ndarray], m: [int], n: [int]) -> np.ndarray:
    """Изменяет форму массива.

    Args:
        arr: Исходный массив
        m: Новое количество строк
        n: Новое количество столбцов

    Returns:
        Массив новой формы m×n
    """
    res = arr.reshape(m, n)
    return res


def transpose_matrix(mat: [np.ndarray]) -> np.ndarray:
    """Транспонирует матрицу.

    Args:
        mat: Входная матрица

    Returns:
        Транспонированная матрица
    """
    return np.transpose(mat)


# ============================================================
# 2. ВЕКТОРНЫЕ ОПЕРАЦИИ
# ============================================================

def vector_add(vec1: [np.array], vec2: np.array) -> np.array:
    """Выполняет поэлементное сложение двух векторов.

    Args:
        vec1: Первый вектор
        vec2: Второй вектор

    Returns:
        Результат поэлементного сложения
    """
    return vec1 + vec2


def scalar_multiply(vec: [np.array], scalar: [int, float]) -> np.array:
    """Умножает вектор на скаляр.

    Args:
        vec: Исходный вектор
        scalar: Скалярное значение

    Returns:
        Вектор, умноженный на скаляр
    """
    return vec * scalar


def elementwise_multiply(vec1: [np.array], vec2: np.array) -> np.array:
    """Выполняет поэлементное умножение двух векторов.

    Args:
        vec1: Первый вектор
        vec2: Второй вектор

    Returns:
        Результат поэлементного умножения
    """
    return vec1 * vec2


def dot_product(vec1: [np.array], vec2: np.array) -> [int, float]:
    """Вычисляет скалярное произведение двух векторов.

    Args:
        vec1: Первый вектор
        vec2: Второй вектор

    Returns:
        Скалярное произведение
    """
    return np.dot(vec1, vec2)


# ============================================================
# 3. МАТРИЧНЫЕ ОПЕРАЦИИ
# ============================================================

def matrix_multiply(mat1: [np.ndarray], mat2: [np.ndarray]) -> np.ndarray:
    """Выполняет умножение двух матриц.

    Args:
        mat1: Первая матрица
        mat2: Вторая матрица

    Returns:
        Результат умножения матриц
    """
    return mat1 @ mat2


def matrix_determinant(mat: [np.ndarray]) -> [int, float]:
    """Вычисляет определитель квадратной матрицы.

    Args:
        mat: Квадратная матрица

    Returns:
        Определитель матрицы
    """
    return np.linalg.det(mat)


def matrix_inverse(mat: [np.ndarray]) -> np.ndarray:
    """Вычисляет обратную матрицу.

    Args:
        mat: Квадратная матрица

    Returns:
        Обратная матрица
    """
    return np.linalg.inv(mat)


def solve_linear_system(mat1: [np.ndarray], mat2: [np.ndarray]) -> np.ndarray:
    """Решает систему линейных уравнений Ax = b.

    Args:
        mat1: Матрица коэффициентов A
        mat2: Вектор свободных членов b

    Returns:
        Решение системы x
    """
    return np.linalg.solve(mat1, mat2)


# ============================================================
# 4. СТАТИСТИЧЕСКИЙ АНАЛИЗ
# ============================================================

def load_dataset(path: [str] = "data/students_scores.csv") -> np.ndarray:
    """Загружает данные из CSV файла в NumPy массив.

    Args:
        path: Путь к CSV файлу

    Returns:
        Данные в виде NumPy массива
    """
    return pd.read_csv(path).to_numpy()


def statistical_analysis(data: np.ndarray) -> dict:
    """Выполняет статистический анализ данных.

    Анализирует результаты экзамена по математике, вычисляя:
    - средний балл
    - медиану
    - стандартное отклонение
    - минимум и максимум
    - 25 и 75 перцентили

    Args:
        data: Одномерный массив с оценками

    Returns:
        Словарь со статистическими показателями
    """
    dictmath = {
        'средний балл': np.mean(data),
        'медиана': np.median(data),
        'отклонение': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'перцентили 25': np.percentile(data, 25),
        'перцентили 75': np.percentile(data, 75)
    }
    return dictmath


def normalize_data(data: [np.ndarray], left: [int, float] = 0, right: [int, float] = 1) -> np.ndarray:
    """Выполняет Min-Max нормализацию данных.

    Приводит данные к заданному диапазону [left, right].

    Args:
        data: Входной массив данных
        left: Левая граница целевого диапазона
        right: Правая граница целевого диапазона

    Returns:
        Нормализованные данные

    Raises:
        Возвращает сообщение об ошибке при делении на ноль
    """
    min_val = np.min(data)
    max_val = np.max(data)

    if max_val - min_val == 0:
        return "На 0 делить нельзя"
    else:
        normalized = left + (data - min_val) * (right - left) / (max_val - min_val)

    return normalized


# ============================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ============================================================

def plot_histogram(data: [np.ndarray]):
    """Строит гистограмму распределения оценок по математике.

    Args:
        data: Данные для построения гистограммы

    Note:
        Использует matplotlib.pyplot.hist для построения
        Сохраняет результат в файл ../-NumPy/plots/histogram.png
    """
    plt.hist(data)
    plt.savefig('../-NumPy/plots/histogram.png')
    pass


def plot_heatmap(dat: [np.ndarray]):
    """Строит тепловую карту корреляции предметов.

    Args:
        dat: Матрица корреляции для визуализации

    Note:
        Использует seaborn.heatmap для построения
        Сохраняет результат в файл ../-NumPy/plots/heatmap.png
    """
    sns.heatmap(dat)
    plt.savefig('../-NumPy/plots/heatmap.png')
    pass


def plot_line(x: [np.ndarray], y: [np.ndarray]):
    """Строит график зависимости оценок от номера студента.

    Args:
        x: Номера студентов (ось X)
        y: Оценки студентов (ось Y)

    Note:
        Добавляет заголовок, подписи осей и сетку
        Сохраняет результат в файл ../-NumPy/plots/line.png
    """
    plt.plot(x, y)
    plt.title("график зависимости: студент -> оценка")
    plt.xlabel('студенты')
    plt.ylabel('оценки')
    plt.grid(True)
    plt.savefig('../-NumPy/plots/line.png')
    pass


if __name__ == "__main__":
    dat = load_dataset()
    plot_histogram(dat)
    plot_line(create_vector(1, len(transpose_matrix(dat)[0]) + 1), transpose_matrix(dat)[0])
