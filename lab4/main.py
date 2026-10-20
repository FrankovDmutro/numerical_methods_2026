import math
import matplotlib.pyplot as plt
import numpy as np


# =====================================================================
# ПУНКТ 1: Модель вологості ґрунту та аналітична похідна

def moisture_function(t: float) -> float:
    """
    Модель вологості ґрунту M(t) = 50 * e^(-0.1 * t) + 5 * sin(t).
    Визначає вологість у момент часу t.
    """
    return 50.0 * math.exp(-0.1 * t) + 5.0 * math.sin(t)


def exact_derivative(t: float) -> float:
    """
    Явний аналітичний вираз для першої похідної:
    dM/dt = -5 * e^(-0.1 * t) + 5 * cos(t).
    """
    return -5.0 * math.exp(-0.1 * t) + 5.0 * math.cos(t)


def central_diff(f, x0: float, h: float) -> float:
    """
    Центральна різницева формула другого порядку точності:
    y'_0(h) = (f(x0 + h) - f(x0 - h)) / (2 * h).
    """
    return (f(x0 + h) - f(x0 - h)) / (2.0 * h)


def main():

    # Задана точка розрахунку
    x0 = 1.0
    exact_val = exact_derivative(x0)
    print(f"\n[Пункт 1]")
    print(f"Точка диференціювання x0: {x0}")
    print(f"Точне аналітичне значення похідної y'(x0): {exact_val:.15f}")

    # Дослідження залежності похибки від кроку h у діапазоні [10^-20 ... 10^3]
    print(f"\n[Пункт 2]")
    print("Дослідження похибки R(h) = |y'_0(h) - y'(x0)| на діапазоні кроків h від 10^-20 до 10^3...")

    # Генеруємо логарифмічну сітку з 1000 значень кроку
    powers = np.linspace(-20, 3, 1000)
    h_values = 10.0**powers

    errors = []
    best_h = None
    min_error = float("inf")

    for h in h_values:
        try:
            approx = central_diff(moisture_function, x0, h)
            err = abs(approx - exact_val)
            # Якщо виникає переповнення або NaN через нульові ділення
            if math.isnan(err) or math.isinf(err):
                err = 1e10
        except ZeroDivisionError:
            err = 1e10

        errors.append(err)

        # Пошук точки глобального мінімуму сумарної похибки (оптимального кроку)
        if err < min_error:
            min_error = err
            best_h = h

    print(f"  Оптимальне значення кроку h0: {best_h:.2e}")
    print(f"  Мінімальна досягнута точність R0: {min_error:.2e}")

    # Розрахунок при базовому кроці h = 10^-3 та 2h
    print(f"\n[Пункти 3, 4, 5]")
    h_base = 1e-3  # Пункт 3: приймаємо h = 10^-3
    print(f"Базовий крок h: {h_base:.4f}")

    # Обчислення для двох кроків сітки h та 2h
    y0_h = central_diff(moisture_function, x0, h_base)
    y0_2h = central_diff(moisture_function, x0, 2.0 * h_base)

    # Похибка при базовому кроці сітки R1
    R1 = abs(y0_h - exact_val)

    print(f"  Наближення з кроком h  : y'_0(h)  = {y0_h:.12f}")
    print(f"  Наближення з кроком 2h : y'_0(2h) = {y0_2h:.12f}")
    print(f"  Похибка апроксимації R1: {R1:.6e}")

    # Метод Рунге-Ромберга
    print(f"\n[Пункт 6]")
    # Для формули другого порядку точності (p = 2) коефіцієнт становить 2^2 - 1 = 3
    y_runge = y0_h + (y0_h - y0_2h) / 3.0
    R2 = abs(y_runge - exact_val)

    print(f"  Уточнене значення за методом Рунге-Ромберга y'_R: {y_runge:.15f}")
    print(f"  Похибка апроксимації R2: {R2:.6e}")
    print(f"  Характер зміни: похибка зменшилася з {R1:.2e} до {R2:.2e} (у {R1 / R2:.1f} разів)")

    # Метод Ейткена з трьома кроками h, 2h, 4h
    print(f"\n[Пункт 7]")
    # Обчислюємо значення похідної на потрійній сітці: h, 2h та 4h
    y0_4h = central_diff(moisture_function, x0, 4.0 * h_base)

    # Формула Ейткена для уточненого значення похідної
    numerator = (y0_2h**2) - (y0_4h * y0_h)
    denominator = 2.0 * y0_2h - (y0_4h + y0_h)
    y_aitken = numerator / denominator

    # Оцінка порядку точності p за трьома наближеннями
    p_order = (1.0 / math.log(2.0)) * math.log(abs((y0_4h - y0_2h) / (y0_2h - y0_h)))

    R3 = abs(y_aitken - exact_val)

    print(f"  Наближення з кроком 4h : y'_0(4h) = {y0_4h:.12f}")
    print(f"  Уточнене значення за методом Ейткена y'_E: {y_aitken:.15f}")
    print(f"  Похибка апроксимації R3: {R3:.6e}")
    print(f"  Оцінений порядок точності p: {p_order:.4f} (теоретичний p = 2)")
    print(f"  Характер зміни: метод Ейткена досягає точності рівня машинної похибки")

    # =====================================================================
    # ЗВЕДЕНА ТАБЛИЦЯ РЕЗУЛЬТАТІВ ДЛЯ ЗВІТУ
    # =====================================================================
    print("\n" + "=" * 75)
    print(f"{'Метод':<28} | {'Крок h':<15} | {'Значення':<16} | {'Похибка R':<10}")
    print("-" * 75)
    print(f"{'Точне аналітичне':<28} | {'-':<15} | {exact_val:<16.10f} | {'0.0000e+00':<10}")
    print(f"{'Центральна різниця (h)':<28} | {h_base:<15.4f} | {y0_h:<16.10f} | {R1:<10.4e}")
    print(f"{'Центральна різниця (2h)':<28} | {2*h_base:<15.4f} | {y0_2h:<16.10f} | {abs(y0_2h - exact_val):<10.4e}")
    print(f"{'Центральна різниця (4h)':<28} | {4*h_base:<15.4f} | {y0_4h:<16.10f} | {abs(y0_4h - exact_val):<10.4e}")
    print(f"{'Центральна різниця (h0)':<28} | {best_h:<15.2e} | {central_diff(moisture_function, x0, best_h):<16.10f} | {min_error:<10.4e}")
    print(f"{'Рунге-Ромберг':<28} | {'h, 2h':<15} | {y_runge:<16.10f} | {R2:<10.4e}")
    print(f"{'Ейткен':<28} | {'h, 2h, 4h':<15} | {y_aitken:<16.10f} | {R3:<10.4e}")
    print("=" * 75)


    # Побудова графіків для звіту

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Графік 1: U-подібна крива похибки в подвійному логарифмічному масштабі
    ax1.loglog(h_values, errors, color="steelblue", lw=2, label="R(h) = |y'_0(h) - y'(x0)|")
    ax1.scatter([best_h], [min_error], color="red", s=100, zorder=5,
                label=f"Оптимум: h0 = {best_h:.2e}\nR0 = {min_error:.2e}")
    ax1.axvline(h_base, color="green", linestyle="--", lw=1.5,
                label=f"Робочий h = {h_base:.3f}")

    ax1.set_title("Залежність похибки R від кроку h\n(Ефект регуляризації)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Крок сітки h (log scale)")
    ax1.set_ylabel("Абсолютна похибка R (log scale)")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper center", fontsize=9)

    # Графік 2: Фізична модель M(t) та швидкість зміни
    t_plot = np.linspace(0, 20, 500)
    m_plot = [moisture_function(t) for t in t_plot]
    dm_plot = [exact_derivative(t) for t in t_plot]

    ax2.plot(t_plot, m_plot, color="navy", lw=2, label="Вологість ґрунту M(t)")
    ax2.plot(t_plot, dm_plot, color="darkorange", lw=1.5, linestyle="--", label="Швидкість зміни M'(t)")
    ax2.scatter([x0], [exact_val], color="crimson", s=80, zorder=5, label=f"M'(1) = {exact_val:.4f}")

    ax2.set_title("Динаміка моделі M(t) та швидкість зміни", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Час t")
    ax2.set_ylabel("M(t) / M'(t)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("lab4_differentiation.png", dpi=300)
    print("\nГрафік успішно збережено у файл 'lab4_differentiation.png'.")
    plt.show()


if __name__ == "__main__":
    main()