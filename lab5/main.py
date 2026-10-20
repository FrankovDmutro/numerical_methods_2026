import math
import matplotlib.pyplot as plt
import numpy as np


def f(x: float) -> float:
    return 50.0 + 20.0 * math.sin(math.pi * x / 12.0) + 5.0 * math.exp(-0.2 * (x - 12.0) ** 2)


def simpson_composite(func, a: float, b: float, n: int) -> float:
    if n % 2 != 0:
        raise ValueError("Число N має бути парним")

    h = (b - a) / n
    s_odd = sum(func(a + i * h) for i in range(1, n, 2))
    s_even = sum(func(a + i * h) for i in range(2, n, 2))

    return (h / 3.0) * (func(a) + 4.0 * s_odd + 2.0 * s_even + func(b))


class AdaptiveSimpson:
    def __init__(self, func, tol: float):
        self.func = func
        self.tol = tol
        self.eval_count = 0

    def _eval(self, x: float) -> float:
        self.eval_count += 1
        return self.func(x)

    def integrate(self, a: float, b: float) -> float:
        self.eval_count = 0
        fa = self._eval(a)
        fb = self._eval(b)
        m = (a + b) / 2.0
        fm = self._eval(m)
        i1 = (b - a) / 6.0 * (fa + 4.0 * fm + fb)
        return self._recursive(a, b, fa, fm, fb, i1, self.tol)

    def _recursive(self, a, b, fa, fm, fb, i1, tol) -> float:
        m = (a + b) / 2.0
        lm = (a + m) / 2.0
        rm = (m + b) / 2.0
        flm = self._eval(lm)
        frm = self._eval(rm)

        i_left = (m - a) / 6.0 * (fa + 4.0 * flm + fm)
        i_right = (b - m) / 6.0 * (fm + 4.0 * frm + fb)
        i2 = i_left + i_right

        if abs(i1 - i2) <= 15.0 * tol:
            return i2 + (i2 - i1) / 15.0

        return (
            self._recursive(a, m, fa, flm, fm, i_left, tol / 2.0)
            + self._recursive(m, b, fm, frm, fb, i_right, tol / 2.0)
        )


def main():
    a, b = 0.0, 24.0
    i_exact = 1200.0 + 5.0 * math.sqrt(math.pi / 0.2) * math.erf(12.0 * math.sqrt(0.2))

    print(f"Точний інтеграл I0: {i_exact:.15f}")

    target_eps = 1e-12
    n_list = list(range(10, 1002, 2))
    errors = []
    n_opt = None
    eps_opt = None

    for n in n_list:
        val = simpson_composite(f, a, b, n)
        err = abs(val - i_exact)
        errors.append(err)
        if n_opt is None and err <= target_eps:
            n_opt = n
            eps_opt = err

    if n_opt is None:
        n_opt = min(zip(n_list, errors), key=lambda x: x[1])[0]
        eps_opt = abs(simpson_composite(f, a, b, n_opt) - i_exact)

    print(f"N_opt (точність <= 1e-12): {n_opt}, похибка: {eps_opt:.4e}")

    raw_n0 = int(round(n_opt / 10.0))
    n0 = max(16, (raw_n0 // 8) * 8)

    i_n0 = simpson_composite(f, a, b, n0)
    eps0 = abs(i_n0 - i_exact)
    print(f"N0: {n0}, I(N0): {i_n0:.15f}, eps0: {eps0:.4e}")

    n0_half = n0 // 2
    i_n0_half = simpson_composite(f, a, b, n0_half)
    i_runge = i_n0 + (i_n0 - i_n0_half) / 15.0
    eps_runge = abs(i_runge - i_exact)
    print(f"I_R (Рунге-Ромберг): {i_runge:.15f}, epsR: {eps_runge:.4e}")

    n0_quarter = n0 // 4
    i_n0_quarter = simpson_composite(f, a, b, n0_quarter)
    num = (i_n0_half) ** 2 - i_n0 * i_n0_quarter
    den = 2.0 * i_n0_half - (i_n0 + i_n0_quarter)
    i_aitken = num / den
    p_order = (1.0 / math.log(2.0)) * math.log(abs((i_n0_quarter - i_n0_half) / (i_n0_half - i_n0)))
    eps_aitken = abs(i_aitken - i_exact)
    print(f"I_E (Ейткен): {i_aitken:.15f}, epsE: {eps_aitken:.4e}, p: {p_order:.4f}")

    print("\nАдаптивний алгоритм:")
    delta_list = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
    adapt_errors = []
    adapt_evals = []

    for delta in delta_list:
        ad = AdaptiveSimpson(f, tol=delta)
        val = ad.integrate(a, b)
        err = abs(val - i_exact)
        adapt_errors.append(err)
        adapt_evals.append(ad.eval_count)
        print(f"delta: {delta:.1e} | викликів f(x): {ad.eval_count:<5} | похибка: {err:.4e}")

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))

    x_vals = np.linspace(0, 24, 400)
    axs[0].plot(x_vals, [f(x) for x in x_vals], color="tab:blue", lw=2)
    axs[0].set_title("Навантаження f(x)")
    axs[0].set_xlabel("Час (год)")
    axs[0].set_ylabel("f(x)")
    axs[0].grid(True, linestyle="--", alpha=0.6)

    axs[1].loglog(n_list, errors, color="tab:red", lw=1.5)
    axs[1].scatter([n_opt], [eps_opt], color="black", zorder=5, label=f"N_opt={n_opt}")
    axs[1].axvline(n0, color="green", linestyle="--", label=f"N0={n0}")
    axs[1].set_title("Похибка Сімпсона від N")
    axs[1].set_xlabel("N")
    axs[1].set_ylabel("Похибка")
    axs[1].grid(True, which="both", linestyle="--", alpha=0.5)
    axs[1].legend()

    axs[2].loglog(delta_list, adapt_errors, "o-", color="purple", label="Похибка")
    axs[2].set_title("Адаптивний метод")
    axs[2].set_xlabel("delta")
    axs[2].set_ylabel("Похибка")
    axs[2].grid(True, which="both", linestyle="--", alpha=0.5)

    ax2_twin = axs[2].twinx()
    ax2_twin.semilogx(delta_list, adapt_evals, "s--", color="orange", label="Виклики")
    ax2_twin.set_ylabel("Кількість викликів f(x)")

    lines1, labels1 = axs[2].get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    axs[2].legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig("lab5_integration.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()