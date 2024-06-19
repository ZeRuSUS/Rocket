from time import time
import math
import numpy as np
from scipy.integrate import solve_ivp, simps
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt

# Константы
q_t = 0.04
g = 0.01
d = 2
C = 0.05
Y = 0.01
t_s = 20
m_t = 0.8

# Начальные условия
x0, t0 = [1.0, 0.0, 0.0], 0.0
# Временные границы
tStart = 0.0
tMax = 100.0

# Определение точек, в которых производятся вычисления
nEval = 100
tEval = np.linspace(t0, tMax, nEval)

# Метод решения ОДУ(Рунге Кутта)
nMethod = 'RK23'


# Зависимость расхода топлива от времени
def q(t, ts):
    if t <= ts:
        try:
            qu = m_t / ts
        except ZeroDivisionError:
            qu = m_t
    else:
        qu = 0
    return qu


# Константа плотности воздуха
p = lambda z: np.exp(-Y * z ** 2)


# Система ОДУ (правая часть)
def Phi(t, y):
    m, z, u = y
    w1 = -q(t, t_s)
    w2 = u
    w3 = -g + 1 / m * (d * q(t, t_s) - C * p(z) * u ** 2)
    return [w1, w2, w3]


# Решение системы ОДУ
def DSolveProblem(printResult=True, tempStart=0.0):
    # Изменяем значение глобальной переменной t_s
    global t_s
    t_s = tempStart

    # Засечем время начала расчетов
    tic1 = time()

    # Решаем систему ОДУ
    nSol = solve_ivp(Phi, [t0, tMax], x0, method=nMethod, t_eval=tEval)

    tSol = nSol.t
    xSol = nSol.y

    # Подсчитаем время выполнения расчетов
    tic2 = time()

    if printResult:
        print('Метод - {0}\nВремя расчетов (сек.):{1}\nКоличество точек:{2}'.
              format(nMethod, tic2 - tic1, len(nSol.t)))

        if nSol.status == -1:
            print('Интегрироваие привело к сбою.')
        elif nSol.status == 0:
            print('Решатель(solver) успешно достиг конца интервала.')
        elif nSol.status == 1:
            print('Интегрирование было завершено.')
    return tSol, xSol


# Графики решения
def PlotResult(tSol, xSol):
    fig = plt.figure(facecolor='white', figsize=(1.4 * 6.4, 1.2 * 4.8))

    # Общий график
    plt.subplot(2, 2, 1)
    plt.plot(tSol, xSol[0, :], '-b', label=r'$m(t)$', linewidth=2)
    plt.plot(tSol, xSol[1, :], '-r', label=r'$z(t)$', linewidth=2)
    plt.plot(tSol, xSol[2, :], '-g', label=r'$u(t)$', linewidth=2)
    plt.xlim(left=t0, right=tMax)
    plt.legend(loc='best')
    plt.title("Общий график")
    plt.grid(True)

    # График Скорости
    plt.subplot(2, 2, 3)
    plt.plot(tSol, xSol[2, :], '-g', label=r'$u(t)$', linewidth=2)
    plt.xlim(left=t0, right=tMax)
    plt.legend(loc='best')
    plt.xlabel(r"$t$")
    plt.title("Скорость")
    plt.grid(True)

    # График высоты подъема
    plt.subplot(1, 2, 2)
    plt.plot(tSol, xSol[1, :], '-r', label=r'$z(t)$', linewidth=2)
    plt.xlim(left=t0, right=tMax)
    plt.ylim(bottom=0.0, top=350.0)
    plt.legend(loc='best')
    plt.xlabel(r"$t$")
    plt.title("Высота")
    plt.grid(True)
    pass


# Определим функцию, которая вычисляет выход продукта
def x3Max(var_tStart):
    tSol, xSol = DSolveProblem(printResult=False, tempStart=var_tStart)
    return -simps(xSol[2:], tSol)


# Main
if __name__ == '__main__':
    tSol, xSol = DSolveProblem(tempStart=3.0)
    PlotResult(tSol, xSol)

    tSol, xSol = DSolveProblem(tempStart=20.0)
    PlotResult(tSol, xSol)

    tSol, xSol = DSolveProblem(tempStart=30.0)
    PlotResult(tSol, xSol)

    plt.show()  # display

    # Находим минимумум функции -ВыходПродукта
    res = minimize_scalar(x3Max, bounds=(t0, tMax), method='Bounded',
                          options={'xatol': 1e-09, 'maxiter': 500, 'disp': 3})

    print('Нахождение оптимального времени сгорания топлива для максимального взлета "z"\n', res)

    tSol, xSol = DSolveProblem(printResult=False, tempStart=res.x)
    PlotResult(tSol, xSol)

    plt.show()  # display

    print(x3Max(0.0))
