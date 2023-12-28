# метод перебора, метод поразрядного поиска, метод дихотомии, метод золотого сечения, метод парабол, метод Ньютона, метод Марквардта
#
# метод средней точки, метод ломаных

import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
import collections


def function(x):
    return x ** 2 - 2 * x - 2 * np.cos(x)


def der_function(x):
    return 2 * x - 2 + 2 * np.sin(x)


def second_der_func(x):
    return 2 + 2 * np.cos(x)


L = 6


def live_plot(data_x, data_f, a, b, title=''):
    clear_output(wait=True)
    plt.figure()
    x_space = np.linspace(a, b, 100)
    for i in range(len(data_f)):
        plt.scatter(data_x, data_f, color='red', s=4)
    plt.plot(x_space, function(x_space), label="function")
    plt.title(title)
    plt.grid(True)
    # plt.xlabel('Brute force')
    plt.legend(loc='center left')  # the plot evolves to the right
    plt.show()


def brute_force(a, b, size=100, live=False):
    h = np.abs((b - a) / size)
    f_min = function(a)
    x_min = a
    list_x = []
    list_f = []
    for i in range(size):
        f_i = function(a + i * h)
        x_i = a + i * h
        if f_i < f_min:
            f_min = function(x_i)
            x_min = a + i * h
            list_f.append(f_min)
            list_x.append(x_min)
            if live and i % 10 == 0:
                live_plot(list_x, list_f, a, b, title='Brute force')
    return f_min, x_min, size


def bitwise(a, b, epsilon=0.001, live=False):
    # assuming b>a
    delta = np.abs(b - a) / 4
    x_0 = a
    f_0 = function(x_0)
    flag = 0
    list_x = []
    list_f = []
    list_x.append(x_0)
    list_f.append(f_0)
    size = 0
    while np.abs(delta) > epsilon:
        if flag:
            x_0 = x_step
            f_0 = f_step
            delta = - delta / 4
        x_step = x_0 + delta
        f_step = function(x_step)
        if f_0 > f_step:
            x_0 = x_step
            f_0 = f_step
            if a < x_0 < b:
                flag = 0
            else:
                flag = 1
        else:
            flag = 1
        list_x.append(x_0)
        list_f.append(f_0)
        if live:
            live_plot(list_x, list_f, a, b, title='Bitwise')
        size += 1
    return f_0, x_0, size


def dichotomy(a, b, epsilon=0.001, live=False):
    a_loc = a
    b_loc = b
    delta = epsilon
    list_x = []
    list_f = []
    x_out = (a + b) / 2
    f_out = function(x_out)
    list_f.append(f_out)
    list_x.append(x_out)
    size = 0
    while np.abs(b - a) >= 2 * epsilon:
        x1 = (a + b) / 2 - delta / 2
        f1 = function(x1)
        x2 = (a + b) / 2 + delta / 2
        f2 = function(x2)
        if f1 <= f2:
            b = x2
        else:
            a = x1
        x_out = (a + b) / 2
        f_out = function(x_out)
        list_f.append(f_out)
        list_x.append(x_out)
        if live:
            live_plot(list_x, list_f, a_loc, b_loc, title='Dichotomy')
        size += 1
    return f_out, x_out, size


def gold_section(a, b, epsilon=0.001, live=False):
    a_loc = a
    b_loc = b
    tau = (np.sqrt(5) - 1) / 2
    l = b - a
    x1 = b - tau * l
    x2 = a + tau * l
    f1 = function(x1)
    f2 = function(x2)
    list_x = []
    list_f = []
    x_out = (a + b) / 2
    f_out = function(x_out)
    list_f.append(f_out)
    list_x.append(x_out)
    size = 0
    while l > 2 * epsilon:
        if f1 <= f2:
            b = x2
            l = b - a
            x2 = x1
            f2 = f1
            x1 = b - tau * l
            f1 = function(x1)
        else:
            a = x1
            l = b - a
            x1 = x2
            f1 = f2
            x2 = a + tau * l
            f2 = function(x2)
        x_out = (a + b) / 2
        f_out = function(x_out)
        list_f.append(f_out)
        list_x.append(x_out)
        if live:
            live_plot(list_x, list_f, a_loc, b_loc, title='Gold section')
        size += 1
    return f_out, x_out, size


def live_plot_parabolic(data_x, data_f, list_parabolic, a, b, title=''):
    clear_output(wait=True)
    plt.figure()
    x_space = np.linspace(a, b, 100)
    for i in range(len(data_f)):
        plt.scatter(data_x, data_f, color='red', s=4)
    a1, b1, c = calc_parabola_vertex(list_parabolic)
    x_pos = np.linspace(a, b, 100)
    y_pos = []
    for x in range(len(x_pos)):
        x_val = x_pos[x]
        y = (a1 * (x_val ** 2)) + (b1 * x_val) + c
        y_pos.append(y)
    plt.plot(x_pos, y_pos)
    plt.plot(x_space, function(x_space), label="function")
    plt.title(title)
    plt.grid(True)
    # plt.xlabel('Brute force')
    plt.legend(loc='center left')  # the plot evolves to the right
    plt.show()


def calc_parabola_vertex(list_loc):
    '''
    Adapted and modifed to get the unknowns for defining a parabola:
    http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
    '''
    x1, y1, x2, y2, x3, y3 = list_loc
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    B = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
    C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom
    return A, B, C


def parabolic(a, b, epsilon=0.001, live=False):
    a_loc = a
    b_loc = b
    x1 = a
    x2 = (b - a) / 2
    x3 = b - epsilon
    f1 = function(x1)
    f2 = function(x2)
    f3 = function(x3)
    a1 = (f2 - f1) / (x2 - x1)
    a2 = ((f3 - f1) / (x3 - x1) - (f2 - f1) / (x2 - x1)) / (x3 - x2)
    x_med = 1 / 2 * (x1 + x2 - a1 / a2)
    f_med = function(x_med)
    list_x = []
    list_f = []
    list_f.append(f_med)
    list_x.append(x_med)
    step = 0
    x_str = x_med
    while step == 0 or np.abs(x_med - x_str) > epsilon:
        x_str = x_med
        if x_med > x2:
            x1 = x2
            x2 = x_med
        else:
            x3 = x2
            x2 = x_med
        f1 = function(x1)
        f2 = function(x2)
        f3 = function(x3)
        a1 = (f2 - f1) / (x2 - x1)
        a2 = ((f3 - f1) / (x3 - x1) - (f2 - f1) / (x2 - x1)) / (x3 - x2)
        x_med = 1 / 2 * (x1 + x2 - a1 / a2)
        f_med = function(x_med)
        step += 1
        list_f.append(f_med)
        list_x.append(x_med)
        if live:
            list_parabolic = [x1, f1, x2, f2, x3, f3]
            live_plot_parabolic(list_x, list_f, list_parabolic, a_loc, b_loc, title='Parabolic')
    return f_med, x_med, step


def newton(a, b, epsilon=0.001, live=False):
    x0 = (a + b) / 2
    g = der_function(x0)
    size = 0
    list_x = [x0]
    list_f = [function((x0))]
    while np.abs(g) > epsilon:
        f2 = second_der_func(x0)
        x_temp = x0
        x0 = x_temp - g / f2
        g = der_function(x0)
        size += 1
        list_x.append(x0)
        list_f.append(function(x0))
        if live:
            live_plot(list_x, list_f, a, b, title='Newton')
    func = function(x0)
    return func, x0, size


def markward(a, b, epsilon=0.001, live=False):
    mu = 1 / 2
    x0 = (a + b) / 2
    g = der_function(x0)
    size = 0
    list_x = [x0]
    list_f = [function(x0)]
    x_temp = b
    while np.abs(x_temp - x0) > epsilon:
        f2 = second_der_func(x0)
        x_temp = x0
        x0 = x_temp - g / (f2 + mu)
        g_last = g
        g = der_function(x0)
        if g < g_last:
            mu = mu / 2
        else:
            mu = 2 * mu
        size += 1
        list_x.append(x0)
        list_f.append(function(x0))
        if live:
            live_plot(list_x, list_f, a, b, title='Markward')
    func = function(x0)
    return func, x0, size


def assistant_func(x, x1):
    return function(x) - L * np.abs(x - x1)


def live_plot_polylines(data_x, data_f, a, b, title=''):
    clear_output(wait=True)
    plt.figure()
    x_space = np.linspace(a, b, 100)
    for i in range(len(data_f)):
        plt.scatter(data_x, data_f, color='red', s=4)
    plt.plot(x_space, function(x_space), label="function")
    plt.title(title)
    plt.grid(True)
    # plt.xlabel('Brute force')
    plt.legend(loc='center left')  # the plot evolves to the right
    plt.show()


def p_func_min(p):
    pl_min = p[0][0]
    xl_min = p[1][0]
    for j in range(len(p[0])):
        if p[0][j] < pl_min:
            pl_min = p[0][j]
            xl_min = p[1][j]
    return xl_min, pl_min


def polylines(a, b, epsilon=0.001, live=False):
    N = int(np.abs(b - a) / (2 * epsilon))
    x0 = 1 / (2 * L) * (function(a) - function(b) + L * (a + b))
    y0 = 1 / 2 * (function(a) - function(b) + L * (a + b))
    stop = 0
    p0 = np.zeros(N)
    x = np.linspace(a, b, N)
    p = np.array([p0, x])
    for i in range(N):
        if x[i] <= x0:
            p[0][i] = function(a) - L * (x[i] - a)
        else:
            p[0][i] = function(b) + L * (x[i] - b)
    x_min = x0
    x1_k = x_min
    plt.plot(p[1], p[0])
    plt.plot(x, function(x))
    plt.show()
    step = 1
    poly(p, x1_k, a, b, step, epsilon=0.001)
    return


def try_end_poly(p):
    for i in range(len(p[0])):
        if p[0][i] > function(p[1][i]):
            return 1
    return 0


def poly(p, x1_k, a, b, step, epsilon=0.001):
    x_prev, p_prev = p_func_min(p)
    N = int(np.abs(b - a) / (2 * epsilon))
    x = np.linspace(a, b, N)
    x_111 = np.linspace(a_global, b_global, N)
    x_min = x1_k
    x_bord = x_min
    p1_f = np.empty(N)
    p1 = np.array([p1_f, x])
    for i in range(N):
        p1[0][i] = max(p[0][i], assistant_func(x_min, x[i]))
    x1_opt, p1_opt = p_func_min(p1)
    plt.plot(p1[1], p1[0])
    plt.plot(x_111, function(x_111))
    plt.show()
    delta = 1 / (2 * L) * (function(x_prev) - p_prev)
    x1_k = x_min - delta
    x2_k = x_min + delta
    step += 1
    poly(p1, x2_k, x_bord, b, step)
    poly(p1, x1_k, a, x_bord, step)
    return 0

a_global = -0.5
b_global = 1
polylines(-0.5, 1)
