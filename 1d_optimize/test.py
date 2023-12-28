import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output

def live_plot_parabolic(data_x, data_f, list_parabolic, a, b, title=''):
    clear_output(wait=True)
    plt.figure()
    x_space = np.linspace(a, b, 100)
    for i in range(len(data_f)):
        plt.scatter(data_x, data_f, color='red', s=4)
    a, b, c = calc_parabola_vertex(list_parabolic)
    x_pos = np.arange(a, b, 0.01)
    y_pos = []
    for x in range(len(x_pos)):
        x_val = x_pos[x]
        y = (a * (x_val ** 2)) + (b * x_val) + c
        y_pos.append(y)
    print(y_pos)
    plt.plot(x_pos, y_pos)
    plt.plot(x_space, function(x_space), label="function")
    plt.title(title)
    plt.grid(True)
    plt.xlabel('Brute force')
    plt.legend(loc='center left')  # the plot evolves to the right
    plt.show()


def calc_parabola_vertex(list_loc):
    '''
    Adapted and modifed to get the unknowns for defining a parabola:
    http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
    '''
    x1, y1, x2, y2, x3, y3 = list_loc
    denom = 0.1
    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    B = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
    C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom

    return A, B, C


def function(x):
    return x ** 2 - 2 * x - 2 * np.cos(x)


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
            live_plot_parabolic(list_x, list_f, list_parabolic, a_loc, b_loc)
    return f_med, x_med, step

parabolic(-0.5,1,live=True)