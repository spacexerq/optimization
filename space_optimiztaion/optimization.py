import numpy as np
import scipy as sc
from matplotlib import pyplot as plt
from IPython.display import clear_output
from matplotlib import cm
from copy import deepcopy

epsilon = 1e-4


def function(x1_loc, x2_loc):
    f = lambda x1_lmb, x2_lmb: (x1_lmb * x1_lmb + (x2_lmb + 1) * (x2_lmb + 1)) * (
            x1_lmb * x1_lmb + (x2_lmb - 1) * (x2_lmb - 1))
    output = np.zeros((len(x1_loc), len(x2_loc)))
    for i in range(len(x1_loc)):
        for j in range(len(x2_loc)):
            output[i][j] = f(x1_loc[i], x2_loc[j])
    return output


def function_point(x1_loc, x2_loc):
    f = lambda x1_lmb, x2_lmb: (x1_lmb * x1_lmb + (x2_lmb + 1) * (x2_lmb + 1)) * (
            x1_lmb * x1_lmb + (x2_lmb - 1) * (x2_lmb - 1))
    output = f(x1_loc, x2_loc)
    return output


def quadratic(x1_loc, x2_loc, x10, x20):
    g = lambda x1, x2, x10, x20: 1 + 15 * x10 ** 4 - 2 * x2 ** 2 + 2 * x1 ** 2 * (
            1 + 3 * x10 ** 2 + x20 ** 2) - 4 * x1 * x10 * (
                                         1 + 5 * x10 ** 2 - 2 * x2 * x20 + 5 * x20 ** 2) + 2 * x10 ** 2 * (
                                         2 + x2 ** 2 - 10 * x2 * x20 + 15 * x20 ** 2) + x20 * (
                                         4 * x2 - 4 * x20 + 6 * x2 ** 2 * x20 - 20 * x2 * x20 ** 2 + 15 * x20 ** 3)
    output = np.zeros((len(x1_loc), len(x2_loc)))
    for i in range(len(x1_loc)):
        for j in range(len(x2_loc)):
            output[i][j] = g(x1_loc[i], x2_loc[j], x10, x20)
    return output


def quadratic_point(x1_loc, x2_loc, x10, x20):
    g = lambda x1, x2, x10, x20: 1 + 15 * x10 ** 4 - 2 * x2 ** 2 + 2 * x1 ** 2 * (
            1 + 3 * x10 ** 2 + x20 ** 2) - 4 * x1 * x10 * (
                                         1 + 5 * x10 ** 2 - 2 * x2 * x20 + 5 * x20 ** 2) + 2 * x10 ** 2 * (
                                         2 + x2 ** 2 - 10 * x2 * x20 + 15 * x20 ** 2) + x20 * (
                                         4 * x2 - 4 * x20 + 6 * x2 ** 2 * x20 - 20 * x2 * x20 ** 2 + 15 * x20 ** 3)
    output = g(x1_loc, x2_loc, x10, x20)
    return output


def plot_both(angle1, angle2):
    mycmap = plt.get_cmap('gist_earth')
    x1_sample = np.linspace(-1, 1, 1000)
    x2_sample = np.linspace(-1.5, 1.5, 1000)
    x1_0 = 1 / 2
    x2_0 = 0
    z_exact = function(x1_sample, x2_sample)
    z_approx = quadratic(x1_sample, x2_sample, x1_0, x2_0)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x1_grid, x2_grid = np.meshgrid(x1_sample, x2_sample)
    ax.plot_surface(x1_grid, x2_grid, z_exact, cmap=mycmap)
    ax.plot_wireframe(x1_grid, x2_grid, z_approx, color='red')
    ax.view_init(angle1, angle2)
    plt.show()


# plot_both(10, 10)
# plot_both(10, 110)


def grad_func(x1_l, x2_l):
    grad_x1 = 4 * x1_l * (x1_l ** 2 + x2_l ** 2 + 1)
    grad_x2 = 4 * x2_l * (x1_l ** 2 + x2_l ** 2 - 1)
    return [grad_x1, grad_x2]


def grad_quad_func(x1_l, x2_l, x10_l, x20_l):
    grad_x1 = 4 * x1_l * (1 + 3 * x10_l ** 2 + x20_l ** 2) - 4 * x10_l * (
            1 + 5 * x10_l ** 2 - 2 * x2_l * x20_l + 5 * x20_l ** 2)
    grad_x2 = 4 * (x20_l + 2 * x1_l * x10_l * x20_l - 5 * x20_l * (x10_l ** 2 + x20_l ** 2) + x2_l * (
            -1 + x10_l ** 2 + 3 * x20_l ** 2))
    return [grad_x1, grad_x2]


def vector_norm(vector):
    return np.sqrt(vector[0] ** 2 + vector[1] ** 2)


# x_{k+1} = x_{k} + alpha * grad f(x_{k})
def step_crushing(x0_loc, alpha=1, eps=epsilon):
    print("Method of crushing step")
    print("Given function from point " + str(x0_loc))
    x_alg = x0_loc
    x_next_step = [0, 0]
    x_sample = np.linspace(-1.5, 1.5, 1000)
    f_sample = function(x_sample, x_sample)
    plt.contour(x_sample, x_sample, f_sample, np.logspace(-1, 1, 15), cmap=cm.gnuplot2)
    plt.colorbar()
    x_steps = [x_alg]
    f_steps = [function_point(x_alg[0], x_alg[1])]
    plt.scatter(x_alg[1], x_alg[0], color='red', s=11)
    while vector_norm(grad_func(x_alg[0], x_alg[1])) >= eps:
        x_next_step[0] = x_alg[0] - alpha * grad_func(x_alg[0], x_alg[1])[0]
        x_next_step[1] = x_alg[1] - alpha * grad_func(x_alg[0], x_alg[1])[1]
        if function_point(x_next_step[0], x_next_step[1]) < function_point(x_alg[0], x_alg[1]):
            x_alg = deepcopy(x_next_step)
        else:
            alpha /= 2
        print(x_alg, function_point(x_alg[0], x_alg[1]))
        plt.scatter(x_alg[1], x_alg[0], color='red', s=11)
        x_steps.append(x_alg)
        f_steps.append(function_point(x_alg[0], x_alg[1]))
    plt.plot([i[1] for i in x_steps], [i[0] for i in x_steps], linewidth=2, color="orange")
    plt.show()
    plt.plot(f_steps)
    plt.title("Function value from number of iterations")
    plt.show()
    print("\n")
    return x_alg, function_point(x_alg[0], x_alg[1])


def step_crushing_quad(x0_loc, alpha=0.2, eps=epsilon):
    print("Method of crushing step")
    print("Quadratic function approximation in point " + str(x0_loc))
    x_alg = x0_loc
    x_next_step = [0, 0]
    x_sample = np.linspace(-1.5, 1.5, 1000)
    g_sample = quadratic(x_sample, x_sample, x0_loc[0], x0_loc[1])
    plt.contour(x_sample, x_sample, g_sample, np.logspace(0, 1, 20), cmap=cm.gnuplot2)
    plt.colorbar()
    x_steps = [x_alg]
    f_steps = [quadratic_point(x_alg[0], x_alg[1], x0_loc[0], x0_loc[1])]
    plt.scatter(x_alg[1], x_alg[0], color='red', s=11)
    while vector_norm(grad_quad_func(x_alg[0], x_alg[1], x0_loc[0], x0_loc[1])) >= eps:
        x_next_step[0] = x_alg[0] - alpha * grad_quad_func(x_alg[0], x_alg[1], x0_loc[0], x0_loc[1])[0]
        x_next_step[1] = x_alg[1] - alpha * grad_quad_func(x_alg[0], x_alg[1], x0_loc[0], x0_loc[1])[1]
        if quadratic_point(x_next_step[0], x_next_step[1], x0_loc[0], x0_loc[1]) < quadratic_point(x_alg[0], x_alg[1],
                                                                                                   x0_loc[0],
                                                                                                   x0_loc[1]):
            x_alg = deepcopy(x_next_step)
        else:
            alpha /= 2
        print(x_alg, quadratic_point(x_alg[0], x_alg[1], x0_loc[0], x0_loc[1]))
        plt.scatter(x_alg[1], x_alg[0], color='red', s=11)
        x_steps.append(x_alg)
        f_steps.append(quadratic_point(x_alg[0], x_alg[1], x0_loc[0], x0_loc[1]))
    plt.plot([i[1] for i in x_steps], [i[0] for i in x_steps], linewidth=2, color="orange")
    plt.show()
    plt.plot(f_steps)
    plt.title("Function value from number of iterations")
    plt.show()
    print("\n")
    return x_alg, quadratic_point(x_alg[0], x_alg[1], x0_loc[0], x0_loc[1])


def fast_descent(x0_loc, eps=epsilon):
    print("Method of the fastest gradient descent")
    print("Given function from point " + str(x0_loc))
    x_alg = x0_loc
    x_next_step = [0, 0]
    x_sample = np.linspace(-1.5, 1.5, 1000)
    f_sample = function(x_sample, x_sample)
    plt.contour(x_sample, x_sample, f_sample, np.logspace(-1, 1, 15), cmap=cm.gnuplot2)
    plt.colorbar()
    x_steps = [x_alg]
    f_steps = [function_point(x_alg[0], x_alg[1])]
    plt.scatter(x_alg[1], x_alg[0], color='red', s=11)
    while vector_norm(grad_func(x_alg[0], x_alg[1])) >= eps:
        p0 = -grad_func(x_alg[0], x_alg[1])[0]
        p1 = -grad_func(x_alg[0], x_alg[1])[1]
        phi = lambda alpha_k: function_point(x_alg[0] + alpha_k * p0, x_alg[1] + alpha_k * p1)
        alpha_opt = sc.optimize.minimize_scalar(phi).x
        x_next_step[0] = x_alg[0] - alpha_opt * grad_func(x_alg[0], x_alg[1])[0]
        x_next_step[1] = x_alg[1] - alpha_opt * grad_func(x_alg[0], x_alg[1])[1]
        x_alg = deepcopy(x_next_step)
        print(x_alg, function_point(x_alg[0], x_alg[1]))
        plt.scatter(x_alg[1], x_alg[0], color='red', s=11)
        x_steps.append(x_alg)
        f_steps.append(function_point(x_alg[0], x_alg[1]))
    plt.plot([i[1] for i in x_steps], [i[0] for i in x_steps], linewidth=2, color="orange")
    plt.show()
    plt.plot(f_steps)
    plt.title("Function value from number of iterations")
    plt.show()
    print("\n")
    return x_alg, function_point(x_alg[0], x_alg[1])


def fast_descent_quad(x0_loc, eps=epsilon, max_num=1000):
    print("Method of the fastest gradient descent")
    print("Given function from point " + str(x0_loc))
    x_alg = x0_loc
    x_next_step = [0, 0]
    x_sample = np.linspace(-1.5, 1.5, 1000)
    f_sample = quadratic(x_sample, x_sample, x_alg[0], x_alg[1])
    plt.contour(x_sample, x_sample, f_sample, np.logspace(-1, 1, 15), cmap=cm.gnuplot2)
    plt.colorbar()
    x_steps = [x_alg]
    f_steps = [quadratic_point(x_alg[0], x_alg[1], x0_loc[0], x0_loc[1])]
    plt.scatter(x_alg[1], x_alg[0], color='red', s=11)
    f_mem = quadratic_point(x_alg[0], x_alg[1], x0_loc[0], x0_loc[1])
    while vector_norm(grad_quad_func(x_alg[0], x_alg[1], x_alg[0], x_alg[1])) >= eps and len(x_steps)<=max_num:
        p0 = -grad_quad_func(x_alg[0], x_alg[1], x0_loc[0], x0_loc[1])[0] / vector_norm(
            grad_quad_func(x_alg[0], x_alg[1], x0_loc[0], x0_loc[1]))
        p1 = -grad_quad_func(x_alg[0], x_alg[1], x0_loc[0], x0_loc[1])[1] / vector_norm(
            grad_quad_func(x_alg[0], x_alg[1], x0_loc[0], x0_loc[1]))
        phi = lambda alpha_k: quadratic_point(x_alg[0] + alpha_k * p0, x_alg[1] + alpha_k * p1, x0_loc[0], x0_loc[1])
        alpha_opt = sc.optimize.minimize_scalar(phi).x
        x_next_step[0] = x_alg[0] - alpha_opt * grad_quad_func(x_alg[0], x_alg[1], x0_loc[0], x0_loc[1])[0]
        x_next_step[1] = x_alg[1] - alpha_opt * grad_quad_func(x_alg[0], x_alg[1], x0_loc[0], x0_loc[1])[1]
        x_alg = deepcopy(x_next_step)
        print(x_alg, quadratic_point(x_alg[0], x_alg[1], x0_loc[0], x0_loc[1]))
        plt.scatter(x_alg[1], x_alg[0], color='red', s=11)
        x_steps.append(x_alg)
        f_steps.append(quadratic_point(x_alg[0], x_alg[1], x0_loc[0], x0_loc[1]))
    plt.plot([i[1] for i in x_steps], [i[0] for i in x_steps], linewidth=2, color="orange")
    plt.show()
    plt.plot(f_steps)
    plt.title("Function value from number of iterations")
    plt.show()
    print("\n")
    return x_alg, quadratic_point(x_alg[0], x_alg[1], x0_loc[0], x0_loc[1])


# step_crushing([0.5, 0.5])
# step_crushing([0.5, 0])
# step_crushing([0.5, 0.25])
# step_crushing_quad([0.5, 0])
# print("Theoretical point:", round(9 / 14, 5), 0)
# fast_descent([0.5, 0.1])
# fast_descent([0.5, 0.25])
# fast_descent([0.5, 0])
# fast_descent_quad([0.5, 0], max_num=1000)
