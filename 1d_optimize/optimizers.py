from typing import Callable, Tuple, Literal

import numpy as np


"""
TODO:
* перебор
* Bitwise Search Method
* дихотомия 
*10*x*np.log(x) - np.power(x, 2)/2 метод парабол 
* метод золотого сечения
* метод Ньютона 
    * Марквардт
    * метод одной костальной
    * Метод рафсона
* метод хорд 
* метод ломанных
"""

# def _search(func: Callable[[float], float], ab: Tuple[float, float] = (0, 1), eps: float = 1e-5, direction: Literal["min", "max"] = "min") -> Tuple[float, float]:
#     """
#     Требуется условие унимодальной функции
#     _ Search Method:

#     Returns:
#     Tuple[float, float]: The optimal function value and corresponding x value.
#     """
#     # приводим задачу к нахождению минимума
#     if direction == "max":
#         def target(x) -> float: return -func(x)
#     else:
#         def target(x) -> float: return func(x)

#     pass


def grid_search(func: Callable[[float], float], ab: Tuple[float, float] = (0., 1.), n: int = 10, direction: Literal["min", "max"] = "min") -> Tuple[float, float]:
    """Grid Search Method

    Args:
        func (Callable[[float], float]): objective function
        ab (Tuple[float, float], optional): x range. Defaults to (0., 1.).
        n (int, optional): number of iterations. Defaults to 10.
        direction (Literal['min', 'max'], optional): direction of optimization. Defaults to "min".

    Raises:
        ValueError: if a > b in ab

    Returns:
        Tuple[float, float]: optimal value and point of it
    """
    # приводим задачу к нахождению минимума
    if direction == "max":
        def target(x): return -func(x)
    else:
        def target(x): return func(x)

    a, b = ab
    if a >= b:
        raise ValueError("a >= b")

    # решаем
    t_opt = target(a)
    x_opt = a
    step = (b-a) / n
    for i in range(1, n + 1):
        x = a + step * i
        if target(x) < t_opt:
            x_opt = x
            t_opt = target(x)

    return t_opt if direction == "min" else -t_opt, x_opt


def bitwise_search(func: Callable[[float], float], ab: Tuple[float, float] = (0, 1), direction: Literal["min", "max"] = "min") -> Tuple[float, float]:
    """
    Требуется условие унимодальной функции   
    Bitwise Search Method:

    Returns:
    Tuple[float, float]: The optimal function value and corresponding x value.
    """

    # приводим задачу к нахождению минимума
    if direction == "max":
        def target(x): return -func(x)
    else:
        def target(x): return func(x)

    a, b = ab
    if a >= b:
        raise ValueError("a >= b")

    epsilon = 1e-6  # Small value for step size

    while (b - a) > epsilon:
        x1 = a + (b - a) / 3
        x2 = b - (b - a) / 3
        if target(x1) < target(x2):
            b = x2
        else:
            a = x1

    x_opt = (a + b) / 2
    t_opt = target(x_opt)

    return t_opt if direction == "min" else -t_opt, x_opt


def dichotomous_search(func: Callable[[float], float], ab: Tuple[float, float] = (0, 1), eps: float = 1e-5, direction: Literal["min", "max"] = "min") -> Tuple[float, float]:
    """
    Требуется условие унимодальной функции   
    Dichotomous Search Method:

    Returns:
    Tuple[float, float]: The optimal function value and corresponding x value.
    """
    # приводим задачу к нахождению минимума
    if direction == "max":
        def target(x) -> float: return -func(x)
    else:
        def target(x) -> float: return func(x)

    a, b = ab
    if a >= b:
        raise ValueError("a >= b")

    while b - a > eps:
        mid = (b+a) / 2.0
        x1 = mid - eps
        x2 = mid + eps

        if target(x1) < target(x2):
            b = mid
        else:
            a = mid

    x_opt: float = (a + b) / 2
    t_opt = target(x_opt)

    return t_opt if direction == "min" else -t_opt, x_opt


def golden_section_search(func: Callable[[float], float], ab: Tuple[float, float] = (0, 1), eps: float = 1e-5, direction: Literal["min", "max"] = "min") -> Tuple[float, float]:
    """
    Требуется условие унимодальной функции   
    Golden Section Search Method:

    Returns:
    Tuple[float, float]: The optimal function value and corresponding x value.
    """
    # приводим задачу к нахождению минимума
    if direction == "max":
        def target(x) -> float: return -func(x)
    else:
        def target(x) -> float: return func(x)

    a, b = ab
    if a >= b:
        raise ValueError("a >= b")

    gr = (np.sqrt(5) + 1)/2

    c = b - (b - a) / gr
    d = a + (b - a) / gr

    while abs(b - a) > eps:
        if target(c) < target(d):  # f(c) > f(d) to find the maximum
            b = d
        else:
            a = c

        # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = b - (b - a) / gr
        d = a + (b - a) / gr

    x_opt: float = (a + b) / 2
    t_opt = target(x_opt)

    return t_opt if direction == "min" else -t_opt, x_opt


def parabolic_interpolation(func: Callable[[float], float],
                            ab: Tuple[float, float] = (0, 1),
                            eps: float = 1e-5,
                            max_iter: int = 100,
                            direction: Literal["min", "max"] = "min") -> Tuple[float, float]:
    """Метод парабол 

    Args:
        func (Callable[[float], float]): Целевая функция
        ab (Tuple[float, float], optional): отрезок поиска оптимума. Defaults to (0, 1).
        eps (float, optional): точность оптимального плана. Defaults to 1e-5.
        max_iter (int, optional): максимальное число итераций. Defaults to 100.
        direction (Literal["min", "max"], optional): направление поиска. Defaults to "min".

    Raises:
        ValueError: если a >= b

    Returns:
        Tuple[float, float]: оптимум и оптимальный план
    """

    # приводим задачу к нахождению минимума
    if direction == "max":
        def target(x) -> float: return -func(x)
    else:
        def target(x) -> float: return func(x)

    a, b = ab
    if a >= b:
        raise ValueError("a >= b")

    # Вычисляем значения функции в точках a и b
    x1, x2, x3 = a, (a+b)/2, b
    a0, a1, a2 = target(x1), (target(x2) - target(x1)) / (x2-x1), (1/(x3-x2))*(
        ((target(x3) - target(x1))/(x3-x1)) - ((target(x2) - target(x1))/(x2-x1)))
    x = 0.5*(x1 + x2 - a1/a2)

    if target(x) >= target(x2) and (x > x1):
        x1 = x

    if target(x) <= target(x2) and (x < x3):
        x1 = x2
        x2 = x

    iter_count = 0
    delta = 1
    res = np.inf
    # Пока не достигнута требуемая точность
    while delta > eps and iter_count < max_iter:
        
        a1 = (target(x2) - target(x1)) / (x2-x1)
        a2 = (1/(x3-x2))*(((target(x3) - target(x1))/(x3-x1)) - ((target(x2) - target(x1))/(x2-x1)))
        x = 0.5*(x1 + x2 - a1/a2)
        delta = abs(res - x)
        res = x

        if target(x) >= target(x2) and (x > x1):
            x1 = x

        if target(x) <= target(x2) and (x < x3):
            x1 = x2
            x2 = x

        iter_count += 1
    #print(iter_count) 
    # Возвращаем значение минимума и координату x
    return target(x) if direction == "min" else -target(x), x



def newtons_method(func: Callable[[float], float],
                   ab: Tuple[float, float] = (0, 1),
                   x0: float | None = None,
                   eps: float = 1e-5,
                   dx: float = 1e-2,
                   max_iter: int = 100,
                   direction: Literal["min", "max"] = "min") -> Tuple[float, float]:
    """
    Метод Ньютона для поиска оптимума

    Args:
        func (Callable[[float], float]): Целевая функция
        ab (Tuple[float, float], optional): отрезок поиска оптимума. Defaults to (0, 1).
        x0 (float | None, optional): Начальная точка, если None, то берется середина отрезка. Defaults to None.
        eps (float, optional): пороговое значение для градиента функции. Defaults to 1e-5.
        dx (float, optional): шаг конечных разностей для первой и второй производной. Defaults to 1e-2.
        max_iter (int, optional): максимальное число итераций. Defaults to 100.
        direction (Literal["min", "max"], optional): Направление поиска оптимума. Defaults to "min".

    Returns:
        Tuple[float, float]: Оптимальное значение функции и оптимальный план
    """

    # приводим задачу к нахождению минимума
    if direction == "max":
        def target(x): return -func(x)
        def d_target(x): return -(-func(x-dx) + func(x+dx)) / (2*dx)
        def d2_target(x): return -(func(x-dx) - 2 * func(x) + func(x+dx))/dx**2
    else:
        def target(x): return func(x)
        def d_target(x): return (-func(x-dx) + func(x+dx)) / (2*dx)
        def d2_target(x): return (func(x-dx) - 2 * func(x) + func(x+dx))/dx**2

    a, b = ab
    if x0 is None:
        x0 = (a+b)/2
    x = x0

    iter_count = 0

    while iter_count < max_iter:
        grad = d_target(x)
        hessian = d2_target(x)
        # Newton's update formula
        x = x - grad/hessian

        # Check for convergence
        if np.abs(grad) < eps:
            break

        iter_count += 1

    t_opt = target(x)

    return t_opt if direction == "min" else -t_opt, x


def newtons_raf_method(func: Callable[[float], float],
                       ab: Tuple[float, float] = (0, 1),
                       x0: float | None = None,
                       eps: float = 1e-5,
                       dx: float = 1e-2,
                       max_iter: int = 100,
                       direction: Literal["min", "max"] = "min") -> Tuple[float, float]:
    """

    Args:
        func (Callable[[float], float]): Целевая функция
        ab (Tuple[float, float], optional): отрезок поиска оптимума. Defaults to (0, 1).
        x0 (float | None, optional): Начальная точка, если None, то берется середина отрезка. Defaults to None.
        eps (float, optional): пороговое значение для градиента функции. Defaults to 1e-5.
        dx (float, optional): шаг конечных разностей для первой и второй производной. Defaults to 1e-2.
        max_iter (int, optional): максимальное число итераций. Defaults to 100.
        direction (Literal["min", "max"], optional): Направление поиска оптимума. Defaults to "min".

    Returns:
        Tuple[float, float]: Оптимальное значение функции и оптимальный план
    """

    # приводим задачу к нахождению минимума
    if direction == "max":
        def target(x): return -func(x)
        def d_target(x): return -(-func(x-dx) + func(x+dx)) / (2*dx)
        def d2_target(x): return -(func(x-dx) - 2 * func(x) + func(x+dx))/dx**2
    else:
        def target(x): return func(x)
        def d_target(x): return (-func(x-dx) + func(x+dx)) / (2*dx)
        def d2_target(x): return (func(x-dx) - 2 * func(x) + func(x+dx))/dx**2

    a, b = ab
    if x0 is None:
        x0 = (a+b)/2
    x = x0

    iter_count = 0

    while iter_count < max_iter:
        grad = d_target(x)
        hessian = d2_target(x)
        grad = d_target(x)
        x_t = x - grad/hessian
        x = x - grad/hessian * grad / (grad + d_target(x_t))

        if np.abs(grad) < eps:
            break

        iter_count += 1
    print(iter_count)
    t_opt = target(x)

    return t_opt if direction == "min" else -t_opt, x


def newtons_m_method(func: Callable[[float], float],
                     ab: Tuple[float, float] = (0, 1),
                     x0: float | None = None,
                     eps: float = 1e-5,
                     dx: float = 1e-2,
                     max_iter: int = 100,
                     direction: Literal["min", "max"] = "min") -> Tuple[float, float]:
    """

    Args:
        func (Callable[[float], float]): Целевая функция
        ab (Tuple[float, float], optional): отрезок поиска оптимума. Defaults to (0, 1).
        x0 (float | None, optional): Начальная точка, если None, то берется середина отрезка. Defaults to None.
        eps (float, optional): пороговое значение для градиента функции. Defaults to 1e-5.
        dx (float, optional): шаг конечных разностей для первой и второй производной. Defaults to 1e-2.
        max_iter (int, optional): максимальное число итераций. Defaults to 100.
        direction (Literal["min", "max"], optional): Направление поиска оптимума. Defaults to "min".

    Returns:
        Tuple[float, float]: Оптимальное значение функции и оптимальный план
    """

    # приводим задачу к нахождению минимума
    if direction == "max":
        def target(x): return -func(x)
        def d_target(x): return -(-func(x-dx) + func(x+dx)) / (2*dx)
        def d2_target(x): return -(func(x-dx) - 2 * func(x) + func(x+dx))/dx**2
    else:
        def target(x): return func(x)
        def d_target(x): return (-func(x-dx) + func(x+dx)) / (2*dx)
        def d2_target(x): return (func(x-dx) - 2 * func(x) + func(x+dx))/dx**2

    a, b = ab
    if x0 is None:
        x0 = (a+b)/2
    x = x0

    iter_count = 0
    p = 1
    grad_ = 1

    while iter_count < max_iter:

        grad = d_target(x)

        hessian = d2_target(x)
        # Newton's update formula
        x = x - p * grad/hessian

        if abs(grad) > abs(grad_):
            p *= 2
        else:
            p /= 2

        grad_ = grad

        # Check for convergence
        if abs(grad) < eps:
            break

        iter_count += 1

    t_opt = target(x)

    return t_opt if direction == "min" else -t_opt, x


def newtons_static_method(func: Callable[[float], float],
                          ab: Tuple[float, float] = (0, 1),
                          x0: float | None = None,
                          eps: float = 1e-5,
                          dx: float = 1e-2,
                          max_iter: int = 100,
                          direction: Literal["min", "max"] = "min") -> Tuple[float, float]:
    """
    Метод Ньютона для поиска оптимума

    Args:
        func (Callable[[float], float]): Целевая функция
        ab (Tuple[float, float], optional): отрезок поиска оптимума. Defaults to (0, 1).
        x0 (float | None, optional): Начальная точка, если None, то берется середина отрезка. Defaults to None.
        eps (float, optional): пороговое значение для градиента функции. Defaults to 1e-5.
        dx (float, optional): шаг конечных разностей для первой и второй производной. Defaults to 1e-2.
        max_iter (int, optional): максимальное число итераций. Defaults to 100.
        direction (Literal["min", "max"], optional): Направление поиска оптимума. Defaults to "min".

    Returns:
        Tuple[float, float]: Оптимальное значение функции и оптимальный план
    """

    # приводим задачу к нахождению минимума
    if direction == "max":
        def target(x): return -func(x)
        def d_target(x): return -(-func(x-dx) + func(x+dx)) / (2*dx)
        def d2_target(x): return -(func(x-dx) - 2 * func(x) + func(x+dx))/dx**2
    else:
        def target(x): return func(x)
        def d_target(x): return (-func(x-dx) + func(x+dx)) / (2*dx)
        def d2_target(x): return (func(x-dx) - 2 * func(x) + func(x+dx))/dx**2

    a, b = ab
    if x0 is None:
        x0 = (a+b)/2
    x = x0

    iter_count = 0
    hessian = d2_target(x)

    while iter_count < max_iter:
        grad = d_target(x)

        # Newton's update formula
        x = x - grad/hessian

        # Check for convergence
        if abs(grad) < eps:
            break

        iter_count += 1

    t_opt = target(x)

    return t_opt if direction == "min" else -t_opt, x


def chords_method(func: Callable[[float], float],
                  ab: Tuple[float, float] = (0, 1),
                  eps: float = 1e-5,
                  dx: float = 1e-2,
                  max_iter: int = 100,
                  direction: Literal["min", "max"] = "min") -> Tuple[float, float]:
    """

    Args:
        func (Callable[[float], float]): Целевая функция
        ab (Tuple[float, float], optional): отрезок поиска оптимума. Defaults to (0, 1).
        x0 (float | None, optional): Начальная точка, если None, то берется середина отрезка. Defaults to None.
        eps (float, optional): пороговое значение для градиента функции. Defaults to 1e-5.
        dx (float, optional): шаг конечных разностей для первой и второй производной. Defaults to 1e-2.
        max_iter (int, optional): максимальное число итераций. Defaults to 100.
        direction (Literal["min", "max"], optional): Направление поиска оптимума. Defaults to "min".

    Returns:
        Tuple[float, float]: Оптимальное значение функции и оптимальный план
    """

    # приводим задачу к нахождению минимума
    if direction == "max":
        def target(x): return -func(x)
        def d_target(x): return -(-func(x-dx) + func(x+dx)) / (2*dx)
    else:
        def target(x): return func(x)
        def d_target(x): return (-func(x-dx) + func(x+dx)) / (2*dx)

    a, b = ab

    x = a - d_target(a) / (d_target(a) - d_target(b)) * (a-b)
    grad = d_target(x)
    iter_count = 0

    while abs(grad) > eps and iter_count < max_iter:
        grad = d_target(x)

        if grad > 0:
            b = x
        else:
            a = x

        iter_count += 1

        x = a - d_target(a) / (d_target(a) - d_target(b)) * (a-b)

    t_opt = target(x)

    return t_opt if direction == "min" else -t_opt, x


def b_lines_method(func: Callable[[float], float],
                   ab: Tuple[float, float] = (0, 1),
                   eps: float = 1e-5,
                   max_iter: int = 100,
                   direction: Literal["min", "max"] = "min") -> Tuple[float, float]:
    """_summary_

    Args:
        func (Callable[[float], float]): _description_
        ab (Tuple[float, float], optional): _description_. Defaults to (0, 1).
        eps (float, optional): _description_. Defaults to 1e-5.
        max_iter (int, optional): _description_. Defaults to 100.
        direction (Literal[&quot;min&quot;, &quot;max&quot;], optional): _description_. Defaults to "min".

    Raises:
        ValueError: _description_

    Returns:
        Tuple[float, float]: _description_
    """

    # приводим задачу к нахождению минимума
    if direction == "max":
        def target(x) -> float: return -func(x)
    else:
        def target(x) -> float: return func(x)

    a, b = ab
    if a >= b:
        raise ValueError("a >= b")

    l = max([(-target(x-eps) + target(x+eps)) / (2*eps) for x in np.linspace(a, b, int((b-a)/1e-2))])
    x_opt = 1/(2*l)*(target(a) - target(b) + l*(a+b))
    p = 1/2*(target(a) + target(b) + l * (a-b))
    delta = 1/(2*l)*(target(x_opt) - p)
    iter_count = 0
    while ((2*l*delta) > eps) and iter_count < max_iter:
        x1 = x_opt + delta
        x2 = x_opt - delta
        if target(x1) < target(x2):
            x_opt = x1
        else:
            x_opt = x2
        p = (1/2)*(target(x_opt) + p)
        delta = 1/(2*l)*(target(x_opt) - p)

    return target(x_opt), x_opt
