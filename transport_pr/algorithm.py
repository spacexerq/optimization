import numpy as np

from .class_data_block import TransportVS

epsilon = 1e-10


def get_start_plan(data: TransportVS):
    """North-West method"""
    res = np.zeros((data.m, data.n))
    a = data.a.copy()
    b = data.b.copy()
    i = 0
    j = 0

    while i < data.m and j < data.n:
        x = min(a[i], b[j])
        a[i] -= x
        b[j] -= x

        res[i][j] = x

        if a[i] == 0:
            i += 1

        if b[j] == 0:
            j += 1

    return res


def is_degenerate_plan(x):
    m, n = x.shape
    return True if np.count_nonzero(x) != m + n - 1 else False


def find_cycle_path(x, start_pos):
    def get_posible_moves(bool_table, path):
        posible_moves = np.full(bool_table.shape, False)
        current_pos = path[-1]
        prev_pos = path[-2] if len(path) > 1 else (epsilon, epsilon)

        if current_pos[0] != prev_pos[0]:
            posible_moves[current_pos[0]] = True

        if current_pos[1] != prev_pos[1]:
            posible_moves[:, current_pos[1]] = True

        return list(zip(*np.nonzero(posible_moves * bool_table)))

    res = [start_pos]
    bool_table = x != 0

    while True:
        current_pos = res[-1]

        bool_table[current_pos[0]][current_pos[1]] = False

        if len(res) > 3:
            bool_table[start_pos[0]][start_pos[1]] = True

        posible_moves = get_posible_moves(bool_table, res)

        if start_pos in posible_moves:
            res.append(start_pos)
            return res

        if not posible_moves:
            for i, j in res[1:-1]:
                bool_table[i][j] = True

            res = [start_pos]
            continue

        res.append(posible_moves[0])


def recalculate_plan(x, cycle_path):
    o = np.min([x[i][j] for i, j in cycle_path[1:-1:2]])
    minus_cells_equal_to_o = [(i, j) for i, j in cycle_path[1:-1:2] if np.isnan(x[i][j]) or x[i][j] == o]

    if np.isnan(o):
        i, j = cycle_path[0]
        x[i][j] = epsilon
        i, j = minus_cells_equal_to_o[0]
        x[i][j] = 0
        return o

    for k, (i, j) in enumerate(cycle_path[:-1]):
        if (i, j) in minus_cells_equal_to_o:
            if minus_cells_equal_to_o.index((i, j)) == 0:
                x[i][j] = 0
            else:
                x[i][j] = epsilon
            continue

        if np.isnan(x[i][j]):
            x[i][j] = 0
        if k % 2 == 0:
            x[i][j] += o
        else:
            x[i][j] -= o
    return o


def make_start_plan_non_degenerate(x):
    """Fix!!!"""
    m, n = x.shape

    while np.count_nonzero(x) != m + n - 1:
        for i in range(m):
            if np.count_nonzero(x[i]) == 1:
                j = np.nonzero(x[i])[0][0]

                if np.count_nonzero(x[:, j]) == 1:
                    if np.nonzero(x[:, j])[0][0] == i:
                        if i < m - 1:
                            x[i + 1][j] = epsilon
                        else:
                            x[i - 1][j] = epsilon
                        break


def solve(data: TransportVS):
    def report_text(report):
        for i in report:
            if len(i) != 1 and type(i) != str:
                for j in range(len(i)):
                    print(i[j])
            else:
                print(i)
    report_list = ['Data given:', (data.c, data.a, data.b), '']
    try:
        diff = data.get_supply_demand_difference()
        report_list.append(f'Supply to demand difference: {diff}')
        report_list.append(f'Balance condition: {True if diff == 0 else False}')
        assert not diff, "No balance in the problem"
        x = get_start_plan(data)
        report_list.extend(['Starting plan. Method used - north-west corner:',
                            (x.copy(), data.a, data.b)])

        check_res = is_degenerate_plan(x)
        report_list.extend([f'Degenerate plane: {check_res}'])
        if check_res:
            make_start_plan_non_degenerate(x)
            report_list.extend([
                '',
                'Trying to make plan non degenerative:',
                (x.copy(), data.a, data.b),
            ])

        while True:
            cost = data.calculate_cost(x)
            report_list.append(f'Target function: {cost}')

            p = data.calculate_potentials(x)
            report_list.append(f'Potentials: {p}')
            report_list.append((data.c, p, x.copy()))

            check_res = data.is_plan_optimal(x, p)
            report_list.append(f'Optimal plan: {check_res}')
            if check_res:
                report_list.extend(['', 'Answer:', (x.copy(), data.a, data.b), f'Target function: {cost}'])
                break

            cycle_path = find_cycle_path(x, data.get_best_free_cell(x, p))
            report_list.append(f'Cycle of the recalculation: {cycle_path}')
            report_list.append((x.copy(), cycle_path, data.a, data.b))

            recalculate_plan(x, cycle_path)
            report_list.extend(['Plan after the recalculation:', (x.copy(), data.a, data.b)])

    finally:
        report_text(report_list)
