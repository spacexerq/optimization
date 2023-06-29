from .algorithm import *


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
        report_list.extend([f'Degenerate plane: {check_res}', "\n"])
        if check_res:
            make_start_plan_non_degenerate(x)
            report_list.extend([
                '',
                'Trying to make plan non degenerate:',
                (x.copy(), data.a, data.b), "\n"
            ])

        while True:
            cost = data.calculate_cost(x)
            report_list.append(f'Target function: {cost}')

            p = data.calculate_potentials(x)
            report_list.append(f'Potentials: {p}')
            report_list.append((data.c, x.copy()))

            check_res = data.is_plan_optimal(x, p)
            report_list.append(f'Optimal plan: {check_res}')
            if check_res:
                report_list.extend(['', 'Answer:', (x.copy()), f'Target function: {cost}'])
                break

            cycle_path = find_cycle_path(x, data.get_best_free_cell(x, p))
            report_list.append(f'Cycle of the recalculation: {cycle_path}')
            report_list.append((x.copy()))

            recalculate_plan(x, cycle_path)
            report_list.extend(['Plan after the recalculation:', (x.copy(), data.a, data.b), "\n"])

    finally:
        report_text(report_list)
