import numpy as np

import transport_pr as tp

# if __name__ == '__main__':
#     data = tp.TransportVS(
#         np.array([12, 30, 13]),
#         np.array([23, 10, 12, 10]),
#         np.array([
#             [64, 32, 45, 12],
#             [32, 78, 23, 90],
#             [88, 67, 10, 32],
#         ]),
#     )
#     tp.solve(data)

if __name__ == '__main__':
    data = tp.TransportVS(
        np.array([150, 110, 200]),
        np.array([100, 70, 130, 110, 50]),
        np.array([
            [20, 3, 9, 15, 35],
            [14, 10, 10, 20, 46],
            [25, 11, 16, 16, 48]
        ]),
    )
    tp.solve(data)

# if __name__ == '__main__':
#     """Degenerative problem"""
#     data = tp.TransportVS(
#         np.array([150, 150, 200]),
#         np.array([100, 70, 130, 110, 50]),
#         np.array([
#             [20, 3, 9, 15, 35],
#             [14, 10, 12, 20, 46],
#             [25, 11, 16, 16, 48]
#         ]),
#     )
#     tp.solve(data)
