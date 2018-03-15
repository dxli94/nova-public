import argparse
import sys

import SuppFuncUtils

from DataReader import DataReader
from Plotter import Plotter
from PostOperator import PostOperator


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--path', help='path to a (single) instance file.')
#     parser.add_argument('--dt', type=int, help='direction type. 0 for box; 1 for octagonal.')
#     parser.add_argument('--horizon', type=float, help='time horizon.')
#     parser.add_argument('--sf', type=float, help='sampling time.')
#     parser.add_argument('--opvars', type=int, nargs='*', help='two index of variables to plot. Indexing from 0! First'
#                                                               'two dimensions (0, 1) by default.')
#
#     args = parser.parse_args()
#     if args.opvars is not None:
#         args.opvars = tuple(args.opvars)
#         if len(args.opvars) is not 2:
#             parser.error("Incorrect number of output variables provided! Either omitted, or two are required!")
#     else:
#         args.opvars = (0, 1)
#     return args


def main():
    # args = parse_args()
    # instance_file = args.path
    # direction_type = args.dt
    # time_horizon = args.horizon
    # samp_time = args.sf

    print(sys.argv)
    instance_file_path = sys.argv[1]  # expecting python main.py instance_file.txt

    assert sys.argv[1] is not None

    data_reader = DataReader(path2instance=instance_file_path)
    direction_type, time_horizon, samp_time, opvars, sys_dynamics = data_reader.read_data()

    sys_dim = sys_dynamics.get_dim()
    directions = SuppFuncUtils.generate_directions(direction_type=direction_type,
                                                   dim=sys_dim)

    post_opt = PostOperator(sys_dynamics, directions)

    sf_mat = post_opt.compute_post(time_horizon, samp_time)
    images = post_opt.get_images(opdims=opvars, sf_mat=sf_mat)

    plotter = Plotter(images, opvars)
    plotter.save_polygons_to_file()


if __name__ == '__main__':
    main()
