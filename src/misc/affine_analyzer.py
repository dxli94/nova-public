import argparse

from misc.affine_post_opt import PostOperator
from utils import suppfunc_utils
from utils.data_reader import DataReader
from utils.glpk_wrapper import GlpkWrapper
from utils.plotter import Plotter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to a (single) instance file.')
    parser.add_argument('--dt', type=int, help='direction type. 0 for box; 1 for octagonal.')
    parser.add_argument('--horizon', type=float, help='time horizon.')
    parser.add_argument('--sampling_time', type=float, help='sampling time.')
    parser.add_argument('--opvars', type=int, nargs='*', help='two index of variables to plot. Indexing from 0! First'
                                                              'two dimensions (0, 1) by default.')

    args = parser.parse_args()
    if args.opvars is not None:
        args.opvars = tuple(args.opvars)
        if len(args.opvars) is not 2:
            parser.error("Incorrect number of output variables provided! Either omitted, or two are required!")
    else:
        args.opvars = (0, 1)
    return args


def main():
    args = parse_args()
    instance_file = args.path
    direction_type = args.dt
    time_horizon = args.horizon
    samp_time = args.sampling_time
    opvars = args.opvars

    data_reader = DataReader(path2instance=instance_file)

    sys_dynamics = data_reader.read_data()
    directions = suppfunc_utils.generate_directions(direction_type=direction_type,
                                                    dim=sys_dynamics.get_dim())

    lp = GlpkWrapper(sys_dynamics.get_dim())

    post_opt = PostOperator()

    sf_mat = post_opt.compute_post(sys_dynamics, directions, time_horizon, samp_time, lp)
    images = post_opt.get_projections(directions=directions, opdims=opvars, sf_mat=sf_mat)

    # for i in range(len(images)):
    #     for v in images[i].vertices:
    #         print('%.5f %.5f' % v)
    #     print('\n', end='')

    plotter = Plotter(images)
    plotter.save_polygons_to_file()


if __name__ == '__main__':
    main()
