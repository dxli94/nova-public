import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import utils.ppl_helper


class Plotter:
    images = []

    def __init__(self, images):
        self.images = images
        self.vertices_sorted = list(map(lambda im: self.sort_vertices(im), self.images))

    def sort_vertices(self, im):
        # corners = [(v[self.opvars[0]], v[self.opvars[1]]) for v in im.vertices]
        corners = utils.ppl_helper.get_2dVert_from_poly(im, 2)

        # corners = [v[:2] for v in im.vertices]
        n = len(corners)
        cx = float(sum(x for x, y in corners)) / n
        cy = float(sum(y for x, y in corners)) / n
        cornersWithAngles = []
        for x, y in corners:
            an = (np.arctan2(y - cy, x - cx) + 2.0 * np.pi) % (2.0 * np.pi)
            cornersWithAngles.append((x, y, an))
        cornersWithAngles.sort(key=lambda tup: tup[2])

        return list(map(lambda ca: (ca[0], ca[1]), cornersWithAngles))

    def save_polygons_to_file(self, filename='../out/outfile.out'):
        with open(filename, 'w') as opfile:
            for vertices in self.vertices_sorted:
                x, y = [elem[0] for elem in vertices], [elem[1] for elem in vertices]
                x.append(x[0])
                y.append(y[0])

                for xx, yy in zip(x, y):
                    opfile.write('%.15f %.15f' % (xx, yy) + '\n')
                opfile.write('\n')

    @staticmethod
    def plot_polygons(filelist, xlabel, ylabel):
        # print('Start reading file...')
        if not isinstance(filelist, list):
            filelist = [filelist]

        # 'palegreen', 'navy', 'mediumseagreen'
        colors = ['crimson', 'blue']
        linewidths = [0.5, 0.5]
        linestyles = ['solid', 'dashed']

        fig = plt.figure(1, dpi=90)
        ax = fig.add_subplot(111)
        ax.set_xlabel('$x_{}$'.format(xlabel))
        ax.set_ylabel('$x_{}$'.format(ylabel))

        for ipfile_path, color, lw, ls in zip(filelist, colors[:len(filelist)], linewidths[:len(filelist)],
                                              linestyles[:len(filelist)]):
            # print(ipfile_path)
            try:
                with open(ipfile_path) as ipfile:
                    content = ipfile.read().strip('\n')
                    polygons = content.split('\n\n')
                    vertices_sorted = list(map(lambda poly: poly.split('\n'), polygons))
            except FileExistsError:
                print('File does not exist %s' % ipfile_path)
            # print('Finished. \nStart plotting...')

            i = 0
            # stepsize = max(len(vertices_sorted) // 50, 1)
            stepsize = 1

            for vertices in vertices_sorted:
                if i % stepsize == 0:
                    x, y = [float(elem.split()[0]) for elem in vertices], [float(elem.split()[1]) for elem in vertices]
                    mat = np.transpose(np.array([x, y]))
                    poly1patch = patches.Polygon(mat, fill=False, edgecolor=color, linewidth=lw, linestyle=ls)
                    ax.add_patch(poly1patch)
                i += 1

        plt.autoscale(enable=True)

        return plt

    @staticmethod
    def plot_points_from_file(filelist, opdims, xlabel, ylabel):
        if not isinstance(filelist, list):
            filelist = [filelist]
        # print('Start reading file...')
        colors = ['red', 'blue']
        linewidths = [0.5, 0.5]
        linestyles = ['solid', 'dashed']

        fig = plt.figure(1, dpi=90)
        ax = fig.add_subplot(111)
        ax.set_xlabel('$x_{}$'.format(xlabel))
        ax.set_ylabel('$x_{}$'.format(ylabel))

        for ipfile_path, color, lw, ls in zip(filelist, colors[:len(filelist)], linewidths[:len(filelist)],
                                              linestyles[:len(filelist)]):
            # print(ipfile_path)
            try:
                with open(ipfile_path) as ipfile:
                    content = ipfile.read().strip('\n')
                    points = content.split('\n')
                    points = list(map(lambda p: p.split('\n')[0], points))
                    x = []
                    y = []

                    for p in points:
                        xy = p.split()
                        x.append(float(xy[opdims[0]]))
                        y.append(float(xy[opdims[1]]))
            except FileExistsError:
                print('File does not exist %s' % ipfile_path)

        plt.plot(x, y)
        plt.autoscale(enable=True)

    @staticmethod
    def plot_points(x, y, xlabel, ylabel):
        # print('Start reading file...')
        color = 'darkblue'
        linewidths = 0.2
        linestyles = 'solid'

        fig = plt.figure(1, dpi=90)
        ax = fig.add_subplot(111)
        ax.set_xlabel('$x_{}$'.format(xlabel))
        ax.set_ylabel('$x_{}$'.format(ylabel))

        plt.plot(x, y, color=color, ls=linestyles, lw=linewidths, alpha=0.8)
        plt.autoscale(enable=True)

    @staticmethod
    def plot_pivots(ipfile_path, opdims, color):
        # fig = plt.figure(1, dpi=90)
        # ax = fig.add_subplot(111)
        # ax.set_xlabel('$x_{1}$')
        # ax.set_ylabel('$x_{2}$')

        try:
            with open(ipfile_path) as ipfile:
                content = ipfile.read().strip('\n')
                points = content.split('\n')
                x = []
                y = []

                for point in points:
                    xy = point.split()
                    x.append(float(xy[opdims[0]]))
                    y.append(float(xy[opdims[1]]))
        except FileNotFoundError:
            print('File does not exist %s' % ipfile_path)

        plt.plot(x, y, 'o', color=color, markersize=8, alpha=0.5)
        plt.autoscale(enable=True)

    @staticmethod
    def save_plt(opfile):
        # plt.savefig(opfile, format='eps', dpi=500)
        plt.savefig(opfile, format='png', dpi=500)

    @staticmethod
    def make_2Dproj_pplpoly(directions, opdims, sf_mat):
        assert len(opdims) == 2, 'Support projection on 2d space only.'

        donot_opdims = []
        for i in range(directions.shape[1]):
            if i not in opdims:
                donot_opdims.append(i)
        donot_opdims = tuple(donot_opdims)

        ret = []

        d_mat = []
        d_mat_idx = []
        close_list = {}
        for i, d in enumerate(directions):
            if any(d[list(opdims)]) and not any(d[list(donot_opdims)]):
                projection_dir = d[list(opdims)]
                projection_dir_tuple = tuple(projection_dir.tolist())

                if projection_dir_tuple not in close_list:
                    d_mat.append(projection_dir)
                    d_mat_idx.append(i)
                    close_list[projection_dir_tuple] = True

        for sf_row in sf_mat:
            sf_row_col = np.reshape(sf_row, (len(sf_row), 1))
            sf_row_dir = sf_row_col[d_mat_idx]
            ret.append(utils.ppl_helper.create_polytope(np.array(d_mat), sf_row_dir, len(opdims)))

        return ret

    @staticmethod
    def make_plot(dim, directions, sf_mat, model_name, poly_dir, simu_traj):
        for i in range(dim):
            for j in range(i, dim):
                if i == j:
                    continue
                opdims = (i, j)
                Plotter.config_plt(opdims)
                ppl_polys = Plotter.make_2Dproj_pplpoly(directions=directions, opdims=opdims, sf_mat=sf_mat)

                img_dir_path = os.path.join('../out/imgs', model_name)
                if not os.path.exists(img_dir_path):
                    os.mkdir(img_dir_path)
                img_path = os.path.join(img_dir_path, '{}-{}.png'.format(*opdims))
                plotman = Plotter(ppl_polys)

                # plot simulation
                for xs in simu_traj:
                    x, y = xs[:, opdims[0]], xs[:, opdims[1]]
                    Plotter.plot_points(x, y, xlabel=str(i), ylabel=str(j))

                # plot polygons
                poly_dir_path = os.path.join(poly_dir, model_name)
                if not os.path.exists(poly_dir_path):
                    os.mkdir(poly_dir_path)
                poly_file_path = os.path.join(poly_dir_path, '{}-{}'.format(*opdims))
                plotman.save_polygons_to_file(filename=poly_file_path)
                plotman.plot_polygons(poly_file_path, xlabel=str(i), ylabel=str(j))

                # plot scaling points
                if None:  # has issue with transparency,
                    if os.path.exists('../out/pivots.out'):
                        plotman.plot_pivots('../out/pivots.out', opdims, 'green')
                    if os.path.exists('../out/sca_cent.out'):
                        plotman.plot_pivots('../out/sca_cent.out', opdims, 'yellow')

                plotman.save_plt(opfile=img_path)

    @staticmethod
    def config_plt(opdims):
        plt.clf()

        fig = plt.figure(1, dpi=90)
        ax = fig.add_subplot(111)
        ax.set_xlabel('$x_{}$'.format(opdims[0]))
        ax.set_ylabel('$x_{}$'.format(opdims[1]))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to the input file storing vertex.')
    parser.add_argument('--opdims', type=int, nargs=2, help='2 dimensions to output.')
    parser.add_argument('--type', type=int, help='0 for points, 1 for polygons. If 2, polygon from first; points from '
                                                 'second.')
    # parser.
    # directions = SuppFuncUtils.generate_directions(direction_type, 2)
    args = parser.parse_args()

    ipfile = args.path
    opdims = args.opdims

    assert ipfile, 'No input file specified!'
    assert opdims, 'No output dimension specified!'

    data_type = args.type
    model_name = ipfile.split('/')[-1].split('.')[0]

    if data_type == 1:
        Plotter.plot_polygons(ipfile, xlabel=str(opdims[0]), ylabel=str(opdims[1]))
    elif data_type == 2:
        # Plotter.plot_polygons(filelist)
        Plotter.plot_points_from_file(['../out/simu.out'], opdims, xlabel=str(opdims[0]), ylabel=str(opdims[1]))
    elif data_type == 3:
        Plotter.plot_polygons(ipfile, xlabel=str(opdims[0]), ylabel=str(opdims[1]))
        Plotter.plot_points_from_file(['../out/simu.out'], opdims, xlabel=str(opdims[0]), ylabel=str(opdims[1]))
        print('Showing plot now.')
    else:
        # plot poly
        poly_path = os.path.join('../out/sfvals', model_name, '{}-{}'.format(*opdims))
        Plotter.plot_polygons(poly_path, xlabel=str(opdims[0]), ylabel=str(opdims[1]))

        # plot simulation
        simu_path = os.path.join('../out/simu_traj', '{}.simu'.format(model_name))
        Plotter.plot_points_from_file(simu_path, opdims, xlabel=str(opdims[0]), ylabel=str(opdims[1]))

        # plot scaling points
        # Plotter.plot_pivots('../out/pivots.out', opdims, 'green')
        # Plotter.plot_pivots('../out/sca_cent.out', opdims, 'yellow')

        print('Showing plot now.')
    plt.show()
