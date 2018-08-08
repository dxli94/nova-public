import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Plotter:
    images = []

    def __init__(self, images, opvars):
        self.images = images
        self.opvars = opvars
        self.vertices_sorted = list(map(lambda im: self.sort_vertices(im), self.images))

    def sort_vertices(self, im):
        # corners = [(v[self.opvars[0]], v[self.opvars[1]]) for v in im.vertices]
        corners = im.vertices

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

    def save_polygons_to_file(self, dirpath='../out/', filename='outfile.out'):
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        with open(os.path.join(dirpath, filename), 'w') as opfile:
            for vertices in self.vertices_sorted:
                x, y = [elem[0] for elem in vertices], [elem[1] for elem in vertices]
                x.append(x[0])
                y.append(y[0])

                for xx, yy in zip(x, y):
                    opfile.write('%.15f %.15f' % (xx, yy) + '\n')
                opfile.write('\n')

    @staticmethod
    def plot_polygons(filelist):
        print('Start reading file...')
        colors = ['red', 'blue']
        linewidths = [1, 1]
        linestyles = ['solid', 'dashed']

        fig = plt.figure(1, dpi=90)
        ax = fig.add_subplot(111)
        ax.set_xlabel('$x_{1}$')
        ax.set_ylabel('$x_{2}$')

        for ipfile_path, color, lw, ls in zip(filelist, colors[:len(filelist)], linewidths[:len(filelist)], linestyles[:len(filelist)]):
            print(ipfile_path)
            try:
                with open(ipfile_path) as ipfile:
                    content = ipfile.read().strip('\n')
                    polygons = content.split('\n\n')
                    vertices_sorted = list(map(lambda poly: poly.split('\n'), polygons))
            except FileExistsError:
                print('File does not exist %s' % ipfile_path)
            print('Finished. \nStart plotting...')

            i = 0
            for vertices in vertices_sorted:
                if i % 50 == 0:
                    x, y = [float(elem.split()[0]) for elem in vertices], [float(elem.split()[1]) for elem in vertices]
                    mat = np.transpose(np.array([x, y]))
                    poly1patch = patches.Polygon(mat, fill=False, edgecolor=color, linewidth=lw, linestyle=ls)
                    ax.add_patch(poly1patch)
                i+=1

        plt.autoscale(enable=True)
        return plt

    @staticmethod
    def plot_points(filelist):
        print('Start reading file...')
        colors = ['red', 'blue']
        linewidths = [1, 1]
        linestyles = ['solid', 'dashed']

        fig = plt.figure(1, dpi=90)
        ax = fig.add_subplot(111)
        ax.set_xlabel('$x_{1}$')
        ax.set_ylabel('$x_{2}$')

        for ipfile_path, color, lw, ls in zip(filelist, colors[:len(filelist)], linewidths[:len(filelist)], linestyles[:len(filelist)]):
            print(ipfile_path)
            try:
                with open(ipfile_path) as ipfile:
                    content = ipfile.read().strip('\n')
                    points = content.split('\n')
                    points = list(map(lambda p: p.split('\n')[0], points))
                    x = []
                    y = []

                    for p in points:
                        xy = p.split()
                        x.append(float(xy[0]))
                        y.append(float(xy[1]))
            except FileExistsError:
                print('File does not exist %s' % ipfile_path)
            print('Finished. \nStart plotting...')

        # print([x, y])
            # for vertices in vertices_sorted:
            #     x, y = [float(elem.split()[0]) for elem in vertices], [float(elem.split()[1]) for elem in vertices]
            #     mat = np.transpose(np.array([x, y]))
            #     poly1patch = patches.Polygon(mat, fill=False, edgecolor=color, linewidth=lw, linestyle=ls)
            #     ax.add_patch(poly1patch)
        plt.plot(x, y)
        plt.autoscale(enable=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, nargs='*', help='path to the input file storing vertex.')
    parser.add_argument('--type', type=int, help='0 for points, 1 for polygons. If 2, polygon from first; points from '
                                                 'second.')
    args = parser.parse_args()
    filelist = args.path if args.path else ['../out/outfile.out']
    data_type = args.type
    if data_type == 1:
        Plotter.plot_points(filelist)
    elif data_type == 2:
        # Plotter.plot_polygons(filelist)
        Plotter.plot_points(['../out/simu.out'])
    else:
        Plotter.plot_polygons(filelist)
        Plotter.plot_points(['../out/simu.out'])
        print('Showing plot now.')
        plt.show()
