'''
Stanley Bak
June 2017

Mass-Spring Reachability
'''

import random
import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib import colors
from matplotlib.widgets import Button, Slider, CheckButtons

from scipy.linalg import expm
from scipy.spatial import ConvexHull

import numpy as np

class PlotStatus(object):
    'Container for the plot status'

    def __init__(self):
        self.num_mass_axis = None

        self.num_masses = None
        self.a_matrix = None
        self.time = None
        self.init_set = None # list of pairs
        self.sim_state = None

        self.paused = True

        self.do_vpoly = False
        self.do_zono = False
        self.do_supp = False

        self.vpoly_points_2d = None
        self.vpoly_chull = None
        self.zono_generators = None
        self.zono_optimize_direction = None
        self.supp_corners = None

        self.set_num_masses(1)
        self.update_time_string = ""

    def restart(self):
        'restart the animation'

        num = self.num_masses
        self.num_masses = None

        self.set_num_masses(num)

    def set_num_masses(self, num):
        'set the number of masses'

        if num != self.num_masses:
            self.num_masses = num
            self.paused = True
            self.time = 0.0

            self.vpoly_points_2d = None
            self.vpoly_chull = None
            self.zono_generators = None
            self.zono_optimize_direction = None
            self.supp_corners = None

            self.sim_start = np.array([0] * 2 * num, dtype=float)
            self.sim_start[1] = 0.8 # initial velocity

            self.init_set = [[-0.2, 0.2]] * 2 * num
            self.init_set[1] = [0.6, 1.0]

            self.sim_state = self.sim_start.copy()

            self.a_matrix = make_a_matrix(2 * num)

            if self.num_mass_axis is not None:
                self.num_mass_axis.axis([-1, num, -0.75, 0.75])

            self.update_time_string = ""
            self.update()

    def set_num_mass_axis(self, axis):
        'set the matplotlib axis that is updated whenever num_masses changes'

        self.num_mass_axis = axis
        plt.draw() # redraw everything

    def update_supp(self):
        'update support function data'

        l_list = []

        dims = self.num_masses * 2

        # list of directions in the transformed space
        l_list.append([1 if d == 0 else 0 for d in xrange(dims)])
        l_list.append([1 if d == 1 else 0 for d in xrange(dims)])
        l_list.append([-1 if d == 0 else 0 for d in xrange(dims)])
        l_list.append([-1 if d == 1 else 0 for d in xrange(dims)])

        supp_vecs = []

        a_exp_t = expm(self.a_matrix.T * self.time)

        for l in l_list:

            # direction in the original space
            direction = np.dot(a_exp_t, l)

            # maximize the direction in initial set
            init_pt = []

            for d in xrange(dims):
                init_pt.append(self.init_set[d][0] if direction[d] < 0 else self.init_set[d][1])

            # take dot product with direction to get the support function value
            val = np.dot(direction, init_pt)

            supp_vec = val * np.array(l)
            supp_vecs.append(supp_vec)

        self.supp_corners = []

        # outer hull approximation
        self.supp_corners.append([supp_vecs[0][0], supp_vecs[1][1]])
        self.supp_corners.append([supp_vecs[2][0], supp_vecs[1][1]])
        self.supp_corners.append([supp_vecs[2][0], supp_vecs[3][1]])
        self.supp_corners.append([supp_vecs[0][0], supp_vecs[3][1]])

        #print "supp_corners = {}".format(self.supp_corners)

    def update(self):
        'update the states to the current step'

        start = time.time()

        a_exp = expm(self.a_matrix * self.time)

        self.update_sim(a_exp)

        if self.do_vpoly:
            if self.vpoly_points_2d is None:
                self.update_vpoly(a_exp)
        else:
            self.vpoly_points_2d = None
            self.vpoly_chull = None

        if self.do_zono:
            if self.zono_generators is None:
                self.update_zono(a_exp)
        else:
            self.zono_generators = None
            self.zono_optimize_direction = None

        if self.do_supp:
            if self.supp_corners is None:
                self.update_supp()
        else:
            self.supp_corners = None

        dif_ms = 1000 * (time.time() - start)

        self.update_time_string = "Update Time: {}ms".format(int(dif_ms))

    def step(self, cur_time=None):
        'advance the states by one time step if not paused'

        if not self.paused or cur_time is not None:
            if cur_time is None:
                delta_t = 0.1
                self.time += delta_t
            else:
                self.time = cur_time

            self.vpoly_points_2d = None
            self.vpoly_chull = None
            self.zono_generators = None
            self.supp_corners = None

            self.update()

    def update_zono(self, a_exp):
        'update data structures for zono representation'

        self.zono_generators = []

        for i in xrange(2 * self.num_masses):
            # each generator is 0.2 wide in a particular direction
            vec = np.array([0.0] * 2 * self.num_masses)
            vec[i] = 0.2

            self.zono_generators.append(np.dot(a_exp, vec))

    def update_vpoly(self, a_exp):
        'update data structures for vpoly reachability at current time'

        self.vpoly_points_2d = []
        num_dims = 2 * self.num_masses

        for index in xrange(2**num_dims):
            init_point = []

            for d in xrange(num_dims):
                divider = 2**(d)

                use_min = index / divider % 2 == 0

                init_point.append(self.init_set[d][0 if use_min else 1])

            # init_point is now constructed
            pt = np.dot(a_exp, np.array(init_point, dtype=float))

            self.vpoly_points_2d.append(np.array([pt[0], pt[1]], dtype=float))

        # also do the convex hull for plotting (generally fast since it's 2d)
        self.vpoly_chull = []
        hull = ConvexHull(self.vpoly_points_2d)

        for i in hull.vertices:
            self.vpoly_chull.append(self.vpoly_points_2d[i])


        # wrap back around to start
        self.vpoly_chull.append(self.vpoly_points_2d[hull.vertices[0]])

    def update_sim(self, a_exp):
        'update the data structures at the current time for the simulation'

        self.sim_state = np.dot(a_exp, self.sim_start)

def update_spring(pts, startx, endx):
    'update the list of points for a single spring in the animation'

    xs = []
    ys = []

    num_tips = 6
    difx = endx - startx

    stepx = difx / (num_tips + 1)
    is_top = True

    xs.append(startx)
    ys.append(0)

    for tip in xrange(num_tips):
        xpos = startx + (tip + 1) * stepx
        ypos = 0.3 if is_top else -0.3
        is_top = not is_top

        xs.append(xpos)
        ys.append(ypos)

    xs.append(endx)
    ys.append(0)

    pts.set_data(xs, ys)

def make_a_matrix(num_dims):
    '''get the A matrix corresponding to the dynamics'''

    a = np.zeros((num_dims, num_dims))

    for d in xrange(num_dims / 2):
        a[2*d][2*d+1] = 1 # pos' = vel

        a[2*d+1][2*d] = -2 # cur-state

        if d > 0:
            a[2*d+1][2*d-2] = 1 # prev-state

        if d < num_dims / 2 - 1:
            a[2*d+1][2*d+2] = 1 # next state

    return a

def get_colors():
    'get a list of colors'

    all_colors = []

    # remove any colors with 'white' or 'yellow in the name
    skip_colors_substrings = ['white', 'yellow']
    skip_colors_exact = ['black']

    for col in colors.cnames:
        skip = False

        for col_substring in skip_colors_substrings:
            if col_substring in col:
                skip = True
                break

        if not skip and not col in skip_colors_exact:
            all_colors.append(col)

    # we'll re-add these later; remove them before shuffling
    first_colors = ['red', 'blue', 'green', 'orange', 'cyan', 'magenta', 'lime']

    for col in first_colors:
        all_colors.remove(col)

    # deterministic shuffle of all remaining colors
    random.seed(0)
    random.shuffle(all_colors)

    # prepend first_colors so they get used first
    all_colors = first_colors + all_colors

    return all_colors

def main():
    'main entry point'

    # make the figure before creating PlotState (otherwise draw() creates a new figure)
    fig = plt.figure(figsize=(16, 10))

    status = PlotStatus()
    max_num_masses = 50

    labelsize = 22
    all_colors = get_colors()

    ax_main = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax_main.grid()
    ax_main.axis([-1.1, 1.1, -1.1, 1.1])
    ax_main.set_xlabel('$x_0$', fontsize=labelsize)
    ax_main.set_ylabel('$v_0$', fontsize=labelsize)
    plt.tick_params(axis='both', which='major', labelsize=labelsize)

    annotation = ax_main.annotate("", xy=(0.05, 0.93), xycoords="axes fraction", fontsize=labelsize)
    annotation.set_animated(True)

    # vpoly points and hull
    vpoly_chull, = ax_main.plot([], [], '--', color='black', lw=3)
    vpoly_points, = ax_main.plot([], [], 'o', color=all_colors[0], lw=2)

    # cur_point
    cur_point, = ax_main.plot([], [], '*', color='black', ms=30)

    # supp
    supp_box, = ax_main.plot([], [], '--', color=all_colors[0], lw=2)

    # zonotope
    zonotope_gen_lines = []
    zonotope_curmax_lines = []

    for i in xrange(2 * max_num_masses):
        gen_line, = ax_main.plot([], [], '--', color=all_colors[i % len(all_colors)], lw=3)
        curmax_line, = ax_main.plot([], [], '-', color=all_colors[i % len(all_colors)], lw=5)

        zonotope_gen_lines.append(gen_line)
        zonotope_curmax_lines.append(curmax_line)

    #####
    ax_im = plt.subplot2grid((3, 1), (2, 0))
    ax_im.grid()
    ax_im.axis([-1, status.num_masses, -0.75, 0.75])

    status.set_num_mass_axis(ax_im)
    plt.setp(ax_im.get_yticklabels(), visible=False)
    ax_im.set_xlabel('Position', fontsize=labelsize)
    plt.tick_params(axis='both', which='major', labelsize=labelsize)

    patch_list = []
    mass_width = 0.3

    # masses
    for m in xrange(max_num_masses):
        patch = patches.Rectangle((0, 0), mass_width, 1.0, fc=all_colors[m % len(all_colors)], ec='black')
        patch_list.append(patch)
        ax_im.add_patch(patch)

    # springs
    spring_list = []

    for _ in xrange(max_num_masses + 1):
        pts, = ax_im.plot([], [], '-', color='black')

        spring_list.append(pts)

    def init():
        '''initialize animation'''

        cur_point.set_data([], [])
        annotation.set_text("")

        vpoly_points.set_data([], [])
        vpoly_chull.set_data([], [])

        supp_box.set_data([], [])

        for patch in patch_list:
            patch.set_visible(False)

        for spring in spring_list:
            spring.set_data([], [])

        for lines in zonotope_gen_lines:
            lines.set_data([], [])

        for lines in zonotope_curmax_lines:
            lines.set_data([], [])

        return patch_list + spring_list + [vpoly_points, vpoly_chull, cur_point, annotation] + \
                zonotope_gen_lines + zonotope_curmax_lines + [supp_box]

    def animate(_):
        """perform animation step"""

        status.step()

        annotation.set_text(status.update_time_string)

        xs = []
        ys = []

        xs.append(status.sim_state[0])
        ys.append(status.sim_state[1])

        cur_point.set_data(xs, ys)

        #### vpoly plot
        if status.vpoly_points_2d is not None:
            xs = []
            ys = []

            for pt in status.vpoly_points_2d:
                xs.append(pt[0])
                ys.append(pt[1])

            vpoly_points.set_data(xs, ys)

            xs = []
            ys = []

            for pt in status.vpoly_chull:
                xs.append(pt[0])
                ys.append(pt[1])

            vpoly_chull.set_data(xs, ys)
        else:
            vpoly_points.set_data([], [])
            vpoly_chull.set_data([], [])

        ##### set patches
        prev_xpos = -1

        for m in xrange(max_num_masses):
            patch = patch_list[m]

            if m >= status.num_masses:
                patch.set_visible(False)
            else:
                patch.set_visible(True)

                xpos = m + status.sim_state[2*m]
                patch.set_xy([xpos - mass_width/2.0, -0.5])

                update_spring(spring_list[m], prev_xpos, xpos - mass_width/2.0)

                prev_xpos = xpos + mass_width / 2.0

        # update final spring
        update_spring(spring_list[status.num_masses], prev_xpos, status.num_masses)

        # update zonotopes
        optimize_prev_point = (status.sim_state[0], status.sim_state[1])

        for dim in xrange(2 * max_num_masses):
            if status.zono_generators is None or dim >= 2 * status.num_masses:
                zonotope_gen_lines[dim].set_data([], [])
                zonotope_curmax_lines[dim].set_data([], [])
            else:
                gen = status.zono_generators[dim]

                xs = []
                ys = []

                xs.append(status.sim_state[0] - gen[0])
                ys.append(status.sim_state[1] - gen[1])

                xs.append(status.sim_state[0] + gen[0])
                ys.append(status.sim_state[1] + gen[1])

                zonotope_gen_lines[dim].set_data(xs, ys)

                # logic for curmax
                if status.zono_optimize_direction is None:
                    zonotope_curmax_lines[dim].set_data([], [])
                else:
                    xs = [optimize_prev_point[0]]
                    ys = [optimize_prev_point[1]]

                    pt1 = [optimize_prev_point[0] - gen[0], optimize_prev_point[1] - gen[1]]
                    pt2 = [optimize_prev_point[0] + gen[0], optimize_prev_point[1] + gen[1]]

                    val1 = np.dot(status.zono_optimize_direction, pt1)
                    val2 = np.dot(status.zono_optimize_direction, pt2)

                    nextpt = pt1 if val1 > val2 else pt2

                    optimize_prev_point = nextpt
                    xs.append(nextpt[0])
                    ys.append(nextpt[1])
                    zonotope_curmax_lines[dim].set_data(xs, ys)

        # supp
        if status.supp_corners is None:
            supp_box.set_data([], [])
        else:
            xs = [status.supp_corners[-1][0]]
            ys = [status.supp_corners[-1][1]]

            for pt in status.supp_corners:
                xs.append(pt[0])
                ys.append(pt[1])

            supp_box.set_data(xs, ys)

        return patch_list + spring_list + [vpoly_points, vpoly_chull, cur_point, annotation] + \
                zonotope_gen_lines + zonotope_curmax_lines + [supp_box]

    def start_stop_pressed(_):
        'button event function'
        status.paused = not status.paused

    def restart_pressed(_):
        'button event function'

        status.restart()

    def update_masses(val):
        'slider moved event function'
        status.set_num_masses(int(round(val)))

    def canvas_clicked(event):
        'canvas clicked event'

        if event.inaxes is ax_main and event.xdata is not None:
            dx = event.xdata - status.sim_state[0]
            dy = event.ydata - status.sim_state[1]

            if abs(dx) < 0.1 and abs(dy) < 0.1:
                status.zono_optimize_direction = None
            else:
                status.zono_optimize_direction = (dx, dy)

    # retain reference to keep event alive
    cid = fig.canvas.mpl_connect('button_press_event', canvas_clicked)

    # shrink plot, add buttons
    plt.tight_layout()

    plt.subplots_adjust(bottom=0.15)

    axstart = plt.axes([0.8, 0.02, 0.15, 0.05])
    start_button = Button(axstart, 'Start / Stop', color='0.85', hovercolor='0.85')
    start_button.on_clicked(start_stop_pressed)
    start_button.label.set_fontsize(labelsize)

    axrestart = plt.axes([0.05, 0.02, 0.15, 0.05])
    restart_button = Button(axrestart, 'Restart', color='0.85', hovercolor='0.85')
    restart_button.on_clicked(restart_pressed)
    restart_button.label.set_fontsize(labelsize)

    axmasses = plt.axes([0.6, 0.02, 0.15, 0.05])
    smasses = Slider(axmasses, 'Num Masses', 1, max_num_masses, valinit=1, valfmt='%0.0f')
    smasses.on_changed(update_masses)

    axcheckboxes = plt.axes([0.25, 0.01, 0.15, 0.1])
    checkboxes = CheckButtons(axcheckboxes, ('Vertex-Polytope', 'Zonotope', 'Support Function'), (False, False, False))

    def update_checkboxes(_):
        'a checkbox was clicked'

        status.do_vpoly = checkboxes.lines[0][0].get_visible()
        status.do_zono = checkboxes.lines[1][0].get_visible()
        status.do_supp = checkboxes.lines[2][0].get_visible()

        status.update()

    checkboxes.on_clicked(update_checkboxes)

    # retain reference to keep animation alive
    anim = animation.FuncAnimation(fig, animate, interval=1, blit=True, init_func=init)

    #status.do_supp = True
    #status.step(2.22)

    plt.show()
    #anim.save('heat.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    # print "Done"

if __name__ == "__main__":
    main()
