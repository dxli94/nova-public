from utils.tracked_vars import TrackedVar as tvar


class PostOptStateholder:
    def __init__(self, init_set_lb, init_set_ub):
        self.tvars_list = []
        self.init_set_lb = tvar()
        self.init_set_ub = tvar()

        self.tube_lb = tvar()
        self.tube_ub = tvar()
        self.tvars_list.append(self.tube_lb)
        self.tvars_list.append(self.tube_ub)
        # tube_lb, tube_ub = init_set_lb, init_set_ub

        # temporary variables for reachable states in dense time
        self.temp_tube_lb = tvar()
        self.temp_tube_ub = tvar()
        self.tvars_list.append(self.temp_tube_lb)
        self.tvars_list.append(self.temp_tube_ub)

        # temp_tube_lb, temp_tube_ub = init_set_lb, init_set_ub
        # initial reachable set in discrete time in the current abstract domain
        # changes when the abstract domain is large enough to contain next image in alfa step
        self.current_init_set_lb = tvar()
        self.current_init_set_ub = tvar()
        self.tvars_list.append(self.current_init_set_lb)
        self.tvars_list.append(self.current_init_set_ub)

        # current_init_set_lb, current_init_set_ub = init_set_lb, init_set_ub

        self.input_lb_seq = tvar()
        self.input_ub_seq = tvar()
        self.tvars_list.append(self.input_lb_seq)
        self.tvars_list.append(self.input_ub_seq)
        # input_lb_seq, input_ub_seq = init_set_lb, init_set_ub

        self.phi_list = tvar([])
        self.tvars_list.append(self.phi_list)

        self.__reset(init_set_lb, init_set_ub)

    def rollback(self):
        for tv in self.tvars_list:
            tv.rollback()

    def templify(self, init_set_lb, init_set_ub):
        self.__reset(init_set_lb, init_set_ub)

    def __reset(self, init_set_lb, init_set_ub):
        self.init_set_lb = init_set_lb
        self.init_set_ub = init_set_ub

        self.tube_lb.reset(init_set_lb)
        self.tube_ub.reset(init_set_ub)

        self.temp_tube_lb.reset(init_set_lb)
        self.temp_tube_ub.reset(init_set_ub)

        self.current_init_set_lb.reset(init_set_lb)
        self.current_init_set_ub.reset(init_set_ub)

        self.input_lb_seq.reset(init_set_lb)
        self.input_ub_seq.reset(init_set_ub)

        self.phi_list.reset([])

if __name__ == '__main__':
    init_set_lb = [0, 1]
    init_set_ub = [1, 2]

    posh = PostOptStateholder(init_set_lb, init_set_ub)
    print(posh.init_set_lb)
    posh.templify(init_set_lb, init_set_ub)
    print(posh.init_set_ub)