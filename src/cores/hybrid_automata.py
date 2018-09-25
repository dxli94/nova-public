from cores.sys_dynamics import GeneralDynamics
from utils.utils import Freezable


class NonlinAutomatonMode(Freezable):
    def __init__(self, parent, name):
        self.parent = parent
        self.name = name

        self.dynamics = None
        self.id_to_vars = None
        self.is_linear = None

        self.freeze_attrs()

    def set_dynamics(self, dynamics_str, is_linear):
        self.id_to_vars = {}
        for i, var in enumerate(self.parent.variables):
            self.id_to_vars[i] = var
        self.dynamics = GeneralDynamics(self.id_to_vars, *dynamics_str)
        self.is_linear = is_linear


class NonlinHybridAutomaton(object):
    """The hybrid automaton with nonlinear dyanmics"""

    def __init__(self, name='HybridAutomaton'):
        self.name = name
        self.modes = {}
        self.variables = []  # list of strings

    def new_mode(self, name):
        """add a mode"""

        assert len(self.variables) > 0, "0 variable is given."

        m = NonlinAutomatonMode(self, name)
        self.modes[m.name] = m
        return m
