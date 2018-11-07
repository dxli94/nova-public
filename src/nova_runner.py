import examples.nova_examples.brusselator as brusselator
import examples.nova_examples.buckling_column as buckling_column
import examples.nova_examples.lorentz_system as lorentz_system
import examples.nova_examples.vanderpol_2d as vanderpol2d
import examples.nova_examples.vanderpol_4d as vanderpol4d
import examples.nova_examples.vanderpol_6d as vanderpol6d
import examples.nova_examples.vanderpol_8d as vanderpol8d
import examples.nova_examples.biology_1 as bio1
import examples.nova_examples.biology_2 as bio2
import examples.nova_examples.coupled_oscillators_5d as coupled_osc_5d
import examples.nova_examples.coupled_oscillators_10d as coupled_osc_10d
import examples.nova_examples.coupled_oscillators_15d as coupled_osc_15d

import numpy as np
np.warnings.filterwarnings('ignore')


# ======== Don't change anything above ========

example = brusselator
# example = buckling_column
# example = predator_prey
# example = bio1
# example = bio2
# example = vanderpol2d
# example = vanderpol4d
# example = vanderpol6d
# example = coupled_osc_5d
# example = coupled_osc_10d
# example = coupled_osc_15d
# example = lorentz_system

settings = example.define_settings()
example.run_nova(settings)