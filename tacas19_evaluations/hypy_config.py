def config_hypy():
    rv = dict()

    vanderpol2d_config = {
        'input_model_file': 'vanderpol_2d.xml',
        'order': [5, 6, 7],
        'unsafe_condition': ['x1 >= 2.8', 'x1 >= 3'],
        'min_step_size': 0.01
    }
    rv['vanderpol_2d'] = vanderpol2d_config

    vanderpol4d_config = {
        'input_model_file': 'vanderpol_4d.xml',
        'order': [5, 6, 7, 8],
        'unsafe_condition': ['x1 >= 2.84', 'x1 >= 3'],
        'min_step_size': 0.005
    }
    rv['vanderpol4d'] = vanderpol4d_config

    vanderpol6d_config = {
        'input_model_file': 'vanderpol_6d.xml',
        'order': [5, 6, 7, 8],
        'unsafe_condition': ['x1 >= 2.84', 'x1 >= 3'],
        'min_step_size': 0.005
    }
    rv['vanderpol6d'] = vanderpol6d_config

    vanderpol8d_config = {
        'input_model_file': 'vanderpol_8d.xml',
        'order': [5, 6, 7, 8],
        'unsafe_condition': ['x1 >= 3'],
        'min_step_size': 0.005
    }
    rv['vanderpol8d'] = vanderpol8d_config

    brusselator_config = {
        'input_model_file': 'brusselator.xml',
        'order': [4, 5, 6, 7],
        'unsafe_condition': ['x1 >= 1.91'],
        'min_step_size': 0.01
    }
    rv['brusselator'] = brusselator_config

    biology_1_config = {
        'input_model_file': 'biology_1.xml',
        'order': [7, 8, 9, 10],
        'unsafe_condition': ['x3 >= 1.9'],
        'min_step_size': 0.001
    }
    rv['biology_1'] = biology_1_config

    biology_2_config = {
        'input_model_file': 'biology_2.xml',
        'order': [7, 8, 9, 10],
        'unsafe_condition': ['x3 >= 0.5'],
        'min_step_size': 0.001
    }
    rv['biology_2'] = biology_2_config

    coupled_oscillators_5d = {
        'input_model_file': 'coupled_oscillators_5d.xml',
        'order': [4, 5, 6],
        'unsafe_condition': ['x1 <= 0.08'],
        'min_step_size': 0.001
    }
    rv['coupled_oscillators_5d'] = coupled_oscillators_5d

    coupled_oscillators_10d = {
        'input_model_file': 'coupled_oscillators_10d.xml',
        'order': [5, 6, 7],
        'unsafe_condition': ['x1 <= 0.08'],
        'min_step_size': 0.001
    }
    rv['coupled_oscillators_10d'] = coupled_oscillators_10d

    coupled_oscillators_15d = {
        'input_model_file': 'coupled_oscillators_15d.xml',
        'order': [5, 6, 7],
        'unsafe_condition': ['x1 <= 0.08'],
        'min_step_size': 0.001
    }
    rv['coupled_oscillators_15d'] = coupled_oscillators_15d

    return rv