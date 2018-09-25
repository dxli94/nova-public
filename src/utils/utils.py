import numpy as np


class Freezable:
    """a class where you can freeze the fields (prevent new fields from being created)

    Source: http://stanleybak.com/hylaa/
    Author: Stanley Bak
    """

    _frozen = False

    def freeze_attrs(self):
        """
        prevents any new attributes from being created in the object
        """
        self._frozen = True

    def __setattr__(self, key, value):
        if self._frozen and not hasattr(self, key):
            raise TypeError("{} does not contain attribute '{}' (object was frozen)".format(self, key))

        object.__setattr__(self, key, value)


class TrackedVar:
    """a class where you can access the previous value of the object
    """
    def __init__(self, val=None):
        self.prev = None
        self.curr = val

    def reset(self, val):
        self.prev = None
        self.curr = val

    def set_val(self, val):
        self.prev = self.curr
        self.curr = val

    def get_val(self):
        return self.curr

    def rollback(self):
        assert self.prev is not None, "No previous value found."

        self.curr = self.prev
        self.prev = None


def get_canno_dir_indices(directions):
    ub_indices = []
    lb_indices = []

    for idx, d in enumerate(directions):
        if np.isclose(np.sum(d), 1):  # due to the way of generating directions, exact equality is not proper.
            ub_indices.append(idx)
        elif np.isclose(np.sum(d), -1):
            lb_indices.append(idx)
        else:
            continue

    return lb_indices, ub_indices


def extract_bounds_from_sf(sf_vec, canno_dir_indices):
    lb_indices = canno_dir_indices[0]
    ub_indices = canno_dir_indices[1]

    lb = -sf_vec[lb_indices]
    ub = sf_vec[ub_indices]

    return lb, ub


if __name__ == '__main__':
    # test TrackedVar()
    tv = TrackedVar(1)
    assert tv.get_val() == 1
    tv.set_val(2)
    assert tv.get_val() == 2
    tv.rollback()
    assert tv.get_val() == 1
    tv.set_val(3)
    assert tv.get_val() == 3
    tv.rollback()
    assert tv.get_val() == 1