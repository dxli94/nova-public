class TrackedVar:
    def __init__(self, val):
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





