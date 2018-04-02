import LookupTables.instance_1


def select_instance(instance_no):
    if instance_no == 1:
        return LookupTables.instance_1.get_table()


if __name__ == '__main__':
    print(select_instance(1))
