from LookupTables import instance_1


def main():
    init_hyperbox = instance_1.get_init()
    table = instance_1.get_table()

    print(str(init_hyperbox))
    print(table)


if __name__ == '__main__':
    main()
