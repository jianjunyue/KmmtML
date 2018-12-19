

class Util:
    def __init__(self):
        return

    def rename_columns(pre_name, columns_name):
        name_dict = {}
        for name in columns_name:
            name_dict[name] = pre_name +'_'+ name
        return name_dict