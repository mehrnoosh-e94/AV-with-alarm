import pickle


def write_list(a_list, filename):
    with open(str(filename), 'wb') as fp:
        pickle.dump(a_list, fp)
#         print(f'Done writing {filename} into a binary file')

def read_list(filename):
    with open(filename, 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list