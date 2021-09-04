import codecs
import pickle
idx2pdbs = None
def get_data_bak():
    global idx2pdbs
    if idx2pdbs is not None:
        return idx2pdbs
    # path = "/mnt/e/study/RoseTTAFold-Train/data/anno/uniclust_2016_09/uniclust_2016_09_annotation_pdb.tsv"
    # idx2pdbs = {}
    # for line in codecs.open(path, "r", "utf-8"):
    #     line = line.split()[:2]
    #     idx2pdbs[line[0]] = line[1]
    path = "/mnt/e/study/RoseTTAFold-Train/data/anno/uniclust_2016_09/dbidx.pickle"
    f = open(path, "rb")
    idx2pdbs = pickle.load(f)
    f.close()
    print("read get_data over", idx2pdbs)
    return idx2pdbs
def get_data():
    global idx2pdbs
    if idx2pdbs is not None:
        return idx2pdbs
    path = "/mnt/e/study/RoseTTAFold-Train/data/anno/uniclust_2016_09/dbidx.pickle"
    f = open(path, "rb")
    idx2pdbs = pickle.load(f)
    f.close()
    # path = "/mnt/e/study/RoseTTAFold-Train/data/anno/uniclust_2016_09/dbidx.txt"
    # data = open(path).readlines()
    # data = [i.strip().split() for i in data]
    # idx2pdbs = {}
    # for k,v in data:
    #     idx2pdbs[k] = v
    # print("read get_data over")
    return idx2pdbs
def get_pdb(entry):
    d = get_data()
    if entry in d:
        return d[entry]
    return None
def clear():
    global idx2pdbs
    if idx2pdbs:
        del(idx2pdbs)
