from re import X
import sys
import codecs
import os
import numpy as np
from multiprocessing import Pool,freeze_support
from Bio import PDB
"""
CA x y z
"""

def get_CA_xyz(coords):
    if len(coords) < 2:
        return None
    coords = coords[1]
    coords = coords.split()
    if coords[2] == "CA":
        coords = coords[6:9]
        coords = list(map(float, coords))
        return coords
    return None

def get_pdbCA(input_file):
    pdb_name = input_file.split('/')[-1].split(".")[0]
    lines = []
    all_coords = []
    pre = None
    for line in open(input_file):
        if line[0:4] =="ATOM":
            columns = line.split()
            if pre is None:
                pre = columns[5]
                lines.append(line)
            elif pre == columns[5]:
                lines.append(line)
            else:
                cur_bone = get_CA_xyz(lines)
                if cur_bone:
                    all_coords.append(cur_bone)
                else:
                    return None, None
                lines = [line]
                pre = columns[5]
    if lines:
        cur_bone = get_CA_xyz(lines)
        if cur_bone:
            all_coords.append(cur_bone)
    return pdb_name, np.array(all_coords)
def get_pdb_dis(ca_coords):
    L = len(ca_coords)
    dis = []
    for i in range(L):
        x = ca_coords - ca_coords[i]
        x = abs(x)
        x = x * x
        x = np.sum(x, 1)
        x = x ** 0.5
        x = x.tolist()
        dis.append(x)
    return np.array(dis)

def get_pdbCA2(input_file):
    def calc_residue_dist(residue_one, residue_two) :
        """Returns the C-alpha distance between two residues"""
        diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
        return np.sqrt(np.sum(diff_vector * diff_vector))
    def calc_dist_matrix(chain_one, chain_two) :
        """Returns a matrix of C-alpha distances between two chains"""
        answer = np.zeros((len(chain_one), len(chain_two)), np.float)
        for row, residue_one in enumerate(chain_one) :
            for col, residue_two in enumerate(chain_two) :
                answer[row, col] = calc_residue_dist(residue_one, residue_two)
        return answer
    pdb_name = input_file.split('/')[-1].split(".")[0]
    parser = PDB.PDBParser()
    struct = parser.get_structure("tmp",input_file)
    model = struct[0]
    redidues = [residue for chain in model for residue in chain if "CA" in residue]
    return pdb_name, calc_dist_matrix(redidues, redidues)

def get_pdbCA3(input_file):
    pdb_name = input_file.split('/')[-1].split(".")[0]
    xyz_path = os.path.join(out_base_dir, pdb_name, pdb_name + ".xyz.npy")
    if not os.path.exists(xyz_path):
        return None, None
    coords = np.load(xyz_path)
    coordsCA = coords[:, 3:6]
    return pdb_name, coordsCA
def process(out_base_dir, pdb):
    pdb_name = pdb.split('/')[-1].split(".")[0]
    out_dir = os.path.join(out_base_dir, pdb_name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_path = os.path.join(out_dir, pdb_name + ".dis")
    # if os.path.exists(out_path + ".npy"):
    #     return
    pdb_name, coords = get_pdbCA3(pdb.strip())
    if pdb_name is None:
        return
    dis = get_pdb_dis(coords)
    np.save(out_path, dis)
if __name__ == '__main__':
    out_base_dir = sys.argv[1]
    p = Pool(int(sys.argv[2]))
    for line in open("train-pdb.list"):
        p.apply_async(process, (out_base_dir, line.strip()))
    p.close()
    p.join()
