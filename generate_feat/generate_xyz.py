import sys
import codecs
import os
import numpy as np
from multiprocessing import Pool,freeze_support
from Bio import PDB
"""
骨架 x y z
"""

def get_bone_xyz(coords):
    coords = coords[:3]
    if len(coords) != 3:
        return None
    coords = [line.split() for line in coords]
    if coords[0][2] == "N" and coords[1][2] == "CA" and coords[2][2] == "C":
        coords = [line[6:9] for line in coords]
        coords = [line if "-" not in line else line.split("-")[0] for line in coords]
        coords = np.array(coords).flatten().tolist()
        coords = list(map(float, coords))
        return coords
    return None

def pdb2coords(input_file):
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
                cur_bone = get_bone_xyz(lines)
                if cur_bone:
                    all_coords.append(cur_bone)
                else:
                    return None, None
                lines = [line]
                pre = columns[5]
    
    if lines:
        cur_bone = get_bone_xyz(lines)
        if cur_bone:
            all_coords.append(cur_bone)
    return pdb_name, np.array(all_coords)

def get_pdb_center(redidues):
    all_coords = []
    for residue in redidues:
        coords = [atom.get_coord() for atom in residue]
        coords = np.array(coords)
        all_coords.append(coords)
    all_coords = np.concatenate(all_coords, 0)
    return np.average(all_coords, 0)
def pdb2coords2(input_file):
    pdb_name = input_file.split('/')[-1].split(".")[0]
    parser = PDB.PDBParser()
    struct = parser.get_structure("tmp",input_file)
    model = struct[0]
    redidues = []

    ppb = PDB.PPBuilder()
    for pp in ppb.build_peptides(model):
        seq = str(pp.get_sequence())
        print(seq, len(seq))
    for chain in model:
        for residue in chain:
            if "CA" not in residue or "N" not in residue or "C" not in residue:
                continue
            redidues.append(residue)
        # break

    all_coords = []
    sel_idxs = []
    for residue in redidues:
        atom_name = ["N", "CA", "C"]
        coords = [residue[i].get_coord() for i in atom_name]
        coords = np.concatenate(coords, -1).reshape(-1, 3)
        coords = coords.flatten().tolist()
        sel_idxs.append(residue.id[1] - 1)
        all_coords.append(coords)
    return pdb_name, np.array(all_coords), sel_idxs
def process(out_base_dir, pdb):
    pdb_name, coords, sel_idxs = pdb2coords2(pdb.strip())

    if pdb_name is None:
        return
    out_dir = os.path.join(out_base_dir, pdb_name)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_path = os.path.join(out_dir, pdb_name + ".xyz")
    fasta = codecs.open(os.path.join(out_base_dir, pdb_name, pdb_name + ".fasta")).readlines()[1].strip()
    all_coords = np.full((len(fasta), 9), np.nan)
    try:
        all_coords[sel_idxs] = coords
    except Exception as e:
        print(e)
    all_coords = all_coords.reshape(-1, 3)
    np.save(out_path, all_coords)
if __name__ == '__main__':
    out_base_dir = sys.argv[1]
    p = Pool(int(sys.argv[2]))
    for line in open("train-pdb.list"):
        p.apply_async(process, (out_base_dir, line.strip()))
    p.close()
    p.join()

