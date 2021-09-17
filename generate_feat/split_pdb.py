import sys
import codecs
import os
from Bio.PDB import PDBParser, PDBIO
from multiprocessing import Pool

def process(pdb_f, out_dir):
    try:
        name = pdb_f.split('/')[-1].split(".")[0]
        pdb = PDBParser().get_structure(name, pdb_f)
        io = PDBIO()
        for chain in pdb.get_chains():
            io.set_structure(chain)
            out_path = os.path.join(out_dir, pdb.get_id() + "_" + chain.get_id() + ".pdb")
            io.save(out_path)
    except Exception as e:
        pass

if __name__ == '__main__':
    out_dir = sys.argv[2]
    p = Pool(int(sys.argv[3]))
    for line in open(sys.argv[1]):
        line = line.strip()
        p.apply_async(process, (line.strip(), out_dir))
    p.close()
    p.join()
