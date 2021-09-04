import sys
import codecs
import os
from Bio import PDB
from multiprocessing import Pool

letters = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H',
           'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
           'TYR': 'Y', 'VAL': 'V'}

def pdb2fasta(input_file):
    pdb_name = input_file.split('/')[-1].split(".")[0]
    name = '>' + pdb_name
    lines = []
    lines.append(name)
    acids = []
    for line in open(input_file):
        if line[0:6] =="SEQRES":
            columns = line.split()
            for resname in columns[4:]:
                if resname in letters:
                    acids.append(letters[resname])
                else:
                    return None, None
    lines.append("".join(acids))
    return pdb_name, "\n".join(lines)
def pdb2fasta2(input_file):
    pdb_name = input_file.split('/')[-1].split(".")[0]
    name = '>' + pdb_name
    lines = []
    lines.append(name)
    parser = PDB.PDBParser()
    structure = parser.get_structure(pdb_name, input_file)
    ppb = PDB.PPBuilder()
    model = structure[0]
    seq = "".join([str(pp.get_sequence()) for pp in ppb.build_peptides(model)])
    lines.append(seq)
    return pdb_name, "\n".join(lines)
def process(out_base_dir, pdb):
    pdb_name, fasta = pdb2fasta2(pdb.strip())
    if pdb_name is None:
        return
    out_dir = os.path.join(out_base_dir, pdb_name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_path = os.path.join(out_dir, pdb_name + ".fasta")
    with codecs.open(out_path, "w", "utf-8") as f:
        f.write(fasta)
if __name__ == '__main__':
    out_base_dir = sys.argv[1]
    p = Pool(int(sys.argv[2]))
    for line in open("train-pdb.list"):
        p.apply_async(process, (out_base_dir, line.strip()))
    p.close()
    p.join()

        
