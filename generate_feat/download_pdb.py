from Bio.PDB import PDBList
pdbl = PDBList()
pdbl.retrieve_pdb_file('7CWP', file_format='pdb', pdir="pdb")