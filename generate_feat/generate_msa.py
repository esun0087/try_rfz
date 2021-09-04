import sys
import os
import codecs

def check_fasta(input_f):
    lines = [i.strip() for i in open(input_f)]
    return len(lines) > 1 and lines[1] != ''
def process(ouput_base_dir, input_file):
    pdb_name = input_file.split('/')[-1].split(".")[0]
    input_path = os.path.join(ouput_base_dir, pdb_name, pdb_name +  ".fasta")
    if not os.path.exists(input_path):
        return False
    if not check_fasta(input_path):
        return False
    output_dir = os.path.join(ouput_base_dir, pdb_name)

    cmd = "bash /mnt/e/study/RoseTTAFold-Train/run_generate_feat.sh %s %s" % (input_path, output_dir)
    print(cmd)
    os.system(cmd)
    return True


if __name__ == '__main__':
    ouput_base_dir = sys.argv[1]
    for line in codecs.open("train-pdb.list", "r", "utf-8"):
        state = process(ouput_base_dir, line.strip())
        # if state:
        #     break
