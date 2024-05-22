import os
from multiprocessing import Pool
from optparse import OptionParser
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

cwd = os.getcwd()
parser = OptionParser()
parser.add_option("-o", "--output_file", dest="output_file")
parser.add_option("-t", "--test_file", dest="test_file")
parser.add_option("-c", "--num_cores", dest="num_cores", default=10)
parser.add_option("-n", "--top_n", dest="top_n", default=10)
opts, args = parser.parse_args()

num_cores = int(opts.num_cores)
top_n = int(opts.top_n)

def convert_cano(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    except:
        smiles = smi
    return smiles

def split_generate_result(file,beam_size):
    sample_gap = (2+3*beam_size)
    num_samples = len(file) // sample_gap
    src = []
    pred = []
    trg = []
    for i in range(num_samples):
        sample = file[i*sample_gap:(i+1)*sample_gap]
        src.append(sample[0].split('\t')[1].strip())
        trg.append(sample[1].split('\t')[1].strip())
        beam = map(lambda x: x.split('\t')[2].strip(),sample[2::3])
        pred.append('\t'.join(beam))
    return src,trg,pred

with open(opts.output_file, 'r') as f:
    pred_targets = f.readlines()

with open(opts.test_file, 'r') as f:
    test_targets_list = f.readlines()  # (5004)

beam_size = (len(pred_targets)//len(test_targets_list)-2)//3

if(len(pred_targets)>len(test_targets_list)):
    src,test_targets_list,pred_targets = split_generate_result(pred_targets,beam_size)

pred_targets_beam_10_list = [line.strip().split('\t') for line in pred_targets]  # (5004,10)

num_rxn = len(test_targets_list)  # (5004)
# convert_cano: smile->mol->smile
test_targets_strip_list = [convert_cano(line.strip()) for line in test_targets_list]

def smi_valid_eval(ix):
    invalid_smiles = 0
    for j in range(top_n):
        output_pred_strip = pred_targets_beam_10_list[ix][j].strip()
        mol = AllChem.MolFromSmiles(output_pred_strip)
        if mol:
            pass
        else:
            invalid_smiles += 1
    return invalid_smiles

def pred_topn_eval(ix):
    pred_true = 0
    for j in range(top_n):
        output_pred_split_list = pred_targets_beam_10_list[ix][j].strip()
        test_targets_split_list = test_targets_strip_list[ix]
        if convert_cano(output_pred_split_list) == convert_cano(test_targets_split_list):
            pred_true += 1
            break
        else:
            continue
    return pred_true

if __name__ == "__main__":
    # calculate invalid SMILES rate
    pool = Pool(num_cores)
    invalid_smiles = pool.map(smi_valid_eval, range(num_rxn), chunksize=1)
    invalid_smiles_total = sum(invalid_smiles)
    # calculate predicted accuracy
    pool = Pool(num_cores)
    pred_true = pool.map(pred_topn_eval, range(num_rxn), chunksize=1)
    pred_true_total = sum(pred_true)
    pool.close()
    pool.join()

    print("Number of invalid SMILES: {}".format(invalid_smiles_total))
    print("Number of SMILES candidates: {}".format(num_rxn * top_n))
    print("Invalid SMILES rate: {0:.3f}".format(invalid_smiles_total / (num_rxn * top_n)))
    print("Number of matched examples: {}".format((pred_true_total)))
    print("Top-{}".format(top_n) +" accuracy: {0:.3f}".format(pred_true_total / (num_rxn)))
    print('\n')