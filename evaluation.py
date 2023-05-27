import os, sys
from rdkit.Chem import AllChem
from rdkit import Chem
from multiprocessing import Pool
from optparse import OptionParser

cwd = os.getcwd()
parser = OptionParser()
parser.add_option("-o", "--output_file", dest="output_file", default='cache/trm_log/log_50K/USPTO_50K/Transformer/2022-11-30-21-08/test_result', help='模型输出的预测结果')
parser.add_option("-t", "--test_file", dest="test_file", default = 'datasets/data/USPTO_50K/test_targets', help='target标签')
parser.add_option("-c", "--num_cores", dest="num_cores", default=10)
parser.add_option("-n", "--top_n", dest="top_n", default=10)
opts, args = parser.parse_args()

num_cores = int(opts.num_cores)
top_n = int(opts.top_n)

def convert_cano(smi):
    try:
        smi = smi.replace('|', '.')  # 不同数据集需要注意这里
        mol = AllChem.MolFromSmiles(smi)
        smiles = Chem.MolToSmiles(mol)
    except:
        smiles = '####'
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
    pred_targets = f.readlines()  # 读入的文件每一行为10个预测结果，每个预测结果用'\t'符号隔开

with open(opts.test_file, 'r') as f:
    test_targets_list = f.readlines()  # (5004)

beam_size = (len(pred_targets)//len(test_targets_list)-2)//3

if(len(pred_targets)>len(test_targets_list)):
    src,test_targets_list,pred_targets = split_generate_result(pred_targets,beam_size)

pred_targets_beam_10_list = [line.strip().split('\t') for line in pred_targets]  # (5004,10)

num_rxn = len(test_targets_list)  # (5004)
# convert_cano: smile->mol->smile
test_targets_strip_list = [convert_cano(line.replace(' ', '').strip()) for line in test_targets_list]

def smi_valid_eval(ix):
    invalid_smiles = 0
    for j in range(top_n):
        output_pred_strip = pred_targets_beam_10_list[ix][j].replace(' ', '').strip()
        mol = AllChem.MolFromSmiles(output_pred_strip)
        if mol:
            pass
        else:
            invalid_smiles += 1
    return invalid_smiles

def pred_topn_eval(ix):
    pred_true = 0
    for j in range(top_n):
        output_pred_split_list = pred_targets_beam_10_list[ix][j].replace(' ', '').strip()
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

    # 单线程调试
    # invalid_smiles_total = 0
    # for i in range(num_rxn):
    #     invalid_smiles = smi_valid_eval(i)
    #     invalid_smiles_total += invalid_smiles
    
    # pred_true_total = 0
    # for i in range(num_rxn):
    #     pred_true = pred_topn_eval(i)
    #     pred_true_total += pred_true

    save_path = os.path.join(os.path.dirname(opts.output_file),'evaluation')
    if '_avg' in opts.output_file:
        save_path += '_avg'
    with open(save_path, 'a')as f:
        f.write("Number of invalid SMILES: {}\n".format(invalid_smiles_total))
        f.write("Number of SMILES candidates: {}\n".format(num_rxn * top_n))
        f.write("Invalid SMILES rate: {0:.3f}\n".format(invalid_smiles_total / (num_rxn * top_n)))
        f.write("Number of matched examples: {}\n".format((pred_true_total)))
        f.write("Top-{}".format(top_n) +" accuracy: {0:.3f}\n".format(pred_true_total / (num_rxn)))
        f.write('\n')