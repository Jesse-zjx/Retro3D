import argparse
from tqdm import tqdm
import time

from utils import get_saved_info, get_pretrained_model, SequenceGenerator, Data


def main():
    # Parser
    parser = argparse.ArgumentParser(description='Batch autoregressive inference')
    parser.add_argument('--pretrained_path', help='pretrained model path', required=True, type=str)
    parser.add_argument('--output_path', help='output result save path', required=True, type=str)
    args = parser.parse_args()
    config, model_dict = get_saved_info(args.pretrained_path)

    data = Data(config=config, train=False, val=False, test=True)
    model = get_pretrained_model(config, model_dict, data)

    test_loader = data.get_loaders()['test']
    reaction_predictor = SequenceGenerator(config, model, data=data, beam_size=config.TEST.BEAM_SIZE)
    reaction_predictor = reaction_predictor.cuda().eval()
    tic = time.time()

    ground_path = args.output_path + '_gt'
    test_results_object = open(args.output_path, 'w')
    test_truths_object = open(ground_path, 'w')
    for batch in tqdm(test_loader, mininterval=2, desc='(Infer)', leave=False):
        src, tgt, gt_context_alignment, gt_nonreactive_mask, src_graph, src_threed, src_atoms = batch
        bond, _ = src_graph
        dist, _ = src_threed
        atoms_coord, atoms_token, atoms_index, batch_index = src_atoms
        src, tgt, gt_context_alignment, gt_nonreactive_mask = src.cuda(), tgt.cuda(), \
                                                                gt_context_alignment.cuda(), \
                                                                gt_nonreactive_mask.cuda()
        bond = bond.cuda()
        dist = dist.cuda()
        atoms_coord, atoms_token, atoms_index, batch_index = \
            atoms_coord.cuda(), atoms_token.cuda(), atoms_index.cuda(), batch_index.cuda()
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        batch_size = tgt.shape[0]
        hyposis = reaction_predictor.forward((src, bond, dist, \
                                              atoms_coord, atoms_token, atoms_index, batch_index))
        for i in range(batch_size):
            tgt_line = ' '.join(test_loader.dataset.reconstruct_smi(tgt[i, :], src=False))
            tgt_line = tgt_line.replace("<sos>", '').replace("<eos>", '').strip()
            test_truths_object.write(tgt_line + '\n')
            pred_results = hyposis[i]  # [i][j]['tokens', 'score', 'alignment', 'positional_scores']
            pred_lines = []
            for j in range(config.TEST.BEAM_SIZE):
                pred_line = pred_results[j]['tokens'].cpu().numpy().tolist()
                pred_line = ' '.join(test_loader.dataset.reconstruct_smi(pred_line))
                pred_line = pred_line.replace("<sos>", '').replace("<eos>", '').strip()
                pred_lines.append(pred_line.strip())
            test_results_object.write('\t'.join(pred_lines) + '\n')

    test_results_object.close()
    test_truths_object.close()
    print('Inference finished,elapse: {:3.3f} min.'.format((time.time()-tic)/60))


if __name__ == '__main__':
    main()
