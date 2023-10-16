import torch
import torch.distributed as dist


class Metric(object):
    """
    用于单机或分布式训练的评价标准的类
    需要评价训练结果时候在每个机器中单独计算正确词数量和总词数，用distributed_sum函数把不同机器上的参数汇总到0号机器中计算最终的评价结果
    """
    def __init__(self, trg_pad_idx=0, topk=5):
        super(Metric, self).__init__()
        self.trg_pad_idx = trg_pad_idx
        self.topk = topk
        self.n_topk_correct, self.n_word_total, self.n_word_correct = 0, 0, 0
        self.total_loss = 0

    def update(self, pred, gold, loss):
        pred.cpu()
        gold.cpu()
        _, maxk_idx = pred.topk(self.topk, dim=1)
        pred = pred.max(1)[1]

        gold = gold.contiguous().view(-1)
        non_pad_mask = gold.ne(self.trg_pad_idx)
        top1_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
        topk_correct = maxk_idx[:, 0].eq(gold).masked_select(non_pad_mask).sum().item()
        for i in range(1, self.topk):
            # print(topk_correct)
            topk_correct += maxk_idx[:, i].eq(gold).masked_select(non_pad_mask).sum().item()
        n_word = non_pad_mask.sum().item()
        # print(top1_correct,topk_correct,n_word)
        self.n_topk_correct += topk_correct
        self.n_word_correct += top1_correct
        self.n_word_total += n_word
        self.total_loss += loss

    def compute(self):
        loss_per_word = self.total_loss / self.n_word_total
        accuracy = self.n_word_correct / self.n_word_total
        accuracy_topk = self.n_topk_correct / self.n_word_total
        return loss_per_word, accuracy, accuracy_topk


def distributed_sum(data):
    """
    汇总各进程的data并相加
    """
    tensor = torch.tensor(data, dtype=torch.float64)
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]

    dist.all_gather(output_tensors, tensor)
    sum = torch.stack(output_tensors, dim=0)
    sum = sum.sum(dim=0)

    return sum.cpu().numpy()
