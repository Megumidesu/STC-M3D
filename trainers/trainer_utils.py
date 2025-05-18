import os
import pickle

import numpy as np
import torch
import torch.distributed as dist


def merge_results_dist(tmpdir, part_logits, part_labels):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    pickle.dump(torch.cat(part_logits).cpu().numpy(),
                open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(rank)), 'wb'))
    pickle.dump(torch.cat(part_labels).cpu().numpy(),
                open(os.path.join(tmpdir, 'label_part_{}.pkl'.format(rank)), 'wb'))
    dist.barrier()

    if rank == 0:

        part_list = []
        part_label_list = []

        for i in range(world_size):
            part_file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))
            part_label = os.path.join(tmpdir, 'label_part_{}.pkl'.format(i))

            part_list.append(pickle.load(open(part_file, 'rb')))
            part_label_list.append(pickle.load(open(part_label, 'rb')))

        part_list = np.concatenate(part_list, axis=0)
        part_label_list = np.concatenate(part_label_list, axis=0)

        logits_all = torch.from_numpy(part_list)
        labels_all = torch.from_numpy(part_label_list)

        return logits_all, labels_all
    else:
        return None, None
