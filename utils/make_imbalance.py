import numpy as np
import random



def gen_imbalanced_data_unif(dataset_name,dataset, img_num_per_cls):
    new_data = []
    new_targets = []
    if dataset_name == "svhn":
        targets_np = np.array(dataset.unlabeled_labels, dtype=np.int64)
    else:
        targets_np = np.array(dataset.unlabeled_targets, dtype=np.int64)
    classes = np.unique(targets_np)

    dataset.num_per_cls_dict = dict()
    for the_class, the_img_num in zip(classes, img_num_per_cls):
        dataset.num_per_cls_dict[the_class] = the_img_num
        idx = np.where(targets_np == the_class)[0]
        np.random.shuffle(idx)
        selec_idx = idx[:the_img_num]
        new_data.append(dataset.unlabeled_data[selec_idx, ...])
        new_targets.extend([the_class, ] * the_img_num)

    new_data = np.vstack(new_data)
    new_idx = [i for i in range(len(new_data))]
    random.shuffle(new_idx)
    dataset.data = np.vstack((dataset.data, new_data[new_idx]))
    if dataset_name == "svhn":
        dataset.labels = dataset.labels + list(np.array(new_targets)[new_idx])
    else:
        dataset.targets = dataset.targets + list(np.array(new_targets)[new_idx])


    return dataset

