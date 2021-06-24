import torch
import random
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


def set_seed(seed, n_gpu=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def save_model_seperately(model, output_dir, args, optimizer, scheduler):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)

    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    logger.info("Saving optimizer and scheduler states to %s", output_dir)


def load_features(dir):
    """ Load all features from the targeted directory """
    if os.path.isdir(dir):
        files = os.listdir(dir)
        fea = dict()
        for file in files:
            if not os.path.isdir(file):
                this_fea = torch.load(os.path.join(dir, file))
                if len(fea.keys()) == 0:
                    fea = this_fea
                else:
                    for k in fea.keys():
                        if isinstance(fea[k], list):
                            fea[k] = fea[k] + this_fea[k]
                        elif isinstance(fea[k], torch.Tensor):
                            fea[k] = torch.cat([fea[k], this_fea[k]], dim=0)
        return fea
    else:
        return torch.load(dir)


def save_features(features, cached_examples_dir, file_size=100000):
        """ Save features into seperate files under the targeted directory """
        this_fea = dict()
        for i in range(0, len(features[list(features.keys())[0]]), file_size):
            for key in features.keys():
                this_fea[key] = features[key][i: i+file_size]
            logger.info("Saving examples into cached file %s", cached_examples_dir)
            torch.save(this_fea, os.path.join(cached_examples_dir, "fea_" + str(i)))
        return features

