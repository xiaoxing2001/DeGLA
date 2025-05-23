import os.path as op
from typing import List
import json


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj
from src.training.BaseDataset import BaseDataset
class ICFGPEDES(BaseDataset):
    """
    ICFG-PEDES

    Reference:
    Semantically Self-Aligned Network for Text-to-Image Part-aware Person Re-identification arXiv 2107

    URL: http://arxiv.org/abs/2107.12666

    Dataset statistics:
    # identities: 4102
    # images: 34674 (train) + 4855 (query) + 14993 (gallery)
    # cameras: 15
    """
    dataset_dir = '/home/face/kaichengyang/xiaoxinghu/data/ICFG-PEDES'

    def __init__(self, root='', verbose=True):
        super(ICFGPEDES, self).__init__()
        self.dataset_dir = '/home/face/kaichengyang/xiaoxinghu/data/ICFG-PEDES'
        self.img_dir = op.join(self.dataset_dir, 'imgs/')
        self.anno_path = op.join(self.dataset_dir, 'ICFG-PEDES.json')
        self._check_before_run()

        self.train_annos, self.test_annos, self.val_annos = self._split_anno(self.anno_path)

        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)

        if verbose:
            self.logger.info("=> ICFG-PEDES Images and Captions are loaded")
            self.show_dataset_info()


    def _split_anno(self, anno_path: str):
        train_annos, test_annos, val_annos = [], [], []
        annos = read_json(anno_path)
        for anno in annos:
            if anno['split'] == 'train':
                train_annos.append(anno)
            elif anno['split'] == 'test':
                test_annos.append(anno)
            else:
                val_annos.append(anno)
        return train_annos, test_annos, val_annos

  
    def _process_anno(self, annos: List[dict], training=False):
        pid_container = set()
        if training:
            dataset = []
            image_id = 0
            for anno in annos:
                pid = int(anno['id'])
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['file_path'])
                captions = anno['captions'] # caption list
                for caption in captions:
                    dataset.append((pid, image_id, img_path, caption))
                    
                image_id += 1

            for idx, pid in enumerate(pid_container):
                # check pid begin from 0 and no break
                assert idx == pid, f"idx: {idx} and pid: {pid} are not match"
            return dataset, pid_container
        else:
            dataset = {}
            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []
            for anno in annos:
                pid = int(anno['id'])
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['file_path'])
                img_paths.append(img_path)
                image_pids.append(pid)
                caption_list = anno['captions'] # caption list
                for caption in caption_list:
                    captions.append(caption)
                    caption_pids.append(pid)
            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "captions": captions
            }
            return dataset, pid_container


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        if not op.exists(self.anno_path):
            raise RuntimeError("'{}' is not available".format(self.anno_path))
import re

def remove_punctuation_and_spaces(text):
    # 使用正则表达式去掉标点符号和空格
    cleaned_text = re.sub(r'[^\w\s]', ' ', text)
    # cleaned_text = re.sub(r'\s+', '', cleaned_text)
    return cleaned_text