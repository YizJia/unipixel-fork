# Copyright (c) 2026 Yizhen Jia.

import re

import nncore
from torch.utils.data import Dataset

from unipixel.constants import REF_TOKEN, SEG_TOKEN
from unipixel.dataset.hybrid import DATASETS

@DATASETS.register(name='pam_caption')
class PAMCaptionDataset(Dataset):

    ANNO_PATH = 'data/PAM-Test/input_full_md.jsonl'

    DATA_ROOT = 'data/PAM-Test'

    SOURCE = 'pam_caption'

    @classmethod
    def load_annos(self, split='test'):
        assert split == 'test'

        raw_annos = nncore.load(self.ANNO_PATH)

        annos = []
        for raw_anno in raw_annos:
            assert raw_anno.get('type', 'video') == 'video', raw_anno.get('type')

            image_root = raw_anno['image_root']
            frame_root = nncore.join(self.DATA_ROOT, image_root)
            vid = nncore.pure_name(image_root)
            question = 'Please give a detailed description of the highlighted object [0] in the video.'
            oid = 0
            mem_question = f'Please give a detailed description of the highlighted object [{oid}] {REF_TOKEN} in the video.'
            mem_response = f'[{oid}] {SEG_TOKEN}'


            for event in raw_anno['annotations']:
                frames = [nncore.join(frame_root, f) for f in event['frames']]
                obj_frame_inds = [sorted(range(len(frames)))]
                all_frame_inds = sorted(list(set(nncore.flatten(obj_frame_inds))))

                boxes_xywh = event.get('box', [])
                if boxes_xywh and isinstance(boxes_xywh[0][0], (int, float)):
                    boxes_xywh = boxes_xywh[0]
                    x1, y1 = int(boxes_xywh[0]), int(boxes_xywh[1])
                    x2, y2 = int(boxes_xywh[0] + boxes_xywh[2]), int(boxes_xywh[1] + boxes_xywh[3])
                    boxes = [x1, y1, x2, y2]
                # if len(boxes) == 1 and len(frames) > 1:
                #     boxes = boxes * len(frames)

                anno = dict(
                    source=self.SOURCE,
                    data_type='region_{}',
                    vid=vid,
                    event_id=event.get('event_id'),
                    frames=frames,
                    frame_idx=0,
                    obj_frame_inds=obj_frame_inds,
                    all_frame_inds=all_frame_inds,
                    mem_question=mem_question,
                    mem_response=mem_response,
                    question=question,
                    caption=event['gt'].strip(),
                    boxes=boxes,
                    box_format='xyxy')

                annos.append(anno)

        return annos
