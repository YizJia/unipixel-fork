# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import argparse
import copy
import io
import random
import re

import imageio.v3 as iio
import matplotlib.pyplot as plt
import nncore
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T
from matplotlib.patches import Rectangle
from PIL import Image

from sam2.modeling.sam2_utils import get_next_point, sample_box_points
from unipixel.constants import MEM_TOKEN
from unipixel.dataset.hybrid import DATASETS
from unipixel.dataset.utils import process_masks, process_vision_info
from unipixel.model.builder import build_model
from unipixel.utils.io import load_frames
from unipixel.utils.transforms import get_sam2_transform
from unipixel.utils.visualizer import Visualizer, random_color

SEG_MEMORY_PROMPTS = [
    '{} Please localize the relevant objects with IDs.',
    '{} Segment all the relevant target(s) with ID(s) in the video.',
    '{} Can you segment the target object(s) with ID(s) in the video?',
    'Please segment the mentioned objects with IDs in the question: {}',
    'Find and track the relevant objects with IDs in the query: {}',
    "Given the query '{}', please provide segmentation masks for the relevant objects with IDs.",
]


def compute_iou_volume(pred: torch.Tensor, gt: torch.Tensor):
    pred_flat = pred.view(-1)
    gt_flat = gt.view(-1)
    intersection = (pred_flat & gt_flat).sum().float()
    union = (pred_flat | gt_flat).sum().float()
    return (intersection / (union + 1e-6)).item()


def compute_f_score_volume(pred: torch.Tensor, gt: torch.Tensor):
    pred_flat = pred.view(-1)
    gt_flat = gt.view(-1)
    tp = (pred_flat & gt_flat).sum().float()
    fp = (pred_flat & ~gt_flat).sum().float()
    fn = (~pred_flat & gt_flat).sum().float()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return (2 * precision * recall / (precision + recall + 1e-6)).item()


def compute_j_and_f_volume(pred: torch.Tensor, gt: torch.Tensor):
    assert pred.shape == gt.shape
    j = compute_iou_volume(pred, gt)
    f = compute_f_score_volume(pred, gt)
    return j, f, (j + f) / 2


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap('tab10')
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_type')
    parser.add_argument('--model_path')
    parser.add_argument('--res_pred_path')
    parser.add_argument('--vis_pred_path')
    parser.add_argument('--image_size', type=int)
    parser.add_argument('--sample_frames', type=int, default=16)
    parser.add_argument('--num_threads', type=int, default=0)
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--dump', type=int, default=-1)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    assert args.prompt_type in ('point', 'box', 'mix')

    if args.chunk > 1:
        pred_path = nncore.join(args.res_pred_path, f'output_{args.index}.json')
    else:
        pred_path = nncore.join(args.res_pred_path, 'output.json')

    print(f'Chunk: {args.chunk} Index: {args.index} Output Path: {pred_path}')

    model, processor = build_model(args.model_path, image_size=args.image_size, device=args.device)
    device = next(model.parameters()).device

    sam2_transform = get_sam2_transform(model.config.sam2_image_size)

    annos = DATASETS.get('pam_caption').load_annos(split='test')
    annos = [annos[i::args.chunk] for i in range(args.chunk)][args.index]

    for i, anno in enumerate(annos):
        assert '{}' in anno['data_type']
        annos[i]['data_type'] = anno['data_type'].format(args.prompt_type)

    dumps = []
    for i in nncore.ProgressBar(range(len(annos))):
        anno = copy.deepcopy(annos[i])
        dump = copy.deepcopy(annos[i])

        frames, paths, inds = load_frames(
            anno['frames'],
            sample_frames=args.sample_frames,
            sample_type='uniform',
            sample_for_llm_only=False,
            num_threads=args.num_threads)

        frame_size = frames.shape[1:3]

        prompt_frame_idx = inds.index(anno['all_frame_inds'].index(anno['frame_idx']))

        question = SEG_MEMORY_PROMPTS[0].format(anno['mem_question'] + '.')
        response = anno['mem_response']

        object_ids = re.findall(r'\[(\d+)\]', response)

        print()
        print(anno['frames'])
        print(question)
        print(response)

        messages = [{
            'role':
            'user',
            'content': [{
                'type': 'video',
                'video': paths,
                'num_threads': args.num_threads,
                'min_pixels': 128 * 28 * 28,
                'max_pixels': 256 * 28 * 28 * int(args.sample_frames / len(paths))
            }, {
                'type': 'text',
                'text': question
            }]
        }, {
            'role': 'assistant',
            'content': response
        }]

        data_types = ['box']

        point_coords, point_labels, point_frames = [], [], []
        
        box_coords = torch.tensor(args['boxes'], dtype=torch.int32, device=device).view(1, 1, 4)
        obj_point_coords = box_coords.reshape(-1, 2, 2)
        top_left_label = 2
        bottom_right_label = 3
        B = box_coords.shape[0]
        obj_point_labels = torch.tensor([top_left_label, bottom_right_label], dtype=torch.int, device=device).repeat(B)

        # assuming one object has only one point or box
        # point_coords: num_objs * num_points * 2
        # point_labels: num_objs * num_points (0: pos 1: neg 2: top left 3: bottom right)
        # point_frames: num_video (1) * num_obj_per_video
        point_coords.append(obj_point_coords)
        point_labels.append(obj_point_labels)
        point_frames.append(prompt_frame_idx)
        point_frames = [torch.LongTensor(point_frames)]

        text = processor.apply_chat_template(messages)

        images, videos = process_vision_info(messages)

        data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
        data = data.to(device)

        data['frames'] = [sam2_transform(frames).to(dtype=model.sam2.dtype, device=device)]
        data['frame_size'] = [frame_size]
        data['point_coords'] = [[p.to(device) for p in point_coords]]
        data['point_labels'] = [[p.to(device) for p in point_labels]]
        data['point_frames'] = [[p.to(device) for p in point_frames]]

        with torch.inference_mode():
            model(**data)

        assert model.seg[0].size(0) == 1
        assert model.seg[0].size(1) == frames.size(0)

        pred_mask = copy.deepcopy([s.cpu() for s in model.seg])
        label_mask = torch.cat(model.seg[:len(object_ids)]).transpose(0, 1).float()

        # ====================================================================

        question, ans = anno['question'], anno['ans']

        masks = anno['single_frame_masks']

        print(len(anno['all_frame_inds']), anno['frame_idx'], anno['all_frame_inds'])
        print(question)

        frames, paths, inds = load_frames(
            anno['video_path'],
            anno['all_frame_inds'],
            anno['frame_idx'],
            sample_frames=args.sample_frames,
            sample_type='uniform',
            sample_for_llm_only=False,
            num_threads=args.num_threads)

        frame_size = frames.shape[1:3]

        # all samples should be videos
        assert len(paths) > 1

        assert label_mask.shape[2:] == frame_size
        cache_mask = label_mask.clone()
        refer_mask = label_mask.clone()
        label_mask = T.resize(label_mask, (model.config.sam2_image_size, model.config.sam2_image_size))
        label_mask = label_mask > 0

        # ensure refer mask has the correct number of frames
        num_objs = refer_mask.size(1)
        if refer_mask.size(0) % 2 != 0:
            refer_mask = torch.cat((refer_mask, refer_mask[-1, None]))
        refer_mask = refer_mask.flatten(1)
        refer_mask = F.max_pool1d(refer_mask.transpose(-1, -2), kernel_size=2, stride=2).transpose(-1, -2)
        refer_mask = refer_mask.view(-1, num_objs, *frame_size)

        prompt = question

        vids = []
        for obj_idx in range(num_objs):
            if (refer_mask[:, obj_idx] == 0).all():
                refer_mask[:, obj_idx] = 1
            tids = (refer_mask[:, obj_idx].any(dim=(-1, -2)).nonzero()[:, 0] * 2 + 1).tolist()
            vids.append(tids)

        prefix = f'Here is a video with {len(paths)} frames denoted as <1> to <{len(paths)}>. The highlighted regions are as follows:\n'

        matches = re.findall(r'<object\d+><region>', prompt)
        assert len(matches) == len(vids) > 0, prompt
        for j, (match, tids) in enumerate(zip(matches, vids)):
            # '<object0><region>' -> '0' '<object0>'
            oid, obj = match[7:-9], match[:-8]
            assert oid == str(int(oid)), match
            assert oid == object_ids[j], (oid, object_ids, j)
            prefix += f'[{oid}]: ' + ' '.join([f'<{tid}>-<{tid + 1}> ' + MEM_TOKEN for tid in tids]) + '\n'
            prompt = prompt.replace(match, f'[{oid}]').replace(obj, f'[{oid}]')

        assert '<region>' not in prompt and '<object' not in prompt, prompt

        prompt = prefix + prompt
        print(prompt)
        print(ans)

        messages = [{
            'role':
            'user',
            'content': [{
                'type': 'video',
                'video': paths,
                'num_threads': args.num_threads,
                'min_pixels': 128 * 28 * 28,
                'max_pixels': 256 * 28 * 28 * int(args.sample_frames / len(paths))
            }, {
                'type': 'text',
                'text': prompt
            }]
        }]

        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        text += 'The best option is ('

        images, videos = process_vision_info(messages)

        data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
        data = data.to(device)

        # spatial patch size set to 14, spatial merge size set to 2
        refer_mask = T.resize(refer_mask, (data['video_grid_thw'][0][1] * 14, data['video_grid_thw'][0][2] * 14))
        refer_mask = F.max_pool2d(refer_mask, kernel_size=28, stride=28)
        refer_mask = refer_mask > 0

        if not refer_mask.any():
            print('[WARNING] refer mask is empty')

        assert refer_mask.size(0) == data['video_grid_thw'][0][0]
        assert refer_mask.size(2) == data['video_grid_thw'][0][1] // 2
        assert refer_mask.size(3) == data['video_grid_thw'][0][2] // 2

        data['refer_mask'] = [refer_mask.to(device)]

        output_ids = model.generate(
            **data,
            do_sample=False,
            temperature=None,
            top_k=None,
            top_p=None,
            repetition_penalty=None,
            max_new_tokens=512)

        assert data.input_ids.size(0) == output_ids.size(0) == 1
        output_ids = output_ids[0, data.input_ids.size(1):]

        if output_ids[-1] == processor.tokenizer.eos_token_id:
            output_ids = output_ids[:-1]

        response = processor.decode(output_ids, clean_up_tokenization_spaces=False)
        print(response)

        dump['response'] = response

        # dummy output mask
        out_mask = cache_mask.transpose(0, 1)
        assert out_mask.size(1) == len(inds)

        masks = process_masks(dict(mask_type='rle', masks=anno['masks']), frame_size, inds)
        ious, j_values, f_values, jf_values = [], [], [], []
        for k in range(len(masks)):
            mask = torch.stack(masks[k]).bool()
            pred = out_mask[k].bool()
            j_value, f_value, jf_value = compute_j_and_f_volume(pred, mask)
            j_values.append(j_value)
            f_values.append(f_value)
            jf_values.append(jf_value)
            assert mask.size() == pred.size(), (mask.size(), pred.size())
            inter = (mask * pred).sum().item()
            union = (mask + pred).sum().item()
            iou = 1 if union == 0 else inter / union
            ious.append(iou)
        iou = sum(ious) / len(ious)
        jf_value = sum(jf_values) / len(jf_values)

        dump['iou'] = iou
        dump['j_values'] = j_values
        dump['f_values'] = f_values

        if args.dump > 0 and i % args.dump == 0:
            nncore.mkdir(args.vis_pred_path)
            jpg_path = nncore.join(args.vis_pred_path, f"{args.index}_{i}_{anno['vid'].replace('/', '_')}.jpg")
            gif_path = nncore.join(args.vis_pred_path, f"{args.index}_{i}_{anno['vid'].replace('/', '_')}.gif")

            for obj_id in range(out_mask.size(0)):
                point_coords[obj_id][:, :,
                                     0] = point_coords[obj_id][:, :, 0] / model.config.sam2_image_size * frame_size[1]
                point_coords[obj_id][:, :,
                                     1] = point_coords[obj_id][:, :, 1] / model.config.sam2_image_size * frame_size[0]

            plt.figure()
            plt.title(' '.join(question.replace('<region>', '').split('\n')[:-1]).replace(' Options:', '\n'))
            plt.imshow(Image.fromarray(frames[prompt_frame_idx].numpy()))
            cmap = plt.get_cmap('tab10')
            for obj_id in range(out_mask.size(0)):
                if (point_labels[obj_id] == 1).all():
                    plt.plot(
                        point_coords[obj_id][0, 0, 0],
                        point_coords[obj_id][0, 0, 1],
                        marker='*',
                        color=cmap(obj_id),
                        markersize=15)
                else:
                    rect = Rectangle((point_coords[obj_id][0, 0, 0], point_coords[obj_id][0, 0, 1]),
                                     point_coords[obj_id][0, 1, 0] - point_coords[obj_id][0, 0, 0],
                                     point_coords[obj_id][0, 1, 1] - point_coords[obj_id][0, 0, 1],
                                     linewidth=2,
                                     edgecolor=cmap(obj_id),
                                     facecolor='none')
                    plt.gca().add_patch(rect)
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(jpg_path)
            plt.close()

            colors = [random_color(rgb=True, maximum=1) for _ in range(out_mask.size(0))]

            visualizer = Visualizer(frames[prompt_frame_idx].numpy())
            for obj_id in range(out_mask.size(0)):
                if (point_labels[obj_id] == 1).all():
                    visualizer.output.ax.plot(
                        point_coords[obj_id][0, 0, 0],
                        point_coords[obj_id][0, 0, 1],
                        marker='*',
                        color=colors[obj_id],
                        markersize=40)
                else:
                    rect = Rectangle((point_coords[obj_id][0, 0, 0], point_coords[obj_id][0, 0, 1]),
                                     point_coords[obj_id][0, 1, 0] - point_coords[obj_id][0, 0, 0],
                                     point_coords[obj_id][0, 1, 1] - point_coords[obj_id][0, 0, 1],
                                     linewidth=5,
                                     edgecolor=colors[obj_id],
                                     facecolor='none')
                    visualizer.output.ax.add_patch(rect)

            visualizer.output.save(
                nncore.join(args.vis_pred_path, f"{args.index}_{i}_{anno['vid'].replace('/', '_')}_prompt.jpg"))

            figs, imgs = [], []
            for idx in range(len(inds)):
                buffer = io.BytesIO()
                visualizer = Visualizer(frames[idx].numpy())
                for obj_id in range(out_mask.size(0)):
                    fig = visualizer.draw_binary_mask_with_number(
                        out_mask[obj_id, idx].bool().numpy(), color=colors[obj_id], alpha=0.3, anno_mode=['Mask'])
                figs.append(fig)
                fig.save(buffer)
                buffer.seek(0)
                imgs.append(iio.imread(buffer))

            iio.imwrite(
                nncore.join(args.vis_pred_path, f"{args.index}_{i}_{anno['vid'].replace('/', '_')}_{jf_value:.3f}.gif"),
                imgs,
                duration=100,
                loop=0)

            fig_inds = np.linspace(0, len(figs) - 1, 6, dtype=int)
            images = []
            for fig_idx in fig_inds:
                buffer = io.BytesIO()
                figs[fig_idx].save(buffer)
                buffer.seek(0)
                img = Image.open(buffer)
                images.append(img.convert('RGB'))
                buffer.close()

            widths, heights = zip(*(img.size for img in images))
            total_width = sum(widths)
            max_height = max(heights)

            new_img = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))

            x_offset = 0
            for img in images:
                new_img.paste(img, (x_offset, 0))
                x_offset += img.width

            new_img.save(
                nncore.join(args.vis_pred_path, f"{args.index}_{i}_{anno['vid'].replace('/', '_')}_{jf_value:.3f}.jpg"))

            nncore.dump(f'{question}\n{ans}\n{response}\n{fig_inds}\n{prompt_frame_idx}',
                        nncore.join(args.vis_pred_path, f"{args.index}_{i}_{anno['vid'].replace('/', '_')}.txt"))

        dumps.append(dump)

    nncore.dump(dumps, pred_path)
