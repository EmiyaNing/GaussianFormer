try:
    from vis_open3d_voxel import (
        save_occ, save_gaussian, save_gaussian_topdown,
        save_gaussian_point,
        save_occ_error, save_gaussian_with_gt_occ,
        vis_gaussian_occ_match,
    )
except:
    try:
        from vis import save_occ, save_gaussian, save_gaussian_topdown
        save_gaussian_point = None
        # save_occ_error / save_gaussian_with_gt_occ / vis_gaussian_occ_match 可能在 vis.py 中不存在，导入兜底
        try:
            from vis_open3d_voxel import (
                save_gaussian_point, save_occ_error,
                save_gaussian_with_gt_occ, vis_gaussian_occ_match)
        except:
            pass
    except:
        print('Load Occupancy Visualization Tools Failed.')
import time, argparse, os.path as osp, os, json
import torch, numpy as np
import torch.distributed as dist

from PIL import Image
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor

import warnings
warnings.filterwarnings("ignore")


def pass_print(*args, **kwargs):
    pass


ALLOCATION_OP_NAMES = {
    0: 'pass_through',
    1: 'clone_parent',
    2: 'clone_child',
    3: 'split_child',
    4: 'opacity_attenuation',
}

ALLOCATION_COLOR_MAP = {
    0: [0.55, 0.55, 0.55],
    1: [0.05, 0.20, 0.95],
    2: [0.15, 0.65, 1.00],
    3: [1.00, 0.10, 0.10],
    4: [1.00, 0.85, 0.05],
}


class AllocationOperationCollector:
    """Runtime collector for AdaptiveAllocationV5 visualization."""

    def __init__(self, model, enabled=False, logger=None):
        self.enabled = enabled
        self.logger = logger
        self._handles = []
        self._original_methods = []
        self.records = []
        self.current_iter = None
        self.current_tag = None
        self.latest_record = None
        self.summary = self._empty_summary()
        if self.enabled:
            self._install(model)

    @staticmethod
    def _empty_summary():
        return {
            'input_gaussians': 0,
            'selected_topk': 0,
            'non_topk_pass_through': 0,
            'clone_parent': 0,
            'clone_child': 0,
            'split_child': 0,
            'opacity_attenuation': 0,
            'output_candidate_count': 0,
            'expected_clone': 0.0,
            'expected_split': 0.0,
            'expected_atten': 0.0,
            'effective_opacity_clone_parent': 0.0,
            'effective_opacity_clone_child': 0.0,
            'effective_opacity_split_child_1': 0.0,
            'effective_opacity_split_child_2': 0.0,
            'effective_opacity_atten': 0.0,
            'num_forwards': 0,
            'num_batch_items': 0,
        }

    def _install(self, model):
        modules = [
            (name, module)
            for name, module in model.named_modules()
            if module.__class__.__name__ == 'AdaptiveAllocationV5'
        ]
        if len(modules) == 0:
            self.enabled = False
            if self.logger is not None:
                self.logger.warning(
                    '[AllocationStatistic] AdaptiveAllocationV5 not found; disabled.')
            return

        for name, module in modules:
            self._handles.append(module.register_forward_hook(self._make_forward_hook(name)))

        if self.logger is not None:
            self.logger.info(
                f'[AllocationStatistic] enabled on {len(modules)} AdaptiveAllocationV5 module(s).')

    def _make_forward_hook(self, module_name):
        def hook(module, inputs, output):
            if not self.enabled:
                return
            labels = getattr(module, 'latest_output_candidate_labels', None)
            stats = getattr(module, 'latest_risky_operation_stats', None)
            selected_mask = getattr(module, 'latest_selected_mask', None)
            risky_prob = getattr(module, 'latest_risky_operation_prob', None)
            if labels is None:
                return
            if output is None or len(output) < 3:
                return
            output_size = int(output[2].shape[1])
            record = self._build_record(
                module_name, labels, stats, selected_mask, risky_prob, output_size)
            self.latest_record = record
            self.records.append(record)
            self._accumulate(record)
        return hook

    @staticmethod
    def _stats_item(stats, batch_idx):
        if isinstance(stats, (list, tuple)) and batch_idx < len(stats):
            return stats[batch_idx] or {}
        return {}

    def _build_record(
            self, module_name, labels, stats, selected_mask, risky_prob,
            output_size):
        labels_cpu = labels.detach().to('cpu').long()
        selected_cpu = selected_mask.detach().to('cpu') if selected_mask is not None else None
        risky_prob_cpu = risky_prob.detach().to('cpu') if risky_prob is not None else None
        per_batch = []
        output_ids = []

        for b in range(labels_cpu.shape[0]):
            cur_label = labels_cpu[b].reshape(-1)
            out = torch.full((output_size,), -1, dtype=torch.long)
            copy_count = min(output_size, int(cur_label.numel()))
            if copy_count > 0:
                out[:copy_count] = cur_label[:copy_count]

            pass_mask = out == 0
            clone_parent_mask = out == 1
            clone_child_mask = out == 2
            split_child_mask = out == 3
            atten_mask = out == 4

            stat = self._stats_item(stats, b)
            selected_topk = int(stat.get(
                'selected_topk',
                int(selected_cpu[b].sum().item()) if selected_cpu is not None else 0))
            input_gaussians = int(stat.get('input_gaussians', 0))
            if input_gaussians <= 0:
                input_gaussians = int(pass_mask.sum().item()) + selected_topk
            output_count = int(stat.get('output_candidate_count', int((out >= 0).sum().item())))

            p_clone_mean = float(stat.get('topk_p_clone_mean', 0.0))
            p_split_mean = float(stat.get('topk_p_split_mean', 0.0))
            p_atten_mean = float(stat.get('topk_p_atten_mean', 0.0))
            if risky_prob_cpu is not None and selected_cpu is not None:
                cur_selected = selected_cpu[b].to(torch.bool)
                if cur_selected.any():
                    cur_prob = risky_prob_cpu[b][cur_selected]
                    p_clone_mean = float(cur_prob[:, 0].mean().item())
                    p_split_mean = float(cur_prob[:, 1].mean().item())
                    p_atten_mean = float(cur_prob[:, 2].mean().item())

            item = {
                'batch_index': b,
                'input_gaussians': input_gaussians,
                'selected_topk': selected_topk,
                'non_topk_pass_through': int(pass_mask.sum().item()),
                'clone_parent': int(clone_parent_mask.sum().item()),
                'clone_child': int(clone_child_mask.sum().item()),
                'split_child': int(split_child_mask.sum().item()),
                'opacity_attenuation': int(atten_mask.sum().item()),
                'output_candidate_count': output_count,
                'padded_output_gaussians': int(output_size),
                'expected_clone': float(stat.get('expected_clone', 0.0)),
                'expected_split': float(stat.get('expected_split', 0.0)),
                'expected_atten': float(stat.get('expected_atten', 0.0)),
                'topk_p_clone_mean': p_clone_mean,
                'topk_p_split_mean': p_split_mean,
                'topk_p_atten_mean': p_atten_mean,
                'effective_opacity_clone_parent': float(stat.get('effective_opacity_clone_parent', 0.0)),
                'effective_opacity_clone_child': float(stat.get('effective_opacity_clone_child', 0.0)),
                'effective_opacity_split_child_1': float(stat.get('effective_opacity_split_child_1', 0.0)),
                'effective_opacity_split_child_2': float(stat.get('effective_opacity_split_child_2', 0.0)),
                'effective_opacity_atten': float(stat.get('effective_opacity_atten', 0.0)),
            }
            per_batch.append(item)
            output_ids.append(out)

        return {
            'iter': self.current_iter,
            'tag': self.current_tag,
            'module': module_name,
            'per_batch': per_batch,
            'output_op_ids': output_ids,
        }

    def _accumulate(self, record):
        self.summary['num_forwards'] += 1
        for item in record['per_batch']:
            self.summary['num_batch_items'] += 1
            for key in (
                'input_gaussians', 'selected_topk', 'non_topk_pass_through',
                'clone_parent', 'clone_child', 'split_child',
                'opacity_attenuation', 'output_candidate_count',
                'expected_clone', 'expected_split', 'expected_atten',
                'effective_opacity_clone_parent',
                'effective_opacity_clone_child',
                'effective_opacity_split_child_1',
                'effective_opacity_split_child_2',
                'effective_opacity_atten',
            ):
                self.summary[key] += item[key]

    def start_iter(self, iter_idx, tag=None):
        if not self.enabled:
            return
        self.current_iter = int(iter_idx)
        self.current_tag = tag
        self.latest_record = None

    def get_output_op_ids(self, batch_idx=0):
        if not self.enabled or self.latest_record is None:
            return None
        output_ids = self.latest_record.get('output_op_ids', [])
        if batch_idx >= len(output_ids):
            return None
        return output_ids[batch_idx].numpy()

    def get_draw_params(self, base_params, batch_idx=0, enable_color=True):
        params = dict(base_params)
        if not enable_color:
            return params
        op_ids = self.get_output_op_ids(batch_idx=batch_idx)
        if op_ids is not None:
            params.update({
                'allocation_color': True,
                'allocation_op_ids': op_ids,
                'allocation_color_map': ALLOCATION_COLOR_MAP,
            })
        return params

    @staticmethod
    def _safe_name(name):
        return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in str(name))

    @staticmethod
    def _item_with_ratios(item):
        output = dict(item)
        input_total = max(output['input_gaussians'], 1)
        output_total = max(output['output_candidate_count'], 1)
        topk_total = max(output['selected_topk'], 1)
        output['ratios'] = {
            'selected_topk': output['selected_topk'] / input_total,
            'non_topk_pass_through': output['non_topk_pass_through'] / input_total,
            'clone_parent': output['clone_parent'] / output_total,
            'clone_child': output['clone_child'] / output_total,
            'split_child': output['split_child'] / output_total,
            'opacity_attenuation': output['opacity_attenuation'] / output_total,
            'output_candidate_count': output['output_candidate_count'] / input_total,
            'expected_clone': output['expected_clone'] / topk_total,
            'expected_split': output['expected_split'] / topk_total,
            'expected_atten': output['expected_atten'] / topk_total,
        }
        return output

    @staticmethod
    def _bar(value, total, width=32):
        if total <= 0:
            return ''
        n = int(round(width * value / total))
        return '#' * n + '.' * (width - n)

    def dump_frame(self, save_dir, batch_idx=0, frame_name=None, print_to_terminal=True):
        if not self.enabled or self.latest_record is None:
            return
        per_batch = self.latest_record.get('per_batch', [])
        if batch_idx >= len(per_batch):
            return

        os.makedirs(save_dir, exist_ok=True)
        tag = frame_name or self.latest_record.get('tag') or f"iter_{self.latest_record.get('iter')}"
        name = self._safe_name(f'{tag}_batch{batch_idx}')
        item = self._item_with_ratios(per_batch[batch_idx])
        payload = {
            'iter': self.latest_record.get('iter'),
            'tag': self.latest_record.get('tag'),
            'module': self.latest_record.get('module'),
            'batch_index': batch_idx,
            'stats': item,
        }

        json_path = os.path.join(save_dir, f'{name}.json')
        with open(json_path, 'w') as f:
            json.dump(payload, f, indent=2)

        md_path = os.path.join(save_dir, f'{name}.md')
        total = item['output_candidate_count']
        bars = [
            ('pass_through', item['non_topk_pass_through']),
            ('clone_parent', item['clone_parent']),
            ('clone_child', item['clone_child']),
            ('split_child', item['split_child']),
            ('atten', item['opacity_attenuation']),
        ]
        with open(md_path, 'w') as f:
            f.write(f"# AdaptiveAllocationV5 Frame Report\n\n")
            f.write(f"- frame: `{tag}`\n")
            f.write(f"- module: `{self.latest_record.get('module')}`\n")
            f.write(f"- batch: `{batch_idx}`\n\n")
            f.write('| Metric | Count | Ratio |\n')
            f.write('| --- | ---: | ---: |\n')
            for key in (
                'input_gaussians', 'selected_topk', 'non_topk_pass_through',
                'clone_parent', 'clone_child', 'split_child',
                'opacity_attenuation', 'output_candidate_count',
                'padded_output_gaussians', 'expected_clone',
                'expected_split', 'expected_atten',
                'topk_p_clone_mean', 'topk_p_split_mean',
                'topk_p_atten_mean',
            ):
                ratio = '-' if key not in item['ratios'] else f"{item['ratios'][key]:.6f}"
                f.write(f"| {key} | {item[key]} | {ratio} |\n")
            f.write('\n## Operation Bars\n\n')
            f.write('| Operation | Count | Bar |\n')
            f.write('| --- | ---: | --- |\n')
            for label, value in bars:
                f.write(f"| {label} | {value} | `{self._bar(value, total)}` |\n")

        if print_to_terminal:
            ratios = item['ratios']
            msg = (
                f"[AllocationStatistic][Frame {tag}][batch {batch_idx}] "
                f"pass={item['non_topk_pass_through']}, "
                f"clone_parent={item['clone_parent']}, "
                f"clone_child={item['clone_child']}, "
                f"split_child={item['split_child']}, "
                f"atten={item['opacity_attenuation']}, "
                f"p_mean=({item['topk_p_clone_mean']:.3f}, "
                f"{item['topk_p_split_mean']:.3f}, "
                f"{item['topk_p_atten_mean']:.3f}), "
                f"input={item['input_gaussians']}, "
                f"output={item['output_candidate_count']} "
                f"({ratios['output_candidate_count']:.2f}x)"
            )
            print(msg)
            if self.logger is not None:
                self.logger.info(msg)

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._original_methods = []
        self.latest_record = None

    def dump(self, save_dir):
        if not self.enabled:
            return
        os.makedirs(save_dir, exist_ok=True)
        summary = dict(self.summary)
        input_total = max(summary['input_gaussians'], 1)
        output_total = max(summary['output_candidate_count'], 1)
        topk_total = max(summary['selected_topk'], 1)
        summary['ratios'] = {
            'selected_topk': summary['selected_topk'] / input_total,
            'non_topk_pass_through': summary['non_topk_pass_through'] / input_total,
            'clone_parent': summary['clone_parent'] / output_total,
            'clone_child': summary['clone_child'] / output_total,
            'split_child': summary['split_child'] / output_total,
            'opacity_attenuation': summary['opacity_attenuation'] / output_total,
            'output_candidate_count': summary['output_candidate_count'] / input_total,
            'expected_clone': summary['expected_clone'] / topk_total,
            'expected_split': summary['expected_split'] / topk_total,
            'expected_atten': summary['expected_atten'] / topk_total,
        }
        json_path = os.path.join(save_dir, 'allocation_stats_summary.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        csv_path = os.path.join(save_dir, 'allocation_stats_per_iter.csv')
        with open(csv_path, 'w') as f:
            f.write(
                'iter,tag,module,batch,input,topk,pass_through,'
                'clone_parent,clone_child,split_child,atten,output,'
                'expected_clone,expected_split,expected_atten,'
                'topk_p_clone_mean,topk_p_split_mean,topk_p_atten_mean,'
                'padded_output\n')
            for record in self.records:
                for item in record['per_batch']:
                    f.write(
                        f"{record['iter']},{record['tag']},{record['module']},"
                        f"{item['batch_index']},{item['input_gaussians']},"
                        f"{item['selected_topk']},{item['non_topk_pass_through']},"
                        f"{item['clone_parent']},{item['clone_child']},"
                        f"{item['split_child']},{item['opacity_attenuation']},"
                        f"{item['output_candidate_count']},"
                        f"{item['expected_clone']},{item['expected_split']},"
                        f"{item['expected_atten']},{item['topk_p_clone_mean']},"
                        f"{item['topk_p_split_mean']},{item['topk_p_atten_mean']},"
                        f"{item['padded_output_gaussians']}\n")

        report_path = os.path.join(save_dir, 'allocation_stats_report.md')
        with open(report_path, 'w') as f:
            f.write('# AdaptiveAllocationV5 Operation Statistics\n\n')
            f.write('| Metric | Count | Ratio |\n')
            f.write('| --- | ---: | ---: |\n')
            for key in (
                'input_gaussians', 'selected_topk',
                'non_topk_pass_through', 'clone_parent', 'clone_child',
                'split_child', 'opacity_attenuation',
                'output_candidate_count', 'expected_clone',
                'expected_split', 'expected_atten',
            ):
                ratio = '-' if key == 'input_gaussians' else f"{summary['ratios'][key]:.6f}"
                f.write(f"| {key} | {summary[key]} | {ratio} |\n")
            f.write('\n## Color Legend\n\n')
            f.write('| Operation | Color |\n')
            f.write('| --- | --- |\n')
            f.write('| non-TopK pass-through | gray |\n')
            f.write('| clone parent | deep blue |\n')
            f.write('| clone child | cyan blue |\n')
            f.write('| split child | red |\n')
            f.write('| opacity attenuation | yellow |\n')

        if self.logger is not None:
            self.logger.info(f'[AllocationStatistic] saved to: {save_dir}')

# ──────────────────────────────────────────────
# FIFO 时序融合 import（流式可视化使用）
# ──────────────────────────────────────────────
try:
    from fifo_eval import (
        FIFOQueue,
        fuse_predictions,
        fused_soft_to_hard,
        soft_pred_to_grid,
        GaussianFIFOQueue,
        gaussian_fifo_fuse_and_render,
    )
    from model.encoder.gaussian_encoder.utils import GaussianPrediction
except ImportError as e:
    _fifo_available = False
    _gaussian_fifo_available = False
    def fuse_predictions(*args, **kwargs):
        raise RuntimeError('fifo_eval 不可用，请确保 fifo_eval.py 存在')
else:
    _fifo_available = True
    _gaussian_fifo_available = True

# ─── 网格参数（与 eval_stream.py 保持一致）───
_H, _W, _D = 200, 200, 16
_PC_MIN = torch.tensor([-50.0, -50.0, -5.0], dtype=torch.float32)
_PC_MAX = torch.tensor([50.0, 50.0, 3.0], dtype=torch.float32)
_GRID_PARAMS = {
    'H': _H, 'W': _W, 'D': _D,
    'pc_min': _PC_MIN,
    'pc_max': _PC_MAX,
}


def main(local_rank, args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    # init DDP
    if args.gpus > 1:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "20507")
        hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
        rank = int(os.environ.get("RANK", 0))  # node id
        gpus = torch.cuda.device_count()  # gpus per node
        print(f"tcp://{ip}:{port}")
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}", 
            world_size=hosts * gpus, rank=rank * gpus + local_rank)
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)

        if local_rank != 0:
            import builtins
            builtins.print = pass_print
    else:
        distributed = False
        world_size = 1
    
    writer = None
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    os.makedirs(args.work_dir, exist_ok=True)
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger('selfocc', log_file=log_file)
    MMLogger._instance_dict['selfocc'] = logger
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    import model
    from dataset import get_dataloader

    my_model = build_segmentor(cfg.model)
    my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    if distributed:
        if cfg.get('syncBN', True):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            logger.info('converted sync bn.')

        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        raw_model = my_model.module
    else:
        my_model = my_model.cuda()
        raw_model = my_model
    logger.info('done ddp model')

    # 注意: 不添加 num_samples 到 val_dataset_config，
    #       NuScenesDataset.__init__() 不接受该参数（只在模型中用到）
    cfg.val_dataset_config.update({
        "vis_indices": args.vis_index,
        "vis_scene_index": args.vis_scene_index})

    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=distributed,
        val_only=True)
    
    # resume and load
    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from
    
    logger.info('resume from: ' + cfg.resume_from)
    logger.info('work dir: ' + args.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        try:
            # raw_model.load_state_dict(ckpt['state_dict'], strict=True)
            raw_model.load_state_dict(ckpt.get('state_dict', ckpt), strict=True)
        except:
            os.system(f"python modify_weight.py --work-dir {args.work_dir} --epoch {args.epoch}")
            cfg.resume_from = os.path.join(args.work_dir, f"epoch_{args.epoch}_mod.pth")
            ckpt = torch.load(cfg.resume_from, map_location=map_location)
            raw_model.load_state_dict(ckpt['state_dict'], strict=True)
        print(f'successfully resumed.')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        print(raw_model.load_state_dict(state_dict, strict=False))
        
    print_freq = cfg.print_freq
    from misc.metric_util import MeanIoU
    miou_metric = MeanIoU(
        list(range(1, 17)),
        17, #17,
        ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation'],
         True, 17, filter_minmax=False)
    miou_metric.reset()

    my_model.eval()
    os.environ['eval'] = 'true'
    allocation_vis_enabled = (
        args.vis_gaussian_allocation_color or args.allocation_statistic)
    if (args.vis_occ or args.vis_occ_error or args.vis_gaussian or
            args.vis_gaussian_point or args.vis_gaussian_topdown or
            args.vis_gaussian_match or allocation_vis_enabled):
        save_dir = os.path.join(args.work_dir, f'vis_ep{args.epoch}')
        os.makedirs(save_dir, exist_ok=True)
    if args.model_type == "base":
        draw_gaussian_params = dict(
            scalar = 1.5,
            ignore_opa = False,
            filter_zsize = False
        )
    elif args.model_type == "prob":
        draw_gaussian_params = dict(
            scalar = 2.0,
            ignore_opa = True,
            filter_zsize = True
        )
    draw_gaussian_params['adaptive_color'] = args.vis_gaussian_adaptive_color
    draw_gaussian_params['adaptive_color_seed'] = args.seed

    allocation_collector = AllocationOperationCollector(
        raw_model,
        enabled=(local_rank == 0 and allocation_vis_enabled),
        logger=logger)

    try:
        with torch.no_grad():
            for i_iter_val, data in enumerate(val_dataset_loader):
                allocation_collector.start_iter(i_iter_val, f'val_{i_iter_val}')

                for k in list(data.keys()):
                    if isinstance(data[k], torch.Tensor):
                        data[k] = data[k].cuda()
                input_imgs = data.pop('img')
                ori_imgs = data.pop('ori_img')
                for i in range(ori_imgs.shape[-1]):
                    ori_img = ori_imgs[0, ..., i].cpu().numpy()
                    ori_img = ori_img[..., [2, 1, 0]]
                    ori_img = Image.fromarray(ori_img.astype(np.uint8))
                    ori_img.save(os.path.join(save_dir, f'{i_iter_val}_image_{i}.png'))

                result_dict = my_model(imgs=input_imgs, metas=data)
                if args.vis_gaussian_gt:
                    gaussian_ctr = result_dict['gaussian'].means[0]
                    gt_occ = result_dict['sampled_label'][0]
                    from vis_open3d_voxel import get_grid_coords
                    grids = get_grid_coords([200, 200, 16], [0.4, 0.4, 0.4])
                    grids = torch.tensor(grids, device=gaussian_ctr.device) - torch.tensor([40, 40, 1], device=gaussian_ctr.device)
                    occ_mask = gt_occ < 17
                    gt_grid_occ = grids[occ_mask]
                    cated_points = torch.cat([gaussian_ctr, gt_grid_occ], dim=0)
                    color_gauss = torch.ones_like(gaussian_ctr, dtype=torch.float32) * torch.tensor([1.0, 1.0, 1.0], device=gt_occ.device)
                    color_gts = torch.ones_like(gt_grid_occ, dtype=torch.float32) * torch.tensor([0, 1.0, 1.0], device=gt_occ.device)
                    cated_colors = torch.cat([color_gauss, color_gts], dim=0)

                    origin_poitns = data['lidar_points'][0][:, :3]
                    origin_mask_x = (origin_poitns[:, 0] > -40) & (origin_poitns[:, 0] < 40)
                    origin_mask_y = (origin_poitns[:, 1] > -40) & (origin_poitns[:, 1] < 40)
                    origin_mask_z = (origin_poitns[:, 2] > -1) & (origin_poitns[:, 2] < 5.4)
                    origin_mask = origin_mask_x & origin_mask_y & origin_mask_z
                    filter_points = origin_poitns[origin_mask]

                    cated_points = torch.cat([cated_points, filter_points], dim=0)
                    color_points = torch.ones_like(filter_points, dtype=torch.float32) * torch.tensor([1.0, 1.0, 0], device=gt_occ.device)
                    cated_colors = torch.cat([cated_colors, color_points], dim=0)

                    from open3d_vis_utils import draw_scenes
                    draw_scenes(points=cated_points.detach().cpu().numpy(), point_colors=cated_colors.detach().cpu().numpy())

                for idx, pred in enumerate(result_dict['final_occ']):
                    pred_occ = pred
                    gt_occ = result_dict['sampled_label'][idx]
                    occ_shape = [200, 200, 16]
                    alloc_draw_params = allocation_collector.get_draw_params(
                        draw_gaussian_params, batch_idx=idx,
                        enable_color=args.vis_gaussian_allocation_color)

                    if args.vis_gaussian_topdown:
                        save_gaussian_topdown(
                            save_dir,
                            result_dict['anchor_init'],
                            result_dict['gaussians'],
                            f'val_{i_iter_val}_topdown'
                        )
                    if args.vis_occ:
                        save_occ(
                            save_dir,
                            pred_occ.reshape(1, *occ_shape),
                            f'val_{i_iter_val}_pred',
                            True, 0, dataset=args.dataset)
                        save_occ(
                            save_dir,
                            gt_occ.reshape(1, *occ_shape),
                            f'val_{i_iter_val}_gt',
                            True, 0, dataset=args.dataset)
                    if args.vis_gaussian:
                        save_gaussian(
                            save_dir,
                            result_dict['gaussian'],
                            f'val_{i_iter_val}_gaussian',
                            **alloc_draw_params)
                    if args.vis_gaussian_point:
                        if save_gaussian_point is None:
                            raise ImportError('save_gaussian_point 需要 vis_open3d_voxel.py / Open3D 可用')
                        save_gaussian_point(
                            save_dir,
                            result_dict['gaussian'],
                            f'val_{i_iter_val}_gaussian',
                            **alloc_draw_params)
                    if args.vis_gaussian_each_stage:
                        for gaussian in result_dict['gaussians']:
                            save_gaussian(
                                save_dir,
                                gaussian,
                                f'val_{i_iter_val}_gaussian',
                                **alloc_draw_params
                            )
                    if args.vis_occ_error:
                        save_occ_error(
                            save_dir,
                            pred_occ.reshape(*occ_shape),
                            gt_occ.reshape(*occ_shape),
                            f'val_{i_iter_val}',
                            dataset=args.dataset)

                    if args.vis_gaussian_match:
                        vis_gaussian_occ_match(
                            save_dir,
                            result_dict['gaussian'],
                            gt_occ.reshape(*occ_shape),
                            f'val_{i_iter_val}',
                            dataset=args.dataset,
                            **draw_gaussian_params)

                    miou_metric._after_step(pred_occ, gt_occ)
                    if local_rank == 0 and allocation_vis_enabled:
                        allocation_collector.dump_frame(
                            os.path.join(save_dir, 'allocation_stats', 'per_frame'),
                            batch_idx=idx,
                            frame_name=f'val_{i_iter_val}')

                if i_iter_val % print_freq == 0 and local_rank == 0:
                    logger.info('[EVAL] Iter %5d'%(i_iter_val))
    finally:
        if local_rank == 0 and allocation_vis_enabled:
            allocation_collector.dump(os.path.join(save_dir, 'allocation_stats'))
        allocation_collector.close()

    miou, iou2 = miou_metric._after_epoch()
    logger.info(f'mIoU: {miou}, iou2: {iou2}')
    miou_metric.reset()
    
    if writer is not None:
        writer.close()


# ══════════════════════════════════════════════
# 流式评估可视化 (NuScenesFlowDataset)
# ══════════════════════════════════════════════

def main_stream(local_rank, args):
    """流式评估可视化主函数。

    使用 NuScenesFlowDataset + SceneStream 按场景顺序逐帧读取数据，
    支持 FIFO 时序融合可视化以及 Semantic Gaussian FIFO 融合可视化，
    并按场景组织可视化结果。
    """
    # 检查 FIFO 依赖
    if args.fifo and not _fifo_available:
        raise ImportError(
            '启用 FIFO 时序融合需要 fifo_eval.py，请确保该文件存在。'
        )
    if args.stream_gaussian_fusion and not _gaussian_fifo_available:
        raise ImportError(
            '启用 Gaussian FIFO 融合需要 fifo_eval.py 和 gaussian_encoder，请确保文件存在。'
        )

    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    # init DDP
    if args.gpus > 1:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "20507")
        hosts = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        gpus = torch.cuda.device_count()
        print(f"tcp://{ip}:{port}")
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}",
            world_size=hosts * gpus, rank=rank * gpus + local_rank)
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)

        if local_rank != 0:
            import builtins
            builtins.print = pass_print
    else:
        distributed = False
        world_size = 1

    writer = None
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    os.makedirs(args.work_dir, exist_ok=True)
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger('selfocc', log_file=log_file)
    MMLogger._instance_dict['selfocc'] = logger
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    import model
    from dataset import get_stream_dataloader

    my_model = build_segmentor(cfg.model)
    my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    if distributed:
        if cfg.get('syncBN', True):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            logger.info('converted sync bn.')

        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        raw_model = my_model.module
    else:
        my_model = my_model.cuda()
        raw_model = my_model
    logger.info('done ddp model')

    # ── 流式 Config 适配 ──
    cfg.val_dataset_config['type'] = 'NuScenesFlowDataset'
    cfg.val_dataset_config.pop('phase', None)

    # 构建流式 DataLoader
    val_dataset_loader = get_stream_dataloader(
        dataset_config=cfg.val_dataset_config,
        loader_config=cfg.val_loader,
        dist=distributed,
        shuffle_scenes=False,
    )

    # resume and load
    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from

    logger.info('resume from: ' + cfg.resume_from)
    logger.info('work dir: ' + args.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        try:
            raw_model.load_state_dict(ckpt.get('state_dict', ckpt), strict=True)
        except:
            os.system(f"python modify_weight.py --work-dir {args.work_dir} --epoch {args.epoch}")
            cfg.resume_from = os.path.join(args.work_dir, f"epoch_{args.epoch}_mod.pth")
            ckpt = torch.load(cfg.resume_from, map_location=map_location)
            raw_model.load_state_dict(ckpt['state_dict'], strict=True)
        print(f'successfully resumed.')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        try:
            print(raw_model.load_state_dict(state_dict, strict=False))
        except:
            from misc.checkpoint_util import refine_load_from_sd
            print(raw_model.load_state_dict(
                refine_load_from_sd(state_dict), strict=False))

    print_freq = cfg.print_freq

    # ── draw_gaussian_params ──
    if args.model_type == "base":
        draw_gaussian_params = dict(
            scalar=1.5,
            ignore_opa=False,
            filter_zsize=False,
        )
    elif args.model_type == "prob":
        draw_gaussian_params = dict(
            scalar=2.0,
            ignore_opa=True,
            filter_zsize=True,
        )
    draw_gaussian_params['adaptive_color'] = args.vis_gaussian_adaptive_color
    draw_gaussian_params['adaptive_color_seed'] = args.seed

    # ── FIFO 队列初始化 ──
    if args.fifo:
        fifo_queue = FIFOQueue(maxlen=args.temporal_windows)
        logger.info(
            f'[FIFO] 启用时序融合可视化: '
            f'windows={args.temporal_windows}, '
            f'alpha={args.fusion_alpha}'
        )
    else:
        fifo_queue = None

    # ── Gaussian FIFO 队列初始化 ──
    if args.stream_gaussian_fusion:
        gaussian_fifo_queue = GaussianFIFOQueue(maxlen=args.temporal_windows)
        logger.info(
            f'[GaussianFIFO] 启用高斯流式融合可视化: '
            f'windows={args.temporal_windows}'
        )
    else:
        gaussian_fifo_queue = None

    # ── 可视化根目录 ──
    stream_vis_root = osp.join(args.work_dir, f'stream_vis_ep{args.epoch}')
    os.makedirs(stream_vis_root, exist_ok=True)
    logger.info(f'Stream visualization root: {stream_vis_root}')

    # ── 指标 ──
    from misc.metric_util import MeanIoU
    CLASS_INDICES = list(range(1, 17))
    NUM_CLASSES = 17
    CLASS_NAMES = [
        'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
        'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
        'vegetation',
    ]
    global_miou = MeanIoU(
        CLASS_INDICES, NUM_CLASSES, CLASS_NAMES,
        True, NUM_CLASSES, filter_minmax=False)
    global_miou.reset()
    per_scene_mious = {}
    scene_results = {}
    scene_frame_counts = {}

    # ─── 流式评估 & 可视化主循环 ──────────────────────────────
    my_model.eval()
    os.environ['eval'] = 'true'

    current_scene = None
    total_frames = 0
    allocation_vis_enabled = (
        args.vis_gaussian_allocation_color or args.allocation_statistic)
    allocation_collector = AllocationOperationCollector(
        raw_model,
        enabled=(local_rank == 0 and allocation_vis_enabled),
        logger=logger)

    with torch.no_grad():
        for i_iter_val, data in enumerate(val_dataset_loader):

            # ── 场景元数据 ──
            scene_token = data['scene_token'][0]
            frame_in_scene = data.get('frame_index_in_scene', [0])[0]
            is_first = data.get('is_first_frame', [False])[0]
            is_last = data.get('is_last_frame', [False])[0]
            allocation_collector.start_iter(
                i_iter_val, f'{scene_token}_frame_{frame_in_scene:04d}')

            # ── 场景切换检测 ──
            if scene_token != current_scene:
                if current_scene is not None:
                    # 上一个场景收尾
                    scene_miou, scene_occiou = per_scene_mious[current_scene]._after_epoch()
                    scene_results[current_scene] = {
                        'mIoU': round(float(scene_miou), 2),
                        'occIoU': round(float(scene_occiou), 2),
                        'num_frames': scene_frame_counts[current_scene],
                    }
                    if local_rank == 0:
                        logger.info(
                            f'{"═" * 55}\n'
                            f' Scene {current_scene} completed: '
                            f'{scene_frame_counts[current_scene]} frames\n'
                            f'   mIoU   = {float(scene_miou):.2f}%\n'
                            f'   occIoU = {float(scene_occiou):.2f}%\n'
                            f'{"═" * 55}'
                        )

                current_scene = scene_token
                per_scene_mious[current_scene] = MeanIoU(
                    CLASS_INDICES, NUM_CLASSES, CLASS_NAMES,
                    True, NUM_CLASSES, filter_minmax=False,
                    name=current_scene)
                per_scene_mious[current_scene].reset()
                scene_frame_counts[current_scene] = 0

                # FIFO: 场景切换时清空队列
                if args.fifo and fifo_queue is not None:
                    fifo_queue.clear()
                    logger.info(f'[FIFO] 场景切换, FIFO 队列已清空')
                if args.stream_gaussian_fusion and gaussian_fifo_queue is not None:
                    gaussian_fifo_queue.clear()
                    logger.info(f'[GaussianFIFO] 场景切换, Gaussian FIFO 队列已清空')

                # 场景级可视化目录
                scene_vis_dir = osp.join(stream_vis_root, current_scene)
                os.makedirs(scene_vis_dir, exist_ok=True)

                if local_rank == 0:
                    logger.info(f'[STREAM VIS] Starting scene: {scene_token}')

            # ── 场景级可视化目录（已在上方创建）──
            scene_vis_dir = osp.join(stream_vis_root, current_scene)

            # ── 数据移至 GPU ──
            for k in list(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda()
                elif isinstance(data[k], np.ndarray):
                    data[k] = torch.from_numpy(data[k]).cuda()

            input_imgs = data.pop('img')
            ori_imgs = data.pop('ori_img', None)

            # ── 保存原始图片 ──
            if ori_imgs is not None and local_rank == 0:
                for i in range(ori_imgs.shape[-1]):
                    ori_img = ori_imgs[0, ..., i].cpu().numpy()
                    ori_img = ori_img[..., [2, 1, 0]]
                    ori_img = Image.fromarray(ori_img.astype(np.uint8))
                    ori_img.save(osp.join(
                        scene_vis_dir,
                        f'frame_{frame_in_scene:04d}_image_{i}.png'))

            # ── 模型推理 ──
            result_dict = my_model(imgs=input_imgs, metas=data)

            # ── 可视化 Gaussian GT ──
            if args.vis_gaussian_gt and local_rank == 0:
                gaussian_ctr = result_dict['gaussian'].means[0]
                gt_occ = result_dict['sampled_label'][0]
                from vis_open3d_voxel import get_grid_coords
                grids = get_grid_coords([200, 200, 16], [0.4, 0.4, 0.4])
                grids = torch.tensor(grids, device=gaussian_ctr.device) - torch.tensor(
                    [40, 40, 1], device=gaussian_ctr.device)
                occ_mask = gt_occ < 17
                gt_grid_occ = grids[occ_mask]
                cated_points = torch.cat([gaussian_ctr, gt_grid_occ], dim=0)
                color_gauss = torch.ones_like(
                    gaussian_ctr, dtype=torch.float32) * torch.tensor(
                    [1.0, 1.0, 1.0], device=gt_occ.device)
                color_gts = torch.ones_like(
                    gt_grid_occ, dtype=torch.float32) * torch.tensor(
                    [0, 1.0, 1.0], device=gt_occ.device)
                cated_colors = torch.cat([color_gauss, color_gts], dim=0)

                origin_points = data['lidar_points'][0][:, :3]
                origin_mask_x = (origin_points[:, 0] > -40) & (origin_points[:, 0] < 40)
                origin_mask_y = (origin_points[:, 1] > -40) & (origin_points[:, 1] < 40)
                origin_mask_z = (origin_points[:, 2] > -1) & (origin_points[:, 2] < 5.4)
                origin_mask = origin_mask_x & origin_mask_y & origin_mask_z
                filter_points = origin_points[origin_mask]

                cated_points = torch.cat([cated_points, filter_points], dim=0)
                color_points = torch.ones_like(
                    filter_points, dtype=torch.float32) * torch.tensor(
                    [1.0, 1.0, 0], device=gt_occ.device)
                cated_colors = torch.cat([cated_colors, color_points], dim=0)

                from open3d_vis_utils import draw_scenes
                draw_scenes(
                    points=cated_points.detach().cpu().numpy(),
                    point_colors=cated_colors.detach().cpu().numpy())

            # ── 时序融合 ──
            if args.fifo and fifo_queue is not None:
                soft_pred_batch = result_dict['pred_occ'][-1]  # (B, C, N)
                batch_size = soft_pred_batch.shape[0]

                fused_occ_list = []
                merged_gaussian_list = []
                for idx in range(batch_size):
                    soft_pred = soft_pred_batch[idx]           # (C, N)
                    curr_lidar2prev = data['lidar2prev'][idx]  # (4, 4)
                    curr_xyz = result_dict['sampled_xyz'][idx]  # (N, 3)

                    fused_soft_flat = fuse_predictions(
                        curr_soft_flat=soft_pred,
                        fifo_queue=fifo_queue,
                        alpha=args.fusion_alpha,
                        curr_lidar2prev=curr_lidar2prev,
                        sampled_xyz_curr=curr_xyz,
                        grid_params=_GRID_PARAMS,
                    )
                    fused_hard = fused_soft_to_hard(fused_soft_flat)
                    fused_occ_list.append(fused_hard)
                    merged_gaussian_list.append(None)

                    # 推入当前帧到 FIFO 队列
                    soft_pred_grid = soft_pred_to_grid(
                        soft_pred, _W, _H, _D)
                    fifo_queue.push(
                        soft_pred_grid=soft_pred_grid,
                        lidar2prev=curr_lidar2prev,
                        frame_idx=scene_frame_counts.get(current_scene, 0) + idx,
                        scene_token=scene_token,
                    )

                pred_occ_for_metric = fused_occ_list  # list of (N,)
            elif args.stream_gaussian_fusion and gaussian_fifo_queue is not None:
                # ── Gaussian FIFO 融合 ──
                gaussian_batch = result_dict['gaussian']
                batch_size = gaussian_batch.means.shape[0]
                fused_occ_list = []
                merged_gaussian_list = []
                for idx in range(batch_size):
                    curr_g = GaussianPrediction(
                        means=gaussian_batch.means[idx:idx+1],
                        scales=gaussian_batch.scales[idx:idx+1],
                        rotations=gaussian_batch.rotations[idx:idx+1],
                        opacities=gaussian_batch.opacities[idx:idx+1],
                        semantics=gaussian_batch.semantics[idx:idx+1],
                        original_means=(
                            gaussian_batch.original_means[idx:idx+1]
                            if gaussian_batch.original_means is not None else None),
                        delta_means=(
                            gaussian_batch.delta_means[idx:idx+1]
                            if gaussian_batch.delta_means is not None else None),
                    )
                    fused_hard, merged_g = gaussian_fifo_fuse_and_render(
                        curr_g, gaussian_fifo_queue,
                        data['lidar2prev'][idx],
                        result_dict['sampled_xyz'][idx:idx+1],
                        raw_model.head,
                        _GRID_PARAMS,
                    )
                    fused_occ_list.append(fused_hard)
                    merged_gaussian_list.append(merged_g)

                    # 推入当前帧到 Gaussian FIFO 队列（CPU 存储）
                    gaussian_fifo_queue.push(
                        gaussian=GaussianPrediction(
                            means=gaussian_batch.means[idx:idx+1].detach().cpu(),
                            scales=gaussian_batch.scales[idx:idx+1].detach().cpu(),
                            rotations=gaussian_batch.rotations[idx:idx+1].detach().cpu(),
                            opacities=gaussian_batch.opacities[idx:idx+1].detach().cpu(),
                            semantics=gaussian_batch.semantics[idx:idx+1].detach().cpu(),
                            original_means=(
                                gaussian_batch.original_means[idx:idx+1].detach().cpu()
                                if gaussian_batch.original_means is not None else None),
                            delta_means=(
                                gaussian_batch.delta_means[idx:idx+1].detach().cpu()
                                if gaussian_batch.delta_means is not None else None),
                        ),
                        lidar2prev=data['lidar2prev'][idx],
                        frame_idx=scene_frame_counts.get(current_scene, 0) + idx,
                        scene_token=scene_token,
                    )
                pred_occ_for_metric = fused_occ_list
            else:
                batch_size = result_dict['final_occ'].shape[0]
                pred_occ_for_metric = [
                    result_dict['final_occ'][idx] for idx in range(batch_size)
                ]
                merged_gaussian_list = [None] * batch_size

            # ── 帧计数 ──
            scene_frame_counts[current_scene] += batch_size
            total_frames += batch_size

            # ── 指标累积与可视化 ──
            for idx in range(batch_size):
                pred_occ = pred_occ_for_metric[idx]
                gt_occ = result_dict['sampled_label'][idx]
                occ_shape = [_H, _W, _D]
                alloc_draw_params = allocation_collector.get_draw_params(
                    draw_gaussian_params, batch_idx=idx,
                    enable_color=args.vis_gaussian_allocation_color)

                # 指标
                occ_mask = result_dict.get(
                    'occ_cam_mask',
                    torch.ones_like(gt_occ, dtype=torch.bool))[idx].flatten()
                global_miou._after_step(pred_occ, gt_occ, occ_mask)
                per_scene_mious[current_scene]._after_step(
                    pred_occ, gt_occ, occ_mask)

                # ── 可视化（仅 rank 0）──
                if local_rank != 0:
                    continue

                # 命名前缀：包含场景和帧索引
                frame_tag = f'frame_{frame_in_scene:04d}'

                # ── 预测错误分类可视化 ──
                if args.vis_occ_error:
                    save_occ_error(
                        scene_vis_dir,
                        pred_occ.reshape(*occ_shape),
                        gt_occ.reshape(*occ_shape),
                        frame_tag,
                        dataset=args.dataset)

                # 俯视图 Gaussian
                if args.vis_gaussian_topdown:
                    save_gaussian_topdown(
                        scene_vis_dir,
                        result_dict['anchor_init'],
                        result_dict['gaussians'],
                        f'{frame_tag}_topdown',
                    )

                # 占用预测 / GT
                if args.vis_occ:
                    save_occ(
                        scene_vis_dir,
                        pred_occ.reshape(1, *occ_shape),
                        f'{frame_tag}_pred',
                        True, 0, dataset=args.dataset)
                    save_occ(
                        scene_vis_dir,
                        gt_occ.reshape(1, *occ_shape),
                        f'{frame_tag}_gt',
                        True, 0, dataset=args.dataset)

                # Gaussian 分布
                if args.vis_gaussian:
                    save_gaussian(
                        scene_vis_dir,
                        result_dict['gaussian'],
                        f'{frame_tag}_gaussian',
                        **alloc_draw_params)
                if args.vis_gaussian_point:
                    if save_gaussian_point is None:
                        raise ImportError('save_gaussian_point 需要 vis_open3d_voxel.py / Open3D 可用')
                    save_gaussian_point(
                        scene_vis_dir,
                        result_dict['gaussian'],
                        f'{frame_tag}_gaussian',
                        **alloc_draw_params)

                # Gaussian FIFO 融合后的高斯可视化
                if args.vis_gaussian and merged_gaussian_list[idx] is not None:
                    save_gaussian(
                        scene_vis_dir,
                        merged_gaussian_list[idx],
                        f'{frame_tag}_gaussian_fused',
                        **draw_gaussian_params)

                # Gaussian + GT Occupancy 叠加可视化
                if args.vis_gaussian_occ:
                    save_gaussian_with_gt_occ(
                        scene_vis_dir,
                        result_dict['gaussian'],
                        gt_occ.reshape(*occ_shape),
                        f'{frame_tag}',
                        dataset=args.dataset,
                        **draw_gaussian_params)

                # 高斯球与GT Occupancy几何匹配可视化
                if args.vis_gaussian_match:
                    vis_gaussian_occ_match(
                        scene_vis_dir,
                        result_dict['gaussian'],
                        gt_occ.reshape(*occ_shape),
                        f'{frame_tag}',
                        dataset=args.dataset,
                        **draw_gaussian_params)

                # 每阶段 Gaussian
                if args.vis_gaussian_each_stage:
                    for stage_i, gaussian in enumerate(result_dict['gaussians']):
                        save_gaussian(
                            scene_vis_dir,
                            gaussian,
                            f'{frame_tag}_gaussian_stage{stage_i}',
                            **alloc_draw_params)

                if allocation_vis_enabled:
                    allocation_collector.dump_frame(
                        osp.join(stream_vis_root, 'allocation_stats', 'per_frame'),
                        batch_idx=idx,
                        frame_name=f'{scene_token}_{frame_tag}')

            # ── 日志 ──
            if i_iter_val % print_freq == 0 and local_rank == 0:
                logger.info(
                    f'[STREAM VIS] Iter {i_iter_val:6d} | '
                    f'Scene: {scene_token} | '
                    f'Frame: {frame_in_scene:04d}/{data.get("num_frames_in_scene", ["?"])[0]} | '
                    f'Total in scene: {scene_frame_counts[current_scene]}'
                )

    # ──────────────────────────────────────────────
    # 最后一个场景收尾
    # ──────────────────────────────────────────────
    if current_scene is not None:
        scene_miou, scene_occiou = per_scene_mious[current_scene]._after_epoch()
        scene_results[current_scene] = {
            'mIoU': round(float(scene_miou), 2),
            'occIoU': round(float(scene_occiou), 2),
            'num_frames': scene_frame_counts[current_scene],
        }
        if local_rank == 0:
            logger.info(
                f'{"═" * 55}\n'
                f' Scene {current_scene} completed: '
                f'{scene_frame_counts[current_scene]} frames\n'
                f'   mIoU   = {float(scene_miou):.2f}%\n'
                f'   occIoU = {float(scene_occiou):.2f}%\n'
                f'{"═" * 55}'
            )

    # ── 结果汇总 ──
    if distributed:
        all_scene_results = [None] * world_size
        dist.all_gather_object(all_scene_results, scene_results)
        if local_rank == 0:
            merged_scene_results = {}
            for rank_results in all_scene_results:
                merged_scene_results.update(rank_results)
            scene_results = merged_scene_results

    if local_rank == 0:
        global_miou_result, global_occiou_result = global_miou._after_epoch()
        global_miou_result = float(global_miou_result)
        global_occiou_result = float(global_occiou_result)

        scene_miou_list = [r['mIoU'] for r in scene_results.values()]
        scene_occiou_list = [r['occIoU'] for r in scene_results.values()]

        mean_scene_miou = float(np.mean(scene_miou_list)) if scene_miou_list else 0.0
        std_scene_miou = float(np.std(scene_miou_list)) if scene_miou_list else 0.0
        mean_scene_occiou = float(np.mean(scene_occiou_list)) if scene_occiou_list else 0.0

        logger.info('')
        logger.info(f'{"═" * 55}')
        logger.info(' Streaming Visualization Results')
        logger.info(f'{"═" * 55}')
        logger.info(f' Total scenes     : {len(scene_results)}')
        logger.info(f' Total frames     : {total_frames}')
        logger.info(f'{"─" * 55}')
        logger.info(f' Global mIoU      : {global_miou_result:.2f}%')
        logger.info(f' Global occIoU    : {global_occiou_result:.2f}%')
        logger.info(f'{"─" * 55}')
        logger.info(f' Per-scene mIoU   : {mean_scene_miou:.2f}% ± {std_scene_miou:.2f}%')
        logger.info(f' Per-scene occIoU : {mean_scene_occiou:.2f}%')
        logger.info(f'{"─" * 55}')
        logger.info(f' Per-scene details:')
        for sc_token in sorted(scene_results.keys()):
            r = scene_results[sc_token]
            logger.info(
                f'   {sc_token:20s}: '
                f'mIoU={r["mIoU"]:6.2f}%  '
                f'occIoU={r["occIoU"]:6.2f}%  '
                f'frames={r["num_frames"]:4d}'
            )
        logger.info(f'{"═" * 55}')

        # 保存结果
        eval_results = {
            'config': args.py_config,
            'work_dir': args.work_dir,
            'global_miou': round(global_miou_result, 2),
            'global_occiou': round(global_occiou_result, 2),
            'per_scene': scene_results,
            'per_scene_stats': {
                'mean_miou': round(mean_scene_miou, 2),
                'std_miou': round(std_scene_miou, 2),
                'mean_occiou': round(mean_scene_occiou, 2),
            },
            'num_scenes': len(scene_results),
            'total_frames': total_frames,
            'fifo_enabled': args.fifo,
        }
        save_path = osp.join(args.work_dir, 'stream_vis_results.json')
        with open(save_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        logger.info(f'Stream vis results saved to: {save_path}')

    if local_rank == 0 and allocation_vis_enabled:
        allocation_collector.dump(osp.join(stream_vis_root, 'allocation_stats'))
    allocation_collector.close()

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vis-occ', action='store_true', default=False)
    parser.add_argument('--vis-gaussian', action='store_true', default=False)
    parser.add_argument('--vis-gaussian-point', action='store_true', default=False,
                        help='使用点云形式可视化 Semantic Gaussian')
    parser.add_argument('--vis-gaussian-adaptive-color', action='store_true', default=False,
                        help='根据 Gaussian opacity 自适应选择红/蓝/灰颜色')
    parser.add_argument('--vis-gaussian-allocation-color', action='store_true', default=False,
                        help='根据 AdaptiveAllocationV5 的 pass-through/clone/split/attenuation 候选类型着色')
    parser.add_argument('--allocation-statistic', action='store_true', default=False,
                        help='统计 AdaptiveAllocationV5 的 TopK risky bank 与候选输出分布')
    parser.add_argument('--vis_gaussian_topdown', action='store_true', default=False)
    parser.add_argument('--vis-index', type=int, nargs='+', default=[])
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--vis_scene_index', type=int, default=-1)
    parser.add_argument('--vis-scene', action='store_true', default=False)
    parser.add_argument('--vis-gaussian-each-stage', action='store_true', default=False)
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='nusc')
    parser.add_argument('--model-type', type=str, default="base", choices=["base", "prob"])
    parser.add_argument('--vis-gaussian-gt', action='store_true', default=False)
    parser.add_argument('--vis-occ-error', action='store_true', default=False,
                        help='可视化预测错误分类: 绿色(正确) / 深红(类别错) / 黑色(假阳性)')
    parser.add_argument('--vis-gaussian-occ', action='store_true', default=False,
                        help='同时可视化 Semantic Gaussian 和 GT Occupancy（GT 灰色背景）')
    parser.add_argument('--vis-gaussian-match', action='store_true', default=False,
                        help='可视化高斯球与GT Occupancy的几何匹配程度（仅展示包裹GT网格的高斯球，按语义匹配着色）')
    # 流式可视化参数
    parser.add_argument('--stream', action='store_true', default=False,
                        help='启用流式可视化模式 (使用 NuScenesFlowDataset)')
    parser.add_argument('--fifo', action='store_true', default=False,
                        help='启用 FIFO 时序融合可视化')
    parser.add_argument('--temporal-windows', type=int, default=3,
                        help='FIFO 队列长度（时序窗口大小）')
    parser.add_argument('--fusion-alpha', type=float, default=0.7,
                        help='时序融合权重 α: P_fused = P_curr * α + P_history * (1-α)')
    # Gaussian FIFO 流式融合可视化
    parser.add_argument('--stream-gaussian-fusion', action='store_true', default=False,
                        help='启用基于 Semantic Gaussian 的 FIFO 流式融合可视化')
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    # 自动模式选择：
    #   - 显式指定 --stream → main_stream
    #   - 未指定 --stream 但指定了 --fifo → 自动启用流式模式（因为 FIFO 依赖 NuScenesFlowDataset）
    #   - 未指定 --stream 但指定了 --stream-gaussian-fusion → 自动启用流式模式
    #   - 其他情况 → 传统 eval 可视化 main
    if args.stream or args.fifo or args.stream_gaussian_fusion:
        if not args.stream:
            print('[INFO] --fifo/--stream-gaussian-fusion 已启用，自动切换到流式可视化模式 (--stream)')
        args.stream = True
        entry_func = main_stream
    else:
        entry_func = main

    if ngpus > 1:
        torch.multiprocessing.spawn(entry_func, args=(args,), nprocs=args.gpus)
    else:
        entry_func(0, args)
