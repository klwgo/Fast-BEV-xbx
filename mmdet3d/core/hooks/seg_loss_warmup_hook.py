from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class SegLossWarmupHook(Hook):
    """在训练早期平滑提升 BEV seg head 的类别权重."""

    def __init__(self,
                 module_attr='seg_head',
                 warmup_iters=2000,
                 start_scale=0.3,
                 end_scale=1.0):
        assert 0 <= start_scale <= end_scale
        self.module_attr = module_attr.split('.')
        self.warmup_iters = max(1, warmup_iters)
        self.start_scale = start_scale
        self.end_scale = end_scale

    def before_train_iter(self, runner):
        scale = self._compute_scale(runner.iter)
        seg_head = self._get_module(runner.model)
        if seg_head is not None and hasattr(seg_head, 'set_class_weight_scale'):
            seg_head.set_class_weight_scale(scale)

    def _compute_scale(self, current_iter: int) -> float:
        if current_iter >= self.warmup_iters:
            return self.end_scale
        progress = current_iter / float(self.warmup_iters)
        return self.start_scale + (self.end_scale - self.start_scale) * progress

    def _get_module(self, model):
        if hasattr(model, 'module'):
            model = model.module
        target = model
        for attr in self.module_attr:
            if not hasattr(target, attr):
                return None
            target = getattr(target, attr)
        return target
