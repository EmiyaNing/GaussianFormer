"""A2: renderer responsibility targets; all other settings identical to A1."""
_base_ = ['./a1_softmax.py']
loss = dict(assignment='render', trace_dir='work_dirs/gaussian_responsibility/train/A2')
