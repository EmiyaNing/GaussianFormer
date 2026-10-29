"""A3: sparse KL-unbalanced transport, same candidate graph and CE as A2."""
_base_ = ['./a1_softmax.py']
loss = dict(assignment='transport', trace_dir='work_dirs/gaussian_responsibility/train/A3')
