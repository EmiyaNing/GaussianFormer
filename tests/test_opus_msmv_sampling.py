import torch

from model.ops.opus_msmv_sampling import wrapper


def test_msmv_c2345_backward_unpacks_extension_gradient_list(monkeypatch):
    """The pybind vector result must be expanded for autograd.Function."""
    tensors = tuple(torch.rand(1, 1, 1, 1, 1) for _ in range(6))

    class Context:
        saved_tensors = tensors

    captured = {}

    def fake_backward(*args):
        captured['args'] = args
        return [torch.zeros_like(tensor) for tensor in tensors]

    monkeypatch.setattr(wrapper, '_ms_deform_attn_cuda_c2345_backward', fake_backward)
    gradients = wrapper._MSMVSamplingC2345.backward(Context(), torch.ones(1))

    assert isinstance(gradients, tuple)
    assert len(gradients) == 6
    assert len(captured['args']) == 7

