import torch
import numpy as np
from pyhessian import hessian
from collections.abc import Mapping

def compute_hessian_metrics(model, data_source, device="cuda"):
    """
    Computes the top eigenvalue and trace of the model's loss landscape.

    ``data_source`` may be a DataLoader, a single batch dict, or a
    one‑element list containing a batch dict.  The helper wrappers below
    adapt HuggingFace models (which expect keyword args) to the tensor‑only
    interface that :class:`pyhessian` assumes.
    """
    model.eval()

    # ------------------------------------------------------------------
    # Helpers for pyhessian compatibility
    # ------------------------------------------------------------------
    class _BatchWrapper:
        """Wrap a dict so that calling ``.cuda()`` moves every tensor."""
        def __init__(self, data_dict):
            self.data = data_dict
        def cuda(self):
            return _BatchWrapper({k: v.cuda() for k, v in self.data.items()})

    class _HFWrapper(torch.nn.Module):
        """Make ``model(batch)`` behave like ``model(**batch)`` when needed."""
        def __init__(self, base_model):
            super().__init__()
            self.base = base_model
        def forward(self, batch):
            if isinstance(batch, _BatchWrapper):
                return self.base(**batch.data)
            elif isinstance(batch, dict):
                return self.base(**batch)
            else:
                return self.base(batch)

    def criterion(outputs, labels):
        return outputs.loss

    # extract a single batch regardless of input type
    if isinstance(data_source, Mapping):
        # covers dict, BatchEncoding, etc.
        batch = data_source
    elif isinstance(data_source, list):
        batch = data_source[0]
    else:
        batch = next(iter(data_source))

    inputs = {k: v.to(device) for k, v in batch.items()}

    wrapped_model = _HFWrapper(model)
    hessian_comp = hessian(
        wrapped_model,
        criterion,
        data=(_BatchWrapper(inputs), inputs["labels"]),
        cuda=(device != "cpu"),
    )

    # compute metrics, guarding against operations that don't support
    # second derivatives (e.g. flash attention kernels)
    try:
        top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=1)
    except Exception as exc:  # pragma: no cover
        print(f"[curvature] eigenvalue computation failed: {exc}")
        top_eigenvalues = [float('nan')]

    try:
        trace = hessian_comp.trace()
    except Exception as exc:  # pragma: no cover
        print(f"[curvature] trace computation failed: {exc}")
        trace = float('nan')

    return top_eigenvalues[0], np.mean(trace)