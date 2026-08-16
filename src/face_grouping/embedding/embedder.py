"""IR-SE50 embedding inference wrapper.

Turns a 112x112 aligned RGB face crop into a 512-D L2-normalized
embedding ready for cosine-similarity matching.
"""
import numpy as np
import torch

from face_grouping.embedding.irse50 import IRSE50

# Preprocessing used by the trained IR-SE50 checkpoint. A mismatch here
# silently degrades embeddings, so keep it tied to the model artifact.
_MEAN = 0.5
_STD = 0.5


class EmbedderWrapper:
    """Load the production IR-SE50 model and produce face embeddings."""

    def __init__(self, weights_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = IRSE50()
        # weights_only=False is required for this known legacy checkpoint
        # format on PyTorch 2.6+. Do not load untrusted checkpoint files.
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, aligned_crop_rgb: np.ndarray) -> torch.Tensor:
        """Convert a (112, 112, 3) uint8 RGB crop to model input."""
        img = aligned_crop_rgb.astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def embed(self, aligned_crop_rgb: np.ndarray) -> np.ndarray:
        """Return one (512,) L2-normalized embedding."""
        output = self.model(self._preprocess(aligned_crop_rgb))
        return output.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def embed_batch(self, aligned_crops_rgb: list) -> np.ndarray:
        """Return an (N, 512) array for a list of aligned crops."""
        if not aligned_crops_rgb:
            return np.empty((0, 512), dtype=np.float32)
        batch = torch.cat([self._preprocess(crop) for crop in aligned_crops_rgb], dim=0)
        return self.model(batch).cpu().numpy()
