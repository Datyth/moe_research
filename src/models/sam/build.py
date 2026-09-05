"""Builder for the adapter-free, promptless SAM ViT-B baseline."""

from __future__ import annotations

from ..esam._vendor.build import build_sam_vit_b
from ..esam._vendor.sam_moe import Sam_my


class _UpstreamArgs:
    """Minimal namespace retained by the vendored SAM components."""

    batch_size = 1


def build_promptless_sam_vit_b(
    *,
    image_size: int,
    num_classes: int,
    checkpoint: str | None,
) -> Sam_my:
    """Build plain SAM ViT-B with empty prompts and a task-specific decoder.

    The shared builder owns checkpoint compatibility and positional-embedding
    resizing. Disabling all three E-SAM additions leaves the original SAM
    transformer blocks, an empty sparse prompt, and SAM's learned dense
    ``no_mask_embed`` prompt.
    """

    return build_sam_vit_b(
        args=_UpstreamArgs(),
        image_size=image_size,
        num_classes=num_classes,
        checkpoint=checkpoint,
        moe_top_k_ratio=0.5,
        moe_num_experts=1,
        use_moe=False,
        use_lpeg=False,
        use_adapters=False,
    )
