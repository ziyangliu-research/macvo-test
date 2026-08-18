from __future__ import annotations

from typing import Any

import torchvision.transforms.functional as TF

from .contracts import Observation, StereoFrameInput
from .resplat_runtime import process_image_and_intrinsic, process_pil_image_and_intrinsic


def make_resplat_domain_observation(
    packet_generator: Any,
    frame_input: StereoFrameInput,
) -> Observation:
    """Build the backend RGB/K in exactly the ReSplat resize/crop camera domain.

    ReSplat may resize and center-crop a source frame before inference. GraphDECO
    needs pixel-space intrinsics rather than ReSplat's optionally normalized K,
    so this function deliberately reuses the same image transformation with
    ``normalize_intrinsics=False``.

    The raw StereoFrame remains untouched and can still be consumed by MAC-VO.
    """

    packet_generator.initialize()
    frame_input.validate(deep=packet_generator.config.strict_validation)

    if packet_generator.config.input_mode == "file_paths":
        image, K_pixel = process_image_and_intrinsic(
            frame_input.descriptor.left_path,
            packet_generator.K_pixel,
            packet_generator.image_shape,
            False,
        )
    else:
        left_pil = TF.to_pil_image(
            frame_input.left_image.detach().cpu().clamp(0, 1)
        )
        image, K_pixel = process_pil_image_and_intrinsic(
            left_pil,
            packet_generator.K_pixel,
            packet_generator.image_shape,
            False,
        )

    observation = Observation(
        descriptor=frame_input.descriptor,
        image=image.contiguous(),
        intrinsic_pixel=K_pixel.contiguous(),
    )
    observation.validate(deep=packet_generator.config.strict_validation)
    return observation
