# SPDX-License-Identifier: GPL-3.0-or-later
"""
ComfyUI-Point2CAD - Point Cloud to CAD Reconstruction

Originally from ComfyUI-CADabra: https://github.com/PozzettiAndrea/ComfyUI-CADabra

Paper: "Point2CAD: Reverse Engineering CAD Models from 3D Point Clouds" (CVPR 2024)
Project: https://www.obukhov.ai/point2cad.html
"""

import sys

# Only run initialization when loaded by ComfyUI, not during pytest
if 'pytest' not in sys.modules:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
