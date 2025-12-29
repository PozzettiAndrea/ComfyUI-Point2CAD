----------
Work in Progress! This node is not finished.
----------

# ComfyUI-Point2CAD

Point cloud to CAD reconstruction using neural network based segmentation.

**Originally from [ComfyUI-CADabra](https://github.com/PozzettiAndrea/ComfyUI-CADabra)**

## Paper

**Point2CAD: Reverse Engineering CAD Models from 3D Point Clouds** (CVPR 2024)

- Project: https://www.obukhov.ai/point2cad.html
- GitHub: https://github.com/prs-eth/point2cad

## Installation

### Via ComfyUI Manager
Search for "Point2CAD" in ComfyUI Manager

### Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/PozzettiAndrea/ComfyUI-Point2CAD
pip install -r ComfyUI-Point2CAD/requirements.txt
```

## Nodes

- **LoadPoint2CADModel** - Download/load segmentation network (ParseNet or HPNet)
- **Point2CADSegmentation** - Segment point cloud into surface clusters
- **Point2CADSurfaceFitting** - Fit primitives to segments (plane, sphere, cylinder, cone)
- **Point2CADSurfaceFittingOCC** - Surface fitting with OpenCASCADE output
- **Point2CADTopologyExtraction** - Extract edges and corners via intersection
- **Point2CADExportBrep** - Export to STEP format
- **SaveSurfaceParams** / **LoadSurfaceParams** - Save/load fitted surface parameters
- **Point2CADToWireframeInfo** - Extract wireframe information

## Requirements

- torch>=2.0.0
- numpy>=1.24.0
- trimesh>=3.20.0
- open3d>=0.17.0
- scipy>=1.11.0
- geomdl>=5.3.0 (B-spline operations)

## Community

Questions or feature requests? Open a [Discussion](https://github.com/PozzettiAndrea/ComfyUI-Point2CAD/discussions) on GitHub.

Join the [Comfy3D Discord](https://discord.gg/PN743tE5) for help, updates, and chat about 3D workflows in ComfyUI.

## Credits

- Original CADabra: [PozzettiAndrea/ComfyUI-CADabra](https://github.com/PozzettiAndrea/ComfyUI-CADabra)
- Point2CAD paper authors

## License

GPL-3.0
