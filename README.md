<p align="center">
  <img src="demos/graph_abstract_v3_SDU.png" alt="OpenCap Monocular graphical abstract" width="100%">
</p>

<p align="center">
<a href="https://utahmobl.github.io/OpenCap-monocular-project-page" target="_blank"><img src="https://img.shields.io/badge/Project_Page-View-9B72F0?logo=book&logoColor=white" alt="Project page"></a>
<a href="https://arxiv.org/abs/2603.24733" title="Paper">
  <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg?logo=arxiv&logoColor=white" alt="Paper">
</a>
<a href="https://www.opencap.ai" title="Get Started"><img src="https://img.shields.io/badge/Get_Started-Try_OpenCap-00adff?logo=rocket&logoColor=white" alt="Get Started"></a>
</p>

# OpenCap Monocular

**3D Human Kinematics and Dynamics From a Single Smartphone Video**

OpenCap-Monocular estimates 3D human movement kinematics and musculoskeletal kinetics from a single smartphone video, combining pose estimation, camera optimization, and biomechanical modeling.

## Features

- Single-camera processing
- 3D pose estimation (WHAM)
- Camera and pose optimization
- OpenSim integration
- Activity classification (walking, sit-to-stand, squats, …)

---

## Installation

### Bare-metal conda

Requires Ubuntu 20.04/22.04, Python 3.9, NVIDIA driver ≥ 520.

```bash
git clone https://github.com/utahmobl/opencap-monocular.git --recursive
cd opencap-monocular
```

See **[installation/INSTALL_SLIM.md](installation/INSTALL_SLIM.md)** for step-by-step instructions.

---

## Pipeline

1. Video preprocessing & rotation correction
2. WHAM 3D pose estimation
3. Camera extrinsics & pose optimization
4. OpenSim IK and export
5. Visualization (`mono.json` for [OpenCap Visualizer](https://visualizer.opencap.ai))

## Outputs

- `mono.json` — OpenCap viewer
- `*.trc`, `*.mot` — OpenSim formats
- `*_scaled.osim` — Scaled model

## Acknowledgments

[WHAM](https://github.com/yohanshin/WHAM) · [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) · [SLAHMR](https://github.com/vye16/slahmr) · [OpenCap](https://opencap.ai) · [OpenSim](https://opensim.stanford.edu) · [VideoLLaMA3](https://github.com/DAMO-NLP-SG/VideoLLaMA3)

## License

This project is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE) — **non-commercial use only**.

Commercial use requires a separate agreement. Contact the authors for inquiries.

### Third-Party Licenses

This project depends on third-party components with their own licenses. Some impose additional restrictions:

| Component | License | Commercial Use |
|---|---|---|
| [WHAM](https://github.com/yohanshin/WHAM) | MIT | Allowed |
| [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) | Apache 2.0 | Allowed |
| [DPVO](https://github.com/princeton-vl/DPVO) | MIT | Allowed |
| [SMPL / SMPL-X body models](https://smpl.is.tue.mpg.de/) | Custom (MPI) | **Not allowed** |

The SMPL body models required by this pipeline are licensed by the Max Planck Institute for non-commercial scientific research only. Users must agree to the [SMPL license](https://smpl.is.tue.mpg.de/modellicense) before downloading the model files. For commercial SMPL licensing, contact ps-license@tue.mpg.de.
