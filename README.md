#### *¿Te gustan las Gambas? ¡Esto es FAMBAS! 🦐⚽*

# FAMBAS (Football mAMba for Ball Action Spotting)

FAMBAS is a video understanding model for Football Action Spotting. Detect and analyze ball actions in football (soccer) matches using Mamba architecture. This project is a part of [SoccerNet Challenge 2025](https://www.soccer-net.org/challenges/2025).

![Architecture](imgs/FAMBAS_screenshot.png)

## Installation

```bash
# Clone the repository
git clone https://github.com/ho0-kim/FAMBAS.git

# Navigate to project directory
cd FAMBAS

# Install dependencies
pip install -r requirements.txt

# install mamba
cd causal-conv1d
python setup.py develop
cd ..
cd mamba
python setup.py develop
cd ..
```

## Usage

### 1. Dataset Preparation

TBD

### 2. Training

TBD

## Acknowledgments

This repository is heavily based on:
- [SN-TeamSpotting (T-DEED)](https://github.com/SoccerNet/sn-teamspotting): For ball action spotting
- [Video-Mamba-Suite](https://github.com/OpenGVLab/video-mamba-suite): For the MAMBA architecture implementation in video analysis

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/ho0-kim/FAMBAS/blob/main/LICENSE) file for details.
