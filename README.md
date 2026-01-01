# PowerGS: Display-Rendering Power Co-Optimization for Foveated Radiance-Field Rendering in Power-Constrained XR Systems ⚡️👓

## 📄 [Paper](https://arxiv.org/abs/2509.21702) 🌐 [Project page](https://powergs.netlify.app/)

This is the official implementation of **PowerGS**.

## 🚀 Getting Started

### Step 1: Clone the Repository
Clone this repository recursively to include all submodules:

```bash
git clone --recursive https://github.com/horizon-research/PowerGS.git
cd PowerGS
```

### Step 2: Environment Setup
We recommend using a virtual environment. You can use the provided `install.sh` script to install the necessary dependencies:

```bash
# Create a virtual environment (optional but recommended)
python3 -m venv env
source env/bin/activate

# Install dependencies
bash install.sh
```

*Note: Ensure you have CUDA installed as the submodules require it for compilation.*

### Step 3: Fit a Scene
To fit a scene, use the `scripts/fit_a_scene.sh` script. You need to provide the scene name, required quality, and a port for training:

```bash
bash scripts/fit_a_scene.sh <scene_name> <q_required> <port>
```

Example:
```bash
bash scripts/fit_a_scene.sh bicycle 0.99 6009
```

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

