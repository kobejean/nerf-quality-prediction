# nerf-quality-prediction
Template repository for creating and registering methods in Nerfstudio.

## Registering with Nerfstudio
Ensure that nerfstudio has been installed according to the [instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html). Clone or fork this repository and run the commands:

```
conda activate nerfstudio
cd nerf-quality-prediction/
pip install -e .
pip install -U pyopenssl cryptography
ns-install-cli
```

## Running the new method
This repository creates a new Nerfstudio method named "nqp-{method_name}". To train with it, run the command:
```
ns-train nqp-{method_name} --data [PATH]
```