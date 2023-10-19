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

## Running the new method run_queue.bash
This repository creates a new Nerfstudio method named "nqp-{method_name}". To train with it, run the command:
```
ns-train nqp-{method_name} --data [PATH] --viewer.quit-on-train-completion [true/false] [data parser]

Example
ns-train nqp-tensorf-half-res --data [PATH] --viewer.quit-on-train-completion True nqp-blender-data
```

## Evaluating the method eval.bash 
```
ns-eval --load-config "DATA_PATH/config.yml" --render-output-path "DATA_PATH/renders" --output-path "DATA_PATH/results.json"
```
## Creating new method and changing parameters 
1. Write new method into method_configs. Follow the exact same format as the rest just change the method name and parameters
  
2. Choose which parameters to change e.g these are the default values you can lower them if you want 
Final_res: 300 //change size and quality
init_res : 128 //change size and quality
Num_color:48 //change size and quality
Num_den:16 //changes size and quality
Num_samples:50 //change quality 
num_uniform_samples:200 //change quality

3.After save that save the file

4.Open up pyproject.toml and add your method name into the list of methods:
```
method_name = 'nqp.configs.method_configs: method_name'
Example: 
nqp-tensorf-half-res = 'nqp.configs.method_configs:nqp-tensorf-half-res'
```

5.Save that file and run the commands in the bash terminal these commands:
```
ns-train --help  //use this to check if your method has been added to the list, if it is not run the commands below 
ns-install-cli   //to add your new method to configs 
pip install -e . //if it still doesn't work use this
```
6.Now you can run the new method using the command stated above

## Files to check 
run_queue.bash : Has script to run the models
eval.bash : Has script to eval the models (will create config file + render model images + evaluation metrics in json)
nqp/configs/method_configs.py : contains all definitions of each method to run (this is where you will write the method and change the parameters) 
