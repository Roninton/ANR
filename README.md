# Project Name

Released code for ECCV 2024 paper: Attention Beats Linear for Fast Implicit Neural Representation Generation.
Arxiv paper link: https://arxiv.org/abs/2407.15355

## Table of Contents

- [Project Name](#project-name)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Steps](#steps)
  - [Usage](#usage)
    - [Configuration Parameters](#configuration-parameters)
    - [Example Run](#example-run)
  - [License](#license)
  - [Contact](#contact)

## Installation

### Prerequisites
- Python version 3.8+
- Operating System: Windows/Linux/macOS

### Steps

1. Clone or download the project to your local machine.
2. Navigate to the project directory:

```bash
cd path/to/your/project
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. download datasets

Download any image dataset or nerf dataset you want, 
for example:
+ CelebA: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
+ ShapeNet: https://paperswithcode.com/paper/shapenet-an-information-rich-3d-model

If you want to use nerf dataset, we expect the folder structure is like:

```
|--dataset_root
|  |--subset_1.json
|  |--subset_1
|  |  |--id_of_item1
|  |  |  |--transforms.json
|  |  |  |--r_0.png
|  |  |  |--r_1.png
|  |  |  |--   ...
|  |  |  |--r_n1.png
|  |  |--id_of_item2
|  |     |--transforms.json
|  |     |--r_0.png
|  |     |--r_1.png
|  |     |--   ...
|  |     |--r_n2.png
|  |--subset_2.json
|  |--subset_2
|     |--id_of_item1
|     |  |--transforms.json
|     |  |--r_0.png
|     |  |--   ...
|     |  |--r_m1.png
|     |--id_of_item2
|     ...
|
```


## Usage

### Configuration Parameters
We give the example configs in `./configs`, and here's some important parameters:
+ `milestone`: The path to an trained model, set to empty if it's a new experiment.
+ `model_comment`: The training outputs will be saved at `./output/{model_comment}`.
+ `image_shape`: An int or list of int. The size of target reconstruction resolution, all target images will be reshape to `image_shape`.
+ `hypo_hiddim`: Neural representation's hidden dim size.
+ `hypo_depth`: Depth of MLP in Representation.
+ `train_target`: Must be one of `image` or `nerf`, for different dataset setting.
+ `hyper_network`,`model_framework`: Model parameter setting.
+ `dataset_kwargs`: Dataset setting.
 

### Example Run
1. make sure that your installation is correct

2. modify the configuration file

You can choose those in `./configs/test` to test whether your setup is correct.

3. run script

The scripts for training/testing/evaling an model are present in `./script`. If you want to train a model, just modify the first few rows in the main function to choose the correspoding config file.

for example:
```python
if __name__ == "__main__":
    global config_file
    config_file = "./configs/Celeba128_anr_d5.yaml"
```

and then, run the following command in terminal:

```python
python ./script/train_inr.py 
```

## License

```BibTex
@article{zhang2024attention,
  title={Attention Beats Linear for Fast Implicit Neural Representation Generation},
  author={Zhang, Shuyi and Liu, Ke and Gu, Jingjun and Cai, Xiaoxu and Wang, Zhihua and Bu, Jiajun and Wang, Haishuai},
  journal={arXiv preprint arXiv:2407.15355},
  year={2024}
}
```

## Contact
If you have any questions, you can reach me via the following:

Email: keliu99@zju.edu.cn
