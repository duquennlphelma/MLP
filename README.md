# Real NVP in Pytorch - TUB

## Real NVP

### Description of the project

This project is an implementation of the Real NVP in Pytorch based on the paper :

> **Density estimation using Real NVP**
>
> Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio
> 
> https://arxiv.org/abs/1605.08803

### Use the code

> **Environment**
> 
> Miniconda environment - Python 3.7

> **Datasets**
> 
> 1. MoonDataset (2D) : generated from `sklearn.datasets.make_moons`
> 2. FunDataset (2D) : samples generated from an image
> 3. MNIST (images)

> **Run**
> 
> 1. Main code to run : `RNVP_shell_script.sh` directly calls `train.py`
> 2. Modify `RNVP_shell_script.sh` with the hyperparameters you want to use 
> (use `python train.py --help` to see the parameters you can modify)
>
> You can train a model or load an already trained one on a specific Dataset.


> **Outputs**
>
> Running the code generates different figures in the `plots` directory: 
> - Training loss per epoch (if you trained a model)
> - Example of an output of the model (data space -> latent space)
> - Example of a reconstructed data (latent space -> data space)
> 
> You also save the parameters of your trained model in the `trained_model` directory

### Results

1. MoonDataset : model ![MoonModel](/trained_models/model_trained_MoonDataset_250_epochs_250_batchsize_0.0001_lr.pth)

![MoonResults](/plots/readme/epochs_loss_FunDataset_250_epochs_250_batchsize_0.0001_lr.png)
![MoonResults](/plots/readme/test_output_MoonDataset_250_epochs_1000_points_250_batchsize_0.0001_lr.png)
![MoonResults](/plots/readme/test_invert_MoonDataset_250_epochs_1000_points_250_batchsize_0.0001_lr.png)

