This repository is the forked version of [adamxyang/laplace-lora](https://github.com/adamxyang/laplace-lora).

I have made modifications to the code for experimental purpose. 

I ran all the scripts on google cab as follows:

# fine tuning LLM using LoRA
```
!python3 /content/laplace-lora/run_gpt.py --model_name_or_path="NousResearch/Llama-2-7b-hf" --max_train_steps=100
```


# fine tuning LLM using Laplace-LoRA
```
!python3 /content/laplace-lora/run_gpt_laplace.py --model_name_or_path="NousResearch/Llama-2-7b-hf" --max_train_steps=100
```


I added one more script of training LoRA parameters using Cyclical Stochastic Gradient descent technique [ruqizhang/
csgmcmc](https://github.com/ruqizhang/csgmcmc.git)

# fine tuning LLM using CSG-MCMC loRA
```
!python3 /content/laplace-lora/run_gpt_csgmcmc.py --model_name_or_path="NousResearch/Llama-2-7b-hf"
```



# laplace-lora
Code for [Bayesian low-rank adaptation for large language models](https://arxiv.org/abs/2308.13111)

This library is largely based on [Laplace](https://github.com/aleximmer/Laplace) and [ASDL](https://github.com/kazukiosawa/asdl/tree/master).

Before installing laplace-lora, first install ASDL from source
```
pip install git+https://github.com/kazukiosawa/asdl
```

To install `laplace-lora`, change directory to `laplace-lora` and run 
```
pip install -e.
```

# LLM fine-tuning with LoRA
To fine-tune LlaMA2 or any GPT-like model on common sense reasoning tasks, use 
```
accelerate launch run_gpt.py
``` 
or the bash file 
```
bash run_gpt.sh
``` 
for submission to a slurm server. Customize training arguments like `lora_alpha`, `lora_r`, `lora_dropout`, etc. Set `testing_set` argument to `val` if using the full training set; set `testing_set` argument to `train_val` to split the training set into training and validation set.

### Hyperparameters for LoRA fine-tuning
There are several hyperparameters that can be tuned for LoRA fine-tuning, e.g. `lora_alpha`, `lora_r`, `lora_dropout`, `learning_rate`, etc.

To use the full training set and Laplace model evidence for optimizing Laplace prior precision, set  the `testing_set` argument to `val`; to split training set into a training set and a validation set and use minibatch gradient descent on the validation negative log-likelihood for optimizing Laplace prior precision, set the `testing_set` argument to `train_val`.

# Post-hoc Laplace-LoRA
To run post-hoc Laplace approximation on saved checkpoints, use 
``` 
accelerate launch run_gpt_laplace.py
``` 
or the bash file 
```
bash run_gpt_laplace.sh
``` 
for submission to a slurm server.

### Hyperparameters for Laplace-LoRA
To use full Laplace-LoRA, set the `laplace_sub` argument to `all`; to use last-layer Laplace-LoRA, set the `laplace_sub` argument to `last_layer`.
