# !/bin/bash

/home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:0 --optimizer adam --initial_epochs 100 --lr 1e-3 --lr_decay 0.8 --epoch_decay 0.8 --replay --store_samples 3 --wandb_log --seed 2000 --sampling_strategy random &
/home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:0 --optimizer adam --initial_epochs 100 --lr 1e-3 --lr_decay 0.8 --epoch_decay 0.8 --replay --store_samples 3 --wandb_log --seed 3000 --sampling_strategy random &
/home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:0 --optimizer adam --initial_epochs 100 --lr 1e-3 --lr_decay 0.8 --epoch_decay 0.8 --replay --store_samples 3 --wandb_log --seed 4000 --sampling_strategy random &
/home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:0 --optimizer adam --initial_epochs 100 --lr 1e-3 --lr_decay 0.8 --epoch_decay 0.8 --replay --store_samples 3 --wandb_log --seed 5000 --sampling_strategy random &
/home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:1 --optimizer adam --initial_epochs 100 --lr 1e-3 --lr_decay 0.8 --epoch_decay 0.8 --replay --store_samples 3 --wandb_log --seed 2000 --sampling_strategy random --order_reverse &
/home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:1 --optimizer adam --initial_epochs 100 --lr 1e-3 --lr_decay 0.8 --epoch_decay 0.8 --replay --store_samples 3 --wandb_log --seed 3000 --sampling_strategy random --order_reverse &
/home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:1 --optimizer adam --initial_epochs 100 --lr 1e-3 --lr_decay 0.8 --epoch_decay 0.8 --replay --store_samples 3 --wandb_log --seed 4000 --sampling_strategy random --order_reverse &
/home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:1 --optimizer adam --initial_epochs 100 --lr 1e-3 --lr_decay 0.8 --epoch_decay 0.8 --replay --store_samples 3 --wandb_log --seed 5000 --sampling_strategy random --order_reverse


# /home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:0 --optimizer sgd --initial_epochs 100 --lr 1e-3 --lr_decay 1 --epoch_decay 1  --wandb_log --seed 2000  &
# /home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:0 --optimizer sgd --initial_epochs 100 --lr 1e-3 --lr_decay 1 --epoch_decay 1  --wandb_log --seed 3000  &
# /home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:0 --optimizer sgd --initial_epochs 100 --lr 1e-3 --lr_decay 1 --epoch_decay 1  --wandb_log --seed 4000  &
# /home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:0 --optimizer sgd --initial_epochs 100 --lr 1e-3 --lr_decay 1 --epoch_decay 1  --wandb_log --seed 5000  &
# /home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:1 --optimizer sgd --initial_epochs 100 --lr 1e-3 --lr_decay 1 --epoch_decay 1  --wandb_log --seed 2000  --order_reverse &
# /home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:1 --optimizer sgd --initial_epochs 100 --lr 1e-3 --lr_decay 1 --epoch_decay 1  --wandb_log --seed 3000  --order_reverse &
# /home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:1 --optimizer sgd --initial_epochs 100 --lr 1e-3 --lr_decay 1 --epoch_decay 1  --wandb_log --seed 4000  --order_reverse &
# /home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:1 --optimizer sgd --initial_epochs 100 --lr 1e-3 --lr_decay 1 --epoch_decay 1  --wandb_log --seed 5000  --order_reverse 

