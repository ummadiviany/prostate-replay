# Worst case
/home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:0 --optimizer adam --alpha 1 --initial_epochs 100 --lr 1e-3 --lr_decay 0.8 --epoch_decay 0.8 --wandb_log --seed 2000  &
/home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:0 --optimizer adam --alpha 1 --initial_epochs 100 --lr 1e-3 --lr_decay 0.8 --epoch_decay 0.8 --wandb_log --seed 3000  &
/home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:0 --optimizer adam --alpha 1 --initial_epochs 100 --lr 1e-3 --lr_decay 0.8 --epoch_decay 0.8 --wandb_log --seed 4000  &
/home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:0 --optimizer adam --alpha 1 --initial_epochs 100 --lr 1e-3 --lr_decay 0.8 --epoch_decay 0.8 --wandb_log --seed 5000

# Practical case
/home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:0 --optimizer adam --alpha 1 --initial_epochs 100 --lr 1e-3 --lr_decay 0.8 --epoch_decay 0.8 --wandb_log --seed 2000  --order_reverse &
/home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:0 --optimizer adam --alpha 1 --initial_epochs 100 --lr 1e-3 --lr_decay 0.8 --epoch_decay 0.8 --wandb_log --seed 3000  --order_reverse &
/home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:0 --optimizer adam --alpha 1 --initial_epochs 100 --lr 1e-3 --lr_decay 0.8 --epoch_decay 0.8 --wandb_log --seed 4000  --order_reverse &
/home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:0 --optimizer adam --alpha 1 --initial_epochs 100 --lr 1e-3 --lr_decay 0.8 --epoch_decay 0.8 --wandb_log --seed 5000  --order_reverse 

