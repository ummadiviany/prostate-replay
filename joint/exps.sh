

/home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:0 --optimizer adam --epochs 50 --lr 1e-3 --wandb_log --seed 2000  &
/home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:0 --optimizer adam --epochs 50 --lr 1e-3 --wandb_log --seed 3000  &
/home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:0 --optimizer adam --epochs 50 --lr 1e-3 --wandb_log --seed 4000  &
/home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:0 --optimizer adam --epochs 50 --lr 1e-3 --wandb_log --seed 5000  & 
/home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:0 --optimizer adam --epochs 100 --lr 1e-3 --wandb_log --seed 2000  &
/home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:0 --optimizer adam --epochs 100 --lr 1e-3 --wandb_log --seed 3000  &
/home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:0 --optimizer adam --epochs 100 --lr 1e-3 --wandb_log --seed 4000  &
/home/amritesh/anaconda3/envs/torch/bin/python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:0 --optimizer adam --epochs 100 --lr 1e-3 --wandb_log --seed 5000  

echo "Done with all experiments"