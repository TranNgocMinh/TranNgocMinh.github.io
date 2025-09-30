
!python myconvnext_train.py \
  --csv fer2013.csv \
  --img-size 224 --epochs 100 --warmup 5 \
  --batch-size 64 --lr 5e-4 --weight-decay 5e-2 \
  --drop-path 0.2 --label-smoothing 0.1 \
  --mixup 0.2 --cutmix 0.2 --ema --ema-decay 0.9997 \
  --amp --balanced-sampler --tta \
  --checkpoint-dir checkpoints --out best_myconvnext_srblock.pth
  
