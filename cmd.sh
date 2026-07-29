# 1000 step sweep
uv run cs336_basics/train.py \
    --train-data output/encoding/encoded_tinystories_train.npy \
    --val-data output/encoding/encoded_tinystories_valid.npy \
    --vocab-size 10000 \
    --context-length 256 \
    --d-model 512 \
    --num-layers 4 \
    --num-heads 16 \
    --d-ff 1344 \
    --batch-size 32 \
    --max-iters 1000 \
    --device cuda \
    --checkpoint-dir checkpoints/tinystories \
    --wandb-project cs336-basics \
    --wandb-run-name sweep_def3e-4 \
    --lr-max 3e-4 --lr-min 3e-5 --cosine-cycle-iters 1000

uv run cs336_basics/train.py \
    --train-data output/encoding/encoded_tinystories_train.npy \
    --val-data output/encoding/encoded_tinystories_valid.npy \
    --vocab-size 10000 \
    --context-length 256 \
    --d-model 512 \
    --num-layers 4 \
    --num-heads 16 \
    --d-ff 1344 \
    --batch-size 32 \
    --max-iters 1000 \
    --device cuda \
    --checkpoint-dir checkpoints/tinystories \
    --wandb-project cs336-basics \
    --wandb-run-name sweep_lr1e-4 \
    --lr-max 1e-4 --lr-min 1e-5 --cosine-cycle-iters 1000

uv run cs336_basics/train.py \
    --train-data output/encoding/encoded_tinystories_train.npy \
    --val-data output/encoding/encoded_tinystories_valid.npy \
    --vocab-size 10000 \
    --context-length 256 \
    --d-model 512 \
    --num-layers 4 \
    --num-heads 16 \
    --d-ff 1344 \
    --batch-size 32 \
    --max-iters 1000 \
    --device cuda \
    --checkpoint-dir checkpoints/tinystories \
    --wandb-project cs336-basics \
    --wandb-run-name sweep_lr1e-3 \
    --lr-max 1e-3 --lr-min 1e-4 --cosine-cycle-iters 1000

uv run cs336_basics/train.py \
    --train-data output/encoding/encoded_tinystories_train.npy \
    --val-data output/encoding/encoded_tinystories_valid.npy \
    --vocab-size 10000 \
    --context-length 256 \
    --d-model 512 \
    --num-layers 4 \
    --num-heads 16 \
    --d-ff 1344 \
    --batch-size 32 \
    --max-iters 1000 \
    --device cuda \
    --checkpoint-dir checkpoints/tinystories \
    --wandb-project cs336-basics \
    --wandb-run-name sweep_lr3e-3 \
    --lr-max 3e-3 --lr-min 3e-4 --cosine-cycle-iters 1000

uv run cs336_basics/train.py \
    --train-data output/encoding/encoded_tinystories_train.npy \
    --val-data output/encoding/encoded_tinystories_valid.npy \
    --vocab-size 10000 \
    --context-length 256 \
    --d-model 512 \
    --num-layers 4 \
    --num-heads 16 \
    --d-ff 1344 \
    --batch-size 32 \
    --max-iters 1000 \
    --device cuda \
    --checkpoint-dir checkpoints/tinystories \
    --wandb-project cs336-basics \
    --wandb-run-name sweep_lr1e-2 \
    --lr-max 1e-2 --lr-min 1e-3 --cosine-cycle-iters 1000

uv run cs336_basics/train.py \
    --train-data output/encoding/encoded_tinystories_train.npy \
    --val-data output/encoding/encoded_tinystories_valid.npy \
    --vocab-size 10000 \
    --context-length 256 \
    --d-model 512 \
    --num-layers 4 \
    --num-heads 16 \
    --d-ff 1344 \
    --batch-size 32 \
    --max-iters 1000 \
    --device cuda \
    --checkpoint-dir checkpoints/tinystories \
    --wandb-project cs336-basics \
    --wandb-run-name sweep_lr3e-2 \
    --lr-max 3e-2 --lr-min 3e-3 --cosine-cycle-iters 1000

# Troubleshoot overfit training loss
uv run cs336_basics/train.py \
    --train-data output/encoding/encoded_tinystories_train.npy \
    --val-data output/encoding/encoded_tinystories_valid.npy \
    --vocab-size 10000 \
    --context-length 256 \
    --d-model 512 \
    --num-layers 4 \
    --num-heads 16 \
    --d-ff 1344 \
    --batch-size 32 \
    --max-iters 1000 \
    --device cuda \
    --checkpoint-dir checkpoints/tinystories_test \
    --wandb-project cs336-basics \
    --wandb-run-name fix3_overfit_nodecay \
    --lr-max 3e-3 --lr-min 3e-3 --weight-decay 0 --grad-clip 1e9

# Local troubleshoot training loss single batch
uv run cs336_basics/train.py \
    --train-data output/encoding/encoded_tinystories_train.npy \
    --val-data output/encoding/encoded_tinystories_valid.npy \
    --vocab-size 10000 \
    --context-length 256 \
    --d-model 512 \
    --num-layers 4 \
    --num-heads 16 \
    --d-ff 1344 \
    --batch-size 32 \
    --max-iters 5000 \
    --device cuda \
    --checkpoint-dir checkpoints/tinystories_test \
    --wandb-project cs336-basics \
    --wandb-run-name fix3_overfit_lr3e-3_nodecay \
    --lr-max 3e-3 --lr-min 3e-3 --weight-decay 0 --grad-clip 1e9

# 5000 steps with 2 best lr
uv run cs336_basics/train.py \
    --train-data output/encoding/encoded_tinystories_train.npy \
    --val-data output/encoding/encoded_tinystories_valid.npy \
    --vocab-size 10000 \
    --context-length 256 \
    --d-model 512 \
    --num-layers 4 \
    --num-heads 16 \
    --d-ff 1344 \
    --batch-size 32 \
    --max-iters 5000 \
    --device cuda \
    --checkpoint-dir checkpoints/tinystories \
    --wandb-project cs336-basics \
    --wandb-run-name sweep_3e-4def_5000 \
    --lr-max 3e-4 --lr-min 3e-5 --cosine-cycle-iters 5000

uv run cs336_basics/train.py \
    --train-data output/encoding/encoded_tinystories_train.npy \
    --val-data output/encoding/encoded_tinystories_valid.npy \
    --vocab-size 10000 \
    --context-length 256 \
    --d-model 512 \
    --num-layers 4 \
    --num-heads 16 \
    --d-ff 1344 \
    --batch-size 32 \
    --max-iters 5000 \
    --device cuda \
    --checkpoint-dir checkpoints/tinystories \
    --wandb-project cs336-basics \
    --wandb-run-name sweep_lr1e-3_5000 \
    --lr-max 1e-3 --lr-min 1e-4 --cosine-cycle-iters 5000

# full token
uv run cs336_basics/train.py \
    --train-data output/encoding/encoded_tinystories_train.npy \
    --val-data output/encoding/encoded_tinystories_valid.npy \
    --vocab-size 10000 \
    --context-length 256 \
    --d-model 512 \
    --num-layers 4 \
    --num-heads 16 \
    --d-ff 1344 \
    --batch-size 32 \
    --max-iters 40000 \
    --device cuda \
    --checkpoint-dir checkpoints/tinystories \
    --wandb-project cs336-basics \
    --wandb-run-name sweep_lr1e-3_full \
    --lr-max 1e-3 --lr-min 1e-4 --cosine-cycle-iters 40000

# with torch.compmile and torch.setmatmultf32 and without
uv run cs336_basics/train.py \
    --train-data output/encoding/encoded_tinystories_train.npy \
    --val-data output/encoding/encoded_tinystories_valid.npy \
    --vocab-size 10000 \
    --context-length 256 \
    --d-model 512 \
    --num-layers 4 \
    --num-heads 16 \
    --d-ff 1344 \
    --batch-size 32 \
    --max-iters 1000 \
    --device cuda \
    --checkpoint-dir checkpoints/tinystories \
    --wandb-project cs336-basics \
    --wandb-run-name wspeedup_s1000_h200 \
    --lr-max 1e-3 --lr-min 1e-4 --cosine-cycle-iters 1000

uv run cs336_basics/train.py \
    --train-data output/encoding/encoded_tinystories_train.npy \
    --val-data output/encoding/encoded_tinystories_valid.npy \
    --vocab-size 10000 \
    --context-length 256 \
    --d-model 512 \
    --num-layers 4 \
    --num-heads 16 \
    --d-ff 1344 \
    --batch-size 32 \
    --max-iters 1000 \
    --device cuda \
    --checkpoint-dir checkpoints/tinystories \
    --wandb-project cs336-basics \
    --wandb-run-name wospeedup_s1000_h200 \
    --lr-max 1e-3 --lr-min 1e-4 --cosine-cycle-iters 1000

# batch size 
# bs 1 
uv run cs336_basics/train.py \
    --train-data output/encoding/encoded_tinystories_train.npy \
    --val-data output/encoding/encoded_tinystories_valid.npy \
    --vocab-size 10000 \
    --context-length 256 \
    --d-model 512 \
    --num-layers 4 \
    --num-heads 16 \
    --d-ff 1344 \
    --batch-size 1 \
    --max-iters 32000 \
    --device cuda \
    --checkpoint-dir checkpoints/tinystories \
    --wandb-project cs336-basics \
    --wandb-run-name bs1 \
    --lr-max 3e-5 --lr-min 3e-6 --cosine-cycle-iters 32000

# bs 16
uv run cs336_basics/train.py \
    --train-data output/encoding/encoded_tinystories_train.npy \
    --val-data output/encoding/encoded_tinystories_valid.npy \
    --vocab-size 10000 \
    --context-length 256 \
    --d-model 512 \
    --num-layers 4 \
    --num-heads 16 \
    --d-ff 1344 \
    --batch-size 16 \
    --max-iters 10000 \
    --device cuda \
    --checkpoint-dir checkpoints/tinystories \
    --wandb-project cs336-basics \
    --wandb-run-name bs16 \
    --lr-max 5e-4 --lr-min 5e-5 --cosine-cycle-iters 10000

# bs 32
uv run cs336_basics/train.py \
    --train-data output/encoding/encoded_tinystories_train.npy \
    --val-data output/encoding/encoded_tinystories_valid.npy \
    --vocab-size 10000 \
    --context-length 256 \
    --d-model 512 \
    --num-layers 4 \
    --num-heads 16 \
    --d-ff 1344 \
    --batch-size 32 \
    --max-iters 5000 \
    --device cuda \
    --checkpoint-dir checkpoints/tinystories \
    --wandb-project cs336-basics \
    --wandb-run-name bs32 \
    --lr-max 1e-3 --lr-min 1e-4 --cosine-cycle-iters 5000

# bs 64
uv run cs336_basics/train.py \
    --train-data output/encoding/encoded_tinystories_train.npy \
    --val-data output/encoding/encoded_tinystories_valid.npy \
    --vocab-size 10000 \
    --context-length 256 \
    --d-model 512 \
    --num-layers 4 \
    --num-heads 16 \
    --d-ff 1344 \
    --batch-size 64 \
    --max-iters 2500 \
    --device cuda \
    --checkpoint-dir checkpoints/tinystories \
    --wandb-project cs336-basics \
    --wandb-run-name bs64 \
    --lr-max 2e-3 --lr-min 2e-4 --cosine-cycle-iters 2500

# bs 128
uv run cs336_basics/train.py \
    --train-data output/encoding/encoded_tinystories_train.npy \
    --val-data output/encoding/encoded_tinystories_valid.npy \
    --vocab-size 10000 \
    --context-length 256 \
    --d-model 512 \
    --num-layers 4 \
    --num-heads 16 \
    --d-ff 1344 \
    --batch-size 128 \
    --max-iters 1250 \
    --device cuda \
    --checkpoint-dir checkpoints/tinystories \
    --wandb-project cs336-basics \
    --wandb-run-name bs128 \
    --lr-max 4e-3 --lr-min 4e-4 --cosine-cycle-iters 1250

# bs 256
uv run cs336_basics/train.py \
    --train-data output/encoding/encoded_tinystories_train.npy \
    --val-data output/encoding/encoded_tinystories_valid.npy \
    --vocab-size 10000 \
    --context-length 256 \
    --d-model 512 \
    --num-layers 4 \
    --num-heads 16 \
    --d-ff 1344 \
    --batch-size 256 \
    --max-iters 625 \
    --device cuda \
    --checkpoint-dir checkpoints/tinystories \
    --wandb-project cs336-basics \
    --wandb-run-name bs256 \
    --lr-max 8e-3 --lr-min 8e-4 --cosine-cycle-iters 625

# bs 512
uv run cs336_basics/train.py \
    --train-data output/encoding/encoded_tinystories_train.npy \
    --val-data output/encoding/encoded_tinystories_valid.npy \
    --vocab-size 10000 \
    --context-length 256 \
    --d-model 512 \
    --num-layers 4 \
    --num-heads 16 \
    --d-ff 1344 \
    --batch-size 313 \
    --max-iters 625 \
    --device cuda \
    --checkpoint-dir checkpoints/tinystories \
    --wandb-project cs336-basics \
    --wandb-run-name bs512-correct \
    --lr-max 1.6e-2 --lr-min 1.6e-3 --cosine-cycle-iters 313

# ablations
# rmsnorm
uv run cs336_basics/train.py \
    --train-data output/encoding/encoded_tinystories_train.npy \
    --val-data output/encoding/encoded_tinystories_valid.npy \
    --vocab-size 10000 \
    --context-length 256 \
    --d-model 512 \
    --num-layers 4 \
    --num-heads 16 \
    --d-ff 1344 \
    --batch-size 32 \
    --max-iters 5000 \
    --device cuda \
    --checkpoint-dir checkpoints/tinystories_ablation \
    --wandb-project cs336-basics \
    --wandb-run-name ablation_rmsnorm \
    --lr-max 1e-3 --lr-min 1e-4 --cosine-cycle-iters 5000
uv run cs336_basics/train.py \
    --train-data output/encoding/encoded_tinystories_train.npy \
    --val-data output/encoding/encoded_tinystories_valid.npy \
    --vocab-size 10000 \
    --context-length 256 \
    --d-model 512 \
    --num-layers 4 \
    --num-heads 16 \
    --d-ff 1344 \
    --batch-size 32 \
    --max-iters 5000 \
    --device cuda \
    --checkpoint-dir checkpoints/tinystories_ablation \
    --wandb-project cs336-basics \
    --wandb-run-name ablation_normsnorm \
    --lr-max 1e-3 --lr-min 1e-4 --cosine-cycle-iters 5000