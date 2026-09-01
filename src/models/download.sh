mkdir -p tmp/checkpoints/imagenet/dino/official
mkdir -p tmp/checkpoints/dinov2

wget -O tmp/checkpoints/imagenet/dino/official/dino_vitbase16_pretrain_full_checkpoint.pth https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain_full_checkpoint.pth
wget -O tmp/checkpoints/dinov2/dinov2_vitb14_pretrain.pth https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
wget -O tmp/checkpoints/dinov2/dinov2_vits14_pretrain.pth https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth