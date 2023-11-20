# CT3D: Consistent Text-to-3D Generations For Custom Subjects

## [Website](https://fishbotwilleatyou.com/ct3d/)

This is an implementtaion of **CT3D: Consistent Text-to-3D Generations For Custom Subjects**  with [Stable Diffusion](https://github.com/CompVis/stable-diffusion). This method can generate 3D model of custom subjects according to a given text prompt with just 3-5 input image of the custom subject. This implementation is based on [Dreambooth](https://arxiv.org/abs/2208.12242), [Dreamfusion](https://dreamfusion3d.github.io/) and [Dreambooth3D](https://dreambooth3d.github.io/). This method can generate more 3D consistent custom subject 3D models compared to **Dreambooth3D** with Stable Diffusion backbone.


## Preparation

```
git clone <repo-name>

cd ./<rep-name>
conda env create -f environment_db1.yaml
conda env create -f environment_zero123.yaml
conda env create -f environment_ip2p.yaml
```

Download custom images for custom ```dog```, ```cat toy```, ```doll```, ```bag``` from [Drive](https://drive.google.com/drive/folders/1UaIa4yQ9CTZQKS7ZIDdibnsk0RUUfKEj?usp=sharing).

Download Stable Diffusion weights on [HuggingFace](https://huggingface.co/CompVis). You can decide which version of checkpoint to use, but I use sd-v1-4-full-ema.ckpt. Then put the weights in ```./Dreambooth-Stable-Diffusion/models/ldm``` folder.

Download Zero123 pretrained model weights:

```
cd ./stable-dreamfusion/pretrained/zero123
wget https://huggingface.co/cvlab/zero123-weights/resolve/main/105000.ckpt
```

Download Instruct-pix2pix checkpoint:

```
cd instruct-pix2pix
bash scripts/download_checkpoints.sh
```

## Running

Here we show how to run all the 3 methods, namely **Naive Dreambooth + Dreamfusion**, **Dreambooth3D** and **CT3D** for a custom subject ```<subject>``` 

Download the ```<subject>``` images from [Drive](https://drive.google.com/drive/folders/1UaIa4yQ9CTZQKS7ZIDdibnsk0RUUfKEj?usp=sharing) and save in ```./Dreambooth-Stable-Diffusion/dataset/<subject>/db_images``` folder.

### Generate Regularization images for Dreambooth
```
cd ./Dreambooth-Stable-Diffusion

conda activate db1

#generate 100 regularization images
CUDA_VISIBLE_DEVICES=0 python scripts/stable_txt2img.py --ddim_eta 0.0 \
                                   --n_samples 4 \
                                   --n_iter 25 \
                                   --scale 10.0 \
                                   --ddim_steps 50  \
                                   --ckpt ./models/ldm/sd-v1-4-full-ema.ckpt \
                                   --prompt "a photo of a <subject>" \
                                   --outdir ./dataset/<subject>/db_reg_images
```

### Training Dreambooth (full training)

```
#train dreambooth on given images (fully trained)
CUDA_VISIBLE_DEVICES=$DEVICE python main.py --base configs/stable-diffusion/v1-finetune_unfrozen.yaml \
                              -t \
                              --actual_resume ./models/ldm/sd-v1-4-full-ema.ckpt \
                              -n <subject>_0 \
                              --gpus 0, \
                              --data_root ./dataset/<subject>/db_images \
                              --reg_data_root ./dataset/<subject>/db_reg_images \
                              --class_word <subject> \
                              --no-test True \
                              --steps 800

#convert dreambooth weights to diffusers format
python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path ./logs/<subject>_0/checkpoints/last.ckpt \
                                                      --original_config_file ./configs/stable-diffusion/v1-inference.yaml \
                                                      --scheduler_type ddim \
                                                      --dump_path sd2diffusers/<subject>_0 \
                                                      --device cuda:0
```

### Training Dreambooth (partial training)

```
#train dreambooth on given images (partial trained)
CUDA_VISIBLE_DEVICES=$DEVICE python main.py --base configs/stable-diffusion/v1-finetune_unfrozen.yaml \
                              -t \
                              --actual_resume ./models/ldm/sd-v1-4-full-ema.ckpt \
                              -n <subject>_1 \
                              --gpus 0, \
                              --data_root ./dataset/<subject>/db_images \
                              --reg_data_root ./dataset/<subject>/db_reg_images \
                              --class_word <subject> \
                              --no-test True \
                              --steps 150

#convert dreambooth weights to diffusers format
python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path ./logs/<subject>_1/checkpoints/last.ckpt \
                                                      --original_config_file ./configs/stable-diffusion/v1-inference.yaml \
                                                      --scheduler_type ddim \
                                                      --dump_path sd2diffusers/<subject>_1 \
                                                      --device cuda:0
```

### Naive Dreambooth + Dreamfusion

```
cd ../stable-dreamfusion

#prompt is 'a marble statue of a sks <subject> wearing a hat'
CUDA_VISIBLE_DEVICES=$DEVICE python main.py \
                              --text a marble statue of a sks <subject> wearing a hat \
                              --workspace <subject>_0 \
                              -O \
                              --sd_version custom:../Dreambooth-Stable-Diffusion/sd2diffusers/<subject>_0

#dmtet finetuning (suggested)
CUDA_VISIBLE_DEVICES=$DEVICE python main.py -O \
              --texta marble statue of a sks <subject> wearing a hat\
              --workspace <subject>_0_dmtet \
              --dmtet \
              --iters 5000 \
              --init_ckpt <subject>_0_dmtet/checkpoints/df.pth \
              --sd_version custom:../Dreambooth-Stable-Diffusion/sd2diffusers/<subject>_0
```

The reults of Naive Dreambooth + Dreamfusion will be saved in ```./stable-dreamfusion/<subject>_0_dmtet/results```.

### Dreambooth3D

```
#dreamfusion on partially trained Dreambooth
cd ./stable-dreamfusion

#prompt is 'a marble statue of a sks <subject> wearing a hat'
CUDA_VISIBLE_DEVICES=$DEVICE python main.py \
                              --text a marble statue of a sks <subject> wearing a hat \
                              --workspace <subject>_1 \
                              -O \
                              --sd_version custom:../Dreambooth-Stable-Diffusion/sd2diffusers/<subject>_1

#dmtet finetuning (suggested) and saving NeRF images
CUDA_VISIBLE_DEVICES=$DEVICE python main.py -O \
              --text a marble statue of a sks <subject> wearing a hat \
              --workspace <subject>_1_dmtet \
              --dmtet \
              --iters 5000 \
              --init_ckpt <subject>_1_dmtet/checkpoints/df.pth \
              --sd_version custom:../Dreambooth-Stable-Diffusion/sd2diffusers/<subject>_1
              --save_inter_images ../Dreambooth-Stable-Diffusion/dataset/<subject>/db_partial_images

cd ../Dreambooth-Stable-Diffusion

#img2img translation
python scripts/img2img.py --prompt a marble statue of a sks <subject> wearing a hat \
        --init-img ./dataset/<subject>/db_partial_images \
        --strength 0.5 \
        --outdir ./dataset/<subject>/db_partial_mod_images \
        --ckpt ./logs/<subject>_0/checkpoints/last.ckpt

mkdir -p ./dataset/<subject>/db_images_combined
cp -RT ./dataset/<subject>/db_images ./dataset/<subject>/db_images_combined
cp -RT ./dataset/<subject>/db_partial_mod_images ./dataset/<subject>/db_images_combined

#train dreambooth on combined images
CUDA_VISIBLE_DEVICES=$DEVICE python main.py --base configs/stable-diffusion/v1-finetune_unfrozen.yaml \
                              -t \
                              --actual_resume './logs/<subject>_0/checkpoints/last.ckpt' \
                              -n <subject>_2 \
                              --gpus 0, \
                              --data_root ./dataset/<subject>/db_images_combined \
                              --reg_data_root ./dataset/<subject>/db_reg_images \
                              --class_word <subject> \
                              --no-test True \
                              --steps 150

#convert dreambooth weights to diffusers format
python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path ./logs/<subject>_2/checkpoints/last.ckpt \
                                                      --original_config_file ./configs/stable-diffusion/v1-inference.yaml \
                                                      --scheduler_type ddim \
                                                      --dump_path sd2diffusers/<subject>_2 \
                                                      --device cuda:0

#dreamfusion on combined dreambooth
cd ../stable-dreamfusion

#prompt is 'a marble statue of a sks <subject> wearing a hat'
CUDA_VISIBLE_DEVICES=$DEVICE python main.py \
                              --text a marble statue of a sks <subject> wearing a hat \
                              --workspace <subject>_2 \
                              -O \
                              --sd_version custom:../Dreambooth-Stable-Diffusion/sd2diffusers/<subject>_2

#dmtet finetuning (suggested)
CUDA_VISIBLE_DEVICES=$DEVICE python main.py -O \
              --text a marble statue of a sks <subject> wearing a hat \
              --workspace <subject>_2_dmtet \
              --dmtet \
              --iters 5000 \
              --init_ckpt <subject>_2_dmtet/checkpoints/df.pth \
              --sd_version custom:../Dreambooth-Stable-Diffusion/sd2diffusers/<subject>_2
```

The reults of Dreambooth3D will be saved in ```./stable-dreamfusion/<subject>_2_dmtet/results```.

### CT3D

```
conda deactivate
conda activate zero123
cd ./stable-dreamfusion

#dreamfusion on zero123 (initial stage), move one image from ../Dreambooth-Stable-Diffusion/dataset/<subject>/db_images to ./data and rename <subject>.png

#remove background
python preprocess_image.py ./data/<subject>.png

#prompt is 'a photo sks <subject>'
CUDA_VISIBLE_DEVICES=$DEVICE python main.py \
                              --image ./data/<subject>_rgba.png \
                              --workspace <subject>_1_zero \
                              -O \
                              --iters 5000

#dmtet finetuning (suggested) and saving NeRF images
CUDA_VISIBLE_DEVICES=$DEVICE python main.py -O \
              --image ./data/<subject>_rgba.png \
              --workspace <subject>_1_zero_dmtet \
              --dmtet \
              --iters 5000 \
              --init_ckpt <subject>_1_zero_dmtet/checkpoints/df.pth \
              --save_inter_images ../Dreambooth-Stable-Diffusion/dataset/<subject>/db_partial_images_zero

#instruct-pix2pix editing (for prompt: 'a marble statue of a sks <subject> wearing a hat')
#convert the texture of sks <subject> to 'marble statue'
cd ../instruct-pix2pix/
conda deactivate 
conda activate ip2p

CUDA_VISIBLE_DEVICES=$DEVICE python edit_cli_all.py --steps 100 \
                   --resolution 512 \
                   --seed 1371 \
                   --cfg-text 7.5 \
                   --cfg-image 1.5 \
                   --indir ./dataset/<subject>/db_partial_images_zero \
                   --output ../Dreambooth-Stable-Diffusion/dataset/<subject>/db_partial_mod_images_zero \
                   --edit 'make it a marble statue'

conda deactivate
conda activate db1
cd ../Dreambooth-Stable-Diffusion/

#train dreambooth on combined images
CUDA_VISIBLE_DEVICES=$DEVICE python main.py --base configs/stable-diffusion/v1-finetune_unfrozen.yaml \
                              -t \
                              --actual_resume ./models/ldm/sd-v1-4-full-ema.ckpt \
                              -n <subject>_2_zero \
                              --gpus 0, \
                              --data_root ./dataset/<subject>/db_partial_mod_images_zero \
                              --reg_data_root ./dataset/<subject>/db_reg_images \
                              --class_word <subject> \
                              --no-test True \
                              --steps 150

#convert dreambooth weights to diffusers format
python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path ./logs/<subject>_2_zero/checkpoints/last.ckpt \
                                                      --original_config_file ./configs/stable-diffusion/v1-inference.yaml \
                                                      --scheduler_type ddim \
                                                      --dump_path sd2diffusers/<subject>_2_zero \
                                                      --device cuda:0

#dreamfusion on combined dreambooth
cd ../stable-dreamfusion

#prompt is 'a marble statue of a sks <subject> wearing a hat', here we replace it with 'a sks <subject> wearing a hat' because the dreambooth already knows sks <subject> is made of marble
CUDA_VISIBLE_DEVICES=$DEVICE python main.py \
                              --text a sks <subject> wearing a hat \
                              --workspace <subject>_2_zero \
                              -O \
                              --sd_version custom:../Dreambooth-Stable-Diffusion/sd2diffusers/<subject>_2_zero

#dmtet finetuning (suggested)
CUDA_VISIBLE_DEVICES=$DEVICE python main.py -O \
              --text a sks <subject> wearing a hat \
              --workspace <subject>_2_zero_dmtet \
              --dmtet \
              --iters 5000 \
              --init_ckpt <subject>_2_zero_dmtet/checkpoints/df.pth \
              --sd_version custom:../Dreambooth-Stable-Diffusion/sd2diffusers/<subject>_2_zero
```

The reults of CT3D will be saved in ```./stable-dreamfusion/<subject>_2_zero_dmtet/results```.

