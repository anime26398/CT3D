GPUS=(0 1 2 3)
IDS=(1 2 3 4)
for i in ${!GPUS[@]}; do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} python edit_cli_all.py --steps 100 \
                        --resolution 512 \
                        --seed 1371 \
                        --cfg-text 5. \
                        --cfg-image 1.2 \
                        --indir /data/amahapat/eulerian_data/validation \
                        --id ${IDS[$i]}
done