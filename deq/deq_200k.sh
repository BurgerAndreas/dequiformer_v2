python main_oc20.py \
    --num-gpus 1 \
    --num-nodes 1 \
    --mode train \
    --config-yml 'deq/deq_200k.yml' \
    --run-dir 'models/deq/200k' \
    --print-every 200 \
    --amp \
    --submit
