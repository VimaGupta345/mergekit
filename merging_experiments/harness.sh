MODEL_ARGS="pretrained=/storage/coda1/p-apadmanabh3/0/vgupta345/merging_exp/mergekit/op_1/config.json"
TASK="jsquad-1.1-0.6,jcommonsenseqa-1.1-0.6,jnli-1.1-0.6,marc_ja-1.1-0.6"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot "2,3,3,3" \
    --device "cuda" \
    --output_path "result.json"