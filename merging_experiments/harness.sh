MODEL_ARGS="pretrained=/storage/coda1/p-apadmanabh3/0/vgupta345/merging_exp/mergekit/op_1/config.json"
TASK="jsquad-1.1-0.6,jcommonsenseqa-1.1-0.6,jnli-1.3-0.6,marc_ja-1.1-0.6"
TASK="jsquad-1.1-0.6"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot "2,3,3,3" \
    --device "cuda" \
    --output_path "result.json"



    Test-sociology', 'hendrycksTest-us_foreign_policy', 'hendrycksTest-virology', 'hendrycksTest-world_religions', 'iwslt17-ar-en', 'iwslt17-en-ar', 'jaqket_v1', 'jaqket_v1-0.1-0.1', 'jaqket_v1-0.1-0.2', 'jaqket_v1-0.1-0.3', 'jaqket_v1-0.1-0.4', 'jaqket_v2', 'jaqket_v2-0.2-0.1', 'jaqket_v2-0.2-0.2', 'jaqket_v2-0.2-0.3', 'jaqket_v2-0.2-0.4', 'jaqket_v2-0.2-0.5', 'jaqket_v2-0.2-0.6', 'jaquad', 'jaquad-0.1-0.1', 'jaquad-0.1-0.2', 'jaquad-0.1-0.3', 'jaquad-0.1-0.4', 'jaquad-0.1-0.5', 'jaquad-0.1-0.6', 'jblimp', 'jcola', 'jcola-0.2-0.0', 'jcola-0.2-0.3', 'jcola-0.2-0.4', 'jcola-0.2-0.5', 'jcola-0.2-0.6', 'jcommonsenseqa', 'jcommonsenseqa-1.1-0.1', 'jcommonsenseqa-1.1-0.2', 'jcommonsenseqa-1.1-0.2.1', 'jcommonsenseqa-1.1-0.3', 'jcommonsenseqa-1.1-0.4', 'jcommonsenseqa-1.1-0.5', 'jcommonsenseqa-1.1-0.6', 'jnli', 'jnli-1.3-0.2', 'jnli-1.3-0.3', 'jnli-1.3-0.4', 'jnli-1.3-0.5', 'jnli-1.3-0.6', 'jsquad', 'jsquad-1.1-0.1', 'jsquad-1.1-0.2', 'jsquad-1.1-0.3', 'jsquad-1.1-0.4', 'jsquad-1.1-0.5', 'jsquad-1.1-0.6', 'jsquad-1.2-0.2', 'jsquad-1.2-0.3', 'jsquad-1.2-0.4', 'jsquad-1.2-0.5', 'jsquad-1.2-0.6', 'lambada_openai', 'lambada_openai_cloze', 'lambada_openai_mt_de', 'lambada_openai_mt_en', 'lambada_openai_mt_es', 'lambada_openai_mt_fr', 'lambada_openai_mt_it', 'lambada_openai_mt_ja', 'lambada_standard', 'lambada_standard_cloze', 'logiqa', 'marc_ja', 'marc_ja-1.1-0.2', 'marc_ja-1.1-0.3', 'marc_ja-1.1-0.4', 'marc_ja-1.1-0.5', 'marc_ja-1.1-0.6', 'math_algebra', 'math_asdiv', 'math_counting_and_prob', 'math_geometry', 'math_intermediate_algebra', 'math_num_theory', 'math_prealgebra', 'math_precalc', 'mathqa', 'mc_taco', 'mgsm', 'mgsm-1.0-0.0', 'mgsm-1.0-0.3', 'mgsm-1.0-0.4', 'mgsm-1.0-0.5', 'mgsm-1.0-0.6', 'mnli', 'mnli_mismatched', 'mrpc', 'multirc', 'mutual', 'mutual_plus', 'openbookqa', 'pile_arxiv', 'pile_bookcorpus2', 'pile_books3', 'pile_dm-mathematics', 'pile_enron', 'pile_europarl', 'pile_freelaw', 'pile_github', 'pile_gutenberg', 'pile_hackernews', 'pile_nih-exporter', 'pile_opensubtitle