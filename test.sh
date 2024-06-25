MODEL_DIR=/opt/data/private/code/NLP/prompt_bbt/local_model/model_dir/
# sst2 subj mpqa cr mr rte trec agnews mrpc cb dbpedia snli yelpp
# gpt2 gpt2-medium gpt2-large gpt-j-6b
for LLM_DIR in gpt2 gpt2-medium gpt2-large; do
LLM_DIR=${MODEL_DIR}${LLM_DIR}
echo ${LLM_DIR}
done