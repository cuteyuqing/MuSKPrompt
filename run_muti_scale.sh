# export CUDA_VISIBLE_DEVICES=0

# LLM=opt-2.7b
# LLM_DIR=./llm/${LLM}
LLM_DIR=/opt/data/private/code/NLP/prompt_bbt/local_model/model_dir
DATA_DIR=./data/

# Set max demonstration shot w.r.t. context length
if [[ "${LLM}" == "gpt2-xl" ]] || [[ "{$LLM}" == "gpt2-large" ]]; then
# max context length = 1024
array1=(mpqa) # maxshot = 32
array2=(sst2) # maxshot = 16
array3=(subj cr mr trec) # maxshot = 8
array4=(rte) # maxshot = 4
array5=(agnews cb) # maxshot = 2
array6=(dbpedia) # maxshot = 1
else
# max context length = 2048
array1=(sst2 mpqa)
array2=(subj cr mr trec)
array3=(rte)
array4=(agnews cb)
array5=(none)
array6=(dbpedia)
fi

# for DATASET in sst2 subj mpqa agnews cb cr dbpedia mr rte trec; do add yelpp mrpc snli
DATASET=trec
# "${array1[@]}" =~ "${DATASET}" 
if [[ "${array1[@]}" =~ "${DATASET}" ]]; then
N_DEMO_SHOT=32
elif [[ "${array2[@]}" =~ "${DATASET}" ]]; then
N_DEMO_SHOT=16
elif [[ "${array3[@]}" =~ "${DATASET}" ]]; then
N_DEMO_SHOT=8
elif [[ "${array4[@]}" =~ "${DATASET}" ]]; then
N_DEMO_SHOT=4
elif [[ "${array5[@]}" =~ "${DATASET}" ]]; then
N_DEMO_SHOT=2
else
N_DEMO_SHOT=1
fi
# 指定训练shot
N_TRAIN_SHOT=16
N_DEMO_SHOT=1
KNN=2
p=(1 2 3)

# sleep 1.5h
echo "开始执行"
# MODEL_DIR=/opt/data/private/code/NLP/prompt_bbt/local_model/model_dir/
# sst2 subj mpqa cr mr rte trec agnews mrpc cb dbpedia snli yelpp
# gpt2 gpt2-medium gpt2-large gpt-j-6b
# LLM_DIR=${MODEL_DIR}${LLM_DIR}
# for alpha in 0 0.2 0.4 0.6 0.8 1.0;do
for LLM in gpt2-xl;do
for prompt_level in 4;do
# for prompt_level in 4;do
# for DATASET in sst2 subj mpqa cr mr rte trec agnews mrpc cb dbpedia; do
# for DATASET in sst2 subj mpqa cr mr rte agnews; do
for DATASET in sst2 subj mpqa cr mr; do
# for SEED in 1 2 3 4 5; do
for SEED in 1 2 3; do
            python3 muti_scale_prompt.py \
                --llm_dir ${LLM_DIR}/${LLM} \
                --dataset ${DATASET} \
                --data_dir ${DATA_DIR} \
                --n_train_shot ${N_TRAIN_SHOT} \
                --n_demo_shot ${N_DEMO_SHOT} \
                --seed $SEED \
                --output_dir ./output \
                --cmd_p -1 \
                --device "cuda:0" \
                --use_class_center 1 \
                --alpha 0.5\
                --prompt_level ${prompt_level}\
                --more_memory\
                --voteorsum 1\
                --use_knn 0\
                --knn ${KNN} &> ./log/ours/${DATASET}/${LLM}_${N_TRAIN_SHOT}_${N_DEMO_SHOT}_${KNN}_${SEED}.log &
                # --knn ${KNN}  
                
wait
done
done
done
done
# done
# ${alpha[$SEED]}
# for DATASET in "subj" "mpqa" "cr" "mr" ;do
#     for N_DEMO_SHOT in 2 4 6 8 10 12 14; do
#         for SEED in 1 2 3 4 5; do
#         # --more_icl\
#         python3 knn_prompting.py \
#             --llm_dir ${LLM_DIR} \
#             --dataset ${DATASET} \
#             --data_dir ${DATA_DIR} \
#             --n_train_shot ${N_TRAIN_SHOT} \
#             --n_demo_shot ${N_DEMO_SHOT} \
#             --seed $SEED \
#             --output_dir ./output \
#             --cmd_p -1 \
#             --device "cuda:0" \
#             --use_class_center 0 \
#             --alpha 0.5\
#             --use_knn 1\
#             --knn ${KNN} &> ./log/ours/${DATASET}/${LLM}_${N_TRAIN_SHOT}_${N_DEMO_SHOT}_${KNN}_${SEED}.log &
#         done
#         wait
#     done
#     wait
# done
echo "所有进程已执行完毕"


