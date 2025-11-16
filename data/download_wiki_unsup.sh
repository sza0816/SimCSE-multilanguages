cd data
wget https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt
cd ..

# run this command directly in terminal
# make sure data is downloaded into data folder

# *** clean unusual terminators in vscode ***
# Ctrl/Cmd + Shift + P
# search: Preferences: Open Settings (UI)
# look up: editor.unusualLineTerminators
# choose: auto


# to test with small subset of wiki data: 

# head -n 2000 data/wiki1m_for_simcse.txt > data/wiki1k.txt
# bash bash/train.sh data/wiki1k.txt outputs/test_small
# python train_unsup.py \
#   --data_path data/wiki1k.txt \
#   --output_dir outputs/en/unsup/test_small \
#   --model_name bert-base-uncased \
#   --epochs 0.1 \
#   --batch_size 64 \
#   --lr 3e-5 \
#   --max_len 32

# head -n 50000 data/wiki1m_for_simcse.txt > data/wiki50k.txt
#  python train_unsup.py \
#   --data_path data/wiki50k.txt \
#   --output_dir outputs/en/unsup/test_small \
#   --model_name bert-base-uncased \
#   --epochs 0.1 \
#   --batch_size 64 \
#   --lr 3e-5 \
#   --max_len 32