

# Notes for Teammates — How to Run Our SimCSE Code on Vast.ai

This is **not** a formal project README. It’s a practical checklist so you can:

1. Rent a GPU machine on Vast.ai
2. Connect with VS Code (SSH)
3. Clone **this** repo and switch to the correct branch
4. Install dependencies and download data
5. Run training / evaluation for English SimCSE (unsupervised & supervised)

If anything is unclear, ping Zian and mention which step you are stuck at.

---

## 1. Rent a GPU on Vast.ai

1. Log in to **vast.ai**.
2. Go to **Search** and pick a GPU machine (I usually choose **A5000** or **RTX 4090** with a standard PyTorch image).
3. For the Docker image:
   - Use a PyTorch image that already has CUDA (e.g. something like `pytorch` with Python ≥ 3.10).
   - You don’t need to customize anything else for now.
4. Click **Rent** and wait until the instance status becomes **running**.

> You pay only while the instance is ON. Remember to shut it down when you’re done.

---

## 2. Get the SSH command from Vast.ai

On the instance page, Vast.ai shows a **Direct SSH** command, something like:

```bash
ssh -p PORT root@sshX.vast.ai
```

Copy this command. We will use it inside VS Code Remote‑SSH.

---

## 3. Connect from VS Code (Remote‑SSH)

1. Open **VS Code** on your laptop.
2. Install the extension **Remote‑SSH** (if you don’t have it yet).
3. Open the command palette (⇧⌘P / Ctrl+Shift+P) → **Remote‑SSH: Add New SSH Host**.
4. Paste the Direct SSH command from Vast.ai (for example: `ssh -p 16235 root@ssh1.vast.ai`).
5. Save to your SSH config.
6. Again open the command palette → **Remote‑SSH: Connect to Host** → pick the host you just added.
7. After connecting, VS Code will show that you’re in a remote session (bottom‑left corner).

You should now be inside the container as **root**.

---

## 4. Create workspace folder and clone the repo

Inside the remote terminal **in VS Code**:

```bash
mkdir -p /workspace/simcse
cd /workspace/simcse
```

Then clone the repo via **HTTPS** (easier for this setup):

```bash
git clone https://github.com/sza0816/SimCSE-multilanguages.git
cd SimCSE-multilanguages
```

Now **very important**:

1. Check that you are on the correct branch (the Vast‑compatible code is on branch `zian`):

   ```bash
   git branch
   ```

2. If you are not on `zian`, switch to it:

   ```bash
   git checkout zian
   ```

3. Confirm that your working directory is the repo root:

   ```bash
   pwd
   ```

   It should be something like:

   ```
   /workspace/simcse/SimCSE-multilanguages
   ```

All training / evaluation commands below assume this path.

---

## 5. Environment setup on Vast.ai

On Vast.ai we **do not** use a Python virtualenv for now — everything is installed into the container’s main environment.

In the repo root (`/workspace/simcse/SimCSE-multilanguages`):

```bash
bash bash/setup_vast.sh
```

(If the script name changes later, check the `bash/` folder or comments from Zian.)

This script does roughly the following:

- Prints Python version (should be 3.10+)
- Installs core packages:
  - `transformers`, `tokenizers`, `datasets`, `accelerate`
  - `torch` with CUDA
  - `sentencepiece`, `scikit-learn`, `pandas`, `tqdm`
- Downloads English SimCSE data via the small scripts in `data/`:
  - `wiki1m_for_simcse.txt` (unsupervised)
  - `nli_for_simcse.csv` (supervised)
  - STS‑B English files (for evaluation)

This may take a while the first time (lots of packages / data).

---

## 6. Running English training (unsup / sup)

Right now, English training is unified in **`train_english.py`**, which supports two modes:

- `mode=unsup`  → **unsupervised SimCSE** on `wiki1m_for_simcse.txt`
- `mode=sup`    → **supervised SimCSE** on `nli_for_simcse.csv`

### 6.1 Example: Unsupervised English training

From the repo root:

```bash
python train_english.py \
  --mode unsup \
  --model_name bert-base-uncased \
  --epochs 1 \
  --batch_size 64 \
  --lr 5e-5 \
  --max_len 32 \
  --warmup_ratio 0.1 \
  --data_path data/wiki1m_for_simcse.txt \
  --output_dir outputs/en/unsup
```

This will:

- Load `bert-base-uncased` from Hugging Face
- Train SimCSE in unsupervised mode using the Wikipedia sentences
- Save weights and config into `outputs/en/unsup`:
  - `pytorch_model.bin` (contrastive head + encoder weights)
  - `config.txt` (backbone name)
  - tokenizer files

### 6.2 Example: Supervised English training

After unsup works, you can try supervised:

```bash
python train_english.py \
  --mode sup \
  --model_name bert-base-uncased \
  --epochs 1 \
  --batch_size 64 \
  --lr 5e-5 \
  --max_len 32 \
  --warmup_ratio 0.1 \
  --data_path data/nli_for_simcse.csv \
  --output_dir outputs/en/sup
```

This uses NLI pairs `(sent0, sent1, hard_neg)` and implements the supervised SimCSE objective from the paper.

You can adjust hyper‑parameters (epochs, batch size, LR, warmup_ratio) depending on GPU memory and how long you want to train.

> **For the progress report:** you can mention that we trained unsupervised and supervised English SimCSE on Vast.ai using `bert-base-uncased` and recorded Spearman correlations on STS‑B as our main metric.

---

## 7. English STS‑B evaluation

We use **`evaluate_english.py`** plus a small bash script to run evaluation for STS‑B.

From the repo root:

```bash
bash bash/evaluate_english.sh
```

(If the script name changes, check the `bash/` folder.)

The script will:

- Load the checkpoint from `outputs/en/unsup` or `outputs/en/sup` (depending on the config inside the script)
- Compute sentence embeddings for STS‑B pairs
- Compute **Spearman correlation** between cosine similarity and gold scores
- Write a summary (per split) into the corresponding `outputs/en/.../eval/` folder

These numbers are what we will later compare across:

- unsupervised vs supervised English
- English vs Chinese vs Hindi (once we add multilingual experiments)

---

## 8. Shutting down the Vast.ai instance

When you are done:

1. Save / push any changes you want to keep:

   ```bash
   git status
   git add ...
   git commit -m "message"
   git push
   ```

2. Go back to Vast.ai web UI and **stop** or **destroy** the instance, so it stops charging.

---

## 9. What to mention in the report

Some phrases you can reuse in the progress report:

- We rented GPU instances (A5000/RTX4090) from Vast.ai and connected via VS Code Remote‑SSH.
- For computing resources, I tried Google Cloud (complex setup procedure & no free node found) and my brother's PC with RTX5060 GPU instance(too new that it supports sm120 while current pytorch versions only supports up to sm90, search yourself for better explanation).
- The codebase is organized in a GitHub repository (`SimCSE-multilanguages`, branch `zian`) with scripts for data download, training, and evaluation. 
- branch `main` is currently storing previous code version that adapts to SeaWulf. Do mention that we tried but only A100 nodes have GPUs and it was too difficult to obtain one single GPU node while there were plenty of other more important projects waiting in queue. 
- We reproduced the **unsupervised** and **supervised** SimCSE training objectives on **English** using `bert-base-uncased`. (unsupervised outputs are available in outputs folder, corr ~0.2-0.4, still working on supb version)
- We used `wiki1m_for_simcse` (unsupervised) and NLI triples (`nli_for_simcse.csv`) for supervised training.
- We evaluated models on STS‑B (English) using Spearman correlation between cosine similarity of sentence embeddings and human‑annotated similarity scores.
- Next steps: 
    1. train & evaluate **supervised** SimCSE in **English**
    2. extend the same pipeline to **Chinese** and **Hindi** sentence embedding benchmarks.

You don’t have to write code details in the report; just focus on the workflow and what we measured.