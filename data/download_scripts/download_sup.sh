#!/bin/bash
set -e

# Download supervised NLI data for English, Chinese, Hindi into their respective folders.

echo "Downloading multilingual NLI supervised data..."

LANGS=("english" "chinese" "hindi")

for LANG in "${LANGS[@]}"; do
    TARGET_DIR="data/${LANG}/sup"
    mkdir -p "${TARGET_DIR}"
    echo ""
    echo "---> Processing language: ${LANG}"

    # Select URL + output filename per language
    if [[ "$LANG" == "english" ]]; then
        URL="https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/nli_for_simcse.csv"
        OUT="nli_english.csv"
        echo "English NLI source: $URL"

    elif [[ "$LANG" == "chinese" ]]; then
        URL="https://huggingface.co/datasets/Peach23333/simcse-nli-multilingual/resolve/main/chinese.csv"
        OUT="nli_chinese.csv"
        echo "Chinese NLI source: $URL"

    elif [[ "$LANG" == "hindi" ]]; then
        URL="https://huggingface.co/datasets/Peach23333/simcse-nli-multilingual/resolve/main/hindi.csv"
        OUT="nli_hindi.csv"
        echo "Hindi NLI source: $URL"
    fi

    # Only attempt download if URL is provided
    if [[ -n "$URL" ]]; then
        echo "Downloading from: $URL"
        wget -O "${TARGET_DIR}/${OUT}" "$URL"
        echo "Saved to ${TARGET_DIR}/${OUT}"
    else
        echo "Skipped downloading for ${LANG} (URL missing)."
    fi
done

echo ""
echo "All multilingual NLI downloads finished."