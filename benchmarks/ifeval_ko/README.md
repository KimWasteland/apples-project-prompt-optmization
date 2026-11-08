---
dataset_info:
  features:
  - name: key
    dtype: int64
  - name: prompt
    dtype: string
  - name: instruction_id_list
    sequence: string
  - name: kwargs
    list:
    - name: capital_frequency
      dtype: 'null'
    - name: capital_relation
      dtype: 'null'
    - name: end_phrase
      dtype: string
    - name: first_word
      dtype: string
    - name: forbidden_words
      sequence: string
    - name: frequency
      dtype: int64
    - name: keyword
      dtype: string
    - name: keywords
      sequence: string
    - name: language
      dtype: string
    - name: let_frequency
      dtype: 'null'
    - name: let_relation
      dtype: 'null'
    - name: letter
      dtype: 'null'
    - name: nth_paragraph
      dtype: int64
    - name: num_bullets
      dtype: int64
    - name: num_highlights
      dtype: int64
    - name: num_paragraphs
      dtype: int64
    - name: num_placeholders
      dtype: int64
    - name: num_sections
      dtype: int64
    - name: num_sentences
      dtype: int64
    - name: num_words
      dtype: int64
    - name: postscript_marker
      dtype: string
    - name: prompt_to_repeat
      dtype: string
    - name: relation
      dtype: string
    - name: section_spliter
      dtype: string
  splits:
  - name: train
    num_bytes: 169148
    num_examples: 342
  download_size: 67128
  dataset_size: 169148
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
language:
- ko
license: apache-2.0
task_categories:
- text-generation
tags:
- InstructionFollowing
- IF
size_categories:
- n<1K
---

# IFEval-Ko: Korean Instruction-Following Benchmark for LLMs

> This dataset is originated from [IFEval](https://huggingface.co/datasets/google/IFEval/) Dataset

[Korean Version README](https://huggingface.co/datasets/allganize/IFEval-Ko/blob/main/README_Ko.md)

`IFEval-Ko` is a Korean adaptation of Google's open-source **IFEval** benchmark utilized with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework. 
It enables evaluation of large language models (LLMs) for their instruction-following capabilities in the Korean language.

## Dataset Details

- **Original Source**: [google/IFEval](https://huggingface.co/datasets/google/IFEval/)
- **Adaptation Author**: [Allganize Inc. LLM TEAM](https://www.allganize.ai/) | Keonmo Lee
- **Repository**: [allganize/IFEval-Ko](https://huggingface.co/datasets/allganize/IFEval-Ko)
- **Languages**: Korean
- **Translation Tool**: GPT-4o
- **License**: Follows original [google/IFEval](https://huggingface.co/datasets/google/IFEval/) license
- **Benchmarked with**: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)



## Benchmark Scores
![plot](https://huggingface.co/datasets/allganize/IFEval-Ko/resolve/main/benchmark-score-dot-plot.png)

## How to Use

Clone `lm-evaluation-harness` and create the `ifeval_ko` folder into the `lm_eval/tasks` directory.

```bash
# Install lm-evaluation-harness and task dependencies
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
pip install langdetect immutabledict 

# Download task files from Hugging Face Repository
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='allganize/IFEval-Ko',
    repo_type='dataset',
    local_dir='lm_eval/tasks/',
    allow_patterns='ifeval_ko/*',
    local_dir_use_symlinks=False
) "

```

***Please check usage of `lm_eval` on original [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) repository before use.***

### Evaluation with Hugging Face Transformers

```bash
lm_eval --model hf \
    --model_args pretrained={HF_MODEL_REPO} \ 
    --tasks ifeval_ko \
    --device cuda:0 \
    --batch_size 8
```
e.g., {HF_MODEL_REPO} =  google/gemma-3-4b-it

### Evaluation with vLLM

Install vLLM-compatible backend:

```bash
pip install lm-eval[vllm]
```

Then run the evaluation:

```bash
lm_eval --model vllm \
    --model_args pretrained={HF_MODEL_REPO},trust_remote_code=True \
    --tasks ifeval_ko
```

---

## Modifications from Original IFEval

### Data Transformation

- **Translation**: Prompts were translated using the **gpt-4o** model, with a custom prompt designed to preserve the original structure.
- **Removed Items**:  
  - 84 case-sensitive (`change_case`) tasks  
  - 28 alphabet-dependent (`letter_frequency`) tasks  
  - Other erroneous or culturally inappropriate prompts

- **Unit Conversions**:  
  - Gallons → Liters  
  - Feet/Inches → Meters/Centimeters  
  - Dollars → Korean Won (USD:KRW ≈ 1:1500)

- **Standardizations**:  
  - Unified headings \<\<Title\>\> or \<\<title\>\> to \<\<제목\>\>  
  - Ensured consistent tone across answers

### Code Changes

- Translated instruction options:
  - `instruction._CONSTRAINED_RESPONSE_OPTIONS`
  - `instruction._ENDING_OPTIONS`

- Modified scoring classes:
  - `KeywordChecker`, `KeywordFrequencyChecker`, `ParagraphFirstWordCheck`, `KeySentenceChecker`, `ForbiddenWords`, `RepeatPromptThenAnswer`, `EndChecker`
  - Applied `unicodedata.normalize('NFC', ...)` for normalization
  - Removed fallback keyword generator for missing fields (now throws error)

- Removed dependency on `nltk` by modifying `count_sentences()` logic

---

## Evaluation Metrics

Please refer to [original IFEval paper](https://arxiv.org/pdf/2311.07911):

### Strict vs. Loose Accuracy

- **Strict**: Checks if the model followed the instruction *without* transformation of response.
- **Loose**: Applies 3 transformations to response before comparison:
  1. Remove markdown symbols (`*`, `**`)
  2. Remove the first line (e.g., "Here is your response:")
  3. Remove the last line (e.g., "Did that help?")

  A sample is marked correct if *any* of the 8 combinations match.

### Prompt-level vs. Instruction-level

- **Prompt-level**: All instructions in a single prompt must be followed to count as True.
- **Instruction-level**: Evaluates each instruction separately for finer-grained metrics.

Created by  
Allganize LLM TEAM  
[**Keonmo Lee (이건모)**](https://huggingface.co/whatisthis8047)


### Original Citation Information
```bibtex
@misc{zhou2023instructionfollowingevaluationlargelanguage,
      title={Instruction-Following Evaluation for Large Language Models}, 
      author={Jeffrey Zhou and Tianjian Lu and Swaroop Mishra and Siddhartha Brahma and Sujoy Basu and Yi Luan and Denny Zhou and Le Hou},
      year={2023},
      eprint={2311.07911},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2311.07911}, 
}
```