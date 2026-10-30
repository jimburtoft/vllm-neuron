# Workload data

| File | What |
|---|---|
| `wiki_2048.json`, `wiki_4096.json` | 8 prompts each, as token IDs for `google/gemma-3-12b-it`: the chat template with the system prompt "Translate the text below to Hindi." and the start of an English Wikipedia article, trimmed so the whole prompt is exactly 2,048 / 4,096 tokens. |
| `ref_fp32_wiki_2048.json`, `ref_fp32_wiki_4096.json` | HF transformers fp32 greedy outputs for those prompts (CPU, eager attention, natural EOS, cap 2,048 / 4,096 tokens). No row reached EOS before the cap. Used by `fp32_gate.py`. |

Built by `long_workload.py prompts-wiki --lengths 2048 4096 --n 8` and `long_workload.py reference --natural-eos`.

## Attribution and license

The prompt text comes from the English Wikipedia articles listed under `source.articles` in the prompt files
(Anarchism, Albedo, Alabama, Achilles, Abraham Lincoln, Aristotle, Academy Awards, Altruism), taken from the
Hugging Face dataset `wikimedia/wikipedia` (config `20231101.en`). Wikipedia text is licensed under the
[Creative Commons Attribution-ShareAlike 4.0 License](https://creativecommons.org/licenses/by-sa/4.0/);
the authors are the contributors listed in each article's history on <https://en.wikipedia.org/>.
These files (the tokenized excerpts and the model's translations of them) are shared under the same license.
