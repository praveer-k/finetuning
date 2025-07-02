# fine tuning

```bash
git clone <link to the repository>
uv sync
brew install ggerganov/ggerganov/llama.cpp
cd ..
git clone https://github.com/ggerganov/llama.cpp 
cd llama.cpp
uv run python convert_hf_to_gguf.py ./tinyllama-merged --outfile ../finetuning/models/tinyllama-chat.gguf --outtype f16
cd ../finetuning
llama-server -m ~/models/
```
https://llama-cpp-python.readthedocs.io/en/latest/install/macos/

https://www.oecd.org/content/dam/oecd/en/publications/reports/2022/01/oecd-transfer-pricing-guidelines-for-multinational-enterprises-and-tax-administrations-2022_57104b3a/0e655865-en.pdf

https://artificialintelligenceact.eu/wp-content/uploads/2021/08/The-AI-Act.pdf

https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf

uv run format.py