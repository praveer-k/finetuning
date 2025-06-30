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
