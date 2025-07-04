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

CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" uv run pip install llama-cpp-python
uv run huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False


```bash
uv run format.py
uv run merge.py
uv run main.py
uv run test.py
```
