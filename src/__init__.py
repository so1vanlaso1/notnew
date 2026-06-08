"""NEWpipeline — cascade NL-QA pipeline.

Two 4B judges (Qwen3.5-4B + Gemma-E2B) answer each question. If they agree,
the answer stands and Qwen writes the explanation. If they disagree, both 4B
models are UNLOADED and a single larger model (Gemma-E4B, the "8B") is loaded to
decide and explain. At most {two 4B models} OR {one 8B model} are resident at a
time, so peak VRAM stays bounded.
"""
