# Grounded RAG IT Support Agent

A high-precision, hallucination-free IT support generation agent built on **Qwen2.5-7B** (4-bit Unsloth quantization) and **FAISS**.

## Technical Finding: The Alignment Tax

During evaluation, comparative tests revealed that fine-tuning the LoRA adapter on conversational IT support logs injected a behavioral prior to *diagnose and elaborate*, which interfered with strict RAG text extraction constraints.

### Solution
The pipeline utilizes a hybrid approach:
- **LoRA Adapter:** Handles intent classification and query formulation.
- **Base Model:** Executed via `model.disable_adapter()` during RAG generation to ensure 100%% source-text fidelity without parametric hallucinations.

## Repository Structure
- `techqa/`: Raw and parsed domain dataset splits.
- `format_prompts.py` & `schema_reformat.py`: ChatML conversion utilities.
- `parse_serverfault.py`: Data ingestion pipelines.
