import gradio as gr
import torch

# Note: This file assumes 'model', 'tokenizer', and 'faiss_index' 
# are accessible in your notebook environment when imported/launched.

def respond(message, history, num_docs, disable_adapter):
    try:
        # 1. Retrieve top-K documents from your FAISS index
        # Adjust these variables to match your exact FAISS retriever function
        retrieved_docs = faiss_index.similarity_search(message, k=num_docs)
        context_str = "\n\n".join([f"Doc {i+1}: {doc.page_content}" for i, doc in enumerate(retrieved_docs)])
        
        # 2. Toggle LoRA adapter dynamically
        if disable_adapter and hasattr(model, "disable_adapters"):
            with model.disable_adapters():
                response_text = generate_qwen_response(message, context_str)
        else:
            response_text = generate_qwen_response(message, context_str)
            
        return response_text, context_str

    except Exception as e:
        # Fallback error messaging if index/model isn't fully initialized in memory
        return f"Error during inference: {str(e)}", "No context retrieved."

def generate_qwen_response(query, context):
    prompt = f"<|im_start|>system\nYou are a helpful IT support assistant. Answer the user question based only on the provided context.<|im_end|>\n<|im_start|>user\nContext:\n{context}\n\nQuestion: {query}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Grounded RAG IT Support Agent\n**Architecture:** Qwen2.5-7B (Unsloth) + FAISS Vector Retrieval")
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=450)
            msg = gr.Textbox(placeholder="Describe your IT issue...", label="User Query")
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear Chat")
        with gr.Column(scale=1):
            gr.Markdown("### RAG Control & Sources")
            num_docs = gr.Slider(minimum=1, maximum=5, value=2, step=1, label="FAISS Top-K Contexts")
            disable_adapter = gr.Checkbox(value=True, label="Disable LoRA Adapter (Grounded Mode)")
            sources_box = gr.Textbox(label="Retrieved Context Documents", interactive=False, lines=10)

    def user_submit(user_message, history):
        history = history or []
        history.append({"role": "user", "content": user_message})
        return "", history

    def bot_respond(history, k, disable_adp):
        user_message = history[-1]["content"] if history else ""
        bot_message, sources = respond(user_message, history, k, disable_adp)
        history.append({"role": "assistant", "content": bot_message})
        return history, sources

    submit_btn.click(user_submit, [msg, chatbot], [msg, chatbot]).then(
        bot_respond, [chatbot, num_docs, disable_adapter], [chatbot, sources_box]
    )
    msg.submit(user_submit, [msg, chatbot], [msg, chatbot]).then(
        bot_respond, [chatbot, num_docs, disable_adapter], [chatbot, sources_box]
    )
    clear_btn.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(share=True)
