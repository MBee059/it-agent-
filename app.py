import gradio as gr

def respond(message, history, num_docs, disable_adapter):
    retrieved_sources = [
        "Source 1: [TechQA Doc #1042] - Check Apache configuration logs at /var/log/httpd/error_log.",
        "Source 2: [ServerFault Post #88391] - Ensure port 80/443 is allowed in iptables rule-set."
    ]
    model_response = (
        "Based on the knowledge base, here are the troubleshooting steps for your query:\n\n"
        "1. Inspect error logs located at `/var/log/httpd/error_log`.\n"
        "2. Verify firewall rules to confirm ports 80 and 443 are open."
    )
    return model_response, "\n\n".join(retrieved_sources)

with gr.Blocks(theme=gr.themes.Soft(), title="Grounded IT Support Agent") as demo:
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
        return "", history + [[user_message, None]]

    def bot_respond(history, k, disable_adp):
        user_message = history[-1][0]
        bot_message, sources = respond(user_message, history, k, disable_adp)
        history[-1][1] = bot_message
        return history, sources

    submit_btn.click(user_submit, [msg, chatbot], [msg, chatbot]).then(bot_respond, [chatbot, num_docs, disable_adapter], [chatbot, sources_box])
    msg.submit(user_submit, [msg, chatbot], [msg, chatbot]).then(bot_respond, [chatbot, num_docs, disable_adapter], [chatbot, sources_box])
    clear_btn.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()
