import gradio as gr

def sample_func():
    return "My First app in Gradio AI!"

if __name__=="__main__":
    demo = gr.Interface(
        fn=sample_func,
        inputs=None,
        outputs="text",
        title="My First App with Gradio",
        description="Welcome to my Gradio app."
        )
    demo.launch()