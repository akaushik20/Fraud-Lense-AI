import gradio as gr

if __name__=="__main__":
    demo = gr.Interface(
    inputs="text",
    outputs="text",
    title="Greeter App",
    description="Type your name below to get a greeting."
    )
    demo.launch()