import streamlit as st
from diffusers import DiffusionPipeline
import torch

# Initialize the Stable Diffusion pipeline
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
pipe.to("cuda")

# Set page configuration
st.set_page_config(
    page_title="Text to Image Generator",
    page_icon="icon4.jpg",  # Ensure the file exists in the working directory
    layout="centered"
)

# Streamlit layout
col1, col2 = st.columns([1, 3])  # Adjust column width ratios for better layout

with col1: 
    st.image("images.jpg", width=170)  # Add a logo or an icon if needed

with col2: 
    st.write("")
    st.write("")
    st.title("Text to Image Generator")

st.write("Generate stunning images from text prompts using Stable Diffusion!")

# Input prompt from the user
prompt = st.text_input("Enter your image description (prompt):")

# Optional negative prompt
negative_prompt = st.text_input(
    "Optional: Enter a negative prompt (things to avoid):",
    placeholder="E.g., Avoid including any water elements in the scene."
)

# Button to generate the image
if st.button("Generate Image"):
    if prompt:
        with st.spinner("Generating image..."):
            try:
                # Generate image
                result = pipe(prompt=prompt, negative_prompt=negative_prompt).images[0]
                st.image(result, caption="Generated Image", use_column_width=True)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a prompt to generate an image.")
