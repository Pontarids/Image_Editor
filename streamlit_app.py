import streamlit as st
from rembg import remove
from PIL import Image as PILImage, ImageFilter
from io import BytesIO
import requests
import cv2
import numpy as np
import os
import asyncio
from pyngrok import ngrok
import threading
import webbrowser

def remove_and_replace_background(subject, background, blur_radius, replace_background, use_color_picker, color):
    with open(subject, 'rb') as subject_img_file:
        subject_img = subject_img_file.read()
    subject_no_bg = remove(subject_img, alpha_matting=True, alpha_matting_foreground_threshold=10)
    subject_img_no_bg = PILImage.open(BytesIO(subject_no_bg)).convert("RGBA")
    
    if replace_background:
        if use_color_picker:
            background_img = PILImage.new("RGBA", subject_img_no_bg.size, color)
        else:
            background_img = PILImage.open(background).convert("RGBA")
            background_img = background_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            background_img = background_img.resize(subject_img_no_bg.size)
        combined_img = PILImage.alpha_composite(background_img, subject_img_no_bg)
        combined_img.save("combined_image.png")
        return "combined_image.png"
    else:
        subject_img_no_bg.save("subject_no_bg.png")
        return "subject_no_bg.png"

def upscale_image(input_image_path, output_image_path, engine_id, api_key, api_host="https://api.stability.ai", width=None, height=None):
    with open(input_image_path, "rb") as file:
        image_data = file.read()

    headers = {
        "Accept": "image/png",
        "Authorization": f"Bearer {api_key}",
    }

    files = {
        "image": image_data,
    }

    data = {}
    if width:
        data["width"] = width
    if height:
        data["height"] = height

    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/image-to-image/upscale",
        headers=headers,
        files=files,
        data=data
    )

    if response.status_code != 200:
        raise Exception(f"Non-200 response: {response.text}")

    try:
        nparr = np.frombuffer(response.content, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise Exception(f"Invalid image data: {e}")

    cv2.imwrite(output_image_path, img_np)
    return output_image_path

def upscale_streamlit(input_image):
    output_image_path = "upscaled_image.png"
    input_image_path = "input_image.png"

    cv2.imwrite(input_image_path, np.array(input_image)[:, :, ::-1])

    upscale_image(input_image_path, output_image_path, "esrgan-v1-x2plus", "sk-snxMfG2LVsLyezE46G9GSxgEBMy9a2rBVsIBQWCrd3n6L5pP", width=1024)
    return output_image_path

def gray(input_img):
    image_path = 'image_gray.png'
    image = cv2.cvtColor(np.array(input_img)[:, :, ::-1], cv2.COLOR_RGB2GRAY)
    cv2.imwrite(image_path, image)
    return image_path

def adjust_brightness_and_darkness(input_img, brightness_enabled, brightness_value, darkness_enabled, darkness_value):
    image = np.array(input_img)[:, :, ::-1].copy()

    if brightness_enabled:
        mat = np.ones(image.shape, dtype='uint8') * brightness_value
        image = cv2.add(image, mat)

    if darkness_enabled:
        mat = np.ones(image.shape, dtype='uint8') * darkness_value
        image = cv2.subtract(image, mat)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_path = 'adjusted_image.png'
    cv2.imwrite(image_path, image_rgb)
    return image_path

def rotate_image(img_input, degrees):
    image_path = 'rotated.png'
    image = np.array(img_input)[:, :, ::-1]
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    rotated_image_rgb = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(image_path, rotated_image_rgb)
    return image_path

def install_packages():
    os.system("pip install iopaint pyngrok > install_logs.txt")
    os.system("iopaint install-plugins-packages > plugins_install_logs.txt")

def start_ngrok(ngrok_token):
    if not ngrok_token:
        return "Ngrok token not set"
    
    ngrok.set_auth_token(ngrok_token)
    active_tunnels = ngrok.get_tunnels()
    for tunnel in active_tunnels:
        public_url = tunnel.public_url
        ngrok.disconnect(public_url)
    url = ngrok.connect(addr="8000", bind_tls=True)
    return url

def start_iopaint(ngrok_token, model, device, enable_interactive_seg, interactive_seg_model, enable_remove_bg, remove_bg_model, enable_realesrgan, realesrgan_model, enable_gfpgan, enable_restoreformer):
    if device == 'cuda':
        os.environ.update({'LD_LIBRARY_PATH': '/usr/lib64-nvidia'})

    url = start_ngrok(ngrok_token)
    if url == "Ngrok token not set":
        return url

    cmds = [
        'iopaint', 'start', '--model', model, '--device', device, '--port', '8000',
        '--interactive-seg-device', device,
        '--interactive-seg-model', interactive_seg_model,
        '--remove-bg-model', remove_bg_model,
        '--realesrgan-model', realesrgan_model,
        '--realesrgan-device', device,
        '--gfpgan-device', device,
        '--restoreformer-device', device
    ]

    if enable_interactive_seg:
        cmds.append('--enable-interactive-seg')
    if enable_remove_bg:
        cmds.append('--enable-remove-bg')
    if enable_realesrgan:
        cmds.append('--enable-realesrgan')
    if enable_gfpgan:
        cmds.append('--enable-gfpgan')
    if enable_restoreformer:
        cmds.append('--enable-restoreformer')

    async def run_process():
        print('>>> starting', *cmds)
        p = await asyncio.create_subprocess_exec(
            *cmds,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def pipe(lines):
            async for line in lines:
                print(line.strip().decode('utf-8'))

        await asyncio.gather(
            pipe(p.stdout),
            pipe(p.stderr),
        )

    threading.Thread(target=lambda: asyncio.run(run_process())).start()

    def open_url(url):
        webbrowser.open(url, new=0)

    open_url("http://localhost:8000")
    return "IOPaint server started on http://localhost:8000"

def run_iopaint_server(ngrok_token, model, device, enable_interactive_seg, interactive_seg_model, enable_remove_bg, remove_bg_model, enable_realesrgan, realesrgan_model, enable_gfpgan, enable_restoreformer):
    install_packages()
    result = start_iopaint(ngrok_token, model, device, enable_interactive_seg, interactive_seg_model, enable_remove_bg, remove_bg_model, enable_realesrgan, realesrgan_model, enable_gfpgan, enable_restoreformer)
    return result

st.title("IMAGE EDITOR")

with st.sidebar:
    selected_tab = st.selectbox("Choose Tab", ["Remove and Replace Background", "Upscale Image", "Gray", "Brightness and Darkness", "Rotate Image", "IOPaint"])

if selected_tab == "Remove and Replace Background":
    st.header("Remove and Replace Background")
    subject_img_input = st.file_uploader("Upload Subject Image", type=["jpg", "jpeg", "png"])
    background_img_input = st.file_uploader("Upload Background Image", type=["jpg", "jpeg", "png"])
    blur_radius = st.slider("Blur Radius", 0, 100)
    replace_bg = st.checkbox("Replace Background")
    use_color_picker = st.checkbox("Use Color Picker")
    color = st.color_picker("Background Color")
    if st.button("Submit"):
        if subject_img_input:
            subject_img_path = subject_img_input.name
            with open(subject_img_path, "wb") as f:
                f.write(subject_img_input.getbuffer())
            if background_img_input:
                background_img_path = background_img_input.name
                with open(background_img_path, "wb") as f:
                    f.write(background_img_input.getbuffer())
                output_path = remove_and_replace_background(subject_img_path, background_img_path, blur_radius, replace_bg, use_color_picker, color)
            else:
                output_path = remove_and_replace_background(subject_img_path, None, blur_radius, replace_bg, use_color_picker, color)
            st.image(output_path)

elif selected_tab == "Upscale Image":
    st.header("Upscale Image")
    img_input_upscale = st.file_uploader("Upload Image to Upscale", type=["jpg", "jpeg", "png"])
    if st.button("Submit"):
        if img_input_upscale:
            input_image = PILImage.open(img_input_upscale)
            output_path = upscale_streamlit(input_image)
            st.image(output_path)

elif selected_tab == "Gray":
    st.header("Convert Image to Gray")
    img_input_gray = st.file_uploader("Upload Image to Convert to Gray", type=["jpg", "jpeg", "png"])
    if st.button("Submit"):
        if img_input_gray:
            input_image = PILImage.open(img_input_gray)
            output_path = gray(input_image)
            st.image(output_path)

elif selected_tab == "Brightness and Darkness":
    st.header("Adjust Brightness and Darkness")
    img_input_contrast = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    brightness_enabled = st.checkbox("Enable Brightness Adjustment")
    brightness_value = st.slider("Brightness Value", 0, 255)
    darkness_enabled = st.checkbox("Enable Darkness Adjustment")
    darkness_value = st.slider("Darkness Value", 0, 255)
    if st.button("Submit"):
        if img_input_contrast:
            input_image = PILImage.open(img_input_contrast)
            output_path = adjust_brightness_and_darkness(input_image, brightness_enabled, brightness_value, darkness_enabled, darkness_value)
            st.image(output_path)

elif selected_tab == "Rotate Image":
    st.header("Rotate Image")
    img_input_rotate = st.file_uploader("Upload Image to Rotate", type=["jpg", "jpeg", "png"])
    degrees = st.slider("Rotation Angle", 0, 360, 90)
    if st.button("Submit"):
        if img_input_rotate:
            input_image = PILImage.open(img_input_rotate)
            output_path = rotate_image(input_image, degrees)
            st.image(output_path)

elif selected_tab == "IOPaint":
    st.header("Start IOPaint Server")
    ngrok_token = st.text_input("Insert Ngrok Token")
    model = st.selectbox("Model", ["lama", "mat", "migan", "runwayml/stable-diffusion-inpainting", "Sanster/PowerPaint-V1-stable-diffusion-inpainting", "Uminosachi/realisticVisionV51_v51VAE-inpainting", "Sanster/anything-4.0-inpainting", "redstonehero/dreamshaper-inpainting", "Sanster/AnyText", "timbrooks/instruct-pix2pix", "Fantasy-Studio/Paint-by-Example"])
    device = st.selectbox("Device", ["cuda", "cpu"])
    enable_interactive_seg = st.checkbox("Enable Interactive Segmentation")
    interactive_seg_model = st.selectbox("Interactive Segmentation Model", ["sam_hq_vit_b", "sam_hq_vit_l", "sam_hq_vit_h"])
    enable_remove_bg = st.checkbox("Enable Remove Background")
    remove_bg_model = st.selectbox("Remove Background Model", ["u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta", "isnet-general-use", "briaai/RMBG-1.4"])
    enable_realesrgan = st.checkbox("Enable RealESRGAN")
    realesrgan_model = st.selectbox("RealESRGAN Model", ["realesr-general-x4v3", "RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"])
    enable_gfpgan = st.checkbox("Enable GFPGAN")
    enable_restoreformer = st.checkbox("Enable RestoreFormer")

    if st.button("Start IOPaint Server"):
        result = run_iopaint_server(ngrok_token, model, device, enable_interactive_seg, interactive_seg_model, enable_remove_bg, remove_bg_model, enable_realesrgan, realesrgan_model, enable_gfpgan, enable_restoreformer)
        st.text(result)

if __name__ == '__main__':
    st.set_page_config(page_title="Image Editor", layout="wide")
    st.title("Image Editor")
