import gradio as gr
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

def upscale_gradio(input_image):
    output_image_path = "upscaled_image.png"
    input_image_path = "input_image.png"

    if np.max(input_image) > 1:
        cv2.imwrite(input_image_path, np.array(input_image))
    else:
        cv2.imwrite(input_image_path, np.array(input_image) * 255)

    upscale_image(input_image_path, output_image_path, "esrgan-v1-x2plus", "sk-snxMfG2LVsLyezE46G9GSxgEBMy9a2rBVsIBQWCrd3n6L5pP", width=1024)
    return output_image_path

def gray(input_img):
    image_path = 'image_gray.png'
    image = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(image_path, image)
    return image_path

def adjust_brightness_and_darkness(input_img, brightness_enabled, brightness_value, darkness_enabled, darkness_value):
    image = input_img.copy()

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
    height, width = img_input.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1)
    rotated_image = cv2.warpAffine(img_input, rotation_matrix, (width, height))
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

with gr.Blocks() as demo:
    gr.Markdown("IMAGE EDITOR")
    
    with gr.Tab("Remove and Replace Background"):
        subject_img_input = gr.Image(type="filepath")
        background_img_input = gr.Image(type="filepath")
        blur_radius_slider = gr.Slider(0, 100, label="Blur Radius")
        replace_bg_checkbox = gr.Checkbox(label="Replace Background")
        use_color_picker_checkbox = gr.Checkbox(label="Use Color Picker")
        color_picker = gr.ColorPicker(label="Background Color")
        processed_img_output = gr.Image()
        submit_button = gr.Button("Submit")
        submit_button.click(remove_and_replace_background, inputs=[subject_img_input, background_img_input, blur_radius_slider, replace_bg_checkbox, use_color_picker_checkbox, color_picker], outputs=processed_img_output)
    
    with gr.Tab("Upscale Image"):
        img_input_upscale = gr.Image()
        img_output_upscale = gr.Image()
        img_button_upscale = gr.Button("Submit")
        img_button_upscale.click(upscale_gradio, inputs=img_input_upscale, outputs=img_output_upscale)
    
    with gr.Tab("Gray"):
        img_input_gray = gr.Image()
        img_output_gray = gr.Image()
        img_button_gray = gr.Button("Submit")
        img_button_gray.click(gray, inputs=img_input_gray, outputs=img_output_gray)
    
    with gr.Tab("Brightness and Darkness"):
        img_input_contrast = gr.Image()
        brightness_checkbox = gr.Checkbox(label="Enable Brightness Adjustment")
        brightness_slider = gr.Slider(0, 255, label="Brightness Value")
        darkness_checkbox = gr.Checkbox(label="Enable Darkness Adjustment")
        darkness_slider = gr.Slider(0, 255, label="Darkness Value")
        img_output_contrast = gr.Image()
        img_button_contrast = gr.Button("Submit")
        img_button_contrast.click(adjust_brightness_and_darkness, inputs=[img_input_contrast, brightness_checkbox, brightness_slider, darkness_checkbox, darkness_slider], outputs=img_output_contrast)
    
    with gr.Tab("Rotate Image"):
        temp_slider = gr.Slider(minimum=0, maximum=360, value=90, step=90, interactive=True, label="Slide me")
        img_input_rotate = gr.Image()
        img_output_rotate = gr.Image()
        img_button_rotate = gr.Button("Submit")
        img_button_rotate.click(rotate_image, inputs=[img_input_rotate, temp_slider], outputs=img_output_rotate)
    
    with gr.Tab("IOPaint"):
        gr.Markdown("## Object Remover Use IOPaint, Thanks to https://github.com/Sanster/IOPaint For This Tool")

        with gr.Column():
            gr.Markdown("### This Is a Video For How To Get Ngrok Token")
            ngrok_video_button = gr.Button("Click Here")
            ngrok_video_button.click(lambda: webbrowser.open_new_tab("https://youtu.be/rRBNdwTQ9HQ?si=TsZIp6vWxPbsIfXz"))

            ngrok_token_input = gr.Textbox(label="Insert Ngrok Token")
            model_input = gr.Dropdown(choices=["lama", "mat", "migan", "runwayml/stable-diffusion-inpainting", "Sanster/PowerPaint-V1-stable-diffusion-inpainting", "Uminosachi/realisticVisionV51_v51VAE-inpainting", "Sanster/anything-4.0-inpainting", "redstonehero/dreamshaper-inpainting", "Sanster/AnyText", "timbrooks/instruct-pix2pix", "Fantasy-Studio/Paint-by-Example"], label="Model", value="lama")
            device_input = gr.Dropdown(choices=["cuda", "cpu"], label="Device", value="cpu")
            enable_interactive_seg_input = gr.Checkbox(label="Enable Interactive Segmentation")
            interactive_seg_model_input = gr.Dropdown(choices=["sam_hq_vit_b", "sam_hq_vit_l", "sam_hq_vit_h"], label="Interactive Segmentation Model", value="sam_hq_vit_b")
            enable_remove_bg_input = gr.Checkbox(label="Enable Remove Background")
            remove_bg_model_input = gr.Dropdown(choices=["u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta", "isnet-general-use", "briaai/RMBG-1.4"], label="Remove Background Model", value="briaai/RMBG-1.4")
            enable_realesrgan_input = gr.Checkbox(label="Enable RealESRGAN")
            realesrgan_model_input = gr.Dropdown(choices=["realesr-general-x4v3", "RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"], label="RealESRGAN Model", value="realesr-general-x4v3")
            enable_gfpgan_input = gr.Checkbox(label="Enable GFPGAN")
            enable_restoreformer_input = gr.Checkbox(label="Enable RestoreFormer")

            run_button = gr.Button("Start IOPaint Server")
            output_text = gr.Textbox()

            run_button.click(
                run_iopaint_server,
                inputs=[
                    ngrok_token_input, model_input, device_input, enable_interactive_seg_input, interactive_seg_model_input, 
                    enable_remove_bg_input, remove_bg_model_input, enable_realesrgan_input, realesrgan_model_input, 
                    enable_gfpgan_input, enable_restoreformer_input
                ],
                outputs=output_text
            )

demo.launch(share=True)
