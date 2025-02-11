import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

def add_directory_and_subdirectories_to_sys_path(base_path):
    """
    지정된 디렉터리와 하위 디렉터리를 sys.path에 추가합니다.
    :param base_path: 기준 디렉터리의 절대 경로
    """
    for root, dirs, files in os.walk(base_path):
        if root not in sys.path:  # 중복 방지
            sys.path.append(root)

# 절대 경로로 기본 디렉터리 지정
base_directory = "/root/app"
# 경로 및 하위 디렉터리 모두 sys.path에 추가
add_directory_and_subdirectories_to_sys_path(base_directory)

import base64
from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
import json
import numpy as np
import random
import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths
import node_helpers
from nodes import CLIPTextEncode, VAEDecode, CheckpointLoaderSimple, LoraLoader, common_ksampler
from datetime import datetime
import threading


app = Flask(__name__)

# 샘플러와 스케줄러 옵션 정의
SAMPLER_NAMES = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
                 "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral",
                 "dpmpp_sde", "dpmpp_2m", "ddim", "uni_pc", "uni_pc_bh2"]

SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]

# 전역 변수로 모델들을 초기화
model = None
clip = None
vae = None


# 작업 상태를 관리하기 위한 Lock과 상태 변수
endpoint_locks = {
    "A" : threading.Lock(),
    "B": threading.Lock()
}


def initialize_models(model_type="A"):

    if model_type == "A":

        global model, clip, vae

        try:
            with torch.no_grad():
                # CheckpointLoaderSimple 사용
                checkpoint_loader = CheckpointLoaderSimple()
                model, clip, vae = checkpoint_loader.load_checkpoint(ckpt_name="flux1-schnell-fp8.safetensors")

                # LoraLoader 사용
                lora_loader = LoraLoader()
                model, clip = lora_loader.load_lora(
                    model=model,
                    clip=clip,
                    lora_name="FluxDFaeTasticDetails.safetensors",
                    strength_model=1.0,
                    strength_clip=1.0
                )
                print(f"Model Loaded Completed with model_type '{model_type}'")
                return model, clip, vae
        except Exception as e:
            print(f"Error initializing models: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    elif model_type == "B":
        
        global model_b, clip_b, vae_b

        try:
            with torch.no_grad():
                # CheckpointLoaderSimple 사용
                checkpoint_loader = CheckpointLoaderSimple()
                model_b, clip_b, vae_b = checkpoint_loader.load_checkpoint(ckpt_name="flux1-schnell-fp8.safetensors")

                # LoraLoader 사용
                lora_loader = LoraLoader()
                model_b, clip_b = lora_loader.load_lora(
                    model=model_b,
                    clip=clip,
                    lora_name="FluxDFaeTasticDetails.safetensors",
                    strength_model=1.0,
                    strength_clip=1.0
                )

                print(f"Model Loaded Completed with model_type '{model_type}'")
                return model_b, clip_b, vae_b
        except Exception as e:
            print(f"Error initializing models: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

def create_empty_latent(width=1024, height=1024, batch_size=1):
    """SD3용 빈 레이턴트 이미지 생성"""
    device = comfy.model_management.intermediate_device()
    # Flux/SD3 모델용 16채널 latent
    latent = torch.zeros([batch_size, 16, height // 8, width // 8], device=device)
    return {"samples": latent}

def sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0):
    """KSampler 실행"""
    with torch.no_grad():
        result = common_ksampler(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent=latent,
            denoise=denoise,
            disable_noise=False,
            start_step=None,
            last_step=None,
            force_full_denoise=True
        )[0]

        return result["samples"]


def save_json_with_timestamp(data, seed):
    """JSON 파일을 타임스탬프와 시드값으로 저장"""
    # logs 디렉토리 생성 (없는 경우)
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{seed}_server.json"
    filepath = os.path.join(log_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return filepath


def save_image_with_timestamp(image, seed):
    """이미지를 타임스탬프와 시드값으로 저장"""
    # generated_images 디렉토리 생성 (없는 경우)
    image_dir = "generated_images"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_{timestamp}_{seed}_server.png"
    filepath = os.path.join(image_dir, filename)

    image.save(filepath)
    print(f"Image saved as: {filepath}")
    return filepath


@app.route('/api/t2i', methods=['POST'])
def text2image():
    global is_generating

    # 현재 작업 중인지 확인
    with generation_lock:
        if is_generating:
            return jsonify({
                'status': 'error',
                'message': 'Engine is currently busy. Please try again later.'
            }), 503
        is_generating = True

    try:
        with torch.no_grad():
            data = request.get_json()
            prompt = data.get('prompt', '')
            negative_prompt = data.get('negative_prompt', '')
            width = data.get('width', 1024)
            height = data.get('height', 1024)
            guidance_value = data.get('guidance_scale', 3.0)
            steps = data.get('steps', 20)
            cfg = data.get('cfg_scale', 8.0)
            seed = data.get('seed', random.randint(0, 2 ** 32 - 1))
            sampler_name = data.get('sampler_name', 'euler')
            scheduler = data.get('scheduler', 'normal')
            denoise = data.get('denoise', 1.0)

            print(f"Using seed: {seed}")

            # CLIP Text Encode
            clip_encode = CLIPTextEncode()
            positive_cond = clip_encode.encode(clip, prompt)[0]
            negative_cond = clip_encode.encode(clip, negative_prompt)[0]

            # FluxGuidance 적용
            positive_cond = node_helpers.conditioning_set_values(positive_cond, {"guidance": guidance_value})
            # SD3용 빈 레이턴트 생성
            latent = create_empty_latent(width=width, height=height)

            # Sampling 실행
            samples = sample(
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive_cond,
                negative=negative_cond,
                latent=latent,
                denoise=denoise
            )

            # VAE Decode
            vae_decoder = VAEDecode()
            decoded = vae_decoder.decode(vae, {"samples": samples})[0]

            # 텐서 형태 변환
            decoded = decoded.cpu().numpy()
            if len(decoded.shape) == 5:
                decoded = decoded[0, 0]
            elif len(decoded.shape) == 4:
                decoded = decoded[0]

            # uint8로 변환
            decoded = np.clip(decoded * 255, 0, 255).astype(np.uint8)

            # 이미지로 변환
            image = Image.fromarray(decoded)

            # 이미지 파일 저장
            image_filepath = save_image_with_timestamp(image, seed)

            # base64 포맷 변환
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # 응답 데이터 생성
            response_data = {
                'status': 'success',
                'image': img_str,
                'parameters': {
                    'prompt': prompt,
                    'negative_prompt': negative_prompt,
                    'width': width,
                    'height': height,
                    'guidance_scale': guidance_value,
                    'steps': steps,
                    'cfg_scale': cfg,
                    'seed': seed,
                    'sampler_name': sampler_name,
                    'scheduler': scheduler,
                    'denoise': denoise
                }
            }

            # JSON 파일 저장
            json_filepath = save_json_with_timestamp(response_data, seed)
            print(f"JSON saved as: {json_filepath}")

            return jsonify(response_data)

    except Exception as e:
        print("Error:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

    finally:
        # 작업 상태 해제
        with generation_lock:
            is_generating = False

@app.route('/api/t2i/state', methods=['GET'])
def get_generation_state():
    """이미지 생성 엔진의 현재 상태를 반환"""
    with generation_lock:
        status = "1" if is_generating else "0"

    return jsonify({"status": status})




def generation_image_by_model(
    model_var_name:str,
    clip_var_name:str,
    vae_var_name:str
):

    model = globals()[model_var_name] if model_var_name in globals() else None
    clip = globals()[clip_var_name] if model_var_name in globals() else None
    vae = globals()[vae_var_name] if model_var_name in globals() else None

    request_type = request.get_json().get('request_type', None)

    try:
        with torch.no_grad():
            data = request.get_json()
            prompt = data.get('prompt', '')
            negative_prompt = data.get('negative_prompt', '')
            width = data.get('width', 128)
            height = data.get('height', 128)
            guidance_value = data.get('guidance_scale', 3.0)
            steps = data.get('steps', 2)
            cfg = data.get('cfg_scale', 8.0)
            seed = data.get('seed', random.randint(0, 2 ** 32 - 1))
            sampler_name = data.get('sampler_name', 'euler')
            scheduler = data.get('scheduler', 'normal')
            denoise = data.get('denoise', 1.0)

            print(f"Using seed: {seed}")

            # CLIP Text Encode
            clip_encode = CLIPTextEncode()
            positive_cond = clip_encode.encode(clip, prompt)[0]
            negative_cond = clip_encode.encode(clip, negative_prompt)[0]

            # FluxGuidance 적용
            positive_cond = node_helpers.conditioning_set_values(positive_cond, {"guidance": guidance_value})
            # SD3용 빈 레이턴트 생성
            latent = create_empty_latent(width=width, height=height)

            # Sampling 실행
            samples = sample(
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive_cond,
                negative=negative_cond,
                latent=latent,
                denoise=denoise
            )

            # VAE Decode
            vae_decoder = VAEDecode()
            decoded = vae_decoder.decode(vae, {"samples": samples})[0]

            # 텐서 형태 변환
            decoded = decoded.cpu().numpy()
            if len(decoded.shape) == 5:
                decoded = decoded[0, 0]
            elif len(decoded.shape) == 4:
                decoded = decoded[0]

            # uint8로 변환
            decoded = np.clip(decoded * 255, 0, 255).astype(np.uint8)

            # 이미지로 변환
            image = Image.fromarray(decoded)

            # 이미지 파일 저장
            image_filepath = save_image_with_timestamp(image, seed)

            # base64 포맷 변환
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # 응답 데이터 생성
            response_data = {
                'status': 'success',
                'image': img_str,
                'parameters': {
                    'prompt': prompt,
                    'negative_prompt': negative_prompt,
                    'width': width,
                    'height': height,
                    'guidance_scale': guidance_value,
                    'steps': steps,
                    'cfg_scale': cfg,
                    'seed': seed,
                    'sampler_name': sampler_name,
                    'scheduler': scheduler,
                    'denoise': denoise
                },
                'request_type' :request_type
            }

            # JSON 파일 저장
            json_filepath = save_json_with_timestamp(response_data, seed)
            print(f"[{model_var_name} | {clip_var_name} | {vae_var_name}] JSON saved as: {json_filepath}")
            return response_data
    except Exception as e:
        raise e

try:
    name = os.environ['TLO_APP_ID']
except KeyError as e:
    name = "T2I_MODEL_APP"
model_name = "t2i"

@app.route(f"/v1/models", methods=["GET"])
def health():
    return "OK"

@app.route(f"/v1/models/{name}", methods=["GET"])
def mhealth():
    return "OK"


@app.route(f"/v1/models/{model_name}:predict", methods=["POST"])
def model_predict():

    request_type = request.get_json()['request_type']
    
    if "A" == request_type:
        lock = endpoint_locks['A']
        if not lock.acquire(blocking=False):
              return jsonify({
                'status': 'error',
                'message': 'Engine is currently busy. Please try again later.'
            }), 503
        else:
            try:
                response_body = generation_image_by_model("model", 'clip', 'vae')
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()  # 메모리 해제
                return jsonify({'error': 'GPU memory overflow, try again with smaller input'}), 500
            except Exception as e:
                raise e
            finally:
                lock.release()
            return jsonify(response_body)
    elif "B" == request_type:
        lock = endpoint_locks['B']
        if not lock.acquire(blocking=False):
              return jsonify({
                'status': 'error',
                'message': 'Engine is currently busy. Please try again later.'
            }), 503
        else:
            try:
                response_body = generation_image_by_model("model_b", 'clip_b', 'vae_b')
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()  # 메모리 해제
                return jsonify({'error': 'GPU memory overflow, try again with smaller input'}), 500
            except Exception as e:
                raise e
            finally:
                lock.release()
            return jsonify(response_body)


if __name__ == '__main__':
    print("Initializing models...")
    model, clip, vae = initialize_models("A")
    model_b, clip_b, vae_b = initialize_models("B")
    print("Models loaded successfully!")
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False, threaded=True)
else:
    print("Initializing models...")
    model, clip, vae = initialize_models("A")
    model_b, clip_b, vae_b = initialize_models("B")
    print("Models loaded successfully!")