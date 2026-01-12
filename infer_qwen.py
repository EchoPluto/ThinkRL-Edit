import os
import json
import torch
import torch.distributed as dist
from PIL import Image
from diffusers import QwenImageEditPipeline
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model, PeftModel
from qwen_cot import QwenVLModel


def setup_distributed():
    """初始化分布式环境"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()


def cleanup():
    """销毁分布式进程"""
    dist.destroy_process_group()


def main():
    # ======== 参数设置 ========
    reflection = True
    cot = True
    use_pretrain = False
    use_text = True
    iters = 80

    root_dir = 'Kris_Bench/KRIS_Bench/'
    qwen_dir = ''
    model_path = f'model.safetensors'

    # ======== 初始化分布式 ========
    local_rank, world_size = setup_distributed()
    device = f"cuda:{local_rank}"

    # ======== 加载图像编辑模型 ========
    pipeline = QwenImageEditPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit"
    )

    if not use_pretrain:
        if not os.path.exists(model_path):
            if local_rank == 0:
                print('model not found')
            cleanup()
            return

        # 加载 LoRA 权重
        state_dict_train = load_file(model_path)
        target_modules = [
            "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
            "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj",
            "attn.to_add_out", "img_mlp.net.0.proj", "img_mlp.net.2",
            "txt_mlp.net.0.proj", "txt_mlp.net.2",
        ]
        transformer_lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
        pipeline.transformer.load_state_dict(state_dict_train, strict=True)
        if use_text:
            state_dict_text = load_file(model_path.replace('model.safetensors', 'text_encoder.safetensors'))
            pipeline.text_encoder.load_state_dict(state_dict_text, strict=True)
    else:
        qwen_dir = qwen_dir + '_pretrain'

    pipeline.to(torch.bfloat16)
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=(local_rank != 0))

    if cot:
        qwen_model = QwenVLModel(device)

    # ======== 按文件为单位均衡划分任务 ========
    all_task_items = []

    all_dirs = sorted(os.listdir(root_dir))
    for d in all_dirs:
        dir_path = os.path.join(root_dir, d)
        json_path = os.path.join(dir_path, 'annotation.json')
        if not os.path.exists(json_path):
            continue

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            if local_rank == 0:
                print(f"Failed to read {json_path}: {e}")
            continue

        for key, item in data.items():
            img_name = item.get('ori_img', None)
            if not isinstance(img_name, str):
                continue
            img_path = os.path.join(dir_path, img_name)
            if not os.path.exists(img_path):
                continue
            all_task_items.append((d, key, item, img_path))

    total_tasks = len(all_task_items)
    tasks_per_rank = total_tasks // world_size
    remainder = total_tasks % world_size

    start_idx = local_rank * tasks_per_rank + min(local_rank, remainder)
    end_idx = start_idx + tasks_per_rank + (1 if local_rank < remainder else 0)
    my_tasks = all_task_items[start_idx:end_idx]

    if local_rank == 0:
        print("===== Task Distribution Summary (by file) =====")
        print(f"Total files: {total_tasks}")
        for i in range(world_size):
            s = i * tasks_per_rank + min(i, remainder)
            e = s + tasks_per_rank + (1 if i < remainder else 0)
            print(f"Rank {i}: {e - s} files")
        print("================================================")

    print(f"\n[Rank {local_rank}] Assigned {len(my_tasks)} files:")
    for _, _, item, img_path in my_tasks:
        print(f"[Rank {local_rank}]  -> {img_path}")
    print("------------------------------------------------\n")

    # ======== 主循环 ========
    for d, key_to_read, item, img_path in my_tasks:
        img_name = item.get('ori_img', None)
        dir_path = os.path.join(root_dir, d)
        save_dir = os.path.join(qwen_dir, str(iters), d)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, img_name)

        if reflection:
            os.makedirs(save_dir.replace('train', 'reflect_train'), exist_ok=True)

        print(f"[Rank {local_rank}] Processing: {img_path}")

        # 跳过已存在的文件
        if (not reflection and os.path.exists(save_path)) or (
            reflection and os.path.exists(save_path.replace('train', 'reflect_train')) and os.path.exists(save_path)
        ):
            print(f"[Rank {local_rank}] ⏩ Skip existing file: {save_path}")
            continue

        try:
            origin_image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[Rank {local_rank}] ❌ Failed to open {img_path}: {e}")
            continue

        text = item.get('ins_en', '')
        if cot:
            with torch.inference_mode():
                text = qwen_model.rewrite([text], [origin_image])
            print(f"[Rank {local_rank}] Rewritten prompt: {text}")

        inputs = {
            "image": origin_image,
            "prompt": text,
            "generator": torch.manual_seed(0),
            "true_cfg_scale": 4.0,
            "negative_prompt": " ",
            "num_inference_steps": 50,
        }

        # 第一次生成
        try:
            with torch.inference_mode():
                output = pipeline(**inputs)
                output_image = output.images[0]
                output_image.save(save_path)
            print(f"[Rank {local_rank}] ✅ Saved: {save_path}")
        except Exception as e:
            print(f"[Rank {local_rank}] ❌ Generation failed for {img_path}: {e}")
            continue

        # 反思增强阶段
        if reflection:
            try:
                with torch.inference_mode():
                    reflect_text = qwen_model.reflect(text, [origin_image], [output_image])
                inputs["prompt"] = reflect_text
                save_path_reflect = save_path.replace('train', 'reflect_train')

                with torch.inference_mode():
                    output = pipeline(**inputs)
                    output_image = output.images[0]
                    output_image.save(save_path_reflect)
                print(f"[Rank {local_rank}] 🔁 Reflection saved: {save_path_reflect}")
            except Exception as e:
                print(f"[Rank {local_rank}] ⚠️ Reflection failed: {e}")
                continue

    cleanup()


if __name__ == "__main__":
    main()
