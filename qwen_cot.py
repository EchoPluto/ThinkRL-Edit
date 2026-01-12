from transformers import AutoProcessor
import os
import json
import re
import torch

from PIL import Image
from qwen_vl_utils import process_vision_info

import torch.distributed as dist

def extract_reflect(response: str) -> dict:
    try:
        data = json.loads(response)
        if "edit_suggestion" in data:
            return data["edit_suggestion"]
    except json.JSONDecodeError:
        pass

    # Step 2. 用正则作为 fallback
    # 要求：必须在一对 { } 内，匹配多种写法
    pattern = re.compile(
        r"""\{
        [^{}]*?                             # 匹配花括号内部内容（非贪婪）
        ["']?\s*edit_suggestion\s*["']?\s*[:：]\s* # 匹配 Instruction（可带引号），支持中英文冒号
        ["']?([^"'}]+)["']?                  # 匹配值，可以无引号
        [^{}]*?
        \}""",
        re.VERBOSE | re.DOTALL
    )

    match = pattern.search(response)
    if match:
        return match.group(1).strip()

    return None

def extract_reflect2(text: str) -> dict:
    try:
        data = json.loads(text)
        if "Edit Suggestion" in data:
            return data["Edit Suggestion"]
    except json.JSONDecodeError:
        pass

    # 1️⃣ 匹配花括号内出现的 Instruction Match 段
    pattern = re.compile(
        r"""\{
        [^{}]*?                                   # 花括号内容
        ["']?\s*Edit\s+Suggestion\s*["']?\s*[:：]\s*  # 匹配关键字及中英文冒号
        ["']?([^"'}]+)["']?                       # 匹配值（允许无引号）
        [^{}]*?
        \}""",
        re.VERBOSE | re.DOTALL
    )

    match = pattern.search(text)
    if match:
        return match.group(1).strip()

    # 2️⃣ 若不在花括号内，则直接匹配普通行（兼容多行）
    pattern_fallback = re.compile(
        r"""["']?\s*Edit\s+Suggestion\s*["']?\s*[:：]\s*["']?(.+?)["']?$""",
        re.VERBOSE | re.MULTILINE
    )
    match = pattern_fallback.search(text)
    if match:
        return match.group(1).strip()

    return None

def extract_rewrite(text):
    # Step 1. 尝试 JSON 解析
    try:
        data = json.loads(text)
        if "Rewrited" in data:
            return data["Rewrited"]
    except json.JSONDecodeError:
        pass

    # Step 2. 用正则作为 fallback
    # 要求：必须在一对 { } 内，匹配多种写法
    pattern = re.compile(
        r"""\{
        [^{}]*?                             # 匹配花括号内部内容（非贪婪）
        ["']?\s*Rewrited\s*["']?\s*[:：]\s* # 匹配 Instruction（可带引号），支持中英文冒号
        ["']?([^"'}]+)["']?                  # 匹配值，可以无引号
        [^{}]*?
        \}""",
        re.VERBOSE | re.DOTALL
    )

    match = pattern.search(text)
    if match:
        return match.group(1).strip()

    return None


prompt_instruction_following = """
You are a professional AI image specialist. Your role is to critically determine an AI-generated edited image by comparing it to the original image and a given editing instruction. You must identify exactly what parts of the instruction were fulfilled, partially fulfilled, or missing, and provide concrete suggestions for the next edit to fully satisfy the instruction.

You will be given:
1. **Image A**: the original image <image>.
2. **Image B**: the edited image <image>.
3. **Editing Instruction**: a directive describing the intended modification to Image A to produce Image B. {instruction} 

### Your Objectives
- Detect all visible differences between Image A and Image B accurately.
- Determine whether these differences match the editing instruction exactly.
- Identify missing, incorrect, or insufficient edits.
- Provide clear, actionable suggestions for the next editing attempt to fully fulfill the instruction.
- Ignore unrelated visual changes; focus strictly on the instruction.
- Do not provide any scores.

### Reasoning Steps
- Detect Difference: Describe all visible changes between Image A and Image B (size, shape, color, position, texture, presence/absence of objects, etc.), without referencing the instruction yet.
- Instruction Match: Compare actual changes to expected changes:
   - Was the correct object modified (not replaced)?
   - Was the requested attribute (color, size, position, texture, etc.) modified correctly?
   - Was the degree/extent of modification accurate?
- Edit Suggestion: Provide specific instructions for the next edit to fix missing or incorrect modifications. Be concrete and concise; target only the unsatisfied parts of the instruction.

### Output Format
Return your results in strict JSON:

{{
  "detect_difference": "",
  "instruction_match": "",
  "edit_suggestion": ""
}}

### Example
Instruction: Adjust the size of the apple to match the size of the watermelon

{{
  "detect_difference": "In the original image, the apple is smaller than the watermelon. In the edited image, the apple has been enlarged but is still smaller than the watermelon.",
  "instruction_match": "The edit increases the apple's size, partially fulfilling the instruction. However, the apple does not fully match the watermelon's size.",
  "edit_suggestion": "Increase the apple's size further so that its height and overall volume visually match the watermelon."
}}
"""


system_prompt = """
You are a Visual Instruction Rewriter for multimodal image editing models.
Your task is to rewrite a user's image editing instruction so that an image editing model can easily understand and correctly perform the edit. 

You will be given:
1. **Input Image**: the original image before editing
2. **Instruction**: the original user instruction describing the desired edit

Your Objective:
- Clarify and Ground the Instruction: Make all edits visually explicit, using concrete visual concepts (objects, positions, colors, lighting, textures, shapes, materials).
- Add a Factual Rationale (if needed): If the instruction requires external or domain knowledge (e.g., physical realism, color consistency, biological accuracy, material reflection, geometry), provide a short explanation under the section “Factual Rationale”, describing what the correct result should visually look like

Example: 
Instruction: This bread after being left for a long time
{{
"Rewrited": "Make the bread appear moldy. It should be covered in splotches of green, black, and white fuzzy mold, particularly on the crusts and cut surfaces, indicating significant spoilage from being left out for a long time."
}}
       
## Input
**Input Image**: <image>
**Instruction**: {instruction}          
## Output Format:
Provide the rewrited instruction in the following JSON format:
{{
"Rewrited": "" 
}}

"""
from io import BytesIO
import base64
root_dir = '/mnt/bn/voyager-useast/users/hengjia.li/Kris_Bench/KRIS_Bench'
def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
    base64_qwen = f"data:image;base64,{encoded_image_text}"
    return base64_qwen

class QwenVLModel:
    def __init__(self, device, use_qwen3=True):
        self.device = device
        if use_qwen3:
            from transformers import Qwen3VLMoeForConditionalGeneration
            model_path = 'Qwen/Qwen3-VL-30B-A3B-Instruct'
            
            self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_path, torch_dtype="auto", device_map=None,
            ).to(self.device)
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model_path = 'Qwen/Qwen2.5-VL-7B-Instruct'
        
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype="auto", device_map=None,
            ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.tokenizer.padding_side = "left"

    def reflect(self, prompts, ref_images, pre_edit_images, max_new_tokens=512):
        self.model.eval()
        texts = [prompt_instruction_following.format(instruction=prompt) for prompt in prompts]
        imgs = [pil_image_to_base64(ref_image) for ref_image in ref_images]

        edit_images = []
        for i, edit_image in enumerate(pre_edit_images):
            if isinstance(edit_image, torch.Tensor):
                save_image = (edit_image * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                save_image = save_image.transpose(1, 2, 0) 
                save_image = Image.fromarray(save_image)
                save_image = pil_image_to_base64(save_image)
                edit_images.append(save_image)
            else:
                edit_images.append(edit_image)
        
        messages = [
            [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img,
                    },
                    {
                        "type": "image",
                        "image": edit_img,
                    },
                    {"type": "text", "text": text},
                ],
            }]
            for text, img, edit_img in zip(texts, imgs, edit_images)
        ]
        apply_texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=apply_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Batch Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        feedbacks = []
        for prompt, output_text in zip(prompts, output_texts):
            match = extract_reflect(output_text)
            if match is not None:
                feedback = prompt + ' In addition, the following is the feedback on the last editing results. Please refer to these feedbacks to regenerate. ' + match
            else:
                match2 = extract_reflect2(output_text)
                if match2 is not None:
                    feedback = prompt + ' In addition, the following is the feedback on the last editing results. Please refer to these feedbacks to regenerate. ' + match2
                else:
                    feedback = prompt
            feedbacks.append(feedback)
        return feedbacks
    
    def rewrite(self, prompts, ref_images, max_new_tokens=512):
        self.model.eval()
        texts = [system_prompt.format(instruction=prompt) for prompt in prompts]
        imgs = [pil_image_to_base64(ref_image) for ref_image in ref_images]
        messages = [
            [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img,
                    },
                    {"type": "text", "text": text},
                ],
            }]
            for text, img in zip(texts, imgs)
        ]
        apply_texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]

        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=apply_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Batch Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        feedbacks = []
        for i, output_text in enumerate(output_texts):

            match = extract_rewrite(output_text)

            if match is None or len(match) == 0:
                rewrite = prompts[i]
            else:
                rewrite = match
            feedbacks.append(rewrite)
        
        return feedbacks
    
if __name__ == '__main__':
    qwen_model = QwenVLModel(use_qwen3=True, device='cuda')

    root_dir = 'Kris_Bench/KRIS_Bench'
    for dir_name in os.listdir(root_dir):
        if 'bio' not in dir_name:
            continue
        dir_path = os.path.join(root_dir, dir_name)
        
        # 步骤1: 定义JSON文件名
        json_file = os.path.join(dir_path, f'annotation.json')

        # 步骤2: 读取JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 步骤3: 根据键获取图像路径
        # 这里我们以键 '1' 为例，您可以替换成其他任何您想要的键
        for index, key_to_read in enumerate(data.keys()):
            # if index > 5:
            #     break
            # 获取图像文件名，例如 '1.jpg'
            img_name = data[key_to_read]['ori_img']
            if isinstance(img_name, str):
                img_path = os.path.join(dir_path, img_name)
            else:
                continue
            rewrites = qwen_model.rewrite([data[key_to_read]['ins_en']], [Image.open(img_path)])
            print(rewrites)
            reflects = qwen_model.reflect([data[key_to_read]['ins_en']], [Image.open(img_path)], [Image.open(img_path)])
            print(reflects)