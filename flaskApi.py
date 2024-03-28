# Create>>>> static folder
# Create>>>> allow ufw 7474
# GPU Server IP
# 

from flask import Flask, request
from io import BytesIO
from PIL import Image
import urllib.request
from pathlib import Path
import sys
from PIL import Image
from utils_ootd import get_mask_location
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import torch
import os
import json
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.inference_ootd_dc import OOTDiffusionDC
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:200"
torch.cuda.memory_summary(device=None, abbreviated=False)

# python -m pip install codetiming

# print("************************************Graphic***********************************************")
# print("Device name:", torch.cuda.get_device_properties('cuda').name)
# print("FlashAttention available:", torch.backends.cuda.flash_sdp_enabled())
# print(f'torch version: {torch.version}')
# print(f'torch.cuda.device_count: {torch.cuda.device_count()}')
# print(f'torch.cuda.current_device(): {torch.cuda.current_device()}')
# print("*************************************Graphic End**********************************************")

openpose_model = OpenPose(0)
parsing_model = Parsing(0)
category_dict = ['upperbody', 'lowerbody', 'dress']
category_dict_utils = ['upper_body', 'lower_body', 'dresses']
model_type = "dc" 
category = 0
cloth_path = None
model_path = None
image_scale = 2.0
n_steps = 20
n_samples = 1
seed = 1


app = Flask(__name__)

def imageWebpToJPGModel(data):
    modelURL = "tryon/"+"model_"+str(data.get('category'))+"_"+str(data.get('customer_id'))+"_"+str(data.get('product_id'))+"_"+str(data.get('timestamp'))
    urllib.request.urlretrieve(data.get('model'),modelURL+".webp")
    im = Image.open(modelURL+".webp").convert("RGB")
    im.save(modelURL+".jpg")
    os.remove(modelURL+".webp") 
    return modelURL+".jpg"
              
def imageWebpToJPGCloth(data):
    clothURL = "tryon/"+"cloth_"+str(data.get('category'))+"_"+str(data.get('customer_id'))+"_"+str(data.get('product_id'))+"_"+str(data.get('timestamp'))
    urllib.request.urlretrieve(data.get('cloth'),clothURL+".webp")
    im = Image.open(clothURL+".webp").convert("RGB")
    im.save(clothURL+".jpg")
    os.remove(clothURL+".webp") 
    return clothURL+".jpg"


@app.route('/tryon2d', methods=['POST'])
def handle_json():
    data = request.json
    # print(data.get('model'))q
    # print(data.get('cloth'))
    # print(data.get('customer_id"'))
    # print(data.get('product_id'))
    # print(data.get('types'))
    # print(data.get('timestamp'))
    model_path = imageWebpToJPGModel(data)
    cloth_path = imageWebpToJPGCloth(data)

    if(data.get('category')=="upper"):
        # upperbody
        category   = 0
        model_type = "hd"
    elif(data.get('category')=="lower"):
        # lowerbody
        category   = 1
        model_type = "dc"
    else:
        # dress
        category   = 2
        model_type = "dc"

    if model_type == "hd":
        model = OOTDiffusionHD(0)
    elif model_type == "dc":
        model = OOTDiffusionDC(0)
    else:
        raise ValueError("model_type must be \'hd\' or \'dc\'!")

    cloth_img = Image.open(cloth_path).resize((768, 1024))
    model_img = Image.open(model_path).resize((768, 1024))
    keypoints = openpose_model(model_img.resize((384, 512)))
    model_parse, _ = parsing_model(model_img.resize((384, 512)))
    mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
    mask = mask.resize((768, 1024), Image.NEAREST)
    mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
    masked_vton_img = Image.composite(mask_gray, model_img, mask)
    # print("mode2love mask")
    masked_vton_img.save('mode2love/mask.jpg')
    linkArray = []
    images = model(
        model_type=model_type,
        category=category_dict[category],
        image_garm=cloth_img,
        image_vton=masked_vton_img,
        mask=mask,
        image_ori=model_img,
        num_samples=n_samples,
        num_steps=n_steps,
        image_scale=image_scale,
        seed=seed,
    )
    # print("model end")
    image_idx = 0
    for image in images:
        image.save('./static/out_' + model_type + '_' + str(image_idx) + '.png')
        print('./static/out_' + model_type + '_' + str(image_idx) + '.png')
        image_idx += 1
    # print("********************************************* Timer ************************:  " )
    
    jsonData = [{"imageLink": item} for item in linkArray]
    json_str = json.dumps(jsonData)
    print(json_str)
    return json_str
 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7474)
