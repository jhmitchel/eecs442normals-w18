import os
import sys
import numpy as np

from PIL import Image

import net
from utils import depth_montage, normals_montage

def main():
    # location of depth module, config and parameters
        
    #model_name = 'depthnormals_nyud_alexnet'
    model_name = 'depthnormals_nyud_vgg'

    module_fn = 'models/iccv15/%s.py' % model_name
    config_fn = 'models/iccv15/%s.conf' % model_name
    params_dir = 'weights/iccv15/%s' % model_name

    # load depth network
    machine = net.create_machine(module_fn, config_fn, params_dir)

    assert len(sys.argv) == 2
    image_dir = sys.argv[1] 
    color_dir = image_dir + '/color'
    normals_dir = image_dir + '/normal'
    rgb_imgs = []
    image_names = []
    og_sizes = []

    for image_name in os.listdir(color_dir):
        if image_name.endswith('.png'):
            # collect image and make it correct size
            image_names.append(image_name)
            image_path = color_dir + '/' + image_name
            rgb = Image.open(image_path)
            og_sizes.append(rgb.size)
            rgb = rgb.resize((320, 240), Image.BICUBIC)
            rgb = np.asarray(rgb).reshape((1, 240, 320, 3))
            rgb_imgs.append(rgb)

    input_array = np.asarray(rgb_imgs).reshape((-1,240,320,3))
    (pred_depths, pred_normals) = machine.infer_depth_and_normals(input_array)

    # save prediction
    for i in range(len(image_names)):
        dims3 = pred_normals[i,:,:,:].shape
        dims4 = (1, dims3[0], dims3[1], dims3[2])
        normals_img_np = pred_normals[i,:,:,:].reshape(dims4)
        normals_img_np = normals_montage(normals_img_np)
        normals_img = Image.fromarray((255*normals_img_np).astype(np.uint8))
        normals_img = normals_img.resize(og_sizes[i], Image.BICUBIC)
        image_path = normals_dir + '/' + image_names[i]
        normals_img.save(image_path)

if __name__ == '__main__':
    main()

