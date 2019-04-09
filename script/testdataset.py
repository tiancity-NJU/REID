import os
import shutil
import re
import random

def rename():
    root = '/home/ztc/Projects/batch-feature-erasing-network/data/combined/bounding_box_train/'
    imgs = os.listdir(root)

    for name in imgs:
        new_param = name.split('_')
        id = int(new_param[0])
        cam = int(new_param[1])
        frame = int(new_param[2][:-4])
        new_name = '{:04d}_c{:d}_f{:07d}.jpg'.format(id, cam, frame)
        os.rename(os.path.join(root, name), os.path.join(root, new_name))




def test():
    root = '/home/ztc/Projects/batch-feature-erasing-network/data/combined/bounding_box_test'
    dst_root = '/home/ztc/Projects/batch-feature-erasing-network/data/combined/query'
    imgs = os.listdir(root)

    for img in imgs:
        if random.random() < 0.08:
            src_name = os.path.join(root, img)
            dst_name = os.path.join(dst_root, img)
            shutil.move(src_name, dst_name)


if __name__ == '__main__':
    test()