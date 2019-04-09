
from datasets import data_manager
from PIL import Image
import os

root = '/home/ztc/Projects/batch-feature-erasing-network/data/msmt17/bounding_box_train'
a =data_manager.init_dataset('msmt17', 'retrieval')

# for img_path, _, _ in a.train:
#     img = Image.open(os.path.join(root,img_path)).convert('RGB')
#
# print('olk')
