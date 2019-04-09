from __future__ import print_function, absolute_import

from datasets import data_manager
from PIL import Image
from torch.utils.data import Dataset

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass

    return img

class ImageData(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    def __getitem__(self, item):
        img, pid, camid = self.dataset[item]
        img = read_image(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':

    root = '/home/ztc/Projects/batch-feature-erasing-network/data'
    dataset=data_manager.MSMT17('msmt17','retrieval',root = root)
    print(dataset.query)
    #img = Image.open(os.path.join(root, 'msmt17', 'bounding_box_train', '0000_002_01_0303morning_0009_0.jpg'))
    # img = cv2.imread(os.path.join(root, 'msmt17', 'bounding_box_train', '0000_002_01_0303morning_0009_0.jpg'))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    # cv2.namedWindow('test')
    # cv2.imshow('test',img)
    # cv2.waitKey(0)
    # p=input('gegegevvw')
    # for item in dataset.train:
    #     img, pid, camid = item
    #     p=input(img)
    #     img = Image.open(os.path.join(root,'msmt17','bounding_box_train',img))
    #     img.show()

    # queryloader = DataLoader(
    #     ImageData(dataset.query, TrainTransform('person')),
    #     #sampler=RandomIdentitySampler(dataset.train, 4),
    #     batch_size=32,num_workers=0,pin_memory=True
    # )
    #
    # galleryloader = DataLoader(
    #     ImageData(dataset.gallery, TestTransform('person')),
    #     batch_size=32, num_workers=4,
    #     pin_memory=True
    # )
    #
    # for i,inputs in enumerate(queryloader):
    #     print(inputs.size())

    #read_image('/home/ztc/Projects/batch-feature-erasing-network/data/msmt17/bounding_box_train/1040_008_13_0302morning_0119_0.jpg')
