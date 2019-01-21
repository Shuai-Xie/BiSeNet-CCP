import argparse
from model.build_BiSeNet import BiSeNet
import os
import torch
import cv2
import skimage.io as sio
from PIL import Image
from torchvision import transforms
import numpy as np
from utils import reverse_one_hot, get_label_info, colour_code_segmentation
import datetime

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/epoch_90.pth', help='The path to the pretrained weights of model')
parser.add_argument('--context_path', type=str, default="Xception", help='The context path model you are using.')
parser.add_argument('--num_classes', type=int, default=151, help='num of object classes (with void)')
parser.add_argument('--crop_height', type=int, default=480, help='Height of cropped/resized input image to network')
parser.add_argument('--crop_width', type=int, default=640, help='Width of cropped/resized input image to network')
parser.add_argument('--cuda', type=str, default='1', help='GPU ids used for training')
parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to user gpu for training')
parser.add_argument('--csv_path', type=str, default='/home/disk1/xs/ADEChallengeData2016/ade150.csv', help='Path to label info csv file')

args = parser.parse_args()

# read csv label path
label_info = get_label_info(args.csv_path)

scale = (args.crop_height, args.crop_width)

# build model
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
model = BiSeNet(args.num_classes, args.context_path)

# load pretrained model if exists
print('load model from %s ...' % args.checkpoint_path)

if torch.cuda.is_available() and args.use_gpu:
    model = torch.nn.DataParallel(model).cuda()
    model.module.load_state_dict(torch.load(args.checkpoint_path))  # GPU -> GPU
else:
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage))  # GPU -> CPU

print('Done!')

resize_img = transforms.Resize(scale, Image.BILINEAR)
resize_depth = transforms.Resize(scale, Image.NEAREST)
to_tensor = transforms.ToTensor()


def predict_on_RGB(image):  # nd convenient both for img and video
    # pre-processing on image
    image = resize_img(image)
    image = transforms.ToTensor()(image).float().unsqueeze(0)

    # predict
    model.eval()
    predict = model(image).squeeze()
    predict = reverse_one_hot(predict)
    predict = colour_code_segmentation(np.array(predict), label_info)  # RGB
    predict = predict.astype(np.uint8)

    return predict


def predict_img_dir(in_dir, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    cnt = 0
    for img in os.listdir(in_dir):
        image = Image.open(in_dir + img)  # RGB mode
        seg = predict_on_RGB(image)
        img = img.replace('.jpg', '.png')
        sio.imsave(out_dir + img, seg)
        print(cnt)
        cnt += 1
        if cnt >= 200:
            break


def test_ade():
    print(datetime.datetime.now())
    predict_img_dir(in_dir='/home/disk1/xs/ADEChallengeData2016/images/validation/',
                    out_dir='/home/disk1/xs/ADEChallengeData2016/ade-epoch90/')
    print(datetime.datetime.now())


if __name__ == '__main__':
    test_ade()
