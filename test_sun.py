import argparse
from model.build_BiSeNet import BiSeNet
import os
import torch
import cv2
from imgaug import augmenters as iaa
from PIL import Image
from torchvision import transforms
import numpy as np
from utils import reverse_one_hot, get_label_info, colour_code_segmentation
import datetime

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/epoch_130.pth', help='The path to the pretrained weights of model')
parser.add_argument('--context_path', type=str, default="Xception", help='The context path model you are using.')
parser.add_argument('--num_classes', type=int, default=38, help='num of object classes (with void)')
parser.add_argument('--crop_height', type=int, default=480, help='Height of cropped/resized input image to network')
parser.add_argument('--crop_width', type=int, default=640, help='Width of cropped/resized input image to network')
parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to user gpu for training')
parser.add_argument('--csv_path', type=str, default='/home/disk2/xs/sun/seg37_class_dict.csv', help='Path to label info csv file')

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


def predict_on_RGBD(image, depth):  # nd convenient both for img and video
    # pre-processing on image
    image = resize_img(image).convert('RGB')
    image = transforms.ToTensor()(image).float().unsqueeze(0)

    depth = resize_depth(depth)
    depth = np.array(depth)
    depth = depth[:, :, np.newaxis]
    depth = depth / 255
    depth = transforms.ToTensor()(depth).float().unsqueeze(0)

    rgbd = torch.cat((image, depth), 1)

    # predict
    model.eval()
    predict = model(rgbd).squeeze()
    predict = reverse_one_hot(predict)
    predict = colour_code_segmentation(np.array(predict), label_info)
    predict = np.uint8(predict)

    return cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR)


def predict_image(image, depth, out_path):
    image = Image.open(image)
    depth = Image.open(depth)
    cv2.imwrite(out_path, predict_on_RGBD(image, depth))


def predict_video(rgb_video, dep_video, out_img_dir):
    if not os.path.exists(out_img_dir):
        os.mkdir(out_img_dir)
    cap1 = cv2.VideoCapture(rgb_video)
    cap2 = cv2.VideoCapture(dep_video)
    cnt = 0
    while True:
        ret1, image = cap1.read()
        ret2, depth = cap2.read()  # 3 channel
        if not ret1 or not ret2:
            break
        depth = cv2.split(depth)[0]
        image = Image.fromarray(image)
        depth = Image.fromarray(depth)
        seg = predict_on_RGBD(image, depth)
        # save image
        cv2.imwrite(out_img_dir + str(cnt) + '.png', seg)
        print(cnt)
        cnt += 1
    cap1.release()
    cap2.release()


def predict_eg_image():
    for i in range(1, 4):
        predict_image(image='./img/sun/' + str(i) + '.jpg',
                      depth='./img/sun/' + str(i) + '.png',
                      out_path='./img/sun/' + str(i) + '_seg.png')
        print(i)


def predict_img_dir(img_dir, depth_dir, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    cnt = 0
    for img in os.listdir(depth_dir):
        image = Image.open(img_dir + img.replace('.png', '.jpg'))
        depth = Image.open(depth_dir + img)
        seg = predict_on_RGBD(image, depth)
        cv2.imwrite(out_dir + img, seg)
        print(cnt)
        cnt += 1


def test_sun():
    print(datetime.datetime.now())
    predict_img_dir(img_dir='/home/disk2/xs/sun/test/image/',
                    depth_dir='/home/disk2/xs/sun/test/depth/',
                    out_dir='/home/disk2/xs/sun/test/xcep-epoch130/')
    print(datetime.datetime.now())


def test_nyu():
    print(datetime.datetime.now())
    predict_img_dir(img_dir='/home/disk2/xs/nyu/image/',
                    depth_dir='/home/disk2/xs/nyu/depth/',
                    out_dir='/home/disk2/xs/nyu/xcep-epoch130/')
    print(datetime.datetime.now())


def predict_lab_dir(img_dir, depth_dir, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    cnt = 0
    for img in os.listdir(depth_dir):
        image = Image.open(img_dir + img.replace('disp', 'left'))
        # filter depth
        depth = Image.open(depth_dir + img)

        # ori depth
        # depth = cv2.imread(depth_dir + img)
        # depth = cv2.split(depth)[0]
        # depth = Image.fromarray(depth)

        # no depth
        # depth = np.zeros((480, 640)).astype('uint8')
        # depth = Image.fromarray(depth)
        seg = predict_on_RGBD(image, depth)
        cv2.imwrite(out_dir + img.replace('disp', 'seg'), seg)
        print(cnt)
        cnt += 1


def test_lab():
    print(datetime.datetime.now())
    predict_lab_dir(img_dir='/home/disk2/xs/lab/image/',
                    depth_dir='/home/disk2/xs/lab/depth/',
                    out_dir='/home/disk2/xs/lab/xcep-epoch130/')
    print(datetime.datetime.now())


def test_3_videos():
    predict_video(rgb_video='./img/lab2/rgb.avi',
                  dep_video='./img/lab2/dep_ori.avi',
                  out_img_dir='./img/lab2/seg_ori/')
    predict_video(rgb_video='./img/lab2/rgb.avi',
                  dep_video='./img/lab2/dep_disp.avi',
                  out_img_dir='./img/lab2/seg_disp/')
    predict_video(rgb_video='./img/lab2/rgb.avi',
                  dep_video='./img/lab2/dep_mix.avi',
                  out_img_dir='./img/lab2/seg_mix/')


if __name__ == '__main__':
    # test_lab()
    # test_nyu()
    test_3_videos()
