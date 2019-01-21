import numpy as np
import matplotlib.pyplot as plt
from plt_label_map import get_label_name_colors
from skimage.measure import label, regionprops
import skimage.io as sio

label_names, label_colors = get_label_name_colors(csv_path='csv/ade150.csv')
label_names.insert(0, 'null')
label_colors.insert(0, (0, 0, 0))

color2val_dict = {}
for i, color in enumerate(label_colors):
    color2val_dict[str(color[0]) + str(color[1]) + str(color[2])] = i


def label_color2value(label_color_img):
    """
    :param label_color_img: label values image, grayscale
    :return: label_value_image
    """
    rows, cols = label_color_img.shape[0], label_color_img.shape[1]
    label_value_img = np.zeros((rows, cols)).astype('uint8')
    for i in range(rows):
        for j in range(cols):
            color = list(label_color_img[i][j])
            label_value_img[i][j] = color2val_dict[str(color[0]) + str(color[1]) + str(color[2])]
    return label_value_img


def get_ins_bbox(label_value):
    """
    结合使用 label_npy 和 depth 得到 3d bbox，三维的分割结果
    :param label_value: label values, 0-150
    :return: all_ins, a dict with all class dict as cat_ins
    cat_ins has keys:
     - name: class name
     - instance: super pixel with (x,y)
     - bbox: instance boundary in 2D (min_row, min_col, max_row, max_col)
     - center: instance center in 3D (x_mid, y_mid, z_mid) z ~ depth
     - num: instance number of a class
    """
    lmin, lmax = int(np.min(label_value)), int(np.max(label_value))
    if lmin == 0:  # 去掉背景
        lmin = 1

    # 从 lbl 中先得到 所有种类
    categories = []
    for i in range(lmin, lmax + 1):
        pix_num = len(np.where(label_value == i)[0])  # 存在某类像素的个数
        if pix_num > 0:
            # print('%2d %-13s %-6d' % (i, label_names[i], pix_num))  # - 左对齐
            categories.append(i)
    # print(categories)

    # 获得 超像素(x,y,z) 和 所有种类对应的 instance
    all_ins = {}
    for cat_idx in categories:  # cat idx
        cat_pix = np.zeros(label_value.shape)
        cat_pix[label_value == cat_idx] = 1

        cat_cnt_domain = label(cat_pix, connectivity=2)  # 8连通 判断 实例个数
        cat_num = np.max(cat_cnt_domain)  # instacne 总数
        # print(idx, cat_num)

        cat_ins = []  # 存储所有 instance
        cat_ins_bbox = []  # 存储所有 instance 的边界
        cat_ins_center = []  # 中心点

        # cat_ins_area = []  # 像素总数
        # 遍历每个 instance 得到 每个物体的 superpix 和 相对位置
        for i in range(1, cat_num + 1):
            # make ins
            one_cat = np.zeros(label_value.shape)
            one_cat[cat_cnt_domain == i] = 1  # one_cat 也可以作为 label 后的 二值图像
            one_cat = one_cat.astype('uint8')
            cat_ins.append(one_cat)

            # get ins(domain) properties
            props = regionprops(label_image=one_cat)
            if props[0].area < 100:  # 去掉 总像素数目 < 100 的实例
                continue
            # 添加 ins 属性
            # bbox
            cat_ins_bbox.append(props[0].bbox)  # tuple (min_row, min_col, max_row, max_col)
            # center (x,y,z)
            center = [int(p) for p in props[0].centroid]  # (row, col)
            cat_ins_center.append(center)

        cat_dict = {
            'name': label_names[cat_idx],
            'instances': cat_ins,
            'bbox': cat_ins_bbox,
            'center': cat_ins_center,
            'num': len(cat_ins_center)
        }
        all_ins[cat_idx] = cat_dict

    return all_ins


def instance_seg(img, show_bbox=True):
    # 显示彩色 label 图片
    lbl_color_img = sio.imread(img)
    figsize = [lbl_color_img.shape[1] / 100, lbl_color_img.shape[0] / 100]
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0, 1., 1.])  # 不显示边界白边
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(lbl_color_img)
    # 实例 bbox
    lbl_val_img = label_color2value(lbl_color_img)
    all_ins = get_ins_bbox(lbl_val_img)
    # 遍历所有 实例
    for cat_idx in all_ins.keys():  # 所有的种类名
        cat_dict = all_ins[cat_idx]
        for i in range(cat_dict['num']):
            # 显示 instance 名称
            ax.annotate(cat_dict['name'] + str(i),
                        xy=(cat_dict['bbox'][i][1], cat_dict['bbox'][i][0]), fontsize=7,
                        xycoords='data', xytext=(2, -10), textcoords='offset points',
                        # bbox=dict(boxstyle='round, pad=0.3',  # linewidth=0 可以不显示边框
                        #           facecolor=[c / 255 for c in label_colors[cat_idx]], lw=0),
                        color='w')
            # 显示边界
            if show_bbox:
                # show bbox
                ax.add_patch(plt.Rectangle(xy=(cat_dict['bbox'][i][1], cat_dict['bbox'][i][0]),
                                           width=cat_dict['bbox'][i][3] - cat_dict['bbox'][i][1],
                                           height=cat_dict['bbox'][i][2] - cat_dict['bbox'][i][0],
                                           edgecolor=[c / 255 for c in label_colors[cat_idx]],
                                           fill=False, linewidth=2))
                # show LT, RD
                # ax.annotate('LT', xy=(cat_dict['bbox'][i][1], cat_dict['bbox'][i][0]), fontsize=6,
                #             xycoords='data', xytext=(+2, -10), textcoords='offset points', color='w')
                ax.annotate('RD', xy=(cat_dict['bbox'][i][3], cat_dict['bbox'][i][2]), fontsize=6,
                            xycoords='data', xytext=(-10, +5), textcoords='offset points', color='w')
    new_name = img.replace('_P', '_S')
    fig.savefig(new_name, dpi=100)  # 和原图一样大


if __name__ == '__main__':
    # img_dir = './ade/'
    # for img in os.listdir(img_dir):
    #     instance_seg(img=img_dir + img)
    instance_seg(img='./ade/ADE_val_00000761_P.png')
