# BiSeNet-CCP
Extend Scene Segmentation to Instance Segmentation using 8-Connected Components Decision.

## Bilateral Segmentation Network (BiSeNet)
Bilateral Segmentation Network (BiSeNet) is designed to segment scenes in Real-time, proposed by Face++.
- PAPER: [BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation](https://arxiv.org/abs/1808.00897)
- CODE: https://github.com/ooooverflow/BiSeNet

We use BiSeNet as our backbone network to finish scene segmentation task. The architure is shown as below.

![](results/bisenet-architecture.png)

## Connected Components Postprocessing
To extend scene segmentation to instance segmentation, we use 8-connected components decision to postprocess the BiSeNet outputs. The connected components theory is shown as below.

![](results/connect-components.png)

The idea is quite easy : **As the segmentation result of each class is a Binary Image, Connected Components Decision can be used to segment the class superpixel to instance superpixels.**

## Our Results (scene & instance)

<img src="results/ADE20K/ADE_val_00000034.jpg">

### ADE20K
<table>
  <tr>
    <td><center><img src="results/ADE20K/ADE_val_00000034.jpg"></center></td>
    <td><center><img src="results/ADE20K/ADE_val_00000034.png"></center></td>
    <td><center><img src="results/ADE20K/ADE_val_00000034_P.png"></center></td>
  </tr>
</table>
