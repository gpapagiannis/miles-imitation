import argparse
import torch
from pathlib import Path
from extractor import ViTExtractor
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import List, Tuple
import correspondences
# from torchvision.transforms.functional import adjust_brightness, adjust_contrast
import random
def resize_pil_to_128_and_save(image_pil, save_path):
    # Center crop image1_pil
    image_pil = image_pil.crop((0, 0, image_pil.size[0], image_pil.size[0]))
    # Resize image1_pil to 128x128
    image1_pil = image_pil.resize((128, 128))
    #S ave image1_pil
    image1_pil.save(save_path)


@torch.no_grad()
def find_similarity(image1, image2, num_pairs: int = 10, load_size: int = 128, layer: int = 1,
                         facet: str = 'key', bin: bool = False, thresh: float = 0.05, model_type: str = 'dino_vits8',
                         stride: int = 8):
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    extractor = ViTExtractor(model_type, stride, device=device)
    image1_batch, image1_pil = extractor.preprocess_image(image1, load_size)
    # image1_batch = adjust_contrast(image1_batch, random.uniform(.2, .5))
    # print("Image1 batch shape:", image1_batch.shape)
    # image1_batch = .3 * image1_batch
    descriptors1 = extractor.extract_descriptors(image1_batch.to(device), layer, facet, bin)
    num_patches1, load_size1 = extractor.num_patches, extractor.load_size
    image2_batch, image2_pil = extractor.preprocess_image(image2, load_size)
    # image2_batch = adjust_contrast(image2_batch, random.uniform(1.2, 1.5))
    descriptors2 = extractor.extract_descriptors(image2_batch.to(device), layer, facet, bin)
    num_patches2, load_size2 = extractor.num_patches, extractor.load_size
    similarities = torch.squeeze(torch.squeeze(correspondences.chunk_cosine_sim(descriptors1, descriptors2, per_patch_similarity=True), dim=-1), dim=0)
    # print("Mean similarity:", torch.mean(similarities))
    mean_sim = torch.mean(similarities)
    similarities = similarities.resize(1, num_patches2[0], num_patches2[1])
    similarities_patch_map = similarities.repeat_interleave(stride, dim=1).repeat_interleave(stride, dim=2).permute(1, 2,
                                                                                                                0).clip(0, 1)
    similarities = similarities.repeat_interleave(stride, dim=1).repeat_interleave(stride, dim=2).repeat(3, 1, 1).permute(1, 2, 0).clip(0, 1)
    # print("Similarities shape:", similarities.shape)

    print("number of patches:", num_patches1, num_patches2);input()

    return mean_sim, image1_batch, image2_batch, extractor, similarities_patch_map, similarities


#@markdown
def sim(image1, image2):
    image_path1 = 'im5_128.png'  # @param
    image_path2 = 'im6_128.png'  # @param
    # @markdown Choose number of points to output:
    num_pairs = 10  # @param
    # @markdown Choose loading size:
    load_size = 224  # @param
    # @markdown Choose layer of descriptor:
    layer = 9  # @param
    # @markdown Choose facet of descriptor:
    facet = 'key'  # @param
    # @markdown Choose if to use a binned descriptor:
    bin = True  # @param
    # @markdown Choose fg / bg threshold:
    thresh = 0.05  # @param
    # @markdown Choose model type:
    model_type = 'dino_vits8'  # @param
    # @markdown Choose stride:
    stride = 14  # @param

    mean_sim, image1_batch, image2_batch, extractor, similarities_patch_map, similarities = find_similarity(image1, image2, num_pairs, load_size, layer, facet, bin, thresh, model_type, stride)
    return mean_sim, image1_batch, image2_batch, extractor, similarities_patch_map, similarities
def corresponds():
    image_path1 = 'im1_128.png'  # @param
    image_path2 = 'im4_128.png'  # @param
    # @markdown Choose number of points to output:
    num_pairs = 30  # @param3
    # @markdown Choose loading size:
    load_size = 224  # @param
    # @markdown Choose layer of descriptor:
    layer = 9  # @param
    # @markdown Choose facet of descriptor:
    facet = 'key'  # @param
    # @markdown Choose if to use a binned descriptor:
    bin = True  # @param
    # @markdown Choose fg / bg threshold:
    thresh = 0.05  # @param
    # @markdown Choose model type:
    model_type = 'dino_vits8'  # @param
    # @markdown Choose stride:
    stride = 7  # @param
    with torch.no_grad():
        points1, points2, image1_pil, image2_pil = correspondences.find_correspondences(image_path1, image_path2, num_pairs, load_size,
                                                                        layer,
                                                                        facet, bin, thresh, model_type, stride)
    fig_1, ax1 = plt.subplots()
    ax1.axis('off')
    ax1.imshow(image1_pil)
    fig_2, ax2 = plt.subplots()
    ax2.axis('off')
    ax2.imshow(image2_pil)

    fig1, fig2 = correspondences.draw_correspondences(points1, points2, image1_pil, image2_pil)
    plt.show()

def coosegment():

    from cosegmentation import find_cosegmentation, draw_cosegmentation, draw_cosegmentation_binary_masks

    #@title Configuration:
    #@markdown Choose image paths:
    images_paths = ['im1_128.png', 'im2_128.png'] #@param
    #@markdown Choose loading size:
    load_size = 224 #@param
    #@markdown Choose layer of descriptor:
    layer = 11 #@param
    #@markdown Choose facet of descriptor:
    facet = 'key' #@param
    #@markdown Choose if to use a binned descriptor:
    bin=False #@param
    #@markdown Choose fg / bg threshold:
    thresh=0.065 #@param
    #@markdown Choose model type:
    model_type='dino_vits8' #@param
    #@markdown Choose stride:
    stride=14 #@param
    #@markdown Choose elbow coefficient for setting number of clusters
    elbow=0.975 #@param
    #@markdown Choose percentage of votes to make a cluster salient.
    votes_percentage=75 #@param
    #@markdown Choose whether to remove outlier images
    remove_outliers=False #@param
    #@markdown Choose threshold to distinguish inliers from outliers
    outliers_thresh=0.7 #@param
    #@markdown Choose interval for sampling descriptors for training
    sample_interval=100 #@param
    #@markdown Use low resolution saliency maps -- reduces RAM usage.
    low_res_saliency_maps=False #@param

    with torch.no_grad():
        # computing cosegmentation
        seg_masks, pil_images = find_cosegmentation(images_paths, elbow, load_size, layer, facet, bin, thresh, model_type,
                                                    stride, votes_percentage, sample_interval, remove_outliers,
                                                    outliers_thresh, low_res_saliency_maps)

        figs, axes = [], []
        for pil_image in pil_images:
            fig, ax = plt.subplots()
            ax.axis('off')
            ax.imshow(pil_image)
            figs.append(fig)
            axes.append(ax)

        # saving cosegmentations
        binary_mask_figs = draw_cosegmentation_binary_masks(seg_masks)
        chessboard_bg_figs = draw_cosegmentation(seg_masks, pil_images)

        plt.show()


def partcoosegment():
    # @title Configuration:
    # @markdown Choose image paths:
    images_paths = ['im1_128.png', 'im2_128.png'] #@param
    # @markdown Choose loading size:
    load_size = 360  # @param
    # @markdown Choose layer of descriptor:
    layer = 11  # @param
    # @markdown Choose facet of descriptor:
    facet = 'key'  # @param
    # @markdown Choose if to use a binned descriptor:
    bin = False  # @param
    # @markdown Choose fg / bg threshold:
    thresh = 0.065  # @param
    # @markdown Choose model type:
    model_type = 'dino_vits8'  # @param
    # @markdown Choose stride:
    stride = 4  # @param
    # @markdown Choose elbow coefficient for setting number of clusters
    elbow = 0.975  # @param
    # @markdown Choose percentage of votes to make a cluster salient.
    votes_percentage = 75  # @param
    # @markdown Choose interval for sampling descriptors for training
    sample_interval = 100  # @param
    # @markdown Use low resolution saliency maps -- reduces RAM usage.
    low_res_saliency_maps = True  # @param
    # @markdown number of final object parts.
    num_parts = 4  # @param
    # @markdown number of crop augmentations to apply on each input image. relevant for small sets.
    num_crop_augmentations = 20  # @param
    # @markdown If true, use three clustering stages instead of two. relevant for small sets.
    three_stages = True  # @param
    # @markdown elbow method for finding amount of clusters when using three clustering stages.
    elbow_second_stage = 0.94  # @param
    import matplotlib.pyplot as plt
    import torch
    from part_cosegmentation import find_part_cosegmentation, draw_part_cosegmentation

    with torch.no_grad():
        # computing part cosegmentation
        parts_imgs, pil_images = find_part_cosegmentation(images_paths, elbow, load_size, layer, facet, bin, thresh,
                                                          model_type,
                                                          stride, votes_percentage, sample_interval,
                                                          low_res_saliency_maps,
                                                          num_parts, num_crop_augmentations, three_stages,
                                                          elbow_second_stage)

        figs, axes = [], []
        for pil_image in pil_images:
            fig, ax = plt.subplots()
            ax.axis('off')
            ax.imshow(pil_image)
            figs.append(fig)
            axes.append(ax)

        # saving part cosegmentations
        part_figs = draw_part_cosegmentation(num_parts, parts_imgs, pil_images)

    plt.show()

def match_template():
    im1 =  Image.open('im1_128.png').convert('RGB')
    im2 = Image.open('im1_128.png').convert('RGB')

    # Convert to grayscale
    im1 = im1.convert('L')
    im2 = im2.convert('L')


    # Convert images to numpy
    im1 = np.array(im1) / 255
    im2 = np.array(im2) / 255

    # Plot images
    # plt.imshow(im1)
    # plt.imshow(im2)
    # plt.show()

    # Center crop im2
    im2 = im2[32:96, 32:96]

    # Convert images to grayscale

    # Plot images im2
    # plt.imshow(im2)
    # plt.show()
    from skimage import data
    from skimage.feature import match_template

    image = im1
    coin = im2

    result = match_template(image, coin)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]

    fig = plt.figure(figsize=(15, 10))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)

    ax1.imshow(coin, cmap=plt.cm.gray)
    ax1.set_axis_off()
    ax1.set_title('template')

    ax2.imshow(image, cmap=plt.cm.gray)
    ax2.set_axis_off()
    ax2.set_title('image')
    # highlight matched region
    hcoin, wcoin = coin.shape
    rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r', facecolor='none')
    ax2.add_patch(rect)

    ax3.imshow(result)
    ax3.set_axis_off()
    ax3.set_title('`match_template`\nresult')
    # highlight matched region
    ax3.autoscale(False)
    ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

    plt.show()



if __name__ == '__main__':

    match_template()