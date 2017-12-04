import glob
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
import boxes_utility
import svc_utility
from moviepy.editor import VideoFileClip


def find_vehicles(img):
    windows = []

    # Testing pipeline on training images:
    boxes_utility.slide_window(windows, img, xy_window=(64, 64))

    # Actually processing sliding windows from a bigger image: (uncomment)
    # boxes_utility.slide_window(windows, img, xy_window=(160, 160), y_start_stop=[380, 650])
    # boxes_utility.slide_window(windows, img, xy_window=(128, 128), y_start_stop=[400, 528])
    # boxes_utility.slide_window(windows, img, xy_window=(96, 96), y_start_stop=[400, 592])
    # boxes_utility.slide_window(windows, img, xy_window=(64, 64), y_start_stop=[400, 496])

    # boxes_utility.draw_boxes(img, windows)

    on_windows = boxes_utility.search_windows(img, windows, svc, cspace, orient, pix_per_cell, cells_per_block,
                                              hog_channel)

    heatmap = np.zeros_like(img[:, :, 0])
    heatmap = boxes_utility.add_heat(heatmap, on_windows)
    heatmap = boxes_utility.apply_threshold(heatmap, 2)

    labels = label(heatmap)

    boxed_image = boxes_utility.draw_labeled_bboxes(img, labels)

    return boxed_image

cspace = 'YUV'
orient = 16
pix_per_cell = 8
cells_per_block = 2
hog_channel = 'ALL'

# svc = svc_utility.train_classifier(cspace, orient, pix_per_cell, cells_per_block, hog_channel)
svc = svc_utility.load_classifier()

car_images = glob.glob('../training_data/vehicles/**/*.png')
noncar_images = glob.glob('../training_data/non_vehicles/**/*.png')

for car_image in car_images:
    img = mpimg.imread(car_image)
    boxed = find_vehicles(img)
    plt.imshow(boxed)
    plt.show()


# test_images = glob.glob('test_images/*.jpg')
#
# for test_image in test_images:
#     img = mpimg.imread(test_image)
#     boxed = find_vehicles(img)
#     plt.imshow(boxed)
#     plt.show()

video = VideoFileClip('project_video.mp4').fl_image(find_vehicles)
video.write_videofile('project_output.mp4', audio=False)


