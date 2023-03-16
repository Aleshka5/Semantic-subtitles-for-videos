import time
import cv2
import numpy as np

def divade_part_video_to_cadrs(video_path,W,H, part_size, part_count):
    cap = cv2.VideoCapture(video_path)
    # cap.set(cv2.CAP_PROP_FPS, 1)
    numpy_images = np.zeros((part_size, 2, H, W, 3),dtype='uint8')
    success, img1 = cap.read()
    resized1 = cv2.resize(img1, (W, H), interpolation=cv2.INTER_AREA)
    i = 0
    while i != part_count:
        start_time = time.time()
        success, img2 = cap.read()
        #cv2.imshow('window',cv2.resize(cv2.resize(img1, (W, H), interpolation=cv2.INTER_AREA), (W*resize_koef, H*resize_koef), interpolation=cv2.INTER_AREA))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        real_size = 1
        while ((real_size < part_size) and (success)):
            resized2 = cv2.resize(img2, (W, H), interpolation=cv2.INTER_AREA)

            numpy_images[real_size, 0], numpy_images[real_size, 1] = resized1, resized2

            resized1 = resized2
            success, img2 = cap.read()
            real_size += 1
        #print(real_size)
        print(f'{time.time() - start_time}sec')
        yield numpy_images[:real_size], real_size
        i+=1

def plot_two_images(cadrs, index):
    print(index)
    cv2.imshow('render',np.hstack([cadrs[index][0]/255,cadrs[index+1][0]/255]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()