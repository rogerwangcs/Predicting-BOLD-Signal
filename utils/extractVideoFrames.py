import os
import cv2
import math
from tqdm import tqdm_notebook


def extractRunFrames(videoPath, snapshotPath, runName):

    if not os.path.exists(videoPath):
        raise OSError('The file {} was not found.'.format(videoPath))

    pbar = tqdm_notebook(total=600)
    cap = cv2.VideoCapture(videoPath)
    frameRate = cap.get(5) * 2  # frame rate times 2 for 2 seconds
    count = 0
    while(cap.isOpened()):
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            filename = os.path.join(
                snapshotPath, 'forrest-%s-%d.jpg' % (runName, count))
            cv2.imwrite(filename, frame)
            pbar.update(1)
            count = count + 1
    cap.release()
    pbar.close()
