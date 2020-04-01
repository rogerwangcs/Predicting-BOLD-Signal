import os
import cv2
import math
from tqdm import tqdm_notebook

def extract_video_frames(video_path, snapshot_path, run_name):

    """ Extracts video frames from video every two seconds
    :type video_path: string
    :param video_path: Directory path to source video directory

    :type snapshot_path: string
    :param snapshot_path: Directory path to save frames

    :type run_name: string
    :param run_name: name of run

    :raises: OSError

    :return: None
    """

    # Makes sure the source video exists
    if not os.path.exists(video_path):
        raise OSError('The file {} was not found.'.format(video_path))

    pbar = tqdm_notebook(total=600) # Most runs have slightly less than 600 frames
    cap = cv2.VideoCapture(video_path)
    frameRate = cap.get(5) * 2  # frame rate * 2 for 2 seconds
    count = 0
    while(cap.isOpened()):
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            filename = os.path.join(
                snapshot_path, 'forrest-%s-%d.jpg' % (run_name, count))
            cv2.imwrite(filename, frame)
            pbar.update(1)
            count = count + 1
    cap.release()
    pbar.close()
