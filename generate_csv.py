import os
import pandas as pd
import numpy as np
import glob

def get_measurements(file):
    df = pd.read_csv(file, names=['loc1', 'loc2', 'speed', 'c_v', 'c_p', 'c_o', 'other', 'off_road', 'agents', 'throttle', 'steer'], index_col=None)
    speed = pd.to_numeric(df['speed'].str[:-4], downcast='float')
    throttle = pd.to_numeric(df['throttle'].str[10:], downcast='float')
    steer = pd.to_numeric(df['steer'].str[7:], downcast='float')
    return speed.tolist(),throttle.tolist(),steer.tolist()
    
# measure_path = "/home/bhushan/work/college/Fall18/projects/cv/CARLA/data/measurements/"
# measure_path = "/home/mihir/Downloads/CARLA_0.8.2/PythonClient/_out/measurements/"
measure_path = "/home/nborude/CARLA_0.8.2/PythonClient/_out/measurements/"

# Loading measurement data
speed_arr = []
throttle_arr = []
steer_arr = []

# for file in glob.glob(path):
for i in range(len(os.listdir(measure_path))):
    with open(measure_path+str(i)+".txt") as file:
        speed,throttle,steer = get_measurements(file)
        speed_arr += (speed)
        throttle_arr += (throttle)
        steer_arr += (steer)
        
# Loading image data
# img_dir_path = "/home/bhushan/work/college/Fall18/projects/cv/CARLA/data/episode*"
img_dir_path = "/home/nborude/CARLA_0.8.2/PythonClient/_out/episode*"
img_path = "/CameraRGB/*.png"


episode_num_arr = []
img_path_arr = []
for directory in sorted(glob.glob(img_dir_path)):
    episode_num = directory[directory.rfind('/')+1:]
    for img in sorted(glob.glob(directory+img_path)):
        episode_num_arr.append(episode_num)
        img = img[img.rfind('/')+1:]
        img_path_arr.append(img)
        
        
# Creating dataframe: episode_number, center_image_path, steer, speed, throttle
df = pd.DataFrame(list(zip(episode_num_arr, img_path_arr, steer_arr, speed_arr, throttle_arr)), columns=['episode_number', 'center','steer','speed','throttle'])
# Writing to CSV
df.to_csv("train.csv")
