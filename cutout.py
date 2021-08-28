from PIL import Image
import os

def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

if __name__ == '__main__':
    dir = 'test_frames/cd33a266-0c0d-428c-b758-f3ba80e6d154.mp4'
    arr = os.listdir(dir)
    # test it
    save_dir = 'test_frames\\croped'
    ensure_dir(save_dir)


    for x in arr:
        im = Image.open(dir+'/'+x)
        im = im.crop((278,93,1095,710))
        im.save(save_dir+'\\'+x+'.jpg')

    print("\r\ndone")
