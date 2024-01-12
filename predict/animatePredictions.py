import matplotlib.pyplot as plt
import matplotlib.animation as animation

import PIL.Image as Image

import os
import datetime

folder_path = 'predict/'

file_list = []
for filename in os.listdir(folder_path):
    if filename.endswith('.png'):
        file_list.append(filename)
sorted_list = sorted(file_list, key=lambda x: int(x.split(".")[0][7:]))

image_list = []
for filename in sorted_list:
    image = Image.open(os.path.join(folder_path, filename))
    image_list.append(image.copy())
    image.close() # close the image object
    os.remove(os.path.join(folder_path, filename))
if image_list == []:
    print('No Files Found. Exiting..')
    exit()
fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(image_list[0])
ax.axis('off')

def update(frame):
    im.set_data(image_list[frame])
    return im,

ani = animation.FuncAnimation(fig, update, frames=len(image_list), interval=200)

now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")

ani.save(folder_path + 'animation_{}.gif'.format(timestamp), writer='imagemagick',dpi=300)
