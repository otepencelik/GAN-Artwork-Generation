import os
from PIL import Image

# set wd
my_dir = "D:/GANart/Datasets/"
os.chdir(my_dir)

# List style
style_names = os.listdir('./wikiart/wikiart')

# Loop to resize
i = 0

for i in range(len(style_names)):
    # create a folder
    if style_names[i] not in os.listdir('./wikiart_128/wikiart'):
        os.mkdir('./wikiart_128/wikiart/' + style_names[i])
    # read images
    images = os.listdir('./wikiart/wikiart/' + style_names[i])
    
    for j in range(len(images)):
        my_img = Image.open('./wikiart/wikiart/' + style_names[i] + '/' + images[j])
        new_image = my_img.resize((128, 128))
        new_image.save('./wikiart_128/wikiart/' + style_names[i] + '/' + images[j])
        if j % 100 == 0:
            print(f'folder: {i+1}/{len(style_names)}, image:{j+1}/{len(images)}')
    
    