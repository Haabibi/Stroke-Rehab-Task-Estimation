from PIL import Image

im = Image.open('/home/abigail/Desktop/Stroke-Rehab-Task-Estimation/action-models/survey_imgs/ic_25.jpg')

width, height = im.size

print(width, height)


left = width / 6.4
top = height / 8 
right = 5.4 * width / 6.4
bottom = 5 * height / 8
im1= im.crop((left, top, right, bottom))
print(im1.size)
im1.show()
