#import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2


# Ordinary license plate value is 0.95, new energy license plate is changed to 0.9
segmentation_spacing = 0.9

'''1 read the picture, and do grayscale processing'''
img = cv2.imread('C:/Users/Ironpritam/Desktop/stage1/yolov4/preds/predictions.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


'''2 Binary the grayscale image'''
ret, img_threshold = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)


'''3 Split characters'''
white = []  # Record the sum of white pixels in each column
black = []  # Record the sum of black pixels in each column
height = img_threshold.shape[0]
width = img_threshold.shape[1]

white_max = 0
black_max = 0


'''4 Cycle through the sum of black and white pixels for each column'''
for i in range(width):
    white_count = 0
    black_count = 0
    for j in range(height):
        if img_threshold[j][i] == 255:
            white_count += 1
        else:
            black_count += 1

    white.append(white_count)
    black.append(black_count)

white_max = max(white)
black_max = max(black)


'''5 Split the image, given the starting point of the character to be split'''
def find_end(start):
    end = start + 1
    for m in range(start + 1, width - 1):
        if(black[m] > segmentation_spacing * black_max):
            end = m
            break
    return end


n = 1
start = 1
end = 2
while n < width - 1:
    n += 1
    if(white[n] > (1 - segmentation_spacing) * white_max):
        start = n
        end = find_end(start)
        n = end
        if end - start > 5:
            print(start, end)
            character = img_threshold[1:height, start:end]
            cv2.imwrite('C:/Users/Ironpritam/Desktop/stage1/yolov4/stage2/final/img{0}.png'.format(n), character)
            cv2.imshow('character', character)
            cv2.waitKey(550)
        #   cv2.destroyAllWindows()
            cv2.imshow("RESULT",img)
            cv2.waitKey(100)
