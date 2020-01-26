import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def edge_detector():
    img = cv.imread("ag-sn_borderless.png", 0)
    canny = cv.Canny(img, 400, 500)

    titles = ["image", "canny"]
    images = [img, canny]

    for i in range(1, len(titles) + 1):
        plt.subplot(1, len(titles), i), plt.imshow(images[i - 1], 'gray')
        plt.title(titles[i - 1])
        plt.xticks([]), plt.yticks([])

    plt.show()


def harris_corner_detector():
    img = cv.imread('ag-sn.png')
    cv.imshow('ag-sn', img)

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    gray_img = np.float32(gray_img)
    dst = cv.cornerHarris(gray_img, 2, 0, 0.04)

    dst = cv.dilate(dst, None)

    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv.imshow('dst', img)

    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()


def shi_corner_detector():
    img = cv.imread('ag-sn_borderless.png')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, bw_img = cv.threshold(gray, 230, 255, cv.THRESH_BINARY)

    # corners = cv.goodFeaturesToTrack(bw_img, 200, 0.01, 10)
    #
    # corners = np.int0(corners)
    #
    # for i in corners:
    #     x, y = i.ravel()
    #     print(f"x = {x:3}, y = {y:3}")
    #
    # for i in corners:
    #     x, y = i.ravel()
    #     cv.circle(bw_img, (x, y), 3, 255, -1)

    cv.imshow('dst', bw_img)

    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()


if __name__ == '__main__':
    # edge_detector()
    shi_corner_detector()
