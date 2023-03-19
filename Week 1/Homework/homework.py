import cv2
import numpy as np


def read_images(image_dir):
    gray_img = cv2.imread(image_dir + "GRAY.JPG")
    rgb_img_half = cv2.imread(image_dir + "RGB_half.JPG")
    rgb_img_quarter = cv2.imread(image_dir + "RGB_quater.JPG")
    return gray_img, rgb_img_half, rgb_img_quarter


def convert_image_to_grayscale(image_rgb):
    return cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)


def detect_keypoints(gray_img, rgb_img, detector):
    kp1, des1 = detector(gray_img, None)
    kp2, des2 = detector(rgb_img, None)
    return kp1, des1, kp2, des2


def match_keypoints(des1, des2, matcher):
    matches_gray = matcher.match(des1, des2)
    matches = [match for match in matches_gray if match.distance < 0.7 * max(len(des1), len(des2))]
    return matches


def extract_matched_keypoints(kp1, kp2, matches):
    query_1 = np.float32([kp1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    train_1 = np.float32([kp2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)
    return query_1, train_1


def find_homography(src_pts, dst_pts):
    homography, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    return homography


def warp_images(homography, gray_img, rgb_img):
    colored_img_warped = cv2.warpPerspective(rgb_img, homography, (gray_img.shape[1], gray_img.shape[0]))
    return colored_img_warped


def overlap_black_pixels(gray_img, colored_img):
    # Wherever there is a black pixel in the first image, replace it with the pixel from the gray image(best way to infer in my opinion)
    colored_img[colored_img == 0] = gray_img[colored_img == 0]
    return colored_img


def enhance_color(colored_img, gray_img):
    # Let's convert image to HSV and up the V channel based on the grayscale image
    # (higher value there is higher the intensity of the pixel)
    colored_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2HSV)
    colored_img[..., 2] = gray_img[..., 0]
    colored_img = cv2.cvtColor(colored_img, cv2.COLOR_HSV2BGR)
    return colored_img


def colorize_gray_picture(gray_img, rgb_img_rotated, detector_func, matcher):
    image_rgb_grayed = convert_image_to_grayscale(rgb_img_rotated)
    kp1, des1, kp2, des2 = detect_keypoints(gray_img,
                                               image_rgb_grayed,
                                               detector_func)
    matches = match_keypoints(des1, des2, matcher)
    query_1, train_1 = extract_matched_keypoints(kp1, kp2, matches)

    homography = find_homography(query_1, train_1)
    colored_img = warp_images(homography, gray_img, rgb_img_rotated)
    colored_img = overlap_black_pixels(gray_img, colored_img)
    colored_img = enhance_color(colored_img, gray_img)
    return colored_img


def main(detector_func, matcher):
    gray_img, rgb_img_half, rgb_img_quarter = read_images("Week 1/Homework/")
    colored_img_result_half = colorize_gray_picture(gray_img, rgb_img_half, detector_func, matcher)
    colored_img_result_quarter = colorize_gray_picture(gray_img, rgb_img_quarter, detector_func, matcher)
    return colored_img_result_half, colored_img_result_quarter


if __name__ == "__main__":
    sift = cv2.SIFT_create().detectAndCompute
    matcher = cv2.BFMatcher()
    colored_img_result_half, colored_img_result_quarter = main(sift, matcher)
    # Show images
    cv2.imwrite("processed_half.jpg", colored_img_result_half)
    cv2.imwrite("processed_quarter.jpg", colored_img_result_quarter)