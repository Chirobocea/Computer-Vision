from copy import deepcopy
import cv2
import numpy as np
import pickle
from sklearn import linear_model, utils
import os
import timeit
import glob
import ntpath
from skimage.feature import hog
from Parameters import *


class FacialDetector:

    def __init__(self, params:Parameters):
        self.params = params
        self.batch_size = 100
        self.max_samples = 13_600
        self.n_btaches = self.max_samples // self.batch_size
        self.best_model = None
        self.hog_nbins = 9
        self.threshold_iou = 0.3


    def compute_hog(self, image, two_dimensions = False, gray = True, blue = False, green = False, red = False):
    
        h = []

        if blue or green or red:
            b, g, r = cv2.split(image)

        if blue:
            h_b = hog(b, orientations=self.hog_nbins, pixels_per_cell = (self.params.dim_hog_cell, self.params.dim_hog_cell),
                           cells_per_block = (2, 2), feature_vector = False)
            h.append(h_b)
        if green:
            h_g = hog(g, orientations=self.hog_nbins, pixels_per_cell = (self.params.dim_hog_cell, self.params.dim_hog_cell),
                           cells_per_block = (2, 2), feature_vector = False)
            h.append(h_g)
        if red:
            h_r = hog(r, orientations=self.hog_nbins, pixels_per_cell = (self.params.dim_hog_cell, self.params.dim_hog_cell),
                           cells_per_block = (2, 2), feature_vector = False)
            h.append(h_r)

        if gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h_gray = hog(image, orientations=self.hog_nbins, pixels_per_cell = (self.params.dim_hog_cell, self.params.dim_hog_cell),
                           cells_per_block = (2, 2), feature_vector = False)
            h.append(h_gray)

        hogs = np.concatenate(h, axis=4)

        if two_dimensions == False:
            hogs = hogs.flatten()

        return hogs


    def train(self):

        svm_file_name = os.path.join(self.params.dir_models, 'task1_svm')
        if os.path.exists(svm_file_name):
            self.best_model = pickle.load(open(svm_file_name, 'rb'))
            return None


        def get_training_data(k):


            def get_descriptors(current_batch, type_samples:int):

                features = []
                
                if type_samples == 1:
                    dir_samples = self.params.dir_pos_samples
                    dir_features = self.params.dir_save_pos_features
                elif type_samples == 0:
                    dir_samples = self.params.dir_neg_samples
                    dir_features = self.params.dir_save_neg_features

                samples = sorted(os.listdir(dir_samples))
                no_samples = len(samples)

                if no_samples - current_batch*self.batch_size < self.batch_size:
                    return features

                for i in range(current_batch*self.batch_size, (current_batch+1)*self.batch_size):
                    if i < no_samples and i < self.max_samples:
                        # Check if HOG features have already been computed and saved
                        features_save_path = os.path.join(dir_features, str(current_batch+1) + '_' + samples[i][:-4] + '.npy')
                        if os.path.exists(features_save_path):
                            # Load the precomputed HOG features
                            features.append(np.load(features_save_path))
                        else:
                            # Compute and save the HOG features
                            image = cv2.imread(os.path.join(dir_samples, samples[i]))
                            features.append(self.compute_hog(image))
                            # Save the HOG features to a file
                            np.save(features_save_path, features[-1])
                
                features = np.array(features)

                return features


            pos_features = []
            neg_features = []
            pos_features = get_descriptors(k, 1)
            neg_features = get_descriptors(k, 0)

            # Concatenate the positive and negative HOGs into a single array
            if len(pos_features) == 0:
                X = neg_features
                Y = np.zeros(len(neg_features))
            elif len(neg_features) == 0:
                X = pos_features
                Y = np.ones(len(pos_features))
            else:
                X = np.concatenate((pos_features, neg_features))

                # Create a label array with 1s for the positive examples and 0s for the negative examples
                Y = np.concatenate((np.ones(len(pos_features)), np.zeros(len(neg_features))))

                # Shuffle the positive and negative examples
                X, Y = utils.shuffle(X, Y)

            return X, Y


        best_accuracy = 0
        best_c = 0
        Cs = [10**k for k in range(-12, -2)]

        for c in Cs:

            model = linear_model.SGDClassifier(alpha = c)
            accuracies = []

            for k in range(self.n_btaches):
                print("Procesing batch no ", k+1)
                X, Y = get_training_data(k)
                
                model.partial_fit(X, Y, classes = [0, 1])

            for k in range(self.n_btaches):
                print("Evaluating batch no ", k+1)
                X, Y = get_training_data(k)

                accuracy = model.score(X, Y)
                accuracies.append(accuracy)
            
            accuracy = np.mean(accuracies)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_c = c
                self.best_model = deepcopy(model)

            # Print the mean accuracy of the model on the test set
            print(f"Mean accuracy for C = {c}: {accuracy}")

        # Save the best model to a file
        print(f"Best mean accuracy for C = {best_c}: {best_accuracy}")
        pickle.dump(self.best_model, open(svm_file_name, 'wb'))


    def intersection_over_union(self, bbox_a, bbox_b):
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_a[3], bbox_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        return iou


    def run(self):

        test_images_path = os.path.join(self.params.dir_test_examples, '*.jpg')
        test_files = glob.glob(test_images_path)

        no_test_images = len(test_files)

        detections = None
        scores = np.array([])
        file_names = np.array([])

        w = self.best_model.coef_.T
        bias = self.best_model.intercept_[0]


        def fit_img_to_cells(image, k):

            k = 2*k

            height, width = image.shape[:2]
            new_width, new_height = None, None

            if width/k != width//k:
                reaming = width - k*(width//k)
                add = k - reaming
                new_width = width + add
            
            if height/k != height//k:
                reaming = height - k*(height//k)
                add = k - reaming
                new_height = height + add

            if new_width is None and new_height is None:
                return image

            if new_width is None:
                new_width = width
            elif new_height is None:
                new_height = height

            white_image = np.full((new_height, new_width, 3), 0, dtype=np.uint8)

            white_image[0:height, 0:width] = image

            return white_image
             

        def distort_image(image, scale, ratio):
            
            if ratio < 1:
                width_distorsion = ratio * scale
                height_distorsion = scale
            else:
                width_distorsion = scale
                height_distorsion = (1/ratio) * scale

            # Resize the image using bilinear interpolation
            return cv2.resize(image, None, fx=width_distorsion, fy=height_distorsion, interpolation=cv2.INTER_LINEAR), 1/width_distorsion, 1/height_distorsion


        def non_maximal_suppression(image_detections, image_scores, image_size):

            sorted_indices = np.flipud(np.argsort(image_scores))
            sorted_image_detections = image_detections[sorted_indices]
            sorted_scores = image_scores[sorted_indices]

            is_maximal = np.ones(len(image_detections)).astype(bool)

            for i in range(len(sorted_image_detections) - 1):
                if is_maximal[i] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                    for j in range(i + 1, len(sorted_image_detections)):
                        if is_maximal[j] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                            if self.intersection_over_union(sorted_image_detections[i],sorted_image_detections[j]) > self.threshold_iou:
                                is_maximal[j] = False
                            else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                                c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                                c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                                if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                        sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                    is_maximal[j] = False
            return sorted_image_detections[is_maximal], sorted_scores[is_maximal]


        def remove_included_detections(detections, scores):
            new_detections = []
            new_scores = []
            for i in range(len(detections)):
                is_included = False
                for j in range(len(detections)):
                    if i != j:
                        if (detections[i][0] >= detections[j][0] and detections[i][1] >= detections[j][1] and detections[i][2] <= detections[j][2] and detections[i][3] <= detections[j][3])\
                        or (detections[j][0] >= detections[i][0] and detections[j][1] >= detections[i][1] and detections[j][2] <= detections[i][2] and detections[j][3] <= detections[i][3]):
                            if scores[i] < scores[j] and abs(scores[i]-scores[j]) > 4:
                                is_included = True
                                break
                if is_included == False:
                    new_detections.append(detections[i])
                    new_scores.append(scores[i])
            return new_detections, new_scores


        for i in range(no_test_images):

            start_time = timeit.default_timer()
            print('Procesing image test no %d of %d...' % (i, no_test_images))
            
            img = cv2.imread(test_files[i])

            image_scores = []
            image_detections = []

            for ratio in self.params.window_aspect_ratios:
                for scale in self.params.window_scale:

                    img_distort, width_distor, height_distor = distort_image(img, scale, ratio)

                    img_distort = fit_img_to_cells(img_distort, self.params.dim_hog_cell)

                    height, width = img_distort.shape[:2]

                    hogs = self.compute_hog(img_distort, True)

                    no_cols = img_distort.shape[1] // self.params.dim_hog_cell - 1
                    no_rows = img_distort.shape[0] // self.params.dim_hog_cell - 1

                    no_cells_in_window = self.params.dim_window // self.params.dim_hog_cell - 1
                    
                    for y in range(0, no_rows - no_cells_in_window):
                        for x in range(0, no_cols - no_cells_in_window):
                            hog_descriptor = hogs[y:y + no_cells_in_window, x:x + no_cells_in_window].flatten()

                            descr = hog_descriptor
                            
                            score = np.dot(descr, w)[0] + bias

                            if score > self.params.threshold_task1:

                                x_min = int(x * self.params.dim_hog_cell)
                                y_min = int(y * self.params.dim_hog_cell)
                                x_max = int(x * self.params.dim_hog_cell + self.params.dim_window)
                                y_max = int(y * self.params.dim_hog_cell + self.params.dim_window)

                                if x_max > width:
                                    x_max = width
                                if y_max > height:
                                    y_max = height

                                x_min, x_max = int(x_min * width_distor), int(x_max * width_distor)
                                y_min, y_max = int(y_min * height_distor), int(y_max * height_distor)

                                image_detections.append([x_min, y_min, x_max, y_max])
                                image_scores.append(score)
                                image_names = None


            if len(image_scores) > 0:
                image_detections, image_scores = non_maximal_suppression(np.array(image_detections), np.array(image_scores), img.shape)
                image_detections, image_scores = remove_included_detections(image_detections, image_scores)

                                                                              
            if len(image_scores) > 0:
                if detections is None:
                    detections = image_detections
                else:
                    detections = np.concatenate((detections, image_detections))
                scores = np.append(scores, image_scores)
                short_name = ntpath.basename(test_files[i])
                image_names = [short_name for ww in range(len(image_scores))]
                file_names = np.append(file_names, image_names)


            end_time = timeit.default_timer()
            print('Procesing time for iamge %d of %d is %f sec.'
                  % (i, no_test_images, end_time - start_time))


        np.save(self.params.dir_solutions+"/task1/detections_all_faces", detections)
        np.save(self.params.dir_solutions+"/task1/scores_all_faces",scores)
        np.save(self.params.dir_solutions+"/task1/file_names_all_faces", file_names)

        return detections, scores, file_names