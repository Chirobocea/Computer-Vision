import os

class Parameters:
    def __init__(self):
        self.base_dir = "D:/Cursuri/5. Computer Vision/Proiect 2/Final project - Good one/312_Chirobocea_Mihail"
        # self.dir_pos_samples = os.path.join(self.base_dir, 'Train/Positive selected')
        # self.dir_neg_samples = os.path.join(self.base_dir, 'Train/Negative x5')
        # self.dir_save_pos_features = os.path.join(self.base_dir, 'HOGs/Positive selected')
        # self.dir_save_neg_features = os.path.join(self.base_dir, 'HOGs/Negative x5')
        self.dir_test_examples = 'D:/Cursuri/5. Computer Vision/Proiect 2/testare/testare'
        self.dir_models = os.path.join(self.base_dir, 'Models')
        self.dir_solutions = os.path.join(self.base_dir, 'Solutions')
        if not os.path.exists(self.dir_solutions):
            os.makedirs(self.dir_solutions)
            print('directory created: {} '.format(self.dir_solutions))


        self.dim_window = 80
        self.window_aspect_ratios = [0.4, 0.7, 0.8, 0.95, 1.1, 1.35, 1.5, 1.75]
        self.window_scale = [0.3, 0.4, 0.5, 0.75, 1, 1.2] 
        self.dim_hog_cell = 10
        self.threshold_task1 = 0
        self.threshold_task2 = 0.35