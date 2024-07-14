
import numpy as np 
import tensorflow as tf
import glob
import os
import csv


def merge_csi_label(csifile, labelfile, win_len=1000, thrshd=0.6, step=200):
    """
    Merge CSV files into a Numpy Array  X,  csi amplitude feature
    Returns Numpy Array X, Shape(Num, Win_Len, 90)
    Args:
        csifile  :  str, csv file containing CSI data
        labelfile:  str, csv fiel with activity label 
        win_len  :  integer, window length
        thrshd   :  float,  determine if an activity is strong enough inside a window
        step     :  integer, sliding window by step
    """
    activity = []
    with open(labelfile, 'r') as labelf:# opening the label file in read mode
        reader = csv.reader(labelf) 
        for line in reader: # iterating over the reader
            label  = line[0] # checks the activity
            if label == 'NoActivity': 
                activity.append(0)
            else:
                activity.append(1)
    activity = np.array(activity) # changing into numpy array
    csi = []
    with open(csifile, 'r') as csif: # opening the csv file in reading mode
        reader = csv.reader(csif)
        for line in reader:
            #[float(v) for v in line] converts each value in the line from a string to a float and creates a list of floats.
            line_array = np.array([float(v) for v in line])
            # extract the amplitude only
            line_array = line_array[1:91]
            csi.append(line_array[np.newaxis,...])# adds a new axis to line_array, converting it from a 1D array to a 2D array with shape (1, 90).
    csi = np.concatenate(csi, axis=0)
    #concatenates all the arrays in the csi list along the first axis, resulting in a single 2D array where each row corresponds to a processed line from the CSV file.
    assert(csi.shape[0] == activity.shape[0])#ensures that the number of rows in csi matches the number of rows in activity
    # screen the data with a window
    index = 0
    feature = []
    while index + win_len <= csi.shape[0]:
        cur_activity = activity[index:index+win_len]#cur_activity is a slice of the activity array from index to index + win_len.
        if np.sum(cur_activity)  <  thrshd * win_len:#he total activity in the window does not meet the threshold
            index += step
            continue
        cur_feature = np.zeros((1, win_len, 90)) #fills the cur_feature with the corresponding csi data from index to index + win_len.
        cur_feature[0] = csi[index:index+win_len, :]
        feature.append(cur_feature)
        index += step
    return np.concatenate(feature, axis=0)# combien all features along the row;


def extract_csi_by_label(raw_folder, label, labels, save=False, win_len=1000, thrshd=0.6, step=200):
    """
    Returns all the samples (X,y) of "label" in the entire dataset
    Args:
        raw_foler: The path of Dataset folder
        label    : str, could be one of labels
        labels   : list of str, ['bed', 'fall', 'pickup', 'run', 'sitdown', 'standup', 'walk']
        save     : boolean, choose whether save the numpy array 
        win_len  :  integer, window length
        thrshd   :  float,  determine if an activity is strong enough inside a window
        step     :  integer, sliding window by step
    """
    print('Starting Extract CSI for Label {}'.format(label))
    label = label.lower() # changes the label to lowercase 
    if not label in labels:#  checks if the lowercase label is not in the list labels.
        raise ValueError("The label {} should be among 'bed','fall','pickup','run','sitdown','standup','walk'".format(labels))
    #if the label is not found in labels, a ValueError is raised
    
    data_path_pattern = os.path.join(raw_folder, 'input_*' + label + '*.csv')
    #'input_*' + label + '*.csv' will match any CSV files in raw_folder that have filenames starting with input_, followed by any characters (*), then the label, 
    # followed by any characters (*), and ending with .csv. 
    #  constructs a file path pattern by joining raw_folder with a pattern string that includes the label.
    input_csv_files = sorted(glob.glob(data_path_pattern))#glob.glob(data_path_pattern) finds all files matching the data_path_pattern
    #sorted(...) sorts the list of found file paths.
    annot_csv_files = [os.path.basename(fname).replace('input_', 'annotation_') for fname in input_csv_files]
    #os.path.basename(fname) extracts the base filename from the full path.
    #.replace('input_', 'annotation_') replaces the 'input_' prefix with 'annotation_' in each filename, generating the corresponding annotation filenames.
    annot_csv_files = [os.path.join(raw_folder, fname) for fname in annot_csv_files]
    # os.path.join(raw_folder, fname) constructs the full path for each annotation file by joining raw_folder with the base filename.
    feature = []
    index = 0
    #zip(input_csv_files, annot_csv_files) pairs each input CSV file with its corresponding annotation CSV file.
    for csi_file, label_file in zip(input_csv_files, annot_csv_files):#
        index += 1
        if not os.path.exists(label_file):
            print('Warning! Label File {} doesn\'t exist.'.format(label_file))
            #If the annotation file does not exist,
            # a warning is printed and the loop continues to the next pair (continue skips the remaining code in the current iteration).
            continue
        feature.append(merge_csi_label(csi_file, label_file, win_len=win_len, thrshd=thrshd, step=step))
        print('Finished {:.2f}% for Label {}'.format(index / len(input_csv_files) * 100,label))
    
    feat_arr = np.concatenate(feature, axis=0)
    if save:
        np.savez_compressed("X_{}_win_{}_thrshd_{}percent_step_{}.npz".format(
            label, win_len, int(thrshd*100), step), feat_arr)
    # one hot
    feat_label = np.zeros((feat_arr.shape[0], len(labels)))#initializes a zero array of shape
    feat_label[:, labels.index(label)] = 1 #labels.index(label) gets the index of the current label in the labels list.
    return feat_arr, feat_label


def train_valid_split(numpy_tuple, train_portion=0.9, seed=379):
    """
    Returns Train and Valid Datset with the format of (x_train, y_train, x_valid, y_valid),
    where x_train and y_train are shuffled randomly.

    Args:
        numpy_tuple  : tuple of numpy array: (x_bed, x_fall, x_pickup, x_run, x_sitdown, x_standup, x_walk)
        train_portion: float, range (0,1)
        seed         : random seed
    """
    #Setting a seed initializes the random number generator to a specific state, 
    # which means that you will get the same sequence of random numbers every time you run your code with that seed. 
    np.random.seed(seed=seed)
    x_train = []
    x_valid = []
    y_valid = []
    y_train = []

    for i, x_arr in enumerate(numpy_tuple):
        index = np.random.permutation([i for i in range(x_arr.shape[0])]) #reates a random permutation of the indices [0, 1, ..., x_arr.shape[0] - 1].
        split_len = int(train_portion * x_arr.shape[0])
        x_train.append(x_arr[index[:split_len], ...])#index[:split_len] gets the indices for the training set.
        #x_arr[index[:split_len], ...] selects the training samples based on the indices and appends them to x_train.
        tmpy = np.zeros((split_len,7))
        #sets the column corresponding to the current class i to 1, creating a one-hot encoded label array.
        tmpy[:, i] = 1
        y_train.append(tmpy)
        #index[split_len:] gets the indices for the validation set.
        #x_arr[index[split_len:], ...] selects the validation samples based on the indices and appends them to x_valid.
        x_valid.append(x_arr[index[split_len:],...])
        tmpy = np.zeros((x_arr.shape[0]-split_len,7))
        #sets the column corresponding to the current class i to 1, creating a one-hot encoded label array for the validation set.
        tmpy[:, i] = 1
        y_valid.append(tmpy)#appends the one-hot encoded labels to y_valid.
    
    #resulting in single arrays for training and validation data and labels.
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    x_valid = np.concatenate(x_valid, axis=0)
    y_valid = np.concatenate(y_valid, axis=0)

    index = np.random.permutation([i for i in range(x_train.shape[0])])# generates a random permutation of the indices [0, 1, ..., x_train.shape[0] - 1].
    x_train = x_train[index, ...]#reorders the training data based on the shuffled indices.
    y_train = y_train[index, ...]#reorders the training labels data based on the shuffled indices.
    return x_train, y_train, x_valid, y_valid
    
    

def extract_csi(raw_folder, labels, save=False, win_len=1000, thrshd=0.6, step=200):
    """
    Return List of Array in the format of [X_label1, y_label1, X_label2, y_label2, .... X_Label7, y_label7]
    Args:
        raw_folder: the folder path of raw CSI csv files, input_* annotation_*
        labels    : all the labels existing in the folder
        save      : boolean, choose whether save the numpy array 
        win_len   :  integer, window length
        thrshd    :  float,  determine if an activity is strong enough inside a window
        step      :  integer, sliding window by step
    """
    ans = []
    for label in labels:
        feature_arr, label_arr = extract_csi_by_label(raw_folder, label, labels, save, win_len, thrshd, step)
        ans.append(feature_arr)  # x label 
        ans.append(label_arr)    # y label
    return tuple(ans)


class AttenLayer(tf.keras.layers.Layer):
    """
    Attention Layer used to Compute Weighted Features along the Time axis
    Args:
        num_state : number of hidden Attention states
    """
    #**kwargs is used to pass a variable number of keyword arguments to a function. 
    def __init__(self, num_state, **kwargs):
        super(AttenLayer, self).__init__(**kwargs)
        self.num_state = num_state
    
#     self.kernel: Creates a weight matrix (kernel) with shape [input_shape[-1], self.num_state], 
#     initialized using Glorot uniform initializer (glorot_uniform). This weight matrix transforms the input features.
# self.bias: Creates a bias vector (bias) with shape [self.num_state], initialized to zeros (initializer='zeros'). 
# This bias term allows the attention mechanism to introduce a shift in the computation.
# self.prob_kernel: Creates a weight vector (prob_kernel) with shape [self.num_state], also initialized using Glorot uniform initializer.
#  This vector is used in computing attention weights (e.g., through a dot product with intermediate values).
    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=[input_shape[-1], self.num_state],
            initializer='glorot_uniform',
            dtype=tf.float32,
            trainable=True
        )
        self.bias = self.add_weight(
            name='bias',
            shape=[self.num_state],
            initializer='zeros',
            dtype=tf.float32,
            trainable=True
        )
        self.prob_kernel = self.add_weight(
            name='prob_kernel',
            shape=[self.num_state],
            initializer='glorot_uniform',
            dtype=tf.float32,
            trainable=True
        )

    def call(self, input_tensor):
        ## Compute intermediate attention state
        atten_state = tf.tanh(tf.tensordot(input_tensor, self.kernel, axes=1) + self.bias)
        # Compute logits (scalar attention scores)
        logits = tf.tensordot(atten_state, self.prob_kernel, axes=1)
         # Apply softmax to compute attention probabilities
        prob = tf.nn.softmax(logits)
        # # Weighted sum of input_tensor using attention probabilities
        weighted_feature = tf.reduce_sum(tf.multiply(input_tensor, tf.expand_dims(prob, -1)), axis=1)
        return weighted_feature
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_state': self.num_state,
        })
        return config

class CSIModelConfig:
    """
    class for Human Activity Recognition ("bed", "fall", "pickup", "run", "sitdown", "standup", "walk")
    Using CSI (Channel State Information)
    Args:
        win_len   :  integer (1000 default) window length for batching sequence
        step      :  integer (200  default) sliding window by this step
        thrshd    :  float   (0.6  default) used to check if the activity is intensive inside a window
        downsample:  integer >=1 (2 default) downsample along the time axis
    """
    def __init__(self, win_len=1000, step=200, thrshd=0.6, downsample=2):
        self._win_len = win_len
        self._step = step
        self._thrshd = thrshd
        self._labels = ("bed", "fall", "pickup", "run", "sitdown", "standup", "walk")
        self._downsample = downsample

    def preprocessing(self, raw_folder, save=False):
        """
        Returns the Numpy Array for training within the format of (X_lable1, y_label1, ...., X_label7, y_label7)
        Args:
            raw_folder: the folder containing raw CSI 
            save      : choose if save the numpy array
        """
        numpy_tuple = extract_csi(raw_folder, self._labels, save, self._win_len, self._thrshd, self._step)
        if self._downsample > 1: # downsampling needed or not
            return tuple([v[:, ::self._downsample,...] if i%2 ==0 else v for i, v in enumerate(numpy_tuple)]) 
        # For elements at even indices (i % 2 == 0), it performs downsampling: v[:, ::self._downsample, ...].
        # For elements at odd indices (i % 2 != 0), it leaves the element unchanged: v.
        # v[:, ::self._downsample, ...]:means that for the array v, take all elements along the first dimension (:), 
        # then take every self._downsample-th element along the second dimension (::self._downsample), and keep all elements along the remaining dimensions (...).
        return numpy_tuple
    
    def load_csi_data_from_files(self, np_files):
        """
        Returns the Numpy Array for training within the format of (X_lable1, y_label1, ...., X_label7, y_label7)
        Args:
            np_files: ('x_bed.npz', 'x_fall.npz', 'x_pickup.npz', 'x_run.npz', 'x_sitdown.npz', 'x_standup.npz', 'x_walk.npz')
        """
        if len(np_files) != 7:
            raise ValueError('There should be 7 numpy files for bed, fall, pickup, run, sitdown, standup, walk.')
        x = [np.load(f)['arr_0'] for f in np_files]
        if self._downsample > 1:
            x = [arr[:,::self._downsample, :] for arr in x]
        y = [np.zeros((arr.shape[0], len(self._labels))) for arr in x]
        numpy_list = []
        for i in range(len(self._labels)):
            y[i][:,i] = 1
            numpy_list.append(x[i])
            numpy_list.append(y[i])
        return tuple(numpy_list)


    
    def build_model(self, n_unit_lstm=200, n_unit_atten=400):
        """
        Returns the Tensorflow Model which uses AttenLayer
        """
        #The input shape is determined based on whether downsampling is used. If self._downsample > 1, the input length is reduced accordingly.
        if self._downsample > 1:
            length = len(np.ones((self._win_len,))[::self._downsample])
            x_in = tf.keras.Input(shape=(length, 90))
        else:
            x_in = tf.keras.Input(shape=(self._win_len, 90))

        #A bidirectional LSTM layer with n_unit_lstm units is applied to the input. 
        # The return_sequences=True parameter ensures that the output contains the full sequence of hidden states for the attention layer.
        x_tensor = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=n_unit_lstm, return_sequences=True))(x_in)
        #The output from the LSTM layer is passed through a custom attention layer (AttenLayer) with n_unit_atten units
        x_tensor = AttenLayer(n_unit_atten)(x_tensor)
        pred = tf.keras.layers.Dense(len(self._labels), activation='softmax')(x_tensor)
        model = tf.keras.Model(inputs=x_in, outputs=pred)
        return model
    
    
    @staticmethod
    def load_model(hdf5path):
        """
        Returns the Tensorflow Model for AttenLayer
        Args:
            hdf5path: str, the model file path
        """
        model = tf.keras.models.load_model(hdf5path, custom_objects={'AttenLayer':AttenLayer})
        return model
    

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Error! Correct Command: python3 csimodel.py Dataset_folder_path")
    raw_data_foler = sys.argv[1]

    # preprocessing
    cfg = CSIModelConfig(win_len=1000, step=200, thrshd=0.6, downsample=2)
    numpy_tuple = cfg.preprocessing('Dataset/Data/', save=True)# the format of (X_lable1, y_label1, ...., X_label7, y_label7)
    # load previous saved numpy files, ignore this if you haven't saved numpy array to files before
    # numpy_tuple = cfg.load_csi_data_from_files(('x_bed.npz', 'x_fall.npz', 'x_pickup.npz', 'x_run.npz', 'x_sitdown.npz', 'x_standup.npz', 'x_walk.npz'))
    x_bed, y_bed, x_fall, y_fall, x_pickup, y_pickup, x_run, y_run, x_sitdown, y_sitdown, x_standup, y_standup, x_walk, y_walk = numpy_tuple
    x_train, y_train, x_valid, y_valid = train_valid_split(
        (x_bed, x_fall, x_pickup, x_run, x_sitdown, x_standup, x_walk),
        train_portion=0.9, seed=379)
    # parameters for Deep Learning Model
    model = cfg.build_model(n_unit_lstm=200, n_unit_atten=400)
    # train
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy', #loss fubnction for multiclss classification
        metrics=['accuracy'])
    model.summary()
    model.fit(
        x_train,
        y_train,
        batch_size=128, epochs=60,#Number of samples per gradient update.
        validation_data=(x_valid, y_valid),
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint('best_atten.keras',
                                                monitor='val_accuracy',
                                                save_best_only=True,
                                                save_weights_only=False)
            ])
    # load the best model
    model = cfg.load_model('best_atten.keras')
    y_pred = model.predict(x_valid)

    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(np.argmax(y_valid, axis=1), np.argmax(y_pred, axis=1)))
