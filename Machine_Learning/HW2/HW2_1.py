import gzip
import struct
import math
import sys

while True:
    flag = input("Enter mode (0 for discrete mode, 1 for continuous mode): ")
    if flag == 0 or flag == 1:
        break

dir_path = './MNIST/'
X_train_path = dir_path + 'train-images-idx3-ubyte.gz'
y_train_path = dir_path + 'train-labels-idx1-ubyte.gz'
X_test_path = dir_path + 't10k-images-idx3-ubyte.gz'
y_test_path = dir_path + 't10k-labels-idx1-ubyte.gz'
bin_num = 32
class_num = 10



# read files
y_train = []
with gzip.open(y_train_path, 'rb') as f:
    y_train_magic, y_train_size = struct.unpack(">II", f.read(8))
    for idx in xrange(y_train_size):
        label = ord(f.read(1))
        y_train.append(label)

X_train = []
with gzip.open(X_train_path, 'rb') as f:
    X_train_magic, X_train_size, X_train_row, X_train_col = struct.unpack(">IIII", f.read(16))
    for idx in xrange(X_train_size):
        img_px = []
        for pxIdx in xrange(X_train_row * X_train_col):
            grey_scale = ord(f.read(1))
            img_px.append(grey_scale)
        X_train.append(img_px)

y_test = []
with gzip.open(y_test_path, 'rb') as f:
    y_test_magic, y_test_size = struct.unpack(">II", f.read(8))
    for idx in xrange(y_test_size):
        label = ord(f.read(1))
        y_test.append(label)

X_test = []
with gzip.open(X_test_path, 'rb') as f:
    X_test_magic, X_test_size, X_test_row, X_test_col = struct.unpack(">IIII", f.read(16))
    for i in xrange(X_test_size):
        img_px = []
        for pxIdx in xrange(X_test_row * X_test_col):
            grey_scale = ord(f.read(1))
            img_px.append(grey_scale)
        X_test.append(img_px)



y_count = [0 for i in xrange(class_num)]
for y in y_train:
    y_count[y] += 1
py = [float(count) / y_train_size for count in y_count]

if flag == 1:
    # conti
    x_sum = [[0 for j in xrange(X_train_row * X_train_col)] for i in xrange(class_num)]
    for px_idx in xrange(X_train_row * X_train_col):
        for idx in xrange(X_train_size):
            grey_scale = X_train[idx][px_idx]
            label = y_train[idx]
            x_sum[label][px_idx] += grey_scale

    x_mean = [[0 for j in xrange(X_train_row * X_train_col)] for i in xrange(class_num)]
    for label in xrange(class_num):
        for px_idx in xrange(X_train_row * X_train_col):
            x_mean[label][px_idx] = float(x_sum[label][px_idx]) / y_count[label]

    x_varience = [[0 for j in xrange(X_train_row * X_train_col)] for i in xrange(class_num)]
    for px_idx in xrange(X_train_row * X_train_col):
        for idx in xrange(X_train_size):
            grey_scale = float(X_train[idx][px_idx])
            label = y_train[idx]
            mean = x_mean[label][px_idx]
            x_varience[label][pxIdx] += (grey_scale - mean) ** 2
    for label in xrange(class_num):
        for px_idx in xrange(X_train_row * X_train_col):
            x_varience[label][px_idx] /= y_count[label]

    # peudo varience
    for label in xrange(class_num):
        for px_idx in xrange(X_train_row * X_train_col):
            if x_varience[label][px_idx] < 16:
                x_varience[label][px_idx] = 16.0

    # testing
    predict = []
    for idx in xrange(X_test_size):
        curr_predict = 0
        max_posterior = -1e9
        for label in xrange(class_num):
            log_likelihood = 0
            for px_idx in xrange(X_test_row * X_test_col):
                grey_scale = float(X_test[idx][px_idx])
                mean = x_mean[label][px_idx]
                varience = x_varience[label][px_idx]
                log_guassian = math.log(1.0 / math.sqrt(varience)) -(math.pow(grey_scale - mean, 2) / (2.0 * varience))
                log_likelihood += log_guassian
            posterior = log_likelihood + math.log(py[label])
            print '{: 12.3f}'.format(posterior),
            if posterior > max_posterior:
                max_posterior = posterior
                curr_predict = label
        predict.append(curr_predict)
        print ""

else:
    # discrete
    x_count = [[[0 for k in xrange(bin_num)] for j in xrange(X_train_row * X_train_col)] for i in xrange(class_num)]
    for idx in xrange(X_train_size):
        for px_idx in xrange(X_train_row * X_train_col):
            grey_scale = X_train[idx][px_idx]
            bin_idx = grey_scale / 8
            label = y_train[idx]
            x_count[label][px_idx][bin_idx] += 1

    # pseudo count
    for label in xrange(class_num):
        for px_idx in xrange(X_train_row * X_train_col):
            min_count = X_train_size
            for bin_idx in xrange(bin_num):
                count = x_count[label][px_idx][bin_idx]
                if count > 0 and count < min_count:
                    min_count = count
            for bin_idx in xrange(bin_num):
                if x_count[label][px_idx][bin_idx] == 0:
                    x_count[label][px_idx][bin_idx] = min_count

    # likelihood
    p_x_given_y = x_count
    for label in xrange(class_num):
        for px_idx in xrange(X_train_row * X_train_col):
            for bin_idx in xrange(bin_num):
                p_x_given_y[label][px_idx][bin_idx] = float(x_count[label][px_idx][bin_idx]) / y_count[label]

    # testing
    predict = []
    for idx in xrange(X_test_size):
        curr_predict = 0
        max_posterior = -1e9
        for label in xrange(class_num):
            log_likelihood = 0
            for px_idx in xrange(X_test_row * X_test_col):
                grey_scale = X_test[idx][px_idx]
                bin_idx = grey_scale / 8
                log_likelihood += math.log(p_x_given_y[label][px_idx][bin_idx])
            posterior = log_likelihood + math.log(py[label])
            print '{: 10.3f}'.format(posterior),
            if posterior > max_posterior:
                max_posterior = posterior
                curr_predict = label
        predict.append(curr_predict)
        print ""

# accuracy
correct_count = 0
for i in xrange(X_test_size):
    if predict[i] == y_test[i]:
        correct_count += 1
error_rate = float(X_test_size - correct_count) / X_test_size
print correct_count
print "\nerror_rate: " + str(error_rate) + "\n"
