import random
import numpy
from sklearn import metrics
import cPickle


class CampactLatentFactor(object):
    def __init__(self, form, dim=100, diag=False, fold_iter=10000000, learn_rate=0.1, regular=0.003, margin=1., init_stdev=0.1, buffer_size_pos=2000000, buffer_size_neg=2000000, percent_history_pos=.5, percent_history_neg=.5, shuffle_history_buffer=False, verbose=True):
        try:
            self.load()
        except:
            self.verbose = verbose
            self.form = form
            self.test_data = None
            self.latent_factor = {}
            self.dim = dim
            self.diag = diag
            self.fold_iter = fold_iter
            self.learn_rate = learn_rate                        # learning rate
            self.regular = regular                              # regularization term
            self.margin = margin                                # negative score + margin > positive score
            self.init_stdev = init_stdev                        # parameter initialization ~N(0, stdDev^2)
            self.buffer_size_pos = buffer_size_pos
            self.buffer_size_neg = buffer_size_neg
            self.percent_history_pos = percent_history_pos
            self.percent_history_neg = percent_history_neg
            self.shuffle_history_buffer = shuffle_history_buffer
            numpy.seterr(all="raise")

        print "----------------------------------------------------------------------------\n"\
              "libCLF\n"\
              "Version: 2.0\n"\
              "Author:  Xuan-Wei Wu, seanappler@gmail.com\n"\
              "WWW:\n"\
              "License: Free for academic use. See license.txt.\n"\
              "----------------------------------------------------------------------------"

        print "Parameters:"
        for info in ["dim", "margin", "diag", "fold_iter", "learn_rate", "regular", "init_stdev", "buffer_size_pos", "buffer_size_neg", "percent_history_pos", "percent_history_neg", "shuffle_history_buffer"]:
            print "\t--%s: %s" % (info, self.__dict__[info])

    def save(self):
        with open("model.pkl", "wb") as f:
            cPickle.dump(self.__dict__, f, protocol=cPickle.HIGHEST_PROTOCOL)

    def load(self):
        with open("model.pkl", "rb") as f:
            self.__dict__ = cPickle.load(f)

    def __tidy_feature(self, feature_lst):
        feature_set = set()

        for feature in feature_lst:
            feature_pair = feature.split(":")
            feature_pair = (int(feature_pair[0]), float(feature_pair[1]))
            feature_set.add(feature_pair)

        return tuple(sorted(feature_set))

    def train(self, train_file, test_file):
        global_iteration = 0
        print "[*] Buffer %s, auc: %s" % (global_iteration, self.evaluate_with_auc(test_file))
        print "[*] Start training..."

        init_learn_rate = self.learn_rate
        train_fd = open(train_file)
        buffer_pos = []
        buffer_neg = []

        while 1:
            if self.shuffle_history_buffer:
                random.shuffle(buffer_pos)
                random.shuffle(buffer_neg)

            buffer_pos = buffer_pos[-int(self.percent_history_pos * len(buffer_pos) + 1): -1]
            buffer_neg = buffer_neg[-int(self.percent_history_neg * len(buffer_neg) + 1): -1]

            for line in train_fd:
                try:
                    label, user_features, item_features, query_features, scen_features = line.rstrip().split(",")

                    user_feature_set = self.__tidy_feature(user_features.split())
                    item_feature_set = self.__tidy_feature(item_features.split())
                    query_feature_set = self.__tidy_feature(query_features.split())
                    scen_feature_set = self.__tidy_feature(scen_features.split())

                    if self.form == 4:
                        if not all((user_feature_set, item_feature_set, query_feature_set, scen_feature_set)):
                            continue
                    elif self.form == 3:
                        if not all((user_feature_set, item_feature_set, query_feature_set)):
                            continue
                    elif self.form == 2:
                        if not all((user_feature_set, item_feature_set)):
                            continue

                        query_feature_set = user_feature_set
                        user_feature_set = ()

                except ValueError:
                    print("[!] The format of log: %s is illegal." % line)
                    continue

                if label == "1":
                    buffer_pos.append((user_feature_set, item_feature_set, query_feature_set, scen_feature_set))
                else:
                    buffer_neg.append((user_feature_set, item_feature_set, query_feature_set, scen_feature_set))

                if len(buffer_pos) > self.buffer_size_pos and len(buffer_neg) > self.buffer_size_neg:
                    break

            if len(buffer_pos) == 0 or len(buffer_neg) == 0:
                print "[x] There is not data."
                return

            local_iteration = 0

            while 1:
                while 1:
                    pos_data = buffer_pos[int(random.random() * len(buffer_pos))]
                    pos_data_score = self.predict(*pos_data)

                    neg_data = buffer_neg[int(random.random() * len(buffer_neg))]
                    neg_data_score = self.predict(*neg_data)

                    if neg_data_score + self.margin > pos_data_score or int(random.random() * 10) == 0:
                        break

                if neg_data_score + self.margin > pos_data_score:
                    user_feature_set_pos, item_feature_set_pos, query_feature_set_pos, scen_feature_set_pos = pos_data
                    user_feature_set_neg, item_feature_set_neg, query_feature_set_neg, scen_feature_set_neg = neg_data

                    if self.form == 4:
                        scen_feature_num_pos = 0
                        sum_scen_matrix_pos = 0

                        for t in scen_feature_set_pos:
                            scen_feature_num_pos += t[1]
                            sum_scen_matrix_pos += self.latent_factor[t[0]][1] * t[1]

                        sum_scen_matrix_pos /= scen_feature_num_pos

                        scen_feature_num_neg = 0
                        sum_scen_matrix_neg = 0

                        for t in scen_feature_set_neg:
                            scen_feature_num_neg += t[1]
                            sum_scen_matrix_neg += self.latent_factor[t[0]][1] * t[1]

                        sum_scen_matrix_neg /= scen_feature_num_neg

                    if self.form >= 3:
                        user_feature_num_pos = 0
                        sum_user_matrix_pos = 0

                        for t in user_feature_set_pos:
                            user_feature_num_pos += t[1]
                            sum_user_matrix_pos += self.latent_factor[t[0]][1] * t[1]

                        sum_user_matrix_pos /= user_feature_num_pos

                        user_feature_num_neg = 0
                        sum_user_matrix_neg = 0

                        for t in user_feature_set_neg:
                            user_feature_num_neg += t[1]
                            sum_user_matrix_neg += self.latent_factor[t[0]][1] * t[1]

                        sum_user_matrix_neg /= user_feature_num_neg

                        scen_feature_num_pos = 1
                        sum_scen_matrix_pos = numpy.eye(self.dim)
                        scen_feature_num_neg = 1
                        sum_scen_matrix_neg = numpy.eye(self.dim)
                    else:
                        user_feature_num_pos = 1
                        sum_user_matrix_pos = numpy.eye(self.dim)
                        user_feature_num_neg = 1
                        sum_user_matrix_neg = numpy.eye(self.dim)

                        scen_feature_num_pos = 1
                        sum_scen_matrix_pos = numpy.eye(self.dim)
                        scen_feature_num_neg = 1
                        sum_scen_matrix_neg = numpy.eye(self.dim)

                    item_feature_num_pos = 0
                    sum_item_matrix_pos = 0

                    for t in item_feature_set_pos:
                        item_feature_num_pos += t[1]
                        sum_item_matrix_pos += self.latent_factor[t[0]][1] * t[1]

                    sum_item_matrix_pos /= item_feature_num_pos

                    item_feature_num_neg = 0
                    sum_item_matrix_neg = 0

                    for t in item_feature_set_neg:
                        item_feature_num_neg += t[1]
                        sum_item_matrix_neg += self.latent_factor[t[0]][1] * t[1]

                    sum_item_matrix_neg /= item_feature_num_neg

                    query_feature_num_pos = 0
                    sum_query_matrix_pos = 0

                    for t in query_feature_set_pos:
                        query_feature_num_pos += t[1]
                        sum_query_matrix_pos += self.latent_factor[t[0]][1] * t[1]

                    sum_query_matrix_pos /= query_feature_num_pos

                    query_feature_num_neg = 0
                    sum_query_matrix_neg = 0

                    for t in query_feature_set_neg:
                        query_feature_num_neg += t[1]
                        sum_query_matrix_neg += self.latent_factor[t[0]][1] * t[1]

                    sum_query_matrix_neg /= query_feature_num_neg

                    latent_factor_tmp = {}

                    for user_feature_pair in user_feature_set_pos:
                        # regularization
                        factor_bias = self.regular * self.latent_factor[user_feature_pair[0]][0]
                        factor_matrix = self.regular * self.latent_factor[user_feature_pair[0]][1]

                        factor_bias -= user_feature_pair[1] / user_feature_num_pos
                        factor_matrix -= user_feature_pair[1] / user_feature_num_pos * numpy.dot(sum_query_matrix_pos, numpy.dot(sum_scen_matrix_pos, sum_item_matrix_pos).T)

                        if user_feature_pair[0] not in latent_factor_tmp:
                            latent_factor_tmp[user_feature_pair[0]] = [factor_bias, factor_matrix]
                        else:
                            latent_factor_tmp[user_feature_pair[0]][0] += factor_bias
                            latent_factor_tmp[user_feature_pair[0]][1] += factor_matrix

                    for item_feature_pair in item_feature_set_pos:
                        # regularization
                        factor_bias = self.regular * self.latent_factor[item_feature_pair[0]][0]
                        factor_matrix = self.regular * self.latent_factor[item_feature_pair[0]][1]

                        factor_bias -= item_feature_pair[1] / item_feature_num_pos
                        factor_matrix -= item_feature_pair[1] / item_feature_num_pos * numpy.dot(numpy.dot(sum_query_matrix_pos.T, sum_user_matrix_pos), sum_scen_matrix_pos).T

                        if item_feature_pair[0] not in latent_factor_tmp:
                            latent_factor_tmp[item_feature_pair[0]] = [factor_bias, factor_matrix]
                        else:
                            latent_factor_tmp[item_feature_pair[0]][0] += factor_bias
                            latent_factor_tmp[item_feature_pair[0]][1] += factor_matrix

                    for query_feature_pair in query_feature_set_pos:
                        # regularization
                        factor_bias = self.regular * self.latent_factor[query_feature_pair[0]][0]
                        factor_matrix = self.regular * self.latent_factor[query_feature_pair[0]][1]

                        factor_bias -= query_feature_pair[1] / query_feature_num_pos
                        factor_matrix -= query_feature_pair[1] / query_feature_num_pos * numpy.dot(numpy.dot(sum_user_matrix_pos, sum_scen_matrix_pos), sum_item_matrix_pos)

                        if query_feature_pair[0] not in latent_factor_tmp:
                            latent_factor_tmp[query_feature_pair[0]] = [factor_bias, factor_matrix]
                        else:
                            latent_factor_tmp[query_feature_pair[0]][0] += factor_bias
                            latent_factor_tmp[query_feature_pair[0]][1] += factor_matrix

                    for scen_feature_pair in scen_feature_set_pos:
                        # regularization
                        factor_bias = self.regular * self.latent_factor[scen_feature_pair[0]][0]
                        factor_matrix = self.regular * self.latent_factor[scen_feature_pair[0]][1]

                        factor_bias -= scen_feature_pair[1] / scen_feature_num_pos
                        factor_matrix -= scen_feature_pair[1] / scen_feature_num_pos * numpy.dot(sum_query_matrix_pos, numpy.dot(sum_user_matrix_pos, sum_item_matrix_pos).T)

                        if scen_feature_pair[0] not in latent_factor_tmp:
                            latent_factor_tmp[scen_feature_pair[0]] = [factor_bias, factor_matrix]
                        else:
                            latent_factor_tmp[scen_feature_pair[0]][0] += factor_bias
                            latent_factor_tmp[scen_feature_pair[0]][1] += factor_matrix

                    for user_feature_pair in user_feature_set_neg:
                        # regularization
                        factor_bias = self.regular * self.latent_factor[user_feature_pair[0]][0]
                        factor_matrix = self.regular * self.latent_factor[user_feature_pair[0]][1]

                        factor_bias += user_feature_pair[1] / user_feature_num_neg
                        factor_matrix += user_feature_pair[1] / user_feature_num_neg * numpy.dot(sum_query_matrix_neg, numpy.dot(sum_scen_matrix_neg, sum_item_matrix_neg).T)

                        if user_feature_pair[0] not in latent_factor_tmp:
                            latent_factor_tmp[user_feature_pair[0]] = [factor_bias, factor_matrix]
                        else:
                            latent_factor_tmp[user_feature_pair[0]][0] += factor_bias
                            latent_factor_tmp[user_feature_pair[0]][1] += factor_matrix

                    for item_feature_pair in item_feature_set_neg:
                        # regularization
                        factor_bias = self.regular * self.latent_factor[item_feature_pair[0]][0]
                        factor_matrix = self.regular * self.latent_factor[item_feature_pair[0]][1]

                        factor_bias += item_feature_pair[1] / item_feature_num_neg
                        factor_matrix += item_feature_pair[1] / item_feature_num_neg * numpy.dot(numpy.dot(sum_query_matrix_neg.T, sum_user_matrix_neg), sum_scen_matrix_neg).T

                        if item_feature_pair[0] not in latent_factor_tmp:
                            latent_factor_tmp[item_feature_pair[0]] = [factor_bias, factor_matrix]
                        else:
                            latent_factor_tmp[item_feature_pair[0]][0] += factor_bias
                            latent_factor_tmp[item_feature_pair[0]][1] += factor_matrix

                    for query_feature_pair in query_feature_set_neg:
                        # regularization
                        factor_bias = self.regular * self.latent_factor[query_feature_pair[0]][0]
                        factor_matrix = self.regular * self.latent_factor[query_feature_pair[0]][1]

                        factor_bias += query_feature_pair[1] / query_feature_num_neg
                        factor_matrix += query_feature_pair[1] / query_feature_num_neg * numpy.dot(numpy.dot(sum_user_matrix_neg, sum_scen_matrix_neg), sum_item_matrix_neg)

                        if query_feature_pair[0] not in latent_factor_tmp:
                            latent_factor_tmp[query_feature_pair[0]] = [factor_bias, factor_matrix]
                        else:
                            latent_factor_tmp[query_feature_pair[0]][0] += factor_bias
                            latent_factor_tmp[query_feature_pair[0]][1] += factor_matrix

                    for scen_feature_pair in scen_feature_set_neg:
                        # regularization
                        factor_bias = self.regular * self.latent_factor[scen_feature_pair[0]][0]
                        factor_matrix = self.regular * self.latent_factor[scen_feature_pair[0]][1]

                        factor_bias -= scen_feature_pair[1] / scen_feature_num_neg
                        factor_matrix -= scen_feature_pair[1] / scen_feature_num_neg * numpy.dot(sum_query_matrix_neg, numpy.dot(sum_user_matrix_neg, sum_item_matrix_neg).T)

                        if scen_feature_pair[0] not in latent_factor_tmp:
                            latent_factor_tmp[scen_feature_pair[0]] = [factor_bias, factor_matrix]
                        else:
                            latent_factor_tmp[scen_feature_pair[0]][0] += factor_bias
                            latent_factor_tmp[scen_feature_pair[0]][1] += factor_matrix

                    for feature, factor in latent_factor_tmp.iteritems():
                        self.latent_factor[feature][0] -= self.learn_rate * factor[0]
                        self.latent_factor[feature][1] -= self.learn_rate * factor[1]

                self.learn_rate = max(self.learn_rate * 0.96, 0.5 * init_learn_rate)
                local_iteration += 1

                if local_iteration > self.fold_iter:
                    global_iteration += 1

                    if self.verbose:
                        current_auc = self.evaluate_with_auc(test_file)
                        print "[*] Buffer %s, auc: %s, \n pos: %s -> %s,\n neg: %s -> %s" % (global_iteration, current_auc, pos_data_score, self.predict(*pos_data), neg_data_score, self.predict(*neg_data))

                    break
            if len(buffer_pos) <= self.buffer_size_pos or len(buffer_neg) <= self.buffer_size_neg:
                if not self.verbose:
                    current_auc = self.evaluate_with_auc(test_file)
                    print "[*] Buffer %s, auc: %s, \n pos: %s -> %s,\n neg: %s -> %s" % (global_iteration, current_auc, pos_data_score, self.predict(*pos_data), neg_data_score, self.predict(*neg_data))

                break

    def predict(self, user_feature_set, item_feature_set, query_feature_set, scen_feature_set):
        sum_user_bias = 0
        sum_item_matrix = 0
        sum_item_bias = 0
        sum_query_matrix = 0
        sum_query_bias = 0
        sum_scen_bias = 0
        item_feature_num = 0
        query_feature_num = 0

        if self.form == 4:
            sum_scen_matrix = 0
            scen_feature_num = 0

            for scen_feature_pair in scen_feature_set:
                if scen_feature_pair[0] not in self.latent_factor:
                    if self.diag:
                        self.latent_factor[scen_feature_pair[0]] = [self.init_stdev * numpy.random.randn(), self.init_stdev * numpy.diag(numpy.random.randn(self.dim, 1))]
                    else:
                        self.latent_factor[scen_feature_pair[0]] = [self.init_stdev * numpy.random.randn(), self.init_stdev * numpy.random.randn(self.dim, self.dim)]

                sum_scen_matrix += scen_feature_pair[1] * self.latent_factor[scen_feature_pair[0]][1]
                sum_scen_bias += scen_feature_pair[1] * self.latent_factor[scen_feature_pair[0]][0]
                scen_feature_num += scen_feature_pair[1]

        if self.form >= 3:
            sum_user_matrix = 0
            user_feature_num = 0

            for user_feature_pair in user_feature_set:
                if user_feature_pair[0] not in self.latent_factor:
                    if self.diag:
                        self.latent_factor[user_feature_pair[0]] = [self.init_stdev * numpy.random.randn(), self.init_stdev * numpy.diag(numpy.random.randn(self.dim, 1))]
                    else:
                        self.latent_factor[user_feature_pair[0]] = [self.init_stdev * numpy.random.randn(), self.init_stdev * numpy.random.randn(self.dim, self.dim)]

                sum_user_matrix += user_feature_pair[1] * self.latent_factor[user_feature_pair[0]][1]
                sum_user_bias += user_feature_pair[1] * self.latent_factor[user_feature_pair[0]][0]
                user_feature_num += user_feature_pair[1]

            sum_scen_matrix = numpy.eye(self.dim)
            scen_feature_num = 1
        else:
            sum_user_matrix = numpy.eye(self.dim)
            user_feature_num = 1

            sum_scen_matrix = numpy.eye(self.dim)
            scen_feature_num = 1

        for item_feature_pair in item_feature_set:
            if item_feature_pair[0] not in self.latent_factor:
                self.latent_factor[item_feature_pair[0]] = [self.init_stdev * numpy.random.randn(), self.init_stdev * numpy.random.randn(self.dim, 1)]

            sum_item_matrix += item_feature_pair[1] * self.latent_factor[item_feature_pair[0]][1]
            sum_item_bias += item_feature_pair[1] * self.latent_factor[item_feature_pair[0]][0]
            item_feature_num += item_feature_pair[1]

        for query_feature_pair in query_feature_set:
            if query_feature_pair[0] not in self.latent_factor:
                self.latent_factor[query_feature_pair[0]] = [self.init_stdev * numpy.random.randn(), self.init_stdev * numpy.random.randn(self.dim, 1)]

            sum_query_matrix += query_feature_pair[1] * self.latent_factor[query_feature_pair[0]][1]
            sum_query_bias += query_feature_pair[1] * self.latent_factor[query_feature_pair[0]][0]
            query_feature_num += query_feature_pair[1]

        return (numpy.dot(numpy.dot(numpy.dot(sum_query_matrix.T, sum_user_matrix), sum_scen_matrix), sum_item_matrix) / (query_feature_num * user_feature_num * scen_feature_num * item_feature_num))[0][0] + sum_user_bias / user_feature_num + sum_item_bias / item_feature_num + sum_query_bias / query_feature_num + sum_scen_bias / scen_feature_num

    def read_test_file(self, test_file):
        print("[*] Loading test...")

        self.test_data = [[], []]

        for line in open(test_file):
            try:
                label, user_features, item_features, query_features, scen_features = line.strip().split(",")

                user_feature_set = self.__tidy_feature(user_features.split())
                item_feature_set = self.__tidy_feature(item_features.split())
                query_feature_set = self.__tidy_feature(query_features.split())
                scen_feature_set = self.__tidy_feature(scen_features.split())

                if self.form == 4:
                    if not all((user_feature_set, item_feature_set, query_feature_set, scen_feature_set)):
                        continue
                elif self.form == 3:
                    if not all((user_feature_set, item_feature_set, query_feature_set)):
                        continue
                elif self.form == 2:
                    if not all((user_feature_set, item_feature_set)):
                        continue

                    query_feature_set = user_feature_set
                    user_feature_set = ()
            except ValueError:
                print("[!] The format of log: %s is illegal." % line)
                continue

            self.test_data[0].append(int(label))
            self.test_data[1].append((user_feature_set, item_feature_set, query_feature_set, scen_feature_set))

    def evaluate_with_auc(self, test_file):
        if not self.test_data:
            self.read_test_file(test_file)

        pred = [self.predict(*data) for data in self.test_data[1]]

        fpr, tpr, thresholds = metrics.roc_curve(self.test_data[0], pred, pos_label=1)

        return metrics.auc(fpr, tpr)

if __name__ == "__main__":
    import sys
    base = sys.argv[1]
    test = sys.argv[2]

    CLF = CampactLatentFactor(form=3)
    CLF.train(base, test)
