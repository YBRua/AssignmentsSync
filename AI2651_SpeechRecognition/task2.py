import os
import pickle
import vad.task2.pipeline as task2Ppl
from vad.classifiers.dualGMM import DualGMMClassifier

train_set_path = './wavs/train'
train_label_path = './data/train_label.txt'
dev_set_path = './wavs/dev'
dev_label_path = './data/dev_label.txt'
test_set_path = './wavs/test'


if __name__ == "__main__":
    # training and evaluation
    VADClassifier = DualGMMClassifier(
        n_components=3,
        covariance_type='full',
        max_iter=500,
        verbose=1,
        random_state=1919810,)

    if os.path.exists('./vad/task2/pretrained_model.pkl'):
        print('Loading pretrained model...')
        VADClassifier = pickle.load(
            open('./vad/task2/pretrained_model.pkl', 'rb'))
    else:
        print('Training model...')
        task2Ppl.train(VADClassifier, train_set_path, train_label_path)
        # save model
        pickle.dump(
            VADClassifier,
            open('./vad/task2/pretrained_model.pkl', 'wb'))

    task2Ppl.evaluate(VADClassifier, dev_set_path, dev_label_path)

    # testing
    task2Ppl.run_on_test(VADClassifier, test_set_path)
