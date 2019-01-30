from collections import defaultdict

from surprise import Dataset
from surprise import SVD
from surprise.model_selection import KFold
import time

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        # print('estimation and true_r : ',est,true_r)
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls

start_time = time.time()


data = Dataset.load_builtin('ml-100k')
kf = KFold(n_splits=5)
algo = SVD()
counter = 0
prec_sum = 0
recall_sum = 0
prec_mean = 0
recall_mean = 0
for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)

    precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=4)

    # Precision and recall can then be averaged over all users
    # print('percision average for fold ',counter,': ',sum(prec for prec in precisions.values()) / len(precisions))
    # print('recall average for fold ',counter,': ',sum(rec for rec in recalls.values()) / len(recalls))
    prec_sum += sum(prec for prec in precisions.values()) / len(precisions)
    recall_sum += sum(rec for rec in recalls.values()) / len(recalls)
    # print('\n--------------------------------------------------\n')
    counter +=1

prec_mean = prec_sum / counter
recall_mean = recall_sum / counter


print("precision mean kfold = 5 is: ",prec_mean, time.time() - start_time)
print("recall mean kfold = 5 is: ",recall_mean, time.time() - start_time)
