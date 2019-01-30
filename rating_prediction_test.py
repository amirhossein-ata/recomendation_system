import zipfile
from surprise import Reader, Dataset, SVD, evaluate

# Unzip ml-100k.zip
# zipfile = zipfile.ZipFile('ml-100k.zip', 'r')
# zipfile.extractall()
# zipfile.close()

# Read data into an array of strings
with open('./ml-100k/u.data') as f:
    all_lines = f.readlines()

# Prepare the data to be used in Surprise
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file('./ml-100k/u.data', reader=reader)

# data = Dataset.load_builtin('ml-100k/u.data')

# Split the dataset into 5 folds and choose the algorithm
data.split(n_folds=4)
algo = SVD()

# Train and test reporting the RMSE and MAE scores
evaluate(algo, data, measures=['RMSE', 'MAE'])

# Retrieve the trainset.
trainset = data.build_full_trainset()
algo.train(trainset)

# Predict a certain item
movie_rating_dic_1={264:3,303:5,361:5,357:4,260:4,356:3,294:5,288:4,50:5,354:5,271:4,300:5,328:3,258:5,210:3,329:5,11:4,327:5,324:5,359:5,362:5,358:2,360:5,301:5,2:3}
movie_rating_dic_2={298:5,691:5,521:5,487:5,286:5,6:5,479:4,340:4,527:3,507:4,276:4,615:4,690:1,294:4,483:5,402:4,371:5,242:4,201:5,50:5,7:4,385:5}
sum = 0
for i in movie_rating_dic_1.keys():
    predict = algo.predict(str(9),str(i),movie_rating_dic_1[i])
    sum += (predict.r_ui - predict.est)**2
    # print(predict.r_ui - predict.est)
    print(predict)
# print('RMSE: ',(sum/len(movie_rating_dic_1.keys())**.5))