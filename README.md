# ANNBank
Given a dataset of 10000 customers, assess the factors behind the churn rate of the bank.

In order to deal with the data pre processing, one hot label encoder was used for the 'countries' column. There were three different countries throughout the dataset. 

Data was normalized using StandardScalar. 

#### An initial accuracy of 79% was achieved. 

#### In order to evaluate the model- 
        1. The first metric used was a confusion matrix to get an idea of the number of True positive and negative values. 
        2. The second metric considered was using cross validation and divide the data into 10 folds. 
        3. The third metric to further optimize the model was Grid Search. 
            i. The combination of hyperparameters that were considered were optimizers, batch size and loss functions.
            ii. It was observed that an accuracy of 85% percent was achieved using the adam optimizer with a batch size of 40. 
