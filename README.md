Face Recognition Challenge


Since I was using a remote server to speed up training I was unable to properly develop a jupyter notebook in parallel. However I can summarize my steps and what I did over the 12 hours or so I spent actively working on this. 


I initially tried to develop a multi output model that would train a single sequential model with two separate branches of dense layers. (train_model_v1) One for gender and one for race. This resulted in a poor model so I switched to a parallel sequential model that would train two vgg16 based models which resulted in significant improvement. (train_model_v2-4) These two initial training attempts were ‘vanilla’ in the sense that there was no attempt at dealing with potential issues like overfitting and data imbalance. 


Once the latter model proved viable. I did data analysis to view the data imbalance in the classes. This can be run using train_mode_v4.py and passing the ‘-p’ flag which will open up a window that shows the breakdown of races and genders. After viewing the discrepancies across race and gender - I first tried training using stratified cross validation. This proved harmful mainly due to the small size of the minority classes in data. I then trained three separate models based off of it with varying batch sizes (16, 32, 64) along with weighted classes to deal with the imbalance of different races. Training sessions over 100 epochs took very long and even though I was developing on a dedicated server these still took hours to complete. If I had more time I would have tried varying class weights along with more in depth data augmentation for the minority classes and more research and development of other data balancing solutions. 


The best model seemed to be the vanilla model which used a batch size of 32 and did not use class weights. Each of the v4 models can be tested using the test.py file i’ve provided which will output metrics including a confusion matrix of the predictions on a test set. 


I did not have time to develop a script to load and predict linked-in profiles. However I do think it would be easy to do properly. I would have built a script that would webscrape the user's name and profile picture. Then I would have cropped the face to fit the datasets format of cropped faces. This would be done using the semantic segmentation tool body-pix which I used extensively for my senior project which can detail the pixels of an image that make up a persons face (along with other body parts). With the cropped images all that I would need to do would be to resize the image to the dimensions specified by the model (200x200). And the script would be complete.


Again, I would like to apologize for not using a jupyter notebook. Given my resource constraints I had to develop in a unix server based environment and was unable to develop a parallel notebook. Had I had more time and had I not had plans over the last weekend I’m sure my results would be better and I would have explored different models, data balancing methods and been able to build the linkedin webscraper/evaluator. I appreciate the opportunity presented and hope this is a good enough fit to move forward. Thank you for your consideration.
