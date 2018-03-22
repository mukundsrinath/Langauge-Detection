**Dataset:** European Parliament Proceedings Parallal Corpus  
**Model:** Logistic Regression  
**Accuracy:** 0.775398502017  
<br />
**Problem**  
The problem in question is that of detection of the language of a given sentence based on the datasest of the the European parliament proceedings parallal corpus which contains 21 spoken European languages. The full dataset contains over 8000 document samples for each of the 21 languages.
<br />  
**Training**  
500 documents for each of the language were used for the purpose of training the logistic regression model. Although the model will perform better with the use of the larger dataset, only a small subset was used due to a tradeoff with time and computationl power. I suspect that the model will perform best when trained using a decision tree. Comparison with other model will be updated soon. 
Each word was split using whitespace and TF-IDF of a sliding-window 3-gram was used to create the features. 
<br />  
**Usage**  
root_dir in train.py needs to be changed to the appropriate folder containing all the language folders. Running train.py will create regression.pkl, lang_list.pkl and vocab.pkl. The file test.py consumes these three files to provide the result below.
Note: test.py can be run separately as the 3 .pkl files have been uploaded
<br />  
**Result**  
Overall Accuracy: 0.775398502017  

             precision    recall  f1-score   support

         bg       1.00      1.00      1.00      1000
         hu       0.98      0.41      0.58      1000
         sk       0.89      0.99      0.94      1000
         it       0.90      1.00      0.95      1000
         es       1.00      1.00      1.00       992
         el       0.85      1.00      0.92      1000
         et       0.75      0.98      0.85      1000
         lv       0.90      0.11      0.20      1000
         ro       0.55      1.00      0.71      1000
         fi       0.76      0.99      0.86      1000
         pl       0.98      0.68      0.80      1000
         sl       0.45      1.00      0.62      1000
         sv       0.99      0.49      0.66      1000
         en       1.00      0.69      0.82       979
         lt       0.90      1.00      0.95      1000
         nl       1.00      0.78      0.87      1000
         da       0.44      1.00      0.61      1000
         de       0.99      0.67      0.80       928
         pt       0.81      0.25      0.38       929
         fr       0.87      0.22      0.35      1000
         cs       0.90      0.99      0.94      1000
         
                  0.85      0.78      0.75     20828

