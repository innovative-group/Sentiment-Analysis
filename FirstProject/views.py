from django.http import HttpResponse
from django.shortcuts import render
from datetime import datetime
from django.shortcuts import redirect


import re
import random
from math import log


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np




def about(request):
    return render(request, 'about.html')

def contact(request):
    return HttpResponse("Contact done!")



def registration(request):
    
    cur_time= datetime.now()

     # Read data from CSV file and store in list

 
    
   # Sample data for training and testing
    data = [
        ['do not like', 'negative'],
        ['have good', 'positive'],
        ['do not have good', 'negative'],
        ['do not have bad', 'positive'],
        ['not impressed', 'negative'],
        ['do nice', 'positive'],
        ['It is improving', 'positive'],
        ['more theoritical but less practical', 'negative'],
        ['Nepal education system have both positive & negative side.', 'neutral'],
        ['Nepal education system is  like average.', 'neutral'],
        ['education system is bad.', 'negative'],
        ['good.', 'positive'],
      
    ]


    # Split the data into training and testing sets & extracting the testing_labels
    def split_data(data, test_ratio):
        """
        Split the input data into training and testing data sets & also extract 'labels' of each sentence of 'test_data'.
        :param data: List of tuples where each tuple represents a data point and its corresponding label.
        :param test_ratio: The ratio of data to be used for testing. The value must be between 0 and 1.
        :return: A tuple of two lists containing the training, testing data  & labels of each sentence of 'test_data' respectively.
        """
        print("\n\n ---------->> Split function is called. <<-----------\n\n")

        random.shuffle(data)  # Shuffle the data randomly.
        split_index = int(len(data) * test_ratio)  # Compute the index to split the data.
        test_data = data[:split_index]  # Extract the testing data.
        train_data = data[split_index:]  # Extract the training data.
        test_labels = [label for (_, label) in test_data]  # Extract the labels for the testing data.
        test_data = [data_point for (data_point, _) in test_data]  # Extract the testing data points.

        return train_data, test_data, test_labels


    # Split the data into training and testing sets & extracting the testing_labels
    def test_data_only(data, test_ratio):
        
        print("\n\n%%%%%>> extracting only test_data <<%%%%%%")

        n = len(data)
        
        test_size = int(n * test_ratio)
        random.shuffle(data)                                                                   # here shuffle() function reorder the position of list element randomly

        return data[:test_size]        #-------->> The split is done by slicing the data array in two parts using Python's slice notation.

    # Preprocess the text data
    def preprocess_text(text, location):    
        print("===========>> preprocess() called from: {} <<============".format(location))
        print("  Before preprocessing:", text)

        text = text.lower() # Convert to lowercase
        text = re.sub(r'http\S+', '', text) # Remove URLs
        text = re.sub(r'\d+', '', text) # Remove digits
        text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
        text = re.sub(r'\s+', ' ', text) # Replace multiple spaces with single space
        text = text.strip() # Remove leading and trailing whitespace
        print("  After preprocessing:", text)
        print("---------------|end of preprocess|--------------------\n")
        return text


    def preprocess_feedBackOnly(text):    

        
        print(" \n -------->> Process_feedbackOnly ")
        text = text.lower() # Convert to lowercase
        text = re.sub(r'http\S+', '', text) # Remove URLs
        text = re.sub(r'\d+', '', text) # Remove digits
        text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
        text = re.sub(r'\s+', ' ', text) # Replace multiple spaces with single space
        text = text.strip() # Remove leading and trailing whitespace
        
        return text




    # Compute the class prior probability of each class
    def compute_prior(data):
        print("%%%%%%%%%%%%%%>> I am compute_prior Function <<%%%%%%%%%%%%%%")
        n = len(data)

        positive_count = sum(1 for row in data if row[1] == 'positive')
        print("Total positive data= ",positive_count)
        negative_count = sum(1 for row in data if row[1] == 'negative')
        print("\nTotal negative data= ",negative_count)
        neutral_count = n - positive_count - negative_count
        print("\nTotal neutral data= ",neutral_count)
        print("-------------------|end of comput_prior|-------------------\n\n")
        return {'positive': positive_count / n, 'negative': negative_count / n, 'neutral': neutral_count / n}        #---> here we have key: value_pair so, its a dictionary.
    


        '''

        To calculate the proportion of each label type in the dataset, the count of each label type is divided by the total number of items in the dataset 👎.
        This normalization is done to get the proportions of the different label types, which can then be used to analyze the distribution of the labels in the dataset.        

        '''


    # Compute the conditional probability of each word given each class
    def compute_likelihood(data):
        print("%%%%%%%%%%%%%%>> I am compute_likelihood Function <<%%%%%%%%%%%%%%")
        word_count = {'positive': {}, 'negative': {}, 'neutral': {}}           #--> here "word_count" is a dictionary because it has key & value pair
        word_total = {'positive': 0, 'negative': 0, 'neutral': 0}              #--> example "word_count": {'positive': {good, nice, like}, 'negative': {not, bad, theoritical}, 'neutral': {not bad, normal}}

        for ix, row in enumerate(data):                                                #  enumerate returns an enumerate object, which is an iterator that generates a series of tuples containing both
         
            print()
            #print(ix, row[0])
            # above print statement print as below i.e index & sentence
            # 0 I have Nepal education system.
            text = preprocess_text(row[0], "[ likelyhood() ]")
            text_label= row[1]

            words = text.split()
            print("Tokenize: ",words, ":", text_label)                 #-------> print data as:: ['nepal', 'education', 'system', 'is', 'not', 'so', 'bad']

            print("")

            for index, word in enumerate(words): 
                #print(word)

                if word not in word_count[row[1]]:
                    word_count[row[1]][word] = 0                                    #------>> As word count have "keys(either positive, negative or neutral)" and its  value "{}" and here "row[1]" give the class label and "[word]" gives a particular split word

                word_count[row[1]][word] += 1                                       #------->> weather if condition is valid or not but this statement execute so, if condition not satisfied then that particular word will be stored in respective Keys of word_count dictionary
                word_total[row[1]] += 1
                print("--->>", index, row[1], word_count[row[1]])

        for cls in ['positive', 'negative', 'neutral']:
            print("")
            print("")
            print("[",cls,"]", "class word length= ", word_total[cls])
            print("------------------------------")
            #------>> "cls" means class
            for word in word_count[cls]:
                word_freq= word_count[cls][word],
                
                #conditional_probability_of_word: cp_word
                cp_word=  word_count[cls][word]/word_total[cls]
                print("[",word,"]", "count= ",word_freq, "class[",word_total[cls],"]","word probability= ", cp_word)

            print("")
            
            
        
        print("----------------------|end of likelyhood()|----------------------------\n\n\n")
    
        return word_count






   
    '''
        The function initializes a dictionary called log_prob with the log prior probabilities of each class. The log probabilities are used instead
          of raw probabilities to avoid underflow issues when multiplying small probabilities together.


          
          If the word is present in the likelihood dictionary for the positive class (likelihood['positive']), its log likelihood probability is added to
         the log_prob['positive'] value. This is done by taking the logarithm of the likelihood probability using the log() function. The corresponding
         log probability is also printed.

        If the word is not present in the likelihood dictionary for the positive class, a very small log likelihood probability (log(1e-10)) is added to the
        log_prob['positive'] value. This is a form of smoothing to handle words that were not seen during training.
    
    '''


    '''
     if the word is present in the likelihood['positive'] dictionary, and if it is, it adds the logarithm of the corresponding likelihood value to 
     the log_prob['positive'] variable.
    
    '''
    # Compute the log posterior probability of each class given a feedback
    def predict(feedback, prior, called_from_msg, likelihood):
        print("%%%%%%%%%%%%%%>> I am main predict Function <<%%%%%%%%%%%%%%")
  
        feedback = preprocess_text(feedback, "[ predict() ]")
        words = feedback.split()
        
        #--> It initializes a dictionary called log_prob with the log prior probabilities of each class.
        log_prob = {'positive': log(prior['positive']), 'negative': log(prior['negative']), 'neutral': log(prior['neutral'])}  
        for word in words:
            if word in likelihood['positive']:
                #print(" found ", word, "=", likelihood)
                log_prob['positive'] += log(likelihood['positive'][word])
                print("probability value of positive sentence word [",word, "] is= ", log_prob['positive'])
            

            else:
                #print("Not found ", word, "=", likelihood)
                log_prob['positive'] += log(1e-10)

            if word in likelihood['negative']:
                log_prob['negative'] += log(likelihood['negative'][word])
                print("probability value of negative sentence word [",word, "] is= ", log_prob['negative'])

            else:
                log_prob['negative'] += log(1e-10)                  

            if word in likelihood['neutral']:                                   
                log_prob['neutral'] += log(likelihood['neutral'][word]) 
                print("probability value of neutral sentence word [",word, "] is= ", log_prob['neutral'])

            else:
                log_prob['neutral'] += log(1e-10)
         
        print("---------------|end of predict() |--------------------")


        '''

            log(1e-10) is the natural logarithm (base e) of 1e-10.
            the line log_prob['positive'] += log(1e-10) is executed when a word in the feedback is not found in the likelihood dictionary
            for the 'positive' class. The value 1e-10 represents 1 multiplied by 10 raised to the power of -10, which is a very small 
             number close to zero.


            A small value like 1e-10 is used to solve the problem of zero probability.
            When a word in the feedback is not found in the training data, the corresponding             
            probability will be zero. And the zero probability of that single word won't make the total probability as zero.

        '''  
        return max(log_prob, key=log_prob.get)

    

    """
        Notes about below "compute_confusion_matrix()"
        
        i> classes = ['positive', 'negative', 'neutral']
           print(range(len(classes))

           what will be the output ?

           Ans: The output of the above code will be:
                range(0, 3)
                Here, len(classes) is 3, so range(len(classes)) returns an iterable object containing numbers from 0 up to (but not including) 3,
                which are 0, 1 and 2. This iterable object is passed to the range() function, which returns a range object that represents the sequence
                of numbers from 0 to 2. The print() function then prints this range object.

    
    
    """



  
    


    def compute_confusion_matrix(test_data, predicted_labels, actual_labels):
        print("\n\n================>> confusion_matrix() is called <<==================\n")
        classes = ['positive', 'negative', 'neutral']
        confusion_matrix = np.zeros((3, 3), dtype=int)


        print(predicted_labels)
        print(actual_labels)
        print("\n\n")
        
        for i in range(len(test_data)):
            predicted_class = predicted_labels[i]
            actual_class = actual_labels[i]
            predicted_index = classes.index(predicted_class)
            actual_index = classes.index(actual_class)
            confusion_matrix[actual_index][predicted_index] += 1


            print(confusion_matrix)
            print("\n")

            # Calculate total number of predictions
            total = sum(sum(confusion_matrix))  # ---> inner sum() function performs sum of all values in columns wise which convert our matrix to only one row with 3 dimensional element
                                                # and then the outer sum() function will perfom sumation on that single row to generate a single value. 
                                                # for reference visit: "sum().jpg": D:\seven Semester\Project\sentiment analysis\Final Final project\Notes



            # Calculate true positives, true negatives, false positives, and false negatives for each class
            tp = [confusion_matrix[i][i] for i in range(len(classes))]  # range(len(classes)) ---> generate the list of index of 
            
            # tn = [sum([confusion_matrix[j][k] for j in range(len(classes)) for k in range(len(classes)) if j != i and k != i]) for i in range(len(classes))]  #OR
            # ------>> calculating the "True Negative" [TN] <<------------                                                  
            tn = []
            for i in range(len(classes)):
                tn_i = 0
                for j in range(len(classes)):
                    for k in range(len(classes)):
                        if j != i and k != i:
                            tn_i += confusion_matrix[j][k]
                tn.append(tn_i)   
            # -------------------------------------------------------------      



            # fp = [sum([confusion_matrix[j][i] for j in range(len(classes))]) - tp[i] for i in range(len(classes))]    OR
            # ------>> calculating the "True Negative" [TN] <<------------                                                  
            fp = []
            for i in range(len(classes)):
                fp_count = 0
                for j in range(len(classes)):
                    if j != i:
                        fp_count += confusion_matrix[j][i]
                fp_count -= tp[i]
                fp.append(fp_count)
            # -------------------------------------------------------------          
            
            
            # fn = [sum([confusion_matrix[i][j] for j in range(len(classes))]) - tp[i] for i in range(len(classes))]          OR
            fn = []
            for i in range(len(classes)):
                sum_fn = 0
                for j in range(len(classes)):
                    if i != j:
                        sum_fn += confusion_matrix[i][j]
                fn.append(sum_fn)


            # Calculate accuracy, error rate, precision, and recall for each class
            accuracy = [tp[i] / total for i in range(len(classes))]
            error_rate = [1 - accuracy[i] for i in range(len(classes))]
            precision = [tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0 for i in range(len(classes))]
            recall = [tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0 for i in range(len(classes))]

            

        # print("\n\nAccuracy:", accuracy)
        # print("Error Rate:", error_rate)
        # print("Precision:", precision)
        # print("Recall:", recall)
        # print("\n============>> end of confusion matrix <<=============\n\n")
        # print("\n\n")
        
        



    print("\n--------->> This registration() will be called twice.<<-----------")
    print("i> when form load for ist time.")
    print("ii> when we click on 'classify' button.")


    def compute_accuracy():
        # Make predictions on the test data
        
        test_data_= test_data_only(data, 0.2)        
        predicted_labels = []
        actual_labels = []
        for row in test_data_ :
            feedback, actual_label = row
            predicted_label = predict(feedback, prior, "predict() called from accuracy value",  likelihood)
            predicted_labels.append(predicted_label)
            actual_labels.append(actual_label)

        # Compute accuracy
        correct = sum(1 for i in range(len(test_data)) if predicted_labels[i] == actual_labels[i])
        accuracy = correct / len(test_data) * 100
            
        return accuracy

   
    # spliting the data into training, testing  & extracting 'label' of "text"
    train_data, test_data, test_labels = split_data(data, 0.2)

    # -------->> printing the training data <<-------------------

    print("---------------->> Training Dataset <<------------------------")
    for a, b in enumerate(train_data):
        print(" index: ", a, "   input= ", b)

    print("---------------------------------------------------------------")

    print("--------------->> Testing data <<-----------------------------")
    for a, b in enumerate(test_data):
        print(" index: ", a, "   input= ", b)

    print("--------------------------------------------------------------")    
                   


    if request.method == 'POST':
        
           

        opinion = request.POST.get('opinion')
        if opinion:
            feedback = opinion



            feedbackPreProcess= preprocess_feedBackOnly(feedback)
            



            
            # Train the model
            print("\n\n Here [compute_prior() is called.")
            prior = compute_prior(train_data)
            print("\n\n Here compute_likelihood() is called.")
            likelihood = compute_likelihood(train_data)

            print("---------------------->> printing frequency of each word in each class <<---------------------")
            for a, b in enumerate(likelihood.items()):
               print(":", a, ":", b)

            print("----------------------------------------------------------------------------------------------------------------------\n\n\n")   

            
            
            classification = predict(feedback, prior, "\npredict() from_classification", likelihood)
            print("\n\n\n")
            print("==================================================")
            print("|                 The given feedback             |")
            print("==================================================")
            print("    ", feedback,"= ", classification , "\n\n")
            print("\nAccuracy of model: {}\n\n".format(compute_accuracy()))


         


            
            predicted_labels = [predict(text, prior, "\npredict() for confusion matrix", likelihood) for text in test_data]
            compute_confusion_matrix(test_data, predicted_labels, test_labels)
           

            
        

            if classification == "positive":
                img_path = 'image/pos_img.jpg'

            elif classification == "negative":
                img_path = 'image/neg_img.jpg'

            elif classification == "neutral":
                img_path = 'image/neu_img.jpg'

            else:
                img_path = 'image/default.jpg'  # a default image for unknown classification


            feedback_tokenize= feedbackPreProcess.split()
            context = {'classification': classification, 'img_path': img_path, 'inputText': feedback, 'feedbackPreProcessText': feedbackPreProcess, 'feedback_tokenize': feedback_tokenize}
        

            
            return render(request, 'reg.html', context)
    else:
        print("Opinion is empty.")    
            
        

    return render(request, 'reg.html')