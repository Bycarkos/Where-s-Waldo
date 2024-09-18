
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score


import pdb

def record_linkage_with_logistic_regression(X_train: np.array, 
                                            y_train: np.array, 
                                            X_test: np.array, 
                                            y_test: np.array,
                                            candidate_sa,
                                          **kwargs):

    print(np.sum(y_test))

    # Create a logistic regression model
    model = LogisticRegression(**kwargs)

    # Train the model
    model.fit(X_train, y_train)
    
    ## population consistency
    final_y_pred = []
    final_y_test = []
    
    ## specificity
    specificity = []
    
    mask_true_pairs = candidate_sa[-1] == 1
    true_src_nodes = np.unique(candidate_sa[0])

    probabilities = model.predict_proba(X_test)

    for idx in range(true_src_nodes.shape[0]):
      true_s, true_t, label = candidate_sa[:, idx]
      
      final_y_test.append(label)

      # Extract the index of the possible individuals connected to the src individual        
      possible_links_individuals = candidate_sa[1, candidate_sa[0,:] == true_s]
      
      #extract the probabilities of this possible links     
      possible_links = probabilities[candidate_sa[0,:] == true_s, 1]
      
      #possible_gt = y_test[candidate_sa[0,:] == true_s]
      decission_link = np.argmax(possible_links)


      decission_prob = possible_links[decission_link]
      decission_individual = possible_links_individuals[decission_link]
      
      
      discovered_link = (int(decission_prob > 0.5))
      
      final_y_pred.append(discovered_link)
      
      if discovered_link == 1:
        correct_individual = int(true_t == decission_individual)
        specificity.append(correct_individual) 


    print("MEAN Probability: ", np.mean(model.predict_proba(X_test)))
    print("STD Probability: ", np.std(model.predict_proba(X_test)))
    
    
    # Evaluate the model
    accuracy = accuracy_score(final_y_test, final_y_pred)
    conf_matrix = confusion_matrix(final_y_test, final_y_pred)
    precision = precision_score(final_y_test, final_y_pred)
    recall = recall_score(final_y_test, final_y_pred)
    f_score = (2*precision*recall) / (precision + recall)

    # Print the evaluation results
    print("Confusion Matrix:\n", conf_matrix)

    print("Accuracy:", accuracy)
    print("Recall: ", recall)
    print("Precission: ", precision)
    print("F-score: ", f_score)
    
    print("Specificity: ", np.sum(specificity)/len(specificity))

    return dict(
      Accuracy= accuracy,
      Conf_Matrix = conf_matrix.tolist(),
      Precssion = precision,
      Recall = recall,
      F_score = f_score,
      Specificity = np.sum(specificity)/len(specificity)   
    )
