import pandas as pd
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

class text_process:
    """
    Class for preprocessing text.
    
    """
    def __init__(self,text):
        """
        Args:
            text : a pd column of 'documents' to be vectorized 
            
            """
        self.text = text
        
    
    def pd_col_to_list(self):
        """
        
            Converts pandas column to a LIST of comma-separated text ('documents') before it is to be vectorized.
            For example, index 0 of the list = the first question (or answer), and so on.
        
        """
        
        doc_list = []
        for i in self.text:
            doc_list.append(i)
            
        return doc_list
        
        
    def remove_stopwords(self, doc_list):    
        """
        Args:
            doc_list: output from pd_col_to_list, a list of comma separated text ('documents')
        
            Removes stop_words, according to the stop words nltk corpus, before the list is to be vectorized.
            
        """
        
        stop_words = stopwords.words('english')
        for j in range(len(doc_list)):
            doc_list[j] = text_to_word_sequence(doc_list[j])
            for i in stop_words:
                count=0
                while count==0:
                    try:
                        doc_list[j].remove(i)
                    except:
                        count+=1
                    finally:
                        pass
        
        return doc_list
        

    def text_to_vec(self, doc_list):
        """
        Args:
            doc_list: output from remove_stopwords(), a list of comma separated text ('documents') where stop words
                       have been removed
        
        
            Converts list of comma-separated 'documents' to a vector, using tf-idf. This is done after stop words 
            have been removed (with remove_stopwords()). Utilizes the keras tokenizer, which separates on spaces, 
            makes all tokens lowercase and removes punctuation. Function then stems the tokens, using the 
            PorterStemmer from nltk. Finally, it returns the preprocessed tokens in vector form.
            
        """

        
        ## Keras tokenizer
        
        tok = Tokenizer()
        tok.fit_on_texts(doc_list)
        
        tokens = list(tok.word_counts)
        
        ## Stem words 
        
        ps = PorterStemmer()
        for i in range(len(tokens)): #i = question index (i.e. question1, question2, etc...)
            tokens[i] = (ps.stem(tokens[i]))
        
        ## Return tf-idf matrix
        
        matrx = tok.texts_to_matrix(tokens, mode='tfidf')
        
        return matrx, tok.get_config()
    
    
    def text_preprocess(self):

        list_form = text_process.pd_col_to_list(self)
        no_stops = text_process.remove_stopwords(self,list_form)
        final_vec = text_process.text_to_vec(self,no_stops)
        
        return final_vec
        
        
        
