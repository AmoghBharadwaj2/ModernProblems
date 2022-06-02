from cgitb import text
import gpt_2_simple as gpt2
import tensorflow as tf
def produce(runname, input):
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess,run_name=runname,reuse=tf.compat.v1.AUTO_REUSE)
    text = gpt2.generate(sess, run_name=runname, prefix= input, return_as_list=True, length=70)[0]
    return text

def produce2(runname, input, s):
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess,run_name=runname,reuse=tf.compat.v1.AUTO_REUSE)
    temp = s + 70
    text = gpt2.generate(sess, run_name=runname, prefix= input, return_as_list=True, length= temp)[0]
    return text

def generate(question, nums):
    starter = question
    size = len(question)
    #num = input("Enter the number of conversations you want: ")
    num = nums
    i = 0
    #save result of models
    result = []
    text1 = produce("run1", starter)
    text1 =  " ".join(text1.split())
    index = text1.rfind(",")
    text1 = text1[:index]
    text1 = text1 + "."
    size1 = len(text1)
    
    print("Religion: " + text1 + "\n")
    result.append("Religion: " + text1)

    text3 = produce("run3", starter)
    text3 =  " ".join(text3.split())
    index = text3.rfind(",")
    text3 = text3[:index]
    text3 = text3 + "."
    size3 = len(text3)
    print("Science: " + text3 + "\n")
    result.append("Science: " + text3)

    #generate philosophy response
    text2 = produce("run2", starter)
    starter =  " ".join(text2.split())
    index = text2.rfind(",")
    text2 = text2[:index]
    text2 = text2 + "."
    size2 = len(text2)
    print("Philosophy: " + text2 + "\n")
    result.append("Philosophy: " + text2)
    n1 = size1
    n2 = size2
    n3 = size3  
    
    defaultsentence = "I don't know"

    while i < int(num):    
        #generate science response 
        text3new = produce2("run3", text1, n1)
        text3new =  " ".join(text3new.split())
        text3new = text3new[size1:]
        while text3new[0].isalpha() == False:
            text3new = text3new[1:]
        index = text3new.rfind(",")
        text3new = text3new[:index]
        text3new = text3new + "."
        print("Science: Hi Religion, " + text3new + "\n")
        result.append("Science: Hi Religion, " + text3new)
     
        #generate religion response
        text1new = produce2("run1", text2, n2)
        text1new =  " ".join(text1new.split())
        text1new = text1new[size2:]
        while text1new[0].isalpha() == False:
            text1new = text1new[1:]
        index = text1new.rfind(",")
        text1new = text1new[:index]
        text1new = text1new + "."
        print("Religion: Hi Philosophy, " + text1new + "\n")
        result.append("Religion: Hi Philosophy, " + text1new)      

        #generate philosophy response
        text2new = produce2("run1", text3, n3)
        text2new =  " ".join(text2new.split())
        text2new = text2new[size3:]
        while text2new[0].isalpha() == False:
            text2new = text2new[1:]
        index = text2new.rfind(",")
        text2new = text2new[:index]
        text2new = text2new + "."
        print("Philosophy: Hi Science, " + text2new + "\n")
        result.append("Philosophy: Hi Science, " + text2new) 
        
        n1 = len(text1new)
        n2 = len(text2new)
        n3 = len(text3new) 
        i += 1
    return result

#generate("Love is", 1)