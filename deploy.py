import streamlit as st
import pickle
import numpy as np
import pandas as pd

model =  pickle.load(open('rfmodel.pkl','rb'))

df = pd.read_csv(r'cleaned_data.csv')

items_to_remove = ['ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6','PAY_1'
                   ,'EDUCATION_CAT', 'graduate school', 'high school', 
                   'others', 'university','default payment next month']

x = df.drop(items_to_remove,axis=1)

k = np.array(x.columns)

def prediction(values):
    input = np.array([values]).astype(np.float64)
    pred = model.predict(input)
    return pred

def main():
    
    page_bg_img = '''
    <style>
    body {
    background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTSQO7akXaqxBfaCy7dX2Dmx9jboCJVwVIX0Q&usqp=CAU");
    background-size: cover;
    } 
    </style>
    '''
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    
    html_temp="""
    
    <div style="background-image:url('https://wpamelia.com/wp-content/uploads/2019/02/astronomy-constellation-dark-998641.jpg');height=200px; width:80vw;margin-bottom:10px">
    <h2 style="text-align:center;font-family:verdana; size:100px; color:white;">ACCOUNT DEFAULT PREDICTION(NEXT MONTH)</h2>
    </div>
    <div>
    <p style="color:black;size:70px;font-family:aerial">FOR STATUS 1 ACCCOUNT WILL DEFAULT AND FOR 0 ACCOUNT WON'T DEFAULT NEXT MONTH</p>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    l=[]
    for i in range(0,16):
        l.append(st.text_input(k[i]))
        
    nd_html="""
    <div style="background-color:red;padding:12px;">
    <h3 style="color:white;text-align:center">Account will not default payment</h3>
    </div>
    """
    
    d_html="""
    <div style="background-color:blue;padding:12px;">
    <h3 style="color:white;text-align:center">Account will default payment next month</h3>
    </div>
    """
    
    if st.button("Check"):
        count =0
        for j in range(0,16):
            if not l[j]:
                count = 1
        if count == 0:
            output = prediction(l)
            st.success("The status is :{}".format(output))
            if output==0:
                 st.markdown(nd_html,unsafe_allow_html=True)
                 st.warning('oh no!! the account will not default the payment next month')
            else:
                 st.markdown(d_html, unsafe_allow_html=True)
                 st.balloons()
        else:
            st.error('all fields are not filled kindly fill all the data for prediction')
if __name__=='__main__':
    main()
