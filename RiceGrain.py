import streamlit as st
import pickle
import sklearn
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
# Load the model using joblib

# Now try predicting
st.title('Rice Grain Quality Detection')
st.sidebar.header('Provide Input')
model = st.sidebar.selectbox('Select Methodology',["Logistic Regression","Decision Tree","K Nearest Neighbor","Hybrid"])
def user_report():
    length = st.sidebar.slider('Length', 0.0,1.0,0.5 )
    width = st.sidebar.slider("Width",0.0,1.0,0.4)    
    color = st.sidebar.radio('Color', ['Brown','White'])
    if(color=="White"):
        color_ww=1.0
        color_br = 0.0
    elif(color=="Brown"):
        color_ww=0.0
        color_br = 1.0
    Texture = st.sidebar.radio('Texture', ['Smooth','Rough','Medium'])    
    if(Texture=="Smooth"):
        Texture_s=1.0
        Texture_r =0.0
        Texture_m = 0.0
    elif(Texture=="Rough"):
        Texture_r=1.0
        Texture_s=0.0
        Texture_m=0.0
    else:
        Texture_m=1.0
        Texture_s=0.0
        Texture_r=0.0
    Weight = st.sidebar.slider("Weight",0.0,1.0)
    user_report_data = {
        'Length' : length,
        'Width' : width,        
        'Brown':color_br,
        'White':color_ww,
        'Smooth':Texture_s,
        'Rough':Texture_r,
        'Medium':Texture_m,
        'Weight':Weight
    }
    user_data1 = [length,width,Weight,color_br,color_ww,Texture_m,Texture_r,Texture_s]
    report = pd.DataFrame(user_report_data,index=[0])
    st.write(report)
    return user_data1
user_data=user_report()
original_title = '<p style="font-family:Courier; color:White; font-size: 25px;font-weight: bold;">Quality of Rice as per selected criteria</p>'
st.markdown(original_title, unsafe_allow_html=True)

oe = OrdinalEncoder()
le = LabelEncoder()
data = pd.read_csv("dataset.csv")

x1,x2,y1,y2=train_test_split(data.iloc[:  , 1:9] , data.iloc[: , 9:].astype('int') , train_size=0.8)

if (model == "Logistic Regression"):
    lr = LogisticRegression()
    lr.fit(x1,y1)
    probability = lr.predict([user_data])
    
elif (model == "Decision Tree"):   
    lr = DecisionTreeClassifier(criterion='entropy')
    lr.fit(x1,y1)
    probability = lr.predict([user_data])
    
elif (model == "K Nearest Neighbor"):    
    lr = KNeighborsClassifier(n_neighbors=5)
    lr.fit(x1,y1)
    probability = lr.predict([user_data])
    
else:
    from sklearn.ensemble import VotingClassifier
    model1 = LogisticRegression(random_state=1)
    model2 = DecisionTreeClassifier(random_state=1)
    model3 = KNeighborsClassifier()
    lr= VotingClassifier(estimators=[('lr',model1),('dtc',model2),('knc',model3)],voting='hard')
    lr.fit(x1,y1)
    probability = lr.predict([user_data])
    
if (probability[0] == 2):
    original_title = '<p style="color:#00f900; font-size: 25px;font-weight: bold;">FAIR</p>'
    st.markdown(original_title, unsafe_allow_html=True)
    if(user_data[4]==1.0):
        st.image("white_smooth.jpg",width=300)
    else:
        st.image("brown_smooth.jpg",width=300)
        
elif(probability[0] == 1):
    original_title = '<p style="color:#00f900; font-size: 25px;font-weight: bold;">GOOD</p>'
    st.markdown(original_title, unsafe_allow_html=True)
    if(user_data[4]==1.0):
        st.image("white_rough.jpg",width=300)
    else:
        st.image("brown_rough.jpg",width=300)
        
elif(probability[0] == 0):
    original_title = '<p style="color:#00f900; font-size: 25px;font-weight: bold;">EXCELLENT</p>'
    st.markdown(original_title, unsafe_allow_html=True)
    if(user_data[4]==1.0):
        st.image("white_medium.jpg",width=300)
    else:
        st.image("brown_medium.jpg",width=300)
        
else:
    original_title = '<p style="color:#00f900; font-size: 25px;font-weight: bold;">POOR</p>'
    st.markdown(original_title, unsafe_allow_html=True)
    if(user_data[4]==1.0):
        st.image("white_poor.jpeg",width=300)
    else:
        st.image("brown_poor.jpeg",width=300)
