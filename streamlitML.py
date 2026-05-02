import streamlit as st
import joblib 
import pandas as pd 

model = joblib.load('fraud_detection.pkl')

st.title('The Fraud Detection App')
st.markdown("Let's Detect the fraud by Prediction through entering some details!")
st.divider()

trans_type = st.selectbox('select transaction_type:',['CASH_OUT','PAYMENT',"TRANSFER","DEBIT"])
amount= st.number_input('enter the Amount', min_value=0.0, value = 10000.0)
oldbalanceOrg=st.number_input('enter the oldbalance(Sender)', min_value=0.0, value = 10000.0)
newbalanceDest= st.number_input('enter the newbalance (Receiver)', min_value=0.0, value = 0.0)
newbalanceOrig= st.number_input('enter the orignal value(Sender)', min_value=0.0, value = 10000.0)
oldbalanceDest= st.number_input('enter the amount(Receiver)', min_value=0.0, value = 0.0)

if st.button("Predict"):
    data = pd.DataFrame([{
        'type': trans_type,
        'Amount' : amount,
        'oldbalanceSender': oldbalanceOrg, 
        'oldbalanceReceiver':oldbalanceDest, 
        'newbalanceSender':newbalanceOrig,
        'newbalanceReceiver': newbalanceDest
    }])
    prediction = model.predict(data)[0]
    st.subheader(f"prediction: {int(prediction)}")
    if prediction==1:
        st.error("The system seems to be Fraud")
    else:
        st.success('The system is not Fraud ')