import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import GenericUnivariateSelect, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tkinter import *
from tkinter import ttk,IntVar
from tkinter import Entry,Label,Tk,Button,messagebox ,Checkbutton
from sklearn.neighbors import KNeighborsClassifier
dataset = pd.read_csv("E:\Files\Projects\Service_Cancellation_Predictor\CustomersDataset.csv")
print(dataset.head())
print(dataset.info())
dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'], errors='coerce')
dataset['TotalCharges'] = dataset['TotalCharges'].astype("float")
print(dataset.isna().any())
dataset.dropna(inplace=True)
for x in dataset.index :
    if dataset.loc[x,"TotalCharges"] <=0 :
        dataset.drop(x,inplace = True)
for x in dataset.index :
    if dataset.loc[x,"MonthlyCharges"] <=0 :
        dataset.drop(x,inplace = True)
print (dataset.info())
print(dataset.duplicated())
le = LabelEncoder()
lepartner = LabelEncoder()
ledependent = LabelEncoder()
lemultiplines = LabelEncoder()
leinternetservice = LabelEncoder()
leonlinesecurity = LabelEncoder()
leonlinebackup = LabelEncoder()
ledeviceprotection = LabelEncoder()
letechsupport = LabelEncoder()
lecontract = LabelEncoder()
lepaper = LabelEncoder()
lepayment= LabelEncoder()
label = le.fit_transform(dataset['gender'])
label1 = lepartner.fit_transform(dataset['Partner'])
label2 = ledependent.fit_transform(dataset['Dependents'])
label3 = le.fit_transform(dataset['PhoneService'])
label4 = lemultiplines.fit_transform(dataset['MultipleLines'])
label5 = leinternetservice.fit_transform(dataset['InternetService'])
label6 = leonlinesecurity.fit_transform(dataset['OnlineSecurity'])
label7 = leonlinebackup.fit_transform(dataset['OnlineBackup'])
label8 = ledeviceprotection.fit_transform(dataset['DeviceProtection'])
label9 = letechsupport.fit_transform(dataset['TechSupport'])
label10 = le.fit_transform(dataset['StreamingTV'])
label11 = le.fit_transform(dataset['StreamingMovies'])
label12 = lecontract.fit_transform(dataset['Contract'])
label13 = lepaper.fit_transform(dataset['PaperlessBilling'])
label14 = lepayment.fit_transform(dataset['PaymentMethod'])
label15= le.fit_transform(dataset['Churn'])
print (label1)
cols = [col for col in dataset.columns if col not in 'MonthlyCharges' +'TotalCharges'+'tenure'+'SeniorCitizen']
dataset.drop(cols,axis = 1 , inplace=True)
dataset["gender"] = label
dataset["Partner"] = label1
dataset["Dependent"] = label2
dataset["PhoneService"] = label3
dataset["MultipleLines"] = label4
dataset["InternetService"] = label5
dataset["OnlineSecurity"] = label6
dataset["OnlineBackup"] = label7
dataset["DeviceProtection"] = label8
dataset["TechSupport"] = label9
dataset["StreamingTV"] = label10
dataset["StreamingMovies"] = label11
dataset["contract" ] = label12
dataset["PaperlessBilling"] = label13
dataset["PaymentMethod"] = label14
dataset["Churn"] = label15
print (dataset.info())
cols = [col for col in dataset.columns if col not in 'Churn']
featureselection1=GenericUnivariateSelect(score_func=chi2,mode='k_best',param=15)
colnew= featureselection1.fit_transform(dataset[cols],dataset[['Churn']])
print (featureselection1.get_support())
cols = [col for col in dataset.columns if col in 'gender'+'PhoneService'+'StreamingTV'+'StreamingMovies']
dataset.drop(cols,axis = 1 , inplace=True)
print (dataset.info())
print (dataset.head())
cols = [col for col in dataset.columns if col not in 'Churn']
dataset_data = dataset[cols]
dataset_target= dataset['Churn']
print (dataset_data.shape)
print (dataset_target.shape)
datatrain,datatest,targettrain,targettest=train_test_split(dataset_data, dataset_target ,test_size = 0.2,random_state=0)
model1= LogisticRegression(solver='liblinear' ,max_iter=1000)
model2= SVC(kernel = 'linear',C=10,max_iter=90000000)
kn= KNeighborsClassifier(n_neighbors =10)
model4=DecisionTreeClassifier (criterion='entropy',max_depth=7)
app = Tk()
app.geometry("800x400")
app.title("Service cancellation predictor")
label1 = Label(app, text="Mehtodology").grid(row=0, column=1)
var1 = IntVar()
var2 = IntVar()
var3 = IntVar()
var4 = IntVar()
def train () :
    if var1.get() >= 1:
       model1.fit(datatrain,targettrain)
       model1_train_predict =model1.predict(datatrain)
       messagebox.showinfo("logestic regression","train Accuracy of logestic regression = " + str(accuracy_score(model1_train_predict,targettrain)))
    if var2.get() >= 1:
       model2.fit(datatrain,targettrain)
       model2_train_predict=model2.predict(datatrain)
       messagebox.showinfo("Svm","train Accuracy of  svm = " + str(accuracy_score(model2_train_predict,targettrain)))
    if var3.get() >= 1:
        model4.fit(datatrain,targettrain)
        model4_train_predict=model4.predict(datatrain)
        messagebox.showinfo("Decisiontree","train Accuracy of  Decision tree = " + str(accuracy_score(model4_train_predict,targettrain)))
    if var4.get() >= 1:
         kn.fit(datatrain,targettrain)
         kn_train_predict=kn.predict(datatrain)
         messagebox.showinfo("KNeighborsClassifier","train Accuracy of KNeighborsClassifier = " + str(accuracy_score(kn_train_predict,targettrain)))

def test() :
   try:
        if var1.get() >= 1:
           model1_test_predict =model1.predict(datatest)
           messagebox.showinfo("logestic regression","test Accuracy of logestic regression = " + str(accuracy_score(model1_test_predict,targettest)))
        if var2.get() >= 1:
           model2_test_predict =model2.predict(datatest)
           messagebox.showinfo("SVM","test Accuracy of SVM = " + str(accuracy_score(model2_test_predict,targettest)))
        if var3.get() >= 1:
           model4_test_predict =model4.predict(datatest)
           messagebox.showinfo("Decision Tree","test Accuracy of Decision Tree = " + str(accuracy_score(model4_test_predict,targettest)))
        if var4.get() >= 1:
          kn_test_predict =kn.predict(datatest)
          messagebox.showinfo("KNeighborsClassifier","test Accuracy of KNeighborsClassifier= " + str(accuracy_score(kn_test_predict,targettest)))
   except:
       messagebox.showerror("Test error" ,"You can't test before train")
logisticRegression_button = Checkbutton(app, text="Logistic Regression", variable=var1).grid(row=1, column=0)
SVM_Button = Checkbutton(app, text="SVM", variable=var2).grid(row=1, column=1)
ID3_Button = Checkbutton(app, text="ID3", variable=var3).grid(row=1, column=2)
KNNbutton = Checkbutton(app, text="KNeighbors", variable=var4).grid(row=1, column=3)
trainButton = Button(app, text="Train", padx=50,command=train).grid(row=2, column=0)
testButton = Button(app, text="Test", padx=50,command = test).grid(row=2, column=1)
label2 = Label(app, text="Customer Data").grid(row=3, column=0)
customerIdEntry = Entry(app, width=20)
customerIdEntry.grid(row=4, column=1)
customerIdLabel = Label(app, text="Customer Id").grid(row=4, column=0)
partnerLabel = Label(app, text="Partner").grid(row=5, column=0)
partnerEntry = Entry(app, width=20)
partnerEntry.grid(row=5, column=1)
phoneServiceLabel = Label(app, text="Phone Service").grid(row=6, column=0)
phoneServiceEntry = Entry(app, width=20)
phoneServiceEntry.grid(row=6, column=1)
onlineSecurityLabel = Label(app, text="Online Security").grid(row=7, column=0)
onlineSecurityEntry = Entry(app, width=20)
onlineSecurityEntry.grid(row=7, column=1)
techSupportLabel = Label(app, text="Tech Support").grid(row=8, column=0)
techSupportEntry = Entry(app, width=20)
techSupportEntry.grid(row=8, column=1)
contractLabel = Label(app, text="Contract").grid(row=9, column=0)
contractEntry = Entry(app, width=20)
contractEntry.grid(row=9, column=1)
monthlyChargesLabel = Label(app, text="Monthly Charges").grid(row=10, column=0)
monthlyChargesEntry = Entry(app, width=20)
monthlyChargesEntry.grid(row=10, column=1)
genderLabel = Label(app, text="Gender").grid(row=4, column=2)
genderEntry = Entry(app, width=20)
genderEntry.grid(row=4, column=3)
dependentsLabel = Label(app, text="Dependents").grid(row=5, column=2)
dependentsEntry = Entry(app, width=20)
dependentsEntry.grid(row=5, column=3)
multipleLinesLabel = Label(app, text="Multiple Lines").grid(row=6, column=2)
multipleLinesEntry = Entry(app, width=20)
multipleLinesEntry.grid(row=6, column=3)
onlineBackupLabel = Label(app, text="Online Backup").grid(row=7, column=2)
onlineBackupEntry = Entry(app, width=20)
onlineBackupEntry.grid(row=7, column=3)
streamingTVLabel = Label(app, text="Streaming TV").grid(row=8, column=2)
streamingTVEntry = Entry(app, width=20)
streamingTVEntry.grid(row=8, column=3)
paperlessBillingLabel = Label(app, text="Paperless Billing").grid(row=9, column=2)
paperlessBillingEntry = Entry(app, width=20)
paperlessBillingEntry.grid(row=9, column=3)
totalChargesLabel = Label(app, text="Total Charges").grid(row=10, column=2)
totalChargesEntry = Entry(app, width=20)
totalChargesEntry.grid(row=10, column=3)
seniorCitizenLabel = Label(app, text="Senior Citizen").grid(row=4, column=4)
seniorCitizenEntry = Entry(app, width=20)
seniorCitizenEntry.grid(row=4, column=5)
tenureLabel = Label(app, text="Tenure").grid(row=5, column=4)
tenureEntry = Entry(app, width=20)
tenureEntry.grid(row=5, column=5)
internetServiceLabel = Label(app, text="Internet Service").grid(row=6, column=4)
internetServiceEntry = Entry(app, width=20)
internetServiceEntry.grid(row=6, column=5)
deviceProtectionLabel = Label(app, text="Device Protection").grid(row=7, column=4)
deviceProtectionEntry = Entry(app, width=20)
deviceProtectionEntry.grid(row=7, column=5)
streamingMoviesLabel = Label(app, text="Streaming Movies").grid(row=8, column=4)
streamingMoviesEntry = Entry(app, width=20)
streamingMoviesEntry.grid(row=8, column=5)
paymentMethodLabel = Label(app, text="Payment Method").grid(row=9, column=4)
paymentMethodEntry = Entry(app, width=20)
paymentMethodEntry.grid(row=9, column=5)
arr=[]
def predictation() :
    global arr
    text1 = seniorCitizenEntry.get()
    text2 = tenureEntry.get()
    text3 = monthlyChargesEntry.get()
    text4 = totalChargesEntry.get()
    text5 = partnerEntry.get()
    text6 = dependentsEntry.get()
    text7 = multipleLinesEntry.get()
    text8 = internetServiceEntry.get()
    text9 = onlineSecurityEntry.get()
    text10= onlineBackupEntry.get()
    text11= deviceProtectionEntry.get()
    text12= techSupportEntry.get()
    text13= contractEntry.get()
    text14= paperlessBillingEntry.get()
    text15=paymentMethodEntry.get()
    arr=[]
    arr.append(text1)
    arr.append(text2)
    arr.append(text3)
    arr.append(text4)
    arr.append(lepartner.transform([text5]))
    arr.append(ledependent.transform([text6]))
    arr.append(lemultiplines.transform([text7]))
    arr.append(leinternetservice.transform([text8]))
    arr.append(leonlinesecurity.transform([text9]))
    arr.append(leonlinebackup.transform([text10]))
    arr.append(ledeviceprotection.transform([text11]))
    arr.append(letechsupport.transform([text12]))
    arr.append(lecontract.transform([text13]))
    arr.append(lepaper.transform([text14]))
    arr.append(lepayment.transform([text15]))
    try:
        predicate = model1.predict([arr])
        if predicate==[0]:
            messagebox.showinfo("result","the user may cancel the service")
        else:
            messagebox.showinfo("result","the user may not cancel the service")
    except:
        messagebox.showerror("Predict error" ,"You can't predict before train")
predictButton = Button(app, text="Predict", padx=50, command=predictation, bg="#EB5353").grid(row=11, column=2)
app.mainloop()