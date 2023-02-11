#!/usr/bin/env python
# coding: utf-8

# In[296]:


import pandas as pd
import numpy as np
import matplotlib as plt
import scipy.stats as stats
import array as ar
import seaborn as sns


# In[20]:


#Dataframe yüklenmesi ve ilk beş satırının gösterilmesi
verilerKalpKriziRiskiTenYears=pd.read_csv('train.csv')
verilerKalpKriziRiskiTenYears.head(5)


# In[21]:


#verilerin son 5 satırı
verilerKalpKriziRiskiTenYears.tail(5)


# In[22]:


verilerKalpKriziRiskiTenYears.describe()


# In[23]:


verilerKalpKriziRiskiTenYears.info


# In[24]:


#Eksik veri kontrolü
verilerKalpKriziRiskiTenYears.isnull()


# In[25]:


verilerKalpKriziRiskiTenYears.isnull().sum()


# In[26]:


#Sutunlardaki eksik verileri ortalama ile doldurmak için her sutun ortalaması hesapla
#age sutun ortalaması
ageOrtalama=verilerKalpKriziRiskiTenYears['age'].mean()
#education sutun ortalaması
educationOrtalama=verilerKalpKriziRiskiTenYears['education'].mean()
#cigsPerDay  sutun ortalaması
cigsPerDayOrtalama=verilerKalpKriziRiskiTenYears['cigsPerDay'].mean()
#BMI sutun ortalaması
BMIOrtalama=verilerKalpKriziRiskiTenYears['BMI'].mean()
#totChol sutun ortalaması
totCholOrtalama=verilerKalpKriziRiskiTenYears['totChol'].mean()
#BPMeds  sutun ortalaması
BPMedsOrtalama=verilerKalpKriziRiskiTenYears['BPMeds'].mean()
#glucose sutun ortalaması,
glucoseOrtalama=verilerKalpKriziRiskiTenYears['glucose'].mean()
#heartRate sutun ortalaması
heartRateOrtalama=verilerKalpKriziRiskiTenYears['heartRate'].mean()
#prevalentStroke sutun ortalaması
prevalentStrokeOrtalama=verilerKalpKriziRiskiTenYears['prevalentStroke'].mean()
#prevalentHyp sutun ortalaması
prevalentHypOrtalama=verilerKalpKriziRiskiTenYears['prevalentHyp'].mean()
#diabetes sutun ortalaması
diabetesOrtalama=verilerKalpKriziRiskiTenYears['diabetes'].mean()
#sysBP sutun ortalaması
sysBPOrtalama=verilerKalpKriziRiskiTenYears['sysBP'].mean()
#diaBP sutun ortalaması
diaBPOrtalama=verilerKalpKriziRiskiTenYears['diaBP'].mean()




# In[27]:


#şimdi fillna fonksiyonu kullanarak ekssik verileri tamamlayalım.
verilerKalpKriziRiskiTenYears['education'].fillna(educationOrtalama, inplace=True)
verilerKalpKriziRiskiTenYears['cigsPerDay'].fillna(cigsPerDayOrtalama, inplace=True)
verilerKalpKriziRiskiTenYears['BMI'].fillna(BMIOrtalama, inplace=True)
verilerKalpKriziRiskiTenYears['totChol'].fillna(totCholOrtalama, inplace=True)
verilerKalpKriziRiskiTenYears['BPMeds'].fillna(BPMedsOrtalama, inplace=True)
verilerKalpKriziRiskiTenYears['glucose'].fillna(glucoseOrtalama, inplace=True)
verilerKalpKriziRiskiTenYears['heartRate'].fillna(heartRateOrtalama, inplace=True)


# In[28]:


#aykırı veri analizi , aykırı veri var mı yok mu 
print('EDUCATİON GRAFİĞİ ve AYKIRI VERİ LİSTESİ')
verilerKalpKriziRiskiTenYears['education'].plot(kind='box')
#görselde ki üst sınır 4.0e yakın olan çizgi min çizgi altta 1.0 a yakın olan çizgi medyan ise 2.0
#çizgilerin dışında değer görünmüyor yani education sutununda aykırı değer yok
Q1=verilerKalpKriziRiskiTenYears.education.quantile(0.25)
Q3=verilerKalpKriziRiskiTenYears.education.quantile(0.75)
IQR=Q3-Q1
#print(Q1,Q3,IQR)
print('Bu değer üstü aykırı veridir: ',Q3+1.5*IQR)
print('Bu değer altı aykırı veridir: ',Q1-1.5*IQR)
ust_sinir_edu=Q3+1.5*IQR
alt_sinir_edu=Q1-1.5*IQR

verilerKalpKriziRiskiTenYears[(verilerKalpKriziRiskiTenYears['education']> ust_sinir_edu) | (verilerKalpKriziRiskiTenYears['education']< alt_sinir_edu)]


# In[29]:



print('CİGSPERDAY GRAFİĞİ ve AYKIRI VERİ LİSTESİ')
verilerKalpKriziRiskiTenYears['cigsPerDay'].plot(kind='box')
Q1=verilerKalpKriziRiskiTenYears.cigsPerDay.quantile(0.25)
Q3=verilerKalpKriziRiskiTenYears.cigsPerDay.quantile(0.75)
IQR=Q3-Q1
#print(Q1,Q3,IQR)
print('Bu değer üstü aykırı veridir: ',Q3+1.5*IQR)
print('Bu değer altı aykırı veridir: ',Q1-1.5*IQR)
ust_sinir_cigsPerDay=Q3+1.5*IQR
alt_sinir_cigsPerDay=Q1-1.5*IQR
verilerKalpKriziRiskiTenYears[(verilerKalpKriziRiskiTenYears['cigsPerDay']> ust_sinir_cigsPerDay) | (verilerKalpKriziRiskiTenYears['cigsPerDay']< alt_sinir_cigsPerDay)]
#bütün satırı getirdi


# In[30]:


print('PREVALENTSTROKE GRAFİĞİ ve AYKIRI VERİ LİSTESİ')
verilerKalpKriziRiskiTenYears['prevalentStroke'].plot(kind='box')
Q1=verilerKalpKriziRiskiTenYears.prevalentStroke.quantile(0.25)
Q3=verilerKalpKriziRiskiTenYears.prevalentStroke.quantile(0.75)
IQR=Q3-Q1
print('Bu değer üstü aykırı veridir: ',Q3+1.5*IQR)
print('Bu değer altı aykırı veridir: ',Q1-1.5*IQR)
ust_sinir_prevalentStroke=Q3+1.5*IQR
alt_sinir_prevalentStroke=Q1-1.5*IQR
verilerKalpKriziRiskiTenYears[(verilerKalpKriziRiskiTenYears['prevalentStroke']> ust_sinir_prevalentStroke) | (verilerKalpKriziRiskiTenYears['prevalentStroke']< alt_sinir_prevalentStroke)]
#bütün satırı getirdi
#bütün satırı getirdi


# In[31]:


print('BPMeds GRAFİĞİ ve AYKIRI VERİ LİSTESİ')
verilerKalpKriziRiskiTenYears['BPMeds'].plot(kind='box')
Q1=verilerKalpKriziRiskiTenYears.BPMeds.quantile(0.25)
Q3=verilerKalpKriziRiskiTenYears.BPMeds.quantile(0.75)
IQR=Q3-Q1
print('Bu değer üstü aykırı veridir: ',Q3+1.5*IQR)
print('Bu değer altı aykırı veridir: ',Q1-1.5*IQR)
ust_sinir_BPMeds=Q3+1.5*IQR
alt_sinir_BPMeds=Q1-1.5*IQR
verilerKalpKriziRiskiTenYears[(verilerKalpKriziRiskiTenYears['BPMeds']> ust_sinir_BPMeds) | (verilerKalpKriziRiskiTenYears['BPMeds']< alt_sinir_BPMeds)]
#bütün satırı getirdi
#bütün satırı getirdi


# In[32]:


print('totChol GRAFİĞİ ve AYKIRI VERİ LİSTESİ')
verilerKalpKriziRiskiTenYears['totChol'].plot(kind='box')
Q1=verilerKalpKriziRiskiTenYears.totChol.quantile(0.25)
Q3=verilerKalpKriziRiskiTenYears.totChol.quantile(0.75)
IQR=Q3-Q1
print('Bu değer üstü aykırı veridir: ',Q3+1.5*IQR)
print('Bu değer altı aykırı veridir: ',Q1-1.5*IQR)
ust_sinir_totChol=Q3+1.5*IQR
alt_sinir_totChol=Q1-1.5*IQR
verilerKalpKriziRiskiTenYears[(verilerKalpKriziRiskiTenYears['totChol']> ust_sinir_totChol) | (verilerKalpKriziRiskiTenYears['totChol']< alt_sinir_totChol)]
#bütün satırı getirdi
#bütün satırı getirdi


# In[33]:


print('diabetes GRAFİĞİ ve AYKIRI VERİ LİSTESİ')
verilerKalpKriziRiskiTenYears['diabetes'].plot(kind='box')
Q1=verilerKalpKriziRiskiTenYears.diabetes.quantile(0.25)
Q3=verilerKalpKriziRiskiTenYears.diabetes.quantile(0.75)
IQR=Q3-Q1
print('Bu değer üstü aykırı veridir: ',Q3+1.5*IQR)
print('Bu değer altı aykırı veridir: ',Q1-1.5*IQR)
ust_sinir_diabetes=Q3+1.5*IQR
alt_sinir_diabetes=Q1-1.5*IQR
verilerKalpKriziRiskiTenYears[(verilerKalpKriziRiskiTenYears['diabetes']> ust_sinir_diabetes) | (verilerKalpKriziRiskiTenYears['diabetes']< alt_sinir_diabetes)]
#bütün satırı getirdi
#bütün satırı getirdi


# In[34]:


print('prevalentHyp GRAFİĞİ ve AYKIRI VERİ LİSTESİ')
verilerKalpKriziRiskiTenYears['prevalentHyp'].plot(kind='box')
Q1=verilerKalpKriziRiskiTenYears.prevalentHyp.quantile(0.25)
Q3=verilerKalpKriziRiskiTenYears.prevalentHyp.quantile(0.75)
IQR=Q3-Q1
print('Bu değer üstü aykırı veridir: ',Q3+1.5*IQR)
print('Bu değer altı aykırı veridir: ',Q1-1.5*IQR)
ust_sinir_prevalentHyp=Q3+1.5*IQR
alt_sinir_prevalentHyp=Q1-1.5*IQR
verilerKalpKriziRiskiTenYears[(verilerKalpKriziRiskiTenYears['prevalentHyp']> ust_sinir_prevalentHyp) | (verilerKalpKriziRiskiTenYears['prevalentHyp']< alt_sinir_prevalentHyp)]
#bütün satırı getirdi
#bütün satırı getirdi


# In[35]:


print('sysBP GRAFİĞİ ve AYKIRI VERİ LİSTESİ')
verilerKalpKriziRiskiTenYears['sysBP'].plot(kind='box')
Q1=verilerKalpKriziRiskiTenYears.sysBP.quantile(0.25)
Q3=verilerKalpKriziRiskiTenYears.sysBP.quantile(0.75)
IQR=Q3-Q1
print('Bu değer üstü aykırı veridir: ',Q3+1.5*IQR)
print('Bu değer altı aykırı veridir: ',Q1-1.5*IQR)
ust_sinir_sysBP=Q3+1.5*IQR
alt_sinir_sysBP=Q1-1.5*IQR
verilerKalpKriziRiskiTenYears[(verilerKalpKriziRiskiTenYears['sysBP']> ust_sinir_sysBP) | (verilerKalpKriziRiskiTenYears['sysBP']< alt_sinir_sysBP)]
#bütün satırı getirdi
#bütün satırı getirdi


# In[36]:


print('BMI GRAFİĞİ ve AYKIRI VERİ LİSTESİ')
verilerKalpKriziRiskiTenYears['BMI'].plot(kind='box')
Q1=verilerKalpKriziRiskiTenYears.BMI.quantile(0.25)
Q3=verilerKalpKriziRiskiTenYears.BMI.quantile(0.75)
IQR=Q3-Q1
print('Bu değer üstü aykırı veridir: ',Q3+1.5*IQR)
print('Bu değer altı aykırı veridir: ',Q1-1.5*IQR)
ust_sinir_BMI=Q3+1.5*IQR
alt_sinir_BMI=Q1-1.5*IQR
verilerKalpKriziRiskiTenYears[(verilerKalpKriziRiskiTenYears['BMI']> ust_sinir_BMI) | (verilerKalpKriziRiskiTenYears['BMI']< alt_sinir_BMI)]
#bütün satırı getirdi
#bütün satırı getirdi


# In[37]:


print('diaBP GRAFİĞİ ve AYKIRI VERİ LİSTESİ')
verilerKalpKriziRiskiTenYears['diaBP'].plot(kind='box')
Q1=verilerKalpKriziRiskiTenYears.diaBP.quantile(0.25)
Q3=verilerKalpKriziRiskiTenYears.diaBP.quantile(0.75)
IQR=Q3-Q1
print('Bu değer üstü aykırı veridir: ',Q3+1.5*IQR)
print('Bu değer altı aykırı veridir: ',Q1-1.5*IQR)
ust_sinir_diaBP=Q3+1.5*IQR
alt_sinir_diaBP=Q1-1.5*IQR
verilerKalpKriziRiskiTenYears[(verilerKalpKriziRiskiTenYears['diaBP']> ust_sinir_diaBP) | (verilerKalpKriziRiskiTenYears['diaBP']< alt_sinir_diaBP)]
#bütün satırı getirdi
#bütün satırı getirdi


# In[38]:


print('heartRate GRAFİĞİ ve AYKIRI VERİ LİSTESİ')
verilerKalpKriziRiskiTenYears['heartRate'].plot(kind='box')
Q1=verilerKalpKriziRiskiTenYears.heartRate.quantile(0.25)
Q3=verilerKalpKriziRiskiTenYears.heartRate.quantile(0.75)
IQR=Q3-Q1
print('Bu değer üstü aykırı veridir: ',Q3+1.5*IQR)
print('Bu değer altı aykırı veridir: ',Q1-1.5*IQR)
ust_sinir_heartRate=Q3+1.5*IQR
alt_sinir_heartRate=Q1-1.5*IQR
verilerKalpKriziRiskiTenYears[(verilerKalpKriziRiskiTenYears['heartRate']> ust_sinir_heartRate) | (verilerKalpKriziRiskiTenYears['heartRate']< alt_sinir_heartRate)]
#bütün satırı getirdi
#bütün satırı getirdi


# In[39]:


print('glucose GRAFİĞİ ve AYKIRI VERİ LİSTESİ')
verilerKalpKriziRiskiTenYears['glucose'].plot(kind='box')
Q1=verilerKalpKriziRiskiTenYears.glucose.quantile(0.25)
Q3=verilerKalpKriziRiskiTenYears.glucose.quantile(0.75)
IQR=Q3-Q1
print('Bu değer üstü aykırı veridir: ',Q3+1.5*IQR)
print('Bu değer altı aykırı veridir: ',Q1-1.5*IQR)
ust_sinir_glucose=Q3+1.5*IQR
alt_sinir_glucose=Q1-1.5*IQR
verilerKalpKriziRiskiTenYears[(verilerKalpKriziRiskiTenYears['glucose']> ust_sinir_glucose) | (verilerKalpKriziRiskiTenYears['glucose']< alt_sinir_glucose)]
#bütün satırı getirdi
#bütün satırı getirdi


# In[40]:


#Aykırı verileri bulduk ve şimdi aykırı verileri alt sınıra ya da üst sınıra eşitleyelim


# In[41]:


#education sutununda aykırı veri yoktu işlemi geçiyorum


# In[42]:


cigs_veri=verilerKalpKriziRiskiTenYears.cigsPerDay.copy()
cigs_veri[(cigs_veri> ust_sinir_cigsPerDay) | (cigs_veri< alt_sinir_cigsPerDay)]=ust_sinir_cigsPerDay

print('\n  aykırı veri eşitlendiği Grafiği')
cigs_veri.plot(kind='box')


# In[43]:


prevalentStroke_veri=verilerKalpKriziRiskiTenYears.prevalentStroke.copy()
prevalentStroke_veri[(prevalentStroke_veri> ust_sinir_prevalentStroke) | (prevalentStroke_veri< alt_sinir_prevalentStroke)]=ust_sinir_prevalentStroke
print('\n  aykırı veri eşitlendiği Grafiği')
prevalentStroke_veri.plot(kind='box')


# In[44]:


BPMeds_veri=verilerKalpKriziRiskiTenYears.BPMeds.copy()
BPMeds_veri[(BPMeds_veri> ust_sinir_BPMeds) | (BPMeds_veri< alt_sinir_BPMeds)]=ust_sinir_BPMeds
print('\n  aykırı veri eşitlendiği Grafiği')
BPMeds_veri.plot(kind='box')


# In[45]:


totChol_veri=verilerKalpKriziRiskiTenYears.totChol.copy()
totChol_veri[(totChol_veri> ust_sinir_totChol) | (totChol_veri< alt_sinir_totChol)]=ust_sinir_totChol

print('\n  aykırı veri eşitlendiği Grafiği')
totChol_veri.plot(kind='box')


# In[46]:


diabetes_veri=verilerKalpKriziRiskiTenYears.diabetes.copy()
diabetes_veri[(diabetes_veri> ust_sinir_diabetes) | (diabetes_veri< alt_sinir_diabetes)]=ust_sinir_diabetes
print('\n  aykırı veri eşitlendiği Grafiği')
diabetes_veri.plot(kind='box')


# In[47]:


prevalentHyp_veri=verilerKalpKriziRiskiTenYears.prevalentHyp.copy()
prevalentHyp_veri[(prevalentHyp_veri> ust_sinir_prevalentHyp) | (prevalentHyp_veri< alt_sinir_prevalentHyp)]=ust_sinir_prevalentHyp
print('\n  aykırı veri eşitlendiği Grafiği')
prevalentHyp_veri.plot(kind='box')


# In[48]:


sysBP_veri=verilerKalpKriziRiskiTenYears.sysBP.copy()
sysBP_veri[(sysBP_veri> ust_sinir_sysBP) | (sysBP_veri< alt_sinir_sysBP)]=ust_sinir_sysBP

print('\n  aykırı veri eşitlendiği Grafiği')
sysBP_veri.plot(kind='box')


# In[49]:


BMI_veri=verilerKalpKriziRiskiTenYears.BMI.copy()
BMI_veri[(BMI_veri> ust_sinir_BMI) | (BMI_veri< alt_sinir_BMI)]=ust_sinir_BMI
print('\n  aykırı veri eşitlendiği Grafiği')
BMI_veri.plot(kind='box')


# In[50]:


diaBP_veri=verilerKalpKriziRiskiTenYears.diaBP.copy()
diaBP_veri[(diaBP_veri> ust_sinir_diaBP) | (diaBP_veri< alt_sinir_diaBP)]=ust_sinir_diaBP
print('\n  aykırı veri eşitlendiği Grafiği')
diaBP_veri.plot(kind='box')


# In[51]:


heartRate_veri=verilerKalpKriziRiskiTenYears.heartRate.copy()
heartRate_veri[(heartRate_veri> ust_sinir_heartRate) | (heartRate_veri< alt_sinir_heartRate)]=ust_sinir_heartRate
print('\n  aykırı veri eşitlendiği Grafiği')
heartRate_veri.plot(kind='box')


# In[52]:


glucose_veri=verilerKalpKriziRiskiTenYears.glucose.copy()
glucose_veri[(glucose_veri> ust_sinir_glucose) | (glucose_veri< alt_sinir_glucose)]=ust_sinir_glucose
print('\n  aykırı veri eşitlendiği Grafiği')
glucose_veri.plot(kind='box')


# In[53]:


#Kategorik veri yapma işlemi ,daha az bellek kullanmak için
verilerKalpKriziRiskiTenYears.dtypes
# veriTiplerini yazdırdıp tiplerine göre değiştireceğim obj olanları sayısallaştıracağım


# In[54]:


verilerKalpKriziRiskiTenYears.columns.get_loc('sex')  # işlem yapacağım sutun indexi 


# In[55]:


cinsiyet=verilerKalpKriziRiskiTenYears.iloc[:,3:4].values
cinsiyet


# In[56]:


from sklearn import preprocessing 
sex_kategori=preprocessing.LabelEncoder()
cinsiyet[:,0]=sex_kategori.fit_transform(verilerKalpKriziRiskiTenYears.iloc[:,3:4])
cinsiyet


# In[57]:


verilerKalpKriziRiskiTenYears.columns.get_loc('is_smoking') # hangi sutun index i bulalım


# In[58]:


sigara_icme_durumu=verilerKalpKriziRiskiTenYears.iloc[:,4:5].values
sigara_icme_durumu


# In[59]:


is_smoking_kategorize=preprocessing.LabelEncoder()
sigara_icme_durumu[:,0]=is_smoking_kategorize.fit_transform(verilerKalpKriziRiskiTenYears.iloc[:,4:5])
sigara_icme_durumu


# In[60]:


#KENDİ YAZDIĞIM ALGORİTMA İLE YAPALIM
cinsiyet=verilerKalpKriziRiskiTenYears.iloc[:,3:4].values
cinsiyet_satir=cinsiyet.shape[0]


# In[61]:


#F olanlar 1 # m olanlar 0 yapacağız
for i in range (cinsiyet_satir):
    if cinsiyet[i]=='F':
        cinsiyet[i]=1
    if cinsiyet[i]=='M':
        cinsiyet[i]=0


# In[62]:


cinsiyet


# In[63]:


#obj olan sigar içme durumunu da yapalım
sigara_icme_durumu=verilerKalpKriziRiskiTenYears.iloc[:,4:5].values
sigara_icme_durumu_satir=sigara_icme_durumu.shape[0]


# In[64]:


#yes olanlar 1 no olanlar 0 yapalım
for i in range (sigara_icme_durumu_satir):
    if sigara_icme_durumu[i]=='YES':
        sigara_icme_durumu[i]=1
    if sigara_icme_durumu[i]=='NO':
        sigara_icme_durumu[i]=0


# In[65]:


sigara_icme_durumu


# In[66]:


verilerKalpKriziRiskiTenYears.is_smoking=sigara_icme_durumu.copy() #burada kopyalayıp verileri kategorik yazdırdı
verilerKalpKriziRiskiTenYears.sex=cinsiyet.copy() 


# In[67]:


#değişti mi diye kontroll edelim
verilerKalpKriziRiskiTenYears.head(5)


# In[68]:


normalizasyonVeri=verilerKalpKriziRiskiTenYears.drop(['id','TenYearCHD'], axis=1)
normalizasyonVeri


# In[69]:


satirSayi=normalizasyonVeri.shape[1]
sutunSayi=normalizasyonVeri.shape[0]


# In[70]:


#Hazır Fonksiyon ile Normalizasyon ve Standardizasyon işlemi


# In[71]:


from sklearn.preprocessing import MinMaxScaler
scalerNorm= MinMaxScaler() 
normalizasyon_scale = scalerNorm.fit_transform(normalizasyonVeri) 


# In[72]:


normalizasyon_scale


# In[73]:


np_dizi_nrm=np.array(normalizasyon_scale)
boyutuDegisti_dizi_nrm=np.reshape(np_dizi_nrm,(sutunSayi,satirSayi))
listeDF_nrm=pd.DataFrame(boyutuDegisti_dizi_nrm)
listeDF_nrm


# In[74]:


from sklearn.preprocessing import StandardScaler
standardizasyon_veri=verilerKalpKriziRiskiTenYears.drop(['id','TenYearCHD'], axis=1)
scaleStd=StandardScaler()
std_veri=scaleStd.fit_transform(standardizasyon_veri)


# In[75]:


std_veri


# In[76]:


np_dizi_std=np.array(std_veri)
boyutuDegisti_dizi_std=np.reshape(np_dizi_std,(sutunSayi,satirSayi))
listeDF_std=pd.DataFrame(boyutuDegisti_dizi_std)
listeDF_std


# In[77]:


#Kendi yazdığım fonk ile normalizasyon işlemi
#her sutunun max ve min bul 
#her sutunun ort bul ve işlemi her satır için incele


# In[78]:


normalizasyonVeri_Fonksiyon=verilerKalpKriziRiskiTenYears.drop(['id','TenYearCHD'], axis=1)


# In[79]:


# sutunların ortalama min ve max ları listede tuttum

minListe=[]
maxListe=[]
ortListe=[]
ortalama=0.0
toplam=0.0
for i in range(satirSayi):
    en_kucuk_sayi=normalizasyonVeri_Fonksiyon.iloc[i,0]
    en_buyuk_sayi=normalizasyonVeri_Fonksiyon.iloc[i,0]
    for j in range(sutunSayi):
        veri=normalizasyonVeri_Fonksiyon.iloc[j,i]
        if en_kucuk_sayi>veri:
            en_kucuk_sayi=veri
        if en_buyuk_sayi<veri:
            en_buyuk_sayi=veri
        toplam=toplam+veri
    ortalama=toplam/sutunSayi
        
    ortListe.append(ortalama)
    minListe.append(en_kucuk_sayi)
    maxListe.append(en_buyuk_sayi)
    
    toplam=0.0
    ortalama=0.0


# In[80]:


normalizasyonListe=[]
for i in range(satirSayi):
    
    for j in range(sutunSayi):
        veri=normalizasyonVeri_Fonksiyon.iloc[j,i]
        normalizasyon=((veri-minListe[i])/(maxListe[i]-minListe[i]))
        normalizasyonListe.append(normalizasyon)
      
        
        
    
    


# In[81]:


np_dizi_nrm=np.array(normalizasyonListe)
boyutuDegisti_dizi_nrm=np.reshape(np_dizi_nrm,(sutunSayi,satirSayi))
listeDF_nrm=pd.DataFrame(boyutuDegisti_dizi_nrm)
listeDF_nrm


# In[82]:


#KORELASYON
# değişkenler bizim TenYearsCHD yi ne kadar etkiliyor bunu analiz edeceğiz. 
verilerKalpKriziRiskiTenYears.corr()


# In[83]:


corelasyon= verilerKalpKriziRiskiTenYears.corr()
sns.heatmap(corelasyon)


# In[84]:


ageCorr=verilerKalpKriziRiskiTenYears['age'].corr(verilerKalpKriziRiskiTenYears['TenYearCHD'])
ageCorr


# In[85]:


educationCorr=verilerKalpKriziRiskiTenYears['education'].corr(verilerKalpKriziRiskiTenYears['TenYearCHD'])
educationCorr


# In[86]:


cigsPerDayCorr=verilerKalpKriziRiskiTenYears['cigsPerDay'].corr(verilerKalpKriziRiskiTenYears['TenYearCHD'])
# tek tek butun hepsinin kalpkrizi riski
#ile ilişkisini inceleyip büyükten kücüğe değerleri sıralayıp riskte en etkili faktörü bulacağım.
cigsPerDayCorr


# In[87]:


BPMedsCorr=verilerKalpKriziRiskiTenYears['BPMeds'].corr(verilerKalpKriziRiskiTenYears['TenYearCHD'])
BPMedsCorr


# In[88]:


prevalentStrokeCorr=verilerKalpKriziRiskiTenYears['prevalentStroke'].corr(verilerKalpKriziRiskiTenYears['TenYearCHD'])
prevalentStrokeCorr


# In[89]:


prevalentHypCorr=verilerKalpKriziRiskiTenYears['prevalentHyp'].corr(verilerKalpKriziRiskiTenYears['TenYearCHD'])
prevalentHypCorr


# In[90]:


diabetesCorr=verilerKalpKriziRiskiTenYears['diabetes'].corr(verilerKalpKriziRiskiTenYears['TenYearCHD'])
diabetesCorr


# In[91]:


totCholCorr=verilerKalpKriziRiskiTenYears['totChol'].corr(verilerKalpKriziRiskiTenYears['TenYearCHD'])
totCholCorr


# In[92]:


sysBPCorr=verilerKalpKriziRiskiTenYears['sysBP'].corr(verilerKalpKriziRiskiTenYears['TenYearCHD'])
sysBPCorr


# In[93]:


diaBPCorr=verilerKalpKriziRiskiTenYears['diaBP'].corr(verilerKalpKriziRiskiTenYears['TenYearCHD'])
diaBPCorr


# In[94]:


heartRateCorr=verilerKalpKriziRiskiTenYears['heartRate'].corr(verilerKalpKriziRiskiTenYears['TenYearCHD'])
heartRateCorr


# In[95]:


glucoseCorr=verilerKalpKriziRiskiTenYears['glucose'].corr(verilerKalpKriziRiskiTenYears['TenYearCHD'])
glucoseCorr


# In[96]:


dictCorr = dict(ageCorrDeger=ageCorr, educationCorrDeger=educationCorr, cigsPerDayCorrDeger=cigsPerDayCorr,BPMedsCorrDeger=BPMedsCorr
               ,prevalentStrokeCorrDeger=prevalentStrokeCorr,prevalentHypCorrDeger=prevalentHypCorr,diabetesCorrDeger=diabetesCorr,
               totCholCorrDeger=totCholCorr,sysBPCorrDeger=sysBPCorr,diaBPCorrDeger=diaBPCorr,heartRateCorrDeger=heartRateCorr,
               glucoseCorrDeger=glucoseCorr)
print('Sözlük ',dictCorr, '\n')


# In[97]:


print('Sıralanmış ilişki değerleri  küçükten büyüğe doğru verilmiştir ')
dictCorr_siralanmis = {k: v for k, v in sorted(dictCorr.items(), key=lambda x: x[1])}
print(dictCorr_siralanmis)


# In[98]:


#en ayırt edici 5 özelliği kullanacağız
#ageCoor
#sysBPCorr
#prevalentHypCorr
#diaBPCorr
#glucoseCorr
#şimdi sınıflandırma işlemi yapacağız. 5 algoritma kullanılacak !!!!!


# In[ ]:





# In[99]:


test_edilecek_veri=pd.read_csv('test.csv')
test_edilecek_veri.head()


# In[100]:


test_edilecek_veri.isnull()


# In[101]:


test_edilecek_veri.isnull().sum()


# In[102]:


from sklearn.impute import SimpleImputer


# In[103]:


imputer=SimpleImputer(missing_values=np.NaN, strategy='mean')


# In[104]:


test_edilecek_veri['education']=imputer.fit_transform(test_edilecek_veri['education'].values.reshape(-1,1))
test_edilecek_veri['cigsPerDay']=imputer.fit_transform(test_edilecek_veri['cigsPerDay'].values.reshape(-1,1))
test_edilecek_veri['BPMeds']=imputer.fit_transform(test_edilecek_veri['BPMeds'].values.reshape(-1,1))
test_edilecek_veri['totChol']=imputer.fit_transform(test_edilecek_veri['totChol'].values.reshape(-1,1))
test_edilecek_veri['BMI']=imputer.fit_transform(test_edilecek_veri['BMI'].values.reshape(-1,1))
test_edilecek_veri['glucose']=imputer.fit_transform(test_edilecek_veri['glucose'].values.reshape(-1,1))


# In[105]:


test_edilecek_veri.isnull().sum()


# In[164]:


test_edilecek_veri.columns.get_loc('is_smoking') # hangi sutun index i bulalım


# In[165]:


sigara_icme_durumu_test=test_edilecek_veri.iloc[:,4:5].values
sigara_icme_durumu_test


# In[ ]:





# In[178]:


is_smoking_kategorize_test=preprocessing.LabelEncoder()


# In[179]:


sigara_icme_durumu_test[:,0]=is_smoking_kategorize_test.fit_transform(test_edilecek_veri.iloc[:,4:5])
sigara_icme_durumu_test


# In[180]:


test_edilecek_veri.columns.get_loc('sex')  # işlem yapacağım sutun indexi 
cinsiyet_test=test_edilecek_veri.iloc[:,3:4].values
cinsiyet_test


# In[181]:


from sklearn import preprocessing 
sex_kategori_test=preprocessing.LabelEncoder()
cinsiyet_test[:,0]=sex_kategori_test.fit_transform(test_edilecek_veri.iloc[:,3:4])
cinsiyet_test


# In[182]:


test_edilecek_veri.is_smoking=sigara_icme_durumu_test.copy() #burada kopyalayıp verileri kategorik yazdırdı
test_edilecek_veri.sex=cinsiyet_test.copy() 


# In[ ]:





# In[183]:


frames = [verilerKalpKriziRiskiTenYears,test_edilecek_veri]  
result = pd.concat(frames)


# In[184]:


#result=result.drop(columns=['id','sex','is_smoking'], axis='columns')


# In[185]:


result.shape


# In[186]:


result.tail()


# In[187]:


#1Algoritma   Sınıflandırma 
from sklearn.model_selection import train_test_split
#en ayırt edici 5 özelliği kullanacağız
#ageCoor
#sysBPCorr
#prevalentHypCorr
#diaBPCorr
#glucoseCorr
#şimdi sınıflandırma işlemi yapacağız. 5 algoritma kullanılacak !!!!!


# In[ ]:





# In[188]:


#verileri eğitmeye başlayalım
X=result.drop(columns=['TenYearCHD'], axis=1)
y=result['TenYearCHD']


# In[189]:


X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.20, random_state=0, shuffle=False)


# In[190]:


X_train.shape


# In[191]:


X_test.shape


# In[192]:


y_train.shape


# In[193]:


y_test.shape


# In[ ]:





# In[194]:


#Linear Regresyon Modeli


# In[195]:


from sklearn.linear_model import LinearRegression


# In[196]:


linear_regresyon_model=LinearRegression()
linear_regresyon_model.fit(X_train,y_train)
linear_regresyon_tahmin=linear_regresyon_model.predict(X_test)
linear_regresyon_DataFrame=pd.DataFrame(linear_regresyon_tahmin)
linear_regresyon_DataFrame


# In[197]:


#scor bul
X_Control=verilerKalpKriziRiskiTenYears.drop(columns=['TenYearCHD'], axis=1)
y_Control=verilerKalpKriziRiskiTenYears['TenYearCHD']


# In[198]:


X_train_linear_regresyon,X_test_linear_regresyon,y_train_linear_regresyon,y_test_linear_regresyon=train_test_split(X_Control,y_Control, test_size=0.20, random_state=0, shuffle=False)


# In[199]:


linear_regresyon_model_control=LinearRegression()
linear_regresyon_model_control.fit(X_train_linear_regresyon,y_train_linear_regresyon)
linear_regresyon_tahmin_control=linear_regresyon_model_control.predict(X_test_linear_regresyon)
linear_regresyon_DataFrame_control=pd.DataFrame(linear_regresyon_tahmin_control)
linear_regresyon_DataFrame_control


# In[200]:


print('İntercept', linear_regresyon_model_control.intercept_)
print('Coef', linear_regresyon_model_control.coef_)


# In[201]:


df_p=pd.DataFrame({'Gerçek:':y_test_linear_regresyon ,'tahmin': linear_regresyon_tahmin_control})
df_p


# In[206]:


from sklearn.metrics import accuracy_score,mean_squared_error,r2_score


# In[317]:


mean_error_linear_regresyon=mean_squared_error(y_test_linear_regresyon,linear_regresyon_tahmin_control)
r2_linear_regresyon=r2_score(y_test_linear_regresyon,linear_regresyon_tahmin_control)
print('Mean error', mean_error_linear_regresyon)
print('r2', r2_linear_regresyon)


# In[223]:


#Karar Ağacı


# In[224]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


# In[225]:


karar_agaci_model=DecisionTreeClassifier()
karar_agaci_model.fit(X_train,y_train)
karar_agaci_tahmin=karar_agaci_model.predict(X_test)
karar_agaci_DataFrame=pd.DataFrame(karar_agaci_tahmin)
karar_agaci_DataFrame


# In[226]:


#Conf matris vs bulmak için elimizde çıktı değerleri olanı bölüp oranlarını bulacağım


# In[227]:


#scor bul
X_Control=verilerKalpKriziRiskiTenYears.drop(columns=['TenYearCHD'], axis=1)
y_Control=verilerKalpKriziRiskiTenYears['TenYearCHD']


# In[228]:


X_train_karar_agaci,X_test_karar_agaci,y_train_karar_agaci,y_test_karar_agaci=train_test_split(X_Control,y_Control, test_size=0.20, random_state=0, shuffle=False)


# In[229]:


karar_agaci_model_control=DecisionTreeClassifier()
karar_agaci_model_control.fit(X_train_karar_agaci,y_train_karar_agaci)
karar_agaci_tahmin_control=karar_agaci_model_control.predict(X_test_karar_agaci)
karar_agaci_DataFrame_control=pd.DataFrame(karar_agaci_tahmin_control)
karar_agaci_DataFrame_control


# In[326]:


confusion_matrix_karar_agaci=confusion_matrix(y_test_karar_agaci,karar_agaci_tahmin_control)
print('Conf matris',confusion_matrix_karar_agaci)
accuracy_score_karar_agaci=accuracy_score(y_test_karar_agaci,karar_agaci_tahmin_control)
print('Acc score' ,accuracy_score_karar_agaci)
score_karar_agaci=karar_agaci_model_control.score(X_test_karar_agaci,y_test_karar_agaci)
print('Score', score_karar_agaci)
mean_error_karar_agaci=mean_squared_error(y_test_karar_agaci,karar_agaci_tahmin_control)
print('Mean error', mean_error_karar_agaci)
r2_karar_agaci=r2_score(y_test_karar_agaci,karar_agaci_tahmin_control)
print('R2 score', r2_karar_agaci)


# In[ ]:


#Logistic model    


# In[233]:


from sklearn.linear_model import LogisticRegression #lojistik regresyon 
from sklearn.metrics import  confusion_matrix,  f1_score, recall_score


# In[235]:


logistic_regresyon_model=LogisticRegression(max_iter=10000)
logistic_regresyon_model.fit(X_train,y_train)
logistic_regresyon_tahmin=logistic_regresyon_model.predict(X_test)
logistic_regresyon_DataFrame=pd.DataFrame(logistic_regresyon_tahmin)
logistic_regresyon_DataFrame


# In[237]:


#scor bul
X_Logistic_Regresyon=verilerKalpKriziRiskiTenYears.drop(columns=['TenYearCHD'], axis=1)
y_Logistic_Regresyon=verilerKalpKriziRiskiTenYears['TenYearCHD']
X_train_Logistic_Regresyon,X_test_Logistic_Regresyon,y_train_Logistic_Regresyon,y_test_Logistic_Regresyon=train_test_split(X_Logistic_Regresyon,y_Logistic_Regresyon, test_size=0.2)
logistic_regresyon_model.fit(X_train_Logistic_Regresyon,y_train_Logistic_Regresyon)
logistic_regresyon_tahmin_control=logistic_regresyon_model.predict(X_test_Logistic_Regresyon)
logistic_regresyon_DataFrame_control = pd.DataFrame({'Gerçek Değer':y_test_Logistic_Regresyon, 'Tahmin Değeri':logistic_regresyon_tahmin_control})
logistic_regresyon_DataFrame_control


# In[328]:


confusion_matrix_logistic_regresyon=confusion_matrix(y_test_Logistic_Regresyon,logistic_regresyon_tahmin_control)
print('Conf matris',confusion_matrix_logistic_regresyon)
accuracy_score_logistic_regresyon=accuracy_score(y_test_Logistic_Regresyon,logistic_regresyon_tahmin_control)
print('Acc score' ,accuracy_score_logistic_regresyon)
score_logistic_regresyon=karar_agaci_model_control.score(X_test_Logistic_Regresyon,y_test_Logistic_Regresyon)
print('Score', score_logistic_regresyon)
mean_error_logistic_regresyon=mean_squared_error(y_test_Logistic_Regresyon,logistic_regresyon_tahmin_control)
print('Mean error', mean_error_logistic_regresyon)
r2_logistic_regresyon=r2_score(y_test_Logistic_Regresyon,logistic_regresyon_tahmin_control)
print('r2 score', r2_logistic_regresyon)


# In[253]:


#Model Svm 
from sklearn.svm import SVC
svm_Model=DecisionTreeClassifier()
svm_Model.fit(X_train,y_train)
svm_Tahmin=svm_Model.predict(X_test)
svm_DataFrame=pd.DataFrame({'Tahminler': svm_Tahmin})
svm_DataFrame


# In[264]:


X_SVM_control=verilerKalpKriziRiskiTenYears.drop(columns=['TenYearCHD'], axis=1)
y_SVM_control=verilerKalpKriziRiskiTenYears['TenYearCHD']


# In[268]:


X_train_SVM,X_test_SVM,y_train_SVM,y_test_SVM=train_test_split(X_SVM_control,y_SVM_control, test_size=0.2)
svm_Model_control=SVC()
svm_Model_control.fit(X_train_SVM,y_train_SVM)
SVM_tahmin_control=svm_Model_control.predict(X_test_SVM)
dfSVM= pd.DataFrame({'Gerçek Değer':y_test_SVM, 'Tahmin Değeri':SVM_tahmin_control})
dfSVM


# In[270]:


confusion_matrix_SVM=confusion_matrix(y_test_SVM,SVM_tahmin_control)
r2_score_SVM=r2_score(y_test_SVM,SVM_tahmin_control)
mean_error_SVM=mean_squared_error(y_test_SVM,SVM_tahmin_control)
accuarry_SVM= accuracy_score(y_test_SVM,SVM_tahmin_control)
print('Conf matris',confusion_matrix_SVM)
print('R2 skoru:', r2_score_SVM)
print('Ortalama kare hatası', mean_error_SVM)
print('Kesinlik değeri', accuarry_SVM)


# In[271]:


#Model Neighborn


# In[272]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score


# In[273]:


KNeighborsClassifierModel=KNeighborsClassifier(n_neighbors=5)
KNeighborsClassifierModel
KNeighborsClassifierModel.fit(X_train,y_train)
KN_tahmin=KNeighborsClassifierModel.predict(X_test)
KN_DataFrame=pd.DataFrame(KN_tahmin)
KN_DataFrame


# In[276]:


X_KN=verilerKalpKriziRiskiTenYears.drop(columns=['TenYearCHD'], axis=1)
y_KN=verilerKalpKriziRiskiTenYears['TenYearCHD']
X_train_KN,X_test_KN,y_train_KN,y_test_KN=train_test_split(X_KN,y_KN, test_size=0.20, random_state=0, shuffle=False)
KNeighborsClassifierModel.fit(X_train_KN,y_train_KN)
KN_tahmin_control=KNeighborsClassifierModel.predict(X_test_KN)
KN_DataFrame_control = pd.DataFrame({'Gerçek Değer':y_test_KN, 'Tahmin Değeri':KN_tahmin_control})
KN_DataFrame_control


# In[279]:


confusion_matrix_KN=confusion_matrix(y_test_KN,KN_tahmin_control)
r2_score_KN=r2_score(y_test_KN,KN_tahmin_control)
mean_error_KN=mean_squared_error(y_test_KN,KN_tahmin_control)
accuarry_KN= accuracy_score(y_test_KN,KN_tahmin_control)
print('Conf matris', confusion_matrix_KN)
print('R2 skoru:', r2_score_KN)
print('Ortalama kare hatası', mean_error_KN)
print('Kesinlik değeri',accuarry_KN)


# In[309]:


import matplotlib.pyplot as plt


# In[324]:


x = np.array([ "KararAgacı","LogisticRegresyon", "SVM", "KNeighborn"])
y = np.array([accuracy_score_karar_agaci, accuracy_score_logistic_regresyon, accuarry_SVM, accuarry_KN])

plt.bar(x,y, width=0.5)
plt.show()


# In[335]:


x = np.array(["LinearRegresyon", "KararAgacı","LogisticRegresyon", "SVM", "KNeighborn"])
y = np.array([r2_linear_regresyon,r2_karar_agaci,r2_logistic_regresyon,r2_score_SVM,r2_score_KN])
plt.plot(x,y, marker = 'o')
plt.show()


# In[338]:


x = np.array(["LinearRegresyon", "KararAgacı","LogisticRegresyon", "SVM", "KNeighborn"])
ypoints = np.array([mean_error_linear_regresyon,mean_error_karar_agaci,mean_error_logistic_regresyon,mean_error_SVM,mean_error_KN])

plt.plot(x, y, 'o')
plt.show()


# In[339]:


x = np.array(["AccDegeri", "r2Scoru","scoru", "hataOranı"])
y = np.array([accuracy_score_logistic_regresyon,r2_logistic_regresyon,score_logistic_regresyon,mean_error_logistic_regresyon])
plt.plot(x,y, marker = 'o')
plt.show()


# In[ ]:




