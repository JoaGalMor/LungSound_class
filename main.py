from data_treatment import *
from utils import *
from CNN import *

data,data_reduced=DataTreatment().read_txt(all_sound_files)
data_noised=DataTreatment().add_noise(data_reduced)
data_shifted=DataTreatment().add_shifted(data_reduced)
all_data=[data,data_noised,data_shifted]
data_final=pd.concat(all_data)
dt=DataTreatment()
dt.perform_feature_engineering(data_final)


net=CNN(dt.mfcc_shape,dt.croma_shape,dt.mspec_shape)
features=["['croma', 'mfcc']","['croma', 'mfcc','mspec']"]
for feature in features:
    net.create_net(chronic_yn=False,features=feature)
    history=net.run_net(dt,epochs=30,features=feature)



