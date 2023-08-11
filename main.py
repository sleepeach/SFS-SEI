from utils import *
from config import *
from model.cvnn import base_complex_model,prune_complex_model

if __name__ == '__main__':
   # 设置随机种子
   set_seed(300)

   # load dataset
   train_dataloader, val_dataloader, test_dataloader = Data_prepared(Dataset)

   # load model
   model = base_complex_model()

   # train
   train_and_val(model, HP_train, HP_val, train_dataloader, val_dataloader)

   # test
   model_prune = prune_complex_model()
   prune_model_and_test(model_prune, HP_test, test_dataloader)