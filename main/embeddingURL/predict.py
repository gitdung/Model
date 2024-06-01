from transformers import AutoTokenizer , AutoModelForSequenceClassification
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


tokenizer = AutoTokenizer.from_pretrained('gechim/phobert-base-v2-finetuned')
phoBert = AutoModel.from_pretrained("gechim/phobert-base-v2-finetuned")

class NN(nn.Module):
  def __init__(self, input_size, num_classes):
    super(NN, self).__init__()
    self.phoBert = phoBert # (batchsize , 1 , 768)
    self.num_classes = num_classes
    self.fc1 = nn.Linear(input_size, 256)
    self.fc2 = nn.Linear(256, 768) #(batchsize , 1 , 768)
    self.dropout_nn = nn.Dropout(0.1)
    self.dropout_lm = nn.Dropout(0.1)


    # self.out = nn.Linear(768, num_classes)
    self.out = nn.Linear(1536, num_classes)

  def forward(self, features, input_ids, token_type_ids, attention_mask , labels):
    # output bên sang
    x_nn = F.relu(self.fc1(features))
    x_nn = F.relu(self.fc2(x_nn))

    # output bên bảo
    x_phoBert = self.phoBert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).last_hidden_state[:,0,:]

    #drop out trước khi concat
    x_nn = self.dropout_nn(x_nn)
    x_phoBert = self.dropout_lm(x_phoBert)

    # print(x_phoBert.shape)
    logits = self.out(torch.cat(( x_nn , x_phoBert) , dim=1)) #self.out( x_nn + x_phoBert)


    # tính loss cái này chỉ để hiện kq loss tập valid
    loss = None
    if labels is not None:
      loss_fct = nn.CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
    return SequenceClassifierOutput(loss = loss , logits=logits) # hàm trainer cần cái này nó mới chịu train

print("Model loaded \n\n\n")
model = torch.load('D:/Workspace/Project_VNNIC/models/model_concat_dataV2.pt' , map_location=torch.device('cpu'))
print(model)
print("\n\n\n")


def normalize_url(url):
    if url.startswith("http://"):
        url = url[7:]
        url = url.replace("www.", "")
    if url.startswith("https://"):
        url = url[8:]
        url = url.replace("www.", "")
        url = url.replace(".", " ")
    url = url.replace(".", " ")
    url = url.replace("/", "")
    url = url.replace("edu vn", "")
    url = url.replace("com vn", "")
    url = url.replace("net vn", "")
    url = url.replace("org vn", "")
    url = url.replace("gov vn", "")
    url = url.replace("vn", "")
    return url

def predict(url):
    url = normalize_url(url)
    url_tokenize = tokenizer(url , return_tensors='pt')
    x_feature = torch.tensor([[9, 2.725480557 ,	0 ,	 0 , 0 , 0 , 0 , 1 , 0 , 0 ]])
    y = model(features = x_feature,input_ids = url_tokenize['input_ids'] , token_type_ids = url_tokenize['token_type_ids'] , attention_mask = url_tokenize['attention_mask']  , labels = torch.tensor([1])).logits
    
    print(torch.argmax(y).item())
    
    if torch.argmax(y).item() == 0:
        return "Bình thường"
    if torch.argmax(y).item() == 1:
        return "Có tín nhiệm thấp"
    
    
if __name__ == "__main__":
    predict("https://www.google.com.vn")