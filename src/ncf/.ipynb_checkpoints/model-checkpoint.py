import torch
import torch.nn as nn
from utils import ndcg, hr
import numpy as np
import os
import random

class GMF(nn.Module):
    
    def __init__(self, config):
        super(GMF, self).__init__()
        # Слои эмбеддингов пользователей и артистов
        self.user_embeddings = nn.Embedding(config.user_count, config.gmf_dim)
        self.item_embeddings = nn.Embedding(config.item_count, config.gmf_dim)
        # Слой, который отобразит значения поэлементного умножения в вероятность
        self.out = nn.Sequential(nn.Linear(config.gmf_dim, 1), nn.Sigmoid())
        
    def forward(self, items, users):
        # Репрезентация пользователей и артистов
        user_embed = self.user_embeddings(users)
        item_embed = self.item_embeddings(items)
        # Поэлементное умножение, результат которого репрезентуется как MF
        product = user_embed * item_embed
        prob = self.out(product)
        return prob
    
    
class MLP(nn.Module):
    
    def __init__(self, config):
        super(MLP, self).__init__()
        # Как было сказано в статье - более хорошего результата можно добиться с помощью
        # Пиромидаидальной структуры скрытых слоёв - те основание сети намного шире чем его верхняя часть
        self.user_embeddings = nn.Embedding(config.user_count, config.mlp_dim * 2**(config.layer_count - 1))
        self.item_embeddings = nn.Embedding(config.item_count, config.mlp_dim * 2**(config.layer_count - 1))
        Hidden_layers = []
        for i in range(config.layer_count):
            input_size = config.mlp_dim * 2**(config.layer_count - i)
            Hidden_layers.extend([nn.Linear(input_size, input_size // 2), nn.LeakyReLU(config.slope), nn.Dropout(config.dropout)])
        self.hidden = nn.Sequential(*Hidden_layers)
        self.out = nn.Sequential(nn.Linear(input_size // 2, 1), nn.Sigmoid())
        
    
    def forward(self, items, users):
        user_embed = self.user_embeddings(users)
        item_embed = self.item_embeddings(items)
        hidden_input = torch.cat((user_embed, item_embed), dim=1)
        hidden_output = self.hidden(hidden_input)
        
        out = self.out(hidden_output)
        return out
    
# NeuMF - по факту является ансамблем GMF и MLP; При использовании pre-trained моделей GMF и MLP
# веса последнего слоя от 1й модели берутся с некоторым коэффициентом альфа (гиперпараметр); соответсвенно от второй берутся значения умноженные на (1 - альфа)
class NeuMF(nn.Module):
    
    def __init__(self, config):
        super(NeuMF, self).__init__()
        
        self.user_embeddings_mf = nn.Embedding(config.user_count, config.gmf_dim)
        self.item_embeddings_mf = nn.Embedding(config.item_count, config.gmf_dim)
        
        self.user_embeddings_mlp = nn.Embedding(config.user_count, config.mlp_dim * 2**(config.layer_count - 1))
        self.item_embeddings_mlp = nn.Embedding(config.item_count, config.mlp_dim * 2**(config.layer_count - 1))
        Hidden_layers = []
        for i in range(config.layer_count):
            input_size = config.mlp_dim * 2**(config.layer_count - i)
            Hidden_layers.extend([nn.Linear(input_size, input_size // 2), nn.LeakyReLU(config.slope), nn.Dropout(config.dropout)])
        self.hidden = nn.Sequential(*Hidden_layers)
        # Всё выше написанное можно увидеть в GMF и MLP, отличие только в out слое
        # его input_dim = gmf + output_dim^(l)
        # тк там происходит конкатенация выходов MLP и GMF
        self.out = nn.Sequential(nn.Linear(config.gmf_dim + (input_size) // 2, 1), nn.Sigmoid())
        
        
    def forward(self, items, users):
        # GMF ветвь сети
        user_emb_mf = self.user_embeddings_mf(users)
        item_emb_mf = self.item_embeddings_mf(items)
        
        mf_dot_product = user_emb_mf * item_emb_mf
        
        # MLP ветвь сети
        user_emb_mlp = self.user_embeddings_mlp(users)
        item_emb_mlp = self.item_embeddings_mlp(items)
        
        mlp_input = torch.cat((user_emb_mlp, item_emb_mlp), dim=1)
        mlp_output = self.hidden(mlp_input)
        
        # Конкатенация выходов
        result_vector = torch.cat((mf_dot_product, mlp_output), dim=1)
        prob = self.out(result_vector)
        return prob

    
class UserNeuMF(nn.Module):
    def __init__(self, config):
        super(UserNeuMF, self).__init__()
        
        self.user_embeddings_mf = nn.Embedding(config.user_count, config.gmf_dim)
        self.item_embeddings_mf = nn.Embedding(config.item_count, config.gmf_dim)
        
        self.user_embeddings_mlp = nn.Embedding(config.user_count, config.mlp_dim * 2**(config.layer_count - 1))
        self.item_embeddings_mlp = nn.Embedding(config.item_count, config.mlp_dim * 2**(config.layer_count - 1))
        Hidden_layers = []
        additional_features = config.user_features
        for i in range(config.layer_count):
            input_size = config.mlp_dim * 2**(config.layer_count - i)
            if i != 0:
                additional_features = 0
            Hidden_layers.extend([nn.Linear(input_size + additional_features, input_size // 2), nn.LeakyReLU(config.slope), nn.Dropout(config.dropout)])
        self.hidden = nn.Sequential(*Hidden_layers)
        # Всё выше написанное можно увидеть в GMF и MLP, отличие только в out слое
        # его input_dim = gmf + output_dim^(l)
        # тк там происходит конкатенация выходов MLP и GMF
        self.out = nn.Sequential(nn.Linear(config.gmf_dim + (input_size) // 2, 1), nn.Sigmoid())
        
        
    def forward(self, items, users, user_features):
        # GMF ветвь сети
        user_emb_mf = self.user_embeddings_mf(users)
        item_emb_mf = self.item_embeddings_mf(items)
        
        mf_dot_product = user_emb_mf * item_emb_mf
        
        # MLP ветвь сети
        user_emb_mlp = self.user_embeddings_mlp(users)
        item_emb_mlp = self.item_embeddings_mlp(items)
        
        mlp_input = torch.cat((user_emb_mlp, item_emb_mlp, user_features), dim=1)
        mlp_output = self.hidden(mlp_input)
        
        # Конкатенация выходов
        result_vector = torch.cat((mf_dot_product, mlp_output), dim=1)
        prob = self.out(result_vector)
        return prob

    
    
def load_pretrained_weights_neu_mf(gmf_model, mlp_model, neu_mf_config):
    neu_mf = NeuMF(neu_mf_config)
    # GMF Веса эмбеддингов
    neu_mf.user_embeddings_mf.weight = gmf_model.user_embeddings.weight
    neu_mf.item_embeddings_mf.weight = gmf_model.item_embeddings.weight
    # MLP Веса эмбеддингов
    neu_mf.user_embeddings_mlp.weight = mlp_model.user_embeddings.weight
    neu_mf.item_embeddings_mlp.weight = mlp_model.item_embeddings.weight
    # Обновление весов в linear слоях, которые лежат в seqential под названием hidden
    neu_mf_state_dict = neu_mf.state_dict()
    mlp_state_dict = mlp_model.hidden.state_dict()
    mlp_state_dict = {'hidden.'+k : v for k,v in mlp_state_dict.items()}
    neu_mf_state_dict.update(mlp_state_dict)
    neu_mf.load_state_dict(neu_mf_state_dict)
    
    # Перенос весов последих out слоев
    gmf_out_layer_weights, gmf_out_layer_bias = gmf_model.out[0].weight, gmf_model.out[0].bias
    mlp_out_layer_weights, mlp_out_layer_bias = mlp_model.out[0].weight, mlp_model.out[0].bias
    # Как было сказано в статье, нужно ввести параметр альфа, определяющий трейдоф от pretrained моделей
    # Так же как и в статье, возьмём альфа равное 0.5
    # Но так как коэффициент симметричен (те 1-0.5 = 0.5), то можно умножение добавить в самом конце
    out_weights = torch.cat((gmf_out_layer_weights, mlp_out_layer_weights), dim=1)
    out_bias = gmf_out_layer_bias + mlp_out_layer_bias
    alpha = neu_mf_config.alpha
    neu_mf.out[0].weight = nn.Parameter(alpha * out_weights)
    neu_mf.out[0].bias = nn.Parameter(alpha * out_bias)
    return neu_mf


def load_pretrained_weights_neu_mf_from_path(gmf_model_path, gmf_config, mlp_model_path, mlp_config, neu_mf_config):
    gmf_model = GMF(gmf_config)
    gmf_model.load_state_dict(torch.load(gmf_model_path))
    gmf_model.eval()
    mlp_model = MLP(mlp_config)
    mlp_model.load_state_dict(torch.load(mlp_model_path))
    mlp_model.eval()
    return load_pretrained_weights_neu_mf(gmf_model, mlp_model, neu_mf_config)

def train_user_neu(model, train, test, val, optim, crit, device, sum_writer, config):
    stop = 500
    epochs = config.epochs
    train_tag = 'user_neu_mf_train_loss'
    test_tag = 'user_neu_mf_test_loss'
    ndcg_tag = 'ndcg_neu_mf'
    hr_tag = 'hr_neu_mf'
    topK = config.top_k
    item_count = config.item_count
    save_path = config.save_path + 'USR'
    items_indexes = [i for i in range(item_count)]
    val_items_input = torch.stack([torch.tensor(i).long() for i in range(item_count)])
    val_items_input = val_items_input.to(device)
    
    for epoch in range(epochs):
        print('epoch[{}]'.format(epoch))
        for batch_num, data in enumerate(train, 0):
            optim.zero_grad()
            
            users_items, label = data
            users, items, features = users_items
            
            label = label.to(device).float()
            users = users.long().to(device)
            items = items.long().to(device)
            features = features.to(device)
            
            output = model(items, users, features).squeeze(-1)
            loss = crit(output, label)
            loss.backward()
            
            optim.step()
            sum_writer.add_scalar(train_tag, loss.item(), batch_num + len(train)*epoch)
            
        for batch_num, data in enumerate(test, 0):
            with torch.no_grad():
                users_items, label = data
                users, items, features = users_items
                
                label = label.to(device).float()
                users = users.long().to(device)
                items = items.long().to(device)
                features = features.to(device)
                
                output = model(items, users, features).squeeze(-1)
                loss = crit(output, label)
                
                sum_writer.add_scalar(test_tag, loss.item(), batch_num + len(test)*epoch)

        torch.save(model.state_dict(), os.path.join(save_path, 'epoch_{}.ckpt'.format(epoch + 1)))
    for i, data in enumerate(val, 0):
        if i > stop:
            break
        user, items, features = data
        with torch.no_grad():
            user_input = torch.stack([torch.tensor(user).long() for _ in range(item_count)])
            user_input = user_input.to(device)
            features = features.to(device)
            probs = model(val_items_input, user_input, features).squeeze(-1).detach().cpu().numpy()
            top_k_predictions = sorted(list(zip(items_indexes, probs)), key=lambda x: x[1], reverse=True)[:topK]
            top_k_predictions = np.asarray(top_k_predictions)
            predicted_items = top_k_predictions[:,0]
            predicted_scores = top_k_predictions[:,1]
            ndcg_score = ndcg(items, predicted_items, predicted_scores)
            hr_score = hr(items, predicted_items)

            sum_writer.add_scalar(ndcg_tag, ndcg_score, i + (epoch // 5)*len(val))
            sum_writer.add_scalar(hr_tag, hr_score, i + (epoch // 5)*len(val))
    
    
def train_ncf(model, train, test, val, optim, crit, device, sum_writer, config):
    stop = 500
    print('start training')
    epochs = config.epochs
    train_tag = config.train
    test_tag = config.test
    ndcg_tag = config.ndcg
    hr_tag = config.hr
    validate = config.evaluate
    topK = config.top_k
    item_count = config.item_count
    save_path = config.save_path
    items_indexes = [i for i in range(item_count)]
    val_items_input = torch.stack([torch.tensor(i).long() for i in range(item_count)])
    val_items_input = val_items_input.to(device)
    
    for epoch in range(epochs):
        for batch_num, data in enumerate(train, 0):
            optim.zero_grad()
            
            users_items, label = data
            users, items = users_items
            
            label = label.to(device).float()
            users = users.long().to(device)
            items = items.long().to(device)
            
            output = model(items, users).squeeze(-1)
            loss = crit(output, label)
            loss.backward()
            
            optim.step()
            sum_writer.add_scalar(train_tag, loss.item(), batch_num + len(train)*epoch)
            
        for batch_num, data in enumerate(test, 0):
            with torch.no_grad():
                users_items, label = data
                users, items = users_items
                
                label = label.to(device).float()
                users = users.long().to(device)
                items = items.long().to(device)
                
                output = model(items, users).squeeze(-1)
                loss = crit(output, label)
                
                sum_writer.add_scalar(test_tag, loss.item(), batch_num + len(test)*epoch)

        torch.save(model.state_dict(), os.path.join(save_path, 'epoch_{}.ckpt'.format(epoch + 1)))
    for i, data in enumerate(val, 0):
        if i > stop:
            break
        user, items = data
        with torch.no_grad():
            user_input = torch.stack([torch.tensor(user).long() for _ in range(item_count)])
            user_input = user_input.to(device)
            probs = model(val_items_input, user_input).squeeze(-1).detach().cpu().numpy()
            top_k_predictions = sorted(list(zip(items_indexes, probs)), key=lambda x: x[1], reverse=True)[:topK]
            top_k_predictions = np.asarray(top_k_predictions)
            predicted_items = top_k_predictions[:,0]
            predicted_scores = top_k_predictions[:,1]
            ndcg_score = ndcg(items, predicted_items, predicted_scores)
            hr_score = hr(items, predicted_items)

            sum_writer.add_scalar(ndcg_tag, ndcg_score, i + (epoch // 5)*len(val))
            sum_writer.add_scalar(hr_tag, hr_score, i + (epoch // 5)*len(val))
            
    
def evaluate_neu(model, val, device, sum_writer, config, stop=500):
    ndcg_tag = config.ndcg
    hr_tag = config.hr
    topK = config.top_k
    item_count = config.item_count
    items_indexes = [i for i in range(item_count)]
    val_items_input = torch.stack([torch.tensor(i).long() for i in range(item_count)])
    val_items_input = val_items_input.to(device)
    for i, data in enumerate(val, 0):
        if i > stop:
            break
        user, items = data
        with torch.no_grad():
            user_input = torch.stack([torch.tensor(user).long() for _ in range(item_count)])
            user_input = user_input.to(device)
            probs = model(val_items_input, user_input).squeeze(-1).detach().cpu().numpy()
            top_k_predictions = sorted(list(zip(items_indexes, probs)), key=lambda x: x[1], reverse=True)[:topK]
            top_k_predictions = np.asarray(top_k_predictions)
            predicted_items = top_k_predictions[:,0]
            predicted_scores = top_k_predictions[:,1]
            ndcg_score = ndcg(items, predicted_items, predicted_scores)
            hr_score = hr(items, predicted_items)

            sum_writer.add_scalar(ndcg_tag, ndcg_score, i)
            sum_writer.add_scalar(hr_tag, hr_score, i)

def evaluate_user_neu(model, val, device, sum_writer, config, stop=500):
    ndcg_tag = config.ndcg
    hr_tag = config.hr
    topK = config.top_k
    item_count = config.item_count
    items_indexes = [i for i in range(item_count)]
    val_items_input = torch.stack([torch.tensor(i).long() for i in range(item_count)])
    val_items_input = val_items_input.to(device)
    for i, data in enumerate(val, 0):
        if i > stop:
            break
        user, items, features = data
        with torch.no_grad():
            user_input = torch.stack([torch.tensor(user).long() for _ in range(item_count)])
            features = torch.stack([features for i in range(item_count)])
            user_input = user_input.to(device)
            features = features.to(device)
            probs = model(val_items_input, user_input, features).squeeze(-1).detach().cpu().numpy()
            top_k_predictions = sorted(list(zip(items_indexes, probs)), key=lambda x: x[1], reverse=True)[:topK]
            top_k_predictions = np.asarray(top_k_predictions)
            predicted_items = top_k_predictions[:,0]
            predicted_scores = top_k_predictions[:,1]
            ndcg_score = ndcg(items, predicted_items, predicted_scores)
            hr_score = hr(items, predicted_items)

            sum_writer.add_scalar(ndcg_tag, ndcg_score, i)
            sum_writer.add_scalar(hr_tag, hr_score, i)