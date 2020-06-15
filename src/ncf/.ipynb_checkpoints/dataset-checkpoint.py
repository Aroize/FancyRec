from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import torch


class ItemDataset(Dataset):
    def __init__(self, data, user_mapper, item_mapper, user_features=None):
        self.data = data
        self.user_mapper = user_mapper
        self.item_mapper = item_mapper
        self.user_features = user_features
            
    def __getitem__(self, idx):
        row = self.data[idx]
        user, item, label = row[0], row[1], row[2]
        user_idx = user
        user = self.user_mapper[user]
        item = self.item_mapper[item]
        user = torch.tensor(user).long()
        item = torch.tensor(item).long()
        label = torch.tensor(label).float()
        net_input = [user, item]
        if self.user_features:
            if user_idx in self.user_features:
                user_info = self.user_features[user_idx]
            else:
                user_info = [0, 0, 0, 0.0]
            user_info_tensor = torch.tensor(user_info)
            net_input.append(user_info_tensor)
        return (net_input, label)
    
    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def get_mappers(data):
        user_to_idx = {}
        item_to_idx = {}
        for u,i in data:
            if u not in user_to_idx:
                idx = len(user_to_idx)
                user_to_idx[u] = idx
            if i not in item_to_idx:
                idx = len(item_to_idx)
                item_to_idx[i] = idx
        return (user_to_idx, item_to_idx)
    
    @staticmethod
    def user_info_preprocessed(users):
        user_to_info = {}
        max_playcount = -1
        for _, _, _, _, playcount in users:
            max_playcount = max(max_playcount, playcount)
        country_indices = dict()
        for uid, age, country, gender, playcount in users:
            gender = int(gender == 'f')
            if country not in country_indices:
                country_indices[country] = len(country_indices)
            country = country_indices[country]
            user_to_info[uid] = [age, country, gender, float(playcount) / max_playcount]
        return user_to_info
    
    
class ValidationDataset(Dataset):
    
    def __init__(self, data, user_mapper, item_mapper, user_features=None):
        user_to_items = {}
        for u,i in tqdm(data):
            user_idx = user_mapper[u]
            item_idx = item_mapper[i]
            if user_idx not in user_to_items:
                user_to_items[user_idx] = set()
            user_to_items[user_idx].add(item_idx)
        self.data = list(user_to_items.items())
        self.user_features = user_features
        
    def __getitem__(self, idx):
        data = self.data[idx]
        user, items = data
        result = [user, items]
        if self.user_features:
            user_info = self.user_features[user]
            user_info_tensor = torch.tensor(user_info)
            result.append(user_info_tensor)
        return result
    
    def __len__(self):
        return len(self.data)