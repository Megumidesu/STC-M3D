from torch import nn
from torch.nn import functional as F

class ViewWeightsModule(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim 
        self.hidden_dim = input_dim // 2
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim // 2, 1) 
        )
    
    def forward(self, img_feat):
        B, V, D = img_feat.shape
        
        encoded_features = self.feature_encoder(img_feat) 
        
        attention_scores = self.attention(encoded_features).squeeze(-1)  
        
        weights = F.softmax(attention_scores, dim=1)  
        
        return weights


class ViewFusionModule(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim 
        self.hidden_dim = input_dim // 2
        self.output_dim = input_dim
        
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        self.recalibration = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.output_dim, self.output_dim)
        )
    
    def forward(self, img_feat, weights):
        B, V, D = img_feat.shape
        
        transformed_feat = self.feature_transform(img_feat)  
        
        weights = weights.unsqueeze(-1) 
        weighted_features = transformed_feat * weights  
        
        fused_feature = weighted_features.sum(dim=1) 
        
        fused_feature = self.recalibration(fused_feature)
        
        return fused_feature