import torch

class Config:
    def __init__(self):
        #Log Paths
        self.logPath_baseline = 'logs/baseline.txt'
        self.logPath_regularized = 'logs/regularized.txt'
        self.logPath_attack_baseline = 'logs/adversarial_baseline'
        self.logPath_attack_regularized = 'logs/adversarial_regularized'
        #Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #Data Paths
        self.train_dataPath = 'CV Dataset\Train'
        self.valid_dataPath = 'CV Dataset\Validtion'
        self.test_dataPath = 'CV Dataset\Test'
        #Loss parameters
        self.r = 0.05
        self.alpha = 0.75
        #Model Paths
        self.best_model_normal = ''
        self.best_model_regularized = ''
