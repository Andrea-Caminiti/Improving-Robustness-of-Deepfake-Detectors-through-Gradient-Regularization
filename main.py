import torch
import gc
import numpy as np
from Network.model import CBAMEfficientNet
from Train.train import train
from Train.loss import GradientRegularizedLoss
from Evaluation.test import test, plots
from Data.dataset import create_dataset
from globals import Config
from Evaluation.attacks import attacks

def main():
    config = Config()
    print(f'Using {config.device}')
    #Train Baseline
    train(config.train_dataPath, config.valid_dataPath, epochs=50, patience=5, device=config.device)
    #Train Gradient Regularized model
    train(config.train_dataPath, config.valid_dataPath, epochs=50, patience=5, loss_fn=GradientRegularizedLoss(config.r, config.alpha), device=config.device)
    
    model = CBAMEfficientNet(2)
    model.load_state_dict(config.best_model_normal)
    #Test Baseline
    test(model, config.test_dataPath, device=config.device, log_path=config.logPath_baseline)
    #Attack Baseline
    attacks(model, config.test_dataPath, config.device, config.logPath_attack_baseline)
    del model
    gc.collect()
    model = CBAMEfficientNet(2)
    model.load_state_dict(config.best_model_regularized)
    #Test Regularized model
    test(model, config.test_dataPath, device=config.device, log_path=config.logPath_regularized)
    attacks(model, config.test_dataPath, config.device, config.logPath_attack_regularized)


if __name__ == '__main__':
    seed = 3233
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    main()