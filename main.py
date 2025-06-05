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
    train(config.train_dataPath, config.valid_dataPath, epochs=20, patience=3, threshold = 0.1, device=config.device)
    # #Train Gradient Regularized model
    train(config.train_dataPath, config.valid_dataPath, epochs=20, patience=3, threshold = 0.1, loss_fn=GradientRegularizedLoss(config.r, config.alpha, device=config.device), device=config.device)
    plots('logs/Baseline')
    plots('logs/Gradient Regularized')
    model = CBAMEfficientNet(2)
    model.load_state_dict(torch.load(config.best_model_normal, weights_only=True))
    #Test Baseline
    test(model, config.test_dataPath, device=config.device, log_path=config.logPath_baseline)
    #Attack Baseline
    model = CBAMEfficientNet(2, attack=True)
    model.load_state_dict(torch.load(config.best_model_normal, weights_only=True))
    attacks(model, config.test_dataPath, config.device, config.logPath_attack_baseline)
    del model
    gc.collect()
    model = CBAMEfficientNet(2)
    model.load_state_dict(torch.load(config.best_model_regularized, weights_only=True))
    # #Test Regularized model
    test(model, config.test_dataPath, device=config.device, log_path=config.logPath_regularized)
    model = CBAMEfficientNet(2, attack=True)
    model.load_state_dict(torch.load(config.best_model_regularized, weights_only=True))
    attacks(model, config.test_dataPath, config.device, config.logPath_attack_regularized)


if __name__ == '__main__':
    seed = 2025
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    main()