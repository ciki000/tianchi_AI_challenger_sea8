args_resnet = {
    'epochs': 200,
    'optimizer_name': 'SGD',
    'optimizer_hyperparameters': {
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 1e-4
    },
    'scheduler_name': 'CosineAnnealingLR',
    'scheduler_hyperparameters': {
        'T_max': 200
    },
    # 'scheduler_name': 'MultiStepLR',
    # 'scheduler_hyperparameters': {
    #     'milestones': [100, 150]
    # },
    'batch_size': 256,
}
args_densenet = {
    'epochs': 200,
    'optimizer_name': 'SGD',
    'optimizer_hyperparameters': {
        'lr': 0.1,
        'momentum': 0.9,
        # 'nesterov': True,
        'weight_decay': 1e-4
    },
    'scheduler_name': 'CosineAnnealingLR',
    'scheduler_hyperparameters': {
        'T_max': 200
    },
    'batch_size': 256,
}