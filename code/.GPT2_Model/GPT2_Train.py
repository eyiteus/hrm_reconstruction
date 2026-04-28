from Utils.trainer import evaluate_standard_model, train_standard_model


def evaluate_gpt2(model, val_loader, device):
    return evaluate_standard_model(model, val_loader, device)


def train_gpt2(*args, **kwargs):
    return train_standard_model(*args, model_name="gpt2", **kwargs)