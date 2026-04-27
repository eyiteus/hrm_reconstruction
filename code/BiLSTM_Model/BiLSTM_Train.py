from Utils.trainer import evaluate_standard_model, train_standard_model


def evaluate_bilstm(model, val_loader, device):
    return evaluate_standard_model(model, val_loader, device)


def train_bilstm(*args, **kwargs):
    return train_standard_model(*args, model_name="bilstm", **kwargs)