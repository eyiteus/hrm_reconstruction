from Utils.trainer import evaluate_standard_model, train_standard_model


def evaluate_bert(model, val_loader, device):
    return evaluate_standard_model(model, val_loader, device)


def train_bert(*args, **kwargs):
    return train_standard_model(*args, model_name="bert", **kwargs)