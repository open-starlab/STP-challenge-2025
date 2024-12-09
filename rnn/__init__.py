from .rnn import RNN

def load_model(model_name, params, parser=None):
    model_name = model_name.lower()

    if model_name == 'rnn':
        return RNN(params, parser)
    else:
        raise NotImplementedError
