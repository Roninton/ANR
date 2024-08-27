models = dict()

def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator

def make_model(name):
    return models[name]
