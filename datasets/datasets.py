datasets = dict()

def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator

def make_dataset(name):
    return datasets[name]