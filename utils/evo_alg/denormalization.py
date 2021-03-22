def denormalize(X, xl, xu):
    _range = 1 if xu is None else (xu - xl)
    return X * _range + xl