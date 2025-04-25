from .func_models import *

def load_models(opt):
    print("Load {} model...".format(opt["model_type"]))
    if opt["model_type"] == "BlankCox":
        return BlankCoxModel(opt)
    elif opt["model_type"] == "BlankClass":
        return BlankClassModel(opt)
    elif opt["model_type"] == "ClassCox":
        return ClassCoxModel(opt)
    elif opt["model_type"] == "ClassClass":
        return ClassClassModel(opt)
    else:
        raise ValueError("{} haven't been implemented yet!".format(opt["model_type"]))
        