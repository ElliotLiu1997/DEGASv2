# set up configurations, we may copy these configurations into a single file
# if total folds = 1, we set fold = -1 which indicate we doesn't split subfolds
BlankClass_opt = { "gpu_id": 0, "input_shape": 25, "feature_dim": 25, "low_reso_output_shape": 2, "high_reso_output_shape": 1, # do not need, dummy variable 
        "dropout_rate": 0.5, "lr": 0.01, "beta1": 0.9, "save_freq": 50, # parameters do not change
        "transfer_type": "Wasserstein", "graph_type": None, "model_type": "BlankClass", "loss_type": "cross_entropy",
        "lambda2": 3.0, "lambda3": 3.0, "gp_weight": 10.0, "n_critic": 2, "tot_iters": 300, # model hyper-parameter information (to change)
        "data_name": "my_data", "batch_size": 200, "pat_batch_size": 200, "sample_method": "balance",
        "is_save": False, "extract_embs": False, "fold": -1, "seed": 0, "tot_folds": 1, "tot_seeds": 10,
        }

ClassClass_opt = { "gpu_id": 0, "input_shape": 25, "feature_dim": 25, "low_reso_output_shape": 2, "high_reso_output_shape": -1, 
        "dropout_rate": 0.5, "lr": 0.01, "beta1": 0.9, "save_freq": 50, # parameters do not change
        "transfer_type": "Wasserstein", "graph_type": None, "model_type": "ClassClass", "loss_type": "cross_entropy",
        "lambda1": 1.0, "lambda2": 3.0, "lambda3": 3.0, "gp_weight": 10.0, "n_critic": 2, "tot_iters": 300, # model hyper-parameter information (to change)
        "data_name": "my_data", "batch_size": 200, "pat_batch_size": 200, "sample_method": "balance",
        "is_save": False, "extract_embs": False, "fold": -1, "seed": 0, "tot_folds": 1, "tot_seeds": 10,
        }

BlankCox_opt = { "gpu_id": 0, "input_shape": 25, "feature_dim": 25, "low_reso_output_shape": 1, "high_reso_output_shape": 1, # do not need, dummy variable 
        "dropout_rate": 0.5, "lr": 0.01, "beta1": 0.9, "save_freq": 50, # parameters do not change
        "transfer_type": "Wasserstein", "graph_type": None, "model_type": "BlankCox", "loss_type": "log_neg",  
        "lambda2": 3.0, "lambda3": 3.0, "gp_weight": 10.0, "n_critic": 2, "tot_iters": 300, # model hyper-parameter information (to change)
        "data_name": "my_data", "batch_size": 200, "pat_batch_size": 200, "sample_method": "balance",
        "is_save": False, "extract_embs": False, "fold": -1, "seed": 0, "tot_folds": 1, "tot_seeds": 10,
        }

ClassCox_opt = { "gpu_id": 0, "input_shape": 25, "feature_dim": 25, "low_reso_output_shape": 1, "high_reso_output_shape": -1, 
        "dropout_rate": 0.5, "lr": 0.01, "beta1": 0.9, "save_freq": 50, # parameters do not change
        "transfer_type": "Wasserstein", "graph_type": None, "model_type": "ClassCox", "loss_type": "log_neg",  
        "lambda1": 1.0, "lambda2": 3.0, "lambda3": 3.0, "gp_weight": 10.0, "n_critic": 2, "tot_iters": 300, # model hyper-parameter information (to change)
        "data_name": "my_data", "batch_size": 200, "pat_batch_size": 200, "sample_method": "balance",
        "is_save": False, "extract_embs": False, "fold": -1, "seed": 0, "tot_folds": 1, "tot_seeds": 10,
        }


