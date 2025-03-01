def change_hyperparameters(best_params, search_space, change_factor=0.2, change_proba=0.5):
    """
    Change a subset of the best hyperparameters within a small range.
    Uses gaussian noise for perturbations.
    """

    new_params = best_params.copy()
    for key in best_params.keys():
        if np.random.rand() < change_proba:
            lower, upper = search_space[key]
            
            if isinstance(search_space[key][0], int):
                noise = int(round(np.random.normal(0, change_factor * best_params[key])))  # Using random gaussian noise to perturb the hyperparameters.
                new_params[key] = best_params[key] + noise  # Can then apply np.clip to constrict the search space.
                
            elif isinstance(search_space[key][0], float):  
                if lower > 0 and upper / lower > 10:  # Large range -> log perturbation
                    log_value = np.log10(best_params[key])
                    log_noise = np.random.normal(0, change_factor)
                    new_params[key] = 10 ** (log_value + log_noise)
                else:  # Small range -> normal perturbation
                    noise = np.random.normal(0, change_factor * best_params[key])
            else:  
                new_params[key] = np.random.choice(search_space[key]) 
    return new_params

def guided_hyperparameter_search(df, max_trials=50, patience=10, change_factor=0.2, change_proba=0.5, output_dim=10, initial_epochs=5):
    """
    Iterative hyperparameter search algorithm that records the best set of hyperparameters based on validation accuracy.
    Slowly constricts the change_factor (and the search space) if the iterated models stop performing better.
    Can be modified based on different model architectures.
    """
    
    import logging
    import tensorflow as tf
    import numpy as np
    from model import predictor_model, rnn_cnn_layer
    from train import train

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)  # Setting up the logger.

    search_space = {
        "rnn_cnn_layers": [1, 10],
        "optimizer": ["Nadam", "Adam", "RMSprop", "SGD"],
        "kernel_size": [1, 20],
        "dense_layers": [1, 5],
        "dense_neurons": [16, 128],
        "rnn_neurons": [16, 128],
        "cnn_filters": [16, 128],
        "dropout_rate": [0.1, 0.3],
        "recurrent_dropout_rate": [0.0, 0.2],
        "activation": ["relu", "swish"],
        "learning_rate": [1e-4, 1e-2],
        "decay_steps": [1e3, 1e6],
        "decay_rate": [0.85, 0.999]
    } 
    # I chose the ranges arbitrarily, the search process could be made more efficient with better selections.

    best_acc = 0
    best_params = None
    no_improvement_count = 0
    n_epochs = initial_epochs

    for trial in range(1, max_trials + 1):
        logger.info(f"\nTrial {trial}/{max_trials}")
        
        if trial == 1 or best_params is None:
            params = {key: np.random.choice(search_space[key]) if isinstance(search_space[key][0], str) 
                     else np.random.uniform(*search_space[key]) if isinstance(search_space[key][0], float)
                     else np.random.randint(search_space[key][0], search_space[key][1] + 1) 
                     for key in search_space}
        else:
            params = change_hyperparameters(best_params, search_space, change_factor, change_proba)
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=params["learning_rate"],
            decay_steps=params["decay_steps"],
            decay_rate=params["decay_rate"],
            staircase=False
        )
        optimizer = getattr(tf.keras.optimizers, params["optimizer"])(learning_rate=lr_schedule)

        model = predictor_model(rnn_cnn_layers=params["rnn_cnn_layers"], rnn_neurons=params["rnn_neurons"], cnn_filters=params["cnn_filters"],
                              kernel_size=params["kernel_size"], dense_layers=params["dense_layers"], dense_neurons=params["dense_neurons"],
                              dropout_rate=params["dropout_rate"], recurrent_dropout_rate=params["recurrent_dropout_rate"], 
                              activation=params["activation"])
        model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                     metrics=["accuracy"])

        logger.info(f"Testing: {params}")
        
        model, valid_acc = train(df, model, initial_size=0.9, test_pct=0.1, seq_length=100, patience=patience)  # I set the initial_size to 0.9 here to get faster results.

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_params = params
            no_improvement_count = 0
            n_epochs += 2
        else:
            no_improvement_count += 1

        logger.info(f"Best so far: {best_params} with accuracy {best_acc:.4f}")

        if no_improvement_count > 0:
            change_factor *= (1 - ((no_improvement_count - 1) / patience)**2)  # Squaring to slow down the convergence speed.

        if no_improvement_count >= patience:
            logger.info("No improvement for several trials. Stopping search.")
            break

    return best_params, best_acc