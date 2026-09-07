
from ray import tune
from .solver import Solver
from functools import partial
import torch
import torch.nn as nn

def _getoptions(size):
    array = []
    option = 4/size
    while option <= size/4:
        array.append(option)
        option *= 4

    print(array)
    return array

def _hypertrain(config, params, ts, vol = None):
    print(params, ts, config)
    solver = Solver(ts)
    solver._setfromconfig(config, params)
    if vol is not None:
        solver.eval(vol)
    else:
        solver.eval()
    solver.train(ts)
    print(solver.results._tables)
    tune.report(loss=solver.results._tables['losses_losses'], accuracy=solver.results._tables['tiltseries_psnrs'])


def hypertrain(trials, params, ts, vol = None):
    size = ts.data.shape[0]
    options = _getoptions(size)
    config = {
        "depth": tune.choice([i for i in range(20)]),
        "option": tune.choice(options),
        "gamma": tune.uniform(0.1,0.9),
        "lr": tune.loguniform(1e-6, 1e-1),
        "batch_size": tune.choice([1, 2]),
    }

    scheduler = tune.schedulers.ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=100000,
        grace_period=1,
        reduction_factor=2)

    result = tune.run(
        partial(_hypertrain, params=params, ts=ts, vol=vol),
        config=config,
        resources_per_trial={'cpu':1, 'gpu':1},
        num_samples=trials,
        scheduler=scheduler,
        progress_reporter=tune.JupyterNotebookReporter(overwrite=True)
    )

    best_trial = result.get_best_trial("accuracy", "max", "last")
    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="accuracy", mode="max")
    #best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    if torch.cuda.is_available():
        device = "cuda:0"
        best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)
